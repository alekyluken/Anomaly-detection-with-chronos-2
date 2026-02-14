"""
Training Loop per SimpleQuantileAnomalyDetector.

Struttura allineata con trainGNN.py:
- trainLoop / evalLoop separati
- CombinedAnomalyLoss (Focal + Ranking + Dice + Consistency)
- Warmup + Cosine Annealing scheduler
- Resume training da ultimo checkpoint
- Salvataggio per-epoch con training_log.json
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from json import load as jsonLoad, dump as jsonDump

# Ensure Nunzio/ is on sys.path for sibling imports
_NUNZIO_DIR = os.path.dirname(os.path.abspath(__file__))
if _NUNZIO_DIR not in sys.path:
    sys.path.insert(0, _NUNZIO_DIR)

from SimplerNN import SimpleQuantileAnomalyDetector
from customDataLoaderV2 import CustomDataset


if torch.cuda.is_available():
    torch.cuda.empty_cache()


def getScheduler(optimizer:torch.optim.Optimizer, total_epochs:int, fractionWarmUp:float=0.1)->torch.optim.lr_scheduler.SequentialLR:
    """Create a learning rate scheduler with warm-up and cosine annealing.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        total_epochs (int): Total number of training epochs.
        fractionWarmUp (float): Fraction of total epochs to use for warm-up. Default is 0.1.
        
    Returns:
        scheduler (torch.optim.lr_scheduler.SequentialLR): The learning rate scheduler.
    """
    warmup_epochs = int(total_epochs * fractionWarmUp)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,  
    total_iters=warmup_epochs
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )



# ═══════════════════════════════════════════════════════════════
# LOSS COMBINATA (Focal + Ranking + Dice + Consistency)
# ═══════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss per gestire severe class imbalance (numericamente stabile).
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: peso per la classe positiva (anomalie). Default 0.9
        gamma: focusing parameter. Default 2.0 (è stabile, >2.5 rischia NaN)
        reduction: 'mean' (consigliato), 'sum', o 'none'
    """
    def __init__(self, alpha: float = 0.9, gamma: float = 2.0, reduction: str = 'sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-7
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp logits per stabilità
        # pred = pred.clamp(-20, 20)
        
        # Sigmoid con clamp per evitare log(0)
        p = torch.sigmoid(pred).clamp(self.eps, 1 - self.eps)
        
        # Focal weight: clamp per evitare instabilità con gamma alto
        focal_weight = ((1 - p * target - (1 - p) * (1 - target)) ** self.gamma).clamp(max=10)
        
        focal_loss = (self.alpha * target + (1 - self.alpha) * (1 - target)) * focal_weight * F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        match self.reduction:
            case 'mean': return focal_loss.mean()
            case 'sum':  return focal_loss.sum()
            case _:      return focal_loss

class WeightedBCELoss(nn.Module):
    """
    BCE con peso esplicito per la classe positiva.
    Semplice ed efficace per imbalance noti.
    
    Args:
        pos_weight: peso per classe positiva. Se anomalie=1%, usa ~99 o calcola dinamicamente
    """
    def __init__(self, pos_weight: float = 70.0):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=pred.device)
        else:
            pos_weight = torch.clamp(((1 - target).sum() + 1e-6) / (target.sum() + 1e-6), min=5.0, max=200.0) 

        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)


class RankingLoss(nn.Module):
    """
    Margin ranking loss: gli score delle anomalie devono essere > score normali + margin.
    Ottimo per separare bene le distribuzioni degli score.
    
    Args:
        margin: margine minimo tra anomalia e normale
    """
    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        anomaly_mask = labels == 1
        normal_mask = labels == 0
        
        # Se non ci sono entrambe le classi, return 0
        if not anomaly_mask.any() or not normal_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        return F.relu(self.margin - scores[anomaly_mask].unsqueeze(1) + scores[normal_mask].unsqueeze(0) ).mean()


class DiceLoss(nn.Module):
    """
    Dice Loss - molto usata per segmentazione con imbalance.
    Ottimizza direttamente il Dice coefficient (simile a F1).
    
    Args:
        smooth: smoothing factor per evitare divisione per zero
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred_prob = torch.sigmoid(pred.clamp(-20, 20))
        pred_prob = torch.sigmoid(pred)

        return (1 - (2. * (pred_prob * target).sum() + self.smooth) / (pred_prob.sum() + target.sum() + self.smooth))


class ConsistencyLoss(nn.Module):
    """
    Consistency Loss: the two heads should agree on normal or abnormal points.
    
    Args:
        weight: peso della consistency loss
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
        
    def forward(self, head1: torch.Tensor, head2: torch.Tensor) -> torch.Tensor:        
        return self.weight * F.mse_loss(torch.sigmoid(head1), torch.sigmoid(head2))

class CombinedAnomalyLoss(nn.Module):
    """
    Loss combinata ottimizzata per anomaly detection con severe imbalance (~1-2% anomalie).
    
    Combina:
    1. Focal Loss (α=0.96, γ=3): gestisce imbalance, focus su campioni difficili
    2. Ranking Loss: garantisce separazione tra score anomalie/normali  
    3. Dice Loss (opzionale): ottimizza direttamente F1-like metric
    
    Args:
        focal_alpha: peso classe positiva in focal loss
        focal_gamma: focusing parameter
        ranking_margin: margine per ranking loss
        ranking_weight: peso del ranking loss nel totale
        dice_weight: peso del dice loss (0 per disabilitare)
        only_ce: se True, usa solo CrossEntropyLoss con pesi specificati
        ce_weight: pesi per le classi nella CrossEntropyLoss
    """
    def __init__(
        self, 
        focal_alpha: float = 0.9,
        focal_gamma: float = 3.0,
        ranking_margin: float = 2.0,
        ranking_weight: float = 1,
        dice_weight: float = 2,
        consistency_weight: float = 1.0,
        only_ce: bool = False,
        ce_weight: list[float] = [1, 99]
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.ranking = RankingLoss(margin=ranking_margin) if ranking_weight > 0 else None
        self.dice = DiceLoss() if dice_weight > 0 else None
        self.consistency = ConsistencyLoss(weight=consistency_weight) if consistency_weight > 0 else None
        self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ce_weight[1]/ce_weight[0])) if only_ce else None


        self.ranking_weight = ranking_weight
        self.dice_weight = dice_weight
        self.consistency_weight = consistency_weight
        self.only_ce = only_ce
    def forward(self, pred: torch.Tensor, binary:torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Focal loss (principale)
        if self.only_ce and self.ce is not None:
            total_loss = self.ce(binary, target)
            if self.consistency is not None:
                total_loss = total_loss + self.consistency_weight * self.consistency(pred, binary)

        else:
            total_loss = self.focal(binary, target)
            
            # Ranking loss (separazione)
            if self.ranking is not None:
                total_loss = total_loss + self.ranking_weight * self.ranking(pred, target)
            
            # Dice loss (opzionale, per F1)
            if self.dice is not None:
                total_loss = total_loss + self.dice_weight * self.dice(binary, target)

            if self.consistency is not None:
                total_loss = total_loss + self.consistency_weight * self.consistency(pred, binary)
                
        return total_loss
    
    def forward_with_components(self, pred: torch.Tensor, binary: torch.Tensor, target: torch.Tensor) -> dict:
        """Ritorna loss totale + componenti per logging."""
        if self.only_ce and self.ce is not None:
            total_loss = self.ce(binary, target)
        else:
            total_loss = self.focal(binary, target)
        components = {'focal': total_loss.item()}

        if self.ranking is not None:
            loss_ranking = self.ranking(pred, target)
            total_loss = total_loss + self.ranking_weight * loss_ranking
            components['ranking'] = loss_ranking.item()

        if self.dice is not None:
            loss_dice = self.dice(binary, target)
            components['dice'] = loss_dice.item()
            total_loss = total_loss + self.dice_weight * loss_dice
            
        if self.consistency is not None :
            loss_consistency = self.consistency(pred, binary)
            components['consistency'] = loss_consistency.item()
            total_loss = total_loss + self.consistency_weight * loss_consistency

        components['total'] = total_loss.item()
        return total_loss, components


# ═══════════════════════════════════════════════════════════════
# TRAIN LOOP
# ═══════════════════════════════════════════════════════════════

def trainLoop(model: SimpleQuantileAnomalyDetector, train_loader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device):
    """Training loop per un singolo DataLoader (un file)."""
    model.train()

    total_loss = 0.0
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    for quantiles, values, gt in train_loader:
        quantiles, values, gt = quantiles.to(device), values.to(device), gt.to(device)

        optimizer.zero_grad()
        cont, binary_logits = model(quantiles, values)
        loss = criterion(cont, binary_logits, gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        outputs_flat = (torch.sigmoid(binary_logits).detach().cpu().numpy().flatten() >= 0.5).astype(int)
        gt_flat = gt.detach().cpu().numpy().flatten().astype(int)

        accuracy  += accuracy_score(gt_flat, outputs_flat)
        precision += precision_score(gt_flat, outputs_flat, zero_division=0)
        recall    += recall_score(gt_flat, outputs_flat, zero_division=0)
        f1        += f1_score(gt_flat, outputs_flat, zero_division=0)


    n = len(train_loader)
    return total_loss / n, accuracy / n, precision / n, recall / n, f1 / n


# ═══════════════════════════════════════════════════════════════
# EVAL LOOP
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def evalLoop(model: SimpleQuantileAnomalyDetector, eval_loader: torch.utils.data.DataLoader,
             criterion: nn.Module, device: torch.device):
    """Validation loop per un singolo DataLoader (un file)."""
    model.eval()

    total_loss = 0.0
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    for quantiles, values, gt in eval_loader:
        quantiles, values, gt = quantiles.to(device), values.to(device), gt.to(device)

        cont, binary_logits = model(quantiles, values)
        loss = criterion(cont, binary_logits, gt)
        total_loss += loss.item()

        outputs_flat = (torch.sigmoid(binary_logits).cpu().numpy().flatten() >= 0.5).astype(int)
        gt_flat = gt.cpu().numpy().flatten().astype(int)

        accuracy  += accuracy_score(gt_flat, outputs_flat)
        precision += precision_score(gt_flat, outputs_flat, zero_division=0)
        recall    += recall_score(gt_flat, outputs_flat, zero_division=0)
        f1        += f1_score(gt_flat, outputs_flat, zero_division=0)

    n = len(eval_loader)
    return total_loss / n, accuracy / n, precision / n, recall / n, f1 / n


# ═══════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def mainLoop(cfg: dict, device: torch.device) -> None:
    """Main training loop – struttura identica a trainGNN.mainLoop."""

    # ── Modello ──
    model = SimpleQuantileAnomalyDetector(
        in_features=cfg.get('in_features', 12),
        hidden_dim=cfg.get('hidden_dim', 32),
        kernel_size=cfg.get('kernel_size', 7),
        num_layers=cfg.get('num_layers', 3),
        num_attention_heads=cfg.get('num_attention_heads', 4),
        num_attention_layers=cfg.get('num_attention_layers', 2),
        dropout=cfg.get('dropout', 0.2)
    ).to(device)

    # ── Resume training ──
    filteredData = [f for f in os.listdir(cfg['model_save_path'])
                    if f.startswith("model_epoch_") and f.endswith(".pth")]
    if cfg.get("RESUME_TRAINING", False) and len(filteredData) > 0:
        latest_model = max(filteredData, key=lambda x: int(x.split("_")[2].split(".")[0]))
        model.load_state_dict(
            torch.load(os.path.join(cfg['model_save_path'], latest_model),
                        map_location=device)['model_state_dict']
        )
        startingEpoch = int(latest_model.split("_")[2].split(".")[0])
        print(f"Resuming training from epoch {startingEpoch} using {latest_model}.")
    else:
        startingEpoch = 0

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get('lr', 1e-3),
        weight_decay=cfg.get('weight_decay', 1e-4)
    )

    # ── Loss combinata ──
    criterion = CombinedAnomalyLoss(
        focal_alpha=cfg.get('focal_alpha', 0.9),
        focal_gamma=cfg.get('focal_gamma', 2.0),
        ranking_margin=cfg.get('ranking_margin', 1.0),
        ranking_weight=cfg.get('ranking_weight', 0.3),
        dice_weight=cfg.get('dice_weight', 0.2),
        consistency_weight=cfg.get('consistency_weight', 0.5),
        only_ce=cfg.get('only_ce', False),
        ce_weight=cfg.get('ce_weight', [1, 99])
    )

    # ── Scheduler (Warmup + Cosine) ──
    scheduler = getScheduler(optimizer, cfg['num_epochs'], cfg.get('warmup_fraction', 0.1))

    # ── File list ──
    trainFiles, valFiles = os.listdir(cfg['train_data_path']), os.listdir(cfg['val_data_path'])
    
    # ── Training loop ──
    for epoch in range(startingEpoch, cfg['num_epochs']):
        np.random.shuffle(trainFiles)
        print(f"\nEpoch {epoch+1}/{cfg['num_epochs']}  (LR: {optimizer.param_groups[0]['lr']:.6f})")

        # ── Train ──
        tLoss, tAcc, tPrec, tRec, tF1 = 0.0, 0.0, 0.0, 0.0, 0.0; nTrain = 0
        for file in tqdm(trainFiles, desc="Training Files", leave=False):
            try:
                loader = torch.utils.data.DataLoader(
                    CustomDataset(file, cfg['processed_data_dir'], cfg['train_data_path'], skipNames=cfg.get("exclude_names", {"WADI"})),
                    batch_size=cfg.get('batch_size', 32) if "WADI" not in file.upper() else 4,
                    shuffle=True,
                    num_workers=cfg.get('num_workers', 0),
                    pin_memory=True
                )
            except (ValueError, FileNotFoundError):
                continue

            loss, acc, prec, rec, f1 = trainLoop(model, loader, optimizer, criterion, device)
            tLoss += loss; tAcc += acc; tPrec += prec; tRec += rec; tF1 += f1; nTrain += 1

        if nTrain > 0:
            print(f"  Train  | Loss: {tLoss/nTrain:.6f}  Acc: {tAcc/nTrain:.4f}  "
                f"Prec: {tPrec/nTrain:.4f}  Rec: {tRec/nTrain:.4f}  F1: {tF1/nTrain:.4f}")

        # ── Validation ──
        vLoss, vAcc, vPrec, vRec, vF1 = 0.0, 0.0, 0.0, 0.0, 0.0; nVal = 0
        for file in tqdm(valFiles, desc="Validating Files", leave=False):
            try:
                loader = torch.utils.data.DataLoader(
                    CustomDataset(file, cfg['processed_data_dir'], cfg['val_data_path'], skipNames=cfg.get("exclude_names", {"WADI"}) if cfg.get("restrict_also_validation", True) else set()),
                    batch_size=cfg.get('batch_size', 32) if "WADI" not in file.upper() else 4,
                    shuffle=False,
                    num_workers=cfg.get('num_workers', 0),
                    pin_memory=True
                )
            except (ValueError, FileNotFoundError):
                continue

            loss, acc, prec, rec, f1 = evalLoop(model, loader, criterion, device)
            vLoss += loss; vAcc += acc; vPrec += prec; vRec += rec; vF1 += f1; nVal += 1

        if nVal > 0:
            print(f"  Valid  | Loss: {vLoss/nVal:.6f}  Acc: {vAcc/nVal:.4f}  "
                    f"Prec: {vPrec/nVal:.4f}  Rec: {vRec/nVal:.4f}  F1: {vF1/nVal:.4f}")

        # ── Scheduler step ──
        scheduler.step()

        # ── Save checkpoint ──
        torch.save(
            {'model_state_dict': model.state_dict(), 'config': cfg},
            os.path.join(cfg['model_save_path'], f"model_epoch_{epoch+1}.pth")
        )

        # ── Training log ──
        log_path = os.path.join(cfg['model_save_path'], "training_log.json")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                training_log = jsonLoad(f)
        else:
            training_log = []

        training_log.append({
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': tLoss / max(nTrain, 1),
            'train_accuracy': tAcc / max(nTrain, 1),
            'train_precision': tPrec / max(nTrain, 1),
            'train_recall': tRec / max(nTrain, 1),
            'train_f1': tF1 / max(nTrain, 1),
            'val_loss': vLoss / max(nVal, 1),
            'val_accuracy': vAcc / max(nVal, 1),
            'val_precision': vPrec / max(nVal, 1),
            'val_recall': vRec / max(nVal, 1),
            'val_f1': vF1 / max(nVal, 1),
        })
        with open(log_path, 'w') as f:
            jsonDump(training_log, f, indent=4)

        print(f"  Epoch {epoch+1} completata. Modello salvato.")
        print("-" * 60)


# ═══════════════════════════════════════════════════════════════
# CONFIG & MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ENABLE_TRAINING = True
    if ENABLE_TRAINING:
        cfg = {
            # ── Data ──
            'train_data_path': './TRAIN_SPLIT/',
            'val_data_path': './TEST_SPLIT/',
            'processed_data_dir': './PROCESSED_TRAIN_DATAV2/',
            'model_save_path': './SAVED_MODELS_SIMPLE_CONS/',

            # ── Model ──
            'in_features': 12,              # 12 feature dal QuantileFeatureExtractor
            'hidden_dim': 64,               # dimensione hidden Conv1D + Attention
            'kernel_size': 7,               # kernel Conv1D
            'num_layers': 3,                # strati Conv1D
            'num_attention_heads': 4,       # teste attention cross-variabile
            'num_attention_layers': 4,      # strati self-attention
            'dropout': 0.25,

            # ── Training ──
            'num_epochs': 40,
            'batch_size': 64,
            'lr': 5e-4,
            'weight_decay': 1e-4,
            'warmup_fraction': 0.25,
            'num_workers': 0,

            # ── Loss (CombinedAnomalyLoss) ──
            'focal_alpha': 0.999999,
            'focal_gamma': 2.5,
            'ranking_margin': 0.3,
            'ranking_weight': 1.5,
            'dice_weight': 1.5,
            'consistency_weight': 1.8,
            'only_ce': False,
            "only_ce+consistency":True,
            'ce_weight': [1, 1],

            # ── Misc ──
            'RESUME_TRAINING': False,
            "exclude_names": [], 
            "restrict_also_validation": True
        }

        os.makedirs(cfg['model_save_path'], exist_ok=True)
        with open(os.path.join(cfg['model_save_path'], "final_config.json"), 'w') as f:
            jsonDump(cfg, f, indent=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        mainLoop(cfg, device)
    else:
        print("Testing mode: loading model and evaluating on validation set.")
        TESTING_MODEL_PATH = "./SAVED_MODELS_SIMPLE_CONS/model_epoch_35.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = torch.load(TESTING_MODEL_PATH, map_location=device)

        model = SimpleQuantileAnomalyDetector(cfg['config']['in_features'], cfg['config']['hidden_dim'], cfg['config']['kernel_size'],
                                            cfg['config']['num_layers'], cfg['config']['num_attention_heads'], cfg['config']['num_attention_layers'], cfg['config']['dropout']).to(device)
        model.load_state_dict(cfg['model_state_dict'])
        model.eval()
        cfg = cfg['config']
        criterion = CombinedAnomalyLoss(focal_alpha=cfg.get('focal_alpha', 0.9), focal_gamma=cfg.get('focal_gamma', 2.0),
            ranking_margin=cfg.get('ranking_margin', 1.0), ranking_weight=-1,
            dice_weight=-1, consistency_weight=-1,
            only_ce=cfg.get('only_ce', True), ce_weight=cfg.get('ce_weight', [1, 1])
        )

        vLoss, vAcc, vPrec, vRec, vF1, nVal = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for file in sorted(os.listdir(cfg['val_data_path'])):
            try:
                loader = torch.utils.data.DataLoader(
                    CustomDataset(file, cfg['processed_data_dir'], cfg['val_data_path'], skipNames=set()),
                    batch_size=cfg['batch_size'],
                    shuffle=False,
                    num_workers=cfg['num_workers'],
                    pin_memory=True
                )
            except (ValueError, FileNotFoundError):
                continue

            (loss, acc, prec, rec, f1), nVal = evalLoop(model, loader, criterion, device), nVal + 1
            vLoss += loss; vAcc += acc; vPrec += prec; vRec += rec; vF1 += f1
            print(f"{file} | Loss: {loss:.6f}  Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")
        if nVal > 0:
            print("-" * 40)
            print(f"\nOverall Validation | Loss: {vLoss/nVal:.6f}  Acc: {vAcc/nVal:.4f}  "
                    f"Prec: {vPrec/nVal:.4f}  Rec: {vRec/nVal:.4f}  F1: {vF1/nVal:.4f}")
