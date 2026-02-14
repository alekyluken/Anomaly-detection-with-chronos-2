import torch, os, numpy as np
import torch.nn as nn
import torch.nn.functional as F

from customGNN import SpatioTemporalAnomalyGNN
from customDataLoader import CustomDataset

from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning

import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from json import load as jsonLoad, dump as jsonDump

if torch.cuda.is_available():
    torch.cuda.empty_cache()



class FocalLoss(nn.Module):
    """
    Focal Loss per gestire severe class imbalance (numericamente stabile).
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: peso per la classe positiva (anomalie). Default 0.9
        gamma: focusing parameter. Default 2.0 (è stabile, >2.5 rischia NaN)
        reduction: 'mean' (consigliato), 'sum', o 'none'
    """
    def __init__(self, alpha: float = 0.9, gamma: float = 2.0, reduction: str = 'mean'):
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
            
        if self.consistency is not None:
            loss_consistency = self.consistency(pred, binary)
            components['consistency'] = loss_consistency.item()
            total_loss = total_loss + self.consistency_weight * loss_consistency

        components['total'] = total_loss.item()
        return total_loss, components


def trainLoop(gnn:SpatioTemporalAnomalyGNN, train_loader:torch.utils.data.DataLoader, optimizer:torch.optim.Optimizer, 
            criterion:torch.nn.Module, device:torch.device)->list[float]:
    gnn.train()

    total_loss = 0.0
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    for cont, mat, gt in tqdm(train_loader, desc="Training Batches", leave=False):
        optimizer.zero_grad()
        cont, mat, gt = cont.to(device), mat.to(device), gt.to(device)

        cont, binary_logits = gnn(cont, mat)  # [B, T, 1], [B, T, 1]

        loss = criterion(cont, binary_logits, gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item()
        
        # Flatten per metriche: [B, T, 1] → [B*T]
        outputs_flat = (torch.sigmoid(binary_logits).detach().cpu().numpy().flatten() > 0.5).astype(int)
        gt_flat = gt.detach().cpu().numpy().flatten().astype(int)
        
        accuracy += accuracy_score(gt_flat, outputs_flat)
        precision += precision_score(gt_flat, outputs_flat, zero_division=0)
        recall += recall_score(gt_flat, outputs_flat, zero_division=0)
        f1 += f1_score(gt_flat, outputs_flat, zero_division=0)

    return total_loss / len(train_loader), accuracy / len(train_loader), precision / len(train_loader), recall / len(train_loader), f1 / len(train_loader)


def evalLoop(gnn:SpatioTemporalAnomalyGNN, eval_loader:torch.utils.data.DataLoader, 
            criterion:torch.nn.Module, device:torch.device)->None:
    gnn.eval()
    total_loss = 0.0
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for cont, mat, gt in eval_loader:
            cont, mat, gt = cont.to(device), mat.to(device), gt.to(device)

            cont, binary_logits = gnn(cont, mat)  # [B, T, 1], [B, T, 1]
            
            loss = criterion(cont, binary_logits, gt)
            
            total_loss += loss.item()
            
            # Flatten per metriche: [B, T, 1] → [B*T]
            outputs_flat = (torch.sigmoid(binary_logits).cpu().numpy().flatten() >= 0.5).astype(int)
            gt_flat = gt.cpu().numpy().flatten().astype(int)
            
            accuracy += accuracy_score(gt_flat, outputs_flat)
            precision += precision_score(gt_flat, outputs_flat, zero_division=0)
            recall += recall_score(gt_flat, outputs_flat, zero_division=0)
            f1 += f1_score(gt_flat, outputs_flat, zero_division=0)

    return total_loss / len(eval_loader), accuracy / len(eval_loader), precision / len(eval_loader), recall / len(eval_loader), f1 / len(eval_loader)


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


def mainLoop(cfg:dict, device: torch.device)->None:
    gnn = SpatioTemporalAnomalyGNN(hidden_dim=cfg.get('gnn_hidden_dim', 64), num_gnn_layers=cfg.get('gnn_num_layers', 3), num_temporal_layers=cfg.get('gnn_num_temporal_layers', 3), dropout=cfg.get('gnn_dropout', 0.3)).to(device)
    filteredData = [f for f in os.listdir(cfg['model_save_path']) if f.startswith("gnn_epoch_") and f.endswith(".pth")]
    if cfg.get("RESUME_TRAINING", False) and len(filteredData) > 0:
        latest_model = max(filteredData, key=lambda x: int(x.split("_")[2].split(".")[0]))
        gnn.load_state_dict(torch.load(os.path.join(cfg['model_save_path'], latest_model), map_location=device))
        startingEpoch = int(latest_model.split("_")[2].split(".")[0])
        print(f"Resuming training from epoch {startingEpoch} using model {latest_model}.")
    else:
        startingEpoch = 0
    optimizer = torch.optim.SGD(gnn.parameters(), lr=cfg.get("lr", 0.01), momentum=cfg.get("momentum", 0.9), weight_decay=cfg.get("weight_decay", 1e-4), nesterov=cfg.get("nesterov", True))
    
    # Loss combinata per class imbalance (~1-2% anomalie)
    criterion = CombinedAnomalyLoss(
        focal_alpha=cfg.get('focal_alpha', 0.75),      # peso per classe positiva
        focal_gamma=cfg.get('focal_gamma', 2.0),       # focusing parameter
        ranking_margin=cfg.get('ranking_margin', 1.0), # margine separazione score
        ranking_weight=cfg.get('ranking_weight', 0.3), # peso ranking loss
        dice_weight=cfg.get('dice_weight', 0.2),       # peso dice loss (F1-like)
        consistency_weight=cfg.get('consistency_weight', 0.5), # peso consistency loss
        only_ce=cfg.get('only_ce', False),
        ce_weight=cfg.get('ce_weight', [1, 99])
    )
    
    scheduler: torch.optim.lr_scheduler._LRScheduler = getScheduler(optimizer, cfg['num_epochs'], cfg.get('warmup_fraction', 0.1))
    trainFiles, valFiles = os.listdir(cfg['train_data_path']), os.listdir(cfg['val_data_path'])
    if cfg['exclude_WADI']:
        trainFiles, valFiles = list(filter(lambda x: "WADI" not in x, trainFiles)), list(filter(lambda x: "WADI" not in x, valFiles))

    for epoch in range(startingEpoch, cfg['num_epochs']):
        np.random.shuffle(trainFiles)
        print(f"Epoch {epoch+1}/{cfg['num_epochs']}")

        trainLoss, trainAcc, trainPrecision, trainRecall, trainF1 = 0.0, 0.0, 0.0, 0.0, 0.0
        for file in tqdm(trainFiles, desc="Training", leave=False):
            try:
                trainLoader = torch.utils.data.DataLoader(CustomDataset(file, cfg['data_path_custom']), batch_size=cfg['batch_size'] if "WADI" not in file else 4, shuffle=True)
            except ValueError as e:
                continue
            tLoss, tAcc, tPrecision, tRecall, tF1 = trainLoop(gnn, trainLoader, optimizer, criterion, device)
            trainLoss, trainAcc, trainPrecision, trainRecall, trainF1 = trainLoss + tLoss, trainAcc + tAcc, trainPrecision + tPrecision, trainRecall + tRecall, trainF1 + tF1

        print(f"Average Training Loss: {trainLoss/len(trainFiles):.6f}")
        print(f"Average Training Accuracy: {trainAcc / len(trainFiles):.6f}")
        print(f"Average Training Precision: {trainPrecision / len(trainFiles):.6f}")
        print(f"Average Training Recall: {trainRecall / len(trainFiles):.6f}")
        print(f"Average Training F1 Score: {trainF1 / len(trainFiles):.6f}", "\n")

        valLoss, valAcc, valPrecision, valRecall, valF1 = 0.0, 0.0, 0.0, 0.0, 0.0
        for file in tqdm(valFiles, desc="Validating", leave=False):
            vLoss, vAcc, vPrecision, vRecall, vF1 = evalLoop(gnn, torch.utils.data.DataLoader(CustomDataset(file, cfg['data_path_custom']), batch_size=cfg['batch_size'] if "WADI" not in file else 4, shuffle=False), criterion, device)
            valLoss, valAcc, valPrecision, valRecall, valF1 = valLoss + vLoss, valAcc + vAcc, valPrecision + vPrecision, valRecall + vRecall, valF1 + vF1

        print(f"Average Validation Loss: {valLoss/ len(valFiles):.6f}")
        print(f"Average Validation Accuracy: {valAcc / len(valFiles):.6f}")
        print(f"Average Validation Precision: {valPrecision / len(valFiles):.6f}")
        print(f"Average Validation Recall: {valRecall / len(valFiles):.6f}")
        print(f"Average Validation F1 Score: {valF1 / len(valFiles):.6f}", "\n")
        
        scheduler.step()

        torch.save({'model_state_dict': gnn.state_dict(), 'config': cfg}, os.path.join(cfg['model_save_path'], f"gnn_epoch_{epoch+1}.pth"))
        if os.path.exists(os.path.join(cfg['model_save_path'], "training_log.json")):
            with open(os.path.join(cfg['model_save_path'], "training_log.json"), 'r') as f:
                training_log = jsonLoad(f)
        else:
            training_log = []

        training_log.append({
            'epoch': epoch+1, 'train_loss': trainLoss/len(trainFiles), 'train_accuracy': trainAcc/len(trainFiles), 'train_precision': trainPrecision/len(trainFiles), 'train_recall': trainRecall/len(trainFiles), 'train_f1': trainF1/len(trainFiles), 
            'val_loss': valLoss/len(valFiles), 'val_accuracy': valAcc/len(valFiles), 'val_precision': valPrecision/len(valFiles), 'val_recall': valRecall/len(valFiles), 'val_f1': valF1/len(valFiles)
        })
        with open(os.path.join(cfg['model_save_path'], "training_log.json"), 'w') as f:
            jsonDump(training_log, f, indent=4)
        print(f"Epoch {epoch+1} completed. Model saved and log updated.")
        print("\n", "-" * 50, "\n")


if __name__ == "__main__":
    for v1,v2 in zip([True, False], ["SAVED_MODELSCE", "SAVED_MODELSCE_WADI"]):
        cfg = {
            'train_data_path': "./TRAIN_SPLIT",
            "data_path_custom": "./PROCESSED_TRAIN_DATA",
            'val_data_path': "./TEST_SPLIT",
            'model_save_path': f"./{v2}",
            'exclude_WADI': v1,  # Escludi dataset WADI se True


            'num_epochs': 40,

            'batch_size': 32*2,
            'lr': 0.002,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'nesterov': True,

            'gnn_hidden_dim': 128,
            'gnn_num_layers': 6,
            'gnn_num_temporal_layers': 6,
            'gnn_dropout': 0.3,

            "percentile_threshold": 95.0,
            'warmup_fraction': 0.15,

            "RESUME_TRAINING": False,  # Se True, carica ultimo modello salvato e riprendi training
            # Loss parameters per class imbalance (~1-2% anomalie) - STABILI
            'focal_alpha': 0.9,        # peso classe positiva (0.9 = 9x importanza)
            'focal_gamma': 2.0,        # focusing: 2.0 è stabile, >2.5 rischia NaN
            'ranking_margin': 1.0,     # margine minimo score(anomalia) - score(normale)
            'ranking_weight': -1,     # peso ranking loss nella loss totale
            'dice_weight': 1,        # peso dice loss (ottimizza F1-like)
            'consistency_weight': 1.2,  # peso consistency loss tra i due head

            'only_ce':True,
            'ce_weight':[1, 200] # peso per le classi nella CrossEntropy Loss
        }

        os.makedirs(cfg['model_save_path'], exist_ok=True)
        with open(os.path.join(cfg['model_save_path'], "final_config.json"), 'w') as f:
            jsonDump(cfg, f, indent=4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mainLoop(cfg, device)
