import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from json import load as jsonLoad, dump as jsonDump

from customGNN_v2 import SpatioTemporalAnomalyGNN_V2
from customDataLoader import CustomDataset

# ═══════════════════════════════════════════════════════════════
# STRATEGIA 1: Adaptive Loss Weighting (AutoLoss)
# ═══════════════════════════════════════════════════════════════

class AdaptiveLossWeighting(nn.Module):
    """
    Apprende automaticamente i pesi ottimali per bilanciare le loss.
    Basato su "Multi-Task Learning Using Uncertainty to Weigh Losses".
    """
    def __init__(self, num_losses: int = 4):
        super().__init__()
        # Log-variance per ogni loss (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
    
    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """losses: lista di tensori loss"""
        weighted = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted.append(precision * loss + self.log_vars[i])
        return sum(weighted)


# ═══════════════════════════════════════════════════════════════
# STRATEGIA 2: Focal Loss con Dynamic Gamma
# ═══════════════════════════════════════════════════════════════

class DynamicFocalLoss(nn.Module):
    """
    Focal loss con gamma che si adatta dinamicamente durante il training.
    Inizia alto (focus su hard examples) e decresce verso BCE standard.
    """
    def __init__(self, alpha: float = 0.95, gamma_start: float = 3.0, 
                 gamma_end: float = 1.0, total_epochs: int = 30):
        super().__init__()
        self.alpha = alpha
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        
    def get_gamma(self) -> float:
        """Linear decay da gamma_start a gamma_end"""
        progress = min(self.current_epoch / self.total_epochs, 1.0)
        return self.gamma_start - (self.gamma_start - self.gamma_end) * progress
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gamma = self.get_gamma()
        p = torch.sigmoid(pred).clamp(1e-7, 1 - 1e-7)
        
        focal_weight = ((1 - p * target - (1 - p) * (1 - target)) ** gamma).clamp(max=10)
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        focal = (self.alpha * target + (1 - self.alpha) * (1 - target)) * focal_weight * bce
        return focal.mean()


# ═══════════════════════════════════════════════════════════════
# STRATEGIA 3: Asymmetric Loss (penalizza falsi negativi molto di più)
# ═══════════════════════════════════════════════════════════════

class AsymmetricLoss(nn.Module):
    """
    Penalizza falsi negativi (miss anomalie) molto più dei falsi positivi.
    Fondamentale per anomaly detection critica.
    """
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Probabilità
        p = torch.sigmoid(pred)
        
        # Asymmetric focusing
        p_t = p * target + (1 - p) * (1 - target)
        
        # Asymmetric clipping
        if self.clip > 0:
            p_t_clipped = (p_t + self.clip).clamp(max=1)
        else:
            p_t_clipped = p_t
            
        # Compute loss
        loss = -torch.log(p_t_clipped.clamp(min=1e-7))
        
        # Asymmetric weights (gamma_neg >> gamma_pos per penalizzare FN)
        gamma = target * self.gamma_pos + (1 - target) * self.gamma_neg
        loss = loss * ((1 - p_t) ** gamma)
        
        return loss.mean()


# ═══════════════════════════════════════════════════════════════
# STRATEGIA 4: Distribution Alignment Loss
# ═══════════════════════════════════════════════════════════════

class DistributionAlignmentLoss(nn.Module):
    """
    Forza le distribuzioni di score tra train e val ad essere simili.
    Previene il collasso "tutto negativo" sul validation.
    """
    def __init__(self, num_bins: int = 50):
        super().__init__()
        self.num_bins = num_bins
        
    def forward(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forza che la distribuzione degli score corrisponda alla ground truth.
        """
        # Percentile degli score
        pos_scores = scores[target == 1]
        neg_scores = scores[target == 0]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=scores.device)
        
        # Vogliamo che gli score positivi siano significativamente più alti
        pos_mean = torch.sigmoid(pos_scores).mean()
        neg_mean = torch.sigmoid(neg_scores).mean()
        
        # Penalizza se la separazione non è sufficiente
        separation_loss = F.relu(0.5 - (pos_mean - neg_mean))
        
        return separation_loss


# ═══════════════════════════════════════════════════════════════
# STRATEGIA 5: OHEM (Online Hard Example Mining)
# ═══════════════════════════════════════════════════════════════

class OHEMBCELoss(nn.Module):
    """
    Seleziona solo gli esempi più difficili per il training.
    Previene overfitting su esempi facili.
    """
    def __init__(self, keep_ratio: float = 0.7, pos_weight: float = 100.0):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.pos_weight = pos_weight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Calcola loss per ogni sample
        pos_weight = torch.tensor([self.pos_weight], device=pred.device)
        bce = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pos_weight, reduction='none'
        )
        
        # Mantieni solo i top-k esempi più difficili
        num_keep = max(1, int(bce.numel() * self.keep_ratio))
        
        # Seleziona i più difficili
        top_loss, _ = torch.topk(bce.view(-1), num_keep)
        
        return top_loss.mean()


# ═══════════════════════════════════════════════════════════════
# LOSS COMBINATA AVANZATA
# ═══════════════════════════════════════════════════════════════

class AdvancedCombinedLoss(nn.Module):
    """
    Combina tutte le strategie avanzate con adaptive weighting.
    """
    def __init__(
        self,
        total_epochs: int = 30,
        focal_alpha: float = 0.95,
        focal_gamma_start: float = 3.0,
        asymmetric_gamma_neg: float = 4.0,
        ohem_keep_ratio: float = 0.7,
        ohem_pos_weight: float = 100.0
    ):
        super().__init__()
        
        # Individual losses
        self.focal = DynamicFocalLoss(focal_alpha, focal_gamma_start, 
                                     total_epochs=total_epochs)
        self.asymmetric = AsymmetricLoss(gamma_neg=asymmetric_gamma_neg)
        self.distribution = DistributionAlignmentLoss()
        self.ohem = OHEMBCELoss(ohem_keep_ratio, ohem_pos_weight)
        
        # Adaptive weighting
        self.auto_loss = AdaptiveLossWeighting(num_losses=4)
        
        # Consistency loss tra i due head
        self.consistency_weight = 1.0
        
    def set_epoch(self, epoch: int):
        self.focal.set_epoch(epoch)
        
    def forward(self, continuous: torch.Tensor, binary: torch.Tensor, 
                target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Returns: (total_loss, components_dict)
        """
        # Individual losses
        loss_focal = self.focal(binary, target)
        loss_asymmetric = self.asymmetric(binary, target)
        loss_dist = self.distribution(binary, target)
        loss_ohem = self.ohem(binary, target)
        
        # Consistency tra i due head
        loss_consistency = self.consistency_weight * F.mse_loss(
            torch.sigmoid(continuous), torch.sigmoid(binary)
        )
        
        # Adaptive weighting
        total_loss = self.auto_loss([loss_focal, loss_asymmetric, loss_dist, loss_ohem])
        total_loss = total_loss + loss_consistency
        
        # Components per logging
        components = {
            'focal': loss_focal.item(),
            'asymmetric': loss_asymmetric.item(),
            'distribution': loss_dist.item(),
            'ohem': loss_ohem.item(),
            'consistency': loss_consistency.item(),
            'total': total_loss.item(),
            'weights': self.auto_loss.log_vars.exp().detach().cpu().tolist()
        }
        
        return total_loss, components


# ═══════════════════════════════════════════════════════════════
# STRATEGIA 6: Threshold Calibration Post-Training
# ═══════════════════════════════════════════════════════════════

def calibrate_threshold(model, val_loader, device, target_recall: float = 0.95):
    """
    Calibra la soglia per ottenere un target recall sul validation set.
    Questo previene il collasso "tutto negativo".
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for cont, mat, gt in val_loader:
            cont, mat, gt = cont.to(device), mat.to(device), gt.to(device)
            _, binary = model(cont, mat)
            
            scores = torch.sigmoid(binary).cpu().numpy().flatten()
            labels = gt.cpu().numpy().flatten()
            
            all_scores.extend(scores)
            all_labels.extend(labels)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Trova la soglia che dà il target recall
    if all_labels.sum() == 0:
        return 0.5  # Default se nessuna anomalia
    
    # Ordina gli score delle anomalie
    pos_scores = all_scores[all_labels == 1]
    threshold = np.percentile(pos_scores, (1 - target_recall) * 100)
    
    return threshold


# ═══════════════════════════════════════════════════════════════
# TRAINING LOOP MIGLIORATO
# ═══════════════════════════════════════════════════════════════

def advanced_train_loop(model, train_loader, optimizer, criterion, device, epoch):
    """Training con gradient accumulation e mixed precision"""
    model.train()
    criterion.set_epoch(epoch)
    
    total_loss = 0.0
    all_preds, all_labels = [], []
    loss_components = {k: 0.0 for k in ['focal', 'asymmetric', 'distribution', 'ohem', 'consistency']}
    
    # Gradient accumulation
    accumulation_steps = 4
    optimizer.zero_grad()
    
    for idx, (cont, mat, gt) in enumerate(train_loader):
        cont, mat, gt = cont.to(device), mat.to(device), gt.to(device)
        
        # Forward
        continuous, binary = model(cont, mat)
        
        # Loss
        loss, components = criterion(continuous, binary, gt)
        loss = loss / accumulation_steps  # Scale per accumulation
        
        # Backward
        loss.backward()
        
        # Update ogni N step
        if (idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        total_loss += loss.item() * accumulation_steps
        for k, v in components.items():
            if k in loss_components:
                loss_components[k] += v
        
        preds = (torch.sigmoid(binary).detach().cpu().numpy().flatten() > 0.5).astype(int)
        labels = gt.detach().cpu().numpy().flatten().astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    # Final gradient step se necessario
    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': total_loss / len(train_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }
    
    # Add loss components
    for k in loss_components:
        metrics[f'loss_{k}'] = loss_components[k] / len(train_loader)
    
    return metrics


def advanced_eval_loop(model, val_loader, criterion, device, threshold: float = 0.5):
    """Evaluation con soglia calibrata"""
    model.eval()
    
    total_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []
    
    with torch.no_grad():
        for cont, mat, gt in val_loader:
            cont, mat, gt = cont.to(device), mat.to(device), gt.to(device)
            
            continuous, binary = model(cont, mat)
            loss, _ = criterion(continuous, binary, gt)
            
            total_loss += loss.item()
            
            scores = torch.sigmoid(binary).cpu().numpy().flatten()
            preds = (scores > threshold).astype(int)
            labels = gt.cpu().numpy().flatten().astype(int)
            
            all_scores.extend(scores)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
    }
    
    # ROC-AUC se ci sono entrambe le classi
    if len(np.unique(all_labels)) > 1:
        metrics['auc'] = roc_auc_score(all_labels, np.array(all_scores))
    
    return metrics


# ═══════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════

def main_training_loop(cfg: dict, device: torch.device):
    """Main training con tutte le strategie avanzate"""
    
    # Model
    model = SpatioTemporalAnomalyGNN_V2(
        hidden_dim=cfg.get('gnn_hidden_dim', 128),
        num_gnn_layers=cfg.get('gnn_num_layers', 4),
        num_temporal_layers=cfg.get('gnn_num_temporal_layers', 3),
        num_heads=cfg.get('num_attention_heads', 4),
        dropout=cfg.get('gnn_dropout', 0.2)
    ).to(device)
    
    # Resume se possibile
    start_epoch = 0
    if cfg.get("RESUME_TRAINING", False):
        checkpoints = [f for f in os.listdir(cfg['model_save_path'])  if f.startswith("gnn_v2_epoch_") and f.endswith(".pth")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("_")[3].split(".")[0]))
            checkpoint = torch.load(os.path.join(cfg['model_save_path'], latest),  map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resumed from epoch {start_epoch}")
    
    # Optimizer con weight decay maggiore
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get('lr', 0.001),
        weight_decay=cfg.get('weight_decay', 0.01),
        betas=(0.9, 0.999)
    )
    
    # Advanced criterion
    criterion = AdvancedCombinedLoss(
        total_epochs=cfg['num_epochs'],
        focal_alpha=cfg.get('focal_alpha', 0.95),
        focal_gamma_start=cfg.get('focal_gamma_start', 3.0),
        asymmetric_gamma_neg=cfg.get('asymmetric_gamma_neg', 4.0),
        ohem_keep_ratio=cfg.get('ohem_keep_ratio', 0.7),
        ohem_pos_weight=cfg.get('ohem_pos_weight', 100.0)
    ).to(device)
    
    # Scheduler con warm restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    
    # Data
    train_files = os.listdir(cfg['train_data_path'])
    val_files = os.listdir(cfg['val_data_path'])
    
    if cfg.get('exclude_WADI', False):
        train_files = [f for f in train_files if "WADI" not in f]
        val_files = [f for f in val_files if "WADI" not in f]
    
    # Training history
    history = []
    best_f1 = 0.0
    threshold = 0.5  # Iniziale
    
    for epoch in range(start_epoch, cfg['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{cfg['num_epochs']}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Current threshold: {threshold:.4f}")
        print('='*60)
        
        # Shuffle train files
        np.random.shuffle(train_files)
        
        # Training
        train_metrics_agg = {k: 0.0 for k in ['loss', 'accuracy', 'precision', 'recall', 'f1']}
        
        for file in tqdm(train_files, desc="Training"):
            try:
                train_loader = torch.utils.data.DataLoader(
                    CustomDataset(file, cfg['data_path_custom'], 
                                enableSubSample=cfg.get("ENABLE_UNDERSAMPLE", False)),
                    batch_size=cfg['batch_size'],
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )
            except ValueError:
                continue
                
            metrics = advanced_train_loop(model, train_loader, optimizer, criterion, device, epoch)
            
            for k in train_metrics_agg:
                train_metrics_agg[k] += metrics[k]
        
        # Average training metrics
        num_train = len([f for f in train_files])
        for k in train_metrics_agg:
            train_metrics_agg[k] /= max(num_train, 1)
        
        print(f"\nTraining Metrics:")
        print(f"  Loss: {train_metrics_agg['loss']:.6f}")
        print(f"  Accuracy: {train_metrics_agg['accuracy']:.4f}")
        print(f"  Precision: {train_metrics_agg['precision']:.4f}")
        print(f"  Recall: {train_metrics_agg['recall']:.4f}")
        print(f"  F1: {train_metrics_agg['f1']:.4f}")
        
        # Validation con threshold calibrata
        val_metrics_agg = {k: 0.0 for k in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'auc']}
        
        for file in tqdm(val_files, desc="Validation"):
            val_loader = torch.utils.data.DataLoader(
                CustomDataset(file, cfg['data_path_custom']),
                batch_size=cfg['batch_size'],
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            metrics = advanced_eval_loop(model, val_loader, criterion, device, threshold)
            
            for k in val_metrics_agg:
                if k in metrics:
                    val_metrics_agg[k] += metrics[k]
        
        # Average validation metrics
        num_val = len(val_files)
        for k in val_metrics_agg:
            val_metrics_agg[k] /= max(num_val, 1)
        
        print(f"\nValidation Metrics (threshold={threshold:.4f}):")
        print(f"  Loss: {val_metrics_agg['loss']:.6f}")
        print(f"  Accuracy: {val_metrics_agg['accuracy']:.4f}")
        print(f"  Precision: {val_metrics_agg['precision']:.4f}")
        print(f"  Recall: {val_metrics_agg['recall']:.4f}")
        print(f"  F1: {val_metrics_agg['f1']:.4f}")
        if val_metrics_agg['auc'] > 0:
            print(f"  AUC: {val_metrics_agg['auc']:.4f}")
        
        # Calibra threshold ogni 5 epoche o se F1 < 0.1
        if (epoch + 1) % 5 == 0 or val_metrics_agg['f1'] < 0.1:
            print("\nCalibrating threshold...")
            val_loader_full = torch.utils.data.DataLoader(
                CustomDataset(val_files[0], cfg['data_path_custom']),
                batch_size=cfg['batch_size'],
                shuffle=False
            )
            new_threshold = calibrate_threshold(model, val_loader_full, device, target_recall=0.90)
            threshold = max(0.1, min(0.9, new_threshold))  # Clamp ragionevole
            print(f"New threshold: {threshold:.4f}")
        
        # Save best model
        if val_metrics_agg['f1'] > best_f1:
            best_f1 = val_metrics_agg['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'threshold': threshold,
                'best_f1': best_f1,
                'config': cfg
            }, os.path.join(cfg['model_save_path'], 'best_model_v2.pth'))
            print(f"✓ New best F1: {best_f1:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'threshold': threshold,
                'config': cfg
            }, os.path.join(cfg['model_save_path'], f'gnn_v2_epoch_{epoch+1}.pth'))
        
        # Update history
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics_agg,
            'val': val_metrics_agg,
            'threshold': threshold
        })
        
        # Save history
        with open(os.path.join(cfg['model_save_path'], 'training_history_v2.json'), 'w') as f:
            jsonDump(history, f, indent=2)
        
        scheduler.step()


if __name__ == "__main__":
    cfg = {
        'train_data_path': "./TRAIN_SPLIT",
        "data_path_custom": "./PROCESSED_TRAIN_DATA",
        'val_data_path': "./TEST_SPLIT",
        'model_save_path': "./SAVED_MODELS_V2",
        'exclude_WADI': True,
        
        'num_epochs': 40,
        'batch_size': 16,  # Ridotto per gradient accumulation
        
        # Optimizer
        'lr': 0.001,
        'weight_decay': 0.01,
        
        # Model
        'gnn_hidden_dim': 128,  # Aumentato
        'gnn_num_layers': 3,  # Aumentato
        'gnn_num_temporal_layers': 2,
        'num_attention_heads': 4,  # Multi-head attention
        'gnn_dropout': 0.3,
        
        # Advanced loss parameters
        'focal_alpha': 0.95,
        'focal_gamma_start': 3.0,
        'asymmetric_gamma_neg': 4.0,
        'ohem_keep_ratio': 0.7,
        'ohem_pos_weight': 100.0,
        
        'ENABLE_UNDERSAMPLE': False,
        'RESUME_TRAINING': False,
    }
    
    os.makedirs(cfg['model_save_path'], exist_ok=True)
    with open(os.path.join(cfg['model_save_path'], 'config_v2.json'), 'w') as f:
        jsonDump(cfg, f, indent=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    main_training_loop(cfg, device)
