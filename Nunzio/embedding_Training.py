import torch
import os 
import pandas as pd
import numpy as np


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from json import load as jsonLoad, dump as jsonDump


if torch.cuda.is_available():
    torch.cuda.empty_cache()


class embeddingLoading(torch.utils.data.Dataset):
    def __init__(self, embeddingFile: str, embeddingDir:str, enableBalancing:bool=False):
        self.embeddings = np.load(os.path.join(embeddingDir, "embeddings", embeddingFile), allow_pickle=True)
        self.gt = pd.read_csv(os.path.join(embeddingDir, "ground_truth_labels", embeddingFile.replace("embeddings.npy", "ground_truth_labels.csv")), index_col=None, header=None).iloc[:, 0].values.astype(int).T
        self.predictions = pd.read_csv(os.path.join(embeddingDir, "predictions", embeddingFile.replace("embeddings.npy", "predictions.csv")), index_col=None).iloc[:, 0].astype(int)
        self.groups = {item-1:self.predictions[self.predictions == item].index.tolist() for item in self.predictions.unique()}
        self.randomMapping = np.random.permutation(list(self.groups.keys()))
        self.gt = {idx:int(self.gt[self.groups[idx]].sum() >= 1) for idx in self.groups.keys()} # se almeno un'anomalia è presente nel gruppo, il gruppo è anomalo (1), altrimenti normale (0)

        if enableBalancing:
            normal, anomalies = [idx for idx in self.groups.keys() if self.gt[idx] == 0], [idx for idx in self.groups.keys() if self.gt[idx] == 1]
            minimum = min(len(normal), len(anomalies))
            balancedIndices = np.random.choice(normal, minimum, replace=False).tolist() + np.random.choice(anomalies, minimum, replace=False).tolist()
            self.randomMapping = np.random.permutation(balancedIndices)

    def __len__(self):
        return len(self.randomMapping)

    def __getitem__(self, idx):
            idx = self.randomMapping[idx]
            return torch.tensor(self.embeddings[idx, ...], dtype=torch.float32), torch.tensor([1-self.gt[idx], self.gt[idx]], dtype=torch.int16)


class customNetwork(torch.nn.Module):
    def __init__(self, hiddenDim:int, numLayersToken:int, numLayerData:int, dropout:float=0.30, classifier:bool=False):
        super().__init__()

        if numLayerData < 1 or numLayerData > 6:
            raise ValueError("numLayerData must be between 1 and 7")

        self.__tokenBranch(hiddenDim, numLayersToken, dropout)

        self.numLayersData = numLayerData
        self.__dataBranch(hiddenDim, numLayerData, dropout)

        self.__jointBranch(hiddenDim, dropout)

        if classifier:
            self.classificationHead = torch.nn.Sequential(
                torch.nn.Linear(hiddenDim, hiddenDim//2),
                torch.nn.LayerNorm(hiddenDim//2),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hiddenDim//2, 2),
            )
        else:
            self.classificationHead = torch.nn.Identity()


    # Token branch: INPUT DIMENSION: Nx1x2*768  --> Nx1x2x32x24 --> Nx2x32x24
    def __tokenBranch(self, hiddenDim:int, numLayers:int, dropout:float):
        # INPUT DIMENSION: Nx1x2*768  --> Nx1x2x32x24 --> Nx2x32x24
        self.tokenBranchInit = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=hiddenDim, kernel_size=3), # NxhiddenDimx30x22
            torch.nn.BatchNorm2d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )

        self.sequentialLayersToken = torch.nn.ModuleList()
        for i in range(numLayers):
            self.sequentialLayersToken.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3, padding=1), # NxhiddenDimx30x22
                torch.nn.BatchNorm2d(hiddenDim),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout / (i+1))
            ))

        self.tokenBranchFinal = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3), # NxhiddenDimx28x20
            torch.nn.BatchNorm2d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )

    def __forwardTokenBranch(self, x:torch.Tensor):
        # x: Nx2x32x24 --> NxhiddenDimx28x20
        x = self.tokenBranchInit(x)
        for layer in self.sequentialLayersToken:
            x = layer(x) + x
        return self.tokenBranchFinal(x)

    # Data branch: INPUT DIMENSION: Nx1x768*W
    def __dataBranch(self, hiddenDim:int, numLayers:int, dropout:float):
        # INPUT DIMENSION: Nx1x768*W
        self.dataBranchInitial = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=hiddenDim, kernel_size=2, stride=2), # NxhiddenDimx((768*W-2) / 2 +1) = NxhiddenDimx384*W
            torch.nn.BatchNorm1d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )

        self.sequentialLayers = torch.nn.ModuleList()
        for i in range(numLayers):
            self.sequentialLayers.append(torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=2, stride=2), # NxhiddenDimx((384*W-2) / 2 +1) = NxhiddenDimx192*W
                torch.nn.BatchNorm1d(hiddenDim),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout / (i+1))
            ))

        self.dataBranchFinal = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=1),
            torch.nn.BatchNorm1d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )

    def __forwardDataBranch(self, x:torch.Tensor):
        # x: Nx1xWx768 --> NxhiddenDimx384//2**numLayersData 
        N, C, W, E = x.shape

        x = torch.reshape(x, (N, C, W*E)) # Nx1x768*W

        x = self.dataBranchInitial(x)
        x2 = x.clone() # skip connection 

        N, H, _ = x.shape

        for layer in self.sequentialLayers:
            x = layer(x)
        
        xPool = x2.reshape((*x.shape, -1)).mean(dim=-1) # NxhiddenDimxL//(2**numLayersData)
        
        x = self.dataBranchFinal(x + xPool)
        return torch.reshape(x, (N, H, E//(2**(self.numLayersData+1)), -1)).mean(dim=-1) # NxhiddenDimxE//(2**(self.numLayersData+1))

    # Joint Branch: INPUT DIMENSION: NxhiddenDimx28x20 + NxhiddenDimxE//(2**(self.numLayersData+1)) --> NxhiddenDim
    def __jointBranch(self, hiddenDim:int, dropout:float):
        self.JConv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=4, dilation=3), # NxhiddenDimx((28*20+E//(2**(self.numLayersData+1))-3) / 2 +1) = NxhiddenDimx558
            torch.nn.BatchNorm1d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )

        self.JConv2_1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3, padding=1),
            torch.nn.BatchNorm1d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )

        self.JConv2_2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=5, padding=6, dilation=3),
            torch.nn.BatchNorm1d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )


        self.JConv3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=1),
            torch.nn.BatchNorm1d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )


    def __forwardJointBranch(self, tokenFeatures:torch.Tensor, dataFeatures:torch.Tensor):
        # tokenFeatures: NxhiddenDimx28x20 --> NxhiddenDimx560
        # dataFeatures: NxhiddenDimxE//(2**(self.numLayersData+1)) --> NxhiddenDimxE//(2**(self.numLayersData+1))
        N, H, W, L = tokenFeatures.shape
        tokenFeatures = torch.reshape(tokenFeatures, (N, H, -1)) # NxhiddenDimx560

        x = self.JConv1(torch.cat((tokenFeatures, dataFeatures), dim=-1)) # NxhiddenDimx(560+E//(2**(self.numLayersData+1))-3*(4-1)-1) / 2 +1) = NxhiddenDimxY

        x1 = self.JConv2_2(x) # NxhiddenDimxY
        x2 = self.JConv2_1(x) # NxhiddenDimxY

        x = self.JConv3(x + x1 + x2) # NxhiddenDimxY 

        return x.mean(dim=-1) # NxhiddenDim
    

    # final forward function
    def forward(self, x: torch.Tensor):
        # TOKEN BRANCH: Nx1x(1+W+1)x768 --> NxhiddenDimx28x20
        if len(x.shape) == 3:
            x = x.unsqueeze(1) # Nx1x(1+W+1)x768
        
        tokens = self.__forwardTokenBranch(x[:, 0, [0, -1], :].reshape((x.shape[0], 2, 32, -1))) # NxhiddenDimx28x20
        data = self.__forwardDataBranch(x[:, :, 1:-1, :]) # NxhiddenDimxE//(2**(self.numLayersData+1))
        joint = self.__forwardJointBranch(tokens, data) # NxhiddenDim 

        return self.classificationHead(joint)


class onlyFirstLastTokensNetwork(torch.nn.Module):
    def __init__(self, hiddenDim:int=64, numInternalLayers:int=4, dropout:float=0.30, classifier:bool=False):
        super().__init__()
        # The network will process files of the kind Nx2x768, where the 2 tokens are the first and the last of the sequence, and 768 is the embedding dimension.
        # Recall the reshape of the vector Nx2x768 -> Nx2x32x24

        if numInternalLayers < 0 or numInternalLayers > 6:
            raise ValueError("numInternalLayers must be between 0 and 6")

        # Lets start with an initial DSConv so we don't mix branch information too early
        self.initialConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, groups=2, padding=1, stride=2), # Nx2x32x24 -> Nx2x16x12
            torch.nn.BatchNorm2d(2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv2d(in_channels=2, out_channels=hiddenDim//2, kernel_size=3, padding=1), # Nx2x16x12 -> NxhiddenDim//2x16x12
            torch.nn.BatchNorm2d(hiddenDim//2),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv2d(in_channels=hiddenDim//2, out_channels=hiddenDim, kernel_size=3, padding=1), # NxhiddenDim//2x16x12 -> NxhiddenDimx16x12
            torch.nn.BatchNorm2d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )

        self.sequentialLayers = torch.nn.ModuleList()
        for i in range(numInternalLayers):
            self.sequentialLayers.append(torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3, padding=1), # NxhiddenDimx16x12
                torch.nn.BatchNorm2d(hiddenDim),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout / (i+1))
            ))

        self.finalConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3, stride=2, padding=1), # NxhiddenDimx8x6
            torch.nn.BatchNorm2d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv2d(in_channels=hiddenDim, out_channels=hiddenDim, kernel_size=3, stride=2, padding=1), # NxhiddenDimx4x3
            torch.nn.BatchNorm2d(hiddenDim),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        if classifier:
            self.classificationHead = torch.nn.Sequential(
                torch.nn.Linear(hiddenDim, hiddenDim//2),
                torch.nn.LayerNorm(hiddenDim//2),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hiddenDim//2, 2),
            )
        else:
            self.classificationHead = torch.nn.Identity()

    def forward(self, x: torch.Tensor):
        # x: Nx1x(1+W+1)x768 --> Nx2x32x24
        if len(x.shape) == 3:
            x = x.unsqueeze(1) # Nx1x(1+W+1)x768

        x = x[:, 0, [0, -1], :].reshape((x.shape[0], 2, 32, -1)) # Nx2x32x24
        
        x = self.initialConv(x)
        for layer in self.sequentialLayers:
            x = layer(x) + x

        x = self.finalConv(x)

        x = self.gap(x) # NxhiddenDimx1x1

        return self.classificationHead(x.view(x.size(0), -1)) # NxhiddenDim



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
    start_factor=0.1,  
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



class customLoss(torch.nn.Module):
    """
    BCE con peso esplicito per la classe positiva.
    Semplice ed efficace per imbalance noti.
    
    Args:
        pos_weight: peso per classe positiva. Se anomalie=1%, usa ~99 o calcola dinamicamente
    """
    def __init__(self, pos_weight: list = [1, 20.0], diceWeight: float = 1.0, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.pos_weight = pos_weight
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(pos_weight, device=device))
        self.diceWeight = diceWeight
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: Nx2 (logits), target: Nx2 (one-hot)
        loss = self.loss(pred, target.argmax(dim=-1))
        if self.diceWeight > 0:
            dice_loss = self.computeDiceLoss(pred, target)
            return loss + dice_loss * self.diceWeight
        return loss

    def computeDiceLoss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: Nx2 (logits), target: Nx2 (one-hot)
        pred_probs = torch.softmax(pred, dim=-1)[:, 1]  # Probabilità della classe positiva
        target_pos = target[:, 1]  # Ground truth per la classe positiva

        return 1 - (2. * (pred_probs * target_pos).sum() + 1e-6) / (pred_probs.sum() + target_pos.sum() + 1e-6)
    

class FocalLoss(torch.nn.Module):
    """Focal Loss per classificazione binaria.
    
    Args:
        alpha: peso per la classe positiva (default 0.95)
        gamma: fattore di focalizzazione (default 2.0)
    """
    def __init__(self, alpha: float = 0.95, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [N, 2] logits
        target: [N, 2] one-hot
        """
        # Convert target to class indices if one-hot
        if target.dim() == 2 and target.size(1) == 2:
            target_idx = target.argmax(dim=-1)
        else:
            target_idx = target.long()
        
        # Cross entropy
        ce = torch.nn.functional.cross_entropy(pred, target_idx, reduction='none')
        
        # p_t: probabilità della classe vera
        p = torch.exp(-ce)
        
        # Focal weight: (1-p)^gamma
        focal_weight = (1 - p) ** self.gamma
        
        # Alpha weighting
        # Se anomalia (class=1) → peso = alpha
        # Se normale (class=0) → peso = 1-alpha
        alpha_t = torch.where(target_idx == 1, torch.tensor(self.alpha, device=pred.device),torch.tensor(1-self.alpha, device=pred.device))
        
        return (alpha_t * focal_weight * ce).mean()
    
class AnomalyAugmentation:
    """
    Augmentation specifico per anomalie - oversampling + noise.
    """
    def __init__(self, oversample_ratio: float = 2.0, noise_std: float = 0.05):
        self.oversample_ratio = oversample_ratio
        self.noise_std = noise_std
    
    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        embeddings: [N, 1, W, 768]
        labels: [N, 2] one-hot
        """
        # Trova anomalie
        anomaly_mask = labels[:, 1] == 1
        
        if anomaly_mask.sum() == 0:
            return embeddings, labels
        
        anomaly_embeds = embeddings[anomaly_mask]
        anomaly_labels = labels[anomaly_mask]
        
        # Oversample
        n_oversample = int(len(anomaly_embeds) * (self.oversample_ratio - 1))
        if n_oversample > 0:
            indices = torch.randint(0, len(anomaly_embeds), (n_oversample,))
            oversampled_embeds = anomaly_embeds[indices]
            
            # Concatenate
            embeddings = torch.cat([embeddings, oversampled_embeds + torch.randn_like(oversampled_embeds) * self.noise_std], dim=0)
            labels = torch.cat([labels, anomaly_labels[indices]], dim=0)
        
        return embeddings, labels
    


def trainLoop(model: customNetwork, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, device: torch.device):
    """Training loop per un singolo DataLoader (un file)."""
    model.train()

    total_loss = 0.0
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    for data, labels in train_loader:
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        cls = model(data)

        loss = criterion(cls, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)  # Gradient clipping
        optimizer.step()


        total_loss += loss.item()

        outputs_flat = torch.argmax(cls, dim=-1).detach().cpu().numpy().flatten().astype(int)
        gt_flat = labels.argmax(dim=-1).detach().cpu().numpy().flatten().astype(int)

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
def evalLoop(model: customNetwork, eval_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module, device: torch.device):
    """Validation loop per un singolo DataLoader (un file)."""
    model.eval()

    total_loss = 0.0
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0

    for data, labels in eval_loader:
        data, labels = data.to(device), labels.to(device)

        cls = model(data)
        scores = torch.argmax(cls, dim=-1)  # Nx1

        loss = criterion(cls.to(device), labels.float())
        total_loss += loss.item()

        outputs_flat = scores.cpu().numpy().flatten().astype(int)
        gt_flat = labels.argmax(dim=-1).cpu().numpy().flatten().astype(int)

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
    if cfg.get("simplerModel", False):
        model = onlyFirstLastTokensNetwork(
            hiddenDim=cfg['hidden_dim'],
            numInternalLayers=cfg['numInternalLayers'] if 'numInternalLayers' in cfg else 4,
            dropout=cfg['dropout'],
            classifier=True
        ).to(device)
    else:
        model = customNetwork(
            hiddenDim=cfg['hidden_dim'],
            numLayersToken=cfg['numLayersToken'] if 'numLayersToken' in cfg else cfg['num_layers'],
            numLayerData=cfg['numLayerData'] if 'numLayerData' in cfg else cfg['num_layers'],
            dropout=cfg['dropout'],
            classifier=True
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
    criterion = FocalLoss()
    # ── Scheduler (Warmup + Cosine) ──
    scheduler = getScheduler(optimizer, cfg['num_epochs'], cfg.get('warmup_fraction', 0.1))

    # ── File list ──
    trainFiles, valFiles = os.listdir(cfg['train_data_path']), os.listdir(cfg['val_data_path'])
    
    # ── Training loop ──
    for epoch in range(startingEpoch, cfg['num_epochs']):
        print(f"\nEpoch {epoch+1}/{cfg['num_epochs']}  (LR: {optimizer.param_groups[0]['lr']:.6f})")
        np.random.shuffle(trainFiles)

        # ── Train ──
        tLoss, tAcc, tPrec, tRec, tF1 = 0.0, 0.0, 0.0, 0.0, 0.0; nTrain = 0
        for file in tqdm(trainFiles, desc="Training Files", leave=False):
            try:
                loader = torch.utils.data.DataLoader(
                    embeddingLoading(file, cfg['processed_data_dir'], enableBalancing=cfg.get("enable_balancing", False)),
                    batch_size=cfg.get('batch_size', 32),
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
                    embeddingLoading(file, cfg['processed_data_dir']),
                    batch_size=cfg.get('batch_size', 32),
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
        cfg = {
            # ── Data ──
            'train_data_path': './TRAIN_SPLIT_UNIVARIATE/',
            'val_data_path': './TEST_SPLIT_UNIVARIATE/',
            'processed_data_dir': './PROCESSED_TRAIN_DATAV3/',
            'model_save_path': './Saved_Models_Simpler/',

            # ── Model ──
            'hidden_dim': 32,
            "numLayersToken": 1,
            "numLayerData": 1,
            "dropout": 0.30,

            "numInternalLayers": 2, # solo per simplerModel
            "simplerModel": True, # se True usa onlyFirstLastTokensNetwork, altrimenti customNetwork

            # ── Training ──
            'num_epochs': 60,
            'batch_size': 64,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'warmup_fraction': 0.20,
            'num_workers': 0,

            
            # ── Misc ──
            'RESUME_TRAINING': False,
            "exclude_names": [], 
            "restrict_also_validation": True,
            "enable_balancing": False
        }

        cfg = {**cfg,
            'pos_weight': [1, 1.0] if cfg.get("enable_balancing", False) else [1, 21.0],
            }

        os.makedirs(cfg['model_save_path'], exist_ok=True)
        with open(os.path.join(cfg['model_save_path'], "final_config.json"), 'w') as f:
            jsonDump(cfg, f, indent=4)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        mainLoop(cfg, device)

