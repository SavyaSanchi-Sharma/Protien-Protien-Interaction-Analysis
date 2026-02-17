import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve
from datetime import datetime
import pandas as pd
import numpy as np

from createdatset import PPISDataset, SCALAR_COLS, STRUCT_COLS, NORM_PATH
from model.esm_projection import ESMProjection
from model.fusion import GatedFusion
from model.gcn import GCNEncoder
from model.tcn import BiTCN
from model.classifier import Classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 100
BATCH_SIZE = 1
LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
PATIENCE = 15

TRAIN_CSV = "data/features/Train_335_36D.csv"
VAL_CSV   = "data/features/Test_315_36D.csv"
ESM_DIR   = "data/esm"
PDB_DIR   = "data/pdbs"
CKPT_DIR  = "checkpoints"

os.makedirs(CKPT_DIR, exist_ok=True)

scaler = GradScaler("cuda")

def log_info(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def ensure_normalization(train_csv_path, output_path=NORM_PATH):
    if os.path.exists(output_path):
        log_info(f"✓ Normalization file already exists: {output_path}")
        return
    
    log_info(f"[NORM] Computing normalization from TRAIN: {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    
    scalar_mean = df[SCALAR_COLS].mean().values
    scalar_std = df[SCALAR_COLS].std().values
    scalar_std = np.where(scalar_std < 1e-6, 1.0, scalar_std)
    
    log_info(f"[NORM] Scalar feature means:\n{scalar_mean}")
    log_info(f"[NORM] Scalar feature stds:\n{scalar_std}")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, mean=scalar_mean, std=scalar_std)
    log_info(f"✓ Saved normalization to {output_path}\n")

def compute_dataset_alpha(dataset):
    pos, neg = 0, 0
    for i in range(len(dataset)):
        y = dataset[i].y
        pos += int((y == 1).sum())
        neg += int((y == 0).sum())
    pos = max(pos, 1)
    N = pos + neg
    log_info(f"Dataset Stats | Total: {N} | Positive: {pos} ({pos/N*100:.1f}%) | Negative: {neg} ({neg/N*100:.1f}%)")
    return neg / pos  # Weight for positive class (ratio, not normalized)

def hybrid_focal_cost_loss(logits, targets, alpha_pos, gamma=2.0):
    """
    Numerically stable focal loss with class weighting for imbalanced data.
    Uses log_softmax + log1p for numerical stability under AMP.
    
    Args:
        logits: [N, 2] raw logits from classifier
        targets: [N] binary labels (0 or 1)
        alpha_pos: weight for positive class (neg/pos ratio)
        gamma: focusing parameter for hard examples
    
    Loss formulation:
        L = -α * (1-p)^γ * log(p) for positives
        L = -p^γ * log(1-p) for negatives
    """
    logp = torch.log_softmax(logits, dim=1)[:, 1]  # log P(class=1)
    p = logp.exp()  # P(class=1), computed from logp for stability

    pos = targets == 1
    neg = targets == 0

    # Focal loss for positives: -alpha * (1-p)^gamma * log(p)
    loss_pos = -alpha_pos * ((1 - p[pos]) ** gamma) * logp[pos]
    
    # Focal loss for negatives: -p^gamma * log(1-p)
    # Use log1p for numerical stability: log(1-p) = log1p(-p)
    loss_neg = -(p[neg] ** gamma) * torch.log1p(-p[neg])
    return (loss_pos.sum() + loss_neg.sum()) / targets.numel()

def find_optimal_threshold(probs, labels):
    best_f1 = 0.0
    best_thresh = 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    best_thresh = thresholds[best_idx]
    best_f1 = f1[best_idx]
   
    # Grid search for optimal threshold
    # for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #     preds = (probs >= thresh).astype(int)
    #     f1 = f1_score(labels, preds, zero_division=0)
    #     if f1 > best_f1:
    #         best_f1 = f1
    #         best_thresh = thresh
    
    log_info(f"[THRESHOLD] Grid search complete: optimal={best_thresh:.2f} → F1={best_f1:.4f}")
    return best_thresh, best_f1

@torch.no_grad()
def evaluate(proj, fusion, gcn, tcn, clf, loader, alpha_pos):
    proj.eval()
    fusion.eval()
    gcn.eval()
    tcn.eval()
    clf.eval()

    losses, probs_all, labels_all = [], [], []

    with autocast(device_type="cuda"):
        for batch_idx, data in enumerate(loader):
            data = data.to(DEVICE)

            esm = data.x[:, :2560]
            struct = data.x[:, 2560:]

            h0 = proj(esm)
            h  = fusion(h0, struct)
            h  = gcn(h, data.edge_index, edge_weight=data.edge_weight)

            seq_lens = torch.bincount(data.batch)
            out, s = [], 0
            for L in seq_lens.tolist():
                out.append(tcn(h[s:s+L].unsqueeze(0)).squeeze(0))
                s += L

            h = torch.cat(out, dim=0)
            logits = clf(h)

            y = data.y.long().view(-1)
            loss = hybrid_focal_cost_loss(logits, y, alpha_pos)

            losses.append(loss.item())
            probs_all.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            labels_all.append(y.cpu())

    probs = torch.cat(probs_all).numpy()
    labels = torch.cat(labels_all).numpy()

    # Threshold-free metrics
    auc = roc_auc_score(labels, probs)
    pr  = average_precision_score(labels, probs)
    
    # Metrics at default threshold (0.5)
    preds_default = (probs >= 0.5).astype(int)
    f1_default = f1_score(labels, preds_default, zero_division=0)
    mcc_default = matthews_corrcoef(labels, preds_default)
    
    # Find optimal threshold for F1 (on validation data)
    optimal_thresh, f1_optimal = find_optimal_threshold(probs, labels)
    preds_optimal = (probs >= optimal_thresh).astype(int)
    mcc_optimal = matthews_corrcoef(labels, preds_optimal)

    return (
        sum(losses) / len(losses), 
        auc, pr, 
        f1_default, mcc_default,
        f1_optimal, mcc_optimal, optimal_thresh
    )

def main():
    log_info("=" * 80)
    log_info("Starting Training Pipeline")
    log_info(f"Device: {DEVICE} | Epochs: {EPOCHS} | LR: {LR} | Batch Size: {BATCH_SIZE}")
    log_info(f"Loss: Numerically Stable Focal Loss (log_softmax + log1p)")
    log_info("=" * 80)

    # Ensure normalization exists (compute if missing)
    log_info("Checking normalization statistics...")
    ensure_normalization(TRAIN_CSV, NORM_PATH)

    log_info("Loading datasets...")
    train_ds = PPISDataset(TRAIN_CSV, ESM_DIR, PDB_DIR)
    val_ds   = PPISDataset(VAL_CSV,   ESM_DIR, PDB_DIR)
    log_info(f"Train set size: {len(train_ds)} | Val set size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    log_info("Computing class weights...")
    alpha_pos = compute_dataset_alpha(train_ds)
    log_info(f"Class weight for positives: {alpha_pos:.4f}")

    log_info("Initializing models...")
    proj   = ESMProjection(2560, dropout=0.2).to(DEVICE)
    fusion = GatedFusion(256, 36, 256, dropout=0.01).to(DEVICE)
    gcn    = GCNEncoder(256, hidden=256, num_layers=6, alpha=0.2, dropout=0.1).to(DEVICE)
    tcn    = BiTCN(256, channels=[64,128,256,512],  dropout=0.1).to(DEVICE)
    clf    = Classifier(1024, dropout=0.3).to(DEVICE)

    total_params = sum(p.numel() for p in 
        list(proj.parameters()) + list(fusion.parameters()) + 
        list(gcn.parameters()) + list(tcn.parameters()) + list(clf.parameters()))
    log_info(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(
        list(proj.parameters()) +
        list(fusion.parameters()) +
        list(gcn.parameters()) +
        list(tcn.parameters()) +
        list(clf.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    warmup = LambdaLR(optimizer, lambda e: min(1.0, (e + 1) / WARMUP_EPOCHS))
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

    best_pr, best_f1, best_mcc = 0.0, 0.0, -1.0
    best_pr_epoch, best_f1_epoch, best_mcc_epoch = 0, 0, 0
    patience = 0
    log = []

    log_info("=" * 80)
    log_info("Starting Training")
    log_info("=" * 80)

    for epoch in range(EPOCHS):
        proj.train()
        fusion.train()
        gcn.train()
        tcn.train()
        clf.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                esm = data.x[:, :2560]
                struct = data.x[:, 2560:]

                h0 = proj(esm)
                h  = fusion(h0, struct)
                h  = gcn(h, data.edge_index, edge_weight=data.edge_weight)

                seq_lens = torch.bincount(data.batch)
                out, s = [], 0
                for L in seq_lens.tolist():
                    out.append(tcn(h[s:s+L].unsqueeze(0)).squeeze(0))
                    s += L

                h = torch.cat(out, dim=0)
                logits = clf(h)
                y = data.y.long().view(-1)

                loss = hybrid_focal_cost_loss(logits, y, alpha_pos)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

        train_loss = epoch_loss / len(train_loader)
        (val_loss, val_auc, val_pr, 
         val_f1, val_mcc, 
         val_f1_opt, val_mcc_opt, opt_thresh) = evaluate(
            proj, fusion, gcn, tcn, clf, val_loader, alpha_pos
        )

        if epoch < WARMUP_EPOCHS:
            warmup.step()
            sched_name = "Warmup"
        else:
            cosine.step()
            sched_name = "Cosine"

        current_lr = optimizer.param_groups[0]['lr']
        log_info(f"[Epoch {epoch+1}/{EPOCHS}] LR={current_lr:.2e} ({sched_name}) | "
                f"Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")
        log_info(f"  → AUC={val_auc:.4f} | PR={val_pr:.4f} | "
                f"F1(0.5)={val_f1:.4f} | MCC(0.5)={val_mcc:.4f}")
        log_info(f"  → F1(opt@{opt_thresh:.2f})={val_f1_opt:.4f} | MCC(opt)={val_mcc_opt:.4f} | "
                f"Patience={patience}/{PATIENCE}")
        log.append({
            "epoch": int(epoch + 1),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auc": float(val_auc),
            "val_pr": float(val_pr),
            "val_f1_default": float(val_f1),
            "val_mcc_default": float(val_mcc),
            "val_f1_optimal": float(val_f1_opt),
            "val_mcc_optimal": float(val_mcc_opt),
            "optimal_threshold": float(opt_thresh),
            "lr": float(current_lr)
        })


        with open("training_log.json", "w") as f:
            json.dump(log, f, indent=2)

        # Multi-metric early stopping: PR, F1, MCC
        improved = False
        if val_pr > best_pr:
            best_pr = val_pr
            best_pr_epoch = epoch + 1
            improved = True
            log_info(f"  ✓ New best PR: {best_pr:.4f} @ epoch {best_pr_epoch}")
        
        if val_f1_opt > best_f1:
            best_f1 = val_f1_opt
            best_f1_epoch = epoch + 1
            improved = True
            log_info(f"  ✓ New best F1: {best_f1:.4f} @ epoch {best_f1_epoch}")
        
        if val_mcc_opt > best_mcc:
            best_mcc = val_mcc_opt
            best_mcc_epoch = epoch + 1
            improved = True
            log_info(f"  ✓ New best MCC: {best_mcc:.4f} @ epoch {best_mcc_epoch}")

        if improved:
            patience = 0
            log_info(f"  → Saving checkpoint (epoch {epoch + 1})...")
            torch.save({
                "proj": proj.state_dict(),
                "fusion": fusion.state_dict(),
                "gcn": gcn.state_dict(),
                "tcn": tcn.state_dict(),
                "clf": clf.state_dict(),
                "epoch": epoch + 1,
                "best_pr": best_pr,
                "best_f1": best_f1,
                "best_mcc": best_mcc,
                "optimal_threshold": opt_thresh,
                "val_auc": val_auc,
                "val_pr": val_pr,
                "val_f1_default": val_f1,
                "val_mcc_default": val_mcc,
                "val_f1_optimal": val_f1_opt,
                "val_mcc_optimal": val_mcc_opt
            }, os.path.join(CKPT_DIR, "best.pt"))
        else:
            patience += 1
            if patience >= PATIENCE:
                log_info(f"✗ Early stopping at epoch {epoch+1} - Patience limit ({PATIENCE}) reached")
                break

    log_info("=" * 80)
    log_info(f"Training Complete")
    log_info(f"  Best PR: {best_pr:.4f} @ epoch {best_pr_epoch}")
    log_info(f"  Best F1 (optimized): {best_f1:.4f} @ epoch {best_f1_epoch}")
    log_info(f"  Best MCC (optimized): {best_mcc:.4f} @ epoch {best_mcc_epoch}")
    log_info("=" * 80)

if __name__ == "__main__":
    main()
