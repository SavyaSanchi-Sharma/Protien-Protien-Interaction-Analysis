import os
import sys
import json
import random
import warnings
warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")

from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_recall_curve,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, DATA_ROOT)

from imbalance_optim import Lion
from createdatset import (PPISDataset, SCALAR_COLS, STRUCT_COLS, NORM_PATH,
                          EDGE_ATTR_DIM, NUM_PLM_LAYERS, PLM_DIM)
from model.esm_projection import MultiLayerProjection
from model.fusion import CrossAttentionFusion
from model.gcn import GCNEncoder
from model.tcn import BiTCN
from model.classifier import Classifier

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS        = 100
WARMUP_EPOCHS = 5
PATIENCE      = 15
NUM_WORKERS   = int(os.environ.get("PPI_NUM_WORKERS", "4"))
PIN_MEMORY    = DEVICE == "cuda"

DEFAULT_HP = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "focal_gamma": 2.0,
    "lambda_mcc": 0.5,
    "proj_dropout": 0.2,
    "fusion_dropout": 0.01,
    "gcn_layers": 6,
    "gcn_alpha": 0.2,
    "gcn_dropout": 0.1,
    "tcn_dropout": 0.1,
    "clf_dropout": 0.3,
    "batch_size": 1,
}

TRAIN_FULL_CSV = os.path.join(DATA_ROOT, "data", "structural", "Train_335_17D.csv")
TRAIN_CSV      = os.path.join(DATA_ROOT, "data", "structural", "Train_split_train.csv")
VAL_CSV        = os.path.join(DATA_ROOT, "data", "structural", "Train_split_val.csv")
ESM_DIR        = os.path.join(DATA_ROOT, "data", "esm_multi")
CKPT_DIR       = os.path.join(ROOT, "checkpoints")
HP_PATH        = os.path.join(ROOT, "best_hp.json")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log_info(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def load_hp():
    hp = dict(DEFAULT_HP)
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            tuned = json.load(f).get("params", {})
        hp.update({k: v for k, v in tuned.items() if k in DEFAULT_HP})
        log_info(f"[HP] Loaded tuned hyperparameters from {HP_PATH}")
    return hp


def ensure_train_val_split(full_csv, train_out, val_out, seed=42, val_frac=0.2):
    if os.path.exists(train_out) and os.path.exists(val_out):
        return
    df = pd.read_csv(full_csv)
    proteins = df[["PDB", "Chain"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(proteins))
    n_val = max(1, int(round(len(proteins) * val_frac)))
    val_keys = set(zip(proteins.iloc[perm[:n_val]]["PDB"].tolist(),
                       proteins.iloc[perm[:n_val]]["Chain"].tolist()))
    is_val = np.array([k in val_keys for k in zip(df["PDB"], df["Chain"])])
    df[~is_val].to_csv(train_out, index=False)
    df[is_val].to_csv(val_out, index=False)
    print(f"[SPLIT] {len(proteins) - n_val} train / {n_val} val proteins (seed={seed})")


def ensure_normalization(train_csv_path, output_path=None):
    output_path = output_path or NORM_PATH
    if os.path.exists(output_path):
        log_info(f"[NORM] Using existing normalization: {output_path}")
        return
    df = pd.read_csv(train_csv_path)
    mean = df[SCALAR_COLS].mean().values
    std  = np.where(df[SCALAR_COLS].std().values < 1e-6, 1.0, df[SCALAR_COLS].std().values)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, scalar_mean=mean, scalar_std=std)
    log_info(f"[NORM] Computed and saved → {output_path}")


def compute_dataset_alpha(dataset):
    pos = neg = 0
    for i in range(len(dataset)):
        y = dataset[i].y
        pos += int((y == 1).sum())
        neg += int((y == 0).sum())
    pos = max(pos, 1)
    log_info(f"Dataset Stats | Total: {pos+neg} | Pos: {pos} ({pos/(pos+neg)*100:.1f}%) | Neg: {neg}")
    return neg / pos


def soft_mcc(logits, targets, eps=1e-7):
    p = torch.softmax(logits, dim=1)[:, 1]
    y = targets.float()
    tp = (p * y).sum()
    fn = ((1 - p) * y).sum()
    fp = (p * (1 - y)).sum()
    tn = ((1 - p) * (1 - y)).sum()
    num = tp * tn - fp * fn
    den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    return num / den


def hybrid_focal_cost_loss(logits, targets, alpha_pos, gamma=2.0, lambda_mcc=0.0,
                           return_components=False):
    logp = torch.log_softmax(logits, dim=1)[:, 1]
    p = logp.exp()
    pos = targets == 1
    neg = targets == 0
    loss_pos = -alpha_pos * ((1 - p[pos]) ** gamma) * logp[pos]
    loss_neg = -(p[neg] ** gamma) * torch.log1p(-p[neg])
    focal = (loss_pos.sum() + loss_neg.sum()) / targets.numel()
    if lambda_mcc > 0.0:
        mcc_s = soft_mcc(logits, targets)
        total = focal - lambda_mcc * mcc_s
    else:
        mcc_s = torch.zeros((), device=focal.device, dtype=focal.dtype)
        total = focal
    if return_components:
        return total, focal.detach(), mcc_s.detach()
    return total


def find_optimal_threshold(probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def find_mcc_threshold(probs, labels):
    best_thresh, best_mcc = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probs >= t).astype(int)
        s = int(preds.sum())
        if s == 0 or s == len(preds):
            continue
        m = matthews_corrcoef(labels, preds)
        if m > best_mcc:
            best_mcc = m
            best_thresh = float(t)
    return best_thresh, float(best_mcc)


def forward_step(modules, data):
    proj, fusion, gcn, tcn, clf = modules
    plm_proj = proj(data.emb)
    plm_dense, mask = to_dense_batch(plm_proj, data.batch)
    struct_dense, _ = to_dense_batch(data.x, data.batch)
    fused = fusion(plm_dense, struct_dense, mask=mask)
    h = fused[mask]
    h = gcn(h, data.edge_index, edge_weight=data.edge_weight, edge_attr=data.edge_attr)
    padded, mask2 = to_dense_batch(h, data.batch)
    h = tcn(padded, lengths=mask2.sum(dim=1))[mask2]
    return clf(h), data.y.long().view(-1)


@torch.no_grad()
def evaluate(modules, loader, alpha_pos):
    for m in modules: m.eval()
    losses, probs_all, labels_all = [], [], []
    with autocast(device_type="cuda"):
        for data in loader:
            data = data.to(DEVICE)
            logits, y = forward_step(modules, data)
            losses.append(hybrid_focal_cost_loss(logits, y, alpha_pos).item())
            probs_all.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            labels_all.append(y.cpu())
    probs = torch.cat(probs_all).numpy()
    labels = torch.cat(labels_all).numpy()
    auc = roc_auc_score(labels, probs)
    pr  = average_precision_score(labels, probs)
    preds_default = (probs >= 0.5).astype(int)
    f1_default = f1_score(labels, preds_default, zero_division=0)
    mcc_default = matthews_corrcoef(labels, preds_default)
    _, f1_opt = find_optimal_threshold(probs, labels)
    opt_thresh, mcc_opt = find_mcc_threshold(probs, labels)
    return (sum(losses) / len(losses), auc, pr, f1_default, mcc_default,
            f1_opt, mcc_opt, opt_thresh)


def build_modules(hp):
    return [
        MultiLayerProjection(NUM_PLM_LAYERS, PLM_DIM, 1024, 256,
                             hp["proj_dropout"]).to(DEVICE),
        CrossAttentionFusion(d_esm=256, d_struct=17, d_out=256,
                             n_heads=hp.get("fusion_heads", 4),
                             dropout=hp["fusion_dropout"]).to(DEVICE),
        GCNEncoder(256, 256, hp["gcn_layers"], hp["gcn_alpha"], hp["gcn_dropout"],
                   edge_dim=EDGE_ATTR_DIM).to(DEVICE),
        BiTCN(256, [64, 128, 256, 512], hp["tcn_dropout"]).to(DEVICE),
        Classifier(1024, hp["clf_dropout"]).to(DEVICE),
    ]


def save_checkpoint(modules, info, path):
    proj, fusion, gcn, tcn, clf = modules
    torch.save({
        "proj":   proj.state_dict(),
        "fusion": fusion.state_dict(),
        "gcn":    gcn.state_dict(),
        "tcn":    tcn.state_dict(),
        "clf":    clf.state_dict(),
        **info,
    }, path)


def main():
    set_seed(int(os.environ.get("PPI_SEED", "42")))
    os.makedirs(CKPT_DIR, exist_ok=True)
    scaler = GradScaler("cuda")

    hp = load_hp()
    log_info("=" * 80)
    log_info(f"esm_model: ESM-2 3B multi-layer ({NUM_PLM_LAYERS}×{PLM_DIM}D) + 17D struct | CA fusion | GCN + BiTCN")
    log_info(f"Device: {DEVICE} | Epochs: {EPOCHS} | LR: {hp['lr']} | Batch: {hp['batch_size']}")
    log_info(f"HP: {hp}")
    log_info("=" * 80)

    ensure_train_val_split(TRAIN_FULL_CSV, TRAIN_CSV, VAL_CSV)
    ensure_normalization(TRAIN_CSV)

    log_info("Loading datasets...")
    train_ds = PPISDataset(TRAIN_CSV, ESM_DIR)
    val_ds   = PPISDataset(VAL_CSV,   ESM_DIR)
    log_info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
    )
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False, **loader_kwargs)
    alpha_pos    = compute_dataset_alpha(train_ds)

    modules = build_modules(hp)
    params = list(chain.from_iterable(m.parameters() for m in modules))
    log_info(f"Total parameters: {sum(p.numel() for p in params):,}")

    optimizer = Lion(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    warmup = LambdaLR(optimizer, lambda e: min(1.0, (e + 1) / WARMUP_EPOCHS))
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

    best_pr = best_f1 = 0.0
    best_mcc = -1.0
    best_pr_epoch = best_f1_epoch = best_mcc_epoch = 0
    patience = 0
    log = []

    for epoch in range(EPOCHS):
        for m in modules: m.train()
        epoch_loss = epoch_focal = epoch_mcc_s = 0.0

        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                logits, y = forward_step(modules, data)
                loss, focal_v, mcc_v = hybrid_focal_cost_loss(
                    logits, y, alpha_pos,
                    gamma=hp["focal_gamma"], lambda_mcc=hp["lambda_mcc"],
                    return_components=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss  += loss.item()
            epoch_focal += focal_v.item()
            epoch_mcc_s += mcc_v.item()

        n_batches = max(len(train_loader), 1)
        train_loss  = epoch_loss  / n_batches
        train_focal = epoch_focal / n_batches
        train_mcc_s = epoch_mcc_s / n_batches

        (val_loss, val_auc, val_pr, val_f1, val_mcc,
         val_f1_opt, val_mcc_opt, opt_thresh) = evaluate(modules, val_loader, alpha_pos)

        sched_name = "Warmup" if epoch < WARMUP_EPOCHS else "Cosine"
        (warmup if epoch < WARMUP_EPOCHS else cosine).step()
        current_lr = optimizer.param_groups[0]["lr"]

        log_info(f"[Epoch {epoch+1}/{EPOCHS}] LR={current_lr:.2e} ({sched_name}) | "
                 f"Train={train_loss:+.4f} | Val={val_loss:+.4f}")
        log_info(f"  Loss: focal={train_focal:.4f} | soft_mcc={train_mcc_s:+.4f} | "
                 f"λ·mcc_s={hp['lambda_mcc']*train_mcc_s:+.4f}")
        log_info(f"  AUC={val_auc:.4f} | PR={val_pr:.4f} | F1(0.5)={val_f1:.4f} | "
                 f"F1_max={val_f1_opt:.4f} | MCC_max={val_mcc_opt:.4f}@{opt_thresh:.2f} | "
                 f"Patience={patience}/{PATIENCE}")
        log.append({
            "epoch": epoch + 1,
            "train_loss":  float(train_loss),
            "train_focal": float(train_focal),
            "train_soft_mcc": float(train_mcc_s),
            "val_loss": float(val_loss), "val_auc": float(val_auc),
            "val_pr": float(val_pr), "val_f1_default": float(val_f1),
            "val_mcc_default": float(val_mcc), "val_f1_optimal": float(val_f1_opt),
            "val_mcc_optimal": float(val_mcc_opt),
            "optimal_threshold": float(opt_thresh), "lr": float(current_lr),
        })
        with open(os.path.join(ROOT, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        if val_pr > best_pr:
            best_pr, best_pr_epoch = val_pr, epoch + 1
            log_info(f"  • New best PR: {best_pr:.4f}")
        if val_f1_opt > best_f1:
            best_f1, best_f1_epoch = val_f1_opt, epoch + 1
            log_info(f"  • New best F1: {best_f1:.4f}")

        if val_mcc_opt > best_mcc:
            best_mcc, best_mcc_epoch = val_mcc_opt, epoch + 1
            patience = 0
            log_info(f"  ✓ New best MCC: {best_mcc:.4f}  (saving checkpoint)")
            save_checkpoint(modules, {
                "epoch": epoch + 1,
                "best_pr": best_pr, "best_f1": best_f1, "best_mcc": best_mcc,
                "optimal_threshold": opt_thresh,
            }, os.path.join(CKPT_DIR, "best.pt"))
        else:
            patience += 1
            if patience >= PATIENCE:
                log_info(f"✗ Early stopping at epoch {epoch+1}")
                break

    log_info(f"Best PR={best_pr:.4f}@{best_pr_epoch} | "
             f"F1={best_f1:.4f}@{best_f1_epoch} | "
             f"MCC={best_mcc:.4f}@{best_mcc_epoch}")


if __name__ == "__main__":
    main()
