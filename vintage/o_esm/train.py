import os, sys, json, random, warnings
warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, precision_recall_curve
from datetime import datetime
import numpy as np
import pandas as pd


def set_seed(seed: int = 42):
    """Seed all RNGs for reproducible training/eval."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "..")
sys.path.insert(0, ROOT)

from createdatset import PPISDataset
from model.esm_projection import ESMProjection
from model.gcn import GCNEncoder
from model.tcn import BiTCN
from model.classifier import Classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
WARMUP_EPOCHS = 5
PATIENCE = 15

DEFAULT_HP = {
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "focal_gamma": 2.0,
    "lambda_mcc": 0.5,
    "proj_dropout": 0.2,
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
ESM_DIR        = os.path.join(DATA_ROOT, "data", "esm")
PDB_DIR        = os.path.join(DATA_ROOT, "data", "pdbs")
CKPT_DIR       = os.path.join(ROOT, "checkpoints")
HP_PATH        = os.path.join(ROOT, "best_hp.json")

os.makedirs(CKPT_DIR, exist_ok=True)
scaler = GradScaler("cuda")


def load_hp():
    hp = dict(DEFAULT_HP)
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            tuned = json.load(f).get("params", {})
        hp.update({k: v for k, v in tuned.items() if k in DEFAULT_HP})
        print(f"[HP] Loaded tuned hyperparameters from {HP_PATH}")
    return hp


def log_info(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def ensure_train_val_split(full_csv, train_out, val_out, seed=42, val_frac=0.2):
    """Carve a protein-level val split from the full Train CSV. Idempotent.
    Same seed across all four model runs → same split, no leakage."""
    if os.path.exists(train_out) and os.path.exists(val_out):
        return
    df = pd.read_csv(full_csv)
    proteins = df[["PDB", "Chain"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(proteins))
    n_val = max(1, int(round(len(proteins) * val_frac)))
    val_rows = proteins.iloc[perm[:n_val]]
    val_keys = set(zip(val_rows["PDB"].tolist(), val_rows["Chain"].tolist()))
    is_val = np.array([k in val_keys for k in zip(df["PDB"], df["Chain"])])
    df[~is_val].to_csv(train_out, index=False)
    df[is_val].to_csv(val_out, index=False)
    print(f"[SPLIT] {len(proteins) - n_val} train / {n_val} val proteins (seed={seed}) "
          f"→ {os.path.basename(train_out)}, {os.path.basename(val_out)}")


def compute_dataset_alpha(dataset):
    pos, neg = 0, 0
    for i in range(len(dataset)):
        y = dataset[i].y
        pos += int((y == 1).sum())
        neg += int((y == 0).sum())
    pos = max(pos, 1)
    N = pos + neg
    log_info(f"Dataset Stats | Total: {N} | Pos: {pos} ({pos/N*100:.1f}%) | Neg: {neg} ({neg/N*100:.1f}%)")
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


def hybrid_focal_cost_loss(logits, targets, alpha_pos, gamma=2.0, lambda_mcc=0.0):
    logp = torch.log_softmax(logits, dim=1)[:, 1]
    p = logp.exp()
    pos = targets == 1
    neg = targets == 0
    loss_pos = -alpha_pos * ((1 - p[pos]) ** gamma) * logp[pos]
    loss_neg = -(p[neg] ** gamma) * torch.log1p(-p[neg])
    loss = (loss_pos.sum() + loss_neg.sum()) / targets.numel()
    if lambda_mcc > 0.0:
        loss = loss - lambda_mcc * soft_mcc(logits, targets)
    return loss


def find_optimal_threshold(probs, labels):
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def find_mcc_threshold(probs, labels):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh, best_mcc = 0.5, -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        s = int(preds.sum())
        if s == 0 or s == len(preds):
            continue
        m = matthews_corrcoef(labels, preds)
        if m > best_mcc:
            best_mcc = m
            best_thresh = float(t)
    return best_thresh, float(best_mcc)


@torch.no_grad()
def evaluate(loader, alpha_pos):
    proj.eval(); gcn.eval(); tcn.eval(); clf.eval()
    losses, probs_all, labels_all = [], [], []

    with autocast(device_type="cuda"):
        for data in loader:
            data = data.to(DEVICE)
            h = proj(data.x)
            h = gcn(h, data.edge_index, edge_weight=data.edge_weight)
            padded, mask = to_dense_batch(h, data.batch)
            lengths = mask.sum(dim=1)
            h = tcn(padded, lengths=lengths)[mask]
            logits = clf(h)
            y = data.y.long().view(-1)
            loss = hybrid_focal_cost_loss(logits, y, alpha_pos)
            losses.append(loss.item())
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
    return sum(losses)/len(losses), auc, pr, f1_default, mcc_default, f1_opt, mcc_opt, opt_thresh


def main():
    global proj, gcn, tcn, clf

    set_seed(int(os.environ.get("PPI_SEED", "42")))
    hp = load_hp()
    log_info("=" * 80)
    log_info("o_esm: ESM-only | GCN + BiTCN")
    log_info(f"Device: {DEVICE} | Epochs: {EPOCHS} | LR: {hp['lr']} | Batch: {hp['batch_size']}")
    log_info(f"HP: {hp}")
    log_info("=" * 80)

    ensure_train_val_split(TRAIN_FULL_CSV, TRAIN_CSV, VAL_CSV)

    log_info("Loading datasets...")
    train_ds = PPISDataset(TRAIN_CSV, ESM_DIR, PDB_DIR)
    val_ds   = PPISDataset(VAL_CSV,   ESM_DIR, PDB_DIR)
    log_info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False)

    alpha_pos = compute_dataset_alpha(train_ds)

    proj = ESMProjection(2560, hidden_dim=512, out_dim=256, dropout=hp["proj_dropout"]).to(DEVICE)
    gcn  = GCNEncoder(256, hidden=256, num_layers=hp["gcn_layers"], alpha=hp["gcn_alpha"], dropout=hp["gcn_dropout"]).to(DEVICE)
    tcn  = BiTCN(256, channels=[64, 128, 256, 512], dropout=hp["tcn_dropout"]).to(DEVICE)
    clf  = Classifier(1024, dropout=hp["clf_dropout"]).to(DEVICE)

    total_params = sum(p.numel() for m in [proj, gcn, tcn, clf] for p in m.parameters())
    log_info(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(
        list(proj.parameters()) + list(gcn.parameters()) +
        list(tcn.parameters()) + list(clf.parameters()),
        lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    warmup = LambdaLR(optimizer, lambda e: min(1.0, (e + 1) / WARMUP_EPOCHS))
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

    best_pr, best_f1, best_mcc = 0.0, 0.0, -1.0
    best_pr_epoch, best_f1_epoch, best_mcc_epoch = 0, 0, 0
    patience_count = 0
    log = []

    for epoch in range(EPOCHS):
        proj.train(); gcn.train(); tcn.train(); clf.train()
        epoch_loss = 0.0

        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                h = proj(data.x)
                h = gcn(h, data.edge_index, edge_weight=data.edge_weight)
                padded, mask = to_dense_batch(h, data.batch)
                lengths = mask.sum(dim=1)
                h = tcn(padded, lengths=lengths)[mask]
                logits = clf(h)
                y = data.y.long().view(-1)
                loss = hybrid_focal_cost_loss(logits, y, alpha_pos,
                                              gamma=hp["focal_gamma"],
                                              lambda_mcc=hp["lambda_mcc"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        (val_loss, val_auc, val_pr, val_f1, val_mcc,
         val_f1_opt, val_mcc_opt, opt_thresh) = evaluate(val_loader, alpha_pos)

        if epoch < WARMUP_EPOCHS:
            warmup.step(); sched_name = "Warmup"
        else:
            cosine.step(); sched_name = "Cosine"

        current_lr = optimizer.param_groups[0]["lr"]
        log_info(f"[Epoch {epoch+1}/{EPOCHS}] LR={current_lr:.2e} ({sched_name}) | "
                 f"Train={train_loss:.4f} | Val={val_loss:.4f}")
        log_info(f"  AUC={val_auc:.4f} | PR={val_pr:.4f} | F1(0.5)={val_f1:.4f} | "
                 f"F1_max={val_f1_opt:.4f} | MCC_max={val_mcc_opt:.4f}@{opt_thresh:.2f} | "
                 f"Patience={patience_count}/{PATIENCE}")
        log.append({
            "epoch": int(epoch + 1), "train_loss": float(train_loss),
            "val_loss": float(val_loss), "val_auc": float(val_auc),
            "val_pr": float(val_pr), "val_f1_default": float(val_f1),
            "val_mcc_default": float(val_mcc), "val_f1_optimal": float(val_f1_opt),
            "val_mcc_optimal": float(val_mcc_opt),
            "optimal_threshold": float(opt_thresh), "lr": float(current_lr),
        })
        with open(os.path.join(ROOT, "training_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        if val_pr > best_pr:
            best_pr = val_pr; best_pr_epoch = epoch + 1
            log_info(f"  • New best PR: {best_pr:.4f}")
        if val_f1_opt > best_f1:
            best_f1 = val_f1_opt; best_f1_epoch = epoch + 1
            log_info(f"  • New best F1: {best_f1:.4f}")
        improved = val_mcc_opt > best_mcc
        if improved:
            best_mcc = val_mcc_opt; best_mcc_epoch = epoch + 1
            log_info(f"  ✓ New best MCC: {best_mcc:.4f}  (saving checkpoint)")

        if improved:
            patience_count = 0
            torch.save({
                "proj": proj.state_dict(), "gcn": gcn.state_dict(),
                "tcn": tcn.state_dict(), "clf": clf.state_dict(),
                "epoch": epoch + 1, "best_pr": best_pr, "best_f1": best_f1,
                "best_mcc": best_mcc, "optimal_threshold": opt_thresh,
            }, os.path.join(CKPT_DIR, "best.pt"))
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log_info(f"✗ Early stopping at epoch {epoch+1}")
                break

    log_info(f"Best PR={best_pr:.4f}@{best_pr_epoch} | F1={best_f1:.4f}@{best_f1_epoch} | MCC={best_mcc:.4f}@{best_mcc_epoch}")


if __name__ == "__main__":
    main()
