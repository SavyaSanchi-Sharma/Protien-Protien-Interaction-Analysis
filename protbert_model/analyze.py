import os, sys, json, warnings
warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.amp import autocast
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve,
                              confusion_matrix, f1_score, matthews_corrcoef)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "..")
sys.path.insert(0, ROOT)

from createdatset import PPISDataset, EDGE_ATTR_DIM
from train import set_seed
from test import get_hp, load_models, run_eval, PROTBERT_DIR, TEST_CSVS, CKPT, NUM_WORKERS, TEST_BATCH_SIZE

ANALYSIS_DIR = os.path.join(ROOT, "analysis")
TRAIN_LOG    = os.path.join(ROOT, "training_log.json")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


def _epoch_axis(log):
    return [r["epoch"] for r in log]


def plot_loss_curves(log, out_path):
    epochs = _epoch_axis(log)
    train  = [r["train_loss"] for r in log]
    val    = [r["val_loss"]   for r in log]
    best_i = max(range(len(log)), key=lambda i: log[i]["val_mcc_optimal"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, color="tab:blue",   label="Train loss",   marker="o", markersize=3)
    ax.plot(epochs, val,   color="tab:red",    label="Val loss",     marker="s", markersize=3)
    ax.axvline(log[best_i]["epoch"], color="tab:green", linestyle="--", alpha=0.6,
               label=f"Best MCC @ epoch {log[best_i]['epoch']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (focal + λ·MCC penalty)")
    ax.set_title("Training / Validation loss (protbert_model)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_loss_components(log, out_path):
    if "train_focal" not in log[0]:
        print("  [skip] no train_focal in log — re-run training to capture loss components")
        return
    epochs = _epoch_axis(log)
    focal  = [r["train_focal"]    for r in log]
    mcc_s  = [r["train_soft_mcc"] for r in log]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, focal, color="tab:blue", marker="o", markersize=3,
             label="Focal-cost loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Focal loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, mcc_s, color="tab:red", marker="s", markersize=3,
             label="Soft MCC")
    ax2.set_ylabel("Soft MCC (training)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    fig.suptitle("Loss components (protbert_model) — focal drives quality, soft-MCC drives correlation")
    ax1.grid(alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_val_metrics(log, out_path):
    epochs = _epoch_axis(log)
    best_i = max(range(len(log)), key=lambda i: log[i]["val_mcc_optimal"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [r["val_auc"]         for r in log], label="AUROC")
    ax.plot(epochs, [r["val_pr"]          for r in log], label="AUPRC")
    ax.plot(epochs, [r["val_f1_optimal"]  for r in log], label="F1 (max)")
    ax.plot(epochs, [r["val_mcc_optimal"] for r in log], label="MCC (max)",
            linewidth=2.0, color="tab:red")
    ax.axvline(log[best_i]["epoch"], color="tab:green", linestyle="--", alpha=0.6,
               label=f"Best MCC @ epoch {log[best_i]['epoch']}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Metric value")
    ax.set_title("Validation metrics over epochs (protbert_model)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_lr_schedule(log, out_path):
    epochs = _epoch_axis(log)
    lrs    = [r["lr"] for r in log]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, color="tab:purple", marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning rate (log scale)")
    ax.set_yscale("log")
    ax.set_title("LR schedule (warmup → cosine annealing)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_pr_curves(probs_dict, labels_dict, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in probs_dict:
        prec, rec, _ = precision_recall_curve(labels_dict[name], probs_dict[name])
        ap = average_precision_score(labels_dict[name], probs_dict[name])
        ax.plot(rec, prec, label=f"{name}  AP={ap:.4f}")
    pos_rate = np.mean([labels_dict[n].mean() for n in labels_dict])
    ax.axhline(pos_rate, color="gray", linestyle="--", alpha=0.4,
               label=f"Random baseline (pos rate ≈ {pos_rate:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curves on test sets (protbert_model)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_roc_curves(probs_dict, labels_dict, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in probs_dict:
        fpr, tpr, _ = roc_curve(labels_dict[name], probs_dict[name])
        auc = roc_auc_score(labels_dict[name], probs_dict[name])
        ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.4,
            label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves on test sets (protbert_model)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_prob_distributions(probs_dict, labels_dict, saved_thresh, out_path):
    fig, axes = plt.subplots(1, len(probs_dict), figsize=(7 * len(probs_dict), 5),
                             squeeze=False)
    for ax, name in zip(axes[0], probs_dict):
        probs = probs_dict[name]; labels = labels_dict[name]
        ax.hist(probs[labels == 0], bins=50, alpha=0.55, color="tab:blue",
                label=f"Negative (n={int((labels == 0).sum())})", density=True)
        ax.hist(probs[labels == 1], bins=50, alpha=0.55, color="tab:red",
                label=f"Positive (n={int((labels == 1).sum())})", density=True)
        ax.axvline(saved_thresh, color="tab:green", linestyle="--",
                   label=f"saved thresh = {saved_thresh:.2f}")
        ax.set_xlabel("Predicted P(binding)"); ax.set_ylabel("Density")
        ax.set_title(f"{name} — predicted probability distribution")
        ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_confusion_matrices(probs_dict, labels_dict, saved_thresh, out_path):
    fig, axes = plt.subplots(1, len(probs_dict), figsize=(6 * len(probs_dict), 5),
                             squeeze=False)
    for ax, name in zip(axes[0], probs_dict):
        probs = probs_dict[name]; labels = labels_dict[name]
        preds = (probs >= saved_thresh).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                ax.text(j, i, f"{v:,}", ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black",
                        fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred non-bind", "Pred bind"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True non-bind", "True bind"])
        f1 = f1_score(labels, preds, zero_division=0)
        mcc = matthews_corrcoef(labels, preds)
        ax.set_title(f"{name} @ thresh={saved_thresh:.2f}\nF1={f1:.3f}  MCC={mcc:.3f}")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def collect_test_predictions():
    print(f"[ANALYZE] Loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location=DEVICE)
    saved_thresh = float(ckpt.get("optimal_threshold", 0.5))
    hp = get_hp()
    proj, fusion, gcn, tcn, clf = load_models(ckpt, hp)
    probs_dict, labels_dict = {}, {}
    for name, csv_path in TEST_CSVS.items():
        if not os.path.exists(csv_path):
            print(f"  [skip] {name}: {csv_path} missing"); continue
        ds = PPISDataset(csv_path, PROTBERT_DIR)
        loader = DataLoader(ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=DEVICE == "cuda",
                            persistent_workers=NUM_WORKERS > 0)
        probs, labels = run_eval(loader, proj, fusion, gcn, tcn, clf)
        probs_dict[name]  = probs
        labels_dict[name] = labels
        print(f"  [{name}] AUC={roc_auc_score(labels, probs):.4f}  "
              f"AP={average_precision_score(labels, probs):.4f}")
    return probs_dict, labels_dict, saved_thresh


def write_summary(log, probs_dict, labels_dict, saved_thresh, out_path):
    best_i = max(range(len(log)), key=lambda i: log[i]["val_mcc_optimal"])
    best   = log[best_i]
    lines = []
    lines.append("=" * 72)
    lines.append("protbert_model — analysis summary")
    lines.append("=" * 72)
    lines.append(f"Best epoch:         {best['epoch']}")
    lines.append(f"  val_loss:         {best['val_loss']:.4f}")
    lines.append(f"  val_auc:          {best['val_auc']:.4f}")
    lines.append(f"  val_pr (AUPRC):   {best['val_pr']:.4f}")
    lines.append(f"  val_f1_max:       {best['val_f1_optimal']:.4f}")
    lines.append(f"  val_mcc_max:      {best['val_mcc_optimal']:.4f}")
    lines.append(f"  saved threshold:  {best['optimal_threshold']:.4f}")
    lines.append(f"Total epochs run:   {len(log)}")
    lines.append("")
    lines.append(f"Test-set evaluation (saved threshold = {saved_thresh:.4f}):")
    for name in probs_dict:
        probs  = probs_dict[name]; labels = labels_dict[name]
        preds  = (probs >= saved_thresh).astype(int)
        auc    = roc_auc_score(labels, probs)
        ap     = average_precision_score(labels, probs)
        f1     = f1_score(labels, preds, zero_division=0)
        mcc    = matthews_corrcoef(labels, preds)
        lines.append(f"  {name:10s} AUROC={auc:.4f}  AUPRC={ap:.4f}  "
                     f"F1={f1:.4f}  MCC={mcc:.4f}  (n={len(labels)})")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    set_seed(42)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_LOG):
        raise FileNotFoundError(f"Missing {TRAIN_LOG} — run train.py first")
    with open(TRAIN_LOG) as f:
        log = json.load(f)
    if not log:
        raise RuntimeError("training_log.json is empty")

    print(f"[ANALYZE] Generating plots in {ANALYSIS_DIR}")
    plot_loss_curves     (log, os.path.join(ANALYSIS_DIR, "01_loss_curves.png"))
    plot_loss_components (log, os.path.join(ANALYSIS_DIR, "02_loss_components.png"))
    plot_val_metrics     (log, os.path.join(ANALYSIS_DIR, "03_val_metrics.png"))
    plot_lr_schedule     (log, os.path.join(ANALYSIS_DIR, "04_lr_schedule.png"))

    probs_dict, labels_dict, saved_thresh = {}, {}, 0.5
    if os.path.exists(CKPT):
        try:
            probs_dict, labels_dict, saved_thresh = collect_test_predictions()
        except Exception as e:
            print(f"  [skip test-time plots] checkpoint load failed: {e.__class__.__name__}: {e}")
            print(f"  (this is expected if the architecture changed since best.pt was saved)")
        if probs_dict:
            plot_pr_curves            (probs_dict, labels_dict,
                                       os.path.join(ANALYSIS_DIR, "05_pr_curves.png"))
            plot_roc_curves           (probs_dict, labels_dict,
                                       os.path.join(ANALYSIS_DIR, "06_roc_curves.png"))
            plot_prob_distributions   (probs_dict, labels_dict, saved_thresh,
                                       os.path.join(ANALYSIS_DIR, "07_prob_distributions.png"))
            plot_confusion_matrices   (probs_dict, labels_dict, saved_thresh,
                                       os.path.join(ANALYSIS_DIR, "08_confusion_matrix.png"))
            write_summary(log, probs_dict, labels_dict, saved_thresh,
                          os.path.join(ANALYSIS_DIR, "summary.txt"))
    else:
        print(f"  [skip test-time plots] checkpoint missing: {CKPT}")

    print(f"\n[ANALYZE] Done → {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
