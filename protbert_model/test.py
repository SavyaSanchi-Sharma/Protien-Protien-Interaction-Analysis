import os
import sys
import json
import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.amp import autocast
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_recall_curve, accuracy_score, precision_score, recall_score,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "..")
sys.path.insert(0, ROOT)

from createdatset import PPISDataset, EDGE_ATTR_DIM, NUM_PLM_LAYERS, PLM_DIM
from model.protbert_projection import MultiLayerProjection
from model.fusion import CrossAttentionFusion
from model.gcn import GCNEncoder
from model.tcn import BiTCN
from model.classifier import Classifier
from train import DEFAULT_HP, HP_PATH, set_seed

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
PROTBERT_DIR = os.path.join(DATA_ROOT, "data", "protbert_multi")
CKPT         = os.path.join(ROOT, "checkpoints", "best.pt")
NUM_WORKERS = int(os.environ.get("PPI_NUM_WORKERS", "4"))
TEST_BATCH_SIZE = int(os.environ.get("PPI_TEST_BATCH_SIZE", "4"))
TEST_CSVS = {
    "Test_60":  os.path.join(DATA_ROOT, "data", "structural", "Test_60_17D.csv"),
    "Test_315": os.path.join(DATA_ROOT, "data", "structural", "Test_315_17D.csv"),
}


def get_hp():
    hp = dict(DEFAULT_HP)
    if os.path.exists(HP_PATH):
        with open(HP_PATH) as f:
            tuned = json.load(f).get("params", {})
        hp.update({k: v for k, v in tuned.items() if k in DEFAULT_HP})
    return hp


def load_models(ckpt, hp):
    proj   = MultiLayerProjection(NUM_PLM_LAYERS, PLM_DIM, 512, 256,
                                  hp["proj_dropout"]).to(DEVICE)
    fusion = CrossAttentionFusion(d_esm=256, d_struct=17, d_out=256,
                                  n_heads=hp.get("fusion_heads", 4),
                                  dropout=hp["fusion_dropout"]).to(DEVICE)
    gcn    = GCNEncoder(256, 256, hp["gcn_layers"], hp["gcn_alpha"], hp["gcn_dropout"],
                        edge_dim=EDGE_ATTR_DIM).to(DEVICE)
    tcn    = BiTCN(256, [64, 128, 256, 512], hp["tcn_dropout"]).to(DEVICE)
    clf    = Classifier(1024, hp["clf_dropout"]).to(DEVICE)
    proj.load_state_dict(ckpt["proj"])
    fusion.load_state_dict(ckpt["fusion"])
    gcn.load_state_dict(ckpt["gcn"])
    tcn.load_state_dict(ckpt["tcn"])
    clf.load_state_dict(ckpt["clf"])
    for m in (proj, fusion, gcn, tcn, clf):
        m.eval()
    return proj, fusion, gcn, tcn, clf


@torch.no_grad()
def run_eval(loader, proj, fusion, gcn, tcn, clf):
    probs_all, labels_all = [], []
    with autocast(device_type="cuda"):
        for data in loader:
            data = data.to(DEVICE)
            plm_proj = proj(data.emb)
            plm_dense, mask = to_dense_batch(plm_proj, data.batch)
            struct_dense, _ = to_dense_batch(data.x, data.batch)
            fused = fusion(plm_dense, struct_dense, mask=mask)
            h = fused[mask]
            h = gcn(h, data.edge_index, edge_weight=data.edge_weight, edge_attr=data.edge_attr)
            padded, mask2 = to_dense_batch(h, data.batch)
            h = tcn(padded, lengths=mask2.sum(dim=1))[mask2]
            logits = clf(h)
            probs_all.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            labels_all.append(data.y.long().view(-1).cpu())
    return torch.cat(probs_all).numpy(), torch.cat(labels_all).numpy()


def metrics(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy":  float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall":    float(recall_score(labels, preds, zero_division=0)),
        "f1":        float(f1_score(labels, preds, zero_division=0)),
        "mcc":       float(matthews_corrcoef(labels, preds)),
        "auroc":     float(roc_auc_score(labels, probs)),
        "auprc":     float(average_precision_score(labels, probs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    print(f"[TEST] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    hp = get_hp()
    proj, fusion, gcn, tcn, clf = load_models(ckpt, hp)
    saved_thresh = float(ckpt.get("optimal_threshold", 0.5))
    print(f"[TEST] Saved threshold (from training): {saved_thresh:.4f}")

    results = {}
    for name, csv_path in TEST_CSVS.items():
        if not os.path.exists(csv_path):
            print(f"[SKIP] {name}: {csv_path} not found")
            continue
        ds = PPISDataset(csv_path, PROTBERT_DIR)
        loader = DataLoader(ds, batch_size=TEST_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS,
                            pin_memory=DEVICE == "cuda",
                            persistent_workers=NUM_WORKERS > 0)
        probs, labels = run_eval(loader, proj, fusion, gcn, tcn, clf)

        m_saved = metrics(probs, labels, saved_thresh)
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1_curve = 2 * precision * recall / (precision + recall + 1e-8)
        opt_thresh = float(thresholds[int(np.argmax(f1_curve))])
        m_opt = metrics(probs, labels, opt_thresh)
        results[name] = {"at_saved_threshold": m_saved,
                         "at_test_optimal_threshold": m_opt}

        print(f"\n=== {name} ===")
        print(f"  AUROC={m_saved['auroc']:.4f}  AUPRC={m_saved['auprc']:.4f}")
        print(f"  @saved_thresh={saved_thresh:.2f} -> "
              f"F1={m_saved['f1']:.4f} MCC={m_saved['mcc']:.4f} "
              f"P={m_saved['precision']:.4f} R={m_saved['recall']:.4f} "
              f"Acc={m_saved['accuracy']:.4f}")
        print(f"  @opt_thresh={opt_thresh:.2f}  -> "
              f"F1={m_opt['f1']:.4f} MCC={m_opt['mcc']:.4f}")

    out = os.path.join(ROOT, "test_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved test results -> {out}")


if __name__ == "__main__":
    main()
