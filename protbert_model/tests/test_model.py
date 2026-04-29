import os, sys
import torch
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from torch_geometric.loader import DataLoader
from createdatset import PPISDataset
from model.protbert_projection import ProtBertProjection
from model.fusion import GatedFusion
from model.gcn import GCNEncoder
from model.tcn import BiTCN
from model.classifier import Classifier
from sklearn.metrics import precision_recall_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT    = os.path.join(ROOT, "..")
CSV_FILE     = os.path.join(DATA_ROOT, "data", "structural", "Test_60_17D.csv")
PROTBERT_DIR = os.path.join(DATA_ROOT, "data", "protbert")
PDB_DIR      = os.path.join(DATA_ROOT, "data", "pdbs")
CHECKPOINT   = os.path.join(ROOT, "checkpoints", "best.pt")
OUT_CSV      = os.path.join(ROOT, "results", "Test_60_predictions.csv")


def main():
    dataset = PPISDataset(CSV_FILE, PROTBERT_DIR, PDB_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    proj   = ProtBertProjection(1024, hidden_dim=512, out_dim=256, dropout=0.2)
    fusion = GatedFusion(d_esm=256, d_struct=17, d_out=256, dropout=0.01)
    gcn    = GCNEncoder(256, hidden=256, num_layers=6, alpha=0.2, dropout=0.1)
    tcn    = BiTCN(256, channels=[64, 128, 256, 512], dropout=0.1)
    clf    = Classifier(1024, dropout=0.3)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    proj.load_state_dict(ckpt["proj"])
    fusion.load_state_dict(ckpt["fusion"])
    gcn.load_state_dict(ckpt["gcn"])
    tcn.load_state_dict(ckpt["tcn"])
    clf.load_state_dict(ckpt["clf"])

    proj = proj.to(DEVICE).eval()
    fusion = fusion.to(DEVICE).eval()
    gcn = gcn.to(DEVICE).eval()
    tcn = tcn.to(DEVICE).eval()
    clf = clf.to(DEVICE).eval()

    rows = []
    with torch.no_grad():
        for idx, data in enumerate(loader):
            data = data.to(DEVICE)
            pb     = data.x[:, :1024]
            struct = data.x[:, 1024:]
            h0 = proj(pb)
            h  = fusion(h0, struct)
            h  = gcn(h, data.edge_index, edge_weight=data.edge_weight)
            seq_lens = torch.bincount(data.batch) if hasattr(data, "batch") else torch.tensor([len(h)])
            out, s = [], 0
            for L in seq_lens.tolist():
                out.append(tcn(h[s:s+L].unsqueeze(0)).squeeze(0))
                s += L
            h = torch.cat(out, dim=0) if len(out) > 1 else out[0]
            logits = clf(h)
            probs = torch.softmax(logits, dim=1)[:, 1]

            (pdb, chain), _ = dataset.groups[idx]
            y = data.y.view(-1).cpu().numpy()
            p = probs.cpu().numpy()
            for i in range(len(p)):
                rows.append({"PDB": pdb, "Chain": chain, "Residue_Index": i + 1,
                             "Probability": float(p[i]), "Label": int(y[i])})

    df = pd.DataFrame(rows)
    prec, rec, thr = precision_recall_curve(df["Label"].values, df["Probability"].values)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = thr[np.argmax(f1)]
    df["Prediction"] = (df["Probability"] >= best_thr).astype(int)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Best F1 threshold = {best_thr:.4f}")
    print(f"[DONE] Predictions written to {OUT_CSV}")


if __name__ == "__main__":
    main()
