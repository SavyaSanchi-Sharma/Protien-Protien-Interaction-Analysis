import os, sys
import torch
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from torch_geometric.loader import DataLoader
from createdatset import PPISDataset

from model.esm_projection import ESMProjection
from model.fusion import AttnFusion
from model.gcn import GCNEncoder
from model.attn_encoder import AttnEncoder
from model.tcn import BiTCN
from model.classifier import Classifier

from sklearn.metrics import precision_recall_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "checkpoints/best.pt"
CSV_FILE   = "data/features/Test_60_36D.csv"
ESM_DIR    = "data/esm"
PDB_DIR    = "data/pdbs"
OUT_CSV    = "results/Test_60_predictions.csv"

USE_AMP = True


def main():

    dataset = PPISDataset(
        csv_path=CSV_FILE,
        esm_dir=ESM_DIR,
        pdb_dir=PDB_DIR,
        cutoff=10.0,
        sigma=4.0
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    proj = ESMProjection(2560, hidden_dim=1024, out_dim=512, dropout=0.2)
    fusion = AttnFusion(36, 512, 512)  # struct_dim=36, esm_dim=512
    gcn = GCNEncoder(
        in_dim=512,
        hidden=512,
        num_layers=12,  # Matches train.py
        alpha=0.7,
        dropout=0.1
    )
    attn = AttnEncoder(512, num_layers=3, heads=8, dropout=0.1)
    tcn = BiTCN(512, channels=[128,256,512,1024], dropout=0.1)
    clf = Classifier(2048, dropout=0.3)

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE,weights_only=False)

    proj.load_state_dict(ckpt["proj"])
    fusion.load_state_dict(ckpt["fusion"])
    gcn.load_state_dict(ckpt["gcn"])
    attn.load_state_dict(ckpt["attn"])
    tcn.load_state_dict(ckpt["tcn"])
    clf.load_state_dict(ckpt["clf"])

    proj = proj.to(DEVICE).eval()
    fusion = fusion.to(DEVICE).eval()
    gcn = gcn.to(DEVICE).eval()
    attn = attn.to(DEVICE).eval()
    tcn = tcn.to(DEVICE).eval()
    clf = clf.to(DEVICE).eval()

    rows = []

    with torch.no_grad():
        for idx, data in enumerate(loader):

            data = data.to(DEVICE)

            esm = data.x[:, :2560]
            struct = data.x[:, 2560:]

            if USE_AMP and DEVICE == "cuda":
                ctx = torch.amp.autocast(device_type="cuda")
            else:
                ctx = torch.amp.autocast(device_type="cpu", enabled=False)

            with ctx:
                h_esm = proj(esm)
                h = fusion(struct, h_esm)  # struct queries into ESM
                h = gcn(h, data.edge_index, edge_weight=data.edge_weight)
                h = attn(h.unsqueeze(0)).squeeze(0)  # Self-attention
                
                # BiTCN requires per-protein processing
                seq_lens = torch.bincount(data.batch) if hasattr(data, 'batch') else torch.tensor([len(h)])
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
                rows.append({
                    "PDB": pdb,
                    "Chain": chain,
                    "Residue_Index": i + 1,
                    "Probability": float(p[i]),
                    "Label": int(y[i])
                })

    df = pd.DataFrame(rows)

    y_all = df["Label"].values
    p_all = df["Probability"].values

    prec, rec, thr = precision_recall_curve(y_all, p_all)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = thr[np.argmax(f1)]

    df["Prediction"] = (df["Probability"] >= best_thr).astype(int)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"[OK] Best F1 threshold = {best_thr:.4f}")
    print(f"[DONE] Predictions written to {OUT_CSV}")


if __name__ == "__main__":
    main()
