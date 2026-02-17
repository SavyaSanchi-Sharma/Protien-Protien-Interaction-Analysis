import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from createdatset import PPISDataset
from model.esm_projection import ESMProjection
from model.fusion import GatedFusion
from model.gcn import GCNEncoder
from model.tcn import BiTCN
from model.classifier import Classifier

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 100
BATCH_SIZE = 4          # recommended for 4GB RTX3050
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = "checkpoints"
PATIENCE = 10

# loss controls
USE_FOCAL = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

USE_SMOOTHNESS = True
SMOOTHNESS_LAMBDA = 0.05  # tune: 0.01 to 0.1

# dataset
TRAIN_CSV = "data/structural/Train_335_15D.csv"
ESM_DIR = "data/esm"
PDB_DIR = "data/pdbs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -------------------------
# LOSS HELPERS
# -------------------------
def focal_with_logits(logits, targets, alpha=0.25, gamma=2.0, pos_weight=None):
    """
    logits: [N]
    targets: [N] float 0/1
    """
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)

    # class weighting alpha
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    focal = alpha_t * ((1 - pt) ** gamma) * bce
    return focal.mean()


def smoothness_regularizer(probs, edge_index, edge_weight=None):
    """
    Patch prior: encourages neighboring residues to have similar probabilities.
    sum_{(i,j) in E} w_ij (p_i - p_j)^2
    """
    src, dst = edge_index
    diff2 = (probs[src] - probs[dst]).pow(2)
    if edge_weight is not None:
        return (edge_weight * diff2).mean()
    return diff2.mean()


# -------------------------
# POS WEIGHT COMPUTATION
# -------------------------
def compute_pos_weight(dataset):
    """
    Compute pos_weight = N_neg / N_pos from training labels.
    dataset returns one protein at a time (Data object with y).
    """
    pos = 0
    neg = 0
    for i in range(len(dataset)):
        y = dataset[i].y
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())

    # avoid division by zero
    pos = max(pos, 1)
    pw = neg / pos
    return pw, pos, neg


# -------------------------
# MAIN
# -------------------------
def main():
    print("[INFO] Loading training dataset...")
    dataset = PPISDataset(
        csv_path=TRAIN_CSV,
        esm_dir=ESM_DIR,
        pdb_dir=PDB_DIR,
        cutoff=14.0,
        sigma=5.0
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[OK] Dataset loaded: {len(dataset)} protein chains")

    # compute pos weight from training data
    pw, pos, neg = compute_pos_weight(dataset)
    pos_weight = torch.tensor([pw], device=DEVICE, dtype=torch.float32)

    print(f"[INFO] Train pos={pos} neg={neg} => pos_weight={pw:.3f}")

    # -------------------------
    # MODEL INIT
    # -------------------------
    proj = ESMProjection(in_dim=2560, dropout=0.2)
    fusion = GatedFusion(d_esm=256, d_struct=15, d_out=256, dropout=0.25)
    gcn = GCNEncoder(in_dim=256, hidden=256, num_layers=6, alpha=0.1, dropout=0.2)
    tcn = BiTCN(in_dim=256, dropout=0.2)   # expects 256-d now
    clf = Classifier(dim=1024, dropout=0.4)

    proj = proj.to(DEVICE)
    fusion = fusion.to(DEVICE)
    gcn = gcn.to(DEVICE)
    tcn = tcn.to(DEVICE)
    clf = clf.to(DEVICE)

    print("[OK] model initialized")

    optimizer = optim.AdamW(
        list(proj.parameters()) +
        list(fusion.parameters()) +
        list(gcn.parameters()) +
        list(tcn.parameters()) +
        list(clf.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")
    patience_counter = 0

    # -------------------------
    # TRAIN LOOP
    # -------------------------
    for epoch in range(EPOCHS):
        proj.train()
        fusion.train()
        gcn.train()
        tcn.train()
        clf.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, data in enumerate(loader):
            data = data.to(DEVICE)

            # split raw node features
            esm = data.x[:, :2560]        # [N,2560]
            struct = data.x[:, 2560:]     # [N,15]

            # 1) ESM projection
            esm_proj = proj(esm)          # [N,256]

            # 2) gated fusion (forces structural influence)
            fused = fusion(esm_proj, struct)   # [N,256]

            # 3) GCN with edge weights
            edge_weight = getattr(data, "edge_weight", None)
            gcn_out = gcn(fused, data.edge_index, edge_weight=edge_weight)  # [N,256]

            # 4) Per-protein BiTCN
            # because multiple proteins are batched -> split by data.batch
            seq_lens = torch.bincount(data.batch)
            tcn_outs = []
            start = 0
            for seq_len in seq_lens.tolist():
                seq_gcn = gcn_out[start:start + seq_len]  # [L,256]
                seq_tcn = tcn(seq_gcn.unsqueeze(0)).squeeze(0)  # [L,1024]
                tcn_outs.append(seq_tcn)
                start += seq_len
            tcn_out = torch.cat(tcn_outs, dim=0)  # [N,1024]

            # 5) classifier head -> logits
            logits = clf(tcn_out)  # [N]

            targets = data.y.float().view(-1)

            # classification loss
            if USE_FOCAL:
                loss_cls = focal_with_logits(
                    logits, targets,
                    alpha=FOCAL_ALPHA,
                    gamma=FOCAL_GAMMA,
                    pos_weight=pos_weight
                )
            else:
                loss_cls = bce_loss_fn(logits.view(-1), targets)

            # smoothness regularizer
            if USE_SMOOTHNESS:
                probs = torch.sigmoid(logits.detach())  # detach prevents weird exploding coupling
                loss_smooth = smoothness_regularizer(
                    probs, data.edge_index,
                    edge_weight=edge_weight
                )
                loss = loss_cls + SMOOTHNESS_LAMBDA * loss_smooth
            else:
                loss = loss_cls

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(proj.parameters()) +
                list(fusion.parameters()) +
                list(gcn.parameters()) +
                list(tcn.parameters()) +
                list(clf.parameters()),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n[Epoch {epoch+1}/{EPOCHS}] Average Loss: {avg_epoch_loss:.4f}")

        # early stopping + checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0

            torch.save({
                "proj_state_dict": proj.state_dict(),
                "fusion_state_dict": fusion.state_dict(),
                "gcn_state_dict": gcn.state_dict(),
                "tcn_state_dict": tcn.state_dict(),
                "clf_state_dict": clf.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss": best_loss,
                "pos_weight": float(pw),
            }, os.path.join(CHECKPOINT_DIR, "model_best.pt"))

            print(f"[SAVED] Best model checkpoint at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[EARLY STOP] No improvement for {PATIENCE} epochs")
                break

    print("\n[DONE] Training complete!")


if __name__ == "__main__":
    main()
