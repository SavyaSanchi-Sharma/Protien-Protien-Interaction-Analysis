# Architecture & Data — current state

Snapshot of the active pipeline as of 2026-05-02. Two parallel models share
every module except the PLM input dim and which embedding directory they read.

> Older narrative + math derivations live in [RESEARCH.md](RESEARCH.md).
> This doc is the *current* code-and-data reference; treat it as the
> source of truth when they disagree.

---

## 1. End-to-end pipeline

```
┌──────────────────────┐    ┌────────────────────────┐    ┌────────────────────┐
│  data/fasta/*.fa     │    │  data/pdbs/*.pdb       │    │ data/structural/   │
│  (Train_335,         │    │  (downloaded from      │    │   *_17D.csv        │
│   Test_60, Test_315) │    │   RCSB by              │    │   17 features +    │
└──────────┬───────────┘    │   pdb_download.py)     │    │   6 coords +       │
           │                └───────────┬────────────┘    │   meta             │
           │                            │                 └─────────┬──────────┘
           │      ┌─────────────────────▼────────────────┐          │
           │      │  dataprep/dataprep.py                │──────────┘
           │      │   PDB → DSSP → 17D + Cα/N coords     │
           │      └──────────────────────────────────────┘
           │
           ▼
 ┌──────────────────────────────────┐    ┌──────────────────────────────────┐
 │ extract_esm_multilayer.py        │    │ extract_protbert_multilayer.py   │
 │ FASTA → ESM-2 3B (L,6,2560) fp16 │    │ FASTA → ProtBert (L,6,1024) fp16 │
 │ → data/esm_multi/<pdb>_<chain>.pt│    │ → data/protbert_multi/...        │
 └──────────────────┬───────────────┘    └──────────────────┬───────────────┘
                    │                                       │
                    ▼                                       ▼
              esm_model/                             protbert_model/
              ├ tune.py    (Optuna TPE → best_hp.json)
              ├ train.py   (Lion → checkpoints/best.pt + training_log.json)
              ├ test.py    (best.pt → test_results.json)
              └ analyze.py (training_log + best.pt → analysis/{plots,summary})
```

`run_esmModel.sh` and `run_protbertModel.sh` chain `tune → train → test → analyze`
end-to-end and `tee` to `logs/<model>.log`.

---

## 2. Data

### 2.1 Datasets

| Dataset | Chains | Residues | Pos rate | Use |
|---|---:|---:|---:|---|
| `Train_335` | 335 | 66,366 | 15.6 % | Training (split 80/20 protein-level, seed 42 → 268 train / 67 val) |
| `Test_60`   | 60  | 13,141 | 11.0 % | Held out, never seen during training/tuning |
| `Test_315`  | 315 | 65,331 | 11.6 % | Held out, never seen during training/tuning |

Train/val split is computed once and cached in
`data/structural/Train_split_{train,val}.csv` so all phases use the same
partition.

### 2.2 Structural CSV — 17 features + 6 coordinates per residue

Produced by [`dataprep/dataprep.py`](../dataprep/dataprep.py) from the PDB +
DSSP. CSV columns (in order): `PDB, Chain, ResIdx, AA, Label`, then 17
features, then 6 coordinates.

| Group | Cols | Definition |
|---|---|---|
| Solvent + flexibility | `RSA`, `ResFlex` | DSSP ASA / max-ASA(aa); min-max-normalized B-factor |
| Biochemistry | `Hydrophobicity` | Mean of 6 hydrophobicity scales (Kyte-Doolittle, Hopp-Woods, Eisenberg, Wimley-White, Hessa, Janin) |
| Local atomic | `PackingDensity` | Atoms within 3.5 Å of Cα, normalized |
| Half-sphere | `HSE_up`, `HSE_down` | Up/down ratio along Cβ−Cα within 8 Å |
| **NEW** geometry | `ResidueDepth` | Min Cα distance to convex-hull vertex of all heavy atoms (min-max norm) |
| **NEW** geometry | `SurfaceCurvature` | Signed projection of Cα-neighborhood centroid offset onto outward normal |
| **NEW** physics | `ElecPotential` | Σⱼ qⱼ/max(d_ij, 1.0) over charged side chains in 12 Å (R/K=+1, D/E=−1, H=+0.5) |
| **NEW** geometry | `LocalPlanarity` | λ₀ / Σλ from local Cα-cloud covariance |
| Backbone angle | `BondAngle` | Inter-peptide Cᵢ₋₁−Nᵢ vs Oᵢ₊₁−Nᵢ angle |
| Torsion | `sin/cos_phi`, `sin/cos_psi`, `sin/cos_omega` | DSSP φ/ψ + computed ω, sin/cos encoded |
| Coordinates | `CA_x`, `CA_y`, `CA_z`, `N_x`, `N_y`, `N_z` | Cα and N atom positions in PDB frame |

The 17 features replace the previous schema (Poly_bias / Poly_RSA / Poly_Flex
/ Poly_interaction were removed as redundant and replaced by the four NEW
features above).

### 2.3 Multi-layer PLM embeddings

For each chain, a fp16 tensor of shape `(L, 6, D)` is saved to
`data/esm_multi/` or `data/protbert_multi/`.

| | ESM-2 3B (`esm2_t36_3B_UR50D`) | ProtBert-BFD (`Rostlab/prot_bert_bfd`) |
|---|---|---|
| Total transformer layers | 36 | 30 |
| Hidden | 2560 | 1024 |
| Layers extracted | `[6, 18, 24, 30, 33, 36]` | `[5, 12, 18, 24, 27, 30]` |

Layers chosen to span the biochemistry → secondary-structure → contact →
task-specific gradient. The model learns the mixing weights at training time
via the `MultiLayerProjection` block ([§3.1](#31-multi-layer-plm-projection)).

### 2.4 Graph construction

For each chain, a contact graph is built from the cached Cα coordinates (no
PDB re-parsing at training time):

- **Edges**: all `(i, j)` with `‖Cα_i − Cα_j‖ ≤ 14 Å`, both directions
  (`KDTree.query_pairs` then mirror).
- **Edge attribute** `e_ij ∈ ℝ¹⁸`:
  - 16-D Gaussian RBF over distance with centers
    `c_k = k · 14/16` (`k = 0..15`) and bandwidth `σ = 14/16`.
  - 2-D `(sin θ_ij, cos θ_ij)` angle between residue *i*'s backbone direction
    `(Cα_i − N_i) / ‖·‖` and the edge direction
    `(Cα_j − Cα_i) / ‖·‖`. Degenerate edges fall back to `(0, 1)`.

Edges + edge_attr are precomputed at `Dataset.__init__` and cached. The
`Data` object exposed at training time carries:

```
data.x          (N, 17)        — normalized scalar struct features (z-score for first 11; sin/cos kept raw)
data.emb        (N, 6, D)      — multi-layer PLM tensor, fp32 at consumption
data.pos        (N, 3)         — Cα coordinates (read from CSV, used by EGNN + fusion bias)
data.edge_index (2, E)         — contact-graph edges (both directions)
data.edge_attr  (E, 18)        — RBF + (sin, cos) angle
data.y          (N,)           — binary residue labels
data.batch      (N,)           — auto-added by PyG DataLoader
```

`data.edge_weight` is also produced (ones tensor) but no current module
consumes it; harmless leftover for potential future use.

---

## 3. Architecture

For one batch of `B` chains with total `N = Σ Lᵦ` residues and max length `L_max`:

```
PLM (N, 6, D) ──► proj ──► (N, 256)
                                    ╲
                                     fusion(plm, struct, mask, pos) ──► (N, 256)
struct (N, 17) ─────────────────────╱      │
                                            │   ⮕  RBF(distance) bias on attention
pos (N, 3) ─► to_dense_batch ──► (B,L,3) ──┘   ⮕  gated combination
                                                ⮕  multiplicative merge

(N, 256) ──► EGNN(in_dim=256, hidden=256, num_layers=L_g, edge_dim=18) ──► (N, 256)
                  │
                  ▼ uses data.pos, data.edge_index, data.edge_attr
                  ▼ updates pos internally each layer (E(3)-equivariant message passing)

(N, 256) ──► to_dense_batch ──► (B, L_max, 256)
                  │
                  ▼ BiTCN: forward + reverse stacks, dilations [1,2,4,8], LayerNorm
                  ▼

(B, L_max, 1024) ──► node-list selection via mask ──► (N, 1024) ──► classifier ──► (N, 2)
```

All modules use **GELU + LayerNorm** throughout (no BatchNorm — variable
chain lengths and batch sizes 1–4 make batch statistics unreliable).

### 3.1 Multi-layer PLM projection

[`esm_projection.py`](../esm_model/model/esm_projection.py) /
[`protbert_projection.py`](../protbert_model/model/protbert_projection.py).

Input `(L, 6, D)` → output `(L, 256)`. Three steps:

1. **Per-layer LayerNorm**: each of the 6 layers gets its own
   `LayerNorm(D)`. Critical for ESM-2 because intermediate layers carry
   ~100× larger magnitudes than the final layer.
2. **Learned softmax mix**: weights `w ∈ ℝ⁶` start at zeros (uniform 1/6
   mix), `softmax`-normalized. A scalar `γ` (init 1.0) rescales the mixed
   tensor.
3. **MLP projection** D → H → 256 with GELU + Dropout + LayerNorm.
   `H = 1024` for ESM, `H = 512` for ProtBert.

### 3.2 Geo-biased cross-attention fusion

[`fusion.py`](../esm_model/model/fusion.py) (identical in both models).

Inputs (dense-batched):
- `plm: (B, L_max, 256)` — projected PLM features
- `struct: (B, L_max, 17)` — raw structural features
- `pos: (B, L_max, 3)` — Cα coordinates (already cast to plm.dtype at the call site for autocast)
- `mask: (B, L_max)` bool — `True` at valid residues

Six sub-blocks:

1. **Deep struct projection** `Linear(17, 128) → GELU → Linear(128, 256) → LayerNorm`.
   Replaces a single `Linear(17, 256)` so the project-and-merge has more
   expressive headroom on the small 17-D side.
2. **Geometric attention bias.** From `pos`, compute pairwise distance
   matrix `d ∈ ℝ^{B×L×L}`, expand via 16-D RBF centered on `[0, 20] Å`,
   pass through a 2-layer MLP per stream → `(B, L, L, n_heads)` additive
   logits. Reshaped to `(B·n_heads, L, L)` and passed to MHA as the
   `attn_mask` argument. Soft, not a hard cutoff — preserves long-range
   while preferring local. Padding positions get `-inf` baked in.
3. **Bidirectional cross-attention.**
   `s2e`: struct queries plm; `e2s`: plm queries (post-attn) struct.
   Each stream has its own MHA + residual + LayerNorm + FFN + LayerNorm.
4. **Gated combination.**
   `g = σ(Linear([plm_h ; struct_h]))`,
   `gated = g · plm_h + (1 − g) · struct_h`.
   Lets the model dial reliance per residue (buried vs exposed needs
   different mixes).
5. **Multiplicative merge.**
   `Linear([plm_h ; struct_h ; plm_h ⊙ struct_h ; gated]) → GELU → LN`.
   Hadamard product captures interactions a linear merge can't.
6. Final dropout + LayerNorm; padded positions zeroed.

### 3.3 EGNN encoder (replacing DeepGCN)

[`gcn.py`](../esm_model/model/gcn.py) (identical in both models). The
filename is preserved for backward git history; the contents are now
`EGNNEncoder` (the old `GCNEncoder` is commented out at the top).

`forward(x, pos, edge_index, edge_attr)`:

1. Input MLP `Linear(256, 256) → GELU → LayerNorm → Dropout` on `x` to get `h⁽⁰⁾`.
2. `num_layers` (tuned 4–10) `EGNNLayer` blocks. Each layer:
   - For every edge `(i, j)`: `rel = pos_i − pos_j`, `d² = ‖rel‖²`.
   - Edge MLP: `m_ij = MLP([h_i ; h_j ; d² ; e_ij])`, output dim 256.
   - **Coordinate update**: `dx_i = Σⱼ rel · MLP_coord(m_ij)`, then `pos ← pos + dx`. (Coords are updated, not consumed downstream — used to drive geometry-aware messages in subsequent layers.)
   - **Feature update**: `m_i = Σⱼ m_ij`, `h_i ← h_i + MLP_node(m_i)`, then LayerNorm + Dropout.

Equivariance: this layer is E(3)-equivariant in `pos` (translations + rotations
of all coords produce the same `h` output) and invariant in features. That's
the inductive bias we want for protein backbones.

Edge messages and node aggregation use `torch_scatter.scatter_add`.

### 3.4 BiTCN

[`tcn.py`](../esm_model/model/tcn.py). After EGNN we re-batch to
`(B, L_max, 256)` and run two parallel residual TCN stacks:

- 4 blocks per direction, channels `[64, 128, 256, 512]`, dilations `[1, 2, 4, 8]`, kernel 3.
- Forward stack on `x`, backward stack on `flip_per_sequence(x, lengths)` — a length-aware reversal that keeps padding zeros in place so dilated convolutions don't leak one chain's padding into another's boundary residues.
- Outputs concatenated → `(B, L_max, 1024)` and LayerNormed.

Receptive field at the top: `2 · (1+2+4+8) · 2 + 1 = 61` residues per direction.

### 3.5 Classifier

[`classifier.py`](../esm_model/model/classifier.py). Standard 2-layer MLP:

```
Linear(1024, 256) → GELU → LayerNorm → Dropout → Linear(256, 2)
```

`softmax` on the (N, 2) logits gives `P(binding)` per residue.

### 3.6 Loss & optimizer (unchanged from previous architecture)

- **Cost-sensitive focal loss** with `γ ∈ [1.0, 3.0]` (tuned), `α₊ = N_neg / N_pos ≈ 5.43`.
- **Soft-MCC penalty** (differentiable Matthews correlation surrogate) added with weight `λ_MCC ∈ [0, 2]` (tuned); total loss is `L_focal − λ_MCC · MCC_s`.
- **Lion optimizer** (sign-of-momentum step), `lr ∈ [10⁻⁶, 3·10⁻⁵]`,
  `wd ∈ [10⁻⁵, 10⁻²]`, betas `(0.9, 0.99)`. See
  [`imbalance_optim.py`](../imbalance_optim.py).
- **LR schedule**: 5-epoch linear warmup → cosine annealing to `1e-6`.
- **Mixed precision**: `torch.amp.autocast("cuda")` + `GradScaler`. Coordinate and edge_attr tensors are explicitly cast to `h.dtype` at the EGNN/fusion call sites for dtype consistency under autocast.

### 3.7 Hyperparameters tuned by Optuna

12 keys, TPE sampler, median pruner, MCC-on-validation objective, 20 trials × 15 epochs:

```
lr, weight_decay, focal_gamma, lambda_mcc,
proj_dropout, fusion_dropout, fusion_heads ∈ {2,4,8},
gcn_layers ∈ [4, 10], gcn_dropout, tcn_dropout, clf_dropout,
batch_size ∈ {1, 2, 4}
```

`gcn_alpha` (initial-residual coefficient of the old DeepGCN) is gone —
EGNN doesn't use it.

---

## 4. What differs between the two models

Everything below is identical in `esm_model/` and `protbert_model/`:

- `model/fusion.py`, `model/gcn.py`, `model/tcn.py`, `model/classifier.py` (file-level identical, verified)
- `tune.py`, `test.py`, `analyze.py`, `train.py` algorithmic structure (loss, optimizer, training loop)
- HP search space, default HPs, optimizer settings, LR schedule, threshold logic

What differs:

| | esm_model | protbert_model |
|---|---|---|
| `PLM_DIM` | 2560 | 1024 |
| Multi-layer projection hidden `H` | 1024 | 512 |
| Embedding dir | `data/esm_multi/` | `data/protbert_multi/` |
| Projection module file | `esm_projection.py` | `protbert_projection.py` |

Both projection modules expose the same class name (`MultiLayerProjection`)
with the same constructor signature and the same forward contract — only the
default `H` differs.

---

## 5. Module / code map

```
ppi/Model/
├── data/
│   ├── fasta/{Train_335, Test_60, Test_315}.fa
│   ├── pdbs/*.pdb                          ← from pdb_download.py
│   ├── structural/                         ← from dataprep/dataprep.py
│   │   ├── Train_335_17D.csv               ← 17 features + 6 coords + meta
│   │   ├── Train_split_{train,val}.csv     ← derived (268 / 67)
│   │   ├── Test_60_17D.csv
│   │   └── Test_315_17D.csv
│   ├── struct_norm.npz                     ← from train.py first run
│   ├── esm_multi/*.pt                      ← (L, 6, 2560) fp16
│   └── protbert_multi/*.pt                 ← (L, 6, 1024) fp16
│
├── pdb_download.py                         ← FASTA → PDB
├── dataprep/dataprep.py                    ← PDB → structural CSV
├── extract_esm_multilayer.py
├── extract_protbert_multilayer.py
├── imbalance_optim.py                      ← Lion optimizer
│
├── esm_model/
│   ├── tune.py
│   ├── train.py
│   ├── test.py
│   ├── analyze.py
│   ├── createdatset.py                     ← PyG Dataset wrapper (caches edges)
│   ├── best_hp.json                        ← from tune.py
│   ├── checkpoints/best.pt                 ← from train.py
│   └── model/
│       ├── esm_projection.py
│       ├── fusion.py
│       ├── gcn.py                          ← EGNN (filename historical)
│       ├── tcn.py
│       └── classifier.py
│
└── protbert_model/
    └── (same structure, ProtBert-BFD specific dims)
```

---

## 6. Environment

Verified working (`ppi` conda env):

```
torch          2.3.0+cu121
torch_scatter  2.1.2+pt23cu121
torch_geometric 2.7.0
fair-esm       2.0.0       # ESM-2 3B
transformers   4.46.3      # ProtBert-BFD
biopython      1.86        # PDBParser, DSSP, ResidueDepth
scipy          1.15.3      # KDTree, ConvexHull, CubicSpline
sklearn        1.7.2       # all metrics + threshold scans
optuna         4.0.0       # TPE sampler + median pruner
pandas         2.3.3
numpy          2.2.6
```

DSSP binary on PATH (`mkdssp` ≥ 4.0). MSMS not required — residue depth uses
`scipy.ConvexHull` instead, no extra dep.

---

## 7. How to run

```bash
# One-time: regenerate everything (uses the new feature schema)
rm -f data/struct_norm.npz data/structural/{*.csv,*.partial}
rm -rf esm_model/checkpoints protbert_model/checkpoints
rm -f esm_model/best_hp.json protbert_model/best_hp.json
./dataprep.sh

# Train both in parallel (two GPUs ideal; one GPU works but slower)
./run_esmModel.sh & ./run_protbertModel.sh & wait
```

Each `run_*.sh` chains `tune → train → test → analyze` and writes
`logs/<model>.log`. Outputs land in `<model>/{checkpoints/best.pt,
training_log.json, test_results.json, analysis/}`.
