# PPI Site Prediction — Research Documentation

End-to-end mathematical and procedural description of the active pipeline
([esm_model/](esm_model/), [protbert_model/](protbert_model/)). Covers the
problem, the data, every transformation in the model, the loss surface, the
optimizer, the training loop, the hyperparameter search, threshold selection,
and the evaluation protocol.

---

## Table of Contents

1. [Problem statement](#1-problem-statement)
2. [Datasets and splits](#2-datasets-and-splits)
3. [Feature pipeline](#3-feature-pipeline)
   - 3.1 [17D structural per-residue features](#31-17d-structural-per-residue-features)
   - 3.2 [Multi-layer PLM embeddings](#32-multi-layer-plm-embeddings)
   - 3.3 [Graph construction and edge features](#33-graph-construction-and-edge-features)
4. [Model architecture](#4-model-architecture)
   - 4.1 [Multi-layer projection (ELMo-style scalar mix)](#41-multi-layer-projection-elmo-style-scalar-mix)
   - 4.2 [Bidirectional cross-attention fusion](#42-bidirectional-cross-attention-fusion)
   - 4.3 [DeepGCN encoder](#43-deepgcn-encoder)
   - 4.4 [Bidirectional temporal convolutional network (BiTCN)](#44-bidirectional-temporal-convolutional-network-bitcn)
   - 4.5 [Classifier head](#45-classifier-head)
5. [Loss function](#5-loss-function)
   - 5.1 [Cost-sensitive focal loss](#51-cost-sensitive-focal-loss)
   - 5.2 [Soft Matthews correlation penalty](#52-soft-matthews-correlation-penalty)
   - 5.3 [Combined objective](#53-combined-objective)
6. [Optimization](#6-optimization)
   - 6.1 [Lion optimizer](#61-lion-optimizer)
   - 6.2 [Learning-rate schedule](#62-learning-rate-schedule)
   - 6.3 [Gradient clipping and mixed precision](#63-gradient-clipping-and-mixed-precision)
7. [Training algorithm](#7-training-algorithm)
8. [Hyperparameter tuning](#8-hyperparameter-tuning)
   - 8.1 [Search space](#81-search-space)
   - 8.2 [TPE sampler and median pruner](#82-tpe-sampler-and-median-pruner)
   - 8.3 [Tuning objective](#83-tuning-objective)
9. [Threshold selection](#9-threshold-selection)
10. [Evaluation protocol](#10-evaluation-protocol)
11. [Current best numbers](#11-current-best-numbers)
12. [Pipeline files](#12-pipeline-files)

---

## 1. Problem statement

**Task.** Per-residue binary classification: for each amino-acid position in a
protein chain, decide whether it is a *protein-protein interaction (PPI)
site* — i.e. it makes contact with a partner chain in a biological complex.

Formally, let a protein chain of length \(L\) be represented by a sequence
\(s = (s_1, s_2, \dots, s_L)\) with \(s_i \in \mathcal{A}\) (the 20 standard
amino acids plus `X`). The dataset provides per-residue labels
\(y_i \in \{0, 1\}\). The goal is to learn a function

$$
f_\theta : (s, X^{\text{struct}}) \rightarrow \hat{p} \in [0, 1]^L
$$

where \(X^{\text{struct}} \in \mathbb{R}^{L \times 17}\) is a structural-feature
matrix derived from the chain's PDB structure, and \(\hat{p}_i\) is the
predicted probability that residue \(i\) is a PPI site.

**Difficulty drivers.**

1. **Class imbalance.** ≈ 15.6 % of residues are positive across the training
   set (8,204 positives / 52,746 residues). A constant-zero predictor scores
   84 % accuracy but 0 MCC.
2. **Dependency on context.** Whether residue \(i\) is an interaction site
   depends on its 3D neighbours, its local backbone geometry, and the global
   evolutionary context — all of which require sequence-aware *and*
   structure-aware reasoning.
3. **Small protein-level training set.** Only 268 train + 67 validation
   chains. The total residue count is large, but the *number of independent
   chains* is small, which is the unit that matters for generalization.

The chosen evaluation metric is the **Matthews correlation coefficient
(MCC)**, the most informative single number under heavy class imbalance.

---

## 2. Datasets and splits

| Dataset | Source | Chains | Residues | Pos rate | Use |
|---|---|---:|---:|---:|---|
| `Train_335` | ESGTC-PPIS supplementary | 335 | 66,366 | 15.6 % | Training (80/20 split below) |
| `Test_60`   | held-out | 60 | 13,141 | 11.0 % | Final evaluation |
| `Test_315`  | held-out | 315 | 65,331 | 11.6 % | Final evaluation |

The `Train_335` set is split protein-level (so the same chain never appears in
both partitions), with a fixed seed (42) and 20 % validation fraction:

$$
|\text{train}| = 268, \quad |\text{val}| = 67.
$$

The split is computed once and persisted to
`data/structural/Train_split_train.csv` and `Train_split_val.csv` so all
subsequent runs (tune, train, test, analyze) use exactly the same split. See
[`ensure_train_val_split`](esm_model/train.py).

**Train/val/test data are protein-disjoint by construction.**

---

## 3. Feature pipeline

Each residue is described by three feature streams: a 17-D structural vector,
a multi-layer PLM embedding, and a contact-graph adjacency. Edges carry a
geometric attribute vector. The full per-residue input is the concatenation of
these streams; node features and edge features are then fed to the model.

### 3.1 17D structural per-residue features

Computed once per chain by [`dataprep/dataprep.py`](dataprep/dataprep.py) from
the PDB file. Equations are documented at residue level in
[docs/equations.md](equations.md); summary:

| # | Column | Definition |
|---|---|---|
| 1 | `RSA` | DSSP solvent-accessible-surface-area, normalized: \(\text{RSA}_i = \text{clip}(\text{ASA}_i / \text{ASA}_{\max}(\text{aa}_i), 0, 1)\) |
| 2 | `ResFlex` | Min-max normalized B-factor: \(\text{Flex}_i = (B_i - \min B) / (\max B - \min B)\) |
| 3 | `Hydrophobicity` | Mean across 6 hydrophobicity scales (Kyte-Doolittle, Hopp-Woods, Eisenberg, Wimley-White, Hessa, Janin) |
| 4 | `PackingDensity` | Number of atoms within 3.5 Å of the C\(\alpha\), normalized |
| 5 | `HSE_up` | Half-sphere exposure, "up" side (along Cβ−Cα), within 8 Å |
| 6 | `HSE_down` | Half-sphere exposure, "down" side |
| 7 | `Poly_bias` | Constant 1.0 (acts as the bias term of a polynomial-interaction kernel) |
| 8 | `Poly_RSA` | RSA again, used as the linear term of the polynomial interaction |
| 9 | `Poly_Flex` | ResFlex again, linear term |
| 10 | `Poly_interaction` | RSA · ResFlex (cross term) |
| 11 | `BondAngle` | \(\theta_i = \arccos\big( (C_{i-1} - N_i) \cdot (O_{i+1} - N_i) / (\|\cdot\|\cdot\|\cdot\|) \big)\) |
| 12-13 | `sin_phi`, `cos_phi` | DSSP \(\phi\) angle (cubic-spline interpolated for missing) |
| 14-15 | `sin_psi`, `cos_psi` | DSSP \(\psi\) |
| 16-17 | `sin_omega`, `cos_omega` | Backbone torsion |

Scalar columns (1-10, 11) are z-score-normalized using statistics computed on
the training set only and saved to
[`data/struct_norm.npz`](data/struct_norm.npz):

$$
\hat{X}_{i, k} = \frac{X_{i, k} - \mu_k^{\text{train}}}{\sigma_k^{\text{train}}}.
$$

Angle features (`sin/cos`) are *not* normalized (they're already bounded in
\([-1, 1]\)).

The structural CSV also carries six **coordinate** columns per residue:
`CA_x, CA_y, CA_z, N_x, N_y, N_z`. These are *not* model inputs; they exist
so the data loader can build the contact graph and compute edge geometry
without re-parsing the PDB at training time. Adding them to the CSV makes
the per-residue coordinates inspectable alongside the rest of the features
([`dataprep/dataprep.py`](dataprep/dataprep.py) writes them via the
`COORD_NAMES` block).

### 3.2 Multi-layer PLM embeddings

Two PLMs are used independently — they produce two parallel models
([esm_model/](esm_model/) and [protbert_model/](protbert_model/)) that share
everything else.

| | ESM-2 3B (`esm2_t36_3B_UR50D`) | ProtBert-BFD (`Rostlab/prot_bert_bfd`) |
|---|---|---|
| Architecture | Encoder transformer | BERT-Large encoder |
| Layers | 36 | 30 |
| Hidden | 2560 | 1024 |
| Layers extracted | `[6, 18, 24, 30, 33, 36]` | `[5, 12, 18, 24, 27, 30]` |
| Pretrain | UniRef50 (evolution-clustered) | BFD (raw) |

For each chain of length \(L\), the extractor saves a tensor

$$
E \in \mathbb{R}^{L \times K \times D}, \qquad K = 6,\ D \in \{2560, 1024\}
$$

to `data/esm_multi/<pdb>_<chain>.pt` (or `data/protbert_multi/`), stored as
fp16 to halve disk. See [`extract_esm_multilayer.py`](extract_esm_multilayer.py)
and [`extract_protbert_multilayer.py`](extract_protbert_multilayer.py). The
chosen layer indices span the biochemistry-→-secondary-structure-→-contact-→-task
gradient documented in [MULTILAYER_EMBEDDINGS.md](MULTILAYER_EMBEDDINGS.md).

The 6 layers are *not* averaged in the extractor; the model learns the mixing
weights — see [§4.1](#41-multi-layer-projection-elmo-style-scalar-mix).

### 3.3 Graph construction and edge features

For each chain, a contact graph \(G = (V, E)\) is built where
\(V = \{1, \dots, L\}\) and an undirected edge \((i, j)\) exists iff

$$
\| x_i^{C\alpha} - x_j^{C\alpha} \|_2 \le 14 \text{ Å}, \qquad i \ne j.
$$

This is implemented with a `KDTree.query_ball_point` over the C\(\alpha\)
coordinates read from the structural CSV (see
[`PPISDataset.build_edges`](esm_model/createdatset.py)). If a chain has no
valid coordinates the graph falls back to a sequential chain \((i, i+1)\).

Each edge carries an attribute vector \(e_{ij} \in \mathbb{R}^{18}\) composed
of two parts:

**(a) 16-D Gaussian RBF over distance.** With centers
\(c_k = k \cdot \frac{14}{16},\ k = 0, \dots, 15\) and bandwidth \(\sigma = 14/16\):

$$
\phi_k(d_{ij}) = \exp\!\left(-\frac{(d_{ij} - c_k)^2}{2\sigma^2}\right),
\quad d_{ij} = \|x_i^{C\alpha} - x_j^{C\alpha}\|_2.
$$

**(b) 2-D backbone-vs-edge angle.** Let \(\hat{e}_{ij} = (x_j^{C\alpha} - x_i^{C\alpha}) / d_{ij}\) and
\(\hat{b}_i = (x_i^{C\alpha} - x_i^N) / \|x_i^{C\alpha} - x_i^N\|_2\). Then

$$
\cos\theta_{ij} = \hat{b}_i \cdot \hat{e}_{ij}, \qquad
\sin\theta_{ij} = \sqrt{\max(0, 1 - \cos^2\theta_{ij})}.
$$

If \(d_{ij}\) is degenerate or N-coordinates are missing, the angle defaults
to \((\sin, \cos) = (0, 1)\). The full edge attribute is

$$
e_{ij} = [\phi_0(d_{ij}), \dots, \phi_{15}(d_{ij}), \sin\theta_{ij}, \cos\theta_{ij}] \in \mathbb{R}^{18}.
$$

Edges also carry a scalar `edge_weight` initialized to 1.0; this is multiplied
by the GCN's learned per-edge gate ([§4.3](#43-deepgcn-encoder)).

---

## 4. Model architecture

For one chain of length \(L\), the forward pass is

```
PLM (L,K,D) ──► proj ──► (L, 256)
                                  ╲
                                   fusion ──► (L, 256) ──► GCN ──► BiTCN ──► clf ──► (L, 2)
                                  ╱
struct (L, 17) ───────────────────
```

All linear layers use **GELU** activation and **LayerNorm** (no BatchNorm; the
batch size during training is 1-4 chains, which is far too small for stable
batch statistics across heterogeneous lengths).

### 4.1 Multi-layer projection (ELMo-style scalar mix)

Defined in [`esm_model/model/esm_projection.py`](esm_model/model/esm_projection.py)
(and the parallel ProtBert version). Input
\(E \in \mathbb{R}^{L \times K \times D}\), output \(P \in \mathbb{R}^{L \times 256}\).

**Step 1 — Per-layer LayerNorm.** ESM-2's intermediate layers carry magnitudes
~100× larger than the final layer (no internal normalization). Without
normalization the scalar mix would simply route around the smaller-magnitude
late layers. So each layer is independently normalized:

$$
\tilde{E}_{i, k, :} = \text{LayerNorm}_k(E_{i, k, :}), \qquad k = 1, \dots, K.
$$

**Step 2 — Learned softmax mix.** With layer logits
\(w \in \mathbb{R}^K\) (initialized to **0**, so the initial mix is a
uniform \(1/K\) average) and a global learned scale \(\gamma \in \mathbb{R}\)
(initialized to **1**):

$$
\alpha_k = \text{softmax}(w)_k, \qquad
M_i = \gamma \sum_{k=1}^{K} \alpha_k \tilde{E}_{i, k, :} \in \mathbb{R}^D.
$$

Since the weights start uniform and each layer is on the same scale, the
optimizer can later concentrate weight on whichever layers help most.

**Step 3 — MLP projection to 256.** Two-layer MLP with hidden size
\(H \in \{1024, 512\}\) (1024 for ESM, 512 for ProtBert):

$$
P_i = \text{LayerNorm}\big( \text{GELU}(W_2 \cdot \text{Dropout}(\text{GELU}(W_1 M_i + b_1))+b_2 ) \big),
\quad P_i \in \mathbb{R}^{256}.
$$

Trainable parameters here include \(K\) LayerNorm pairs, the \(K\)-vector
\(w\), the scalar \(\gamma\), and the two MLP layers.

### 4.2 Bidirectional cross-attention fusion

Defined in [`esm_model/model/fusion.py`](esm_model/model/fusion.py). Replaces
the original gated fusion of the ESGTC-PPIS paper with a transformer-style
two-stream attention.

**Input.** Dense-batched tensors:

- \(P \in \mathbb{R}^{B \times L_{\max} \times 256}\) — projected PLM features.
- \(S^{\text{raw}} \in \mathbb{R}^{B \times L_{\max} \times 17}\) — raw structural features.
- \(M \in \{0, 1\}^{B \times L_{\max}}\) — `True` at valid residue positions
  (PyG `to_dense_batch` convention).

**Step 1 — Lift structural features.** Linear projection 17 → 256:

$$
S = S^{\text{raw}} W^{\text{up}}, \quad S \in \mathbb{R}^{B \times L_{\max} \times 256}.
$$

**Step 2 — Two cross-attention streams.** Standard scaled dot-product attention
with multiple heads (\(h = 2, 4, \) or \(8\), tuned). For
\(Q, K, V \in \mathbb{R}^{B \times L_{\max} \times 256}\),

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V,
\qquad d_k = 256/h.
$$

The two streams (`s2e` = struct queries PLM, `e2s` = PLM queries struct):

$$
\begin{aligned}
S' &= \text{LN}\big(S + \text{MHA}_{s2e}(S, P, P; M)\big), \\
S'' &= \text{LN}\big(S' + \text{FFN}_s(S')\big), \\
P' &= \text{LN}\big(P + \text{MHA}_{e2s}(P, S'', S''; M)\big), \\
P'' &= \text{LN}\big(P' + \text{FFN}_e(P')\big).
\end{aligned}
$$

The structural side gets updated *first*; the PLM side then queries the
already-updated struct stream. Each FFN is a two-layer GELU MLP with hidden
\(2 \cdot 256 = 512\) and dropout. Padding positions are zeroed at the end.

**Step 3 — Merge.**

$$
F = \text{Dropout}(\text{LayerNorm}( [P''\,;\,S''] W^{\text{merge}})), \quad F \in \mathbb{R}^{B \times L_{\max} \times 256}.
$$

**Why MHA's `key_padding_mask` is inverted.** PyTorch MHA expects `True` to
mean *ignore*, while PyG's `to_dense_batch` mask uses `True` for *valid*. The
code negates before passing in.

**Output.** \(F\) is reduced back to a node-list tensor by selecting valid
positions: \(H^{(0)} \in \mathbb{R}^{N \times 256}\) where \(N = \sum_b L_b\)
is the total number of residues in the batch.

### 4.3 DeepGCN encoder

Defined in [`esm_model/model/gcn.py`](esm_model/model/gcn.py). Stacks
\(L_g \in [4, 10]\) residual GCN layers with an initial-residual term à la
GCNII. Tuned LR of layers is one of the most important HP. Each layer:

**Input gate over edge attributes.** A two-layer MLP \(g\) maps each edge's
18-D attribute to a scalar in \((0, 1)\):

$$
\tilde{w}_{ij} = \sigma(g(e_{ij})), \qquad
w_{ij} \leftarrow w_{ij} \cdot \tilde{w}_{ij}.
$$

This gate is computed once per encoder call and reused by all layers.

**Single layer.** Let \(H^{(\ell)} \in \mathbb{R}^{N \times 256}\) be the
features at layer \(\ell\). Standard symmetric GCN propagation
([Kipf & Welling, 2017]):

$$
\hat{A} = \tilde{D}^{-1/2}(A + I)\tilde{D}^{-1/2},
\qquad \tilde{D}_{ii} = 1 + \sum_j A_{ij},
$$

with weighted edges from the gate. The layer output is

$$
\text{GCNLayer}(H^{(\ell)}) = \text{Dropout}(\text{LayerNorm}(\text{GELU}(\hat{A} H^{(\ell)} W^{(\ell)}))).
$$

**Initial-residual + dense residual.** With \(\alpha\) tunable in \([0.1, 0.7]\):

$$
\tilde{H}^{(\ell+1)} = (1 - \alpha)\, \text{GCNLayer}(H^{(\ell)}) + \alpha\, H^{(0)},
\qquad H^{(\ell+1)} = H^{(\ell)} + \tilde{H}^{(\ell+1)}.
$$

The initial-residual term is the GCNII trick: it keeps a fraction of the
*input* signal threaded through the entire stack, which controls
over-smoothing in deep GCNs. Empirically the optimizer prefers small
\(\alpha \approx 0.2-0.4\).

**Input projection.** The first step of `GCNEncoder.forward` is an MLP
\(\text{Linear} \to \text{GELU} \to \text{LayerNorm} \to \text{Dropout}\)
that maps \(\mathbb{R}^{256} \to \mathbb{R}^{256}\) and produces \(H^{(0)}\).

### 4.4 Bidirectional temporal convolutional network (BiTCN)

Defined in [`esm_model/model/tcn.py`](esm_model/model/tcn.py). Captures
sequence-order patterns the GCN does not see (the GCN treats each residue's
neighbours as a set, not a sequence).

**One TCN block.** With kernel size \(k = 3\), dilation \(d\), input/output
channels \((c_{\text{in}}, c_{\text{out}})\), and dropout \(p\):

$$
\begin{aligned}
y &= \text{Dropout}(\text{LayerNorm}(\text{GELU}(\text{Conv1d}_{k, d}(x)))), \\
y &= \text{Dropout}(\text{LayerNorm}(\text{GELU}(\text{Conv1d}_{k, d}(y)))), \\
\text{out} &= y + \text{ResProj}(x).
\end{aligned}
$$

`ResProj` is identity if \(c_{\text{in}} = c_{\text{out}}\), else a linear
projection. Conv1d uses zero-padding so length is preserved.

**Stack.** Four blocks with channel widths \([64, 128, 256, 512]\) and
dilations \([1, 2, 4, 8]\); receptive field at the top
\(= \sum_i 2 d_i (k-1) + 1 = 2(1+2+4+8)\cdot 2 + 1 = 61\) residues.

**Bidirectional combination.** Two independent stacks `f` (forward) and `b`
(reverse). For batched input \(x \in \mathbb{R}^{B \times L_{\max} \times C}\)
with sequence lengths \(\ell \in \mathbb{R}^B\):

$$
\begin{aligned}
x_f &= \text{ResidualTCN}_f(x \odot v), \\
x_b &= \text{flip}_\ell(\text{ResidualTCN}_b(\text{flip}_\ell(x) \odot v)), \\
\text{out} &= \text{LayerNorm}([x_f; x_b]) \in \mathbb{R}^{B \times L_{\max} \times 2C_{\text{top}}},
\end{aligned}
$$

where \(v \in \{0, 1\}^{B \times L_{\max} \times 1}\) is the validity mask
(1 inside the protein, 0 in padding). \(\text{flip}_\ell\) reverses each
sequence's valid prefix and leaves the padding zeros in place. **Why the
flip-per-sequence matters:** if `torch.flip` were used naively, padding zeros
would land at the front of short sequences, and the dilated convolutions of
the reverse branch would mix one protein's padding into another's boundary
residues. The per-sequence flip ensures the reverse branch sees exactly the
same boundary as a single-protein call.

The output channel width is \(2 \cdot 512 = 1024\).

### 4.5 Classifier head

Defined in [`esm_model/model/classifier.py`](esm_model/model/classifier.py).
Standard 2-layer MLP:

$$
\text{logits}_i = W_2\, \text{Dropout}(\text{LayerNorm}(\text{GELU}(W_1 h_i + b_1))) + b_2,
$$

with \(W_1 \in \mathbb{R}^{256 \times 1024}\), \(W_2 \in \mathbb{R}^{2 \times 256}\).
The two-class softmax then gives \(\hat{p}_i = \text{softmax}(\text{logits}_i)_1\)
(probability of "is binding site").

Total trainable parameters: ~6.1 M for ProtBERT, slightly more for ESM
(driven by the 2560→1024 first layer of the projection MLP).

---

## 5. Loss function

The training objective combines a residue-level focal classification loss
with a soft, differentiable surrogate for MCC. Both terms operate on the
same per-residue softmax probabilities.

### 5.1 Cost-sensitive focal loss

Let \(p_i = \text{softmax}(\text{logits}_i)_1 \in (0, 1)\). The standard
[Lin et al. 2017] focal loss is

$$
\mathcal{L}_i^{\text{focal}} = \begin{cases}
-\alpha_+ (1 - p_i)^\gamma \log p_i & y_i = 1, \\
-(p_i)^\gamma \log(1 - p_i) & y_i = 0.
\end{cases}
$$

The two free parameters:

- **\(\gamma \in [1, 3]\) (focusing).** Down-weights easy examples (already
  well-predicted) by a factor \((1 - p_i)^\gamma\) for positives and \(p_i^\gamma\)
  for negatives. \(\gamma = 0\) recovers cross-entropy; the tuned values land
  near \(\gamma \approx 1.3-2.0\).

- **\(\alpha_+\) (class weight).** Cost-sensitive multiplier for the positive
  class. Computed automatically as

$$
\alpha_+ = \frac{N_-}{N_+} \approx \frac{44542}{8204} \approx 5.43
$$

  on the training partition. Negatives are *not* weighted (their term gets
  weight 1.0), so the ratio between positive and negative gradients at
  identical confidence is \(\alpha_+ \approx 5.43\).

The batch-level focal loss is the per-residue mean:

$$
\mathcal{L}^{\text{focal}} = \frac{1}{N_b} \sum_{i \in \text{batch}} \mathcal{L}_i^{\text{focal}}.
$$

The implementation in
[`hybrid_focal_cost_loss`](esm_model/train.py) uses `log_softmax` for numerical
stability:

```
logp = log_softmax(logits, dim=1)[:, 1]
p    = exp(logp)
loss_pos = -alpha_pos * (1 - p[pos])^gamma * logp[pos]
loss_neg = -p[neg]^gamma * log1p(-p[neg])
focal    = (loss_pos.sum() + loss_neg.sum()) / N_b
```

### 5.2 Soft Matthews correlation penalty

The hard MCC is non-differentiable (it depends on thresholding). The soft
version drops the threshold and uses probabilities directly:

$$
\begin{aligned}
\widetilde{TP} &= \sum_i p_i y_i, & \widetilde{FN} &= \sum_i (1 - p_i) y_i, \\
\widetilde{FP} &= \sum_i p_i (1 - y_i), & \widetilde{TN} &= \sum_i (1 - p_i)(1 - y_i),
\end{aligned}
$$

$$
\text{MCC}_s = \frac{\widetilde{TP}\,\widetilde{TN} - \widetilde{FP}\,\widetilde{FN}}
                  {\sqrt{(\widetilde{TP}+\widetilde{FP})(\widetilde{TP}+\widetilde{FN})(\widetilde{TN}+\widetilde{FP})(\widetilde{TN}+\widetilde{FN}) + \varepsilon}},
$$

with \(\varepsilon = 10^{-7}\) preventing division by zero. \(\text{MCC}_s\) is
bounded in \([-1, 1]\) and continuously differentiable. It collapses to the
hard MCC when probabilities are pushed to \(\{0, 1\}\). See
[`soft_mcc`](esm_model/train.py).

### 5.3 Combined objective

The total loss is

$$
\boxed{ \mathcal{L} = \mathcal{L}^{\text{focal}} - \lambda_{\text{MCC}} \cdot \text{MCC}_s }
$$

with \(\lambda_{\text{MCC}} \in [0, 2]\) tuned. The minus sign comes from the
fact that we *minimize* loss while *maximizing* MCC. At the optimum the
gradients are pulling in two different directions:

- Focal pushes individual probabilities toward the correct extreme.
- Soft-MCC pushes the *whole-batch* confusion structure toward correlated
  prediction-vs-truth.

A side-effect of the combination: the total loss can become **negative**.
This is harmless because optimization only depends on the gradient direction
and not the absolute sign. In the logs the loss will routinely cross zero by
epoch 3-5 on the training set. The validation loss generally does *not* go
negative — its focal term is much larger because predictions are less
confident.

The tuned \(\lambda_{\text{MCC}}\) values for the latest run sit between
1.5 and 2.0, indicating soft-MCC pressure is roughly the same magnitude as
the focal term.

---

## 6. Optimization

### 6.1 Lion optimizer

Defined in [`imbalance_optim.py`](imbalance_optim.py). Lion (Chen et al. 2023,
[arXiv:2302.06675](https://arxiv.org/abs/2302.06675)) replaces AdamW. The
update rule, for each parameter \(\theta\) with gradient \(g\):

$$
\begin{aligned}
\theta &\leftarrow \theta \cdot (1 - \eta \cdot \lambda_{\text{wd}}), \\
u &= \text{sign}(\beta_1 m + (1 - \beta_1) g), \\
\theta &\leftarrow \theta - \eta \cdot u, \\
m &\leftarrow \beta_2 m + (1 - \beta_2) g.
\end{aligned}
$$

**Why Lion for class imbalance.** The update direction is the *sign* of a
momentum-blended gradient, so each parameter receives a step of magnitude
exactly \(\eta\) regardless of per-parameter gradient magnitude. Under heavy
imbalance the majority-class gradient direction dominates the per-parameter
gradient magnitude that AdamW's adaptive scaling normalizes. Lion sidesteps
that scaling step entirely — direction comes from sign(momentum), magnitude is
a constant.

Default betas \((\beta_1, \beta_2) = (0.9, 0.99)\). Tuning rules-of-thumb
relative to AdamW: \(\eta \approx \eta_{\text{AdamW}} / 10\),
\(\lambda_{\text{wd}} \approx 10\text{-}100 \cdot \lambda_{\text{wd}}^{\text{AdamW}}\).
Search-space bounds reflect this: \(\eta \in [10^{-6}, 3 \cdot 10^{-5}]\),
\(\lambda_{\text{wd}} \in [10^{-5}, 10^{-2}]\).

**Implementation note.** Lion is wired through `torch.amp.GradScaler.unscale_()`
identically to AdamW because the parameter-group layout is standard. Inside
`step` the running EMA `exp_avg` is *cloned before* the in-place blend so the
EMA used in step \(t+1\) is not corrupted by the sign computation at step \(t\).

### 6.2 Learning-rate schedule

Two-phase: linear warmup followed by cosine annealing.

- **Warmup, \(e < W = 5\):**
  $$ \eta_e = \eta_0 \cdot \frac{e + 1}{W}. $$
- **Cosine, \(e \ge W\):**
  $$ \eta_e = \eta_{\min} + \frac{\eta_0 - \eta_{\min}}{2}\left(1 + \cos\frac{\pi (e - W)}{T - W}\right), $$
  with \(\eta_{\min} = 10^{-6}\) and \(T = 100\).

Wired as two `torch.optim.lr_scheduler` instances (`LambdaLR` for warmup,
`CosineAnnealingLR` for cosine), switched per epoch. In practice early
stopping fires at epoch ≈ 21, so the cosine never completes its decay; the
LR ends near \(0.99 \cdot \eta_0\).

### 6.3 Gradient clipping and mixed precision

**AMP.** Forward and loss are computed under
`torch.amp.autocast(device_type="cuda")` (defaults to bfloat16 on Ampere+ or
float16 fallback). Backward uses `GradScaler` to prevent fp16 underflow
on small gradients:

```
scaler.scale(loss).backward()
scaler.unscale_(optimizer)         # gradients are now fp32, scaled back
clip_grad_norm_(params, 1.0)
scaler.step(optimizer)
scaler.update()
```

**Gradient clipping.** Global L2-norm clip at threshold 1:

$$
g \leftarrow g \cdot \min\!\left(1,\ \frac{1}{\|g\|_2}\right).
$$

Applied between `unscale_` and `step` so clipping operates on the unscaled
gradient. Prevents the loss-spike epochs (where soft-MCC saturates near ±1)
from sending the optimizer into a destructive step.

---

## 7. Training algorithm

Pseudocode for [`train.py.main()`](esm_model/train.py):

```
seed everything (default 42)
ensure_train_val_split(Train_335 → 268 train + 67 val)
ensure_normalization(train CSV → struct_norm.npz)
load HP (defaults overridden by best_hp.json if present)

train_ds, val_ds = PPISDataset(...)
α₊ = compute_dataset_alpha(train_ds)        # ≈ 5.43
modules = [proj, fusion, gcn, tcn, clf]

optimizer = Lion(params, lr=hp.lr, weight_decay=hp.wd)
warmup    = LambdaLR(optimizer, e ↦ min(1, (e+1)/5))
cosine    = CosineAnnealingLR(optimizer, T_max=95, η_min=1e-6)
scaler    = GradScaler("cuda")

best_mcc = -1
patience = 0
for epoch in 0 .. 99:
    for module in modules: module.train()
    for batch in train_loader:
        optimizer.zero_grad()
        with autocast:
            logits, y = forward_step(modules, batch)
            loss, focal_v, mcc_v = hybrid_focal_cost_loss(logits, y, α₊,
                                                          γ=hp.γ, λ=hp.λ_MCC,
                                                          return_components=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(params, 1.0)
        scaler.step(optimizer); scaler.update()

    val_loss, val_auc, val_pr, val_f1, val_mcc, val_f1_opt, val_mcc_opt, t_opt = evaluate(...)

    (warmup if epoch < 5 else cosine).step()

    if val_mcc_opt > best_mcc:
        best_mcc = val_mcc_opt
        save_checkpoint({...modules, "optimal_threshold": t_opt})    # MCC-optimal
        patience = 0
    else:
        patience += 1
        if patience >= 15:
            break
```

Notes:

- **Checkpoint trigger is val MCC, not val loss or val F1.** This guarantees
  `best.pt` ↔ best-MCC weights.
- The MCC-optimal threshold computed on the *validation* set at that epoch is
  saved into the checkpoint so production inference uses the same threshold
  the training run picked (see [§9](#9-threshold-selection)).
- The `evaluate` function returns *both* the F1-optimal threshold and the
  MCC-optimal threshold (independent searches). Only the MCC one is saved.
- All shapes in the forward pass are reconstructed by PyG's `to_dense_batch`
  (residue-list → padded-batch and back) — see
  [`forward_step`](esm_model/train.py).

---

## 8. Hyperparameter tuning

[Optuna](https://optuna.org/) drives the search.
[`tune.py`](esm_model/tune.py) creates a study, runs `trials` trials of
`epochs` mini-training runs, and writes the best parameters to
`best_hp.json` for `train.py` to pick up.

### 8.1 Search space

| HP | Type | Range | Notes |
|---|---|---|---|
| `lr` | log-float | `[1e-6, 3e-5]` | Lion convention (~AdamW/10) |
| `weight_decay` | log-float | `[1e-5, 1e-2]` | Lion convention (~AdamW × 10) |
| `focal_gamma` | float | `[1.0, 3.0]` | Below 1 ≈ cross-entropy; above 3 over-suppresses easy examples |
| `proj_dropout` | float | `[0.1, 0.4]` | Multi-layer projection MLP |
| `fusion_dropout` | float | `[0.05, 0.3]` | Cross-attention + FFN |
| `fusion_heads` | categorical | `{2, 4, 8}` | MHA heads |
| `gcn_layers` | int | `[4, 10]` | DeepGCN depth |
| `gcn_alpha` | float | `[0.1, 0.7]` | Initial-residual fraction |
| `gcn_dropout` | float | `[0.05, 0.3]` | Per-layer GCN |
| `tcn_dropout` | float | `[0.05, 0.3]` | Per-block BiTCN |
| `clf_dropout` | float | `[0.2, 0.5]` | Classifier head |
| `lambda_mcc` | float | `[0, 2]` | Soft-MCC penalty weight |
| `batch_size` | categorical | `{1, 2, 4}` | Chains per step |

Defaults (used if `best_hp.json` is missing) are at the centre of these ranges.

### 8.2 TPE sampler and median pruner

**Tree-structured Parzen Estimator (TPE).** TPE models the joint distribution
of HP given trial outcome. After observing past trials it splits them into
"good" (top \(\gamma\) fraction by objective) and "bad" sets, fits density
estimates \(\ell(x)\) and \(g(x)\) to each, and samples the next trial by
maximizing the ratio \(\ell(x) / g(x)\). This concentrates search where past
good trials have been while still exploring. With seed 42 the sampler is
deterministic given the same trial history.

**Median pruner.** Each trial reports its validation MCC after every epoch.
After 5 startup trials and 3 warmup epochs per trial, Optuna kills any trial
whose intermediate value at epoch \(e\) is below the median of all completed
trials at that same epoch. About 4 of 20 trials were pruned in each of the
latest searches.

### 8.3 Tuning objective

Each trial runs a shorter training loop (defaults to 15 epochs) with the same
forward pass and Lion optimizer. After each epoch:

1. Compute val probabilities.
2. Sweep MCC over thresholds \(\{0.01, 0.02, \dots, 0.99\}\) (see
   [`find_mcc_threshold`](esm_model/train.py)).
3. Report the best MCC at that epoch to Optuna.

The trial value is `max(reported MCC across epochs)`. Optuna maximizes this.
After all trials, the best trial's HP are written to `best_hp.json`. The full
training run then re-instantiates from these, trains up to 100 epochs (with
patience 15) on the same train/val split, and writes the final checkpoint.

A typical Optuna best result: 20 trials × 15 epochs × ~25 sec/epoch ≈ 7 GPU
hours per model. ~25 % of trials prune early.

---

## 9. Threshold selection

After softmax, each residue has a probability \(\hat{p}_i \in (0, 1)\). To
emit a hard prediction we apply a threshold \(\tau\): \(\hat{y}_i = \mathbb{1}[\hat{p}_i \ge \tau]\).
Different thresholds optimize different metrics, so the pipeline reports two:

**(a) MCC-optimal threshold (saved into `best.pt`).** Found by linear scan:

$$
\tau^*_{\text{MCC}} = \arg\max_{\tau \in \{0.01, 0.02, \dots, 0.99\}} \text{MCC}(\hat{y}(\tau), y)\quad \text{on val.}
$$

This is what `train.py` saves into the checkpoint. It is used at inference
time as the default for production / deployment.

**(b) F1-optimal threshold (computed at test time).** Found via the
precision-recall curve directly:

$$
F_1(\tau) = \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)},
\qquad \tau^*_{F_1} = \arg\max_{\tau \in \mathcal{T}_{\text{PR}}} F_1(\tau)
$$

where \(\mathcal{T}_{\text{PR}}\) is the discrete set of thresholds returned by
`sklearn.metrics.precision_recall_curve` (one per unique probability). This
is what the ESGTC-PPIS paper reports for fair comparison.

Both are printed by `test.py` for every test set:

```
=== Test_60 ===
  AUROC=0.8533  AUPRC=0.6147
  @saved_thresh=0.73 -> F1=0.5589 MCC=0.4982 P=0.6580 R=0.4858 Acc=0.8789
  @opt_thresh=0.51   -> F1=0.5683 MCC=0.4884
```

The two often differ by 0.02-0.05 in MCC (the saved threshold favours MCC,
not F1, so its F1 number is slightly lower than the F1-optimal version). For
evaluation against the paper, the F1-optimal numbers are used.

---

## 10. Evaluation protocol

[`test.py`](esm_model/test.py) performs the following for each held-out test
set:

1. Load `best.pt` (architecture-aware: each module loads its own state dict).
2. Build the dataset (validates that ESM/ProtBERT embeddings exist for every
   chain).
3. Run inference with `torch.no_grad()` + `autocast` on `batch_size=1`.
4. Compute probabilities, then for both \(\tau^*_{\text{MCC}}\) and
   \(\tau^*_{F_1}\):
   - Accuracy, precision, recall, F1, MCC at the threshold.
   - Threshold-independent: AUROC, AUPRC.
5. Save full results to `test_results.json`.

Threshold-independent metrics:

- **AUROC** = area under ROC curve (TPR vs FPR). Robust to class imbalance
  but can be over-optimistic when negatives dominate (most thresholds give
  low FPR easily).
- **AUPRC** (= average precision) = area under precision-recall curve.
  Far more sensitive under imbalance — drops fast when the model can't
  separate the rare positive class from negatives. Our preferred
  threshold-free metric for this task.

Per-residue confusion-matrix metrics:

- **F1** = harmonic mean of precision and recall.
- **MCC** = Matthews correlation coefficient:
  $$
  \text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \in [-1, 1].
  $$
  Defined to be 0 when any factor in the denominator is 0 (constant
  predictor, total miss, etc.). MCC is the headline metric because it is
  symmetric in classes and balanced under imbalance.

---

## 11. Current best numbers

Latest run, 2026-05-01. See [analyses](esm_model/analysis/summary.txt) and
[protbert_model/analysis/summary.txt](protbert_model/analysis/summary.txt).

**Validation (best epoch):**

| Model | Best epoch | val MCC | val AUPRC | val F1 | Saved \(\tau^*_{\text{MCC}}\) |
|---|---:|---:|---:|---:|---:|
| `esm_model` | 6 / 21 | 0.5011 | 0.6120 | 0.5753 | 0.73 |
| `protbert_model` | 6 / 21 | 0.4767 | 0.5863 | 0.5608 | 0.83 |

**Test sets at saved threshold:**

| Set | Model | AUROC | AUPRC | F1 | MCC |
|---|---|---:|---:|---:|---:|
| Test_60  | esm | 0.8533 | 0.6147 | 0.5589 | **0.4982** |
| Test_60  | protbert | 0.8391 | 0.5724 | 0.5264 | 0.4609 |
| Test_315 | esm | 0.8564 | 0.5967 | 0.5496 | **0.4900** |
| Test_315 | protbert | 0.8502 | 0.5762 | 0.5292 | 0.4663 |

vs ESGTC-PPIS paper (their best is **MCC 42.7 / 45.3** on Test_60 / Test_315
respectively at F1-optimal threshold). At F1-optimal we beat the paper by
~+5-7 points on both sets. See [COMPARISON.md §3-4](COMPARISON.md) for the
full leaderboard.

---

## 12. Pipeline files

```
ppi/Model/
├── data/
│   ├── fasta/{Train_335, Test_60, Test_315}.fa
│   ├── pdbs/*.pdb                          ← downloaded by pdb_download.py
│   ├── structural/                         ← per-residue: 17 features + 6 coord cols
│   │   ├── Train_335_17D.csv               ← (PDB, Chain, ResIdx, AA, Label, 17 feats, CA_xyz, N_xyz)
│   │   ├── Train_split_train.csv           ← derived (268 chains)
│   │   ├── Train_split_val.csv             ← derived (67 chains)
│   │   ├── Test_60_17D.csv
│   │   └── Test_315_17D.csv
│   ├── struct_norm.npz                     ← scalar mean/std from train split
│   ├── esm_multi/*.pt                      ← (L, 6, 2560) fp16 per chain
│   └── protbert_multi/*.pt                 ← (L, 6, 1024) fp16 per chain
│
├── pdb_download.py                         ← FASTA → PDB downloader
├── dataprep/dataprep.py                    ← PDB → 17D structural CSV
├── extract_esm_multilayer.py               ← FASTA → ESM-2 3B multi-layer
├── extract_protbert_multilayer.py          ← FASTA → ProtBert-BFD multi-layer
│
├── imbalance_optim.py                      ← Lion optimizer
│
├── esm_model/
│   ├── tune.py                             ← Optuna TPE search
│   ├── train.py                            ← final training (writes best.pt)
│   ├── test.py                             ← evaluation on Test_60 + Test_315
│   ├── analyze.py                          ← post-hoc plots + summary
│   ├── createdatset.py                     ← PyG Dataset wrapper
│   ├── best_hp.json                        ← from tune.py
│   ├── checkpoints/best.pt                 ← from train.py
│   ├── training_log.json                   ← per-epoch metrics
│   ├── test_results.json                   ← from test.py
│   ├── analysis/                           ← from analyze.py
│   └── model/
│       ├── esm_projection.py               ← multi-layer scalar mix
│       ├── fusion.py                       ← cross-attention fusion
│       ├── gcn.py                          ← DeepGCN encoder
│       ├── tcn.py                          ← BiTCN
│       └── classifier.py
│
└── protbert_model/                         ← parallel of esm_model/
    └── (same structure, dim 1024 instead of 2560)
```

The `vintage/` directory contains the older single-layer-PLM, no-fusion
ablations (`o_esm`, `o_protbert`) — preserved for historical comparison and
not used by the active pipeline.

---

## Cross-references

- Multi-layer PLM extraction rationale: [MULTILAYER_EMBEDDINGS.md](MULTILAYER_EMBEDDINGS.md)
- Comparison vs paper and SOTA: [COMPARISON.md](COMPARISON.md), [SOTA_COMPARISON.md](SOTA_COMPARISON.md)
- Per-feature equation derivations (RSA, Flex, HSE, etc.): [equations.md](equations.md)
- Run history and log inventory: [LOG_HISTORY.md](LOG_HISTORY.md)
