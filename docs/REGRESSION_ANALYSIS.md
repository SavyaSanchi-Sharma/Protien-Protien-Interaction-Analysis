# Latest Run Analysis vs COMPARISON.md baseline

**Run date:** 2026-04-30
**Scope:** `esm_model` and `protbert_model` only (ablations excluded)
**Baseline:** [COMPARISON.md](COMPARISON.md)

---

## Table of Contents

1. [Headline summary](#1-headline-summary)
2. [esm_model — improved on Test_60](#2-esm_model--improved-on-test_60)
3. [protbert_model — regressed everywhere](#3-protbert_model--regressed-everywhere)
4. [vs the paper (ESGTC-PPIS)](#4-vs-the-paper-esgtc-ppis)
5. [Why protbert_model regressed (deep dive)](#5-why-protbert_model-regressed-deep-dive)
6. [Architecture: not the cause](#6-architecture-not-the-cause)
7. [Edge-attr gate: wired correctly, but suspect](#7-edge-attr-gate-wired-correctly-but-suspect)
8. [Recommended fixes](#8-recommended-fixes)

---

## 1. Headline summary

| Model | Direction | Best test MCC change |
|---|---|---|
| **`esm_model`** | **IMPROVED** on Test_60 | **+1.72 MCC** (46.90 → 48.62) — new best ever |
| **`protbert_model`** | **REGRESSED** on both test sets | −1.67 MCC on Test_60, −2.22 MCC on Test_315 |

Both models received the same code changes (edge-attr gate in GCN, refactored loss/AMP, +81 lines in `createdatset.py`). Only the HP search outcomes diverged.

---

## 2. esm_model — IMPROVED on Test_60

| Metric | Previous (COMPARISON.md) | Latest | Δ |
|---|---:|---:|---:|
| Val MCC | 0.4845 | 0.4707 | −0.014 |
| **Test_60 F1@F1-opt** | 55.71 | **57.24** | **+1.53** |
| **Test_60 MCC@F1-opt** | 46.90 | **48.62** | **+1.72** ⭐ |
| Test_60 AUROC | 85.64 | 85.29 | −0.35 |
| Test_60 AUPRC | 59.87 | 60.42 | +0.55 |
| Test_315 F1@F1-opt | 52.64 | 52.95 | +0.31 |
| Test_315 MCC@F1-opt | 44.39 | 44.68 | +0.29 |
| Test_315 AUROC | 83.61 | 83.63 | +0.02 |
| Test_315 AUPRC | 55.48 | 54.58 | −0.90 |

**Key HP shifts (from [logs/esm_model_train.log:2](logs/esm_model_train.log#L2)):**

| HP | Previous | Latest | Direction |
|---|---:|---:|---|
| `lr` | 4.73e-4 | **1.94e-4** | lower |
| `batch_size` | 2 | **2** | unchanged |
| `λ_mcc` | 1.15 | **1.69** | more MCC pressure |
| `focal_gamma` | 2.24 | 1.47 | lower |
| `gcn_layers` | 7 | 7 | unchanged |
| Saved threshold | 0.80 | 0.32 | less peaked outputs |

The new tune found a better-regularized configuration on the same architecture.

---

## 3. protbert_model — REGRESSED everywhere

| Metric | Previous | Latest | Δ |
|---|---:|---:|---:|
| Val MCC | 0.4788 | 0.4602 | −0.019 |
| Test_60 F1@F1-opt | 56.51 | 54.97 | −1.54 |
| **Test_60 MCC@F1-opt** | **47.72** | 46.05 | **−1.67** |
| Test_60 AUROC | 85.05 | 83.60 | −1.45 |
| Test_60 AUPRC | 59.92 | 57.33 | −2.59 |
| Test_315 F1@F1-opt | 55.04 | 53.48 | −1.56 |
| **Test_315 MCC@F1-opt** | **47.31** | 45.09 | **−2.22** |
| Test_315 AUROC | 85.17 | 84.25 | −0.92 |
| Test_315 AUPRC | 58.33 | 55.90 | −2.43 |

**Key HP shifts (from [logs/protbert_model_train.log:5](logs/protbert_model_train.log#L5)):**

| HP | Old (good) | New (bad) | Direction |
|---|---:|---:|---|
| `lr` | 1.55e-4 | **4.60e-4** | 3× higher |
| `batch_size` | 2 | **8** | 4× larger |
| `proj_dropout` | 0.28 | **0.056** | 5× less |
| `clf_dropout` | 0.36 | **0.10** | 3.6× less |
| `focal_gamma` | 2.58 | 1.53 | lower |
| `λ_mcc` | 1.23 | 1.18 | similar |
| Saved threshold | 0.38 | 0.70 | more peaked outputs |

Every change is in the *less-regularization* direction.

---

## 4. vs the paper (ESGTC-PPIS)

| Model | Test_60 MCC vs paper (42.7) | Test_315 MCC vs paper (45.3) |
|---|---:|---:|
| `esm_model` (latest) | **+5.9** ⭐ best so far | −0.6 |
| `esm_model` (previous) | +4.2 | −0.9 |
| `protbert_model` (latest) | +3.4 | −0.2 |
| `protbert_model` (previous) | +5.0 | +2.0 |

The new headline is `esm_model`'s **48.62 MCC on Test_60** — the strongest result across all runs to date, beating the paper by ~6 MCC points.

---

## 5. Why protbert_model regressed (deep dive)

### 5.1 The HP search landed in an overfitting valley (primary cause)

Each of the five HP shifts in §3 is a known overfitting risk on a small (268-protein) dataset:

- **batch_size 2 → 8.** Larger batches = lower gradient noise = less implicit SGD regularization.
- **lr 1.55e-4 → 4.60e-4.** 3× larger optimizer steps.
- **proj_dropout 0.28 → 0.056.** Almost no dropout right after the 1024-dim ProtBert embedding — the model can memorize per-residue ProtBert patterns.
- **clf_dropout 0.36 → 0.10.** Final classifier sees nearly raw BiTCN features.
- **focal_gamma 2.58 → 1.53.** Less down-weighting of easy negatives → model focuses less on the hard positive class that drives MCC.

All five at once is a recipe for memorization of the 268 training proteins.

### 5.2 The training log shows textbook overfitting

From [logs/protbert_model_train.log](logs/protbert_model_train.log):

```
ep  4: train=+0.24  val=+0.45      (warmup, healthy)
ep 12: train=-0.20  val=+0.42      ← best val MCC 0.4602
ep 27: train=-0.78  val=+2.57      ← early-stopped, train still falling
```

Train-vs-val loss divergence by ep 27 is **~3.35**. The previous run's gap at ep 25 was about **1.3**. Val MCC peaks at ep 12 and never recovers — sharp peak followed by capacity-driven drift.

### 5.3 Effective-step issue from the larger batch

With `batch_size=8`: **268/8 ≈ 34 optimizer steps per epoch** vs **134 steps** at `batch_size=2`. Combined with `WARMUP_EPOCHS=5` ([train.py:67](protbert_model/train.py#L67)), warmup completes in ~170 steps instead of ~670 — so the model hits peak LR **4× faster**. Combined with cosine decay being gentle once at peak, that means too many high-LR steps spent fitting once the model has found a workable solution.

### 5.4 The Optuna→final-train gap

`best_hp.json` reports `value: 0.4823` (val MCC during tuning), final `train.py` reached only `0.4602` — a **−0.022 gap**.

Likely sources:
- **Tune runs ~15 epochs max** with median pruner ([tune.py](protbert_model/tune.py)). The tuned trial likely peaked early and didn't have time to overfit. Final training to early-stop=27 epochs **gave the model the rope** to overfit.
- **Tune-vs-train RNG drift**: AMP scaler, dropout RNG, and DataLoader worker ordering can diverge slightly.

The gap means the tuned config was right on the edge of overfitting, not safely past it. The pruner's "best" was a fragile peak.

### 5.5 The saved threshold confirms overfitting calibration

Saved threshold went **0.38 → 0.70** — output distributions are more peaked.

| Set | Threshold | Precision | Recall |
|---|---:|---:|---:|
| Test_60 (previous, 0.38) | 0.38 | 54.3% | 57.8% |
| Test_60 (latest, 0.70) | 0.70 | 60.5% | 46.8% |

The new model is **abstaining on borderline residues** — confident on training-similar patterns, uncertain elsewhere. That's the calibration signature of overfitting: sure about what it memorized, unsure about everything else.

---

## 6. Architecture: not the cause

The architecture is essentially unchanged between runs. Both old and new `train.py` build:

```python
ProtBertProjection(1024, 512, 256)
GatedFusion(256, 17, 256)
GCNEncoder(256, 256, 8 layers)
BiTCN(256, [64, 128, 256, 512])
Classifier(1024)
```

The only architectural delta: a tiny `edge_proj` MLP added to [protbert_model/model/gcn.py:41-44](protbert_model/model/gcn.py#L41-L44):

```python
nn.Linear(edge_dim, edge_dim)  # 18×18
nn.GELU()
nn.Linear(edge_dim, 1)         # 18×1
```

This adds ~342 parameters — negligible.

The "~17M params" figure in COMPARISON.md was an unverified estimate. The trainer's actual measurement is **5,270,557 params** ([logs/protbert_model_train.log:22](logs/protbert_model_train.log#L22)). That's the true count, and it almost certainly was the same in the previous run too.

**`esm_model` improved with the same code changes**, which rules out architecture as the cause.

---

## 7. Edge-attr gate: wired correctly, but suspect

End-to-end plumbing (verified):

```
createdatset.py:122-124       # builds 18D edge_attr (16 RBF + sin,cos)
   ↓
DataLoader (PyG batches edge_attr automatically)
   ↓
train.py:188 / tune.py:66 / test.py:73
   gcn(h, data.edge_index, edge_weight=data.edge_weight, edge_attr=data.edge_attr)
   ↓
gcn.py:47-50                  # gate: score = sigmoid(MLP(edge_attr))
   ↓
gcn.py:53-54                  # passed as edge_weight into all 8 GCNConv layers
```

Two design issues that may interact badly with the new high-LR regime:

### 7.1 The gate halves message strength at initialization

The final `Linear(18, 1)` is randomly initialized → output logit ≈ 0 → `sigmoid(...)` ≈ 0.5 across all edges at init.

In the previous run (no gate), `edge_weight = ones` meant every edge contributed at full magnitude. Now every edge starts at ~0.5. Self-loops added by `GCNConv(add_self_loops=True)` use weight 1.0 and **bypass the gate**. So at init, every node receives full-strength self-loop + dampened-strength neighbor messages → the network behaves **closer to a per-node MLP** for the first several epochs and has to climb out of that hole.

### 7.2 With high LR, the gate becomes a memorization knob

With `lr=4.6e-4`, the gate's parameters can rapidly learn highly skewed per-edge weights tailored to the training set. With the old `lr=1.55e-4`, the gate would have had less room to "solve" training-protein edges by raw fitting.

This is plausibly meaningful but **not the dominant cause** — `esm_model` has the same gate and improved.

---

## 8. Recommended fixes

In order of cost/impact:

### A. Cheapest test: restore the old HPs and re-run

Swap `best_hp.json` back to the previous values (`lr=1.55e-4`, `batch_size=2`, `proj_dropout=0.28`, `clf_dropout=0.36`, `focal_gamma=2.58`, `λ_mcc=1.23`) and run `train.py` + `test.py`. If val MCC recovers to ≥ 0.477, the issue is fully isolated to the HP search.

### B. Constrain the search space in `tune.py`

The old config sat squarely in this box; the new one walked right out:

- `batch_size ∈ {2, 4}` (drop 8)
- `lr ∈ [1e-4, 3e-4]`
- `proj_dropout ∈ [0.15, 0.4]`
- `clf_dropout ∈ [0.2, 0.5]`

### C. Penalize fragile peaks in the tune objective

Currently the tune objective is the best val MCC during the trial — peak-snipping. Two safer alternatives:

- **Median of last 3 epochs' val MCC** instead of best.
- **Best val MCC penalized by train-val gap**: `val_mcc - α · max(0, train_loss_collapse_signal)`.

This would have rejected the `batch=8` / high-LR config because its peak was followed by collapse.

### D. Bias-init the edge gate to ~1.0

In [protbert_model/model/gcn.py:41-45](protbert_model/model/gcn.py#L41-L45), after constructing `self.edge_proj`:

```python
nn.init.zeros_(self.edge_proj[-1].weight)
nn.init.constant_(self.edge_proj[-1].bias, 4.0)   # sigmoid(4) ≈ 0.98
```

Makes the gate start as near-identity (no dampened init), and lets the model *learn* to attenuate specific edges if useful. Standard residual-style gate trick.

### E. Optional: A/B-test the gate

Temporarily pass `edge_attr=None` in [train.py:188](protbert_model/train.py#L188) (or `edge_dim=None` at construction in [train.py:223](protbert_model/train.py#L223)) and re-run with the *current* HPs. If val MCC recovers, the gate is contributing to the regression. If not, it's purely the HP regime.

---

## Bottom line

The new `protbert_model` regression is **HP-search drift, not an architecture or code regression.** The same code path produced an *improvement* on `esm_model` — the difference is that `esm_model`'s tune kept `batch_size=2` and lowered LR, while `protbert_model`'s tune jumped to `batch_size=8` with 3× higher LR and 5× less dropout, walking into an overfitting valley. The training log confirms classic overfitting dynamics (train collapses to −0.78, val climbs to +2.57, val MCC peaks at ep 12 and never recovers).

Fix A alone (restore old HPs) should recover the previous numbers. Fix B + C will prevent it from happening again.
