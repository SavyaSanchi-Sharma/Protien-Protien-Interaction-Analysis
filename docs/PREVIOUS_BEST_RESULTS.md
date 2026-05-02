# Previous best results — frozen reference

Snapshot taken **2026-05-02 03:10**, while the new (EGNN + geo-biased fusion + new
features) tuning run is in progress. This document captures the **best results
achieved before that change**, so we have a stable baseline to compare against
and a known-good configuration to revert to if needed.

The numbers below come from the run completed on **2026-05-01 11:13 → 18:49**
(esm_model 7.5h, protbert_model 7.6h end-to-end). The corresponding artifacts
on disk at the time of writing this doc:

- `esm_model/best_hp.json`, `esm_model/training_log.json`, `esm_model/test_results.json`, `esm_model/checkpoints/best.pt` (34.6 MB)
- `protbert_model/best_hp.json`, `protbert_model/training_log.json`, `protbert_model/test_results.json`, `protbert_model/checkpoints/best.pt` (24.6 MB)

These files **will be overwritten** when the in-progress run finishes; this
document is the durable record.

---

## 1. Headline numbers

### Best on each test set, at the production (saved) threshold

| Test set | Model | Threshold | F1 | **MCC** | AUROC | AUPRC | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Test_60** (n=13,141) | `esm_model` | 0.73 | 0.5589 | **0.4982** | 0.8533 | 0.6147 | 0.658 | 0.486 | 0.879 |
| Test_60 | `protbert_model` | 0.83 | 0.5264 | 0.4609 | 0.8391 | 0.5724 | 0.623 | 0.456 | 0.870 |
| **Test_315** (n=65,331) | `esm_model` | 0.73 | 0.5496 | **0.4900** | 0.8564 | 0.5967 | 0.628 | 0.489 | 0.885 |
| Test_315 | `protbert_model` | 0.83 | 0.5292 | 0.4663 | 0.8502 | 0.5762 | 0.604 | 0.471 | 0.880 |

### Same evaluation, but at the F1-optimal threshold (paper convention)

| Test set | Model | Threshold | F1 | MCC | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|
| Test_60 | `esm_model` | 0.51 | **0.5683** | 0.4884 | 0.8533 | 0.6147 |
| Test_60 | `protbert_model` | 0.68 | 0.5498 | 0.4591 | 0.8391 | 0.5724 |
| Test_315 | `esm_model` | 0.46 | **0.5656** | 0.4898 | 0.8564 | 0.5967 |
| Test_315 | `protbert_model` | 0.73 | 0.5494 | 0.4698 | 0.8502 | 0.5762 |

### vs published baselines

| Method | Year | Test_60 MCC | Test_315 MCC |
|---|---|---:|---:|
| GraphPPIS | 2022 | 33.7 | 33.6 |
| DeepProSite | 2023 | 37.9 | 35.5 |
| **ESGTC-PPIS (paper)** | 2025 | **42.7** | **45.3** |
| **Ours: `esm_model` (this doc)** | 2026-05-01 | **49.8** *(at saved threshold)* | **49.0** |
| **Ours: `protbert_model` (this doc)** | 2026-05-01 | 46.1 | 46.6 |

`esm_model` beat the published paper by **+7.1 MCC on Test_60** and **+3.7 MCC on Test_315** at the production threshold.

---

## 2. Architecture used

This was the architecture **before** the EGNN + geo-biased-fusion changes.
Files referenced are the state on 2026-05-01.

```
PLM (L, 6, D)  ──► proj ──► (L, 256) ──┐
                                        ├─► fusion(plm, struct, mask) ──► (L, 256) ──► DeepGCN ──► BiTCN ──► clf ──► (L, 2)
struct (L, 17) ────────────────────────┘     ▲
                                              │ no geometric bias
                                              │ no gating
                                              │ no multiplicative merge
```

| Component | Setting | File |
|---|---|---|
| PLM extraction | Multi-layer (6 layers) | `extract_esm_multilayer.py`, `extract_protbert_multilayer.py` |
| Multi-layer projection | ELMo-style scalar mix + MLP, hidden=1024 (esm) / 512 (protbert) | `model/{esm,protbert}_projection.py` |
| Cross-attention fusion | Bidirectional MHA (s2e + e2s), single-layer struct projection (17→256), simple linear merge `Linear([plm; struct])` | `model/fusion.py` |
| GNN encoder | **DeepGCN** with initial-residual term (GCNII-style), per-edge attention gate from `edge_attr` | `model/gcn.py` |
| BiTCN | 4 blocks, channels [64, 128, 256, 512], dilations [1, 2, 4, 8] | `model/tcn.py` |
| Classifier | Linear(1024, 256) → GELU → LN → Dropout → Linear(256, 2) | `model/classifier.py` |
| Loss | Focal (cost-sensitive, α₊ = N_neg/N_pos ≈ 5.43) − λ_MCC · soft_MCC | `train.py` |
| Optimizer | Lion (sign-of-momentum) | `imbalance_optim.py` |
| LR schedule | 5-epoch warmup → cosine annealing to 1e-6 | `train.py` |
| Mixed precision | autocast + GradScaler (fp16 on V100) | `train.py` |

### 17D structural feature set (the *original* one, with Poly_*)

```
RSA, ResFlex, Hydrophobicity, PackingDensity,
HSE_up, HSE_down,
Poly_bias, Poly_RSA, Poly_Flex, Poly_interaction,
BondAngle,
sin_phi, cos_phi, sin_psi, cos_psi, sin_omega, cos_omega
```

`Poly_bias` is the constant 1.0; `Poly_RSA = RSA`; `Poly_Flex = ResFlex`;
`Poly_interaction = RSA × ResFlex`. These four were removed in the new run
in favour of `ResidueDepth, SurfaceCurvature, ElecPotential, LocalPlanarity`.

### Edge features (same in both old and new pipelines)

- 16-D Gaussian RBF over Cα-Cα distance (cutoff 14 Å)
- 2-D `(sin θ, cos θ)` between the residue's backbone direction `(Cα − N)` and the edge direction
- Total `edge_attr_dim = 18`

Graph: undirected, 14 Å Cα-Cα cutoff, no self-loops.

---

## 3. Best hyperparameters (frozen)

### `esm_model` — found at Optuna trial 14 of 20 (val MCC = 0.5136)

```json
{
  "value": 0.5136320524037461,
  "params": {
    "lr": 2.9812282251334696e-05,
    "weight_decay": 0.0003829744481435841,
    "focal_gamma": 1.2950646082201973,
    "proj_dropout": 0.10627406994502112,
    "fusion_dropout": 0.11668600391439436,
    "fusion_heads": 8,
    "gcn_layers": 9,
    "gcn_alpha": 0.2691545055420094,
    "gcn_dropout": 0.23796209591792133,
    "tcn_dropout": 0.11902140476132765,
    "clf_dropout": 0.38604467233536727,
    "lambda_mcc": 1.977960685348398,
    "batch_size": 2
  }
}
```

### `protbert_model` — found at Optuna trial 16 of 20 (val MCC = 0.5108)

```json
{
  "value": 0.5107806208688125,
  "params": {
    "lr": 1.2791453106231851e-05,
    "weight_decay": 0.0004869302537251107,
    "focal_gamma": 2.036475116822786,
    "proj_dropout": 0.22122153600427188,
    "fusion_dropout": 0.1439868339542001,
    "fusion_heads": 2,
    "gcn_layers": 5,
    "gcn_alpha": 0.43721567580761067,
    "gcn_dropout": 0.09103949880201281,
    "tcn_dropout": 0.11397679784281059,
    "clf_dropout": 0.24397648165885627,
    "lambda_mcc": 1.5288357906457397,
    "batch_size": 1
  }
}
```

### What the HP search converged on

- `batch_size`: 1-2 (small enough that soft-MCC saw enough variety per step)
- `gcn_layers`: deep (9) for ESM, moderate (5) for ProtBERT — ESM's bigger embedding rewarded more graph depth
- `lambda_mcc` ≈ 1.5-2.0 — soft-MCC pressure was substantial in both
- `lr`: ~1e-5 to 3e-5 — mid-range of the Lion search space
- `gcn_alpha`: ~0.27 (esm) / ~0.44 (protbert) — both moderate initial-residual fractions

---

## 4. Training-time observations

Both models followed the **same convergence pattern**:

| | esm_model | protbert_model |
|---|---|---|
| Best epoch (val MCC) | **6 / 21** | **6 / 21** |
| Saved MCC-optimal threshold | 0.73 | 0.83 |
| val_loss at best | 1.255 | 0.548 |
| val AUC at best | 0.847 | 0.842 |
| val AUPRC at best | 0.612 | 0.586 |
| val F1_max at best | 0.575 | 0.561 |
| val MCC_max at best | **0.501** | 0.477 |
| Total epochs run (early-stopped) | 21 | 21 |
| TUNE phase wall-clock | 6.85 h | 6.83 h |
| TRAIN phase wall-clock | 37 min | 39 min |
| TEST phase wall-clock | 1.5 min | 1.5 min |

Both peaked at epoch 6, then plateaued and rode out 15 patience epochs.
A patience of ~8 would have ended training in ~14 epochs without quality
loss; left at 15 here for extra robustness.

---

## 5. Per-test-set deep dive (saved threshold = production number)

### `esm_model`

| | Test_60 | Test_315 |
|---|---|---|
| Threshold | 0.73 | 0.73 |
| Accuracy | 0.8789 | 0.8853 |
| Precision | 0.6580 | 0.6276 |
| Recall | 0.4858 | 0.4888 |
| F1 | 0.5589 | 0.5496 |
| **MCC** | **0.4982** | **0.4900** |
| AUROC | 0.8533 | 0.8564 |
| AUPRC | 0.6147 | 0.5967 |
| n residues | 13,141 | 65,331 |

### `protbert_model`

| | Test_60 | Test_315 |
|---|---|---|
| Threshold | 0.83 | 0.83 |
| Accuracy | 0.8705 | 0.8800 |
| Precision | 0.6228 | 0.6037 |
| Recall | 0.4559 | 0.4711 |
| F1 | 0.5264 | 0.5292 |
| **MCC** | **0.4609** | **0.4663** |
| AUROC | 0.8391 | 0.8502 |
| AUPRC | 0.5724 | 0.5762 |
| n residues | 13,141 | 65,331 |

### Vintage ablations (single-layer PLM, no structural fusion) — for reference

These are the `vintage/o_esm/` and `vintage/o_protbert/` historical baselines,
showing what the structural fusion + multi-layer extraction contributes.

| Model | Test_60 MCC | Test_315 MCC | Δ vs current best |
|---|---:|---:|---:|
| `vintage/o_esm` (ESM-2 3B alone, no fusion) | 0.295 | 0.253 | −0.20 / −0.24 |
| `vintage/o_protbert` (ProtBert alone, no fusion) | 0.179 | 0.153 | −0.28 / −0.31 |

So **structural fusion + multi-layer PLM extraction** contributes **~+0.20-0.31 MCC** depending on the PLM. That entire stack is preserved in the new run; only the *fusion operator* and the *graph encoder* changed.

---

## 6. Files containing the actual model weights

```
esm_model/checkpoints/best.pt      34.6 MB  ← esm_model state dict (proj, fusion, gcn, tcn, clf) + saved_thresh 0.73
protbert_model/checkpoints/best.pt 24.6 MB  ← protbert_model state dict + saved_thresh 0.83
```

These checkpoints are **specific to the old architecture**. They will not load
into the new EGNN + geo-biased-fusion pipeline because the module class
definitions changed (different parameter shapes for fusion's `merge`,
different module class for `gcn`).

If you want to **deploy or further evaluate this checkpoint**, restore the
old code from git:

```bash
# revert just the model code to its state on 2026-05-01
git log --oneline -- esm_model/model protbert_model/model
git show <commit-hash>:esm_model/model/fusion.py > esm_model/model/fusion.py
git show <commit-hash>:esm_model/model/gcn.py    > esm_model/model/gcn.py
# (mirror to protbert_model/, regenerate dataset, re-evaluate)
```

Or save the entire model+code snapshot as a frozen tar:

```bash
mkdir -p ../Model_snapshots/2026-05-01_pre-egnn
cp -r esm_model protbert_model docs imbalance_optim.py extract_*.py \
      ../Model_snapshots/2026-05-01_pre-egnn/
cd .. && tar -czf Model_snapshots/2026-05-01_pre-egnn.tar.gz Model_snapshots/2026-05-01_pre-egnn
```

---

## 7. How to reproduce these numbers exactly

If you need to roll back and reproduce these results:

1. **Code state**: revert to the commit immediately preceding the EGNN
   migration. As of writing, that's the commit before the 2026-05-02 changes
   to `esm_model/model/gcn.py` and `esm_model/model/fusion.py`.

2. **Data state**:
   - `data/structural/Train_335_17D.csv` etc. **must contain the original Poly_*** columns. The current CSVs (regenerated 2026-05-02) have the new feature schema; the old CSVs are gone unless you have a backup.
   - To regenerate the old CSVs you'd need to revert `dataprep/dataprep.py` to its pre-2026-05-02 state and re-run `dataprep.sh`.

3. **Hyperparameters**: drop the JSON blobs above into `esm_model/best_hp.json` and `protbert_model/best_hp.json`.

4. **Run**: `./run_esmModel.sh` and `./run_protbertModel.sh`. With seed=42, results should reproduce to within float32 round-off and Optuna RNG-state variance.

---

## 8. Recommendation

Treat this as the **fall-back baseline**. The new architecture currently in
training (EGNN + geo-biased-fusion + new features) needs to **beat the
numbers in §5** to be worth keeping. If the new run lands at:

- **Test_60 MCC ≥ 0.50 and Test_315 MCC ≥ 0.49** → the new architecture is at least as good; keep it.
- **Test_60 MCC ∈ [0.45, 0.50)** → marginal; ablate to figure out which change cost the most and selectively roll back.
- **Test_60 MCC < 0.45** → revert wholesale (use the rollback recipe in §7).

The numbers in this document are the bar.
