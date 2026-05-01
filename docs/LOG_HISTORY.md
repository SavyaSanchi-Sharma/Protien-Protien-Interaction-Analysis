# Log history

Snapshot of every log under `logs/` before it was cleared on 2026-05-01.
Captures what was run, when, and the headline outcome — so the raw 1-2 MB of
training transcripts don't have to be retained for reference.

---

## Index

1. [Run summary table](#1-run-summary-table)
2. [Active-pipeline runs (April-May 2026)](#2-active-pipeline-runs-april-may-2026)
3. [Vintage / ablation runs (April 2026)](#3-vintage--ablation-runs-april-2026)
4. [Early dev artefacts (February 2026)](#4-early-dev-artefacts-february-2026)
5. [How to regenerate](#5-how-to-regenerate)

---

## 1. Run summary table

| Date | Phase | Log file(s) | Outcome |
|---|---|---|---|
| 2026-02-01 → 02-03 | Early dev (PSSM pipeline, since abandoned) | `pssm_*`, `pdb_download_2026*`, `dataprep_2026*`, `esm_extract_2026*`, `train_2026*`, `esm_test_2026*`, `create_dataset_2026*` | Pipeline scaffolding before PLM-only pivot |
| 2026-04-29 02:44 | Dataprep — PDB download | `dataprep_pdb.log` | 454 PDB structures present locally |
| 2026-04-29 02:53 | Dataprep — 17D structural CSVs | `dataprep_structural.log` | Train_335, Test_60, Test_315 CSVs written |
| 2026-04-29 02:57 | Dataprep — ESM multi-layer | `dataprep_esm.log` | (L, 6, 2560) fp16 tensors → `data/esm_multi/` |
| 2026-04-29 02:58 | Dataprep — ProtBERT multi-layer | `dataprep_protbert.log` | (L, 6, 1024) fp16 tensors → `data/protbert_multi/` |
| 2026-04-29 06:14-06:18 | Vintage train+test (`o_protbert`, single-layer, no fusion) | `o_protbert_train.log`, `o_protbert_test.log` | Test_60 MCC = 0.18 (single-layer baseline) |
| 2026-05-01 11:13-18:45 | TUNE → TRAIN → TEST → ANALYZE — `esm_model` | `esm_model.log` | Test_60 MCC = **0.498** (val 0.501 @ epoch 6) |
| 2026-05-01 11:13-18:49 | TUNE → TRAIN → TEST → ANALYZE — `protbert_model` | `protbert_model.log` | Test_60 MCC = 0.461 (val 0.477 @ epoch 6) |

---

## 2. Active-pipeline runs (April-May 2026)

These are the runs that produced the current `best.pt` checkpoints in
[esm_model/checkpoints/](esm_model/checkpoints/) and
[protbert_model/checkpoints/](protbert_model/checkpoints/).

### 2.1 Data preparation (2026-04-29)

Re-runs of the four-stage `dataprep.sh` flow against the current FASTA
inputs. All four logs are deterministic (idempotent — second runs hit `[SKIP]
already exists` lines). Total fresh dataprep takes ~15 min on the workstation
GPU; subsequent runs return instantly.

| Stage | Log | Notes |
|---|---|---|
| 1. PDB download | `dataprep_pdb.log` | 454 unique PDB IDs across all FASTA splits |
| 2. 17D structural | `dataprep_structural.log` | Per-residue features written to `data/structural/*_17D.csv` |
| 3. ESM-2 3B multi-layer | `dataprep_esm.log` | Layers `[6, 18, 24, 30, 33, 36]`, fp16 |
| 4. ProtBERT-BFD multi-layer | `dataprep_protbert.log` | Layers `[5, 12, 18, 24, 27, 30]`, fp16 |

### 2.2 ESM model run (2026-05-01, full TUNE→TRAIN→TEST→ANALYZE)

Single transcript: `esm_model.log` (1,330 lines). Runs the four phases
back-to-back via [`run_esmModel.sh`](run_esmModel.sh).

**TUNE phase (11:13:31 → 18:04:53, ~6.9 h).**
20 Optuna TPE trials, MCC objective, 4 pruned. Best:

```
Trial #14: MCC=0.5136
{lr=2.98e-5, weight_decay=3.83e-4, focal_gamma=1.30, proj_dropout=0.106,
 fusion_dropout=0.117, fusion_heads=8, gcn_layers=9, gcn_alpha=0.269,
 gcn_dropout=0.238, tcn_dropout=0.119, clf_dropout=0.386,
 lambda_mcc=1.978, batch_size=2}
```

Written to [esm_model/best_hp.json](esm_model/best_hp.json).

**TRAIN phase (18:04:54 → 18:41:15, ~37 min, 21 epochs).**
Best val MCC at epoch 6, 15 patience epochs after. Final:

```
Best PR=0.6120 @ epoch 6
Best F1=0.5779 @ epoch 7
Best MCC=0.5011 @ epoch 6   ← saved as best.pt
Saved threshold τ*_MCC=0.73
```

**TEST phase (18:41:16 → 18:43:02, ~2 min, 78,472 residues across both sets).**

```
Test_60   AUROC=0.8533  AUPRC=0.6147  F1=0.5589  MCC=0.4982
Test_315  AUROC=0.8564  AUPRC=0.5967  F1=0.5496  MCC=0.4900
```

(at saved threshold 0.73; F1-optimal thresholds 0.51 / 0.46 give marginally
higher F1 but slightly lower MCC.)

**ANALYZE phase.** Wrote 8 PNGs + `summary.txt` to
[esm_model/analysis/](esm_model/analysis/).

### 2.3 ProtBERT model run (2026-05-01, full TUNE→TRAIN→TEST→ANALYZE)

Same flow, log: `protbert_model.log` (1,318 lines).

**TUNE.** 20 Optuna TPE trials, 4 pruned. Best:

```
Trial #16: MCC=0.5108
{lr=1.28e-5, weight_decay=4.87e-4, focal_gamma=2.04, proj_dropout=0.221,
 fusion_dropout=0.144, fusion_heads=2, gcn_layers=5, gcn_alpha=0.437,
 gcn_dropout=0.091, tcn_dropout=0.114, clf_dropout=0.244,
 lambda_mcc=1.529, batch_size=1}
```

Written to [protbert_model/best_hp.json](protbert_model/best_hp.json).

**TRAIN.** 21 epochs, best at epoch 6 (matching ESM's convergence pattern):

```
Best PR=0.5888 @ epoch 8
Best F1=0.5608 @ epoch 6
Best MCC=0.4767 @ epoch 6   ← saved as best.pt
Saved threshold τ*_MCC=0.83
```

**TEST.**

```
Test_60   AUROC=0.8391  AUPRC=0.5724  F1=0.5264  MCC=0.4609
Test_315  AUROC=0.8502  AUPRC=0.5762  F1=0.5292  MCC=0.4663
```

(at saved threshold 0.83; F1-optimal at 0.68 / 0.73.)

**Cross-model take.** ESM beats ProtBERT by +3.7 MCC on Test_60 and +2.4 on
Test_315. Both peak at val-epoch 6 then plateau — patience could be shortened
to ~8 with no quality loss.

---

## 3. Vintage / ablation runs (April 2026)

Pre-architecture-rev runs of the single-layer-PLM, no-fusion ablations now
parked in [vintage/](vintage/). Useful as the "what does the model look like
without structural fusion?" baseline.

### `o_protbert` (2026-04-29 06:14-06:18)

`o_protbert_train.log`, `o_protbert_test.log`, `o_protbert_tune.log` (empty).

**Training** ran 27 epochs before early-stop, best val at epoch 12:

```
Best PR=0.3159 @ epoch 12
Best F1=0.3584 @ epoch 17
Best MCC=0.2194 @ epoch 12
```

**Test:**

```
Test_60   AUROC=0.6876  AUPRC=0.2907  F1=0.3037  MCC=0.1789  (saved τ=0.64)
Test_315  AUROC=0.6701  AUPRC=0.2483  F1=0.2737  MCC=0.1530
```

≈ +28 MCC delta from "single-layer ProtBERT alone" → "multi-layer ProtBERT
+ structural cross-attention fusion". The structural fusion is the dominant
contributor, in line with the ablation table in [COMPARISON.md §5](COMPARISON.md).

`o_esm` was tuned + tested in the same session; numbers preserved in
[vintage/o_esm/test_results.json](vintage/o_esm/test_results.json) (Test_60
MCC ≈ 0.29 — better than `o_protbert` thanks to the larger PLM, but still
~20 points below the fused models).

---

## 4. Early dev artefacts (February 2026)

Predates the PLM-only pivot. The original plan included PSSM (position-specific
scoring matrix) + HMM features — these were extracted via PSI-BLAST against
UniRef-90, which is what the long log timestamps reflect.

| Log file | Content |
|---|---|
| `pssm_setup_2026020*.log` | PSI-BLAST DB build / setup attempts (the 735 KB log captures the BLAST iteration noise) |
| `pssm_extraction.log` | Full PSSM extraction across Train_335 + Test_60 + Test_315 (~12 hours, 0.02 proteins/sec) |
| `pssm_extract_20260201_015336.log` | Mid-run snapshot |
| `pssm_setup_current.log` | Last-run PSSM setup status |
| `pdb_download_20260201_*.log` | First successful PDB downloads (FASTA → mmCIF/PDB) |
| `dataprep_20260201_*.log` | First structural-CSV runs (DSSP, polynomial features) |
| `esm_extract_20260201_*.log` | Single-layer ESM-2 3B extraction (`data/esm/*.pt`) — superseded by multi-layer |
| `create_dataset_20260201_*.log` | PyG `Dataset` construction sanity checks |
| `train_20260201_*.log` | Two early training-run snapshots; later superseded by the architecture rev |
| `esm_test_20260201_*.log` | Early test runs against the single-layer pipeline |

These were retained while validating the migration from the
PSSM+HMM+single-layer pipeline to the current multi-layer-PLM-only one.
They are not regenerated by the current `dataprep.sh`.

---

## 5. How to regenerate

The clearable logs were re-generatable from these scripts:

```bash
# Full data prep — writes data/pdbs, data/structural, data/esm_multi, data/protbert_multi
./dataprep.sh

# Active-pipeline model runs (each does TUNE → TRAIN → TEST → ANALYZE)
./run_esmModel.sh        # → logs/esm_model.log
./run_protbertModel.sh   # → logs/protbert_model.log
```

Vintage runs (`o_esm`, `o_protbert`) have their data dependencies in
`data/esm/` and `data/protbert/` (single-layer outputs). Those scripts
([run_oEsm.sh](run_oEsm.sh), [run_oProtbert.sh](run_oProtbert.sh)) still
reference the pre-move paths and would need their working-directory
updated to `vintage/o_esm` / `vintage/o_protbert` before re-running.

The PSSM pipeline is no longer wired up; reviving it would require restoring
`extract_pssm.py` and the BLAST DB setup scripts from git history pre-Apr 28.
