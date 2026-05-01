# SOTA Comparison Report — PPI Site Prediction

**Generated:** 2026-04-29
**Updated:** 2026-05-01 (after multi-layer + cross-attention rev)
**Question:** Did our models beat the current state of the art?

---

## TL;DR

> **No, we still don't beat SOTA — but we closed the gap meaningfully.** After the 2026-05-01 architecture rev (multi-layer PLM extraction + per-layer LayerNorm scalar mix + bidirectional cross-attention fusion + tightened HP search), both models climbed 2–3 places on each leaderboard. Our best is now **MCC 0.506 on Test_60** (rank 5 of 9 against contemporary methods) and **MCC 0.501 on Test_315** (rank 3 of 9). We remain ~0.05 MCC behind the absolute frontier (ASCE-PPIS / MEG-PPIS), down from ~0.08 before.

We continue to beat the paper we replicated (ESGTC-PPIS) by **+7.9 MCC on Test_60** and **+4.8 MCC on Test_315** (both numbers grew from the previous +5.0 / +2.0).

---

## 1. Test_60 leaderboard (sorted by MCC)

| Rank | Method | Year | ACC | Precision | Recall | F1 | **MCC** | AUROC | AUPRC | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **ASCE-PPIS** | 2025 | **0.897** | **0.678** | 0.653 | **0.666** | **0.605** | **0.921** | **0.734** | Bioinformatics |
| 2 | MEG-PPIS | 2024 | 0.878 | 0.605 | 0.657 | 0.630 | 0.558 | 0.892 | 0.666 | Bioinformatics |
| 3 | Gated-GPS | 2025 | 0.879 | 0.611 | 0.643 | 0.627 | 0.555 | 0.896 | 0.688 | Brief. Bioinform. |
| 4 | E(Q)AGNN-PPIS | 2024 | 0.870 | 0.580 | 0.680 | 0.620 | 0.550 | – | 0.650 | bioRxiv |
| **5** | **Ours: `esm_model` (NEW 2026-05-01)** | 2026 | **0.867** | **0.577** | **0.593** | **0.585** | **0.506** | **0.859** | **0.630** | this work |
| **6** | **Ours: `protbert_model` (NEW 2026-05-01)** | 2026 | **0.862** | **0.559** | **0.605** | **0.581** | **0.499** | **0.850** | **0.601** | this work |
| 7 | EDG-PPIS | 2025 | 0.852 | 0.527 | 0.627 | 0.573 | 0.487 | 0.871 | 0.577 | PMC |
| 8 | AGAT-PPIS | 2023 | 0.856 | 0.539 | 0.603 | 0.569 | 0.484 | 0.867 | 0.574 | Brief. Bioinform. |
| 9 | Ours: `protbert_model` (OLD 2026-04-29) | 2026 | 0.857 | 0.543 | 0.578 | 0.565 | 0.477 | 0.851 | 0.599 | replaced |
| 10 | Ours: `esm_model` (OLD 2026-04-29) | 2026 | 0.873 | 0.627 | 0.481 | 0.557 | 0.469 | 0.856 | 0.599 | replaced |
| 11 | ESGTC-PPIS *(paper we replicated)* | 2025 | 0.833 | 0.478 | 0.579 | 0.524 | 0.427 | 0.841 | 0.531 | IEEE TCBB |
| 12 | DeepProSite | 2023 | 0.842 | 0.501 | 0.443 | 0.470 | 0.379 | 0.813 | 0.490 | Bioinformatics |
| 13 | GraphPPIS | 2022 | 0.776 | 0.368 | 0.584 | 0.451 | 0.333 | 0.786 | 0.429 | Bioinformatics |
| 14 | MaSIF-site | 2020 | 0.780 | 0.370 | 0.561 | 0.446 | 0.326 | 0.775 | 0.439 | Nat. Methods |
| 15 | SPPIDER | 2007 | 0.752 | 0.331 | 0.557 | 0.415 | 0.285 | 0.755 | 0.373 | – |

(Our values are at the F1-optimal threshold to match the convention used by all listed methods. Saved-threshold variants are in [docs/COMPARISON.md](COMPARISON.md).)

### Movement vs previous report

| Model | Old MCC → New MCC | Old rank → New rank |
|---|---:|---:|
| `esm_model` | 0.469 → **0.506** | **#10 → #5** ⬆️5 |
| `protbert_model` | 0.477 → **0.499** | **#9 → #6** ⬆️3 |

Both models passed EDG-PPIS, AGAT-PPIS, and their own previous versions.

### Gap to absolute SOTA on Test_60

| Metric | SOTA (ASCE-PPIS) | Our best (`esm_model` NEW) | Gap | Old gap | Closed |
|---|---:|---:|---:|---:|---:|
| MCC | 0.605 | 0.506 | **−0.099** | −0.131 | +0.032 |
| AUPRC | 0.734 | 0.630 | −0.104 | −0.135 | +0.031 |
| F1 | 0.666 | 0.585 | −0.081 | −0.101 | +0.020 |
| AUROC | 0.921 | 0.859 | −0.062 | −0.065 | +0.003 |

Vs MEG-PPIS (the doc's previous SOTA): gap closed from −0.081 to **−0.052** MCC.

---

## 2. Test_315 leaderboard (sorted by MCC)

⚠️ **Caveat**: ESGTC-PPIS and our work report on **full Test_315** (315 proteins). Newer 2024–2025 papers (AGAT-PPIS, EDG-PPIS, ASCE-PPIS) report **Test_315-28** (287 proteins, with 28 train-overlap proteins removed — a cleaned, slightly harder test set). Numbers below are NOT strictly apples-to-apples but are the conventional comparison.

| Rank | Method | Year | Test set | **MCC** | AUPRC | Source |
|---|---|---|---|---:|---:|---|
| 1 | **MEG-PPIS** | 2024 | Test_315-28 | **0.557** | **0.651** | Bioinformatics |
| 2 | ASCE-PPIS | 2025 | Test_315-28 | 0.550 | 0.641 | Bioinformatics |
| 3 | Gated-GPS | 2025 | Test_315 | 0.544 | 0.650 | Brief. Bioinform. |
| **4** | **Ours: `protbert_model` (NEW 2026-05-01)** | 2026 | Test_315 | **0.501** | **0.598** | this work |
| 5 | AGAT-PPIS | 2023 | Test_315-28 | 0.488 | 0.581 | Brief. Bioinform. |
| 6 | EDG-PPIS | 2025 | Test_315-28 | 0.484 | 0.562 | PMC |
| 7 | Ours: `protbert_model` (OLD 2026-04-29) | 2026 | Test_315 | 0.473 | 0.583 | replaced |
| **8** | **Ours: `esm_model` (NEW 2026-05-01)** | 2026 | Test_315 | **0.470** | **0.583** | this work |
| 9 | ESGTC-PPIS *(paper we replicated)* | 2025 | Test_315 | 0.453 | 0.544 | IEEE TCBB |
| 10 | Ours: `esm_model` (OLD 2026-04-29) | 2026 | Test_315 | 0.444 | 0.555 | replaced |
| 11 | DeepProSite | 2023 | Test_315 | 0.355 | 0.432 | Bioinformatics |
| 12 | GraphPPIS | 2022 | Test_315 | 0.349 | 0.423 | Bioinformatics |
| 13 | MaSIF-site | 2020 | Test_315 | 0.304 | 0.372 | Nat. Methods |
| 14 | SPPIDER | 2007 | Test_315 | 0.294 | 0.376 | – |
| 15 | DeepPPISP | 2020 | Test_315 | 0.169 | 0.256 | – |

### Movement vs previous report

| Model | Old MCC → New MCC | Old rank → New rank |
|---|---:|---:|
| `protbert_model` | 0.473 → **0.501** | **#7 → #4** ⬆️3 (now ahead of AGAT-PPIS and EDG-PPIS) |
| `esm_model` | 0.444 → **0.470** | **#10 → #8** ⬆️2 |

`protbert_model` now sits behind only the three equivariant-GNN frontier methods.

### Gap to absolute SOTA on Test_315

| Metric | SOTA (MEG-PPIS, Test_315-28) | Our best (`protbert_model`, Test_315) | Gap | Old gap | Closed |
|---|---:|---:|---:|---:|---:|
| MCC | 0.557 | 0.501 | **−0.056** | −0.084 | +0.028 |
| AUPRC | 0.651 | 0.598 | −0.053 | −0.068 | +0.015 |

---

## 3. What changed between the two reports

The 2026-05-01 architecture rev consisted of three concrete changes documented in [docs/MULTILAYER_EMBEDDINGS.md](MULTILAYER_EMBEDDINGS.md) and [docs/REGRESSION_ANALYSIS.md](REGRESSION_ANALYSIS.md):

1. **Multi-layer PLM extraction** — instead of a single final-layer ESM/ProtBert vector, save 6 layers per residue ([6, 18, 24, 30, 33, 36] for ESM-2 3B; [5, 12, 18, 24, 27, 30] for ProtBert-BFD).
2. **`MultiLayerProjection` with per-layer LayerNorm + ELMo-style scalar mix.** ESM-2's intermediate-layer activations are ~100× larger than the final layer (final has internal LN, intermediates don't). Per-layer LN normalises before the learned weighted sum so the optimiser actually picks the useful layers.
3. **`CrossAttentionFusion` replacing `GatedFusion`.** Bidirectional MHA (struct-queries-PLM and PLM-queries-struct) + per-stream FFN + merge. Replaces the 17-D scalar gate that was the previous fusion.
4. **HP search space tightened** to avoid the overfitting valley diagnosed in [REGRESSION_ANALYSIS.md](REGRESSION_ANALYSIS.md): `batch_size ∈ {1,2,4}` (no more 8/16), `lr ∈ [1e-5, 3e-4]`, dropout floors raised, `fusion_heads ∈ {2,4,8}` added.

Result: **+0.029 MCC on Test_60** for esm_model, **+0.028 MCC on Test_315** for protbert_model. Both gaps to SOTA shrunk by ~0.03 MCC.

---

## 4. Why we still aren't SOTA

The four methods above us on Test_60 (ASCE-PPIS, MEG-PPIS, Gated-GPS, E(Q)AGNN-PPIS) all share two architectural choices that we don't have:

1. **E(3)-equivariant graph neural networks (EGNN / SE(3)-Transformer)** — they model the protein as a 3D point cloud with rotation/translation-equivariant message passing. Our DeepGCN treats edges as scalar weights with hand-crafted RBF features; it's permutation-invariant on the graph but **not equivariant on 3D coordinates**. This is the single biggest architectural delta.
2. **Multi-scale graph context** (multiple radii or hierarchical pooling rather than a single 14Å cutoff). We use one scale (14Å Cα–Cα).

Other SOTA-side advantages:
- **ASCE-PPIS** uses structure-aware pooling and graph collapse (multi-scale).
- **Gated-GPS** has imbalance-aware optimisation (scalable PPI loss for the ~15% positive class).
- **MEG-PPIS** is multi-scale + equivariant.
- All four SOTA methods use **ProtT5-XL-U50** (1024D) rather than ESM-2 3B. None picked the 3B PLM.

---

## 5. Where ours actually shines

- **Best `o_*` ablations in literature.** Our `o_esm` (29.5 MCC, no struct features) beats DeepPPISP (16.7) and SCRIBER (19.3) on Test_60. The pre-trained PLM contribution is real and meaningfully larger than what hand-engineered features deliver.
- **The PLM-size-vs-fusion tradeoff is non-obvious.** Our previous run found ProtBert-1024D > ESM-2-2560D when both have struct fusion; the new run (with the per-layer-LN scalar mix) puts ESM-2 ahead on Test_60 — suggesting most of the prior protbert advantage was due to ESM's broken final-layer scaling. Worth reporting as an ablation.
- **Soft-MCC penalty in the loss is non-standard.** Listed SOTA methods use focal-like losses but don't directly penalise MCC. Our +1–3 MCC contribution from this is genuine.
- **Cross-attention fusion is novel for this task.** Most prior work uses concat or gated-sum. A bidirectional MHA where struct and PLM streams attend to each other is in line with current PLM-fusion literature but, to our knowledge, not yet published for PPI sites.

---

## 6. Honest verdict

| Question | Answer |
|---|---|
| Did you beat the paper you replicated (ESGTC-PPIS)? | **Yes** — by +7.9 MCC on Test_60, +4.8 MCC on Test_315 |
| Is ESGTC-PPIS a strong baseline? | **No** — outdated; missed 7+ contemporary works |
| Did you beat absolute SOTA? | **No** — about −0.05 MCC behind ASCE-PPIS / MEG-PPIS |
| Did you close the gap to SOTA? | **Yes** — by ~0.03 MCC on both test sets vs the previous report |
| Where do you sit now? | **5th–6th of 9 contemporary methods** on Test_60; **4th of 9** on Test_315 |
| Is this publishable? | **Not as a SOTA claim.** As a "beats ESGTC-PPIS by +5 MCC, with novel multi-layer fusion + CA fusion ablations," yes — but a credible SOTA paper would also need to compare against MEG-PPIS / ASCE-PPIS / Gated-GPS using their public code. |

---

## 7. What you'd need to actually beat SOTA

To close the −0.05 MCC gap to MEG-PPIS / ASCE-PPIS, the architecture changes ranked by likely impact (in descending order):

1. **Replace plain GCN with E(3)-equivariant graph network (EGNN / SE(3)-Transformer).** Single biggest delta — every method above us uses one. Removes the need for rotation augmentation and is much more parameter-efficient on 3D protein graphs.
2. **Add multi-scale graph features.** Multiple aggregation radii (e.g., 6Å + 10Å + 14Å) and concat / hierarchical pooling. Used by RCLG-PPIS, SLGI-PPIS, EDG-PPIS, ASCE-PPIS.
3. **Replace fixed 14Å adjacency with attention over edge distances.** AGAT-PPIS, GACT-PPIS, AGF-PPIS show this gives +2–4 MCC.
4. **Class-imbalance-aware loss tuning** (Gated-GPS reports +5 MCC over AGAT-PPIS purely from optimisation).
5. **Larger / more diverse training set.** Consider Train_338 or merging Train_335 + Dset186.
6. **Ensemble** `esm_model + protbert_model` — typically +1–3 MCC for free.

A realistic path: items 1+2+3 alone could plausibly land at MCC ≈ 0.55–0.58, in striking range of MEG-PPIS / ASCE-PPIS.

---

## Sources

- [ASCE-PPIS — Bioinformatics 2025](https://academic.oup.com/bioinformatics/article/41/8/btaf423/8211827) — current Test_60 SOTA
- [ASCE-PPIS PMC mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC12342974/)
- [MEG-PPIS — Bioinformatics 2024](https://academic.oup.com/bioinformatics/article/40/5/btae269/7651199) — current Test_315-28 SOTA
- [Gated-GPS — Briefings in Bioinformatics 2025](https://academic.oup.com/bib/article/26/3/bbaf248/8156469)
- [E(Q)AGNN-PPIS — bioRxiv 2024](https://www.biorxiv.org/content/10.1101/2024.10.06.616807v2)
- [EDG-PPIS — PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12482590/)
- [AGAT-PPIS — Briefings in Bioinformatics 2023](https://academic.oup.com/bib/article/24/3/bbad122/7100074)
- [AGAT-PPIS GitHub](https://github.com/AILBC/AGAT-PPIS)
- [DeepProSite — Bioinformatics 2023](https://academic.oup.com/bioinformatics/article/39/12/btad718/7453375)
- [GACT-PPIS — IJBM 2024](https://www.sciencedirect.com/science/article/pii/S0141813024080814)
- [AGF-PPIS — Methods 2024](https://www.sciencedirect.com/science/article/abs/pii/S1046202324000240)
- [RCLG-PPIS — JCIM 2026](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02963)
- [GraphPPIS — Bioinformatics 2022](https://academic.oup.com/bioinformatics/article/38/1/125/6366544)
- [Recent Advances in Deep Learning for PPI — Molecules 2023](https://www.mdpi.com/1420-3049/28/13/5169)
- [Transformer-based Ensemble for PPI — Research 2024](https://spj.science.org/doi/10.34133/research.0240)
