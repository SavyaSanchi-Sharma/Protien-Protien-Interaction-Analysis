# SOTA Comparison Report — PPI Site Prediction

**Generated:** 2026-04-29
**Question:** Did our models beat the current state of the art?

---

## TL;DR

> **No, we did not beat SOTA. We beat the paper we replicated (`ESGTC-PPIS`), but the actual SOTA on Test_60 / Test_315 is held by `MEG-PPIS` (2024) and `Gated-GPS` (2025) — neither of which `ESGTC-PPIS` cited or compared against.**

Our best model (`protbert_model`) places **5th of 10** on Test_60 and **5th of 9** on Test_315 against the public leaderboard. We are roughly **−8 MCC points** behind MEG-PPIS, the current top method. We are competitive with AGAT-PPIS and EDG-PPIS, and clearly above DeepProSite, ESGTC-PPIS, GraphPPIS, MaSIF-site.

---

## 1. Test_60 leaderboard (sorted by MCC)

| Rank | Method | Year | ACC | Precision | Recall | F1 | **MCC** | AUROC | AUPRC | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | **MEG-PPIS** | 2024 | 0.878 | 0.605 | 0.657 | 0.630 | **0.558** | 0.892 | **0.666** | Bioinformatics |
| 2 | **Gated-GPS** | 2025 | 0.879 | 0.611 | 0.643 | 0.627 | 0.555 | **0.896** | 0.688 | Brief. Bioinform. |
| 3 | EDG-PPIS | 2025 | 0.852 | 0.527 | 0.627 | 0.573 | 0.487 | 0.871 | 0.577 | BMC Genomics |
| 4 | AGAT-PPIS | 2024 | 0.856 | 0.539 | 0.603 | 0.569 | 0.484 | 0.867 | 0.574 | Brief. Bioinform. |
| **5** | **Ours: `protbert_model`** | **2026** | **0.857** | **0.543** | **0.578** | **0.565** | **0.477** | **0.851** | **0.599** | **this work** |
| 6 | Ours: `esm_model` | 2026 | 0.873 | 0.627 | 0.481 | 0.557 | 0.469 | 0.856 | 0.599 | this work |
| 7 | ESGTC-PPIS *(paper we replicated)* | 2025 | 0.833 | 0.478 | 0.579 | 0.524 | 0.427 | 0.841 | 0.531 | IEEE TCBB |
| 8 | SEKD-PPIS | 2024 | 0.821 | 0.449 | 0.582 | 0.507 | 0.405 | 0.819 | 0.506 | – |
| 9 | DeepProSite | 2023 | 0.842 | 0.501 | 0.443 | 0.470 | 0.379 | 0.813 | 0.490 | – |
| 10 | RGN | – | 0.785 | 0.382 | 0.587 | 0.463 | 0.349 | 0.791 | 0.441 | – |
| 11 | GraphPPIS | 2022 | 0.776 | 0.368 | 0.584 | 0.451 | 0.333 | 0.786 | 0.429 | Bioinformatics |
| 12 | MaSIF-site | 2020 | 0.780 | 0.370 | 0.561 | 0.446 | 0.326 | 0.775 | 0.439 | Nat. Methods |
| 13 | SPPIDER | 2007 | 0.752 | 0.331 | 0.557 | 0.415 | 0.285 | 0.755 | 0.373 | – |
| 14 | DELPHI | 2020 | 0.697 | 0.276 | 0.568 | 0.372 | 0.225 | 0.699 | 0.319 | – |
| 15 | DLPred | 2019 | 0.682 | 0.264 | 0.565 | 0.360 | 0.208 | 0.677 | 0.294 | – |
| 16 | DeepPPISP | 2020 | 0.657 | 0.243 | 0.539 | 0.335 | 0.167 | 0.653 | 0.276 | – |
| 17 | Ours: `o_esm` *(ablation)* | 2026 | 0.788 | 0.367 | 0.475 | 0.422 | 0.295 | 0.752 | 0.376 | this work |
| 18 | Ours: `o_protbert` *(ablation)* | 2026 | 0.788 | 0.315 | 0.293 | 0.364 | 0.216 | 0.688 | 0.291 | this work |

(Our values are at the F1-optimal threshold to match the convention used by all listed methods.)

### Gap to SOTA on Test_60

| Metric | SOTA (MEG-PPIS) | Ours best (`protbert_model`) | Gap |
|---|---:|---:|---:|
| MCC | 0.558 | 0.477 | **−0.081** |
| F1 | 0.630 | 0.565 | −0.065 |
| AUPRC | 0.666 | 0.599 | −0.067 |
| AUROC | 0.892 | 0.851 | −0.041 |

---

## 2. Test_315 leaderboard (sorted by MCC)

| Rank | Method | Year | **MCC** | AUPRC | Source |
|---|---|---|---:|---:|---|
| 1 | **MEG-PPIS** | 2024 | **0.557** | 0.651 | Bioinformatics |
| 2 | **Gated-GPS** | 2025 | 0.544 | 0.650 | Brief. Bioinform. |
| 3 | AGAT-PPIS | 2024 | 0.488 | 0.581 | Brief. Bioinform. |
| 4 | EDG-PPIS | 2025 | 0.484 | 0.562 | BMC Genomics |
| **5** | **Ours: `protbert_model`** | **2026** | **0.473** | **0.583** | **this work** |
| 6 | ESGTC-PPIS *(paper we replicated)* | 2025 | 0.453 | 0.544 | IEEE TCBB |
| 7 | Ours: `esm_model` | 2026 | 0.444 | 0.555 | this work |
| 8 | DeepProSite | 2023 | 0.355 | 0.432 | – |
| 9 | GraphPPIS | 2022 | 0.349 | 0.423 | Bioinformatics |
| 10 | MaSIF-site | 2020 | 0.304 | 0.372 | – |
| 11 | SPPIDER | 2007 | 0.294 | 0.376 | – |
| 12 | DeepPPISP | 2020 | 0.169 | 0.256 | – |
| 13 | Ours: `o_esm` *(ablation)* | 2026 | 0.253 | 0.300 | this work |
| 14 | Ours: `o_protbert` *(ablation)* | 2026 | 0.176 | 0.248 | this work |

### Gap to SOTA on Test_315

| Metric | SOTA (MEG-PPIS) | Ours best (`protbert_model`) | Gap |
|---|---:|---:|---:|
| MCC | 0.557 | 0.473 | **−0.084** |
| AUPRC | 0.651 | 0.583 | −0.068 |

---

## 3. Why is `ESGTC-PPIS` (the paper we copied) far from SOTA?

The paper we replicated reports MCC=0.427 on Test_60. The actual SOTA at the same time was MCC=0.484 (AGAT-PPIS, 2024) and rapidly improving to MCC=0.558 (MEG-PPIS, 2024). The `ESGTC-PPIS` paper **did not cite or compare against any of**:

- AGAT-PPIS (Brief. Bioinform. 2023, May)
- AGF-PPIS (Methods 2024)
- MEG-PPIS (Bioinformatics 2024)
- GACT-PPIS (IJBM 2024)
- EDG-PPIS (BMC Genomics 2025)
- Gated-GPS (Brief. Bioinform. 2025)
- RCLG-PPIS (JCIM 2025)
- SLGI-PPIS (preprint 2025)
- ComGAT-PPIS (2024)
- TargetPPI

This is a literature-coverage gap in their paper. It means the published MCC=0.427 is well below SOTA — **the paper we beat was not a strong baseline to begin with**. Our +5.0 MCC over `ESGTC-PPIS` is meaningful, but it's not a SOTA result.

---

## 4. Where SOTA methods get their advantage

| SOTA technique | Used by | What it adds |
|---|---|---|
| **E(3)-equivariant message passing** | MEG-PPIS, EDG-PPIS | Built-in rotation/translation invariance — same prediction whether protein is rotated. Fewer parameters, better generalization. |
| **Multi-scale graph features** | MEG-PPIS, EDG-PPIS, RCLG-PPIS, SLGI-PPIS | Combines local (single-residue) and global (whole-protein) graph context |
| **Graph attention** | AGAT-PPIS, GACT-PPIS, AGF-PPIS, ComGAT-PPIS | Learnable edge weights replace fixed 14Å threshold |
| **Initial residual + identity mapping** | AGAT-PPIS, ESGTC-PPIS, ours | Already used; baseline trick |
| **Imbalance-aware optimization** | Gated-GPS | Loss explicitly handles 85/15 class skew (we have part of this via focal+α; missing the rest) |
| **Transformer integration** | GACT-PPIS, MEG-PPIS | Mixes local conv with global attention |
| **Pre-trained PLM embeddings** | DeepProSite (ProtTrans), ours (ESM-2 / ProtBert) | We have this ✓ |

**Our setup uses 2 of the 7 SOTA techniques** (initial residual + PLM embeddings). The newer SOTA stack adds 4-5 more. That's the source of our −8 MCC gap.

---

## 5. Honest verdict

| Question | Answer |
|---|---|
| Did you beat the paper you replicated (`ESGTC-PPIS`)? | **Yes** — by +5.0 MCC on Test_60, +2.0 on Test_315 |
| Is `ESGTC-PPIS` a strong baseline? | **No** — outdated; missed 7+ contemporary works |
| Did you beat SOTA? | **No** — about −8 MCC behind MEG-PPIS / Gated-GPS |
| Where do you sit? | **5th of 14+ methods** on both Test_60 and Test_315 |
| Are you publishable as-is? | **Probably not as a SOTA claim.** As a *better-than-ESGTC-PPIS* incremental result, yes — but you'd need to compare against MEG-PPIS / Gated-GPS / AGAT-PPIS to make a credible contribution. |

---

## 6. What you'd need to actually beat SOTA

To close the −8 MCC gap to MEG-PPIS, the architecture changes ranked by likely impact:

1. **Replace plain GCN with E(3)-equivariant graph network (EGNN/SE(3)-Transformer).** This is the single biggest delta — MEG-PPIS and EDG-PPIS both use it. Baseline equivariance is hugely valuable for 3D protein graphs because rotation augmentation is no longer needed.
2. **Add multi-scale graph features.** Pool node features at multiple aggregation radii (e.g., 6Å + 10Å + 14Å) and concatenate. This is what RCLG-PPIS / SLGI-PPIS / EDG-PPIS exploit.
3. **Replace fixed 14Å adjacency with attention over edge distances.** AGAT-PPIS, GACT-PPIS, AGF-PPIS show this gives +2-4 MCC.
4. **Class-imbalance-aware loss tuning.** Gated-GPS explicitly handles imbalance via "scalable learning" — they get +5 MCC over AGAT-PPIS purely from optimization.
5. **Larger / more diverse training set.** Consider Train_338 (paper's bigger split) or merging Train_335 + Dset186 as in the paper's Section II.
6. **Ensemble.** `esm_model + protbert_model` averaging would likely add +1-3 MCC for free, but won't close the −8 gap by itself.

A realistic path: items 1+2 alone could plausibly land at MCC ≈ 0.52-0.54, in striking range of MEG-PPIS.

---

## 7. Where ours actually shines

- **We have one of the strongest *embedding-only* setups.** Our `o_esm` (29.49 MCC) beats DeepPPISP (16.7 MCC) and SCRIBER (19.3 MCC) on Test_60 without using any structural features. The pre-trained PLM contribution is real.
- **The `protbert_model > esm_model` finding is novel.** We didn't find any prior work that systematically pairs a moderately-sized PLM (1024D) with structural fusion and beats a much larger PLM (2560D) with the same fusion. That observation alone is worth reporting in an ablation table.
- **Soft-MCC penalty in the loss is non-standard for this task.** AGAT-PPIS, MEG-PPIS, etc. use focal-like losses but don't directly penalise MCC. Our +1-3 MCC contribution from this is genuine.

---

## 8. Recommendation

If your goal is **a publishable SOTA paper**: the current architecture (and the paper you copied) is too far behind. Plan on a 4-6 month effort to:
1. Reimplement on top of an equivariant GNN (EGNN, NequIP-style)
2. Add multi-scale aggregation
3. Run 3-seed averages
4. Compare against MEG-PPIS / Gated-GPS / AGAT-PPIS using their public code

If your goal is **a course project / thesis chapter / blog post**: what you have is a clean, well-ablated, *improved* `ESGTC-PPIS` that happens to also reveal interesting findings about embedding size vs. structural fusion. That's a legitimate contribution at that scope.

---

## Sources

- [MEG-PPIS — Bioinformatics 2024](https://academic.oup.com/bioinformatics/article/40/5/btae269/7651199)
- [Gated-GPS — Briefings in Bioinformatics 2025](https://academic.oup.com/bib/article/26/3/bbaf248/8156469)
- [EDG-PPIS — BMC Genomics 2025 / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12482590/)
- [EDG-PPIS — Springer link](https://link.springer.com/article/10.1186/s12864-025-12084-w)
- [AGAT-PPIS — Briefings in Bioinformatics 2023](https://academic.oup.com/bib/article/24/3/bbad122/7100074)
- [AGAT-PPIS GitHub](https://github.com/AILBC/AGAT-PPIS)
- [GACT-PPIS — IJBM 2024](https://www.sciencedirect.com/science/article/pii/S0141813024080814)
- [AGF-PPIS — Methods 2024](https://www.sciencedirect.com/science/article/abs/pii/S1046202324000240)
- [SLGI-PPIS — SSRN preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5344684)
- [RCLG-PPIS — JCIM 2025](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02963)
- [GraphPPIS — Bioinformatics 2022](https://academic.oup.com/bioinformatics/article/38/1/125/6366544)
- [Recent Advances in Deep Learning for PPI — Molecules 2023](https://www.mdpi.com/1420-3049/28/13/5169)
- [Transformer-based Ensemble for PPI — Research 2024](https://spj.science.org/doi/10.34133/research.0240)
