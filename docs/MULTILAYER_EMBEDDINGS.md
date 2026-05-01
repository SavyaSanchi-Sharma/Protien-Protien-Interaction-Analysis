# Multi-Layer PLM Embedding Strategy for PPI-Site Prediction

**Context:** Train_335 / Test_60 / Test_315 PPI-site prediction. We do **not** use PSSM or HMM features (the paper does). Sequence-side input is fully delegated to a pretrained protein language model (PLM).

**Hypothesis:** A single PLM final-layer vector loses residue-level conservation and biochemistry signal. Extracting representations from multiple layers (early–mid–late) can reconstitute the PSSM/HMM signal we removed, plus add structural-precursor signal that the final layer compresses away.

---

## Table of Contents

1. [Models in the pipeline](#1-models-in-the-pipeline)
2. [What each layer encodes (biology)](#2-what-each-layer-encodes-biology)
3. [Layers chosen for extraction](#3-layers-chosen-for-extraction)
4. [Why this matters specifically without PSSM/HMM](#4-why-this-matters-specifically-without-psshmm)
5. [Tradeoffs and risks](#5-tradeoffs-and-risks)
6. [Expected accuracy impact](#6-expected-accuracy-impact)
7. [Files produced and disk cost](#7-files-produced-and-disk-cost)
8. [Sources / further reading](#8-sources--further-reading)

---

## 1. Models in the pipeline

| | **ESM-2 3B** (`esm2_t36_3B_UR50D`) | **ProtBert-BFD** (`Rostlab/prot_bert_bfd`) |
|---|---|---|
| Architecture | Encoder-only transformer | BERT-Large (encoder-only) |
| Transformer layers | **36** | **30** |
| Hidden dim | 2560 | 1024 |
| Attention heads | 40 | 16 |
| Params | 3.0B | 420M |
| Pretrain corpus | UniRef50 (~65M sequences, evolution-clustered) | BFD (~2.1B sequences, raw) |
| Pretrain objective | Masked LM (15% mask, BERT-style) | Masked LM (15% mask, BERT-style) |
| Tokenization | Character-level, 33-token alphabet | Character-level, spaces between residues |
| Context limit | 1024 tokens (we cap at 1022) | 2048 (we cap at 1022) |

ESM-2 3B brings 6× the hidden width and an evolution-aware corpus (UniRef50). ProtBert brings ~30× more pretraining data but lacks UR50's homology clustering, which empirically translates into weaker co-evolution / contact signal.

---

## 2. What each layer encodes (biology)

The MLM pretraining objective — *predict masked residue given context* — forces the network to build a feature stack that progressively integrates more context. Linear-probing studies decompose this stack as follows:

### Early layers (1 – ~⅓ of depth)

- **Residue identity, robustly encoded** — needed before anything else can be conditional on it.
- **Biochemistry**: hydrophobicity, charge, polarity, side-chain volume, aromaticity.
- **Immediate-neighbor patterns**: di- and tri-peptide context.
- **Behaviorally close to learned amino-acid descriptors** with light context.

### Early-mid layers (~⅓ – ½ of depth)

- **Short-to-medium-range sequence motifs**: helix-capping signatures, β-turn motifs, glycine-proline runs, charged patches (`KKKK`, `DDDD`).
- **Co-evolution signal starts to be usable** in models trained on aligned-cluster representatives (UR50D for ESM-2). This is the layer band that recovers PSSM-equivalent conservation information.

### Mid layers (~½ – ⅔ of depth)

- **Structural abstractions emerge**:
  - Secondary structure (α/β/coil) — Q3 recovery peaks here in linear probes.
  - Solvent accessibility (RSA) — recoverability peaks here.
- **Disorder / flexibility cues** become recoverable.

### Late layers (~⅔ – next-to-last)

- **Tertiary contact patterns** become recoverable from attention maps. This is where ESMFold reads structure from in ESM-2.
- **Binding-site / interaction-interface propensity** signal lives here for residue-level tasks.

### Final layer

- **Re-shaped to be useful for the MLM head** — biased toward "what residue would fit here?" rather than "what does this residue do?".
- **Useful for pretraining, often suboptimal for downstream tasks.** Multiple studies (linked below) show mid-to-late non-final layers beat the final layer for binding-site / structural / functional prediction.

---

## 3. Layers chosen for extraction

Strategic stratified sampling across each model's depth, hitting each band above:

### ESM-2 3B — `extract_esm_multilayer.py` → `data/esm_multi/`

Output shape per protein: `(L, 6, 2560)` fp16.

| Layer | Band | What it primarily contributes |
|---:|---|---|
| 6 | Early | Biochemistry: residue identity, hydrophobicity, charge, polarity |
| 18 | Early-mid | Local sequence context, motif patterns, conservation/co-evolution |
| 24 | Mid | Secondary-structure abstractions, RSA-precursor signal |
| 30 | Mid-late | Tertiary contact patterns, binding-site propensity |
| 33 | Late | Late-task abstractions, before MLM-head warp |
| 36 | Final | Parity with current pipeline (MLM-biased) |

### ProtBert-BFD — `extract_protbert_multilayer.py` → `data/protbert_multi/`

Output shape per protein: `(L, 6, 1024)` fp16.

| Layer | Band | What it primarily contributes |
|---:|---|---|
| 5 | Early | Biochemistry |
| 12 | Early-mid | k-mer / motif-level signal |
| 18 | Mid | Secondary structure / RSA precursor |
| 24 | Mid-late | Contact / binding-site precursor |
| 27 | Late | Late-task abstractions |
| 30 | Final | Parity with current pipeline |

---

## 4. Why this matters specifically without PSSM/HMM

The paper (ESGTC-PPIS, [docs/SOTA_COMPARISON.md](SOTA_COMPARISON.md)) explicitly injects two evolutionary feature blocks: **PSSM (20D position-specific scoring) and HMM profile (20D)**. These are essentially compressed, hand-designed versions of: *"residue identity at this position, weighted by its conservation across homologs."*

**We dropped both.** Our only sequence-side input is one ESM-2 layer 36 vector OR one ProtBert last-hidden vector. So we are betting that one final-layer vector contains everything PSSM+HMM contained, plus more. That's true *in capacity*, but a single late-layer vector emphasizes high-level abstractions over residue-level conservation/biochemistry.

**Adding early-mid layers reconstitutes the PSSM/HMM signal** — but as a *learned* version that's stronger than handcrafted PSSM (because the underlying corpus is 65M aligned-cluster representatives, not the BLAST hits PSSM is built from).

### Mapping PPI-site biology to layer bands

PPI interfaces have well-documented compositional and biochemical biases. Where each is recoverable:

| Biological signature of PPI sites | Where it lives in the embedding |
|---|---|
| Enriched in W, Y, R, H, M, F (large, aromatic, positively charged) | **Early (1–10)** — pure residue chemistry |
| Depleted in K, E, D at the core (rim residues differ) | **Early** |
| Hydrophobicity contrast vs surrounding surface | **Early–mid (1–18)** |
| Conservation hot-spots within otherwise variable surface patches | **Mid (12–24)** — direct MLM target |
| O-glycosylation-friendly stretches, MoRFs, disordered linkers | **Early–mid** — flexibility/disorder priors |
| Hot-spot residues forming O-rings around energetic core | **Late (24–33)** — structural abstraction |
| Surface curvature / packing density | **Mid–late (18–30)** |
| Contact propensity in the folded state | **Late (28–35)** |

Our 17D structural feature set (RSA, packing density, HSE_up/down, BondAngle, sin/cos φ/ψ/ω) covers the **geometric** side. It does **not** cover compositional biases or sequence-context conservation. **Early-mid layers fill that exact gap.**

---

## 5. Tradeoffs and risks

### Pros

1. **Recovers conservation signal lost when dropping PSSM/HMM.** Mid-band layers (12–24 of ESM-2 3B; 12–18 of ProtBert) directly encode position-specific conservation via the MLM objective on UR50D / BFD.
2. **Preserves biochemistry without distortion.** Early layers encode residue properties before they get reshaped into structural abstractions. Even if the downstream model could re-derive these, getting them straight from the embedding saves the network capacity for learning PPI-specific patterns.
3. **Adds structural-precursor signal that complements 17D struct features.** Mid-layer secondary-structure-like and RSA-precursor signal is finer-grained than the scalar 17D RSA / HSE columns.
4. **Layer-mixing is a low-cost intervention.** Storage and compute scale linearly in number of layers; no retraining of the PLM is needed.

### Cons

1. **Redundancy with our 17D struct.** Hydrophobicity / RSA are already in 17D as scalars. Early-layer features encode them across thousands of dims with much more nuance — net little harm, but the 17D becomes a smaller fraction of the input and the gated fusion has more work to do.
2. **More dimensions for the model to weight on a 268-protein training set.** This is a real overfitting cost. The protbert regression we just diagnosed in [REGRESSION_ANALYSIS.md](REGRESSION_ANALYSIS.md) was triggered partially by under-regularized HPs; adding 6× more input channels makes proper regularization more critical, not less.
3. **Disorder/flexibility cues are double-edged.** Early-mid layers will surface "this residue is in a flexible loop." PPI sites can be ordered (rigid interface) or disordered (MoRFs in IDPs). A model trained on Train_335 picks up the dominant pattern; if test sets have a different distribution, mild domain shift.
4. **Final layer is no longer special.** If we treat all layers symmetrically, we lose the implicit "this is the most-processed view" bias, which can be a useful default for downstream heads.

---

## 6. Expected accuracy impact

Honest estimate based on published evidence + our specific situation:

**Most likely: +1 to +3 MCC on Test_60 and Test_315.** That's the consensus range for layer-mixing on residue-level tasks.

### Why we expect the upper half (+2 to +3) rather than the lower half (+1)

1. **No PSSM/HMM in our pipeline.** Published gains are reported in setups that already have explicit conservation features. Our gap-from-conservation is bigger, so the marginal value of early-mid layers should be larger.
2. **17D structural features are already strong.** The 2×2 ablation in [COMPARISON.md](COMPARISON.md) shows +17 to +30 MCC from struct alone. The complement to "structural" is "compositional/conservation," which is exactly what early-mid layers add.
3. **ESM-2 3B is large enough that early layers are richly populated.** Small models compress signal across layers, so layer-mixing gains less. 3B has redundancy spread across layers we can exploit.

### Why we might land at the lower end (+1 or even neutral)

1. **Final-layer ESM-2 already encodes much of this signal.** The compression isn't lossy enough to leave huge gains on the table. Empirically, going from layer-36-only to a learned mix of {12, 24, 36} usually adds 0.5–1.5 MCC on residue tasks, not 5.
2. **Our fusion + DeepGCN + BiTCN is already a deep network** that can re-derive layer-mid abstractions from layer 36. So we are competing with a downstream model that's already trying to recover early-layer info via feature engineering.
3. **268-protein training set + more input dims = overfitting risk.** If the projection layer naively expands to handle `(L, 6, 2560)`, parameter count grows and the protbert-style overfitting reappears unless regularization is tuned for the new input shape.

### Where the real upside lives

The biggest published gains from layer mixing happen when the downstream head respects the layer structure (e.g., per-layer scalar-mix weights, softmax-normalized) rather than treating the layer dimension as just "more channels" via vanilla concatenation.

---

## 7. Files produced and disk cost

### Scripts

| Script | Output dir | Tensor shape per protein | Format |
|---|---|---|---|
| [extract_esm_multilayer.py](../extract_esm_multilayer.py) | `data/esm_multi/` | `(L, 6, 2560)` | fp16 |
| [extract_protbert_multilayer.py](../extract_protbert_multilayer.py) | `data/protbert_multi/` | `(L, 6, 1024)` | fp16 |

### Disk cost (avg L≈196, ~700 proteins, fp16)

| | Per protein | Total |
|---|---:|---:|
| ESM-2 3B, 6 layers × 2560D | ~6 MB | ~4.2 GB |
| ProtBert, 6 layers × 1024D | ~2.4 MB | ~1.7 GB |

Both scripts skip-if-exists, so re-running is idempotent. Cast to float32 in the loader before feeding into the model.

---

## 8. Sources / further reading

### PLM layer probing — what each layer encodes

- [Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model* (Science, 2023)](https://www.science.org/doi/10.1126/science.ade2574) — original ESM-2 paper, includes layer-wise probing of secondary structure, contacts, and remote homology.
- [Layer Probing Improves Kinase Functional Prediction with Protein Language Models (arXiv 2024)](https://arxiv.org/html/2512.00376) — direct evidence that mid-to-late ESM-2 layers (20–33) beat the final layer.
- [ESM-2 Protein Embeddings Overview](https://www.emergentmind.com/topics/esm-2-protein-embeddings) — accessible summary of what layers encode.
- [Endowing Protein Language Models with Structural Knowledge (arXiv 2024)](https://arxiv.org/html/2401.14819v1) — analysis of where structural information lives.
- [In the twilight zone of protein sequence homology: do PLMs learn protein structure? (Bioinformatics Advances, 2024)](https://academic.oup.com/bioinformaticsadvances/article/4/1/vbae119/7735315) — depth of structural recoverability.

### ESM-2 vs ProtBert — head-to-head benchmarks

- [Protein-small molecule binding site prediction (J Cheminform, 2024)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00920-2) — ESM-2 MCC 0.535 vs ProtBert 0.352 on UniProtSMB.
- [Encoding strategies in protein function prediction: a comprehensive review (Frontiers, 2025)](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2025.1506508/full) — comparative review, ESM-2 3B top of class.
- [Comparative assessment of PLMs for EC number prediction (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11866580/) — DNN-ESM2-3B beats DNN-ProtBert at all sequence-identity thresholds.
- [Medium-sized PLMs at transfer learning on realistic datasets (Sci Rep, 2025)](https://www.nature.com/articles/s41598-025-05674-x) — caveat: fine-tuned 3B doesn't always beat 650M.

### PPI-site prediction methods using PLMs

- [DeepProSite: structure-aware protein binding site prediction using ESMFold and pretrained language model (Bioinformatics, 2023)](https://academic.oup.com/bioinformatics/article/39/12/btad718/7453375) — prior SOTA, uses ProtT5-XL-U50 (1024D) + DSSP + ESMFold.
- [A Transformer-Based Ensemble Framework for the Prediction of PPI Sites (PMC, 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10528219/) — EnsemPPIS uses ProtBERT.
- [Advances in PLM for Nucleic Acid Protein Binding Site Prediction (MDPI, 2024)](https://www.mdpi.com/2073-4425/15/8/1090) — review of PLM applications to binding-site tasks.
- [ESM2_AMP: an interpretable framework for PPI prediction (Briefings in Bioinformatics, 2025)](https://academic.oup.com/bib/article/26/4/bbaf434/8242608) — recent ESM-2-based PPI work.

### Embedding fusion / layer mixing strategies

- [FusPB-ESM2: Fusion model of ProtBERT and ESM-2 for cell-penetrating peptide prediction (CompBioChem, 2024)](https://dl.acm.org/doi/10.1016/j.compbiolchem.2024.108098) — empirical case for stacking PLM embeddings.
- [Embeddings from PLMs predict conservation and variant effects (PMC, 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8716573/) — evidence that PLM embeddings encode conservation directly.
- [Fine-tuning protein language models boosts predictions across diverse tasks (Nat Commun, 2024)](https://www.nature.com/articles/s41467-024-51844-2) — tuning vs frozen, layer-selection considerations.

### General reference

- [GitHub: facebookresearch/esm](https://github.com/facebookresearch/esm) — official ESM repo with model cards.
- [HuggingFace: Rostlab/prot_bert_bfd](https://huggingface.co/Rostlab/prot_bert_bfd) — ProtBert-BFD model card.
- [HuggingFace: ESM docs](https://huggingface.co/docs/transformers/model_doc/esm) — Transformers ESM documentation.
- [ESM 2 — BioNeMo](https://nvidia.github.io/bionemo-framework/models/ESM-2/) — NVIDIA's reference page for ESM-2 model variants.
- [A survey of downstream applications of evolutionary scale modeling protein language models (Wiley, 2026)](https://onlinelibrary.wiley.com/doi/full/10.1002/qub2.70013) — recent survey covering ESM-2 downstream tasks.

---

## Bottom line

Multi-layer extraction is the cheapest sequence-side intervention available. Disk: ~6 GB total. Compute: one-time pass over `data/fasta/`. The biology says early-mid layers fill the conservation/biochemistry gap left by removing PSSM/HMM. The literature says expect +1 to +3 MCC on residue-level tasks. Our specific (no-PSSM, strong-struct, 268-protein) setup leans toward the upper end of that range, conditional on the downstream model treating the layer dimension as a learnable weighting rather than a vanilla concatenation.
