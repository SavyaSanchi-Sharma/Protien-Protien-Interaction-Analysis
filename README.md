# Protein-Protein Interaction Site Prediction

Deep learning framework for predicting protein-protein interaction (PPI) sites at the residue level using multi-modal features.

## Overview

This model combines ESM-2 protein language model embeddings with structural and evolutionary features through a graph neural network architecture:

- **ESM-2 Embeddings**: 2560D sequence representations from Meta's 3B parameter model
- **Structural Features**: 16D/36D features including RSA, flexibility, hydrophobicity, PSSM (optional)
- **Graph Architecture**: GCN + Bidirectional TCN for spatial and sequential modeling

## Architecture

```
ESM-2 (2560D) ──→ Projection (256D) ──┐
                                        ├──→ Fusion (256D) ──→ GCN (6 layers) ──→ BiTCN ──→ Classifier
Structural (36D) ──────────────────────┘
```

**Components:**
- **ESM Projection**: 2560D → 256D with GELU + LayerNorm
- **Gated Fusion**: Learnable gate to combine ESM + structural features
- **GCN Encoder**: 6-layer graph convolution on protein structure (10Å spatial cutoff)
- **Bidirectional TCN**: 4-stage temporal convolution (64→128→256→512 channels)
- **Classifier**: Binary interface/non-interface prediction

## Quick Start

### 1. Installation

```bash
cd Model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PyTorch 2.9+ with CUDA
- 12GB+ GPU memory

### 2. Run Full Pipeline

```bash
cd Model
chmod +x run1.sh
./run1.sh
```

This script will:
1. Create virtual environment and install dependencies
2. Configure feature mode (16D structural or 36D with PSSM)
3. Create PyTorch dataset
4. Train model (~2-3 hours)
5. Test on benchmark dataset

**Outputs:**
- `checkpoints/best.pt`: Best model checkpoint
- `training_log.json`: Training metrics
- `Test_60_predictions.csv`: Test predictions
- `logs/`: Detailed logs

### 3. Optional: Add PSSM Features (Recommended)

PSSM features improve performance by 15-25%. This requires PSI-BLAST and UniRef50 database (~100GB, 6-12 hours):

```bash
cd Model
chmod +x setup_pssm.sh
./setup_pssm.sh
```

This script will:
1. Install PSI-BLAST (if needed)
2. Download UniRef50 database (~50GB download, ~100GB uncompressed)
3. Extract PSSM features for all proteins (6-12 hours)
4. Merge features (16D → 36D)
5. Update training configuration

**Expected Performance:**
- **With PSSM (36D)**: F1: 0.72-0.75, MCC: 0.56-0.61, AUPR: 0.65-0.72
- **Without PSSM (16D)**: F1: 0.60-0.63, MCC: 0.45-0.50, AUPR: 0.55-0.60

## Data Structure

```
Model/
├── model/                    # Neural network components
│   ├── esm_projection.py
│   ├── fusion.py
│   ├── gcn.py
│   ├── tcn.py
│   └── classifier.py
├── data/
│   ├── fasta/               # Input sequences
│   ├── pdbs/                # PDB structures
│   ├── esm/                 # ESM-2 embeddings (.pt files)
│   ├── features/            # Merged features (36D)
│   ├── structural/          # Structural features (16D)
│   └── pssm/                # PSSM features (20D)
├── tests/                   # Test scripts
│   ├── test_model.py
│   └── test_esm.py
├── checkpoints/             # Saved models
├── logs/                    # Training logs
├── train.py                 # Training script
├── createdatset.py          # Dataset loader
├── extract_esm.py           # ESM-2 extraction
├── extract_pssm.py          # PSSM extraction
├── merge_features.py        # Feature merging
├── run1.sh                  # Full pipeline script
├── setup_pssm.sh            # PSSM setup script
└── requirements.txt
```

## Features

**Structural Features (16D):**
- RSA (Relative Solvent Accessibility)
- Residue Flexibility
- Hydrophobicity
- Packing Density
- HSE (Half-Sphere Exposure)
- Polymer properties
- Backbone angles (sin/cos of φ, ψ, ω)

**Evolutionary Features (20D - optional):**
- PSSM (Position-Specific Scoring Matrix) from PSI-BLAST

**Total:** 2560D (ESM) + 16D/36D (structural/evolutionary) = 2576D or 2596D

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 (early stopping: 15 patience) |
| Batch Size | 1 protein |
| Learning Rate | 1e-4 (warmup: 5 epochs, cosine decay) |
| Loss | Hybrid Focal Loss (γ=2.0, class-weighted) |
| Precision | Mixed (FP16/BF16) |
| Parameters | ~15M |

**Metrics:** ROC-AUC, PR-AUC, F1, MCC (multi-metric early stopping)

## Testing

Unit tests are available in `tests/`:

```bash
# Test trained model
python tests/test_model.py

# Test ESM extraction
python tests/test_esm.py
```

## Graph Construction

**Nodes:** Residues (C-alpha atoms)  
**Edges:**
- Sequential: (i, i+1) for all residues
- Spatial: distance < 10Å cutoff
- Weights: Gaussian kernel exp(-d²/2σ²) with σ=4Å

## Loss Function

Numerically stable focal loss with class weighting:

```python
L = -α * (1-p)^γ * log(p)     # positives
L = -p^γ * log(1-p)           # negatives
```

- **α**: Negative/positive sample ratio (handles imbalance)
- **γ**: Focusing parameter (2.0) - down-weights easy examples

## Dataset

- **Training:** 335 proteins
- **Validation:** 315 proteins
- **Test:** 60 proteins

Features are z-score normalized using training set statistics.

