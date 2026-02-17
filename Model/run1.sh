#!/usr/bin/env bash
set -euo pipefail

# ============================================
# PPI Prediction Pipeline - 512D ESM + Features
# Supports: 16D (structural only) or 36D (with PSSM)
# ============================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python -u"

LOG_DIR="$PROJECT_ROOT/logs"
CKPT_DIR="$PROJECT_ROOT/checkpoints"

mkdir -p "$LOG_DIR" "$CKPT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# -----------------------------
# Create/activate venv
# -----------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment (.venv)"
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# -----------------------------
# Install dependencies
# -----------------------------
echo "[INFO] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# -----------------------------
# Environment variables
# -----------------------------
export CUDA_VISIBLE_DEVICES=0
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

# -----------------------------
# GPU check
# -----------------------------
echo "======================================"
echo "[INFO] GPU Status"
echo "======================================"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "[WARN] nvidia-smi not found (CPU mode?)"
fi
echo ""

# -----------------------------
# Detect feature mode
# -----------------------------
FEATURE_MODE="16D"
if [ -f "data/features/Train_335_36D.csv" ]; then
    FEATURE_MODE="36D"
    echo "[INFO] ✅ Found 36D merged features (structural + PSSM)"
elif [ -f "data/pssm/Train_335_pssm.csv" ]; then
    echo "[INFO] ⚠️  Found PSSM features but not merged yet"
    echo "[INFO] Run: python merge_features.py"
else
    echo "[INFO] ⚠️  Using 16D structural features only (PSSM not available)"
fi

echo ""
echo "======================================"
echo "[INFO] Architecture Configuration"
echo "======================================"
echo "ESM Projection:    2560 → 1024 → 512"
echo "Fusion:            512D ESM + ${FEATURE_MODE} features"
echo "GCN:               512D, 6 layers"
echo "TCN:               [128,256,512,1024]"
echo "Classifier:        2048 → 512 → 256 → 2"
echo "Total Parameters:  ~15M"
echo ""
if [ "$FEATURE_MODE" = "36D" ]; then
    echo "Features: 36D = 16D structural + 20D PSSM"
else
    echo "Features: 16D = structural only"
fi
echo ""

# ============================================
# Pipeline Steps
# ============================================

# Optional: Uncomment if needed
# echo "======================================"
# echo "Step 1: Download PDB Structures"
# echo "======================================"
# $PYTHON pdb_download.py \
#   2>&1 | tee "$LOG_DIR/pdb_download_$TIMESTAMP.log"
# echo "[✓] PDB download complete"
# echo ""

# echo "======================================"
# echo "Step 2: Extract ESM2 Embeddings (2560D)"
# echo "======================================"
# $PYTHON extract_esm.py \
#   2>&1 | tee "$LOG_DIR/esm_extract_$TIMESTAMP.log"
# echo "[✓] ESM extraction complete"
# echo ""

# echo "======================================"
# echo "Step 3: Extract Structural Features (16D)"
# echo "======================================"
# $PYTHON dataprep/dataprep.py \
#   2>&1 | tee "$LOG_DIR/dataprep_$TIMESTAMP.log"
# echo "[✓] Structural features extracted"
# echo ""

# ============================================
# PSSM Extraction (Optional but Recommended)
# ============================================
# Uncomment to extract PSSM features
# NOTE: This takes 6-12 hours. Run setup_pssm.sh instead for overnight runs.
#
# echo "======================================"
# echo "Step 4: Extract PSSM Features (20D)"
# echo "======================================"
# if command -v psiblast &> /dev/null; then
#     if [ -d "$HOME/ppi/Model/blast_db/uniref50" ]; then
#         $PYTHON extract_pssm.py \
#           2>&1 | tee "$LOG_DIR/pssm_extract_$TIMESTAMP.log"
#         echo "[✓] PSSM extraction complete"
#         
#         echo ""
#         echo "======================================"
#         echo "Step 5: Merge Features (16D + 20D → 36D)"
#         echo "======================================"
#         $PYTHON merge_features.py \
#           2>&1 | tee "$LOG_DIR/merge_features_$TIMESTAMP.log"
#         echo "[✓] Features merged"
#         FEATURE_MODE="36D"
#     else
#         echo "[WARN] UniRef50 database not found!"
#         echo "       Run: ./setup_pssm.sh"
#     fi
# else
#     echo "[WARN] PSI-BLAST not installed"
#     echo "       Install: sudo apt-get install ncbi-blast+"
# fi
# echo ""

# ============================================
# Update Training Scripts for Current Mode
# ============================================
echo "======================================"
echo "Configuring Training for ${FEATURE_MODE} Mode"
echo "======================================"

if [ "$FEATURE_MODE" = "36D" ]; then
    # Update to 36D feature paths
    sed -i 's|data/structural/Train_335_16D.csv|data/features/Train_335_36D.csv|g' train.py
    sed -i 's|data/structural/Test_315_16D.csv|data/features/Test_315_36D.csv|g' train.py
    sed -i 's|data/structural/Test_60_16D.csv|data/features/Test_60_36D.csv|g' tests/test_model.py
    
    # Delete old normalization if exists
    if [ -f "data/struct_norm.npz" ]; then
        rm -f data/struct_norm.npz
        echo "[INFO] Deleted old normalization file"
    fi
    
    echo "[✓] Configured for 36D features (structural + PSSM)"
else
    # Ensure using 16D structural paths
    sed -i 's|data/features/Train_335_36D.csv|data/structural/Train_335_16D.csv|g' train.py
    sed -i 's|data/features/Test_315_36D.csv|data/structural/Test_315_16D.csv|g' train.py
    sed -i 's|data/features/Test_60_36D.csv|data/structural/Test_60_16D.csv|g' tests/test_model.py
    
    echo "[✓] Configured for 16D features (structural only)"
fi
echo ""

# ============================================
# Create PyTorch Dataset
# ============================================
echo "======================================"
echo "Create PyTorch Dataset"
echo "======================================"
$PYTHON createdatset.py \
  2>&1 | tee "$LOG_DIR/create_dataset_$TIMESTAMP.log"
echo "[✓] Dataset created"
echo ""

# ============================================
# Train Model
# ============================================
echo "======================================"
echo "Train Model (512D ESM + ${FEATURE_MODE})"
echo "======================================"
echo "[INFO] Expected training time: 2-3 hours"
echo "[INFO] Model parameters: ~15M"
echo ""
$PYTHON train.py \
  2>&1 | tee "$LOG_DIR/train_$TIMESTAMP.log"
echo "[✓] Training complete"
echo ""

# ============================================
# Test Model
# ============================================
echo "======================================"
echo "Test Model on Test_60"
echo "======================================"
$PYTHON tests/test_model.py \
  2>&1 | tee "$LOG_DIR/test_$TIMESTAMP.log"
echo "[✓] Testing complete"
echo ""

# ============================================
# Summary
# ============================================
echo "======================================"
echo "Pipeline Complete! ✅"
echo "======================================"
echo ""
echo "Configuration: ${FEATURE_MODE} features"
echo ""
echo "Results:"
echo "  - Model checkpoint: best_model.pt"
echo "  - Test predictions: Test_60_predictions.csv"
echo "  - Benchmarks: per_protein_benchmarks.csv"
echo ""
echo "Logs: $LOG_DIR"
echo "Checkpoints: $CKPT_DIR"
echo ""

if [ "$FEATURE_MODE" = "36D" ]; then
    echo "Expected Performance (with PSSM):"
    echo "  - F1:   0.72-0.75 (+15-25% vs baseline)"
    echo "  - MCC:  0.56-0.61 (+24-36% vs baseline)"
    echo "  - AUPR: 0.65-0.72 (+18-31% vs baseline)"
else
    echo "Expected Performance (16D baseline):"
    echo "  - F1:   0.60-0.63"
    echo "  - MCC:  0.45-0.50"
    echo "  - AUPR: 0.55-0.60"
    echo ""
    echo "💡 TIP: Add PSSM features for +15-25% performance boost!"
    echo "   Run: ./setup_pssm.sh (overnight, 6-12 hours)"
fi
echo ""
echo "======================================"
