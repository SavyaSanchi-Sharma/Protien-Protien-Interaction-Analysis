#!/bin/bash
set -e

# ============================================
# PSSM Setup & Extraction - Overnight Script
# ============================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$PROJECT_ROOT/logs/pssm_setup_current.log"

mkdir -p "$PROJECT_ROOT/logs"

exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo "============================================"
echo "PSSM Setup Started: $(date)"
echo "============================================"
echo "Log file: $LOG_FILE"
echo ""

# ============================================
# Step 1: Check Prerequisites
# ============================================
echo "[1/7] Checking prerequisites..."
echo ""

AVAILABLE_GB=$(df -BG "$HOME" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available disk space: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_GB" -lt 150 ]; then
    echo "[ERROR] Insufficient disk space! Need 150GB, have ${AVAILABLE_GB}GB"
    exit 1
fi

echo "[✓] Disk space OK"
echo ""

if ! command -v psiblast &> /dev/null; then
    echo "[WARN] PSI-BLAST not found, installing..."
    sudo apt-get update -qq
    sudo apt-get install -y ncbi-blast+
    echo "[✓] PSI-BLAST installed"
else
    echo "[✓] PSI-BLAST already installed"
    psiblast -version
fi
echo ""

# ============================================
# Step 2: Download UniRef50
# ============================================
DB_DIR="$HOME/ppi/Model/blast_db/uniref50"
mkdir -p "$DB_DIR"
cd "$DB_DIR"

echo "[2/7] Downloading UniRef50 database..."
echo ""

if [ -f "uniref50.fasta" ]; then
    echo "[✓] uniref50.fasta already exists, skipping download"
elif [ -f "uniref50.fasta.gz" ]; then
    echo "[✓] uniref50.fasta.gz already exists, skipping download"
else
    echo "Starting download: $(date)"
    wget -c ftp://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz \
        --progress=bar:force:noscroll
    echo "[✓] Download complete: $(date)"
fi
echo ""

# ============================================
# Step 3: Uncompress Database
# ============================================
echo "[3/7] Uncompressing database..."
echo ""

if [ -f "uniref50.fasta" ]; then
    FASTA_SIZE=$(du -h uniref50.fasta | cut -f1)
    echo "[✓] uniref50.fasta already exists (${FASTA_SIZE})"
else
    echo "Starting decompression: $(date)"
    gunzip -v uniref50.fasta.gz
    echo "[✓] Decompression complete: $(date)"
fi
echo ""

# ============================================
# Step 4: Format Database for BLAST
# ============================================
echo "[4/7] Formatting database for BLAST..."
echo ""

if blastdbcmd -db uniref50 -info >/dev/null 2>&1; then
    echo "[✓] BLAST database already formatted"
else
    echo "Starting makeblastdb: $(date)"
    makeblastdb \
        -in uniref50.fasta \
        -dbtype prot \
        -out uniref50 \
        -parse_seqids \
        -hash_index \
        -max_file_sz 4GB
    echo "[✓] Database formatting complete: $(date)"
fi

# ============================================
# Step 5: Verify Database (ROBUST)
# ============================================
echo ""
echo "[5/7] Verifying BLAST database..."

if blastdbcmd -db uniref50 -info >/dev/null 2>&1; then
    DB_SIZE=$(du -sh . | cut -f1)
    PIN_COUNT=$(ls uniref50*.pin | wc -l)
    echo "[✓] Database ready"
    echo "    Volumes      : $PIN_COUNT"
    echo "    Total size   : $DB_SIZE"
else
    echo "[ERROR] BLAST database validation failed!"
    exit 1
fi

echo ""
echo "============================================"
echo "[✓] UniRef50 BLAST DB setup completed"
echo "Completed at: $(date)"
echo "============================================"

# ============================================
# Step 6: Extract PSSM Features
# ============================================
echo "[6/7] Extracting PSSM features..."
echo "This will process all proteins (~400 total)"
echo "Estimated time: 6-12 hours"
echo ""

echo "Starting PSSM extraction: $(date)"
source .venv/bin/activate
python extract_pssm.py 2>&1 | tee -a "$LOG_FILE"
echo "[✓] PSSM extraction complete: $(date)"
echo ""

# ============================================
# Step 7: Merge Features and Retrain
# ============================================
echo "[7/7] Merging features and preparing for training..."
echo ""

# Merge structural + PSSM
echo "Merging features..."
python merge_features.py 2>&1 | tee -a "$LOG_FILE"
echo "[✓] Features merged"
echo ""

# Update train.py to use merged features
echo "Updating train.py to use 36D features..."
sed -i 's|data/structural/Train_335_16D.csv|data/features/Train_335_36D.csv|g' train.py
sed -i 's|data/structural/Test_315_16D.csv|data/features/Test_315_36D.csv|g' train.py
echo "[✓] Updated CSV paths in train.py"
echo ""

# Update test_model.py
sed -i 's|data/structural/Test_60_16D.csv|data/features/Test_60_36D.csv|g' tests/test_model.py
echo "[✓] Updated CSV paths in test_model.py"
echo ""

# Delete old normalization
if [ -f "data/struct_norm.npz" ]; then
    rm data/struct_norm.npz
    echo "[✓] Deleted old normalization file"
fi
echo ""

# ============================================
# Summary
# ============================================
echo "============================================"
echo "PSSM Setup Complete! ✅"
echo "============================================"
echo "Completed: $(date)"
echo ""
echo "Database location: $DB_DIR"
echo "Database size: $(du -sh $DB_DIR | cut -f1)"
echo ""
echo "PSSM features:"
echo "  - data/pssm/Train_335_pssm.csv"
echo "  - data/pssm/Test_60_pssm.csv"
echo "  - data/pssm/Test_315_pssm.csv"
echo ""
echo "Merged features (36D):"
echo "  - data/features/Train_335_36D.csv"
echo "  - data/features/Test_60_36D.csv"
echo "  - data/features/Test_315_36D.csv"
echo ""
echo "============================================"
echo "Ready to Train!"
echo "============================================"
echo ""
echo "Run training with:"
echo "  python train.py"
echo ""
echo "Expected performance with PSSM:"
echo "  F1:   0.72-0.75 (+0.12-0.15 vs baseline)"
echo "  MCC:  0.56-0.61 (+0.11-0.16 vs baseline)"
echo "  AUPR: 0.65-0.72 (+0.10-0.17 vs baseline)"
echo ""
echo "============================================"
echo "Full log: $LOG_FILE"
echo "============================================"
