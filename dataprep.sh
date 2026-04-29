#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOGS="$ROOT/logs"; mkdir -p "$LOGS"

if [[ ! -d data/fasta ]]; then
    echo "ERROR: data/fasta/ not found. Place Train_335.fa, Test_315.fa, Test_60.fa there first."
    exit 1
fi

echo "[$(date +%H:%M:%S)] (1/4) Downloading PDBs from data/fasta/*.fa  -> data/pdbs/"
"$PYTHON_BIN" pdb_download.py 2>&1 | tee "$LOGS/dataprep_pdb.log"

echo "[$(date +%H:%M:%S)] (2/4) Generating structural CSVs  -> data/structural/*_17D.csv"
"$PYTHON_BIN" dataprep/dataprep.py 2>&1 | tee "$LOGS/dataprep_structural.log"

echo "[$(date +%H:%M:%S)] (3/4) Extracting ESM-2 3B embeddings  -> data/esm/"
"$PYTHON_BIN" extract_esm.py 2>&1 | tee "$LOGS/dataprep_esm.log"

echo "[$(date +%H:%M:%S)] (4/4) Extracting ProtBERT embeddings  -> data/protbert/"
"$PYTHON_BIN" extract_protbert.py 2>&1 | tee "$LOGS/dataprep_protbert.log"

echo
echo "Data prep complete. Expected layout:"
echo "  data/pdbs/*.pdb"
echo "  data/structural/Train_335_17D.csv  Test_315_17D.csv  Test_60_17D.csv"
echo "  data/esm/*.pt        (2560D per residue)"
echo "  data/protbert/*.pt   (1024D per residue)"
