#!/usr/bin/env python3
"""
PSSM Feature Extraction Pipeline
Generates Position-Specific Scoring Matrix (PSSM) features for protein sequences
using PSI-BLAST against UniRef50 database.
"""

import os
import sys
import subprocess
import tempfile
import logging
import multiprocessing as mp
from functools import partial
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Database paths - UPDATE THESE FOR YOUR SYSTEM
BLAST_DB = "/home/iiitdwd/ppi/Model/blast_db/uniref50/uniref50"  # Path to UniRef50 BLAST database
PSIBLAST_BIN = "psiblast"  # or full path if not in PATH

# System configuration (auto-detected)
CPU_COUNT = mp.cpu_count()
MAX_WORKERS = 2  # Limit to 2 parallel workers to reduce DB contention
BLAST_THREADS_PER_WORKER = 6  # 6 threads per worker = 12 threads total

# PSSM parameters
PSSM_DIM = 20  # 20 amino acids
NUM_ITERATIONS = 3  # PSI-BLAST iterations (reduced from default for speed)
EVALUE_THRESHOLD = 0.001  # E-value cutoff
BLAST_TIMEOUT = 6000  # 10 minutes per protein (generous timeout for large proteins)

# Configure logging
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "pssm_extraction.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_fmt)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_fmt)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

def run_psiblast(sequence, num_iterations=NUM_ITERATIONS, evalue=EVALUE_THRESHOLD, threads=BLAST_THREADS_PER_WORKER):
    """
    Run PSI-BLAST and extract PSSM matrix
    
    Args:
        sequence: Protein sequence string
        num_iterations: Number of PSI-BLAST iterations
        evalue: E-value threshold
    
    Returns:
        np.array of shape (len(sequence), 20) or None if failed
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write query sequence to FASTA
        query_fasta = os.path.join(tmpdir, "query.fasta")
        with open(query_fasta, 'w') as f:
            f.write(f">query\n{sequence}\n")
        
        # Output paths
        pssm_ascii = os.path.join(tmpdir, "pssm.txt")
        
        # PSI-BLAST command
        cmd = [
            PSIBLAST_BIN,
            "-query", query_fasta,
            "-db", BLAST_DB,
            "-num_iterations", str(num_iterations),
            "-evalue", str(evalue),
            "-max_target_seqs", "500",  # Limit hits to prevent MSA explosion
            "-out_ascii_pssm", pssm_ascii,
            "-num_threads", str(threads),
            "-out", os.devnull,  # Suppress alignment output for speed
        ]
        
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=BLAST_TIMEOUT,
                check=False
            )
            
            if result.returncode != 0:
                logger.warning(f"PSI-BLAST failed: {result.stderr.decode()[:200]}")
                return None
            
            # Parse PSSM ASCII file
            if not os.path.exists(pssm_ascii):
                logger.warning("PSSM file not created")
                return None
            
            return parse_pssm_file(pssm_ascii, len(sequence))
            
        except subprocess.TimeoutExpired:
            logger.warning(f"PSI-BLAST timeout (>{BLAST_TIMEOUT//60}min)")
            return None
        except Exception as e:
            logger.error(f"PSI-BLAST error: {e}")
            return None


def parse_pssm_file(pssm_path, expected_length):
    """
    Parse PSI-BLAST ASCII PSSM file
    
    Format:
    [Header lines]
    1 A  -1  -2   3  ...  (20 scores)
    2 C   2   0  -1  ...
    ...
    
    Returns:
        np.array of shape (expected_length, 20)
    """
    try:
        pssm_matrix = []
        
        with open(pssm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) < 22:  # pos + AA + 20 scores
                    continue
                
                # Check if first column is a number (position)
                try:
                    pos = int(parts[0])
                except ValueError:
                    continue
                
                # Extract 20 PSSM scores (columns 2-21)
                try:
                    scores = [int(parts[i]) for i in range(2, 22)]
                    pssm_matrix.append(scores)
                except (ValueError, IndexError):
                    continue
        
        if len(pssm_matrix) != expected_length:
            logger.warning(f"PSSM length mismatch: {len(pssm_matrix)} vs {expected_length}")
            return None
        
        return np.array(pssm_matrix, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"PSSM parsing error: {e}")
        return None


def get_default_pssm(length):
    """
    Generate default zero PSSM for sequences without homologs
    
    Args:
        length: Sequence length
    
    Returns:
        np.array of shape (length, 20) filled with zeros
    """
    return np.zeros((length, 20), dtype=np.float32)


def normalize_pssm(pssm):
    """
    Normalize PSSM scores to [0, 1] range
    
    PSSM scores typically range from -10 to +10
    We use sigmoid-like normalization
    
    Args:
        pssm: np.array of shape (length, 20)
    
    Returns:
        Normalized PSSM
    """
    # Clip extreme values
    pssm = np.clip(pssm, -10, 10)
    
    # Min-max normalization to [0, 1]
    pssm_norm = (pssm + 10) / 20.0
    
    return pssm_norm.astype(np.float32)


def process_single_protein(args):
    """
    Process a single protein (for multiprocessing)
    
    Args:
        args: Tuple of (protein_num, total_proteins, header, seq, labels, use_default)
    
    Returns:
        Tuple of (rows, success, protein_info)
    """
    protein_num, total_proteins, header, seq, labels, use_default = args
    
    # Parse header
    header = header.lstrip('>')
    if '_' in header:
        pdb, chain = header.split('_', 1)
    else:
        pdb = header[:4]
        chain = header[4:]
    
    # Run PSI-BLAST
    pssm = run_psiblast(seq)
    
    success = False
    if pssm is None:
        if use_default:
            pssm = get_default_pssm(len(seq))
        else:
            return [], False, f"{pdb}{chain}"
    else:
        success = True
    
    # Normalize PSSM
    pssm_norm = normalize_pssm(pssm)
    
    # Create rows
    rows = []
    for res_idx, (aa, label, pssm_row) in enumerate(zip(seq, labels, pssm_norm), start=1):
        row = [pdb, chain, res_idx, aa, int(label)] + pssm_row.tolist()
        rows.append(row)
    
    return rows, success, f"{pdb}{chain}"


def extract_pssm_for_fasta(fasta_path, output_csv, use_default_on_fail=True, n_workers=MAX_WORKERS):
    """
    Extract PSSM features for all proteins in FASTA file (parallelized)
    
    Args:
        fasta_path: Path to FASTA file (format: >PDB, seq, labels)
        output_csv: Output CSV path
        use_default_on_fail: If True, use zero PSSM when PSI-BLAST fails
    
    Output CSV format:
        PDB, Chain, ResIdx, AA, Label, PSSM_1, ..., PSSM_20
    """
    logger.info("="*80)
    logger.info(f"PSSM extraction: {fasta_path}")
    logger.info("="*80)
    
    # Read FASTA
    with open(fasta_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    total_proteins = len(lines) // 3
    logger.info(f"Total proteins: {total_proteins}")
    logger.info(f"Using {n_workers} parallel workers (CPU cores: {CPU_COUNT})")
    logger.info(f"PSI-BLAST threads per worker: {BLAST_THREADS_PER_WORKER}")
    logger.info(f"Total concurrent threads: {n_workers * BLAST_THREADS_PER_WORKER}")
    
    # Prepare arguments for parallel processing
    protein_args = []
    for i in range(0, len(lines), 3):
        protein_num = i // 3 + 1
        header, seq, labels = lines[i:i+3]
        protein_args.append((protein_num, total_proteins, header, seq, labels, use_default_on_fail))
    
    rows = []
    success_count = 0
    failed_count = 0
    start_time = time.time()
    
    # Process proteins in parallel with progress tracking
    logger.info("Starting parallel processing...")
    processed = 0
    
    with mp.Pool(processes=n_workers) as pool:
        for protein_rows, success, protein_id in pool.imap_unordered(process_single_protein, protein_args):
            processed += 1
            
            # Update counts
            if success:
                success_count += 1
            else:
                failed_count += 1
            
            # Add rows
            rows.extend(protein_rows)
            
            # Progress logging (every 10 proteins or key milestones)
            if processed % 10 == 0 or processed in [1, 5, total_proteins]:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_proteins - processed) / rate if rate > 0 else 0
                
                logger.info(
                    f"Progress: {processed}/{total_proteins} "
                    f"({processed/total_proteins*100:.1f}%) | "
                    f"Success: {success_count} | Failed: {failed_count} | "
                    f"Rate: {rate:.2f} proteins/sec | "
                    f"ETA: {eta/60:.1f} min"
                )
    
    # Save to CSV
    columns = ['PDB', 'Chain', 'ResIdx', 'AA', 'Label'] + [f'PSSM_{i}' for i in range(1, 21)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    
    # Calculate timing
    total_time = time.time() - start_time
    avg_time_per_protein = total_time / total_proteins if total_proteins > 0 else 0
    
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total proteins: {total_proteins}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed (using default): {failed_count}")
    logger.info(f"Success rate: {success_count/total_proteins*100:.1f}%")
    logger.info(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    logger.info(f"Avg time per protein: {avg_time_per_protein:.2f} seconds")
    logger.info(f"Processing rate: {total_proteins/total_time:.2f} proteins/second")
    logger.info(f"Output: {output_csv}")
    logger.info(f"Shape: {df.shape}")
    logger.info("="*80)


if __name__ == "__main__":
    # Check PSI-BLAST availability
    logger.info(f"Starting PSSM extraction pipeline")
    logger.info(f"Log file: logs/pssm_extraction.log")
    
    try:
        subprocess.run([PSIBLAST_BIN, "-version"], check=True, capture_output=True)
        logger.info(f"PSI-BLAST found: {PSIBLAST_BIN}")
    except:
        logger.error(f"PSI-BLAST not found: {PSIBLAST_BIN}")
        logger.error("Install BLAST+: sudo apt-get install ncbi-blast+")
        exit(1)
    
    # Check database (support multi-volume databases like .00.pin, .01.pin, etc.)
    db_path = Path(BLAST_DB)
    db_name = db_path.name
    db_dir = db_path.parent
    
    # Check for main .pin file or first volume (.00.pin)
    if not (db_path.with_suffix('.pin').exists() or 
            db_dir.joinpath(f"{db_name}.00.pin").exists() or
            db_dir.joinpath(f"{db_name}.pal").exists()):
        logger.error(f"BLAST database not found: {BLAST_DB}")
        logger.error(f"Expected: {BLAST_DB}.pin or {BLAST_DB}.00.pin")
        logger.error("Download UniRef50 and run makeblastdb")
        exit(1)
    
    logger.info(f"BLAST database found: {BLAST_DB}")
    
    # Create output directory
    os.makedirs("data/pssm", exist_ok=True)
    
    # Extract PSSM for all datasets
    extract_pssm_for_fasta("data/fasta/Train_335.fa", "data/pssm/Train_335_pssm.csv")
    extract_pssm_for_fasta("data/fasta/Test_60.fa", "data/pssm/Test_60_pssm.csv")
    extract_pssm_for_fasta("data/fasta/Test_315.fa", "data/pssm/Test_315_pssm.csv")
