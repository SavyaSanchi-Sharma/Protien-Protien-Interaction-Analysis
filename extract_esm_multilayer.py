"""Per-residue ESM-2 3B representations from selected transformer layers.

Layers chosen across ESM-2 3B's 36 transformer layers to span the
biochemistry → secondary-structure → contact → task-specific gradient:

  6, 18, 24, 30, 33, 36

Stored fp16 in data/esm_multi/<pdb>_<chain>.pt; cast to float32 in the loader.
"""

import os
import random

import numpy as np
import torch
import esm

FASTA_DIR = "data/fasta"
OUT_DIR   = "data/esm_multi"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS    = [6, 18, 24, 30, 33, 36]
MAX_LEN   = 1022
VALID_AA  = set("ACDEFGHIKLMNPQRSTVWYX")

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
np.random.seed(42); random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.to(DEVICE, dtype=torch.bfloat16).eval()
batch_converter = alphabet.get_batch_converter()


@torch.no_grad()
def extract(sequence):
    _, _, tokens = batch_converter([("protein", sequence)])
    tokens = tokens.to(DEVICE)
    out = model(tokens, repr_layers=LAYERS)
    reps = out["representations"]
    stacked = torch.stack(
        [reps[L][0, 1:len(sequence) + 1] for L in LAYERS], dim=1
    )
    del out, tokens, reps
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return stacked.to(torch.float16).cpu()


def parse_fasta(path):
    # 3 lines per record: header, sequence, label.
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    records = []
    for i in range(0, len(lines), 3):
        if not lines[i].startswith(">"):
            raise ValueError(f"Malformed FASTA at line {i+1}: {lines[i]}")
        records.append((lines[i][1:], lines[i + 1].upper()))
    return records


for fasta_file in os.listdir(FASTA_DIR):
    fasta_path = os.path.join(FASTA_DIR, fasta_file)
    print(fasta_path)
    for rid, seq in parse_fasta(fasta_path):
        if len(rid) < 5:
            raise ValueError(f"Malformed FASTA header: {rid}")
        bad = set(seq) - VALID_AA
        if bad:
            raise ValueError(f"Invalid residues {bad} in {rid}")

        pdb   = rid[:4].lower()
        chain = rid[-1].upper()
        if not chain.isalpha():
            raise ValueError(f"Invalid chain ID in header: {rid}")

        if not seq:
            print(f"[SKIP] Empty sequence: {rid}")
            continue
        if len(seq) > MAX_LEN:
            print(f"[SKIP] Too long for ESM ({len(seq)} aa): {rid}")
            continue

        save_path = os.path.join(OUT_DIR, f"{pdb}_{chain}.pt")
        if os.path.exists(save_path):
            print(f"[SKIP] Exists: {pdb}_{chain}")
            continue

        emb = extract(seq)
        if emb.shape[0] != len(seq):
            raise RuntimeError(
                f"Length mismatch for {rid}: emb={emb.shape[0]} seq={len(seq)}"
            )
        if emb.shape[1] != len(LAYERS) or emb.shape[2] != 2560:
            raise RuntimeError(
                f"Shape mismatch for {rid}: got {tuple(emb.shape)}, "
                f"expected (L, {len(LAYERS)}, 2560)"
            )
        torch.save(emb, save_path)
        print(f"[OK] {pdb}_{chain} → {tuple(emb.shape)} fp16")
