import os
import torch
import esm
import random
import numpy as np

FASTA_DIR = "data/fasta"
OUT_DIR = "data/esm"
DEVICE = "cuda"
LAYER = 36
MAX_LEN = 1022

os.makedirs(OUT_DIR, exist_ok=True)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
model = model.to(DEVICE, dtype=torch.bfloat16).eval()
batch_converter = alphabet.get_batch_converter()

@torch.no_grad()
def extract(sequence):
    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)

    out = model(tokens, repr_layers=[LAYER])
    emb = out["representations"][LAYER][0, 1:len(sequence) + 1]

    del out, tokens
    torch.cuda.empty_cache()

    return emb.float().cpu()

def clean_protein(seq):
    seq = seq.upper()
    for c in seq:
        if c not in VALID_AA:
            raise ValueError(f"Invalid residue {c}")
    return seq

def parse_fasta(path):
    records = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            raise ValueError(f"Malformed FASTA at line: {lines[i]}")

        rid = lines[i][1:]
        seq = lines[i + 1]

        if not all(c in "ACDEFGHIKLMNPQRSTVWYX" for c in seq.upper()):
            raise ValueError(f"Invalid AA sequence for {rid}")

        records.append((rid, seq))
        i += 3

    return records

for fasta_file in os.listdir(FASTA_DIR):
    fasta_path = os.path.join(FASTA_DIR, fasta_file)
    print(fasta_path)

    records = parse_fasta(fasta_path)

    for rid, seq in records:
        if len(rid) < 5:
            raise ValueError(f"Malformed FASTA header: {rid}")

        pdb = rid[:4].lower()
        chain = rid[-1].upper()

        if not chain.isalpha():
            raise ValueError(f"Invalid chain ID in header: {rid}")

        seq = clean_protein(seq)

        if len(seq) == 0:
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

        torch.save(emb, save_path)
        print(f"[OK] {pdb}_{chain} → {emb.shape}")
