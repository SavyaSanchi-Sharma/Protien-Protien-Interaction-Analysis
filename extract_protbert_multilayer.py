"""Per-residue ProtBert-BFD representations from selected transformer layers.

Layers chosen across ProtBert-BFD's 30 transformer layers to span the
biochemistry → secondary-structure → contact → task-specific gradient:

  5, 12, 18, 24, 27, 30

Stored fp16 in data/protbert_multi/<pdb>_<chain>.pt; cast to float32 in the loader.
"""

import os
import re
import random

import numpy as np
import torch
from transformers import BertTokenizer, BertModel

FASTA_DIR = "data/fasta"
OUT_DIR   = "data/protbert_multi"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS    = [5, 12, 18, 24, 27, 30]
MAX_LEN   = 1022

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
np.random.seed(42); random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
model = BertModel.from_pretrained(
    "Rostlab/prot_bert_bfd", output_hidden_states=True
).to(DEVICE).eval()


@torch.no_grad()
def extract(sequence):
    seq = " ".join(re.sub(r"[UZOB]", "X", sequence.upper()))
    enc = tokenizer(seq, return_tensors="pt")
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    # hidden_states[0] is the input embedding; layers 1..30 are transformer outputs.
    hidden_states = out.hidden_states
    # Slice [1:-1] strips [CLS] and [SEP] so the output aligns with the residue sequence.
    stacked = torch.stack(
        [hidden_states[L][0, 1:-1] for L in LAYERS], dim=1
    )
    del out, input_ids, attention_mask, hidden_states
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

        pdb   = rid[:4].lower()
        chain = rid[-1].upper()
        if not chain.isalpha():
            raise ValueError(f"Invalid chain ID in header: {rid}")

        if not seq:
            print(f"[SKIP] Empty sequence: {rid}")
            continue
        if len(seq) > MAX_LEN:
            print(f"[SKIP] Too long for ProtBERT ({len(seq)} aa): {rid}")
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
        if emb.shape[1] != len(LAYERS) or emb.shape[2] != 1024:
            raise RuntimeError(
                f"Shape mismatch for {rid}: got {tuple(emb.shape)}, "
                f"expected (L, {len(LAYERS)}, 1024)"
            )
        torch.save(emb, save_path)
        print(f"[OK] {pdb}_{chain} → {tuple(emb.shape)} fp16")
