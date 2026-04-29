import os
import re
import torch
import random
import numpy as np
from transformers import BertTokenizer, BertModel

FASTA_DIR = "data/fasta"
OUT_DIR = "data/protbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 1022

os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
model = model.to(DEVICE).eval()

@torch.no_grad()
def extract(sequence):
    seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence.upper())))
    encoding = tokenizer(seq, return_tensors="pt")
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    emb = out.last_hidden_state[0, 1:-1]  # strip [CLS] and [SEP] → (L, 1024)

    del out, input_ids, attention_mask
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return emb.float().cpu()

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

        seq = seq.upper()

        if len(seq) == 0:
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

        torch.save(emb, save_path)
        print(f"[OK] {pdb}_{chain} → {emb.shape}")
