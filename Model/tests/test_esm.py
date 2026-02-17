import os
import re
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTA_DIR = os.path.join(PROJECT_ROOT, "data/fasta")
ESM_DIR = os.path.join(PROJECT_ROOT, "data/esm")
EXPECTED_DIM = 2560
MAX_ESM_LEN = 1022
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX")
VALID_LABEL_CHARS = set("01")
MAX_PER_FASTA = None

def parse_custom_fasta(path, max_entries=None):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    i = 0
    while i + 2 < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue

        pid = lines[i][1:].strip()
        seq = lines[i + 1].strip().upper()
        labels = lines[i + 2].strip()

        records.append({
            "pid": pid,
            "seq": seq,
            "labels": labels,
        })

        i += 3
        if max_entries is not None and len(records) >= max_entries:
            break

    return records


def validate_sequence(seq):
    bad = set(c for c in seq if c not in VALID_AA)
    return len(bad) == 0, bad

def validate_labels(labels):
    bad = set(c for c in labels if c not in VALID_LABEL_CHARS)
    return len(bad) == 0, bad


def pid_to_pdb_chain(pid):
    """
    Enforces strict PDB format: 4 chars + 1 chain
    Example: 1abcA
    """
    m = re.match(r"^([0-9][A-Za-z0-9]{3})([A-Za-z0-9])$", pid)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def find_esm_embedding(pdb, chain):
    pdb_u = pdb.upper()
    pdb_l = pdb.lower()
    chain_u = chain.upper()
    chain_l = chain.lower()

    candidates = [
        f"{pdb_u}_{chain_u}.pt",
        f"{pdb_u}{chain_u}.pt",
        f"{pdb_u}.pt",
        f"{pdb_l}_{chain_u}.pt",
        f"{pdb_l}{chain_u}.pt",
        f"{pdb_l}.pt",
        f"{pdb_l}{chain_l}.pt",
    ]

    for name in candidates:
        path = os.path.join(ESM_DIR, name)
        if os.path.exists(path):
            return path

    return None


def test_fasta_file(path):
    print(f"\n[TESTING FASTA] {os.path.basename(path)}")

    records = parse_custom_fasta(path, MAX_PER_FASTA)
    if not records:
        print("  ✗ No valid FASTA records found")
        return False

    passed = 0
    failed = 0

    for r in records:
        pid = r["pid"]
        seq = r["seq"]
        labels = r["labels"]
        L = len(seq)

        print(f"\n[CHECK] {pid} | L={L}")

        if L == 0:
            print("  ✗ Empty sequence")
            failed += 1
            continue

        if L > MAX_ESM_LEN:
            print(f"  ✗ Sequence too long for ESM2: {L} > {MAX_ESM_LEN}")
            failed += 1
            continue

        ok_seq, bad_seq = validate_sequence(seq)
        if not ok_seq:
            print(f"  ✗ Invalid amino acids: {sorted(bad_seq)}")
            failed += 1
            continue

        ok_lab, bad_lab = validate_labels(labels)
        if not ok_lab:
            print(f"  ✗ Invalid label characters: {sorted(bad_lab)}")
            failed += 1
            continue

        if len(labels) != L:
            print(f"  ✗ Label length mismatch: seq={L}, labels={len(labels)}")
            failed += 1
            continue

        pdb, chain = pid_to_pdb_chain(pid)
        if pdb is None:
            print("  ✗ PID does not match PDB+chain format")
            failed += 1
            continue

        emb_path = find_esm_embedding(pdb, chain)
        if emb_path is None:
            print(f"  ✗ Missing ESM embedding for {pdb}{chain}")
            failed += 1
            continue

        try:
            emb = torch.load(emb_path, map_location="cpu")

            if not isinstance(emb, torch.Tensor):
                raise TypeError("Not a torch.Tensor")

            if emb.ndim != 2:
                raise ValueError(f"Expected 2D tensor, got {emb.ndim}D")

            if emb.shape != (L, EXPECTED_DIM):
                raise ValueError(f"Shape mismatch: {tuple(emb.shape)} vs ({L}, {EXPECTED_DIM})")

            if torch.isnan(emb).any():
                raise ValueError("NaNs detected")

            if torch.isinf(emb).any():
                raise ValueError("Infs detected")

            print(f"  ✔ OK | emb={tuple(emb.shape)} | {os.path.basename(emb_path)}")
            passed += 1

        except Exception as e:
            print(f"  ✗ Embedding error: {e}")
            failed += 1

    print(f"\n[RESULT] {passed}/{passed + failed} passed")
    return failed == 0


# -----------------------------
# Entry point
# -----------------------------
def main():
    print("[START] ESM embedding validation")

    if not os.path.isdir(FASTA_DIR):
        print(f"[ERROR] FASTA dir not found: {FASTA_DIR}")
        raise SystemExit(1)

    if not os.path.isdir(ESM_DIR):
        print(f"[ERROR] ESM dir not found: {ESM_DIR}")
        raise SystemExit(1)

    fasta_files = sorted(f for f in os.listdir(FASTA_DIR) if f.endswith(".fa"))
    if not fasta_files:
        print("[ERROR] No FASTA files found")
        raise SystemExit(1)

    all_ok = True
    for f in fasta_files:
        ok = test_fasta_file(os.path.join(FASTA_DIR, f))
        all_ok = all_ok and ok

    print("\n==============================")
    if all_ok:
        print("[FINAL] ALL ESM EMBEDDINGS VALID ✅")
        raise SystemExit(0)
    else:
        print("[FINAL] ESM VALIDATION FAILED ❌")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
