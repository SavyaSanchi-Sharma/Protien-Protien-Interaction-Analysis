import os
import requests

FASTA_DIR = "data/fasta"
PDB_DIR = "data/pdbs"

os.makedirs(PDB_DIR, exist_ok=True)


def download_pdb(pdb_id):
    pdb_id = pdb_id.lower()
    out_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        print(f"[SKIP] {pdb_id}.pdb already exists")
        return
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        print(f"[ERROR] Failed to download {pdb_id} (HTTP {r.status_code})")
        return
    with open(out_path, "w") as f:
        f.write(r.text)
    print(f"[OK] Downloaded {pdb_id}.pdb")


pdb_ids = set()
for fasta_file in os.listdir(FASTA_DIR):
    fasta_path = os.path.join(FASTA_DIR, fasta_file)
    with open(fasta_path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            header = line[1:].strip().split()[0]
            if len(header) < 5:
                print(f"[WARN] Skipping malformed header: {line.strip()}")
                continue
            pdb_ids.add(header[:4].lower())

print(f"[INFO] Found {len(pdb_ids)} unique PDB IDs")
for pdb in sorted(pdb_ids):
    download_pdb(pdb)
