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
        print(f"[ERROR] Failed to download {pdb_id}")
        return

    with open(out_path, "w") as f:
        f.write(r.text)

    print(f"[OK] Downloaded {pdb_id}.pdb")

# collect unique PDB IDs
pdb_ids = set()

for fasta_file in os.listdir(FASTA_DIR):
    fasta_path = os.path.join(FASTA_DIR, fasta_file)

    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                rid = line[1:].strip()   # e.g. 2v9tA
                pdb = rid[:-1].lower()   # remove chain
                pdb_ids.add(pdb)

print(f"[INFO] Found {len(pdb_ids)} unique PDB IDs")

# download all PDBs
for pdb in sorted(pdb_ids):
    download_pdb(pdb)
