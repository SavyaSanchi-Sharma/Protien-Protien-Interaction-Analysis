import os
import shutil
import time
import requests
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.vectors import Vector
from Bio.SeqUtils import seq1
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline

PDB_DIR = "data/pdbs"
os.makedirs(PDB_DIR, exist_ok=True)
np.random.seed(42)

FEATURE_NAMES = [
    "RSA", "ResFlex", "Hydrophobicity", "PackingDensity",
    "HSE_up", "HSE_down",
    "Poly_bias", "Poly_RSA", "Poly_Flex", "Poly_interaction",
    "BondAngle",
    "sin_phi", "cos_phi", "sin_psi", "cos_psi",
    "sin_omega", "cos_omega"
]

# Cα and N atom coordinates per residue. Used by the data loader to build the
# residue contact graph and edge geometry without re-parsing the PDB.
COORD_NAMES = ["CA_x", "CA_y", "CA_z", "N_x", "N_y", "N_z"]

MAX_ASA = {
    'A': 121, 'R': 265, 'N': 187, 'D': 187, 'C': 148,
    'Q': 214, 'E': 214, 'G': 97,  'H': 216, 'I': 195,
    'L': 191, 'K': 230, 'M': 203, 'F': 228, 'P': 154,
    'S': 143, 'T': 163, 'W': 264, 'Y': 255, 'V': 165
}

HYDRO_SCALES = [
    {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2},
    {'A':-0.5,'R':3.0,'N':0.2,'D':3.0,'C':-1.0,'Q':0.2,'E':3.0,'G':0.0,'H':-0.5,'I':-1.8,'L':-1.8,'K':3.0,'M':-1.3,'F':-2.5,'P':0.0,'S':0.3,'T':-0.4,'W':-3.4,'Y':-2.3,'V':-1.5},
    {'A':0.62,'R':-2.53,'N':-0.78,'D':-0.90,'C':0.29,'Q':-0.85,'E':-0.74,'G':0.48,'H':-0.40,'I':1.38,'L':1.06,'K':-1.50,'M':0.64,'F':1.19,'P':0.12,'S':-0.18,'T':-0.05,'W':0.81,'Y':0.26,'V':1.08},
    {'A':0.74,'R':0.64,'N':0.63,'D':0.62,'C':0.91,'Q':0.62,'E':0.62,'G':0.72,'H':0.78,'I':0.88,'L':0.85,'K':0.52,'M':0.85,'F':0.88,'P':0.64,'S':0.66,'T':0.70,'W':0.85,'Y':0.76,'V':0.86},
    {'A':0.31,'R':-1.01,'N':-0.60,'D':-0.77,'C':1.54,'Q':-0.22,'E':-0.64,'G':0.00,'H':0.13,'I':1.80,'L':1.70,'K':-0.99,'M':1.23,'F':1.79,'P':0.72,'S':-0.04,'T':0.26,'W':2.25,'Y':0.96,'V':1.22},
    {'A':0.20,'R':-1.50,'N':-0.60,'D':-1.40,'C':1.00,'Q':-0.70,'E':-1.30,'G':0.00,'H':-0.10,'I':1.80,'L':1.50,'K':-1.50,'M':1.30,'F':1.40,'P':0.00,'S':-0.10,'T':0.40,'W':1.00,'Y':0.60,'V':1.50}
]

MAX_RETRIES = 3
RETRY_DELAYS = [2, 5, 10]

DSSP_BIN = next((b for b in ("mkdssp", "dssp", "dssp3") if shutil.which(b)), None)
if DSSP_BIN is None:
    print("[WARN] No DSSP binary found. Install: sudo apt-get install dssp")
else:
    print(f"[INFO] DSSP binary: {DSSP_BIN}")


def _valid_pdb_text(text):
    """True if `text` (str) starts with a plausible PDB record."""
    if len(text) < 200:
        return False
    head = text[:300]
    return any(t in head for t in ("ATOM", "HETATM", "HEADER", "REMARK"))


def _valid_file(path):
    if not os.path.exists(path) or os.path.getsize(path) < 200:
        return False
    with open(path, "r", errors="ignore") as f:
        return _valid_pdb_text(f.read(300))


def _valid_content(content):
    return _valid_pdb_text(content[:300].decode("utf-8", errors="ignore"))


def _clean_pdb_for_dssp(path):
    """Strip HETATM/ANISOU so newer mkdssp doesn't choke on HOH/ligands.
    Caches output as <stem>_clean.pdb; reused if newer than the source."""
    stem, ext = os.path.splitext(path)
    clean_path = f"{stem}_clean{ext or '.pdb'}"
    if os.path.exists(clean_path) and os.path.getmtime(clean_path) >= os.path.getmtime(path):
        return clean_path
    with open(path, "r", errors="ignore") as src, open(clean_path, "w") as dst:
        for line in src:
            if not line.startswith(("HETATM", "ANISOU")):
                dst.write(line)
    return clean_path


def _run_dssp(model, path):
    if DSSP_BIN is None:
        raise RuntimeError("No DSSP binary found. Install: sudo apt-get install dssp")
    return DSSP(model, _clean_pdb_for_dssp(path), dssp=DSSP_BIN)


def parse_header(header):
    raw = header.replace(">", "").strip().split()[0].upper()
    if len(raw) >= 5:
        return raw[:4], raw[4]
    if len(raw) == 4:
        return raw, None
    print(f"[WARN] Invalid header format: {header}")
    return None, None


def download_structure(pdb_id):
    """Download PDB with retry logic and content validation."""
    dest = os.path.join(PDB_DIR, f"{pdb_id.lower()}.pdb")

    if _valid_file(dest):
        print(f"[CACHE] {pdb_id}.pdb")
        return dest

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 404:
                print(f"[ERROR] {pdb_id}.pdb not found on RCSB")
                return None
            if r.status_code == 200 and _valid_content(r.content):
                with open(dest, "wb") as f:
                    f.write(r.content)
                print(f"[DOWNLOAD] {pdb_id}.pdb")
                return dest
            print(f"[WARN] {pdb_id}.pdb attempt {attempt+1}: status={r.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"[WARN] {pdb_id}.pdb attempt {attempt+1}: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAYS[attempt])

    print(f"[ERROR] {pdb_id}.pdb unavailable after {MAX_RETRIES} attempts")
    return None


def parse_structure(pdb_id, path):
    try:
        return PDBParser(QUIET=True).get_structure(pdb_id, path)
    except Exception as e:
        print(f"[ERROR] parse_structure {pdb_id}: {e}")
        return None


def select_best_chain(structure, seq):
    """Pick the chain whose CA-only sequence length is closest to `seq`."""
    best, diff = None, 1e9
    for c in structure[0]:
        try:
            s = "".join(seq1(r.resname) for r in c if "CA" in r)
        except Exception as e:
            print(f"[WARN] Chain {c.id} error: {e}")
            continue
        if abs(len(s) - len(seq)) < diff:
            best, diff = c.id, abs(len(s) - len(seq))
    if best is not None:
        print(f"[DEBUG] Auto-selected chain: {best} (length diff: {diff})")
    return best


def extract_17D_features(structure, path, chain_id, fasta_seq):
    model = structure[0]
    try:
        dssp = _run_dssp(model, path)
    except Exception as e:
        print(f"[ERROR] DSSP failed: {e}")
        return None

    if chain_id not in model:
        print(f"[ERROR] Chain '{chain_id}' not found. Available: {[c.id for c in model]}")
        return None
    chain = model[chain_id]

    residues = [r for r in chain if all(a in r for a in ("CA", "N", "C", "O"))]
    if len(residues) < 3:
        print(f"[ERROR] Too few residues: {len(residues)}")
        return None

    coverage = len(residues) / len(fasta_seq)
    if coverage < 0.7:
        print(f"[ERROR] Low coverage: {len(residues)}/{len(fasta_seq)} = {coverage*100:.1f}%")
        return None

    print(f"[DEBUG] Processing {len(residues)} residues (coverage {coverage*100:.1f}%)")

    ca_coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)
    ca_tree = KDTree(ca_coords)

    all_atoms = []
    for r in residues:
        for a in r.get_atoms():
            if a.is_disordered():
                a = a.selected_child
            all_atoms.append(a.coord)
    atom_tree = KDTree(np.array(all_atoms, dtype=np.float32))

    b = np.array([r["CA"].get_bfactor() for r in residues], dtype=np.float32)
    b_range = b.max() - b.min()
    flex = (b - b.min()) / b_range if b_range > 1e-6 else np.zeros_like(b)

    dens = np.array([
        len(atom_tree.query_ball_point(r["CA"].coord, 3.5)) / max(len(list(r.get_atoms())), 1)
        for r in residues
    ], dtype=np.float32)
    d_range = dens.max() - dens.min()
    dens = (dens - dens.min()) / d_range if d_range > 1e-6 else np.zeros_like(dens)

    # Phi/psi from DSSP with cubic-spline interpolation over missing values.
    phis = np.full(len(residues), np.nan)
    psis = np.full(len(residues), np.nan)
    for i, r in enumerate(residues):
        k = (chain_id, r.id)
        if k in dssp:
            if dssp[k][4] != 360:
                phis[i] = np.radians(dssp[k][4])
            if dssp[k][5] != 360:
                psis[i] = np.radians(dssp[k][5])
    idx = np.arange(len(residues))
    valid_phi = ~np.isnan(phis)
    valid_psi = ~np.isnan(psis)
    phis = CubicSpline(idx[valid_phi], phis[valid_phi])(idx) if valid_phi.sum() >= 3 else np.nan_to_num(phis)
    psis = CubicSpline(idx[valid_psi], psis[valid_psi])(idx) if valid_psi.sum() >= 3 else np.nan_to_num(psis)

    feats = []
    bad = 0
    for i, r in enumerate(residues):
        try:
            aa = seq1(r.resname)
            dssp_entry = dssp.get((chain_id, r.id)) if hasattr(dssp, "get") else None
            asa = dssp_entry[3] if dssp_entry is not None else 0.0
            rsa = float(np.clip(asa / MAX_ASA.get(aa, 1.0), 0.0, 1.0))
            hydro = float(np.mean([s.get(aa, 0.0) for s in HYDRO_SCALES]))
            poly = [1.0, rsa, float(flex[i]), rsa * float(flex[i])]

            bond = 0.0
            if 0 < i < len(residues) - 1:
                bond = float(Vector.angle(
                    residues[i-1]["C"].get_vector() - r["N"].get_vector(),
                    residues[i+1]["O"].get_vector() - r["N"].get_vector(),
                ))
            omega = float(Vector.angle(
                r["N"].get_vector() - r["CA"].get_vector(),
                r["O"].get_vector() - r["C"].get_vector(),
            ))

            ca = r["CA"].coord
            n_coord = r["N"].coord
            ref = (r["CB"].coord - ca) if "CB" in r else (n_coord - ca)
            neigh = ca_tree.query_ball_point(ca, 8.0)
            up = sum(1 for j in neigh if j != i and np.dot(ca_coords[j] - ca, ref) >= 0)
            down = max(len(neigh) - 1, 0) - up
            total = max(len(neigh) - 1, 1)

            feats.append([
                rsa, float(flex[i]), hydro, float(dens[i]),
                up / total, down / total,
                *poly,
                bond,
                float(np.sin(phis[i])), float(np.cos(phis[i])),
                float(np.sin(psis[i])), float(np.cos(psis[i])),
                float(np.sin(omega)), float(np.cos(omega)),
                float(ca[0]), float(ca[1]), float(ca[2]),
                float(n_coord[0]), float(n_coord[1]), float(n_coord[2]),
            ])
        except Exception as e:
            print(f"[WARN] Residue {i} ({r.resname} {r.id}): {e} — using zeros")
            feats.append([0.0] * (len(FEATURE_NAMES) + len(COORD_NAMES)))
            bad += 1

    if bad:
        print(f"[WARN] {bad}/{len(residues)} residues used zero fallback")

    result = np.array(feats, dtype=np.float32)
    print(f"[DEBUG] Features extracted: {result.shape}")
    return result


def build_dataset(fasta_file, out_csv, checkpoint_every=50):
    print(f"\n{'='*80}")
    print(f"[START] {fasta_file}")
    print(f"{'='*80}\n")

    with open(fasta_file) as f:
        lines = [l.strip() for l in f if l.strip()]

    total_proteins = len(lines) // 3
    print(f"[INFO] Total proteins: {total_proteins}\n")

    partial_path = out_csv + ".partial"
    rows = []
    done_keys = set()
    if os.path.exists(partial_path):
        try:
            df_p = pd.read_csv(partial_path)
            rows = df_p.values.tolist()
            done_keys = set(zip(df_p["PDB"].str.lower(), df_p["Chain"]))
            print(f"[RESUME] {len(done_keys)} chains already processed\n")
        except Exception as e:
            print(f"[WARN] Could not load checkpoint: {e} — starting fresh")

    csv_columns = ["PDB", "Chain", "ResIdx", "AA", "Label"] + FEATURE_NAMES + COORD_NAMES
    success = failed = resumed = 0

    for i in range(0, len(lines), 3):
        protein_num = i // 3 + 1
        try:
            h, seq, lab = lines[i:i+3]
            print(f"\n[{protein_num}/{total_proteins}] {h}")

            pdb, chain = parse_header(h)
            if pdb is None:
                failed += 1; continue
            if len(seq) != len(lab):
                print(f"[SKIP] seq/label length mismatch ({len(seq)} vs {len(lab)})")
                failed += 1; continue
            if not all(c in "01" for c in lab):
                print(f"[SKIP] Invalid label characters")
                failed += 1; continue

            key = (pdb.lower(), chain or "?")
            if key in done_keys:
                print(f"[SKIP] Already done (resuming)")
                resumed += 1; success += 1; continue

            path = download_structure(pdb)
            if path is None:
                failed += 1; continue

            structure = parse_structure(pdb, path)
            if structure is None:
                failed += 1; continue

            if chain is None:
                chain = select_best_chain(structure, seq)
                if chain is None:
                    print(f"[SKIP] Chain selection failed")
                    failed += 1; continue

            feats = extract_17D_features(structure, path, chain, seq)
            if feats is None or feats.shape[1] != len(FEATURE_NAMES) + len(COORD_NAMES):
                print(f"[SKIP] Feature extraction failed")
                failed += 1; continue

            L = min(len(seq), len(feats))
            for j in range(L):
                rows.append([pdb.lower(), chain, j+1, seq[j], int(lab[j])] + feats[j].tolist())
            print(f"[OK] {pdb}_{chain}: {L} residues")
            success += 1

            if success % checkpoint_every == 0:
                pd.DataFrame(rows, columns=csv_columns).to_csv(partial_path, index=False)
                print(f"[CHECKPOINT] {success} proteins / {len(rows)} residues saved")

        except Exception as e:
            print(f"[ERROR] Protein {protein_num}: {e}")
            failed += 1

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df = pd.DataFrame(rows, columns=csv_columns)
    df.to_csv(out_csv, index=False)
    if os.path.exists(partial_path):
        os.unlink(partial_path)

    print(f"\n{'='*80}")
    print(f"[SUMMARY] {fasta_file}")
    print(f"  Total:       {total_proteins}")
    print(f"  Succeeded:   {success}  (resumed: {resumed})")
    print(f"  Failed:      {failed}")
    print(f"  Success rate:{success/total_proteins*100:.1f}%")
    print(f"  Output:      {out_csv}  {df.shape}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    build_dataset("data/fasta/Train_335.fa", "data/structural/Train_335_17D.csv")
    build_dataset("data/fasta/Test_60.fa",   "data/structural/Test_60_17D.csv")
    build_dataset("data/fasta/Test_315.fa",  "data/structural/Test_315_17D.csv")
