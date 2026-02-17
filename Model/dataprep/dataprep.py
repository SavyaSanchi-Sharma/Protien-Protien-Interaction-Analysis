import os
import requests
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.vectors import Vector
from Bio.SeqUtils import seq1
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
import time

PDB_DIR = "data/pdbs"
os.makedirs(PDB_DIR, exist_ok=True)
np.random.seed(42)

FEATURE_NAMES = [
    "RSA", "ResFlex", "Hydrophobicity", "PackingDensity",
    "HSE_up", "HSE_down",
    "Poly_RSA", "Poly_Flex", "Poly_interaction",
    "BondAngle",
    "sin_phi", "cos_phi", "sin_psi", "cos_psi",
    "sin_omega", "cos_omega"
]

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

def parse_header(header):
    try:
        raw = header.replace(">", "").strip().split()[0].upper()
        if len(raw) >= 5:
            return raw[:4], raw[4]
        if len(raw) == 4:
            return raw, None
        print(f"[WARN] Invalid header format: {header}")
        return None, None
    except Exception as e:
        print(f"[ERROR] parse_header failed for '{header}': {e}")
        return None, None

def download_structure(pdb_id):
    """Download PDB structure, trying PDB format first, then mmCIF fallback"""
    try:
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id.lower()}.pdb")
        cif_path = os.path.join(PDB_DIR, f"{pdb_id.lower()}.cif")
        
        # Check cache
        if os.path.exists(pdb_path):
            print(f"[DEBUG] Using cached: {pdb_id}.pdb")
            return pdb_path, "pdb"
        if os.path.exists(cif_path):
            print(f"[DEBUG] Using cached: {pdb_id}.cif")
            return cif_path, "cif"
        
        # Try downloading both formats
        for ext in ["pdb", "cif"]:
            url = f"https://files.rcsb.org/download/{pdb_id}.{ext}"
            try:
                print(f"[DEBUG] Trying {pdb_id}.{ext}...")
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    path = os.path.join(PDB_DIR, f"{pdb_id.lower()}.{ext}")
                    with open(path, "w") as f:
                        f.write(r.text)
                    print(f"[DEBUG] Successfully downloaded: {pdb_id}.{ext}")
                    return path, ext
            except Exception as e:
                print(f"[WARN] {ext.upper()} download failed: {e}")
                continue
        
        print(f"[ERROR] {pdb_id} not available in PDB or CIF format")
        return None, None
    except Exception as e:
        print(f"[ERROR] download_structure failed for {pdb_id}: {e}")
        return None, None

def select_best_chain(structure, seq):
    try:
        model = structure[0]
        best, diff = None, 1e9
        for c in model:
            try:
                s = "".join(seq1(r.resname) for r in c if "CA" in r)
                if abs(len(s) - len(seq)) < diff:
                    best, diff = c.id, abs(len(s) - len(seq))
            except Exception as e:
                print(f"[WARN] Chain {c.id} processing error: {e}")
                continue
        
        if best is not None:
            print(f"[DEBUG] Auto-selected chain: {best} (length diff: {diff})")
        return best
    except Exception as e:
        print(f"[ERROR] select_best_chain failed: {e}")
        return None

def extract_17D_features(structure, pdb_path, chain_id, fasta_seq):
    try:
        model = structure[0]
        
        # DSSP
        try:
            dssp = DSSP(model, pdb_path, dssp='mkdssp')
            print(f"[DEBUG] DSSP calculated successfully")
        except Exception as e:
            print(f"[ERROR] DSSP failed: {e}")
            return None
        
        # Get chain
        try:
            chain = model[chain_id]
        except KeyError:
            print(f"[ERROR] Chain '{chain_id}' not found. Available: {[c.id for c in model]}")
            return None
        
        residues = [r for r in chain if all(a in r for a in ("CA","N","C","O"))]
        if len(residues) < 3:
            print(f"[ERROR] Too few residues: {len(residues)}")
            return None
        
        if len(residues)/len(fasta_seq) < 0.7:
            print(f"[ERROR] Too many missing residues: {len(residues)}/{len(fasta_seq)} = {len(residues)/len(fasta_seq)*100:.1f}%")
            return None
        
        print(f"[DEBUG] Processing {len(residues)} residues")
        
        # CA coords
        ca_coords = np.array([r["CA"].coord for r in residues], dtype=np.float32)
        ca_tree = KDTree(ca_coords)
        
        # All atoms
        all_atoms = []
        for r in residues:
            for a in r.get_atoms():
                if a.is_disordered():
                    a = a.selected_child
                all_atoms.append(a.coord)
        atom_tree = KDTree(np.array(all_atoms, dtype=np.float32))
        
        # Flexibility
        b = np.array([r["CA"].get_bfactor() for r in residues], dtype=np.float32)
        flex = (b - b.min())/(b.max()-b.min()) if b.max()>b.min() else np.zeros_like(b)
        
        # Packing density
        dens = []
        for r in residues:
            a0 = r["CA"]  # Use CA for deterministic results
            dens.append(len(atom_tree.query_ball_point(a0.coord, 3.5))/max(len(list(r.get_atoms())),1))
        dens = np.array(dens, dtype=np.float32)
        dens = (dens-dens.min())/(dens.max()-dens.min()) if dens.max()>dens.min() else np.zeros_like(dens)
        
        # Phi and Psi
        phis = np.full(len(residues),np.nan)
        psis = np.full(len(residues),np.nan)
        for i,r in enumerate(residues):
            k=(chain_id,r.id)
            if k in dssp:
                if dssp[k][4]!=360: phis[i]=np.radians(dssp[k][4])
                if dssp[k][5]!=360: psis[i]=np.radians(dssp[k][5])
        
        idx=np.arange(len(residues))
        phis=CubicSpline(idx[~np.isnan(phis)],phis[~np.isnan(phis)])(idx) if np.sum(~np.isnan(phis))>=3 else np.nan_to_num(phis)
        psis=CubicSpline(idx[~np.isnan(psis)],psis[~np.isnan(psis)])(idx) if np.sum(~np.isnan(psis))>=3 else np.nan_to_num(psis)
        
        # Extract per-residue features
        feats=[]
        for i,r in enumerate(residues):
            try:
                aa=seq1(r.resname)
                
                # RSA
                try:
                    rsa=dssp[(chain_id,r.id)][3]/MAX_ASA.get(aa,1.0)
                except:
                    rsa=0.0
                
                # Hydrophobicity
                hydro=np.mean([s.get(aa,0.0) for s in HYDRO_SCALES])
                
                # Polynomial
                poly=[rsa,flex[i],rsa*flex[i]]
                
                # Bond angle
                if 0<i<len(residues)-1:
                    try:
                        bond=Vector.angle(residues[i-1]["C"].get_vector()-r["N"].get_vector(),
                                         residues[i+1]["O"].get_vector()-r["N"].get_vector())
                    except:
                        bond=0.0
                else:
                    bond=0.0
                
                # Omega
                try:
                    omega=Vector.angle(r["N"].get_vector()-r["CA"].get_vector(),
                                      r["O"].get_vector()-r["C"].get_vector())
                except:
                    omega=0.0
                
                # HSE (Half-Sphere Exposure) - normalized by neighbor count
                ca=r["CA"].coord
                ref=(r["CB"].coord-ca) if "CB" in r else (r["N"].coord-ca)
                neigh=ca_tree.query_ball_point(ca,8.0)
                up=down=0
                for j in neigh:
                    if j==i: continue
                    if np.dot(ca_coords[j]-ca,ref)>=0: up+=1
                    else: down+=1
                # Normalize by total neighbors
                total = max(len(neigh)-1, 1)
                up = up / total
                down = down / total
                
                feats.append([
                    rsa,flex[i],hydro,dens[i],
                    up,down,
                    *poly,
                    bond,
                    np.sin(phis[i]),np.cos(phis[i]),
                    np.sin(psis[i]),np.cos(psis[i]),
                    np.sin(omega),np.cos(omega)
                ])
            except Exception as e:
                print(f"[ERROR] Feature extraction failed for residue {i}: {e}")
                return None
        
        result = np.array(feats,dtype=np.float32)
        print(f"[DEBUG] Extracted features: {result.shape}")
        return result
        
    except Exception as e:
        print(f"[ERROR] extract_17D_features failed: {e}")
        return None

def build_dataset(fasta_file,out_csv):
    print(f"\n{'='*80}")
    print(f"[START] Processing: {fasta_file}")
    print(f"{'='*80}\n")
    
    rows=[]
    success_count = 0
    failed_count = 0
    
    try:
        with open(fasta_file) as f:
            lines=[l.strip() for l in f if l.strip()]
    except Exception as e:
        print(f"[FATAL] Cannot read {fasta_file}: {e}")
        return
    
    total_proteins = len(lines) // 3
    print(f"[INFO] Total proteins to process: {total_proteins}\n")
    
    for i in range(0,len(lines),3):
        protein_num = i // 3 + 1
        try:
            h,seq,lab=lines[i:i+3]
            print(f"\n[{protein_num}/{total_proteins}] {h}")
            
            pdb,chain=parse_header(h)
            if pdb is None:
                print(f"[SKIP] Header parsing failed")
                failed_count += 1
                continue
            
            path, fmt = download_structure(pdb)
            if path is None:
                print(f"[SKIP] Download failed")
                failed_count += 1
                continue
            
            try:
                s=PDBParser(QUIET=True).get_structure(pdb,path)
            except Exception as e:
                print(f"[SKIP] PDB parsing failed: {e}")
                failed_count += 1
                continue
            
            if chain is None:
                chain=select_best_chain(s,seq)
                if chain is None:
                    print(f"[SKIP] Chain selection failed")
                    failed_count += 1
                    continue
            
            feats=extract_17D_features(s,path,chain,seq)
            if feats is None:
                print(f"[SKIP] Feature extraction failed")
                failed_count += 1
                continue
            
            # Validate feature dimensions
            if feats.shape[1] != len(FEATURE_NAMES):
                print(f"[ERROR] Feature dimension mismatch: {feats.shape[1]} != {len(FEATURE_NAMES)}")
                failed_count += 1
                continue
            
            # Add to rows
            L = min(len(seq),len(feats))
            for j in range(L):
                rows.append([pdb,chain,j+1,seq[j],int(lab[j])]+feats[j].tolist())
            
            print(f"[SUCCESS] Added {L} residues")
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR] Protein {protein_num} failed: {e}")
            failed_count += 1
            continue
    
    # Save dataset
    try:
        df=pd.DataFrame(rows,columns=["PDB","Chain","ResIdx","AA","Label"]+FEATURE_NAMES)
        df.to_csv(out_csv,index=False)
        
        print(f"\n{'='*80}")
        print(f"[SUMMARY]")
        print(f"{'='*80}")
        print(f"Total proteins: {total_proteins}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"Success rate: {success_count/total_proteins*100:.1f}%")
        print(f"Output: {out_csv}")
        print(f"Shape: {df.shape}")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"[FATAL] Could not save CSV: {e}")

if __name__=="__main__":
    build_dataset("data/fasta/Train_335.fa","data/structural/Train_335_16D.csv")
    build_dataset("data/fasta/Test_60.fa","data/structural/Test_60_16D.csv")
    build_dataset("data/fasta/Test_315.fa","data/structural/Test_315_16D.csv")
