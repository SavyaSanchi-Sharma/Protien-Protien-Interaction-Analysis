import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from Bio.PDB import PDBParser
from scipy.spatial import KDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "..")

SCALAR_COLS = ['RSA', 'ResFlex', 'Hydrophobicity', 'PackingDensity', 'HSE_up', 'HSE_down', 'Poly_RSA', 'Poly_Flex', 'Poly_interaction', 'BondAngle']
CONST_COLS = ["Poly_bias"]
ANGLE_COLS = ["sin_phi", "cos_phi", "sin_psi", "cos_psi", "sin_omega", "cos_omega"]
STRUCT_COLS = SCALAR_COLS + CONST_COLS + ANGLE_COLS
NORM_PATH = os.path.join(DATA_ROOT, "data", "struct_norm.npz")


def compute_normalization_from_train(train_csv_path, output_path=None):
    if output_path is None:
        output_path = NORM_PATH
    print(f"[NORM] Computing normalization from TRAIN: {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    scalar_mean = df[SCALAR_COLS].mean().values
    scalar_std = df[SCALAR_COLS].std().values
    scalar_std = np.where(scalar_std < 1e-6, 1.0, scalar_std)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, scalar_mean=scalar_mean, scalar_std=scalar_std)
    print(f"Saved normalization to {output_path}")


class PPISDataset(Dataset):
    def __init__(self, csv_path, esm_dir, pdb_dir, cutoff=14.0, norm_path=None):
        self.df = pd.read_csv(csv_path)
        self.esm_dir = esm_dir
        self.pdb_dir = pdb_dir
        self.cutoff = cutoff
        self.parser = PDBParser(QUIET=True)

        if norm_path is None:
            norm_path = NORM_PATH
        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"Normalization file not found: {norm_path}\n"
                f"Run: compute_normalization_from_train(...)"
            )
        stats = np.load(norm_path)
        self.scalar_mean = torch.from_numpy(stats["scalar_mean"]).float()
        self.scalar_std = torch.from_numpy(stats["scalar_std"]).float()
        print(f"[NORM] Loaded normalization from {norm_path}")

        self.groups = []
        missing = 0
        for (pdb, chain), g in self.df.groupby(["PDB", "Chain"]):
            if self._find_emb_path(pdb, chain, self.esm_dir) is None:
                missing += 1
                print(f"[SKIP] Missing ESM: {pdb}_{chain}")
                continue
            self.groups.append(((pdb, chain), g))

        print(f"[INFO] Total valid groups: {len(self.groups)}")
        print(f"[INFO] Skipped (missing ESM): {missing}\n")

    def __len__(self):
        return len(self.groups)

    def _find_emb_path(self, pdb, chain, emb_dir):
        pdb_u, pdb_l = pdb.upper(), pdb.lower()
        chain_u = chain.upper()
        for name in [f"{pdb_u}_{chain_u}.pt", f"{pdb_l}_{chain_u}.pt",
                     f"{pdb_u}{chain_u}.pt", f"{pdb_l}{chain_u}.pt",
                     f"{pdb_u}.pt", f"{pdb_l}.pt"]:
            path = os.path.join(emb_dir, name)
            if os.path.exists(path):
                return path
        return None

    def __getitem__(self, idx):
        (pdb, chain), g = self.groups[idx]
        struct_raw = torch.tensor(g[STRUCT_COLS].values, dtype=torch.float32)
        struct_raw[:, :len(SCALAR_COLS)] = (
            struct_raw[:, :len(SCALAR_COLS)] - self.scalar_mean
        ) / self.scalar_std

        labels = torch.tensor(g["Label"].values, dtype=torch.long)

        path = self._find_emb_path(pdb, chain, self.esm_dir)
        esm = torch.load(path)
        if esm.ndim != 2 or esm.shape[1] != 2560:
            raise RuntimeError(f"Invalid ESM shape for {pdb}_{chain}: {tuple(esm.shape)}")

        L = min(len(struct_raw), len(esm))
        x = torch.cat([esm[:L], struct_raw[:L]], dim=-1)
        labels = labels[:L]

        edge_index, edge_weight = self.build_edges(pdb, chain, L)
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    def build_edges(self, pdb, chain, L):
        def sequential_graph():
            edges = []
            for i in range(L - 1):
                edges += [(i, i + 1), (i + 1, i)]
            return edges

        edges = None
        pdb_path = os.path.join(self.pdb_dir, f"{pdb.lower()}.pdb")
        if os.path.exists(pdb_path):
            structure = self.parser.get_structure(pdb, pdb_path)
            if chain in structure[0]:
                chain_obj = structure[0][chain]
                residues = [r for r in chain_obj if "CA" in r and r.id[0] == " "]
                if len(residues) != L:
                    print(f"[ALIGN] {pdb}_{chain}: PDB CA count={len(residues)} vs L={L}")
                residues = residues[:L]
                if len(residues) > 0:
                    coords = np.array([r["CA"].coord for r in residues])
                    tree = KDTree(coords)
                    edges = []
                    n = len(coords)
                    for i in range(min(L, n)):
                        for j in tree.query_ball_point(coords[i], self.cutoff):
                            if i == j:
                                continue
                            edges.append((i, j))

        if edges is None or len(edges) == 0:
            edges = sequential_graph()
        if len(edges) == 0:
            edges = [(0, 0)]

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
        return edge_index, edge_weight

