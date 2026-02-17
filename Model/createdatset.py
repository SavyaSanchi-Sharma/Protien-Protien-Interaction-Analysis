import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from Bio.PDB import PDBParser
from scipy.spatial import KDTree
import math

SCALAR_COLS = [
    "RSA", "ResFlex", "Hydrophobicity", "PackingDensity",
    "HSE_up", "HSE_down",
    "Poly_RSA", "Poly_Flex", "Poly_interaction", "BondAngle"
]

PSSM_COLS = [f"PSSM_{i}" for i in range(1, 21)]

ANGLE_COLS = [
    "sin_phi", "cos_phi",
    "sin_psi", "cos_psi",
    "sin_omega", "cos_omega"
]

STRUCT_COLS = SCALAR_COLS + PSSM_COLS + ANGLE_COLS
NORM_PATH = "data/struct_norm.npz"


def compute_normalization_from_train(train_csv_path, output_path=NORM_PATH):
    """
    Compute normalization statistics from TRAIN set ONLY.
    Run this ONCE before training. Never call from dataset.
    Normalizes scalar (10D) and PSSM (20D) features separately.
    """
    print(f"[NORM] Computing normalization from TRAIN: {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    
    # Normalize scalar features (10D)
    scalar_mean = df[SCALAR_COLS].mean().values
    scalar_std = df[SCALAR_COLS].std().values
    scalar_std = np.where(scalar_std < 1e-6, 1.0, scalar_std)
    
    # Normalize PSSM features (20D)
    pssm_mean = df[PSSM_COLS].mean().values
    pssm_std = df[PSSM_COLS].std().values
    pssm_std = np.where(pssm_std < 1e-6, 1.0, pssm_std)
    
    print(f"[NORM] Scalar feature means (10D):\n{scalar_mean}")
    print(f"[NORM] Scalar feature stds (10D):\n{scalar_std}")
    print(f"[NORM] PSSM feature means (20D):\n{pssm_mean}")
    print(f"[NORM] PSSM feature stds (20D):\n{pssm_std}")
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, 
             scalar_mean=scalar_mean, scalar_std=scalar_std,
             pssm_mean=pssm_mean, pssm_std=pssm_std)
    print(f"✓ Saved normalization to {output_path}\n")


class PPISDataset(Dataset):
    def __init__(self, csv_path, esm_dir, pdb_dir, cutoff=10.0, sigma=4.0, norm_path=NORM_PATH):
        self.df = pd.read_csv(csv_path)
        self.esm_dir = esm_dir
        self.pdb_dir = pdb_dir
        self.cutoff = cutoff
        self.sigma = sigma
        self.parser = PDBParser(QUIET=True)

        # Load precomputed normalization (must exist — computed once from TRAIN)
        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"Normalization file not found: {norm_path}\n"
                f"Run: compute_normalization_from_train('data/features/Train_335_36D.csv')"
            )
        
        stats = np.load(norm_path)
        self.scalar_mean = torch.from_numpy(stats["scalar_mean"]).float()
        self.scalar_std = torch.from_numpy(stats["scalar_std"]).float()
        self.pssm_mean = torch.from_numpy(stats["pssm_mean"]).float()
        self.pssm_std = torch.from_numpy(stats["pssm_std"]).float()
        
        print(f"[NORM] Loaded normalization from {norm_path}")
        print(f"[NORM] Scalar means (10D): {self.scalar_mean.numpy()}")
        print(f"[NORM] Scalar stds (10D):  {self.scalar_std.numpy()}")
        print(f"[NORM] PSSM means (20D): {self.pssm_mean.numpy()}")
        print(f"[NORM] PSSM stds (20D):  {self.pssm_std.numpy()}")

        self.groups = []
        missing = 0
        for (pdb, chain), g in self.df.groupby(["PDB", "Chain"]):
            esm_path = self._find_esm_path(pdb, chain)
            if esm_path is None:
                missing += 1
                print(f"[SKIP] Missing ESM: {pdb}_{chain}")
                continue
            self.groups.append(((pdb, chain), g))

        print(f"[INFO] Total valid groups: {len(self.groups)}")
        print(f"[INFO] Skipped groups missing ESM: {missing}\n")

    def __len__(self):
        return len(self.groups)
    
    def _find_esm_path(self, pdb, chain):
        pdb_u = pdb.upper()
        pdb_l = pdb.lower()
        chain_u = chain.upper()
        chain_l = chain.lower()

        candidates = [
            f"{pdb_u}_{chain_u}.pt",
            f"{pdb_l}_{chain_u}.pt",
            f"{pdb_u}{chain_u}.pt",
            f"{pdb_l}{chain_u}.pt",
            f"{pdb_u}.pt",
            f"{pdb_l}.pt",
        ]

        for name in candidates:
            path = os.path.join(self.esm_dir, name)
            if os.path.exists(path):
                return path

        return None

    def _load_esm(self, pdb, chain):
        path = self._find_esm_path(pdb, chain)
        if path is None:
            raise FileNotFoundError(f"ESM embedding not found for {pdb}_{chain}")
        return torch.load(path)

    def __getitem__(self, idx):
        (pdb, chain), g = self.groups[idx]

        # Load all structural features (36D: 10 scalar + 20 PSSM + 6 angle)
        struct_raw = torch.tensor(g[STRUCT_COLS].values, dtype=torch.float32)
        
        # Normalize scalar features (first 10 columns)
        struct_raw[:, :len(SCALAR_COLS)] = (
            struct_raw[:, :len(SCALAR_COLS)] - self.scalar_mean
        ) / self.scalar_std
        
        # Normalize PSSM features (next 20 columns)
        pssm_start = len(SCALAR_COLS)
        pssm_end = pssm_start + len(PSSM_COLS)
        struct_raw[:, pssm_start:pssm_end] = (
            struct_raw[:, pssm_start:pssm_end] - self.pssm_mean
        ) / self.pssm_std
        
        # Angles (last 6 columns) remain untouched [-1, 1]
        struct = struct_raw
        
        labels = torch.tensor(g["Label"].values, dtype=torch.long)

        esm = self._load_esm(pdb, chain)
        if esm.ndim != 2 or esm.shape[1] != 2560:
            raise RuntimeError(
                f"Invalid ESM shape for {pdb}_{chain}: got {tuple(esm.shape)}"
            )
        
        L = min(len(struct), len(esm))
        struct = struct[:L]
        esm = esm[:L]
        labels = labels[:L]

        x = torch.cat([esm, struct], dim=-1)
        edge_index, edge_weight = self.build_edges(pdb, chain, L)

        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=labels)

    def build_edges(self, pdb, chain, L):
        # fallback sequential edges
        def sequential_graph():
            edges = []
            dists = []
            for i in range(L - 1):
                edges += [(i, i+1), (i+1, i)]
                dists += [1.0, 1.0]
            return edges, dists

        pdb_path = os.path.join(self.pdb_dir, f"{pdb.lower()}.pdb")
        if not os.path.exists(pdb_path):
            edges, dists = sequential_graph()
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
            return edge_index, edge_weight

        structure = self.parser.get_structure(pdb, pdb_path)
        if chain not in structure[0]:
            edges, dists = sequential_graph()
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
            return edge_index, edge_weight

        chain_obj = structure[0][chain]
        residues = [r for r in chain_obj if "CA" in r][:L]
        if len(residues) == 0:
            edges, dists = sequential_graph()
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
            return edge_index, edge_weight

        coords = np.array([r["CA"].coord for r in residues])
        tree = KDTree(coords)

        edges = []
        dists = []

        # sequential edges (strong)
        for i in range(L-1):
            edges += [(i, i+1), (i+1, i)]
            dists += [1.0, 1.0]

        # spatial edges
        for i in range(L):
            nbrs = tree.query_ball_point(coords[i], self.cutoff)
            for j in nbrs:
                if i == j or abs(i - j) == 1:
                    continue
                dij = float(np.linalg.norm(coords[i] - coords[j]))
                edges.append((i, j))
                dists.append(dij)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # exp kernel weights
        sigma = self.sigma
        edge_weight = torch.tensor(
            [math.exp(-(d*d) / (2 * sigma * sigma)) for d in dists],
            dtype=torch.float32
        )
        return edge_index, edge_weight
