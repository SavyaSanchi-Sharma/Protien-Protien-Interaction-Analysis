import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from Bio.PDB import PDBParser
from scipy.spatial import KDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "..")


class PPISDataset(Dataset):
    def __init__(self, csv_path, protbert_dir, pdb_dir, cutoff=14.0):
        self.df = pd.read_csv(csv_path)
        self.protbert_dir = protbert_dir
        self.pdb_dir = pdb_dir
        self.cutoff = cutoff
        self.parser = PDBParser(QUIET=True)

        self.groups = []
        missing = 0
        for (pdb, chain), g in self.df.groupby(["PDB", "Chain"]):
            if self._find_emb_path(pdb, chain, self.protbert_dir) is None:
                missing += 1
                print(f"[SKIP] Missing ProtBERT: {pdb}_{chain}")
                continue
            self.groups.append(((pdb, chain), g))

        print(f"[INFO] Total valid groups: {len(self.groups)}")
        print(f"[INFO] Skipped (missing ProtBERT): {missing}\n")

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
        labels = torch.tensor(g["Label"].values, dtype=torch.long)

        path = self._find_emb_path(pdb, chain, self.protbert_dir)
        emb = torch.load(path)
        if emb.ndim != 2 or emb.shape[1] != 1024:
            raise RuntimeError(f"Invalid ProtBERT shape for {pdb}_{chain}: {tuple(emb.shape)}")

        L = min(len(labels), len(emb))
        x = emb[:L]
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

