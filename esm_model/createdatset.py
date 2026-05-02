import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from scipy.spatial import KDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "..")

SCALAR_COLS = ['RSA', 'ResFlex', 'Hydrophobicity', 'PackingDensity', 'HSE_up', 'HSE_down',
               'ResidueDepth', 'SurfaceCurvature', 'ElecPotential', 'LocalPlanarity',
               'Poly_interaction',
               'BondAngle']
ANGLE_COLS = ["sin_phi", "cos_phi", "sin_psi", "cos_psi", "sin_omega", "cos_omega"]
STRUCT_COLS = SCALAR_COLS + ANGLE_COLS
COORD_COLS = ["CA_x", "CA_y", "CA_z", "N_x", "N_y", "N_z"]
NORM_PATH = os.path.join(DATA_ROOT, "data", "struct_norm.npz")

EDGE_RBF_NUM = 16
EDGE_MAX_DIST = 14.0
EDGE_ATTR_DIM = EDGE_RBF_NUM + 2

# Must match the layer set in extract_esm_multilayer.py.
NUM_PLM_LAYERS = 6
PLM_DIM = 2560

_RBF_CENTERS = np.linspace(0.0, EDGE_MAX_DIST, EDGE_RBF_NUM, dtype=np.float32)
_RBF_SIGMA = EDGE_MAX_DIST / EDGE_RBF_NUM


def _compute_edge_features_vec(coords_ca, coords_n, edges_arr):
    # coords_ca, coords_n: (L, 3) float32
    # edges_arr: (E, 2) int64 of (src, dst) pairs
    # Returns: (E, EDGE_ATTR_DIM) float32, identical math to the per-edge loop.
    src = edges_arr[:, 0]
    dst = edges_arr[:, 1]
    diff = coords_ca[dst] - coords_ca[src]
    d = np.linalg.norm(diff, axis=1)

    rbf = np.exp(-((d[:, None] - _RBF_CENTERS[None, :]) ** 2)
                 / (2.0 * _RBF_SIGMA * _RBF_SIGMA))

    d_safe = np.where(d < 1e-12, 1.0, d)
    edge_dir = diff / d_safe[:, None]

    bb = coords_ca[src] - coords_n[src]
    bb_norm = np.linalg.norm(bb, axis=1)
    bb_norm_safe = np.where(bb_norm < 1e-12, 1.0, bb_norm)
    bb_unit = bb / bb_norm_safe[:, None]

    cos_t = np.einsum("ij,ij->i", bb_unit, edge_dir)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))

    degenerate = (d < 1e-6) | (bb_norm < 1e-6)
    cos_t = np.where(degenerate, 1.0, cos_t)
    sin_t = np.where(degenerate, 0.0, sin_t)

    out = np.empty((edges_arr.shape[0], EDGE_ATTR_DIM), dtype=np.float32)
    out[:, :EDGE_RBF_NUM] = rbf.astype(np.float32, copy=False)
    out[:, EDGE_RBF_NUM] = sin_t.astype(np.float32, copy=False)
    out[:, EDGE_RBF_NUM + 1] = cos_t.astype(np.float32, copy=False)
    return out


def _build_edges_for_chain(coords_ca, coords_n, L, cutoff):
    edges_arr = np.empty((0, 2), dtype=np.int64)
    if len(coords_ca) > 0:
        tree = KDTree(coords_ca)
        pairs = tree.query_pairs(cutoff, output_type="ndarray")
        if len(pairs) > 0:
            edges_arr = np.concatenate([pairs, pairs[:, ::-1]], axis=0).astype(np.int64, copy=False)

    if len(edges_arr) == 0 and L > 1:
        i = np.arange(L - 1, dtype=np.int64)
        fwd = np.stack([i, i + 1], axis=1)
        bwd = np.stack([i + 1, i], axis=1)
        edges_arr = np.concatenate([fwd, bwd], axis=0)

    if len(edges_arr) == 0:
        edges_arr = np.array([[0, 0]], dtype=np.int64)

    edge_index = torch.from_numpy(edges_arr.T.copy())
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)
    if len(coords_ca) > 0:
        edge_attr_np = _compute_edge_features_vec(coords_ca, coords_n, edges_arr)
    else:
        edge_attr_np = np.zeros((edges_arr.shape[0], EDGE_ATTR_DIM), dtype=np.float32)
        edge_attr_np[:, EDGE_RBF_NUM + 1] = 1.0
    edge_attr = torch.from_numpy(edge_attr_np)
    return edge_index, edge_weight, edge_attr


class PPISDataset(Dataset):
    def __init__(self, csv_path, esm_dir, cutoff=14.0, norm_path=None):
        self.df = pd.read_csv(csv_path)
        missing_cols = [c for c in STRUCT_COLS + COORD_COLS if c not in self.df.columns]
        if missing_cols:
            raise RuntimeError(
                f"{csv_path} is missing columns {missing_cols}. "
                f"Re-run dataprep/dataprep.py to regenerate the structural CSV."
            )
        self.esm_dir = esm_dir
        self.cutoff = cutoff

        if norm_path is None:
            norm_path = NORM_PATH
        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"Normalization file not found: {norm_path}\n"
                f"Run train.py first — it computes normalization from the training set."
            )
        stats = np.load(norm_path)
        if stats["scalar_mean"].shape[0] != len(SCALAR_COLS):
            raise RuntimeError(
                f"Stale {norm_path}: expected {len(SCALAR_COLS)} scalar entries, "
                f"got {stats['scalar_mean'].shape[0]}. Delete it and re-run train.py."
            )
        self.scalar_mean = torch.from_numpy(stats["scalar_mean"]).float()
        self.scalar_std = torch.from_numpy(stats["scalar_std"]).float()
        print(f"[NORM] Loaded normalization from {norm_path}")

        # Eager per-chain cache: struct (normalized), labels, edge_index,
        # edge_weight, edge_attr — everything except the PLM tensor, which is
        # loaded lazily in __getitem__ to keep RAM usage bounded.
        self._cache = []
        missing = 0
        for (pdb, chain), g in self.df.groupby(["PDB", "Chain"]):
            plm_path = self._find_emb_path(pdb, chain, self.esm_dir)
            if plm_path is None:
                missing += 1
                print(f"[SKIP] Missing ESM: {pdb}_{chain}")
                continue

            struct = torch.tensor(g[STRUCT_COLS].values, dtype=torch.float32)
            struct[:, :len(SCALAR_COLS)] = (
                struct[:, :len(SCALAR_COLS)] - self.scalar_mean
            ) / self.scalar_std
            labels = torch.tensor(g["Label"].values, dtype=torch.long)

            coords = g[COORD_COLS].values.astype(np.float32, copy=False)
            coords_ca = np.ascontiguousarray(coords[:, 0:3])
            coords_n  = np.ascontiguousarray(coords[:, 3:6])
            L = struct.shape[0]
            edge_index, edge_weight, edge_attr = _build_edges_for_chain(
                coords_ca, coords_n, L, cutoff
            )

            self._cache.append({
                "plm_path": plm_path,
                "struct": struct.contiguous(),
                "labels": labels,
                "edge_index": edge_index,
                "edge_weight": edge_weight,
                "edge_attr": edge_attr,
                "pos": torch.from_numpy(coords_ca).contiguous(),
            })

        print(f"[INFO] Total valid groups: {len(self._cache)}")
        print(f"[INFO] Skipped (missing ESM): {missing}\n")

    def __len__(self):
        return len(self._cache)

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
        c = self._cache[idx]
        plm = torch.load(c["plm_path"]).float()
        if (plm.ndim != 3 or plm.shape[1] != NUM_PLM_LAYERS
                or plm.shape[2] != PLM_DIM):
            raise RuntimeError(
                f"Invalid ESM shape at {c['plm_path']}: {tuple(plm.shape)}, "
                f"expected (L, {NUM_PLM_LAYERS}, {PLM_DIM})"
            )

        struct_full = c["struct"]
        labels_full = c["labels"]
        L_csv = struct_full.shape[0]
        L_plm = plm.shape[0]
        L = min(L_csv, L_plm)

        if L < L_csv:
            mask = (c["edge_index"][0] < L) & (c["edge_index"][1] < L)
            edge_index = c["edge_index"][:, mask].contiguous()
            edge_weight = c["edge_weight"][mask].contiguous()
            edge_attr = c["edge_attr"][mask].contiguous()
            struct = struct_full[:L].contiguous()
            labels = labels_full[:L]
            pos = c["pos"][:L].contiguous()
        else:
            edge_index = c["edge_index"]
            edge_weight = c["edge_weight"]
            edge_attr = c["edge_attr"]
            struct = struct_full
            labels = labels_full
            pos = c["pos"]

        emb = plm[:L].contiguous()
        return Data(x=struct, emb=emb, pos=pos,
                    edge_index=edge_index, edge_weight=edge_weight,
                    edge_attr=edge_attr, y=labels)
