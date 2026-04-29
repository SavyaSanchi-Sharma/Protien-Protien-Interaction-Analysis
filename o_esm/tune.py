import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from torch.amp import autocast, GradScaler
import optuna

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "..")
sys.path.insert(0, ROOT)

from createdatset import PPISDataset
from model.esm_projection import ESMProjection
from model.gcn import GCNEncoder
from model.tcn import BiTCN
from model.classifier import Classifier
from train import (TRAIN_FULL_CSV, TRAIN_CSV, VAL_CSV, ESM_DIR, PDB_DIR,
                   ensure_train_val_split, hybrid_focal_cost_loss,
                   find_mcc_threshold, compute_dataset_alpha, set_seed)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build(trial):
    hp = {
        "lr":            trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "focal_gamma":   trial.suggest_float("focal_gamma", 1.0, 3.0),
        "proj_dropout":  trial.suggest_float("proj_dropout", 0.05, 0.4),
        "gcn_layers":    trial.suggest_int("gcn_layers", 4, 10),
        "gcn_alpha":     trial.suggest_float("gcn_alpha", 0.1, 0.7),
        "gcn_dropout":   trial.suggest_float("gcn_dropout", 0.05, 0.3),
        "tcn_dropout":   trial.suggest_float("tcn_dropout", 0.05, 0.3),
        "clf_dropout":   trial.suggest_float("clf_dropout", 0.1, 0.5),
        "lambda_mcc":    trial.suggest_float("lambda_mcc", 0.0, 2.0),
        "batch_size":    trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16]),
    }
    proj = ESMProjection(2560, 512, 256, hp["proj_dropout"]).to(DEVICE)
    gcn  = GCNEncoder(256, 256, hp["gcn_layers"], hp["gcn_alpha"], hp["gcn_dropout"]).to(DEVICE)
    tcn  = BiTCN(256, [64, 128, 256, 512], hp["tcn_dropout"]).to(DEVICE)
    clf  = Classifier(1024, hp["clf_dropout"]).to(DEVICE)
    return hp, proj, gcn, tcn, clf


def step(modules, data, alpha_pos, gamma, lambda_mcc):
    proj, gcn, tcn, clf = modules
    h = proj(data.x)
    h = gcn(h, data.edge_index, edge_weight=data.edge_weight)
    padded, mask = to_dense_batch(h, data.batch)
    lengths = mask.sum(dim=1)
    h = tcn(padded, lengths=lengths)[mask]
    logits = clf(h)
    y = data.y.long().view(-1)
    loss = hybrid_focal_cost_loss(logits, y, alpha_pos, gamma=gamma, lambda_mcc=lambda_mcc)
    return logits, y, loss


def objective(trial, train_ds, val_ds, alpha_pos, max_epochs):
    hp, *modules = build(trial)
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False)
    params = [p for m in modules for p in m.parameters()]
    optimizer = optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])
    scaler = GradScaler("cuda")

    best_mcc = -1.0
    for epoch in range(max_epochs):
        for m in modules: m.train()
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                _, _, loss = step(modules, data, alpha_pos, hp["focal_gamma"], hp["lambda_mcc"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer); scaler.update()

        for m in modules: m.eval()
        probs_all, labels_all = [], []
        with torch.no_grad(), autocast(device_type="cuda"):
            for data in val_loader:
                data = data.to(DEVICE)
                logits, y, _ = step(modules, data, alpha_pos, hp["focal_gamma"], hp["lambda_mcc"])
                probs_all.append(torch.softmax(logits, dim=1)[:, 1].cpu())
                labels_all.append(y.cpu())
        probs = torch.cat(probs_all).numpy()
        labels = torch.cat(labels_all).numpy()
        _, mcc_opt = find_mcc_threshold(probs, labels)
        best_mcc = max(best_mcc, mcc_opt)
        trial.report(mcc_opt, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_mcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    ap.add_argument("--no-prune", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_train_val_split(TRAIN_FULL_CSV, TRAIN_CSV, VAL_CSV)
    train_ds = PPISDataset(TRAIN_CSV, ESM_DIR, PDB_DIR)
    val_ds   = PPISDataset(VAL_CSV,   ESM_DIR, PDB_DIR)
    alpha_pos = compute_dataset_alpha(train_ds)

    sampler = (optuna.samplers.RandomSampler(seed=args.seed)
               if args.sampler == "random"
               else optuna.samplers.TPESampler(seed=args.seed))
    pruner = (optuna.pruners.NopPruner()
              if args.no_prune
              else optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda t: objective(t, train_ds, val_ds, alpha_pos, args.epochs),
        n_trials=args.trials,
    )
    print(f"\nBest trial #{study.best_trial.number}: MCC={study.best_trial.value:.4f}")
    print("Best params:", study.best_trial.params)
    out = os.path.join(ROOT, "best_hp.json")
    with open(out, "w") as f:
        json.dump({"value": study.best_trial.value, "params": study.best_trial.params}, f, indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
