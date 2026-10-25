"""Models: tree baselines and an LSTM regressor (PyTorch)."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch import nn

from . import config as C
from .metrics import rmse


class LSTMRegressor(nn.Module):
    """Stacked LSTM -> MLP head. Outputs RUL scaled to [0, 1] (multiply by RUL_CAP)."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_lstm(model: nn.Module, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Predict RUL in cycles (clipped to [0, RUL_CAP])."""
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i : i + batch])
        outs.append(model(xb).numpy())
    return np.clip(np.concatenate(outs) * C.RUL_CAP, 0, C.RUL_CAP)


def train_lstm(
    x_tr, y_tr, x_va, y_va, seed: int = 0, epochs: int = 60, lr: float = 1e-3,
    batch: int = 256, patience: int = 10, hidden: int = 64, verbose: bool = True,
):
    """Train with Adam + early stopping on validation RMSE (in cycles).

    Returns (best_model, history) where history has per-epoch train loss / val RMSE.
    """
    set_seed(seed)
    model = LSTMRegressor(x_tr.shape[-1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)
    loss_fn = nn.MSELoss()
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr / C.RUL_CAP)
    g = torch.Generator().manual_seed(seed)

    best, best_state, bad = float("inf"), None, 0
    hist = {"train_loss": [], "val_rmse": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        total = 0.0
        for i in range(0, len(xt), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        val = rmse(y_va, predict_lstm(model, x_va))
        sched.step(val)
        hist["train_loss"].append(total / len(xt))
        hist["val_rmse"].append(val)
        if verbose and (ep == 1 or ep % 5 == 0):
            print(f"  seed {seed} epoch {ep:3d}  train_mse {hist['train_loss'][-1]:.5f}  val_rmse {val:6.2f}  ({time.time()-t0:.0f}s)")
        if val < best - 1e-3:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    hist["best_val_rmse"] = best
    hist["epochs_run"] = len(hist["val_rmse"])
    return model, hist


def make_random_forest(seed: int = 0):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=seed
    )


def make_xgboost(seed: int = 0):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
        random_state=seed, n_jobs=-1, tree_method="hist",
    )
