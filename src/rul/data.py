"""Loading, labelling, scaling and windowing the NASA C-MAPSS turbofan data."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def load_subset(data_dir: Path = C.DATA_DIR, subset: str = C.SUBSET):
    """Return (train_df, test_df, test_rul) for one C-MAPSS subset.

    train_df: run-to-failure trajectories for every engine.
    test_df:  trajectories that stop *before* failure.
    test_rul: true remaining useful life at the last observed test cycle.
    """
    data_dir = Path(data_dir)

    def read(name: str) -> pd.DataFrame:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python scripts/download_data.py` first."
            )
        return pd.read_csv(path, sep=r"\s+", header=None, names=C.COLUMNS)

    train = read(f"train_{subset}.txt")
    test = read(f"test_{subset}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None, names=["RUL"])["RUL"].to_numpy()
    if len(rul) != test["unit"].nunique():
        raise ValueError("RUL file does not match number of test engines")
    return train, test, rul


def add_rul(train: pd.DataFrame, cap: int = C.RUL_CAP) -> pd.DataFrame:
    """Add a piecewise-linear RUL label: remaining cycles, clipped at `cap`."""
    out = train.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    out["RUL"] = (max_cycle - out["cycle"]).clip(upper=cap)
    return out


def split_units(train: pd.DataFrame, n_val: int = C.VAL_UNITS, seed: int = C.SPLIT_SEED):
    """Split by *engine* (never by row) so no engine leaks across train/val."""
    units = np.sort(train["unit"].unique())
    rng = np.random.default_rng(seed)
    val_units = np.sort(rng.choice(units, size=n_val, replace=False))
    is_val = train["unit"].isin(val_units)
    return train[~is_val].copy(), train[is_val].copy()


@dataclass
class MinMaxScaler:
    """Tiny min-max scaler that serialises to JSON (no pickle needed)."""

    features: list
    min_: np.ndarray
    max_: np.ndarray

    @classmethod
    def fit(cls, df: pd.DataFrame, features=C.FEATURES) -> "MinMaxScaler":
        arr = df[features].to_numpy(dtype=np.float64)
        return cls(list(features), arr.min(axis=0), arr.max(axis=0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        span = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        out[self.features] = (df[self.features].to_numpy(np.float64) - self.min_) / span
        return out

    def to_json(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"features": self.features, "min": self.min_.tolist(), "max": self.max_.tolist()}
            )
        )

    @classmethod
    def from_json(cls, path: Path) -> "MinMaxScaler":
        d = json.loads(Path(path).read_text())
        return cls(d["features"], np.array(d["min"]), np.array(d["max"]))


@dataclass
class Windows:
    x: np.ndarray  # (N, window, n_features) float32
    unit: np.ndarray  # (N,)
    cycle: np.ndarray  # (N,) cycle at the end of each window
    rul: np.ndarray | None  # (N,) capped RUL label, if available


def make_windows(
    df: pd.DataFrame,
    features=C.FEATURES,
    window: int = C.WINDOW,
    last_only: bool = False,
) -> Windows:
    """Sliding windows ending at every cycle of every engine.

    Engines shorter than `window` are left-padded by repeating their first
    observation, so every cycle gets a window (needed for trajectory plots).
    With `last_only=True` only the final window per engine is returned, which
    is how the NASA test set is scored.
    """
    xs, units, cycles, ruls = [], [], [], []
    has_rul = "RUL" in df.columns
    for unit, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        arr = g[list(features)].to_numpy(np.float32)
        n = len(arr)
        padded = np.concatenate([np.repeat(arr[:1], window - 1, axis=0), arr], axis=0)
        idx = np.arange(n)[:, None] + np.arange(window)[None, :]
        w = padded[idx]
        cyc = g["cycle"].to_numpy()
        r = g["RUL"].to_numpy() if has_rul else None
        if last_only:
            w, cyc = w[-1:], cyc[-1:]
            r = r[-1:] if r is not None else None
        xs.append(w)
        units.append(np.full(len(w), unit))
        cycles.append(cyc)
        if r is not None:
            ruls.append(r)
    return Windows(
        np.concatenate(xs),
        np.concatenate(units),
        np.concatenate(cycles),
        np.concatenate(ruls).astype(np.float32) if has_rul else None,
    )
