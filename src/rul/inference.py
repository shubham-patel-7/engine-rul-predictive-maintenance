"""Load trained models and predict RUL trajectories (used by the dashboard and tests)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import MinMaxScaler, make_windows
from .models import LSTMRegressor, predict_lstm


class RULPredictor:
    """Ensemble of LSTM seeds. Mean = prediction, std across members = uncertainty."""

    def __init__(self, model_dir: Path = C.MODEL_DIR):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text())
        self.window: int = meta["window"]
        self.features: list = meta["features"]
        self.val_units: list = meta["val_units"]
        self.scaler = MinMaxScaler.from_json(model_dir / "scaler.json")
        self.models = []
        for s in meta["seeds"]:
            m = LSTMRegressor(len(self.features))
            m.load_state_dict(torch.load(model_dir / f"lstm_seed{s}.pt", map_location="cpu"))
            m.eval()
            self.models.append(m)

    def predict_trajectory(self, engine_df: pd.DataFrame) -> pd.DataFrame:
        """Predicted RUL at every cycle of one engine (using only data up to that cycle)."""
        scaled = self.scaler.transform(engine_df.sort_values("cycle"))
        w = make_windows(scaled, self.features, self.window)
        member = np.stack([predict_lstm(m, w.x) for m in self.models])
        return pd.DataFrame(
            {"cycle": w.cycle, "rul_pred": member.mean(0), "rul_std": member.std(0)}
        )

    def predict_last(self, df: pd.DataFrame) -> pd.DataFrame:
        """One row per engine: predicted RUL at its last observed cycle."""
        scaled = self.scaler.transform(df)
        w = make_windows(scaled, self.features, self.window, last_only=True)
        member = np.stack([predict_lstm(m, w.x) for m in self.models])
        return pd.DataFrame(
            {"unit": w.unit, "cycle": w.cycle, "rul_pred": member.mean(0), "rul_std": member.std(0)}
        )


def alert_status(rul_pred: float, alert_threshold: float, warn_threshold: float) -> str:
    """Traffic-light status from predicted RUL."""
    if rul_pred <= alert_threshold:
        return "ALERT"
    if rul_pred <= warn_threshold:
        return "WARNING"
    return "OK"
