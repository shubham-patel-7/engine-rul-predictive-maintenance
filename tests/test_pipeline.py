"""Smoke tests that need trained artefacts (skipped if `python -m rul.train` hasn't been run)."""
import json
from pathlib import Path

import pytest

from rul import config as C

needs_models = pytest.mark.skipif(
    not (C.MODEL_DIR / "meta.json").exists(), reason="models not trained yet"
)


@needs_models
def test_predictor_trajectory_and_last():
    from rul.data import load_subset
    from rul.inference import RULPredictor

    _, test, _ = load_subset()
    p = RULPredictor()
    eng = test[test["unit"] == 1]
    traj = p.predict_trajectory(eng)
    assert len(traj) == len(eng)
    assert traj["rul_pred"].between(0, C.RUL_CAP).all()
    last = p.predict_last(test)
    assert len(last) == 100


@needs_models
def test_saved_metrics_are_sane():
    m = json.loads((C.RESULTS_DIR / "metrics.json").read_text())
    for name in ("Random Forest", "XGBoost", "LSTM ensemble"):
        assert 5 < m["models"][name]["test"]["rmse"] < 40


@needs_models
def test_dashboard_runs_headless():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 4
