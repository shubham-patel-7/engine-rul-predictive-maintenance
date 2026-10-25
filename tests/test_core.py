import numpy as np
import pytest

from rul import config as C
from rul.data import MinMaxScaler, add_rul, load_subset, make_windows, split_units
from rul.features import window_features
from rul.metrics import alert_metrics, nasa_score, r2, rmse


@pytest.fixture(scope="module")
def data():
    return load_subset()


def test_dataset_shapes(data):
    train, test, rul = data
    assert train["unit"].nunique() == 100 and test["unit"].nunique() == 100
    assert len(rul) == 100 and train.shape[1] == 26


def test_rul_label_is_capped_and_hits_zero(data):
    train = add_rul(data[0])
    assert train["RUL"].max() == C.RUL_CAP
    assert (train.groupby("unit")["RUL"].min() == 0).all()


def test_split_is_by_engine_with_no_overlap(data):
    tr, va = split_units(add_rul(data[0]))
    assert set(tr["unit"]).isdisjoint(set(va["unit"]))
    assert va["unit"].nunique() == C.VAL_UNITS


def test_scaler_fit_on_train_only_and_roundtrip(data, tmp_path):
    tr, _ = split_units(add_rul(data[0]))
    sc = MinMaxScaler.fit(tr)
    z = sc.transform(tr)[C.FEATURES]
    assert z.min().min() >= -1e-9 and z.max().max() <= 1 + 1e-9
    sc.to_json(tmp_path / "s.json")
    sc2 = MinMaxScaler.from_json(tmp_path / "s.json")
    assert np.allclose(sc.min_, sc2.min_) and np.allclose(sc.max_, sc2.max_)


def test_windows_shape_padding_and_last_only(data):
    train = add_rul(data[0])
    w = make_windows(train, window=30)
    assert w.x.shape == (len(train), 30, len(C.FEATURES))
    # first window of an engine is its first row repeated (left padding)
    first = train[train["unit"] == 1].iloc[0][C.FEATURES].to_numpy(np.float32)
    assert np.allclose(w.x[0, 0], first) and np.allclose(w.x[0, -1], first)
    # last window label is RUL 0 for a run-to-failure engine
    last = make_windows(train, window=30, last_only=True)
    assert len(last.x) == 100 and (last.rul == 0).all()


def test_window_does_not_look_into_the_future(data):
    train = add_rul(data[0])
    g = train[train["unit"] == 1]
    w = make_windows(g, window=5)
    t = 10  # window ending at row t must equal rows t-4..t
    assert np.allclose(w.x[t], g[C.FEATURES].to_numpy(np.float32)[t - 4 : t + 1])


def test_window_features_slope_detects_trend():
    x = np.tile(np.arange(30, dtype=np.float32)[None, :, None], (2, 1, 3))
    f = window_features(x)
    assert f.shape == (2, 12)
    assert np.allclose(f[:, 9:], 1.0, atol=1e-5)  # slope of a unit ramp is 1


def test_metrics_known_values():
    assert rmse([10, 20], [12, 18]) == pytest.approx(2.0)
    assert r2([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    # late (pred > true) must be punished more than early by the same amount
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert nasa_score([50], [50]) == 0


def test_alert_metrics_confusion_counts():
    true = np.array([5, 10, 40, 80])
    pred = np.array([8, 50, 20, 90])
    m = alert_metrics(true, pred, horizon=30)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
