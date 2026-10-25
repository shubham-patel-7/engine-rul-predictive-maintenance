"""Train and evaluate Random Forest, XGBoost and LSTM models on C-MAPSS.

Usage:  python -m rul.train --seeds 0 1 2
Writes models/, results/metrics.json, results/predictions_test.csv, results/history.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data import MinMaxScaler, add_rul, load_subset, make_windows, split_units
from .features import window_features
from .metrics import alert_metrics, rmse, summarize
from .models import make_random_forest, make_xgboost, predict_lstm, train_lstm


def _score_block(true_te, pred_te, true_va, pred_va) -> dict:
    test = summarize(true_te, pred_te)
    test["rmse_capped_truth"] = rmse(np.minimum(true_te, C.RUL_CAP), pred_te)
    return {
        "test": test,
        "val_all_cycles": summarize(true_va, pred_va),
        "alert_at_horizon": alert_metrics(true_te, pred_te, horizon=C.ALERT_HORIZON),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--window", type=int, default=C.WINDOW)
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    args = ap.parse_args(argv)
    if args.quick:
        args.epochs = 2
    torch.set_num_threads(2)

    C.MODEL_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)

    # ---- data -------------------------------------------------------------
    train, test, test_rul = load_subset()
    train = add_rul(train)
    tr_df, va_df = split_units(train)
    scaler = MinMaxScaler.fit(tr_df)  # fit on training engines only: no leakage
    tr_s, va_s, te_s = scaler.transform(tr_df), scaler.transform(va_df), scaler.transform(test)

    w_tr = make_windows(tr_s, window=args.window)
    w_va = make_windows(va_s, window=args.window)
    w_te = make_windows(te_s, window=args.window, last_only=True)
    assert np.array_equal(w_te.unit, np.sort(test["unit"].unique())), "test order mismatch"
    print(f"train engines {tr_df.unit.nunique()}  val engines {va_df.unit.nunique()}  test engines {len(w_te.x)}")
    print(f"train windows {len(w_tr.x)}  val windows {len(w_va.x)}")

    f_tr, f_va, f_te = (window_features(w.x) for w in (w_tr, w_va, w_te))
    out: dict = {"config": {"window": args.window, "rul_cap": C.RUL_CAP, "alert_horizon": C.ALERT_HORIZON,
                            "features": C.FEATURES, "seeds": args.seeds,
                            "n_train_engines": int(tr_df.unit.nunique()),
                            "n_val_engines": int(va_df.unit.nunique()),
                            "n_test_engines": int(len(w_te.x))},
                 "models": {}}
    preds = {"unit": w_te.unit, "true_rul": test_rul}

    # ---- tree baselines ---------------------------------------------------
    for name, maker in [("Random Forest", make_random_forest), ("XGBoost", make_xgboost)]:
        print(f"Training {name} ...")
        t0 = time.time()
        m = maker(seed=0)
        m.fit(f_tr, w_tr.rul)
        fit_s = time.time() - t0
        t1 = time.time()
        p_te = np.clip(m.predict(f_te), 0, C.RUL_CAP)
        infer_ms = (time.time() - t1) * 1000 / len(f_te)
        p_va = np.clip(m.predict(f_va), 0, C.RUL_CAP)
        out["models"][name] = {**_score_block(test_rul, p_te, w_va.rul, p_va),
                               "fit_seconds": fit_s, "infer_ms_per_engine": infer_ms}
        preds[name] = p_te
        if name == "XGBoost":
            m.save_model(str(C.MODEL_DIR / "xgboost.json"))
        print(f"  test RMSE {out['models'][name]['test']['rmse']:.2f}")

    # ---- LSTM (several seeds -> mean +/- std and an ensemble) --------------
    seed_te, seed_va, histories, seed_blocks = [], [], {}, []
    for s in args.seeds:
        print(f"Training LSTM seed {s} ...")
        t0 = time.time()
        model, hist = train_lstm(w_tr.x, w_tr.rul, w_va.x, w_va.rul, seed=s, epochs=args.epochs)
        fit_s = time.time() - t0
        t1 = time.time()
        p_te = predict_lstm(model, w_te.x)
        infer_ms = (time.time() - t1) * 1000 / len(w_te.x)
        p_va = predict_lstm(model, w_va.x)
        seed_te.append(p_te)
        seed_va.append(p_va)
        histories[str(s)] = hist
        blk = _score_block(test_rul, p_te, w_va.rul, p_va)
        blk.update(seed=s, fit_seconds=fit_s, infer_ms_per_engine=infer_ms)
        seed_blocks.append(blk)
        torch.save(model.state_dict(), C.MODEL_DIR / f"lstm_seed{s}.pt")
        print(f"  seed {s}: test RMSE {blk['test']['rmse']:.2f}  NASA score {blk['test']['nasa_score']:.0f}")

    rm = [b["test"]["rmse"] for b in seed_blocks]
    out["models"]["LSTM (per-seed)"] = {
        "seeds": seed_blocks,
        "test_rmse_mean": float(np.mean(rm)), "test_rmse_std": float(np.std(rm)),
        "test_nasa_mean": float(np.mean([b["test"]["nasa_score"] for b in seed_blocks])),
        "test_nasa_std": float(np.std([b["test"]["nasa_score"] for b in seed_blocks])),
    }
    ens_te, ens_va = np.mean(seed_te, axis=0), np.mean(seed_va, axis=0)
    out["models"]["LSTM ensemble"] = {**_score_block(test_rul, ens_te, w_va.rul, ens_va),
                                      "n_members": len(args.seeds)}
    preds["LSTM ensemble"] = ens_te
    for s, p in zip(args.seeds, seed_te):
        preds[f"LSTM seed{s}"] = p

    # ---- alert threshold sweep (best model) -------------------------------
    sweep = [alert_metrics(test_rul, ens_te, horizon=C.ALERT_HORIZON, threshold=t) for t in range(10, 81, 5)]
    out["alert_sweep_lstm_ensemble"] = sweep

    # ---- save -------------------------------------------------------------
    scaler.to_json(C.MODEL_DIR / "scaler.json")
    (C.MODEL_DIR / "meta.json").write_text(json.dumps(
        {"window": args.window, "features": C.FEATURES, "rul_cap": C.RUL_CAP, "seeds": args.seeds,
         "val_units": sorted(int(u) for u in va_df.unit.unique())}, indent=2))
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))
    (C.RESULTS_DIR / "history.json").write_text(json.dumps(histories, indent=2))
    pd.DataFrame(preds).to_csv(C.RESULTS_DIR / "predictions_test.csv", index=False)

    print("\n=== Test-set results (100 engines, last cycle) ===")
    for k, v in out["models"].items():
        if "test" in v:
            print(f"{k:16s} RMSE {v['test']['rmse']:6.2f}  MAE {v['test']['mae']:6.2f}  NASA {v['test']['nasa_score']:8.0f}")
    print(f"{'LSTM per-seed':16s} RMSE {out['models']['LSTM (per-seed)']['test_rmse_mean']:.2f} +/- {out['models']['LSTM (per-seed)']['test_rmse_std']:.2f}")


if __name__ == "__main__":
    main()
