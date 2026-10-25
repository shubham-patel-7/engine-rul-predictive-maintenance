"""Three demo runs of the dashboard logic on held-out engines (never used for training).

Picks the shortest-lived, median-lived and longest-lived held-out engine, replays each one cycle by
cycle exactly as the dashboard does, and reports when the ALERT / WARNING would have fired.

    python scripts/demo_runs.py            # writes results/demo_runs.json and results/demo_runs.md
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rul import config as C  # noqa: E402
from rul.data import add_rul, load_subset  # noqa: E402
from rul.inference import RULPredictor  # noqa: E402

ALERT, WARN = C.ALERT_HORIZON, 50


def first_cycle(mask, cycles):
    idx = np.flatnonzero(mask)
    return int(cycles[idx[0]]) if len(idx) else None


def run_one(pred, eng, label):
    eng = eng.sort_values("cycle")
    traj = pred.predict_trajectory(eng)
    cyc = traj["cycle"].to_numpy()
    life = int(eng["cycle"].max())
    true = eng.set_index("cycle").loc[cyc, "RUL"].to_numpy()   # capped at RUL_CAP, as in training
    p = traj["rul_pred"].to_numpy()
    alert_on, warn_on = p <= ALERT, p <= WARN
    c_alert, c_warn = first_cycle(alert_on, cyc), first_cycle(warn_on, cyc)
    last30 = true <= 30
    return {
        "label": label, "engine": int(eng["unit"].iloc[0]), "life_cycles": life,
        "warning_first_cycle": c_warn,
        "warning_lead_cycles": None if c_warn is None else life - c_warn,
        "alert_first_cycle": c_alert,
        "alert_lead_cycles": None if c_alert is None else life - c_alert,
        "false_alert_cycles": int((alert_on & (true > ALERT)).sum()),     # alert while >30 cycles remain
        "missed_alert_cycles": int((~alert_on & (true <= ALERT)).sum()),  # no alert although <=30 remain
        "alert_toggles": int(np.abs(np.diff(alert_on.astype(int))).sum()),  # on/off flapping
        "rul_error_last_cycle": float(p[-1] - true[-1]),
        "mae_last_30_cycles": float(np.abs(p[last30] - true[last30]).mean()),
        "mae_healthy_phase": float(np.abs(p[true >= C.RUL_CAP] - true[true >= C.RUL_CAP]).mean()) if (true >= C.RUL_CAP).any() else None,
        "std_healthy_phase": float(traj["rul_std"][true >= C.RUL_CAP].mean()) if (true >= C.RUL_CAP).any() else None,
        "std_last_30_cycles": float(traj["rul_std"][last30].mean()),
    }


def main():
    pred = RULPredictor()
    train, _, _ = load_subset()
    train = add_rul(train)
    val = train[train["unit"].isin(pred.val_units)]
    lives = val.groupby("unit")["cycle"].max().sort_values()
    picks = [("Shortest-lived", lives.index[0]), ("Median-lived", lives.index[len(lives) // 2]),
             ("Longest-lived", lives.index[-1])]
    runs = [run_one(pred, val[val["unit"] == u], lab) for lab, u in picks]

    (C.RESULTS_DIR / "demo_runs.json").write_text(json.dumps(runs, indent=2))
    f = lambda v, nd=0: "never" if v is None else f"{v:.{nd}f}"  # noqa: E731
    L = [f"| Demo run | Engine | Life (cycles) | WARNING (pred <= {WARN}) fires at cycle | ALERT (pred <= {ALERT}) fires at cycle | Alert lead time | False-alert cycles | Missed-alert cycles | On/off toggles | Error at last cycle | MAE last 30 cycles |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        L.append(f"| {r['label']} | {r['engine']} | {r['life_cycles']} | {f(r['warning_first_cycle'])} | {f(r['alert_first_cycle'])} | "
                 f"{f(r['alert_lead_cycles'])} cycles | {r['false_alert_cycles']} | {r['missed_alert_cycles']} | {r['alert_toggles']} | "
                 f"{r['rul_error_last_cycle']:+.1f} | {r['mae_last_30_cycles']:.1f} |")
    md = "\n".join(L)
    (C.RESULTS_DIR / "demo_runs.md").write_text(md + "\n")
    print(md)
    print()
    for r in runs:
        print(r["label"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k.startswith(("mae_h", "std_"))})


if __name__ == "__main__":
    main()
