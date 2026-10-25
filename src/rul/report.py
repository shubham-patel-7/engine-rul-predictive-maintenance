"""Generate figures and a results table from saved training outputs.

Usage:  python -m rul.report
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .data import add_rul, load_subset
from .metrics import alert_metrics

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # fixed categorical order
MODEL_COLOR = {"Random Forest": AQUA, "XGBoost": ORANGE, "LSTM ensemble": BLUE, "LSTM": BLUE}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 9.5, "legend.frameon": False, "figure.dpi": 150, "lines.linewidth": 2,
    })


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def fig_model_comparison(m):
    mods = m["models"]
    seed = mods["LSTM (per-seed)"]
    rmse = [mods["Random Forest"]["test"]["rmse"], mods["XGBoost"]["test"]["rmse"],
            seed["test_rmse_mean"], mods["LSTM ensemble"]["test"]["rmse"]]
    nasa = [mods["Random Forest"]["test"]["nasa_score"], mods["XGBoost"]["test"]["nasa_score"],
            seed["test_nasa_mean"], mods["LSTM ensemble"]["test"]["nasa_score"]]
    err = [0, 0, seed["test_rmse_std"], 0], [0, 0, seed["test_nasa_std"], 0]
    colors = [AQUA, ORANGE, BLUE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, vals, e, title, fmt in [(axes[0], rmse, err[0], "Test RMSE (cycles, lower is better)", "{:.1f}"),
                                    (axes[1], nasa, err[1], "NASA score (lower is better)", "{:,.0f}")]:
        bars = ax.bar(range(4), vals, color=colors, width=0.55, yerr=e, ecolor=INK2, capsize=3,
                      edgecolor=SURFACE, linewidth=1.5)
        bars[3].set_hatch("////")
        bars[3].set_edgecolor(SURFACE)
        for i, v in enumerate(vals):
            ax.text(i, v + max(e[i], 0) + max(vals) * 0.02, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["Random\nForest", "XGBoost", "LSTM\n(mean of seeds)", "LSTM\nensemble"])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(vals) * 1.18)
    fig.tight_layout()
    _save(fig, "model_comparison.png")


def fig_pred_vs_true(pred):
    cols = ["Random Forest", "XGBoost", "LSTM ensemble"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    lim = max(pred["true_rul"].max(), 130) + 5
    for ax, c in zip(axes, cols):
        r = float(np.sqrt(np.mean((pred[c] - pred["true_rul"]) ** 2)))
        ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
        ax.scatter(pred["true_rul"], pred[c], s=26, color=MODEL_COLOR[c], edgecolor=SURFACE, linewidth=1, alpha=0.9)
        ax.set_title(f"{c}  (RMSE {r:.1f})")
        ax.set_xlabel("True RUL (cycles)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted RUL (cycles)")
    fig.tight_layout()
    fig.text(0.5, -0.07, "Dashed line = perfect prediction. Points above it are late (optimistic) predictions - the dangerous kind.",
             ha="center", color=INK2, fontsize=9)
    _save(fig, "predicted_vs_true.png")


def fig_training_curves(hist):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    styles = ["-", "--", ":"]
    for (seed, h), ls in zip(hist.items(), styles):
        ax.plot(range(1, len(h["val_rmse"]) + 1), h["val_rmse"], color=BLUE, ls=ls, label=f"seed {seed}")
        b = int(np.argmin(h["val_rmse"]))
        ax.scatter([b + 1], [h["val_rmse"][b]], s=36, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation RMSE (cycles)")
    ax.set_title("LSTM validation error by epoch (dot = early-stopping checkpoint)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "training_curves.png")


MODELS3 = ["Random Forest", "XGBoost", "LSTM ensemble"]


def sweep_all(pred, horizon):
    """Alert precision/recall for every model across a range of alert thresholds."""
    rows = []
    for name in MODELS3:
        for t in range(10, 81, 5):
            a = alert_metrics(pred["true_rul"], pred[name], horizon=horizon, threshold=t)
            rows.append({"model": name, **a})
    return pd.DataFrame(rows)


def ranking_metrics(pred, horizon):
    """Threshold-free comparison: how well does predicted RUL *rank* engines that fail within `horizon`?"""
    truth = (pred["true_rul"] <= horizon).astype(int)
    out = {}
    for name in MODELS3:
        out[name] = {"ap": float(average_precision_score(truth, -pred[name])),
                     "auc": float(roc_auc_score(truth, -pred[name]))}
    return out, int(truth.sum())


def paired_bootstrap_rmse(pred, a, b, n=5000, seed=0):
    """95% CI for RMSE(a) - RMSE(b), resampling test engines with replacement (paired)."""
    rng = np.random.default_rng(seed)
    y = pred["true_rul"].to_numpy()
    ea, eb = (pred[a].to_numpy() - y) ** 2, (pred[b].to_numpy() - y) ** 2
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d = np.sqrt(ea[idx].mean(1)) - np.sqrt(eb[idx].mean(1))
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fig_alert_tradeoff(m, pred):
    h = m["config"]["alert_horizon"]
    sw = sweep_all(pred, h)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharex=True)
    for ax, key, title in [(axes[0], "recall", "Recall: real failures caught"),
                           (axes[1], "precision", "Precision: alerts that were real")]:
        for name in MODELS3:
            d = sw[sw["model"] == name]
            ax.plot(d["threshold"], d[key], color=MODEL_COLOR[name], label=name)
        ax.axvline(h, color=INK2, lw=1, ls=":")
        ax.set_xlabel("Alert fires when predicted RUL <= threshold (cycles)")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
    axes[0].text(h + 0.8, 0.04, f"{h}-cycle failure horizon", color=INK2, fontsize=8.5)
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "alert_tradeoff.png")


def fig_example_trajectories():
    from .inference import RULPredictor

    train, _, _ = load_subset()
    train = add_rul(train)
    pred = RULPredictor()
    val = train[train["unit"].isin(pred.val_units)]
    life = val.groupby("unit")["cycle"].max().sort_values()
    picks = [life.index[0], life.index[len(life) // 2], life.index[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
    for ax, u in zip(axes, picks):
        g = val[val["unit"] == u]
        t = pred.predict_trajectory(g)
        ax.fill_between(t["cycle"], t["rul_pred"] - t["rul_std"], t["rul_pred"] + t["rul_std"], color=BLUE, alpha=0.18, lw=0)
        ax.plot(g["cycle"], g["RUL"], color=INK2, lw=1.5, ls="--", label="True RUL (capped at 125)")
        ax.plot(t["cycle"], t["rul_pred"], color=BLUE, label="Predicted RUL (LSTM ensemble)")
        ax.axhline(C.ALERT_HORIZON, color=ORANGE, lw=1.2)
        ax.text(g["cycle"].min() + 2, C.ALERT_HORIZON + 3, "alert threshold", color=INK2, fontsize=8)
        ax.set_title(f"Held-out engine {u} (life {life[u]} cycles)")
        ax.set_xlabel("Cycle")
    axes[0].set_ylabel("RUL (cycles)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, "example_trajectories.png")


def fig_degradation_signals():
    train, _, _ = load_subset()
    train = add_rul(train)
    train["ttf"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    sensors = ["s11", "s4", "s12", "s9"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    rng = np.random.default_rng(0)
    for ax, s in zip(axes, sensors):
        for u in rng.choice(train["unit"].unique(), 12, replace=False):
            g = train[train["unit"] == u]
            ax.plot(g["ttf"], g[s], color=INK2, alpha=0.25, lw=0.8)
        mean = train[train["ttf"] <= 250].groupby("ttf")[s].mean()
        ax.plot(mean.index, mean.values, color=BLUE, lw=2)
        ax.set_xlim(250, 0)
        ax.set_title(f"{s}  {C.SENSOR_NAMES[s].split(' - ')[0]}", fontsize=10)
        ax.set_xlabel("Cycles before failure")
    fig.suptitle("Sensors drift as the engine degrades (grey = 12 engines, blue = fleet mean)", x=0.01, ha="left",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    _save(fig, "degradation_signals.png")


def results_table(m, pred) -> str:
    mods = m["models"]
    rank, n_pos = ranking_metrics(pred, m["config"]["alert_horizon"])
    h = m["config"]["alert_horizon"]
    rows = []
    for name in MODELS3:
        t = mods[name]["test"]
        a = mods[name]["alert_at_horizon"]
        rows.append((name, t["rmse"], t["mae"], t["nasa_score"], t["r2"], a["recall"], a["precision"],
                     rank[name]["ap"], rank[name]["auc"]))
    s = mods["LSTM (per-seed)"]
    near = pred["true_rul"] <= h
    near_rmse = {n: float(np.sqrt(np.mean((pred.loc[near, n] - pred.loc[near, "true_rul"]) ** 2))) for n in MODELS3}
    lines = [f"| Model | RMSE | RMSE (true RUL ≤ {h}) | MAE | NASA score | R² | Recall @{h} | Precision @{h} | Ranking AP | Ranking AUC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | {r[1]:.2f} | {near_rmse[r[0]]:.2f} | {r[2]:.2f} | {r[3]:,.0f} | {r[4]:.3f} | {r[5]:.2f} | {r[6]:.2f} | {r[7]:.3f} | {r[8]:.3f} |")
    lines.append(f"| LSTM single seed (mean ± std, n={len(s['seeds'])}) | {s['test_rmse_mean']:.2f} ± {s['test_rmse_std']:.2f} | - | - | "
                 f"{s['test_nasa_mean']:,.0f} ± {s['test_nasa_std']:,.0f} | - | - | - | - | - |")
    lines.append("")
    for x, y in [("LSTM ensemble", "XGBoost"), ("Random Forest", "XGBoost"), ("LSTM ensemble", "Random Forest")]:
        d, lo, hi = paired_bootstrap_rmse(pred, x, y)
        lines.append(f"- RMSE({x}) - RMSE({y}) = {d:+.2f} cycles, 95% bootstrap CI [{lo:+.2f}, {hi:+.2f}]"
                     f"{' (CI includes 0: not a clear difference)' if lo < 0 < hi else ''}")
    lines.append("")
    lines.append(f"*Ranking AP/AUC: how well predicted RUL separates the {n_pos} test engines that really fail within {h} cycles "
                 "from the rest; threshold-free, so models that simply predict lower values get no advantage.*")
    return "\n".join(lines)


def main():
    _style()
    m = json.loads((C.RESULTS_DIR / "metrics.json").read_text())
    pred = pd.read_csv(C.RESULTS_DIR / "predictions_test.csv")
    hist = json.loads((C.RESULTS_DIR / "history.json").read_text())
    fig_model_comparison(m)
    fig_pred_vs_true(pred)
    fig_training_curves(hist)
    fig_alert_tradeoff(m, pred)
    fig_example_trajectories()
    fig_degradation_signals()
    table = results_table(m, pred)
    (C.RESULTS_DIR / "results_table.md").write_text(table + "\n")
    print(table)
    print(f"Figures written to {C.FIG_DIR}")


if __name__ == "__main__":
    main()
