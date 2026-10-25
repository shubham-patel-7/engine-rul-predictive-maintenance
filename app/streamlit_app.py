"""Engine fleet health dashboard.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works even without `pip install -e .`

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from rul import config as C
from rul.data import add_rul, load_subset
from rul.inference import RULPredictor, alert_status

st.set_page_config(page_title="Engine RUL dashboard", page_icon="🛠️", layout="wide")

BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e1"
STATUS_ICON = {"OK": "🟢 OK", "WARNING": "🟡 WARNING", "ALERT": "🔴 ALERT"}


@st.cache_resource
def get_predictor():
    return RULPredictor()


@st.cache_data
def get_data():
    train, test, test_rul = load_subset()
    return add_rul(train), test, test_rul


@st.cache_data
def fleet_table(_pred, alert_thr: float, warn_thr: float):
    _, test, test_rul = get_data()
    last = _pred.predict_last(test)
    last["true_rul"] = test_rul
    last["status"] = [alert_status(r, alert_thr, warn_thr) for r in last["rul_pred"]]
    return last


def main():
    st.title("🛠️ Turbofan engine fleet - remaining useful life")
    st.caption(
        "LSTM ensemble trained on NASA C-MAPSS FD001. Predicts how many cycles each engine has left "
        "and raises an alert before it gets there."
    )
    try:
        pred = get_predictor()
        train, test, test_rul = get_data()
    except FileNotFoundError as e:
        st.error(f"{e}\n\nTrain the models first: `python -m rul.train`")
        st.stop()

    # ---------------- sidebar ----------------
    st.sidebar.header("Controls")
    pool = st.sidebar.radio(
        "Engine pool",
        ["Held-out engines (run to failure)", "Test fleet (stops before failure)"],
        help="Held-out engines were never used for training and we know their full life, so the "
        "true RUL curve can be drawn. Test-fleet engines stop before failure; only the final RUL is known.",
    )
    alert_thr = st.sidebar.slider("Alert threshold (cycles)", 10, 60, C.ALERT_HORIZON, 5,
                                  help="ALERT when predicted RUL falls to or below this value.")
    warn_thr = st.sidebar.slider("Warning threshold (cycles)", alert_thr + 5, 100, max(alert_thr + 20, 50), 5)

    if pool.startswith("Held-out"):
        df_pool = train[train["unit"].isin(pred.val_units)]
        units = sorted(pred.val_units)
    else:
        df_pool = test
        units = sorted(test["unit"].unique().tolist())
    unit = st.sidebar.selectbox("Engine", units)
    eng = df_pool[df_pool["unit"] == unit].sort_values("cycle")

    traj = pred.predict_trajectory(eng)
    last_cycle = int(eng["cycle"].max())
    replay = st.sidebar.slider("Replay up to cycle", int(eng["cycle"].min()) + 1, last_cycle, last_cycle,
                               help="Scrub back in time to see what the model knew at that point.")
    shown = traj[traj["cycle"] <= replay]
    now = shown.iloc[-1]
    status = alert_status(now["rul_pred"], alert_thr, warn_thr)

    # ---------------- headline ----------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted RUL", f"{now['rul_pred']:.0f} cycles", help="Mean of the LSTM ensemble")
    c2.metric("Model disagreement", f"± {now['rul_std']:.1f}", help="Std. dev. across ensemble members; a wide gap means low confidence.")
    c3.metric("Cycles run", f"{replay}")
    c4.metric("Status", STATUS_ICON[status])
    if status == "ALERT":
        st.error(f"Engine {unit}: predicted to fail within ~{now['rul_pred']:.0f} cycles. Schedule maintenance.")
    elif status == "WARNING":
        st.warning(f"Engine {unit}: degradation detected. Plan an inspection.")
    else:
        st.success(f"Engine {unit}: healthy.")

    # ---------------- RUL trajectory ----------------
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Predicted RUL over time")
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ax.fill_between(shown["cycle"], shown["rul_pred"] - shown["rul_std"], shown["rul_pred"] + shown["rul_std"],
                        color=BLUE, alpha=0.18, lw=0)
        ax.plot(shown["cycle"], shown["rul_pred"], color=BLUE, lw=2, label="Predicted RUL")
        if pool.startswith("Held-out"):
            truth = eng[eng["cycle"] <= replay]
            ax.plot(truth["cycle"], truth["RUL"], color=INK2, lw=1.4, ls="--", label=f"True RUL (capped at {C.RUL_CAP})")
        else:
            ax.scatter([last_cycle], [test_rul[units.index(unit)]], s=60, color=INK2, zorder=4, label="True RUL at last cycle")
        ax.axhline(alert_thr, color=ORANGE, lw=1.3, label="Alert threshold")
        ax.axhline(warn_thr, color=ORANGE, lw=1, ls=":", label="Warning threshold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL (cycles)")
        ax.set_ylim(0, 150)
        ax.grid(color=GRID)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Sensor trends")
        picks = st.multiselect("Sensors", C.FEATURES, default=["s11", "s4", "s12"],
                               format_func=lambda s: f"{s} - {C.SENSOR_NAMES[s]}")
        if picks:
            sc = pred.scaler.transform(eng[eng["cycle"] <= replay])
            st.line_chart(sc.set_index("cycle")[picks], height=300)
            st.caption("Min-max normalised using training-engine ranges.")

    # ---------------- fleet view ----------------
    st.subheader("Fleet view: test engines ranked by urgency")
    fleet = fleet_table(pred, alert_thr, warn_thr).sort_values("rul_pred")
    counts = fleet["status"].value_counts()
    m1, m2, m3 = st.columns(3)
    m1.metric("🔴 Alerts", int(counts.get("ALERT", 0)))
    m2.metric("🟡 Warnings", int(counts.get("WARNING", 0)))
    m3.metric("🟢 Healthy", int(counts.get("OK", 0)))

    truth_close = fleet["true_rul"] <= alert_thr
    caught = int(((fleet["status"] == "ALERT") & truth_close).sum())
    missed = int(((fleet["status"] != "ALERT") & truth_close).sum())
    false_alarm = int(((fleet["status"] == "ALERT") & ~truth_close).sum())
    st.caption(
        f"Check against ground truth at an alert threshold of {alert_thr}: "
        f"**{caught} real failures caught, {missed} missed, {false_alarm} false alarms.**"
    )
    show = fleet.rename(columns={"unit": "Engine", "cycle": "Cycles run", "rul_pred": "Predicted RUL",
                                 "rul_std": "± disagreement", "true_rul": "True RUL (NASA)", "status": "Status"})
    show["Status"] = show["Status"].map(STATUS_ICON)
    st.dataframe(show.round(1), hide_index=True, use_container_width=True, height=300)

    with st.expander("How this works / limitations"):
        st.markdown(
            f"""
- **Model**: 2-layer LSTM over the last {pred.window} cycles of 14 sensors, ensemble of {len(pred.models)} seeds.
- **Target**: RUL clipped at {C.RUL_CAP} cycles, because early-life engines are indistinguishable from each other.
- **Alert logic**: ALERT when predicted RUL <= threshold. Lowering it means fewer false alarms but more missed failures.
- **Limitation**: FD001 is simulated data with one operating condition and one fault mode (HPC degradation).
  Real fleets need per-engine calibration, multiple fault modes and handling of missing sensors.
"""
        )


main()
