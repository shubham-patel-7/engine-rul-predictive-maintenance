"""Central configuration for the C-MAPSS RUL project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "CMAPSS"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SUBSET = "FD001"  # 1 operating condition, 1 fault mode (HPC degradation)
RUL_CAP = 125  # piecewise-linear RUL target: early life is "healthy", not countable
WINDOW = 30  # cycles of history the models see
ALERT_HORIZON = 30  # "will this engine fail within N cycles?"

COLUMNS = ["unit", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

# FD001 has 7 channels that are constant (setting3, s1, s5, s10, s16, s18, s19)
# plus s6 which is near-constant. The 14 below are the informative sensors.
SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
FEATURES = [f"s{i}" for i in SENSORS]

SENSOR_NAMES = {
    "s2": "T24 - LPC outlet temp",
    "s3": "T30 - HPC outlet temp",
    "s4": "T50 - LPT outlet temp",
    "s7": "P30 - HPC outlet pressure",
    "s8": "Nf - fan speed",
    "s9": "Nc - core speed",
    "s11": "Ps30 - HPC static pressure",
    "s12": "phi - fuel flow / Ps30",
    "s13": "NRf - corrected fan speed",
    "s14": "NRc - corrected core speed",
    "s15": "BPR - bypass ratio",
    "s17": "htBleed - bleed enthalpy",
    "s20": "W31 - HPT coolant bleed",
    "s21": "W32 - LPT coolant bleed",
}

# Reproducibility
VAL_UNITS = 20  # engines held out of training for early stopping / demos
SPLIT_SEED = 42
