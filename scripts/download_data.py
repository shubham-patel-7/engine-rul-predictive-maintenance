"""Download C-MAPSS FD001 into data/CMAPSS (the repo already ships FD001; use this to restore or fetch others).

Official source: NASA Prognostics Center of Excellence Data Set Repository
  https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
This script uses a public GitHub mirror of the same files. Usage:
  python scripts/download_data.py            # FD001
  python scripts/download_data.py FD002 FD003 FD004
"""
import sys
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData",
    "https://raw.githubusercontent.com/schwxd/LSTM-Keras-CMAPSS/master/C-MAPSS-Data",
]
OUT = Path(__file__).resolve().parents[1] / "data" / "CMAPSS"


def fetch(name: str) -> None:
    dest = OUT / name
    if dest.exists():
        print(f"exists   {name}")
        return
    for base in MIRRORS:
        try:
            with urllib.request.urlopen(f"{base}/{name}", timeout=60) as r:
                dest.write_bytes(r.read())
            print(f"fetched  {name}  <- {base}")
            return
        except Exception as e:  # try next mirror
            print(f"  {base} failed: {e}")
    sys.exit(f"Could not download {name}. Get it from the NASA link in this file's docstring.")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in sys.argv[1:] or ["FD001"]:
        for kind in ("train", "test", "RUL"):
            fetch(f"{kind}_{sub}.txt")
