# paths.py
"""
Project paths, resolved from this file's location (the Python/ folder), never from the CWD.
Every module that reads Data/ or writes plots/ goes through here, so scripts behave the same
whichever folder they are launched from.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent      # .../Code/Python
DATA = ROOT / "Data"
PLOTS = ROOT / "plots"


def under_root(p):
    """Absolute paths are returned as is; relative ones (e.g. "plots/park") are taken from ROOT."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p
