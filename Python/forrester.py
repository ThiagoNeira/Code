# forrester.py (adapted from branin.py / currin.py)
# Two configurations of the 1D Forrester function on [0, 1]:
#   2 fidelities  (Forrester et al., 2007):        Data/forrester/12_6
#   4 fidelities  (Mainini et al., 2022; f4 -> f1): Data/forrester/20_15_10_6
# RNAmf supports at most 3 levels, so the 4-fidelity run compares against
# recursive co-kriging ("cokm") only; no rna_* files exist for that design.
import os
import numpy as np
from experiment import run_experiment

# ----------------------------------------------------------------------------
# MF_RFF_bayes parameters. Every key below is forwarded to MF_RFF_bayes(...), and every one can be
# overridden per model. Values shown are the class defaults except where noted.
# Not listed on purpose:
#   d      -> set by run_experiment
#   seed   -> defaults to the replicate seed (set it here only to freeze the RFF draws across replicates)
#   kernel -> defaults to run_experiment(kernel=...)  ('gaussian' | 'matern32' | 'matern52')
#   linear -> inferred from the model name (..._l -> True, ..._nl -> False); set explicitly to override
# ----------------------------------------------------------------------------
BAYES_COMMON = dict(
    nuggets=1e-6,            # fixed nugget (broadcasts to all levels; used as initial value if tune_nugget=True)
    n_restarts=10,           # experiment default is 5; the class default is 10
    n_samples=250,           # posterior draws (class default 500)
    tune_nugget=True,        # tune the nuggets by marginal likelihood
    nugget_bounds=(-8, -2),  # log10 box, fallback for per-level nugget bounds
    freq_method='sobol',     # 'sobol' | 'mc' | 'orthogonal'
    n_screen=5,              # screening pool = n_screen * n_restarts LHS points
    local_frac=1.0,          # fraction of the n_restarts best screened starts that get a local search
    verbose=True,
)

# Independent search boxes per level (log10 units, k(r) = exp(-eta r^2) on [0, 1]).
#   eta_bounds    : list of L (lo, hi) pairs for the spatial bandwidth
#   eta_bounds_f  : list of L-1 (lo, hi) pairs for the f_{l-1} coordinate (non-linear model only)
#   nugget_bounds : list of L (lo, hi) pairs per level
#
# Level 0 (the cheap fidelity) oscillates like sin(12x): ML optima eta ~ 10^1.1-10^1.5
# (length-scale ~0.15), so (0, 2.5) brackets it.
# 2 fidelities: y_high = 2 y_low - 20(x - 0.5) - 10 exactly, so the linear discrepancy is
# a straight line (eta -> 10^-3, lambda -> lower bound); the box allows near-flat kernels.
models_2fid = {

    "MF_RFF_Bayes_l": {
        "type": "MF_RFF_Bayes_l",
        **BAYES_COMMON,
        "feature_mults": [6, 10],
        "eta_bounds": [(0, 2.5), (-4, 1)],
        "nugget_bounds": [(-8, -2), (-8, -2)],
    },

    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "feature_mults": [6, 10],
        "eta_bounds": [(0, 2.5), (-4, 1)],
        "eta_bounds_f": [(-5, 0)],
        "nugget_bounds": [(-8, -2), (-8, -2)],
    },

    "rna": {},
    "cokm": {},
}

# 4 fidelities (levels f4, f3, f2, f1):
#   f4 -> f3 : exactly linear (f3 = 1.5 f4 - 10(x - 0.5) + 5.5) -> smooth discrepancy box.
#   f3 -> f2 and f2 -> f1 : f2 = (5.5x - 2.5)^2 sin(12x - 4) is a distorted fidelity, so the
#   discrepancies oscillate at the scale of f itself (eta ~ 10^1.2). The top level has only
#   ~6 points (spacing ~0.17); its upper bound is capped at 10^2 (length-scale >= 0.07)
#   because wide-box fits ran away to eta ~ 10^3.5 (interpolating spikes).
models_4fid = {

    "MF_RFF_Bayes_l": {
        "type": "MF_RFF_Bayes_l",
        **BAYES_COMMON,
        "feature_mults": [6, 10, 10, 10],
        "eta_bounds": [(0, 2.5), (-4, 1), (-1, 2.5), (-1, 2)],
        "nugget_bounds": [(-8, -2), (-8, -2), (-8, -1), (-8, -1)],
    },

    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "feature_mults": [6, 10, 10, 10],
        "eta_bounds": [(0, 2.5), (-4, 1), (-4, 2), (-4, 2)],
        "eta_bounds_f": [(-5, 0), (-5, 0.5), (-5, 0.5)],
        "nugget_bounds": [(-8, -2), (-8, -2), (-8, -1), (-8, -1)],
    },

    # RNAmf is limited to 3 fidelity levels: recursive co-kriging only.
    "cokm": {},
}

if __name__ == "__main__":
    # ── Forrester, 2 fidelities ──────────────────────────────────────────────
    os.makedirs("plots/forrester_2fid", exist_ok=True)
    run_experiment(
        d=1,
        N_list=[12, 6],
        n_replicates=100, base_seed=0,
        data_dir="Data/forrester",
        plot_dir="plots/forrester_2fid",
        model_names=models_2fid,
        kernel='gaussian'
    )

    # ── Forrester, 4 fidelities ──────────────────────────────────────────────
    os.makedirs("plots/forrester_4fid", exist_ok=True)
    run_experiment(
        d=1,
        N_list=[20, 15, 10, 6],
        n_replicates=100, base_seed=0,
        data_dir="Data/forrester",
        plot_dir="plots/forrester_4fid",
        model_names=models_4fid,
        kernel='gaussian'
    )
