# hartmann.py (adapted from branin.py / currin.py)
# Hartmann-3D and Hartmann-6D on [0, 1]^d, each with 3 fidelities (Kandasamy et al., 2016):
#   f_m(x) = - sum_i alpha_i^(m) exp(- sum_j A_ij (x_j - P_ij)^2),
#   alpha^(m) = alpha + (3 - m) * delta,  delta = (0.01, -0.01, -0.1, 0.1),  m = 1 (lowest) .. 3 (high)
# Data: Data/hartmann3/60_30_15 and Data/hartmann6/120_60_30 (RNAmf + co-kriging precomputed).
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
    nuggets=1e-6,              # fixed nugget (broadcasts to all levels; used as initial value if tune_nugget=True)
    n_restarts=10,             # experiment default is 5; the class default is 10
    n_samples=250,             # posterior draws (class default 500)
    tune_nugget=True,          # tune the nuggets by marginal likelihood
    nugget_bounds=(-8, 0),     # log10 box, fallback for per-level nugget bounds
    freq_method='sobol',       # 'sobol' | 'mc' | 'orthogonal'
    n_screen=5,                # screening pool = n_screen * n_restarts LHS points
    local_frac=1.0,            # fraction of the n_restarts best screened starts that get a local search
    feature_mults=[6, 10, 10], # m_l = _resolve_m(N_l, feature_mults[l]) for 3 levels
    verbose=True,
)

# Independent search boxes per level (log10 units, k(r) = exp(-sum_j eta_j r_j^2) on [0, 1]^d).
#   eta_bounds    : list of L (lo, hi) pairs (same box for every x-dim)
#   eta_bounds_f  : list of L-1 (lo, hi) pairs for the f_{l-1} coordinate (non-linear model only)
#   nugget_bounds : list of L (lo, hi) pairs per level
#
# Notes from wide-box ML fits (eta in (-5, 5), lambda in (-10, 0)) on replicates 1-3:
# * Outputs are O(1) (sd ~0.95 for 3D, ~0.39 for 6D) and the fidelities are correlated at
#   r > 0.997, so the discrepancies are small. The tuned prior precision lambda is
#   10^-1 to 10^-3.5 on all levels: the Branin/Currin box (-8, -3) would clip it, so the
#   upper nugget bound is raised to 0.
# * Some dimensions are irrelevant at some levels (eta -> lower bound; e.g. x3 at level 0
#   of Hartmann-6). Hitting the lower bound only means a flat direction, so the lower bounds
#   stay wide and the boundary warnings for those dims are benign.
# * Upper bounds on levels >= 1 are capped (10^2 in 3D, 10^1.5 in 6D, length-scale >= 0.1-0.2):
#   without the cap, the 6D linear level 1 ran away to eta ~ 10^5 in some dims (features
#   become white noise absorbing the small discrepancy).
# * Non-linear levels: the f_{l-1} coordinate carries almost all the signal
#   (eta_f ~ 10^-1.3 to 10^-2.8 on the standardized f), and most x-dims go flat.
models_hartmann3 = {

    "MF_RFF_Bayes_l": {
        "type": "MF_RFF_Bayes_l",
        **BAYES_COMMON,
        "eta_bounds": [(-2, 2.5), (-3, 2), (-3, 2)],
        "nugget_bounds": [(-8, -1), (-7, 0), (-7, 0)],
    },

    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "eta_bounds": [(-2, 2.5), (-5, 1.5), (-5, 1.5)],
        "eta_bounds_f": [(-4, 0), (-4, 0)],
        "nugget_bounds": [(-8, -1), (-8, 0), (-8, 0)],
    },

    "rna": {},
    "cokm": {},
}

models_hartmann6 = {

    "MF_RFF_Bayes_l": {
        "type": "MF_RFF_Bayes_l",
        **BAYES_COMMON,
        "eta_bounds": [(-4, 2), (-4, 1.5), (-4, 1.5)],
        "nugget_bounds": [(-8, 0), (-7, 0), (-7, 0)],
    },

    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "eta_bounds": [(-4, 2), (-5, 1), (-5, 1)],
        "eta_bounds_f": [(-4, 0), (-4, 0)],
        "nugget_bounds": [(-8, 0), (-8, -1), (-8, -1)],
    },

    "rna": {},
    "cokm": {},
}

if __name__ == "__main__":
    # ── Hartmann 3D, 3 fidelities ────────────────────────────────────────────
    os.makedirs("plots/hartmann3", exist_ok=True)
    run_experiment(
        d=3,
        N_list=[60, 30, 15],
        n_replicates=100, base_seed=0,
        data_dir="Data/hartmann3",
        plot_dir="plots/hartmann3",
        model_names=models_hartmann3,
        kernel='gaussian'
    )

    # ── Hartmann 6D, 3 fidelities ────────────────────────────────────────────
    os.makedirs("plots/hartmann6", exist_ok=True)
    run_experiment(
        d=6,
        N_list=[120, 60, 30],
        n_replicates=100, base_seed=0,
        data_dir="Data/hartmann6",
        plot_dir="plots/hartmann6",
        model_names=models_hartmann6,
        kernel='gaussian'
    )
