# perdikaris.py
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # Python/, so a plain `python <file>` also works
from harness.experiment import run_experiment

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
    nugget_bounds=(-7, -1),  # log10 box, fallback for per-level nugget bounds
    freq_method='sobol',     # 'sobol' | 'mc' | 'orthogonal'
    n_screen=5,              # screening pool = n_screen * n_restarts LHS points
    local_frac=1.0,          # fraction of the n_restarts best screened starts that get a local search
    feature_mults=[10, 20],   # m_l = _resolve_m(feature_mults[l] * N_l)
    verbose=True,
)

# Independent search boxes per level (log10 units). None -> class defaults.
#   eta_bounds    : list of L (lo, hi) pairs for spatial bandwidths
#   eta_bounds_f  : list of L-1 (lo, hi) pairs for the f_{l-1} coordinate (non-linear model only)
#   nugget_bounds : list of L (lo, hi) pairs per level (None -> nugget_bounds)
models_to_run = {

    
    "MF_RFF_Bayes_l": {
        "type": "MF_RFF_Bayes_l",
        **BAYES_COMMON,
        "eta_bounds": [(2, 3.7), (2, 3.7)],
        "nugget_bounds": [(-8, -3), (-8, -3)],
    },

    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "eta_bounds": [(0.5, 2.5), (-3, 0.5)],
        "eta_bounds_f": [(-3, 0.5)],
        "nugget_bounds": [(-8, -3), (-8, -3)],
    },

    "rna": {},
}

if __name__ == "__main__":

    run_experiment(
        d=1,
        N_list=[50, 14],
        n_replicates=100, 
        base_seed=0,
        data_dir="Data/perdikaris",
        plot_dir="plots/perdikaris",
        model_names=models_to_run,
        kernel='gaussian'
    )