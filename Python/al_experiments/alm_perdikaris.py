# alm_perdikaris.py
"""
ALM comparison on the Perdikaris function: MF_RFF_bayes vs RNAmf.
Run ../R/ALM Perdikaris.R first (it writes Data/perdikaris/alm_13_8), then, from Python/:
    python -m al_experiments.alm_perdikaris
Costs, budget and the acquisition search are read from the R output (alm_settings.npy).
Saves plots/perdikaris/alm/alm_comparison.png and alm_results.npz.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # Python/, so a plain `python <file>` also works
from harness.alm_experiment import run_alm_experiment

# Same functions as R/ALM Perdikaris.R (level 0 = low fidelity f1, level 1 = high fidelity f2)
from harness.test_functions import perdikaris_f1 as f1, perdikaris_f2 as f2

# MF_RFF_bayes settings from perdikaris.py (non-linear model); every key goes to MF_RFF_bayes(...).
BAYES_COMMON = dict(
    nuggets=1e-6,
    n_restarts=10,
    n_samples=250,
    tune_nugget=True,
    nugget_bounds=(-7, -1),
    freq_method='sobol',
    n_screen=5,              # hyperparameter screening (not the ALM screening pool)
    local_frac=1.0,
    feature_mults=[10, 20],
    verbose=False,
)

models_to_run = {
    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "eta_bounds": [(0.5, 2.5), (-3, 0.5)],
        "eta_bounds_f": [(-3, 0.5)],
        "nugget_bounds": [(-8, -3), (-8, -3)],
    },
}

if __name__ == "__main__":
    run_alm_experiment(
        d=1,
        N_list=[13, 8],
        oracles=[f1, f2],
        model_names=models_to_run,
        data_dir="Data/perdikaris",
        plot_dir="plots/perdikaris",
        kernel='gaussian',
    )
