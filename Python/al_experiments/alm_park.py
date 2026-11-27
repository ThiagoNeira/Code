# alm_park.py
"""
ALM comparison on the Park function: MF_RFF_bayes vs RNAmf.
Run ../R/ALM Park.R first (it writes Data/park/alm_40_20), then, from Python/:
    python -m al_experiments.alm_park
Costs, budget and the acquisition search are read from the R output (alm_settings.npy).
Saves plots/park/alm/alm_comparison.png and alm_results.npz.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # Python/, so a plain `python <file>` also works
from harness.alm_experiment import run_alm_experiment
from harness.test_functions import park91a, park91alc   # same as R/ALM Park.R (level 0, level 1)


# MF_RFF_bayes settings from park.py (non-linear model); every key goes to MF_RFF_bayes(...).
BAYES_COMMON = dict(
    nuggets=1e-6,
    n_restarts=10,
    n_samples=250,
    tune_nugget=True,
    nugget_bounds=(-7, -1),
    freq_method='sobol',
    n_screen=5,              # hyperparameter screening (not the ALM screening pool)
    local_frac=1.0,
    feature_mults=[6, 10],
    verbose=False,
)

models_to_run = {
    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "eta_bounds": [(-3, 0), (-4, 0)],
        "eta_bounds_f": [(-5, -1)],
        "nugget_bounds": [(-8, -3), (-8, -3)],
    },
}

if __name__ == "__main__":
    run_alm_experiment(
        d=4,
        N_list=[40, 20],
        oracles=[park91alc, park91a],
        model_names=models_to_run,
        data_dir="Data/park",
        plot_dir="plots/park",
        kernel='gaussian',
    )
