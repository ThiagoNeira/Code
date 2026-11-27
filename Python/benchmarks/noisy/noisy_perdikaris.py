# noisy_perdikaris.py
"""
Perdikaris with noisy outputs on non-nested designs: NARGP, KO, One-shot GP and MF_RFF_bayes
(linear and non-linear), 100 replicates. Run from Python/:   python -m benchmarks.noisy.noisy_perdikaris

Data: Data/perdikaris_noisy/50_14, written by noisy_data.py on the first run and reused after.
Each level gets its own random LHS (not nested); y_l = f_l(x) + sigma_l * eps, with
sigma_l = NOISE_FRAC * sd(f_l).
Scoring (experiment.py): RMSE and R2 against the noise-free test outputs; CRPS and coverage against
noisy test observations. Plots: plots/perdikaris_noisy/.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # Python/, so a plain `python <file>` also works
from harness.experiment import run_experiment
from harness.noisy_data import make_noisy_data
from harness.test_functions import perdikaris_f1 as f1, perdikaris_f2 as f2
from paths import DATA

NOISE_FRAC = 0.05           # sigma_l / sd(f_l), both levels
N_LIST = [50, 14]
N_REP = 100
DATA_SEED = 0

# MF_RFF_bayes settings from perdikaris.py, except nugget_bounds: lambda is a noise-to-signal ratio
# (about NOISE_FRAC^2 = 2.5e-3 here), above the noise-free upper bound 1e-3.
BAYES_COMMON = dict(
    nuggets=1e-6,
    n_restarts=10,
    n_samples=250,
    tune_nugget=True,
    nugget_bounds=(-8, 0),
    freq_method='sobol',
    n_screen=5,
    local_frac=1.0,
    feature_mults=[10, 20],
    verbose=False,
)

# The GPy baselines see standardized outputs (one mean and sd pooled over the levels).
models_to_run = {
    "NARGP": {"n_restarts": 5, "standardize": True},
    "KO": {"n_restarts": 5, "standardize": True},
    "One-shot GP": {"type": "GP", "n_restarts": 5, "standardize": True},
    "MF_RFF_Bayes_l": {
        "type": "MF_RFF_Bayes_l",
        **BAYES_COMMON,
        "eta_bounds": [(2, 3.7), (2, 3.7)],
        "nugget_bounds": [(-8, 0), (-8, 0)],
    },
    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "eta_bounds": [(0.5, 2.5), (-3, 0.5)],
        "eta_bounds_f": [(-3, 0.5)],
        "nugget_bounds": [(-8, 0), (-8, 0)],
    },
}

if __name__ == "__main__":
    make_noisy_data(os.path.join(DATA, "perdikaris_noisy", "_".join(map(str, N_LIST))),
                    [f1, f2], d=1, N_list=N_LIST, n_rep=N_REP, noise_frac=NOISE_FRAC, seed=DATA_SEED)
    run_experiment(
        d=1,
        N_list=N_LIST,
        n_replicates=N_REP,
        base_seed=0,
        data_dir="Data/perdikaris_noisy",
        plot_dir="plots/perdikaris_noisy",
        model_names=models_to_run,
        kernel='gaussian',
    )
