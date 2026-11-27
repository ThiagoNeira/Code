# noisy_park.py
"""
Park with noisy outputs on non-nested designs: NARGP, KO, One-shot GP and MF_RFF_bayes
(linear and non-linear), 100 replicates. Run from Python/:   python -m benchmarks.noisy.noisy_park

Data: Data/park_noisy/40_20, written by noisy_data.py on the first run and reused after.
Each level gets its own random LHS (not nested); y_l = f_l(x) + sigma_l * eps, with
sigma_l = NOISE_FRAC * sd(f_l).
Scoring (experiment.py): RMSE and R2 against the noise-free test outputs; CRPS and coverage against
noisy test observations. Plots: plots/park_noisy/.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # Python/, so a plain `python <file>` also works
from harness.experiment import run_experiment
from harness.noisy_data import make_noisy_data
from harness.test_functions import park91a, park91alc
from paths import DATA

NOISE_FRAC = 0.05           # sigma_l / sd(f_l), both levels
N_LIST = [40, 20]
N_REP = 100
DATA_SEED = 0

# MF_RFF_bayes settings from park.py, except nugget_bounds: lambda is a noise-to-signal ratio
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
    feature_mults=[6, 10],
    verbose=False,
)

# The GPy baselines see standardized outputs (one mean and sd pooled over the levels): Park's
# outputs (mean about 10, variance about 22) exceed KO's and GP's kernel-variance bound of 10.
models_to_run = {
    "NARGP": {"n_restarts": 5, "standardize": True},
    "KO": {"n_restarts": 5, "standardize": True},
    "One-shot GP": {"type": "GP", "n_restarts": 5, "standardize": True},
    "MF_RFF_Bayes_l": {
        "type": "MF_RFF_Bayes_l",
        **BAYES_COMMON,
        "eta_bounds": [(-3, 1), (-3, 1)],
        "nugget_bounds": [(-8, 0), (-8, 0)],
    },
    "MF_RFF_Bayes_nl": {
        "type": "MF_RFF_Bayes_nl",
        **BAYES_COMMON,
        "eta_bounds": [(-3, 0), (-4, 0)],
        "eta_bounds_f": [(-5, -1)],
        "nugget_bounds": [(-8, 0), (-8, 0)],
    },
}

if __name__ == "__main__":
    make_noisy_data(os.path.join(DATA, "park_noisy", "_".join(map(str, N_LIST))),
                    [park91alc, park91a], d=4, N_list=N_LIST, n_rep=N_REP, noise_frac=NOISE_FRAC,
                    seed=DATA_SEED)
    run_experiment(
        d=4,
        N_list=N_LIST,
        n_replicates=N_REP,
        base_seed=0,
        data_dir="Data/park_noisy",
        plot_dir="plots/park_noisy",
        model_names=models_to_run,
        kernel='gaussian',
    )
