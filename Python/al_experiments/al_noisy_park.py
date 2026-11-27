# al_noisy_park.py
"""
Park with noisy outputs on non-nested designs: MF_RFF_bayes (non-linear) with ALM (MacKay) vs ALC
(Cohn) active learning, 100 replicates. Run from Python/:   python -m al_experiments.al_noisy_park

Initial designs: Data/park_noisy/alm_40_20, written by noisy_data.py on the first run (an
independent random LHS per level, noise as in noisy_park.py). Costs, budget and search settings
are those of the noise-free ALM runs (Data/park/alm_40_20/alm_settings.npy).
Acquisition is single-level (x is evaluated at the chosen level only, at cost c_l), and noisy
oracles may re-evaluate a point already in the design.
Saves plots/park_noisy/al/alm_comparison.png and alm_results.npz.
"""
import os
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # Python/, so a plain `python <file>` also works
from harness.alm_experiment import run_alm_experiment
from harness.noisy_data import make_noisy_data
from paths import DATA
from benchmarks.noisy.noisy_park import park91a, park91alc, NOISE_FRAC, DATA_SEED, models_to_run

N_INIT = [40, 20]
N_REP = 100
AL_SETTINGS = [1, 3, 30, 256, 3, 0.05]      # c_1, c_2, budget, n_screen, n_starts, min_sep

if __name__ == "__main__":
    make_noisy_data(os.path.join(DATA, "park_noisy", "alm_" + "_".join(map(str, N_INIT))),
                    [park91alc, park91a], d=4, N_list=N_INIT, n_rep=N_REP, noise_frac=NOISE_FRAC,
                    seed=DATA_SEED + 1, al_settings=AL_SETTINGS)
    run_alm_experiment(
        d=4,
        N_list=N_INIT,
        oracles=[park91alc, park91a],
        model_names={"MF_RFF_Bayes_nl": models_to_run["MF_RFF_Bayes_nl"]},
        data_dir="Data/park_noisy",
        plot_dir="plots/park_noisy",
        kernel='gaussian',
        criteria=('alm', 'alc'),
        include_rna=False,
        out_name="al",
        nested=False,
        allow_repeats=True,
    )
