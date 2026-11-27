# alc_perdikaris.py
"""
Perdikaris: MF_RFF_bayes with ALC (Cohn) vs ALM (MacKay) active learning.
Run from Python/:   python -m al_experiments.alc_perdikaris

Both criteria start from the initial designs in Data/perdikaris/alm_13_8 (written by
../R/ALM Perdikaris.R) and use its test set, costs, budget and search settings. Oracles and model
settings are those of alm_perdikaris.py. ALC averages over a fixed Latin hypercube of 100 * d points.
Saves plots/perdikaris/alc/alm_comparison.png and alm_results.npz.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # Python/, so a plain `python <file>` also works
from harness.alm_experiment import run_alm_experiment
from al_experiments.alm_perdikaris import f1, f2, models_to_run

if __name__ == "__main__":
    run_alm_experiment(
        d=1,
        N_list=[13, 8],
        oracles=[f1, f2],
        model_names=models_to_run,
        data_dir="Data/perdikaris",
        plot_dir="plots/perdikaris",
        kernel='gaussian',
        criteria=('alm', 'alc'),
        include_rna=False,
        out_name="alc",
    )
