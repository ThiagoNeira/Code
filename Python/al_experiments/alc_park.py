# alc_park.py
"""
Park: MF_RFF_bayes with ALC (Cohn) vs ALM (MacKay) active learning.
Run from Python/:   python -m al_experiments.alc_park

Both criteria start from the initial designs in Data/park/alm_40_20 (written by ../R/ALM Park.R)
and use its test set, costs, budget and search settings. Oracles and model settings are those of
alm_park.py. ALC averages over a fixed Latin hypercube of 100 * d points.
Saves plots/park/alc/alm_comparison.png and alm_results.npz.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # Python/, so a plain `python <file>` also works
from harness.alm_experiment import run_alm_experiment
from al_experiments.alm_park import park91a, park91alc, models_to_run

if __name__ == "__main__":
    run_alm_experiment(
        d=4,
        N_list=[40, 20],
        oracles=[park91alc, park91a],
        model_names=models_to_run,
        data_dir="Data/park",
        plot_dir="plots/park",
        kernel='gaussian',
        criteria=('alm', 'alc'),
        include_rna=False,
        out_name="alc",
    )
