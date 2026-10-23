import numpy as np
from experiment import run_experiment


if __name__ == "__main__":
    import os

    models_to_run = {
            "MF_RFF_Bayes_l": {"type": "MF_RFF_Bayes_l", "n_restarts": 5, "n_samples": 250},
            "MF_RFF_Bayes_nl": {"type": "MF_RFF_Bayes_nl", "n_restarts": 5, "n_samples": 250},
            "GP": {"n_restarts": 5},
            "KO": {"n_restarts": 5},
            "NARGP": {"n_restarts": 5},
            "rna": {},
            "cokm": {}   
            }
 
    # ── Park ─────────────────────────────────────────────────────────────────
    os.makedirs("plots/borehole",   exist_ok=True)
    run_experiment(
        d=8,
        N1=80, N2=40,
        n_replicates=100, base_seed=0,
        data_dir="Data/borehole",
        plot_dir="plots/borehole",
        criterion='nlml',
        tuning_method='optim',
        model_names=models_to_run,
        kernel='gaussian')
