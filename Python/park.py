import numpy as np
from experiment import run_experiment

if __name__ == "__main__":
    import os

    models_to_run = {
            "MF_RFF_Bayes_l": {"type": "MF_RFF_Bayes_l", "n_restarts": 10, "n_samples": 250},
            "MF_RFF_Bayes_nl": {"type": "MF_RFF_Bayes_nl", "n_restarts": 10, "n_samples": 250},
            "rna": {},
            "cokm": {},
            "NARGP": {"n_restarts": 5}
            }
 
    # ── Park ─────────────────────────────────────────────────────────────────
    os.makedirs("plots/park",   exist_ok=True)
    run_experiment(
        d=4, 
        N1=40, N2=20,
        n_replicates=100, base_seed=0,
        data_dir="Data/park",
        plot_dir="plots/park",
        criterion='nlml',
        model_names=models_to_run,
        kernel='gaussian'
    )       
