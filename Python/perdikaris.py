from experiment import run_experiment

if __name__ == "__main__":
    import os

    os.makedirs("plots/perdikaris", exist_ok=True)

    
    # Specify models and their individual tuning parameters cleanly here

    """
        model_names = {
            "KO": {"n_restarts": 5},
            "NARGP": {"n_restarts": 5},
            "MF_RFF_linear": {"n_restarts": 5, "n_samples": 250, "number_etas_grid": 100},
            "MF_RFF_nonl": {"n_restarts": 5, "n_samples": 500, "number_etas_grid": 100},
            "MF_RFF_Bayes_l": {"n_restarts": 5, "number_etas_grid": 100},
            "MF_RFF_Bayes_nl": {"n_restarts": 5, "number_etas_grid": 100},
            "GP": {"n_restarts": 5},
            "rna": {},
            "cokm": {}
        }
    """

    models_to_run = {
            "MF_RFF_Bayes_l": {"type": "MF_RFF_Bayes_l", "n_restarts": 5, "n_samples": 250},
            "MF_RFF_Bayes_nl": {"type": "MF_RFF_Bayes_nl", "n_restarts": 5, "n_samples": 250},
            "GP": {"n_restarts": 5},
            "NARGP": {"n_restarts": 5},
            "rna": {},
            "cokm": {}   
            }
    
    run_experiment(
        d=1,
        N1=13, N2=8,
        n_replicates=100, 
        base_seed=0,
        data_dir="Data/perdikaris",
        plot_dir="plots/perdikaris",
        criterion='nlml',
        tuning_method='optim',
        model_names=models_to_run,
        kernel='gaussian'
    )