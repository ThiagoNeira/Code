import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from utils import timer, crps_ensemble, crps_gaussian
from MF_RFF_Bayes import MF_RFF_bayes
from MF_RFF import MF_RFF
from NARGP import NARGP
from KO_model import KO
from One_shot_GP import GP

from plotting import plot_boxplot, plot_slices_high_1d

# ============================================================================
# Single run
# ============================================================================
def single_run(N1, N2, d, seed, 
               kernel,
               data_dir, 
               criterion='nlpd', 
               tuning_method='grid',
               model_names=None):

    # 1. Load Data
    X1 = np.load(f'{data_dir}/{N1}_{N2}/X1_{N1}_rep{seed}.npy')
    z  = np.load(f'{data_dir}/{N1}_{N2}/y1_{N1}_rep{seed}.npy').reshape(-1, 1)
    X2 = np.load(f'{data_dir}/{N1}_{N2}/X2_{N2}_rep{seed}.npy')
    y  = np.load(f'{data_dir}/{N1}_{N2}/y2_{N2}_rep{seed}.npy').reshape(-1, 1)

    X_test   = np.load(f'{data_dir}/{N1}_{N2}/x_test.npy')
    y_test   = np.load(f'{data_dir}/{N1}_{N2}/yhigh_test.npy')

    # 2. Setup Models via Dictionary Configuration (supporting unique aliases)
    if model_names is None:
        model_names = {
            "KO": {"n_restarts": 5},
            "NARGP": {"n_restarts": 5},
            "MF_RFF_linear": {"n_starts": 5, "number_etas_grid": 100},
            "MF_RFF_nonl": {"n_starts": 5, "number_etas_grid": 100},
            "MF_RFF_Bayes_l": {"n_starts": 5, "number_etas_grid": 100},
            "MF_RFF_Bayes_nl": {"n_starts": 5, "number_etas_grid": 100},
            "GP": {"n_restarts": 5},
            "rna": {},
            "cokm": {}
        }
    elif isinstance(model_names, list):
        model_names = {name: {} for name in model_names}

    models = {}
    model_types = {}

    for name, params in model_names.items():
        # Determine the underlying model type from 'type' param or key name prefix
        model_type = params.get("type", None)
        if model_type is None:
            for known in ["KO", "NARGP", "GP", "MF_RFF_linear", "MF_RFF_nonl", "MF_RFF_Bayes_l", "MF_RFF_Bayes_nl", "rna", "cokm"]:
                if name == known or name.startswith(known + "_") or name.startswith(known + "-"):
                    model_type = known
                    break
            if model_type is None:
                model_type = name

        model_types[name] = model_type

        if model_type == "KO":
            models[name] = KO(n_restarts=params.get("n_restarts", 5), kernel=kernel)
        elif model_type == "NARGP":
            models[name] = NARGP(n_restarts=params.get("n_restarts", 5), kernel=kernel)
        elif model_type == "GP":
            models[name] = GP(n_restarts=params.get("n_restarts", 5), kernel=kernel)
        elif any(k in model_type for k in ["MF_RFF", "MF_RFF_Bayes"]):
            # Determine linearity (default to True if 'l' or 'linear' is in the type/name)
            is_linear = params.get("linear", True if ("l" in model_type.lower() or "linear" in model_type.lower()) and "nl" not in model_type.lower() else False)
            
            rff_kwargs = {
                'd': d, 'nugget1': 1e-6, 'nugget2': 1e-4, 
                'number_etas_grid': params.get("number_etas_grid", 100), 
                'criterion': criterion, 'seed': seed,
                'kernel': kernel, 'tuning_method': tuning_method,
                'n_restarts': params.get("n_restarts", 5),
                'linear': is_linear
            }
            # Pass n_samples if provided
            if "n_samples" in params:
                rff_kwargs['n_samples'] = params["n_samples"]

            if "Bayes" in model_type:
                models[name] = MF_RFF_bayes(**rff_kwargs)
            else:
                models[name] = MF_RFF(**rff_kwargs)
        elif model_type in ["rna", "cokm"]:
            models[name] = "precomputed"

    # 3. Fit and Evaluate 
    timing = {}
    metrics = {}
    coverages = {}

    for name, model in models.items():
        metrics[name] = {}
        m_type = model_types[name]
        
        if m_type in ["rna", "cokm"]:
            time_elapsed = float(np.load(f'{data_dir}/{N1}_{N2}/{m_type}_time_rep{seed}.npy'))
            print(f"[TIMER] {name} ({m_type}): {time_elapsed:.3f}s")
            timing[name] = time_elapsed
            y_pred = np.load(f'{data_dir}/{N1}_{N2}/{m_type}_mean_rep{seed}.npy').reshape(-1,1)
            y_var = np.load(f'{data_dir}/{N1}_{N2}/{m_type}_sigma2_rep{seed}.npy').reshape(-1,1)
            y_lower = np.load(f'{data_dir}/{N1}_{N2}/{m_type}_lower_rep{seed}.npy').reshape(-1,1)
            y_upper = np.load(f'{data_dir}/{N1}_{N2}/{m_type}_upper_rep{seed}.npy').reshape(-1,1)

            metrics[name]["crps"] = crps_gaussian(y_pred, y_var, y_test)
            
        else:
            with timer(name, timing):
                if m_type == "GP":
                    model.fit(X2, y)
                else:
                    model.fit(X1, z, X2, y)

                if m_type in ["MF_RFF_Bayes_l", "MF_RFF_Bayes_nl"]:
                    y_pred, _, y_lower, y_upper, y_samples = model.predict(X_test, return_ci=True, return_samples=True)

                    metrics[name]["crps"] = crps_ensemble(y_samples, y_test)

                else:
                    y_pred, y_var, y_lower, y_upper = model.predict(X_test, return_ci=True)
                    metrics[name]["crps"] = crps_gaussian(y_pred, y_var, y_test)

        coverages[name] = ((y_test >= y_lower.ravel()) & (y_test <= y_upper.ravel())).mean()
        metrics[name]["rmse_high"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics[name]["r2"] = r2_score(y_test, y_pred)
        
    return metrics, timing, coverages

# ============================================================================
# Replicates (Fully Dynamic)
# ============================================================================
def run_replicates(d, N1, N2, n_replicates, base_seed,
                   data_dir,
                   criterion='nlpd', 
                   tuning_method='grid',
                   model_names=None,
                   kernel='gaussian'): 

    errors = {"high": {}}
    r2_scores = {}
    crps = {}
    timings = {}
    coverages = {}

    for r in range(1, n_replicates+1):
        print(f"\nReplicate {r} of {n_replicates}")
        seed = base_seed + r

        metrics, timing, coverage = single_run(
            N1, N2, d, seed, 
            kernel,
            data_dir=data_dir,            
            criterion=criterion, 
            tuning_method=tuning_method,   
            model_names=model_names
        )

        for model, m_vals in metrics.items():
            errors["high"].setdefault(model, []).append(m_vals["rmse_high"])
            r2_scores.setdefault(model, []).append(m_vals["r2"])
            crps.setdefault(model, []).append(m_vals["crps"])
            timings.setdefault(model, []).append(timing[model])
            coverages.setdefault(model, []).append(coverage[model])

    return errors, r2_scores, crps, timings, coverages

# ============================================================================
# Top-level convenience function
# ============================================================================
def run_experiment(d, N1, N2,
                   n_replicates, base_seed,
                   data_dir, plot_dir,
                   criterion='nlpd',
                   tuning_method='grid',
                   model_names=None,
                   kernel='gaussian'):

    errors, r2_scores, crps, timings, coverages = run_replicates(
        d, N1, N2, n_replicates, base_seed,
        data_dir, 
        criterion=criterion, 
        tuning_method=tuning_method,
        model_names=model_names,
        kernel=kernel
    )

    # RMSE plots
    plot_boxplot(
        errors["high"], N2, plot_dir,
        high_fidelity=True,
        criterion=criterion,
        tuning_method=tuning_method,
        metric_name="RMSE", 
        file_prefix="rmse_boxplot",
        kernel=kernel)

    # R2 Score plots
    plot_boxplot(
        r2_scores, N2, plot_dir,
        high_fidelity=True,
        criterion=criterion,
        tuning_method=tuning_method,
        metric_name="R2 Score", 
        file_prefix="r2_boxplot",
        kernel=kernel)

    plot_boxplot(
        crps, N2, plot_dir,
        high_fidelity=True,
        criterion=criterion,
        tuning_method=tuning_method,
        metric_name="crps", 
        file_prefix="crps_boxplot",
        kernel=kernel)

    # Coverage plot
    plot_boxplot(
        coverages, N2, plot_dir, 
        high_fidelity=True, 
        criterion=criterion, 
        tuning_method=tuning_method, 
        metric_name="Coverage", 
        file_prefix="coverage_boxplot",
        target_value=0.95,
        kernel=kernel
    )

    # Timing box plot
    plot_boxplot(
        timings, plot_dir=plot_dir,
        criterion=criterion,
        tuning_method=tuning_method,
        metric_name="Time (s)",
        is_timing=True,
        N1=N1, N2=N2,
        n_replicates=n_replicates,
        kernel=kernel
    )