# experiment.py
import os
import inspect
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from utils import timer, crps_ensemble, crps_gaussian
from MF_RFF_Bayes import MF_RFF_bayes
from NARGP import NARGP
from KO_model import KO
from One_shot_GP import GP
from plotting import plot_boxplot

# ============================================================================
# MF_RFF_bayes configuration
# ============================================================================
# Dynamically read constructor arguments so it naturally supports generalized L-level args
# like `nuggets`, `feature_mults`, `eta_bounds`, etc.
_BAYES_PARAMS = {p for p in inspect.signature(MF_RFF_bayes.__init__).parameters if p not in ("self", "d")}
_RESERVED_KEYS = {"type"}  # keys used by this file, not forwarded to the model

def _build_bayes(name, params, d, seed, kernel, is_linear):
    """
    Builds an MF_RFF_bayes from the model's config dict.
    Defaults to the generalized L-level API parameters.
    """
    unknown = set(params) - _BAYES_PARAMS - _RESERVED_KEYS
    if unknown:
        raise ValueError(f"Model '{name}': unknown MF_RFF_bayes parameter(s) {sorted(unknown)}. "
                         f"Valid: {sorted(_BAYES_PARAMS)} (+ 'type'; 'd' is set by the experiment).")
    
    # nuggets=1e-6 will broadcast across all L levels inside the MF_RFF_bayes class
    kwargs = dict(d=d, seed=seed, kernel=kernel, nuggets=1e-6,
                  n_restarts=5, linear=is_linear)
    kwargs.update({k: v for k, v in params.items() if k not in _RESERVED_KEYS})
    return MF_RFF_bayes(**kwargs)

# ============================================================================
# Single run (Generalized for L levels)
# ============================================================================
def single_run(N_list, d, seed, 
               kernel,
               data_dir, 
               model_names=None):
    """
    Executes a single replicate training and evaluation pass across all models.
    
    Parameters
    ----------
    N_list : list of int
        Sample sizes for each fidelity level, ordered from lowest to highest.
    d : int
        Input dimensionality.
    seed : int
        Random seed for this specific replicate.
    kernel : str
        Kernel family (e.g., 'gaussian', 'matern52').
    data_dir : str
        Base directory containing the generated dataset.
    model_names : dict or list, optional
        Dictionary mapping model aliases to their hyperparameter configurations.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    clean_data_dir = data_dir.replace("Data/", "").replace("Data\\", "")
    
    # Data folder is formatted as N0_N1_..._NL (e.g., "100_50_10")
    folder_name = "_".join(str(n) for n in N_list)
    base_data_dir = os.path.join(script_dir, 'Data', clean_data_dir, folder_name)

    # 1. Load L-Level Data dynamically
    X_list = []
    y_list = []
    for lvl, N in enumerate(N_list):
        # 1-based indexing for filenames (X1, X2, etc.)
        X_list.append(np.load(os.path.join(base_data_dir, f'X{lvl+1}_{N}_rep{seed}.npy')))
        y_list.append(np.load(os.path.join(base_data_dir, f'y{lvl+1}_{N}_rep{seed}.npy')).reshape(-1, 1))

    X_test = np.load(os.path.join(base_data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(base_data_dir, 'yhigh_test.npy'))

    # 2. Setup Models via Dictionary Configuration
    if model_names is None:
        model_names = {
            "KO": {"n_restarts": 5},
            "NARGP": {"n_restarts": 5},
            "MF_RFF_Bayes_l": {"n_restarts": 5},
            "MF_RFF_Bayes_nl": {"n_restarts": 5},
            "GP": {"n_restarts": 5},
            "rna": {},
            "cokm": {}
        }
    elif isinstance(model_names, list):
        model_names = {name: {} for name in model_names}

    models = {}
    model_types = {}

    for name, params in model_names.items():
        model_type = params.get("type", None)
        if model_type is None:
            for known in ["KO", "NARGP", "GP", "MF_RFF_Bayes_l", "MF_RFF_Bayes_nl", "rna", "cokm"]:
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
        elif "MF_RFF_Bayes" in model_type:
            # Determine linearity across L-1 links (default to True if 'l' is in the type/name)
            is_linear = params.get("linear", True if ("_l" in model_type.lower() or "linear" in model_type.lower()) and "nl" not in model_type.lower() else False)
            models[name] = _build_bayes(name, params, d, seed, kernel, is_linear)
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
            time_file = os.path.join(base_data_dir, f'{m_type}_time_rep{seed}.npy')
            time_elapsed = float(np.load(time_file))
            print(f"[TIMER] {name} ({m_type}): {time_elapsed:.3f}s")
            timing[name] = time_elapsed

            y_pred  = np.load(os.path.join(base_data_dir, f'{m_type}_mean_rep{seed}.npy')).reshape(-1, 1)
            y_var   = np.load(os.path.join(base_data_dir, f'{m_type}_sigma2_rep{seed}.npy')).reshape(-1, 1)
            y_lower = np.load(os.path.join(base_data_dir, f'{m_type}_lower_rep{seed}.npy')).reshape(-1, 1)
            y_upper = np.load(os.path.join(base_data_dir, f'{m_type}_upper_rep{seed}.npy')).reshape(-1, 1)

            metrics[name]["crps"] = crps_gaussian(y_pred, y_var, y_test)
            
        else:
            with timer(name, timing):
                if m_type == "GP":
                    # Single fidelity baselines only train on the highest fidelity data
                    model.fit(X_list[-1], y_list[-1])
                    y_pred, y_var, y_lower, y_upper = model.predict(X_test, return_ci=True)
                    metrics[name]["crps"] = crps_gaussian(y_pred, y_var, y_test)
                
                else:
                    # Generalized L-level training for all MF models
                    model.fit(X_list, y_list)

                    if "MF_RFF_Bayes" in m_type:
                        # Extract predictions specifically for the target fidelity (level = -1)
                        # MF_RFF_bayes internally caches all L levels' statistics in `self.last_predictions_`
                        y_pred, _, y_lower, y_upper, y_samples = model.predict(X_test, level=-1, return_samples=True)
                        metrics[name]["crps"] = crps_ensemble(y_samples, y_test)
                    else:
                        # Fallback for standard MF baselines (KO, NARGP)
                        y_pred, y_var, y_lower, y_upper = model.predict(X_test, return_ci=True)
                        metrics[name]["crps"] = crps_gaussian(y_pred, y_var, y_test)

        coverages[name] = ((y_test >= y_lower.ravel()) & (y_test <= y_upper.ravel())).mean()
        metrics[name]["rmse_high"] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics[name]["r2"] = r2_score(y_test, y_pred)
        
    return metrics, timing, coverages

# ============================================================================
# Replicates (Fully Dynamic)
# ============================================================================
def run_replicates(d, N_list, n_replicates, base_seed,
                   data_dir,
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
            N_list, d, seed, 
            kernel,
            data_dir=data_dir,            
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
def run_experiment(d, N_list,
                   n_replicates, base_seed,
                   data_dir, plot_dir,
                   model_names=None,
                   kernel='gaussian'):
    
    # Ensure N_list is a list (provides backwards compatibility if someone passes N1, N2 by accident)
    if isinstance(N_list, int):
        raise TypeError("N_list must be a sequence of sample sizes, e.g., [N1, N2] or [N1, N2, N3].")

    errors, r2_scores, crps, timings, coverages = run_replicates(
        d, N_list, n_replicates, base_seed,
        data_dir, 
        model_names=model_names,
        kernel=kernel
    )

    # Use the highest fidelity sample size for boxplot labeling
    N_target = N_list[-1]

    # RMSE plots
    plot_boxplot(
        errors["high"], N_target, plot_dir,
        high_fidelity=True,
        metric_name="RMSE", 
        file_prefix="rmse_boxplot",
        kernel=kernel)

    # R2 Score plots
    plot_boxplot(
        r2_scores, N_target, plot_dir,
        high_fidelity=True,
        metric_name="R2 Score", 
        file_prefix="r2_boxplot",
        kernel=kernel)

    # CRPS plots
    plot_boxplot(
        crps, N_target, plot_dir,
        high_fidelity=True,
        metric_name="crps", 
        file_prefix="crps_boxplot",
        kernel=kernel)
    
    # Coverage plot
    plot_boxplot(
        coverages, N_target, plot_dir, 
        high_fidelity=True, 
        metric_name="Coverage", 
        file_prefix="coverage_boxplot",
        target_value=0.95,
        kernel=kernel
    )
    
    # Timing box plot (using N_list for full display if the plotting utility supports it, else strings)
    plot_boxplot(
        timings, plot_dir=plot_dir,
        metric_name="Time (s)",
        is_timing=True,
        N1=N_list[0], N2=N_target,  # Passed for backwards compatibility in plotting labels
        n_replicates=n_replicates,
        kernel=kernel
    )