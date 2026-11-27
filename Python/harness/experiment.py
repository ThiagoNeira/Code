# experiment.py
import os
import re
import inspect
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from harness.utils import timer, crps_ensemble, crps_gaussian
from model.MF_RFF_Bayes import MF_RFF_bayes
from baselines.NARGP import NARGP
from baselines.KO_model import KO
from baselines.One_shot_GP import GP
from harness.plotting import plot_boxplot
from paths import DATA

# ============================================================================
# MF_RFF_bayes configuration
# ============================================================================
# Dynamically read constructor arguments so it naturally supports generalized L-level args
# like `nuggets`, `feature_mults`, `eta_bounds`, etc.
_BAYES_PARAMS = {p for p in inspect.signature(MF_RFF_bayes.__init__).parameters if p not in ("self", "d")}
_RESERVED_KEYS = {"type", "standardize"}  # keys used by this file, not forwarded to the model

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


class _Standardized:
    """
    Wraps a GPy baseline (KO, NARGP, GP): fits it on y standardized by one mean and one sd pooled
    over all levels, and maps predictions back (mean * s + m, var * s^2). One common transform keeps
    the relation between levels intact. Used through {"standardize": True} in a model config: the
    baselines' kernel-variance and noise bounds assume roughly unit-scale, zero-mean outputs.
    """

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        multi = isinstance(y, (list, tuple))
        pooled = np.concatenate([np.ravel(v) for v in y]) if multi else np.ravel(y)
        self.loc = float(pooled.mean())
        self.scale = float(pooled.std()) if pooled.std() > 0 else 1.0
        z = ([(np.asarray(v, dtype=float) - self.loc) / self.scale for v in y] if multi
             else (np.asarray(y, dtype=float) - self.loc) / self.scale)
        self.model.fit(X, z)
        return self

    def predict(self, X, return_ci=False, return_var=False):
        mean, var = self.model.predict(X, return_var=True)
        mean = mean * self.scale + self.loc
        var = var * self.scale ** 2
        if return_ci:
            std = np.sqrt(var)
            return mean, var, mean - 1.96 * std, mean + 1.96 * std
        if return_var:
            return mean, var
        return mean

# ============================================================================
# Data loading: stacked R output, with fallback to legacy per-replicate files
# ============================================================================
# Stacked layout (written by the R scripts, one file per quantity):
#   X{l}.npy, y{l}.npy        rows of all replicates; column 0 = replicate index
#   {rna,cokm}_{mean,sigma2,lower,upper}.npy   (n_rep, n_test); row i-1 = replicate i
#   {rna,cokm}_time.npy       (n_rep,); entry i-1 = replicate i
# Legacy layout: X{l}_{N}_rep{i}.npy, y{l}_{N}_rep{i}.npy, {rna,cokm}_{field}_rep{i}.npy
_PRECOMPUTED_FIELDS = ("mean", "sigma2", "lower", "upper", "time")
_stacked_cache = {}  # path -> (mtime, array)


def _load_stacked(path):
    """Loads a stacked file once (reloaded if it changes on disk). Returns None if absent."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _stacked_cache.get(path)
    if cached is None or cached[0] != mtime:
        cached = (mtime, np.load(path))
        _stacked_cache[path] = cached
    return cached[1]


def _load_level(base_data_dir, prefix, lvl, N, seed, legacy=False):
    """Training inputs (prefix 'X', shape (n, d)) or outputs (prefix 'y', shape (n, 1))
    of 1-based level `lvl` for replicate `seed`."""
    path = os.path.join(base_data_dir, f'{prefix}{lvl}.npy')
    stacked = None if legacy else _load_stacked(path)
    if stacked is None:
        arr = np.load(os.path.join(base_data_dir, f'{prefix}{lvl}_{N}_rep{seed}.npy'))
    else:
        arr = stacked[stacked[:, 0] == seed, 1:]
        if arr.shape[0] == 0:
            raise ValueError(f"Replicate {seed} not found in {path}")
    return arr if prefix == 'X' else arr.reshape(-1, 1)


def _load_precomputed(base_data_dir, m_type, field, seed, legacy=False):
    """Precomputed R baseline result ('rna' / 'cokm') for replicate `seed`:
    a float for field 'time', otherwise an (n_test, 1) array."""
    path = os.path.join(base_data_dir, f'{m_type}_{field}.npy')
    stacked = None if legacy else _load_stacked(path)
    if stacked is None:
        arr = np.load(os.path.join(base_data_dir, f'{m_type}_{field}_rep{seed}.npy'))
    else:
        if not 1 <= seed <= stacked.shape[0]:
            raise ValueError(f"Replicate {seed} not found in {path} ({stacked.shape[0]} replicates)")
        arr = stacked[seed - 1]
        if np.all(np.isnan(arr)):
            raise ValueError(f"Replicate {seed} in {path} was not computed (R run incomplete?)")
    if field == 'time':
        return float(np.ravel(arr)[0])
    return np.asarray(arr).reshape(-1, 1)

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
    clean_data_dir = data_dir.replace("Data/", "").replace("Data\\", "")
    
    # Data folder is formatted as N0_N1_..._NL (e.g., "100_50_10")
    folder_name = "_".join(str(n) for n in N_list)
    base_data_dir = os.path.join(DATA, clean_data_dir, folder_name)

    # 1. Load L-Level Data dynamically
    X_list = []
    y_list = []
    for lvl, N in enumerate(N_list):
        # 1-based indexing for filenames (X1, X2, etc.)
        X_list.append(_load_level(base_data_dir, 'X', lvl + 1, N, seed))
        y_list.append(_load_level(base_data_dir, 'y', lvl + 1, N, seed))

    X_test = np.load(os.path.join(base_data_dir, 'x_test.npy'))
    y_test = np.ravel(np.load(os.path.join(base_data_dir, 'yhigh_test.npy')))
    # Noisy data (noisy_data.py): RMSE and R2 against the noise-free outputs, CRPS and coverage
    # against noisy test observations, since every model's predictive includes the noise.
    noisy_path = os.path.join(base_data_dir, 'yhigh_test_noisy.npy')
    y_obs = np.ravel(np.load(noisy_path)) if os.path.exists(noisy_path) else y_test

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
            models[name] = KO(n_restarts=params.get("n_restarts", 5), kernel=kernel, seed=seed)
        elif model_type == "NARGP":
            models[name] = NARGP(n_restarts=params.get("n_restarts", 5), nsamples=params.get("nsamples", 1000), kernel=kernel, seed=seed)
        elif model_type == "GP":
            models[name] = GP(n_restarts=params.get("n_restarts", 5), kernel=kernel)
        elif "MF_RFF_Bayes" in model_type:
            # Determine linearity across L-1 links (default to True if 'l' is in the type/name)
            is_linear = params.get("linear", True if ("_l" in model_type.lower() or "linear" in model_type.lower()) and "nl" not in model_type.lower() else False)
            models[name] = _build_bayes(name, params, d, seed, kernel, is_linear)
        elif model_type in ["rna", "cokm"]:
            models[name] = "precomputed"

        if params.get("standardize", False):
            if model_type not in ("KO", "NARGP", "GP"):
                raise ValueError(f"Model '{name}': 'standardize' applies to KO, NARGP and GP only.")
            models[name] = _Standardized(models[name])

    # 3. Fit and Evaluate 
    timing = {}
    metrics = {}
    coverages = {}

    for name, model in models.items():
        metrics[name] = {}
        m_type = model_types[name]
        
        if m_type in ["rna", "cokm"]:
            time_elapsed = _load_precomputed(base_data_dir, m_type, 'time', seed)
            print(f"[TIMER] {name} ({m_type}): {time_elapsed:.3f}s")
            timing[name] = time_elapsed

            y_pred  = _load_precomputed(base_data_dir, m_type, 'mean', seed)
            y_var   = _load_precomputed(base_data_dir, m_type, 'sigma2', seed)
            y_lower = _load_precomputed(base_data_dir, m_type, 'lower', seed)
            y_upper = _load_precomputed(base_data_dir, m_type, 'upper', seed)

            metrics[name]["crps"] = crps_gaussian(y_pred, y_var, y_obs)

        else:
            with timer(name, timing):
                if m_type == "GP":
                    # Single fidelity baselines only train on the highest fidelity data
                    model.fit(X_list[-1], y_list[-1])
                    y_pred, y_var, y_lower, y_upper = model.predict(X_test, return_ci=True)
                    metrics[name]["crps"] = crps_gaussian(y_pred, y_var, y_obs)
                
                else:
                    # Generalized L-level training for all MF models
                    model.fit(X_list, y_list)

                    if "MF_RFF_Bayes" in m_type:
                        # Extract predictions specifically for the target fidelity (level = -1)
                        # MF_RFF_bayes internally caches all L levels' statistics in `self.last_predictions_`
                        y_pred, _, y_lower, y_upper, y_samples = model.predict(X_test, level=-1, return_samples=True)
                        metrics[name]["crps"] = crps_ensemble(y_samples, y_obs)
                    else:
                        # Fallback for standard MF baselines (KO, NARGP)
                        y_pred, y_var, y_lower, y_upper = model.predict(X_test, return_ci=True)
                        metrics[name]["crps"] = crps_gaussian(y_pred, y_var, y_obs)

        coverages[name] = ((y_obs >= y_lower.ravel()) & (y_obs <= y_upper.ravel())).mean()
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

# ============================================================================
# One-off conversion: legacy per-replicate folders -> stacked layout
# ============================================================================
def pack_legacy_folder(folder, delete_legacy=False):
    """
    Converts one data folder (e.g. Data/branin/20_15_10) from the legacy per-replicate
    files to the stacked layout read by `single_run`, without re-running R.

    The folder name gives N_list; only files matching it are packed (others are reported
    as leftovers). After writing, every replicate is loaded through both layouts and
    compared with np.array_equal (values, shape, dtype). If anything differs, the new
    files are removed and an error is raised. Legacy files are deleted only when
    `delete_legacy=True` and the check passed.
    """
    folder = os.path.normpath(folder)
    N_list = [int(n) for n in os.path.basename(folder).split("_")]
    files = set(os.listdir(folder))

    def seeds_for(pattern):
        rx = re.compile(pattern)
        return sorted(int(m.group(1)) for f in files for m in [rx.fullmatch(f)] if m)

    # --- Build the stacked arrays -------------------------------------------
    out = {}
    used = set()
    seeds = seeds_for(rf"X1_{N_list[0]}_rep(\d+)\.npy")
    for lvl, N in enumerate(N_list, start=1):
        for prefix in ("X", "y"):
            blocks = []
            for s in seeds:
                fname = f"{prefix}{lvl}_{N}_rep{s}.npy"
                arr = np.load(os.path.join(folder, fname))
                arr = arr.reshape(arr.shape[0], -1)
                blocks.append(np.column_stack([np.full(arr.shape[0], float(s)), arr]))
                used.add(fname)
            out[f"{prefix}{lvl}.npy"] = np.vstack(blocks)

    pred_seeds = {}
    for m_type in ("rna", "cokm"):
        s_list = seeds_for(rf"{m_type}_mean_rep(\d+)\.npy")
        if not s_list:
            continue
        pred_seeds[m_type] = s_list
        n_rep = max(s_list)
        for field in _PRECOMPUTED_FIELDS:
            stacked = None
            for s in s_list:
                fname = f"{m_type}_{field}_rep{s}.npy"
                arr = np.ravel(np.load(os.path.join(folder, fname)))
                if stacked is None:
                    stacked = np.full((n_rep,) if field == "time" else (n_rep, arr.size), np.nan)
                stacked[s - 1] = arr[0] if field == "time" else arr
                used.add(fname)
            out[f"{m_type}_{field}.npy"] = stacked

    clash = [f for f in out if f in files]
    if clash:
        raise FileExistsError(f"{folder}: stacked files already exist: {clash}")
    for fname, arr in out.items():
        np.save(os.path.join(folder, fname), arr)

    # --- Verify: stacked loader == legacy loader, bit for bit ---------------
    def same(a, b):
        return (np.array_equal(a, b) and a.shape == b.shape and a.dtype == b.dtype) \
            if isinstance(a, np.ndarray) else a == b

    try:
        for s in seeds:
            for lvl, N in enumerate(N_list, start=1):
                for prefix in ("X", "y"):
                    if not same(_load_level(folder, prefix, lvl, N, s),
                                _load_level(folder, prefix, lvl, N, s, legacy=True)):
                        raise ValueError(f"mismatch: {prefix}{lvl} replicate {s}")
        for m_type, s_list in pred_seeds.items():
            for s in s_list:
                for field in _PRECOMPUTED_FIELDS:
                    if not same(_load_precomputed(folder, m_type, field, s),
                                _load_precomputed(folder, m_type, field, s, legacy=True)):
                        raise ValueError(f"mismatch: {m_type}_{field} replicate {s}")
    except Exception:
        for fname in out:
            os.remove(os.path.join(folder, fname))
        raise

    leftovers = sorted(f for f in files if re.search(r"_rep\d+\.npy$", f) and f not in used)
    if delete_legacy:
        for fname in used:
            os.remove(os.path.join(folder, fname))

    summary = {"folder": folder, "replicates": len(seeds),
               "precomputed": {k: len(v) for k, v in pred_seeds.items()},
               "files_written": sorted(out), "legacy_files_verified": len(used),
               "legacy_deleted": delete_legacy, "unpacked_leftovers": leftovers}
    print(f"[PACK] {folder}: {len(seeds)} replicates, {len(used)} legacy files -> "
          f"{len(out)} stacked files, verified identical"
          + (", legacy deleted" if delete_legacy else "")
          + (f"; {len(leftovers)} leftover per-rep files not matching N_list" if leftovers else ""))
    return summary


def pack_all_legacy(data_root=None, delete_legacy=False):
    """Runs `pack_legacy_folder` on every Data/<fn>/<N1>_..._<NL>/ folder that still
    has legacy files and no stacked files yet."""
    if data_root is None:
        data_root = DATA
    summaries = []
    for fn in sorted(os.listdir(data_root)):
        fn_dir = os.path.join(data_root, fn)
        if not os.path.isdir(fn_dir):
            continue
        for design in sorted(os.listdir(fn_dir)):
            folder = os.path.join(fn_dir, design)
            if not (os.path.isdir(folder) and re.fullmatch(r"\d+(_\d+)+", design)):
                continue
            if os.path.exists(os.path.join(folder, "X1.npy")):
                print(f"[PACK] {folder}: already stacked, skipped")
                continue
            summaries.append(pack_legacy_folder(folder, delete_legacy=delete_legacy))
    return summaries