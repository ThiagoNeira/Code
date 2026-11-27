# alm_experiment.py
"""
Active learning (MacKay ALM) comparison: MF_RFF_bayes (active_learning.py) vs RNAmf (R/ALM <fn>.R).

Run the R script first. It writes Data/<fn>/alm_<N1>_..._<NL>/:
  X{l}.npy, y{l}.npy          initial nested designs, column 0 = replicate index
  x_test.npy, yhigh_test.npy  fixed test set at the target level
  alm_settings.npy            [c_1, ..., c_L, budget, n_screen, n_starts, min_sep]
  rna_alm_history.npy         one row per RNAmf fit: rep, step, spent, N_1..N_L, level, x_1..x_d
                              (level is 1-based and NaN on the last fit)
  rna_alm_mean.npy, rna_alm_sigma2.npy   test predictions, one row per history row
  rna_alm_time.npy            (n_rep,) seconds per replicate (all fits and acquisitions)

Fairness: for every replicate the MF-RFF models start from RNAmf's initial design, check that the
Python oracles reproduce R's outputs, and use the costs, budget and acquisition search read from
alm_settings.npy (same budget rule, nested acquisition and repeated-point rule on both sides).
Both methods are scored here with the functions of experiment.py: RMSE, Gaussian CRPS for RNAmf
and sample-based CRPS for MF-RFF.

Outputs in <plot_dir>/<out_name>/ (default "alm"): alm_comparison.png (metric vs cost and metric
at the budget) and alm_results.npz (per-fit curves and per-replicate summaries).

The same driver also compares acquisition criteria for MF-RFF (criteria=('alm', 'alc'),
include_rna=False): every criterion starts from the same initial designs, test set, costs, budget
and search settings. ALC averages the target-level variance reduction over a fixed reference set
(a Latin hypercube of 100 * d points, seed 0, the same for every replicate).

Noisy data (folders written by noisy_data.py, with noise_sd.npy and yhigh_test_noisy.npy): the
oracles are called as f_l(x) + sigma_l * eps (one noise stream per replicate, the same for every
criterion), the initial outputs are checked against f_l within 6 sigma_l, RMSE is measured against
the noise-free test outputs and CRPS against the noisy ones. Without RNAmf output (include_rna=False)
the replicates are read from X1.npy.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from harness.utils import crps_ensemble, crps_gaussian
from harness.experiment import _build_bayes, _load_level
from scipy.stats import qmc

from al.active_learning import active_learning_loop
from paths import under_root


def _crps_samples(samples, y_true, chunk=100):
    """crps_ensemble over chunks of test points (its pairwise term needs N_test * S^2 memory)."""
    y_true = np.ravel(y_true)
    n = len(y_true)
    return sum(crps_ensemble(samples[i:i + chunk], y_true[i:i + chunk]) * len(y_true[i:i + chunk])
               for i in range(0, n, chunk)) / n


def _rmse(mean, y_true):
    return float(np.sqrt(np.mean((np.ravel(mean) - np.ravel(y_true)) ** 2)))


def _on_grid(spent, values, grid):
    """Step function: the value of the last fit whose cost spent is <= c, for every c in grid."""
    idx = np.searchsorted(spent, grid, side='right') - 1
    return np.asarray(values)[np.clip(idx, 0, None)]


def _load_alm_dir(alm_dir, include_rna=True):
    names = ["alm_settings", "x_test", "yhigh_test"]
    if include_rna:
        names += ["rna_alm_history", "rna_alm_mean", "rna_alm_sigma2", "rna_alm_time"]
    data = {name: np.load(os.path.join(alm_dir, f"{name}.npy")) for name in names}
    for name in ("noise_sd", "yhigh_test_noisy"):            # noisy data only (noisy_data.py)
        path = os.path.join(alm_dir, f"{name}.npy")
        data[name] = np.load(path) if os.path.exists(path) else None
    s = data["alm_settings"]
    L = len(s) - 4
    data.update(costs=s[:L], budget=float(s[L]), n_screen=int(s[L + 1]), n_starts=int(s[L + 2]),
                min_sep=float(s[L + 3]), L=L)
    return data


def _rna_replicate(data, rep, y_test):
    """RNAmf curves for one replicate from the stacked R output."""
    H, L = data["rna_alm_history"], data["L"]
    rows = np.flatnonzero(H[:, 0] == rep)
    if rows.size == 0:
        raise ValueError(f"Replicate {rep} not found in rna_alm_history.npy (R run incomplete?)")
    lev = H[rows, 3 + L]
    return {
        'spent': H[rows, 2],
        'rmse': np.array([_rmse(data["rna_alm_mean"][r], y_test) for r in rows]),
        'crps': np.array([crps_gaussian(data["rna_alm_mean"][r], data["rna_alm_sigma2"][r], y_test)
                          for r in rows]),
        'levels': lev[~np.isnan(lev)].astype(int) - 1,          # 0-based, one per acquisition
        'N_final': H[rows[-1], 3:3 + L].astype(int),
        'time': float(data["rna_alm_time"][rep - 1]),
    }


def _noisy_oracle(f, sd, rng):
    return lambda X: np.ravel(f(X)) + sd * rng.standard_normal(len(X))


def run_alm_experiment(d, N_list, oracles, model_names, data_dir, plot_dir,
                       n_replicates=None, kernel='gaussian', jac='analytic',
                       criteria=('alm',), include_rna=True, out_name="alm", n_ref_per_dim=100,
                       nested=True, allow_repeats=False):
    """
    d            : input dimension.
    N_list       : nominal initial sizes per level (names the folder Data/<fn>/alm_<N1>_..._<NL>).
    oracles      : list of L functions, oracles[l](X) -> y at level l (0 = lowest), X of shape (n, d).
                   Noise-free; with noisy data the noise is added here (noise_sd.npy).
    model_names  : {name: MF_RFF_bayes config} as in the benchmark scripts.
    data_dir     : e.g. "Data/perdikaris" (relative to Python/, see paths.py);  plot_dir : e.g. "plots/perdikaris".
    n_replicates : replicates 1..n (default: all replicates in the R output).
    criteria     : MF-RFF acquisition criteria to run, 'alm' and/or 'alc'. With one criterion the
                   methods are named after the models; with several, "<model> (ALM)" etc.
    include_rna  : include RNAmf's precomputed ALM run (from the R output).
    out_name     : output subfolder of plot_dir.
    n_ref_per_dim: ALC reference set size per input dimension.
    nested       : True: an acquisition at level l evaluates levels 0..l (cost C_l); False: level l
                   only (cost c_l). See active_learning.active_learning_loop.
    allow_repeats: re-evaluate a point already in the design (for noisy oracles).
    """
    alm_dir = os.path.join(under_root(data_dir), "alm_" + "_".join(str(n) for n in N_list))
    out_dir = os.path.join(under_root(plot_dir), out_name)
    os.makedirs(out_dir, exist_ok=True)

    data = _load_alm_dir(alm_dir, include_rna=include_rna)
    L, costs, budget = data["L"], data["costs"], data["budget"]
    X_test = data["x_test"].reshape(len(data["yhigh_test"]), d)
    y_test = np.ravel(data["yhigh_test"])
    noise_sd = data["noise_sd"]
    y_obs = y_test if data["yhigh_test_noisy"] is None else np.ravel(data["yhigh_test_noisy"])
    if include_rna:
        n_rep_avail = int(np.nanmax(data["rna_alm_history"][:, 0]))
    else:
        n_rep_avail = int(np.max(np.load(os.path.join(alm_dir, "X1.npy"))[:, 0]))
    n_rep = n_rep_avail if n_replicates is None else min(int(n_replicates), n_rep_avail)
    X_ref = qmc.LatinHypercube(d=d, seed=0).random(n_ref_per_dim * d) if 'alc' in criteria else None
    print(f"Active learning comparison ({', '.join(c.upper() for c in criteria)}): {alm_dir}\n  costs {costs}, budget {budget}, screen {data['n_screen']}, "
          f"starts {data['n_starts']}, min_sep {data['min_sep']}, replicates 1..{n_rep}, "
          f"{'nested' if nested else 'single-level'} acquisition"
          + ("" if noise_sd is None else f", noise sd {np.round(noise_sd, 4).tolist()}"))

    def evaluate(model):
        mean, _, _, _, samples = model.predict(X_test, level=-1, return_samples=True)
        return {'rmse': _rmse(mean, y_test), 'crps': float(_crps_samples(samples, y_obs))}

    runs = [(name, params, crit) for name, params in model_names.items() for crit in criteria]
    label = (lambda name, crit: name) if len(criteria) == 1 else (lambda name, crit: f"{name} ({crit.upper()})")
    methods = (["RNAmf"] if include_rna else []) + [label(name, crit) for name, _, crit in runs]
    results = {m: [] for m in methods}
    for rep in range(1, n_rep + 1):
        X_list = [_load_level(alm_dir, 'X', l, None, rep) for l in range(1, L + 1)]
        y_list = [_load_level(alm_dir, 'y', l, None, rep).ravel() for l in range(1, L + 1)]
        for l in range(L):
            f_X = np.ravel(oracles[l](X_list[l]))
            if noise_sd is None:
                if not np.allclose(f_X, y_list[l], rtol=1e-10, atol=1e-12):
                    raise ValueError(f"Replicate {rep}, level {l + 1}: Python oracle does not reproduce R's outputs.")
            elif np.max(np.abs(y_list[l] - f_X)) > 6 * noise_sd[l]:
                raise ValueError(f"Replicate {rep}, level {l + 1}: outputs differ from the oracle by more than 6 noise sd.")

        if include_rna:
            results["RNAmf"].append(_rna_replicate(data, rep, y_test))
        for name, params, crit in runs:
            is_linear = params.get("linear", name.endswith("_l"))
            make_model = lambda: _build_bayes(name, params, d, rep, kernel, is_linear)
            if noise_sd is None:
                run_oracles = oracles
            else:                                   # same noise stream for every criterion
                noise_rng = np.random.default_rng([rep, 1])
                run_oracles = [_noisy_oracle(f, s, noise_rng) for f, s in zip(oracles, noise_sd)]
            t0 = time.perf_counter()
            _, X_fin, _, hist = active_learning_loop(
                make_model, X_list, y_list, run_oracles, costs, budget,
                n_screen=data["n_screen"], n_starts=data["n_starts"], min_sep=data["min_sep"],
                seed=rep, evaluate=evaluate, jac=jac, criterion=crit, X_ref=X_ref,
                nested=nested, allow_repeats=allow_repeats,
            )
            results[label(name, crit)].append({
                'spent': np.array([h['spent'] for h in hist]),
                'rmse': np.array([h['rmse'] for h in hist]),
                'crps': np.array([h['crps'] for h in hist]),
                'levels': np.array([h['level'] for h in hist if 'level' in h], dtype=int),
                'N_final': np.array([len(X) for X in X_fin]),
                'time': time.perf_counter() - t0,
            })

        line = f"Replicate {rep}/{n_rep}"
        for m in methods:
            r = results[m][-1]
            line += f" | {m}: RMSE {r['rmse'][-1]:.4g}, CRPS {r['crps'][-1]:.4g}, N {r['N_final'].tolist()}, {r['time']:.0f}s"
        print(line)

    # ---- Summaries on a common cost grid
    grid = np.linspace(0.0, budget, 401)
    curves = {m: {k: np.array([_on_grid(r['spent'], r[k], grid) for r in results[m]])
                  for k in ('rmse', 'crps')} for m in methods}
    final = {m: {k: np.array([r[k][-1] for r in results[m]]) for k in ('rmse', 'crps')} for m in methods}
    acq = {m: np.array([np.bincount(r['levels'], minlength=L) for r in results[m]]) for m in methods}
    times = {m: np.array([r['time'] for r in results[m]]) for m in methods}

    print(f"\nAt the budget ({budget:g}), median [IQR] over {n_rep} replicates:")
    for m in methods:
        q = {k: np.percentile(final[m][k], [25, 50, 75]) for k in ('rmse', 'crps')}
        print(f"  {m:18s} RMSE {q['rmse'][1]:.4g} [{q['rmse'][0]:.4g}, {q['rmse'][2]:.4g}] | "
              f"CRPS {q['crps'][1]:.4g} [{q['crps'][0]:.4g}, {q['crps'][2]:.4g}] | "
              f"acquisitions per level {np.round(acq[m].mean(axis=0), 1).tolist()} | "
              f"time {times[m].mean():.1f}s per replicate")

    # ---- Plot: median and IQR vs cost, and the metric at the budget
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for j, (k, label) in enumerate((('rmse', 'RMSE'), ('crps', 'CRPS'))):
        ax = axes[j]
        for c, m in zip(colors, methods):
            lo, med, hi = np.percentile(curves[m][k], [25, 50, 75], axis=0)
            ax.step(grid, med, where='post', color=c, lw=1.8, label=m)
            ax.fill_between(grid, lo, hi, step='post', color=c, alpha=0.2)
        ax.set_yscale('log')
        ax.set_xlabel('cost spent')
        ax.set_ylabel(f'{label} (target level)')
        ax.set_title(f'{label} vs cost: median and IQR over {n_rep} replicates')
        ax.legend(fontsize=8)

        ax = axes[2 + j]
        ax.boxplot([final[m][k] for m in methods])
        ax.set_xticks(range(1, len(methods) + 1), methods)
        ax.set_yscale('log')
        ax.set_ylabel(label)
        ax.set_title(f'{label} at the budget ({budget:g})')
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "alm_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- Raw results
    npz = {'grid': grid, 'costs': costs, 'budget': budget, 'methods': np.array(methods)}
    for m in methods:
        for k in ('rmse', 'crps'):
            npz[f"{m}_{k}_on_grid"] = curves[m][k]
            npz[f"{m}_{k}_final"] = final[m][k]
        npz[f"{m}_acquisitions"] = acq[m]
        npz[f"{m}_time"] = times[m]
        for rep, r in enumerate(results[m], start=1):
            for k in ('spent', 'rmse', 'crps', 'levels'):
                npz[f"{m}_rep{rep}_{k}"] = r[k]
    npz_path = os.path.join(out_dir, "alm_results.npz")
    np.savez(npz_path, **npz)
    print(f"Saved {fig_path}\nSaved {npz_path}")
    return results
