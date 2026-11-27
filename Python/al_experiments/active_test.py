# active_test.py
"""
Check of active_learning.py (MacKay ALM and Cohn ALC, next point by multi-start L-BFGS-B with
exact gradients) on the 1-D, two-level Perdikaris problem:
    f_0(x) = sin(8 pi x)                    (low fidelity)
    f_1(x) = (x - sqrt(2)) * sin(8 pi x)^2  (high fidelity, target)

Run from Python/:   python -m al_experiments.active_test
Saves plots/perdikaris/active_learning_alm.png and active_learning_alc.png.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # Python/, so a plain `python <file>` also works
from model.MF_RFF_Bayes import MF_RFF_bayes
import al.active_learning as al
from al.active_learning import active_learning_loop, alm_select, alc_select
from paths import PLOTS

f0 = lambda X: np.sin(8 * np.pi * X[:, 0])
f1 = lambda X: (X[:, 0] - np.sqrt(2)) * np.sin(8 * np.pi * X[:, 0]) ** 2

# ---- Settings
SEED = 0
N_INIT = [13, 8]          # initial points per level (nested: level-1 points are a subset of level-0 points)
COSTS = [1.0, 3.0]        # cost of one evaluation at each level (assumed: high fidelity 3x the low one)
BUDGET = 80.0             # total cost allowed for the acquisitions
BOUNDS = [(0.0, 1.0)]     # input box searched by L-BFGS-B
N_SCREEN = 256            # Sobol screening points per iteration (only pick the L-BFGS-B starts)
N_STARTS = 3              # L-BFGS-B starts per level
CRITERIA = ['alm', 'alc'] # acquisition criteria to run (MacKay, Cohn)
N_REF = 100               # ALC reference points (Latin hypercube)

# Model settings from perdikaris.py (non-linear model), with fewer restarts for speed.
def make_model():
    return MF_RFF_bayes(
        d=1,
        linear=False,
        n_restarts=5,                      # multi-start L-BFGS-B runs for the hyperparameters
        n_samples=250,                     # posterior trajectories S
        tune_nugget=True,
        feature_mults=[6, 10],             # RFF size multiplier per level
        eta_bounds=[(0.5, 2.5), (-3, 0.5)],
        eta_bounds_f=[(-3, 0.5)],
        nugget_bounds=[(-8, -3), (-8, -3)],
        seed=SEED,
        verbose=False,
    )

if __name__ == "__main__":
    # ---- Initial nested design, test set, ALC reference set
    X0 = qmc.LatinHypercube(d=1, seed=SEED).random(N_INIT[0])
    X1 = X0[:N_INIT[1]]
    X_test = np.linspace(0, 1, 500).reshape(-1, 1)
    y_test = f1(X_test)
    X_ref = qmc.LatinHypercube(d=1, seed=SEED).random(N_REF)

    # ---- Gradient checks on the initial model (central differences, step h)
    m0 = make_model().fit([X0, X1], [f0(X0), f1(X1)])
    X_chk = np.vstack([[0.0], [1e-3], [1.0 - 1e-3], [1.0],
                       np.random.default_rng(SEED).random((16, 1))])
    h = 1e-5
    print("Gradient check, ALM (initial model, 20 points): epistemic_var_grad vs predict and FD")
    for l in range(m0.L):
        v, g = m0.epistemic_var_grad(X_chk, level=l)
        v_pred = m0.predict(X_chk, level='all')[l]['var_epistemic'].ravel()
        v_p = m0.predict(X_chk + h, level='all')[l]['var_epistemic'].ravel()
        v_m = m0.predict(X_chk - h, level='all')[l]['var_epistemic'].ravel()
        g_fd = (v_p - v_m) / (2 * h)
        print(f"  level {l}: value vs predict max rel err {np.max(np.abs(v - v_pred) / v_pred):.1e} | "
              f"gradient vs FD max err / max |grad| {np.max(np.abs(g[:, 0] - g_fd)) / np.max(np.abs(g_fd)):.1e}")
    print("Gradient check, ALC (initial model, 20 points): grad log R_l vs FD")
    G0 = al.alc_reference(m0, X_ref)
    for l in range(m0.L):
        noise = al._noise_vars(m0, l)
        res = [(al._log_alc_and_grad(m0, x, l, G0, noise)[1],
                al._log_alc_and_grad_fd(m0, x, l, G0, noise, np.array([h]))[1]) for x in X_chk]
        err = max(abs(a[0] - b[0]) for a, b in res) / max(abs(b[0]) for _, b in res)
        print(f"  level {l}: gradient vs FD max err / max |grad| {err:.1e}")

    for crit_name in CRITERIA:
        print(f"\n===== {crit_name.upper()} =====")
        t0 = time.perf_counter()
        model, X_list, y_list, hist = active_learning_loop(
            make_model, [X0, X1], [f0(X0), f1(X1)], [f0, f1], COSTS, BUDGET, bounds=BOUNDS,
            X_test=X_test, y_test=y_test, n_screen=N_SCREEN, n_starts=N_STARTS, seed=SEED,
            criterion=crit_name, X_ref=X_ref,
        )
        elapsed = time.perf_counter() - t0

        # ---- Log of the acquisitions
        for i, rec in enumerate(hist):
            line = f"fit {i:2d} | spent {rec['spent']:5.1f} | N {rec['N']} | RMSE {rec['rmse']:.4f}"
            if 'level' in rec:
                line += f" | -> level {rec['level']} at x = {rec['x'][0]:.4f}, crit = {np.round(rec['crit'], 6)}"
            print(line)
        print(f"Active learning loop ({crit_name.upper()}): {elapsed:.1f} s")

        # ---- Optimizer check on the final model: L-BFGS-B maximum vs a dense-grid maximum
        X_grid = np.linspace(BOUNDS[0][0], BOUNDS[0][1], 4001).reshape(-1, 1)
        cum_costs = np.cumsum(COSTS)
        if crit_name == 'alm':
            pred_grid = model.predict(X_grid, level='all')
            grid_vals = [pred_grid[l]['var_epistemic'].ravel() for l in range(model.L)]
            _, _, crit, X_best = alm_select(model, COSTS, BOUNDS, n_screen=N_SCREEN, n_starts=N_STARTS, rng=SEED)
        else:
            G = al.alc_reference(model, X_ref)
            Z_grid = [model._evaluate_trajectories_at(k, X_grid) for k in range(model.L)]
            grid_vals = [al.alc_values(Z_grid, l, G, al._noise_vars(model, l)) for l in range(model.L)]
            _, _, crit, X_best = alc_select(model, COSTS, BOUNDS, X_ref, n_screen=N_SCREEN, n_starts=N_STARTS, rng=SEED)
        print(f"Optimizer check (final model), max_x of the {crit_name.upper()} criterion:")
        for l in range(model.L):
            i = int(np.argmax(grid_vals[l]))
            v_opt = crit[l] * cum_costs[l]
            print(f"  level {l}: L-BFGS-B {v_opt:.4e} at x = {X_best[l, 0]:.5f} | "
                  f"grid {grid_vals[l][i]:.4e} at x = {X_grid[i, 0]:.5f} | ratio {v_opt / grid_vals[l][i]:.4f}")

        # ---- Plot: final fit per level (initial vs acquired points) and RMSE vs cost
        pred = model.predict(X_test, level='all')
        truths = [f0(X_test), y_test]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for l in range(2):
            ax = axes[l]
            mean = pred[l]['mean'].ravel()
            sd_epi = np.sqrt(pred[l]['var_epistemic'].ravel())
            ax.plot(X_test, truths[l], 'k--', lw=1, label='truth')
            ax.plot(X_test, mean, 'b', lw=1.5, label='mean')
            ax.fill_between(X_test[:, 0], mean - 2 * sd_epi, mean + 2 * sd_epi, color='b', alpha=0.2,
                            label='mean ± 2 sd (epistemic)')
            n0 = N_INIT[l]
            ax.plot(X_list[l][:n0, 0], y_list[l][:n0], 'ko', ms=5, label='initial')
            ax.plot(X_list[l][n0:, 0], y_list[l][n0:], 'r^', ms=7, label='acquired')
            ax.set_title(f"{crit_name.upper()} - level {l}: {n0} initial + {len(X_list[l]) - n0} acquired")
            ax.set_xlabel('x')
            ax.legend(fontsize=8)

        axes[2].plot([rec['spent'] for rec in hist], [rec['rmse'] for rec in hist], 'o-')
        axes[2].set_xlabel('cost spent')
        axes[2].set_ylabel('RMSE (level 1)')
        axes[2].set_title(f'{crit_name.upper()}: test RMSE vs cost')
        fig.tight_layout()

        os.makedirs(PLOTS / "perdikaris", exist_ok=True)
        path = PLOTS / "perdikaris" / f"active_learning_{crit_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")
