# active_test.py
"""
Smoke test of active_learning.py (ALD / ALM) on the 1-D Perdikaris two-level problem.

  f_0(x) = sin(8 pi x)                      (low fidelity)
  f_1(x) = (x - sqrt(2)) sin(8 pi x)^2      (target)

Run from the Python/ directory:   python active_test.py [ald|alm]

What it checks
  1. `acquire` runs on a fitted model and returns a single (x, level) with sensible shapes,
     and (ALD) the decomposition V_l(x) sums to the total latent variance.
  2. `active_learning_loop` refits / acquires / evaluates until the budget is spent.
  3. A plot of how the design evolves, one panel per level (plots/perdikaris/active_learning_<method>.png).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from MF_RFF_Bayes import MF_RFF_bayes
from active_learning import acquire, active_learning_loop, candidate_set

# ----------------------------------------------------------------------------
# Problem
# ----------------------------------------------------------------------------
def f0(x):
    return np.sin(8 * np.pi * x)


def f1(x):
    return (x - np.sqrt(2)) * np.sin(8 * np.pi * x) ** 2


FUNCS = [f0, f1]
ORACLES = [lambda x, f=f: float(f(np.asarray(x).ravel()[0])) for f in FUNCS]   # noise-free, scalar out

D = 1
L = 2
COSTS = [1.0, 5.0]
NESTED = True                  # evaluating level 1 at x also evaluates level 0 at x (as in the benchmark data)
N_INIT = [6, 3]                # small start so the acquisition has room to work
BUDGET = 60.0
N_CAND = 1024
SEED = 42

METHOD = (sys.argv[1] if len(sys.argv) > 1 else "ald").lower()
PLOT_DIR = "plots/perdikaris"


def make_model():
    return MF_RFF_bayes(
        d=D, kernel="gaussian", seed=SEED, linear=False,
        nuggets=1e-6, tune_nugget=True, nugget_bounds=[(-8, -3), (-8, -3)],
        eta_bounds=[(0.5, 2.5), (-3, 0.5)], eta_bounds_f=[(-3, 0.5)],
        feature_mults=[20, 100], n_restarts=3, n_samples=100, freq_method="sobol",
        verbose=True,
    )


def initial_design(seed=SEED):
    """Nested initial design: X_1 is the first N_INIT[1] points of X_0 (random, scrambled Sobol)."""
    X0 = candidate_set(D, 8, seed=seed)[:N_INIT[0]]
    return [X0, X0[:N_INIT[1]].copy()], [f0(X0[:, 0]), f1(X0[:N_INIT[1], 0])]


# ----------------------------------------------------------------------------
# 1. Single acquisition
# ----------------------------------------------------------------------------
def check_single_acquisition(X_list, y_list):
    model = make_model().fit(X_list, y_list)
    X_cand = candidate_set(D, N_CAND, seed=SEED)
    acq = acquire(model, X_list, y_list, X_cand, method=METHOD, costs=COSTS, nested=NESTED, seed=SEED)
    print(f"[acquire] method={METHOD}")
    print("   x          :", np.round(acq.x, 4))
    print("   level      :", acq.level)
    print("   eval_levels:", acq.eval_levels)
    print("   cost       :", acq.cost)
    print("   max score  :", np.round(acq.score.max(axis=0), 5), "(per level, gain / cost)")
    assert acq.x.shape == (D,)
    assert 0 <= acq.level < L
    assert acq.eval_levels == (list(range(acq.level + 1)) if NESTED else [acq.level])
    assert np.isclose(acq.cost, sum(COSTS[k] for k in acq.eval_levels))
    assert acq.score.shape == (N_CAND, L) and np.all(np.isfinite(acq.score))
    assert np.allclose(X_cand[acq.index], acq.x)
    assert np.unravel_index(np.argmax(acq.score), acq.score.shape) == (acq.index, acq.level)
    if METHOD == "ald":
        dec = acq.detail
        assert dec.V.shape == (N_CAND, L) and dec.var_g.shape == (N_CAND, L) and dec.target == L - 1
        assert np.allclose(dec.V.sum(axis=1), dec.total)       # sigma_T^2 = sum_l V_l
        assert np.all(dec.total > 0)
        assert np.all(dec.V[:, -1] >= 0)                       # top-level term is analytic (no MC noise)
    print("[acquire] shape / consistency checks passed.\n")


# ----------------------------------------------------------------------------
# 2. Full loop with a callback that records per-iteration diagnostics
# ----------------------------------------------------------------------------
def run_loop(X_list, y_list):
    grid = np.linspace(0, 1, 400).reshape(-1, 1)
    snaps = []                                         # per-iteration: acquisition surface (gain / cost)

    def callback(model, acq, it, spent):
        # The loop calls this once more for the acquisition that would exceed the budget; skip it.
        if spent + acq.cost > BUDGET:
            print(f"  it={it:2d} spent={spent:5.1f}  next cost={acq.cost} would exceed the budget -> stop")
            return
        # acq.score is (N_CAND, L) = criterion / cost on the loop's candidate set (same seed rule).
        xc = candidate_set(D, N_CAND, seed=SEED + it)[:, 0]
        order = np.argsort(xc)
        snaps.append(dict(it=it, spent=spent, x=xc[order], score=acq.score[order],
                          picked=acq.x.copy(), level=acq.level))
        print(f"  it={it:2d} spent={spent:5.1f}  next x={np.round(acq.x, 4)} "
              f"level={acq.level} eval_levels={acq.eval_levels} cost={acq.cost}")

    model, X_list, y_list, hist = active_learning_loop(
        make_model, X_list, y_list, ORACLES, budget=BUDGET, costs=COSTS, method=METHOD,
        nested=NESTED, n_cand=N_CAND, seed=SEED, callback=callback)
    return model, X_list, y_list, hist, snaps, grid


# ----------------------------------------------------------------------------
# 3. Plots
# ----------------------------------------------------------------------------
def acquisition_log(X_init, hist):
    """Rows (iteration, level, x). Iteration 0 = initial design; AL iteration i -> i + 1."""
    rows = [(0, k, float(x)) for k, X in enumerate(X_init) for x in X[:, 0]]
    for h in hist:
        rows += [(h["it"] + 1, k, float(h["x"][0])) for k in h["eval_levels"]]
    return np.array(rows)


def plot_evolution(X_init, hist, model, grid, snaps, path):
    log = acquisition_log(X_init, hist)
    n_it = int(log[:, 0].max())
    xs = grid[:, 0]
    cmap = plt.get_cmap("viridis")
    names = ["Level 0  (f$_0$ = sin 8$\\pi$x)", "Level 1  (f$_1$, target)"]

    fig, axes = plt.subplots(2, L, figsize=(6.2 * L, 8), gridspec_kw=dict(height_ratios=[1, 1.6]),
                             constrained_layout=True)
    pred = model.predict(grid, level="all")
    for k in range(L):
        # ---- top: function, final fit and the points coloured by acquisition iteration
        ax = axes[0, k]
        ax.plot(xs, FUNCS[k](xs), "k-", lw=1.2, label="true")
        mean, var = np.ravel(pred[k]["mean"]), np.ravel(pred[k]["var"])
        ax.plot(xs, mean, "C3--", lw=1, label="final posterior mean")
        ax.fill_between(xs, mean - 2 * np.sqrt(var), mean + 2 * np.sqrt(var), color="C3", alpha=0.15)
        sel = log[log[:, 1] == k]
        sc = ax.scatter(sel[:, 2], FUNCS[k](sel[:, 2]), c=sel[:, 0], cmap=cmap, vmin=0, vmax=max(n_it, 1),
                        s=38, edgecolor="k", linewidth=0.5, zorder=3)
        ax.set_title(names[k] + f"  -  {len(sel)} pts")
        ax.set_xlabel("x"); ax.set_ylabel(f"f$_{k}$(x)")
        if k == 0:
            ax.legend(loc="lower left", fontsize=8)
        fig.colorbar(sc, ax=ax, label="acquisition iteration (0 = initial)")

        # ---- bottom: sampling evolution = x location vs iteration
        ax = axes[1, k]
        ini, new = sel[sel[:, 0] == 0], sel[sel[:, 0] > 0]
        ax.scatter(ini[:, 2], ini[:, 0], marker="s", c="0.5", s=40, label="initial design")
        ax.scatter(new[:, 2], new[:, 0], c=new[:, 0], cmap=cmap, vmin=0, vmax=max(n_it, 1),
                   s=48, edgecolor="k", linewidth=0.5, label="acquired", zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, n_it + 0.5)
        ax.set_xlabel("x"); ax.set_ylabel("iteration")
        ax.set_title(f"Sampling evolution - level {k}")
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"{METHOD.upper()} on Perdikaris - costs={COSTS}, nested={NESTED}, budget={BUDGET:g}")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_acquisition_curves(snaps, grid, path):
    """Criterion / cost per level over x at the first, a middle and the last iteration."""
    if not snaps:
        return
    pick = sorted({0, len(snaps) // 2, len(snaps) - 1})
    fig, axes = plt.subplots(1, len(pick), figsize=(5.2 * len(pick), 3.8), constrained_layout=True, squeeze=False)
    for ax, i in zip(axes[0], pick):
        s = snaps[i]
        for k in range(L):
            ax.plot(s["x"], s["score"][:, k], label=f"score/cost level {k}")
        ax.axvline(s["picked"][0], color=f"C{s['level']}", ls=":", lw=1.5)
        ax.set_title(f"iteration {s['it']+1}  (spent {s['spent']:g})")
        ax.set_xlabel("x"); ax.set_yscale("symlog", linthresh=1e-6)
        ax.legend(fontsize=8)
    fig.suptitle(f"{METHOD.upper()}: acquisition surface (dotted = chosen point, colour = level)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    X_init, y_init = initial_design()
    print(f"Initial design: N = {[len(X) for X in X_init]}, method = {METHOD}\n")

    check_single_acquisition(X_init, y_init)

    print("Active learning loop:")
    model, X_list, y_list, hist, snaps, grid = run_loop(X_init, y_init)
    print(f"\nFinal N per level: {[len(X) for X in X_list]}  "
          f"(spent {hist[-1]['spent'] if hist else 0:g} of {BUDGET:g})")

    # accuracy of the target level before / after
    x_test = np.linspace(0, 1, 1000).reshape(-1, 1)
    rmse = lambda m: float(np.sqrt(np.mean(
        (np.ravel(m.predict(x_test, level=L - 1)[0]) - f1(x_test[:, 0])) ** 2)))
    print(f"RMSE target level: initial={rmse(make_model().fit(X_init, y_init)):.4f}  "
          f"after AL={rmse(model):.4f}")

    p1 = os.path.join(PLOT_DIR, f"active_learning_{METHOD}.png")
    p2 = os.path.join(PLOT_DIR, f"active_learning_{METHOD}_acquisition.png")
    plot_evolution(X_init, hist, model, grid, snaps, p1)
    plot_acquisition_curves(snaps, grid, p2)
    print(f"Saved {p1}\nSaved {p2}")


if __name__ == "__main__":
    main()
