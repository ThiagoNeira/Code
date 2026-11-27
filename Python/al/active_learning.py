# active_learning.py
"""
Active learning for the multi-fidelity MF_RFF_bayes model: MacKay (ALM) and Cohn (ALC).

Each iteration:
  1. Fit the model on the current design.
  2. For every level l, maximize the acquisition criterion over the input box with multi-start
     L-BFGS-B, using its exact gradient. A fresh Sobol screening pool only picks the starts.
       - ALM: the epistemic variance Var[f_l(x)] = var_prop + var_weights at x
              (MF_RFF_bayes.epistemic_var_grad). The noise variance is left out: it does not
              shrink with more data.
       - ALC: the reduction of the target-level epistemic variance, averaged over a reference
              set X_ref, from observing y_0(x), ..., y_l(x) (a nested run at level l).
  3. Pick the level l* with the largest  max_x criterion_l(x) / C_l,  where C_l = c_0 + ... + c_l
     is the cost of a nested run (level l plus all levels below it).
  4. Evaluate the maximizer x_{l*} at levels 0, ..., l* (nested design), add the points, and
     repeat until the budget is spent.

Single-level acquisition (nested=False; the model does not need nested designs): x is evaluated
at level l* only, the cost is c_l instead of C_l, and ALC conditions on y_l(x) alone (B below
has the single column Z_l(x)). ALM is unchanged apart from the cost.

ALC from the coupled posterior draws. With B (S, l+1) the centered draws [Z_0(x), ..., Z_l(x)],
A (n_ref, S) the centered target-level draws on X_ref, and sigma2_k the mean noise variance of level k:
    Sigma(x) = B^T B / S + diag(sigma2_0, ..., sigma2_l)     covariance of the observations at x
    c(x')    = A(x') B / S                                    Cov(f_top(x'), y_{0..l}(x))
    R_l(x)   = mean_x' c(x')^T Sigma^{-1} c(x') = tr(Sigma^{-1} B^T G B) / S^2,   G = A^T A / n_ref.
This is Gaussian conditioning on the joint law of the draws: exact for a Gaussian model (Cohn's
criterion), an approximation through non-linear links. Hyperparameters stay fixed. G is computed
once per iteration, so the cost per candidate does not depend on the size of X_ref.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

_VAR_FLOOR = 1e-300     # log floor: the dual leverage is clipped at 0 in densely sampled regions


def candidate_set(d, n, seed=None):
    """Scrambled Sobol points in [0, 1]^d (n should be a power of 2)."""
    return qmc.Sobol(d=d, scramble=True, seed=seed).random(n)


def _box(bounds):
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    return lb, ub


def _multistart_max(log_f_and_grad, bounds, X_pool, v_pool, n_starts=3, min_sep=0.05, maxiter=100):
    """
    max_x f(x) over the box by multi-start L-BFGS-B on -log f.

    log_f_and_grad : x -> (log f(x), grad log f(x)).
    X_pool, v_pool : screened points inside the box and their values f.
    n_starts       : number of L-BFGS-B runs. Starts are the best pool points, taken greedily so
                     that they are at least `min_sep` apart in box-scaled units (else, in low d,
                     all starts would be neighbours on the same peak).
    Returns (x_best, f_best). The best pool point is the fallback, so the result is never worse
    than the screen.
    """
    lb, ub = _box(bounds)
    width = ub - lb
    order = np.argsort(-v_pool)
    best_x, best_v = X_pool[order[0]].copy(), float(v_pool[order[0]])

    starts = []
    for i in order:
        u = (X_pool[i] - lb) / width
        if all(np.linalg.norm(u - (s - lb) / width) >= min_sep for s in starts):
            starts.append(X_pool[i])
            if len(starts) == n_starts:
                break

    def neg_log_f(x):
        lv, g = log_f_and_grad(x)
        return -lv, -g

    for x0 in starts:
        res = minimize(neg_log_f, x0=x0, method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': maxiter})
        if np.isfinite(res.fun) and np.exp(-res.fun) > best_v:
            best_x, best_v = np.clip(res.x, lb, ub), float(np.exp(-res.fun))
    return best_x, best_v


def _check_jac(jac):
    if jac not in ('analytic', 'fd'):
        raise ValueError(f"jac must be 'analytic' or 'fd'; got {jac!r}.")


# =====================================================================
# ALM: epistemic variance at x
# =====================================================================
def _log_var_and_grad(model, x, level, steps=None):
    """
    log Var[f_level(x)] and its exact gradient, grad Var / Var, from model.epistemic_var_grad.
    At the log floor (variance clipped to 0) the gradient is set to 0. `steps` is unused; it keeps
    the signature of `_log_var_and_grad_fd`.
    """
    v, g = model.epistemic_var_grad(x.reshape(1, -1), level=level)
    v, g = float(v[0]), g[0]
    if v <= _VAR_FLOOR:
        return np.log(_VAR_FLOOR), np.zeros_like(g)
    return np.log(v), g / v


def _log_var_and_grad_fd(model, x, level, steps):
    """
    log Var[f_level(x)] and its central-difference gradient from ONE predict call.

    The rows [x, x + h_j e_j, x - h_j e_j] are stacked, so a full gradient costs the same as one
    evaluation (predict time is dominated by a fixed per-call overhead). Rows may fall a hair
    outside the box near a bound; the RFF features are defined everywhere, so that is harmless.
    """
    d = x.size
    E = np.diag(steps)
    P = np.vstack([x, x + E, x - E])                                  # (1 + 2d, d)
    v = model.predict(P, level='all')[level]['var_epistemic'].ravel()
    lv = np.log(np.maximum(v, _VAR_FLOOR))
    return lv[0], (lv[1:d + 1] - lv[d + 1:]) / (2.0 * steps)


def maximize_epistemic_var(model, level, bounds, X_pool, v_pool, n_starts=3, min_sep=0.05,
                           jac='analytic', fd_step=1e-5, maxiter=100):
    """
    max_x Var[f_level(x)] over the box `bounds` by multi-start L-BFGS-B on -log Var[f_level(x)].

    model    : fitted MF_RFF_bayes.
    level    : level index l.
    bounds   : list of d (lo, hi) pairs.
    X_pool   : (n_pool, d) screened points inside the box; v_pool : (n_pool,) their variances.
    n_starts, min_sep, maxiter : see _multistart_max.
    jac      : 'analytic' (exact gradient from model.epistemic_var_grad, the default) or 'fd'
               (central differences stacked into one predict call; for checks, or for models
               without epistemic_var_grad).
    fd_step  : finite-difference step relative to the box width (jac='fd' only).

    Returns (x_best, v_best).
    """
    _check_jac(jac)
    lb, ub = _box(bounds)
    steps = fd_step * (ub - lb)
    f = _log_var_and_grad if jac == 'analytic' else _log_var_and_grad_fd
    return _multistart_max(lambda x: f(model, x, level, steps), bounds, X_pool, v_pool,
                           n_starts=n_starts, min_sep=min_sep, maxiter=maxiter)


def _run_costs(costs, nested):
    """Cost of acquiring level l: C_l = c_0 + ... + c_l (nested run) or c_l (single level)."""
    costs = np.asarray(costs, dtype=float)
    return np.cumsum(costs) if nested else costs


def _observed_levels(level, nested):
    """Levels observed at x when level `level` is acquired."""
    return list(range(level + 1)) if nested else [level]


def alm_select(model, costs, bounds, allowed=None, n_screen=256, n_starts=3, rng=None,
               nested=True, **opt_kw):
    """
    One ALM decision on a fitted model.

    model    : fitted MF_RFF_bayes.
    costs    : (L,) cost of one evaluation at each level, c_0, ..., c_{L-1}.
    bounds   : list of d (lo, hi) pairs, the input box searched.
    allowed  : optional (L,) boolean mask of levels that may be chosen (e.g. affordable ones);
               levels not allowed are not optimized.
    n_screen : size of the Sobol screening pool (power of 2), drawn fresh from `rng`.
    n_starts : L-BFGS-B starts per level.
    rng      : seed or np.random.Generator for the screening pool.
    nested   : True: level l costs C_l = c_0 + ... + c_l (nested run); False: c_l (single level).
    opt_kw   : passed to maximize_epistemic_var (min_sep, jac, fd_step, maxiter).

    Returns (level, x, crit, X_best), where crit[l] = max_x Var[f_l(x)] / C_l (or / c_l; -inf if
    not allowed) and X_best[l] is the maximizer at level l (NaN if not allowed).
    """
    L, d = model.L, model.d
    lb, ub = _box(bounds)
    allowed = np.ones(L, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)

    X_pool = qmc.scale(candidate_set(d, n_screen, seed=rng), lb, ub)
    pred = model.predict(X_pool, level='all')                  # one call screens every level
    cum_costs = _run_costs(costs, nested)                      # cost of acquiring level l

    crit = np.full(L, -np.inf)
    X_best = np.full((L, d), np.nan)
    for l in range(L):
        if not allowed[l]:
            continue
        X_best[l], v = maximize_epistemic_var(
            model, l, bounds, X_pool, pred[l]['var_epistemic'].ravel(), n_starts=n_starts, **opt_kw
        )
        crit[l] = v / cum_costs[l]
    level = int(np.argmax(crit))
    return level, X_best[level].copy(), crit, X_best


# =====================================================================
# ALC: integrated reduction of the target-level variance
# =====================================================================
def alc_reference(model, X_ref):
    """G = A^T A / n_ref, with A (n_ref, S) the centered target-level draws on X_ref."""
    A = model._evaluate_trajectories_at(model.L - 1, np.asarray(X_ref, dtype=float))
    A = A - A.mean(axis=1, keepdims=True)
    return A.T @ A / A.shape[0]


def _noise_vars(model, level, levels=None):
    """Mean noise variance sigma2_k of the observed levels k (default 0..level)."""
    obs = range(level + 1) if levels is None else levels
    return np.array([np.mean(model.sig2_draws[k]) for k in obs])


def alc_values(Z_list, level, G, noise, levels=None):
    """
    R_level at n points from their draws: Z_list[k] of shape (n, S) for every observed level k
    (`levels`, default 0..level: a nested run; [level]: a single-level acquisition).
    Returns (n,) integrated variance reductions.
    """
    obs = range(level + 1) if levels is None else levels
    B = np.stack([Z_list[k] for k in obs], axis=-1)                     # (n, S, k_obs)
    B = B - B.mean(axis=1, keepdims=True)
    S = B.shape[1]
    Bt = np.swapaxes(B, 1, 2)
    Sigma = Bt @ B / S + np.diag(noise)                                 # (n, k_obs, k_obs)
    M = Bt @ (G @ B) / S ** 2                                           # (n, k_obs, k_obs)
    return np.trace(np.linalg.solve(Sigma, M), axis1=1, axis2=2)


def _alc_value_and_grad(B, dB, G, noise):
    """
    R(x) = tr(Sigma^{-1} M) and its gradient, for one point.
    B (S, k) and dB (S, k, d): draws and their gradients, both centered over the S draws.
      dR/dx_j = 2/S^2 tr(Sigma^{-1} B^T G dB_j) - 2/S tr(P B^T dB_j),  P = Sigma^{-1} M Sigma^{-1}.
    """
    S = B.shape[0]
    GB = G @ B                                                          # (S, k)
    Sigma = B.T @ B / S + np.diag(noise)
    M = B.T @ GB / S ** 2
    Sinv = np.linalg.inv(Sigma)
    R = float(np.trace(Sinv @ M))
    P = Sinv @ M @ Sinv
    W = (2.0 / S ** 2) * (GB @ Sinv) - (2.0 / S) * (B @ P)              # (S, k)
    return R, np.einsum('sk,skd->d', W, dB)


def _log_alc_and_grad(model, x, level, G, noise, steps=None, levels=None):
    """log R_level(x) and its exact gradient, from model.latent_draws_grad.
    `levels`: observed levels (default 0..level)."""
    obs = range(level + 1) if levels is None else levels
    Z_list, dZ_list = model.latent_draws_grad(x.reshape(1, -1), level=level)
    B = np.stack([Z_list[k][0] for k in obs], axis=-1)                 # (S, k_obs)
    dB = np.stack([dZ_list[k][0] for k in obs], axis=1)                # (S, k_obs, d)
    B = B - B.mean(axis=0)
    dB = dB - dB.mean(axis=0)
    R, g = _alc_value_and_grad(B, dB, G, noise)
    if R <= _VAR_FLOOR:
        return np.log(_VAR_FLOOR), np.zeros_like(g)
    return np.log(R), g / R


def _log_alc_and_grad_fd(model, x, level, G, noise, steps, levels=None):
    """log R_level(x) and its central-difference gradient (rows x, x +- h_j e_j in one call)."""
    obs = list(range(level + 1)) if levels is None else list(levels)
    d = x.size
    E = np.diag(steps)
    P = np.vstack([x, x + E, x - E])
    Z_obs = {k: model._evaluate_trajectories_at(k, P) for k in obs}
    lv = np.log(np.maximum(alc_values(Z_obs, level, G, noise, levels=obs), _VAR_FLOOR))
    return lv[0], (lv[1:d + 1] - lv[d + 1:]) / (2.0 * steps)


def alc_select(model, costs, bounds, X_ref, allowed=None, n_screen=256, n_starts=3, rng=None,
               min_sep=0.05, jac='analytic', fd_step=1e-5, maxiter=100, nested=True):
    """
    One ALC decision on a fitted model.

    X_ref  : (n_ref, d) reference inputs over which the target-level variance reduction is averaged.
    nested : True: acquiring level l observes y_0(x), ..., y_l(x) at cost C_l; False: y_l(x) only,
             at cost c_l.
    Other arguments as in alm_select (min_sep, jac, fd_step, maxiter as in maximize_epistemic_var).

    Returns (level, x, crit, X_best), where crit[l] = max_x R_l(x) / C_l (or / c_l; -inf if not
    allowed) and X_best[l] is the maximizer at level l (NaN if not allowed).
    """
    _check_jac(jac)
    L, d = model.L, model.d
    lb, ub = _box(bounds)
    steps = fd_step * (ub - lb)
    allowed = np.ones(L, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
    cum_costs = _run_costs(costs, nested)

    G = alc_reference(model, X_ref)                            # once per decision
    top = int(np.flatnonzero(allowed).max()) if allowed.any() else 0
    X_pool = qmc.scale(candidate_set(d, n_screen, seed=rng), lb, ub)
    Z_pool = [model._evaluate_trajectories_at(k, X_pool) for k in range(top + 1)]
    f = _log_alc_and_grad if jac == 'analytic' else _log_alc_and_grad_fd

    crit = np.full(L, -np.inf)
    X_best = np.full((L, d), np.nan)
    for l in range(L):
        if not allowed[l]:
            continue
        obs = _observed_levels(l, nested)
        noise = _noise_vars(model, l, levels=obs)
        v_pool = alc_values(Z_pool, l, G, noise, levels=obs)
        X_best[l], v = _multistart_max(lambda x: f(model, x, l, G, noise, steps, levels=obs), bounds,
                                       X_pool, v_pool, n_starts=n_starts, min_sep=min_sep,
                                       maxiter=maxiter)
        crit[l] = v / cum_costs[l]
    level = int(np.argmax(crit))
    return level, X_best[level].copy(), crit, X_best


# =====================================================================
# Loop
# =====================================================================
def active_learning_loop(make_model, X_list, y_list, oracles, costs, budget, bounds=None,
                         X_test=None, y_test=None, n_screen=256, n_starts=3, seed=None,
                         evaluate=None, criterion='alm', X_ref=None, nested=True,
                         allow_repeats=False, **opt_kw):
    """
    Runs active learning until no level fits in the remaining budget.

    make_model : function with no arguments returning a new, unfitted MF_RFF_bayes
                 (the model is refitted from scratch, hyperparameters included, every iteration).
    X_list     : list of L initial input arrays, X_list[l] of shape (N_l, d).
    y_list     : list of L initial output arrays, y_list[l] of shape (N_l,).
    oracles    : list of L functions, oracles[l](X) -> y at level l for X of shape (n, d).
    costs      : (L,) cost of one evaluation at each level.
    budget     : total cost allowed for the acquisitions (the initial design is free).
    bounds     : list of d (lo, hi) pairs, the input box searched (default: unit cube).
    X_test, y_test : optional test set at the target level; if given, RMSE is recorded.
    n_screen, n_starts, opt_kw : see alm_select / alc_select
                 (opt_kw: min_sep, jac='analytic'|'fd', fd_step, maxiter).
    seed       : seed for the screening pools (a fresh pool is drawn at every iteration) and for
                 the default X_ref.
    evaluate   : optional function(model) -> dict, called after every fit; its entries are added
                 to that fit's history record (e.g. test metrics such as CRPS).
    criterion  : 'alm' (MacKay) or 'alc' (Cohn).
    X_ref      : ALC reference inputs (default: a Latin hypercube of 100 * d points in the box).
    nested     : True: acquiring level l evaluates x at levels 0..l, at cost C_l = c_0 + ... + c_l.
                 False: at level l only, at cost c_l (single-level acquisition).
    allow_repeats : False: a level that already has x is skipped, and the loop stops if x is
                 already in the design at every level it would be evaluated at. True (noisy
                 oracles): x is evaluated again, since a replicate brings a new noisy observation.

    Returns (model, X_list, y_list, history). history[i] is a dict for fit i:
      'spent' : cost spent so far,  'N' : points per level,  'rmse' : test RMSE (or None),
      plus the entries returned by `evaluate`,
      'level', 'x', 'crit', 'x_best' : the acquisition chosen after that fit (absent for the
      last fit); 'x_best' holds the per-level maximizers.
    """
    if criterion not in ('alm', 'alc'):
        raise ValueError(f"criterion must be 'alm' or 'alc'; got {criterion!r}.")
    X_list = [np.asarray(X, dtype=float).reshape(len(X), -1) for X in X_list]
    y_list = [np.asarray(y, dtype=float).ravel() for y in y_list]
    costs = np.asarray(costs, dtype=float)
    d = X_list[0].shape[1]
    if bounds is None:
        bounds = [(0.0, 1.0)] * d
    if criterion == 'alc' and X_ref is None:
        lb, ub = _box(bounds)
        X_ref = qmc.scale(qmc.LatinHypercube(d=d, seed=seed).random(100 * d), lb, ub)
    rng = np.random.default_rng(seed)
    spent = 0.0
    history = []

    while True:
        model = make_model().fit(X_list, y_list)
        rec = {'spent': spent, 'N': [len(X) for X in X_list], 'rmse': None}
        if X_test is not None:
            mean = model.predict(X_test, level=-1)[0].ravel()
            rec['rmse'] = float(np.sqrt(np.mean((mean - np.ravel(y_test)) ** 2)))
        if evaluate is not None:
            rec.update(evaluate(model))
        history.append(rec)

        # Only levels whose acquisition cost still fits in the remaining budget can be chosen.
        affordable = _run_costs(costs, nested) <= budget - spent
        if not affordable.any():
            break
        if criterion == 'alm':
            level, x, crit, X_best = alm_select(model, costs, bounds, allowed=affordable,
                                                n_screen=n_screen, n_starts=n_starts, rng=rng,
                                                nested=nested, **opt_kw)
        else:
            level, x, crit, X_best = alc_select(model, costs, bounds, X_ref, allowed=affordable,
                                                n_screen=n_screen, n_starts=n_starts, rng=rng,
                                                nested=nested, **opt_kw)
        rec.update(level=level, x=x, crit=crit, x_best=X_best)

        # Evaluate x at levels 0..level (nested) or at `level` only. Without allow_repeats, a level
        # that already has x is skipped.
        x = x.reshape(1, -1)
        added = False
        for k in _observed_levels(level, nested):
            if not allow_repeats and np.any(np.all(np.isclose(X_list[k], x), axis=1)):
                continue
            X_list[k] = np.vstack([X_list[k], x])
            y_list[k] = np.append(y_list[k], oracles[k](x))
            spent += costs[k]
            added = True
        if not added:                                  # x already in the design: refitting would repeat forever
            print(f"{criterion.upper()} stopped: selected x = {x.ravel()} is already in the design "
                  f"at levels {_observed_levels(level, nested)}.")
            break

    return model, X_list, y_list, history
