# MF_RFF_Bayes.py
"""
Generalized L-Level Multi-Fidelity Bayesian Random Fourier Features (MF-RFF) Regression.

Supports an arbitrary number of fidelity levels L >= 1:
  - Level 0 (base fidelity): f_0(x) = phi_0(x) @ w_0
  - Level l (for l = 1, ..., L-1):
      * Linear (Kennedy-O'Hagan) link : f_l(x) = phi_l(x) @ beta_l + rho_l * f_{l-1}(x)
        Solved across all S trajectories simultaneously via exact Schur-complement rank-1 updates.
      * Non-linear (NARGP-style) link : f_l(x) = phi_l([x, tilde{f}_{l-1}(x)]) @ w_l
        Accelerated via precomputed spatial projections X @ omega_x.T.

At prediction time, computes and saves the predictive mean, total predictive variance,
consistent 3-way variance decomposition (var_prop/varexp, var_weights, var_noise, expvar),
and quantiles (q_lo, q_hi) for EVERY fidelity level l = 0, ..., L-1.
"""
import warnings
import numpy as np
import scipy.linalg as sla
from scipy.stats import qmc, t as sp_t
from scipy.optimize import minimize
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from features_full import base_frequencies, scale_frequencies, phi, phi_from_proj
from rff_linear_algebra import (
    parameter_bayes,
    sample_weights_bayes,
    leverage_bayes,
    score_bayes,
    score_bayes_with_grad,
)

_PENALTY = 1e10


# =====================================================================
# 1. Standalone Optimization & Validation Helpers
# =====================================================================
def _broadcast_per_level(val, L, name, is_bounds=False):
    """
    Normalizes a configuration parameter into a list of length L.
    For bound specifications (`is_bounds=True`), distinguishes between:
      - None: replicated as [None] * L.
      - A single 2-tuple `(lo, hi)` of scalars: replicated as `[(lo, hi)] * L`.
      - A sequence of length L: mapped 1-to-1 across the L levels.
    """
    if is_bounds:
        if val is None:
            return [None] * L
        if isinstance(val, tuple) and len(val) == 2 and all(isinstance(x, (int, float, np.number)) for x in val):
            return [(float(val[0]), float(val[1]))] * L
        if isinstance(val, (list, tuple)) and len(val) == L:
            return list(val)
        arr = np.asarray(val, dtype=float)
        if arr.shape == (2,):
            return [(float(arr[0]), float(arr[1]))] * L
        raise ValueError(
            f"{name} must be a single (lo, hi) tuple or a sequence of length {L} (one per level)."
        )

    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) != L:
            raise ValueError(f"{name} must be a scalar or a sequence of length {L}; got length {len(val)}.")
        return list(val)
    return [val] * L


def _as_box(b, n, name):
    """Normalizes a log10 bound specification into a list of `n` (lo, hi) float tuples."""
    arr = np.asarray(b, dtype=float)
    if arr.shape == (2,):
        boxes = [(float(arr[0]), float(arr[1]))] * n
    elif arr.shape == (n, 2):
        boxes = [(float(lo), float(hi)) for lo, hi in arr]
    else:
        raise ValueError(f"{name} must be (lo, hi) or a sequence of {n} (lo, hi) pairs; got shape {arr.shape}.")
    if any(not lo < hi for lo, hi in boxes):
        raise ValueError(f"{name} requires lo < hi in every coordinate; got {boxes}.")
    return boxes


def _warn_eta_at_bound(level_idx, log_eta, bounds):
    """Warns if any optimized bandwidth sits within 2% of the search box boundary."""
    log_eta = np.atleast_1d(log_eta)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    tol = 0.02 * (hi - lo)
    hit = (log_eta < lo + tol) | (log_eta > hi - tol)
    if hit.any():
        warnings.warn(
            f"Level {level_idx} bandwidth eta at search bound in dims {np.where(hit)[0].tolist()}; "
            f"check input/output scaling."
        )


def _warn_nugget_at_upper(level_idx, lam, nugget_bounds):
    """Warns if the optimized nugget sits at the upper bound (indicating pure noise fit)."""
    lo, hi = nugget_bounds
    if np.log10(lam) > hi - 0.02 * (hi - lo):
        warnings.warn(
            f"Level {level_idx} nugget at upper bound ({lam:.2g}): level is explained mostly as noise; "
            f"check model specification or scaling."
        )


def _multistart_lbfgsb(fun, bounds, n_starts, n_screen, local_frac, rng):
    """
    Screened multi-start L-BFGS-B optimizer using exact analytical gradients.
    Runs sequentially without joblib dispatch overhead for fast small/medium-N execution.
    """
    d = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    errors = (np.linalg.LinAlgError, ValueError, FloatingPointError)

    def value(x):
        try:
            val = float(fun(x, False))
        except errors:
            return _PENALTY
        return val if np.isfinite(val) else _PENALTY

    def value_and_grad(x):
        try:
            val, g = fun(x, True)
            val, g = float(val), np.asarray(g, dtype=float)
        except errors:
            return _PENALTY, np.zeros(d)
        if not (np.isfinite(val) and np.all(np.isfinite(g))):
            return _PENALTY, np.zeros(d)
        return val, g

    n_pool = max(n_starts, n_screen * n_starts)
    sampler = qmc.LatinHypercube(d=d, seed=rng)
    pool = qmc.scale(sampler.random(n_pool), lb, ub)
    vals = np.array([value(x0) for x0 in pool], dtype=float)

    k = max(3, int(np.ceil(local_frac * n_starts)))
    best_indices = np.argsort(vals)[:k]

    best_val, best_x = np.inf, None
    if vals[best_indices[0]] < _PENALTY:
        best_val, best_x = float(vals[best_indices[0]]), pool[best_indices[0]]

    opts = {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7, 'disp': False}
    for i in best_indices:
        if vals[i] >= _PENALTY:
            continue
        res = minimize(value_and_grad, x0=pool[i], method='L-BFGS-B', jac=True, bounds=bounds, options=opts)
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val, best_x = float(res.fun), res.x

    return best_x


# =====================================================================
# 2. L-Level Multi-Fidelity Bayesian RFF Model
# =====================================================================
class MF_RFF_bayes:
    """
    Arbitrary L-Level Multi-Fidelity Bayesian Random Fourier Features Regression Model.
    """
    def __init__(self,
                 d,
                 *,
                 kernel='gaussian',
                 nuggets=1e-6,
                 n_restarts=10,
                 n_samples=500,
                 linear=False,
                 tune_nugget=True,
                 nugget_bounds=(-8.0, 0.0),
                 eta_bounds=(-3.5, 0.8),
                 eta_bounds_f=(-2.5, 0.3),
                 feature_mults=None,
                 quantiles=(0.025, 0.975),
                 seed=None,
                 freq_method='sobol',
                 n_screen=5,
                 local_frac=1.0,
                 verbose=True):

        self.d = int(d)
        self.kernel = kernel
        self.nuggets_init = nuggets
        self.n_starts = int(n_restarts)
        self.samples = int(n_samples)
        self.linear_init = linear
        self.tune_nugget = bool(tune_nugget)
        self.nugget_bounds_init = (-8.0, 0.0) if nugget_bounds is None else nugget_bounds
        self.eta_bounds_init = eta_bounds
        self.eta_bounds_f_init = eta_bounds_f
        self.feature_mults_init = feature_mults
        if len(quantiles) != 2 or not (0.0 < quantiles[0] < quantiles[1] < 1.0):
            raise ValueError(f"quantiles must be (q_lo, q_hi) in (0, 1); got {quantiles}.")
        self.quantiles = (float(quantiles[0]), float(quantiles[1]))
        self.seed = seed
        self.freq_method = freq_method
        self.n_screen = int(n_screen)
        self.local_frac = float(local_frac)
        self.verbose = bool(verbose)

        # Hyperpriors: prior variance tau for linear scaling rho, and NIG shape/rate (a_l, b_l)
        self.tau = 1e5
        self.a_prior = 1e-6
        self.b_prior = 1e-6

        # Attributes populated dynamically for L levels inside fit()
        self.L = None
        self.N = []
        self.m = []
        self.p = []
        self.linear = []
        self.lam = []
        self.mu = []
        self.Sigma_inv = []
        self.base_freq = []
        self.eta = []
        self.omega = []
        self.y_mean = []
        self.f_loc = []
        self.f_scale = []

        # Base level 0 posterior state
        self.beta0_hat = None
        self.Q0 = None
        self.L0 = None
        self.phi0_train = None

        # Per-level posterior state across all L levels:
        #   W[l]          : (S, p_l + int(linear[l])) coupled weight draws
        #   sig2_draws[l] : (S,) coupled Inverse-Gamma noise variance draws (reused in predict!)
        #   Q[l]          : float (at l=0) or (S,) array (at l >= 1) of posterior quadratic forms
        #   level_cache[l]: dict of Schur-complement quantities (linear) or Cholesky/dual cache (non-linear)
        self.W = []
        self.sig2_draws = []
        self.Q = []
        self.level_cache = []

        # Deterministic seed streams spawned in fit()
        self.seeds_freq = []
        self.seeds_opt = []
        self.seed_fit = None
        self.seed_pred = None

        # Cache of all L levels' predictions from the most recent predict() call
        self.last_predictions_ = None

    # ------------------------------------------------------------------ utilities
    @staticmethod
    def _resolve_m(N, feature_mult):
        """Resolves number of RFF frequencies m from effective sample size; min 2 for small N."""
        return max(2, feature_mult*min(int(np.sqrt(N) * np.log(N)), int(0.5 * N) - 1))

    def _prec_vector(self, lvl, lam):
        """Constructs the 1D diagonal prior precision vector for level `lvl`."""
        P = np.full(self.p[lvl], lam, dtype=float)
        if self.linear[lvl]:
            return np.append(P, 1.0 / self.tau)
        return P

    def _nl_design_from_proj(self, lvl, proj_x, f_prev):
        """
        Builds the non-linear level `lvl` design matrix using precomputed spatial
        projection `proj_x = X @ omega_x.T` (N, m_l) plus the rank-1 latent term.
        """
        f_std = (f_prev - self.f_loc[lvl]) / self.f_scale[lvl]
        omega_f = self.omega[lvl][:, self.d:]                 # (m_l, 1)
        return phi_from_proj(proj_x + f_std @ omega_f.T)

    def _eval_nlml(self, design, y, X_eff, omega, mu, Sigma, grad):
        """Evaluates NLML (and exact gradient if grad=True) for a single level."""
        if grad:
            return score_bayes_with_grad(
                design, y, mu=mu, Sigma=Sigma, a=self.a_prior, b=self.b_prior,
                X_eff=X_eff, omega=omega, tune_nugget=self.tune_nugget
            )
        return score_bayes(design, y, mu=mu, Sigma=Sigma, a=self.a_prior, b=self.b_prior)

    # ------------------------------------------------------------------ sequential bandwidth tuning
    def _tune_all_levels(self, X_list, yc_list, eta_bounds_list, eta_bounds_f_list, nugget_bounds_list):
        """
        Sequentially tunes (eta_l [, lam_l]) for levels l = 0, ..., L-1 via multi-start L-BFGS-B.
        """
        default_box = (-2.0 - np.log10(self.d), np.log10(5000.0) - np.log10(self.d))
        plugin_weights = [None] * self.L

        def predict_plugin(target_lvl, X_query):
            """Propagates deterministic plug-in mean from level 0 up to `target_lvl` at X_query."""
            f_val = phi(X_query, self.omega[0]) @ plugin_weights[0]
            for k in range(1, target_lvl + 1):
                if self.linear[k]:
                    psi_k = np.hstack([phi(X_query, self.omega[k]), f_val])
                else:
                    proj_xk = X_query @ self.omega[k][:, :self.d].T
                    psi_k = self._nl_design_from_proj(k, proj_xk, f_val)
                f_val = psi_k @ plugin_weights[k]
            return f_val

        for lvl in range(self.L):
            rng_opt = np.random.default_rng(self.seeds_opt[lvl])
            X_l, y_l = X_list[lvl], yc_list[lvl]
            b_x = default_box if eta_bounds_list[lvl] is None else eta_bounds_list[lvl]
            bounds_x = _as_box(b_x, self.d, f"eta_bounds[level={lvl}]")
            nug_box = [nugget_bounds_list[lvl]] if self.tune_nugget else []

            if lvl == 0:
                self.f_loc.append(0.0)
                self.f_scale.append(1.0)
                bounds_eta = bounds_x
                bounds_total = bounds_eta + nug_box

                def obj0(theta, grad):
                    theta = np.atleast_1d(theta)
                    omega0 = scale_frequencies(self.base_freq[0], 10.0 ** theta[:self.d])
                    Sigma0 = self._prec_vector(0, 10.0 ** theta[self.d]) if self.tune_nugget else self.Sigma_inv[0]
                    return self._eval_nlml(phi(X_l, omega0), y_l, X_l, omega0, self.mu[0], Sigma0, grad)

                theta_opt = _multistart_lbfgsb(
                    obj0, bounds_total, self.n_starts, self.n_screen, self.local_frac, rng_opt
                )
            else:
                f_prev_at_xl = predict_plugin(lvl - 1, X_l)
                n_eta = self.base_freq[lvl].shape[1]

                if self.linear[lvl]:
                    self.f_loc.append(0.0)
                    self.f_scale.append(1.0)
                    X_eff, f_col = X_l, f_prev_at_xl
                    bounds_eta = bounds_x
                    design_buf = np.empty((self.N[lvl], self.p[lvl] + 1), dtype=float)
                    design_buf[:, -1:] = f_col
                else:
                    f_ref = predict_plugin(lvl - 1, X_list[lvl - 1])
                    loc_l = float(f_ref.mean())
                    std_l = float(f_ref.std())
                    scale_l = std_l if std_l > 1e-8 else 1.0
                    self.f_loc.append(loc_l)
                    self.f_scale.append(scale_l)

                    X_eff = np.hstack([X_l, (f_prev_at_xl - loc_l) / scale_l])
                    f_col = None
                    b_f = (-2.0, np.log10(500.0)) if eta_bounds_f_list[lvl - 1] is None else eta_bounds_f_list[lvl - 1]
                    bounds_eta = bounds_x + _as_box(b_f, 1, f"eta_bounds_f[level={lvl}]")

                bounds_total = bounds_eta + nug_box

                def obj_l(theta, grad):
                    theta = np.atleast_1d(theta)
                    omega_l = scale_frequencies(self.base_freq[lvl], 10.0 ** theta[:n_eta])
                    Sigma_l = self._prec_vector(lvl, 10.0 ** theta[n_eta]) if self.tune_nugget else self.Sigma_inv[lvl]
                    if self.linear[lvl]:
                        design_buf[:, :self.p[lvl]] = phi(X_eff, omega_l)
                        design = design_buf
                    else:
                        design = phi(X_eff, omega_l)
                    return self._eval_nlml(design, y_l, X_eff, omega_l, self.mu[lvl], Sigma_l, grad)

                theta_opt = _multistart_lbfgsb(
                    obj_l, bounds_total, self.n_starts, self.n_screen, self.local_frac, rng_opt
                )

            if theta_opt is None:
                raise RuntimeError(f"Bandwidth optimization failed at fidelity level {lvl}.")

            n_eta_lvl = self.base_freq[lvl].shape[1]
            log_eta_lvl = theta_opt[:n_eta_lvl]
            _warn_eta_at_bound(lvl, log_eta_lvl, bounds_eta)

            if self.tune_nugget:
                lam_lvl = float(10.0 ** theta_opt[n_eta_lvl])
                self.lam[lvl] = lam_lvl
                self.Sigma_inv[lvl] = self._prec_vector(lvl, lam_lvl)
                _warn_nugget_at_upper(lvl, lam_lvl, nugget_bounds_list[lvl])

            eta_lvl = 10.0 ** log_eta_lvl
            omega_lvl = scale_frequencies(self.base_freq[lvl], eta_lvl)
            self.eta.append(eta_lvl)
            self.omega.append(omega_lvl)

            if lvl == 0:
                psi_lvl = phi(X_l, omega_lvl)
            elif self.linear[lvl]:
                psi_lvl = np.hstack([phi(X_l, omega_lvl), f_prev_at_xl])
            else:
                proj_xl = X_l @ omega_lvl[:, :self.d].T
                psi_lvl = self._nl_design_from_proj(lvl, proj_xl, f_prev_at_xl)
            w_hat, _, _ = parameter_bayes(psi_lvl, y_l, mu=self.mu[lvl], Sigma=self.Sigma_inv[lvl])
            plugin_weights[lvl] = w_hat

    # ------------------------------------------------------------------ fast linear & nonlinear level fits
    def _fit_level_linear_schur(self, lvl, F_prev, X_l, y_l, rng_fit):
        """
        Exact vectorized Schur-complement solver for a linear level `lvl >= 1`:
            Psi_s = [Phi_l(X_l), f_s],   Sigma = diag(lam * I_p, 1 / tau).
        Factors the base spatial RFF system Phi_l ONCE and computes posterior means,
        quadratic forms Q_s, coupled noise draws sig2_s, and posterior weight draws
        W_s = [beta_s, rho_s] across all S trajectories in batched BLAS calls.
        """
        S = self.samples
        N_l, p_l = self.N[lvl], self.p[lvl]
        lam = self.lam[lvl]
        prec_base = np.full(p_l, lam, dtype=float)
        mu_base = np.zeros((p_l, 1), dtype=float)

        phi_l = phi(X_l, self.omega[lvl])                     # (N_l, p_l)
        beta0_hat, _, L_base = parameter_bayes(phi_l, y_l, mu=mu_base, Sigma=prec_base)
        phi_cache_base = phi_l if p_l > N_l else None

        if p_l <= N_l:
            B_F_temp = sla.solve_triangular(L_base, phi_l.T @ F_prev, lower=True, check_finite=False)
            B_F = sla.solve_triangular(L_base, B_F_temp, lower=True, trans='T', check_finite=False)  # (p_l, S)
            u_y = y_l - phi_l @ beta0_hat                                                            # (N_l, 1)
            U = F_prev - phi_l @ B_F                                                                 # (N_l, S)
        else:
            rhs = np.hstack([y_l, F_prev])                                                           # (N_l, 1 + S)
            v_all = sla.solve_triangular(L_base, rhs, lower=True, check_finite=False)
            u_all = sla.solve_triangular(L_base, v_all, lower=True, trans='T', check_finite=False)
            u_y = u_all[:, :1]                                                                       # (N_l, 1)
            U = u_all[:, 1:]                                                                         # (N_l, S)
            B_F = (1.0 / lam) * (phi_l.T @ U)                                                        # (p_l, S)

        a_vec = (1.0 / self.tau) + np.sum(F_prev * U, axis=0)                                        # (S,)
        uty = (F_prev.T @ u_y).ravel()                                                               # (S,)
        rho_hat = uty / a_vec                                                                        # (S,)
        Beta_hat = beta0_hat - B_F * rho_hat[None, :]                                                # (p_l, S)

        y_M0_y = float((y_l.T @ u_y).item())
        Q_raw_vec = np.maximum(0.0, y_M0_y - (uty ** 2) / a_vec)                                     # (S,)
        Q_vec = Q_raw_vec + 2.0 * self.b_prior                                                       # (S,)

        a_l_N = self.a_prior + N_l / 2.0
        sig2_draws = 1.0 / rng_fit.gamma(shape=a_l_N, scale=2.0 / Q_vec)                             # (S,)

        rho_draws = rho_hat + np.sqrt(sig2_draws / a_vec) * rng_fit.standard_normal(S)               # (S,)
        delta_beta = sample_weights_bayes(
            np.zeros((p_l, 1)), L_base, sig2_draws, phi_cache_base, prec_base, rng_fit
        ).T                                                                                          # (p_l, S)
        Beta_draws = Beta_hat - B_F * (rho_draws - rho_hat)[None, :] + delta_beta                    # (p_l, S)
        W_lvl = np.hstack([Beta_draws.T, rho_draws[:, None]])                                        # (S, p_l + 1)

        cache = {
            'L_base': L_base,
            'phi_cache_base': phi_cache_base,
            'prec_base': prec_base,
            'B_F': B_F,
            'a_vec': a_vec,
            'Beta_hat': Beta_hat,
            'rho_hat': rho_hat,
        }
        return W_lvl, sig2_draws, Q_vec, cache

    def _fit_single_trajectory_nl(self, lvl, f_prev_s, proj_x_l, y_l, a_l_N, child_seed):
        """
        Worker for a single trajectory `s` of a non-linear level `lvl >= 1`:
        builds Psi_s from precomputed `proj_x_l`, solves the NIG posterior, caches
        `psi_s` if in dual mode (p > N), and draws the coupled pair (sig2_s, w_l^{(s)}).
        """
        rng_s = np.random.default_rng(child_seed)
        psi_s = self._nl_design_from_proj(lvl, proj_x_l, f_prev_s)
        gamma_hat_s, Q_raw_s, L_s = parameter_bayes(psi_s, y_l, mu=self.mu[lvl], Sigma=self.Sigma_inv[lvl])
        Q_s = Q_raw_s + 2.0 * self.b_prior

        sig2_s = float(1.0 / rng_s.gamma(shape=a_l_N, scale=2.0 / Q_s))
        psi_cache_s = psi_s if psi_s.shape[1] > psi_s.shape[0] else None
        w_s = sample_weights_bayes(
            gamma_hat_s, L_s, np.array([sig2_s]), psi_cache_s, self.Sigma_inv[lvl], rng_s
        ).ravel()
        return gamma_hat_s, Q_s, L_s, psi_cache_s, w_s, sig2_s

    def _fit_posteriors(self, X_list, yc_list):
        """
        Fits full Bayesian posteriors across all L levels and stores coupled
        (W_l, sig2_draws_l) trajectories.
        """
        rng_fit = np.random.default_rng(self.seed_fit)
        S = self.samples

        # ---- Level 0: Exact NIG Posterior & S Coupled Draws
        phi0 = phi(X_list[0], self.omega[0])
        self.beta0_hat, Q0_raw, self.L0 = parameter_bayes(phi0, yc_list[0], mu=self.mu[0], Sigma=self.Sigma_inv[0])
        self.phi0_train = phi0 if self.p[0] > self.N[0] else None
        self.Q0 = Q0_raw + 2.0 * self.b_prior

        a0_N = self.a_prior + self.N[0] / 2.0
        sig2_0 = 1.0 / rng_fit.gamma(shape=a0_N, scale=2.0 / self.Q0, size=S)
        W0 = sample_weights_bayes(
            self.beta0_hat, self.L0, sig2_0, self.phi0_train, self.Sigma_inv[0], rng_fit
        )

        self.W = [W0]
        self.sig2_draws = [sig2_0]
        self.Q = [self.Q0]
        self.level_cache = [None]

        # ---- Levels l = 1, ..., L-1
        for lvl in range(1, self.L):
            X_l, y_l = X_list[lvl], yc_list[lvl]
            Z_prev_at_Xl = self._evaluate_trajectories_at(lvl - 1, X_l)  # (N_l, S)

            if self.linear[lvl]:
                W_lvl, sig2_lvl, Q_vec, cache = self._fit_level_linear_schur(
                    lvl, Z_prev_at_Xl, X_l, y_l, rng_fit
                )
            else:
                proj_x_l = X_l @ self.omega[lvl][:, :self.d].T           # (N_l, m_l) precomputed once!
                a_l_N = self.a_prior + self.N[lvl] / 2.0
                traj_seeds = [
                    int(c.generate_state(1)[0])
                    for c in np.random.SeedSequence(rng_fit.integers(int(1e9))).spawn(S)
                ]

                with threadpool_limits(limits=1, user_api='blas'):
                    results = Parallel(n_jobs=-1, backend="threading")(
                        delayed(self._fit_single_trajectory_nl)(
                            lvl, Z_prev_at_Xl[:, [s]], proj_x_l, y_l, a_l_N, traj_seeds[s]
                        )
                        for s in range(S)
                    )

                gamma_list, Q_list, L_list, psi_cache_list, w_list, sig2_list = zip(*results)
                Q_vec = np.asarray(Q_list, dtype=float)
                sig2_lvl = np.asarray(sig2_list, dtype=float)
                W_lvl = np.vstack(w_list)                                # (S, p_l)

                cache = {
                    'gamma_hat': list(gamma_list),
                    'L_chol': list(L_list),
                    'psi_train': list(psi_cache_list),
                }

            self.W.append(W_lvl)
            self.sig2_draws.append(sig2_lvl)
            self.Q.append(Q_vec)
            self.level_cache.append(cache)

    def _evaluate_trajectories_at(self, target_lvl, X_query):
        """
        Propagates all S posterior weight trajectories from level 0 up to `target_lvl`
        at query inputs `X_query`, returning the centered latent matrix Z of shape (N_query, S).
        """
        Z = phi(X_query, self.omega[0]) @ self.W[0].T
        for lvl in range(1, target_lvl + 1):
            W_lvl = self.W[lvl]
            if self.linear[lvl]:
                phi_x = phi(X_query, self.omega[lvl])
                Z = phi_x @ W_lvl[:, :self.p[lvl]].T + Z * W_lvl[:, self.p[lvl]][None, :]
            else:
                proj_x = X_query @ self.omega[lvl][:, :self.d].T
                Z_next = np.empty_like(Z)
                for s in range(self.samples):
                    psi_s = self._nl_design_from_proj(lvl, proj_x, Z[:, [s]])
                    Z_next[:, s] = psi_s @ W_lvl[s]
                Z = Z_next
        return Z

    # ------------------------------------------------------------------ main fit entrypoint
    def fit(self, X_list, y_list):
        """
        Fits the L-level Multi-Fidelity Bayesian RFF model.
        """
        if not isinstance(X_list, (list, tuple)) or not isinstance(y_list, (list, tuple)):
            raise TypeError("X_list and y_list must be sequences (lists or tuples) of length L.")
        if len(X_list) != len(y_list) or len(X_list) < 1:
            raise ValueError("X_list and y_list must be non-empty sequences of the same length L >= 1.")

        self.L = len(X_list)
        X_clean, y_clean = [], []
        for lvl in range(self.L):
            Xl = np.asarray(X_list[lvl], dtype=float)
            if Xl.ndim == 1:
                Xl = Xl.reshape(-1, 1)
            yl = np.asarray(y_list[lvl], dtype=float).reshape(-1, 1)
            if Xl.shape[1] != self.d:
                raise ValueError(f"Level {lvl} input X has {Xl.shape[1]} columns; expected d={self.d}.")
            if Xl.shape[0] != yl.shape[0]:
                raise ValueError(f"Level {lvl} X and y have mismatched lengths: {Xl.shape[0]} vs {yl.shape[0]}.")
            if Xl.shape[0] < 3:
                raise ValueError(f"Level {lvl} has {Xl.shape[0]} points; need at least 3 points per level.")
            if not (np.all(np.isfinite(Xl)) and np.all(np.isfinite(yl))):
                raise ValueError(f"Level {lvl} inputs contain NaN or Inf.")
            X_clean.append(Xl)
            y_clean.append(yl)

        self.N = [len(Xl) for Xl in X_clean]

        if self.feature_mults_init is None:
            f_mults = [3] + [6] * (self.L - 1)
        else:
            f_mults = _broadcast_per_level(self.feature_mults_init, self.L, "feature_mults")

        nuggets_list = _broadcast_per_level(self.nuggets_init, self.L, "nuggets")
        nugget_bounds_list = _broadcast_per_level(
            self.nugget_bounds_init, self.L, "nugget_bounds", is_bounds=True
        )
        for lvl, nb in enumerate(nugget_bounds_list):
            if nb is None:
                nugget_bounds_list[lvl] = (-8.0, 0.0)
            elif len(nb) != 2 or not (nb[0] < nb[1]):
                raise ValueError(f"nugget_bounds at level {lvl} must be (lo, hi) with lo < hi; got {nb}.")

        eta_bounds_list = _broadcast_per_level(
            self.eta_bounds_init, self.L, "eta_bounds", is_bounds=True
        )
        if self.L > 1:
            linear_links = _broadcast_per_level(self.linear_init, self.L - 1, "linear")
            eta_bounds_f_list = _broadcast_per_level(
                self.eta_bounds_f_init, self.L - 1, "eta_bounds_f", is_bounds=True
            )
        else:
            linear_links = []
            eta_bounds_f_list = []
        self.linear = [False] + [bool(b) for b in linear_links]

        child_seqs = np.random.SeedSequence(self.seed).spawn(2 * self.L + 2)
        all_seeds = [int(c.generate_state(1)[0]) for c in child_seqs]
        self.seeds_freq = all_seeds[:self.L]
        self.seeds_opt = all_seeds[self.L:2 * self.L]
        self.seed_fit = all_seeds[2 * self.L]
        self.seed_pred = all_seeds[2 * self.L + 1]

        self.m = [self._resolve_m(self.N[lvl],f_mults[lvl]) for lvl in range(self.L)]
        self.p = [2 * m_l for m_l in self.m]
        if self.verbose:
            p_str = ", ".join(f"p_{lvl}={self.p[lvl]}" for lvl in range(self.L))
            print(f"MF_RFF_bayes (L={self.L}): {p_str}")

        self.lam = [float(nuggets_list[lvl]) for lvl in range(self.L)]
        self.mu = [np.zeros((self.p[lvl] + int(self.linear[lvl]), 1)) for lvl in range(self.L)]
        self.Sigma_inv = [self._prec_vector(lvl, self.lam[lvl]) for lvl in range(self.L)]

        self.base_freq = []
        for lvl in range(self.L):
            d_lvl = self.d if (lvl == 0 or self.linear[lvl]) else (self.d + 1)
            self.base_freq.append(
                base_frequencies(
                    self.m[lvl], d_lvl, kernel=self.kernel, seed=self.seeds_freq[lvl], method=self.freq_method
                )
            )

        self.y_mean = [float(yl.mean()) for yl in y_clean]
        yc_list = [y_clean[lvl] - self.y_mean[lvl] for lvl in range(self.L)]

        self.eta, self.omega, self.f_loc, self.f_scale = [], [], [], []
        self._tune_all_levels(X_clean, yc_list, eta_bounds_list, eta_bounds_f_list, nugget_bounds_list)
        self._fit_posteriors(X_clean, yc_list)
        return self

    # ------------------------------------------------------------------ prediction for all L levels
    def predict(self, X, level='all', return_samples=False, quantiles=None, random_state=None):
        """
        Computes and stores the predictive mean, total variance, consistent variance
        decomposition, and quantiles across ALL fidelity levels l = 0, ..., L-1.

        Variance keys saved per level l in `self.last_predictions_[l]`:
          - `'var_prop'`    : Epistemic variance from lower levels (0 at l=0, Var_s(mu_s) at l>=1).
          - `'var_weights'` : Own-level parameter uncertainty E_s[Q_s / (alpha_l - 2) * h_s].
          - `'var_noise'`   : Expected observation noise variance E_s[Q_s / (alpha_l - 2)].
          - `'varexp'`      : Alias for `'var_prop'` (consistent across all levels).
          - `'expvar'`      : `'var_weights' + 'var_noise'` (consistent across all levels).
          - `'var'`         : Total predictive variance `'varexp' + 'expvar'`.
        """
        if self.L is None:
            raise RuntimeError("Call fit(X_list, y_list) before predict().")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.d:
            raise ValueError(f"X has {X.shape[1]} columns; expected d={self.d}.")

        q_lo_prob, q_hi_prob = self.quantiles if quantiles is None else quantiles
        rng = (random_state if isinstance(random_state, np.random.Generator)
               else np.random.default_rng(self.seed_pred if random_state is None else random_state))

        N_test = X.shape[0]
        S = self.samples
        all_levels_out = {}

        # ---- Level 0 (Base Fidelity) Prediction
        phi0_star = phi(X, self.omega[0])
        alpha0 = self.N[0] + 2.0 * self.a_prior

        mean0 = phi0_star @ self.beta0_hat + self.y_mean[0]
        h0 = leverage_bayes(phi0_star, self.L0, self.phi0_train, self.Sigma_inv[0]).reshape(-1, 1)

        var_prop0 = np.zeros((N_test, 1), dtype=float)
        var_weights0 = (self.Q0 / (alpha0 - 2.0)) * h0
        var_noise0 = np.full((N_test, 1), self.Q0 / (alpha0 - 2.0))
        varexp0 = var_prop0
        expvar0 = var_weights0 + var_noise0
        var0 = varexp0 + expvar0

        # Exact closed-form Student-t quantiles at Level 0 (instantaneous & noise-free)
        scale0 = np.sqrt((self.Q0 / alpha0) * (1.0 + h0))
        q_lo0 = mean0 + float(sp_t.ppf(q_lo_prob, df=alpha0)) * scale0
        q_hi0 = mean0 + float(sp_t.ppf(q_hi_prob, df=alpha0)) * scale0

        Z_prev = phi0_star @ self.W[0].T if (self.L > 1 or return_samples) else None

        lvl0_dict = {
            'mean': mean0,
            'var': var0,
            'varexp': varexp0,
            'expvar': expvar0,
            'var_prop': var_prop0,
            'var_weights': var_weights0,
            'var_noise': var_noise0,
            'q_lo': q_lo0,
            'q_hi': q_hi0,
        }
        if return_samples:
            noise0 = rng.standard_normal((N_test, S)) * np.sqrt(self.sig2_draws[0])[None, :]
            lvl0_dict['samples'] = Z_prev + noise0 + self.y_mean[0]
        all_levels_out[0] = lvl0_dict

        # ---- Levels l = 1, ..., L-1 Prediction
        for lvl in range(1, self.L):
            alpha_l = self.N[lvl] + 2.0 * self.a_prior
            Q_vec = self.Q[lvl]                                            # (S,)
            cache = self.level_cache[lvl]

            if self.linear[lvl]:
                # Vectorized Schur-complement prediction across all S trajectories at once
                phi_l_star = phi(X, self.omega[lvl])                       # (N_test, p_l)
                h0_star = leverage_bayes(
                    phi_l_star, cache['L_base'], cache['phi_cache_base'], cache['prec_base']
                )                                                          # (N_test,)
                diff_F = Z_prev - (phi_l_star @ cache['B_F'])              # (N_test, S)
                H_mat = h0_star[:, None] + (diff_F ** 2) / cache['a_vec'][None, :]
                mu_s = phi_l_star @ cache['Beta_hat'] + Z_prev * cache['rho_hat'][None, :]

                W_lvl = self.W[lvl]
                Z_curr = phi_l_star @ W_lvl[:, :self.p[lvl]].T + Z_prev * W_lvl[:, self.p[lvl]][None, :]
            else:
                # Fast direct loop with precomputed spatial projection X @ omega_x.T
                proj_x_star = X @ self.omega[lvl][:, :self.d].T            # (N_test, m_l)
                mu_s = np.empty((N_test, S), dtype=float)
                H_mat = np.empty((N_test, S), dtype=float)
                Z_curr = np.empty((N_test, S), dtype=float)

                gamma_list = cache['gamma_hat']
                L_list = cache['L_chol']
                psi_train_list = cache['psi_train']
                Sigma_inv_l = self.Sigma_inv[lvl]
                W_lvl = self.W[lvl]

                for s in range(S):
                    psi_star_s = self._nl_design_from_proj(lvl, proj_x_star, Z_prev[:, [s]])
                    mu_s[:, s] = (psi_star_s @ gamma_list[s]).ravel()
                    H_mat[:, s] = leverage_bayes(psi_star_s, L_list[s], psi_train_list[s], Sigma_inv_l)
                    Z_curr[:, s] = psi_star_s @ W_lvl[s]

            # Consistent 3-way & 2-way Law of Total Variance decomposition
            noise_s = Q_vec / (alpha_l - 2.0)                              # (S,)
            weights_s = H_mat * noise_s[None, :]                           # (N_test, S)

            mean_l = np.mean(mu_s, axis=1).reshape(-1, 1) + self.y_mean[lvl]
            var_prop_l = np.var(mu_s, axis=1).reshape(-1, 1)
            var_weights_l = np.mean(weights_s, axis=1).reshape(-1, 1)
            var_noise_l = np.full((N_test, 1), float(np.mean(noise_s)))
            varexp_l = var_prop_l
            expvar_l = var_weights_l + var_noise_l
            var_l = varexp_l + expvar_l

            # Coupled posterior predictive draws using self.sig2_draws[lvl] from fit()
            noise_l = rng.standard_normal((N_test, S)) * np.sqrt(self.sig2_draws[lvl])[None, :]
            samples_l = Z_curr + noise_l + self.y_mean[lvl]

            # Single vectorized quantile call for both bounds
            q_both = np.quantile(samples_l, (q_lo_prob, q_hi_prob), axis=1)
            q_lo_l = q_both[0].reshape(-1, 1)
            q_hi_l = q_both[1].reshape(-1, 1)

            lvl_dict = {
                'mean': mean_l,
                'var': var_l,
                'varexp': varexp_l,
                'expvar': expvar_l,
                'var_prop': var_prop_l,
                'var_weights': var_weights_l,
                'var_noise': var_noise_l,
                'q_lo': q_lo_l,
                'q_hi': q_hi_l,
            }
            if return_samples:
                lvl_dict['samples'] = samples_l

            all_levels_out[lvl] = lvl_dict
            Z_prev = Z_curr

        self.last_predictions_ = all_levels_out

        if level == 'all':
            return all_levels_out

        lvl_idx = int(level)
        if lvl_idx < 0:
            lvl_idx += self.L
        if not (0 <= lvl_idx < self.L):
            raise ValueError(f"Requested level={level} out of bounds for L={self.L} levels.")

        res = all_levels_out[lvl_idx]
        if return_samples:
            return res['mean'], res['var'], res['q_lo'], res['q_hi'], res['samples']
        return res['mean'], res['var'], res['q_lo'], res['q_hi']