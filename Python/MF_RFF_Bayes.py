import warnings
import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, block_diag
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from features_full import base_frequencies, scale_frequencies, phi, psi
from rff_linear_algebra import parameter_bayes, score_bayes, score_bayes_with_grad

_PENALTY = 1e10

def _warn_eta_at_bound(name, log_eta, bounds):
    """Flag bandwidths sitting on (or within 2% of the span of) a bound of the search box."""
    log_eta = np.atleast_1d(log_eta)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    tol = 0.02 * (hi - lo)
    hit = (log_eta < lo + tol) | (log_eta > hi - tol)
    if hit.any():
        warnings.warn(f"{name} at search bound in dims {np.where(hit)[0].tolist()}; "
                      f"check input/output scaling.")

def _warn_nugget_at_upper(name, lam, nugget_bounds):
    """Only the UPPER end is suspicious (the level looks like mostly noise); a tiny nugget is normal."""
    lo, hi = nugget_bounds
    if np.log10(lam) > hi - 0.02 * (hi - lo):
        warnings.warn(f"{name} at its upper bound ({lam:.2g}): this level is explained mostly as noise; "
                      f"check the model/scaling.")

class MF_RFF_bayes:
    """
    Multi-Fidelity Bayesian Random Fourier Features (MF-RFF) regression model.
    Combines low-fidelity (cheap) and high-fidelity (expensive) data sources
    using a fully Bayesian treatment of random feature weights.

    Hyperparameters (eta per input dimension, optional nugget) are tuned per level by maximising the
    marginal likelihood with exact analytic gradients and a screened multi-start L-BFGS-B.

    Key options
      freq_method : 'sobol' | 'mc' | 'orthogonal'   how base frequencies are drawn
      feature_mult1/2 : the number of random features is _resolve_m(feature_mult * N) at each level
      n_screen    : pool of n_screen * n_restarts LHS points is screened (value only) to choose starts
      local_frac  : fraction of the n_restarts best screened starts that get a local search

    Search boxes (all in log10 units), independent for each level
      eta_bounds1   : level-1 bandwidths. (lo, hi) applied to all d dims, or a list of d (lo, hi) pairs.
                      None -> [-2 - log10(d), log10(5000) - log10(d)].
      eta_bounds2   : level-2 bandwidths on the x dims (same formats as eta_bounds1). None -> same default as level 1.
      eta_bounds_f1 : level-2 bandwidth on the f1 coordinate (non-linear model only), a single (lo, hi).
                      None -> (-2, log10(500)).
      nugget_bounds1, nugget_bounds2 : (lo, hi) for the level-1 / level-2 nugget.
                      None -> fall back to nugget_bounds (kept for backward compatibility).

    Centering: z (y1) and y (y2) are centred by their sample means before fitting; the means are added back
    in predict(). Predictive variances are unaffected by the shift.
    """
    def __init__(self,
                 d,
                 *,
                 kernel='gaussian',
                 nugget1=1e-6,
                 nugget2=1e-6,
                 n_restarts=10,
                 n_samples=500,
                 linear=False,
                 tune_nugget=True,
                 nugget_bounds=(-8, 0),
                 nugget_bounds1=(-8, 0),
                 nugget_bounds2=(-8, 0),
                 eta_bounds1=(-3.5, 0.8),
                 eta_bounds2=(-3.5, 0.8),
                 eta_bounds_f1=(-2.5, 0.3),
                 seed=None,
                 freq_method='sobol',
                 criterion='nlml',
                 n_screen=5,                                    
                 local_frac=1.0,
                 feature_mult1=3,
                 feature_mult2=6,
                 verbose=True):

        if criterion != 'nlml':
            raise ValueError("Only criterion='nlml' is implemented (the LOO-MSE score was removed).")

        self.d = d
        self.kernel = kernel
        self.nugget1 = nugget1
        self.nugget2 = nugget2
        self.n_starts = n_restarts
        self.samples = n_samples
        self.linear = linear
        self.tune_nugget = tune_nugget
        self.nugget_bounds = tuple(nugget_bounds)       # log10 box (default for both levels)
        self.nugget_bounds1 = tuple(nugget_bounds1) if nugget_bounds1 is not None else self.nugget_bounds
        self.nugget_bounds2 = tuple(nugget_bounds2) if nugget_bounds2 is not None else self.nugget_bounds
        for name, nb in (("nugget_bounds1", self.nugget_bounds1), ("nugget_bounds2", self.nugget_bounds2)):
            if len(nb) != 2 or not nb[0] < nb[1]:
                raise ValueError(f"{name} must be (lo, hi) with lo < hi; got {nb}.")
        self.eta_bounds1 = eta_bounds1                  # log10 boxes, resolved in _tune_bandwidths_optimization
        self.eta_bounds2 = eta_bounds2
        self.eta_bounds_f1 = eta_bounds_f1
        self.seed = seed
        self.criterion = criterion
        self.freq_method = freq_method
        self.n_screen = int(n_screen)
        self.local_frac = local_frac
        self.feature_mult1 = feature_mult1
        self.feature_mult2 = feature_mult2
        self.verbose = verbose

        # The first three children are identical to spawn(3), so seed1/seed2/seed_fit are unchanged.
        s1, s2, s_fit, s_pred = [int(c.generate_state(1)[0])
                                 for c in np.random.SeedSequence(seed).spawn(4)]
        self.seed1, self.seed2 = s1, s2
        self.seed_fit, self.seed_pred = s_fit, s_pred   # posterior sampling at fit / predict

        # Placeholders filled by fit()
        self.tau = None
        self.lam1 = self.lam2 = None
        self.base1 = self.base2 = None
        self.N1 = self.N2 = None
        self.m1 = self.m2 = None
        self.p1 = self.p2 = None
        self.mu1 = self.mu2 = None
        self.Sigma1_inv = self.Sigma2_inv = None
        self.a1 = self.a2 = self.b1 = self.b2 = None
        self.Q1 = self.Q2 = None
        self.sigma1 = self.sigma2 = None
        self.omega1 = self.omega2 = None
        self.eta1 = self.eta2 = None
        self.f1_loc, self.f1_scale = 0.0, 1.0   # standardisation of f1 (non-linear model only)
        self.z_mean = 0.0              # centering constant of the LF responses z (y1)
        self.y_mean = 0.0              # centering constant of the HF responses y (y2)
        self.beta1_hat = None          # level-1 posterior mean (p1, 1)
        self.sigma1_samples = None     # sigma1^2 draws coupled to each beta1 draw
        self.beta1 = self.gamma2 = None
        self.L1 = self.L2 = None

    # ------------------------------------------------------------------ priors
    @staticmethod
    def _resolve_m(N):
        """Number of random frequencies; the max(2, .) guard avoids m <= 0 for tiny N."""
        return max(2, min(int(np.sqrt(N) * np.log(N)), int(0.5 * N) - 1))

    def _prec1(self, lam):
        return lam * np.eye(self.p1)

    def _prec2(self, lam):
        P = lam * np.eye(self.p2)
        if self.linear:
            return block_diag(P, (1.0 / self.tau) * np.eye(1))   # rho keeps its fixed precision
        return P

    def _priors(self, tau):
        self.tau = tau
        self.mu1 = np.zeros((self.p1, 1))
        self.mu2 = np.zeros((self.p2 + int(self.linear), 1))
        self.lam1, self.lam2 = self.nugget1, self.nugget2
        self.Sigma1_inv = self._prec1(self.lam1)
        self.Sigma2_inv = self._prec2(self.lam2)
        self.a1 = self.a2 = self.b1 = self.b2 = 1e-6

    # ------------------------------------------------------------------ objectives
    def _split_theta(self, theta, n_eta):
        theta = np.atleast_1d(theta)
        log_lam = theta[n_eta] if self.tune_nugget else None
        return theta[:n_eta], log_lam

    def _nlml(self, design, y, X_eff, omega, mu, Sigma, a, b, grad):
        """NLML (and its exact gradient w.r.t. [log10 eta, log10 lam]) for one level."""
        if grad:
            return score_bayes_with_grad(design, y, mu=mu, Sigma=Sigma, a=a, b=b,
                                         X_eff=X_eff, omega=omega, tune_nugget=self.tune_nugget)
        return score_bayes(design, y, mu=mu, Sigma=Sigma, a=a, b=b)

    def _objective_eta1(self, theta, X1, z, grad=True):
        log_eta, log_lam = self._split_theta(theta, self.d)
        omega1 = scale_frequencies(self.base1, 10.0 ** log_eta)
        Sigma = self._prec1(10.0 ** log_lam) if self.tune_nugget else self.Sigma1_inv
        return self._nlml(phi(X1, omega1), z, X1, omega1, self.mu1, Sigma, self.a1, self.b1, grad)

    def _objective_eta_high(self, theta, X_eff, f1_col, y, grad=True):
        """
        X_eff : inputs of the level-2 kernel ([X2, standardised f1] for the non-linear model, X2 for the linear one)
        f1_col: plug-in f1(X2), appended as the rho column in the linear model (unused otherwise)
        Both are computed once by the caller, since level 1 is fixed during the level-2 search.
        """
        log_eta, log_lam = self._split_theta(theta, self.base2.shape[1])
        omega2 = scale_frequencies(self.base2, 10.0 ** log_eta)
        Sigma = self._prec2(10.0 ** log_lam) if self.tune_nugget else self.Sigma2_inv

        design = phi(X_eff, omega2)
        if self.linear:
            design = np.hstack([design, f1_col])
        return self._nlml(design, y, X_eff, omega2, self.mu2, Sigma, self.a2, self.b2, grad)

    # ------------------------------------------------------------------ optimisation
    def _multistart_lbfgsb(self, fun, d, n_starts, bounds, rng):
        """
        Screened multi-start L-BFGS-B with exact gradients.
        fun(x, grad) returns the value (grad=False) or (value, gradient) (grad=True).
        """
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

        # 1. Screen a larger LHS pool with value-only evaluations (cheap compared with a local search)
        n_pool = max(n_starts, self.n_screen * n_starts)
        sampler = qmc.LatinHypercube(d=d, seed=rng.integers(int(1e6)))
        pool = qmc.scale(sampler.random(n_pool), lb, ub)
        vals = np.array([value(x0) for x0 in pool])

        # 2. Local searches from the k best screened points; the best raw pool point is the incumbent
        k = max(3, int(np.ceil(self.local_frac * n_starts)))
        best_indices = np.argsort(vals)[:k]

        best_val, best_x = np.inf, None
        if vals[best_indices[0]] < _PENALTY:
            best_val, best_x = vals[best_indices[0]], pool[best_indices[0]]

        opts = {'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-7, 'disp': False}
        for i in best_indices:
            if vals[i] >= _PENALTY:
                continue
            res = minimize(value_and_grad, x0=pool[i], method='L-BFGS-B', jac=True, bounds=bounds, options=opts)
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_x = res.fun, res.x
        return best_x

    @staticmethod
    def _as_box(b, n, name):
        """Normalise a log10 box: (lo, hi) -> n copies; a sequence of n (lo, hi) pairs -> list of tuples."""
        arr = np.asarray(b, dtype=float)
        if arr.shape == (2,):
            boxes = [(float(arr[0]), float(arr[1]))] * n
        elif arr.shape == (n, 2):
            boxes = [(float(lo), float(hi)) for lo, hi in arr]
        else:
            raise ValueError(f"{name} must be (lo, hi) or a list of {n} (lo, hi) pairs; got shape {arr.shape}.")
        if any(not lo < hi for lo, hi in boxes):
            raise ValueError(f"{name} needs lo < hi in every coordinate; got {boxes}.")
        return boxes

    def _tune_bandwidths_optimization(self, X1, z, X2, y):
        """Sequential tuning of (eta, nugget) per level: level 1 on LF data, then level 2 given plug-in f1."""
        rng = np.random.default_rng(self.seed)
        default_box = (-2 - np.log10(self.d), np.log10(5000) - np.log10(self.d))
        bounds_eta1 = self._as_box(default_box if self.eta_bounds1 is None else self.eta_bounds1,
                                   self.d, "eta_bounds1")
        bounds_x2 = self._as_box(default_box if self.eta_bounds2 is None else self.eta_bounds2,
                                 self.d, "eta_bounds2")
        nug_b1 = [self.nugget_bounds1] if self.tune_nugget else []   # appended as the LAST coordinate
        nug_b2 = [self.nugget_bounds2] if self.tune_nugget else []

        # Step 1: low fidelity (eta1 [, nugget1])
        bounds1 = bounds_eta1 + nug_b1
        theta1 = self._multistart_lbfgsb(lambda t, g: self._objective_eta1(t, X1, z, g),
                                         len(bounds1), self.n_starts, bounds1, rng)
        if theta1 is None:
            raise RuntimeError("Optimization for eta1 failed.")

        log_eta1 = theta1[:self.d]
        _warn_eta_at_bound("eta1", log_eta1, bounds_eta1)
        if self.tune_nugget:
            self.lam1 = 10.0 ** theta1[self.d]
            self.Sigma1_inv = self._prec1(self.lam1)        # used by everything downstream
            _warn_nugget_at_upper("nugget1", self.lam1, self.nugget_bounds1)

        # Level-1 quantities are fixed during the level-2 search: compute once
        omega1 = scale_frequencies(self.base1, 10.0 ** log_eta1)
        phi1 = phi(X1, omega1)
        beta1_hat, _, _ = parameter_bayes(phi1, z, mu=self.mu1, Sigma=self.Sigma1_inv)
        f12 = phi(X2, omega1) @ beta1_hat                    # (N2, 1) plug-in f1 at the HF inputs

        # Step 2: high fidelity (eta2 [, nugget2])
        n_eta2 = self.base2.shape[1]
        if self.linear:
            # f1 enters linearly through rho (no bandwidth), so its scale is absorbed: no standardisation
            self.f1_loc, self.f1_scale = 0.0, 1.0
            X_eff, f1_col = X2, f12
            bounds_eta2 = bounds_x2
        else:
            # Non-linear: the kernel acts on (x, f1); standardise f1 with constants fixed here
            f1_ref = phi1 @ beta1_hat
            self.f1_loc = float(f1_ref.mean())
            self.f1_scale = float(f1_ref.std()) or 1.0       # guard against a constant f1
            X_eff, f1_col = np.hstack([X2, (f12 - self.f1_loc) / self.f1_scale]), None
            f1_box = (-2.0, np.log10(500)) if self.eta_bounds_f1 is None else self.eta_bounds_f1
            bounds_eta2 = bounds_x2 + self._as_box(f1_box, 1, "eta_bounds_f1")  # ARD over (x, f1)

        bounds2 = bounds_eta2 + nug_b2
        theta2 = self._multistart_lbfgsb(lambda t, g: self._objective_eta_high(t, X_eff, f1_col, y, g),
                                         len(bounds2), self.n_starts, bounds2, rng)
        if theta2 is None:
            raise RuntimeError("Optimization for eta2 failed.")

        log_eta2 = theta2[:n_eta2]
        _warn_eta_at_bound("eta2", log_eta2, bounds_eta2)
        if self.tune_nugget:
            self.lam2 = 10.0 ** theta2[n_eta2]
            self.Sigma2_inv = self._prec2(self.lam2)
            _warn_nugget_at_upper("nugget2", self.lam2, self.nugget_bounds2)

        return 10.0 ** log_eta1, 10.0 ** log_eta2

    # ------------------------------------------------------------------ fitting
    def _design2(self, X, phi1x, phi2x, beta, f1x):
        """Level-2 design for ONE f1 draw: [phi2 | phi1 @ beta] (linear) or phi([x, standardised f1]) (non-linear)."""
        if self.linear:
            return psi(phi1x, phi2x, beta)
        return phi(np.hstack([X, (f1x - self.f1_loc) / self.f1_scale]), self.omega2)

    def _fit_level2(self, beta, f1x, phi12, phi2, X2, y2):
        """Deterministic level-2 update for one beta1 draw; shared quantities are precomputed."""
        psi2 = self._design2(X2, phi12, phi2, beta, f1x)
        gamma2_hat, Q2_raw, L2 = parameter_bayes(psi2, y2, mu=self.mu2, Sigma=self.Sigma2_inv)
        Q2 = Q2_raw + 2 * self.b2
        sigma2 = Q2 / (self.N2 + 2 * self.a2)
        return gamma2_hat, Q2, sigma2, L2

    def parameters_bayes(self, X1, X2, y1, y2):
        """Fits level-1 weights (beta1) and level-2 weights (gamma2) given the tuned hyperparameters."""
        rng = np.random.default_rng(self.seed_fit)

        phi1 = phi(X1, self.omega1)
        beta1_hat, Q1_raw, self.L1 = parameter_bayes(phi1, y1, mu=self.mu1, Sigma=self.Sigma1_inv)
        self.beta1_hat = beta1_hat
        self.Q1 = Q1_raw + 2 * self.b1
        self.sigma1 = self.Q1 / (self.N1 + 2 * self.a1)

        # Joint NIG draw: sigma^2 ~ IG(a_N, b_N), then beta | sigma^2 ~ N(beta_hat, sigma^2 K^{-1})
        S = self.samples
        a1_N = self.a1 + self.N1 / 2.0
        b1_N = self.Q1 / 2.0
        self.sigma1_samples = 1.0 / rng.gamma(shape=a1_N, scale=1.0 / b1_N, size=S)

        zz = rng.standard_normal((self.p1, S))
        dev = solve_triangular(self.L1, zz, lower=True, trans='T', check_finite=False)      # Cov = L^{-T} L^{-1} = K^{-1}
        self.beta1 = (beta1_hat + np.sqrt(self.sigma1_samples)[None, :] * dev).T   # (S, p1)

        phi12 = phi(X2, self.omega1)                       # (N2, p1), shared by all samples
        Z12 = phi12 @ self.beta1.T                         # (N2, S), all f1 draws at once
        phi2 = phi(X2, self.omega2) if self.linear else None

        with threadpool_limits(limits=1, user_api='blas'):
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(self._fit_level2)(self.beta1[s].reshape(-1, 1), Z12[:, [s]], phi12, phi2, X2, y2)
                for s in range(S)
            )

        gamma2, Q2, sigma2, L2 = zip(*results)
        self.gamma2, self.Q2, self.sigma2, self.L2 = list(gamma2), list(Q2), list(sigma2), list(L2)

    def fit(self, X1, z, X2, y):
        """Main workflow: set feature sizes and priors, draw frequencies, tune bandwidths, fit weights."""
        X1 = np.asarray(X1, dtype=float); z = np.asarray(z, dtype=float).reshape(-1, 1)
        X2 = np.asarray(X2, dtype=float); y = np.asarray(y, dtype=float).reshape(-1, 1)
        if X1.ndim == 1: X1 = X1.reshape(-1, 1)
        if X2.ndim == 1: X2 = X2.reshape(-1, 1)

        self.N1, self.N2 = len(X1), len(X2)
        if self.N1 < 3 or self.N2 < 3:
            raise ValueError("Need at least 3 points per fidelity (predictive variance needs N + 2a > 2).")
        if not all(np.all(np.isfinite(a)) for a in (X1, z, X2, y)):
            raise ValueError("Inputs contain NaN/inf.")

        self.m1 = self._resolve_m(self.feature_mult1 * self.N1)
        self.m2 = self._resolve_m(self.feature_mult2 * self.N2)
        self.p1, self.p2 = 2 * self.m1, 2 * self.m2
        if self.verbose:
            print(f"p1={self.p1}, p2={self.p2}")

        self._priors(tau=1e5)

        d2 = self.d if self.linear else self.d + 1
        self.base1 = base_frequencies(self.m1, self.d, kernel=self.kernel, seed=self.seed1, method=self.freq_method)
        self.base2 = base_frequencies(self.m2, d2, kernel=self.kernel, seed=self.seed2, method=self.freq_method)

        # Centre each level's responses (the zero-mean weight priors have no intercept); means are restored in predict().
        # f1 therefore lives on the centred-z scale at both fit and predict time, which keeps level 2 consistent.
        self.z_mean = float(z.mean())
        self.y_mean = float(y.mean())
        zc = z - self.z_mean
        yc = y - self.y_mean

        self.eta1, self.eta2 = self._tune_bandwidths_optimization(X1, zc, X2, yc)

        self.omega1 = scale_frequencies(self.base1, self.eta1)
        self.omega2 = scale_frequencies(self.base2, self.eta2)

        self.parameters_bayes(X1, X2, zc, yc)
        return self

    # ------------------------------------------------------------------ prediction
    @staticmethod
    def _interval(samples):
        lo = np.quantile(samples, 0.025, axis=1).reshape(-1, 1)
        hi = np.quantile(samples, 0.975, axis=1).reshape(-1, 1)
        return lo, hi

    def predict(self, X, low_fidelity=False, return_ci=False, return_samples=False, random_state=None):
        """
        Predictive mean at X.  Return signatures (unchanged):
          neither flag : y_pred
          return_samples: (y_pred, y_var, samples)                      [low fidelity]
                          (y_pred, y_var, ci_lo, ci_hi, samples)         [high fidelity]
          return_ci    : (y_pred, y_var, ci_lo, ci_hi)                  [low fidelity]
                         (y_pred, y_var, ci_lo, ci_hi, samples)         [high fidelity]
        y_var is the predictive variance of y* (includes the '1 +' observation term).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Reproducible by default; pass a Generator/seed to override
        if isinstance(random_state, np.random.Generator):
            rng = random_state
        else:
            rng = np.random.default_rng(self.seed_pred if random_state is None else random_state)

        S = self.beta1.shape[0]
        N_test = X.shape[0]
        need_samples = return_ci or return_samples
        phi1_star = phi(X, self.omega1)

        # ---- low-fidelity path
        if low_fidelity:
            alpha1 = self.N1 + 2 * self.a1
            y_pred = phi1_star @ self.beta1_hat + self.z_mean         # posterior mean (no MC noise), original scale

            # Student-t: scale^2 * alpha/(alpha-2) = Q1/(alpha1-2) * (1 + h)
            v_mat = solve_triangular(self.L1, phi1_star.T, lower=True, check_finite=False)
            h = np.sum(v_mat ** 2, axis=0).reshape(-1, 1)
            y_var = (self.Q1 / (alpha1 - 2.0)) * (1.0 + h)

            if not need_samples:
                return y_pred

            # noise uses the SAME sigma^2 that generated beta1^s (joint NIG draw)
            noise = rng.standard_normal((N_test, S)) * np.sqrt(self.sigma1_samples)[None, :]
            samples = phi1_star @ self.beta1.T + noise + self.z_mean
            if return_samples:
                return y_pred, y_var, samples
            return (y_pred, y_var) + self._interval(samples)

        # ---- high-fidelity path
        alpha2 = self.N2 + 2 * self.a2
        a2_N = alpha2 / 2.0

        mu_s = np.zeros((N_test, S))
        var_s = np.zeros((N_test, S))
        samples = np.zeros((N_test, S)) if need_samples else None

        Z_pred = phi1_star @ self.beta1.T                              # (N_test, S): all f1 draws at once
        phi2_star = phi(X, self.omega2) if self.linear else None

        for i in range(S):
            psi2 = self._design2(X, phi1_star, phi2_star, self.beta1[i].reshape(-1, 1), Z_pred[:, [i]])

            mu_s[:, i] = (psi2 @ self.gamma2[i]).ravel()
            v_mat2 = solve_triangular(self.L2[i], psi2.T, lower=True, check_finite=False)
            var_s[:, i] = (self.Q2[i] / (alpha2 - 2.0)) * (1.0 + np.sum(v_mat2 ** 2, axis=0))

            if need_samples:
                # Joint NIG draw: sigma^2, then gamma | sigma^2 (shared across test points), then observation noise
                s2 = 1.0 / rng.gamma(shape=a2_N, scale=2.0 / self.Q2[i])          # b_N = Q2/2
                zz = rng.standard_normal(psi2.shape[1])
                dev = solve_triangular(self.L2[i], zz, lower=True, trans='T', check_finite=False)
                gamma_s = self.gamma2[i].ravel() + np.sqrt(s2) * dev
                samples[:, i] = psi2 @ gamma_s + np.sqrt(s2) * rng.standard_normal(N_test)

        y_pred = np.mean(mu_s, axis=1).reshape(-1, 1) + self.y_mean
        # Law of total variance over beta1^s: Var(E[y*|beta1]) + E[Var(y*|beta1)]  (invariant to the shift)
        y_var = np.var(mu_s, axis=1).reshape(-1, 1) + np.mean(var_s, axis=1).reshape(-1, 1)

        if need_samples:
            samples += self.y_mean
            return (y_pred, y_var) + self._interval(samples) + (samples,)
        return y_pred