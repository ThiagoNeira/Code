import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, block_diag
from features_full import base_frequencies, scale_frequencies, phi, psi
from rff_linear_algebra import parameter_bayes, score_bayes
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import warnings

class MF_RFF_bayes:
    """
    Multi-Fidelity Bayesian Random Fourier Features (MF-RFF) regression model.
    Combines low-fidelity (cheap) and high-fidelity (expensive) data sources 
    using a fully Bayesian treatment of random feature weights.
    """
    def __init__(self,
                 d, 
                 *,
                 kernel = 'gaussian',
                 nugget1 = 1e-6,
                 nugget2 = 1e-6,
                 n_restarts = 10,
                 n_samples = 500,
                 criterion = 'nlml',
                 linear = False,
                 tune_nugget=False,
                 nugget_bounds=(-7, -1),
                 seed=None):
         
        self.d = d
        self.kernel = kernel
        self.nugget1 = nugget1
        self.nugget2 = nugget2
        self.n_starts = n_restarts
        self.samples = n_samples
        self.criterion = criterion
        self.linear = linear 
        self.seed = seed
        # Replace the original 3-seed spawn. The first three children are identical
        # to spawn(3), so seed1/2/3 (and omega1/2/3) are unchanged.
        s1, s2, s_fit, s_pred = [int(c.generate_state(1)[0])
              for c in np.random.SeedSequence(seed).spawn(4)]
        self.seed1, self.seed2 = s1, s2
        self.seed_fit, self.seed_pred = s_fit, s_pred   # posterior sampling at fit / predict
        # Placeholders for data sizes, dimensions, priors, coefficients, and tracking variables
        self.tune_nugget = tune_nugget
        self.nugget_bounds = tuple(nugget_bounds)      # log10 box
        self.tau = None
        self.lam1 = self.lam2 = None    
        self.base1 = self.base2 = None 
        self.N1 = self.N2 = None
        self.m1 = self.m2 =  None
        self.p1 = self.p2 =  None
        self.mu1 = self.mu2 = None
        self.Sigma1_inv = self.Sigma2_inv = None
        self.a1 = self.a2 = self.b1 = self.b2 = None
        self.Q1 = self.Q2 = None
        self.sigma1 = self.sigma2 = None
        self.omega1 = self.omega2 = self.omega3 = None
        self.eta1 = self.eta2 = None
        self.beta1_hat = None          # level-1 posterior mean (p1, 1)
        self.sigma1_samples = None     # sigma1^2 draws coupled to each beta1 draw
        self.beta1 = self.gamma2 = None
        self.L1 = self.L2 = None
      
    def _resolve_m(self, N):
        return min(int(np.sqrt(N) * np.log(N)), int(0.5*N)-1)

    def _prec1(self, lam):
        return lam * np.eye(self.p1)
    
    def _prec2(self, lam):
        P = lam * np.eye(self.p2)
        if self.linear:
            return block_diag(P, (1.0 / self.tau) * np.eye(1))   # rho keeps its fixed precision
        return P

    def _priors(self, tau):
        self.tau = tau
        self.mu1 = np.zeros(self.p1).reshape(-1, 1)
        self.mu2 = np.zeros(self.p2 + int(self.linear)).reshape(-1, 1)
        self.lam1, self.lam2 = self.nugget1, self.nugget2
        self.Sigma1_inv = self._prec1(self.lam1)
        self.Sigma2_inv = self._prec2(self.lam2)
        self.a1 = self.a2 = self.b1 = self.b2 = 1e-6

    def _split_theta(self, theta, n_eta):
        theta = np.atleast_1d(theta)
        log_lam = theta[n_eta] if self.tune_nugget else None
        return theta[:n_eta], log_lam

    def _objective_eta1(self, theta, X1, z):
        log_eta, log_lam = self._split_theta(theta, self.d)
        omega1 = scale_frequencies(self.base1, 10.0 ** log_eta)
        Sigma = self._prec1(10.0 ** log_lam) if self.tune_nugget else self.Sigma1_inv
        phi1 = phi(X1, omega1)
        return score_bayes(phi1, z, mu=self.mu1, Sigma=Sigma, a=self.a1, b=self.b1, criterion=self.criterion)

    def _objective_eta_high(self, theta, phi12, beta1, z12, X2, y):
        """Level-1 quantities (phi12, beta1, z12) are precomputed once by the caller."""
        log_eta, log_lam = self._split_theta(theta, self.base2.shape[1])
        omega2 = scale_frequencies(self.base2, 10.0 ** log_eta)
        Sigma = self._prec2(10.0 ** log_lam) if self.tune_nugget else self.Sigma2_inv
 
        if self.linear:
            phi2 = phi(X2, omega2)
            psi2 = psi(phi12, phi2, beta1)
        else:
            psi2 = phi(np.hstack([X2, z12]), omega2)
 
        return score_bayes(psi2, y, mu=self.mu2, Sigma=Sigma, a=self.a2, b=self.b2, criterion=self.criterion)

    def _multistart_lbfgsb(self, objective, d, n_starts, bounds, rng):
        """Multi-start L-BFGS-B optimization (seeded with Latin Hypercube Sampling) to escape local minima."""
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])

        # Finite penalty for failed evaluations (inf/NaN can break the L-BFGS-B line search)
        PENALTY = 1e10

        def safe_objective(x):
            try:
                val = float(objective(x))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return PENALTY
            return val if np.isfinite(val) else PENALTY

        sampler = qmc.LatinHypercube(d=d, seed=rng.integers(int(1e6)))
        starts = qmc.scale(sampler.random(n_starts), lb, ub)

        # 1. Pre-evaluate all starting points to find promising regions
        initial_vals = np.array([safe_objective(x0) for x0 in starts])

        # 2. Filter down to the top-performing candidates (top 50% or at least 3)
        k = max(3, n_starts // 2)
        best_indices = np.argsort(initial_vals)[:k]

        # 3. Seed the incumbent with the best raw sampled point, so the result
        #    is never worse than random search
        best_val = np.inf
        best_x = None
        best_raw_idx = best_indices[0]
        if initial_vals[best_raw_idx] < PENALTY:
            best_val = initial_vals[best_raw_idx]
            best_x = starts[best_raw_idx]

        # 4. Run L-BFGS-B on the refined candidates (skipping failed starts)
        for i in best_indices:
            if initial_vals[i] >= PENALTY:
                continue

            res = minimize(
                safe_objective,
                x0=starts[i],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
            )

            # 5. Accept any finite improvement; do not require res.success
            #    (iteration-limit / line-search exits still often improve the objective)
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        # Returns None only if every evaluation failed; callers already raise RuntimeError
        return best_x
    
    def _tune_bandwidths_optimization(self, X1, z, X2, y):
        """Sequential tuning of (eta, nugget) per level via multi-start L-BFGS-B."""
        rng = np.random.default_rng(self.seed)
        lb = -2 - np.log10(self.d)
        ub = np.log10(5000) - np.log10(self.d)  
        bounds_d = [(lb, ub)] * self.d
        nug_b = [self.nugget_bounds] if self.tune_nugget else []   # appended as the LAST coordinate

        def _warn_if_at_bound(name, log_eta, bounds):
            """Flag bandwidths sitting on (or within 2% of span of) a bound of the search box."""
            log_eta = np.atleast_1d(log_eta)
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            tol = 0.02 * (hi - lo)
            hit = (log_eta < lo + tol) | (log_eta > hi - tol)
            if hit.any():
                warnings.warn(f"{name} at search bound in dims {np.where(hit)[0].tolist()}; "
                            f"check input/output scaling.")

        def _warn_nugget_upper(name, lam):
            """Only the UPPER end is suspicious (the level looks like mostly noise); a tiny nugget is normal."""
            lo, hi = self.nugget_bounds
            if np.log10(lam) > hi - 0.02 * (hi - lo):
                warnings.warn(f"{name} at its upper bound ({lam:.2g}): this level is explained mostly as noise; "
                              f"check the model/scaling.")

        # Step 1: Optimize low-fidelity (eta1 [, nugget1])
        bounds1 = bounds_d + nug_b
        obj1 = lambda theta: self._objective_eta1(theta, X1, z)
        theta1 = self._multistart_lbfgsb(obj1, len(bounds1), self.n_starts, bounds1, rng)
        if theta1 is None:
            raise RuntimeError("Optimization for eta1 failed.")
        log_eta1_best = theta1[:self.d]
        _warn_if_at_bound("eta1", log_eta1_best, bounds_d)
        if self.tune_nugget:
            self.lam1 = 10.0 ** theta1[self.d]
            self.Sigma1_inv = self._prec1(self.lam1)         # used by everything downstream
            _warn_nugget_upper("nugget1", self.lam1)

        # Level-1 quantities are fixed during the high-fidelity search: compute once
        omega1 = scale_frequencies(self.base1, 10.0 ** log_eta1_best)
        phi1 = phi(X1, omega1)
        beta1_hat, _, _ = parameter_bayes(phi1, z, mu=self.mu1, Sigma=self.Sigma1_inv)
        phi12 = phi(X2, omega1)
        f12_raw = phi12 @ beta1_hat                      # (N2, 1) plug-in f1 at the high-fidelity inputs

        # Step 2: Optimize high-fidelity (eta2 [, nugget2])
        n_eta2 = self.base2.shape[1]
        if self.linear:
            # f1 enters linearly through rho (no bandwidth), so its scale is absorbed: no standardization
            self.f1_loc, self.f1_scale = 0.0, 1.0
            z12 = f12_raw
            bounds_eta2 = bounds_d
        else:
            # Non-linear: the kernel acts on (x, f1), so standardize f1 with constants fixed here.
            f1_ref = phi1 @ beta1_hat
            self.f1_loc = float(f1_ref.mean())
            self.f1_scale = float(f1_ref.std()) or 1.0       # guard against a constant f1
            z12 = (f12_raw - self.f1_loc) / self.f1_scale
            bounds_eta2 = bounds_d + [(-2.0, np.log10(500))]  # ARD over (x, f1)

        bounds2 = bounds_eta2 + nug_b
        obj_high = lambda theta: self._objective_eta_high(theta, phi12, beta1_hat, z12, X2, y)
        theta2 = self._multistart_lbfgsb(obj_high, len(bounds2), self.n_starts, bounds2, rng)
        if theta2 is None:
            raise RuntimeError("Optimization for eta2 failed.")
        log_eta2_best = theta2[:n_eta2]
        _warn_if_at_bound("eta2", log_eta2_best, bounds_eta2)
        if self.tune_nugget:
            self.lam2 = 10.0 ** theta2[n_eta2]
            self.Sigma2_inv = self._prec2(self.lam2)
            _warn_nugget_upper("nugget2", self.lam2)

        return 10.0 ** log_eta1_best, 10.0 ** log_eta2_best

    def _fit_level2(self, beta, z12, phi12, phi2, X2, y2):
        """Deterministic level-2 update for one beta1 draw; shared quantities are precomputed."""
        if self.linear:
            psi2 = psi(phi12, phi2, beta)
        else:
            z12_std = (z12 - self.f1_loc) / self.f1_scale
            psi2 = phi(np.hstack([X2, z12_std]), self.omega2)

        gamma2_hat, Q2_raw, L2 = parameter_bayes(psi2, y2, mu=self.mu2, Sigma=self.Sigma2_inv)

        Q2 = Q2_raw + 2 * self.b2
        sigma2 = Q2 / (self.N2 + 2 * self.a2)
        return gamma2_hat, Q2, sigma2, L2
    
    def parameters_bayes(self, X1, X2, y1, y2):
        """Fits level-1 weights (beta1) and level-2 weights (gamma2) given current bandwidth settings."""
        rng = np.random.default_rng(self.seed_fit)

        phi1 = phi(X1, self.omega1)
        # Solve for level-1 parameters and Cholesky factor L1
        beta1_hat, Q1_raw, self.L1 = parameter_bayes(phi1, y1, mu=self.mu1, Sigma=self.Sigma1_inv)
        self.beta1_hat = beta1_hat
        self.Q1 = Q1_raw + 2*self.b1

        alpha1 = self.N1 + 2*self.a1
        self.sigma1 = self.Q1/alpha1

        # Joint NIG draw: sigma^2 ~ IG(a_N, b_N), then beta | sigma^2 ~ N(beta_hat, sigma^2 K^{-1})
        S = self.samples
        a1_N = self.a1 + self.N1 / 2.0
        b1_N = self.Q1 / 2.0
        self.sigma1_samples = 1.0 / rng.gamma(shape=a1_N, scale=1.0 / b1_N, size=S)

        z = rng.standard_normal((self.p1, S))
        dev = solve_triangular(self.L1, z, lower=True, trans='T')      # Cov = L^{-T} L^{-1} = K^{-1}
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
        """Main training workflow: configures dimensions, frequencies, priors, tunes bandwidths, and fits weights."""
        X1 = np.asarray(X1);  z = np.asarray(z).reshape(-1, 1)
        X2 = np.asarray(X2);  y = np.asarray(y).reshape(-1, 1)

        if X1.ndim == 1: X1 = X1.reshape(-1, 1)  
        if X2.ndim == 1: X2 = X2.reshape(-1, 1) 

        self.N1, self.N2 = len(X1), len(X2)

        # Determine feature dimensions using heuristic if not explicitly initialized
        
        self.m1 = self._resolve_m(self.N1) if self.linear else self._resolve_m(1.5*self.N1)
        self.m2 = self._resolve_m(self.N2) if self.linear else self._resolve_m(2*self.N2)

        self.p1 = 2*self.m1
        self.p2 = 2*self.m2

        print(self.p1)
        print(self.p2)

        # Setup prior distributions parameters
        self._priors(tau=1e3)

        d2 = self.d if self.linear else self.d + 1
        self.base1 = base_frequencies(self.m1, self.d, kernel=self.kernel, seed=self.seed1)
        self.base2 = base_frequencies(self.m2, d2,      kernel=self.kernel, seed=self.seed2)

        self.eta1, self.eta2 = self._tune_bandwidths_optimization(X1, z, X2, y)

        self.omega1 = scale_frequencies(self.base1, self.eta1)
        self.omega2 = scale_frequencies(self.base2, self.eta2)
        
        # Fit parameters across model levels
        self.parameters_bayes(X1, X2, z, y)

        return self

    def predict(self, X, low_fidelity=False, return_ci=False, return_samples=False, random_state=None):
        """Computes predictions at new inputs X, returning mean, variance, and optional 95% credible intervals."""
        X = np.asarray(X)
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

        # Low-fidelity prediction path
        if low_fidelity:
            alpha1 = self.N1 + 2 * self.a1

            # Posterior mean (no Monte Carlo noise)
            y_pred = phi1_star @ self.beta1_hat

            # Analytic predictive variance of y*: Student-t scale^2 * alpha/(alpha-2) = Q1/(alpha1-2) * (1 + h)
            # (drop the "1 +" to get the latent-function variance only)
            v_mat = solve_triangular(self.L1, phi1_star.T, lower=True)
            h = np.sum(v_mat**2, axis=0).reshape(-1, 1)
            y_var = (self.Q1 / (alpha1 - 2.0)) * (1.0 + h)

            if need_samples:
                f_samples = phi1_star @ self.beta1.T                                   # (N_test, S)
                # noise uses the SAME sigma^2 that generated beta1^s (joint NIG draw)
                noise = rng.standard_normal((N_test, S)) * np.sqrt(self.sigma1_samples)[None, :]
                pred_samples_matrix = f_samples + noise

                if return_samples:
                    return y_pred, y_var, pred_samples_matrix

                ci_lower = np.quantile(pred_samples_matrix, 0.025, axis=1).reshape(-1, 1)
                ci_upper = np.quantile(pred_samples_matrix, 0.975, axis=1).reshape(-1, 1)
                return y_pred, y_var, ci_lower, ci_upper

            return y_pred

        # High-fidelity prediction path (default)
        else:
            alpha2 = self.N2 + 2 * self.a2
            a2_N = alpha2 / 2.0

            mu_s_array = np.zeros((N_test, S))
            var_s_array = np.zeros((N_test, S))
            if need_samples:
                pred_samples_matrix = np.zeros((N_test, S))

            Z_pred = phi1_star @ self.beta1.T              # (N_test, S): all f1 draws at once
            if self.linear:
                phi2_star = phi(X, self.omega2)            # independent of i


            for i in range(S):
                if self.linear:
                    psi2 = psi(phi1_star, phi2_star, self.beta1[i].reshape(-1, 1))
                else:
                    z_pred_std = (Z_pred[:, [i]] - self.f1_loc) / self.f1_scale
                    psi2 = phi(np.hstack([X, z_pred_std]), self.omega2)

                mu_s_array[:, i] = (psi2 @ self.gamma2[i]).flatten()

                v_mat2 = solve_triangular(self.L2[i], psi2.T, lower=True)
                h = np.sum(v_mat2**2, axis=0)
                var_s_array[:, i] = (self.Q2[i] / (alpha2 - 2.0)) * (1.0 + h)

                if need_samples:
                    # Joint NIG draw: sigma^2 first, then gamma | sigma^2, then the observation noise
                    s2 = 1.0 / rng.gamma(shape=a2_N, scale=2.0 / self.Q2[i])    # b_N = Q2/2
                    z = rng.standard_normal(psi2.shape[1])                       # shared across test points
                    dev = solve_triangular(self.L2[i], z, lower=True, trans='T')
                    gamma_s = self.gamma2[i].flatten() + np.sqrt(s2) * dev
                    pred_samples_matrix[:, i] = psi2 @ gamma_s + np.sqrt(s2) * rng.standard_normal(N_test)

            y_pred = np.mean(mu_s_array, axis=1).reshape(-1, 1)

            # Law of total variance over beta1^s: Var(E[y*|beta1]) + E[Var(y*|beta1)]
            varexp = np.var(mu_s_array, axis=1).reshape(-1, 1)
            expvar = np.mean(var_s_array, axis=1).reshape(-1, 1)
            y_var = varexp + expvar

            if need_samples:
                ci_lower = np.quantile(pred_samples_matrix, 0.025, axis=1).reshape(-1, 1)
                ci_upper = np.quantile(pred_samples_matrix, 0.975, axis=1).reshape(-1, 1)
                return y_pred, y_var, ci_lower, ci_upper, pred_samples_matrix

            return y_pred