import numpy as np
from tqdm import tqdm
from scipy.stats import qmc, multivariate_t, invgamma
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, block_diag, cho_solve
from features_full import sample_frequencies, phi, psi
from rff_linear_algebra import parameter_bayes, score_bayes
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

class MF_RFF_bayes:
    """
    Multi-Fidelity Bayesian Random Fourier Features (MF-RFF) regression model.
    Combines low-fidelity (cheap) and high-fidelity (expensive) data sources 
    using a fully Bayesian treatment of random feature weights.
    """
    def __init__(self,
                 d, 
                 *,
                 # tuning grid
                 kernel = 'gaussian',
                 nugget1 = 1e-6,
                 nugget2 = 1e-6,
                 number_etas_grid = 100,
                 n_restarts = 10,
                 n_samples = 500,
                 criterion = 'nlpd',
                 linear = False,
                 seed=None
        ):
         
        self.d = d
        self.kernel = kernel
        self.nugget1 = nugget1
        self.nugget2 = nugget2
        self.n_starts = n_restarts
        self.samples = n_samples
        self.number_etas_grid = number_etas_grid
        self.criterion = criterion
        self.linear = linear 
        self.seed = seed
        s1, s2, s3 = [int(c.generate_state(1)[0])
              for c in np.random.SeedSequence(seed).spawn(3)]
        self.seed1, self.seed2, self.seed3 = s1, s2, s3
        # Placeholders for data sizes, dimensions, priors, coefficients, and tracking variables
        self.N1 = self.N2 = None
        self.m1 = self.m2 = self.m3 = None
        self.p1 = self.p2 = self.p3 = None
        self.mu1 = self.mu2 = None
        self.Sigma1_inv = self.Sigma2_inv = None
        self.a1 = self.a2 = self.b1 = self.b2 = None
        self.Q1 = self.Q2 = None
        self.sigma1 = self.sigma2 = None
        self.omega1 = self.omega2 = self.omega3 = None
        self.eta1 = self.eta2  = self.eta3 = None
        self.beta1 = self.gamma2 = None
        self.phi1 = self.phi2 = self.phi12 = self.phi32 = None
        self.L1 = self.L2 = None
      
    def _resolve_m(self, N):
        return min(int(0.75*np.sqrt(N) * np.log(N)), int(0.5*N)-1)
    
        
    
    def _build_grids(self):
        """Generates a hyperparameter search grid for bandwidths (eta) using Latin Hypercube Sampling."""
        d = self.d
        lb = -2 - np.log10(d)
        ub = np.log10(500) - np.log10(d)

        if d == 1:
            return np.logspace(lb, ub, self.number_etas_grid)

        else:
            sampler = qmc.LatinHypercube(d=d, seed=self.seed)
            sample = sampler.random(n=self.number_etas_grid)
            boundary  = np.array([np.ones(d), np.zeros(d)])
            beta_vec = np.vstack([sample, boundary])
            eta_grid = 10 ** ((ub - lb)* beta_vec + lb)

            return eta_grid

    def _priors(self, tau):
        """Sets up prior precision matrices and shape/scale hyperparameters for low and high fidelity levels."""
        self.Sigma1_inv  = self.nugget1*np.eye(self.p1)
        self.mu1 = np.zeros(self.p1).reshape(-1,1)

        Sigma_inv22 = self.nugget2* np.eye(self.p2)
        Sigma_inv23 = (1/tau)*np.eye(self.p3) 

        self.mu2 = np.zeros(self.p2+self.p3).reshape(-1,1)
        self.Sigma2_inv = block_diag(Sigma_inv22, Sigma_inv23)

        self.a1 = self.a2 = self.b1 = self.b2 = 1e-6

    def _basis(self, X1, X2):
        self.phi1 = phi(X1, self.omega1)
        self.phi12 = phi(X2, self.omega1)
        self.phi2 = phi(X2, self.omega2)
        self.phi32 = np.ones((X2.shape[0], 1)) if self.linear else phi(X2, self.omega3)

    def _objective_eta1(self, log_eta, X1, z):
        """Generates omega1 dynamically inside optimization for the current eta1."""
        eta1 = 10.0 ** log_eta
        omega1 = sample_frequencies(self.m1, self.d, kernel=self.kernel, eta=eta1, seed=self.seed1)
        phi1 = phi(X1, omega1)
        return score_bayes(phi1, z, mu=self.mu1, Sigma=self.Sigma1_inv, a=self.a1, b=self.b1, criterion=self.criterion)

    def _objective_eta_high(self, log_eta_high, log_eta1_best, X1, X2, z, y):
        """Generates omega1, omega2, omega3 dynamically inside optimization."""
        eta1 = 10.0 ** log_eta1_best
        omega1 = sample_frequencies(self.m1, self.d, kernel=self.kernel, eta=eta1, seed=self.seed1)
        phi1 = phi(X1, omega1)
        beta1, _, _ = parameter_bayes(phi1, z, mu=self.mu1, Sigma=self.Sigma1_inv)

        phi12 = phi(X2, omega1)
        if self.linear:
            eta2 = 10.0 ** log_eta_high[:self.d]
            omega2 = sample_frequencies(self.m2, self.d, kernel=self.kernel, eta=eta2, seed=self.seed2)
            phi2 = phi(X2, omega2)
            psi2 = psi(phi12, phi2, 1, beta1)
        else:
            eta2 = 10.0 ** log_eta_high[:self.d]
            eta3 = 10.0 ** log_eta_high[self.d:]
            omega2 = sample_frequencies(self.m2, self.d, kernel=self.kernel, eta=eta2, seed=self.seed2)
            omega3 = sample_frequencies(self.m3, self.d, kernel=self.kernel, eta=eta3, seed=self.seed3)
            phi2 = phi(X2, omega2)
            phi3 = phi(X2, omega3)
            psi2 = psi(phi12, phi2, phi3, beta1)

        return score_bayes(psi2, y, mu=self.mu2, Sigma=self.Sigma2_inv, a=self.a2, b=self.b2, criterion=self.criterion)

    def _multistart_lbfgsb(self, objective, d, n_starts, bounds, rng):
        """Multi-start L-BFGS-B optimization (seeded with Latin Hypercube Sampling) to escape local minima."""
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])

        sampler = qmc.LatinHypercube(d=d, seed=rng.integers(1e6))
        starts = qmc.scale(sampler.random(n_starts), lb, ub)

        # 1. Pre-evaluate all starting points to find promising regions
        initial_vals = np.full(n_starts, np.inf)
        for i, x0 in enumerate(starts):
            try:
                initial_vals[i] = objective(x0)
            except np.linalg.LinAlgError:
                continue

        # 2. Filter down to the top-performing candidates (top 50% or at least 3)
        k = max(3, n_starts // 2)
        best_indices = np.argsort(initial_vals)[:k]
        refined_starts = starts[best_indices]

        best_val = np.inf
        best_x = None

        # 3. Run L-BFGS-B gradient-based optimization on the refined candidates
        for x0 in refined_starts:
            if np.isinf(objective(x0)):
                continue
                    
            try:
                res = minimize(
                        objective, 
                        x0=x0, 
                        method='L-BFGS-B', 
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-6, 'disp': False}
                    )
                    
                # 4. Strictly require solver success before accepting the minimum
                if res.success and res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
                        
            except np.linalg.LinAlgError:
                continue

        # 5. Fallback mechanism: if all gradient optimizations fail, return the best raw sampled point
        if best_x is None:
            best_raw_idx = best_indices[0]
            if not np.isinf(initial_vals[best_raw_idx]):
                return starts[best_raw_idx]

        return best_x
    
    def _tune_bandwidths_optimization(self, X1, z, X2, y):
        """Sequential bandwidth tuning via multi-start L-BFGS-B optimization."""
        rng = np.random.default_rng(self.seed)
        lb = -2 - np.log10(self.d)
        ub = np.log10(500) - np.log10(self.d)
        bounds_d = [(lb, ub)] * self.d

        # Step 1: Optimize low-fidelity bandwidth (eta1)
        obj1 = lambda log_eta: self._objective_eta1(log_eta, X1, z)
        log_eta1_best = self._multistart_lbfgsb(obj1, self.d, self.n_starts, bounds_d, rng)
        if log_eta1_best is None:
            raise RuntimeError("Optimization for eta1 failed.")

        # Step 2: Optimize high-fidelity bandwidths (eta2 and optionally eta3)
        if self.linear:
            obj_high = lambda log_eta: self._objective_eta_high(log_eta, log_eta1_best, X1, X2, z, y)
            log_eta2_best = self._multistart_lbfgsb(obj_high, self.d, self.n_starts, bounds_d, rng)
            if log_eta2_best is None:
                raise RuntimeError("Optimization for eta2 failed.")
            return 10.0 ** log_eta1_best, 10.0 ** log_eta2_best, 10.0 ** log_eta2_best
        else:
            obj_high = lambda x: self._objective_eta_high(x, log_eta1_best, X1, X2, z, y)
            bounds_2d = bounds_d * 2
            log_eta23_best = self._multistart_lbfgsb(obj_high, 2 * self.d, self.n_starts, bounds_2d, rng)
            if log_eta23_best is None:
                raise RuntimeError("Optimization for eta2 and eta3 failed.")
            
            return 10.0 ** log_eta1_best, 10.0 ** log_eta23_best[:self.d], 10.0 ** log_eta23_best[self.d:]

    def _fit_level2(self, beta, X2, y2):
        """Helper method for parallel level-2 posterior updates."""
        beta = beta.reshape(-1, 1)
        psi2 = psi(self.phi12, self.phi2, self.phi32, beta)
        gamma2_hat, Q2_raw, L2 = parameter_bayes(psi2, y2, mu=self.mu2, Sigma=self.Sigma2_inv)

        K2inv = cho_solve((L2, True), np.eye(L2.shape[0]))
        Q2 = Q2_raw + 2 * self.b2
        alpha2 = self.N2 + 2 * self.a2
        sigma2 = Q2 / alpha2

        gamma2_sample = multivariate_t.rvs(
                loc=gamma2_hat.flatten(), 
                shape=sigma2 * K2inv, 
                df=alpha2, 
                size=1
            ).reshape(-1, 1)
            
        return gamma2_sample, Q2, sigma2, L2
    
    def parameters_bayes(self, X1, X2, y1, y2):
        """Fits level-1 weights (beta1) and level-2 weights (gamma2) given current bandwidth settings."""

        # Solve for level-1 parameters and Cholesky factor L1
        beta1_hat, Q1_raw, self.L1 =  parameter_bayes(self.phi1, y1, mu=self.mu1, Sigma=self.Sigma1_inv) 
        self.Q1 =  Q1_raw + 2*self.b1 

        alpha1 = self.N1 + 2*self.a1
        self.sigma1 = self.Q1/alpha1

        # Use solve_triangular for numerical stability instead of explicit matrix inversion
        K1inv = cho_solve((self.L1, True), np.eye(self.L1.shape[0]))

        # Draw posterior samples for level-1 weights (beta1) from a Student-t distribution
        self.beta1 = multivariate_t.rvs(loc=beta1_hat.flatten(), shape=self.sigma1*K1inv, df=alpha1, size=self.samples)

        self.gamma2 = []
        self.Q2 = []
        self.sigma2 = []
        self.L2 = []

        # PARALLELIZATION: Map level-2 updates across cores instead of running sequentially
        
        with threadpool_limits(limits=1, user_api='blas'):
            results = Parallel(n_jobs=-1, backend="threading")(
                delayed(self._fit_level2)(beta, X2, y2) for beta in self.beta1
            )
        
        self.gamma2, self.Q2, self.sigma2, self.L2 = zip(*results)
        self.gamma2 = list(self.gamma2)
        self.Q2 = list(self.Q2)
        self.sigma2 = list(self.sigma2)
        self.L2 = list(self.L2)

    def fit(self, X1, z, X2, y):
        """Main training workflow: configures dimensions, frequencies, priors, tunes bandwidths, and fits weights."""
        X1 = np.asarray(X1);  z = np.asarray(z).reshape(-1, 1)
        X2 = np.asarray(X2);  y = np.asarray(y).reshape(-1, 1)

        if X1.ndim == 1: X1 = X1.reshape(-1, 1)  
        if X2.ndim == 1: X2 = X2.reshape(-1, 1) 

        self.N1, self.N2 = len(X1), len(X2)

        # Determine feature dimensions using heuristic if not explicitly initialized
        self.m1 = self._resolve_m(self.N1)
        self.m2 = self._resolve_m(self.N2)

        if self.linear:
            self.m3 = 1
        else:
            self.m3 = self._resolve_m(0.5*self.N2)

        self.p1 = 2*self.m1
        self.p2 = 2*self.m2
        self.p3 = self.m3 if self.linear else 2*self.m3 

        print(self.p1)
        print(self.p2)
        print(self.p3)
    
        # Setup prior distributions parameters
        self._priors(tau=1e3)

        self.eta1, self.eta2, self.eta3 = self._tune_bandwidths_optimization(X1, z, X2, y)

        self.omega1 = sample_frequencies(self.m1, self.d, kernel=self.kernel, eta=self.eta1, seed=self.seed1)
        self.omega2 = sample_frequencies(self.m2, self.d, kernel=self.kernel, eta=self.eta2, seed=self.seed2)
        self.omega3 = None if self.linear else sample_frequencies(self.m3, self.d, kernel=self.kernel, eta=self.eta3, seed=self.seed3)

        self._basis(X1, X2)
            
        # Fit parameters across model levels
        self.parameters_bayes(X1, X2, z, y)

        # Validate estimated noise variances
        if self.sigma1 <= 0 or any(s <= 0 for s in self.sigma2):
            raise ValueError("Estimated noise variance is non-positive. Consider increasing the nugget or checking the data.")
             
        return self

    def predict(self, X, low_fidelity=False, return_ci=False, return_samples=False):
        """Computes predictions at new inputs X, returning mean, variance, and optional 95% credible intervals."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        S = self.beta1.shape[0]
        N_test = X.shape[0]

        phi1_star = phi(X, self.omega1)
        # Low-fidelity prediction path
        if low_fidelity:
            y_pred_samples = phi1_star @ self.beta1.T 
            y_pred = np.mean(y_pred_samples, axis=1).reshape(-1, 1)

            varexp = np.var(y_pred_samples, axis=1).reshape(-1, 1)
                
            # Compute expected observation/epistemic variance for level 1
            v_mat = solve_triangular(self.L1, phi1_star.T, lower=True)
            phi_var_term = np.sum(v_mat**2, axis=0).reshape(-1, 1)
            expvar = self.sigma1 * phi_var_term
                
            y_var = varexp + expvar

            if return_ci or return_samples:
                a1_N = self.a1 + (self.N1 / 2.0)
                b1_N = self.Q1 / 2.0
                sigma2_1_samples = invgamma.rvs(a=a1_N, scale=b1_N, size=S)
                noise_matrix = np.random.normal(0, 1, size=(N_test, S)) * np.sqrt(sigma2_1_samples)
                
                # Full predictive distribution (Epistemic + Aleatoric)
                pred_samples_matrix = y_pred_samples + noise_matrix
                
                if return_samples:
                    return y_pred, y_var, pred_samples_matrix
                
                ci_lower = np.quantile(pred_samples_matrix, 0.025, axis=1).reshape(-1, 1)
                ci_upper = np.quantile(pred_samples_matrix, 0.975, axis=1).reshape(-1, 1)
                return y_pred, y_var, ci_lower, ci_upper
        
        # High-fidelity prediction path (default)
        else:
            mu_s_array = np.zeros((N_test, S))
            expvar_array = np.zeros((N_test, S))

            phi2_star = phi(X, self.omega2)
            phi3_star = np.ones((X.shape[0], 1)) if self.linear else phi(X, self.omega3)

            for i, beta1 in enumerate(self.beta1):

                beta1 = beta1.reshape(-1, 1)
                psi2 = psi(phi1_star, phi2_star, phi3_star, beta1)
                prediction = psi2 @ self.gamma2[i].reshape(-1, 1)
                mu_s_array[:, i] = prediction.flatten()
                v_mat2 = solve_triangular(self.L2[i], psi2.T, lower=True)
                expvar_array[:, i] = (self.sigma2[i] * np.sum(v_mat2**2, axis=0)).flatten()

            y_pred = np.mean(mu_s_array, axis=1).reshape(-1, 1)
                
            # Compute variance for standard output
            varexp = np.var(mu_s_array, axis=1).reshape(-1, 1)
            expvar = np.mean(expvar_array, axis=1).reshape(-1, 1)
            y_var = varexp + expvar

            if return_ci or return_samples:
                a2_N = self.a2 + (self.N2 / 2.0)
                b2_N_array = np.array(self.Q2).flatten() / 2.0
                
                sigma2_2_samples = invgamma.rvs(a=a2_N, scale=b2_N_array)
                noise_matrix = np.random.normal(0, 1, size=(N_test, S)) * np.sqrt(sigma2_2_samples)
                
                # Full predictive distribution (Must include noise_matrix for accurate CRPS)
                pred_samples_matrix = mu_s_array + noise_matrix 
                
                ci_lower = np.quantile(pred_samples_matrix, 0.025, axis=1).reshape(-1, 1)
                ci_upper = np.quantile(pred_samples_matrix, 0.975, axis=1).reshape(-1, 1)
                return y_pred, y_var, ci_lower, ci_upper, pred_samples_matrix
                
            return y_pred