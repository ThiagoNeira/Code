import GPy
import numpy as np

class NARGP:

    def __init__(self, n_restarts: int = 20, nsamples: int = 1000, kernel: str = "gaussian"):
        self.n_restarts = n_restarts
        self.nsamples   = nsamples
        self.kernel_type = kernel

        self.model1 = None
        self.model2 = None

    def _check_is_fitted(self):
        if self.model1 is None or self.model2 is None:
            raise RuntimeError("Call fit() before predict().")

    def _get_kernel(self, kt, input_dim, active_dims=None, ARD=True):
        kt_lower = kt.lower()
        kwargs = {"input_dim": input_dim, "ARD": ARD}
        if active_dims is not None:
            kwargs["active_dims"] = active_dims
            
        if kt_lower == "gaussian":
            return GPy.kern.RBF(**kwargs)
        elif kt_lower == "matern32":
            return GPy.kern.Matern32(**kwargs)
        elif kt_lower == "matern52":
            return GPy.kern.Matern52(**kwargs)
        else:
            raise ValueError(
                f"Unsupported kernel_type: '{kt}'. "
                "Choose from 'gaussian', 'matern32', or 'matern52'."
            )

    def _fit_gp(self, X, y, kernel):
        model = GPy.models.GPRegression(X, y, kernel)
        model[".*Gaussian_noise"] = model.Y.var() * 0.01
        model[".*Gaussian_noise"].fix()
        model.optimize(max_iters=500)
        model[".*Gaussian_noise"].unfix()
        model[".*Gaussian_noise"].constrain_positive()
        model.optimize_restarts(self.n_restarts, optimizer="bfgs",
                                max_iters=1000, verbose=False, robust=True)
        return model

    def fit(self, X1, z, X2, y):
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        
        # Force 2D arrays for GPy[cite: 2]
        if X1.ndim == 1: X1 = X1.reshape(-1, 1)
        if X2.ndim == 1: X2 = X2.reshape(-1, 1)
            
        z  = np.asarray(z).reshape(-1, 1)
        y  = np.asarray(y).reshape(-1, 1)

        d = X1.shape[1]

        # Level 1 - low-fidelity GP[cite: 2]
        k1 = self._get_kernel(self.kernel_type, input_dim=d, ARD=True) 
        self.model1 = self._fit_gp(X1, z, k1)

        # Level 2 - augmented input[cite: 2]
        mu1, _ = self.model1.predict(X2)
        XX = np.hstack([X2, mu1])

        x_dims  = list(range(d))                
        f1_dims = [d]   

        k2 = (
            self._get_kernel(self.kernel_type, input_dim=d, active_dims=x_dims, ARD=True) * 
            self._get_kernel(self.kernel_type, input_dim=1, active_dims=f1_dims, ARD=False) + 
            self._get_kernel(self.kernel_type, input_dim=d, active_dims=x_dims, ARD=True)
        )

        self.model2 = self._fit_gp(XX, y, k2)
        return self

    def predict(self, X, return_ci=False):
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        N = X.shape[0]

        mu1, C1 = self.model1.predict(X, full_cov=True)
        C1 = C1 + 1e-9 * np.eye(N)
        Z = np.random.multivariate_normal(mu1.flatten(), C1, self.nsamples)

        tmp_m = np.zeros((self.nsamples, N))
        tmp_v = np.zeros((self.nsamples, N))
        for i in range(self.nsamples):
            mu, v = self.model2.predict(np.hstack([X, Z[i, :, None]]))
            tmp_m[i, :] = mu.flatten()
            tmp_v[i, :] = v.flatten()

        mean = np.mean(tmp_m, axis=0)[:, None]

        # Law of Total Variance for Monte Carlo propagation:[cite: 2]
        varexp = np.var(tmp_m, axis=0)[:, None]     # Epistemic variance across samples[cite: 2]
        expvar = np.mean(tmp_v, axis=0)[:, None]    # Expected model 2 predictive variance[cite: 2]
        var = varexp + expvar

        if return_ci:
            std = np.sqrt(var)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            return mean, var, lower, upper

        return mean

    def predict_low(self, X, return_ci=False):
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1: X = X.reshape(-1, 1)
        
        mean, var = self.model1.predict(X)
        if return_ci:
            std = np.sqrt(var)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            return mean, var, lower, upper
        return mean