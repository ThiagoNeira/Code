import numpy as np
import GPy


class GP:
    def __init__(self, n_restarts=10, kernel="gaussian"):
        self.model = None
        self.n_restarts = n_restarts
        self.kernel = kernel.lower()
        
    def _check_is_fitted(self):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
            
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        d = X.shape[1]
        
        # Setup Kernels based on user choice
        if self.kernel == "gaussian":
            kernel = GPy.kern.RBF(input_dim=d, ARD=True)
        elif self.kernel == "matern32":
            kernel = GPy.kern.Matern32(input_dim=d, ARD=True)
        elif self.kernel == "matern52":
            kernel = GPy.kern.Matern52(input_dim=d, ARD=True)
        else:
            raise ValueError(
                f"Unsupported kernel: '{self.kernel}'. "
                "Choose from 'gaussian', 'matern32', or 'matern52'."
            )
    
        # Model Construction
        self.model = GPy.models.GPRegression(X, y, kernel)
    
        # Constraints
        self.model.kern.lengthscale.constrain_bounded(1e-3, 10.0)
        self.model.kern.variance.constrain_bounded(1e-3, 10.0)
        self.model.Gaussian_noise.variance.constrain_bounded(1e-6, 1e-1)

        # Warm start
        self.model.kern.lengthscale = 0.1
        self.model.kern.variance = 1.0
    
        self.model.optimize_restarts(self.n_restarts, verbose=False, robust=True)
        return self
    
    def predict(self, X, return_ci=False, return_var=False):
        self._check_is_fitted()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        mean, var = self.model.predict(X)

        if return_ci:
            std = np.sqrt(var)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            return mean, var, lower, upper
            
        if return_var:
            return mean, var
        
        return mean