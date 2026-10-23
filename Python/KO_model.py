import numpy as np
import GPy
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers import GPyMultiOutputWrapper

class KO:
    def __init__(self, n_restarts=5, kernel="gaussian"):
        self.model = None
        self.n_restarts = n_restarts
        self.kernel_type = kernel
        
    def _check_is_fitted(self):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
            
    def fit(self, X1, z, X2, y):
        X1 = np.asarray(X1)
        X1 = np.atleast_2d(X1).T if X1.ndim == 1 else X1
        z  = np.asarray(z).reshape(-1, 1)
        X2 = np.asarray(X2)
        X2 = np.atleast_2d(X2).T if X2.ndim == 1 else X2
        y  = np.asarray(y).reshape(-1, 1)

        d = X1.shape[1]
        
        X1_aug = np.hstack([X1, np.zeros((X1.shape[0], 1))])
        X2_aug = np.hstack([X2, np.ones((X2.shape[0], 1))]) 
        X_train = np.vstack([X1_aug, X2_aug])
        Y_train = np.vstack([z, y])

        # 2. Setup Kernels dynamically based on kernel_type
        def _get_kernel(kt, input_dim):
            kt_lower = kt.lower()
            if kt_lower == "gaussian":
                return GPy.kern.RBF(input_dim=input_dim, ARD=True)
            elif kt_lower == "matern32":
                return GPy.kern.Matern32(input_dim=input_dim, ARD=True)
            elif kt_lower == "matern52":
                return GPy.kern.Matern52(input_dim=input_dim, ARD=True)
            else:
                raise ValueError(
                    f"Unsupported kernel_type: '{kt}'. "
                    "Choose from 'gaussian', 'matern32', or 'matern52'."
                )

        kern_low = _get_kernel(self.kernel_type, d)    
        kern_err = _get_kernel(self.kernel_type, d)

        kernels = [kern_low, kern_err]
        multi_fidelity_kernel = LinearMultiFidelityKernel(kernels)
    
        # 3. Model Construction
        gpy_model = GPyLinearMultiFidelityModel(
            X_train, Y_train, multi_fidelity_kernel, n_fidelities=2
        )
        
        # Constrain lengthscales and variances to avoid overflow
        gpy_model.kern.kernels[0].lengthscale.constrain_bounded(1e-3, 10.0)
        gpy_model.kern.kernels[1].lengthscale.constrain_bounded(1e-3, 10.0)
        gpy_model.kern.kernels[0].variance.constrain_bounded(1e-3, 10.0)
        gpy_model.kern.kernels[1].variance.constrain_bounded(1e-3, 10.0)
        gpy_model.likelihood.likelihoods_list[0].variance.constrain_bounded(1e-6, 1.0)
        gpy_model.likelihood.likelihoods_list[1].variance.constrain_bounded(1e-6, 1.0)
    
        # Warm starts
        gpy_model.kern.kernels[0].lengthscale = 0.1
        gpy_model.kern.kernels[1].lengthscale = 0.1
        gpy_model.kern.kernels[0].variance    = 1.0
        gpy_model.kern.kernels[1].variance    = 1.0
    
        self.model = GPyMultiOutputWrapper(
            gpy_model, n_outputs=2, n_optimization_restarts=self.n_restarts,
            verbose_optimization=False
        )
        self.model.optimize()
        return self
            
    def predict(self, X, return_ci=False, return_var=False):
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # High fidelity indicator (1) in the last column
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        
        # Provide Y_metadata so the mixed-noise likelihood knows this is fidelity level 1 (high fidelity)
        Y_metadata = {'output_index': np.ones((X.shape[0], 1), dtype=int)}

        mean, var = self.model.gpy_model.predict(X_aug, Y_metadata=Y_metadata)

        if return_ci:
            std = np.sqrt(var)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            return mean, var, lower, upper
            
        if return_var:
            return mean, var
        
        return mean
    
    def predict_low(self, X, return_ci=False, return_var=False):
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Low fidelity indicator (0) in the last column
        X_aug = np.hstack([X, np.zeros((X.shape[0], 1))])
        
        # Provide Y_metadata for fidelity level 0 (low fidelity)
        Y_metadata = {'output_index': np.zeros((X.shape[0], 1), dtype=int)}

        mean, var = self.model.gpy_model.predict(X_aug, Y_metadata=Y_metadata)

        if return_ci:
            std = np.sqrt(var)
            lower = mean - 1.96 * std
            upper = mean + 1.96 * std
            return mean, var, lower, upper
            
        if return_var:
            return mean, var
        
        return mean