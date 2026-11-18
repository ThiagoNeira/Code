# KO_model.py
"""
L-level Kennedy-O'Hagan (AR1) multi-fidelity co-kriging via emukit.

  f_0(x) = delta_0(x)
  f_l(x) = rho_{l-1} * f_{l-1}(x) + delta_l(x),   l = 1, ..., L-1

Fit as one joint GP over all levels (fidelity index in the last input column),
with independent Gaussian noise per level. Predictions are the exact Gaussian
posterior at the requested level; no sampling is involved. Inputs need not be
nested across levels.
"""
import numpy as np
import GPy
from emukit.multi_fidelity.kernels.linear_multi_fidelity_kernel import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.model_wrappers import GPyMultiOutputWrapper


class KO:
    def __init__(self, n_restarts=5, kernel="gaussian", seed=None):
        self.model = None
        self.n_restarts = n_restarts
        self.kernel_type = kernel
        self.seed = seed
        self.L = None
        self.d = None

    def _check_is_fitted(self):
        if self.model is None:
            raise RuntimeError("Call fit(X_list, y_list) before predict().")

    @staticmethod
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

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X)
        return np.atleast_2d(X).T if X.ndim == 1 else X

    def fit(self, X_list, y_list):
        if not isinstance(X_list, (list, tuple)) or not isinstance(y_list, (list, tuple)):
            raise TypeError("X_list and y_list must be sequences (lists or tuples) of length L.")
        if len(X_list) != len(y_list) or len(X_list) < 1:
            raise ValueError("X_list and y_list must be non-empty sequences of the same length L >= 1.")

        if self.seed is None:
            return self._fit(X_list, y_list)

        # GPy restarts draw from the global np.random; seed it for this fit only.
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            return self._fit(X_list, y_list)
        finally:
            np.random.set_state(state)

    def _fit(self, X_list, y_list):
        X_clean = [self._as_2d(X) for X in X_list]
        y_clean = [np.asarray(y).reshape(-1, 1) for y in y_list]

        d = X_clean[0].shape[1]
        for lvl, (Xl, yl) in enumerate(zip(X_clean, y_clean)):
            if Xl.shape[1] != d:
                raise ValueError(f"Level {lvl} input X has {Xl.shape[1]} columns; expected d={d}.")
            if Xl.shape[0] != yl.shape[0]:
                raise ValueError(f"Level {lvl} X and y have mismatched lengths: {Xl.shape[0]} vs {yl.shape[0]}.")

        L = len(X_clean)

        # 1. Stack all levels with the fidelity index in the last column
        X_train = np.vstack([np.hstack([Xl, np.full((Xl.shape[0], 1), float(lvl))])
                             for lvl, Xl in enumerate(X_clean)])
        Y_train = np.vstack(y_clean)

        # 2. One kernel per level: kernels[0] models f_0, kernels[l >= 1] the discrepancy delta_l
        kernels = [self._get_kernel(self.kernel_type, d) for _ in range(L)]
        multi_fidelity_kernel = LinearMultiFidelityKernel(kernels)

        # 3. Model Construction
        gpy_model = GPyLinearMultiFidelityModel(
            X_train, Y_train, multi_fidelity_kernel, n_fidelities=L
        )

        # Constrain lengthscales and variances to avoid overflow
        # (grouped by parameter type, in the same order as the original 2-level code)
        for k in gpy_model.kern.kernels:
            k.lengthscale.constrain_bounded(1e-3, 10.0)
        for k in gpy_model.kern.kernels:
            k.variance.constrain_bounded(1e-3, 10.0)
        for lik in gpy_model.likelihood.likelihoods_list:
            lik.variance.constrain_bounded(1e-6, 1.0)

        # Warm starts
        for k in gpy_model.kern.kernels:
            k.lengthscale = 0.1
        for k in gpy_model.kern.kernels:
            k.variance = 1.0

        self.model = GPyMultiOutputWrapper(
            gpy_model, n_outputs=L, n_optimization_restarts=self.n_restarts,
            verbose_optimization=False
        )
        self.model.optimize()
        self.L = L
        self.d = d
        return self

    def predict(self, X, level=-1, return_ci=False, return_var=False):
        self._check_is_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.d:
            raise ValueError(f"X has {X.shape[1]} columns; expected d={self.d}.")

        lvl = int(level)
        if lvl < 0:
            lvl += self.L
        if not (0 <= lvl < self.L):
            raise ValueError(f"Requested level={level} out of bounds for L={self.L} levels.")

        # Fidelity indicator in the last column
        X_aug = np.hstack([X, np.full((X.shape[0], 1), float(lvl))])

        # Y_metadata so the mixed-noise likelihood uses this level's noise variance
        Y_metadata = {'output_index': np.full((X.shape[0], 1), lvl, dtype=int)}

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
        """Prediction at the lowest fidelity level (level 0)."""
        return self.predict(X, level=0, return_ci=return_ci, return_var=return_var)
