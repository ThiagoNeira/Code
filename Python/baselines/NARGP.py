# NARGP.py
"""
L-level Nonlinear Autoregressive Gaussian Process (NARGP; Perdikaris et al., 2017).

  - Level 0: f_0(x) ~ GP(0, k_0(x, x'))
  - Level l >= 1: f_l(x) = g_l(x, f_{l-1}(x)), with
        k_l = k_x(x, x') * k_f(f, f') + k_delta(x, x')

Training (recursive, plug-in mean chain): level l is fit on [X_l, m_{l-1}(X_l)], where
    m_0(x) = E[f_0(x)],   m_k(x) = E[g_k(x, m_{k-1}(x))].
Inputs need not be nested across levels.

Prediction (Monte Carlo): S chains are propagated through every level. Latent
(noise-free) samples feed the next level, and the target level combines the
per-chain predictive moments via the law of total variance.
"""
import GPy
import numpy as np

_CHUNK_ROWS = 50_000


class NARGP:

    def __init__(self, n_restarts: int = 20, nsamples: int = 1000, kernel: str = "gaussian", seed=None):
        self.n_restarts = n_restarts
        self.nsamples = int(nsamples)
        self.kernel_type = kernel
        self.seed = seed

        self.models = []
        self.L = None
        self.d = None

    def _check_is_fitted(self):
        if not self.models:
            raise RuntimeError("Call fit(X_list, y_list) before predict().")

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

    def _level_kernel(self, d):
        """k_x(x) * k_f(f) + k_delta(x) on the augmented input [x, f]."""
        x_dims = list(range(d))
        f_dims = [d]
        return (
            self._get_kernel(self.kernel_type, input_dim=d, active_dims=x_dims, ARD=True) *
            self._get_kernel(self.kernel_type, input_dim=1, active_dims=f_dims, ARD=False) +
            self._get_kernel(self.kernel_type, input_dim=d, active_dims=x_dims, ARD=True)
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

    @staticmethod
    def _predict_batched(model, XX, noiseless):
        """GPy predict on a large input matrix in row chunks to bound kernel-matrix memory."""
        pred = model.predict_noiseless if noiseless else model.predict
        means, varis = [], []
        for start in range(0, XX.shape[0], _CHUNK_ROWS):
            mu, v = pred(XX[start:start + _CHUNK_ROWS])
            means.append(mu)
            varis.append(v)
        return np.vstack(means).ravel(), np.vstack(varis).ravel()

    def _plugin_mean(self, target_lvl, X):
        """Deterministic plug-in mean chain m_{target_lvl}(X), shape (N, 1)."""
        m, _ = self.models[0].predict(X)
        for k in range(1, target_lvl + 1):
            m, _ = self.models[k].predict(np.hstack([X, m]))
        return m

    def _as_2d(self, X):
        X = np.asarray(X, dtype=float)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def fit(self, X_list, y_list):
        if not isinstance(X_list, (list, tuple)) or not isinstance(y_list, (list, tuple)):
            raise TypeError("X_list and y_list must be sequences (lists or tuples) of length L.")
        if len(X_list) != len(y_list) or len(X_list) < 1:
            raise ValueError("X_list and y_list must be non-empty sequences of the same length L >= 1.")

        X_clean = [self._as_2d(X) for X in X_list]
        y_clean = [np.asarray(y, dtype=float).reshape(-1, 1) for y in y_list]
        d = X_clean[0].shape[1]
        for lvl, (Xl, yl) in enumerate(zip(X_clean, y_clean)):
            if Xl.shape[1] != d:
                raise ValueError(f"Level {lvl} input X has {Xl.shape[1]} columns; expected d={d}.")
            if Xl.shape[0] != yl.shape[0]:
                raise ValueError(f"Level {lvl} X and y have mismatched lengths: {Xl.shape[0]} vs {yl.shape[0]}.")

        self.L = len(X_clean)
        self.d = d
        self.models = []

        # Level 0 - plain GP on x
        k0 = self._get_kernel(self.kernel_type, input_dim=d, ARD=True)
        self.models.append(self._fit_gp(X_clean[0], y_clean[0], k0))

        # Levels 1..L-1 - augmented input [x, m_{l-1}(x)]
        for lvl in range(1, self.L):
            m_prev = self._plugin_mean(lvl - 1, X_clean[lvl])
            XX = np.hstack([X_clean[lvl], m_prev])
            self.models.append(self._fit_gp(XX, y_clean[lvl], self._level_kernel(d)))

        return self

    def predict(self, X, level=-1, return_ci=False, return_var=False):
        self._check_is_fitted()
        X = self._as_2d(X)
        if X.shape[1] != self.d:
            raise ValueError(f"X has {X.shape[1]} columns; expected d={self.d}.")

        lvl = int(level)
        if lvl < 0:
            lvl += self.L
        if not (0 <= lvl < self.L):
            raise ValueError(f"Requested level={level} out of bounds for L={self.L} levels.")

        N, S = X.shape[0], self.nsamples

        if lvl == 0:
            mean, var = self.models[0].predict(X)
        else:
            rng = np.random.default_rng(self.seed)
            X_rep = np.tile(X, (S, 1))                                    # (S*N, d), chain-major

            # Latent level-0 draws; pointwise draws give exact pointwise marginals
            mu0, v0 = self.models[0].predict_noiseless(X)
            Z = mu0.ravel()[None, :] + np.sqrt(np.maximum(v0.ravel(), 0.0))[None, :] * rng.standard_normal((S, N))

            # Propagate latent draws through intermediate levels 1..lvl-1
            for k in range(1, lvl):
                mu_k, v_k = self._predict_batched(self.models[k], np.hstack([X_rep, Z.reshape(-1, 1)]), noiseless=True)
                Z = (mu_k + np.sqrt(np.maximum(v_k, 0.0)) * rng.standard_normal(S * N)).reshape(S, N)

            # Target level: per-chain predictive moments (including observation noise)
            mu_t, v_t = self._predict_batched(self.models[lvl], np.hstack([X_rep, Z.reshape(-1, 1)]), noiseless=False)
            mu_t, v_t = mu_t.reshape(S, N), v_t.reshape(S, N)

            # Law of Total Variance across chains
            mean = np.mean(mu_t, axis=0)[:, None]
            varexp = np.var(mu_t, axis=0)[:, None]                         # Epistemic variance propagated from lower levels
            expvar = np.mean(v_t, axis=0)[:, None]                         # Expected target-level predictive variance
            var = varexp + expvar

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
