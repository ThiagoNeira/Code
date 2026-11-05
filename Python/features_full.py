# features.py
import warnings
import numpy as np
from scipy.stats import qmc, norm, gamma as gamma_dist
 
_MATERN_NU = {'matern32': 1.5, 'matern52': 2.5}
 
def _orthogonal_gaussian(m, d, rng):
    """Orthogonal random features (Yu et al. 2016): Haar-orthogonal blocks with chi(d) row norms."""
    blocks = []
    for _ in range(int(np.ceil(m / d))):
        G = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(G)
        Q = Q * np.sign(np.diag(R))[None, :]
        norms = np.sqrt(rng.chisquare(d, size=d))
        blocks.append(norms[:, None] * Q)
    return np.vstack(blocks)[:m]
 
def base_frequencies(m, d, kernel='gaussian', seed=None, method='mc'):
    """
    Unit-scale spectral samples.  method: 'mc' (iid), 'sobol' (scrambled QMC, inverse-CDF), 'orthogonal'.
    All three give the same marginal law; 'sobol'/'orthogonal' lower the variance of the kernel approximation.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    kernel = kernel.lower()
    if kernel != 'gaussian' and kernel not in _MATERN_NU:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'gaussian', 'matern32' or 'matern52'.")
    nu = _MATERN_NU.get(kernel)
    method = method.lower()
 
    if method == 'mc':
        Z = rng.standard_normal((m, d))
        g = rng.gamma(shape=nu, scale=1.0, size=m) if nu else None
    elif method == 'sobol':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')   # non-power-of-2 balance warning
            U = qmc.Sobol(d=d + (1 if nu else 0), scramble=True, seed=rng).random(m)
        U = np.clip(U, 1e-12, 1 - 1e-12)
        Z = norm.ppf(U[:, :d])
        g = gamma_dist.ppf(U[:, d], a=nu) if nu else None
    elif method in ('orthogonal', 'orf'):
        Z = _orthogonal_gaussian(m, d, rng)
        g = rng.gamma(shape=nu, scale=1.0, size=m) if nu else None
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'mc', 'sobol' or 'orthogonal'.")
    if nu is None:
        return Z
    return np.sqrt(nu / g)[:, None] * Z      # multivariate Student-t, 2*nu dof
 
def scale_frequencies(base, eta):
    d = base.shape[1]
    eta = np.atleast_1d(eta)
    if eta.size == 1:
        eta = np.full(d, eta.item())
    elif eta.size != d:
        raise ValueError(f"eta must be a scalar or have length {d}, got {eta.size}.")
    return base * np.sqrt(2.0 * eta)
 
def sample_frequencies(m, d, kernel='gaussian', eta=1, seed=None, method='mc'):   # wrapper
    return scale_frequencies(base_frequencies(m, d, kernel=kernel, seed=seed, method=method), eta)
 
def phi(X, omega):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    w = np.asarray(omega, dtype=float)
    m = w.shape[0]
    projection = X @ w.T
    return np.sqrt(1.0 / m) * np.hstack([np.cos(projection), np.sin(projection)])

def psi(phi1, phi2, beta):
    z12 = (phi1 @ beta).reshape(-1, 1)
    return np.hstack([phi2, z12])