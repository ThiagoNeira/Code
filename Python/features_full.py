# features.py
import numpy as np

_MATERN_NU = {'matern32': 1.5, 'matern52': 2.5}

def base_frequencies(m, d, kernel='gaussian', seed=None):
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    kernel = kernel.lower()

    Z = rng.standard_normal((m, d))
    if kernel == 'gaussian':
        return Z
    if kernel in _MATERN_NU:
        nu = _MATERN_NU[kernel]
        g = rng.gamma(shape=nu, scale=1.0, size=m)
        return np.sqrt(nu / g)[:, None] * Z
    raise ValueError(...)

def scale_frequencies(base, eta):
    d = base.shape[1]
    eta = np.atleast_1d(eta)
    if eta.size == 1:
        eta = np.full(d, eta.item())
    elif eta.size != d:
        raise ValueError(...)
    return base * np.sqrt(2.0 * eta)

def sample_frequencies(m, d, kernel='gaussian', eta=1, seed=None):   # wrapper
    return scale_frequencies(base_frequencies(m, d, kernel=kernel, seed=seed), eta)

def phi(X,omega):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    w = np.asarray(omega)
    m = w.shape[0]
    
    projection = X @ w.T
    cos_part = np.cos(projection)
    sin_part = np.sin(projection)

    features = np.hstack([cos_part, sin_part])
    return np.sqrt(1/m) * features 

def psi(phi1, phi2, beta):

    z12 = (phi1 @ beta).reshape(-1, 1)

    return np.hstack([phi2, z12])