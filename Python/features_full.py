# features_full.py
"""
Spectral frequency sampling and trigonometric Random Fourier Feature (RFF) maps
for shift-invariant kernels (Gaussian/RBF, Matérn-3/2, Matérn-5/2).
"""
import warnings
import numpy as np
from scipy.stats import qmc, norm, gamma as gamma_dist

_MATERN_NU = {'matern32': 1.5, 'matern52': 2.5}


def _orthogonal_gaussian(m, d, rng):
    """
    Generates Orthogonal Random Features (Yu et al., 2016) of shape (m, d).
    Constructs Haar-distributed random orthogonal blocks of size (d, d) via QR
    decomposition with sign correction, scaled row-wise by chi(d) norms.
    """
    n_blocks = int(np.ceil(m / d))
    blocks = []
    for _ in range(n_blocks):
        G = rng.standard_normal((d, d))
        Q, R = np.linalg.qr(G)
        signs = np.sign(np.diag(R))
        signs[signs == 0.0] = 1.0
        Q *= signs[None, :]
        norms = np.sqrt(rng.chisquare(d, size=d))
        blocks.append(norms[:, None] * Q)
    return np.vstack(blocks)[:m]


def base_frequencies(m, d, kernel='gaussian', seed=None, method='sobol'):
    """
    Draws m unit-scale spectral frequency vectors in R^d according to Bochner's theorem.
    method: 'mc' (i.i.d.), 'sobol' (scrambled QMC), or 'orthogonal' (ORF).
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    kernel = kernel.lower()
    if kernel != 'gaussian' and kernel not in _MATERN_NU:
        raise ValueError(f"Unknown kernel '{kernel}'. Choose 'gaussian', 'matern32', or 'matern52'.")
    nu = _MATERN_NU.get(kernel)
    method = method.lower()

    if method == 'mc':
        Z = rng.standard_normal((m, d))
        g = rng.gamma(shape=nu, scale=1.0, size=m) if nu else None
    elif method == 'sobol':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            U = qmc.Sobol(d=d + (1 if nu else 0), scramble=True, seed=rng).random(m)
        U = np.clip(U, 1e-12, 1.0 - 1e-12)
        Z = norm.ppf(U[:, :d])
        g = gamma_dist.ppf(U[:, d], a=nu) if nu else None
    elif method in ('orthogonal', 'orf'):
        Z = _orthogonal_gaussian(m, d, rng)
        g = rng.gamma(shape=nu, scale=1.0, size=m) if nu else None
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'mc', 'sobol', or 'orthogonal'.")

    if nu is None:
        return Z
    return np.sqrt(nu / g)[:, None] * Z


def scale_frequencies(base, eta):
    """Scales unit spectral frequencies `base` (m, d) by ARD bandwidths `eta`."""
    d = base.shape[1]
    eta = np.atleast_1d(np.asarray(eta, dtype=float))
    if eta.size not in (1, d):
        raise ValueError(f"eta must be a scalar or have length {d}, got {eta.size}.")
    return base * np.sqrt(2.0 * eta)[None, :]


def sample_frequencies(m, d, kernel='gaussian', eta=1.0, seed=None, method='sobol'):
    """Convenience wrapper that draws base frequencies and scales them by `eta`."""
    return scale_frequencies(base_frequencies(m, d, kernel=kernel, seed=seed, method=method), eta)


def phi_from_proj(projection):
    """
    Evaluates normalized cosine and sine features directly from a precomputed
    projection matrix `projection = X @ omega.T` of shape (N, m) into a single
    preallocated (N, 2*m) array.
    """
    N, m = projection.shape
    scale = np.sqrt(1.0 / m)
    out = np.empty((N, 2 * m), dtype=float)
    np.cos(projection, out=out[:, :m])
    np.sin(projection, out=out[:, m:])
    out *= scale
    return out


def phi(X, omega):
    """
    Evaluates the normalized Random Fourier Feature map:
        phi(X) = sqrt(1 / m) * [cos(X @ omega.T), sin(X @ omega.T)]
    into a preallocated (N, 2*m) array.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    w = np.asarray(omega, dtype=float)
    return phi_from_proj(X @ w.T)