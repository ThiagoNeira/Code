# features.py
import numpy as np

def sample_frequencies(m, d, kernel='gaussian', eta=1, seed=None):
    """
    Generates random frequencies for RFF supporting Gaussian and Matérn kernels,
    with options for orthogonal or non-orthogonal (standard) sampling.
    """
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    eta = np.atleast_1d(eta)
    if eta.size == 1:
        eta = np.full(d, eta.item())
    elif eta.size != d:
        raise ValueError(f"eta must be a scalar or a sequence of length {d}, got size {eta.size}")

    if kernel.lower() == 'gaussian':
        # Precision parameterization: variance is 2*eta
        omega = rng.standard_normal((m, d))
        return np.sqrt(2.0 * eta) * omega

    elif kernel.lower() == 'matern32':
        nu = 1.5
        # 1. Direction components from the first d columns
        W_raw = rng.standard_normal((m, d))
        W = W_raw / np.linalg.norm(W_raw, axis=1, keepdims=True)
        
        # 2. Radial component (Gamma) from the (d+1)-th column
        gamma_samples = rng.gamma(shape=nu, scale=1.0, size=m)
        row_norms = np.sqrt(nu / gamma_samples)
        
        # Since 1/ell = sqrt(2 * eta), we multiply by sqrt(2 * eta) instead of dividing by eta
        scale_factors = np.sqrt(2.0 * eta)
        omega = (row_norms[:, None] * scale_factors) * W
        return omega

    elif kernel.lower() == 'matern52':
        nu = 2.5
        # 1. Direction components from the first d columns
        W_raw = rng.standard_normal((m, d))
        W = W_raw / np.linalg.norm(W_raw, axis=1, keepdims=True)
        
        # Radial component (isotropic Student-t scaling)
        gamma_samples = rng.gamma(shape=nu, scale=1.0, size=m)
        row_norms = np.sqrt(nu / gamma_samples)
        
        # Since 1/ell = sqrt(2 * eta), we multiply by sqrt(2 * eta) instead of dividing by eta
        scale_factors = np.sqrt(2.0 * eta)
        omega = (row_norms[:, None] * scale_factors) * W
        return omega        

    else:
        raise ValueError(f"Unsupported kernel: '{kernel}'. Choose 'gaussian' or 'matern32' or 'matern52'.")

def RFF(X,omega):
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

class MF_RFF_Features:
    def __init__(self, m1, m2, m3, linear=False):
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.linear = linear
    
    def phi(self, X, omega):
        return RFF(X, omega)
    
    def psi(self, X, omega1, omega2, omega3, beta):
        phi1 = self.phi(X, omega1)
        phi2 = RFF(X, omega2)

        z12 = (phi1 @ beta).reshape(-1, 1)

        if self.linear:
            return np.hstack([phi2, z12])
        else:
            phi3 = RFF(X, omega3)
            diagz12 = np.diag(z12.flatten())
            r = diagz12 @ phi3
            return np.hstack([phi2, r])