#rff_linear_algebra.py
import numpy as np
import scipy.linalg as sla
from scipy.special import gammaln

def _ridge_core(psi, y, mu, Sigma):

    K = psi.T @ psi + Sigma
    L = np.linalg.cholesky(K)
 
    gamma = sla.cho_solve((L, True), psi.T @ y + Sigma @ mu)
 
    resid = y - psi @ gamma
    dev = gamma - mu
    Q = float(np.sum(resid ** 2) + np.sum(dev * (Sigma @ dev)))
    return gamma, Q, L

def _loo_mse(psi, y, gamma, L):
    """Analytic leave-one-out MSE for ridge regression: mean( e_i / (1 - H_ii) )^2."""
    v = sla.solve_triangular(L, psi.T, lower=True, check_finite=False)
    diag_F = 1.0 - np.sum(v ** 2, axis=0).reshape(-1, 1)   # 1 - H_ii
    residuals = y - psi @ gamma
    return float(np.mean((residuals / diag_F) ** 2))

def parameter_bayes(psi, y, mu, Sigma):
 
    y = np.asarray(y).reshape(-1, 1)
    mu = np.asarray(mu).reshape(-1, 1)
    p = psi.shape[1]
 
    # Sigma is diagonal, so Sigma^{1/2} is just sqrt of diagonal
    sqrt_Sigma = np.sqrt(np.diag(Sigma))
    A = np.vstack([psi, np.diag(sqrt_Sigma)])
    b_aug = np.vstack([y, (sqrt_Sigma[:, None] * mu)])
 
    # Thin QR: A = Q_qr @ R, where R.T is the lower Cholesky factor L
    Q_qr, R = np.linalg.qr(A, mode='reduced')
    
    # Ensure positive diagonal so L = R.T matches Cholesky convention
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    R = signs[:, None] * R
    Q_qr = Q_qr * signs[None, :]
    L = R.T
 
    # Solve R @ gamma = Q_qr.T @ b_aug (avoids squaring condition number)
    proj = Q_qr.T @ b_aug
    gamma = sla.solve_triangular(R, proj, lower=False)
 
    # Residual sum of squares Q is strictly non-negative: ||b_aug - A @ gamma||^2
    residual_aug = b_aug - A @ gamma
    Q = float(np.sum(residual_aug ** 2))
 
    return gamma, Q, L                                      

def score_bayes(psi, y, mu, Sigma, a, b, criterion='nlml'):
 
    y = np.asarray(y).reshape(-1, 1)
    mu = np.asarray(mu).reshape(-1, 1)
    N = psi.shape[0]
 
    gamma, Q, L = _ridge_core(psi, y, mu, Sigma)
 
    if criterion == 'nlml':
        # Log determinant of K
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        
        # Log determinant of the prior precision matrix Sigma
        diag_Sigma = np.diag(Sigma)      # Sigma is diagonal (as assumed in parameter_bayes): O(p), not O(p^3)
        if np.any(diag_Sigma <= 0):
            raise np.linalg.LinAlgError("Prior precision Sigma must be positive definite.")
        log_det_Sigma = float(np.sum(np.log(diag_Sigma)))
        
        # Updated Inverse-Gamma parameters given the data
        a_N = a + (N / 2.0)
        b_N = b + (0.5 * Q)
        
        # Full Analytical Negative Log Marginal Likelihood
        nlml = (
            (gammaln(a) - a * np.log(b))
            + (a_N * np.log(b_N) - gammaln(a_N))
            + (N / 2.0) * np.log(2.0 * np.pi)
            + 0.5 * log_det_K
            - 0.5 * log_det_Sigma
        )
        
        return float(nlml)
 
    if criterion == 'mse':
        return _loo_mse(psi, y, gamma, L)
 
    raise ValueError(f"Unknown criterion '{criterion}'. Choose 'nlml' or 'mse'.")                                                            