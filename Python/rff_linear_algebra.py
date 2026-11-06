# rff_linear_algebra.py
import numpy as np
import scipy.linalg as sla
from scipy.special import gammaln

LN10 = np.log(10.0)


def _prior_diag(Sigma):
    """Diagonal of the prior precision matrix (dense diagonal matrix or vector accepted)."""
    S = np.asarray(Sigma, dtype=float)
    d = np.diag(S).copy() if S.ndim == 2 else S.ravel().copy()
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise np.linalg.LinAlgError("Prior precision Sigma must be positive definite.")
    return d


def _augmented_qr(psi, y, mu, prec):
    """Primal solver: O(N p^2 + p^3). Best when p <= N."""
    sqrt_prec = np.sqrt(prec)
    A = np.vstack([psi, np.diag(sqrt_prec)])
    b_aug = np.vstack([y, sqrt_prec[:, None] * mu])

    Q_qr, R = np.linalg.qr(A, mode='reduced')
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    R = signs[:, None] * R
    Q_qr = Q_qr * signs[None, :]

    gamma = sla.solve_triangular(R, Q_qr.T @ b_aug, lower=False, check_finite=False)
    Q = float(np.sum((b_aug - A @ gamma) ** 2))
    return Q_qr, R, gamma, Q


def _dual_cholesky(psi, y, mu, prec):
    """
    Dual solver: O(N^2 p + N^3). Best when p > N.
    Forms M = I_N + psi @ diag(1/prec) @ psi.T = L_M @ L_M.T (N x N).
    """
    N = psi.shape[0]
    inv_prec = 1.0 / prec
    psi_scaled = psi * inv_prec[None, :]          # (N, p) = Psi @ Sigma^{-1}
    M = psi_scaled @ psi.T                        # (N, N)
    M[np.diag_indices(N)] += 1.0

    L_M = sla.cholesky(M, lower=True, check_finite=False)
    y_tilde = y - psi @ mu if np.any(mu) else y
    v = sla.solve_triangular(L_M, y_tilde, lower=True, check_finite=False)
    alpha = sla.solve_triangular(L_M, v, lower=True, trans='T', check_finite=False)

    gamma = mu + psi_scaled.T @ alpha             # (p, 1) exact posterior mean
    Q = float(np.sum(v ** 2))                     # y_tilde.T @ M^{-1} @ y_tilde
    return L_M, alpha, gamma, Q, M


def _nlml_value_from_logdet(N, a, b, Q, half_log_det_ratio):
    """NLML given 0.5 * (log|K| - log|Sigma|) = sum(log(diag(L_primal))) - 0.5*sum(log(prec)) = sum(log(diag(L_M)))."""
    a_N = a + N / 2.0
    b_N = b + 0.5 * Q
    nlml = ((gammaln(a) - a * np.log(b))
            + (a_N * np.log(b_N) - gammaln(a_N))
            + (N / 2.0) * np.log(2.0 * np.pi)
            + half_log_det_ratio)
    return float(nlml), a_N, b_N


def parameter_bayes(psi, y, mu, Sigma):
    """
    Posterior mean gamma, Q = ||y - psi g||^2 + (g-mu)' Sigma (g-mu), and lower factor L with L L' = K.
    Currently restricted to the primal solver because MF_RFF_Bayes expects a (p x p) Cholesky factor for MC sampling.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    _, R, gamma, Q = _augmented_qr(psi, y, mu, _prior_diag(Sigma))
    return gamma, Q, R.T


def score_bayes(psi, y, mu, Sigma, a, b):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    N, p = psi.shape
    prec = _prior_diag(Sigma)

    if p <= N:
        _, R, _, Q = _augmented_qr(psi, y, mu, prec)
        half_log_det = float(np.sum(np.log(np.diag(R)))) - 0.5 * float(np.sum(np.log(prec)))
    else:
        L_M, _, _, Q, _ = _dual_cholesky(psi, y, mu, prec)
        half_log_det = float(np.sum(np.log(np.diag(L_M))))

    nlml, _, _ = _nlml_value_from_logdet(N, a, b, Q, half_log_det)
    return nlml


def score_bayes_with_grad(psi, y, mu, Sigma, a, b, X_eff, omega, tune_nugget=False):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    N, p = psi.shape
    m = omega.shape[0]
    two_m = 2 * m
    prec = _prior_diag(Sigma)
    lam = float(prec[0])

    if p <= N:
        Q_qr, R, gamma, Q_raw = _augmented_qr(psi, y, mu, prec)
        L = R.T
        half_log_det = float(np.sum(np.log(np.diag(L)))) - 0.5 * float(np.sum(np.log(prec)))
        nlml, a_N, b_N = _nlml_value_from_logdet(N, a, b, Q_raw, half_log_det)

        Q_psi = Q_qr[:N, :]
        Psi_Kinv = sla.solve_triangular(L, Q_psi.T, lower=True, trans='T', check_finite=False).T
        r = y - psi @ gamma
        scale_Q = a_N / b_N
        G = Psi_Kinv[:, :two_m] - (scale_Q * r) @ gamma[:two_m].T
        lam_tr_Kinv = float(np.sum(Q_qr[N:N + two_m, :] ** 2)) if tune_nugget else 0.0
    else:
        L_M, alpha, gamma, Q_raw, M = _dual_cholesky(psi, y, mu, prec)
        half_log_det = float(np.sum(np.log(np.diag(L_M))))
        nlml, a_N, b_N = _nlml_value_from_logdet(N, a, b, Q_raw, half_log_det)

        # M^{-1} via N x N triangular inverse
        L_M_inv = sla.solve_triangular(L_M, np.eye(N), lower=True, check_finite=False)
        M_inv = L_M_inv.T @ L_M_inv                                       # (N, N)
        scale_Q = a_N / b_N

        # Psi_Kinv[:, :2m] = (1 / lam) * M_inv @ psi[:, :2m], and r = alpha
        G = (1.0 / lam) * (M_inv @ psi[:, :two_m]) - (scale_Q * alpha) @ gamma[:two_m].T

        if tune_nugget:
            # (1/lam) * psi[:, :2m] @ psi[:, :2m].T is M - I_N (minus the rho rank-1 piece if p > 2m)
            Gram_2m = (M - np.eye(N)) if p == two_m else (psi[:, :two_m] @ psi[:, :two_m].T) / lam
            lam_tr_Kinv = float(two_m - np.sum(M_inv * Gram_2m))

    Phi_cos, Phi_sin = psi[:, :m], psi[:, m:two_m]
    M_chain = -G[:, :m] * Phi_sin + G[:, m:two_m] * Phi_cos
    grad_log_eta = 0.5 * LN10 * np.sum((X_eff.T @ M_chain) * omega.T, axis=1)

    if not tune_nugget:
        return nlml, grad_log_eta

    quad_gamma = float(np.sum((gamma[:two_m] - mu[:two_m]) ** 2))
    grad_log_lam = LN10 * (0.5 * lam_tr_Kinv - m + 0.5 * scale_Q * lam * quad_gamma)
    return nlml, np.append(grad_log_eta, grad_log_lam)