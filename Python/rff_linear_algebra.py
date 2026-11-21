# rff_linear_algebra.py
"""
Bayesian linear algebra routines for Random Fourier Feature regression.
Automatically switches between:
  - Primal Augmented Householder QR (p <= N): O(N p^2 + p^3), square-root conditioning.
  - Dual Cholesky with Woodbury & Matheron's Rule (p > N): O(N^2 p + N^3).
Includes mode='r' fast value-only NLML evaluation and exact analytical NLML gradients.
"""
import numpy as np
import scipy.linalg as sla
from scipy.special import gammaln

LN10 = np.log(10.0)


def _prior_diag(Sigma):
    """
    Extracts and validates the 1D diagonal of the prior precision matrix Sigma.
    Accepts either a 1D vector of length p or a 2D diagonal matrix of shape (p, p).
    """
    S = np.asarray(Sigma, dtype=float)
    d = np.diag(S) if S.ndim == 2 else S.ravel()
    if not np.all(np.isfinite(d)) or np.any(d <= 0.0):
        raise np.linalg.LinAlgError("Prior precision Sigma must be finite and positive definite.")
    return d


def _augmented_qr(psi, y, mu, prec):
    """
    Primal solver via thin Householder QR on the augmented system:
        A = [psi; diag(sqrt(prec))],   b_aug = [y; sqrt(prec) * mu]
    Operates at square-root condition number sqrt(kappa(K)). Best when p <= N.
    """
    sqrt_prec = np.sqrt(prec)
    A = np.vstack([psi, np.diag(sqrt_prec)])
    b_aug = np.vstack([y, sqrt_prec[:, None] * mu])

    Q_qr, R = np.linalg.qr(A, mode='reduced')
    signs = np.sign(np.diag(R))
    signs[signs == 0.0] = 1.0
    R *= signs[:, None]
    Q_qr *= signs[None, :]

    gamma = sla.solve_triangular(R, Q_qr.T @ b_aug, lower=False, check_finite=False)
    Q = float(np.sum((b_aug - A @ gamma) ** 2))
    return Q_qr, R, gamma, Q


def _augmented_qr_value_only(psi, y, mu, prec):
    """
    Fast value-only primal solver using QR with mode='r' on the (N + p) x (p + 1)
    extended matrix [A | b_aug].
      - R_ext[:p, :p] is the upper Cholesky factor R of K = A.T @ A.
      - R_ext[p, p]^2 is the exact residual sum of squares Q = min_gamma ||b_aug - A @ gamma||_2^2.
    Skips Q_qr construction and triangular solves during LHS screening.
    """
    p = psi.shape[1]
    sqrt_prec = np.sqrt(prec)
    A = np.vstack([psi, np.diag(sqrt_prec)])
    b_aug = np.vstack([y, sqrt_prec[:, None] * mu])
    R_ext = np.linalg.qr(np.hstack([A, b_aug]), mode='r')
    diag_R = np.abs(np.diag(R_ext[:p, :p]))
    Q = float(R_ext[p, p] ** 2)
    return diag_R, Q


def _dual_cholesky(psi, y, mu, prec):
    """
    Dual solver via the N x N covariance matrix:
        M = I_N + psi @ diag(1 / prec) @ psi.T = L_M @ L_M.T
    Best when p > N.
    """
    N = psi.shape[0]
    inv_prec = 1.0 / prec
    psi_scaled = psi * inv_prec[None, :]          # (N, p) = psi @ Sigma^{-1}
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
    """
    Evaluates the Negative Log Marginal Likelihood (NLML) of the conjugate
    Normal-Inverse-Gamma (NIG) model given 0.5 * (log|K| - log|Sigma|).
    """
    a_N = a + N / 2.0
    b_N = b + 0.5 * Q
    nlml = ((gammaln(a) - a * np.log(b))
            + (a_N * np.log(b_N) - gammaln(a_N))
            + (N / 2.0) * np.log(2.0 * np.pi)
            + half_log_det_ratio)
    return float(nlml), a_N, b_N


def parameter_bayes(psi, y, mu, Sigma):
    """
    Computes exact conjugate Bayesian posterior parameters for design matrix `psi`.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    N, p = psi.shape
    prec = _prior_diag(Sigma)

    if p <= N:
        _, R, gamma, Q = _augmented_qr(psi, y, mu, prec)
        return gamma, Q, R.T
    L_M, _, gamma, Q, _ = _dual_cholesky(psi, y, mu, prec)
    return gamma, Q, L_M


def sample_weights_bayes(gamma_hat, L, sigma_samples, psi_train, Sigma, rng):
    """
    Draws S posterior weight vectors from N(gamma_hat, sigma_s^2 * K^{-1}), returning shape (S, p).
    - When `psi_train` is None (Primal, p <= N): direct triangular solve with L (p, p).
    - When `psi_train` is provided (Dual, p > N): Matheron's rule with L_M (N, N).
    """
    gamma_hat = np.asarray(gamma_hat, dtype=float).reshape(-1, 1)
    p = gamma_hat.shape[0]
    sigma_samples = np.atleast_1d(sigma_samples).astype(float)
    S = sigma_samples.shape[0]
    sqrt_sig = np.sqrt(sigma_samples)[None, :]                        # (1, S)

    if psi_train is None:
        zz = rng.standard_normal((p, S))
        dev = sla.solve_triangular(L, zz, lower=True, trans='T', check_finite=False)
        return (gamma_hat + sqrt_sig * dev).T                         # (S, p)

    N = psi_train.shape[0]
    prec = _prior_diag(Sigma)
    inv_prec = 1.0 / prec                                             # (p,)
    inv_sqrt_prec = np.sqrt(inv_prec)                                 # (p,)

    gamma_0 = inv_sqrt_prec[:, None] * rng.standard_normal((p, S))    # (p, S)
    eps_0 = rng.standard_normal((N, S))                               # (N, S)

    v = sla.solve_triangular(L, psi_train @ gamma_0 + eps_0, lower=True, check_finite=False)
    u = sla.solve_triangular(L, v, lower=True, trans='T', check_finite=False)

    dev = gamma_0 - inv_prec[:, None] * (psi_train.T @ u)             # (p, S)
    return (gamma_hat + sqrt_sig * dev).T                             # (S, p)


def leverage_bayes(psi_star, L, psi_train, Sigma):
    """
    Computes the quadratic leverage form h(x_*) = diag(psi_star @ K^{-1} @ psi_star.T)
    as a 1D array of length N_test.
    """
    if psi_train is None:
        v_mat = sla.solve_triangular(L, psi_star.T, lower=True, check_finite=False)
        return np.sum(v_mat ** 2, axis=0)

    prec = _prior_diag(Sigma)
    psi_star_scaled = psi_star / prec[None, :]                        # (N_test, p)
    h_prior = np.sum(psi_star_scaled * psi_star, axis=1)              # (N_test,)
    cross = psi_train @ psi_star_scaled.T                             # (N, N_test)
    v_mat = sla.solve_triangular(L, cross, lower=True, check_finite=False)
    return np.maximum(0.0, h_prior - np.sum(v_mat ** 2, axis=0))


def dual_leverage_factor(L, psi_train, Sigma):
    """
    Precomputes the test-independent dual leverage factor (Dual, p > N only):
        V = L_M^{-1} @ psi_train @ diag(1 / prec),   shape (N, p),
    so that h(x_*) = h_prior(x_*) - ||V @ psi_star.T||^2 column-wise (see `leverage_from_factor`).
    """
    prec = _prior_diag(Sigma)
    return sla.solve_triangular(L, psi_train / prec[None, :], lower=True, check_finite=False)


def leverage_from_factor(psi_star, V, h_prior):
    """
    Dual leverage h(x_*) = h_prior - rowsum((psi_star @ V.T)^2) from a precomputed factor
    V = dual_leverage_factor(...). Equals `leverage_bayes` in dual mode, with a single GEMM
    and no triangular solve. `h_prior` is diag(psi_star @ diag(1/prec) @ psi_star.T)
    (scalar or (N_test,) array).
    """
    G = psi_star @ V.T                                                # (N_test, N)
    return np.maximum(0.0, h_prior - np.einsum('ij,ij->i', G, G))


def score_bayes(psi, y, mu, Sigma, a, b):
    """
    Evaluates the Negative Log Marginal Likelihood (NLML) without gradients.
    Uses mode='r' augmented QR when p <= N to skip Q_qr construction and triangular solves.
    """
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    N, p = psi.shape
    prec = _prior_diag(Sigma)

    if p <= N:
        diag_R, Q = _augmented_qr_value_only(psi, y, mu, prec)
        half_log_det = float(np.sum(np.log(diag_R))) - 0.5 * float(np.sum(np.log(prec)))
    else:
        L_M, _, _, Q, _ = _dual_cholesky(psi, y, mu, prec)
        half_log_det = float(np.sum(np.log(np.diag(L_M))))

    nlml, _, _ = _nlml_value_from_logdet(N, a, b, Q, half_log_det)
    return nlml


def score_bayes_with_grad(psi, y, mu, Sigma, a, b, X_eff, omega, tune_nugget=False):
    """
    Evaluates the NLML and its exact analytical gradient with respect to
    log10(eta) (and log10(lam) if tune_nugget=True).
    """
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

        L_M_inv = sla.solve_triangular(L_M, np.eye(N), lower=True, check_finite=False)
        M_inv = L_M_inv.T @ L_M_inv
        scale_Q = a_N / b_N

        G = (1.0 / lam) * (M_inv @ psi[:, :two_m]) - (scale_Q * alpha) @ gamma[:two_m].T

        if tune_nugget:
            Phi2m = psi[:, :two_m]
            Gram_2m = (Phi2m @ Phi2m.T) / lam
            lam_tr_Kinv = float(two_m - np.sum(M_inv * Gram_2m))

    Phi_cos, Phi_sin = psi[:, :m], psi[:, m:two_m]
    M_chain = -G[:, :m] * Phi_sin + G[:, m:two_m] * Phi_cos
    grad_log_eta = 0.5 * LN10 * np.sum((X_eff.T @ M_chain) * omega.T, axis=1)

    if not tune_nugget:
        return nlml, grad_log_eta

    quad_gamma = float(np.sum((gamma[:two_m] - mu[:two_m]) ** 2))
    grad_log_lam = LN10 * (0.5 * lam_tr_Kinv - m + 0.5 * scale_Q * lam * quad_gamma)
    return nlml, np.append(grad_log_eta, grad_log_lam)