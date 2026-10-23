#rff_linear_algebra.py
import numpy as np
import scipy.linalg as sla
from scipy.special import gammaln
from scipy.stats import t

def build_system(psi, Sigma):

    K = psi.T @ psi + Sigma

    L = np.linalg.cholesky(K)
    return K, L, psi

def parameter_freq(phi, y, Sigma):

    y = np.asarray(y).reshape(-1,1)
    N = phi.shape[0]
    _, L, psi = build_system(phi, Sigma)

    rhs = psi.T @ y 
    gamma = sla.cho_solve((L, True), rhs)

    v = sla.solve_triangular(L, rhs, lower=True)
    
    # Calculate penalized Q: (y^T * y) - (v^T * v)
    Q = float(y.T @ y - v.T @ v)
    
    # Calculate estimated variance
    a = N / 2
    sigma = Q / (2 * (a - 1))
    
    return gamma, sigma, L

def parameter_bayes(phi, y, mu, Sigma):

    y = np.asarray(y).reshape(-1,1)
    mu = np.asarray(mu).reshape(-1,1)

    N = phi.shape[0]
    _, L, psi = build_system(phi, Sigma)

    rhs = psi.T @ y + Sigma@mu
    gamma = sla.cho_solve((L, True), rhs)

    v = sla.solve_triangular(L, rhs, lower=True)
    
    Q = float(y.T @ y + mu.T@Sigma@mu - v.T @ v)
        
    return gamma, Q, L


def score_freq(phi, y, Sigma, criterion='nlpd'):

    y = np.asarray(y).reshape(-1,1)
    N = phi.shape[0]

    _, L, psi = build_system(phi, Sigma)

    if criterion == 'nlml':
        rhs = psi.T @ y
        v_y = sla.solve_triangular(L, rhs, lower=True)
        
        Q = float(y.T @ y - v_y.T @ v_y)
        
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        
        return (N / 2.0) * np.log(Q) + 0.5 * log_det_K

    v = sla.solve_triangular(L, psi.T, lower=True)
    diag_H = np.sum(v ** 2, axis=0).reshape(-1, 1)
    diag_F = 1.0 - diag_H
    
    rhs = psi.T @ y
    gamma = sla.cho_solve((L, True), rhs)
    residuals = y - (psi @ gamma)

    loo_residuals = residuals / diag_F    

    if criterion == 'mse':
        return float(np.mean(loo_residuals ** 2))
    
    elif criterion == 'nlpd':
            # Variance estimate based on LOO residuals
            sigma_loo = float(np.mean(loo_residuals ** 2))
            
            loo_var = sigma_loo / diag_F

            nlpd = 0.5 * (np.log(2 * np.pi * loo_var) + (loo_residuals ** 2) / loo_var)
            
            return float(np.mean(nlpd))
    
    else:
        raise ValueError(f"Unknown criterion '{criterion}'. Choose 'nlml' or 'nlpd' or 'mse'.")                                                                   

def score_bayes(phi, y, mu, Sigma, a, b, criterion='nlpd'):

    y = np.asarray(y).reshape(-1, 1)
    mu = np.asarray(mu).reshape(-1, 1)
    N = phi.shape[0]

    _, L, psi = build_system(phi, Sigma)

    if criterion == 'nlml':
        rhs = psi.T @ y + Sigma @ mu
        v_y = sla.solve_triangular(L, rhs, lower=True)
        
        # Quadratic form Q
        Q = float(y.T @ y + mu.T @ Sigma @ mu - v_y.T @ v_y)
        
        # Log determinant of K
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))
        
        # Updated Inverse-Gamma parameters given the data
        a_N = a + (N / 2.0)
        b_N = b + (0.5 * Q)
        
        # Full Analytical Negative Log Marginal Likelihood
        nlml = (
            (gammaln(a) - a * np.log(b))
            + (a_N * np.log(b_N) - gammaln(a_N))
            + (N / 2.0) * np.log(2.0 * np.pi)
            + 0.5 * log_det_K
        )
        
        return float(nlml)

    # 2. Efficiently calculate the Hat Matrix diagonal (diag_H)
    v = sla.solve_triangular(L, psi.T, lower=True)
    diag_H = np.sum(v ** 2, axis=0).reshape(-1, 1)
    diag_F = 1.0 - diag_H
    
    # 3. Calculate standard residuals (e = y - y_hat) using the updated rhs
    rhs = psi.T @ y + Sigma @ mu
    gamma = sla.cho_solve((L, True), rhs)
    residuals = y - (psi @ gamma)

    # 4. Calculate Leave-One-Out residuals analytically
    loo_residuals = residuals / diag_F

    # 5. Evaluate Criterion
    if criterion == 'mse':
        return float(np.mean(loo_residuals ** 2))
    
    elif criterion == 'nlpd':
            # Compute posterior scale parameters from full data Q
            v_y = sla.solve_triangular(L, rhs, lower=True)
            Q = float(y.T @ y + mu.T @ Sigma @ mu - v_y.T @ v_y)
            a_N = a + (N / 2.0)
            b_N = b + (0.5 * Q)
            
            # Degrees of freedom for the Student-t predictive distribution
            nu = 2.0 * a_N
            
            # LOO mean predictions
            loo_means = y - loo_residuals
            
            # LOO Student-t predictive scale for each point
            # scale_i^2 = (b_N / a_N) * (1 / diag_F)
            scale_i = np.sqrt((b_N / a_N) / diag_F)
            
            # Evaluate Student-t Negative Log Predictive Density for each point
            log_densities = t.logpdf(
                y.flatten(), 
                df=nu, 
                loc=loo_means.flatten(), 
                scale=scale_i.flatten()
            )
            
            # Return average Negative Log Predictive Density
            return float(-np.mean(log_densities))
    
    else:
        raise ValueError(f"Unknown criterion '{criterion}'. Choose 'nlml' or 'nlpd' or 'mse'.")                                                                