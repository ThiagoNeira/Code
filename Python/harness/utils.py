import os
import time
from contextlib import contextmanager
import numpy as np

# Creates folders where plots and results will be stores
def directory_build(data_type,nested,d,N1,N2,benchmark=False):
    
    # Main folder for plots
    project_folder = 'Plots'
    os.makedirs(project_folder, exist_ok=True)
    
    if benchmark == True:
        # Folder describing the experiment
        datatype_folder = f"{data_type}(Benchmark,{nested})_d{d}_N1_{N1}_N2_{N2}"
        datatype_folder = os.path.join(project_folder, datatype_folder)
    
        os.makedirs(datatype_folder, exist_ok=True)
        
    else:
        # Folder describing the experiment
        datatype_folder = f"{data_type}({nested})_d{d}_N1_{N1}_N2_{N2}"
        datatype_folder = os.path.join(project_folder, datatype_folder)
    
        os.makedirs(datatype_folder, exist_ok=True)
        
    return datatype_folder

@contextmanager

def timer(label, store=None):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[TIMER] {label}: {elapsed:.3f}s")

    if store is not None:
        store[label]=elapsed

import numpy as np

def crps_ensemble(predictions, y_true):
    """
    Computes the Continuous Ranked Probability Score (CRPS) for Monte Carlo samples.
    
    Parameters:
    - predictions: np.ndarray of shape (N_test, N_samples) containing predictive trajectories.
    - y_true: np.ndarray of shape (N_test,) or (N_test, 1) true observations.
    
    Returns:
    - float: Average CRPS across the test set.
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true).flatten()
    
    # Term 1: Mean absolute error between each sample and the true observation
    term1 = np.mean(np.abs(predictions - y_true[:, None]), axis=1)
    
    # Term 2: Mean absolute difference between all pairs of samples
    abs_diff = np.abs(predictions[:, :, None] - predictions[:, None, :])
    term2 = np.mean(abs_diff, axis=(1, 2))
    
    crps_per_point = term1 - 0.5 * term2
    return np.mean(crps_per_point)

from scipy.stats import norm

def crps_gaussian(mu, var, y_true):
    """
    Computes the analytical CRPS for models that output a Gaussian 
    predictive distribution N(mu, var).
    """
    mu = np.asarray(mu).flatten()
    # Add a tiny epsilon to variance to prevent division by zero or NaN issues
    sigma = np.sqrt(np.maximum(np.asarray(var).flatten(), 1e-12))
    y_true = np.asarray(y_true).flatten()
    
    # Standardized residual
    z = (y_true - mu) / sigma
    
    # Closed-form Gaussian CRPS
    crps_per_point = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    
    return np.mean(crps_per_point)