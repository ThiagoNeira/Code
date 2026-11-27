# test_functions.py
"""
Python oracles of the benchmark functions, for the experiments that evaluate them directly
(active learning, and the noisy data written by noisy_data.py). They match the R definitions
used to generate the nested, noise-free data (../R/ALM Park.R, ../R/ALM Perdikaris.R).
"""
import numpy as np

_EPS = np.sqrt(np.finfo(float).eps)   # R: sqrt(.Machine$double.eps), avoids x1 = 0


# ---- Park (d = 4): level 0 = park91alc, level 1 = park91a
def park91a(X):
    """High fidelity (R: park91a), rows of X in [0, 1]^4."""
    x1 = np.where(X[:, 0] == 0, _EPS, X[:, 0])
    x2, x3, x4 = X[:, 1], X[:, 2], X[:, 3]
    term1 = x1 / 2 * (np.sqrt(1 + (x2 + x3 ** 2) * x4 / (x1 ** 2)) - 1)
    term2 = (x1 + 3 * x4) * np.exp(1 + np.sin(x3))
    return term1 + term2


def park91alc(X):
    """Low fidelity (R: park91alc)."""
    x1 = np.where(X[:, 0] == 0, _EPS, X[:, 0])
    x2, x3 = X[:, 1], X[:, 2]
    yh = park91a(X)
    return (1 + np.sin(x1) / 10) * yh - 2 * x1 + x2 ** 2 + x3 ** 2 + 0.5


# ---- Perdikaris (d = 1): level 0 = perdikaris_f1 (low), level 1 = perdikaris_f2 (high)
def perdikaris_f1(X):
    return np.sin(8 * np.pi * X[:, 0])


def perdikaris_f2(X):
    return (X[:, 0] - np.sqrt(2)) * np.sin(8 * np.pi * X[:, 0]) ** 2
