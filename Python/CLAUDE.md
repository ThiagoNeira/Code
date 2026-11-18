# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for an **L-level multi-fidelity Bayesian Random Fourier Feature (MF-RFF) regression model**, benchmarked against GP multi-fidelity baselines on standard test functions (Branin, Borehole, Currin, Park, Perdikaris). There is no package, build system, or test suite.

The core of the project is three files:

| File | Purpose |
|---|---|
| `MF_RFF_Bayes.py` | The model class `MF_RFF_bayes`: per-level configuration, hyperparameter tuning, posterior sampling across levels, and prediction with a variance decomposition. |
| `rff_linear_algebra.py` | Conjugate Normal–Inverse-Gamma (NIG) posterior, weight sampling, leverage, and NLML with exact analytic gradients. Switches between a primal solver (p ≤ N) and a dual solver (p > N). |
| `features_full.py` | Spectral frequency draws for the gaussian, matern32, and matern52 kernels (`sobol` / `mc` / `orthogonal`), bandwidth scaling, and the trigonometric feature map. |

The git root is the parent `Code/` directory.

## Running

`python` is not on PATH in the default shell (it resolves to the Windows Store alias), so use the user's environment or interpreter explicitly. Run from this `Python/` directory, because plot paths are relative to the CWD:

```
python branin.py      # likewise borehole.py, currin.py, park.py, perdikaris.py
```

Each benchmark script defines `models_to_run`, where each entry's keys go straight to the `MF_RFF_bayes(...)` constructor, and calls `experiment.run_experiment`. For a quick check, lower `n_replicates`, `n_restarts`, or `n_samples` in the script.

Dependencies: numpy, scipy, scikit-learn, joblib, threadpoolctl, matplotlib, GPy, emukit.

## The model

Levels run l = 0 (lowest fidelity) to L-1 (target). Each level is a Bayesian linear model on its own RFF basis. With ψ_l(x) the design row, w_l the weights, and σ_l² that level's noise variance:

- **Level 0:** y_0 = φ_0(x)ᵀ w_0 + ε_0
- **Linear (Kennedy–O'Hagan) link, `linear=True`:** y_l = φ_l(x)ᵀ β_l + ρ_l f_{l-1}(x) + ε_l. Here ψ_l = [φ_l, f_{l-1}], and ρ_l is a constant scalar.
- **Non-linear (NARGP-style) link, `linear=False`:** y_l = φ_l([x, f̃_{l-1}(x)])ᵀ w_l + ε_l. This is one stationary ARD kernel on the (d+1)-dimensional input. f̃ is f_{l-1} standardized by `f_loc` / `f_scale`.

Probabilistic assumptions, as the code implements them:

- **Noise:** ε_l ~ N(0, σ_l² I). It is homoscedastic and independent across levels.
- **Weights:** Gaussian **conditionally on σ_l²** (conjugate NIG), with zero prior mean:
  - w_l | σ_l² ~ N(0, σ_l² / λ_l · I) for the RFF weights.
  - ρ_l | σ_l² ~ N(0, σ_l² τ), with τ = 1e5.
  - Marginally over σ_l², the weights are Student-t.
- **Variance:** there is one inverse-gamma prior per level, on σ_l² only: σ_l² ~ IG(a, b), with a = b = 1e-6.
- **Point-estimated hyperparameters:** the "nugget" λ_l (a noise-to-signal ratio) and the log10 bandwidths η_l are not given priors. They are type-II ML point estimates that minimize the NLML. 
- **Inputs need not be nested across levels.** f_{l-1} is evaluated at X_l by propagating posterior draws, not by looking up y_{l-1}. The current R-generated benchmark data *are* nested (`NestedX`), because co-kriging requires nested designs.
- **Fitting is sequential, as in recursive co-kriging and NARGP.** Data at level l never updates the posteriors of levels below it.
- **Centering:** each y_l is centered by its sample mean before fitting, and the mean is added back at predict time.

## Data flow: `fit` → `predict`

### `fit(X_list, y_list)`

1. **Validate and size.** Compute N_l, then m_l = `_resolve_m(N_l, feature_mults[l])` and p_l = 2 m_l. With the usual `feature_mults`, p > N, so the **dual** path is the common one.
2. **Seeds.** `SeedSequence(seed).spawn(2L+2)` produces the frequency seeds, the per-level optimizer seeds, `seed_fit`, and `seed_pred`.
3. **Base frequencies.** Draw `base_freq[l]` of shape (m_l, d), or (m_l, d+1) for non-linear levels. Center each y_l.
4. **`_tune_all_levels`: tune hyperparameters level by level.** For each level, minimize the NLML over log10 η (and log10 λ if `tune_nugget`) with `_multistart_lbfgsb`. That routine screens an LHS pool of `n_screen * n_restarts` points, then runs L-BFGS-B with exact gradients from the best starts.
   - In this stage, f_{l-1} at X_l is the **deterministic plug-in posterior-mean chain** (`predict_plugin`), not samples.
   - This stage sets `eta`, `omega`, `lam`, `Sigma_inv`, `f_loc`, and `f_scale`.
5. **`_fit_posteriors`: fit the full posteriors with S = `n_samples` trajectories.**
   - **Level 0:** an exact NIG posterior, with S coupled draws (σ²_s, w_s).
   - **Level l ≥ 1:** `Z_prev = _evaluate_trajectories_at(l-1, X_l)` gives an (N_l, S) matrix of latent draws, one per trajectory.
     - **Linear:** `_fit_level_linear_schur` factors φ_l once and handles ρ for all S trajectories at once, using rank-1 Schur-complement updates.
     - **Non-linear:** `_fit_single_trajectory_nl` solves one NIG posterior per trajectory. It runs under joblib threading with BLAS limited to 1 thread.
   - The results are stored in `W[l]`, `sig2_draws[l]`, `Q[l]`, and `level_cache[l]`.

### `predict(X, level='all'|int, return_samples=False)`

- **Level 0:** closed-form Student-t mean, variance, and quantiles, with df = N_0 + 2a.
- **Level l ≥ 1:**
  - For each trajectory s, compute the conditional mean `mu_s` and leverage `H`.
  - The law of total variance gives `var = var_prop` (spread of `mu_s`, i.e. uncertainty propagated from lower levels) `+ var_weights + var_noise`. `varexp` and `expvar` are aliases for the two halves of that split.
  - Samples are `Z_curr` + noise drawn with the **same** `sig2_draws` as in fit. Quantiles are empirical over the S samples.
- **Return values:**
  - `level='all'` returns a dict keyed by level.
  - An integer level returns `(mean, var, q_lo, q_hi[, samples])`.
  - Every call also stores all levels in `last_predictions_`.

## Notation (as used in code)

| Symbol | Meaning |
|---|---|
| `L`, `lvl`/`l` | Number of levels; 0-based level index in Python. The data files use 1-based `X1..XL`. |
| `d`, `N[l]`, `m[l]`, `p[l]` | Input dimension, points at level l, number of frequencies, number of features (p = 2m). |
| `S` / `samples` | Number of posterior trajectories (`n_samples`). |
| `eta[l]` | ARD bandwidths. `omega = base_freq * sqrt(2*eta)`, so for the gaussian kernel k = exp(-Σ η_j r_j²). Bounds are in **log10**. |
| `omega[l]` | Scaled frequencies (m_l, d) or (m_l, d+1). Non-linear levels split them into `omega[:, :d]` (x part) and `omega[:, d:]` (f part). |
| `phi` / `psi` | φ = sqrt(1/m)·[cos(Xωᵀ), sin(Xωᵀ)]. ψ is the full design: φ itself, `[φ, f]` (linear), or φ of `[x, f̃]` (non-linear). |
| `lam[l]` | The "nugget": prior **precision** of the RFF weights, relative to σ_l². |
| `Sigma`, `Sigma_inv`, `prec` | **All three name the prior precision diagonal** `[λ,…,λ(, 1/τ)]`, not a covariance. Built by `_prec_vector`. |
| `mu` | Prior mean of the weights (zeros). |
| `a_prior`, `b_prior`, `tau` | IG shape and rate for σ²; prior variance multiplier for ρ. |
| `K`, `M` | Primal K = ψᵀψ + diag(prec). Dual M = I + ψ diag(1/prec) ψᵀ. |
| `L` (factor) | Cholesky factor of K (p×p, primal) **or** of M (N×N, dual). Which one it is is signalled by `psi_train` / `phi_cache` being `None` (primal) or the training design (dual). |
| `gamma`/`gamma_hat`, `beta0_hat` | Posterior mean of the weights. |
| `Q` | Posterior quadratic form + 2b, so that b_N = Q/2. It is a scalar at level 0 and an (S,) array at higher levels. |
| `h`, `H_mat` | Leverage ψ* K⁻¹ ψ*ᵀ. |
| `W[l]`, `sig2_draws[l]` | Coupled posterior draws, of shape (S, p_l [+1]) and (S,). |
| `Z`, `Z_prev`, `Z_curr`, `F_prev` | (N, S) matrices of **centered, noise-free** latent trajectories f_l. |

## Gotchas when editing the core

- **Column layout for the gradients.** `score_bayes_with_grad` assumes the first 2m columns of ψ are `[cos | sin]` with precision `prec[0] = λ`, and that any further columns (the ρ column) are excluded from the η and λ gradients. It also assumes that the columns of `X_eff` line up with the columns of `omega`. Changing the feature layout breaks the gradients silently.
- **Primal vs. dual switch.** `parameter_bayes`, `sample_weights_bayes`, and `leverage_bayes` all branch on p ≤ N. Callers must pass the training design (`psi_train`) exactly when p > N, and `None` otherwise.
- **Per-level arguments.** Per-level constructor arguments go through `_broadcast_per_level`:
  - Length L: `nuggets`, `feature_mults`, `eta_bounds`, `nugget_bounds`.
  - Length L-1: `linear`, `eta_bounds_f`.
  - Level 0 is never linear.

## Other Python files

- **`experiment.py`** only compares models. It loads each replicate's data, fits every configured model, and saves boxplots of RMSE, R², CRPS, coverage, and time to `plots/<fn>/`. CRPS comes from samples for MF-RFF and from a Gaussian approximation for the other models.
- **Baselines** (GPy/emukit, plus the single-fidelity `One_shot_GP.py`). `experiment.py` calls `fit(X_list, y_list)` and `predict(X, return_ci=True) → (mean, var, lower, upper)` for the target level; every multi-fidelity baseline must implement that interface.
  - **`NARGP.py`** supports **L levels**. Fitting is recursive: level l is trained on `[X_l, m_{l-1}(X_l)]`, where m is the deterministic plug-in mean chain. Prediction propagates `nsamples` Monte Carlo chains through every level using latent (`predict_noiseless`) draws, then applies the law of total variance at the target level (`level=-1` by default). Per-point draws are used instead of `full_cov`, because only pointwise marginals are needed. `seed` controls the propagation RNG; GPy's restarts still use the global `np.random`.
  - **`KO_model.py`** supports **L levels**. It is one joint Kennedy–O'Hagan (AR1) GP built from emukit's `LinearMultiFidelityKernel`: L sub-kernels, L−1 unconstrained scale parameters ρ, and one Gaussian noise term per level. The fidelity index is the last input column (0..L-1). Prediction is the exact Gaussian posterior at `level` (default -1); there is no sampling. With L = 2 it reproduces the original two-level code bit for bit, because the constraints and warm starts are applied in the same grouped order. Keep that order if you edit it.
    - `seed` seeds the global `np.random` (which GPy restarts use) for the duration of `fit` only, then restores the previous state.
    - The bounds (variance ≤ 10, noise ≤ 1, length-scale ≤ 10) assume roughly unit-scale outputs. KO is not normalized, so on large-range targets such as Branin its coverage is poor.

## R side: data and R baselines (read-only)

- **Never edit the R scripts or their outputs.** Python only reads them.
- **Scripts:** `../R/<fn> rnamf-cokr.R`, plus `../R/perdikaris rnamf.R`.
- **Outputs:** `Data/<fn>/<N1>_..._<NL>/`. Each folder holds:
  - `X{l}_{N_l}_rep{i}.npy` and `y{l}_{N_l}_rep{i}.npy`
  - `x_test.npy` and `yhigh_test.npy`
  - precomputed RNAmf (`rna_*`) and co-kriging (`cokm_*`) results: `_mean`, `_sigma2`, `_lower`, `_upper`, and `_time`, each as `_rep{i}.npy`
- **How Python uses them:** `N_list` in a script must match an existing folder name, and replicate r reads `rep{base_seed + r}`. The `"rna"` and `"cokm"` model entries just load these files.
