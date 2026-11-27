# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for an **L-level multi-fidelity Bayesian Random Fourier Feature (MF-RFF) regression model** and **cost-aware active learning (AL)** built on it. It is benchmarked against GP multi-fidelity baselines on standard test functions (Branin, Borehole, Currin, Park, Perdikaris). There is no installable package, build system, or test suite. The git root is the parent `Code/` directory: Python lives in `Code/Python/`, and the R data generators and R baselines live in `Code/R/`.

`Code/Python/` is split into importable folders (each has an empty `__init__.py`); `Python/` itself is the import root. Library code lives in `model/`, `al/`, `baselines/` and `harness/`; the scripts that run one experiment each live in `benchmarks/` and `al_experiments/`.

| Group | Folder | Files |
|---|---|---|
| **1. Model** (core) | `model/` | `MF_RFF_Bayes.py` (class `MF_RFF_bayes`), `rff_linear_algebra.py` (conjugate NIG algebra, NLML and exact gradients), `features_full.py` (spectral draws, feature map) |
| **2. Active learning** (core) | `al/` | `active_learning.py` (ALM and ALC acquisition, AL loop) |
| 3. Baselines | `baselines/` | `NARGP.py`, `KO_model.py`, `One_shot_GP.py` (GPy / emukit). RNAmf and co-kriging run in R and their results are precomputed; `../R/RNAmf_0.1.2.tar.gz` is the RNAmf package source |
| Shared experiment code | `harness/` | `experiment.py` (benchmark runner), `alm_experiment.py` (AL runner), `utils.py`, `plotting.py`, `noisy_data.py` (noisy, non-nested data), `test_functions.py` (Python oracles: `park91a`, `park91alc`, `perdikaris_f1`, `perdikaris_f2`) |
| 4. Nested, noise-free benchmarks | `benchmarks/nested/` | drivers `branin.py`, `borehole.py`, `currin.py`, `park.py`, `perdikaris.py`; data from `../R/<fn> rnamf-cokr.R` |
| 5. Noisy, non-nested benchmarks | `benchmarks/noisy/` | `noisy_park.py`, `noisy_perdikaris.py` |
| 6. AL experiments, nested and noise-free | `al_experiments/` | `alm_park.py`, `alm_perdikaris.py` (vs RNAmf); `alc_park.py`, `alc_perdikaris.py` (ALM vs ALC); `active_test.py` (checks) |
| 7. AL experiments, noisy and non-nested | `al_experiments/` | `al_noisy_park.py`, `al_noisy_perdikaris.py` |
| Paths | `Python/paths.py` | `ROOT` (= `Python/`), `DATA`, `PLOTS`, `under_root(p)` |

Imports are absolute from `Python/`: `from model.MF_RFF_Bayes import MF_RFF_bayes`, `from al.active_learning import ...`, `from harness.experiment import run_experiment`, `from paths import DATA`. Dependencies point one way: drivers → `harness/` → `model/`, `al/`, `baselines/`. The only cross-folder driver imports are `al_noisy_<fn>.py` → `benchmarks/noisy/noisy_<fn>.py` (noise settings and models) and `alc_<fn>.py` → `alm_<fn>.py` (models).

## Environment setup (any machine)

The project runs on several computers, so set up the environment from this recipe rather than copying an env.

### Python

Use **Python 3.10** from **conda-forge**, with **numpy < 2**. Install GPy with conda first, then emukit with pip.

```
conda create -n mfrff -c conda-forge python=3.10 "numpy<2" scipy scikit-learn joblib threadpoolctl matplotlib gpy=1.13.2
conda activate mfrff
pip install emukit==0.5.1
python -c "import numpy, GPy, emukit; print(numpy.__version__)"    # must print 1.x
```

Verified working set (Windows, Nov 2026); all project modules import, and KO, NARGP, GP and MF-RFF all fit and predict:

| Package | Version | Needed by |
|---|---|---|
| python | 3.10.22 | everything |
| numpy | 1.26.4 (**must be < 2**) | everything |
| scipy | 1.11.4 | everything (`linalg`, `optimize`, `stats.qmc`, `special`) |
| joblib, threadpoolctl | 1.6.0, 3.6.0 | `MF_RFF_Bayes.py` (threaded non-linear levels) |
| matplotlib | 3.10.9 | `plotting.py`, `alm_experiment.py`, `active_test.py` |
| scikit-learn | 1.7.2 | `experiment.py` (RMSE, R²) |
| GPy (+ paramz 0.9.6) | 1.13.2 | `NARGP.py`, `KO_model.py`, `One_shot_GP.py` |
| emukit | 0.5.1 | `KO_model.py` |

Why the constraints:
- **numpy < 2.** GPy 1.13.2 depends on paramz 0.9.6, which imports `numpy.lib.function_base`; numpy 2 removed it. With numpy 2, `import GPy` fails with `ModuleNotFoundError: No module named 'numpy.lib.function_base'`.
- **Install order.** Install GPy (conda) before emukit (pip). With numpy 1.26 already present, pip keeps it. If a `pip install` ever pulls in numpy 2, fix it with `pip install "numpy<2"` and recheck the import.
- **Activate the env on Windows.** Use `conda activate` or the Anaconda Prompt. Calling `envs\<name>\python.exe` directly without activation leaves `Library\bin` off PATH; numpy's LAPACK then fails to load, and GPy crashes natively (code `0xc06d007f`) when building a kernel.

What needs what:
- The core model (`MF_RFF_Bayes.py`, `rff_linear_algebra.py`, `features_full.py`) and `active_learning.py` need only numpy, scipy, joblib and threadpoolctl.
- `active_test.py` also needs matplotlib.
- `experiment.py` imports `NARGP`, `KO_model` and `One_shot_GP` at module level. So **every benchmark driver and every AL driver** (they import `experiment.py` through `alm_experiment.py`) needs GPy and emukit, even when it runs only MF-RFF.

### R (data generation and the R baselines)

- R ≥ 4.x; R 4.6.2 is installed on one of the machines.
- CRAN packages: `install.packages(c("DiceKriging", "RcppCNPy", "glue", "lhs", "plgp", "foreach", "doParallel"))`. `plgp`, `foreach` and `doParallel` are RNAmf's own imports.
- `MuFiCokriging` (for co-kriging) may have to be installed from the CRAN archive if `install.packages` can't find it.
- RNAmf 0.1.2 comes from the local tarball: `install.packages("<path>/Code/R/RNAmf_0.1.2.tar.gz", repos = NULL, type = "source")`. Every data-generating R script runs this line itself, with the path found automatically (§7).

### Running

- On Windows, `python` alone resolves to the Windows Store alias unless an env is activated.
- Run drivers as modules from `Python/`, for example `python -m benchmarks.nested.perdikaris` or `python -m al_experiments.alm_park`.
  - Running a driver as a plain file also works (`python benchmarks/nested/perdikaris.py`, or VS Code's Run button): each driver starts with a two-line bootstrap that puts `Python/` on `sys.path`. Keep that bootstrap in new drivers, above their local imports.
  - Every path goes through `paths.py`, so all reads and writes are independent of the CWD: data from `Python/Data/`, plots and AL outputs under `Python/plots/`. The `data_dir` / `plot_dir` strings passed to `run_experiment` and `run_alm_experiment` ("Data/park", "plots/park") are taken relative to `Python/`; absolute paths are used as is.
  - `.vscode/settings.json` (in `Code/`) sets `python.analysis.extraPaths: ["Python"]` so Pylance resolves the imports.
- For a quick check, lower `n_replicates`, `n_restarts` or `n_samples` in a driver, or pass `n_replicates=` to `run_alm_experiment`.

Envs on the `santh` machine (Nov 2026):
- `research` (Python 3.12): no GPy or emukit.
- **`research_modelling` (Python 3.10) is the working env**: numpy 1.26.4, scipy 1.15.3, GPy 1.13.2, emukit 0.5.1. KO, NARGP, GP and MF-RFF all fit and predict there.
  - Its unrelated `pydoe` 1.5.0 wants numpy ≥ 2.2.6, so `pip check` complains, and installing or upgrading `pydoe` would pull numpy 2 back in and break GPy. The project does not use `pydoe`.
- The R packages other than `glue` were not installed yet.

---

## 1. The model (`MF_RFF_Bayes.py`, `rff_linear_algebra.py`, `features_full.py`)

### 1.1 Generative model

Levels run l = 0 (lowest fidelity) to L−1 (target). Each level is a conjugate Bayesian linear model on its own RFF basis φ_l (§1.3). f_l denotes the latent, noise-free level-l function.

- **Level 0:** y_0 = φ_0(x)ᵀ w_0 + ε_0.
- **Linear (Kennedy–O'Hagan) link, `linear[l] = True`:** y_l = φ_l(x)ᵀ β_l + ρ_l f_{l−1}(x) + ε_l. The design row is ψ_l = [φ_l(x), f_{l−1}(x)] (p_l + 1 columns), and ρ_l is one constant scalar.
- **Non-linear (NARGP-style) link, `linear[l] = False`:** y_l = φ_l([x, f̃_{l−1}(x)])ᵀ w_l + ε_l, with f̃_{l−1} = (f_{l−1} − `f_loc[l]`) / `f_scale[l]`.
  - The (d+1)-dimensional frequencies are drawn as independent blocks (d, 1), so the kernel is the product k_x(x, x′)·k_f(f, f′), with ARD bandwidths η_x (d of them) and η_f (one).
  - Unlike the `NARGP.py` baseline, there is no additive k_δ(x, x′) term.

Probabilistic assumptions, exactly as coded:

- **Noise:** ε_l ~ N(0, σ_l² I). It is homoscedastic and independent across levels and points.
- **Weights, conditionally on σ_l² (conjugate NIG), with zero prior mean:**
  - w_l | σ_l² ~ N(0, σ_l²/λ_l · I) for the RFF weights.
  - ρ_l | σ_l² ~ N(0, σ_l² τ), with τ = 1e5 (`self.tau`, nearly flat).
  - Marginally over σ_l², the weights are Student-t.
- **Variance:** one prior per level, on σ_l² only: σ_l² ~ IG(a, b), with a = b = 1e-6 (`a_prior`, `b_prior`).
- **Point-estimated hyperparameters:** λ_l (the "nugget") and log10 η_l (the ARD bandwidths) have no priors. They are type-II ML estimates that minimize the NLML (§1.5).
- **What λ means.** Every φ row has unit norm, since (1/m)Σ(cos² + sin²) = 1. The prior is therefore Var[f_l(x) | σ_l²] = σ_l²/λ_l, plus σ_l² τ f_{l−1}(x)² on linear levels.
  - **λ is the noise-to-signal variance ratio.** The noise-free scripts search log10 λ in (−8, −3). At 5% noise, λ ≈ 0.05² = 2.5e-3, hence `nugget_bounds = (−8, 0)` in the noisy scripts.
  - σ² scales signal and noise together. In noise-free fits, σ² ≈ λ·(signal variance) is tiny, and the uncertainty is carried by the epistemic variance σ²·h. The leverage h is about 0 at the data and goes to 1/λ far from it.
- **Centering:** each y_l is centered by its sample mean `y_mean[l]` before fitting, and the mean is added back in `predict`.
  - The latent trajectories passed between levels (`Z`) are the **centered** ones.
  - The linear link therefore reads y_l − ȳ_l = φβ + ρ(f_{l−1} − ȳ_{l−1}) + ε, and φβ absorbs the offset.
- **Fitting is sequential, as in recursive co-kriging and NARGP.** Data at level l never update the posteriors of the levels below it.
- **Inputs need not be nested across levels.** f_{l−1} at X_l comes from propagating posterior draws, never from looking up y_{l−1}.
  - The R-generated benchmark data *are* nested (`NestedX`), because co-kriging and RNAmf need nested designs.
  - The noisy data from `noisy_data.py` are not nested.

### 1.2 Conjugate NIG posterior (one level, design Ψ of size N × q)

P = diag(`prec`) is the prior **precision**: [λ, …, λ], or [λ, …, λ, 1/τ] on linear levels. The prior mean is μ = 0.

- K = ΨᵀΨ + P (q × q) and M = I_N + ΨP⁻¹Ψᵀ (N × N), with log|M| = log|K| − log|P|.
- γ = K⁻¹(Ψᵀy + Pμ).
- Q_raw = ‖y − Ψγ‖² + (γ − μ)ᵀP(γ − μ) = (y − Ψμ)ᵀM⁻¹(y − Ψμ). The model stores **Q = Q_raw + 2b**, so b_N = Q/2 and a_N = a + N/2.
- σ² | y ~ IG(a_N, Q/2) and w | σ², y ~ N(γ, σ²K⁻¹). The posterior mean of σ² is c := Q/(N + 2a − 2).
- NLML = (N/2) log 2π + ½ log|M| + log Γ(a) − a log b − log Γ(a_N) + a_N log b_N.
- Predictive at x* (design row ψ*):
  - Student-t with 2a_N dof, location ψ*γ and scale² (Q/(N + 2a))(1 + h), where h = ψ*K⁻¹ψ*ᵀ is the leverage.
  - Its variance is c(1 + h) = c·h (weights) + c (noise).

### 1.3 Features and spectral draws (`features_full.py`)

- φ(X) = sqrt(1/m)·[cos(Xωᵀ) | sin(Xωᵀ)], of shape (N, 2m), in a **[cos | sin] column layout**. It is computed by `phi`, or by `phi_from_proj` from a precomputed projection Xωᵀ.
- `base_frequencies(m, d, kernel, seed, method, blocks)` draws unit-scale frequencies of shape (m, d):
  - `gaussian`: z ~ N(0, I).
  - `matern32` / `matern52` (ν = 1.5 / 2.5): z·sqrt(ν/g), with g ~ Gamma(ν, 1) per row. This is a multivariate t with 2ν dof, the Matérn spectral density.
  - `method='sobol'` (default): scrambled Sobol in d dims (plus one Gamma coordinate per block for Matérn), mapped through `norm.ppf` / `gamma.ppf` after clipping to [1e-12, 1−1e-12].
  - `method='mc'`: i.i.d. draws.
  - `method='orthogonal'` or `'orf'`: orthogonal random features (Yu et al. 2016), built from Haar blocks of size d × d with χ(d) row norms. Matérn adds i.i.d. Gamma.
  - `blocks=(d, 1)` (used on non-linear levels) splits the columns into independent blocks with one Gamma variable per block, which gives the product kernel k_x·k_f. It has no effect for the Gaussian kernel, whose coordinates are already independent.
- `scale_frequencies(base, eta)`: ω = base · sqrt(2η), per column.
  - So η = 1/(2ℓ²), and for the Gaussian kernel k(r) = exp(−Σ_j η_j r_j²).
  - Bounds are on **log10 η**. Larger η means a wigglier function.
- `phi_vjp(features, R)` contracts with the Jacobian of φ in the projection u = Xωᵀ. It works from the features themselves, with no extra trig:
  - G[n, j] = ∂(φ(u_n)·R_n)/∂u_nj = sqrt(1/m)(cos u_nj R_n[m+j] − sin u_nj R_n[j]).
  - R of shape (N, 2m) gives (N, m). R of shape (N or 1, 2m, k) contracts k vectors at once and gives (N, m, k).
  - The x-gradient is then G @ ω.

### 1.4 Sizes and constructor

- m_l = `_resolve_m(N_l, feature_mults[l])` = max(2, mult · min(⌊√N_l ln N_l⌋, ⌊N_l/2⌋ − 1)), and p_l = 2m_l. The default is `feature_mults = [3, 6, 6, …]`. The drivers' comment "m_l = _resolve_m(feature_mults[l] * N_l)" is stale.
- **Every configuration in the drivers has p_l > N_l, so the dual (N × N) solver is the one that runs.** Examples:
  - Perdikaris 50/14 with [10, 20]: p = 480 / 240.
  - Park 40/20 with [6, 10]: p = 228 / 180.
  - ALM Perdikaris 13/8 with [10, 20]: p = 100 / 120.
- The primal (p × p) path is correct (it agrees with the dual one to about 1e-13), but the experiments essentially never exercise it.
- m is re-derived from N at every `fit`, so p changes as AL adds points. p also varies per replicate, because `NestedX` changes the lower-level sizes (§3).

`MF_RFF_bayes(d, *, ...)` keyword arguments:

| Argument | Default | Meaning |
|---|---|---|
| `kernel` | `'gaussian'` | `'gaussian'`, `'matern32'`, `'matern52'` |
| `nuggets` | `1e-6` | λ per level (scalar or L values). **Ignored when `tune_nugget=True`**: the search then starts from an LHS pool inside `nugget_bounds` |
| `n_restarts` | 10 | number of L-BFGS-B start candidates (`self.n_starts`) |
| `n_samples` | 500 | S, the number of posterior trajectories (all drivers use 250) |
| `linear` | False | per link: a scalar or L−1 values |
| `tune_nugget` | True | tune log10 λ jointly with log10 η |
| `nugget_bounds` | (−8, 0) | log10 λ box: one tuple, or L tuples |
| `eta_bounds` | (−3.5, 0.8) | log10 η_x box: one tuple, or L entries. Each entry is a tuple (all dims) or a list of d tuples (one per dim). A `None` entry means (−2 − log10 d, log10 5000 − log10 d) |
| `eta_bounds_f` | (−2.5, 0.3) | log10 η_f box for the non-linear links: one entry or L−1 entries. `None` means (−2, log10 500) |
| `feature_mults` | None → [3, 6, …] | L multipliers for m_l |
| `quantiles` | (0.025, 0.975) | predictive interval |
| `seed` | None | root seed (the drivers pass the replicate seed) |
| `freq_method` | `'sobol'` | `'sobol'`, `'mc'`, `'orthogonal'` |
| `n_screen` | 5 | the LHS screening pool has `n_screen · n_restarts` points |
| `local_frac` | 1.0 | number of L-BFGS-B runs = max(3, ⌈local_frac · n_restarts⌉), started from the best pool points |
| `verbose` | True | print p_l per level |

Fixed values: τ = 1e5 and a = b = 1e-6.

### 1.5 `fit(X_list, y_list)` step by step

1. **Validate.** Inputs are lists of length L ≥ 1, and each level needs at least 3 points, finite values and d columns. Per-level arguments are broadcast with `_broadcast_per_level`, and `linear = [False] + links`.
2. **Seeds.** `SeedSequence(seed).spawn(2L+2)` produces `seeds_freq[0..L−1]`, `seeds_opt[0..L−1]`, `seed_fit` and `seed_pred`.
3. **Sizes, priors, frequencies.**
   - Compute m_l and p_l. Set `mu[l]` to zeros and `Sigma_inv[l]` to the precision vector (`_prec_vector`).
   - `base_freq[l]` has shape (m_l, d) on level 0 and on linear levels, and shape (m_l, d+1) with `blocks=(d, 1)` on non-linear levels.
   - Center the y_l.
4. **`_tune_all_levels`: hyperparameters, level by level.** θ = [log10 η (d values, or d+1 on non-linear levels), log10 λ (if `tune_nugget`)].
   - f_{l−1} at X_l is the **deterministic plug-in mean chain** (`predict_plugin`: posterior-mean weights ŵ_k, propagated level by level), not the sampled trajectories. So the tuned η_l and λ_l ignore the uncertainty in f_{l−1}.
   - Linear level: the design is [φ_l(X_l; ω(η)), f̂_{l−1}(X_l)], with the buffer reused across evaluations.
   - Non-linear level:
     - `f_loc[l]` and `f_scale[l]` are the mean and std of the plug-in f̂_{l−1} at **X_{l−1}**, the previous level's own inputs (a std below 1e-8 is replaced by 1).
     - The input is X_eff = [X_l, (f̂_{l−1}(X_l) − loc)/scale].
     - On level 0 and on linear levels, loc = 0 and scale = 1.
   - Objective: the NLML of §1.2, minimized by `_multistart_lbfgsb`:
     - An LHS pool of max(n_restarts, n_screen·n_restarts) points (rng from `seeds_opt[l]`) is scored with the value-only NLML (`score_bayes`).
     - L-BFGS-B then runs from the best k starts (maxiter 300, ftol 1e-12, gtol 1e-7), using the exact gradient from `score_bayes_with_grad`.
     - A `LinAlgError`, `ValueError`, `FloatingPointError` or non-finite value becomes a penalty of 1e10.
     - It returns the best point, falling back to the best pool point. If every pool point failed, it raises `RuntimeError`.
   - Warnings are issued when an η is within 2% of a bound ("check input/output scaling") and when λ sits at its upper bound ("level explained mostly as noise").
   - This step sets `eta[l]`, `omega[l]`, `lam[l]`, `Sigma_inv[l]`, `f_loc[l]`, `f_scale[l]`, and the plug-in weights used by the next level.
5. **`_fit_posteriors`: S coupled trajectories**, with the rng from `seed_fit`.
   - **Level 0:** the exact NIG posterior (`parameter_bayes`) gives `beta0_hat`, `Q0`, `L0` and `phi0_train` (the training design in dual mode, else None). Then S draws are taken: σ²_s ~ IG(a + N_0/2, Q0/2) and w_s ~ N(β̂_0, σ²_s K⁻¹) (`sample_weights_bayes`).
   - **Level l ≥ 1:** `Z_prev = _evaluate_trajectories_at(l−1, X_l)` has shape (N_l, S).
     - Column s propagates trajectory s with **its own** weight draws W[k][s] for all k < l.
     - The level-l posterior of trajectory s is conditioned on that lower-level path. The S columns are therefore joint draws of (f_0, …, f_{L−1}), which is what makes ALC (§2.5) possible.
     - Linear levels use `_fit_level_linear_schur` (§1.6). Non-linear levels use `_fit_single_trajectory_nl`, once per s (§1.7).
   - Stores `W[l]`, `sig2_draws[l]`, `Q[l]` and `level_cache[l]`.

### 1.6 Linear level: `_fit_level_linear_schur` (all S trajectories at once)

Here Ψ_s = [Φ, f_s], P = diag(λ I_p, 1/τ) and F = Z_prev (N, S). Φ is factored **once** (`L_base`): K₀ = ΦᵀΦ + λI on the primal path, or M₀ = I + ΦΦᵀ/λ on the dual path.

- β₀ = K₀⁻¹Φᵀy; u_y = M₀⁻¹y = y − Φβ₀; U = M₀⁻¹F; B_F = K₀⁻¹ΦᵀF, of shape (p, S) (on the dual path, B_F = ΦᵀU/λ).
- a_s = 1/τ + f_sᵀM₀⁻¹f_s is the Schur complement of the ρ block of K_s, so Var(ρ | σ², y) = σ²/a_s.
- ρ̂_s = f_sᵀM₀⁻¹y / a_s and β̂_s = β₀ − B_F,s ρ̂_s.
- Q_s = max(0, yᵀM₀⁻¹y − (f_sᵀM₀⁻¹y)²/a_s) + 2b, a Sherman–Morrison rank-1 update.
- Draws:
  - σ²_s ~ IG(a + N/2, Q_s/2).
  - ρ_s = ρ̂_s + sqrt(σ²_s/a_s)·z.
  - β_s = β̂_s − B_F,s(ρ_s − ρ̂_s) + δ_s, with δ_s ~ N(0, σ²_s K₀⁻¹), because β given ρ has precision K₀.
  - `W[l]` = [β_s, ρ_s], of shape (S, p+1).
- Cache: `L_base`, `phi_cache_base`, `prec_base`, `B_F`, `a_vec`, `Beta_hat`, `rho_hat`.
- Prediction: h_s = φ*K₀⁻¹φ*ᵀ + (z*_s − φ*B_F,s)²/a_s and μ_s = φ*β̂_s + z*_s ρ̂_s, where z*_s is trajectory s's value of f_{l−1}(x*).

### 1.7 Non-linear level: `_fit_single_trajectory_nl` (one NIG solve per trajectory)

- ψ_s(x) = φ(xω_xᵀ + f̃_s(x)ω_fᵀ), with ω_x = ω[:, :d] and ω_f = ω[:, d:]. The x-projection `proj_x = X ω_xᵀ` is computed once per call, and the f part is a rank-1 broadcast (`_nl_design_from_proj`).
- For each s: `parameter_bayes` (dual Cholesky of M_s), then σ²_s, then w_s (a Matheron draw).
  - This runs under joblib with `backend="threading"` and `n_jobs=-1`, with BLAS limited to 1 thread (`threadpool_limits`).
  - Per-trajectory seeds come from `SeedSequence(rng_fit.integers(1e9)).spawn(S)`, so results do not depend on thread scheduling.
- The cache holds lists of length S:
  - `gamma_hat`.
  - `L_chol` (N × N).
  - `psi_train` (N × p in dual mode).
  - `GW` = [γ_s | w_s] (p × 2), so one GEMM gives both μ_s and Z_s.
  - `V` = L_s⁻¹ψ_sP⁻¹ (N × p), the dual leverage factor.
- Memory is about S(N² + 2Np) floats per non-linear level.

### 1.8 `predict(X, level='all'|int, return_samples=False, quantiles=None, random_state=None)`

- **Level 0 (closed form).** With α₀ = N₀ + 2a:
  - mean = φ*β̂₀ + ȳ₀, and h₀ is the leverage.
  - `var_weights` = Q₀/(α₀−2)·h₀, `var_noise` = Q₀/(α₀−2), and `var_prop` = 0.
  - The quantiles are **exact Student-t**: mean ± t⁻¹_{α₀}(q)·sqrt((Q₀/α₀)(1 + h₀)).
- **Level l ≥ 1.** For each trajectory s, with c_s = Q_s/(N_l + 2a − 2), compute the conditional mean μ_s and the leverage h_s.
  - Linear levels: §1.6.
  - Non-linear levels: a threaded loop over s, with h_s = `leverage_from_factor(ψ*_s, V_s, 1/λ)`. h_prior = 1/λ is exact because the rows have unit norm and every precision equals λ.
  - The law of total variance over the S trajectories then gives:
    - `mean` = mean_s μ_s + ȳ_l. This is Rao–Blackwellized, so it is not the sample mean of `samples`.
    - `var_prop` = Var_s(μ_s): the uncertainty propagated from the lower levels (alias `varexp`).
    - `var_weights` = mean_s(c_s h_s): the level's own weight uncertainty.
    - `var_noise` = mean_s c_s, which is constant in x.
    - `var` = the sum of the three, and `expvar` = `var_weights + var_noise`.
    - **`var_epistemic` = `var_prop + var_weights`** = Var[f_l(x)]. This is what ALM uses.
- **Samples**, of shape (N, S): Z_s(x*) = ψ*_s w_s + ε, with ε ~ N(0, `sig2_draws[l][s]`).
  - The w_s are the coupled draws, the same ones propagated to the next level.
  - The noise uses the **same** σ²_s that generated w_s, so each column is an exact draw from that trajectory's Student-t predictive.
  - At l ≥ 1 the quantiles are **empirical** over the S samples.
  - The noise RNG is `default_rng(seed_pred)` unless `random_state` (an int or a Generator) is given, so repeated calls return identical samples.
- **Return values.**
  - `level='all'` returns a dict {l: dict of the keys above, plus `'samples'` if requested}.
  - An int (negative values count from the top) returns `(mean, var, q_lo, q_hi[, samples])`, each of shape (N, 1).
  - Every call stores all levels in `last_predictions_`. Every level is always computed, so a call for level 0 still runs them all.

### 1.9 Exact x-gradients for active learning

- `epistemic_var_grad(X, level=-1)` returns (var (n,), grad (n, d)): `predict`'s `var_epistemic` at `level` and its exact gradient. Everything is fixed after `fit`, so the variance is smooth in x. At level 0, var = c₀h₀ and ∇ = 2c₀ JᵀK⁻¹φ₀.
- `latent_draws_grad(X, level=-1)` returns (Z_list, dZ_list) for k = 0..level. Z_list[k] (n, S) holds the centered, noise-free coupled draws of f_k, and dZ_list[k] (n, S, d) holds their gradients. ALC uses it.
- Both are built on `_level0_draws_grad` and `_level_draws_grad(lvl, X, Z, dZ, need_var)`.
  - `_level_draws_grad` carries (Z, ∇Z) up one level. With `need_var`, it also returns μ_s, H_s and their gradients.
  - Levels above `level` are not evaluated.
- Chain rule, with J the Jacobian of a design row and c_s = Q_s/(α_l − 2):
  - **Leverage:** ∇h = 2JᵀK⁻¹ψ, computed from R = ψK⁻¹ (`kinv_psi` / `kinv_psi_from_factor`). These mirror `leverage_bayes` / `leverage_from_factor`, including the clip at 0: R rows are zeroed there, so the gradient is 0 where h is clipped.
  - **Non-linear:** u_s = ω_x x + ω_f(Z_{l−1,s} − loc)/scale, so ∂u/∂x = ω_x + ω_f ∇Z_{l−1,s}ᵀ/scale. Then ∇μ_s = J_sᵀγ_s, ∇Z_s = J_sᵀw_s and ∇h_s = 2J_sᵀK_s⁻¹ψ_s, with the three vectors contracted in one `phi_vjp` call.
  - **Linear:** ∇μ_s = Jᵀβ̂_s + ρ̂_s∇Z_{l−1,s} and ∇Z_s = Jᵀβ_s + ρ_s∇Z_{l−1,s}. Also ∇H_s = ∇h₀ + 2·diff_s(∇Z_{l−1,s} − JᵀB_F,s)/a_s, with diff_s = Z_{l−1,s} − φB_F,s.
  - **Total:** ∇var_prop = (2/S)Σ_s(μ_s − μ̄)∇μ_s and ∇var_weights = (1/S)Σ_s c_s∇H_s.
- The loop over s is serial, about 25 ms per call for one point at S = 250 on a non-linear level. For a few points that is cheaper than `predict`'s threaded loop.
- **If `predict`'s variance formulas change, change these methods too, and rerun `active_test.py`.**

### 1.10 `rff_linear_algebra.py` reference

`Sigma` arguments are always the prior **precision** diagonal (a vector or a diagonal matrix). `_prior_diag` checks that it is finite and positive and otherwise raises `LinAlgError`, which the optimizer turns into the penalty. Every function switches on p ≤ N (primal) vs p > N (dual), and the two paths agree to about 1e-13.

| Function | Primal (p ≤ N) | Dual (p > N) | Returns |
|---|---|---|---|
| `parameter_bayes(psi, y, mu, Sigma)` | Householder QR of A = [Ψ; √P] and b = [y; √P μ], with signs fixed so that Rᵀ = chol(K) | Cholesky L_M of M; α = M⁻¹(y − Ψμ); γ = μ + P⁻¹Ψᵀα | `(gamma (p,1), Q_raw, L)`, where L is chol(K) (p×p) or chol(M) (N×N) |
| `sample_weights_bayes(gamma_hat, L, sigma_samples, psi_train, Sigma, rng)` | w = γ + σ_s L⁻ᵀz | Matheron's rule: γ₀ ~ N(0, P⁻¹), ε₀ ~ N(0, I), dev = γ₀ − P⁻¹ΨᵀM⁻¹(Ψγ₀ + ε₀) ~ N(0, K⁻¹) | (S, p) draws from N(γ, σ²_s K⁻¹) |
| `leverage_bayes(psi_star, L, psi_train, Sigma)` | ‖L⁻¹ψ*ᵀ‖² | Woodbury: ψ*P⁻¹ψ*ᵀ − ‖L_M⁻¹ΨP⁻¹ψ*ᵀ‖², clipped at 0 | h (N_test,) |
| `dual_leverage_factor(L, psi_train, Sigma)` | — | V = L_M⁻¹ΨP⁻¹ (N×p), so that K⁻¹ = P⁻¹ − VᵀV | V |
| `leverage_from_factor(psi_star, V, h_prior)` | — | h_prior − rowsum((ψ*Vᵀ)²), clipped; one GEMM | h |
| `kinv_psi(psi_star, L, psi_train, Sigma)`, `kinv_psi_from_factor(psi_star, V, prec, h_prior)` | h and R = ψ*K⁻¹ | same, with R rows zeroed where h is clipped | (h, R) |
| `score_bayes(psi, y, mu, Sigma, a, b)` | QR `mode='r'` of the stacked [A, b] (R_ext[p,p]² = Q_raw, no Q factor built) | dual Cholesky | NLML |
| `score_bayes_with_grad(psi, y, mu, Sigma, a, b, X_eff, omega, tune_nugget)` | | | NLML, ∂/∂log10 η (one per column of `omega`), and ∂/∂log10 λ if tuned |

NLML gradients. They are exact; checked against finite differences to about 1e-9 on both paths.
- ∂NLML/∂Ψ = ΨK⁻¹ − (a_N/b_N) r γᵀ, with r = y − Ψγ (envelope theorem on Q).
  - Primal: ΨK⁻¹ = Q_Ψ R⁻ᵀ.
  - Dual: ΨK⁻¹ = M⁻¹ΨP⁻¹ and r = α.
- The gradient is restricted to the first 2m columns (G) and then pushed through the features:
  - ∂NLML/∂U = −G_cos ⊙ Φ_sin + G_sin ⊙ Φ_cos, with U = X_eff ωᵀ.
  - ∂ω_jk/∂log10 η_k = ½ ln10 · ω_jk, so ∂NLML/∂log10 η_k = ½ ln10 Σ_j (X_effᵀ ∂NLML/∂U)_kj ω_jk.
- ∂NLML/∂log10 λ = ln10 · [½ λ tr_{2m}(K⁻¹) − m + ½ (a_N/b_N) λ ‖γ_{2m} − μ_{2m}‖²].
  - λ tr_{2m}(K⁻¹) is ‖Q_qr[N:N+2m]‖²_F on the primal path, and 2m − tr(M⁻¹Φ_{2m}Φ_{2m}ᵀ)/λ on the dual path.
- The dual gradient forms M⁻¹ explicitly (N × N), which is fine at the N used here.

### 1.11 Fitted state

| Attribute | Shape | Meaning |
|---|---|---|
| `L`, `N`, `m`, `p` | int; lists | number of levels; points, frequencies and features per level |
| `linear` | list of bool | `linear[0]` is always False |
| `lam`, `eta`, `omega`, `base_freq` | per level | tuned λ; η of shape (d,) or (d+1,); ω = base·sqrt(2η) of shape (m, d) or (m, d+1) |
| `mu`, `Sigma_inv` | (p [+1], 1), (p [+1],) | prior mean (zeros) and prior **precision** |
| `y_mean` | float per level | output centering |
| `f_loc[l]`, `f_scale[l]` | float per level | standardization of f_{l−1} inside level l (0 and 1 on level 0 and on linear levels) |
| `beta0_hat`, `Q0`, `L0`, `phi0_train` | | level-0 posterior; `phi0_train` is the design in dual mode, else None |
| `W[l]` | (S, p_l [+1]) | coupled weight draws ([β, ρ] on linear levels) |
| `sig2_draws[l]` | (S,) | coupled σ² draws, reused by `predict` for the sample noise |
| `Q[l]` | float at l = 0, (S,) above | Q per trajectory |
| `level_cache[l]` | dict | None at l = 0; the keys of §1.6 / §1.7 above |
| `seeds_freq`, `seeds_opt`, `seed_fit`, `seed_pred` | ints | derived from `seed` |
| `last_predictions_` | dict | all levels of the most recent `predict` |

### 1.12 Notation (as used in code)

| Symbol | Meaning |
|---|---|
| `L`, `lvl`/`l` | Number of levels; 0-based level index in Python. The data files use 1-based `X1..XL`. |
| `d`, `N[l]`, `m[l]`, `p[l]` | Input dimension; points, frequencies and features (p = 2m) at level l. |
| `S` / `samples` | Number of posterior trajectories (`n_samples`). |
| `eta[l]` | ARD bandwidths. Bounds are in **log10**. |
| `omega[l]` | Scaled frequencies. Non-linear levels split them into `omega[:, :d]` (x part) and `omega[:, d:]` (f part). |
| `phi` / `psi` | φ = sqrt(1/m)[cos(Xωᵀ), sin(Xωᵀ)]. ψ is the full design: φ itself, `[φ, f]` (linear), or φ of `[x, f̃]` (non-linear). |
| `lam[l]` | The "nugget": prior **precision** of the RFF weights relative to σ², i.e. the noise-to-signal ratio. |
| `Sigma`, `Sigma_inv`, `prec` | **All three name the prior precision diagonal** `[λ,…,λ(, 1/τ)]`, not a covariance. |
| `K`, `M` | Primal K = ψᵀψ + diag(prec). Dual M = I + ψ diag(1/prec) ψᵀ. |
| `L` (factor) | Cholesky factor of K (p×p, primal) **or** of M (N×N, dual). `psi_train` / `phi_cache_base` / `phi0_train` being None (primal) or the training design (dual) tells which. |
| `gamma`/`gamma_hat`, `beta0_hat` | Posterior mean of the weights. |
| `Q` | Posterior quadratic form + 2b, so that b_N = Q/2. |
| `h`, `H_mat`, `H` | Leverage ψ*K⁻¹ψ*ᵀ. |
| `Z`, `Z_prev`, `Z_curr`, `F_prev` | (N, S) matrices of **centered, noise-free** latent trajectories. |
| `c`, `c_s` | Q/(N + 2a − 2), the posterior mean of σ². |

### 1.13 Gotchas when editing the core

- **The feature layout is load-bearing.** `score_bayes_with_grad`, `phi_vjp`, and the h_prior = 1/λ shortcut in `predict` and `_level_draws_grad` all assume that:
  - the first 2m columns of ψ are `[cos | sin]`, with precision λ;
  - any extra column (the ρ column) comes after them and is excluded from the η and λ gradients;
  - rows have unit norm;
  - the columns of `X_eff` line up with the columns of `omega`.

  Adding a bias or intercept column, or changing the normalization, breaks values and gradients silently.
- **Primal vs. dual switch.** `parameter_bayes`, `sample_weights_bayes`, `leverage_bayes` and `kinv_psi` branch on p ≤ N. Callers must pass the training design (`psi_train`) exactly when p > N and None otherwise, and the meaning of the `L` factor flips with it.
- **Per-level arguments** go through `_broadcast_per_level`:
  - Length L: `nuggets`, `feature_mults`, `eta_bounds`, `nugget_bounds`.
  - Length L−1: `linear`, `eta_bounds_f`.
  - Level 0 is never linear.
  - For bounds, a 2-tuple of scalars is one box for every level, and a list or tuple of length L is per level.
  - Trap 1: write a single box as a **tuple**. With L = 2, a list `[lo, hi]` is read as per-level and fails.
  - Trap 2: with L = 2, a list of two pairs is read per **level**, not per dimension. Per-dimension boxes need a list of L lists of d pairs, as in `currin.py`.
- **No incremental update.** `fit` re-tunes and re-samples everything, and AL refits from scratch every iteration.
- **Tuning vs. posterior.** The hyperparameters see the plug-in mean chain, while the posteriors see the sampled trajectories. Don't "fix" one to match the other without intent.
- **`predict` and the gradient methods must agree.** `epistemic_var_grad` and `latent_draws_grad` reproduce `predict`'s formulas, so edit them together and rerun `active_test.py` (value vs. `predict`, gradient vs. finite differences).
- **Cost.**
  - `predict` always computes every level, and the non-linear levels cost O(S·n·p·N).
  - The sample CRPS in `experiment.py` builds an (n_test, S, S) array: 1000 × 250² floats is about 500 MB.
  - `alm_experiment._crps_samples` computes it in chunks.

---

## 2. Active learning (`active_learning.py`)

### 2.1 The loop: `active_learning_loop`

```
active_learning_loop(make_model, X_list, y_list, oracles, costs, budget, bounds=None,
                     X_test=None, y_test=None, n_screen=256, n_starts=3, seed=None,
                     evaluate=None, criterion='alm', X_ref=None, nested=True,
                     allow_repeats=False, **opt_kw)      # opt_kw: min_sep, jac, fd_step, maxiter
    -> (model, X_list, y_list, history)
```

Each iteration:

1. **Refit from scratch:** `model = make_model().fit(X_list, y_list)`, hyperparameters included. `make_model` takes no arguments and returns a new, unfitted `MF_RFF_bayes`.
2. **Record** `{'spent', 'N', 'rmse'}`, where `rmse` is the target-level RMSE if `X_test` is given and None otherwise. The entries returned by `evaluate(model)` are added and overwrite these; `alm_experiment` supplies `rmse` and `crps` this way.
3. **Find the affordable levels:** those with run cost C_l ≤ budget − spent, where C_l = c_0 + … + c_l (`nested=True`) or c_l (`nested=False`). If no level is affordable, stop.
4. **Decide** with `alm_select` or `alc_select`, using `allowed = affordable` and `rng = default_rng(seed)`. There is one generator for the whole loop, so every iteration draws a fresh screening pool. The record gains `level` (0-based), `x`, `crit` (L,) and `x_best` (L, d).
5. **Evaluate** x at levels 0..l (nested) or at level l only.
   - Without `allow_repeats`, a level whose design already holds x is skipped and **not charged**. The match is `np.isclose` on every coordinate, with the default rtol/atol.
   - `spent += c_k` for each level actually evaluated.
   - If no level was added, the loop prints a message and stops, since refitting would repeat the same decision forever.
   - With `allow_repeats=True` (noisy oracles), x is evaluated again.
6. **Repeat.** The initial design is free. `history` has one record per fit; the last record has no `level`.

Notes:
- Affordability is tested with the full C_l, but the charge can be smaller, because lower levels that already hold x are free. So `spent` may not land exactly on the budget.
- `bounds` defaults to the unit cube, which every experiment uses.
- For ALC, `X_ref` defaults to an LHS of 100·d points seeded with `seed`.
- `oracles[l]` receives a (1, d) array.

### 2.2 Level choice

- crit[l] = max_x A_l(x) / (C_l if nested else c_l), where A_l is the criterion at level l.
- Levels that are not allowed get −inf and are not optimized.
- The chosen level is the argmax. Ties go to the lowest level, as with R's `which.max`.
- This is RNAmf's cost-normalized rule (Heo & Sung; `ALM_RNAmf`, `ALC_RNAmf`).

### 2.3 Inner maximization: `_multistart_max`

- **Screen.** `candidate_set(d, n_screen, seed=rng)` draws a scrambled Sobol pool (n_screen should be a power of 2), scaled to the box and drawn fresh at each decision. The criterion is evaluated on the whole pool in one vectorized call:
  - ALM: a single `predict(level='all')` serves every level.
  - ALC: `alc_values` on draws precomputed for levels 0 up to the highest allowed level.
- **Starts.** Pool points are taken greedily in decreasing criterion order, and a point is kept only if it is at least `min_sep` from every start already kept (Euclidean distance, in box-normalized units), up to `n_starts`. Without this, in low d, all starts would sit on the same peak.
- **Local search.** L-BFGS-B minimizes −log A(x) with the exact gradient −∇A/A (`jac=True`, `maxiter=100`) inside the box. The log keeps the scale manageable, since the criteria span orders of magnitude.
- **Result.** The best of the local results and the best pool point, compared on the original scale and clipped to the box. Because the pool point is a fallback, the result is never worse than the screen.
- **Floor.** `_VAR_FLOOR = 1e-300`. Where A ≤ the floor (the dual leverage clips to 0 in densely sampled regions), the log is floored and the gradient is set to 0.
- **`jac='fd'`.** Central differences with step `fd_step` × box width. The 2d + 1 rows [x, x ± h e_j] go into **one** `predict` / `_evaluate_trajectories_at` call. Use it for checks, or for models without analytic gradients.

### 2.4 ALM (MacKay): `alm_select`

- A_l(x) = Var[f_l(x)], the `var_epistemic` at level l (= `var_prop + var_weights`).
- Its value and exact gradient come from `epistemic_var_grad`, via `_log_var_and_grad`.
- Observation noise is left out, because it does not shrink with more data.
- Each level's variance is in its own output scale, as in RNAmf's `ALM_RNAmf` (σ*²_l(x)/C_l).
- At level 0 the variance is closed form (c₀h₀). At l ≥ 1 it is an average over the S trajectories.

`alm_select(model, costs, bounds, allowed=None, n_screen=256, n_starts=3, rng=None, nested=True, **opt_kw)` returns `(level, x, crit, X_best)`. `maximize_epistemic_var(model, level, bounds, X_pool, v_pool, ...)` maximizes a single level.

### 2.5 ALC (Cohn): `alc_select`

A_l(x) = R_l(x) is the reduction in the target level's epistemic variance, averaged over a reference set X_ref, from observing y_k(x) at the observed levels k ∈ obs(l). obs(l) is {0..l} for nested acquisition and {l} for single-level acquisition.

R_l(x) is computed by **Gaussian conditioning on the coupled draws**. Let B (S, k_obs) be the centered draws [Z_k(x)] and A (n_ref, S) the centered target-level draws on X_ref. Then:

- Σ(x) = BᵀB/S + diag(σ̄²_k) is the covariance of the observations at x, with σ̄²_k = mean_s `sig2_draws[k][s]` (`_noise_vars`).
- c(x′) = A(x′)B/S = Cov(f_top(x′), y_obs(x)).
- R_l(x) = mean_{x′} c(x′)Σ⁻¹c(x′)ᵀ = tr(Σ⁻¹BᵀGB)/S², with G = AᵀA/n_ref.
  - G (S × S) comes from `alc_reference` and is computed **once per decision**.
  - So the cost per candidate does not depend on n_ref.

Gradient (`_alc_value_and_grad`; dB comes from `latent_draws_grad`, centered):
- ∂R/∂x_j = (2/S²)tr(Σ⁻¹BᵀG ∂B_j) − (2/S)tr(P Bᵀ∂B_j), with P = Σ⁻¹MΣ⁻¹ and M = BᵀGB/S².
- It is implemented as W = (2/S²)GBΣ⁻¹ − (2/S)BP and ∇R = Σ_{s,k} W_sk ∂B_sk.

Further notes:
- `alc_values(Z_list, level, G, noise, levels=None)` evaluates R at n points at once; it is used for pool screening and grid checks.
- The criterion is exact for a Gaussian model (Cohn's criterion) and an approximation through non-linear links. The hyperparameters stay fixed.
- Monte Carlo error in S: about 20% at S = 250 against the single-level closed form, and about 2% at S = 4000.
- Contrast with RNAmf's `ALC_RNAmf`: it imputes `mc.sample` = 100 outputs at x, does a rank-1 update of each GP inverse and recomputes the predictive variance on Xref, with no gradients (numerical `optim`). Here everything comes in closed form from the draws the model already holds.

`alc_select(model, costs, bounds, X_ref, allowed=None, n_screen=256, n_starts=3, rng=None, min_sep=0.05, jac='analytic', fd_step=1e-5, maxiter=100, nested=True)` returns `(level, x, crit, X_best)`.

### 2.6 Nested vs. single-level acquisition

- `nested=True` (the default, for nested designs such as the R data): acquiring level l evaluates x at levels 0..l at cost C_l, and ALC conditions on y_0(x), …, y_l(x).
- `nested=False` (for non-nested designs such as the noisy data): x is evaluated at level l only, at cost c_l. ALC conditions on y_l(x) alone (`levels=[l]`), and ALM changes only through the cost.
- `allow_repeats=True` (for noisy oracles): a point already in the design is evaluated again, since a replicate brings a new noisy observation.
- The defaults (`nested=True`, `allow_repeats=False`) reproduce the original behavior bit for bit.

### 2.7 Checks: `active_test.py`

- Setup: the 1-D two-level Perdikaris problem. The initial design is an LHS of 13 points, with its first 8 at level 1 (seed 0). Costs are [1, 3] and the budget is 80. The model is non-linear, with `n_restarts=5` and `feature_mults=[6, 10]`.
- Checks, on 20 points that include the box edges:
  - `epistemic_var_grad` vs. `predict`'s `var_epistemic`, and vs. finite differences;
  - the ALC gradient vs. finite differences.
- It then runs ALM and ALC and compares the L-BFGS-B maximum at each level with the maximum over a 4001-point grid.
- Figures are saved to `Python/plots/perdikaris/active_learning_{alm,alc}.png` (`paths.PLOTS`).
- It does not need GPy. Rerun it after touching `predict`, the gradient methods or the criteria.

### 2.8 Gotchas

- **Keep R's `alm_select()` (in `../R/ALM <fn>.R`) in sync with Python's `alm_select`.** They share:
  - the screening-pool size;
  - `n_starts` starts at least `min_sep` apart;
  - L-BFGS-B with `maxit` 100;
  - choosing among affordable levels only;
  - nested acquisition;
  - the skip-a-level-that-has-x rule and the stop rule (R's `in.design` uses numpy.isclose's tolerance).

  They differ by design: R screens with `randomLHS` and uses `optim`'s numerical gradients on RNAmf's `obj.ALM_level_l`, while Python screens with scrambled Sobol and uses exact gradients.
- `crit` values are per unit cost. The maximized criterion value is `crit[l] * C_l`.
- The model is refit from scratch at every iteration, so p and the tuned hyperparameters change as N grows. `alm_experiment` uses the same `seed` throughout a replicate.

---

## 3. Baselines

`experiment.py` calls `fit(X_list, y_list)` and `predict(X, return_ci=True) → (mean, var, lower, upper)` at the target level. Every multi-fidelity baseline must implement that interface.

- **`NARGP.py`** supports L levels (Perdikaris et al. 2017).
  - Level 0 is a GP on x. Level l is a GP on [x, m_{l−1}(x)] with kernel k_x(x)·k_f(f) + k_δ(x), ARD on x. Training uses the deterministic plug-in mean chain m.
  - Each GP is first optimized with the noise fixed at 1% of Var(y) (500 iterations). The noise is then freed and `optimize_restarts(n_restarts, bfgs)` runs.
  - Prediction propagates `nsamples` (default 1000) Monte Carlo chains with latent (`predict_noiseless`) per-point draws, then applies the law of total variance at the target level (`level=-1` by default, noise included at the target).
  - `seed` controls only the propagation RNG; GPy restarts use the global `np.random`. Rows are predicted in chunks of 50 000.
- **`KO_model.py`** supports L levels. It is one joint Kennedy–O'Hagan AR1 GP built from emukit's `LinearMultiFidelityKernel`: L sub-kernels, L−1 unconstrained ρ, and one Gaussian noise term per level. The fidelity index is the last input column.
  - Prediction is the exact Gaussian posterior at `level` (default −1), with that level's noise included through `Y_metadata`.
  - Bounds: length-scale and variance in [1e-3, 10], noise in [1e-6, 1]. Warm starts are 0.1 and 1.0.
  - With L = 2 it reproduces the original two-level code bit for bit, because the constraints and warm starts are applied in the same grouped order. Keep that order if you edit it.
  - `seed` seeds the global `np.random` for the duration of `fit` only, then restores the previous state.
  - The bounds assume roughly unit-scale outputs, so use `"standardize": True`. Without it, coverage on large-range targets such as Branin is poor.
- **`One_shot_GP.py`** (`GP`) is a single-fidelity GPy regression on the target level only. It uses an ARD kernel, length-scale and variance in [1e-3, 10], noise in [1e-6, 0.1], and `optimize_restarts`. It has no seed argument.
- **RNAmf** is the R package 0.1.2 in `../R/RNAmf_0.1.2.tar.gz` (Heo & Sung, arXiv:2309.11772).
  - It supports 2 or 3 levels and requires nested designs (`checknested`).
  - It is recursive: fit1 = GP(X1, y1), fit2 = GP([X2, μ₁(X2)], y2) on the plug-in mean, and fit3 likewise.
  - Each GP uses a separable kernel (`sqex` exp(−Σ(x−x′)²/θ_j), or Matérn 1.5 / 2.5) on inputs standardized by `scale_inputs`. The nugget is fixed at g = sqrt(eps) ≈ 1.5e-8, the mean is a constant μ̂ (`constant=TRUE`, as used here), τ̂² is profiled out, and θ is fitted by L-BFGS-B within data-driven bounds.
  - `predict` returns closed-form moments of the top GP integrated over f_{l−1}(x) ~ N(μ, σ²) (exact for sqex, and for Matérn via `zetafun`), split into `VE` (propagated) and `EV` (own level).
  - AL functions:
    - `ALM_RNAmf`.
    - `ALC_RNAmf`.
    - `ALD_RNAmf`: the VE/EV decomposition.
    - `ALMC_RNAmf`: x chosen by ALM at the top level, the level by ALC.
  - `NestedX(n, d)` builds a maximin LHS per level, then replaces the lower-level point nearest to each higher-level point with that point. So X_{l+1} is the last n_{l+1} rows of X_l.
  - With `NestedX`, **N_l at the lower levels can exceed the request**. In the data: Borehole N₁ = 82–92 for 80, Park 40–45, Currin 20–23, Perdikaris 50–52.
- **Co-kriging** runs in R via `MuFiCokriging::MuFicokm` (Le Gratiet's recursive co-kriging).
  - Settings: `covtype="gauss"`, constant trends and `lower=eps`.
  - Perdikaris uses `nugget=1e-8` and `lower=1e-4` and builds `NestedDesign` directly, because re-substituting with `NestedDesignBuild` breaks in 1-D.
  - It is predicted with `"UK"`, with an interval of ±1.96 sd.

## 4. Nested, noise-free benchmarks (`experiment.py` and the drivers)

- `run_experiment(d, N_list, n_replicates, base_seed, data_dir, plot_dir, model_names, kernel)` calls `run_replicates`, which calls `single_run` for replicate seeds base_seed + 1 … base_seed + n_replicates. With base_seed = 0 it reads rep1..rep100.
- **Model dispatch.** The model type comes from `"type"` or from the name prefix (`KO`, `NARGP`, `GP`, `MF_RFF_Bayes_l`, `MF_RFF_Bayes_nl`, `rna`, `cokm`, followed by `_` or `-`).
  - MF-RFF is built by `_build_bayes`. Its defaults are d, `seed` = the replicate seed, `kernel`, `nuggets=1e-6` and `n_restarts=5`. `linear` is inferred from the name (`_l` or `linear`, and no `nl`) unless given.
  - Every other key goes to the constructor, and unknown keys raise. `"type"` and `"standardize"` are reserved.
  - `"standardize": True` (KO, NARGP and GP only) wraps the model in `_Standardized`. It fits on y standardized by one mean and sd pooled over all levels, and maps predictions back.
  - `"rna"` and `"cokm"` just load the R results.
- **Metrics per replicate.**
  - RMSE and R² of the mean, against the noise-free `yhigh_test`.
  - CRPS: from samples for MF-RFF (`crps_ensemble`), Gaussian for the other models (`crps_gaussian`).
  - Coverage of [lower, upper]:
    - MF-RFF: the empirical 2.5% / 97.5% sample quantiles.
    - GPy models and co-kriging: ±1.96 sd.
    - RNAmf: ±1.96·sqrt(σ²·n/(n−2)), with n the top-level size.
  - Time: the Python fit + predict time, or the time R recorded.
  - RNAmf's saved `rna_sigma2` is **not** inflated by n/(n−2), while `rna_lower` and `rna_upper` are. So its CRPS and its coverage use slightly different spreads.
  - If the folder has `yhigh_test_noisy.npy`, CRPS and coverage are computed against it (every model's predictive includes the noise), while RMSE and R² still use the noise-free outputs.
  - `y_test` is flattened on load; the Perdikaris `yhigh_test` has shape (n, 1).
- **Plots** (`plotting.plot_boxplot`) are saved under `Python/<plot_dir>/` (`paths.under_root`), whatever the CWD: `rmse_boxplot_high_…`, `r2_boxplot_high_…`, `crps_boxplot_high_…`, `coverage_boxplot_high_…` (with a 0.95 reference line) and `timing_…`. Filenames carry the kernel, N (the target size), the number of replicates, and a leftover `criterion=nlpd` tag.
- **`utils.py`:**
  - `timer` is a context manager that stores the elapsed seconds.
  - `crps_ensemble` computes E|X − y| − ½E|X − X′| over all sample pairs; its memory scales as n_test·S².
  - `crps_gaussian` is the closed form, with the variance floored at 1e-12.
  - `directory_build` is legacy and unused.
- **Data layout.** The current R output uses the stacked layout:
  - `X{l}.npy` / `y{l}.npy`, with column 0 = replicate index.
  - `{rna,cokm}_{mean,sigma2,lower,upper}.npy` of shape (n_rep, n_test); row i−1 is replicate i.
  - `{rna,cokm}_time.npy` of shape (n_rep,).
  - The legacy per-replicate layout (`X{l}_{N}_rep{i}.npy`, …) is still read as a fallback. `pack_legacy_folder` / `pack_all_legacy` convert legacy folders and verify that both loaders agree bit for bit.

| Driver | d | Levels | `N_list` (data folder) | MF-RFF `feature_mults` | Models run |
|---|---|---|---|---|---|
| `perdikaris.py` | 1 | 2 | 50, 14 | [10, 20] | `MF_RFF_Bayes_l`, `MF_RFF_Bayes_nl`, `rna` |
| `currin.py` | 2 | 2 | 20, 10 | [6, 10] | same (per-dimension η boxes) |
| `park.py` | 4 | 2 | 40, 20 | [6, 10] | same |
| `borehole.py` | 8 | 2 | 80, 40 | [6, 10] | same |
| `branin.py` | 2 | 3 | 20, 15, 10 | [6, 10, 10] | same; `Data/branin/` does not exist yet, so run `../R/branin rnamf-cokr.R` first |

- Common settings in all drivers: 100 replicates, `base_seed=0`, `n_samples=250`, `n_restarts=10`, `tune_nugget=True`, `nugget_bounds=(-8, -3)` per level, `kernel='gaussian'`, and plots in `plots/<fn>/`.
- KO, NARGP, GP and `cokm` are supported but are not in the current `models_to_run`.
- Some driver comments are stale: `eta_bounds1 / eta_bounds2`, the `_resolve_m` formula, and "nuggets used as initial value".

Test functions, low → high fidelity. These are the R definitions; `harness/test_functions.py` mirrors Park and Perdikaris where AL and the noisy data need oracles.
- **Perdikaris** (d = 1): f_0 = sin(8πx), f_1 = (x − √2) sin²(8πx). The test set is a 1000-point grid on [0, 1] (`x_test.npy` is 1-D).
- **Currin** (d = 2): f_1 is the Currin et al. (1988) exponential. f_0 is the mean of 4 evaluations at (x₁ ± 1/20, x₂ + 1/20 or max(0, x₂ − 1/20)).
- **Park** (d = 4): f_1 = `park91a`, and f_0 = `park91alc` = (1 + sin(x₁)/10)·f_1 − 2x₁ + x₂² + x₃² + 0.5. x₁ = 0 is replaced by sqrt(eps).
- **Borehole** (d = 8): f_1 is the borehole flow rate. f_0 replaces 2π by 5 and the 1 in the denominator by 1.5. [0, 1]⁸ is mapped to the physical ranges.
- **Branin** (d = 2, 3 levels): f_2 is Branin, and f_1, f_0 are shifted and rescaled variants (from the RNAmf paper). [0, 1]² is mapped to [−5, 10] × [0, 15].
- Test sets are `maximinLHS(1000, d)` with `set.seed(0)` (Perdikaris uses the grid). Replicate i uses `set.seed(i)` and `NestedX`.

## 5. Noisy, non-nested benchmarks (Python only)

- **`noisy_data.make_noisy_data(out_dir, funcs, d, N_list, n_rep, noise_frac=0.05, seed=0, n_test=1000, n_sd=100_000, al_settings=None)`:**
  - Each replicate and level gets an independent random LHS, so the design is not nested.
  - Outputs are y_l = f_l(x) + σ_l ε, with σ_l = `noise_frac` · sd(f_l) and sd estimated on an LHS of `n_sd` points.
  - Test set: one fixed LHS, with noise-free `yhigh_test.npy` and one noisy draw `yhigh_test_noisy.npy`.
  - Also written: `noise_sd.npy` (L,), `noise_meta.npy` ([d, N_1..N_L, n_rep, noise_frac, seed, n_test, n_sd]) and, optionally, `alm_settings.npy`. Everything uses the stacked layout.
  - Seeds: `SeedSequence(seed).spawn(n_rep + 2)`. Child [0] is for the sd estimate, [1] for the test set, and [1 + r] for replicate r.
  - An existing folder is reused only if its meta (and AL settings) match; otherwise it raises `FileExistsError`. `noise_meta.npy` is written last and marks the folder as complete.
- **`noisy_perdikaris.py` / `noisy_park.py`:**
  - Data: `Data/<fn>_noisy/50_14` and `40_20`, 100 replicates, `noise_frac=0.05`, data seed 0. σ is [0.0354, 0.0184] for Perdikaris and [0.2524, 0.2374] for Park.
  - Models: NARGP, KO and One-shot GP (all with `"standardize": True`), plus MF-RFF `_l` / `_nl`. MF-RFF uses the noise-free settings except `nugget_bounds=(-8, 0)`, since λ ≈ 2.5e-3 at 5% noise.
  - The oracles are imported from `harness/test_functions.py`.
  - Plots go to `plots/<fn>_noisy/`.

## 6. Active-learning experiments

**`alm_experiment.run_alm_experiment(d, N_list, oracles, model_names, data_dir, plot_dir, n_replicates=None, kernel='gaussian', jac='analytic', criteria=('alm',), include_rna=True, out_name="alm", n_ref_per_dim=100, nested=True, allow_repeats=False)`:**
- **Inputs.** It reads `Data/<fn>/alm_<N1>_<N2>/`: `alm_settings.npy` (costs, budget, n_screen, n_starts, min_sep), the test set, and optionally RNAmf's history. It runs replicates 1..n; by default n is everything available, counted from `rna_alm_history.npy` or, with `include_rna=False`, from `X1.npy`.
- **Per replicate:**
  1. Load the initial design.
  2. Check the Python oracles against the stored outputs: `allclose` with rtol 1e-10 for noise-free data, or within 6σ_l for noisy data.
  3. For every (model, criterion) pair, run `active_learning_loop` with:
     - `make_model = _build_bayes(name, params, d, seed=rep, …)`, with `linear` taken from a name ending in `_l` unless given;
     - `seed=rep`;
     - a fixed ALC reference set: an LHS of `n_ref_per_dim·d` points with seed 0, the same for every replicate and criterion;
     - `evaluate` = the target-level RMSE against the noise-free test outputs, plus the sample CRPS (in chunks of 100) against `yhigh_test_noisy` when present.
- **Noisy folders** (those with `noise_sd.npy`): the oracles return f_l(x) + σ_l ε, with ε from `default_rng([rep, 1])`. The generator is recreated for each run, so every criterion sees the same noise stream.
- **RNAmf curves** come from the R history, with Gaussian CRPS against the noise-free test outputs.
- **Output** goes to `<plot_dir>/<out_name>/`:
  - `alm_comparison.png`: RMSE and CRPS vs. cost as median and IQR step curves on a 401-point cost grid, plus boxplots at the budget, all on a log scale.
  - `alm_results.npz`: `grid`, `costs`, `budget`, `methods`, `<m>_{rmse,crps}_on_grid`, `<m>_{rmse,crps}_final`, `<m>_acquisitions`, `<m>_time` and `<m>_rep<r>_{spent,rmse,crps,levels}`.
  - Method names are the model name when there is one criterion, and `"<name> (ALM)"` / `"<name> (ALC)"` when there are several.
  - The console prints the median [IQR] at the budget, the mean number of acquisitions per level, and the time per replicate.

| Driver | Data | Criteria | RNAmf | Acquisition | Replicates | Output |
|---|---|---|---|---|---|---|
| `alm_perdikaris.py` | `Data/perdikaris/alm_13_8` | ALM | yes | nested | 10 | `plots/perdikaris/alm/` |
| `alm_park.py` | `Data/park/alm_40_20` | ALM | yes | nested | 10 | `plots/park/alm/` |
| `alc_perdikaris.py` | `Data/perdikaris/alm_13_8` | ALM, ALC | no | nested | 10 | `plots/perdikaris/alc/` |
| `alc_park.py` | `Data/park/alm_40_20` | ALM, ALC | no | nested | 10 | `plots/park/alc/` |
| `al_noisy_perdikaris.py` | `Data/perdikaris_noisy/alm_13_8` | ALM, ALC | no | single level, repeats | 100 | `plots/perdikaris_noisy/al/` |
| `al_noisy_park.py` | `Data/park_noisy/alm_40_20` | ALM, ALC | no | single level, repeats | 100 | `plots/park_noisy/al/` |

- **Model.** Only `MF_RFF_Bayes_nl`.
  - The noise-free runs use the settings in `alm_<fn>.py`, which are those of `perdikaris.py` / `park.py` with `nugget_bounds=(-8, -3)`.
  - The noisy runs use `models_to_run["MF_RFF_Bayes_nl"]` from `noisy_<fn>.py`, with `nugget_bounds=(-8, 0)`.
- **AL settings** `[c_1, c_2, budget, n_screen, n_starts, min_sep]`: Perdikaris [1, 3, 50, 256, 3, 0.05] and Park [1, 3, 30, 256, 3, 0.05].
  - The R ALM scripts write the noise-free ones.
  - The noisy drivers hard-code the same values and write their data with `make_noisy_data(..., seed=DATA_SEED + 1, al_settings=...)`.
- **Order of runs.** Run `../R/ALM <fn>.R` before `alm_<fn>.py` / `alc_<fn>.py`. The noisy AL drivers need no R.

## 7. R side: data and R baselines

- **Never edit the R scripts or their outputs unless the user explicitly asks.** Python only reads them.
- **No hard-coded paths.** The data-generating R scripts (`<fn> rnamf-cokr.R`, `perdikaris rnamf.R`, `ALM <fn>.R`) start with an `r_dir <- local({...})` block that finds their own folder, `Code/R/`. It works with `Rscript` (decoding its `~+~` encoding of spaces), `source()` and RStudio; in a plain console it falls back to the working directory, and stops with a clear message unless that folder contains `RNAmf_0.1.2.tar.gz`. They install `file.path(r_dir, "RNAmf_0.1.2.tar.gz")` and write to `data_root <- file.path(dirname(r_dir), "Python", "Data")`, so they run unchanged on any machine. Keep that block in new data-generating scripts.
- **`../R/<fn> rnamf-cokr.R`** (branin, borehole, currin, park, perdikaris):
  - Runs 100 replicates of `NestedX` designs, with RNAmf (`kernel="sqex"`, `constant=TRUE`) and co-kriging fits.
  - Writes to `Data/<fn>/<N1>_..._<NL>/` in the stacked layout, with a checkpoint every 10 replicates.
  - `perdikaris rnamf-cokr.R` currently writes `13_8`. `perdikaris rnamf.R` is the RNAmf-only variant for `50_14`.
- **`../R/ALM Perdikaris.R`, `../R/ALM Park.R`:** RNAmf with ALM, 10 replicates. They write `Data/<fn>/alm_<N1>_<N2>/`:
  - `X{l}.npy`, `y{l}.npy`: the initial nested designs.
  - `x_test.npy`, `yhigh_test.npy`: one fixed test set.
  - `alm_settings.npy`: the AL settings.
  - `rna_alm_history.npy`: one row per RNAmf fit, `rep, step, spent, N_1..N_L, level (1-based, NaN on the last fit), x_1..x_d`.
  - `rna_alm_mean.npy` / `rna_alm_sigma2.npy`: test predictions, one row per history row.
  - `rna_alm_time.npy`.
  - Their `alm_select()` mirrors Python's (§2.8).
- **Unused originals.** `../R/ALC <fn>.R` (the user's originals, using `ALC_RNAmf`) and `../R/ALM Blade.R` (needs MATLAB) are unchanged and not used.
- **Matching folders.** `N_list` in a Python driver must match an existing folder name, and replicate r reads `rep = base_seed + r`.
