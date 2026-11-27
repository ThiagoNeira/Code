# noisy_data.py
"""
Noisy multi-fidelity data on non-nested designs, in the stacked layout read by experiment.py and
alm_experiment.py.

For replicate r = 1..n_rep and level l, X_l is a random Latin hypercube of N_l points in [0, 1]^d
(random positions inside the cells), drawn independently across levels and replicates, so the
design is not nested. The outputs are
    y_l = f_l(X_l) + sigma_l * eps,   eps ~ N(0, 1) independent,   sigma_l = noise_frac * sd(f_l),
with sd(f_l) estimated once on a Latin hypercube of n_sd points. The test set is one fixed Latin
hypercube of n_test points: noise-free target outputs (yhigh_test.npy) and one noisy draw of them
(yhigh_test_noisy.npy).

Files written to out_dir:
  X{l}.npy, y{l}.npy      stacked over replicates, column 0 = replicate index (l is 1-based)
  x_test.npy (n_test, d), yhigh_test.npy (n_test,), yhigh_test_noisy.npy (n_test,)
  noise_sd.npy            (L,) sigma_l
  noise_meta.npy          [d, N_1, ..., N_L, n_rep, noise_frac, seed, n_test, n_sd]
  alm_settings.npy        only with al_settings: [c_1, ..., c_L, budget, n_screen, n_starts, min_sep]
An existing folder is reused if its noise_meta.npy (and alm_settings.npy) match; otherwise an error
is raised rather than overwriting it.
"""
import os
import numpy as np
from scipy.stats import qmc


def _lhs(d, n, rng):
    return qmc.LatinHypercube(d=d, scramble=True, seed=rng).random(n)


def make_noisy_data(out_dir, funcs, d, N_list, n_rep, noise_frac=0.05, seed=0, n_test=1000,
                    n_sd=100_000, al_settings=None):
    """
    out_dir     : folder to write (e.g. Data/perdikaris_noisy/50_14).
    funcs       : list of L noise-free functions, funcs[l](X) -> (n,) for X of shape (n, d); l = 0 lowest.
    N_list      : points per level.
    n_rep       : replicates 1..n_rep.
    noise_frac  : sigma_l / sd(f_l).
    seed        : root seed (sd estimate, test set and every replicate derive from it).
    al_settings : optional [c_1, ..., c_L, budget, n_screen, n_starts, min_sep] for alm_experiment.
    Returns sigma (L,).
    """
    L = len(funcs)
    if len(N_list) != L:
        raise ValueError(f"N_list has {len(N_list)} entries for {L} functions.")
    meta = np.array([d, *N_list, n_rep, noise_frac, seed, n_test, n_sd], dtype=float)
    settings = None if al_settings is None else np.asarray(al_settings, dtype=float)

    meta_path = os.path.join(out_dir, "noise_meta.npy")
    if os.path.exists(meta_path):
        same = np.array_equal(np.load(meta_path), meta)
        if settings is not None:
            s_path = os.path.join(out_dir, "alm_settings.npy")
            same = same and os.path.exists(s_path) and np.array_equal(np.load(s_path), settings)
        if not same:
            raise FileExistsError(f"{out_dir} holds data generated with other settings; "
                                  "delete the folder to regenerate it.")
        sigma = np.load(os.path.join(out_dir, "noise_sd.npy"))
        print(f"[DATA] reusing {out_dir} (noise sd {np.round(sigma, 4).tolist()})")
        return sigma

    os.makedirs(out_dir, exist_ok=True)
    seq = np.random.SeedSequence(seed).spawn(n_rep + 2)

    # Noise level per level, from sd(f_l) on a large Latin hypercube
    X_sd = _lhs(d, n_sd, np.random.default_rng(seq[0]))
    sigma = np.array([noise_frac * np.std(f(X_sd)) for f in funcs])

    # Fixed test set at the target level
    rng = np.random.default_rng(seq[1])
    X_test = _lhs(d, n_test, rng)
    y_test = np.ravel(funcs[-1](X_test))
    y_test_noisy = y_test + sigma[-1] * rng.standard_normal(n_test)

    # Replicates: independent random LHS per level (non-nested) and noisy outputs
    X_rows = [[] for _ in range(L)]
    y_rows = [[] for _ in range(L)]
    for r in range(1, n_rep + 1):
        rng = np.random.default_rng(seq[1 + r])
        for l, (f, N) in enumerate(zip(funcs, N_list)):
            X = _lhs(d, N, rng)
            y = np.ravel(f(X)) + sigma[l] * rng.standard_normal(N)
            X_rows[l].append(np.column_stack([np.full(N, float(r)), X]))
            y_rows[l].append(np.column_stack([np.full(N, float(r)), y]))

    for l in range(L):
        np.save(os.path.join(out_dir, f"X{l + 1}.npy"), np.vstack(X_rows[l]))
        np.save(os.path.join(out_dir, f"y{l + 1}.npy"), np.vstack(y_rows[l]))
    np.save(os.path.join(out_dir, "x_test.npy"), X_test)
    np.save(os.path.join(out_dir, "yhigh_test.npy"), y_test)
    np.save(os.path.join(out_dir, "yhigh_test_noisy.npy"), y_test_noisy)
    np.save(os.path.join(out_dir, "noise_sd.npy"), sigma)
    if settings is not None:
        np.save(os.path.join(out_dir, "alm_settings.npy"), settings)
    np.save(meta_path, meta)                     # last: marks the folder as complete
    print(f"[DATA] wrote {out_dir}: {n_rep} replicates, N {list(N_list)}, "
          f"noise sd {np.round(sigma, 4).tolist()} ({noise_frac:g} x sd(f_l))")
    return sigma
