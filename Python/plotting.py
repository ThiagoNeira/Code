# plotting.py

import numpy as np
import matplotlib.pyplot as plt

def _make_slice_grid(d, n=1000):
    return np.full((n, d), 0.5)

# plotting.py (Add these functions)

def plot_slices_high_1d(high_fn, seed, N1, N2, d, models_dict, plot_dir, criterion='nlpd', tuning_method='hybrid'):
    """
    Plots high-fidelity predictions with confidence intervals (y_lower, y_upper) for each dimension slice.
    """
    n_models = len(models_dict)
    fig, axes = plt.subplots(d, n_models, figsize=(4 * n_models, 4 * d), squeeze=False)
    x_ref = _make_slice_grid(d)
    colors = plt.get_cmap("tab10")

    for i in range(d):
        X_slice = x_ref.copy()
        X_slice[:, i] = np.linspace(0, 1, 1000)
        xi = X_slice[:, i]
        y_true = high_fn(X_slice)

        for col, (name, model) in enumerate(models_dict.items()):
            ax = axes[i, col]
            
            # Predict with confidence intervals
            y_pred, _, y_lower, y_upper = model.predict(X_slice, return_ci=True)

            color = colors(col)
            ax.plot(xi, y_pred.ravel(), color=color, lw=2, label=name)
            ax.fill_between(xi, y_lower.ravel(), y_upper.ravel(), color=color, alpha=0.2, label="Coverage")
            ax.plot(xi, y_true.ravel(), "k--", lw=1.5, label="True")
            
            ax.set_title(f"{name}")
            ax.set_xlabel(f"x_{{ {i+1} }}")
            if col == 0:
                ax.set_ylabel("y")
            ax.legend(loc="best", fontsize=8, framealpha=0.7)

    plt.suptitle(
        f"High-Fidelity Predictive Mean & Coverage \n"
        f"(N1={N1}, N2={N2}, seed={seed} | criterion={criterion}, tuning={tuning_method})"
    )
    plt.tight_layout()
    plt.savefig(
        f"{plot_dir}/high_fidelity_ci_N1={N1}_N2={N2}_seed={seed}_criterion={criterion}_tuning={tuning_method}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_slices_low_1d(low_fn, seed, N1, N2, d, models_dict, plot_dir, criterion='nlpd', tuning_method='hybrid'):
    """
    Plots low-fidelity predictions with confidence intervals for models supporting low fidelity.
    """
    low_models = {k: v for k, v in models_dict.items() if k != "GP"}
    if not low_models:
        return

    n_models = len(low_models)
    fig, axes = plt.subplots(d, n_models, figsize=(4 * n_models, 4 * d), squeeze=False)
    x_ref = _make_slice_grid(d)
    colors = plt.get_cmap("tab10")

    for i in range(d):
        X_slice = x_ref.copy()
        X_slice[:, i] = np.linspace(0, 1, 1000)
        xi = X_slice[:, i]
        z_true = low_fn(X_slice)

        for col, (name, model) in enumerate(low_models.items()):
            ax = axes[i, col]
            
            if name == "KO":
                z_pred, _, z_lower, z_upper = model.predict_low(X_slice, return_ci=True)
            else:
                z_pred, _, z_lower, z_upper = model.predict(X_slice, low_fidelity=True, return_ci=True)

            color = colors(col)
            ax.plot(xi, z_pred.ravel(), color=color, lw=2, label=name)
            ax.fill_between(xi, z_lower.ravel(), z_upper.ravel(), color=color, alpha=0.2, label="Coverage")
            ax.plot(xi, z_true.ravel(), "k--", lw=1.5, label="True")
            
            ax.set_title(f"{name} (Low)")
            ax.set_xlabel(f"x_{{ {i+1} }}")
            if col == 0:
                ax.set_ylabel("z")
            ax.legend(loc="best", fontsize=8, framealpha=0.7)

    plt.suptitle(
        f"Low-Fidelity Predictive Mean & Coverage \n"
        f"(N1={N1}, N2={N2}, seed={seed} | criterion={criterion}, tuning={tuning_method})"
    )
    plt.tight_layout()
    plt.savefig(
        f"{plot_dir}/low_fidelity_ci_N1={N1}_N2={N2}_seed={seed}_criterion={criterion}_tuning={tuning_method}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

def plot_boxplot(
    scores,
    N=None,
    plot_dir="",
    high_fidelity=True,
    criterion='nlpd',
    tuning_method='hybrid',
    metric_name="RMSE",
    file_prefix="boxplot",
    target_value=None,
    is_timing=False,      
    N1=None,              
    N2=None,              
    n_replicates=None,
    kernel='gaussian',     
    n_samples=None         
):
    labels = list(scores.keys())
    data = [scores[k] for k in labels]
    
    if not data or len(data[0]) == 0:
        return

    n_rep = len(data[0]) if n_replicates is None else n_replicates

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(labels)), 4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)

    if target_value is not None:
        ax.axhline(target_value, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Target ({target_value})')
        ax.legend(loc='best', fontsize=8)

    ax.set_ylabel(metric_name)

    # Construct filename tags
    samples_str = f"_n_samples={n_samples}" if n_samples is not None else ""

    if is_timing:
        ax.set_title(f"Fitting time — {n_rep} replicates")
        filename = f"timing_kernel={kernel}{samples_str}_N1={N1}_N2={N2}_rep={n_rep}_criterion={criterion}_tuning={tuning_method}.png"
    else:
        fid_label = "High" 
        ax.set_title(
            f"{fid_label}-Fidelity {metric_name} — {n_rep} replicates, "
            f"{'N2' if high_fidelity else 'N1'}={N}"
        )
        suffix = "high" 
        filename = f"{file_prefix}_{suffix}_kernel={kernel}{samples_str}_N={N}_rep={n_rep}_criterion={criterion}_tuning={tuning_method}.png"

    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        f"{plot_dir}/{filename}",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)