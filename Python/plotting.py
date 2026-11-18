# plotting.py

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_boxplot(
    scores,
    N=None,
    plot_dir="",
    high_fidelity=True,
    criterion='nlpd',
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
        filename = f"timing_kernel={kernel}{samples_str}_N1={N1}_N2={N2}_rep={n_rep}_criterion={criterion}.png"
    else:
        fid_label = "High" 
        ax.set_title(
            f"{fid_label}-Fidelity {metric_name} — {n_rep} replicates, "
            f"{'N2' if high_fidelity else 'N1'}={N}"
        )
        suffix = "high" 
        filename = f"{file_prefix}_{suffix}_kernel={kernel}{samples_str}_N={N}_rep={n_rep}_criterion={criterion}.png"

    # Ensure plot_dir is relative to this script's location if it's not absolute
    if not os.path.isabs(plot_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(script_dir, plot_dir)

    # Create the directory if it doesn't exist yet
    os.makedirs(plot_dir, exist_ok=True)

    save_path = os.path.join(plot_dir, filename)

    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    print(repr(save_path), len(save_path))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)