"""Figures for the report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_grid(grid, path):
    names = list(grid.index)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid.values, cmap='YlGnBu', vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel('Tested at')
    ax.set_ylabel('Trained at')
    ax.set_title('Local helpers: how often each is right (mean of seeds)')
    for i in range(len(names)):
        for j in range(len(names)):
            v = grid.values[i, j]
            ax.text(j, i, f'{v:.0%}', ha='center', va='center', color='white' if v > .75 else 'black',
                    fontweight='bold' if i == j else None)
    plt.colorbar(im, label='share predicted correctly')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_privacy_curve(sweep, ref, path):
    """sweep: long DataFrame (seed, epsilon, hospital, accuracy, accuracy_cal). ref: dict name -> value."""
    per_seed = sweep.groupby(['epsilon', 'seed'])[['accuracy', 'accuracy_cal']].mean().reset_index()
    g = per_seed.groupby('epsilon')['accuracy'].agg(['mean', 'std']).fillna(0)
    gc = per_seed.groupby('epsilon')['accuracy_cal'].agg(['mean', 'std']).fillna(0)
    fin, finc = g[np.isfinite(g.index)], gc[np.isfinite(gc.index)]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.errorbar(fin.index, fin['mean'], yerr=fin['std'], marker='o', capsize=3, label='Federated + DP')
    ax.errorbar(finc.index, finc['mean'], yerr=finc['std'], marker='s', capsize=3, color='tab:orange',
                label='Federated + DP + local calibration')
    for (label, v), style in zip(ref.items(), ['--', ':', '-.']):
        ax.axhline(v, linestyle=style, color='gray', label=label)
    if np.inf in g.index:
        ax.axhline(g.loc[np.inf, 'mean'], color='green', linestyle='--', label='Federated, no privacy')
    ax.set_xscale('log', base=2)
    ax.set_xticks(fin.index, [f'{e:g}' for e in fin.index])
    ax.set_xlabel('epsilon (smaller = more privacy, more blur); bars = std over seeds')
    ax.set_ylabel('average test accuracy over the 4 hospitals')
    ax.set_title('Privacy versus accuracy')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drift(curves, phase_a, path):
    """Old-stage and new-stage accuracy through both phases (seed 0 curves)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    na = len(phase_a)
    for ax, key, title in ((axes[0], 'A', 'Old stage (what it should not forget)'),
                           (axes[1], 'B', 'New stage (after the world changed)')):
        ax.plot(range(na), [r[key] for r in phase_a], color='black', label='learning stage A')
        for name, hist in curves.items():
            ax.plot(range(na, na + len(hist)), [r[key] for r in hist], label=name)
        ax.axvline(na - 0.5, color='red', linestyle=':')
        ax.set_title(title)
        ax.set_xlabel('federated round')
    axes[0].set_ylabel('average test accuracy')
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
