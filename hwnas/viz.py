"""Matplotlib helpers (optional dependency)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .space import OPS, SearchSpace


def plot_pareto(all_cands, front, path, title="Accuracy vs latency", acc_label="accuracy (SYNTHETIC proxy)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([c.lat * 1e3 for c in all_cands], [c.acc for c in all_cands], s=8, alpha=.35, label="evaluated")
    ax.plot([c.lat * 1e3 for c in front], [c.acc for c in front], "o-", c="C3", label="Pareto front")
    ax.set_xlabel("latency (ms)"); ax.set_ylabel(acc_label); ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_arch_grid(archs: dict, space: SearchSpace, path, title="Selected operator per layer"):
    fig, axes = plt.subplots(1, len(archs), figsize=(3.2 * len(archs), 4), sharey=True, squeeze=False)
    for ax, (name, arch) in zip(axes[0], archs.items()):
        ax.imshow(np.array(arch)[:, None], cmap="viridis", vmin=0, vmax=len(OPS) - 1, aspect=.5)
        for i, a in enumerate(arch):
            ax.text(0, i, OPS[a].name.replace("mb_", ""), ha="center", va="center", color="w", fontsize=8)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks(range(space.n_layers))
    axes[0][0].set_ylabel("layer"); fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_history(hists: dict, path, title="Best accuracy vs search budget"):
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, h in hists.items():
        ax.plot(np.arange(1, len(h) + 1), h, label=k)
    ax.set_xlabel("evaluations"); ax.set_ylabel("best feasible accuracy (proxy)"); ax.set_title(title)
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
