"""Visualization utilities for all NK-ABM organizational models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)

_COLORS = sns.color_palette("colorblind", 8)
_CONDITION_LABELS = {
    "centralized": "Centralized",
    "decentralized": "Decentralized",
    "reintegrated": "Reintegrated (Temp-Dec)",
    "hierarchy": "Traditional Hierarchy",
    "autonomous": "Autonomous (Flat)",
    "multiauthority": "Multiauthority (Matrix)",
    "top_down": "Top-Down",
    "bottom_up": "Bottom-Up",
    "jtc_human": "JTC + Human",
    "jtc_ai": "JTC + AI",
    "flat_human": "Flat + Human",
    "flat_ai": "Flat + AI",
}


def _save_or_show(fig: plt.Figure, output_path: Optional[str]) -> None:
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_performance_trajectories(
    results: dict[str, np.ndarray],
    title: str,
    shock_period: Optional[int] = None,
    ylabel: str = "Mean Performance",
    output_path: Optional[str] = None,
) -> None:
    """Line plot of performance trajectories for multiple organizational conditions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (key, traj) in enumerate(results.items()):
        label = _CONDITION_LABELS.get(key, key)
        ax.plot(traj, label=label, color=_COLORS[i % len(_COLORS)], linewidth=2)
    if shock_period is not None:
        ax.axvline(shock_period, color="red", linestyle="--", alpha=0.7, label="Environmental shock")
    ax.set_xlabel("Period")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", framealpha=0.85)
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_rescaled_cost(
    results: dict[int, dict[int, float]],
    title: str,
    N: int = 12,
    output_path: Optional[str] = None,
) -> None:
    """Log-log plot of mean rescaled cost <C> vs group size L for each connectivity M."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (M, l_costs) in enumerate(sorted(results.items())):
        Ls = sorted(l_costs.keys())
        Cs = [l_costs[L] for L in Ls]
        valid = [(L, C) for L, C in zip(Ls, Cs) if not np.isnan(C)]
        if not valid:
            continue
        Lv, Cv = zip(*valid)
        ax.loglog(Lv, Cv, "o-", label=f"M={M}", color=_COLORS[i % len(_COLORS)], linewidth=1.5)

    # Reference line: C = L / 2^N (independent search)
    L_ref = np.logspace(0, 4, 100)
    ax.loglog(L_ref, L_ref / 2**N, "k--", alpha=0.5, label=r"$L/2^N$ (independent)")

    ax.set_xlabel("Group size L")
    ax.set_ylabel(r"$\langle C \rangle = L t^* / 2^N$")
    ax.set_title(title)
    ax.legend(framealpha=0.85)
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_strategy_evolution(
    results: dict,
    title: str,
    output_path: Optional[str] = None,
) -> None:
    """Stacked area chart of HC/PB/FB strategy proportions over generations."""
    fig, axes = plt.subplots(1, len(results), figsize=(12, 5), sharey=True)
    if len(results) == 1:
        axes = [axes]

    strategy_colors = {"HC": _COLORS[0], "PB": _COLORS[1], "FB": _COLORS[2]}

    for ax, (condition, data) in zip(axes, results.items()):
        gens = range(len(data.get("HC", data.get("HC_prop", []))))
        hc = data.get("HC", data.get("HC_prop", []))
        pb = data.get("PB", data.get("PB_prop", []))
        fb = data.get("FB", data.get("FB_prop", []))
        ax.stackplot(
            gens, hc, pb, fb,
            labels=["Hill-climbing (HC)", "Payoff-biased (PB)", "Frequency-biased (FB)"],
            colors=[strategy_colors["HC"], strategy_colors["PB"], strategy_colors["FB"]],
            alpha=0.8,
        )
        ax.set_xlabel("Generation")
        ax.set_title(condition)
        ax.set_ylim(0, 1)

    axes[0].set_ylabel("Strategy proportion")
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save_or_show(fig, output_path)


def plot_jtc_summary(
    perf: dict[str, list],
    diversity: dict[str, list],
    shock_period: int,
    output_path: Optional[str] = None,
) -> None:
    """
    2-panel figure: performance trajectories + diversity over time for JTC hypothesis.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    conditions = ["jtc_human", "jtc_ai", "flat_human", "flat_ai"]
    for i, cond in enumerate(conditions):
        label = _CONDITION_LABELS.get(cond, cond)
        color = _COLORS[i]
        linestyle = "--" if "ai" in cond else "-"
        ax1.plot(perf[cond], label=label, color=color, linestyle=linestyle, linewidth=2)
        ax2.plot(diversity[cond], color=color, linestyle=linestyle, linewidth=2)

    for ax in (ax1, ax2):
        ax.axvline(shock_period, color="red", linestyle=":", alpha=0.8, label="Shock")
        ax.grid(True, alpha=0.4)

    ax1.set_ylabel("Max Performance (best agent)")
    ax1.set_title("JTC vs. Flat × Human vs. AI — Performance")
    ax1.legend(loc="lower right", fontsize=9)

    ax2.set_ylabel("Solution Diversity (mean Hamming / N)")
    ax2.set_xlabel("Period")
    ax2.set_title("Diversity over time")

    fig.tight_layout()
    _save_or_show(fig, output_path)
