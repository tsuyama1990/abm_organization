"""
Leitner (2023) — Top-down vs bottom-up coordination under environmental disruptions.

Top-down: a central coordinator picks the best proposal from all workers each period.
Bottom-up: each worker independently adopts improvements from local search.
Both face periodic correlated landscape shocks (Beta-distribution; NKLandscape.perturb).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape


def _top_down_step(
    landscape: NKLandscape,
    worker_states: np.ndarray,
    fitness_cache: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Central coordinator evaluates one random 1-flip proposal per worker,
    then picks the globally best improvement and applies it.
    """
    n = len(worker_states)
    best_gain = 0.0
    best_i, best_state = -1, None

    for i in range(n):
        bit = int(rng.integers(0, landscape.N))
        candidate = worker_states[i].copy()
        candidate[bit] ^= 1
        new_f = landscape.fitness(candidate)
        gain = new_f - fitness_cache[i]
        if gain > best_gain:
            best_gain = gain
            best_i, best_state = i, candidate

    if best_i >= 0 and best_state is not None:
        worker_states[best_i] = best_state
        fitness_cache[best_i] = fitness_cache[best_i] + best_gain


def _bottom_up_step(
    landscape: NKLandscape,
    worker_states: np.ndarray,
    fitness_cache: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """
    Each worker independently evaluates one random 1-flip proposal
    and adopts it if it improves local fitness.
    """
    for i in range(len(worker_states)):
        bit = int(rng.integers(0, landscape.N))
        candidate = worker_states[i].copy()
        candidate[bit] ^= 1
        new_f = landscape.fitness(candidate)
        if new_f > fitness_cache[i]:
            worker_states[i] = candidate
            fitness_cache[i] = new_f


def _run_single(
    landscape: NKLandscape,
    mode: str,
    n_workers: int,
    n_periods: int,
    shock_period: Optional[int],
    shock_rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Single run; returns performance trajectory of shape (n_periods+1,)."""
    states = np.array([landscape.random_state() for _ in range(n_workers)])
    cache = np.array([landscape.fitness(states[i]) for i in range(n_workers)])

    current = landscape
    traj = [float(np.max(cache))]

    for t in range(1, n_periods + 1):
        if shock_period is not None and t == shock_period:
            current = landscape.perturb(shock_rho, rng)
            cache = np.array([current.fitness(states[i]) for i in range(n_workers)])

        if mode == "top_down":
            _top_down_step(current, states, cache, rng)
        else:
            _bottom_up_step(current, states, cache, rng)

        traj.append(float(np.max(cache)))

    return np.array(traj)


def simulate(
    N: int = 10,
    K: int = 4,
    n_workers: int = 5,
    n_periods: int = 200,
    shock_period: Optional[int] = 100,
    shock_rho: float = 0.3,
    n_runs: int = 100,
    seed: int = 42,
) -> dict[str, list]:
    """
    Compare top-down vs bottom-up coordination under disruption.

    Returns
    -------
    dict with 'top_down' and 'bottom_up', each a list (trajectory of mean performance).
    Also includes 'params' and summary stats.
    """
    rng = np.random.default_rng(seed)
    accum: dict[str, list[np.ndarray]] = {"top_down": [], "bottom_up": []}

    for _ in range(n_runs):
        ls = NKLandscape(N, K, seed=int(rng.integers(0, 2**31)))
        for mode in ("top_down", "bottom_up"):
            traj = _run_single(ls, mode, n_workers, n_periods, shock_period, shock_rho, rng)
            accum[mode].append(traj)

    mean_perf = {m: np.mean(np.array(v), axis=0).tolist() for m, v in accum.items()}

    pre_peak: dict[str, float] = {}
    post_recovery: dict[str, float] = {}
    for m, traj in mean_perf.items():
        arr = np.array(traj)
        sp = shock_period if shock_period is not None else n_periods
        pre_peak[m] = float(np.max(arr[:sp]))
        window = arr[sp + 1 : min(sp + 51, n_periods + 1)]
        post_recovery[m] = float(np.mean(window)) if len(window) > 0 else float(arr[-1])

    return {
        "top_down": mean_perf["top_down"],
        "bottom_up": mean_perf["bottom_up"],
        "pre_shock_peak": pre_peak,
        "post_shock_recovery": post_recovery,
        "params": {
            "N": N, "K": K, "n_workers": n_workers,
            "n_periods": n_periods, "shock_period": shock_period, "shock_rho": shock_rho,
        },
    }
