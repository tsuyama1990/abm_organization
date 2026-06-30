"""
Pykälä et al. (2026) — Heterogeneous learning strategies in collective problem-solving.

Three strategies compete in an evolutionary model:
  HC (Hill-Climbing)      : greedy best-neighbor step
  PB (Payoff-Biased copy) : copy state of highest-fitness peer
  FB (Frequency-Biased copy): copy state of most-common strategy's best representative

Each generation: agents search for n_steps, then strategies evolve via fitness-proportionate selection.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape

STRATEGIES = ("HC", "PB", "FB")


def _hc_step(
    landscape: NKLandscape,
    state: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Take the best 1-flip improvement, or stay if at local max."""
    best_n, best_f = landscape.best_neighbor(state)
    return best_n if best_f > landscape.fitness(state) else state


def _pb_step(
    landscape: NKLandscape,
    state: np.ndarray,
    all_states: np.ndarray,
    all_fitness: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Copy state of the peer with highest fitness (then do 1 HC step)."""
    best_peer = int(np.argmax(all_fitness))
    return _hc_step(landscape, all_states[best_peer].copy(), rng)


def _fb_step(
    landscape: NKLandscape,
    state: np.ndarray,
    all_states: np.ndarray,
    all_strategies: np.ndarray,
    all_fitness: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Copy the best representative of the most frequent strategy."""
    counts = np.bincount(all_strategies, minlength=3)
    dominant_strat = int(np.argmax(counts))
    mask = all_strategies == dominant_strat
    if not np.any(mask):
        return _hc_step(landscape, state, rng)
    best_in_dominant = int(np.argmax(all_fitness * mask - 1e9 * ~mask))
    return _hc_step(landscape, all_states[best_in_dominant].copy(), rng)


def _generation(
    landscape: NKLandscape,
    states: np.ndarray,
    strategies: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one generation: each agent takes n_steps, return updated (states, fitness)."""
    n = len(states)
    fitness = np.array([landscape.fitness(states[i]) for i in range(n)])

    for _ in range(n_steps):
        new_states = states.copy()
        for i in range(n):
            s = strategies[i]
            if s == 0:  # HC
                new_states[i] = _hc_step(landscape, states[i], rng)
            elif s == 1:  # PB
                new_states[i] = _pb_step(landscape, states[i], states, fitness, rng)
            else:  # FB
                new_states[i] = _fb_step(landscape, states[i], states, strategies, fitness, rng)
        states = new_states
        fitness = np.array([landscape.fitness(states[i]) for i in range(n)])

    return states, fitness


def _evolve_strategies(
    strategies: np.ndarray,
    fitness: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fitness-proportionate selection: replace each agent's strategy."""
    n = len(strategies)
    f_shifted = fitness - fitness.min() + 1e-9
    probs = f_shifted / f_shifted.sum()
    parent_indices = rng.choice(n, size=n, p=probs)
    return strategies[parent_indices]


def simulate(
    N: int = 12,
    K: int = 4,
    n_agents: int = 30,
    n_generations: int = 100,
    n_steps_per_gen: int = 10,
    n_runs: int = 50,
    seed: int = 42,
) -> dict:
    """
    Evolutionary simulation of HC/PB/FB strategy competition.

    Returns
    -------
    dict with:
      - 'mean_fitness': list[float] — mean population fitness per generation
      - 'HC_prop', 'PB_prop', 'FB_prop': list[float] — strategy proportions per generation
      - 'params': dict
    """
    rng = np.random.default_rng(seed)

    fit_accum: list[np.ndarray] = []
    prop_accum: dict[str, list[np.ndarray]] = {s: [] for s in STRATEGIES}

    for _ in range(n_runs):
        ls = NKLandscape(N, K, seed=int(rng.integers(0, 2**31)))
        states = np.array([ls.random_state() for _ in range(n_agents)])
        # Start with equal proportions
        strategies = np.array([i % 3 for i in range(n_agents)], dtype=int)

        gen_fitness: list[float] = []
        gen_props: dict[str, list[float]] = {s: [] for s in STRATEGIES}

        for _g in range(n_generations):
            states, fitness = _generation(ls, states, strategies, n_steps_per_gen, rng)
            gen_fitness.append(float(np.mean(fitness)))
            counts = np.bincount(strategies, minlength=3)
            for j, s in enumerate(STRATEGIES):
                gen_props[s].append(counts[j] / n_agents)
            strategies = _evolve_strategies(strategies, fitness, rng)

        fit_accum.append(np.array(gen_fitness))
        for s in STRATEGIES:
            prop_accum[s].append(np.array(gen_props[s]))

    return {
        "mean_fitness": np.mean(np.array(fit_accum), axis=0).tolist(),
        "HC_prop": np.mean(np.array(prop_accum["HC"]), axis=0).tolist(),
        "PB_prop": np.mean(np.array(prop_accum["PB"]), axis=0).tolist(),
        "FB_prop": np.mean(np.array(prop_accum["FB"]), axis=0).tolist(),
        "params": {
            "N": N, "K": K, "n_agents": n_agents,
            "n_generations": n_generations, "n_steps_per_gen": n_steps_per_gen,
        },
    }
