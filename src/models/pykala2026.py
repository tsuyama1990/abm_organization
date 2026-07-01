"""
Pykälä, Dillion & Galesic (2026) — Heterogeneous learning strategies interact with
social network structure and problem complexity to benefit collective search.
Phil. Trans. R. Soc. B 381: 20240450.

Key mechanisms from paper:
- G=30 groups, each of S agents on a social network (ring lattice or random)
- Three strategies per agent:
    HC  (hill-climbing):      random 1-bit flip, keep if higher payoff
    PB  (payoff-biased):      copy solution of highest-payoff NEIGHBOUR
    FB  (frequency-biased):   copy most-frequent solution among NEIGHBOURS
- Search: t=25 synchronous time-steps per generation
- Selection: top 2 groups (by summed payoffs) replicate → new G=30 groups
- Mutation: μ=0.05 per group; increase one random strategy proportion by 0.2
- N=10, K∈{0,8}; network density D∈{0.22, 0.89}
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape

HC, PB, FB = 0, 1, 2
STRATEGY_NAMES = ("HC", "PB", "FB")


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def _ring_lattice(S: int, D: float, rng: np.random.Generator) -> np.ndarray:
    """
    Regular ring lattice: each agent connected to k neighbours on each side,
    where k is chosen so density ≈ D.
    D = m / (S*(S-1)/2), m = S*k  →  k = D*(S-1)/2.
    """
    k = max(1, round(D * (S - 1) / 2))
    adj = np.zeros((S, S), dtype=bool)
    for i in range(S):
        for d in range(1, k + 1):
            j = (i + d) % S
            adj[i, j] = True
            adj[j, i] = True
    return adj


def _random_network(S: int, D: float, rng: np.random.Generator) -> np.ndarray:
    """Random network with global density ≈ D (configuration model style)."""
    adj = np.zeros((S, S), dtype=bool)
    for i in range(S):
        for j in range(i + 1, S):
            if rng.random() < D:
                adj[i, j] = True
                adj[j, i] = True
    return adj


def _make_network(S: int, topology: str, D: float, rng: np.random.Generator) -> np.ndarray:
    if topology == "ring":
        return _ring_lattice(S, D, rng)
    return _random_network(S, D, rng)


# ---------------------------------------------------------------------------
# Learning strategy steps (use neighbour info, not global)
# ---------------------------------------------------------------------------

def _hc_step(
    state: np.ndarray,
    landscape: NKLandscape,
    rng: np.random.Generator,
) -> np.ndarray:
    bit = int(rng.integers(0, landscape.N))
    candidate = state.copy()
    candidate[bit] ^= 1
    return candidate if landscape.fitness(candidate) > landscape.fitness(state) else state


def _pb_step(
    state: np.ndarray,
    neighbours: list[int],
    all_states: np.ndarray,
    all_fitness: np.ndarray,
    landscape: NKLandscape,
    rng: np.random.Generator,
) -> np.ndarray:
    """Copy solution of highest-payoff neighbour (if better than self)."""
    if not neighbours:
        return _hc_step(state, landscape, rng)
    best_nb = neighbours[int(np.argmax(all_fitness[neighbours]))]
    best_state = all_states[best_nb]
    return best_state.copy() if all_fitness[best_nb] > landscape.fitness(state) else state


def _fb_step(
    state: np.ndarray,
    neighbours: list[int],
    all_states: np.ndarray,
    landscape: NKLandscape,
    rng: np.random.Generator,
) -> np.ndarray:
    """Copy the most-frequent solution among neighbours (by majority vote per bit)."""
    if not neighbours:
        return _hc_step(state, landscape, rng)
    nb_states = all_states[neighbours]
    # Most frequent solution: majority vote per bit
    majority = (nb_states.mean(axis=0) >= 0.5).astype(int)
    return majority if landscape.fitness(majority) > landscape.fitness(state) else state


# ---------------------------------------------------------------------------
# Group simulation
# ---------------------------------------------------------------------------

def _search_phase(
    landscape: NKLandscape,
    states: np.ndarray,
    strategies: np.ndarray,
    adj: np.ndarray,
    t_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Run t_steps of synchronous search. Returns (final_states, final_fitness)."""
    S = len(states)
    for _ in range(t_steps):
        fitness = np.array([landscape.fitness(states[i]) for i in range(S)])
        new_states = states.copy()
        for i in range(S):
            neighbours = list(np.where(adj[i])[0])
            s = strategies[i]
            if s == HC:
                new_states[i] = _hc_step(states[i], landscape, rng)
            elif s == PB:
                new_states[i] = _pb_step(states[i], neighbours, states, fitness, landscape, rng)
            else:
                new_states[i] = _fb_step(states[i], neighbours, states, landscape, rng)
        states = new_states

    fitness = np.array([landscape.fitness(states[i]) for i in range(S)])
    return states, fitness


def _evolve(
    groups_states: list[np.ndarray],
    groups_strategies: list[np.ndarray],
    groups_fitness: list[np.ndarray],
    S: int,
    topology: str,
    D: float,
    mu: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Selection: top 2 groups (by summed fitness) replicate to fill G=30 groups.
    Mutation: each new group mutates with prob μ (one strategy +0.2 in proportion).
    """
    G = len(groups_states)
    group_payoffs = [float(np.sum(f)) for f in groups_fitness]
    ranked = sorted(range(G), key=lambda i: -group_payoffs[i])
    top2 = ranked[:2]

    new_states: list[np.ndarray] = []
    new_strats: list[np.ndarray] = []
    new_adj: list[np.ndarray] = []

    for g in range(G):
        parent = top2[g % 2]
        strat = groups_strategies[parent].copy()

        # Mutation: with prob μ, increase one strategy's proportion by 0.2
        if rng.random() < mu:
            chosen_strat = int(rng.integers(0, 3))
            # Increase proportion of chosen_strat by 0.2 (affect ~20% of agents)
            n_switch = max(1, int(0.2 * S))
            candidates = np.where(strat != chosen_strat)[0]
            if len(candidates) > 0:
                to_switch = rng.choice(candidates, size=min(n_switch, len(candidates)), replace=False)
                strat[to_switch] = chosen_strat

        # New agents start at random locations
        init_states = np.array([np.zeros(groups_states[0].shape[1], dtype=int) for _ in range(S)])
        new_states.append(init_states)
        new_strats.append(strat)
        new_adj.append(_make_network(S, topology, D, rng))

    return new_states, new_strats, new_adj


def simulate(
    N: int = 10,
    K_values: Optional[list[int]] = None,
    S_values: Optional[list[int]] = None,
    topology_values: Optional[list[str]] = None,
    D_values: Optional[list[float]] = None,
    G: int = 30,
    t_steps: int = 25,
    n_generations: int = 1000,
    mu: float = 0.05,
    n_runs: int = 200,
    seed: int = 42,
) -> dict:
    """
    Evolutionary simulation of HC/PB/FB strategy competition.
    Parameters match Table 1: N=10, K∈{0,8}, S∈{10,19}, D∈{0.22,0.89}, G=30,
    t=25, generations=1000, μ=0.05, 200 runs.

    Returns strategy proportions and mean fitness over generations for each condition.
    """
    if K_values is None:
        K_values = [0, 8]
    if S_values is None:
        S_values = [10, 19]
    if topology_values is None:
        topology_values = ["ring", "random"]
    if D_values is None:
        D_values = [0.22, 0.89]

    rng = np.random.default_rng(seed)
    results: dict = {}

    for K in K_values:
        # Pre-generate 100 landscapes as in paper
        landscapes = [NKLandscape(N, K, seed=int(rng.integers(0, 2**31))) for _ in range(100)]

        for S in S_values:
            for topology in topology_values:
                for D in D_values:
                    cond = f"K{K}_S{S}_{topology}_D{D}"
                    prop_accum: dict[str, list[np.ndarray]] = {n: [] for n in STRATEGY_NAMES}
                    fit_accum: list[np.ndarray] = []

                    for _ in range(n_runs):
                        # Initialize G groups, all HC
                        ls = landscapes[int(rng.integers(0, 100))]
                        groups_states = [
                            np.array([ls.random_state() for _ in range(S)]) for _ in range(G)
                        ]
                        groups_strategies = [np.zeros(S, dtype=int) for _ in range(G)]
                        groups_adj = [_make_network(S, topology, D, rng) for _ in range(G)]

                        gen_props: dict[str, list[float]] = {n: [] for n in STRATEGY_NAMES}
                        gen_fitness: list[float] = []

                        for _g in range(n_generations):
                            # Sample new landscape each generation
                            ls = landscapes[int(rng.integers(0, 100))]
                            groups_fitness = []
                            new_group_states = []

                            for g in range(G):
                                # Reset positions each generation
                                init = np.array([ls.random_state() for _ in range(S)])
                                final_states, final_fit = _search_phase(
                                    ls, init, groups_strategies[g],
                                    groups_adj[g], t_steps, rng
                                )
                                new_group_states.append(final_states)
                                groups_fitness.append(final_fit)

                            groups_states = new_group_states

                            # Record strategy proportions and mean fitness
                            all_strats = np.concatenate(groups_strategies)
                            counts = np.bincount(all_strats, minlength=3)
                            total = len(all_strats)
                            for j, name in enumerate(STRATEGY_NAMES):
                                gen_props[name].append(counts[j] / total)
                            gen_fitness.append(float(np.mean([f.mean() for f in groups_fitness])))

                            # Evolution
                            groups_states, groups_strategies, groups_adj = _evolve(
                                groups_states, groups_strategies, groups_fitness,
                                S, topology, D, mu, rng
                            )

                        for name in STRATEGY_NAMES:
                            prop_accum[name].append(np.array(gen_props[name]))
                        fit_accum.append(np.array(gen_fitness))

                    results[cond] = {
                        "mean_fitness": np.mean(np.array(fit_accum), axis=0).tolist(),
                        "HC_prop": np.mean(np.array(prop_accum["HC"]), axis=0).tolist(),
                        "PB_prop": np.mean(np.array(prop_accum["PB"]), axis=0).tolist(),
                        "FB_prop": np.mean(np.array(prop_accum["FB"]), axis=0).tolist(),
                        "final_HC": float(np.mean(np.array(prop_accum["HC"])[:, -1])),
                        "final_PB": float(np.mean(np.array(prop_accum["PB"])[:, -1])),
                        "final_FB": float(np.mean(np.array(prop_accum["FB"])[:, -1])),
                    }

    results["params"] = {
        "N": N, "K_values": K_values, "S_values": S_values,
        "topology_values": topology_values, "D_values": D_values,
        "G": G, "t_steps": t_steps, "n_generations": n_generations, "mu": mu,
    }
    return results
