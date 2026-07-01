"""
Fontanari & Rodrigues (2015) "Influence of network topology on cooperative problem-solving systems"
arXiv:1510.03059v2.

L agents on a 1-D ring; each agent sees M neighbours.
Each trial: with prob 1-p flip a random bit (exploration);
            with prob p copy 1 bit from the best-fitness neighbour (exploitation).
Rescaled cost: C = L * t* / 2^N  where t* = first trial a global max is found.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape


class FontanariModel:
    """
    L agents on a periodic 1-D ring, each connected to M neighbours (M//2 each side).
    """

    def __init__(
        self,
        N: int,
        K: int,
        L: int,
        M: int,
        p: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if M >= L:
            raise ValueError(f"M={M} must be < L={L}")
        self.N = N
        self.K = K
        self.L = L
        self.M = M
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.states: np.ndarray = np.zeros((L, N), dtype=int)
        self.fitness_cache: np.ndarray = np.zeros(L)

    def _get_neighbors(self, idx: int) -> list[int]:
        half = self.M // 2
        return [(idx + d) % self.L for d in range(-half, half + 1) if d != 0]

    def reset(self, landscape: NKLandscape) -> None:
        for i in range(self.L):
            self.states[i] = landscape.random_state()
            self.fitness_cache[i] = landscape.fitness(self.states[i])

    def step(self, landscape: NKLandscape) -> None:
        target = int(self.rng.integers(0, self.L))
        if self.rng.random() < self.p:
            # Exploitation: copy 1 bit from best neighbour
            neighbours = self._get_neighbors(target)
            best_nb = neighbours[int(np.argmax(self.fitness_cache[neighbours]))]
            diff = np.where(self.states[target] != self.states[best_nb])[0]
            if len(diff) > 0:
                bit = int(self.rng.choice(diff))
                self.states[target, bit] = self.states[best_nb, bit]
                self.fitness_cache[target] = landscape.fitness(self.states[target])
        else:
            # Exploration: random bit flip
            bit = int(self.rng.integers(0, self.N))
            self.states[target, bit] ^= 1
            self.fitness_cache[target] = landscape.fitness(self.states[target])

    def run(
        self,
        landscape: NKLandscape,
        global_max_fitness: float,
        max_trials: int = 5_000_000,
    ) -> Optional[int]:
        """Return trial count to first hit global max, or None if exceeded."""
        self.reset(landscape)
        for t in range(1, max_trials + 1):
            if float(np.max(self.fitness_cache)) >= global_max_fitness - 1e-10:
                return t
            self.step(landscape)
        return None


def simulate_cost(
    N: int,
    K: int,
    L_values: list[int],
    M_values: list[int],
    p: float = 0.5,
    n_runs: int = 300,
    seed: int = 42,
    max_trials: int = 5_000_000,
) -> dict[int, dict[int, float]]:
    """
    Compute mean rescaled cost C = L * mean(t*) / 2^N for each (M, L) pair.
    Skips pairs where t* is None (all runs exceeded max_trials).
    Returns {M: {L: mean_C}}.
    """
    rng = np.random.default_rng(seed)
    results: dict[int, dict[int, float]] = {}

    for M in M_values:
        results[M] = {}
        for L in L_values:
            if M >= L:
                continue
            t_stars: list[float] = []
            for _ in range(n_runs):
                ls = NKLandscape(N, K, seed=int(rng.integers(0, 2**31)))
                _, global_max_f = ls.global_maximum()
                model = FontanariModel(N, K, L, M, p, seed=int(rng.integers(0, 2**31)))
                t = model.run(ls, global_max_f, max_trials)
                if t is not None:
                    t_stars.append(float(t))

            if t_stars:
                mean_C = L * float(np.mean(t_stars)) / (2**N)
                results[M][L] = mean_C
            else:
                results[M][L] = float("nan")

    return results


def simulate(
    N: int = 12,
    L_values: Optional[list[int]] = None,
    M_values: Optional[list[int]] = None,
    K_values: Optional[list[int]] = None,
    p: float = 0.5,
    n_runs: int = 200,
    seed: int = 42,
) -> dict:
    """
    Full experiment: run for each K in K_values.
    Returns {f'K{K}': {M: {L: mean_C}}}.
    """
    if L_values is None:
        L_values = [2, 4, 8, 16, 32, 64]
    if M_values is None:
        M_values = [2, 4, 8, 16]
    if K_values is None:
        K_values = [0, 4]

    out: dict = {}
    for K in K_values:
        out[f"K{K}"] = simulate_cost(N, K, L_values, M_values, p, n_runs, seed)
    return out
