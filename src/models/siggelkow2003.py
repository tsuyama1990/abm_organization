"""
Siggelkow & Levinthal (2003) "Temporarily Divide to Conquer"
Organization Science, 14(6), 650–669.

Three organizational forms on an NK fitness landscape:
  - Centralized:    firm evaluates one 1-flip alternative per period
  - Decentralized:  two divisions each search within their own decisions
  - Reintegrated:   decentralized for t_dec periods, then centralized
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _partial_fitness(
    landscape: NKLandscape,
    state: np.ndarray,
    division: list[int],
) -> float:
    """Average fitness contribution for a subset of decisions."""
    return float(
        sum(landscape.phi[j][landscape._component_index(state, j)] for j in division)
        / len(division)
    )


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

def centralized_search(
    landscape: NKLandscape,
    initial_state: np.ndarray,
    n_periods: int,
) -> tuple[list[float], np.ndarray]:
    """
    Centralized firm: each period evaluates one random 1-flip alternative.
    Adopts if global fitness improves.
    """
    rng = landscape.rng
    state = np.asarray(initial_state, dtype=int).copy()
    trajectory: list[float] = [landscape.fitness(state)]

    for _ in range(n_periods):
        bit = int(rng.integers(0, landscape.N))
        candidate = state.copy()
        candidate[bit] ^= 1
        if landscape.fitness(candidate) > landscape.fitness(state):
            state = candidate
        trajectory.append(landscape.fitness(state))

    return trajectory, state


def decentralized_search(
    landscape: NKLandscape,
    initial_state: np.ndarray,
    n_periods: int,
    divisions: list[list[int]],
) -> tuple[list[float], np.ndarray]:
    """
    Decentralized firm: each division independently searches its own decisions.
    Division evaluates ONLY its partial fitness (own tasks averaged).
    Both divisions can implement improvements in the same period.
    """
    rng = landscape.rng
    state = np.asarray(initial_state, dtype=int).copy()
    trajectory: list[float] = [landscape.fitness(state)]

    for _ in range(n_periods):
        for div in divisions:
            bit = int(rng.choice(div))
            candidate = state.copy()
            candidate[bit] ^= 1
            if _partial_fitness(landscape, candidate, div) > _partial_fitness(landscape, state, div):
                state = candidate
        trajectory.append(landscape.fitness(state))

    return trajectory, state


def reintegrated_search(
    landscape: NKLandscape,
    initial_state: np.ndarray,
    n_periods: int,
    divisions: list[list[int]],
    t_dec: int,
) -> tuple[list[float], np.ndarray]:
    """
    Temporarily decentralized then reintegrated.
    Decentralized for t_dec periods; centralized for the rest.
    """
    t_dec = min(t_dec, n_periods)
    traj_dec, state_mid = decentralized_search(
        landscape, initial_state, t_dec, divisions
    )
    if t_dec == n_periods:
        return traj_dec, state_mid

    traj_cen, state_final = centralized_search(landscape, state_mid, n_periods - t_dec)
    # traj_cen[0] repeats the last point of traj_dec; drop it
    return traj_dec + traj_cen[1:], state_final


# ---------------------------------------------------------------------------
# Simulation entry point
# ---------------------------------------------------------------------------

def simulate(
    N: int = 6,
    K: int = 2,
    interaction_type: str = "random",
    block_sizes: Optional[list[int]] = None,
    n_periods: int = 100,
    t_dec: int = 50,
    n_runs: int = 250,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Compare all three organizational forms over n_runs independent landscapes.

    Returns
    -------
    dict with keys 'centralized', 'decentralized', 'reintegrated', each an
    ndarray of shape (n_periods+1,) representing the mean performance trajectory.
    """
    if block_sizes is None:
        block_sizes = [N // 2, N // 2]
    divisions = [
        list(range(0, block_sizes[0])),
        list(range(block_sizes[0], N)),
    ]

    rng = np.random.default_rng(seed)
    accum: dict[str, list[list[float]]] = {
        "centralized": [],
        "decentralized": [],
        "reintegrated": [],
    }

    for _ in range(n_runs):
        ls = NKLandscape(
            N,
            K,
            interaction_type=interaction_type,
            block_sizes=block_sizes if interaction_type == "block_diagonal" else None,
            seed=int(rng.integers(0, 2**31)),
        )
        s0 = ls.random_state()

        traj_c, _ = centralized_search(ls, s0.copy(), n_periods)
        traj_d, _ = decentralized_search(ls, s0.copy(), n_periods, divisions)
        traj_r, _ = reintegrated_search(ls, s0.copy(), n_periods, divisions, t_dec)

        accum["centralized"].append(traj_c)
        accum["decentralized"].append(traj_d)
        accum["reintegrated"].append(traj_r)

    return {key: np.mean(np.array(vals), axis=0) for key, vals in accum.items()}
