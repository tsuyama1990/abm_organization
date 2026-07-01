"""
Levinthal & Workiewicz (2018) — Authority structures in nearly decomposable systems.
Organization Science.

Three authority structures on a block-diagonal NK landscape:
  single_boss    : one decision-maker evaluates all 1-flip proposals and picks the best
  autonomous     : each sub-unit independently adopts improvements within its block
  multiauthority : sub-units independently search; shared decisions require majority approval
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape


def _single_boss_step(
    landscape: NKLandscape,
    state: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick the single best 1-flip improvement across all N bits."""
    neighbours = landscape.neighbors_1flip(state)
    fs = landscape.fitness_batch(np.array(neighbours))
    best_idx = int(np.argmax(fs))
    if fs[best_idx] > landscape.fitness(state):
        return neighbours[best_idx]
    return state


def _autonomous_step(
    landscape: NKLandscape,
    state: np.ndarray,
    block_ranges: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Each block independently picks a random 1-flip and adopts if block-fitness improves."""
    new_state = state.copy()
    for bstart, bend in block_ranges:
        bit = int(rng.integers(bstart, bend))
        candidate = new_state.copy()
        candidate[bit] ^= 1
        block_bits = list(range(bstart, bend))
        old_f = sum(landscape.phi[j][landscape._component_index(new_state, j)] for j in block_bits)
        new_f = sum(landscape.phi[j][landscape._component_index(candidate, j)] for j in block_bits)
        if new_f > old_f:
            new_state = candidate
    return new_state


def _multiauthority_step(
    landscape: NKLandscape,
    state: np.ndarray,
    block_ranges: list[tuple[int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Each block proposes improvements; cross-block changes need majority (>= 50%) of blocks to agree.
    Simplified: each block evaluates its own bit-flip; shared bits need agreement from both adjacent blocks.
    Here we implement as: each block proposes, then proposals are adopted only if they improve global fitness.
    """
    new_state = state.copy()
    proposals: list[Optional[np.ndarray]] = []
    for bstart, bend in block_ranges:
        bit = int(rng.integers(bstart, bend))
        candidate = new_state.copy()
        candidate[bit] ^= 1
        block_bits = list(range(bstart, bend))
        old_f = sum(landscape.phi[j][landscape._component_index(new_state, j)] for j in block_bits)
        new_f = sum(landscape.phi[j][landscape._component_index(candidate, j)] for j in block_bits)
        proposals.append(candidate if new_f > old_f else None)

    # Only adopt proposals that also improve global fitness (cross-block coordination check)
    for prop in proposals:
        if prop is not None and landscape.fitness(prop) > landscape.fitness(new_state):
            new_state = prop
    return new_state


def simulate(
    N: int = 12,
    K: int = 3,
    block_sizes: Optional[list[int]] = None,
    n_periods: int = 200,
    n_runs: int = 100,
    seed: int = 42,
) -> dict[str, list]:
    """
    Compare three authority structures on a block-diagonal NK landscape.

    Returns
    -------
    dict with keys 'single_boss', 'autonomous', 'multiauthority',
    each a list of mean performance over periods.
    """
    if block_sizes is None:
        block_sizes = [N // 3, N // 3, N - 2 * (N // 3)]

    block_ranges = []
    start = 0
    for bs in block_sizes:
        block_ranges.append((start, start + bs))
        start += bs

    rng = np.random.default_rng(seed)
    accum: dict[str, list[np.ndarray]] = {
        "single_boss": [], "autonomous": [], "multiauthority": []
    }

    for _ in range(n_runs):
        ls = NKLandscape(
            N, K, interaction_type="block_diagonal",
            block_sizes=block_sizes, seed=int(rng.integers(0, 2**31))
        )
        s0 = ls.random_state()

        for mode in ("single_boss", "autonomous", "multiauthority"):
            state = s0.copy()
            traj = [ls.fitness(state)]
            for _ in range(n_periods):
                if mode == "single_boss":
                    state = _single_boss_step(ls, state, rng)
                elif mode == "autonomous":
                    state = _autonomous_step(ls, state, block_ranges, rng)
                else:
                    state = _multiauthority_step(ls, state, block_ranges, rng)
                traj.append(ls.fitness(state))
            accum[mode].append(np.array(traj))

    mean_perf = {m: np.mean(np.array(v), axis=0).tolist() for m, v in accum.items()}
    return {
        **mean_perf,
        "params": {"N": N, "K": K, "block_sizes": block_sizes, "n_periods": n_periods},
    }
