"""
Levinthal & Workiewicz (2018) — When Two Bosses Are Better Than One.
Organization Science.

Two-level hierarchical model:
  - N_p subunits, each with N_d detailed decisions (DDs)
  - Superiors make policy decisions (PDs): majority vote of DDs in their subunit
  - Three organizational forms:
      SA (single authority): one superior over all N_p subunits
      AU (autonomous):       no superiors; managers search DDs freely
      MA (multiauthority):   N_p/2 superiors, each covering 2 subunits (matrix)
  - α: weight of integration (higher-level) vs specialization (lower-level)
  - Fitness: π = α·V(f_p) + (1-α)·(1/N_p)·ΣV(f_{p,d})
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape


def _make_two_level_landscape(
    N_p: int,
    N_d: int,
    K_p: int,
    K_d: int,
    rng: np.random.Generator,
) -> tuple[NKLandscape, NKLandscape]:
    """
    Create two NK landscapes:
      - detail_ls: N_p*N_d decisions, block-diagonal with K_d within-subunit couplings
      - policy_ls: N_p decisions with K_p cross-subunit couplings
    """
    seed_d = int(rng.integers(0, 2**31))
    seed_p = int(rng.integers(0, 2**31))
    block_sizes = [N_d] * N_p
    detail_ls = NKLandscape(N_p * N_d, K_d, interaction_type="block_diagonal",
                            block_sizes=block_sizes, seed=seed_d)
    policy_ls = NKLandscape(N_p, K_p, interaction_type="random", seed=seed_p)
    return detail_ls, policy_ls


def _dd_to_pd(dd_state: np.ndarray, N_p: int, N_d: int) -> np.ndarray:
    """Convert N_p*N_d detailed decisions to N_p policy decisions via majority vote."""
    pd = np.zeros(N_p, dtype=int)
    for p in range(N_p):
        block = dd_state[p * N_d:(p + 1) * N_d]
        pd[p] = int(block.mean() >= 0.5)
    return pd


def _combined_fitness(
    detail_ls: NKLandscape,
    policy_ls: NKLandscape,
    dd_state: np.ndarray,
    N_p: int,
    N_d: int,
    alpha: float,
) -> float:
    """π = α·V(f_p) + (1-α)·V(f_{p,d}). Eq. 3 in paper."""
    pd = _dd_to_pd(dd_state, N_p, N_d)
    f_policy = policy_ls.fitness(pd)
    f_detail = detail_ls.fitness(dd_state)
    return alpha * f_policy + (1 - alpha) * f_detail


def _manager_search(
    detail_ls: NKLandscape,
    dd_state: np.ndarray,
    subunit: int,
    N_d: int,
    constraint_pd: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Manager searches DDs within own subunit (Hamming-1, within block).
    If constraint_pd is not None, new state must be consistent with that PD value
    (i.e. majority vote of block must equal constraint_pd).
    """
    start = subunit * N_d
    bit = int(rng.integers(start, start + N_d))
    candidate = dd_state.copy()
    candidate[bit] ^= 1

    if constraint_pd is not None:
        block = candidate[start:start + N_d]
        if int(block.mean() >= 0.5) != constraint_pd:
            return dd_state  # violates policy directive

    block_bits = list(range(start, start + N_d))
    old_f = sum(detail_ls.phi[j][detail_ls._component_index(dd_state, j)] for j in block_bits)
    new_f = sum(detail_ls.phi[j][detail_ls._component_index(candidate, j)] for j in block_bits)
    return candidate if new_f > old_f else dd_state


def _superior_search(
    policy_ls: NKLandscape,
    pd_state: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Superior picks random PD and flips it (Hamming-1 at policy level). Returns (new_pd, flipped_subunit)."""
    subunit = int(rng.integers(0, policy_ls.N))
    new_pd = pd_state.copy()
    new_pd[subunit] ^= 1
    return new_pd, subunit


def _run_sa(
    detail_ls: NKLandscape,
    policy_ls: NKLandscape,
    N_p: int,
    N_d: int,
    alpha: float,
    n_periods: int,
    rng: np.random.Generator,
) -> list[float]:
    """Single Authority: one superior + N_p managers."""
    dd = detail_ls.random_state()
    pd = _dd_to_pd(dd, N_p, N_d)
    traj = [_combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha)]

    for _ in range(n_periods):
        # Superior proposes new PD
        new_pd, changed_subunit = _superior_search(policy_ls, pd, rng)

        # Manager of changed subunit searches DDs consistent with new PD
        new_dd = _manager_search(detail_ls, dd, changed_subunit, N_d,
                                 constraint_pd=new_pd[changed_subunit], rng=rng)
        new_pd_actual = _dd_to_pd(new_dd, N_p, N_d)

        # Superior accepts if combined fitness improves
        f_new = _combined_fitness(detail_ls, policy_ls, new_dd, N_p, N_d, alpha)
        f_old = _combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha)
        if f_new > f_old:
            dd, pd = new_dd, new_pd_actual

        traj.append(_combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha))
    return traj


def _run_au(
    detail_ls: NKLandscape,
    N_p: int,
    N_d: int,
    alpha: float,
    n_periods: int,
    rng: np.random.Generator,
    policy_ls: NKLandscape,
) -> list[float]:
    """Autonomous: no superiors, each manager searches DDs freely."""
    dd = detail_ls.random_state()
    traj = [_combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha)]

    for _ in range(n_periods):
        subunit = int(rng.integers(0, N_p))
        new_dd = _manager_search(detail_ls, dd, subunit, N_d, constraint_pd=None, rng=rng)
        f_new = _combined_fitness(detail_ls, policy_ls, new_dd, N_p, N_d, alpha)
        f_old = _combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha)
        if f_new > f_old:
            dd = new_dd
        traj.append(_combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha))
    return traj


def _run_ma(
    detail_ls: NKLandscape,
    policy_ls: NKLandscape,
    N_p: int,
    N_d: int,
    alpha: float,
    n_periods: int,
    rng: np.random.Generator,
) -> list[float]:
    """
    Multiauthority (matrix): N_p/2 superiors, each covers 2 subunits.
    If two superiors that cover the same manager disagree, manager is unconstrained.
    Layout: superior s covers subunits (s, s + N_p//2) for s in range(N_p//2).
    """
    n_sup = N_p // 2
    dd = detail_ls.random_state()
    pd = _dd_to_pd(dd, N_p, N_d)
    # Each superior's current PD proposals (one per subunit they cover)
    sup_pds = pd.copy()  # superiors start aligned with actual PD

    traj = [_combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha)]

    for _ in range(n_periods):
        # Pick a random superior, flip one of their PDs
        s = int(rng.integers(0, n_sup))
        subunits_covered = [s, s + n_sup]
        changed_sub = subunits_covered[int(rng.integers(0, 2))]
        new_sup_pds = sup_pds.copy()
        new_sup_pds[changed_sub] ^= 1

        # Check for conflicts: does any other superior also cover this subunit?
        # In our layout each subunit is covered by exactly 2 superiors (s and s+n_sup)
        other_sup_for_sub = (changed_sub + n_sup) % N_p
        agree = (new_sup_pds[changed_sub] == new_sup_pds[other_sup_for_sub])

        constraint = new_sup_pds[changed_sub] if agree else None
        new_dd = _manager_search(detail_ls, dd, changed_sub, N_d, constraint_pd=constraint, rng=rng)

        f_new = _combined_fitness(detail_ls, policy_ls, new_dd, N_p, N_d, alpha)
        f_old = _combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha)
        if f_new > f_old:
            dd = new_dd
            sup_pds = new_sup_pds

        traj.append(_combined_fitness(detail_ls, policy_ls, dd, N_p, N_d, alpha))
    return traj


def simulate(
    N_p: int = 4,
    N_d: int = 3,
    K_p: int = 3,
    K_d: int = 2,
    alpha_values: Optional[list[float]] = None,
    n_periods: int = 200,
    n_runs: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compare SA / AU / MA across α values (integration vs specialization weight).

    Default parameters match paper: N_p=4, N_d=3, K_p=3, K_d=2.
    α→0: integration important; α→1: specialization important.

    Returns dict with keys like 'SA_a0.2', 'AU_a0.2', 'MA_a0.2', etc.
    """
    if alpha_values is None:
        alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    rng = np.random.default_rng(seed)
    forms = ("SA", "AU", "MA")
    keys = [f"{f}_a{a}" for f in forms for a in alpha_values]
    accum: dict[str, list[list[float]]] = {k: [] for k in keys}

    for _ in range(n_runs):
        detail_ls, policy_ls = _make_two_level_landscape(N_p, N_d, K_p, K_d, rng)
        for alpha in alpha_values:
            traj_sa = _run_sa(detail_ls, policy_ls, N_p, N_d, alpha, n_periods, rng)
            traj_au = _run_au(detail_ls, N_p, N_d, alpha, n_periods, rng, policy_ls)
            traj_ma = _run_ma(detail_ls, policy_ls, N_p, N_d, alpha, n_periods, rng)
            accum[f"SA_a{alpha}"].append(traj_sa)
            accum[f"AU_a{alpha}"].append(traj_au)
            accum[f"MA_a{alpha}"].append(traj_ma)

    mean_perf = {k: np.mean(np.array(v), axis=0).tolist() for k, v in accum.items()}

    # Final performance per alpha for easy comparison
    final = {k: float(np.array(v)[-1]) for k, v in mean_perf.items()}

    return {
        **mean_perf,
        "final_performance": final,
        "params": {
            "N_p": N_p, "N_d": N_d, "K_p": K_p, "K_d": K_d,
            "alpha_values": alpha_values, "n_periods": n_periods,
        },
    }
