"""
Leitner (2023) — Building resilient organizations: top-down vs. bottom-up organizing.

Key mechanisms from paper:
- M agents, each responsible for N/M tasks (task block d_m)
- Utility: U(d_m, d_{-m}) = λ·P(d_m) + (1-λ)·P(d_{-m})
  λ=1 → individualistic (own tasks only); λ=0.33 → altruistic (also cares about residual)
- Search: Hamming-1 neighbourhood of own tasks; adopt if utility improves
- Top-down: task blocks fixed (agent m owns tasks (m-1)*N/M .. m*N/M)
- Bottom-up: task blocks dynamic; every τ periods agents may swap tasks
- Disruptions: correlated NK shock via NKLandscape.perturb(rho)
- Normalized performance: P̄_t = mean over runs of P(d_t) / P^max_st
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape


def _partial_perf(landscape: NKLandscape, state: np.ndarray, task_ids: list[int]) -> float:
    """Mean fitness contribution for a subset of tasks."""
    return float(
        sum(landscape.phi[j][landscape._component_index(state, j)] for j in task_ids)
        / len(task_ids)
    )


def _utility(
    landscape: NKLandscape,
    state: np.ndarray,
    own_tasks: list[int],
    residual_tasks: list[int],
    lam: float,
) -> float:
    """U = λ·P(own) + (1-λ)·P(residual). Eq. 4 in paper."""
    p_own = _partial_perf(landscape, state, own_tasks)
    if not residual_tasks:
        return p_own
    p_res = _partial_perf(landscape, state, residual_tasks)
    return lam * p_own + (1 - lam) * p_res


def _agent_search(
    landscape: NKLandscape,
    state: np.ndarray,
    own_tasks: list[int],
    residual_tasks: list[int],
    lam: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Hamming-1 search within own task block; adopt if utility improves. Eq. 5.
    """
    u_current = _utility(landscape, state, own_tasks, residual_tasks, lam)
    bit = int(rng.choice(own_tasks))
    candidate = state.copy()
    candidate[bit] ^= 1
    if _utility(landscape, candidate, own_tasks, residual_tasks, lam) > u_current:
        return candidate
    return state


def _bottom_up_reallocate(
    task_assignments: list[list[int]],
    landscape: NKLandscape,
    state: np.ndarray,
    lam: float,
    C: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Simplified bottom-up task reallocation (every τ periods).
    Each agent offers its least-valued task; another agent accepts if it improves utility.
    """
    M = len(task_assignments)
    new_assignments = [list(a) for a in task_assignments]

    for m in range(M):
        own = new_assignments[m]
        if len(own) <= 1:
            continue
        # Find task to offer: the one contributing least to own utility
        worst_task = min(own, key=lambda j: landscape.phi[j][landscape._component_index(state, j)])

        # Find another agent with capacity that gains from taking this task
        for n in rng.permutation(M):
            if n == m or len(new_assignments[n]) >= C:
                continue
            # Check if n gains utility from taking worst_task
            n_own = new_assignments[n]
            n_res = [j for k, tasks in enumerate(new_assignments) if k != n for j in tasks]
            u_before = _utility(landscape, state, n_own, n_res, lam)
            trial_n = n_own + [worst_task]
            trial_res = [j for j in n_res if j != worst_task]
            u_after = _utility(landscape, state, trial_n, trial_res, lam)
            if u_after > u_before:
                new_assignments[m].remove(worst_task)
                new_assignments[n].append(worst_task)
                break

    return new_assignments


def _run_single(
    landscape: NKLandscape,
    mode: str,
    M: int,
    lam: float,
    n_periods: int,
    shock_period: Optional[int],
    shock_rho: float,
    tau: int,
    C: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """
    Single run. Returns (trajectory, p_max) where trajectory is raw performance
    and p_max is the max achievable on this landscape (for normalization).
    """
    N = landscape.N
    state = landscape.random_state()

    # Task allocation
    block = N // M
    if mode == "top_down":
        task_assignments = [list(range(m * block, (m + 1) * block)) for m in range(M)]
        # handle remainder
        for j in range(M * block, N):
            task_assignments[-1].append(j)
    else:
        # Bottom-up starts same as top-down, then reallocates dynamically
        task_assignments = [list(range(m * block, (m + 1) * block)) for m in range(M)]
        for j in range(M * block, N):
            task_assignments[-1].append(j)

    # Global max for normalization (only feasible for small N)
    try:
        _, p_max = landscape.global_maximum()
    except ValueError:
        p_max = 1.0

    current = landscape
    traj = [landscape.fitness(state)]

    for t in range(1, n_periods + 1):
        if shock_period is not None and t == shock_period:
            current = landscape.perturb(shock_rho, rng)

        # Bottom-up: reallocate tasks every τ periods
        if mode == "bottom_up" and t % tau == 0:
            task_assignments = _bottom_up_reallocate(
                task_assignments, current, state, lam, C, rng
            )

        # Each agent searches within own tasks
        for m in range(M):
            own = task_assignments[m]
            residual = [j for k, tasks in enumerate(task_assignments) if k != m for j in tasks]
            state = _agent_search(current, state, own, residual, lam, rng)

        traj.append(current.fitness(state))

    return np.array(traj), p_max


def simulate(
    N: int = 15,
    K: int = 3,
    M: int = 5,
    lam_values: Optional[list[float]] = None,
    n_periods: int = 200,
    shock_period: Optional[int] = 50,
    shock_rho: float = 0.5,
    tau: int = 20,
    C: int = 7,
    n_runs: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compare top-down vs bottom-up for each λ (incentive) value.

    Parameters match Table 1: N=15, M=5, C=7, τ=20, shock at t=50,
    ρ ∈ {-0.5, 0.5}, λ ∈ {0.33, 1}.

    Returns dict with keys like 'top_down_lam0.33', 'bottom_up_lam1.0' etc.,
    each a normalized performance trajectory.
    """
    if lam_values is None:
        lam_values = [0.33, 1.0]

    rng = np.random.default_rng(seed)
    keys = [f"{mode}_lam{lam}" for mode in ("top_down", "bottom_up") for lam in lam_values]
    accum: dict[str, list[np.ndarray]] = {k: [] for k in keys}

    for _ in range(n_runs):
        ls = NKLandscape(N, K, seed=int(rng.integers(0, 2**31)))
        for mode in ("top_down", "bottom_up"):
            for lam in lam_values:
                traj, p_max = _run_single(
                    ls, mode, M, lam, n_periods, shock_period, shock_rho, tau, C, rng
                )
                norm_traj = traj / p_max if p_max > 0 else traj
                accum[f"{mode}_lam{lam}"].append(norm_traj)

    mean_perf = {k: np.mean(np.array(v), axis=0).tolist() for k, v in accum.items()}

    sp = shock_period if shock_period is not None else n_periods
    pre_peak = {k: float(np.max(np.array(v)[:sp])) for k, v in mean_perf.items()}
    post_window = slice(sp + 1, min(sp + 51, n_periods + 1))
    post_recovery = {k: float(np.mean(np.array(v)[post_window])) for k, v in mean_perf.items()}

    return {
        **mean_perf,
        "pre_shock_peak": pre_peak,
        "post_shock_recovery": post_recovery,
        "params": {
            "N": N, "K": K, "M": M, "lam_values": lam_values,
            "shock_period": shock_period, "shock_rho": shock_rho,
            "tau": tau, "C": C,
        },
    }
