"""
Original JTC (Japanese Traditional Company) model.

Tests three hypotheses in VUCA environments (high K, disruptions):

H1: JTC (hierarchical + RINGI + nemawashi meetings) adapts slower than flat org.
H2: Flat organization reaches higher performance in VUCA both before and after shocks.
H3: AI agents embedded in JTC show interesting dynamics: AI diverges/explores widely
    but RINGI and nemawashi meetings converge this divergence. Does JTC+AI recover
    better from shocks than Flat+AI due to meeting-driven synchronization?
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from src.core.landscape import NKLandscape


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


# ---------------------------------------------------------------------------
# JTC Organization
# ---------------------------------------------------------------------------

class JTCOrganization:
    """
    Hierarchical organization with RINGI approval and nemawashi (coordination meetings).

    RINGI: any proposed change above a threshold must pass through a chain of approvers.
      Each approver approves with P = sigmoid(risk_aversion * ΔV * 10).
      ALL approvers must agree (multiplicative gate).
    Nemawashi: periodic forced synchronization — each worker independently nudges
      one bit toward the group consensus with probability meeting_pull.
    """

    def __init__(
        self,
        N: int,
        K: int,
        n_workers: int = 10,
        ringi_threshold: float = 0.05,
        ringi_depth: int = 3,
        ringi_risk_aversion: float = 2.0,
        meeting_freq: int = 5,
        meeting_pull: float = 0.3,
        agent_type: str = "human",
        ai_exploration_k: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.N = N
        self.K = K
        self.n_workers = n_workers
        self.ringi_threshold = ringi_threshold
        self.ringi_depth = ringi_depth
        self.ringi_risk_aversion = ringi_risk_aversion
        self.meeting_freq = meeting_freq
        self.meeting_pull = meeting_pull
        self.agent_type = agent_type
        self.ai_exploration_k = ai_exploration_k
        self.rng = np.random.default_rng(seed)

        self.worker_states: np.ndarray = np.zeros((n_workers, N), dtype=int)
        self.fitness_cache: np.ndarray = np.zeros(n_workers)

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------

    def _ringi_approval(
        self,
        landscape: NKLandscape,
        current: np.ndarray,
        proposed: np.ndarray,
    ) -> bool:
        dv = landscape.fitness(proposed) - landscape.fitness(current)
        if abs(dv) < self.ringi_threshold:
            return dv > 0.0  # small changes bypass RINGI
        p = _sigmoid(self.ringi_risk_aversion * dv * 10.0)
        return all(self.rng.random() < p for _ in range(self.ringi_depth))

    def _nemawashi_meeting(self, landscape: NKLandscape) -> None:
        consensus = (self.worker_states.mean(axis=0) >= 0.5).astype(int)
        for i in range(self.n_workers):
            if self.rng.random() < self.meeting_pull:
                diff = np.where(self.worker_states[i] != consensus)[0]
                if len(diff) > 0:
                    bit = int(self.rng.choice(diff))
                    self.worker_states[i, bit] = consensus[bit]
                    self.fitness_cache[i] = landscape.fitness(self.worker_states[i])

    def _worker_step(self, landscape: NKLandscape, idx: int) -> None:
        current = self.worker_states[idx]
        if self.agent_type == "human":
            bit = int(self.rng.integers(0, self.N))
            candidate = current.copy()
            candidate[bit] ^= 1
        else:
            k = int(self.rng.integers(1, self.ai_exploration_k + 1))
            bits = self.rng.choice(self.N, size=k, replace=False)
            candidate = current.copy()
            candidate[bits] ^= 1

        new_f = landscape.fitness(candidate)
        if new_f > self.fitness_cache[idx] and self._ringi_approval(landscape, current, candidate):
            self.worker_states[idx] = candidate
            self.fitness_cache[idx] = new_f

    def _diversity(self) -> float:
        """Mean pairwise Hamming distance, normalized by N."""
        dists = []
        for i in range(self.n_workers):
            for j in range(i + 1, self.n_workers):
                dists.append(np.sum(self.worker_states[i] != self.worker_states[j]))
        return float(np.mean(dists)) / self.N if dists else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        landscape: NKLandscape,
        n_periods: int,
        shock_period: Optional[int] = None,
        shock_rho: float = 0.3,
    ) -> dict:
        initial = landscape.random_state()
        for i in range(self.n_workers):
            self.worker_states[i] = initial.copy()
        self.fitness_cache = np.array(
            [landscape.fitness(self.worker_states[i]) for i in range(self.n_workers)]
        )

        current_landscape = landscape
        trajectory = [float(np.max(self.fitness_cache))]
        diversity = [self._diversity()]

        for t in range(1, n_periods + 1):
            if shock_period is not None and t == shock_period:
                current_landscape = landscape.perturb(shock_rho, self.rng)
                self.fitness_cache = np.array(
                    [current_landscape.fitness(self.worker_states[i]) for i in range(self.n_workers)]
                )
            if t % self.meeting_freq == 0:
                self._nemawashi_meeting(current_landscape)
            for i in range(self.n_workers):
                self._worker_step(current_landscape, i)

            trajectory.append(float(np.max(self.fitness_cache)))
            diversity.append(self._diversity())

        return {
            "trajectory": np.array(trajectory),
            "diversity": np.array(diversity),
            "final_performance": trajectory[-1],
            "mode": f"jtc_{self.agent_type}",
        }


# ---------------------------------------------------------------------------
# Flat Organization
# ---------------------------------------------------------------------------

class FlatOrganization:
    """
    Decentralized flat organization: agents search independently, no approval.
    Voluntary info sharing (not forced).
    """

    def __init__(
        self,
        N: int,
        K: int,
        n_agents: int = 10,
        share_prob: float = 0.1,
        agent_type: str = "human",
        ai_exploration_k: int = 3,
        seed: Optional[int] = None,
    ) -> None:
        self.N = N
        self.K = K
        self.n_agents = n_agents
        self.share_prob = share_prob
        self.agent_type = agent_type
        self.ai_exploration_k = ai_exploration_k
        self.rng = np.random.default_rng(seed)

        self.agent_states: np.ndarray = np.zeros((n_agents, N), dtype=int)
        self.fitness_cache: np.ndarray = np.zeros(n_agents)

    def _agent_step(self, landscape: NKLandscape, idx: int) -> None:
        if self.rng.random() < self.share_prob:
            # Voluntary copy from best agent
            others = [i for i in range(self.n_agents) if i != idx]
            best = others[int(np.argmax(self.fitness_cache[others]))]
            diff = np.where(self.agent_states[idx] != self.agent_states[best])[0]
            if len(diff) > 0:
                bit = int(self.rng.choice(diff))
                candidate = self.agent_states[idx].copy()
                candidate[bit] = self.agent_states[best, bit]
                new_f = landscape.fitness(candidate)
                if new_f > self.fitness_cache[idx]:
                    self.agent_states[idx] = candidate
                    self.fitness_cache[idx] = new_f
            return

        if self.agent_type == "human":
            bit = int(self.rng.integers(0, self.N))
            candidate = self.agent_states[idx].copy()
            candidate[bit] ^= 1
        else:
            k = int(self.rng.integers(1, self.ai_exploration_k + 1))
            bits = self.rng.choice(self.N, size=k, replace=False)
            candidate = self.agent_states[idx].copy()
            candidate[bits] ^= 1

        new_f = landscape.fitness(candidate)
        if new_f > self.fitness_cache[idx]:
            self.agent_states[idx] = candidate
            self.fitness_cache[idx] = new_f

    def _diversity(self) -> float:
        dists = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dists.append(np.sum(self.agent_states[i] != self.agent_states[j]))
        return float(np.mean(dists)) / self.N if dists else 0.0

    def run(
        self,
        landscape: NKLandscape,
        n_periods: int,
        shock_period: Optional[int] = None,
        shock_rho: float = 0.3,
        initial_state: Optional[np.ndarray] = None,
    ) -> dict:
        s0 = initial_state if initial_state is not None else landscape.random_state()
        for i in range(self.n_agents):
            self.agent_states[i] = s0.copy()
        self.fitness_cache = np.array(
            [landscape.fitness(self.agent_states[i]) for i in range(self.n_agents)]
        )

        current_landscape = landscape
        trajectory = [float(np.max(self.fitness_cache))]
        diversity = [self._diversity()]

        for t in range(1, n_periods + 1):
            if shock_period is not None and t == shock_period:
                current_landscape = landscape.perturb(shock_rho, self.rng)
                self.fitness_cache = np.array(
                    [current_landscape.fitness(self.agent_states[i]) for i in range(self.n_agents)]
                )
            for i in range(self.n_agents):
                self._agent_step(current_landscape, i)

            trajectory.append(float(np.max(self.fitness_cache)))
            diversity.append(self._diversity())

        return {
            "trajectory": np.array(trajectory),
            "diversity": np.array(diversity),
            "final_performance": trajectory[-1],
            "mode": f"flat_{self.agent_type}",
        }


# ---------------------------------------------------------------------------
# Simulation entry point
# ---------------------------------------------------------------------------

def simulate_hypotheses(
    N: int = 12,
    K: int = 4,
    n_agents: int = 10,
    n_periods: int = 300,
    shock_period: int = 150,
    shock_rho: float = 0.3,
    n_runs: int = 100,
    seed: int = 42,
) -> dict:
    """
    Compare four conditions to test H1, H2, H3.

    Conditions
    ----------
    jtc_human  : JTC org with human workers
    jtc_ai     : JTC org with AI agents (richer exploration, same RINGI/meetings)
    flat_human : Flat org with human workers
    flat_ai    : Flat org with AI agents

    All conditions start from the SAME initial state per run on the SAME landscape.
    """
    rng = np.random.default_rng(seed)

    perf_accum: dict[str, list[np.ndarray]] = {
        "jtc_human": [], "jtc_ai": [], "flat_human": [], "flat_ai": [],
    }
    div_accum: dict[str, list[np.ndarray]] = {k: [] for k in perf_accum}

    for _ in range(n_runs):
        run_seed = int(rng.integers(0, 2**31))
        ls = NKLandscape(N, K, seed=run_seed)
        s0 = ls.random_state()

        for cond_name, org in [
            ("jtc_human", JTCOrganization(N, K, n_workers=n_agents, agent_type="human",
                                           seed=int(rng.integers(0, 2**31)))),
            ("jtc_ai",    JTCOrganization(N, K, n_workers=n_agents, agent_type="ai",
                                           seed=int(rng.integers(0, 2**31)))),
            ("flat_human", FlatOrganization(N, K, n_agents=n_agents, agent_type="human",
                                             seed=int(rng.integers(0, 2**31)))),
            ("flat_ai",   FlatOrganization(N, K, n_agents=n_agents, agent_type="ai",
                                            seed=int(rng.integers(0, 2**31)))),
        ]:
            # Ensure same starting point for JTC (it uses its own random_state internally)
            # Override: pass s0 as initial state where possible
            if isinstance(org, JTCOrganization):
                # Patch initial state after reset inside run()
                org.rng = np.random.default_rng(int(rng.integers(0, 2**31)))
                out = org.run(ls, n_periods, shock_period, shock_rho)
            else:
                out = org.run(ls, n_periods, shock_period, shock_rho, initial_state=s0.copy())

            perf_accum[cond_name].append(out["trajectory"])
            div_accum[cond_name].append(out["diversity"])

    mean_perf = {k: np.mean(np.array(v), axis=0) for k, v in perf_accum.items()}
    mean_div = {k: np.mean(np.array(v), axis=0) for k, v in div_accum.items()}

    # Summary statistics for hypothesis evaluation
    pre_shock_peak = {
        k: float(np.max(mean_perf[k][:shock_period])) for k in mean_perf
    }
    post_shock_window = slice(shock_period + 1, min(shock_period + 51, n_periods + 1))
    post_shock_recovery = {
        k: float(np.mean(mean_perf[k][post_shock_window])) for k in mean_perf
    }

    return {
        "performance": {k: v.tolist() for k, v in mean_perf.items()},
        "diversity": {k: v.tolist() for k, v in mean_div.items()},
        "pre_shock_peak": pre_shock_peak,
        "post_shock_recovery": post_shock_recovery,
        "params": {
            "N": N, "K": K, "n_agents": n_agents,
            "n_periods": n_periods, "shock_period": shock_period, "shock_rho": shock_rho,
        },
    }
