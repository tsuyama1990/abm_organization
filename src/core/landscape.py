"""
NK fitness landscape — the core engine used by every organizational model.

Kauffman (1993):
  N = number of binary decisions (string length)
  K = epistatic coupling (0 ≤ K ≤ N-1)
  K=0  → smooth, single peak
  K=N-1 → maximally rugged, ~2^N/(N+1) local peaks
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional


class NKLandscape:
    """
    NK model fitness landscape.

    Each binary string x ∈ {0,1}^N maps to a scalar fitness Φ(x) ∈ (0,1).
    Fitness is the average of N component contributions φ_i, where each φ_i
    depends on x_i and K other "influencer" bits (epistasis).
    """

    def __init__(
        self,
        N: int,
        K: int,
        interaction_type: Literal["adjacent", "random", "block_diagonal"] = "adjacent",
        block_sizes: Optional[list[int]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.N = N
        self.K = K
        self.rng = np.random.default_rng(seed)

        # interaction_matrix[i] = [i, influencer_1, ..., influencer_K]  shape (N, K+1)
        self.interaction_matrix = self._build_interactions(interaction_type, block_sizes)

        # phi[i][state_idx] = fitness contribution of component i  (2^(K+1) values each)
        self.phi: list[np.ndarray] = [
            self.rng.uniform(0.0, 1.0, 2 ** (K + 1)) for _ in range(N)
        ]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_interactions(
        self,
        interaction_type: str,
        block_sizes: Optional[list[int]],
    ) -> np.ndarray:
        N, K = self.N, self.K

        if interaction_type == "adjacent":
            # Canonical Kauffman: K right-neighbours on a ring
            return np.array(
                [[i] + [(i + k + 1) % N for k in range(K)] for i in range(N)],
                dtype=int,
            )

        if interaction_type == "random":
            mat = []
            for i in range(N):
                others = [j for j in range(N) if j != i]
                chosen = self.rng.choice(others, size=K, replace=False).tolist()
                mat.append([i] + chosen)
            return np.array(mat, dtype=int)

        if interaction_type == "block_diagonal":
            if block_sizes is None:
                raise ValueError("block_sizes required for block_diagonal")
            if sum(block_sizes) != N:
                raise ValueError(f"sum(block_sizes)={sum(block_sizes)} ≠ N={N}")

            # Map each component to its block's range
            component_block: list[tuple[int, int]] = []
            start = 0
            for bs in block_sizes:
                for _ in range(bs):
                    component_block.append((start, start + bs))
                start += bs

            mat = []
            for i in range(N):
                bstart, bend = component_block[i]
                within = [j for j in range(bstart, bend) if j != i]
                K_eff = min(K, len(within))
                if within and K_eff > 0:
                    chosen = self.rng.choice(within, size=K_eff, replace=False).tolist()
                else:
                    chosen = []
                # Pad with i itself when block is smaller than K
                chosen += [i] * (K - K_eff)
                mat.append([i] + chosen)
            return np.array(mat, dtype=int)

        raise ValueError(f"Unknown interaction_type: {interaction_type!r}")

    # ------------------------------------------------------------------
    # Core fitness calculation
    # ------------------------------------------------------------------

    def _component_index(self, state: np.ndarray, i: int) -> int:
        """Convert the (K+1)-bit sub-state of component i to an integer index."""
        bits = state[self.interaction_matrix[i]]
        # big-endian: first bit is most significant
        return int(bits.dot(1 << np.arange(len(bits) - 1, -1, -1)))

    def fitness(self, state: np.ndarray) -> float:
        """Fitness Φ(x) = (1/N) Σ φ_i(x).  O(N·K)."""
        s = np.asarray(state, dtype=int)
        return float(
            sum(self.phi[i][self._component_index(s, i)] for i in range(self.N)) / self.N
        )

    def fitness_batch(self, states: np.ndarray) -> np.ndarray:
        """Vectorised fitness for a 2-D array of states (n_states × N)."""
        states = np.asarray(states, dtype=int)
        if states.ndim == 1:
            return np.array([self.fitness(states)])
        return np.array([self.fitness(s) for s in states])

    # ------------------------------------------------------------------
    # Neighbourhood
    # ------------------------------------------------------------------

    def neighbors_1flip(self, state: np.ndarray) -> list[np.ndarray]:
        """Return all N states at Hamming distance 1."""
        s = np.asarray(state, dtype=int)
        result = []
        for i in range(self.N):
            n = s.copy()
            n[i] ^= 1
            result.append(n)
        return result

    def best_neighbor(self, state: np.ndarray) -> tuple[np.ndarray, float]:
        """Return the fittest 1-flip neighbour and its fitness."""
        neighbours = self.neighbors_1flip(state)
        fs = self.fitness_batch(np.array(neighbours))
        idx = int(np.argmax(fs))
        return neighbours[idx], float(fs[idx])

    def local_maximum(self, state: np.ndarray) -> bool:
        """True if no single-bit flip improves fitness."""
        f = self.fitness(state)
        return all(self.fitness(n) <= f for n in self.neighbors_1flip(state))

    # ------------------------------------------------------------------
    # Global search (exhaustive — feasible only for N ≤ 20)
    # ------------------------------------------------------------------

    def global_maximum(self) -> tuple[np.ndarray, float]:
        if self.N > 20:
            raise ValueError(f"N={self.N} too large for exhaustive search (max 20)")
        best_f, best_s = -1.0, None
        for i in range(2 ** self.N):
            s = np.array([(i >> j) & 1 for j in range(self.N)], dtype=int)
            f = self.fitness(s)
            if f > best_f:
                best_f, best_s = f, s.copy()
        return best_s, best_f  # type: ignore[return-value]

    def count_local_maxima(self) -> int:
        """Count local maxima by exhaustive enumeration (N ≤ 20 only)."""
        if self.N > 20:
            raise ValueError(f"N={self.N} too large")
        return sum(
            1
            for i in range(2 ** self.N)
            if self.local_maximum(
                np.array([(i >> j) & 1 for j in range(self.N)], dtype=int)
            )
        )

    # ------------------------------------------------------------------
    # Disruption / environmental shock  (Leitner 2023, Eq. 2-3)
    # ------------------------------------------------------------------

    def perturb(
        self,
        rho: float,
        rng: Optional[np.random.Generator] = None,
    ) -> "NKLandscape":
        """
        Generate a correlated landscape shift to simulate VUCA disruption.

        rho ∈ (-1, 1):
          rho → +1 : nearly identical to original (mild shock)
          rho → -1 : strongly anti-correlated (catastrophic shock)
        """
        if rng is None:
            rng = self.rng

        # Beta shape parameter (Leitner 2023, Eq. 2)
        a = 0.5 * (np.sqrt((49.0 + rho) / (1.0 + rho)) - 5.0)

        new = NKLandscape.__new__(NKLandscape)
        new.N = self.N
        new.K = self.K
        new.rng = rng
        new.interaction_matrix = self.interaction_matrix.copy()

        new_phi: list[np.ndarray] = []
        for i in range(self.N):
            old = self.phi[i]
            n = len(old)
            v = rng.uniform(0.0, 1.0, n)
            w = rng.beta(a, 1.0, n)
            # Leitner 2023, Eq. 3
            correlated = np.where(v < 0.5, np.abs(w - old), 1.0 - np.abs(w - old))
            new_phi.append(correlated)
        new.phi = new_phi
        return new

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def random_state(self) -> np.ndarray:
        return self.rng.integers(0, 2, size=self.N).astype(int)

    def hill_climb(self, state: np.ndarray, max_steps: int = 10_000) -> tuple[np.ndarray, float]:
        """Greedy hill-climbing from state; returns local maximum and its fitness."""
        s = np.asarray(state, dtype=int).copy()
        for _ in range(max_steps):
            best_n, best_f = self.best_neighbor(s)
            if best_f <= self.fitness(s):
                break
            s = best_n
        return s, self.fitness(s)

    def __repr__(self) -> str:
        return f"NKLandscape(N={self.N}, K={self.K})"
