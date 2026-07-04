"""
Master runner: execute all available models with modest parameters.
Saves JSON results and plots to experiments/results/ and experiments/plots/.

Usage:
    uv run python experiments/run_all.py
"""

import importlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SEPARATOR = "=" * 60


def _save(name: str, data: dict) -> None:
    path = OUTPUT_DIR / f"{name}.json"
    serializable = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            serializable[k] = {
                kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                for kk, vv in v.items()
            }
        else:
            serializable[k] = v
    path.write_text(json.dumps(serializable, indent=2))
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_siggelkow2003() -> dict:
    from src.models.siggelkow2003 import simulate
    from src.viz.plots import plot_performance_trajectories

    print(f"\n{SEPARATOR}")
    print("1. Siggelkow & Levinthal (2003) — Temp-Divide-to-Conquer")
    results = simulate(N=6, K=2, n_periods=100, t_dec=50, n_runs=100, seed=42)
    for c, t in results.items():
        print(f"  {c:>15s}  final={t[-1]:.4f}")
    plot_performance_trajectories(
        results,
        "Siggelkow & Levinthal 2003: N=6, K=2",
        output_path=str(PLOT_DIR / "siggelkow2003.png"),
    )
    return results


def run_fontanari2015() -> dict | None:
    print(f"\n{SEPARATOR}")
    print("2. Fontanari & Rodrigues (2015) — Network topology & cooperative search")
    try:
        mod = importlib.import_module("src.models.fontanari2015")
        results = mod.simulate(N=10, L_values=[1, 2, 4, 8, 16], n_runs=50, seed=42)
        print(f"  Finished. Keys: {list(results.keys())}")
        return results
    except Exception as e:
        print(f"  SKIPPED (Jules not merged yet): {e}")
        return None


def run_levinthal2018() -> dict | None:
    print(f"\n{SEPARATOR}")
    print("3. Levinthal & Workiewicz (2018) — Near-decomposable systems")
    try:
        mod = importlib.import_module("src.models.levinthal2018")
        results = mod.simulate(N=12, K=3, n_periods=200, n_runs=100, seed=42)
        for c, t in results.items():
            arr = np.asarray(t)
            print(f"  {c:>15s}  final={arr[-1]:.4f}")
        from src.viz.plots import plot_performance_trajectories
        plot_performance_trajectories(
            {k: np.asarray(v) for k, v in results.items()},
            "Levinthal & Workiewicz 2018: N=12, K=3",
            output_path=str(PLOT_DIR / "levinthal2018.png"),
        )
        return results
    except Exception as e:
        print(f"  SKIPPED (Jules not merged yet): {e}")
        return None


def run_leitner2023() -> dict | None:
    print(f"\n{SEPARATOR}")
    print("4. Leitner (2023) — Top-down vs bottom-up under disruptions")
    try:
        mod = importlib.import_module("src.models.leitner2023")
        results = mod.simulate(N=10, K=4, n_periods=200, shock_period=100,
                               shock_rho=0.3, n_runs=50, seed=42)
        for c, t in results.items():
            if isinstance(t, (list, np.ndarray)):
                arr = np.asarray(t)
                print(f"  {c:>15s}  final={arr[-1]:.4f}")
        from src.viz.plots import plot_performance_trajectories
        traj_keys = {k: np.asarray(v) for k, v in results.items()
                     if isinstance(v, (list, np.ndarray))}
        plot_performance_trajectories(
            traj_keys,
            "Leitner 2023: N=10, K=4",
            shock_period=100,
            output_path=str(PLOT_DIR / "leitner2023.png"),
        )
        return results
    except Exception as e:
        print(f"  SKIPPED (Jules not merged yet): {e}")
        return None


def run_pykala2026() -> dict | None:
    print(f"\n{SEPARATOR}")
    print("5. Pykälä et al. (2026) — Heterogeneous learning strategies")
    try:
        mod = importlib.import_module("src.models.pykala2026")
        results = mod.simulate(N=12, K=4, n_agents=20, n_generations=100,
                               n_runs=30, seed=42)
        print(f"  Finished. Keys: {list(results.keys())}")
        return results
    except Exception as e:
        print(f"  SKIPPED (Jules not merged yet): {e}")
        return None


def run_jtc_hypotheses() -> dict:
    from src.models.jtc_model import simulate_hypotheses
    from src.viz.plots import plot_jtc_summary

    print(f"\n{SEPARATOR}")
    print("6. JTC Hypothesis Test (H1, H2, H3)")

    results = simulate_hypotheses(
        N=12, K=6, n_agents=10, n_periods=200,
        shock_period=100, shock_rho=0.3,
        n_runs=30, seed=42,
    )

    pre = results["pre_shock_peak"]
    post = results["post_shock_recovery"]

    print(f"\n  Pre-shock peak:")
    for c, v in pre.items():
        print(f"    {c:>15s}: {v:.4f}")
    print(f"\n  Post-shock recovery (50 periods):")
    for c, v in post.items():
        print(f"    {c:>15s}: {v:.4f}")

    h1 = pre["flat_human"] > pre["jtc_human"]
    h2 = max(pre["flat_human"], pre["flat_ai"]) > max(pre["jtc_human"], pre["jtc_ai"])
    h3 = post["jtc_ai"] > post["flat_ai"]

    print(f"\n  H1 (JTC slower than flat): {'✓' if h1 else '✗'}")
    print(f"  H2 (Flat best > JTC best):  {'✓' if h2 else '✗'}")
    print(f"  H3 (JTC+AI better recovery than Flat+AI): {'✓' if h3 else '✗'}")

    plot_jtc_summary(
        perf=results["performance"],
        diversity=results["diversity"],
        shock_period=100,
        output_path=str(PLOT_DIR / "jtc_hypotheses.png"),
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\nNK-ABM Organizational Models — Full Run")
    print(f"Output dir: {OUTPUT_DIR}")

    all_results = {}
    runners = [
        ("siggelkow2003", run_siggelkow2003),
        ("fontanari2015", run_fontanari2015),
        ("levinthal2018", run_levinthal2018),
        ("leitner2023", run_leitner2023),
        ("pykala2026", run_pykala2026),
        ("jtc_hypotheses", run_jtc_hypotheses),
    ]

    for name, fn in runners:
        try:
            result = fn()
            if result is not None:
                all_results[name] = result
                _save(name, result if isinstance(result, dict) else {"data": result})
        except Exception:
            print(f"  ERROR in {name}:")
            traceback.print_exc()

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    for name, _ in runners:
        status = "✓ complete" if name in all_results else "✗ skipped/error"
        print(f"  {name:>20s}:  {status}")

    print(f"\nDone. Results in {OUTPUT_DIR}, plots in {PLOT_DIR}\n")


if __name__ == "__main__":
    main()
