"""
Experiment: JTC Hypothesis Testing (H1, H2, H3).

H1: JTC adapts slower than flat org in VUCA (high K + shock).
H2: Flat org reaches higher performance peak in VUCA.
H3: JTC+AI shows interesting post-shock resilience due to meeting-driven
    convergence of AI's wide exploration.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.jtc_model import simulate_hypotheses
from src.viz.plots import plot_jtc_summary

OUTPUT_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"

PARAMS = {
    "vuca": dict(N=12, K=8, n_agents=10, n_periods=300, shock_period=150, shock_rho=0.2,
                 n_runs=50, seed=42),
    "stable": dict(N=12, K=2, n_agents=10, n_periods=300, shock_period=150, shock_rho=0.2,
                   n_runs=50, seed=99),
}


def _verdict(condition: bool, label: str) -> str:
    return f"  ✓ CONFIRMED: {label}" if condition else f"  ✗ NOT confirmed: {label}"


def _print_results(results: dict, env_name: str) -> None:
    print(f"\n{'='*60}")
    print(f"Environment: {env_name.upper()}")
    print(f"{'='*60}")
    print(f"Params: {results['params']}")

    print("\nPre-shock peak performance:")
    for cond, v in results["pre_shock_peak"].items():
        print(f"  {cond:>15s}: {v:.4f}")

    print("\nPost-shock recovery (mean over 50 periods after shock):")
    for cond, v in results["post_shock_recovery"].items():
        print(f"  {cond:>15s}: {v:.4f}")

    pre = results["pre_shock_peak"]
    post = results["post_shock_recovery"]

    print(f"\nHypothesis verdicts ({env_name}):")

    # H1: JTC (human) slower than flat (human)
    h1 = pre["flat_human"] > pre["jtc_human"]
    print(_verdict(h1, "H1 — Flat+Human > JTC+Human in pre-shock performance"))

    # H2: Flat outperforms JTC in VUCA (peak)
    flat_best = max(pre["flat_human"], pre["flat_ai"])
    jtc_best = max(pre["jtc_human"], pre["jtc_ai"])
    h2 = flat_best > jtc_best
    print(_verdict(h2, "H2 — Flat best > JTC best (pre-shock peak)"))

    # H3: JTC+AI recovers better post-shock vs Flat+AI
    h3 = post["jtc_ai"] > post["flat_ai"]
    print(_verdict(h3, "H3 — JTC+AI post-shock recovery > Flat+AI (meeting-convergence effect)"))

    # Bonus: does AI help JTC more in post-shock?
    jtc_ai_lift = post["jtc_ai"] - post["jtc_human"]
    flat_ai_lift = post["flat_ai"] - post["flat_human"]
    h3b = jtc_ai_lift > flat_ai_lift
    print(_verdict(h3b, "H3b — AI lifts JTC more than Flat post-shock (RINGI+meetings create synergy)"))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for env_name, params in PARAMS.items():
        print(f"\nSimulating {env_name} environment ({params['n_runs']} runs)...")
        results = simulate_hypotheses(**params)

        _print_results(results, env_name)

        (OUTPUT_DIR / f"jtc_{env_name}.json").write_text(json.dumps(results, indent=2))
        print(f"  Saved: results/jtc_{env_name}.json")

        plot_jtc_summary(
            perf=results["performance"],
            diversity=results["diversity"],
            shock_period=params["shock_period"],
            output_path=str(PLOT_DIR / f"jtc_{env_name}.png"),
        )
        print(f"  Plot: plots/jtc_{env_name}.png")


if __name__ == "__main__":
    main()
