"""
Experiment: Siggelkow & Levinthal (2003) replication.
Compares centralized, decentralized, and reintegrated organizational forms.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.siggelkow2003 import simulate
from src.viz.plots import plot_performance_trajectories

OUTPUT_DIR = Path(__file__).parent / "results"
PLOT_DIR = Path(__file__).parent / "plots"


def main():
    print("=== Siggelkow & Levinthal (2003) ===")
    print("Running N=6, K=2, 250 runs...")
    results = simulate(N=6, K=2, n_periods=100, t_dec=50, n_runs=250, seed=42)

    print(f"\nFinal performance (period=100):")
    for cond, traj in results.items():
        print(f"  {cond:>15s}: {traj[-1]:.4f}")

    print("\nKey finding: reintegrated should outperform both pure forms")
    cent_final = results["centralized"][-1]
    dec_final = results["decentralized"][-1]
    rei_final = results["reintegrated"][-1]

    if rei_final > cent_final and rei_final > dec_final:
        print("  ✓ CONFIRMED: reintegrated > centralized and decentralized")
    elif rei_final > dec_final:
        print("  ~ PARTIAL: reintegrated > decentralized but not centralized")
    else:
        print("  ✗ NOT confirmed with these parameters")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_serializable = {k: v.tolist() for k, v in results.items()}
    (OUTPUT_DIR / "siggelkow2003.json").write_text(json.dumps(results_serializable, indent=2))
    print(f"\nResults saved to {OUTPUT_DIR}/siggelkow2003.json")

    # Plot
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_performance_trajectories(
        results,
        title="Siggelkow & Levinthal (2003): N=6, K=2",
        output_path=str(PLOT_DIR / "siggelkow2003.png"),
    )
    print(f"Plot saved to {PLOT_DIR}/siggelkow2003.png")

    # High K variant
    print("\nRunning N=6, K=4 (rugged landscape)...")
    results_hi = simulate(N=6, K=4, n_periods=100, t_dec=50, n_runs=250, seed=43)
    print(f"Final performance (K=4):")
    for cond, traj in results_hi.items():
        print(f"  {cond:>15s}: {traj[-1]:.4f}")

    (OUTPUT_DIR / "siggelkow2003_highK.json").write_text(
        json.dumps({k: v.tolist() for k, v in results_hi.items()}, indent=2)
    )
    plot_performance_trajectories(
        results_hi,
        title="Siggelkow & Levinthal (2003): N=6, K=4 (rugged)",
        output_path=str(PLOT_DIR / "siggelkow2003_highK.png"),
    )


if __name__ == "__main__":
    main()
