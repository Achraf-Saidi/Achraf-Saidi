"""Run the complete experiment: python run.py --runs 1500 --patients 600."""

import argparse
import json
from pathlib import Path

from src.experiment import monte_carlo
from src.plots import make_plots


def main():
    parser = argparse.ArgumentParser(description="Censoring and sequential survival calibration")
    parser.add_argument("--runs", type=int, default=1500, help="Monte Carlo runs per scenario")
    parser.add_argument("--patients", type=int, default=600, help="Patients per simulation")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.runs < 2 or args.patients < 10:
        parser.error("Use at least two runs and ten patients.")

    output = Path("results")
    output.mkdir(exist_ok=True)
    monitor, horizon, paths = monte_carlo(args.runs, args.patients, args.seed)
    monitor.to_csv(output / "sequential_results.csv", index=False)
    horizon.to_csv(output / "horizon_results.csv", index=False)
    make_plots(monitor, horizon, paths, output / "figures")
    metadata = {"seed": args.seed, "runs_per_scenario": args.runs,
                "patients_per_run": args.patients, "alpha": 0.05,
                "hypothesis": "correct full predictive CDF; independent censoring",
                "monitoring_note": "patient index assumes completed observations, not real calendar time"}
    (output / "run_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\nSequential monitoring — alarm probability at 50% censoring")
    print(monitor[monitor.censor_target.eq(0.5)].pivot(index="forecast_rate", columns="method", values="probability_alarm").round(3).to_string())
    print("\nFive-year survival estimation")
    print(horizon.pivot(index="censor_target", columns="method", values="estimate_mean").round(3).to_string())
    print("\nSaved results in", output.resolve())


if __name__ == "__main__":
    main()
