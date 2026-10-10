"""Figures used in the project report."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"oracle": "#244E78", "naive": "#D4634D", "censor_aware": "#159889",
          "complete_case": "#D4634D", "ipcw_known_g": "#159889"}
LABELS = {"oracle": "Oracle", "naive": "Complete-case PIT (naive)",
          "censor_aware": "Censor-aware e-process", "complete_case": "Complete case",
          "ipcw_known_g": "IPCW (known G)"}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#243143", "text.color": "#243143",
                         "axes.grid": True, "grid.alpha": 0.16,
                         "savefig.dpi": 180})


def save(folder, name):
    plt.savefig(Path(folder) / name, bbox_inches="tight", facecolor="white")
    plt.close()


def make_plots(monitor, horizon, example_paths, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    setup()

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharey=True)
    for ax, rate, title in zip(axes, (0.10, 0.14), ("Correct forecast (null)", "Miscalibrated forecast")):
        sub = monitor[monitor.forecast_rate == rate]
        for method in ("oracle", "naive", "censor_aware"):
            d = sub[sub.method == method].sort_values("censor_target")
            ax.plot(100 * d.censor_target, 100 * d.probability_alarm, marker="o",
                    linewidth=2.5, color=COLORS[method], label=LABELS[method])
        if rate == 0.10:
            ax.axhline(5, color="#777777", linestyle="--", linewidth=1.4, label="5% nominal level")
        ax.set(title=title, xlabel="Censoring rate (%)", ylim=(-2, 103))
        ax.set_xticks([0, 10, 30, 50, 70])
    axes[0].set_ylabel("Probability of at least one alarm (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), frameon=False)
    fig.suptitle("What censoring does to sequential calibration monitoring", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save(folder, "01_false_alarms_and_power.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    d = monitor[(monitor.censor_target == 0.5) & (monitor.forecast_rate != 0.10)]
    xs = sorted(d.forecast_rate.unique())
    for method in ("oracle", "naive", "censor_aware"):
        ys = [100 * float(d[(d.method == method) & (d.forecast_rate == r)].probability_alarm.iloc[0]) for r in xs]
        ax.plot([f"{x:.2f}" for x in xs], ys, marker="o", linewidth=2.6,
                color=COLORS[method], label=LABELS[method])
    ax.set(xlabel="Forecast event hazard (true hazard = 0.10)",
           ylabel="Probability of alarm (%)", ylim=(0, 103))
    ax.legend(frameon=False)
    ax.set_title("Power at 50% censoring", fontweight="bold", fontsize=15)
    fig.tight_layout()
    save(folder, "02_power_by_miscalibration.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for method in ("oracle", "complete_case", "ipcw_known_g"):
        d = horizon[horizon.method == method].sort_values("censor_target")
        ax.errorbar(100 * d.censor_target, d.estimate_mean, yerr=1.96 * d.se_mean,
                    label=LABELS[method], marker="o", linewidth=2.3,
                    capsize=3, color=COLORS[method])
    ax.axhline(horizon.true_survival.iloc[0], linestyle="--", color="#272727",
               label="True survival at t = 5")
    ax.axhline(horizon.forecast_survival.iloc[0], linestyle=":", color="#8053A7",
               label="Miscalibrated forecast (hazard = 0.14)")
    ax.set(xlabel="Censoring rate (%)", ylabel="Estimated 5-year survival", ylim=(0, 1))
    ax.legend(frameon=False, ncol=2)
    ax.set_title("Censoring changes what calibration appears to be", fontweight="bold", fontsize=15)
    fig.tight_layout()
    save(folder, "03_horizon_calibration.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for method in ("oracle", "naive", "censor_aware"):
        path = example_paths[(0.5, 0.14)][method][0]
        ax.plot(np.arange(1, len(path) + 1), path / np.log(10),
                color=COLORS[method], linewidth=2, label=LABELS[method])
    ax.axhline(np.log10(20), linestyle="--", color="#333333", label="Alarm threshold: log10(20)")
    ax.set(xlabel="Patient index (completed observation)", ylabel="log10(e-process)")
    ax.legend(frameon=False)
    ax.set_title("One simulated sequence: 50% censoring, wrong forecast", fontweight="bold", fontsize=15)
    fig.tight_layout()
    save(folder, "04_eprocess_example.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for rate, label, color in [(0.10, "Correct forecast", "#159889"),
                                (0.14, "Wrong forecast", "#D4634D")]:
        d = monitor[(monitor.forecast_rate == rate) & (monitor.method == "censor_aware")]
        d = d.sort_values("censor_target")
        ax.plot(100 * d.censor_target, d.median_alarm_patient,
                marker="o", linewidth=2.5, color=color, label=label)
    ax.set(xlabel="Censoring rate (%)", ylabel="Median detection patient, conditional on alarm")
    ax.legend(frameon=False)
    ax.set_title("How long detection takes when an alarm occurs", fontweight="bold", fontsize=15)
    fig.tight_layout()
    save(folder, "05_detection_delay.png")
