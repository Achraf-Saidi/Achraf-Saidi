"""Monte Carlo summaries for sequential monitoring and horizon calibration."""

import numpy as np
import pandas as pd

from .betting import evidence_paths, first_crossing
from .simulation import HORIZON, TRUE_RATE, pit, sample_patients, survival

CENSOR_FRACTIONS = (0.0, 0.1, 0.3, 0.5, 0.7)
FORECAST_RATES = (0.06, 0.10, 0.14, 0.20)
METHODS = ("oracle", "naive", "censor_aware")


def monte_carlo(repetitions=1500, patients=600, seed=20261010, example_runs=4):
    """Run independent batches; common random numbers help comparisons.

    Alarm = at least one threshold crossing by patient N. A run with no
    alarm has missing detection time, not detection time N+1.
    """
    rng = np.random.default_rng(seed)
    records, paths_for_plots, horizon_records = [], {}, []
    alpha = 0.05
    s_true = survival(HORIZON, TRUE_RATE)

    for fraction in CENSOR_FRACTIONS:
        t, y, delta = sample_patients(rng, repetitions, patients, fraction)
        actual_fraction = float(np.mean(~delta))
        # At horizon tau, only subjects with events before tau or observed
        # event-free at tau have an ascertained binary outcome.
        confirmed_alive = y > HORIZON
        event_before_tau = delta & (y <= HORIZON)
        resolved = confirmed_alive | event_before_tau
        naive_survival = (confirmed_alive.sum(axis=1) / resolved.sum(axis=1))
        lambda_c = TRUE_RATE * fraction / (1 - fraction)
        g_tau = np.exp(-lambda_c * HORIZON)
        ipcw_survival = confirmed_alive.mean(axis=1) / g_tau
        oracle_survival = (t > HORIZON).mean(axis=1)
        for method, estimates in (("oracle", oracle_survival),
                                  ("complete_case", naive_survival),
                                  ("ipcw_known_g", ipcw_survival)):
            horizon_records.append(dict(
                censor_target=fraction, method=method,
                estimate_mean=float(np.mean(estimates)),
                bias=float(np.mean(estimates) - s_true),
                rmse=float(np.sqrt(np.mean((estimates - s_true) ** 2))),
                se_mean=float(np.std(estimates, ddof=1) / np.sqrt(repetitions)),
                true_survival=float(s_true),
                forecast_survival=float(survival(HORIZON, 0.14)),
            ))

        for forecast in FORECAST_RATES:
            paths, boundary = evidence_paths(t, y, delta, forecast, alpha)
            for method in METHODS:
                alarms = first_crossing(paths[method], boundary)
                hit = np.isfinite(alarms)
                records.append(dict(
                    censor_target=fraction,
                    censor_realized=actual_fraction,
                    forecast_rate=forecast,
                    true_rate=TRUE_RATE,
                    model_correct=forecast == TRUE_RATE,
                    method=method,
                    probability_alarm=float(np.mean(hit)),
                    mc_se=float(np.sqrt(np.mean(hit) * (1 - np.mean(hit)) / repetitions)),
                    median_alarm_patient=float(np.median(alarms[hit])) if np.any(hit) else np.nan,
                    alarm_probability_by_200=float(np.mean(alarms <= 200)),
                    horizon_patients=patients,
                    runs=repetitions,
                    seed=seed,
                ))
            if fraction in (0, 0.5, 0.7) and forecast in (0.10, 0.14):
                # Only a few paths are kept for the demonstration figure.
                paths_for_plots[(fraction, forecast)] = {
                    name: path[:example_runs].copy()
                    for name, path in paths.items()
                }

    return pd.DataFrame(records), pd.DataFrame(horizon_records), paths_for_plots
