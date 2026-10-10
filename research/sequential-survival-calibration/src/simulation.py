"""Synthetic event times, independent right-censoring, and a simple forecast."""

import numpy as np

TRUE_RATE = 0.10
HORIZON = 5.0


def censoring_rate(target_fraction, event_rate=TRUE_RATE):
    """For independent exponential T and C, P(C < T) = c / (c + event_rate)."""
    if not 0 <= target_fraction < 1:
        raise ValueError("Censoring fraction must be in [0, 1).")
    return event_rate * target_fraction / (1 - target_fraction)


def sample_patients(rng, repetitions, patients, censor_fraction):
    """Rows are independent Monte Carlo runs; columns are successive patients.

    The simulated true event times are retained for the oracle, but never
    passed to the censoring-aware analysis for censored patients.
    """
    t = rng.exponential(1 / TRUE_RATE, (repetitions, patients))
    rate_c = censoring_rate(censor_fraction)
    c = (rng.exponential(1 / rate_c, (repetitions, patients))
         if rate_c > 0 else np.full_like(t, np.inf))
    observed = np.minimum(t, c)
    event = t <= c
    return t, observed, event


def survival(time, rate):
    return np.exp(-rate * time)


def pit(time, forecast_rate):
    """Probability integral transform under the forecast distribution."""
    return -np.expm1(-forecast_rate * time)
