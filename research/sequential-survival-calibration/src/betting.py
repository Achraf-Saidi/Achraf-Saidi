"""Fixed-mixture Beta bets with elementary formulas."""

import numpy as np
from scipy.special import logsumexp

# These Beta(a,1) and Beta(1,b) alternatives need no numerical integration.
# In the exponential example, the PIT under a wrong event rate has a
# Beta(1,b) law, so these alternatives are particularly easy to interpret.
BETS = ((0.5, 1), (0.8, 1), (2, 1),
        (1, 0.5), (1, 0.7), (1, 1.4), (1, 2))


def log_beta_bet(u, a, b):
    """Log Beta(a,b) density; these bets always have a=1 or b=1."""
    u = np.clip(u, 1e-12, 1 - 1e-12)
    if b == 1:
        return np.log(a) + (a - 1) * np.log(u)
    return np.log(b) + (b - 1) * np.log1p(-u)


def log_censored_bet(v, a, b):
    """log P_Beta(U>v) / P_Uniform(U>v) for a right-censored PIT."""
    v = np.clip(v, 1e-12, 1 - 1e-12)
    if a == 1:
        return (b - 1) * np.log1p(-v)
    # Beta(a,1) survival = 1-v**a; expm1 avoids cancellation near 1.
    return np.log(-np.expm1(a * np.log(v))) - np.log1p(-v)


def evidence_paths(true_times, observed, event, forecast_rate, level=0.05):
    """Oracle, invalid complete-case, and valid censor-aware mixture paths.

    Validity claim: independent continuous event times, independent censoring,
    correct fixed forecasts, fixed bets, and sequentially completed observations.
    """
    from .simulation import pit

    oracle_u = pit(true_times, forecast_rate)
    seen_u = pit(observed, forecast_rate)
    component_logs = {name: [] for name in ("oracle", "naive", "censor_aware")}
    for a, b in BETS:
        observed_log_bet = log_beta_bet(seen_u, a, b)
        component_logs["oracle"].append(log_beta_bet(oracle_u, a, b))
        component_logs["naive"].append(np.where(event, observed_log_bet, 0.0))
        component_logs["censor_aware"].append(np.where(
            event, observed_log_bet, log_censored_bet(seen_u, a, b)))

    paths = {}
    log_threshold = np.log(1 / level)
    for name, entries in component_logs.items():
        # Crucial: mixture of full component e-processes.
        log_components = np.cumsum(np.stack(entries), axis=2)
        paths[name] = logsumexp(log_components, axis=0) - np.log(len(BETS))
    return paths, log_threshold


def first_crossing(log_path, log_threshold):
    """1-indexed first alarm; NaN when no alarm occurs before horizon."""
    hit = log_path >= log_threshold
    first = np.argmax(hit, axis=1) + 1
    return np.where(np.any(hit, axis=1), first, np.nan)
