"""Deterministic mathematical checks; no Monte Carlo pass/fail thresholds."""

import numpy as np
from scipy.integrate import quad

from src.betting import BETS, evidence_paths, log_beta_bet, log_censored_bet
from src.simulation import TRUE_RATE, censoring_rate, pit, survival


def test_rate_matches_target_censoring_probability():
    for fraction in (0, 0.1, 0.3, 0.5, 0.7):
        c = censoring_rate(fraction)
        assert np.isclose(c / (c + TRUE_RATE), fraction)


def test_pit_and_survival():
    assert np.isclose(pit(5, 0.1), 1 - survival(5, 0.1))
    assert np.isclose(pit(0, 0.1), 0)


def test_censored_bet_null_expectation_for_fixed_censor_time():
    # Given C=c, event PIT u is uniform on [0,1]. The exact conditional
    # expectation is integral_0^v q(u)du + (1-v)*E_censored(v) = 1.
    for a, b in BETS:
        for v in (0.05, 0.3, 0.65, 0.9):
            area = quad(lambda u: np.exp(log_beta_bet(u, a, b)), 0, v)[0]
            total = area + (1-v)*np.exp(log_censored_bet(v, a, b))
            assert np.isclose(total, 1, atol=1e-8)


def test_no_censor_all_methods_identical():
    t = np.array([[0.2, 1, 2, 3], [2, 4, 5, 6]])
    paths, _ = evidence_paths(t, t, np.ones_like(t, dtype=bool), 0.1)
    assert np.allclose(paths["oracle"], paths["naive"])
    assert np.allclose(paths["oracle"], paths["censor_aware"])


def test_corrected_censored_bet_can_use_only_observed_data():
    t1 = np.array([[7., 8., 2.]])
    t2 = np.array([[70., 90., 2.]])
    y = np.array([[1., 1., 2.]])
    events = np.array([[False, False, True]])
    p1, _ = evidence_paths(t1, y, events, 0.1)
    p2, _ = evidence_paths(t2, y, events, 0.1)
    assert np.array_equal(p1["censor_aware"], p2["censor_aware"])
    assert not np.array_equal(p1["oracle"], p2["oracle"])


def test_evidence_for_one_run_does_not_depend_on_other_runs():
    t = np.array([[1., 2., 3.], [7., 8., 9.]])
    y = np.array([[1., 1., 3.], [2., 3., 9.]])
    event = np.array([[True, False, True], [False, False, True]])
    full, _ = evidence_paths(t, y, event, 0.1)
    alone, _ = evidence_paths(t[:1], y[:1], event[:1], 0.1)
    for method in full:
        assert np.allclose(full[method][0], alone[method][0])
