# Sequential survival calibration under censoring

An initial simulation study for my PhD on monitoring the reliability of survival predictions.

The question is simple: **if a survival model is wrong, does censoring make that harder to see? And can we monitor calibration continuously without generating unjustified alarms?**

This repository uses synthetic data because the true event times, forecast errors and censoring mechanism are known. We can therefore compare what a test says with what is actually true.

## Run it

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python run.py --runs 1500 --patients 600
```

Outputs appear in `results/`: two CSV tables, run parameters, and five figures. The seed is fixed by default. For a quick check, use `--runs 100 --patients 200` (the estimates will be noisy).

## What is simulated?

- Real event time: `T ~ Exponential(0.10)`.
- Forecast survival: `S_hat(t) = exp(-forecast_rate * t)` with rates `0.06, 0.10, 0.14, 0.20`.
- Independent censoring: `C ~ Exponential(c)` with `c = 0.10 * p / (1-p)` giving a target censoring fraction `p`.
- Observations: `Y = min(T, C)` and `event = (T <= C)`.
- Censoring targets: `0%, 10%, 30%, 50%, 70%`.

The correct forecast has rate `0.10`. The others are deliberately miscalibrated. Censoring does not change the true event-time distribution or the forecast: it changes what we observe.

## Sequential tests

We work with `U = F_hat(T)`. Under a correct continuous predictive CDF, `U` is uniform. Each fixed Beta density `q(u)` is a valid nonnegative bet with unit null expectation. We take an equally weighted **mixture of component e-processes**, not a product of averaged single-step bets.

Three monitoring procedures are compared:

1. **Oracle:** sees every true `T`, including times hidden from a real analyst.
2. **Naive complete case:** multiplies `q(F_hat(Y))` only for observed events. This can mistake censoring-related selection for miscalibration. It is **not** guaranteed to control false alarms.
3. **Censor-aware:** an event contributes `q(F_hat(Y))`; a censored patient contributes `(1-Q(F_hat(Y))) / (1-F_hat(Y))`, where `Q` is the Beta CDF. This is the conditional expected bet for a uniform PIT truncated above the censoring point.

For fixed `C` independent of `T` (and the forecast), the censor-aware single-step bet has expectation one under the correct conditional event-time distribution:

`integral_0^v q(u) du + (1-v) * [(1-Q(v))/(1-v)] = 1`.

Under independent subjects and fixed bets, the product is a martingale under the null. Each component therefore satisfies Ville's anytime inequality, and so does their fixed-weight mixture. The alarm threshold is `e >= 1/0.05 = 20`. Log-evidence is used in the implementation for numerical stability. These claims apply to the **specified simple setup**, not arbitrary informative censoring or changing populations.

The Beta family here is a small, fixed **implementation inspired by the PIT betting framework** of Arnold, Henzi and Ziegel, **not an exact reproduction of their adaptive procedures**.

## A second perspective: calibration at five years

We also compare three estimators of survival at `t = 5`:

- Oracle: the fraction with `T > 5`.
- Complete-case: the survival fraction among those whose five-year status is known.
- IPCW (known censoring survival): `mean(1{Y > 5}) / G(5)` where `G(5) = P(C > 5)`.

Here the known-`G` IPCW estimator is unbiased under independent censoring, but can be noisy when `G(5)` is small. **This estimator is a fixed-horizon diagnostic, not itself a sequential e-value.**

## Reading the outputs

- `01_false_alarms_and_power.png`: type-I error under the correct forecast and power under an incorrect forecast, across censoring rates.
- `02_power_by_miscalibration.png`: power for different forecast errors at 50% censoring.
- `03_horizon_calibration.png`: five-year survival estimates against the truth and a wrong forecast.
- `04_eprocess_example.png`: the evolution of log-evidence in one simulated sequence.
- `05_detection_delay.png`: median patient index of detection **among runs that alerted**. This is not an unconditional expected delay.

The results are Monte Carlo estimates, not a proof that one method dominates another. Use `mc_se` in the sequential results and `se_mean` in the horizon results to assess simulation noise.

## Important limitations

- **Patients are indexed by completed observations, not clinical calendar time.** Delayed outcomes and asynchronous follow-up require a separate filtration and are future work.
- The censoring mechanism is independent of the event time, and its distribution is known in the simulator. Informative censoring and estimated `G` are outside this first experiment.
- The experiment tests full-distribution PIT calibration. A separate five-year plot does not establish conditional calibration for each patient subgroup.
- A mixture of a few fixed Beta bets is easy to inspect, but can be less powerful than adaptively fitted betting strategies.
- Nothing here is evidence of a new result in the literature. The purpose is to build and validate a baseline before developing new theory.

## References

- Arnold, S., Henzi, A., & Ziegel, J. F. (2023). *Sequentially valid tests for forecast calibration*. Annals of Applied Statistics, 17(3), 1909–1935. https://doi.org/10.1214/22-AOAS1697
- A-calibration: assessment of prediction models for survival data under censoring (2025). https://doi.org/10.1186/s12874-025-02671-6

The code is intentionally short and split by responsibility: `simulation.py` generates data, `betting.py` calculates e-processes, `experiment.py` collects outcomes, and `plots.py` creates the figures.
