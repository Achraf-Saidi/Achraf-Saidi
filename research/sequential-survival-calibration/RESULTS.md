# Results from the first simulation

**Run:** 1,500 Monte Carlo replications per setting, 600 patients per replication, seed 20261010. There are 20 scenarios (4 forecast hazards × 5 censoring rates) and three sequential monitoring procedures. The alert threshold is 20 (nominal alpha = 0.05).

## Type-I error: correct forecast (hazard 0.10)

| Censoring | Oracle | Complete-case naive | Censor-aware |
|---:|---:|---:|---:|
| 0% | 3.8% | 3.8% | 3.8% |
| 10% | 2.9% | 18.6% | 3.1% |
| 30% | 3.1% | 100.0% | 4.0% |
| 50% | 3.6% | 100.0% | 3.5% |
| 70% | 2.9% | 100.0% | 3.3% |

The naive method does not control the false alarm rate. Corrected and oracle procedures are below 5% in every setting tested. At a 3.5% observed rate (1,500 independent replicates), the approximate 95% Monte Carlo interval is 2.6–4.4%. These experiments *check* the theorem's behavior, but are not a proof.

## Power: incorrect forecast (hazard 0.14)

| Censoring | Oracle | Complete-case naive | Censor-aware |
|---:|---:|---:|---:|
| 0% | 100.0% | 100.0% | 100.0% |
| 10% | 100.0% | 90.6% | 100.0% |
| 30% | 100.0% | 4.3% | 99.8% |
| 50% | 100.0% | 99.7% | 99.3% |
| 70% | 100.0% | 100.0% | 95.7% |

The apparent oscillation in the naive detection rate is instructive: censoring selection can partly cancel genuine model error, or exaggerate it. Because naive type-I error is uncontrolled, its high detection rates must not be treated as meaningful power.

Among **corrected runs that alert**, median detection patient indices for the hazard-0.14 forecast are 60, 70, 90, 118 and 185 at 0%, 10%, 30%, 50% and 70% censoring respectively. The 70%-censored corrected procedure still alerts in 95.7% of runs by patient 600; runs without alarms are excluded from this conditional median.

## Five-year survival (t = 5)

The true survival probability is `exp(-0.5) = 0.60653`. At 70% target censoring:

- Oracle average: 0.606.
- Complete-case average: 0.437 (strong downward selection bias).
- IPCW with the exact known censoring survival `G(5)`: 0.608 (close to truth).

This is a separate fixed-horizon diagnostic, not an anytime-valid IPCW test. The IPCW weights grow and estimator variance increases as `G(5)` decreases.

## What this does and does not establish

The first baseline is working. It illustrates a real statistical problem and a valid solution **in the simplified iid, independent-censoring setting**, using a fixed family of bets. It does not establish methodological novelty, nor does it solve delayed clinical follow-up, censoring dependent on patient characteristics, estimation of `G`, dataset shift, or sequential nuisance adaptation.

**Next experiment:** allow `C` to depend on baseline features `X`, retain independent censoring only conditionally on `X`, and test what happens when `G(t|X)` must be estimated and updated over time. Define carefully the filtration for arriving and delayed outcomes before making clinical deployment claims.