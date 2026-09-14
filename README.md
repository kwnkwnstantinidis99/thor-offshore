# Thor Offshore Wind Farm — A Comparative Feasibility Analysis

An independent case study comparing three revenue mechanisms — a fixed-price
PPA, a two-sided CfD with a cumulative payment cap, and a multi-criteria
auction — applied to the real **Thor offshore wind farm** (Denmark, ~1 GW,
under construction, COD 2027).

This is not a hypothetical: Thor's actual confirmed contract terms are used
throughout (strike price, payment cap, concession length), sourced publicly
and documented with a confidence tag on every assumption (see
`assumptions.yaml`). It is an independent analysis built on public
information — not RWE's actual financial model, not investment advice, and
not affiliated with RWE or the Danish Energy Agency.

## Why this case study is interesting

Thor's two-sided CfD strike price was set at essentially zero
(~EUR 0.01/MWh) through the tender's lottery mechanism. That single fact
inverts the usual intuition about CfDs: instead of protecting the producer
from low prices, it means almost **all** market revenue flows to the state
as a clawback — until a cumulative payment cap (~EUR 375m) is reached, at
which point the contract terminates entirely and the producer becomes a pure
merchant generator for the remaining ~25+ years of the 30-year concession.
*When* that cap is reached depends entirely on the price path realized, which
is why this analysis is built around Monte Carlo simulation rather than a
single deterministic price forecast.

## Methodology

- **Price uncertainty**: an arithmetic Ornstein-Uhlenbeck (mean-reverting)
  process for DK1 electricity prices, calibrated on 2015-2025 historical
  data (`data/dk1_annual_prices.csv`) and simulated forward across the full
  30-year operating horizon. See `src/price_model.py` for why arithmetic
  (not log) OU is required, and why a half-life prior is preferred over a
  raw statistical fit given the short, crisis-distorted calibration window.
- **CfD settlement**: net cap accounting with pro-rated crossing-year
  settlement — see `src/scenarios.py` for the full mechanics and a worked
  numerical example in the module docstring.
- **Financials**: unlevered (all-equity), real-terms project cash flows;
  Danish 15%/year declining-balance tax depreciation; NPV and IRR are
  reported as *distributions* across Monte Carlo iterations, not point
  estimates, since the CfD's cap-exhaustion timing is path-dependent.
- **Sensitivity**: one-at-a-time tornado analysis on the ranges defined in
  `assumptions.yaml`.

## Project structure

```
assumptions.yaml          Single source of truth for every input — every
                           value is tagged [CONFIRMED] / [ESTIMATED] /
                           [PLACEHOLDER] / [UNRESOLVED]. No numbers are
                           hardcoded anywhere in /src.
data/
  dk1_annual_prices.csv    Historical DK1 annual prices, 2015-2025, with a
                           confidence flag per year (DK1-specific vs. a
                           Nordic/Baltic regional proxy for 2015-2018).
src/
  assumptions.py           Loads + validates assumptions.yaml; blocks the
                           pipeline on any unresolved value.
  cashflow.py               Energy production, OPEX, CAPEX schedule,
                           depreciation, decommissioning — shared across
                           all three mechanisms.
  price_model.py            Stochastic DK1 price simulation (Ornstein-
                           Uhlenbeck), calibrated on history, projected
                           forward.
  scenarios.py               Mechanism-specific revenue: fixed PPA,
                           two-sided CfD (with cap settlement), multi-
                           criteria auction.
  financials.py             Unlevered NPV / IRR, including Danish tax
                           depreciation and loss carryforward.
  sensitivity.py            Tornado sensitivity analysis.
  run_analysis.py           Orchestrates the full pipeline end to end.
outputs/                   Generated charts and CSVs (created on run).
```

## Running it

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

This runs the full pipeline (1,000 Monte Carlo price paths by default,
configurable in `assumptions.yaml` under `price_model.simulation`) and
writes to `outputs/`:

- `mechanism_comparison_summary.csv` — NPV/IRR summary statistics per mechanism
- `npv_distribution_<mechanism>.csv` — full per-iteration NPV/IRR for each mechanism
- `npv_distributions.png` — NPV distribution comparison chart
- `price_fan_chart.png` — simulated DK1 price paths, p5-p95 fan chart
- `tornado_two_sided_cfd.png` — sensitivity tornado chart

Each module also runs standalone with `python src/<module>.py` for a quick
sanity check of that piece in isolation (e.g. `python src/price_model.py`
prints the calibrated OU parameters and a few simulated price statistics).

## Known limitations / documented simplifications

- **Unlevered analysis only.** Debt sizing, interest, amortisation, and DSCR
  covenants are configured in `assumptions.yaml` (`finance.debt`) but not
  yet modeled — a proper sculpted debt schedule sized to a DSCR covenant is
  a distinct piece of work, not approximated here.
- **Decommissioning is treated as non-tax-deductible**, a simplification
  rather than a researched position under Danish tax law.
- **Gross vs. net capacity factor** for the published 52.5% figure could not
  be confirmed against Thor's tender documents; it is treated as net, as a
  stated analyst assumption (see `assumptions.yaml`, `technical.capacity_factor_is_net`).
- **2015-2018 DK1 prices** are a Nordic/Baltic regional proxy (ACER data),
  not DK1-specific — flagged in `data/dk1_annual_prices.csv`.
- **2019** has no reliable DK1 price figure and is left blank rather than
  interpolated.

## Author

Konstantinos Konstantinidis — independent portfolio case study.
