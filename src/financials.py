"""
financials.py
-------------
Combines the cost schedule from cashflow.py with mechanism revenue from
scenarios.py into unlevered, real-terms project cash flows, and computes
NPV and IRR. Because mechanism revenue is a [iterations, n_years] array (one
row per simulated DK1 price path from price_model.py), NPV and IRR come out
as DISTRIBUTIONS across iterations — reflecting the uncertainty in *when* the
CfD cap gets exhausted, not a single point estimate.

Scope note: this computes UNLEVERED (all-equity) project cash flows,
discounted at finance.discount_rate_real. Debt sizing, interest, and DSCR
covenants (finance.debt in assumptions.yaml) are configured but not yet
modeled here — gearing, interest_rate_nominal, tenor_years, and
target_min_dscr remain [PLACEHOLDER] in the assumptions file, and a proper
sculpted/annuity debt schedule with DSCR sizing is a distinct, non-trivial
piece of work flagged as a follow-up rather than approximated silently.
Decommissioning cost is treated as non-tax-deductible in this version (a
documented simplification, not a researched Danish tax position).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import newton

from assumptions import get_path, operating_years
from cashflow import (
    construction_years,
    decommissioning_years,
    build_cost_schedule,
)


def _tax_with_loss_carryforward(ebt_by_year: np.ndarray, tax_rate: float) -> np.ndarray:
    """Vectorized across iterations: ebt_by_year is [iterations, n_years].
    Losses (negative EBT) carry forward and offset future positive EBT before
    any tax is due — per finance.depreciation.loss_carryforward: true. No
    refund is ever paid out for losses (Danish rule as stated), only relief
    against future profits."""
    iterations, n_years = ebt_by_year.shape
    tax = np.zeros((iterations, n_years))
    carryforward = np.zeros(iterations)

    for t in range(n_years):
        ebt_t = ebt_by_year[:, t]
        usable_loss = np.minimum(carryforward, np.maximum(ebt_t, 0.0))
        taxable_income = np.maximum(ebt_t - usable_loss, 0.0)
        tax[:, t] = taxable_income * tax_rate

        carryforward = carryforward - usable_loss
        new_loss = np.maximum(-ebt_t, 0.0)
        carryforward = carryforward + new_loss

    return tax


def build_unlevered_fcf(
    assumptions: dict,
    cost_schedule: dict,
    revenue_eur: np.ndarray,   # [iterations, n_operating_years]
    op_years: list[int],
) -> dict:
    """Returns:
      all_years -> list[int], construction + operating + decommissioning years, sorted
      fcf       -> [iterations, len(all_years)] unlevered post-tax free cash flow
    """
    cons_years = construction_years(assumptions)
    decomm_years = decommissioning_years(assumptions)
    all_years = cons_years + op_years + decomm_years
    tax_rate = get_path(assumptions, "finance.corporate_tax_rate")

    iterations = revenue_eur.shape[0]
    fcf = np.zeros((iterations, len(all_years)))

    # --- Construction years: CAPEX outflow only, no revenue, no tax ---
    for i, year in enumerate(cons_years):
        fcf[:, i] = -cost_schedule["capex"][year]

    # --- Operating years: revenue - opex - tax (with depreciation shield) ---
    opex_arr = np.array([cost_schedule["opex"][y] for y in op_years])
    depreciation_arr = np.array([cost_schedule["depreciation"].get(y, 0.0) for y in op_years])

    ebt = revenue_eur - opex_arr[None, :] - depreciation_arr[None, :]
    tax = _tax_with_loss_carryforward(ebt, tax_rate)

    # Add back depreciation (non-cash): FCF = EBT_after_tax + depreciation
    operating_fcf = (ebt - tax) + depreciation_arr[None, :]

    op_offset = len(cons_years)
    fcf[:, op_offset:op_offset + len(op_years)] = operating_fcf

    # --- Decommissioning years: outflow only, no tax shield (documented simplification) ---
    decomm_offset = op_offset + len(op_years)
    for i, year in enumerate(decomm_years):
        fcf[:, decomm_offset + i] = -cost_schedule["decommissioning"][year]

    return {"all_years": all_years, "fcf": fcf}


def compute_npv(fcf: np.ndarray, all_years: list[int], discount_rate: float) -> np.ndarray:
    """[iterations] NPV, discounted to t=0 = all_years[0] (the FID year)."""
    t0 = all_years[0]
    offsets = np.array([y - t0 for y in all_years])
    discount_factors = 1.0 / (1.0 + discount_rate) ** offsets
    return fcf @ discount_factors  # [iterations]


def _npv_at_rate(cashflows: np.ndarray, offsets: np.ndarray, rate: float) -> float:
    if rate <= -1.0:
        return float("inf") if np.sum(cashflows) < 0 else float("-inf")
    with np.errstate(over="ignore"):
        value = np.sum(cashflows / (1.0 + rate) ** offsets)
    return float(value)


def compute_irr(fcf: np.ndarray, all_years: list[int], initial_guess: float = 0.10) -> np.ndarray:
    """[iterations] IRR via Newton's method, starting from `initial_guess`.

    This cash flow pattern (one large early outflow, decades of positive
    inflow, a small late outflow for decommissioning) has exactly two sign
    changes, so by Descartes' rule of signs it can have up to two real
    positive roots — confirmed empirically here, not just a theoretical
    edge case. A wide bisection scan (e.g. from -50%) does not reliably find
    "the" economically relevant root: it can latch onto a second, spurious
    root at a deeply negative rate, or worse, one so numerically extreme
    (weights blow up as (1+r) shrinks toward zero over a ~35-year horizon)
    that it silently reports a nonsense value. Newton's method from a guess
    near the discount rate converges to the NEAREST root, matching how
    standard financial software (e.g. Excel's IRR, default guess 10%)
    behaves, and lands on the conventional root: the rate at which NPV
    transitions from positive (project beats the hurdle) to negative (it
    doesn't) around the actual discount rate, not a distant artifact.
    Returns np.nan where Newton's method fails to converge within 100
    iterations or leaves the sensible range."""
    t0 = all_years[0]
    offsets = np.array([y - t0 for y in all_years])
    iterations = fcf.shape[0]
    irr = np.full(iterations, np.nan)

    for i in range(iterations):
        cashflows = fcf[i, :]
        try:
            root = newton(
                lambda r: _npv_at_rate(cashflows, offsets, r),
                x0=initial_guess,
                maxiter=100,
                tol=1e-8,
            )
            if -0.99 < root < 5.0:  # sanity bound — outside this, treat as non-convergent
                irr[i] = root
        except (RuntimeError, OverflowError):
            continue
    return irr


def evaluate_mechanism(
    assumptions: dict,
    cost_schedule: dict,
    revenue_eur: np.ndarray,
    op_years: list[int],
) -> dict:
    """Full pipeline for one mechanism: FCF -> NPV distribution -> IRR distribution."""
    discount_rate = get_path(assumptions, "finance.discount_rate_real")

    built = build_unlevered_fcf(assumptions, cost_schedule, revenue_eur, op_years)
    npv = compute_npv(built["fcf"], built["all_years"], discount_rate)
    irr = compute_irr(built["fcf"], built["all_years"])

    return {
        "all_years": built["all_years"],
        "fcf": built["fcf"],
        "npv_eur": npv,
        "irr": irr,
    }


def summarize_distribution(values: np.ndarray, label: str) -> dict:
    valid = values[~np.isnan(values)]
    return {
        "label": label,
        "n_valid": int(valid.size),
        "n_total": int(values.size),
        "mean": float(np.mean(valid)) if valid.size else float("nan"),
        "p5": float(np.percentile(valid, 5)) if valid.size else float("nan"),
        "p25": float(np.percentile(valid, 25)) if valid.size else float("nan"),
        "p50": float(np.percentile(valid, 50)) if valid.size else float("nan"),
        "p75": float(np.percentile(valid, 75)) if valid.size else float("nan"),
        "p95": float(np.percentile(valid, 95)) if valid.size else float("nan"),
    }


if __name__ == "__main__":
    # Quick manual check: python src/financials.py
    from assumptions import load, validate
    from price_model import calibrate, simulate_price_paths
    from scenarios import fixed_ppa_revenue, two_sided_cfd_revenue, multi_criteria_revenue

    a = load("assumptions.yaml")
    validate(a)

    cost_schedule = build_cost_schedule(a)
    op_years = operating_years(a)

    params = calibrate(a)
    sim = simulate_price_paths(a, params)
    captured = sim["captured_paths"]

    energy = cost_schedule["energy_mwh"]

    print("=" * 70)
    print("PROJECT-LEVEL (UNLEVERED) NPV / IRR BY MECHANISM")
    print(f"Discount rate: {get_path(a, 'finance.discount_rate_real'):.1%}  |  "
          f"{captured.shape[0]} Monte Carlo iterations")
    print("=" * 70)

    mechanisms = {
        "Fixed PPA": fixed_ppa_revenue(a, energy, captured, op_years),
        "Two-sided CfD": two_sided_cfd_revenue(a, energy, captured, op_years)["revenue_eur"],
        "Multi-criteria auction": multi_criteria_revenue(a, energy, captured, op_years)["revenue_eur"],
    }

    for name, revenue in mechanisms.items():
        result = evaluate_mechanism(a, cost_schedule, revenue, op_years)
        npv_stats = summarize_distribution(result["npv_eur"], f"{name} NPV")
        irr_stats = summarize_distribution(result["irr"], f"{name} IRR")

        print(f"\n{name}:")
        print(f"  NPV (EUR m): p5={npv_stats['p5']/1e6:,.0f}  p50={npv_stats['p50']/1e6:,.0f}  "
              f"mean={npv_stats['mean']/1e6:,.0f}  p95={npv_stats['p95']/1e6:,.0f}")
        print(f"  IRR:         p5={irr_stats['p5']:.1%}  p50={irr_stats['p50']:.1%}  "
              f"mean={irr_stats['mean']:.1%}  p95={irr_stats['p95']:.1%}  "
              f"({irr_stats['n_valid']}/{irr_stats['n_total']} iterations had a solvable IRR)")
