"""
sensitivity.py
---------------
One-at-a-time sensitivity analysis ("tornado chart" data) using the ranges
already defined in assumptions.yaml's `sensitivity.ranges` block. For each
range, the perturbed assumption is set to its low/high value, the FULL
pipeline (cashflow -> price simulation -> mechanism revenue -> NPV) is
re-run, and the resulting swing in mean NPV is recorded.

One range in the scaffold's assumptions.yaml — "Market price"
(market.base_power_price_eur_per_mwh) — is a leftover from the deterministic
base-case design and has no effect under the stochastic OU pricing approach
actually implemented (price_model.py ignores it entirely; see its module
docstring). Rather than silently computing a no-op sensitivity for it, this
module skips it and says so explicitly. "Capture rate" (market.capture_rate)
IS active — it multiplies the simulated baseload price in price_model.py.

Mechanism-specific ranges (currently just "CfD cap") only apply when running
sensitivity on the two_sided_cfd mechanism; they're skipped for the other two
with the same explicit-skip approach, rather than computing a range that
can't possibly move that mechanism's NPV.
"""

from __future__ import annotations

import numpy as np

from assumptions import get_path, set_path, operating_years
from cashflow import build_cost_schedule
from price_model import calibrate, simulate_price_paths
from scenarios import fixed_ppa_revenue, two_sided_cfd_revenue, multi_criteria_revenue
from financials import evaluate_mechanism

_INACTIVE_UNDER_STOCHASTIC_PRICING = {"market.base_power_price_eur_per_mwh"}

_MECHANISM_ONLY_PATHS = {
    "contracts.two_sided_cfd.cumulative_payment_cap_eur": {"two_sided_cfd"},
}

_REVENUE_FN = {
    "fixed_ppa": fixed_ppa_revenue,
    "two_sided_cfd": lambda a, e, c, y: two_sided_cfd_revenue(a, e, c, y)["revenue_eur"],
    "multi_criteria": lambda a, e, c, y: multi_criteria_revenue(a, e, c, y)["revenue_eur"],
}


def _mean_npv(assumptions: dict, mechanism: str) -> float:
    cost_schedule = build_cost_schedule(assumptions)
    op_years = operating_years(assumptions)
    params = calibrate(assumptions)
    sim = simulate_price_paths(assumptions, params)
    captured = sim["captured_paths"]
    energy = cost_schedule["energy_mwh"]

    revenue = _REVENUE_FN[mechanism](assumptions, energy, captured, op_years)
    result = evaluate_mechanism(assumptions, cost_schedule, revenue, op_years)
    return float(np.mean(result["npv_eur"]))


def run_tornado(assumptions: dict, mechanism: str = "two_sided_cfd") -> list[dict]:
    """Returns a list of dicts, one per active range, sorted by |swing|
    descending (the standard tornado-chart ordering: biggest driver on top):
      name, low_npv, base_npv, high_npv, swing (= high_npv - low_npv)
    """
    if mechanism not in _REVENUE_FN:
        raise ValueError(f"Unknown mechanism '{mechanism}', expected one of {list(_REVENUE_FN)}")

    base_npv = _mean_npv(assumptions, mechanism)
    ranges = get_path(assumptions, "sensitivity.ranges")

    results = []
    for r in ranges:
        path = r["path"]

        if path in _INACTIVE_UNDER_STOCHASTIC_PRICING:
            print(f"[sensitivity] skipping '{r['name']}' ({path}) — inactive under "
                  "stochastic OU pricing, see module docstring.")
            continue
        if path in _MECHANISM_ONLY_PATHS and mechanism not in _MECHANISM_ONLY_PATHS[path]:
            print(f"[sensitivity] skipping '{r['name']}' ({path}) — only affects "
                  f"{_MECHANISM_ONLY_PATHS[path]}, not '{mechanism}'.")
            continue

        base_value = get_path(assumptions, path)
        if r["mode"] == "relative":
            low_value = base_value * r["low"]
            high_value = base_value * r["high"]
        elif r["mode"] == "absolute":
            low_value = r["low"]
            high_value = r["high"]
        else:
            raise ValueError(f"Unknown sensitivity mode '{r['mode']}' for range '{r['name']}'")

        low_assumptions = set_path(assumptions, path, low_value)
        high_assumptions = set_path(assumptions, path, high_value)

        low_npv = _mean_npv(low_assumptions, mechanism)
        high_npv = _mean_npv(high_assumptions, mechanism)

        results.append({
            "name": r["name"],
            "path": path,
            "low_value": low_value,
            "high_value": high_value,
            "low_npv": low_npv,
            "base_npv": base_npv,
            "high_npv": high_npv,
            "swing": high_npv - low_npv,
        })

    results.sort(key=lambda x: abs(x["swing"]), reverse=True)
    return results


def plot_tornado(results: list[dict], mechanism: str, output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    base = results[0]["base_npv"] if results else 0.0
    lows = [r["low_npv"] - base for r in results]
    highs = [r["high_npv"] - base for r in results]

    fig, ax = plt.subplots(figsize=(9, 0.5 * len(results) + 1.5))
    y_pos = np.arange(len(names))

    for y, low, high in zip(y_pos, lows, highs):
        left = min(low, high)
        width = abs(high - low)
        ax.barh(y, width, left=left, color="#4C72B0", alpha=0.85)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # biggest driver at top
    ax.set_xlabel("Mean NPV vs. base case (EUR)")
    ax.set_title(f"Sensitivity (tornado) — {mechanism}, base NPV = EUR {base/1e6:,.0f}m")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Quick manual check: python src/sensitivity.py
    from assumptions import load, validate

    a = load("assumptions.yaml")
    validate(a)

    print("Running tornado sensitivity on two_sided_cfd (this re-runs the full "
          "pipeline ~2x per active range, may take a few seconds)...\n")
    results = run_tornado(a, mechanism="two_sided_cfd")

    print(f"\n{'Range':<20} {'Low NPV (EURm)':>16} {'Base NPV (EURm)':>17} "
          f"{'High NPV (EURm)':>17} {'Swing (EURm)':>14}")
    for r in results:
        print(f"{r['name']:<20} {r['low_npv']/1e6:>16,.0f} {r['base_npv']/1e6:>17,.0f} "
              f"{r['high_npv']/1e6:>17,.0f} {r['swing']/1e6:>14,.0f}")

    import os
    os.makedirs("outputs", exist_ok=True)
    plot_tornado(results, "two_sided_cfd", "outputs/tornado_two_sided_cfd.png")
    print("\nSaved outputs/tornado_two_sided_cfd.png")
