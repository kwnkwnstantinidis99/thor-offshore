"""
run_analysis.py
----------------
Orchestrates the full pipeline end to end:
  assumptions.yaml -> cashflow -> stochastic price simulation -> mechanism
  revenue (all three) -> NPV/IRR distributions -> comparison chart + CSVs
  -> tornado sensitivity for the two-sided CfD.

Run from the project root:
    python src/run_analysis.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from assumptions import load, validate, get_path, operating_years
from cashflow import build_cost_schedule
from price_model import calibrate, simulate_price_paths, get_price_statistics
from scenarios import fixed_ppa_revenue, two_sided_cfd_revenue, multi_criteria_revenue
from financials import evaluate_mechanism, summarize_distribution
from sensitivity import run_tornado, plot_tornado


def run_full_analysis(assumptions_path: str = "assumptions.yaml", allow_unresolved: bool = False) -> dict:
    a = load(assumptions_path)
    validate(a, allow_unresolved=allow_unresolved)

    out_dir = get_path(a, "outputs.directory")
    os.makedirs(out_dir, exist_ok=True)

    cost_schedule = build_cost_schedule(a)
    op_years = operating_years(a)
    energy = cost_schedule["energy_mwh"]

    print("Calibrating stochastic DK1 price model and simulating forward paths...")
    params = calibrate(a)
    sim = simulate_price_paths(a, params)
    captured = sim["captured_paths"]
    price_stats = get_price_statistics(sim)

    print(f"  long-run mean: EUR {params.long_run_mean:.2f}/MWh  |  "
          f"reversion speed: {params.reversion_speed:.4f}/yr "
          f"({'half-life prior' if params.half_life_used else 'fitted'})  |  "
          f"{captured.shape[0]} iterations over {op_years[0]}-{op_years[-1]}")

    mechanisms_revenue = {
        "fixed_ppa": fixed_ppa_revenue(a, energy, captured, op_years),
        "two_sided_cfd": two_sided_cfd_revenue(a, energy, captured, op_years)["revenue_eur"],
        "multi_criteria": multi_criteria_revenue(a, energy, captured, op_years)["revenue_eur"],
    }

    mechanism_results = {}
    summary_rows = []
    for name, revenue in mechanisms_revenue.items():
        result = evaluate_mechanism(a, cost_schedule, revenue, op_years)
        npv_stats = summarize_distribution(result["npv_eur"], f"{name}_npv")
        irr_stats = summarize_distribution(result["irr"], f"{name}_irr")
        mechanism_results[name] = result

        summary_rows.append({
            "mechanism": name,
            "npv_mean_eur": npv_stats["mean"],
            "npv_p5_eur": npv_stats["p5"],
            "npv_p50_eur": npv_stats["p50"],
            "npv_p95_eur": npv_stats["p95"],
            "irr_mean": irr_stats["mean"],
            "irr_p5": irr_stats["p5"],
            "irr_p50": irr_stats["p50"],
            "irr_p95": irr_stats["p95"],
            "irr_solvable_fraction": irr_stats["n_valid"] / irr_stats["n_total"],
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 100)
    print("MECHANISM COMPARISON — unlevered project NPV / IRR "
          f"({captured.shape[0]} Monte Carlo iterations, "
          f"discount rate {get_path(a, 'finance.discount_rate_real'):.1%})")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}"):
        print(summary_df.to_string(index=False))

    if get_path(a, "outputs.save_cashflow_csv"):
        summary_path = os.path.join(out_dir, "mechanism_comparison_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved {summary_path}")

        for name, result in mechanism_results.items():
            npv_path = os.path.join(out_dir, f"npv_distribution_{name}.csv")
            pd.DataFrame({"npv_eur": result["npv_eur"], "irr": result["irr"]}).to_csv(npv_path, index=False)

    _plot_npv_distributions(mechanism_results, out_dir, a)
    _plot_price_fan(price_stats, out_dir)

    print("\nRunning tornado sensitivity on two_sided_cfd...")
    tornado_results = run_tornado(a, mechanism="two_sided_cfd")
    plot_tornado(tornado_results, "two_sided_cfd", os.path.join(out_dir, "tornado_two_sided_cfd.png"))
    print(f"Saved {os.path.join(out_dir, 'tornado_two_sided_cfd.png')}")

    return {
        "assumptions": a,
        "cost_schedule": cost_schedule,
        "price_params": params,
        "price_simulation": sim,
        "mechanism_results": mechanism_results,
        "summary_df": summary_df,
        "tornado_results": tornado_results,
    }


def _plot_npv_distributions(mechanism_results: dict, out_dir: str, assumptions: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"fixed_ppa": "#4C72B0", "two_sided_cfd": "#DD8452", "multi_criteria": "#55A868"}

    for name, result in mechanism_results.items():
        npv_m = result["npv_eur"] / 1e6
        color = colors.get(name)
        if np.std(npv_m) < 1e-6:
            # Deterministic mechanism (e.g. multi_criteria with a fixed contract
            # price for the full term): a histogram of a single repeated value
            # is a near-infinite spike that swamps the y-axis and makes every
            # other series invisible. Draw it as a labelled vertical line instead
            # — it genuinely has no distribution to show.
            ax.axvline(npv_m[0], color=color, linewidth=2, linestyle="--",
                       label=f"{name} (deterministic, EUR {npv_m[0]:,.0f}m)")
        else:
            ax.hist(npv_m, bins=40, alpha=0.5, label=name, density=True, color=color)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("NPV (EUR million)")
    ax.set_ylabel("Density")
    ax.set_title("NPV distribution by mechanism (Monte Carlo over simulated DK1 price paths)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "npv_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_price_fan(price_stats: dict, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = price_stats["years"]
    s = price_stats["captured_paths"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(years, s["p5"], s["p95"], alpha=0.2, color="#4C72B0", label="p5-p95")
    ax.fill_between(years, s["p25"], s["p75"], alpha=0.35, color="#4C72B0", label="p25-p75")
    ax.plot(years, s["p50"], color="#4C72B0", linewidth=2, label="median")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Captured DK1 price (EUR/MWh)")
    ax.set_title("Simulated DK1 captured price paths — Monte Carlo fan chart")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "price_fan_chart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    run_full_analysis()
