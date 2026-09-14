"""
scenarios.py
------------
Mechanism-specific REVENUE calculations. Takes the energy production series
from cashflow.py and a set of simulated captured-price paths from
price_model.py, and returns producer revenue per year (or per year per Monte
Carlo iteration, for the CfD, where the settlement is path-dependent).

Three mechanisms:
  fixed_ppa_revenue()      - flat contracted price for term_years, merchant after
  two_sided_cfd_revenue()  - the critical one, see docstring below
  multi_criteria_revenue() - scored price adjustment on a flat contract

Two-sided CfD mechanics (why producer revenue collapses to ~strike price)
----------------------------------------------------------------------------
While the CfD is active, the state and producer settle the GAP between the
captured market price and the strike price every year:
    state_flow_t = (captured_price_t - strike_price) * energy_t
      > 0  -> producer PAYS the state this amount (clawback)
      < 0  -> the state PAYS the producer this amount (support)

Producer's actual cash that year, net of this settlement, is ALWAYS exactly
strike_price * energy_t, regardless of where the market price landed — that
symmetry is the entire point of a two-sided CfD. With Thor's confirmed strike
of ~EUR 0.01/MWh, this means producer revenue while the CfD is active is
effectively zero per MWh; essentially all market revenue is handed to the
state as clawback.

state_flow_t accumulates into a running total (cap_accounting: net — a
support year, where state_flow_t < 0, REDUCES the running total and restores
headroom rather than being ignored). Once the running total reaches
cumulative_payment_cap_eur, the contract terminates fully
(on_cap_reached: terminate_to_merchant): the producer keeps 100% of market
revenue from then on, and — because clawback_survives_cap: false — owes
nothing further even if the running total would otherwise have kept rising.
The crossing year itself is pro-rated (settle_partial_year: true) so the
running total lands exactly on the cap rather than overshooting.

Because this is path-dependent (WHEN the cap is reached depends on the
simulated price path), this function is vectorized across Monte Carlo
iterations rather than assuming a single deterministic price path.
"""

from __future__ import annotations

import numpy as np

from assumptions import get_path, operating_years


def _energy_array(energy_mwh_by_year: dict[int, float], years: list[int]) -> np.ndarray:
    return np.array([energy_mwh_by_year[y] for y in years])


def fixed_ppa_revenue(
    assumptions: dict,
    energy_mwh_by_year: dict[int, float],
    captured_price_paths: np.ndarray,   # [iterations, n_years]
    years: list[int],
) -> np.ndarray:
    """Flat contracted price for `term_years`, indexed at `price_indexation_rate`,
    covering `contracted_volume_share` of output; the uncontracted share (if
    any) and everything after the contract term is sold at the simulated
    merchant (captured) price. Returns [iterations, n_years] revenue."""
    cfg = assumptions["contracts"]["fixed_ppa"]
    base_price = cfg["contract_price_eur_per_mwh"]
    indexation = cfg["price_indexation_rate"]
    term_years = cfg["term_years"]
    contracted_share = cfg["contracted_volume_share"]

    energy = _energy_array(energy_mwh_by_year, years)  # [n_years]
    iterations, n_years = captured_price_paths.shape

    revenue = np.empty((iterations, n_years))
    for i, year in enumerate(years):
        years_since_cod = i  # years[0] is COD year, contract runs from COD
        in_term = years_since_cod < term_years
        contracted_price_t = base_price * (1 + indexation) ** years_since_cod

        contracted_mwh = energy[i] * contracted_share
        merchant_mwh = energy[i] * (1 - contracted_share)

        if in_term:
            revenue[:, i] = (
                contracted_mwh * contracted_price_t
                + merchant_mwh * captured_price_paths[:, i]
            )
        else:
            # post_term_treatment: "merchant" (the only mode implemented —
            # "rollover" would need a second contract price, not specified)
            revenue[:, i] = energy[i] * captured_price_paths[:, i]

    return revenue


def two_sided_cfd_revenue(
    assumptions: dict,
    energy_mwh_by_year: dict[int, float],
    captured_price_paths: np.ndarray,   # [iterations, n_years]
    years: list[int],
) -> dict:
    """Returns:
      revenue_eur          -> [iterations, n_years] producer revenue
      state_flow_eur        -> [iterations, n_years] producer->state flow (+ve = producer pays)
      cumulative_at_end_eur -> [iterations] running total at horizon end (<= cap by construction)
      termination_year_idx  -> [iterations] index into `years` when cap was reached,
                                 or -1 if never reached within the horizon
    """
    cfg = assumptions["contracts"]["two_sided_cfd"]
    strike = cfg["strike_price_eur_per_mwh"]
    indexation = cfg["strike_indexation_rate"]
    term_years = cfg["term_years"]
    cap = cfg["cumulative_payment_cap_eur"]

    if cfg["cap_accounting"] != "net":
        raise NotImplementedError("Only cap_accounting: 'net' is implemented.")
    if cfg["on_cap_reached"] != "terminate_to_merchant":
        raise NotImplementedError("Only on_cap_reached: 'terminate_to_merchant' is implemented.")
    if cfg["clawback_survives_cap"] is not False:
        raise NotImplementedError("Only clawback_survives_cap: false is implemented.")
    settle_partial_year = cfg["settle_partial_year"]

    energy = _energy_array(energy_mwh_by_year, years)  # [n_years]
    iterations, n_years = captured_price_paths.shape

    revenue = np.empty((iterations, n_years))
    state_flow = np.empty((iterations, n_years))

    running_total = np.zeros(iterations)               # cumulative producer->state, net
    active = np.ones(iterations, dtype=bool)            # CfD still in force this iteration
    termination_year_idx = np.full(iterations, -1)

    for i, year in enumerate(years):
        years_since_cod = i
        strike_t = strike * (1 + indexation) ** years_since_cod
        within_term = years_since_cod < term_years

        market_revenue_t = energy[i] * captured_price_paths[:, i]
        # Gap this year IF the CfD were active for the full year:
        full_year_state_flow = (captured_price_paths[:, i] - strike_t) * energy[i]

        cfd_applies_this_year = active & within_term

        # --- Iterations where the CfD is already terminated, or term has expired: pure merchant ---
        revenue[~cfd_applies_this_year, i] = market_revenue_t[~cfd_applies_this_year]
        state_flow[~cfd_applies_this_year, i] = 0.0

        # --- Iterations where the CfD is still active this year ---
        idx_active = np.where(cfd_applies_this_year)[0]
        if idx_active.size:
            prior_total = running_total[idx_active]
            flow = full_year_state_flow[idx_active]
            would_be_total = prior_total + flow

            crossing = would_be_total >= cap  # only meaningful when flow > 0, see below
            # Crossing can only be triggered by a POSITIVE flow (producer paying
            # state); a support year (flow < 0) moves running_total away from
            # the cap and can never trigger termination.
            crossing = crossing & (flow > 0)

            # --- Non-crossing iterations: full year settles at strike price ---
            idx_no_cross = idx_active[~crossing]
            revenue[idx_no_cross, i] = strike_t * energy[i]
            state_flow[idx_no_cross, i] = full_year_state_flow[idx_no_cross]
            running_total[idx_no_cross] = would_be_total[~crossing]

            # --- Crossing iterations: pro-rate this year, then terminate ---
            idx_cross = idx_active[crossing]
            if idx_cross.size:
                flow_cross = full_year_state_flow[idx_cross]
                prior_cross = running_total[idx_cross]
                if settle_partial_year:
                    frac = (cap - prior_cross) / flow_cross  # fraction of the year still under the CfD
                    frac = np.clip(frac, 0.0, 1.0)
                else:
                    frac = 1.0  # whole crossing year still settles at strike, cap is simply exceeded

                revenue[idx_cross, i] = (
                    frac * strike_t * energy[i]
                    + (1 - frac) * captured_price_paths[idx_cross, i] * energy[i]
                )
                state_flow[idx_cross, i] = frac * flow_cross  # only the pre-cap portion is paid/owed
                running_total[idx_cross] = prior_cross + state_flow[idx_cross, i]

                active[idx_cross] = False
                termination_year_idx[idx_cross] = i

    return {
        "revenue_eur": revenue,
        "state_flow_eur": state_flow,
        "cumulative_at_end_eur": running_total,
        "termination_year_idx": termination_year_idx,
    }


def multi_criteria_revenue(
    assumptions: dict,
    energy_mwh_by_year: dict[int, float],
    captured_price_paths: np.ndarray,
    years: list[int],
) -> dict:
    """Scored auction: a weighted score across price/environmental/hybrid/social
    criteria maps to a price adjustment around a reference price (the only
    outcome_mapping implemented — 'revenue_sharing' is configured in
    assumptions.yaml but not yet wired in, since the scaffold's scores are
    still [PLACEHOLDER]). Returns revenue plus the compliance cost add-ons,
    which belong in cashflow, not revenue."""
    cfg = assumptions["contracts"]["multi_criteria"]
    if cfg["outcome_mapping"] != "price_adjustment":
        raise NotImplementedError(
            f"outcome_mapping = '{cfg['outcome_mapping']}' not implemented — "
            "only 'price_adjustment' is wired up."
        )

    weights = cfg["weights"]
    scores = cfg["scores"]
    weighted_score = sum(weights[c] * scores[c] for c in weights)  # 0-100 scale

    adj_cfg = cfg["price_adjustment"]
    reference_price = adj_cfg["reference_price_eur_per_mwh"]
    max_uplift = adj_cfg["max_uplift_eur_per_mwh"]
    max_discount = adj_cfg["max_discount_eur_per_mwh"]

    if weighted_score >= 50:
        contract_price = reference_price + (weighted_score - 50) / 50 * max_uplift
    else:
        contract_price = reference_price - (50 - weighted_score) / 50 * max_discount

    term_years = cfg["term_years"]
    energy = _energy_array(energy_mwh_by_year, years)
    iterations, n_years = captured_price_paths.shape

    revenue = np.empty((iterations, n_years))
    for i, year in enumerate(years):
        if i < term_years:
            revenue[:, i] = energy[i] * contract_price
        else:
            revenue[:, i] = energy[i] * captured_price_paths[:, i]

    return {
        "revenue_eur": revenue,
        "weighted_score": weighted_score,
        "contract_price_eur_per_mwh": contract_price,
        "extra_capex_eur": cfg["compliance_cost"]["capex_uplift_eur"],
        "extra_opex_eur_per_year": cfg["compliance_cost"]["opex_uplift_eur_per_year"],
    }


if __name__ == "__main__":
    # Quick manual check: python src/scenarios.py
    from assumptions import load, validate
    from cashflow import annual_energy_mwh
    from price_model import calibrate, simulate_price_paths

    a = load("assumptions.yaml")
    validate(a)

    energy = annual_energy_mwh(a)
    years = operating_years(a)

    params = calibrate(a)
    sim = simulate_price_paths(a, params)
    captured = sim["captured_paths"]

    ppa_rev = fixed_ppa_revenue(a, energy, captured, years)
    print(f"Fixed PPA — mean revenue year 1: EUR {ppa_rev[:, 0].mean():,.0f}, "
          f"mean revenue year 30: EUR {ppa_rev[:, -1].mean():,.0f}")

    cfd = two_sided_cfd_revenue(a, energy, captured, years)
    strike = get_path(a, "contracts.two_sided_cfd.strike_price_eur_per_mwh")
    print(f"\nTwo-sided CfD:")
    print(f"  mean revenue year 1 (should be ~= strike * energy = "
          f"EUR {strike * energy[years[0]]:,.0f}): "
          f"EUR {cfd['revenue_eur'][:, 0].mean():,.0f}")
    reached = cfd["termination_year_idx"] >= 0
    print(f"  cap reached in {reached.sum()}/{len(reached)} simulated paths")
    if reached.sum():
        term_years_hit = [years[idx] for idx in cfd["termination_year_idx"][reached]]
        print(f"  termination year across paths that hit the cap: "
              f"min={min(term_years_hit)}, median={int(np.median(term_years_hit))}, "
              f"max={max(term_years_hit)}")
    print(f"  mean revenue final year (post-cap, should approach full merchant): "
          f"EUR {cfd['revenue_eur'][:, -1].mean():,.0f}")

    mc = multi_criteria_revenue(a, energy, captured, years)
    print(f"\nMulti-criteria auction:")
    print(f"  weighted score: {mc['weighted_score']:.1f}/100")
    print(f"  contract price: EUR {mc['contract_price_eur_per_mwh']:.2f}/MWh")
    print(f"  mean revenue year 1: EUR {mc['revenue_eur'][:, 0].mean():,.0f}")
