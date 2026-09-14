"""
cashflow.py
-----------
The technical/cost engine shared by all three revenue mechanisms (PPA, two-
sided CfD, multi-criteria auction). This module knows nothing about strike
prices or auction scoring — it only answers "how much energy, at what cost,
in which year". scenarios.py layers mechanism-specific REVENUE on top of the
energy_mwh series this module produces; financials.py combines that revenue
with the cost series here into NPV/IRR.

All years are calendar years, matching assumptions.operating_years() /
the construction window derived from timeline.fid_year and construction_years.
Nothing here is hardcoded — every number traces back to assumptions.yaml.
"""

from __future__ import annotations

from assumptions import get_path, operating_years


def construction_years(assumptions: dict) -> list[int]:
    fid = get_path(assumptions, "timeline.fid_year")
    n = get_path(assumptions, "timeline.construction_years")
    return list(range(fid, fid + n))


def decommissioning_years(assumptions: dict) -> list[int]:
    op_years = operating_years(assumptions)
    n = get_path(assumptions, "timeline.decommissioning_years")
    last_op_year = op_years[-1]
    return list(range(last_op_year + 1, last_op_year + 1 + n))


def annual_energy_mwh(assumptions: dict) -> dict[int, float]:
    """Net annual energy production per operating year, applying degradation
    from COD. capacity_factor_is_net=true means the published 52.5% already
    nets out wake/electrical/availability losses — the loss factors are left
    neutral (1.0 / 0.0) by construction in assumptions.yaml, so this function
    does not apply them a second time regardless of the flag's value; the
    flag exists so a future analyst flipping it to false knows to also set
    real loss factors, not to silently double-count here."""
    capacity_mw = get_path(assumptions, "technical.installed_capacity_mw")
    cf = get_path(assumptions, "technical.net_capacity_factor")
    availability = get_path(assumptions, "technical.availability")
    wake_loss = get_path(assumptions, "technical.wake_and_electrical_loss")
    degradation = get_path(assumptions, "technical.annual_degradation_rate")

    gross_annual_mwh = capacity_mw * cf * 8760 * availability * (1 - wake_loss)

    years = operating_years(assumptions)
    result = {}
    for i, year in enumerate(years):
        result[year] = gross_annual_mwh * (1 - degradation) ** i
    return result


def annual_opex_eur(assumptions: dict) -> dict[int, float]:
    """Fixed O&M (all-in benchmark, no separate variable leg per assumptions.yaml)
    with real escalation, plus any major overhaul lump sums at their configured
    operating-year offsets."""
    capacity_mw = get_path(assumptions, "technical.installed_capacity_mw")
    base_opex_per_mw = get_path(assumptions, "opex.fixed_opex_eur_per_mw_year")
    escalation = get_path(assumptions, "opex.opex_real_escalation_rate")
    seabed_lease = get_path(assumptions, "opex.seabed_lease_eur_per_year")

    overhaul_offsets = set(get_path(assumptions, "opex.major_overhaul.year_offsets"))
    overhaul_cost = get_path(assumptions, "opex.major_overhaul.cost_eur")

    years = operating_years(assumptions)
    result = {}
    for i, year in enumerate(years):
        base = capacity_mw * base_opex_per_mw * (1 + escalation) ** i
        base += seabed_lease
        if i in overhaul_offsets:
            base += overhaul_cost
        result[year] = base
    return result


def capex_outflows_eur(assumptions: dict) -> dict[int, float]:
    """CAPEX spend spread across construction_years per the S-curve
    spend_profile. Grid connection is already inside total_capex_eur (see
    assumptions.yaml) — nothing added here on top."""
    total_capex = get_path(assumptions, "capex.total_capex_eur")
    profile = get_path(assumptions, "capex.spend_profile")
    years = construction_years(assumptions)

    if len(profile) != len(years):
        raise ValueError(
            f"capex.spend_profile has {len(profile)} entries but "
            f"timeline.construction_years = {len(years)} — they must match."
        )
    if abs(sum(profile) - 1.0) > 1e-6:
        raise ValueError(f"capex.spend_profile must sum to 1.0, got {sum(profile)}")

    return {year: total_capex * share for year, share in zip(years, profile)}


def decommissioning_outflows_eur(assumptions: dict) -> dict[int, float]:
    """Decommissioning cost, spread evenly across decommissioning_years
    immediately after the last operating year."""
    capacity_mw = get_path(assumptions, "technical.installed_capacity_mw")
    cost_per_mw = get_path(assumptions, "decommissioning.cost_eur_per_mw")
    scrap_credit = get_path(assumptions, "decommissioning.scrap_credit_eur_per_mw")

    total_cost = capacity_mw * (cost_per_mw - scrap_credit)
    years = decommissioning_years(assumptions)
    per_year = total_cost / len(years)
    return {year: per_year for year in years}


def depreciation_schedule_eur(assumptions: dict) -> dict[int, float]:
    """Danish tax depreciation: 15%/yr declining balance on total CAPEX,
    starting from COD (assets aren't in service, and can't be depreciated,
    during construction). The declining-balance base never reaches exactly
    zero, so once the remaining balance falls below write_off_threshold_eur
    it is written off in full that year rather than trailing indefinitely.
    Returns the DEPRECIATION EXPENSE (tax shield base) per calendar year,
    covering only the years in which the asset is still being written down —
    not necessarily every operating year, since a fast-declining balance can
    cross the write-off threshold well before year 30."""
    total_capex = get_path(assumptions, "capex.total_capex_eur")
    rate = get_path(assumptions, "finance.depreciation.declining_balance_rate")
    threshold = get_path(assumptions, "finance.depreciation.write_off_threshold_eur")
    method = get_path(assumptions, "finance.depreciation.method")

    if method != "declining_balance":
        raise NotImplementedError(
            f"finance.depreciation.method = '{method}' is not implemented; "
            "only 'declining_balance' is supported (Danish rule for turbines > 1MW)."
        )

    years = operating_years(assumptions)
    schedule: dict[int, float] = {}
    balance = total_capex
    for year in years:
        if balance <= threshold:
            schedule[year] = balance
            balance = 0.0
            break
        expense = balance * rate
        schedule[year] = expense
        balance -= expense
    return schedule


def build_cost_schedule(assumptions: dict) -> dict:
    """Convenience bundle: everything cashflow.py knows, keyed by category,
    each a dict[year -> EUR]. Used by financials.py and scenarios.py so they
    don't need to call every function individually."""
    return {
        "capex": capex_outflows_eur(assumptions),
        "opex": annual_opex_eur(assumptions),
        "decommissioning": decommissioning_outflows_eur(assumptions),
        "depreciation": depreciation_schedule_eur(assumptions),
        "energy_mwh": annual_energy_mwh(assumptions),
    }


if __name__ == "__main__":
    # Quick manual check: python src/cashflow.py
    from assumptions import load, validate

    a = load("assumptions.yaml")
    validate(a)

    schedule = build_cost_schedule(a)

    print(f"Construction years: {construction_years(a)}")
    print(f"Operating years:    {operating_years(a)[0]}-{operating_years(a)[-1]}")
    print(f"Decommissioning yrs:{decommissioning_years(a)}")

    print(f"\nCAPEX outflows (should sum to total_capex_eur):")
    total_capex_check = sum(schedule["capex"].values())
    print(f"  sum = EUR {total_capex_check:,.0f}  "
          f"(target: EUR {get_path(a, 'capex.total_capex_eur'):,.0f})")

    first_op_year = operating_years(a)[0]
    last_op_year = operating_years(a)[-1]
    print(f"\nYear {first_op_year}: energy = {schedule['energy_mwh'][first_op_year]:,.0f} MWh, "
          f"OPEX = EUR {schedule['opex'][first_op_year]:,.0f}")
    print(f"Year {last_op_year}:  energy = {schedule['energy_mwh'][last_op_year]:,.0f} MWh, "
          f"OPEX = EUR {schedule['opex'][last_op_year]:,.0f}")
    print(f"Degradation check: year {last_op_year} energy is "
          f"{schedule['energy_mwh'][last_op_year] / schedule['energy_mwh'][first_op_year]:.3f} "
          f"of year {first_op_year}")

    dep_years = sorted(schedule["depreciation"].keys())
    print(f"\nDepreciation: EUR {sum(schedule['depreciation'].values()):,.0f} total "
          f"written off over {len(dep_years)} years ({dep_years[0]}-{dep_years[-1]})")

    decomm_total = sum(schedule["decommissioning"].values())
    print(f"\nDecommissioning: EUR {decomm_total:,.0f} total "
          f"over years {decommissioning_years(a)}")
