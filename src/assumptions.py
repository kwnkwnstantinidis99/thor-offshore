"""
assumptions.py
---------------
Loads assumptions.yaml (the single source of truth for every numeric input
in this project) and provides:

  - load(path)                    -> nested dict
  - get_path(assumptions, path)   -> value at a dotted path, e.g. "capex.total_capex_eur"
  - set_path(assumptions, path, value) -> returns a NEW dict with that path replaced
                                          (used by sensitivity.py to build scenario copies
                                          without mutating the base assumptions)
  - validate(assumptions, allow_unresolved=False)
  - operating_years(assumptions)  -> list[int] of calendar years the project is operating,
                                      derived from timeline.cod_year and
                                      timeline.operating_life_years. Never hardcode this
                                      elsewhere — price_model.py's horizon and every
                                      cashflow year index must come from here.

No numeric values are hardcoded in this file. If you find yourself typing a number
here, it belongs in assumptions.yaml instead.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# Dotted paths that are LEGITIMATELY null in assumptions.yaml — either because
# they're derived elsewhere (capex_eur_per_kw is redundant once total_capex_eur
# is set) or because they're filled at runtime by calibration (OU parameters).
# Any other null value is treated as a genuine [UNRESOLVED] item and blocks
# validate() unless the caller explicitly passes allow_unresolved=True.
_EXPECTED_NULL_PATHS = {
    "capex.capex_eur_per_kw",
    "price_model.parameters.long_run_mean_eur_per_mwh",
    "price_model.parameters.reversion_speed_per_year",
    "price_model.parameters.volatility_eur_per_mwh",
    "price_model.parameters.initial_price_eur_per_mwh",
}


def load(path: str | Path = "assumptions.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def get_path(assumptions: dict[str, Any], dotted_path: str) -> Any:
    node: Any = assumptions
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"Path '{dotted_path}' not found in assumptions (failed at '{part}')")
        node = node[part]
    return node


def set_path(assumptions: dict[str, Any], dotted_path: str, value: Any) -> dict[str, Any]:
    """Returns a deep-copied dict with the value at dotted_path replaced.
    Used by sensitivity.py so each scenario run gets an isolated copy."""
    new_assumptions = copy.deepcopy(assumptions)
    parts = dotted_path.split(".")
    node = new_assumptions
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    return new_assumptions


def _find_unexpected_nulls(node: Any, prefix: str = "") -> list[str]:
    unexpected: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if value is None:
                if path not in _EXPECTED_NULL_PATHS:
                    unexpected.append(path)
            else:
                unexpected.extend(_find_unexpected_nulls(value, path))
    return unexpected


def validate(assumptions: dict[str, Any], allow_unresolved: bool = False) -> None:
    """Raises ValueError listing every unresolved (null) leaf that isn't on the
    expected-null whitelist, unless allow_unresolved=True. This is the load-bearing
    check that stops the pipeline from silently running on a placeholder."""
    unexpected_nulls = _find_unexpected_nulls(assumptions)

    if unexpected_nulls and not allow_unresolved:
        formatted = "\n".join(f"  - {p}" for p in unexpected_nulls)
        raise ValueError(
            "assumptions.yaml has unresolved [UNRESOLVED] values that must be set "
            "before running an analysis (or pass allow_unresolved=True to proceed "
            f"anyway, at your own risk):\n{formatted}"
        )

    # A few structural sanity checks, independent of the null-scan above —
    # these catch a wrong-but-non-null value (e.g. a stray typo), not a missing one.
    strike = get_path(assumptions, "contracts.two_sided_cfd.strike_price_eur_per_mwh")
    if not (0 <= strike < 5):
        raise ValueError(
            f"contracts.two_sided_cfd.strike_price_eur_per_mwh = {strike} looks wrong. "
            "The confirmed Thor strike price is ~EUR 0.01/MWh (DKK 0.1/MWh). "
            "A value like 30.0 is the old placeholder — check assumptions.yaml."
        )

    cf_is_net = get_path(assumptions, "technical.capacity_factor_is_net")
    if not isinstance(cf_is_net, bool):
        raise ValueError(
            "technical.capacity_factor_is_net must be explicitly true or false, not "
            f"{cf_is_net!r}. This flag changes annual energy output by ~10-15%."
        )


def operating_years(assumptions: dict[str, Any]) -> list[int]:
    """Calendar years the project is generating revenue, derived from timeline —
    never hardcoded. E.g. cod_year=2027, operating_life_years=30 -> 2027..2056."""
    cod_year = get_path(assumptions, "timeline.cod_year")
    life = get_path(assumptions, "timeline.operating_life_years")
    return list(range(cod_year, cod_year + life))


if __name__ == "__main__":
    # Quick manual check: python src/assumptions.py
    a = load("assumptions.yaml")
    validate(a)
    years = operating_years(a)
    print(f"Loaded OK. Operating years: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"Strike price: EUR {get_path(a, 'contracts.two_sided_cfd.strike_price_eur_per_mwh')}/MWh")
    print(f"Capacity factor is net: {get_path(a, 'technical.capacity_factor_is_net')}")
