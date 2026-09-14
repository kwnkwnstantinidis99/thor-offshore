"""
price_model.py
---------------
Stochastic model of DK1 electricity prices: an arithmetic (level) Ornstein-
Uhlenbeck process, calibrated on historical annual data and simulated FORWARD
across the project's full operating horizon.

Calibration vs. simulation — the distinction that matters here
----------------------------------------------------------------
The historical series (data/dk1_annual_prices.csv, 2015-2025) is used ONLY to
estimate the process's statistical parameters: where prices tend to revert to
(long-run mean), how fast (reversion speed / half-life), and how volatile they
are. It is NOT resampled, repeated, or otherwise reused as the future price
path. Once calibrated, the model is run forward from today across the full
operating horizon (derived from assumptions.timeline — see assumptions.py),
producing a DISTRIBUTION of possible future price paths via Monte Carlo. The
historical window and the simulation horizon are different objects serving
different purposes, and this module keeps them structurally separate: history
only ever flows into `calibrate()`; only `calibrate()`'s output flows into
`simulate_price_paths()`.

Why arithmetic, not geometric/log-OU
-------------------------------------
A log-OU process cannot produce negative prices. DK1 has real, recurring
negative-price hours, and those interact directly with the two-sided CfD's
net cap accounting (cap_accounting: net) — a negative captured price widens
the strike-market gap and draws down the cap faster. Arithmetic OU admits
negative values natively, which is required here, not optional.

Why half-life over a raw statistical fit
------------------------------------------
Only ~10 usable annual observations exist (2015-2018 regional proxy + 2020-
2025 DK1-specific; 2019 is a gap), several of them from an energy-crisis
window. That is far too few to pin down a reversion speed with any real
statistical confidence — an AR(1) fit on this sample is fitting noise as much
as signal. Since the purpose of this model is to project FORWARD three
decades, not to explain the past decade precisely, we prefer a judgement-based
`half_life_years` prior (see assumptions.yaml) over the fitted value. The fit
is still computed and reported (see OUParameters.reversion_speed_fitted) so
the difference is visible, not just silently substituted.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from assumptions import get_path, operating_years


@dataclass
class OUParameters:
    long_run_mean: float            # theta / mu, EUR/MWh
    reversion_speed: float          # kappa actually used (per year)
    reversion_speed_fitted: float   # kappa from raw AR(1) fit, for comparison only
    volatility: float               # sigma, EUR/MWh (annual, absolute)
    initial_price: float            # P_0, EUR/MWh
    half_life_used: bool            # True if half_life_years prior overrode the fit
    years_used_for_mean: list[int] = field(default_factory=list)
    years_excluded_from_mean: list[int] = field(default_factory=list)
    n_observations: int = 0


def _read_historical_prices(csv_path: str | Path) -> list[tuple[int, float]]:
    """Reads (year, price) pairs, skipping rows with no price (the documented
    2019 gap). Does NOT interpolate or guess — a missing year is simply absent
    from the returned series."""
    series: list[tuple[int, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(row for row in f if not row.lstrip().startswith("#"))
        for row in reader:
            price_str = row["price_eur_per_mwh"].strip()
            if price_str == "":
                continue  # genuine gap (2019) — left out, not interpolated
            series.append((int(row["year"]), float(price_str)))
    return sorted(series)


def calibrate(assumptions: dict) -> OUParameters:
    """Calibrates OU parameters from data/dk1_annual_prices.csv per the method
    and overrides declared in assumptions.yaml's price_model block."""
    pm = assumptions["price_model"]
    calib = pm["calibration"]
    params_cfg = pm["parameters"]

    csv_path = calib["source_file"]
    series = _read_historical_prices(csv_path)

    n_obs = len(series)
    if n_obs < calib["min_observations"]:
        raise ValueError(
            f"Only {n_obs} historical price observations found in {csv_path}, "
            f"below the configured minimum of {calib['min_observations']}."
        )
    if n_obs < calib.get("warn_below_observations", 15):
        print(
            f"[price_model] WARNING: calibrating on only {n_obs} annual observations. "
            "Reversion-speed identification from this sample is weak — this is exactly "
            "why prefer_half_life_over_fit is kept on. See module docstring."
        )

    exclude_years = set(calib.get("exclude_years") or [])
    years_all = [y for y, _ in series]
    mean_series = [(y, p) for y, p in series if y not in exclude_years]
    years_used_for_mean = [y for y, _ in mean_series]

    # --- Long-run mean ------------------------------------------------------
    long_run_mean = params_cfg.get("long_run_mean_eur_per_mwh")
    if long_run_mean is None:
        long_run_mean = float(np.mean([p for _, p in mean_series]))

    # --- AR(1) fit (for comparison / diagnostics; may or may not be used) ---
    # Only consecutive-year pairs are valid AR(1) transitions — a gap (e.g.
    # 2019 missing, or 2018->2020) is skipped rather than treated as dt=1.
    pairs = [
        (series[i][1], series[i + 1][1])
        for i in range(len(series) - 1)
        if series[i + 1][0] - series[i][0] == 1
    ]
    if len(pairs) >= 2:
        x = np.array([p for p, _ in pairs])
        y = np.array([p for _, p in pairs])
        # y = phi * x + c  (simple OLS)
        A = np.vstack([x, np.ones_like(x)]).T
        phi, c = np.linalg.lstsq(A, y, rcond=None)[0]
        phi = float(np.clip(phi, 1e-6, 0.999))  # keep it a valid mean-reverting AR(1)
        kappa_fitted = -np.log(phi)  # dt = 1 year
        residuals = y - (phi * x + c)
        sigma_fitted = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(y))
    else:
        # Not enough consecutive pairs to fit anything meaningful.
        kappa_fitted = np.log(2) / params_cfg["half_life_years"]
        sigma_fitted = float(np.std([p for _, p in series])) if len(series) > 1 else 10.0

    # --- Reversion speed: half-life prior wins unless explicitly told not to ---
    prefer_half_life = params_cfg.get("prefer_half_life_over_fit", True)
    half_life_years = params_cfg.get("half_life_years")
    explicit_kappa = params_cfg.get("reversion_speed_per_year")

    half_life_used = False
    if explicit_kappa is not None:
        reversion_speed = float(explicit_kappa)
    elif prefer_half_life and half_life_years:
        reversion_speed = float(np.log(2) / half_life_years)
        half_life_used = True
    else:
        reversion_speed = float(kappa_fitted)

    # --- Volatility: fitted residual std (half-life only overrides speed, not sigma) ---
    volatility = params_cfg.get("volatility_eur_per_mwh")
    if volatility is None:
        volatility = sigma_fitted

    # --- Initial price: defaults to the last observed year ---
    initial_price = params_cfg.get("initial_price_eur_per_mwh")
    if initial_price is None:
        initial_price = series[-1][1]

    return OUParameters(
        long_run_mean=float(long_run_mean),
        reversion_speed=float(reversion_speed),
        reversion_speed_fitted=float(kappa_fitted),
        volatility=float(volatility),
        initial_price=float(initial_price),
        half_life_used=half_life_used,
        years_used_for_mean=years_used_for_mean,
        years_excluded_from_mean=sorted(exclude_years),
        n_observations=n_obs,
    )


def simulate_price_paths(assumptions: dict, params: OUParameters) -> dict:
    """Simulates forward DK1 price paths over the FULL operating horizon
    (never the historical window) using exact discrete-time OU sampling.

    Returns a dict with:
      years           -> list[int], the simulated calendar years
      baseload_paths  -> np.ndarray [iterations, n_years], simulated system price
      captured_paths  -> np.ndarray [iterations, n_years], after capture_rate applied
    """
    years = operating_years(assumptions)
    n_years = len(years)

    sim_cfg = assumptions["price_model"]["simulation"]
    iterations = sim_cfg["iterations"]
    seed = sim_cfg["random_seed"]
    antithetic = sim_cfg.get("antithetic_variates", True)

    neg_cfg = assumptions["price_model"]["negative_prices"]
    allow_negative = neg_cfg.get("allow", True)
    floor = neg_cfg.get("floor_eur_per_mwh", -np.inf)

    kappa = params.reversion_speed
    mu = params.long_run_mean
    sigma = params.volatility
    dt = 1.0  # annual steps

    rng = np.random.default_rng(seed)

    # Exact discretization of an OU process over step dt:
    #   P_{t+1} = mu + (P_t - mu) * exp(-kappa*dt) + sigma_d * Z
    #   sigma_d = sigma * sqrt((1 - exp(-2*kappa*dt)) / (2*kappa))     [kappa > 0]
    decay = np.exp(-kappa * dt)
    if kappa > 1e-8:
        sigma_d = sigma * np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa))
    else:
        sigma_d = sigma * np.sqrt(dt)  # kappa ~ 0 fallback: pure random walk step

    if antithetic:
        half = (iterations + 1) // 2
        z_half = rng.standard_normal(size=(half, n_years))
        z = np.vstack([z_half, -z_half])[:iterations]
    else:
        z = rng.standard_normal(size=(iterations, n_years))

    paths = np.empty((iterations, n_years))
    prev = np.full(iterations, params.initial_price)
    for t in range(n_years):
        prev = mu + (prev - mu) * decay + sigma_d * z[:, t]
        if not allow_negative:
            prev = np.maximum(prev, 0.0)
        elif np.isfinite(floor):
            prev = np.maximum(prev, floor)
        paths[:, t] = prev

    captured_paths = paths.copy()
    if sim_cfg.get("apply_capture_rate", True):
        capture_rate = assumptions["market"]["capture_rate"]
        decline = assumptions["market"].get("capture_rate_annual_decline", 0.0)
        for t in range(n_years):
            effective_capture = max(capture_rate - decline * t, 0.0)
            captured_paths[:, t] = paths[:, t] * effective_capture

    return {
        "years": years,
        "baseload_paths": paths,
        "captured_paths": captured_paths,
    }


def get_price_statistics(simulation_result: dict) -> dict:
    """Per-year summary statistics (mean + percentiles) across all iterations,
    for both the baseload and capture-adjusted paths."""
    stats = {}
    for key in ("baseload_paths", "captured_paths"):
        arr = simulation_result[key]
        stats[key] = {
            "mean": arr.mean(axis=0),
            "p5": np.percentile(arr, 5, axis=0),
            "p25": np.percentile(arr, 25, axis=0),
            "p50": np.percentile(arr, 50, axis=0),
            "p75": np.percentile(arr, 75, axis=0),
            "p95": np.percentile(arr, 95, axis=0),
        }
    stats["years"] = simulation_result["years"]
    return stats


if __name__ == "__main__":
    # Quick manual check: python src/price_model.py
    from assumptions import load, validate

    a = load("assumptions.yaml")
    validate(a)

    params = calibrate(a)
    print("Calibrated OU parameters:")
    print(f"  long-run mean (theta):     EUR {params.long_run_mean:.2f}/MWh")
    print(f"  reversion speed (used):    {params.reversion_speed:.4f} /yr "
          f"(half-life {np.log(2)/params.reversion_speed:.1f} yr)"
          f"{'  <- half-life prior' if params.half_life_used else '  <- fitted'}")
    print(f"  reversion speed (fitted):  {params.reversion_speed_fitted:.4f} /yr  (for comparison only)")
    print(f"  volatility (sigma):        EUR {params.volatility:.2f}/MWh")
    print(f"  initial price (P0):        EUR {params.initial_price:.2f}/MWh")
    print(f"  mean estimated from years: {params.years_used_for_mean}")
    print(f"  years excluded from mean:  {params.years_excluded_from_mean}")

    result = simulate_price_paths(a, params)
    stats = get_price_statistics(result)
    print(f"\nSimulated {result['captured_paths'].shape[0]} paths over "
          f"{stats['years'][0]}-{stats['years'][-1]}")
    print(f"  Captured price, year {stats['years'][0]}: "
          f"p5={stats['captured_paths']['p5'][0]:.1f}  "
          f"p50={stats['captured_paths']['p50'][0]:.1f}  "
          f"p95={stats['captured_paths']['p95'][0]:.1f}  EUR/MWh")
    print(f"  Captured price, year {stats['years'][-1]}: "
          f"p5={stats['captured_paths']['p5'][-1]:.1f}  "
          f"p50={stats['captured_paths']['p50'][-1]:.1f}  "
          f"p95={stats['captured_paths']['p95'][-1]:.1f}  EUR/MWh")
