# src/flashcat/indicators/eddi_csv.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------- #
# Thermodynamics / ET components  #
# ------------------------------- #

def _sat_vapor_pressure_kpa(t_c: pd.Series | float) -> pd.Series:
    # Tetens formula (kPa)
    return 0.6108 * np.exp((17.27 * t_c) / (t_c + 237.3))

def _slope_svpc_kpa_per_c(t_c: pd.Series) -> pd.Series:
    es = _sat_vapor_pressure_kpa(t_c)
    return 4098.0 * es / (t_c + 237.3) ** 2

def _pressure_from_elevation_kpa(elev_m: pd.Series | float) -> pd.Series:
    # Standard atmosphere (kPa): P = 101.3 * ((293 - 0.0065*z)/293)^5.26
    return 101.3 * ((293.0 - 0.0065 * np.asarray(elev_m)) / 293.0) ** 5.26

def _actual_vapor_pressure_from_rh_kpa(t_c: pd.Series, rh: pd.Series) -> pd.Series:
    es = _sat_vapor_pressure_kpa(t_c)
    rhf = np.where(rh > 1.0, rh / 100.0, rh)  # accept 0–100 or 0–1
    return es * rhf

def _actual_vapor_pressure_from_tdew_kpa(tdew_c: pd.Series) -> pd.Series:
    return _sat_vapor_pressure_kpa(tdew_c)

@dataclass
class PMCoeffs:
    Cn: float = 1600.0  # tall crop daily
    Cd: float = 0.38

def etref_pm_asce_daily_df(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    pm: PMCoeffs = PMCoeffs(),
) -> pd.Series:
    """
    Compute daily ASCE–PM ETref (mm/day) from a DataFrame of daily inputs.
    Required columns in colmap:
      - t_c, u2_ms, rn_mj_m2_d
      - (rh_pct OR tdew_c)
    Optional:
      - elev_m OR pressure_kpa
    """
    t = df[colmap["t_c"]]
    u2 = df[colmap["u2_ms"]]
    rn = df[colmap["rn_mj_m2_d"]]

    if "tdew_c" in colmap and colmap["tdew_c"] in df.columns:
        ea = _actual_vapor_pressure_from_tdew_kpa(df[colmap["tdew_c"]])
    elif "rh_pct" in colmap and colmap["rh_pct"] in df.columns:
        ea = _actual_vapor_pressure_from_rh_kpa(t, df[colmap["rh_pct"]])
    else:
        raise ValueError("Provide either 'rh_pct' or 'tdew_c' in colmap and DataFrame.")

    delta = _slope_svpc_kpa_per_c(t)

    if "pressure_kpa" in colmap and colmap["pressure_kpa"] in df.columns:
        gamma = 1.013e-3 * df[colmap["pressure_kpa"]] / (0.622 * 2.45)
    else:
        elev = df[colmap.get("elev_m")] if colmap.get("elev_m") in df.columns else 0.0
        p_kpa = _pressure_from_elevation_kpa(elev)
        gamma = 1.013e-3 * p_kpa / (0.622 * 2.45)

    es = _sat_vapor_pressure_kpa(t)
    vpd = np.maximum(es - ea, 0.0)

    num = 0.408 * delta * rn + gamma * (pm.Cn / (t + 273.0)) * u2 * vpd
    den = delta + gamma * (1.0 + pm.Cd * u2)
    et = (num / den).clip(lower=0.0)
    et.name = "etref_pm"
    return et


# --------------------------- #
# Rolling window + baseline   #
# --------------------------- #

def _days_from_scale(scale: str) -> int:
    s = scale.upper()
    if s.endswith("W"):  # weeks
        return int(s[:-1]) * 7
    if s.endswith("D"):  # days
        return int(s[:-1])
    if s.endswith("M"):  # months ~ average days/month
        return max(1, int(round(int(s[:-1]) * 30.4375)))
    raise ValueError(f"Bad scale {scale!r}. Use like '1W','4W','45D','3M'.")

def _rolling_sum(series: pd.Series, window_days: int, min_frac: float = 0.8) -> pd.Series:
    min_periods = max(1, int(math.ceil(min_frac * window_days)))
    return series.rolling(window_days, min_periods=min_periods).sum()

def _invnorm_ppf(p: np.ndarray) -> np.ndarray:
    # Acklam approximation; same as earlier but vectorized for arrays
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p = np.asarray(p, dtype=float)
    x = np.full_like(p, np.nan, dtype=float)
    plow = 0.02425
    phigh = 1 - plow

    mask = (p > 0) & (p < plow)
    q = np.sqrt(-2 * np.log(p[mask]))
    x[mask] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    mask = (p >= plow) & (p <= phigh)
    q = p[mask] - 0.5
    r = q*q
    x[mask] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    mask = (p > phigh) & (p < 1)
    q = np.sqrt(-2 * np.log(1 - p[mask]))
    x[mask] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1))
    return x

def _standardize_empirical_monthly(
    y: pd.Series, baseline: Tuple[str, str]
) -> pd.Series:
    """
    Empirical CDF per calendar month using only baseline years,
    then inverse-normal to z (EDDI).
    """
    s = y.dropna().copy()
    idx = s.index
    years = idx.year
    months = idx.month
    y0, y1 = int(baseline[0]), int(baseline[1])

    out = pd.Series(index=y.index, dtype=float, name="EDDI")

    for m in range(1, 13):
        mask_m = (months == m)
        if not mask_m.any():
            continue
        mask_base = mask_m & (years >= y0) & (years <= y1)
        base_vals = s[mask_base]
        if base_vals.size < 10:
            continue
        base_sorted = np.sort(base_vals.values)
        n = base_sorted.size

        # Map all month-m values to percentile against baseline distribution
        vals = s[mask_m].values
        idxs = np.searchsorted(base_sorted, vals, side="right")
        p = idxs / (n + 1.0)
        z = _invnorm_ppf(np.clip(p, 1e-6, 1 - 1e-6))
        out.loc[s.index[mask_m]] = z

    return out


# --------------------------- #
# Public CSV-facing function  #
# --------------------------- #

def eddi_pm_from_csv(
    input_csv: str,
    output_csv: str,
    *,
    colmap: Dict[str, str] = {
        "date": "date",
        "t_c": "t_c",
        "u2_ms": "u2_ms",
        "rn_mj_m2_d": "rn_mj_m2_d",
        # Use either rh_pct OR tdew_c. If both provided, tdew_c wins.
        "rh_pct": "rh_pct",
        # "tdew_c": "tdew_c",
        # Optional:
        "elev_m": "elev_m",
        # "pressure_kpa": "pressure_kpa",
        # Optional group column for multi-site CSVs:
        # "group": "group",
    },
    scale: str = "4W",
    baseline: Tuple[str, str] = ("1981", "2010"),
    min_window_frac: float = 0.8,
    date_format: Optional[str] = None,
) -> None:
    """
    Read daily meteorology from CSV, compute ASCE–PM ETref -> EDDI on the given scale,
    and write a CSV with ETref, accumulated ET, and EDDI.

    If a 'group' column is present in colmap and the CSV, EDDI is computed per-group.

    Parameters
    ----------
    input_csv : str
    output_csv : str
    colmap : Dict[str,str]
        Maps logical variable names to CSV column names (see defaults).
    scale : str
        Window length like '1W','2W','4W','3M','45D' (rolling sum).
    baseline : (str,str)
        Baseline years for standardization (inclusive).
    min_window_frac : float
        Minimum fraction of non-NaN samples required in the rolling window.
    date_format : Optional[str]
        If dates are not ISO; e.g., '%d/%m/%Y'.
    """
    df = pd.read_csv(input_csv)

    # Parse date and sort
    date_col = colmap["date"]
    df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors="raise")
    df = df.sort_values(date_col)

    # Group support
    group_col = colmap.get("group")
    if group_col and group_col in df.columns:
        grouped = []
        for g, sub in df.groupby(group_col, sort=False):
            out_g = _eddi_pm_single(sub, colmap, scale, baseline, min_window_frac)
            out_g[group_col] = g
            grouped.append(out_g)
        out = pd.concat(grouped, axis=0, ignore_index=True)
    else:
        out = _eddi_pm_single(df, colmap, scale, baseline, min_window_frac)

    out.to_csv(output_csv, index=False)


def _eddi_pm_single(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    scale: str,
    baseline: Tuple[str, str],
    min_window_frac: float,
) -> pd.DataFrame:
    date_col = colmap["date"]
    s_et = etref_pm_asce_daily_df(df, colmap)
    s_et.index = pd.to_datetime(df[date_col].values)

    # Rolling sum over the desired window
    L = _days_from_scale(scale)
    y = _rolling_sum(s_et, L, min_frac=min_window_frac)
    y.name = "etref_accum_mm"

    # Standardize per-month using empirical CDF (baseline years)
    z = _standardize_empirical_monthly(y, baseline)
    z.name = "EDDI"

    out = pd.DataFrame({
        date_col: s_et.index,
        "etref_mm_d": s_et.values,
        "etref_accum_mm": y.values,
        "EDDI": z.values,
    })
    return out
