"""
ESI (Evaporative Stress Index) for a single grid, numpy-first.

Author: Kumar Puran Tripathy
Email:  tripathypuranbdk@gmail.com
Date:   Sept 04, 2025

Design:
- Inputs are 1D daily arrays for one grid: year, month, day, ET (mm/d), ETref (mm/d).
- Compute ESR = ET / ETref (clipped).
- Smooth via rolling mean over N days (scale_days).
- Standardize by calendar month using an empirical CDF over a baseline period, then
  transform percentile to standard normal (z). That z is ESI.

HOW TO USE:
(1) Your data should be in a CSV file with columns: year,month,day,et_mm_d,etref_mm_d.
year,month,day,et_mm_d,etref_mm_d
2000,1,1,2.3,3.8
2000,1,2,2.1,3.6
...
(2) Call the function `esi_from_csv_single_grid` with numpy arrays of these columns.
import numpy as np
from flashcat.indicators.esi_numpy_single import esi_from_csv_single_grid

esi_from_csv_single_grid(
    input_csv=r"C:\Users\ktripat\Dropbox\Python_Projects\data\grid_TX01_daily.csv",
    output_csv=r"C:\Users\ktripat\Dropbox\Python_Projects\data\grid_TX01_esi_28d.csv",
    has_header=True,          # set False if no header row
    scale_days=28,            # choose 7/14/28/56...
    baseline=(1981, 2010),
    min_window_frac=0.8,
    clip_esr=(0.0, 1.5)
)

------------------------ OR ------------------------

(2) import numpy as np
from flashcat.indicators.esi_numpy_single import esi_pipeline_single_grid

# year, month, day, et, etref are 1D numpy arrays (same length)
dates, esi = esi_pipeline_single_grid(
    year, month, day, et, etref,
    scale_days=28, baseline=(1981,2010)
)
"""

from __future__ import annotations
import math
import numpy as np


# ------------------------ small utilities ------------------------ #

def _invnorm_ppf(p: np.ndarray) -> np.ndarray:
    """
    Fast inverse-normal (probit) approximation (Acklam, 2003).
    p must be in (0, 1). Vectorized.
    """
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    p = np.asarray(p, float)
    x = np.full_like(p, np.nan, float)
    pl, ph = 0.02425, 1 - 0.02425

    m = (p > 0) & (p < pl)
    q = np.sqrt(-2 * np.log(p[m]))
    x[m] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1))

    m = (p >= pl) & (p <= ph)
    q = p[m] - 0.5
    r = q*q
    x[m] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
            (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1))

    m = (p > ph) & (p < 1)
    q = np.sqrt(-2 * np.log(1 - p[m]))
    x[m] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
              ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1))
    return x


def _rolling_mean_ignore_nan(x: np.ndarray, L: int, min_frac: float = 0.8) -> np.ndarray:
    """
    Right-aligned rolling mean with NaN tolerance.
    Window is valid if >= min_frac * L finite values are present.
    """
    x = np.asarray(x, float)
    n = x.size
    if L <= 0 or L > n:
        return np.full(n, np.nan, float)

    valid = np.isfinite(x).astype(float)
    x0 = np.nan_to_num(x, nan=0.0)
    k = np.ones(L, float)

    sum_x = np.convolve(x0, k, mode="full")[L-1: L-1+n]
    cnt_x = np.convolve(valid, k, mode="full")[L-1: L-1+n]

    need = math.ceil(min_frac * L)
    out = np.full(n, np.nan, float)
    ok = cnt_x >= need
    out[ok] = sum_x[ok] / cnt_x[ok]
    return out


def _ym_from_ymd(year: np.ndarray, month: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dates[D], years[int], months[int]) from year/month/day arrays."""
    year = np.asarray(year, int)
    month = np.asarray(month, int)
    day = np.asarray(day, int)

    # Build YYYY-MM-DD strings (robust across calendars)
    dates = np.array([np.datetime64(f"{y:04d}-{m:02d}-{d:02d}") for y, m, d in zip(year, month, day)],
                     dtype="datetime64[D]")

    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = (dates.astype("datetime64[M]").astype(int) % 12) + 1
    return dates, years, months


# ------------------------- core ESI routines ------------------------- #

def esr_daily_1d(et_mm_d: np.ndarray, etref_mm_d: np.ndarray, clip: tuple[float, float] = (0.0, 1.5)) -> np.ndarray:
    """
    Daily evaporative stress ratio for one grid:
    ESR = ET / ETref, clipped to [clip_min, clip_max]; ETref <= 0 is treated as missing.
    """
    et = np.asarray(et_mm_d, float)
    etref = np.asarray(etref_mm_d, float)

    etref[~np.isfinite(etref) | (etref <= 0.0)] = np.nan
    esr = et / etref
    return np.clip(esr, clip[0], clip[1])


def esr_scale_1d(esr_daily: np.ndarray, scale_days: int = 28, min_window_frac: float = 0.8) -> np.ndarray:
    """
    Smooth ESR with a right-aligned rolling mean over 'scale_days'.
    Typical choices: 7, 14, 28, 56.
    """
    return _rolling_mean_ignore_nan(esr_daily, int(scale_days), min_frac=min_window_frac)


def esi_standardize_by_month_1d(esr_scaled: np.ndarray,
                                years: np.ndarray,
                                months: np.ndarray,
                                baseline: tuple[int, int] = (1981, 2010)) -> np.ndarray:
    """
    Month-wise empirical standardization:
    For each calendar month, compute percentile vs baseline years, then map to z (standard normal).
    """
    x = np.asarray(esr_scaled, float)
    y = np.asarray(years, int)
    m = np.asarray(months, int)

    out = np.full(x.size, np.nan, float)
    y0, y1 = int(baseline[0]), int(baseline[1])
    eps = 1e-6

    for month_id in range(1, 13):
        mask = (m == month_id) & np.isfinite(x)
        if not np.any(mask):
            continue

        base = mask & (y >= y0) & (y <= y1)
        base_vals = x[base]
        if base_vals.size < 10:
            continue

        s = np.sort(base_vals)
        idx = np.searchsorted(s, x[mask], side="right")
        p = idx / (s.size + 1.0)
        out[np.where(mask)[0]] = _invnorm_ppf(np.clip(p, eps, 1 - eps))

    return out


def esi_pipeline_single_grid(year: np.ndarray,
                              month: np.ndarray,
                              day: np.ndarray,
                              et_mm_d: np.ndarray,
                              etref_mm_d: np.ndarray,
                              *,
                              scale_days: int = 28,
                              baseline: tuple[int, int] = (1981, 2010),
                              min_window_frac: float = 0.8,
                              clip_esr: tuple[float, float] = (0.0, 1.5)) -> tuple[np.ndarray, np.ndarray]:
    """
    All-in-one convenience wrapper for a single grid:
      (1) ESR daily -> (2) rolling mean -> (3) month-wise standardization

    Returns
    -------
    dates : np.ndarray[datetime64[D]]  # aligned to inputs
    esi   : np.ndarray[float]          # z-scores (NaN where insufficient data)
    """
    dates, years, months = _ym_from_ymd(year, month, day)
    esr_d = esr_daily_1d(et_mm_d, etref_mm_d, clip=clip_esr)
    esr_s = esr_scale_1d(esr_d, scale_days=scale_days, min_window_frac=min_window_frac)
    esi = esi_standardize_by_month_1d(esr_s, years, months, baseline=baseline)
    return dates, esi

