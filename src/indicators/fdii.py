"""
Flash Drought Intensification Index (FDII)
1. FD_intensity
2. Drought_severity

Example usage:
# v: your weekly ESI/EDDI/SM z-series (one grid)
# years: same length as v (ints)
base_mask = (years >= 2000) & (years <= 2015)
fd_int, dro_sev, fdii = fdii_weekly(v, base_mask)

"""

import numpy as np

def fdii_weekly(v, base_mask, dt_range=(2,3,4,5,6,7),
                dro_base=-0.85, base_drop=-0.85, base_dt=3, nw=13):
    """
    v: weekly standardized anomalies (z), shape (T,)
    base_mask: True for baseline weeks (e.g., 2000–2015), shape (T,)
    returns: fd_int, dro_sev, fdii  (each shape (T,))
    """
    T = len(v); max_dt = max(dt_range); R_base = base_drop / base_dt  # (~ -0.283)
    fd_int = np.full(T, np.nan); dro_sev = np.full(T, np.nan); fdii = np.full(T, np.nan)

    # precompute diffs and baseline mean/std per ΔT
    diffs = {dt: (v - np.roll(v, dt)).astype(float) for dt in dt_range}
    for dt in dt_range: diffs[dt][:dt] = np.nan  # invalid lead-in
    mu  = {dt: np.nanmean(diffs[dt][base_mask]) for dt in dt_range}
    sig = {dt: (np.nanstd(diffs[dt][base_mask], ddof=1) or 1.0) for dt in dt_range}

    # main loop over valid junction weeks (enough history + 13w forward)
    for t in range(max_dt, T - nw):
        # FD-INT: best (most negative) standardized rate over dt_range
        Rbest = None
        for dt in dt_range:
            d = diffs[dt][t]
            if np.isfinite(d):
                zc = (d - mu[dt]) / (sig[dt] if sig[dt] > 0 else 1.0)
                R  = zc / dt
                if (Rbest is None) or (R < Rbest):  # more negative is "faster"
                    Rbest = R
        if Rbest is None:
            continue
        fd_int[t] = Rbest / R_base if Rbest <= R_base else 0.0

        # DRO-SEV: next 13 weeks, with ≥4 consecutive weeks ≤ dro_base
        fwin = v[t+1:t+1+nw]
        below = (fwin <= dro_base).astype(int)
        has_run4 = np.any(np.convolve(below, np.ones(4, int), 'valid') >= 4)
        if has_run4:
            contrib = np.maximum(0.0, dro_base - fwin)  # only weeks at/under threshold
            dro_sev[t] = 1.0 + np.nanmean(contrib)
        else:
            dro_sev[t] = 0.0

        fdii[t] = fd_int[t] * dro_sev[t]

    return fd_int, dro_sev, fdii
