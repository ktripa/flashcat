import numpy as np
import pandas as pd
from scipy.special import ndtri  # inverse normal CDF

# ==== FILE PATHS ====
et_file    = r"E:\POSTDOC\FD_predcition\processed_data\era5_day_evaporation_conus_2000_2024.txt"
etref_file = r"E:\POSTDOC\FD_predcition\processed_data\era5_day_potential_evaporation_conus_2000_2024.txt"
out_file   = r"E:\POSTDOC\FD_predcition\processed_data\esi_weekly_conus_2000_2024.txt"

# ==== PARAMETERS ====
window_days = 7
clip_min, clip_max = 0.0, 100.0
baseline_start, baseline_end = 2000, 2015

# ==== LOAD DATA ====
print("Loading ET...")
et = np.loadtxt(et_file, delimiter=",")
print("Loading ETref...")
etref = np.loadtxt(etref_file, delimiter=",")

# First 3 cols are dates
years  = et[:,0].astype(int)
months = et[:,1].astype(int)
days   = et[:,2].astype(int)

# Grid data
et_vals    = et[:,3:]
etref_vals = etref[:,3:]

print("Shape:", et_vals.shape)

# ==== CALCULATE ESR ====
etref_vals = np.clip(etref_vals, 1e-6, None)
esr = et_vals / etref_vals
esr = np.clip(esr, clip_min, clip_max)

# ==== 7-day rolling mean ====
df = pd.DataFrame(esr)
esr_smooth = df.rolling(window=window_days, min_periods=6).mean().to_numpy()

# ==== Convert to z-scores month-wise ====
def monthwise_ecdf_to_z(values, base_values):
    out = np.full(values.shape, np.nan, dtype=np.float32)
    b = base_values[np.isfinite(base_values)]
    if b.size < 10:
        return out
    b = np.sort(b)
    n = b.size
    y = (np.arange(1, n+1)) / (n+1.0)
    good = np.isfinite(values)
    if np.any(good):
        eps = 1e-6
        p = np.interp(values[good], b, y, left=y[0]+eps, right=y[-1]-eps)
        out[good] = ndtri(p)
    return out

esi = np.full_like(esr_smooth, np.nan, dtype=np.float32)

baseline_mask = (years >= baseline_start) & (years <= baseline_end)

print("Standardizing month-wise ...")
for m in range(1,13):
    idx_all  = (months == m)
    idx_base = idx_all & baseline_mask
    if not np.any(idx_all) or not np.any(idx_base):
        continue
    vals_all  = esr_smooth[idx_all,:]
    vals_base = esr_smooth[idx_base,:]
    for g in range(esr_smooth.shape[1]):
        esi[idx_all, g] = monthwise_ecdf_to_z(vals_all[:,g], vals_base[:,g])

# ==== SAVE OUTPUT ====
out = np.column_stack([years, months, days, esi])
np.savetxt(out_file, out, fmt="%.6f")
print("Saved:", out_file)


