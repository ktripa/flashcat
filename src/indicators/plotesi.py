#####

import numpy as np
import matplotlib.pyplot as plt

# === paths ===
esi_file = r"E:\POSTDOC\FD_predcition\processed_data\esi_weekly_conus_2000_2024.txt"

# === load ===
print("Loading ESI file ...")
data = np.loadtxt(esi_file, delimiter=" ")

# split
years  = data[:,0].astype(int)
months = data[:,1].astype(int)
days   = data[:,2].astype(int)
esi    = data[:,3:]   # (time, grids)

# make a date axis
import datetime as dt
dates = [dt.date(y,m,d) for y,m,d in zip(years, months, days)]

# === plotting first 16 grids ===
fig, axes = plt.subplots(4,4, figsize=(15,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat[:16]):
    ax.plot(dates, esi[:,i], lw=0.7)
    ax.set_title(f"Grid {i+1}", fontsize=9)
    ax.axhline(0, color='k', lw=0.5)   # 0 line for reference

fig.suptitle("ESI Time Series (First 16 Grids)", fontsize=14)
plt.tight_layout()
plt.show()