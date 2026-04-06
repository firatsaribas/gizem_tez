import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
'''
# -------------------------------------------------
# 1. CSV OKU
# -------------------------------------------------
df = pd.read_csv("vehicle_load.csv", sep=";", decimal=",")
df.columns = df.columns.str.strip()

# Kolonları standardize et
df = df.rename(columns={"Base Case": "Base"})

# Tip dönüşümleri
df["Vehicle"] = pd.to_numeric(df["Vehicle"], errors="coerce").astype(int)
df["Week"]    = pd.to_numeric(df["Week"], errors="coerce").astype(int)
df["Base"]    = pd.to_numeric(df["Base"], errors="coerce")
df["S2"]      = pd.to_numeric(df["S2"], errors="coerce")

df = df.dropna(subset=["Vehicle", "Week", "Base", "S2"])

# -------------------------------------------------
# 2. LOAD CHANGE
# -------------------------------------------------
df["LoadChange"] = df["S2"] - df["Base"]

# -------------------------------------------------
# 3. PIVOT
# -------------------------------------------------
heatmap_data = df.pivot(
    index="Vehicle",
    columns="Week",
    values="LoadChange"
).sort_index()

# -------------------------------------------------
# 4. RENK NORMALIZASYONU (0 merkezli)
# -------------------------------------------------
vmin = heatmap_data.min().min()
vmax = heatmap_data.max().max()

norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# -------------------------------------------------
# 5. HEATMAP
# -------------------------------------------------
plt.figure(figsize=(8, 6))

im = plt.imshow(
    heatmap_data,
    aspect="auto",
    cmap="RdBu",
    norm=norm
)

plt.colorbar(im, label="Load Change (S1 scenario − Base case)")

plt.xticks(
    ticks=np.arange(len(heatmap_data.columns)),
    labels=[f"Week {w}" for w in heatmap_data.columns]
)

plt.yticks(
    ticks=np.arange(len(heatmap_data.index)),
    labels=[f" {v}" for v in heatmap_data.index]
)

plt.xlabel("Time Period")
plt.ylabel("Vehicle")
#plt.title("Temporal Redistribution of Vehicle Loads under Scenario S2 Relative to the Base Case")

plt.tight_layout()
plt.show()
'''


# Time bazında utilization change (%) listeleri
time_data = {
    "Time 1": [-17.62, 24.44, -28.92, -94.37, -56.35, -66.32, 11.62, -16.64, 0.00, 4.70, -34.26, -21.04, -101.76, -150.07, -17.91, -70.27, -67.27, -31.91, 0.00, -144.62, -46.22, -9.33, -17.67, -98.51, -129.30],

    "Time 2": [-15.68, 32.71, 0.00, -17.49, -22.14, -84.37, 8.93, -109.59, -26.62, -38.62, -24.09, -56.03, 0.00, -116.71, -16.62, -35.41, -85.35, -17.07, 0.00, -66.48, -125.89, -4.70, 0.00, -44.32, -70.83],

    "Time 3": [-18.69, 0.48, 0.00, 0.00, -17.44, -17.48, 0.00, -67.02, -52.52, -85.23, -71.53, -32.79, -16.02, -180.40, 0.00, -72.06, -69.53, -179.07, 22.48, -16.27, -23.48, -61.83, 0.00, -80.00, -118.53],

    "Time 4": [-16.18, 0.00, 0.00, -13.21, 0.00, -79.68, 8.19, -55.48, -52.47, 0.00, -25.66, -28.24, 0.00, -107.68, -16.04, 0.89, -17.02, -4.21, -16.27, -87.61, -34.12, -11.44, -16.24, 0.00, -35.80],

    "Time 5": [36.61, 0.00, -39.30, -37.83, -24.95, -104.38, -31.55, 16.27, 47.66, 0.00, -25.73, -21.76, -15.95, -157.55, -16.84, -17.84, -35.69, -34.38, -53.60, 0.00, -34.96, -31.63, -17.02, -87.71, -32.73],

    "Time 6": [-16.85, 0.00, 26.24, -46.65, -22.75, -53.30, 0.00, -15.41, -28.71, -33.96, -22.54, -24.73, -51.93, -88.48, -33.68, -51.09, 0.00, -67.53, 0.00, -50.07, -35.37, -28.36, 0.00, -50.36, -70.99]
}
# Boxplot
plt.figure(figsize=(10, 5))
plt.boxplot(time_data.values(), labels=time_data.keys(), showfliers=True)
plt.axhline(0)  # reference line

plt.ylabel("Vehicle Utilization Change (%)")
plt.xlabel("Time Period")
#plt.title("Distribution of Vehicle Utilization Changes (Base Case vs. S3)")

plt.tight_layout()
plt.show()
