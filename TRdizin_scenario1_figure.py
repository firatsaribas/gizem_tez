
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------

data = {
    "producer": [
        "Kırkağaç", "Salihli", "Saruhanlı", "Şehzadeler",
        "Ahmetli", "Karacasu", "Bozdoğan", "Kuyucak",
        "Efeler", "Söke", "Sultanhisar", "Menemen",
        "Torbalı", "Ödemiş", "Bergama"
    ],
    "gain": [
        98900, 0, 116200, 49800,
        16000, 174300, 141100, 141100,
        0, 0, 49800, 0,
        0, 16000, 24900
    ]
}

df = pd.DataFrame(data)

# Sort from highest gain to lowest gain
df = df.sort_values("gain", ascending=True)

# ------------------------------------------------------------------
# FIGURE
# ------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11
})

fig, ax = plt.subplots(figsize=(8, 6))

bars = ax.barh(
    df["producer"],
    df["gain"],
    edgecolor="black",
    linewidth=0.8
)

# Value labels
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 2000,
        bar.get_y() + bar.get_height()/2,
        f"{width:,.0f}",
        va="center",
        fontsize=10
    )

ax.set_xlabel("Revenue gain from collaborative selling (TL)")
ax.set_ylabel("Producer")
#ax.set_title(
  #  "Revenue gain obtained through collaborative selling\n"
   # "relative to the no-collaboration benchmark")

ax.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "revenue_gain_collaboration.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
