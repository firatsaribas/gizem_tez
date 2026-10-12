"""
plot_routes_horizontal.py
=========================
Generates a publication-quality figure showing optimal vehicle routes
for each period in a horizontal strip layout.

Each period occupies one row with two horizontal strips:
  - Top strip  : Vehicle 1 route (solid blue)
  - Bottom strip: Vehicle 2 route (dashed purple)

Nodes are ordered left to right: Origin → Stop 1 → Stop 2 → ... → Hub

Output: routes_horizontal.pdf  and  routes_horizontal.png

Usage:
    python plot_routes_horizontal.py

Requirements:
    pip install matplotlib
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# 1.  ROUTE DATA  — update these if the solution changes
# =============================================================================

routes = {
    1: {"V1": ["O1", "F8", "F10", "F3", "F7", "F5", "Hub"],
        "V2": ["O2", "F6", "F1",  "F9", "F2", "F4", "Hub"]},
    2: {"V1": ["O1", "F8", "F10", "F3", "F7", "F5", "Hub"],
        "V2": ["O2", "F6", "F1",  "F9", "F2", "F4", "Hub"]},
    3: {"V1": ["O1", "F4", "F8",  "F10","F7", "F5", "Hub"],
        "V2": ["O2", "F6", "F1",  "F9", "F2", "F4", "Hub"]},
    4: {"V1": ["O1", "F4", "F9",  "F6", "F5", "Hub"],
        "V2": ["O2", "F1", "F3",  "F7", "F10","Hub"]},
    5: {"V1": ["O1", "F4", "F2",  "F8", "F5", "Hub"],
        "V2": ["O2", "F6", "F1",  "F9", "F2", "Hub"]},
    6: {"V1": ["O1", "F4", "F9",  "F2", "F8", "F5", "Hub"],
        "V2": ["O2", "F6", "F1",  "F3", "F7", "Hub"]},
}

# =============================================================================
# 2.  VISUAL SETTINGS
# =============================================================================

V1_COLOUR  = "#185FA5"   # solid blue   — Vehicle 1
V2_COLOUR  = "#534AB7"   # dashed purple — Vehicle 2

NODE_STYLES = {
    "O1":  dict(boxstyle="round,pad=0.38", facecolor="#E6F1FB",
                edgecolor="#185FA5", linewidth=1.2),
    "O2":  dict(boxstyle="round,pad=0.38", facecolor="#EEEDFE",
                edgecolor="#534AB7", linewidth=1.2),
    "Hub": dict(boxstyle="round,pad=0.38", facecolor="#FAEEDA",
                edgecolor="#BA7517", linewidth=1.2),
    "F":   dict(boxstyle="round,pad=0.38", facecolor="#E1F5EE",
                edgecolor="#1D9E75", linewidth=1.2),
}
TEXT_COLOURS = {
    "O1":  "#0C447C",
    "O2":  "#3C3489",
    "Hub": "#633806",
    "F":   "#085041",
}

def get_style(name):
    if name in ("O1", "O2", "Hub"):
        return NODE_STYLES[name], TEXT_COLOURS[name]
    return NODE_STYLES["F"], TEXT_COLOURS["F"]


# =============================================================================
# 3.  DRAWING HELPERS
# =============================================================================

def draw_node(ax, name, x, y, fontsize=9):
    bbox, tc = get_style(name)
    ax.text(x, y, name, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=tc, bbox=bbox, zorder=5,
            transform=ax.transAxes)


def draw_arrow(ax, x0, x1, y, colour, dashed=False, lw=1.8):
    ls = (0, (6, 3)) if dashed else "-"
    ax.annotate(
        "",
        xy=(x1, y), xytext=(x0, y),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->,head_width=0.22,head_length=0.012",
            color=colour,
            lw=lw,
            linestyle=ls,
        ),
        zorder=3,
    )


def draw_strip(ax, seq, y_center, colour, dashed=False,
               x_start=0.07, x_end=0.97, node_fs=9, lw=1.8):
    """Draw one horizontal route strip inside an axes."""
    n = len(seq)
    # Evenly space nodes; leave padding at each end for node boxes
    padding = 0.025
    xs = [x_start + padding + (x_end - x_start - 2 * padding)
          * i / (n - 1) for i in range(n)]

    # Node half-width in axes fraction (approx) — for arrow trimming
    hw = 0.030

    # Draw arrows between consecutive nodes
    for i in range(n - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1] - hw, y_center),
            xytext=(xs[i] + hw, y_center),
            xycoords="axes fraction",
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="->,head_width=0.22,head_length=0.010",
                color=colour,
                lw=lw,
                linestyle=(0, (6, 3)) if dashed else "-",
            ),
            zorder=3,
        )

    # Draw nodes
    for i, name in enumerate(seq):
        draw_node(ax, name, xs[i], y_center, fontsize=node_fs)


# =============================================================================
# 4.  FIGURE
# =============================================================================

n_periods = 6
fig, axes = plt.subplots(n_periods, 1, figsize=(14, 10))
fig.subplots_adjust(
    hspace=0.08,
    left=0.05, right=0.99,
    top=0.99, bottom=0.07,
)

for period, ax in zip(range(1, n_periods + 1), axes):

    r = routes[period]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Panel background
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0.02), 1.0, 0.96,
        boxstyle="square,pad=0",
        linewidth=0.7, edgecolor="#CCCCCC",
        facecolor="#FAFAFA",
        transform=ax.transAxes, zorder=0,
    ))

    # Period label on the left
    ax.text(0.022, 0.5, f"t = {period}",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="#333333",
            transform=ax.transAxes)

    # V1 label
    ax.text(0.052, 0.75, "V1",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=V1_COLOUR,
            transform=ax.transAxes)

    # V2 label
    ax.text(0.052, 0.25, "V2",
            ha="center", va="center",
            fontsize=9, fontweight="bold", color=V2_COLOUR,
            transform=ax.transAxes)

    # Thin divider between V1 and V2
    ax.axhline(y=0.50, xmin=0.06, xmax=1.0,
               color="#E0E0E0", linewidth=0.6, zorder=1)

    # Draw routes
    draw_strip(ax, r["V1"], y_center=0.75, colour=V1_COLOUR,
               dashed=False, x_start=0.07, x_end=0.98, node_fs=9, lw=1.9)
    draw_strip(ax, r["V2"], y_center=0.25, colour=V2_COLOUR,
               dashed=True,  x_start=0.07, x_end=0.98, node_fs=9, lw=1.9)

# =============================================================================
# 5.  LEGEND
# =============================================================================

legend_elements = [
    mpatches.Patch(facecolor="#E6F1FB", edgecolor="#185FA5",
                   linewidth=1.1, label="Vehicle 1 origin (O1)"),
    mpatches.Patch(facecolor="#EEEDFE", edgecolor="#534AB7",
                   linewidth=1.1, label="Vehicle 2 origin (O2)"),
    mpatches.Patch(facecolor="#E1F5EE", edgecolor="#1D9E75",
                   linewidth=1.1, label="Farmer node"),
    mpatches.Patch(facecolor="#FAEEDA", edgecolor="#BA7517",
                   linewidth=1.1, label="Hub (İzmir)"),
    plt.Line2D([0], [0], color=V1_COLOUR, linewidth=2.0,
               label="Vehicle 1 route"),
    plt.Line2D([0], [0], color=V2_COLOUR, linewidth=2.0,
               linestyle=(0, (6, 3)), label="Vehicle 2 route"),
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=6,
    fontsize=9.5,
    frameon=True,
    framealpha=0.95,
    edgecolor="#CCCCCC",
    bbox_to_anchor=(0.52, 0.005),
)

# =============================================================================
# 6.  SAVE
# =============================================================================

fig.savefig("routes_horizontal.pdf", dpi=300,
            bbox_inches="tight", facecolor="white")
fig.savefig("routes_horizontal.png", dpi=200,
            bbox_inches="tight", facecolor="white")
print("Saved: routes_horizontal.pdf")
print("Saved: routes_horizontal.png")
plt.close(fig)