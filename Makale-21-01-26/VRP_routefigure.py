import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

# ============================================================
# STYLE
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
})

BLUE = "#1657B8"
ORANGE = "#E66100"
GRAY = "#9C9C9C"
LIGHT_GRAY = "#C7C7C7"
BLACK = "#000000"
BORDER = "#6F6F6F"
DIVIDER = "#B7B7B7"

ROUTES = {
    1: {"Vehicle 1": None, "Vehicle 2": ["O2", "F9", "F8", "F1", "F3", "H"]},
    2: {"Vehicle 1": None, "Vehicle 2": ["O2", "F9", "F8", "F1", "F2", "F3", "H"]},
    3: {"Vehicle 1": ["O1", "F6", "F7", "F8", "F1", "H"],
        "Vehicle 2": ["O2", "F9", "F10", "F3", "F2", "H"]},
    4: {"Vehicle 1": ["O1", "F6", "F7", "F8", "F9", "F1", "H"],
        "Vehicle 2": ["O2", "F10", "F5", "F4", "F3", "F2", "H"]},
    5: {"Vehicle 1": ["O1", "F6", "F7", "F8", "F1", "H"],
        "Vehicle 2": ["O2", "F9", "F10", "F3", "F2", "H"]},
    6: {"Vehicle 1": ["O1", "F6", "F7", "F8", "F1", "H"],
        "Vehicle 2": ["O2", "F9", "F10", "F5", "F4", "F3", "F2", "H"]},
}

def lab(node):
    if node == "H":
        return r"$H$"
    return rf"${node[0]}_{{{node[1:]}}}$"

def arrow(ax, x1, y1, x2, y2, color, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.35,
        linestyle=(0, (4, 4)) if dashed else "-",
        color=color,
        shrinkA=0, shrinkB=0,
        zorder=2
    ))

def origin(ax, x, y, node, color):
    ax.scatter([x], [y], transform=ax.transAxes, marker="s",
               s=105, facecolor=color, edgecolor=BLACK,
               linewidth=0.8, zorder=5)
    ax.text(x, y+0.055, lab(node), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10.3, zorder=6)

def farmer(ax, x, y, node, color):
    ax.scatter([x], [y], transform=ax.transAxes, marker="o",
               s=760, facecolor="white", edgecolor=color,
               linewidth=1.25, zorder=5)
    ax.text(x, y, lab(node), transform=ax.transAxes,
            ha="center", va="center", fontsize=10.0, zorder=6)

def hub(ax, x, y, active=True):
    fc = BLACK if active else LIGHT_GRAY
    ax.scatter([x], [y], transform=ax.transAxes, marker="D",
               s=92, facecolor=fc, edgecolor=BLACK,
               linewidth=0.8, zorder=5)
    ax.text(x, y+0.055, r"$H$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10.3,
            color=BLACK if active else "#555555", zorder=6)

def farmer_positions(n):
    if n == 4:
        return np.linspace(0.235, 0.735, n)
    if n == 5:
        return np.linspace(0.190, 0.770, n)
    if n == 6:
        return np.linspace(0.165, 0.795, n)
    return np.linspace(0.22, 0.75, n)

def lane(ax, y, vehicle_name, route, color, inactive=False):
    # vehicle name kept well above O_k label
    ax.text(0.030, y+0.135, vehicle_name, transform=ax.transAxes,
            ha="left", va="center", fontsize=11.2,
            fontweight="bold", color=color)

    ox, hx = 0.052, 0.948

    if inactive:
        origin(ax, ox, y, "O1", BLUE)
        hub(ax, hx, y, active=False)
        arrow(ax, ox+0.035, y, hx-0.040, y, GRAY, dashed=True)
        return

    origin(ax, ox, y, route[0], color)
    hub(ax, hx, y, active=True)

    fs = route[1:-1]
    xs = farmer_positions(len(fs))
    for x, f in zip(xs, fs):
        farmer(ax, float(x), y, f, color)

    arrow(ax, ox+0.030, y, xs[0]-0.040, y, color)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, x1+0.040, y, x2-0.040, y, color)
    arrow(ax, xs[-1]+0.040, y, hx-0.032, y, color)

def panel(fig, rect, period):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(Rectangle((0.004, 0.008), 0.992, 0.984,
                           transform=ax.transAxes, facecolor="white",
                           edgecolor=BORDER, linewidth=0.85))
    ax.text(0.5, 0.925, f"Period {period}", transform=ax.transAxes,
            ha="center", va="center", fontsize=12.5)
    ax.plot([0.008, 0.992], [0.845, 0.845], transform=ax.transAxes,
            color="#808080", linewidth=0.65)
    ax.plot([0.020, 0.980], [0.420, 0.420], transform=ax.transAxes,
            color=DIVIDER, linewidth=0.85, linestyle=(0, (4, 3)))

    if period in (1, 2):
        lane(ax, 0.655, "Vehicle 2", ROUTES[period]["Vehicle 2"], ORANGE)
        lane(ax, 0.195, "Vehicle 1", None, BLUE, inactive=True)
    else:
        lane(ax, 0.655, "Vehicle 1", ROUTES[period]["Vehicle 1"], BLUE)
        lane(ax, 0.195, "Vehicle 2", ROUTES[period]["Vehicle 2"], ORANGE)

def legend_item(ax, x, y, kind, text, color=None):
    if kind == "square":
        ax.scatter([x], [y], transform=ax.transAxes, marker="s", s=115,
                   facecolor=color, edgecolor=BLACK, linewidth=0.8)
        tx = x + 0.035
    elif kind == "circle":
        ax.scatter([x], [y], transform=ax.transAxes, marker="o", s=230,
                   facecolor="white", edgecolor=BLACK, linewidth=1.0)
        tx = x + 0.035
    elif kind == "diamond":
        ax.scatter([x], [y], transform=ax.transAxes, marker="D", s=100,
                   facecolor=BLACK, edgecolor=BLACK)
        tx = x + 0.035
    elif kind == "line":
        ax.plot([x, x+0.06], [y, y], transform=ax.transAxes, color=color, lw=1.7)
        tx = x + 0.075
    elif kind == "dashed":
        ax.plot([x, x+0.06], [y, y], transform=ax.transAxes, color=GRAY,
                lw=1.45, linestyle=(0, (4, 4)))
        ax.add_patch(FancyArrowPatch((x+0.03, y), (x+0.06, y),
                                    transform=ax.transAxes, arrowstyle="-|>",
                                    mutation_scale=10, color=GRAY, linewidth=1.2))
        tx = x + 0.075
    ax.text(tx, y, text, transform=ax.transAxes,
            ha="left", va="center", fontsize=10.7)

def main():
    # Exact 4:3 canvas matching the reference image.
    fig = plt.figure(figsize=(14.48, 10.86), dpi=100, facecolor="white")


    # Group headers
    fig.text(0.5, 0.922, "(a) Periods 1–2",
             ha="center", va="center", fontsize=15.8, fontweight="bold")
    fig.text(0.5, 0.661, "(b) Periods 3–4",
             ha="center", va="center", fontsize=15.8, fontweight="bold")
    fig.text(0.5, 0.413, "(c) Periods 5–6",
             ha="center", va="center", fontsize=15.8, fontweight="bold")

    # Fixed panel rectangles: [left, bottom, width, height]
    left = 0.008
    gap = 0.013
    width = (0.984 - gap) / 2
    right = left + width + gap
    h = 0.215

    panel(fig, [left, 0.692, width, h], 1)
    panel(fig, [right, 0.692, width, h], 2)
    panel(fig, [left, 0.435, width, h], 3)
    panel(fig, [right, 0.435, width, h], 4)
    panel(fig, [left, 0.190, width, h], 5)
    panel(fig, [right, 0.190, width, h], 6)

    # Legend box with manually positioned items.
    lax = fig.add_axes([0.044, 0.082, 0.912, 0.095])
    lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis("off")
    lax.add_patch(Rectangle((0.002, 0.02), 0.996, 0.96,
                            transform=lax.transAxes, facecolor="white",
                            edgecolor=BORDER, linewidth=0.9))

    legend_item(lax, 0.035, 0.66, "square",
                r"Origin of Vehicle 1 ($O_1$)", BLUE)
    legend_item(lax, 0.035, 0.30, "square",
                r"Origin of Vehicle 2 ($O_2$)", ORANGE)

    legend_item(lax, 0.325, 0.66, "circle",
                r"Farmer location ($F_i$)")
    legend_item(lax, 0.325, 0.30, "diamond",
                r"Collection hub ($H$)")

    legend_item(lax, 0.550, 0.66, "line",
                "Vehicle 1 route", BLUE)
    legend_item(lax, 0.550, 0.30, "line",
                "Vehicle 2 route", ORANGE)

    legend_item(lax, 0.795, 0.66, "dashed",
                "Vehicle 1 inactive")
    lax.text(0.870, 0.31, "(not used)", transform=lax.transAxes,
             ha="left", va="center", fontsize=10.7)

    # Notes
    fig.text(
        0.007, 0.053,
        r"$\it{Note:}$ $O_k$ denotes the origin location of Vehicle $k$, $F_i$ denotes Farmer $i$, and $H$ denotes the collection hub. Arrows indicate the visiting sequence. Layout is illustrative and not to scale. ",
        ha="left", va="center", fontsize=11.2
    )


    # Do not use bbox_inches='tight'; it changes the fixed positions.
    fig.savefig("collection_routes_exact.png", dpi=300, facecolor="white", pad_inches=0)
    fig.savefig("collection_routes_exact.pdf", facecolor="white", pad_inches=0)
    fig.savefig("collection_routes_exact.svg", facecolor="white", pad_inches=0)
    plt.show()

if __name__ == "__main__":
    main()