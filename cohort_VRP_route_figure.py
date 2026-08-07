'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 3 Araçlı Güncel Rotalar ───────────────────────────────────────────────────
routes = {
    1: {
        "V1": None,
        "V2": ["O2", "F9", "F1", "Hub"],
        "V3": ["O3", "F3", "F2", "Hub"],
    },
    2: {
        "V1": ["O1", "F6", "F7", "F8", "F1", "Hub"],
        "V2": ["O2", "F9", "Hub"],
        "V3": ["O3", "F3", "F2", "Hub"],
    },
    3: {
        "V1": ["O1", "F6", "F7", "F8", "F9", "Hub"],
        "V2": ["O2", "F4", "F3", "Hub"],
        "V3": ["O3", "F3", "F2", "F1", "Hub"],
    },
    4: {
        "V1": ["O1", "F6", "F3", "F4", "F5", "Hub"],
        "V2": ["O2", "F9", "F8", "F1", "Hub"],
        "V3": ["O3", "F3", "F2", "Hub"],
    },
    5: {
        "V1": ["O1", "F6", "F7", "F8", "F9", "F1", "Hub"],
        "V2": ["O2", "F9", "F2", "Hub"],
        "V3": ["O3", "F3", "F4", "Hub"],
    },
    6: {
        "V1": ["O1", "F6", "F7", "F8", "F9", "Hub"],
        "V2": ["O2", "F9", "F1", "Hub"],
        "V3": ["O3", "F4", "F5", "F3", "F2", "Hub"],
    },
}

# ── Periyot Bazlı Düğüm Koordinatları (O1, O2, O3 y=0.88 seviyesine çekildi) ──
period_pos = {
    1: {
        "O1": (0.18, 0.88), "O2": (0.50, 0.88), "O3": (0.82, 0.88),
        "F9": (0.50, 0.68), "F1": (0.50, 0.48),
        "F3": (0.82, 0.68), "F2": (0.82, 0.48),
        "Hub": (0.50, 0.20)
    },
    2: {
        "O1": (0.18, 0.88), "O2": (0.50, 0.88), "O3": (0.82, 0.88),
        "F6": (0.18, 0.72), "F7": (0.18, 0.58), "F8": (0.32, 0.45), "F1": (0.32, 0.32),
        "F9": (0.50, 0.55),
        "F3": (0.82, 0.70), "F2": (0.82, 0.48),
        "Hub": (0.50, 0.18)
    },
    3: {
        "O1": (0.18, 0.88), "O2": (0.50, 0.88), "O3": (0.82, 0.88),
        "F6": (0.18, 0.74), "F7": (0.18, 0.60), "F8": (0.18, 0.46), "F9": (0.32, 0.33),
        "F4": (0.50, 0.68),
        "F3": (0.65, 0.52), "F2": (0.82, 0.62), "F1": (0.75, 0.35),
        "Hub": (0.50, 0.18)
    },
    4: {
        "O1": (0.18, 0.88), "O2": (0.50, 0.88), "O3": (0.82, 0.88),
        "F6": (0.18, 0.74), "F4": (0.32, 0.48), "F5": (0.42, 0.34),
        "F9": (0.50, 0.74), "F8": (0.50, 0.58), "F1": (0.50, 0.42),
        "F3": (0.72, 0.64), "F2": (0.82, 0.48),
        "Hub": (0.50, 0.18)
    },
    5: {
        "O1": (0.18, 0.88), "O2": (0.50, 0.88), "O3": (0.82, 0.88),
        "F6": (0.18, 0.74), "F7": (0.18, 0.60), "F8": (0.18, 0.46), "F9": (0.32, 0.33),
        "F2": (0.50, 0.55), "F1": (0.42, 0.34),
        "F3": (0.82, 0.68), "F4": (0.82, 0.48),
        "Hub": (0.50, 0.18)
    },
    6: {
        "O1": (0.18, 0.88), "O2": (0.50, 0.88), "O3": (0.82, 0.88),
        "F6": (0.18, 0.74), "F7": (0.18, 0.60), "F8": (0.18, 0.46),
        "F9": (0.35, 0.55), "F1": (0.35, 0.35),
        "F4": (0.82, 0.74), "F5": (0.82, 0.60), "F3": (0.82, 0.46), "F2": (0.68, 0.33),
        "Hub": (0.50, 0.18)
    }
}

# ── Görsel Stiller ──────────────────────────────────────────────────────────────
NODE_STYLES = {
    "origin":  dict(boxstyle="round,pad=0.35", facecolor="#E6F1FB", edgecolor="#185FA5", linewidth=1.2),
    "origin2": dict(boxstyle="round,pad=0.35", facecolor="#EEEDFE", edgecolor="#534AB7", linewidth=1.2),
    "origin3": dict(boxstyle="round,pad=0.35", facecolor="#FAECE7", edgecolor="#D9534F", linewidth=1.2),
    "farmer":  dict(boxstyle="round,pad=0.35", facecolor="#E1F5EE", edgecolor="#1D9E75", linewidth=1.2),
    "hub":     dict(boxstyle="round,pad=0.35", facecolor="#FAEEDA", edgecolor="#BA7517", linewidth=1.2),
}
TEXT_COLOURS = {
    "origin": "#0C447C",
    "origin2": "#3C3489",
    "origin3": "#A94442",
    "farmer": "#085041",
    "hub": "#633806"
}

V1_COLOUR = "#185FA5"  # Mavi
V2_COLOUR = "#534AB7"  # Mor
V3_COLOUR = "#D9534F"  # Kırmızı/Turuncu

def node_style(n):
    if n == "O1":  return "origin", "origin"
    if n == "O2":  return "origin2", "origin2"
    if n == "O3":  return "origin3", "origin3"
    if n in ["Hub", "H"]: return "hub", "hub"
    return "farmer", "farmer"

def draw_node(ax, name, pos, fs=8.5):
    x, y = pos
    sk, tk = node_style(name)
    display_name = "Hub" if name == "H" else name
    ax.text(x, y, display_name, ha="center", va="center", fontsize=fs, fontweight="bold",
            color=TEXT_COLOURS[tk], bbox=NODE_STYLES[sk], zorder=5, transform=ax.transAxes)

def draw_arrow(ax, p0, p1, colour, linestyle="-", offset=0.0):
    x0, y0 = p0; x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0; L = (dx**2 + dy**2)**0.5
    if L > 0:
        nx, ny = -dy/L, dx/L
        x0 += nx*offset; y0 += ny*offset; x1 += nx*offset; y1 += ny*offset
    rad = 0.08 if offset != 0 else 0.03
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->,head_width=0.20,head_length=0.14",
                                color=colour, lw=1.5, linestyle=linestyle,
                                connectionstyle=f"arc3,rad={rad}"), zorder=3)

def draw_route(ax, seq, pos, colour, linestyle="-", offset=0.0):
    if not seq or len(seq) < 2:
        return
    for i in range(len(seq) - 1):
        if seq[i] in pos and seq[i+1] in pos:
            draw_arrow(ax, pos[seq[i]], pos[seq[i+1]], colour, linestyle=linestyle, offset=offset)

# ── Figure Oluşturma ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
fig.subplots_adjust(hspace=0.08, wspace=0.04,
                    left=0.02, right=0.98, top=0.98, bottom=0.08)

for period, ax in zip(range(1, 7), axes.flat):
    pos = period_pos[period]
    r   = routes[period]

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")

    ax.add_patch(mpatches.FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                 boxstyle="round,pad=0.01", linewidth=0.9,
                 edgecolor="#AAAAAA", facecolor="#FAFAFA",
                 transform=ax.transAxes, zorder=0))

    # Başlık çerçevenin en üstüne taşındı (y=0.985)
    ax.text(0.50, 0.985, f"Period {period}", ha="center", va="top",
            fontsize=11.5, fontweight="bold", color="#333333",
            transform=ax.transAxes, zorder=6)

    # Rotaları Çizdir
    if r.get("V3"):
        draw_route(ax, r["V3"], pos, V3_COLOUR, linestyle=":",   offset=+0.016)
    if r.get("V2"):
        draw_route(ax, r["V2"], pos, V2_COLOUR, linestyle="--",  offset=+0.000)
    if r.get("V1"):
        draw_route(ax, r["V1"], pos, V1_COLOUR, linestyle="-",   offset=-0.016)

    # Düğümleri Çizdir
    for name, xy in pos.items():
        draw_node(ax, name, xy, fs=8.0)

    # Alt Bilgi Metinleri (V1, V2, V3)
    v1_str = "  →  ".join(r["V1"]) if r.get("V1") else "No active route"
    v2_str = "  →  ".join(r["V2"]) if r.get("V2") else "No active route"
    v3_str = "  →  ".join(r["V3"]) if r.get("V3") else "No active route"

    ax.text(0.50, 0.13, f"V1: {v1_str}", ha="center", va="top", fontsize=7.5,
            color=V1_COLOUR if r.get("V1") else "#888888", transform=ax.transAxes, fontweight="semibold")
    ax.text(0.50, 0.08, f"V2: {v2_str}", ha="center", va="top", fontsize=7.5,
            color=V2_COLOUR if r.get("V2") else "#888888", transform=ax.transAxes, fontweight="semibold")
    ax.text(0.50, 0.03, f"V3: {v3_str}", ha="center", va="top", fontsize=7.5,
            color=V3_COLOUR if r.get("V3") else "#888888", transform=ax.transAxes, fontweight="semibold")

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor="#E6F1FB", edgecolor="#185FA5", linewidth=1.2, label="Vehicle 1 origin (O1)"),
    mpatches.Patch(facecolor="#EEEDFE", edgecolor="#534AB7", linewidth=1.2, label="Vehicle 2 origin (O2)"),
    mpatches.Patch(facecolor="#FAECE7", edgecolor="#D9534F", linewidth=1.2, label="Vehicle 3 origin (O3)"),
    mpatches.Patch(facecolor="#E1F5EE", edgecolor="#1D9E75", linewidth=1.2, label="Farmer node"),
    mpatches.Patch(facecolor="#FAEEDA", edgecolor="#BA7517", linewidth=1.2, label="Hub"),
    plt.Line2D([0], [0], color=V1_COLOUR, linewidth=1.8, linestyle="-", label="Vehicle 1 route"),
    plt.Line2D([0], [0], color=V2_COLOUR, linewidth=1.8, linestyle="--", label="Vehicle 2 route"),
    plt.Line2D([0], [0], color=V3_COLOUR, linewidth=1.8, linestyle=":", label="Vehicle 3 route"),
]

fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8.5,
           frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
           bbox_to_anchor=(0.50, 0.002))

fig.savefig("routes_3vehicles_network.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("routes_3vehicles_network.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved successfully.")
plt.close(fig)

'''
#her period line seklinde

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

BLUE = "#1657B8"      # Vehicle 1
ORANGE = "#E66100"    # Vehicle 2
GREEN = "#008856"     # Vehicle 3
GRAY = "#9C9C9C"
LIGHT_GRAY = "#C7C7C7"
BLACK = "#000000"
BORDER = "#6F6F6F"
DIVIDER = "#B7B7B7"

# --- 3 ARAÇLI GÜNCEL ÇÖZÜM ---
ROUTES = {
    1: {
        "Vehicle 1": None,
        "Vehicle 2": ["O2", "F9", "F1", "H"],
        "Vehicle 3": ["O3", "F3", "F2", "H"],
    },
    2: {
        "Vehicle 1": ["O1", "F6", "F7", "F8", "F1", "H"],
        "Vehicle 2": ["O2", "F9", "H"],
        "Vehicle 3": ["O3", "F3", "F2", "H"],
    },
    3: {
        "Vehicle 1": ["O1", "F6", "F7", "F8", "F9", "H"],
        "Vehicle 2": ["O2", "F4", "F3", "H"],
        "Vehicle 3": ["O3", "F3", "F2", "F1", "H"],
    },
    4: {
        "Vehicle 1": ["O1", "F6", "F3", "F4", "F5", "H"],
        "Vehicle 2": ["O2", "F9", "F8", "F1", "H"],
        "Vehicle 3": ["O3", "F3", "F2", "H"],
    },
    5: {
        "Vehicle 1": ["O1", "F6", "F7", "F8", "F9", "F1", "H"],
        "Vehicle 2": ["O2", "F9", "F2", "H"],
        "Vehicle 3": ["O3", "F3", "F4", "H"],
    },
    6: {
        "Vehicle 1": ["O1", "F6", "F7", "F8", "F9", "H"],
        "Vehicle 2": ["O2", "F9", "F1", "H"],
        "Vehicle 3": ["O3", "F4", "F5", "F3", "F2", "H"],
    },
}

def lab(node):
    if node in ["H", "F10"]:  
        return r"$H$"
    return rf"${node[0]}_{{{node[1:]}}}$"

def arrow(ax, x1, y1, x2, y2, color, dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.15,
        linestyle=(0, (4, 4)) if dashed else "-",
        color=color,
        shrinkA=0, shrinkB=0,
        zorder=2
    ))

def origin(ax, x, y, node, color):
    ax.scatter([x], [y], transform=ax.transAxes, marker="s",
               s=75, facecolor=color, edgecolor=BLACK,
               linewidth=0.8, zorder=5)
    # O_1, O_2, O_3 etiketini karenin üstüne çakışmayacak yükseklikte basıyoruz
    ax.text(x, y+0.075, lab(node), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.0, zorder=6)

def farmer(ax, x, y, node, color):
    if node == "F10":
        ax.scatter([x], [y], transform=ax.transAxes, marker="D",
                   s=80, facecolor=BLACK, edgecolor=BLACK,
                   linewidth=0.8, zorder=5)
    else:
        ax.scatter([x], [y], transform=ax.transAxes, marker="o",
                   s=420, facecolor="white", edgecolor=color,
                   linewidth=1.1, zorder=5)
    ax.text(x, y, lab(node), transform=ax.transAxes,
            ha="center", va="center", fontsize=7.5, 
            color="white" if node=="F10" else BLACK, zorder=6)

def hub(ax, x, y, active=True):
    fc = BLACK if active else LIGHT_GRAY
    ax.scatter([x], [y], transform=ax.transAxes, marker="D",
               s=75, facecolor=fc, edgecolor=BLACK,
               linewidth=0.8, zorder=5)
    ax.text(x, y+0.075, r"$H$", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.0,
            color=BLACK if active else "#555555", zorder=6)

def farmer_positions(n):
    if n == 1:
        return [0.55]
    return list(np.linspace(0.25, 0.84, n))

def lane(ax, y, vehicle_name, route, color, default_origin="O1"):
    # "Vehicle 1" yazısını en sola (x=0.015), ortalanmış yukarısına basıyoruz
    ax.text(0.015, y+0.125, vehicle_name, transform=ax.transAxes,
            ha="left", va="center", fontsize=8.5,
            fontweight="bold", color=color)

    # Orijin karesinin x konumunu 0.085 yaparak metinle çakışmasını engelliyoruz
    ox, hx = 0.085, 0.940

    # Eğer araç pasifse
    if not route:
        origin(ax, ox, y, default_origin, color)
        hub(ax, hx, y, active=False)
        arrow(ax, ox+0.030, y, hx-0.035, y, GRAY, dashed=True)
        return

    # Orijin ve Varış Hub'ını çiz
    origin(ax, ox, y, route[0], color)
    hub(ax, hx, y, active=True)

    fs = route[1:-1]
    xs = farmer_positions(len(fs))
    
    gap = 0.020 if len(fs) > 5 else 0.028

    for x, f in zip(xs, fs):
        farmer(ax, float(x), y, f, color)

    # Okları çiz
    arrow(ax, ox+0.025, y, xs[0]-gap, y, color)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, x1+gap, y, x2-gap, y, color)
    arrow(ax, xs[-1]+gap, y, hx-0.028, y, color)

def panel(fig, rect, period):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.add_patch(Rectangle((0.004, 0.008), 0.992, 0.984,
                           transform=ax.transAxes, facecolor="white",
                           edgecolor=BORDER, linewidth=0.85))
    ax.text(0.5, 0.940, f"Period {period}", transform=ax.transAxes,
            ha="center", va="center", fontsize=11.0, fontweight="bold")
    ax.plot([0.008, 0.992], [0.870, 0.870], transform=ax.transAxes,
            color="#808080", linewidth=0.65)
    
    # 3 Şerit için 2 Ayırıcı Çizgi
    ax.plot([0.020, 0.980], [0.580, 0.580], transform=ax.transAxes,
            color=DIVIDER, linewidth=0.75, linestyle=(0, (4, 3)))
    ax.plot([0.020, 0.980], [0.310, 0.310], transform=ax.transAxes,
            color=DIVIDER, linewidth=0.75, linestyle=(0, (4, 3)))

    # ÜST ŞERİT (y=0.680): Vehicle 1 (MAVİ)
    lane(ax, 0.680, "Vehicle 1", ROUTES[period]["Vehicle 1"], BLUE, default_origin="O1")
    
    # ORTA ŞERİT (y=0.410): Vehicle 2 (TURUNCU)
    lane(ax, 0.410, "Vehicle 2", ROUTES[period]["Vehicle 2"], ORANGE, default_origin="O2")

    # ALT ŞERİT (y=0.140): Vehicle 3 (YEŞİL)
    lane(ax, 0.140, "Vehicle 3", ROUTES[period]["Vehicle 3"], GREEN, default_origin="O3")

def legend_item(ax, x, y, kind, text, color=None):
    if kind == "square":
        ax.scatter([x], [y], transform=ax.transAxes, marker="s", s=75,
                   facecolor=color, edgecolor=BLACK, linewidth=0.8)
        tx = x + 0.025
    elif kind == "circle":
        ax.scatter([x], [y], transform=ax.transAxes, marker="o", s=140,
                   facecolor="white", edgecolor=BLACK, linewidth=1.0)
        tx = x + 0.025
    elif kind == "diamond":
        ax.scatter([x], [y], transform=ax.transAxes, marker="D", s=75,
                   facecolor=BLACK, edgecolor=BLACK)
        tx = x + 0.025
    elif kind == "line":
        ax.plot([x, x+0.04], [y, y], transform=ax.transAxes, color=color, lw=1.7)
        tx = x + 0.050
    elif kind == "dashed":
        ax.plot([x, x+0.04], [y, y], transform=ax.transAxes, color=GRAY,
                lw=1.45, linestyle=(0, (4, 4)))
        ax.add_patch(FancyArrowPatch((x+0.02, y), (x+0.04, y),
                                    transform=ax.transAxes, arrowstyle="-|>",
                                    mutation_scale=8, color=GRAY, linewidth=1.1))
        tx = x + 0.050
    ax.text(tx, y, text, transform=ax.transAxes,
            ha="left", va="center", fontsize=9.0)

def main():
    fig = plt.figure(figsize=(14.5, 11.5), dpi=100, facecolor="white")

    # Group headers
    fig.text(0.5, 0.932, "(a) Periods 1–2", ha="center", va="center", fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.671, "(b) Periods 3–4", ha="center", va="center", fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.418, "(c) Periods 5–6", ha="center", va="center", fontsize=13.5, fontweight="bold")

    left = 0.008
    gap = 0.013
    width = (0.984 - gap) / 2
    right = left + width + gap
    h = 0.235

    panel(fig, [left, 0.685, width, h], 1)
    panel(fig, [right, 0.685, width, h], 2)
    panel(fig, [left, 0.430, width, h], 3)
    panel(fig, [right, 0.430, width, h], 4)
    panel(fig, [left, 0.175, width, h], 5)
    panel(fig, [right, 0.175, width, h], 6)

    # Legend box
    lax = fig.add_axes([0.020, 0.065, 0.960, 0.085])
    lax.set_xlim(0, 1); lax.axis("off")
    lax.add_patch(Rectangle((0.002, 0.02), 0.996, 0.96,
                            transform=lax.transAxes, facecolor="white",
                            edgecolor=BORDER, linewidth=0.9))

    # Column 1: Origins
    legend_item(lax, 0.015, 0.72, "square", r"Origin Vehicle 1 ($O_1$)", BLUE)
    legend_item(lax, 0.015, 0.40, "square", r"Origin Vehicle 2 ($O_2$)", ORANGE)
    legend_item(lax, 0.015, 0.08, "square", r"Origin Vehicle 3 ($O_3$)", GREEN)

    # Column 2: Nodes
    legend_item(lax, 0.230, 0.65, "circle", r"Farmer location ($F_i$)")
    legend_item(lax, 0.230, 0.25, "diamond", r"Collection hub ($H$)")

    # Column 3: Routes
    legend_item(lax, 0.480, 0.72, "line", "Vehicle 1 route", BLUE)
    legend_item(lax, 0.480, 0.40, "line", "Vehicle 2 route", ORANGE)
    legend_item(lax, 0.480, 0.08, "line", "Vehicle 3 route", GREEN)

    # Column 4: Inactive
    legend_item(lax, 0.760, 0.55, "dashed", "Vehicle inactive")
    lax.text(0.825, 0.22, "(not used)", transform=lax.transAxes,
             ha="left", va="center", fontsize=9.0)

    # Notes
    fig.text(
        0.007, 0.038,
        r"$\it{Note:}$ $O_k$ denotes the origin location of Vehicle $k$, $F_i$ denotes Farmer $i$, and $H$ denotes the collection hub. Arrows indicate the visiting sequence. Layout is illustrative and not to scale.",
        ha="left", va="center", fontsize=9.8
    )

    fig.savefig("routes_3vehicles_linear.png", dpi=300, facecolor="white", pad_inches=0)
    fig.savefig("routes_3vehicles_linear.pdf", facecolor="white", pad_inches=0)
    plt.show()

if __name__ == "__main__":
    main()
