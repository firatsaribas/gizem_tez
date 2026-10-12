"""
FFH Flowchart Generator
=======================
Generates a publication-quality flowchart of the Feasibility-First Heuristic
using matplotlib.  Outputs both PNG and PDF.

Usage:
    python ffh_flowchart.py

Outputs:
    ffh_flowchart.png  (300 dpi)
    ffh_flowchart.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 10, 20          # inches
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Color palette ─────────────────────────────────────────────────────────────
C_BOX    = "#F5F5F2"           # light gray fill for process boxes
C_HEADER = "#CCCCCC"           # darker gray for phase headers
C_BORDER = "#333333"           # box border / arrow color
C_TEXT   = "#1A1A1A"           # main text
C_SUBTEXT= "#444444"           # secondary text inside boxes
FONT     = "DejaVu Sans"

# ── Helper functions ──────────────────────────────────────────────────────────

def box(ax, x, y, w, h, text, fontsize=8.5, bold=False,
        fill=C_BOX, border=C_BORDER, radius=0.15,
        valign="center", lines=None):
    """Draw a rounded rectangle with centred text.
    lines: list of strings for multi-line content (overrides text).
    """
    fancy = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0.8, edgecolor=border, facecolor=fill, zorder=3
    )
    ax.add_patch(fancy)

    if lines:
        n = len(lines)
        spacing = h / (n + 1)
        for i, line in enumerate(lines):
            ty = y + h / 2 - spacing * (i + 1)
            fw = "bold" if (bold and i == 0) else "normal"
            ax.text(x, ty, line, ha="center", va="center",
                    fontsize=fontsize, fontweight=fw,
                    color=C_TEXT, fontfamily=FONT, zorder=4,
                    wrap=False)
    else:
        fw = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va=valign,
                fontsize=fontsize, fontweight=fw,
                color=C_TEXT, fontfamily=FONT, zorder=4)


def stadium(ax, x, y, w, h, text, fontsize=9, bold=True,
            fill=C_BOX, border=C_BORDER):
    """Draw a stadium (pill) shape for START / END."""
    fancy = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={h/2}",
        linewidth=1.0, edgecolor=border, facecolor=fill, zorder=3
    )
    ax.add_patch(fancy)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            color=C_TEXT, fontfamily=FONT, zorder=4)


def diamond(ax, x, y, w, h, lines, fontsize=8):
    """Draw a diamond decision shape."""
    hw, hh = w / 2, h / 2
    pts = np.array([[x, y + hh], [x + hw, y], [x, y - hh], [x - hw, y]])
    poly = plt.Polygon(pts, closed=True, linewidth=0.8,
                       edgecolor=C_BORDER, facecolor=C_BOX, zorder=3)
    ax.add_patch(poly)
    n = len(lines)
    for i, line in enumerate(lines):
        offset = (i - (n - 1) / 2) * (fontsize * 0.018)
        ax.text(x, y - offset, line, ha="center", va="center",
                fontsize=fontsize, color=C_TEXT, fontfamily=FONT, zorder=4)


def header(ax, x, y, w, h, text, fontsize=9):
    """Draw a shaded phase header bar."""
    fancy = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=0.8, edgecolor=C_BORDER, facecolor=C_HEADER, zorder=3
    )
    ax.add_patch(fancy)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=C_TEXT, fontfamily=FONT, zorder=4)


def arr(ax, x1, y1, x2, y2, label=None, label_side="right"):
    """Draw a straight arrow from (x1,y1) to (x2,y2)."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_BORDER,
                                lw=0.9, mutation_scale=10),
                zorder=5)
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx = 0.18 if label_side == "right" else -0.18
        ax.text(mx + dx, my, label, ha="left" if label_side == "right" else "right",
                va="center", fontsize=7.5, color=C_SUBTEXT, fontfamily=FONT, zorder=6)


def polyline(ax, pts, label=None, label_idx=1, label_side="right"):
    """Draw a multi-segment path (list of (x,y) tuples), arrow at end."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=C_BORDER, lw=0.9, zorder=5, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=C_BORDER,
                                lw=0.9, mutation_scale=10),
                zorder=5)
    if label:
        i = label_idx
        mx = (pts[i][0] + pts[i + 1][0]) / 2
        my = (pts[i][1] + pts[i + 1][1]) / 2
        dx = 0.18 if label_side == "right" else -0.18
        ax.text(mx + dx, my, label, ha="left" if label_side == "right" else "right",
                va="center", fontsize=7.5, color=C_SUBTEXT, fontfamily=FONT, zorder=6)


# ── Layout constants ───────────────────────────────────────────────────────────
CX  = 5.0      # main column centre x
BW  = 7.2      # standard box width
BH1 = 0.42     # single-line box height
BH2 = 0.72     # two-line box height
BH3 = 1.00     # three-line box height
PHH = 0.38     # phase header height
DW  = 3.6      # diamond width
DH  = 0.90     # diamond height
GAP = 0.28     # vertical gap between elements

# Top y coordinate (high = top of figure)
Y = FIG_H - 0.6

# ── START ──────────────────────────────────────────────────────────────────────
y_start = Y
stadium(ax, CX, y_start, 2.2, BH1, "START")
Y -= BH1 / 2 + GAP

# ── Precompute REQ ─────────────────────────────────────────────────────────────
arr(ax, CX, y_start - BH1 / 2, CX, Y + BH2 / 2)
y_pre = Y
box(ax, CX, y_pre, BW, BH2, "",
    lines=["Precompute REQ[i,t]  for all i ∈ D,  t ∈ T",
           "Initialise  Q_cum[i,0] = 0,   W_cum[i,0] = 0   for all i ∈ D"])
Y -= BH2 / 2 + GAP

# ── Set t = 1 ──────────────────────────────────────────────────────────────────
arr(ax, CX, y_pre - BH2 / 2, CX, Y + BH1 / 2)
y_t1 = Y
box(ax, CX, y_t1, 2.0, BH1, "Set  t = 1")
Y -= BH1 / 2 + GAP

# ── Diamond: t <= T? ───────────────────────────────────────────────────────────
arr(ax, CX, y_t1 - BH1 / 2, CX, Y + DH / 2)
y_dia = Y - DH / 2
diamond(ax, CX, y_dia, DW, DH,
        ["t  ≤  T ?"])
Y = y_dia - DH / 2 - GAP

# ── No branch → Evaluate Objective → END ──────────────────────────────────────
ex = CX + DW / 2       # right tip of diamond
ey = y_dia
eval_x = 8.2
eval_y = y_dia

polyline(ax, [(ex, ey), (eval_x, eval_y)], label="No  (t > T)", label_idx=0)
box(ax, eval_x, eval_y, 1.5, BH1, "Evaluate\nObjective", fontsize=8,
    lines=["Evaluate", "Objective"])

end_y = eval_y - BH2 / 2 - GAP * 1.5
arr(ax, eval_x, eval_y - BH2 / 2, eval_x, end_y + BH1 / 2)
stadium(ax, eval_x, end_y, 1.5, BH1, "END")

# ── Yes arrow down ─────────────────────────────────────────────────────────────
arr(ax, CX, y_dia - DH / 2, CX, Y + PHH / 2, label="Yes", label_side="right")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE I
# ══════════════════════════════════════════════════════════════════════════════
y_ph1 = Y
header(ax, CX, y_ph1, BW, PHH,
       "Phase I:  Cover — Downstream Route Selection and Loading")
Y -= PHH / 2 + GAP

arr(ax, CX, y_ph1 - PHH / 2, CX, Y + BH2 / 2)
y_qmin = Y
box(ax, CX, y_qmin, BW, BH2, "",
    lines=["Compute  qmin[i] = max(0,  REQ[i,t] + W_cum[i,t−1] − Q_cum[i,t−1])  for all i ∈ D",
           "Identify mandatory depot set   U = { i ∈ D :  qmin[i] > 0 }"])
Y -= BH2 / 2 + GAP

arr(ax, CX, y_qmin - BH2 / 2, CX, Y + BH2 / 2)
y_sc = Y
box(ax, CX, y_sc, BW, BH2, "",
    lines=["Weighted set-cover greedy:  select routes maximising  Σ qmin[i] / λ[r]",
           "Drop-and-swap local search:  remove redundant routes;  replace costly routes"])
Y -= BH2 / 2 + GAP

arr(ax, CX, y_sc - BH2 / 2, CX, Y + BH2 / 2)
y_load = Y
box(ax, CX, y_load, BW, BH2, "",
    lines=["Load each active route in decreasing order of depot need",
           "Derive  Q[b,i,t] = Σ_{r: hub(r)=b} Z[i,r,t]   and   OUT[b] = Σ_i Q[b,i,t]"])
Y -= BH2 / 2 + GAP

# ══════════════════════════════════════════════════════════════════════════════
# PHASE II
# ══════════════════════════════════════════════════════════════════════════════
arr(ax, CX, y_load - BH2 / 2, CX, Y + PHH / 2)
y_ph2 = Y
header(ax, CX, y_ph2, BW, PHH,
       "Phase II:  Fill — Scenario-Decomposed Upstream Assignment")
Y -= PHH / 2 + GAP

arr(ax, CX, y_ph2 - PHH / 2, CX, Y + BH2 / 2)
y_init = Y
box(ax, CX, y_init, BW, BH2, "",
    lines=["Initialise   OUT*[b] = OUT[b]   for all b ∈ B",
           "For each s ∈ S:   Pass 1 (Tier-1 reuse → Tier-2 AUC)   +   Pass 2 (Vogel regret)"])
Y -= BH2 / 2 + GAP

arr(ax, CX, y_init - BH2 / 2, CX, Y + DH / 2)
y_conv = Y - DH / 2
diamond(ax, CX, y_conv, DW + 0.6, DH,
        ["min_s  achieved[b,s]  ≥  OUT*[b]",
         "for all  b ∈ B ?"])
Y = y_conv - DH / 2 - GAP

# No branch → Tighten box → loop back into greedy box
tighten_x = CX + (DW + 0.6) / 2 + 1.0
tighten_y  = y_conv
tighten_bw = 1.4
tighten_bh = 0.62

polyline(ax, [(CX + (DW + 0.6) / 2, y_conv), (tighten_x, tighten_y)],
         label="No", label_idx=0)
box(ax, tighten_x, tighten_y, tighten_bw, tighten_bh, "",
    lines=["Tighten", "OUT*[b]"], fontsize=8)

# Tighten → up → back into greedy box right side
loop_x = tighten_x + tighten_bw / 2 + 0.2
polyline(ax, [
    (tighten_x, tighten_y + tighten_bh / 2),
    (tighten_x, y_init + BH2 / 2 + 0.15),
    (CX + BW / 2,  y_init + BH2 / 2 + 0.15),
    (CX + BW / 2,  y_init + BH2 / 2),
])

# Yes arrow
arr(ax, CX, y_conv - DH / 2, CX, Y + BH2 / 2, label="Yes", label_side="right")

y_scale = Y
box(ax, CX, y_scale, BW, BH2, "",
    lines=["Scale  Z[i,r,t]  proportionally to OUT*[b];   deactivate zero-load routes",
           "Re-run single-scenario greedy for all s ∈ S   →   finalise  Y,  L"])
Y -= BH2 / 2 + GAP

# ══════════════════════════════════════════════════════════════════════════════
# PHASE III
# ══════════════════════════════════════════════════════════════════════════════
arr(ax, CX, y_scale - BH2 / 2, CX, Y + PHH / 2)
y_ph3 = Y
header(ax, CX, y_ph3, BW, PHH,
       "Phase III:  Inventory and Waste State Update")
Y -= PHH / 2 + GAP

arr(ax, CX, y_ph3 - PHH / 2, CX, Y + BH3 / 2)
y_inv = Y
box(ax, CX, y_inv, BW, BH3, "",
    lines=["Update   Q_cum[i,t] = Q_cum[i,t−1] + Σ_b Q[b,i,t]",
           "Compute  W[i,t]   (shelf-life cohort formula, Eq. 4);   Update  W_cum[i,t]",
           "Compute signed inventory   I[i,t] = Q_cum[i,t] − Σ μ[i,a] − W_cum[i,t]"])
Y -= BH3 / 2 + GAP

arr(ax, CX, y_inv - BH3 / 2, CX, Y + BH1 / 2)
y_inc = Y
box(ax, CX, y_inc, 2.0, BH1, "t  =  t + 1")
Y -= BH1 / 2 + GAP

# Loop back to t<=T diamond  (left side polyline)
loop_left_x = CX - BW / 2 - 0.35
polyline(ax, [
    (CX - 2.0 / 2, y_inc),          # left edge of t=t+1 box
    (loop_left_x,  y_inc),           # far left
    (loop_left_x,  y_dia),           # up to diamond level
    (CX - DW / 2,  y_dia),           # into left tip of diamond
])

# ── Caption ────────────────────────────────────────────────────────────────────
ax.text(CX, 0.25,
        "Fig. X.  Flowchart of the Feasibility-First Heuristic (FFH).",
        ha="center", va="center", fontsize=8, fontstyle="italic",
        color="#666666", fontfamily=FONT)

# ── Save ───────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.3)
plt.savefig(r"C:\Users\gizem\OneDrive\Belgeler\GitHub\gizem_tez\Makale-21-01-26\ffh_flowchart.png", dpi=300,
            bbox_inches="tight", facecolor="white")
plt.savefig(r"C:\Users\gizem\OneDrive\Belgeler\GitHub\gizem_tez\Makale-21-01-26\ffh_flowchart.pdf",
            bbox_inches="tight", facecolor="white")
print("Saved: ffh_flowchart.png and ffh_flowchart.pdf")
plt.close()