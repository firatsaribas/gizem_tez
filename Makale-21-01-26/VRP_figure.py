import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle

routes = {
    1: {"V1": ["O1", "F5", "Hub"], "V2": ["O2", "F6", "F5", "Hub"]},
    2: {"V1": ["O1", "F1", "F6", "Hub"], "V2": ["O2", "F6", "F5", "Hub"]},
    3: {"V1": ["O1", "F7", "Hub"], "V2": ["O2", "F6", "F7", "Hub"]},
    4: {"V1": ["O1", "F5", "Hub"], "V2": ["O2", "F6", "F5", "Hub"]},
    5: {"V1": ["O1", "F7", "Hub"], "V2": ["O2", "F6", "F7", "Hub"]},
    6: {"V1": ["O1", "F5", "Hub"], "V2": ["O2", "F6", "F5", "Hub"]},
}

period_pos = {
    1: {"O1":(0.22,0.92),"O2":(0.78,0.92),"F6":(0.78,0.68),"F5":(0.50,0.50),"Hub":(0.50,0.26)},
    2: {"O1":(0.22,0.92),"O2":(0.78,0.92),"F1":(0.22,0.68),"F5":(0.78,0.68),"F6":(0.50,0.50),"Hub":(0.50,0.26)},
    3: {"O1":(0.22,0.92),"O2":(0.78,0.92),"F6":(0.78,0.68),"F7":(0.50,0.50),"Hub":(0.50,0.26)},
    4: {"O1":(0.22,0.92),"O2":(0.78,0.92),"F6":(0.78,0.68),"F5":(0.50,0.50),"Hub":(0.50,0.26)},
    5: {"O1":(0.22,0.92),"O2":(0.78,0.92),"F6":(0.78,0.68),"F7":(0.50,0.50),"Hub":(0.50,0.26)},
    6: {"O1":(0.22,0.92),"O2":(0.78,0.92),"F6":(0.78,0.68),"F5":(0.50,0.50),"Hub":(0.50,0.26)},
}

NODE_STYLES = {
    "origin":  dict(boxstyle="round,pad=0.40", facecolor="#E6F1FB", edgecolor="#185FA5", linewidth=1.2),
    "origin2": dict(boxstyle="round,pad=0.40", facecolor="#EEEDFE", edgecolor="#534AB7", linewidth=1.2),
    "farmer":  dict(boxstyle="round,pad=0.40", facecolor="#E1F5EE", edgecolor="#1D9E75", linewidth=1.2),
    "hub":     dict(boxstyle="round,pad=0.40", facecolor="#FAEEDA", edgecolor="#BA7517", linewidth=1.2),
}
TEXT_COLOURS = {"origin":"#0C447C","origin2":"#3C3489","farmer":"#085041","hub":"#633806"}
V1_COLOUR = "#185FA5"; V2_COLOUR = "#534AB7"

def node_style(n):
    if n=="O1":  return "origin","origin"
    if n=="O2":  return "origin2","origin2"
    if n=="Hub": return "hub","hub"
    return "farmer","farmer"

def draw_node(ax, name, pos, fs=10):
    x,y=pos; sk,tk=node_style(name)
    ax.text(x,y,name,ha="center",va="center",fontsize=fs,fontweight="bold",
            color=TEXT_COLOURS[tk],bbox=NODE_STYLES[sk],zorder=5,transform=ax.transAxes)

def draw_arrow(ax, p0, p1, colour, dashed=False, offset=0.0):
    x0,y0=p0; x1,y1=p1
    dx,dy=x1-x0,y1-y0; L=(dx**2+dy**2)**0.5
    if L>0:
        nx,ny=-dy/L,dx/L
        x0+=nx*offset; y0+=ny*offset; x1+=nx*offset; y1+=ny*offset
    rad=0.10 if offset!=0 else 0.05
    ls=(0,(6,3)) if dashed else "-"
    ax.annotate("",xy=(x1,y1),xytext=(x0,y0),
                xycoords="axes fraction",textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->,head_width=0.26,head_length=0.18",
                                color=colour,lw=1.8,linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"),zorder=3)

def draw_route(ax, seq, pos, colour, dashed=False, offset=0.0):
    for i in range(len(seq)-1):
        draw_arrow(ax,pos[seq[i]],pos[seq[i+1]],colour,dashed=dashed,offset=offset)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
fig.subplots_adjust(hspace=0.06, wspace=0.04,
                    left=0.02, right=0.98, top=0.98, bottom=0.055)

for period, ax in zip(range(1,7), axes.flat):
    pos = period_pos[period]
    r   = routes[period]

    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal"); ax.axis("off")

    ax.add_patch(mpatches.FancyBboxPatch((0.01,0.01),0.98,0.98,
                 boxstyle="round,pad=0.01",linewidth=0.9,
                 edgecolor="#AAAAAA",facecolor="#FAFAFA",
                 transform=ax.transAxes,zorder=0))

    ax.text(0.50,0.97,f"Period {period}",ha="center",va="top",
            fontsize=12,fontweight="bold",color="#333333",
            transform=ax.transAxes,zorder=6)

    draw_route(ax, r["V2"], pos, V2_COLOUR, dashed=True,  offset=+0.014)
    draw_route(ax, r["V1"], pos, V1_COLOUR, dashed=False, offset=-0.014)

    for name, xy in pos.items():
        draw_node(ax, name, xy, fs=10)

    v1_text = "V1:  " + "  →  ".join(r["V1"])
    v2_text = "V2:  " + "  →  ".join(r["V2"])

    ax.text(0.50,0.165,v1_text,ha="center",va="top",fontsize=8.5,
            color=V1_COLOUR,transform=ax.transAxes,fontweight="semibold")
    ax.text(0.50,0.105,v2_text,ha="center",va="top",fontsize=8.5,
            color=V2_COLOUR,transform=ax.transAxes,fontweight="semibold")

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor="#E6F1FB", edgecolor="#185FA5", linewidth=1.2, label="Vehicle 1 origin (O1)"),
    mpatches.Patch(facecolor="#EEEDFE", edgecolor="#534AB7", linewidth=1.2, label="Vehicle 2 origin (O2)"),
    mpatches.Patch(facecolor="#E1F5EE", edgecolor="#1D9E75", linewidth=1.2, label="Farmer node"),
    mpatches.Patch(facecolor="#FAEEDA", edgecolor="#BA7517", linewidth=1.2, label="Hub (İzmir)"),
    plt.Line2D([0],[0], color=V1_COLOUR, linewidth=2.0, label="Vehicle 1 route"),
    plt.Line2D([0],[0], color=V2_COLOUR, linewidth=2.0, linestyle=(0,(6,3)), label="Vehicle 2 route"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=9,
           frameon=True, framealpha=0.9, edgecolor="#CCCCCC",
           bbox_to_anchor=(0.50, 0.002))


fig.savefig("routes_all_periods.pdf", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("routes_all_periods.png", dpi=200, bbox_inches="tight", facecolor="white")
print("Saved.")
plt.close(fig)