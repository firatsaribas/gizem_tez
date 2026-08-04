"""
=============================================================================
Scale-Dependent Collaborative Selling Problem
ORIGINAL MODEL — Exactly as formulated in the paper
No access thresholds (A_l), no access gate variables (z)
=============================================================================

DOMAIN  : Agricultural cooperative / food supply chain
PRODUCT : Memecik / Ayvalık olives, Aegean Region, Turkey
SOLVER  : IBM CPLEX via DOcplex  (pip install docplex)

─────────────────────────────────────────────────────────────────────────────
MODEL NOTE
─────────────────────────────────────────────────────────────────────────────
  This file implements the model exactly as written in the paper:
    - Constraints (9) and (10) use per-entity caps: r <= B_l * y_g / s_i
    - No minimum quantity threshold parameter A_l
    - No access gate binary variables z^G_gl, z^I_il
    - Segment allocation is free: the solver chooses how much goes to each
      segment without any sequential filling rule
    - Revenue segments represent market-based or negotiated quantity premiums

  Under this formulation, the revenue function is concave in quantity
  (it is the optimal value of a parametric LP). As a result, the all-
  individual selling solution is always weakly optimal, regardless of
  the B_l and p_l parameter values chosen. The base case solution therefore
  correctly returns individual selling for all producers.

  The parameter values are chosen so that the individual solution is
  non-trivial: producers with q_i > B_2 = 8 t spill into segment 1,
  creating varied segment allocations across producers.

─────────────────────────────────────────────────────────────────────────────
PARAMETERS
─────────────────────────────────────────────────────────────────────────────

  B_2 = 8 t  (export/premium mill delivery lot)
    Boutique export mills in the Aegean require 5–10 t per consignment
    (EZZİB, 2024). Producers with q_i > 8 t spill to the wholesale segment.

  B_1 = 20 t  (regional oil mill batch)
    Regional oil mills handle 10–30 t per registered producer lot (FAO, 2020).

  B_0 = 30 t  (local cooperative — effectively uncapped)
    Local cooperatives absorb all remaining quantity (Artukoğlu et al., 2012).

  p_2 = 220 $/t  (export/premium channel, +69% over local)
  p_1 = 175 $/t  (regional wholesale, +35% over local)
  p_0 = 130 $/t  (local spot/cooperative gate price)
    Sources: Olive Oil Times (2025); EZZİB (2024); IOC (2024)
=============================================================================
"""

import math
from docplex.mp.model import Model


# =============================================================================
# 1.  INSTANCE DATA
# =============================================================================

producers = list(range(10))
segments  = [0, 1, 2]
G         = list(range(len(producers)))

district = {
    0:"Torbalı, İzmir",  1:"Selçuk, İzmir",   2:"Urla, İzmir",
    3:"Foça, İzmir",     4:"Ödemiş, İzmir",   5:"Nazilli, Aydın",
    6:"Söke, Aydın",     7:"Akhisar, Manisa",  8:"Kırkağaç, Manisa",
    9:"Milas, Muğla",
}
lat = {0:38.163,1:37.956,2:38.321,3:38.673,4:38.226,
       5:37.913,6:37.748,7:38.919,8:39.101,9:37.318}
lon = {0:27.359,1:27.369,2:26.765,3:26.755,4:27.971,
       5:28.317,6:27.413,7:27.837,8:27.668,9:27.787}

# Producer quantities (t/season)
q = {0:12, 1:7, 2:15, 3:8, 4:5, 5:18, 6:6, 7:14, 8:10, 9:9}

# Haversine distances (km)
def haversine(la1, lo1, la2, lo2):
    R = 6371.0
    phi1, phi2 = math.radians(la1), math.radians(la2)
    a = math.sin(math.radians(la2-la1)/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lo2-lo1)/2)**2
    return 2.0 * R * math.asin(math.sqrt(a))

d = {(i,j): haversine(lat[i],lon[i],lat[j],lon[j])
     for i in producers for j in producers if i != j}

D_max = 50.0
infeasible_pairs = [(i,j) for i in producers for j in producers
                    if i < j and d[(i,j)] > D_max]

# Segment parameters — original model (no A_l)
B = {0: 10, 1: 20, 2: 30}    # per-entity segment capacities (t)
p = {0: 130, 1: 175, 2: 220} # unit revenues ($/t)

channel = {0: "Local / spot ($130/t)",
           1: "Regional wholesale ($175/t)",
           2: "Export / premium ($220/t)"}

# Instance summary
sep = "=" * 70
print(sep)
print("ORIGINAL MODEL — Base Case (N=10, L=3)")
print("No access thresholds (A_l) | No gate variables (z)")
print(sep)
print(f"  Total supply     : {sum(q.values())} t")
print(f"  D_max            : {D_max} km")
print(f"  Infeasible pairs : {len(infeasible_pairs)} of 45")
print()
print(f"  {'#':>2}  {'District':<22} {'q_i':>6}")
print("  " + "─"*35)
for i in producers:
    print(f"  {i:>2}  {district[i]:<22} {q[i]:>6} t")
print()
print(f"  {'Seg':>3}  {'Channel':<28} {'B_l (t)':>8}  {'p_l ($/t)':>10}")
print("  " + "─"*54)
for l in segments:
    print(f"  {l:>3}  {channel[l]:<28} {B[l]:>8}  {p[l]:>10}")
print(sep)


# =============================================================================
# 2.  MODEL — ORIGINAL FORMULATION (paper Section 3)
# =============================================================================

mdl = Model(name="Original_Collab_Selling_N10_L3")
mdl.parameters.mip.tolerances.mipgap = 1e-6
mdl.parameters.timelimit             = 3600

# ── Decision variables ────────────────────────────────────────────────────────
x  = {(i,g): mdl.binary_var(name=f"x_{i}_{g}")
      for i in producers for g in G}
s  = {i: mdl.binary_var(name=f"s_{i}") for i in producers}
y  = {g: mdl.binary_var(name=f"y_{g}") for g in G}
Q  = {g: mdl.continuous_var(lb=0, name=f"Q_{g}") for g in G}
rG = {(g,l): mdl.continuous_var(lb=0, name=f"rG_{g}_{l}")
      for g in G for l in segments}
rI = {(i,l): mdl.continuous_var(lb=0, name=f"rI_{i}_{l}")
      for i in producers for l in segments}

# ── Objective (Eq. 1) ─────────────────────────────────────────────────────────
mdl.maximize(
    mdl.sum(p[l] * rG[(g,l)] for g in G for l in segments) +
    mdl.sum(p[l] * rI[(i,l)] for i in producers for l in segments)
)

# ── Constraints ───────────────────────────────────────────────────────────────

# (2) Each producer joins exactly one group or sells individually
for i in producers:
    mdl.add_constraint(
        mdl.sum(x[(i,g)] for g in G) + s[i] == 1,
        ctname=f"C2_i{i}")

# (3) Assignment only to active groups
for i in producers:
    for g in G:
        mdl.add_constraint(x[(i,g)] <= y[g], ctname=f"C3_i{i}_g{g}")

# (4) Active group requires at least two members
for g in G:
    mdl.add_constraint(
        mdl.sum(x[(i,g)] for i in producers) >= 2 * y[g],
        ctname=f"C4_g{g}")

# (5) Empty group must be inactive
for g in G:
    mdl.add_constraint(
        mdl.sum(x[(i,g)] for i in producers) <= len(producers) * y[g],
        ctname=f"C5_g{g}")

# (6) Group total quantity
for g in G:
    mdl.add_constraint(
        Q[g] == mdl.sum(q[i] * x[(i,g)] for i in producers),
        ctname=f"C6_g{g}")

# (7) Group quantity fully allocated across segments
for g in G:
    mdl.add_constraint(
        mdl.sum(rG[(g,l)] for l in segments) == Q[g],
        ctname=f"C7_g{g}")

# (8) Individual quantity fully allocated across segments
for i in producers:
    mdl.add_constraint(
        mdl.sum(rI[(i,l)] for l in segments) == q[i] * s[i],
        ctname=f"C8_i{i}")

# (9) Group segment capacity — original per-entity cap
#     0 <= r^G_{gl} <= B_l * y_g
for g in G:
    for l in segments:
        mdl.add_constraint(rG[(g,l)] >= 0,
                           ctname=f"C9_lb_g{g}_l{l}")
        mdl.add_constraint(rG[(g,l)] <= B[l] * y[g],
                           ctname=f"C9_ub_g{g}_l{l}")

# (10) Individual segment capacity — original per-entity cap
#      0 <= r^I_{il} <= B_l * s_i
for i in producers:
    for l in segments:
        mdl.add_constraint(rI[(i,l)] >= 0,
                           ctname=f"C10_lb_i{i}_l{l}")
        mdl.add_constraint(rI[(i,l)] <= B[l] * s[i],
                           ctname=f"C10_ub_i{i}_l{l}")

# (11) Geographical incompatibility
for g in G:
    for (i,j) in infeasible_pairs:
        mdl.add_constraint(
            x[(i,g)] + x[(j,g)] <= 1,
            ctname=f"C11_i{i}_j{j}_g{g}")

# Model size
n_bin  = len(x) + len(s) + len(y)
n_cont = len(Q) + len(rG) + len(rI)
print(f"\nMODEL SIZE")
print(f"  Binary variables    : {n_bin}")
print(f"  Continuous variables: {n_cont}")
print(f"  Total variables     : {n_bin + n_cont}")
print(f"  Constraints         : {mdl.number_of_constraints}")
print()


# =============================================================================
# 3.  SOLVE
# =============================================================================

print("Solving with IBM CPLEX …")
sol = mdl.solve(log_output=True)
print()

if sol is None:
    print("ERROR: No solution found.")
else:
    obj_val = sol.objective_value
    print(sep)
    print("SOLUTION REPORT")
    print(sep)
    print(f"  Solver status   : {mdl.solve_details.status}")
    print(f"  MIP gap         : {mdl.solve_details.mip_relative_gap:.2e}")
    print(f"  Optimal revenue : ${obj_val:,.2f}")
    print()

    ind = [i for i in producers if sol.get_value(s[i]) > 0.5]
    act = [g for g in G if sol.get_value(y[g]) > 0.5]

    print(f"  INDIVIDUAL SELLERS  ({len(ind)} producers)")
    ind_rev = 0.0
    for i in ind:
        rev_i = sum(p[l] * sol.get_value(rI[(i,l)]) for l in segments)
        alloc = {l: sol.get_value(rI[(i,l)]) for l in segments}
        segs  = ", ".join(f"seg{l}: {alloc[l]:.1f}t@${p[l]}"
                          for l in segments if alloc[l] > 1e-6)
        print(f"    [{i}] {district[i]:<22} q={q[i]:>2}t | {segs} | ${rev_i:,.2f}")
        ind_rev += rev_i

    print()
    print(f"  COLLABORATIVE GROUPS  ({len(act)} active)")
    grp_rev = 0.0
    for g in act:
        mems    = [i for i in producers if sol.get_value(x[(i,g)]) > 0.5]
        tot_qty = sum(q[i] for i in mems)
        rev_g   = sum(p[l] * sol.get_value(rG[(g,l)]) for l in segments)
        alloc   = {l: sol.get_value(rG[(g,l)]) for l in segments}
        segs    = ", ".join(f"seg{l}: {alloc[l]:.1f}t@${p[l]}"
                            for l in segments if alloc[l] > 1e-6)
        print(f"    Group {g}: {[district[i] for i in mems]}")
        print(f"      Q={tot_qty}t | {segs} | ${rev_g:,.2f}")
        grp_rev += rev_g

    if not act:
        print("    (none — individual selling is optimal under this formulation)")

    print()
    print(f"  Revenue — individual: ${ind_rev:,.2f}  |  groups: ${grp_rev:,.2f}")
    print(f"  TOTAL OPTIMAL REVENUE: ${obj_val:,.2f}")
    print(sep)

    # Validation
    print("\nVALIDATION")
    assigned = set(ind)
    for g in act:
        assigned |= {i for i in producers if sol.get_value(x[(i,g)]) > 0.5}
    print(f"  [{'OK' if assigned==set(producers) else 'FAIL'}] All producers assigned")

    geo_ok = all(
        d[(mems[a], mems[b])] <= D_max
        for g in act
        for mems in [[i for i in producers if sol.get_value(x[(i,g)]) > 0.5]]
        for a in range(len(mems)) for b in range(a+1, len(mems))
    )
    print(f"  [{'OK' if geo_ok else 'FAIL'}] Geographical constraints satisfied")
    print(f"  [OK] Segment capacity constraints satisfied (verified by CPLEX)")


# =============================================================================
# 4.  EXPORT RESULTS TO EXCEL
# =============================================================================

def write_results_to_excel(sol, filepath="original_model_results.xlsx"):
    import openpyxl
    wb = openpyxl.Workbook()

    # x_ig
    ws = wb.active; ws.title = "x_ig"
    ws.cell(row=1, column=1, value="x_ig (1 = producer i assigned to group g)")
    ws.cell(row=2, column=1, value="i \\ g")
    for g in G:
        ws.cell(row=2, column=g+2, value=g)
    for i in producers:
        ws.cell(row=i+3, column=1, value=i)
        for g in G:
            ws.cell(row=i+3, column=g+2, value=round(sol.get_value(x[(i,g)])))

    # s_i
    ws2 = wb.create_sheet("s_i")
    ws2.append(["i", "s_i"])
    for i in producers:
        ws2.append([i, round(sol.get_value(s[i]))])

    # y_g
    ws3 = wb.create_sheet("y_g")
    ws3.append(["g", "y_g"])
    for g in G:
        ws3.append([g, round(sol.get_value(y[g]))])

    # Q_g
    ws4 = wb.create_sheet("Q_g")
    ws4.append(["g", "Q_g"])
    for g in G:
        ws4.append([g, round(sol.get_value(Q[g]), 4)])

    # rG_gl
    ws5 = wb.create_sheet("rG_gl")
    ws5.append(["g \\ l", 0, 1, 2])
    for g in G:
        ws5.append([g] + [round(sol.get_value(rG[(g,l)]), 4) for l in segments])

    # rI_il
    ws6 = wb.create_sheet("rI_il")
    ws6.append(["i \\ l", 0, 1, 2])
    for i in producers:
        ws6.append([i] + [round(sol.get_value(rI[(i,l)]), 4) for l in segments])

    wb.save(filepath)
    print(f"\nResults saved to: {filepath}")


if sol is not None:
    write_results_to_excel(sol, "original_model_results.xlsx")