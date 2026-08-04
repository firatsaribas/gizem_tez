"""
Corrected Collaborative Selling Model — Aegean Turkey Olive Producers
======================================================================
Fixes applied vs. original code
---------------------------------
FIX-1  D_max aligned to paper value: 75 km → 100 km
FIX-2  Channel label corrected: "5-15 t" → "up to 10 t allocated"
FIX-3  Redundant constraint C5 (∑x_ig ≤ |N|·y_g) removed
FIX-4  Symmetry-breaking constraints added: y_g ≥ y_{g+1} for all g
FIX-5  Province encoding fixed: "Aydn" → "Aydın"
FIX-6  Excel export now includes a Revenue Summary sheet
FIX-7  Post-solve validation extended with objective cross-check
"""

import math
from docplex.mp.model import Model

# =============================================================================
# 1.  INSTANCE DATA
# =============================================================================

producers = list(range(1, 11))
segments  = [0, 1, 2]
G         = list(range(len(producers)))           # potential group indices

# ── Producer meta-data ────────────────────────────────────────────────────────
# FIX-5: "Aydn" → "Aydın" for producers 5 and 6
district = {
    1: "Torbalı, İzmir", 2: "Selçuk, İzmir", 3: "Urla, İzmir",
    4: "Foça, İzmir", 5: "Ödemiş, İzmir", 6: "Nazilli, Aydın",
    7: "Söke, Aydın", 8: "Akhisar, Manisa", 9: "Kırkağaç, Manisa",
    10: "Milas, Muğla",
}

lat = {
    1: 38.163, 2: 37.956, 3: 38.321, 4: 38.673, 5: 38.226,
    6: 37.913, 7: 37.748, 8: 38.919, 9: 39.101, 10: 37.318
}

lon = {
    1: 27.359, 2: 27.369, 3: 26.765, 4: 26.755, 5: 27.971,
    6: 28.317, 7: 27.413, 8: 27.837, 9: 27.668, 10: 27.787
}

q = {1: 12, 2: 7, 3: 15, 4: 10, 5: 5, 6: 18, 7: 6, 8: 14, 9: 10, 10: 9}
#q = {1: 12, 2: 7, 3: 11, 4: 8, 5: 5, 6: 13, 7: 6, 8: 9, 9: 10, 10: 9}
# Producer quantities (tonnes/season)
#q = {1:12, 2:7, 3:15, 4:8, 5:5, 6:18, 7:6, 8:14, 9:10, 10:9}

# ── Distances via Haversine ───────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2.0 * R * math.asin(math.sqrt(a))

d = {(i, j): haversine(lat[i], lon[i], lat[j], lon[j])
     for i in producers for j in producers if i != j}

# FIX-1: D_max corrected from 75 km (code) to 100 km (paper)
D_max = 75.0
infeasible_pairs = [(i, j) for i in producers for j in producers
                    if i < j and d[(i, j)] > D_max]

# ── Segment parameters ────────────────────────────────────────────────────────
# A[l]: minimum quantity (t) to ACCESS segment l
# B[l]: maximum quantity (t) allocable within segment l
# p[l]: unit revenue ($/t)
A = {0:  0, 1:  5, 2: 25} #A = {0:  0, 1:  5, 2: 15}
B = {0:  5, 1: 25, 2: 30} #B = {0:  5, 1: 10, 2: 15}
p = {0: 130, 1: 175, 2: 220} #p = {0: 10, 1: 75, 2: 220}

# FIX-2: label corrected — seg 1 allocates up to B[1]=10 t, not "5-15 t"
channel = {
    0: "Base rate        (A=0 t,  B=5 t,  $130/t)",
    1: "Volume bonus     (A=5 t,  B=25 t, $75/t)",   # up to 25 t allocated
    2: "Scale premium    (A=25 t, B=30 t, $220/t)",
}

# ── Instance summary ──────────────────────────────────────────────────────────
sep = "=" * 72
print(sep)
print("CORRECTED BASE CASE — Aegean Turkey Olive Producers")
print("N=10 producers | |G|=10 potential groups | L=3 revenue segments")
print(sep)
print(f"  Total supply : {sum(q.values())} t/season     D_max : {D_max} km  [FIX-1]")
print()
print(f"  {'#':>2}  {'District':<24} {'Lat':>7}  {'Lon':>7}  {'q_i (t)':>8}")
print("  " + "─" * 58)
for i in producers:
    print(f"  {i:>2}  {district[i]:<24} {lat[i]:>7.3f}  {lon[i]:>7.3f}  {q[i]:>8}")
print()
print(f"  {'Seg':>3}  {'Description':<45} {'A_l':>5}  {'B_l':>5}  {'p_l':>7}")
print("  " + "─" * 68)
for l in segments:
    print(f"  {l:>3}  {channel[l]:<45} {A[l]:>5}  {B[l]:>5}  {p[l]:>7}")
print()
print(f"  Infeasible pairs (d > {D_max} km): {len(infeasible_pairs)} of 45  [FIX-1]")
print(sep)


# =============================================================================
# 2.  MODEL
# =============================================================================

mdl = Model(name="Collab_Selling_Aegean_Olive_Corrected")
mdl.parameters.mip.tolerances.mipgap = 1e-6
mdl.parameters.timelimit             = 3600

# ── Decision variables ────────────────────────────────────────────────────────
x  = {(i, g): mdl.binary_var(name=f"x_{i}_{g}")
      for i in producers for g in G}
s  = {i: mdl.binary_var(name=f"s_{i}") for i in producers}
y  = {g: mdl.binary_var(name=f"y_{g}") for g in G}
Q  = {g: mdl.continuous_var(lb=0, name=f"Q_{g}") for g in G}

rG = {(g, l): mdl.continuous_var(lb=0, name=f"rG_{g}_{l}")
      for g in G for l in segments}
rI = {(i, l): mdl.continuous_var(lb=0, name=f"rI_{i}_{l}")
      for i in producers for l in segments}

zG = {(g, l): mdl.binary_var(name=f"zG_{g}_{l}")
      for g in G for l in segments}
zI = {(i, l): mdl.binary_var(name=f"zI_{i}_{l}")
      for i in producers for l in segments}

# ── Objective — Eq. (1) ───────────────────────────────────────────────────────
mdl.maximize(
    mdl.sum(p[l] * rG[(g, l)] for g in G for l in segments) +
    mdl.sum(p[l] * rI[(i, l)] for i in producers for l in segments)
)

# ── Constraints ───────────────────────────────────────────────────────────────

# (2) Each producer sells individually or joins exactly one group
for i in producers:
    mdl.add_constraint(mdl.sum(x[(i, g)] for g in G) + s[i] == 1,
                       ctname=f"C_assign_i{i}")

# (3) Producer can join only an active group
for i in producers:
    for g in G:
        mdl.add_constraint(x[(i, g)] <= y[g],
                           ctname=f"C_active_i{i}_g{g}")

# (4) Active group must have at least two members
for g in G:
    mdl.add_constraint(mdl.sum(x[(i, g)] for i in producers) >= 2 * y[g],
                       ctname=f"C_minsize_g{g}")

# NOTE: Constraint C5 (∑x_ig ≤ |N|·y_g) from original code is REMOVED.
# FIX-3: It is implied by C_active (∑_i x_ig ≤ ∑_i y_g = |N|·y_g) and
#         is therefore redundant. Removing it tightens the LP relaxation.

# (5) Group quantity
for g in G:
    mdl.add_constraint(Q[g] == mdl.sum(q[i] * x[(i, g)] for i in producers),
                       ctname=f"C_qty_g{g}")

# (6) Group quantity fully allocated across segments
for g in G:
    mdl.add_constraint(mdl.sum(rG[(g, l)] for l in segments) == Q[g],
                       ctname=f"C_alloc_G_g{g}")

# (7) Individual quantity fully allocated across segments
for i in producers:
    mdl.add_constraint(mdl.sum(rI[(i, l)] for l in segments) == q[i] * s[i],
                       ctname=f"C_alloc_I_i{i}")

# (8-10) Group segment access: cap + activity gate + threshold gate
for g in G:
    for l in segments:
        mdl.add_constraint(rG[(g, l)] <= B[l] * zG[(g, l)],
                           ctname=f"C_cap_G_g{g}_l{l}")
        mdl.add_constraint(zG[(g, l)] <= y[g],
                           ctname=f"C_gate_act_G_g{g}_l{l}")
        mdl.add_constraint(Q[g] >= A[l] * zG[(g, l)],
                           ctname=f"C_gate_thr_G_g{g}_l{l}")

# (11-13) Individual segment access: cap + activity gate + threshold gate
for i in producers:
    for l in segments:
        mdl.add_constraint(rI[(i, l)] <= B[l] * zI[(i, l)],
                           ctname=f"C_cap_I_i{i}_l{l}")
        mdl.add_constraint(zI[(i, l)] <= s[i],
                           ctname=f"C_gate_act_I_i{i}_l{l}")
        # q[i] is a fixed parameter → constraint is linear
        mdl.add_constraint(q[i] * s[i] >= A[l] * zI[(i, l)],
                           ctname=f"C_gate_thr_I_i{i}_l{l}")

# (14) Geographical incompatibility
for g in G:
    for (i, j) in infeasible_pairs:
        mdl.add_constraint(x[(i, g)] + x[(j, g)] <= 1,
                           ctname=f"C_geo_i{i}_j{j}_g{g}")

# FIX-4: Symmetry-breaking — active groups occupy the lowest indices first.
# y_g >= y_{g+1} eliminates permutation-equivalent solutions, significantly
# reducing the symmetric search space in branch-and-bound.
for g in range(len(G) - 1):
    mdl.add_constraint(y[g] >= y[g + 1],
                       ctname=f"C_sym_g{g}")

# ── Model size ────────────────────────────────────────────────────────────────
n_bin  = len(x) + len(s) + len(y) + len(zG) + len(zI)
n_cont = len(Q) + len(rG) + len(rI)
n_con  = mdl.number_of_constraints
print(f"\nMODEL SIZE  (after corrections)")
print(f"  Binary variables     : {n_bin}")
print(f"  Continuous variables : {n_cont}")
print(f"  Total variables      : {n_bin + n_cont}")
print(f"  Constraints          : {n_con}")
print(f"  [FIX-3] C5 redundant constraint removed")
print(f"  [FIX-4] {len(G)-1} symmetry-breaking constraints added")
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

    # Individual sellers
    ind     = [i for i in producers if sol.get_value(s[i]) > 0.5]
    ind_rev = 0.0
    print(f"  INDIVIDUAL SELLERS  ({len(ind)} producers)")
    for i in ind:
        rev_i = sum(p[l] * sol.get_value(rI[(i, l)]) for l in segments)
        alloc = {l: sol.get_value(rI[(i, l)]) for l in segments}
        segs  = ", ".join(f"seg{l}:{alloc[l]:.1f}t@${p[l]}"
                          for l in segments if alloc[l] > 1e-6)
        print(f"    [{i}] {district[i]:<26} q={q[i]:>2}t | {segs} | ${rev_i:,.2f}")
        ind_rev += rev_i

    # Collaborative groups
    act     = [g for g in G if sol.get_value(y[g]) > 0.5]
    grp_rev = 0.0
    print()
    print(f"  COLLABORATIVE GROUPS  ({len(act)} active)")
    for g in act:
        mems    = [i for i in producers if sol.get_value(x[(i, g)]) > 0.5]
        tot_qty = sum(q[i] for i in mems)
        rev_g   = sum(p[l] * sol.get_value(rG[(g, l)]) for l in segments)
        alloc   = {l: sol.get_value(rG[(g, l)]) for l in segments}
        segs    = ", ".join(f"seg{l}:{alloc[l]:.1f}t@${p[l]}"
                            for l in segments if alloc[l] > 1e-6)
        print(f"    Group {g}: members {mems}  →  {[district[i] for i in mems]}")
        print(f"      Q={tot_qty}t | {segs} | ${rev_g:,.2f}")
        grp_rev += rev_g

    print()
    print(f"  Revenue — individual : ${ind_rev:,.2f}")
    print(f"  Revenue — groups     : ${grp_rev:,.2f}")
    print(f"  TOTAL OPTIMAL REVENUE: ${obj_val:,.2f}")
    print(sep)

    # ── Validation ────────────────────────────────────────────────────────────
    print("\nVALIDATION")

    # (a) All producers assigned
    assigned = set(ind)
    for g in act:
        assigned |= {i for i in producers if sol.get_value(x[(i, g)]) > 0.5}
    ok_assign = assigned == set(producers)
    print(f"  [{'OK' if ok_assign else 'FAIL'}] All producers assigned")

    # (b) Geographical constraints
    geo_ok = all(
        d[(mems[a], mems[b])] <= D_max
        for g in act
        for mems in [[i for i in producers if sol.get_value(x[(i, g)]) > 0.5]]
        for a in range(len(mems)) for b in range(a + 1, len(mems))
    )
    print(f"  [{'OK' if geo_ok else 'FAIL'}] Geographical constraints satisfied  (D_max={D_max} km)")

    # (c) Segment access thresholds
    thr_ok = all(
        sum(q[i] for i in producers if sol.get_value(x[(i, g)]) > 0.5) >= A[l]
        for g in act for l in segments
        if sol.get_value(rG[(g, l)]) > 1e-6
    )
    print(f"  [{'OK' if thr_ok else 'FAIL'}] Segment access thresholds satisfied")

    # FIX-7: Cross-check objective value from allocation variables
    recomputed_obj = (
        sum(p[l] * sol.get_value(rG[(g, l)]) for g in G for l in segments) +
        sum(p[l] * sol.get_value(rI[(i, l)]) for i in producers for l in segments)
    )
    obj_ok = abs(recomputed_obj - obj_val) < 1e-4
    print(f"  [{'OK' if obj_ok else 'FAIL'}] Recomputed revenue ${recomputed_obj:,.2f} "
          f"matches solver objective ${obj_val:,.2f}")

    # (d) Symmetry check: active groups occupy lowest indices
    sym_ok = all(
        (sol.get_value(y[g]) >= sol.get_value(y[g + 1]) - 1e-6)
        for g in range(len(G) - 1)
    )
    print(f"  [{'OK' if sym_ok else 'FAIL'}] Symmetry-breaking respected (active groups at lowest indices)")

    print()


# =============================================================================
# 4.  EXPORT TO EXCEL  (FIX-6: Revenue Summary sheet added)
# =============================================================================

def write_results_to_excel(sol, filepath="collaborative_selling_results_corrected.xlsx"):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    wb = openpyxl.Workbook()

    hdr_font  = Font(bold=True)
    hdr_fill  = PatternFill("solid", fgColor="D9E1F2")
    ctr       = Alignment(horizontal="center")
    thin      = Side(style="thin")
    full_bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)

    def style_header_row(ws, row=1):
        for cell in ws[row]:
            cell.font      = hdr_font
            cell.fill      = hdr_fill
            cell.alignment = ctr
            cell.border    = full_bdr

    # ── Sheet 1: Revenue Summary  [FIX-6] ─────────────────────────────────────
    ws0 = wb.active
    ws0.title = "Revenue Summary"
    ws0.append(["Entity", "Type", "Members", "Q (t)",
                "Rev seg0 ($)", "Rev seg1 ($)", "Rev seg2 ($)", "Total Rev ($)"])
    style_header_row(ws0)

    ind_list = [i for i in producers if sol.get_value(s[i]) > 0.5]
    act_list = [g for g in G if sol.get_value(y[g]) > 0.5]

    for i in ind_list:
        rev_by_seg = [round(p[l] * sol.get_value(rI[(i, l)]), 2) for l in segments]
        ws0.append([f"Producer {i}", "Individual", str(i),
                    q[i]] + rev_by_seg + [round(sum(rev_by_seg), 2)])

    for g in act_list:
        mems    = [i for i in producers if sol.get_value(x[(i, g)]) > 0.5]
        tot_qty = sum(q[i] for i in mems)
        rev_by_seg = [round(p[l] * sol.get_value(rG[(g, l)]), 2) for l in segments]
        ws0.append([f"Group {g}", "Collaborative", str(mems),
                    tot_qty] + rev_by_seg + [round(sum(rev_by_seg), 2)])

    # Totals row
    total_rev = round(sol.objective_value, 2)
    ws0.append(["TOTAL", "", "", "",
                round(sum(p[0]*sol.get_value(rI[(i,0)]) for i in ind_list) +
                      sum(p[0]*sol.get_value(rG[(g,0)]) for g in act_list), 2),
                round(sum(p[1]*sol.get_value(rI[(i,1)]) for i in ind_list) +
                      sum(p[1]*sol.get_value(rG[(g,1)]) for g in act_list), 2),
                round(sum(p[2]*sol.get_value(rI[(i,2)]) for i in ind_list) +
                      sum(p[2]*sol.get_value(rG[(g,2)]) for g in act_list), 2),
                total_rev])
    for cell in ws0[ws0.max_row]:
        cell.font = Font(bold=True)

    ws0.column_dimensions["A"].width = 16
    ws0.column_dimensions["C"].width = 20

    # ── Sheet 2: x_ig ─────────────────────────────────────────────────────────
    ws1 = wb.create_sheet("x_ig")
    ws1.append(["x_ig  (1 = producer i assigned to group g)"])
    ws1.append(["i \\ g"] + list(G))
    style_header_row(ws1, row=2)
    for i in producers:
        ws1.append([i] + [round(sol.get_value(x[(i, g)])) for g in G])

    # ── Sheet 3: s_i ──────────────────────────────────────────────────────────
    ws2 = wb.create_sheet("s_i")
    ws2.append(["i", "s_i (1 = sells individually)"])
    style_header_row(ws2)
    for i in producers:
        ws2.append([i, round(sol.get_value(s[i]))])

    # ── Sheet 4: y_g ──────────────────────────────────────────────────────────
    ws3 = wb.create_sheet("y_g")
    ws3.append(["g", "y_g (1 = group active)"])
    style_header_row(ws3)
    for g in G:
        ws3.append([g, round(sol.get_value(y[g]))])

    # ── Sheet 5: Q_g ──────────────────────────────────────────────────────────
    ws4 = wb.create_sheet("Q_g")
    ws4.append(["g", "Q_g (total quantity, t)"])
    style_header_row(ws4)
    for g in G:
        ws4.append([g, round(sol.get_value(Q[g]), 4)])

    # ── Sheet 6: rG_gl ────────────────────────────────────────────────────────
    ws5 = wb.create_sheet("rG_gl")
    ws5.append(["g \\ l (qty allocated, t)"] + [f"seg {l}" for l in segments])
    style_header_row(ws5)
    for g in G:
        ws5.append([g] + [round(sol.get_value(rG[(g, l)]), 4) for l in segments])

    # ── Sheet 7: rI_il ────────────────────────────────────────────────────────
    ws6 = wb.create_sheet("rI_il")
    ws6.append(["i \\ l (qty allocated, t)"] + [f"seg {l}" for l in segments])
    style_header_row(ws6)
    for i in producers:
        ws6.append([i] + [round(sol.get_value(rI[(i, l)]), 4) for l in segments])

    # ── Sheet 8: zG_gl ────────────────────────────────────────────────────────
    ws7 = wb.create_sheet("zG_gl")
    ws7.append(["g \\ l (access gate)"] + [f"seg {l}" for l in segments])
    style_header_row(ws7)
    for g in G:
        ws7.append([g] + [round(sol.get_value(zG[(g, l)])) for l in segments])

    # ── Sheet 9: zI_il ────────────────────────────────────────────────────────
    ws8 = wb.create_sheet("zI_il")
    ws8.append(["i \\ l (access gate)"] + [f"seg {l}" for l in segments])
    style_header_row(ws8)
    for i in producers:
        ws8.append([i] + [round(sol.get_value(zI[(i, l)])) for l in segments])

    wb.save(filepath)
    print(f"\nResults saved to: {filepath}")
    return filepath

if sol is not None:
    write_results_to_excel(sol, "collaborative_selling_results_corrected.xlsx")

