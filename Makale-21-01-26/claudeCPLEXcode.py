"""
OPL → Python/DOcplex  (Exact 1-to-1 Translation)
===================================================
Data source : claudecplex.xlsx  (your actual file)
Solver      : IBM CPLEX via docplex

Sheet mapping (verified from your Excel):
  'fr'       → lamda[r]       (route fixed cost)
  'Beta'     → beta[f][k]     (assignment cost coefficient)
  'nr'       → As / NR        (route→b→demand nodes)
  'gama'     → gama[f,b,k]    (load cost coefficient)
  'delta'    → delta[b][r]    (route-hub assignment binary)
  'upsilon'  → upsilon[f,s,t] (facility capacity)
  'theta'    → theta[k]       (vehicle capacity)
  'd'        → de[d,t]        (demand)
  'cr'       → c[r]           (route capacity)
  'hi'       → h[d]           (holding cost)

  pr[s]  → hardcoded (not in Excel — adjust below if needed)
  p      → hardcoded (not in Excel — adjust below if needed)

Install: pip install docplex pandas openpyxl
"""

import pandas as pd
from docplex.mp.model import Model
from math import sqrt

# ===========================================================================
# 0. CONFIGURATION
# ===========================================================================
DATA_FILE = "claudecplex.xlsx"   # <-- put full path if not in same folder

EPGAP  = 0.01
m      = 10       # m > T → perishability disabled in main model
mprime = 2        # real shelf life for analysis
co     = 0.1
Zalpha = 2.575

# Parameters NOT in Excel — adjust these if needed
pr = {1: 1/3, 2: 1/3, 3: 1/3}   # scenario probabilities (equal weight)
p  = 5.0                          # waste cost per unit

# ===========================================================================
# 1. INDEX RANGES  (matching OPL)
# ===========================================================================
F = range(1, 101)   # 1..100
S = range(1, 4)     # 1..3
T = range(1, 7)     # 1..6
R = range(1, 31)    # 1..30
D = range(1, 31)    # 1..30
B = range(1, 4)     # 1..3
K = range(1, 26)    # 1..25

# ===========================================================================
# 2. LOAD DATA FROM EXCEL
# ===========================================================================
xl = pd.ExcelFile(DATA_FILE)

# --- lamda[r] : sheet 'fr', columns [Rota, fr] ----------------------------
df_fr = xl.parse("fr", header=0)
df_fr.columns = ["r", "value"]
lamda = {int(row["r"]): float(row["value"]) for _, row in df_fr.iterrows()}

# --- beta[f,k] : sheet 'Beta', wide format --------------------------------
# First col = facility index (Ad), remaining cols = k=1..25
df_beta = xl.parse("Beta", header=0, index_col=0)
beta = {}
for f_idx, row in df_beta.iterrows():
    f = int(f_idx)
    for col, val in row.items():
        k = int(float(col))
        if k in K:
            beta[f, k] = float(val)

# --- nr sheet : route | b | demand_node1 | demand_node2 | ... -------------
# col0 = route r, col1 = b, col2.. = demand nodes (NaN = end)
df_nr = xl.parse("nr", header=None)
As = []
NR_global = set()
for _, row in df_nr.iterrows():
    r_val = int(row.iloc[0])
    b_val = int(row.iloc[1])
    nodes = set()
    for val in row.iloc[2:]:
        if pd.notna(val):
            nodes.add(int(val))
    NR_global |= nodes
    As.append({"r": r_val, "b": b_val, "NR": nodes})
NR = NR_global

# --- gama[f,b,k] : sheet 'gama', columns [f, b, k, gamma] ----------------
df_gama = xl.parse("gama", header=0)
df_gama.columns = ["f", "b", "k", "value"]
gama = {(int(r["f"]), int(r["b"]), int(r["k"])): float(r["value"])
        for _, r in df_gama.iterrows()}

# --- delta[b,r] : sheet 'delta', wide format ------------------------------
# index = b (1,2,3), columns = r (1..30)
df_delta_raw = xl.parse("delta", header=0, index_col=0)
delta = {}
for b_idx, row in df_delta_raw.iterrows():
    b = int(b_idx)
    for col, val in row.items():
        r = int(col)
        delta[b, r] = int(val)

# --- upsilon[f,s,t] : sheet 'upsilon', columns [f, s, t, value] -----------
df_up = xl.parse("upsilon", header=0)
df_up.columns = ["f", "s", "t", "value"]
upsilon = {(int(r["f"]), int(r["s"]), int(r["t"])): float(r["value"])
           for _, r in df_up.iterrows()}

# --- theta[k] : sheet 'theta', columns [K, theta] -------------------------
df_theta = xl.parse("theta", header=0)
df_theta.columns = ["k", "value"]
theta = {int(r["k"]): float(r["value"]) for _, r in df_theta.iterrows()}

# --- de[d,t] : sheet 'd', wide format -------------------------------------
# col0 = d, remaining cols = t=1..6
df_d = xl.parse("d", header=0)
df_d.columns = ["d"] + [str(t) for t in T]
de = {}
for _, row in df_d.iterrows():
    d = int(row["d"])
    for t in T:
        de[d, t] = int(row[str(t)])

# --- c[r] : sheet 'cr', columns [rota, cap] --------------------------------
df_cr = xl.parse("cr", header=0)
df_cr.columns = ["r", "value"]
c = {int(r["r"]): float(r["value"]) for _, r in df_cr.iterrows()}

# --- h[d] : sheet 'hi', columns [D/T, basecase] ---------------------------
df_hi = xl.parse("hi", header=0)
df_hi.columns = ["d", "value"]
h = {int(r["d"]): float(r["value"]) for _, r in df_hi.iterrows()}

# ===========================================================================
# 3. BUILD MODEL  (exact OPL logic, same order)
# ===========================================================================
mdl = Model(name="PerishableDistribution")
mdl.parameters.mip.tolerances.mipgap = EPGAP

# ---------------------------------------------------------------------------
# Decision Variables  — identical to OPL declarations
# ---------------------------------------------------------------------------

# dvar boolean Y[F][B][K][S][T]
Y = {(f, b, k, s, t): mdl.binary_var(name=f"Y_{f}_{b}_{k}_{s}_{t}")
     for f in F for b in B for k in K for s in S for t in T}

# dvar int I[D][T]
I = {(i, t): mdl.integer_var(lb=-mdl.infinity, name=f"I_{i}_{t}")
     for i in D for t in T}

# dvar int+ Ipo[D][T]
Ipo = {(i, t): mdl.integer_var(lb=0, name=f"Ipo_{i}_{t}")
       for i in D for t in T}

# dvar int+ L[F][B][K][S][T]
L = {(f, b, k, s, t): mdl.integer_var(lb=0, name=f"L_{f}_{b}_{k}_{s}_{t}")
     for f in F for b in B for k in K for s in S for t in T}

# dvar int+ Q[B][D][T]
Q = {(b, i, t): mdl.integer_var(lb=0, name=f"Q_{b}_{i}_{t}")
     for b in B for i in D for t in T}

# dvar int+ Z[D][R][T]
Z = {(i, r, t): mdl.integer_var(lb=0, name=f"Z_{i}_{r}_{t}")
     for i in D for r in R for t in T}

# dvar int+ W[D][T]
W = {(i, t): mdl.integer_var(lb=0, name=f"W_{i}_{t}")
     for i in D for t in T}

# dvar boolean X[R][T]
X = {(r, t): mdl.binary_var(name=f"X_{r}_{t}")
     for r in R for t in T}

# dvar int+ Inv
Inv = mdl.integer_var(lb=0, name="Inv")

# dvar int+ Wast
Wast = mdl.integer_var(lb=0, name="Wast")

# dvar int EIprime[D][T]
EIprime = {(i, t): mdl.integer_var(lb=-mdl.infinity, name=f"EIp_{i}_{t}")
           for i in D for t in T}

# dvar int+ EWprime[D][T]
EWprime = {(i, t): mdl.integer_var(lb=0, name=f"EWp_{i}_{t}")
           for i in D for t in T}

# ---------------------------------------------------------------------------
# Objective  — identical to OPL
# ---------------------------------------------------------------------------
invcost        = mdl.sum(Ipo[i, t] * h[i]
                         for i in D for t in T)
wastecost      = mdl.sum(W[i, t] * p
                         for i in D for t in T if t >= m)
routecost      = mdl.sum(lamda[r] * X[r, t]
                         for r in R for t in T)
assignmentcost = mdl.sum(pr[s] * beta[f, k] * Y[f, b, k, s, t]
                         for f in F for b in B for k in K for s in S for t in T)
loadcost       = mdl.sum(pr[s] * gama[f, b, k] * L[f, b, k, s, t]
                         for f in F for b in B for k in K for s in S for t in T)

mdl.minimize(invcost + wastecost + routecost + assignmentcost + loadcost)

# ---------------------------------------------------------------------------
# Constraints  — identical to OPL, same order
# ---------------------------------------------------------------------------

# 1a. sum(b,k) L[f][b][k][s][t] <= upsilon[f,s,t]
for f in F:
    for s in S:
        for t in T:
            mdl.add_constraint(
                mdl.sum(L[f, b, k, s, t] for b in B for k in K)
                <= upsilon[f, s, t]
            )

# 1b. sum(f,b) L[f][b][k][s][t] <= theta[k]
for k in K:
    for s in S:
        for t in T:
            mdl.add_constraint(
                mdl.sum(L[f, b, k, s, t] for f in F for b in B) <= theta[k]
            )

# 1c. L[f][b][k][s][t] <= Y[f][b][k][s][t] * theta[k]
for f in F:
    for b in B:
        for k in K:
            for s in S:
                for t in T:
                    mdl.add_constraint(
                        L[f, b, k, s, t] <= Y[f, b, k, s, t] * theta[k]
                    )

# 1d. sum(b) Y[f][b][k][s][t] <= 1
for f in F:
    for s in S:
        for t in T:
            for k in K:
                mdl.add_constraint(
                    mdl.sum(Y[f, b, k, s, t] for b in B) <= 1
                )

# 2. sum(f,k) L[f][b][k][s][t] == sum(i) Q[b][i][t]
for b in B:
    for s in S:
        for t in T:
            mdl.add_constraint(
                mdl.sum(L[f, b, k, s, t] for f in F for k in K)
                == mdl.sum(Q[b, i, t] for i in D)
            )

# 3a. I[i][t] == sum(a=1..t, b) Q[b][i][a] - sum(a=1..t) (de[i][a]+W[i][a])
for i in D:
    for t in T:
        mdl.add_constraint(
            I[i, t] == (mdl.sum(Q[b, i, a] for a in range(1, t+1) for b in B)
                        - mdl.sum(de[i, a] + W[i, a] for a in range(1, t+1)))
        )

# 3b. Ipo[i][t] >= I[i][t]
for i in D:
    for t in T:
        mdl.add_constraint(Ipo[i, t] >= I[i, t])

# 3c. W[i][t] >= ... for t >= m
for i in D:
    for t in T:
        if t >= m:
            rhs = (I[i, t - m + 1]
                   - mdl.sum(de[i, a] for a in range(t - m + 2, t + 1))
                   - mdl.sum(W[i, a] for a in range(t - m + 2, t)))
            mdl.add_constraint(W[i, t] >= rhs)

# 3d. W[i][t] == 0 for t < m
for i in D:
    for t in T:
        if t < m:
            mdl.add_constraint(W[i, t] == 0)

# 4. Service level
for i in D:
    for t in T:
        demand_sum = sum(de[i, a] for a in range(1, t + 1))
        safety     = sqrt(sum(de[i, a] ** 2 for a in range(1, t + 1))) * co * Zalpha
        lhs = (mdl.sum(Q[b, i, a] for a in range(1, t + 1) for b in B)
               - mdl.sum(W[i, a] for a in range(1, t)))
        mdl.add_constraint(lhs >= demand_sum + safety)

# 5a. Z[i][r][t] == 0 if i not in route nodes
for r in R:
    route_nodes = set()
    for a in As:
        if a["r"] == r:
            route_nodes |= a["NR"]
    for t in T:
        for i in D:
            if i not in route_nodes:
                mdl.add_constraint(Z[i, r, t] == 0)

# 5b. sum(i in route) Z[i][r][t] <= c[r] * X[r][t]
for r in R:
    route_nodes = set()
    for a in As:
        if a["r"] == r:
            route_nodes |= a["NR"]
    for t in T:
        mdl.add_constraint(
            mdl.sum(Z[i, r, t] for i in D if i in route_nodes)
            <= c[r] * X[r, t]
        )

# 5c. sum(r) Z[i][r][t] * delta[b][r] == Q[b][i][t]
for i in D:
    for t in T:
        for b in B:
            mdl.add_constraint(
                mdl.sum(Z[i, r, t] * delta[b, r] for r in R) == Q[b, i, t]
            )

# 6a. EIprime — KISIT40
for i in D:
    for t in T:
        mdl.add_constraint(
            EIprime[i, t] == (
                mdl.sum(Q[b, i, a] for a in range(1, t+1) for b in B)
                - mdl.sum(de[i, a] + EWprime[i, a] for a in range(1, t+1))
            )
        )

# 6b. EWprime == 0 for t < mprime  — KISIT42
for i in D:
    for t in T:
        if t < mprime:
            mdl.add_constraint(EWprime[i, t] == 0)

# 6c. KISIT41
for i in D:
    for t in T:
        if t >= mprime:
            rhs = (EIprime[i, t - mprime + 1]
                   - mdl.sum(de[i, a] for a in range(t - mprime + 2, t + 1))
                   - mdl.sum(EWprime[i, a] for a in range(t - mprime + 2, t)))
            mdl.add_constraint(EWprime[i, t] == mdl.max(rhs, 0))

# 6d. Inv = sum h[i] * max(EIprime[i][t], 0)  — KISIT44
mdl.add_constraint(
    Inv == mdl.sum(h[i] * mdl.max(EIprime[i, t], 0) for i in D for t in T)
)

# 6e. Wast = sum p * EWprime  — KISIT43
mdl.add_constraint(
    Wast == mdl.sum(p * EWprime[i, t]
                    for i in D for t in T if t >= mprime)
)

# ===========================================================================
# 4. SOLVE
# ===========================================================================
print("=" * 55)
print("Model built successfully.")
print(f"  Variables   : {mdl.number_of_variables:,}")
print(f"  Constraints : {mdl.number_of_constraints:,}")
print("=" * 55)
print("Solving...")

sol = mdl.solve(log_output=True)

# ===========================================================================
# 5. OUTPUT
# ===========================================================================
if sol:
    print("\n================ RESULTS ================")
    print(f"Objective          : {mdl.objective_value:.4f}")
    print(f"  Inventory cost   : {sol.get_value(invcost):.4f}")
    print(f"  Waste cost       : {sol.get_value(wastecost):.4f}")
    print(f"  Route cost       : {sol.get_value(routecost):.4f}")
    print(f"  Assignment cost  : {sol.get_value(assignmentcost):.4f}")
    print(f"  Load cost        : {sol.get_value(loadcost):.4f}")
    print(f"\nInv  (analysis)   : {sol.get_value(Inv):.4f}")
    print(f"Wast (analysis)   : {sol.get_value(Wast):.4f}")

    # Export results
    rows = []
    for i in D:
        for t in T:
            rows.append({
                "d": i, "t": t,
                "I":       sol.get_value(I[i, t]),
                "Ipo":     sol.get_value(Ipo[i, t]),
                "W":       sol.get_value(W[i, t]),
                "EIprime": sol.get_value(EIprime[i, t]),
                "EWprime": sol.get_value(EWprime[i, t]),
            })
    pd.DataFrame(rows).to_excel("results_inventory.xlsx", index=False)

    rows_q = [{"b": b, "d": i, "t": t, "Q": sol.get_value(Q[b, i, t])}
              for b in B for i in D for t in T]
    pd.DataFrame(rows_q).to_excel("results_Q.xlsx", index=False)

    rows_x = [{"r": r, "t": t, "X": sol.get_value(X[r, t]),
                "Z_total": sum(sol.get_value(Z[i, r, t]) for i in D)}
              for r in R for t in T]
    pd.DataFrame(rows_x).to_excel("results_routes.xlsx", index=False)

    print("\nResults saved to:")
    print("  results_inventory.xlsx")
    print("  results_Q.xlsx")
    print("  results_routes.xlsx")

else:
    print("No solution found. Status:", mdl.solve_status)
    mdl.export_as_lp("debug_model.lp")
    print("Model exported to debug_model.lp for inspection.")