import pandas as pd
import math
from pyomo.environ import *

# ============================================================
# EXCEL FILE
# ============================================================
file = "claudecplex.xlsx"

# ============================================================
# PARAMETERS FROM EXCEL
# ============================================================

# --- gama[f,b,k] ---
gama_df = pd.read_excel(file, sheet_name="gama", header=0)
gama = {}
for _, row in gama_df.iterrows():
    f, b, k, val = int(row.iloc[0]), int(row.iloc[1]), int(row.iloc[2]), row.iloc[3]
    gama[f, b, k] = val

# --- beta[f,k] ---
beta_df = pd.read_excel(file, sheet_name="Beta", header=0, index_col=0)
beta = {}
for i_idx, row in enumerate(beta_df.itertuples(index=False)):
    f = i_idx + 1
    for k_idx, val in enumerate(row):
        k = k_idx + 1
        beta[f, k] = val

# --- upsilon[f,s,t] ---
upsilon_df = pd.read_excel(file, sheet_name="upsilon", header=0)
upsilon = {}
for _, row in upsilon_df.iterrows():
    f, s, t, val = int(row.iloc[0]), int(row.iloc[1]), int(row.iloc[2]), row.iloc[3]
    upsilon[f, s, t] = val

# --- theta[k] ---
theta_df = pd.read_excel(file, sheet_name="theta", header=0)
theta = {}
for _, row in theta_df.iterrows():
    k = int(row.iloc[0])
    if k <= 25:
        theta[k] = row.iloc[1]

# --- h[i] ---
h_df = pd.read_excel(file, sheet_name="hi", header=0)
h = {}
for i_idx, row in h_df.iterrows():
    i = i_idx + 1
    h[i] = row.iloc[1]

# --- de[i,t] ---
de_df = pd.read_excel(file, sheet_name="d", header=0, index_col=0)
de = {}
for i_idx, row in enumerate(de_df.itertuples(index=False)):
    i = i_idx + 1
    for t_idx, val in enumerate(row):
        t = t_idx + 1
        de[i, t] = int(val)

# --- lamda[r] ---
lamda_df = pd.read_excel(file, sheet_name="fr", header=0)
lamda = {}
for _, row in lamda_df.iterrows():
    r = int(row.iloc[0])
    lamda[r] = row.iloc[1]

# --- c[r] ---
c_df = pd.read_excel(file, sheet_name="cr", header=0)
c = {}
for _, row in c_df.iterrows():
    r = int(row.iloc[0])
    c[r] = row.iloc[1]

# --- delta[b,r] ---
delta_df = pd.read_excel(file, sheet_name="delta", header=0, index_col=0)
delta = {}
for b_idx, row in enumerate(delta_df.itertuples(index=False)):
    b = b_idx + 1
    for r_idx, val in enumerate(row):
        r = r_idx + 1
        delta[b, r] = int(val)

# ============================================================
# ⚠️ MISSING FROM EXCEL — please fill in
# ============================================================
pr = {1: 0.30, 2: 0.5, 3: 0.2}  # probability of each scenario — PLEASE FILL IN
p  = 27.55696873  

# ============================================================
# AS — route to depot mapping
# ============================================================
AS = {
    1:  {25,24,14,13},
    2:  {16,17,23,20},
    3:  {10,22,27,30},
    4:  {28,26,19},
    5:  {11,12,29},
    6:  {8,18,21,15,9},
    7:  {19,20,14,7,2},
    8:  {17,15,21,25},
    9:  {22,8,6,5,1},
    10: {23,13,10,11},
    11: {29,26,24,16,4,3},
    12: {9,18,30,28,27},
    13: {30,23,5,4,2},
    14: {12,15,21,25,28},
    15: {7,9,10,16,22},
    16: {18,19,17,14,12},
    17: {29,24,8,1,3},
    18: {27,20,7,6,4,2},
    19: {7,6,5,3,1},
    20: {20,21,22,24,26,28},
    21: {23,8,16,13,11},
    22: {6,5,4,3,2,1},
    23: {17,30,14,9,10,12},
    24: {18,19,25,26,27,29},
    25: {11,13,15,20,22,26},
    26: {10,16,15,27},
    27: {7,13,14,17,18},
    28: {9,12,11},
    29: {8,19,23,25,24,21},
    30: {18,30,29},
}

# ============================================================
# SETS
# ============================================================
F_set = list(range(1, 101))
S_set = list(range(1, 4))
T_set = list(range(1, 7))
R_set = list(range(1, 31))
D_set = list(range(1, 31))
B_set = list(range(1, 4))
K_set = list(range(1, 26))

# ============================================================
# CONSTANTS
# ============================================================
m      = 2
co     = 0.1
Zalpha = 2.575

# ============================================================
# PYOMO MODEL
# ============================================================
model = ConcreteModel()

# --- Sets ---
model.F = Set(initialize=F_set)
model.S = Set(initialize=S_set)
model.T = Set(initialize=T_set)
model.R = Set(initialize=R_set)
model.D = Set(initialize=D_set)
model.B = Set(initialize=B_set)
model.K = Set(initialize=K_set)

# ============================================================
# DECISION VARIABLES
# ============================================================
model.Y = Var(model.F, model.B, model.K, model.S, model.T, domain=Binary)
model.I = Var(model.D, model.T, domain=Integers)
model.Ipo = Var(model.D, model.T, domain=NonNegativeIntegers)
model.L = Var(model.F, model.B, model.K, model.S, model.T, domain=NonNegativeIntegers)
model.Q = Var(model.B, model.D, model.T, domain=NonNegativeIntegers)
model.Z = Var(model.D, model.R, model.T, domain=NonNegativeIntegers)
model.W = Var(model.D, model.T, domain=NonNegativeIntegers)
model.X = Var(model.R, model.T, domain=Binary)



# ============================================================
# OBJECTIVE FUNCTION
# ============================================================
def obj_rule(model):
    invcost = sum(model.Ipo[i,t] * h[i] for i in model.D for t in model.T)
    wastecost = sum(model.W[i,t] * p for i in model.D for t in model.T if t >= m)
    routecost = sum(lamda[r] * model.X[r,t] for r in model.R for t in model.T)
    assignmentcost = sum(
        pr[s] * beta[f,k] * model.Y[f,b,k,s,t]
        for f in model.F for b in model.B for k in model.K
        for s in model.S for t in model.T)
    loadcost = sum(
        pr[s] * gama[f,b,k] * model.L[f,b,k,s,t]
        for f in model.F for b in model.B for k in model.K
        for s in model.S for t in model.T)
    return invcost + wastecost + routecost + assignmentcost + loadcost

model.obj = Objective(rule=obj_rule, sense=minimize)

# ============================================================
# ORIGINAL CONSTRAINTS
# ============================================================

# Supplier capacity
def supplier_cap(model, f, s, t):
    return sum(model.L[f,b,k,s,t] for b in model.B for k in model.K) <= upsilon[f,s,t]
model.supplier_cap = Constraint(model.F, model.S, model.T, rule=supplier_cap)

# Vehicle capacity
def vehicle_cap(model, k, s, t):
    return sum(model.L[f,b,k,s,t] for f in model.F for b in model.B) <= theta[k]
model.vehicle_cap = Constraint(model.K, model.S, model.T, rule=vehicle_cap)

# Linking L and Y
def link_LY(model, f, b, k, s, t):
    return model.L[f,b,k,s,t] <= model.Y[f,b,k,s,t] * theta[k]
model.link_LY = Constraint(model.F, model.B, model.K, model.S, model.T, rule=link_LY)

# One transshipment center per supplier per vehicle
def one_center(model, f, s, t, k):
    return sum(model.Y[f,b,k,s,t] for b in model.B) <= 1
model.one_center = Constraint(model.F, model.S, model.T, model.K, rule=one_center)

# Flow balance at transshipment center
def flow_balance(model, b, s, t):
    return (sum(model.L[f,b,k,s,t] for f in model.F for k in model.K) ==
            sum(model.Q[b,i,t] for i in model.D))
model.flow_balance = Constraint(model.B, model.S, model.T, rule=flow_balance)

# Inventory balance (2.11)
def inv_balance(model, i, t):
    return (model.I[i,t] ==
            sum(model.Q[b,i,a] for a in range(1, t+1) for b in model.B) -
            sum(de[i,a] + model.W[i,a] for a in range(1, t+1)))
model.inv_balance = Constraint(model.D, model.T, rule=inv_balance)

# Positive inventory (2.12)
def pos_inv(model, i, t):
    return model.Ipo[i,t] >= model.I[i,t]
model.pos_inv = Constraint(model.D, model.T, rule=pos_inv)

# Waste constraint (2.13)
def waste_con(model, i, t):
    if t >= m:
        return (model.W[i,t] >= model.I[i, t-m+1]
                - sum(de[i,a] for a in range(t-m+2, t+1))
                - sum(model.W[i,a] for a in range(t-m+2, t)))
    return Constraint.Skip
model.waste_con = Constraint(model.D, model.T, rule=waste_con)

# No waste before m (2.14)
def no_waste(model, i, t):
    if t < m:
        return model.W[i,t] == 0
    return Constraint.Skip
model.no_waste = Constraint(model.D, model.T, rule=no_waste)

# Safety stock (2.15)
def safety_stock(model, i, t):
    lhs = (sum(model.Q[b,i,a] for a in range(1, t+1) for b in model.B) -
           sum(model.W[i,a] for a in range(1, t)))
    rhs = (sum(de[i,a] for a in range(1, t+1)) +
           math.sqrt(sum(de[i,a]**2 for a in range(1, t+1))) * co * Zalpha)
    return lhs >= rhs
model.safety_stock = Constraint(model.D, model.T, rule=safety_stock)

# Route-depot assignment (2.16)
def route_depot(model, r, t, i):
    if i not in AS.get(r, set()):
        return model.Z[i,r,t] == 0
    return Constraint.Skip
model.route_depot = Constraint(model.R, model.T, model.D, rule=route_depot)

# Route capacity (2.17)
def route_cap(model, r, t):
    return (sum(model.Z[i,r,t] for i in AS.get(r, set())) <= c[r] * model.X[r,t])
model.route_cap = Constraint(model.R, model.T, rule=route_cap)

# Flow from routes to depots (2.18)
def route_flow(model, i, t, b):
    return (sum(model.Z[i,r,t] * delta[b,r] for r in model.R) == model.Q[b,i,t])
model.route_flow = Constraint(model.D, model.T, model.B, rule=route_flow)





# ============================================================
# SOLVE
# ============================================================
print("Building model...")
print("Solving...")

solver = SolverFactory('cplex', executable=r"C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex.exe")
solver.options['mipgap'] = 0.01

results = solver.solve(model, tee=True)

if results.solver.termination_condition == TerminationCondition.optimal or \
   results.solver.termination_condition == TerminationCondition.feasible:
    print("\n--- RESULTS ---")
    print(f"Objective value : {value(model.obj):.2f}")
    print(f"WasteCostNP     : {value(model.WasteCostNP):.2f}")
    print(f"InvCostNP       : {value(model.InvCostNP):.2f}")

    print("\n--- EWprime values ---")
    for i in D_set:
        row = [value(model.EWprime[i,t]) for t in T_set]
        print(f"Depot {i}: {row}")
else:
    print("No solution found.")
    print(f"Termination condition: {results.solver.termination_condition}")
