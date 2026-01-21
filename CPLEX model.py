# ---------------------------------------
# One-time installs (run in terminal):
# pip install pyomo pandas openpyxl
# ---------------------------------------

import math
import pandas as pd
import pyomo.environ as pyo


# =========================
# Helpers
# =========================
def assert_no_duplicates(df, cols, name):
    dup = df.duplicated(subset=cols).sum()
    if dup:
        raise ValueError(f"{name} has {dup} duplicate keys on {cols}!")

def require_columns(df, cols, sheet_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet_name}' missing columns {missing}. Found: {list(df.columns)}")

def read_and_prepare_data(file_path: str):
    # --- Read sheets
    beta_df  = pd.read_excel(file_path, sheet_name="beta")
    gamma_df = pd.read_excel(file_path, sheet_name="gamma")
    theta_df = pd.read_excel(file_path, sheet_name="theta")
    vehicles_df = pd.read_excel(file_path, sheet_name="vehicles")
    stock_costs_df = pd.read_excel(file_path, sheet_name="stock_costs")
    supply_df = pd.read_excel(file_path, sheet_name="supply")
    demand_df = pd.read_excel(file_path, sheet_name="demand")
    route_costs_df = pd.read_excel(file_path, sheet_name="route_costs")
    route_capacity_df = pd.read_excel(file_path, sheet_name="route_capacity")
    delta_df = pd.read_excel(file_path, sheet_name="delta", header=None)

    # --- Column checks (based on your code)
    require_columns(beta_df, ["f", "k", "beta"], "beta")
    require_columns(gamma_df, ["f", "b", "k", "gamma"], "gamma")
    require_columns(theta_df, ["k", "theta"], "theta")
    require_columns(vehicles_df, ["k", "f"], "vehicles")
    require_columns(stock_costs_df, ["d", "stock_cost"], "stock_costs")
    require_columns(supply_df, ["f", "s", "t", "supply"], "supply")
    require_columns(demand_df, ["d", "t", "demand"], "demand")
    require_columns(route_costs_df, ["r", "cost"], "route_costs")
    require_columns(route_capacity_df, ["r", "capacity"], "route_capacity")

    # --- Types
    beta_df  = beta_df.astype({"f": int, "k": int})
    gamma_df = gamma_df.astype({"f": int, "b": int, "k": int})
    theta_df = theta_df.astype({"k": int})
    vehicles_df = vehicles_df.astype({"k": int, "f": int})
    stock_costs_df = stock_costs_df.astype({"d": int})
    supply_df = supply_df.astype({"f": int, "s": int, "t": int})
    demand_df = demand_df.astype({"d": int, "t": int})
    route_costs_df = route_costs_df.astype({"r": int})
    route_capacity_df = route_capacity_df.astype({"r": int})

    # --- Numeric safety
    for df, col in [
        (beta_df, "beta"),
        (gamma_df, "gamma"),
        (theta_df, "theta"),
        (stock_costs_df, "stock_cost"),
        (supply_df, "supply"),
        (demand_df, "demand"),
        (route_costs_df, "cost"),
        (route_capacity_df, "capacity"),
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # --- Duplicate checks
    assert_no_duplicates(beta_df,  ["f","k"],       "beta")
    assert_no_duplicates(gamma_df, ["f","b","k"],   "gamma")
    assert_no_duplicates(theta_df, ["k"],           "theta")
    assert_no_duplicates(vehicles_df, ["k"],        "vehicles")
    assert_no_duplicates(stock_costs_df, ["d"],     "stock_costs")
    assert_no_duplicates(supply_df, ["f","s","t"],  "supply")
    assert_no_duplicates(demand_df, ["d","t"],      "demand")
    assert_no_duplicates(route_costs_df, ["r"],     "route_costs")
    assert_no_duplicates(route_capacity_df, ["r"],  "route_capacity")

    # --- Dicts
    beta_dict  = {(int(r.f), int(r.k)): float(r.beta) for r in beta_df.itertuples(index=False)}
    gamma_dict = {(int(r.f), int(r.b), int(r.k)): float(r.gamma) for r in gamma_df.itertuples(index=False)}
    theta_dict = {int(r.k): float(r.theta) for r in theta_df.itertuples(index=False)}
    h_dict     = {int(r.d): float(r.stock_cost) for r in stock_costs_df.itertuples(index=False)}
    upsilon_dict = {(int(r.f), int(r.s), int(r.t)): float(r.supply) for r in supply_df.itertuples(index=False)}
    demand_dict  = {(int(r.d), int(r.t)): float(r.demand) for r in demand_df.itertuples(index=False)}
    lamda_dict   = {int(r.r): float(r.cost) for r in route_costs_df.itertuples(index=False)}
    c_dict       = {int(r.r): float(r.capacity) for r in route_capacity_df.itertuples(index=False)}

    # --- delta: exactly B4:AE6  -> rows 4..6 and cols B..AE
    delta_block = delta_df.iloc[3:6, 1:31]  # (3 x 30)
    if delta_block.shape != (3, 30):
        raise ValueError(f"delta_block shape is {delta_block.shape}, expected (3,30). Check 'delta!B4:AE6'.")

    delta_block = delta_block.apply(pd.to_numeric, errors="raise")
    delta_dict = {(b, r): int(delta_block.iloc[b-1, r-1]) for b in range(1,4) for r in range(1,31)}

    return beta_dict, gamma_dict, theta_dict, h_dict, upsilon_dict, demand_dict, lamda_dict, c_dict, delta_dict


def build_model(beta_dict, gamma_dict, theta_dict, h_dict, upsilon_dict, demand_dict,
                lamda_dict, c_dict, delta_dict,
                route_to_depots, pr_dict, m, p, co, Zalpha):

    # Sets (OPL ranges)
    F = range(1, 101)
    S = range(1, 4)
    T = range(1, 7)
    R = range(1, 31)
    D = range(1, 31)
    B = range(1, 4)
    K = range(1, 26)

    def de(i, t):
        return float(demand_dict.get((i, t), 0.0))

    model = pyo.ConcreteModel()
    model.F = pyo.Set(initialize=list(F))
    model.S = pyo.Set(initialize=list(S))
    model.T = pyo.Set(initialize=list(T))
    model.R = pyo.Set(initialize=list(R))
    model.D = pyo.Set(initialize=list(D))
    model.B = pyo.Set(initialize=list(B))
    model.K = pyo.Set(initialize=list(K))

    # Vars
    model.Y = pyo.Var(model.F, model.B, model.K, model.S, model.T, domain=pyo.Binary)
    model.I = pyo.Var(model.D, model.T, domain=pyo.Integers)
    model.Ipo = pyo.Var(model.D, model.T, domain=pyo.NonNegativeIntegers)
    model.L = pyo.Var(model.F, model.B, model.K, model.S, model.T, domain=pyo.NonNegativeIntegers)
    model.Q = pyo.Var(model.B, model.D, model.T, domain=pyo.NonNegativeIntegers)
    model.Z = pyo.Var(model.D, model.R, model.T, domain=pyo.NonNegativeIntegers)
    model.W = pyo.Var(model.D, model.T, domain=pyo.NonNegativeIntegers)
    model.X = pyo.Var(model.R, model.T, domain=pyo.Binary)

    # Objective
    def obj_rule(mdl):
        invcost = sum(mdl.Ipo[i,t] * h_dict[i] for i in mdl.D for t in mdl.T)
        wastecost = sum(mdl.W[i,t] * p for i in mdl.D for t in mdl.T if t >= m)
        routecost = sum(lamda_dict[r] * mdl.X[r,t] for r in mdl.R for t in mdl.T)

        assignmentcost = sum(
            pr_dict[s] * sum(beta_dict[(f,k)] * mdl.Y[f,b,k,s,t]
                             for f in mdl.F for b in mdl.B for k in mdl.K)
            for s in mdl.S for t in mdl.T
        )

        loadcost = sum(
            pr_dict[s] * sum(gamma_dict[(f,b,k)] * mdl.L[f,b,k,s,t]
                             for f in mdl.F for b in mdl.B for k in mdl.K)
            for s in mdl.S for t in mdl.T
        )

        return invcost + wastecost + routecost + assignmentcost + loadcost

    model.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints

    # (2.1) Supplier cap
    def supply_cap_rule(mdl, f, s, t):
        return sum(mdl.L[f,b,k,s,t] for b in mdl.B for k in mdl.K) <= upsilon_dict[(f,s,t)]
    model.SupplyCap = pyo.Constraint(model.F, model.S, model.T, rule=supply_cap_rule)

    # (2.2) Vehicle cap
    def veh_cap_rule(mdl, k, s, t):
        return sum(mdl.L[f,b,k,s,t] for f in mdl.F for b in mdl.B) <= theta_dict[k]
    model.VehicleCap = pyo.Constraint(model.K, model.S, model.T, rule=veh_cap_rule)

    # (2.3) Link L <= Y*theta
    def link_LY_rule(mdl, f, b, k, s, t):
        return mdl.L[f,b,k,s,t] <= mdl.Y[f,b,k,s,t] * theta_dict[k]
    model.LinkLY = pyo.Constraint(model.F, model.B, model.K, model.S, model.T, rule=link_LY_rule)

    # (2.4) One hub per (f,k,s,t): sum_b Y <= 1
    def onehub_rule(mdl, f, k, s, t):
        return sum(mdl.Y[f,b,k,s,t] for b in mdl.B) <= 1
    model.OneHub = pyo.Constraint(model.F, model.K, model.S, model.T, rule=onehub_rule)

    # (2.5) Hub balance
    def hub_balance_rule(mdl, b, s, t):
        return sum(mdl.L[f,b,k,s,t] for f in mdl.F for k in mdl.K) == sum(mdl.Q[b,i,t] for i in mdl.D)
    model.HubBalance = pyo.Constraint(model.B, model.S, model.T, rule=hub_balance_rule)

    # (2.11) Inventory definition
    def inv_def_rule(mdl, i, t):
        return mdl.I[i,t] == \
            sum(mdl.Q[b,i,a] for a in range(1, t+1) for b in mdl.B) - \
            sum(de(i,a) + mdl.W[i,a] for a in range(1, t+1))
    model.InvDef = pyo.Constraint(model.D, model.T, rule=inv_def_rule)

    # (2.12) Ipo >= I
    def ipo_rule(mdl, i, t):
        return mdl.Ipo[i,t] >= mdl.I[i,t]
    model.IpoGeI = pyo.Constraint(model.D, model.T, rule=ipo_rule)

    # (2.13) Waste recursion for t >= m
    def waste_rec_rule(mdl, i, t):
        if t < m:
            return pyo.Constraint.Skip
        a1 = t - m + 1
        dem_sum = sum(de(i,a) for a in range(t-m+2, t+1)) if (t-m+2) <= t else 0
        w_sum   = sum(mdl.W[i,a] for a in range(t-m+2, t)) if (t-m+2) <= (t-1) else 0
        return mdl.W[i,t] >= mdl.I[i,a1] - dem_sum - w_sum
    model.WasteRec = pyo.Constraint(model.D, model.T, rule=waste_rec_rule)

    # (2.14) W=0 for t < m
    def waste_zero_rule(mdl, i, t):
        if t >= m:
            return pyo.Constraint.Skip
        return mdl.W[i,t] == 0
    model.WasteZero = pyo.Constraint(model.D, model.T, rule=waste_zero_rule)

    # (2.15) Service level constraint
    def service_rule(mdl, i, t):
        lhs = sum(mdl.Q[b,i,a] for a in range(1, t+1) for b in mdl.B) - sum(mdl.W[i,a] for a in range(1, t))
        dem_total = sum(de(i,a) for a in range(1, t+1))
        dem_sq = sum(de(i,a)**2 for a in range(1, t+1))
        rhs = dem_total + math.sqrt(dem_sq) * co * Zalpha
        return lhs >= rhs
    model.Service = pyo.Constraint(model.D, model.T, rule=service_rule)

    # (2.16) Z=0 if depot i not visited on route r
    def z_zero_rule(mdl, i, r, t):
        if i in route_to_depots[r]:
            return pyo.Constraint.Skip
        return mdl.Z[i,r,t] == 0
    model.ZZero = pyo.Constraint(model.D, model.R, model.T, rule=z_zero_rule)

    # (2.17) Route cap: sum visited Z <= c[r] * X
    def route_cap_rule(mdl, r, t):
        return sum(mdl.Z[i,r,t] for i in route_to_depots[r]) <= c_dict[r] * mdl.X[r,t]
    model.RouteCap = pyo.Constraint(model.R, model.T, rule=route_cap_rule)

    # (2.18) Link Z -> Q via delta
    def z_to_q_rule(mdl, b, i, t):
        return sum(mdl.Z[i,r,t] * delta_dict[(b,r)] for r in mdl.R) == mdl.Q[b,i,t]
    model.ZtoQ = pyo.Constraint(model.B, model.D, model.T, rule=z_to_q_rule)

    return model


def solve_model(model, solver_name="cplex"):
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            f"Solver '{solver_name}' not available. "
            f"Try installing/setting up CPLEX or use another solver (e.g., glpk/cbc) if supported."
        )
    results = solver.solve(model, tee=True)
    print("Status:", results.solver.status)
    print("Termination:", results.solver.termination_condition)
    print("Objective:", pyo.value(model.OBJ))
    return results


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    file_path = "step1.xlsx"

    # Route structure (your As)
    route_to_depots = {
        1: [25,24,14,13], 2: [16,17,23,20], 3: [10,22,27,30], 4: [28,26,19],
        5: [11,12,29], 6: [8,18,21,15,9], 7: [19,20,14,7,2], 8: [17,15,21,25],
        9: [22,8,6,5,1], 10: [23,13,10,11], 11: [29,26,24,16,4,3], 12: [9,18,30,28,27],
        13: [30,23,5,4,2], 14: [12,15,21,25,28], 15: [7,9,10,16,22], 16: [18,19,17,14,12],
        17: [29,24,8,1,3], 18: [27,20,7,6,4,2], 19: [7,6,5,3,1], 20: [20,21,22,24,26,28],
        21: [23,8,16,13,11], 22: [6,5,4,3,2,1], 23: [17,30,14,9,10,12], 24: [18,19,25,26,27,29],
        25: [11,13,15,20,22,26], 26: [10,16,15,27], 27: [7,13,14,17,18], 28: [9,12,11],
        29: [8,19,23,25,24,21], 30: [18,30,29]
    }

    # Scenario probabilities + constants (OPL)
    pr_dict = {1: 0.3, 2: 0.5, 3: 0.2}
    m = 2
    p = 27.55696873
    co = 0.1
    Zalpha = 1.96

    # Load data
    beta_dict, gamma_dict, theta_dict, h_dict, upsilon_dict, demand_dict, lamda_dict, c_dict, delta_dict = \
        read_and_prepare_data(file_path)

    # Build model
    model = build_model(
        beta_dict, gamma_dict, theta_dict, h_dict, upsilon_dict, demand_dict,
        lamda_dict, c_dict, delta_dict,
        route_to_depots, pr_dict, m, p, co, Zalpha
    )

    # Solve
    solve_model(model, solver_name="cplex")
