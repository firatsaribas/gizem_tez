import pandas as pd
import numpy as np
import math
import os
import time
import pyomo.environ as pyo
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pyomo.environ import (
    ConcreteModel, Set, Var, Constraint, Objective, SolverFactory,
    Binary, NonNegativeIntegers, Integers, minimize, value, TerminationCondition
)

# ============================================================
# 1) DATA STRUCTURES
# ============================================================
@dataclass(frozen=True)
class Route:
    r_id: str
    hub: str
    depots: Tuple[str, ...]
    capacity: float
    fixed_cost: float

@dataclass
class Instance:
    T: List[int]; D: List[str]; B: List[str]; F: List[str]; S: List[str]; K: List[str]
    routes: List[Route]
    mu: Dict[Tuple[str, int], float]                      # demand de[i,t]
    alpha: float; cv: float; shelf_life: int              # m
    holding_cost: Dict[str, float]                        # h[i]
    waste_cost: float                                     # p
    supply: Dict[Tuple[str, str, int], float]             # upsilon[f,s,t]
    theta: Dict[str, float]                               # theta[k]
    beta: Dict[Tuple[str, str], float]                    # beta[f,k]
    gamma: Dict[Tuple[str, str, str], float]              # gama[f,b,k] (only valid FBK)
    scenario_probs: Dict[str, float]                      # pr[s]

# ============================================================
# 2) HELPERS
# ============================================================
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def to_str_id(x) -> str:
    if pd.isna(x):
        return "0"
    if isinstance(x, float) and float(x).is_integer():
        x = int(x)
    return str(x)

def z_lookup(alpha: float) -> float:
    lookup = {0.90: 1.281, 0.95: 1.644, 0.975: 1.959, 0.99: 2.326, 0.995: 2.575}
    return lookup.get(alpha, 2.575)

# ============================================================
# 3) LOAD INSTANCE FROM EXCEL
# ============================================================
def load_instance_from_excel(
    file_path: str,
    r_to_d: Dict[str, List[str]],
    r_to_h: Dict[str, str],
    alpha: float,
    cv: float,
    shelf_life: int,
    waste_cost: float,
    scenario_probs: Dict[int, float]
) -> Instance:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} dosyası bulunamadı!")

    xl = pd.ExcelFile(file_path)

    beta_df = norm_cols(pd.read_excel(xl, sheet_name="beta")).astype({"f": int, "k": int})
    gamma_df = norm_cols(pd.read_excel(xl, sheet_name="gamma")).astype({"f": int, "b": int, "k": int})
    theta_df = norm_cols(pd.read_excel(xl, sheet_name="theta")).astype({"k": int})
    stock_costs_df = norm_cols(pd.read_excel(xl, sheet_name="stock_costs")).astype({"d": int})
    supply_df = norm_cols(pd.read_excel(xl, sheet_name="supply")).astype({"f": int, "s": int, "t": int})
    demand_df = norm_cols(pd.read_excel(xl, sheet_name="demand")).astype({"d": int, "t": int})
    route_costs_df = norm_cols(pd.read_excel(xl, sheet_name="route_costs")).astype({"r": int})
    route_capacity_df = norm_cols(pd.read_excel(xl, sheet_name="route_capacity")).astype({"r": int})

    inst = Instance(
        T=sorted(supply_df["t"].unique().tolist()),
        D=sorted([to_str_id(x) for x in stock_costs_df["d"].unique()]),
        B=sorted([to_str_id(x) for x in gamma_df["b"].unique()]),
        F=sorted([to_str_id(x) for x in beta_df["f"].unique()]),
        S=sorted([to_str_id(x) for x in supply_df["s"].unique()]),
        K=sorted([to_str_id(x) for x in theta_df["k"].unique()]),
        routes=[],
        mu={(to_str_id(r.d), int(r.t)): float(r.demand) for r in demand_df.itertuples(index=False)},
        alpha=alpha, cv=cv, shelf_life=shelf_life,
        holding_cost={to_str_id(r.d): float(r.stock_cost) for r in stock_costs_df.itertuples(index=False)},
        waste_cost=waste_cost,
        supply={(to_str_id(r.f), to_str_id(r.s), int(r.t)): float(r.supply) for r in supply_df.itertuples(index=False)},
        theta={to_str_id(r.k): float(r.theta) for r in theta_df.itertuples(index=False)},
        beta={(to_str_id(r.f), to_str_id(r.k)): float(r.beta) for r in beta_df.itertuples(index=False)},
        gamma={(to_str_id(r.f), to_str_id(r.b), to_str_id(r.k)): float(r.gamma) for r in gamma_df.itertuples(index=False)},
        scenario_probs={to_str_id(s): float(p) for s, p in scenario_probs.items()}
    )

    r_costs = {int(r.r): float(r.cost) for r in route_costs_df.itertuples(index=False)}
    r_caps  = {int(r.r): float(r.capacity) for r in route_capacity_df.itertuples(index=False)}

    for r_id_int in sorted(r_costs.keys()):
        r_id = str(r_id_int)
        if r_id not in r_to_h or r_id not in r_to_d:
            raise KeyError(f"Missing route mapping for r_id={r_id}")
        inst.routes.append(Route(
            r_id=r_id,
            hub=str(r_to_h[r_id]),
            depots=tuple(str(d) for d in r_to_d[r_id]),
            capacity=r_caps[r_id_int],
            fixed_cost=r_costs[r_id_int]
        ))

    return inst

# ============================================================
# 4) BUILD PYOMO MODEL = CPLEX EQUIVALENT
# ============================================================
def build_pyomo_model(inst: Instance, r_to_d: Dict[str, List[str]], r_to_h: Dict[str, str]) -> ConcreteModel:
    m = ConcreteModel()

    # ---- Sets (CPLEX ranges) ----
    m.F = Set(initialize=inst.F)
    m.S = Set(initialize=inst.S)
    m.T = Set(initialize=inst.T)
    m.B = Set(initialize=inst.B)
    m.D = Set(initialize=inst.D)
    m.K = Set(initialize=inst.K)
    m.R = Set(initialize=[r.r_id for r in inst.routes])

    # ---- Fast maps ----
    route_dict = {r.r_id: r for r in inst.routes}
    T_sorted = sorted(inst.T)
    t_min = T_sorted[0]
    mval = int(inst.shelf_life)  # m in CPLEX
    co = float(inst.cv)
    Zalpha = float(z_lookup(inst.alpha))  # CPLEX Zalpha = 2.575

    # ---- FBK valid tuples (CPLEX FBK) ----
    FBK = sorted(list(inst.gamma.keys()))  # (f,b,k) only where gamma exists
    m.FBK = Set(dimen=3, initialize=FBK)

    # ---- Z index: only visited depots per route (sparse = CPLEX 2.16) ----
    Z_pairs = []
    for r in m.R:
        for i in r_to_d.get(r, []):
            if i in inst.D:
                Z_pairs.append((i, r))
    m.ZIDX = Set(dimen=2, initialize=sorted(set(Z_pairs)))

    # ---- Variables (same as CPLEX) ----
    # Y exists only for valid FBK tuples (prevents meaningless Y choices)
    m.Y = Var(m.F, m.B, m.K, m.S, m.T, domain=Binary, initialize=0)
    m.L = Var(m.F, m.B, m.K, m.S, m.T, domain=NonNegativeIntegers, initialize=0)
    m.I   = Var(m.D, m.T, domain=Integers, initialize=0)
    m.Ipo = Var(m.D, m.T, domain=NonNegativeIntegers, initialize=0)
    m.Q   = Var(m.B, m.D, m.T, domain=NonNegativeIntegers, initialize=0)
    m.Z   = Var(m.ZIDX, m.T, domain=NonNegativeIntegers, initialize=0)
    m.W   = Var(m.D, m.T, domain=NonNegativeIntegers, initialize=0)
    m.X   = Var(m.R, m.T, domain=Binary, initialize=0)

    # ============================================================
    # Objective (same structure as CPLEX)
    # ============================================================
    invcost = sum(m.Ipo[i,t] * inst.holding_cost[i] for i in m.D for t in m.T)
    wastecost = sum(m.W[i,t] * inst.waste_cost for i in m.D for t in m.T if int(t) >= mval)
    routecost = sum(route_dict[r].fixed_cost * m.X[r,t] for r in m.R for t in m.T)

    # Use the sets directly and be careful with the IF check
    assignmentcost = sum(
        inst.scenario_probs[s] * inst.beta.get((f, k), 0.0) * m.Y[f, b, k, s, t]
        for t in m.T for s in m.S for f in m.F for k in m.K for b in m.B
    )

    loadcost = sum(
        inst.scenario_probs[s] * inst.gamma.get((f, b, k), 0.0) * m.L[f, b, k, s, t]
        for t in m.T for s in m.S for (f, b, k) in inst.gamma.keys()
    )


    m.obj = Objective(expr=invcost + wastecost + routecost + assignmentcost + loadcost, sense=minimize)

    # Keep for reporting
    m.inv_total = invcost
    m.waste_total = wastecost
    m.route_total = routecost
    m.assign_total = assignmentcost
    m.load_total = loadcost

    # ============================================================
    # Constraints (match your CPLEX model)
    # ============================================================

     # (1) Supply: forall(f,s,t) sum_{b,k} L <= upsilon[f,s,t]
    def supply_limit_rule(model, f, s, t):
        # Sol taraf: L değişkenlerinin toplamı
        lhs = sum(model.L[f, b, k, s, t] for b in model.B for k in model.K)
        # Sağ taraf: upsilon parametresi
        rhs = inst.supply.get((f, s, t), 0)
        return lhs <= rhs

    # Kısıtı modele tek seferde bağlıyoruz
    model.supply_con = pyo.Constraint(model.F, model.S, model.T, rule=supply_limit_rule)


    # (2) Araç Kapasite Kısıtı
    def c_vehicle_capacity_rule(model, k, s, t):
        # Sol Taraf (LHS): k aracının taşıdığı tüm yüklerin toplamı
        # sum_{f in F} sum_{b in B} L[f,b,k,s,t]
        total_load = sum(model.L[f, b, k, s, t] for f in model.F for b in model.B)
        
        # Sağ Taraf (RHS): k aracının kapasite parametresi
        capacity_limit = inst.theta.get(k, 0.0)
        
        return total_load <= capacity_limit

    # Kısıtı modele ekliyoruz
    # İndisler: k (Araç), s (Senaryo), t (Zaman)
    model.c_vehicle_capacity = pyo.Constraint(model.K, model.S, model.T, rule=c_vehicle_capacity_rule)


    # (3) Bağlama Kısıtı: Atama varsa sevkiyat miktarına izin ver
    def c_coupling_rule(model, f, b, k, s, t):
        # Sol Taraf (LHS): Atama kararı * Araç Kapasitesi
        # Y[f,b,k,s,t] binary olduğu için (0 veya 1), 
        # Y=0 ise sonuç 0 olur, Y=1 ise sonuç theta[k] olur.
        lhs = model.Y[f, b, k, s, t] * inst.theta.get(k, 0.0)
        
        # Sağ Taraf (RHS): Sevkiyat miktarı
        rhs = model.L[f, b, k, s, t]
        
        # Kural: Kapasite (atama varsa) >= Sevkiyat Miktarı
        return lhs >= rhs

    # Kısıtı modele ekliyoruz
    # İndisler: f (Tedarikçi), b (Merkez), k (Araç), s (Senaryo), t (Zaman)
    model.c_coupling = pyo.Constraint(model.F, model.B, model.K, model.S, model.T, rule=c_coupling_rule)

    # (4) Tekli Atama Kısıtı: Bir aracın tek bir transfer merkezine atanması
    def c_one_hub_rule(model, f, k, s, t):
        # Sol Taraf (LHS): Aracın tüm transfer merkezleri (B) üzerindeki atama toplamı
        # sum_{b in B} Y[f,b,k,s,t]
        total_assignments = sum(model.Y[f, b, k, s, t] for b in model.B)
        
        # Kural: Toplam atama 1'den büyük olamaz (0 veya 1 olabilir)
        return total_assignments <= 1

    # Kısıtı modele ekliyoruz
    # İndisler: f (Tedarikçi), k (Araç), s (Senaryo), t (Zaman)
    # Not: 'b' üzerinden toplam aldığımız için kısıt indisleri arasında 'model.B' yer almaz.
    model.c_one_hub = pyo.Constraint(model.F, model.K, model.S, model.T, rule=c_one_hub_rule)


    # (5) Merkez Akış Dengesi: Gelen miktar = Giden miktar
    def c_hub_balance_rule(model, b, s, t):
        # Sol Taraf (LHS): Tedarikçilerden merkeze (b) gelen toplam yük
        # sum_{k in K} sum_{f in F} L[f,b,k,s,t]
        total_inflow = sum(model.L[f, b, k, s, t] for k in model.K for f in model.F)
        
        # Sağ Taraf (RHS): Merkezden (b) depolara (D) giden toplam yük
        # sum_{i in D} Q[b,i,t]
        total_outflow = sum(model.Q[b, i, t] for i in model.D)
        
        # Kural: Giriş ve çıkış birbirine eşit olmalı
        return total_inflow == total_outflow

    # Kısıtı modele ekliyoruz
    # İndisler: b (Merkez), s (Senaryo), t (Zaman)
    model.c_hub_balance = pyo.Constraint(model.B, model.S, model.T, rule=c_hub_balance_rule)


    # (6) Inventory balance (2.11) - using the same cumulative form as CPLEX for exact match
    # I[i,t] == sum_{a<=t,b} Q[b,i,a] - sum_{a<=t} (de[i,a] + W[i,a])
    def c_inv(mdl, i, t):
        a_set = [a for a in T_sorted if a <= t]
        inflow = sum(mdl.Q[b,i,a] for b in mdl.B for a in a_set)
        outflow = sum(inst.mu.get((i,int(a)),0.0) + mdl.W[i,a] for a in a_set)
        return mdl.I[i,t] == inflow - outflow
    m.c_inventory = Constraint(m.D, m.T, rule=c_inv)

    # (7) Ipo >= I (2.12)
    def c_ipo(mdl, i, t):
        return mdl.Ipo[i,t] >= mdl.I[i,t]
    m.c_ipo = Constraint(m.D, m.T, rule=c_ipo)

    # (8) Waste (2.13) and (2.14)
    # for t < m: W=0
    def c_waste_zero(mdl, i, t):
        if int(t) < mval:
            return mdl.W[i,t] == 0
        return Constraint.Skip
    m.c_waste_zero = Constraint(m.D, m.T, rule=c_waste_zero)

    # for t >= m: W[i,t] >= I[i,t-m+1] - sum_{a=t-m+2..t} de[i,a] - sum_{a=t-m+2..t-1} W[i,a]
    def c_waste(mdl, i, t):
        t_int = int(t)
        if t_int < mval:
            return Constraint.Skip

        # CPLEX uses t-m+1 as an index in T (T=1..6). We'll do the same.
        tm1 = t_int - mval + 1
        demand_sum = sum(inst.mu.get((i,a),0.0) for a in range(t_int - mval + 2, t_int + 1))
        waste_sum  = sum(mdl.W[i,a] for a in range(t_int - mval + 2, t_int))
        return mdl.W[i,t] >= mdl.I[i, tm1] - demand_sum - waste_sum
    m.c_waste = Constraint(m.D, m.T, rule=c_waste)

    # (9) Service level (2.15):
    def c_service(mdl, i, t):
        t_int = int(t)
        # Denklem (19) sol taraf
        # sum_{a=1..t, b} Q
        inflow = sum(mdl.Q[b, i, a] for b in mdl.B for a in range(1, t_int + 1))
        # sum_{a=1..t-1} W
        waste_past = sum(mdl.W[i, a] for a in range(1, t_int)) 
        
        # Denklem (19) sağ taraf
        demand_sum = sum(inst.mu.get((i, a), 0.0) for a in range(1, t_int + 1))
        demand_sq_sum = sum((inst.mu.get((i, a), 0.0)**2) for a in range(1, t_int + 1))
        
        # C * Zalpha * sqrt(...)
        rhs = demand_sum + co * Zalpha * math.sqrt(demand_sq_sum)
        
        return inflow - waste_past >= rhs
    m.c_service = Constraint(m.D, m.T, rule=c_service)

    # (10) Route capacity (2.17): sum_{i in NR_r} Z[i,r,t] <= c[r]*X[r,t]
    def c_routecap(mdl, r, t):
        load = sum(mdl.Z[(i,r), t] for (i,rr) in mdl.ZIDX if rr == r)
        return load <= route_dict[r].capacity * mdl.X[r,t]
    m.c_routecap = Constraint(m.R, m.T, rule=c_routecap)

    # (11) Route -> Q via hub (2.18): sum_r Z[i,r,t]*delta[b,r] == Q[b,i,t]
    # delta[b,r]=1 if route r starts at hub b else 0
    def c_route_to_q(mdl, b, i, t):
        # Sadece hub'ı 'b' olan rotaları al (delta[b,r] == 1 durumu)
        r_list = [r_id for r_id, hub in r_to_h_manual.items() if hub == str(b)]
        
        # Denklem (13)
        lhs = sum(mdl.Z[i, r, t] for r in r_list if (i, r) in mdl.ZIDX)
        return lhs == mdl.Q[b, i, t]
    m.c_route_to_q = Constraint(m.B, m.D, m.T, rule=c_route_to_q)

    return m

# ============================================================
# 5) RUN
# ============================================================
if __name__ == "__main__":
    start = time.time()

    file_name = "step1.xlsx"

    # Manual route maps (your data)
    r_to_d_manual = {
        1: [25,24,14,13], 2: [16,17,23,20], 3: [10,22,27,30], 4: [28,26,19],
        5: [11,12,29], 6: [8,18,21,15,9], 7: [19,20,14,7,2], 8: [17,15,21,25],
        9: [22,8,6,5,1], 10: [23,13,10,11], 11: [29,26,24,16,4,3], 12: [9,18,30,28,27],
        13: [30,23,5,4,2], 14: [12,15,21,25,28], 15: [7,9,10,16,22], 16: [18,19,17,14,12],
        17: [29,24,8,1,3], 18: [27,20,7,6,4,2], 19: [7,6,5,3,1], 20: [20,21,22,24,26,28],
        21: [23,8,16,13,11], 22: [6,5,4,3,2,1], 23: [17,30,14,9,10,12], 24: [18,19,25,26,27,29],
        25: [11,13,15,20,22,26], 26: [10,16,15,27], 27: [7,13,14,17,18], 28: [9,12,11],
        29: [8,19,23,25,24,21], 30: [18,30,29]
    }

    r_to_h_manual = {
        1: 2, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 2, 8: 3, 9: 2, 10: 1,
        11: 2, 12: 3, 13: 2, 14: 1, 15: 3, 16: 2, 17: 2, 18: 2, 19: 1,
        20: 3, 21: 2, 22: 3, 23: 3, 24: 2, 25: 1, 26: 1, 27: 3, 28: 1,
        29: 1, 30: 1
    }

    # String standardization
    r_to_d_manual = {str(k): [str(d) for d in v] for k, v in r_to_d_manual.items()}
    r_to_h_manual = {str(k): str(v) for k, v in r_to_h_manual.items()}

    print("Excel verileri yukleniyor...")
    inst = load_instance_from_excel(
        file_name, r_to_d_manual, r_to_h_manual,
        alpha=0.995, cv=0.1, shelf_life=2, waste_cost=27.55696873,
        scenario_probs={1: 0.3, 2: 0.5, 3: 0.2}
    )


    print("Pyomo modeli (CPLEX-eşdeğer) olusturuluyor...")
    model = build_pyomo_model(inst, r_to_d_manual, r_to_h_manual)


    print(f"Demand Sözlüğü Boyutu: {len(inst.mu)}")
    print(f"Beta Sözlüğü Boyutu: {len(inst.beta)}")
    print(f"Supply Sözlüğü Boyutu: {len(inst.supply)}")
    # Örnek bir veri çekmeyi dene
    sample_f = list(model.F)[0]
    sample_s = list(model.S)[0]
    sample_t = list(model.T)[0]
    print(f"Örnek Kapasite ({sample_f}, {sample_s}, {sample_t}):", inst.supply.get((sample_f, sample_s, sample_t)))

    print("CPLEX cozumu baslatildi (epgap=0.01, timelimit=900)...")
    solver = SolverFactory("cplex")
    print("CPLEX available?:", solver.available())
    print("CPLEX executable:", solver.executable())

    solver.options["timelimit"] = 900
    solver.options["mip_tolerances_mipgap"] = 0.01

    results = solver.solve(model, tee=True)


    print("\n--- DEBUG: data sizes ---")
    print("len(beta) =", len(inst.beta), " | beta min/max =",
        (min(inst.beta.values()) if inst.beta else None),
        (max(inst.beta.values()) if inst.beta else None))

    print("len(gamma) =", len(inst.gamma), " | gamma min/max =",
        (min(inst.gamma.values()) if inst.gamma else None),
        (max(inst.gamma.values()) if inst.gamma else None))

    print("\n--- DEBUG: variable usage ---")
    max_Q = max(value(model.Q[b,i,t]) for b in model.B for i in model.D for t in model.T)
    print("Max Q =", max_Q)

    # Y full index ise:
    max_Y = max(value(model.Y[f,b,k,s,t]) for f in model.F for b in model.B for k in model.K for s in model.S for t in model.T)
    print("Max Y =", max_Y)

    # L full index ise:
    max_L = max(value(model.L[f,b,k,s,t]) for f in model.F for b in model.B for k in model.K for s in model.S for t in model.T)
    print("Max L =", max_L)

    print("\n" + "=" * 60)
    print(f"Termination: {results.solver.termination_condition}")


    if results.solver.termination_condition in (TerminationCondition.optimal, TerminationCondition.feasible):
        print(f"OBJ   : {value(model.obj):.2f}")
        print(f"INV   : {value(model.inv_total):.2f}")
        print(f"WASTE : {value(model.waste_total):.2f}")
        print(f"ROUTE : {value(model.route_total):.2f}")
        print(f"ASSGN : {value(model.assign_total):.2f}")
        print(f"LOAD  : {value(model.load_total):.2f}")
    else:
        print("Cozum bulunamadi / zaman siniri / infeasible olabilir.")
    print("=" * 60)

    print(f"Toplam sure: {time.time() - start:.1f} sn")
