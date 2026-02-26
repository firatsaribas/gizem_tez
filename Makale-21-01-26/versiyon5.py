# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 20:47:58 2026

@author: gizem
"""

# -*- coding: utf-8 -*- 



from dataclasses import dataclass, field 
from typing import Dict, List, Tuple, Set 
import pandas as pd 
import math 
import time 
  
# ============================================================ 
# Timing helper 
# ============================================================ 

  
class PhaseTimer: 
    def __init__(self, name: str): 
        self.name = name 
        self.t0 = None 

    def __enter__(self): 
        self.t0 = time.perf_counter() 
        print(f"\n[START] {self.name}") 
        return self 

    def __exit__(self, exc_type, exc, tb): 
        dt = time.perf_counter() - self.t0 
        if exc is None: 
            print(f"[DONE]  {self.name} | elapsed={dt:.2f} sec") 
        else: 
            print(f"[FAIL]  {self.name} | elapsed={dt:.2f} sec | error={exc}") 
        return False 
  
# ============================================================ 
# Small helpers (integer-safe) ggg
# ============================================================ 

def int_floor(x: float) -> int: 
    return int(math.floor(x + 1e-9))   

def int_round(x: float) -> int: 
    return int(round(x)) 
  

def clamp01(x: float) -> float: 
    return max(0.0, min(1.0, x)) 
  

# ============================================================ 
# 0) Dataclasses 
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
    T: List[int] 
    D: List[str] 
    B: List[str] 
    F: List[str] 
    S: List[str] 
    K: List[str] 
    routes: List[Route] 

    mu: Dict[Tuple[str, int], float]      # mean demand (parameter) 
    alpha: float 
    cv: float 
    shelf_life: int 

    holding_cost: Dict[str, float]        # h_i 
    waste_cost: float                     # p  

    supply: Dict[Tuple[str, str, int], float]      # v_{f,s,t} 
    allowed_f_b: Set[Tuple[str, str]]              # admissible arcs 
    theta: Dict[str, float]                        # theta_k   

    beta: Dict[Tuple[str, str], float]             # beta_{f,k} 
    gamma: Dict[Tuple[str, str, str], float]       # gamma_{f,b,k}   

    scenario_probs: Dict[str, float]               # pr[s] 

  

@dataclass 

class FFHSolution: 
    # downstream (scenario-independent) 
    X: Dict[Tuple[str, int], int] = field(default_factory=dict)                 # (r,t) in {0,1} 
    Z: Dict[Tuple[str, str, int], int] = field(default_factory=dict)            # (i,r,t) int+ 
    Q: Dict[Tuple[str, str, int], int] = field(default_factory=dict)            # (b,i,t) int+ 

    # upstream (scenario-dependent) 
    Y: Dict[Tuple[str, str, str, str, int], int] = field(default_factory=dict)  # (f,b,k,s,t) in {0,1} 
    L: Dict[Tuple[str, str, str, str, int], int] = field(default_factory=dict)  # (f,b,k,s,t) int+  
    
    # inventory/waste trajectories 
    Ipos: Dict[Tuple[str, int], int] = field(default_factory=dict)              # (i,t) int+ 
    W: Dict[Tuple[str, int], int] = field(default_factory=dict)                 # (i,t) int+   

    # cumulative trackers (for qmin) 
    Q_cum: Dict[Tuple[str, int], int] = field(default_factory=dict)             # (i,t) int+ 
    W_cum: Dict[Tuple[str, int], int] = field(default_factory=dict)             # (i,t) int+   

    # cost breakdown 
    cost_breakdown: Dict[str, float] = field(default_factory=dict) 
    cost_breakdown_opl: Dict[str, float] = field(default_factory=dict) 

    

# ============================================================ 
# 1) Excel Loader (robust) 
# ============================================================ 

def norm_cols(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    df.columns = [str(c).strip().lower() for c in df.columns] 
    return df   

def assert_cols(df: pd.DataFrame, needed: List[str], name: str) -> None: 
    missing = [c for c in needed if c not in df.columns] 
    if missing: 
        raise ValueError(f"[{name}] missing columns: {missing}. Existing: {list(df.columns)}") 

  
def to_str_id(x) -> str: 
    if pd.isna(x): 
        raise ValueError("Found NaN in an ID field.") 
    if isinstance(x, float) and float(x).is_integer(): 
        x = int(x) 
    return str(x) 

  

def z_value(alpha: float) -> float: 
    lookup = { 
        0.90: 1.281551566, 
        0.95: 1.644853627, 
        0.975: 1.959963985, 
        0.99: 2.326347874, 
        0.995: 2.575829304 
    } 

    if alpha in lookup: 
        return lookup[alpha] 

  
    # mild fallback approximation 
    x = 2 * alpha - 1 
    ln = math.log(1 - x * x) 
    tt = 2 / (math.pi * 0.147) + ln / 2 
    erfinv = math.copysign(math.sqrt(max(0.0, math.sqrt(tt * tt - ln / 0.147) - tt)), x) 

    return math.sqrt(2) * erfinv 

  

  

def load_instance_from_excel( 
    file_path: str, 
    route_to_depots: Dict[int, List[int]], 
    route_to_hub: Dict[int, int], 
    *, 
    alpha: float, 
    cv: float, 
    shelf_life: int, 
    waste_cost: float, 
    scenario_probs: Dict[int, float], 

) -> Instance:   

    with PhaseTimer("Phase 0 - Data loading (Excel -> Instance)"): 
        beta_df = norm_cols(pd.read_excel(file_path, sheet_name="beta")) 
        gamma_df = norm_cols(pd.read_excel(file_path, sheet_name="gamma")) 
        theta_df = norm_cols(pd.read_excel(file_path, sheet_name="theta")) 
        stock_costs_df = norm_cols(pd.read_excel(file_path, sheet_name="stock_costs")) 
        supply_df = norm_cols(pd.read_excel(file_path, sheet_name="supply")) 
        demand_df = norm_cols(pd.read_excel(file_path, sheet_name="demand")) 
        route_costs_df = norm_cols(pd.read_excel(file_path, sheet_name="route_costs")) 
        route_capacity_df = norm_cols(pd.read_excel(file_path, sheet_name="route_capacity")) 


        assert_cols(beta_df, ["f", "k", "beta"], "beta") 
        assert_cols(gamma_df, ["f", "b", "k", "gamma"], "gamma") 
        assert_cols(theta_df, ["k", "theta"], "theta") 
        assert_cols(stock_costs_df, ["d", "stock_cost"], "stock_costs") 
        assert_cols(supply_df, ["f", "s", "t", "supply"], "supply") 
        assert_cols(demand_df, ["d", "t", "demand"], "demand") 
        assert_cols(route_costs_df, ["r", "cost"], "route_costs") 
        assert_cols(route_capacity_df, ["r", "capacity"], "route_capacity") 

  
        # types 
        beta_df = beta_df.astype({"f": int, "k": int}) 
        gamma_df = gamma_df.astype({"f": int, "b": int, "k": int}) 
        theta_df = theta_df.astype({"k": int}) 
        stock_costs_df = stock_costs_df.astype({"d": int}) 
        supply_df = supply_df.astype({"f": int, "s": int, "t": int}) 
        demand_df = demand_df.astype({"d": int, "t": int}) 
        route_costs_df = route_costs_df.astype({"r": int}) 
        route_capacity_df = route_capacity_df.astype({"r": int}) 

  
        # dicts (string IDs in instance) 
        beta = {(to_str_id(r.f), to_str_id(r.k)): float(r.beta) for r in beta_df.itertuples(index=False)} 
        gamma = {(to_str_id(r.f), to_str_id(r.b), to_str_id(r.k)): float(r.gamma) for r in gamma_df.itertuples(index=False)} 
        theta = {to_str_id(r.k): float(r.theta) for r in theta_df.itertuples(index=False)} 
        holding_cost = {to_str_id(r.d): float(r.stock_cost) for r in stock_costs_df.itertuples(index=False)} 
        supply = {(to_str_id(r.f), to_str_id(r.s), int(r.t)): float(r.supply) for r in supply_df.itertuples(index=False)} 
        mu = {(to_str_id(r.d), int(r.t)): float(r.demand) for r in demand_df.itertuples(index=False)} 

        route_costs_dict = {int(r.r): float(r.cost) for r in route_costs_df.itertuples(index=False)} 
        route_capacity_dict = {int(r.r): float(r.capacity) for r in route_capacity_df.itertuples(index=False)} 
        R_set = sorted(route_costs_dict.keys()) 

 

        # sets 
        F = sorted({to_str_id(x) for x in beta_df["f"].unique()}) 
        K = sorted({to_str_id(x) for x in beta_df["k"].unique()}) 
        B = sorted({to_str_id(x) for x in gamma_df["b"].unique()}) 
        D = sorted({to_str_id(x) for x in stock_costs_df["d"].unique()}) 
        S = sorted({to_str_id(x) for x in supply_df["s"].unique()}) 
        T = sorted({int(x) for x in supply_df["t"].unique()}) 

  
        # check route dictionaries coverage 
        if set(route_to_depots.keys()) != set(R_set): 
            raise ValueError("Route ID mismatch between Excel route_costs and route_to_depots keys.") 

        if set(route_to_hub.keys()) != set(R_set): 
            raise ValueError("Route ID mismatch between Excel route_costs and route_to_hub keys.") 
  

        # allowed arcs 
        allowed_f_b = {(f, b) for (f, b, k) in gamma.keys()} 

  
        # routes list 
        routes: List[Route] = [] 
        for r in R_set: 
            r_id = str(r) 
            hub = str(route_to_hub[r]) 
            depots = tuple(str(int(d)) for d in route_to_depots[r]) 
            cap = float(route_capacity_dict[r]) 
            fixed = float(route_costs_dict[r]) 
            routes.append(Route(r_id=r_id, hub=hub, depots=depots, capacity=cap, fixed_cost=fixed)) 

  
        # scenario_probs to string keys 
        sp = {to_str_id(s): float(p) for s, p in scenario_probs.items()} 


        inst = Instance( 
            T=T, D=D, B=B, F=F, S=S, K=K, routes=routes, 
            mu=mu, alpha=float(alpha), cv=float(cv), shelf_life=int(shelf_life), 
            holding_cost=holding_cost, waste_cost=float(waste_cost), 
            supply=supply, allowed_f_b=allowed_f_b, theta=theta, 
            beta=beta, gamma=gamma, scenario_probs=sp 
        ) 

        print("Instance sizes:", 
              f"|F|={len(inst.F)}, |K|={len(inst.K)}, |B|={len(inst.B)}, |D|={len(inst.D)}, " 
              f"|S|={len(inst.S)}, |T|={len(inst.T)}, |R|={len(inst.routes)}") 
        return inst 

  

  

# ============================================================ 
# 2) FFH Core computations 
# ============================================================ 

  

def precompute_REQ(inst: Instance) -> Dict[Tuple[str, int], int]:
    """
    REQ_{i,t} = sum_{a<=t} mu_{i,a} + z * sqrt(sum_{a<=t} sigma_{i,a}^2)
    with sigma_{i,a} = cv * mu_{i,a}

    REQ is rounded UP to integer (service-level consistent)
    """

    z = z_value(inst.alpha)
    REQ: Dict[Tuple[str, int], int] = {}

    for i in inst.D:
        cum_mu = 0.0
        cum_var = 0.0

        for t in inst.T:
            m = inst.mu[(i, t)]
            s = inst.cv * m

            cum_mu += m
            cum_var += s * s

            req_value = cum_mu + z * math.sqrt(cum_var)
            REQ[(i, t)] = math.ceil(req_value)

    return REQ

  

  

def fifo_update_int(age_buckets: List[int], receipt: int, demand: int) -> Tuple[List[int], int, int]: 
    """ 
    age_buckets: [age0 newest, ..., age(m-1) oldest] (all int) 
    - Add receipt to age0 
    - Serve demand FIFO (oldest first) 
    - Waste: whatever remains in oldest bucket after serving, then it expires. 
    - Age shift: buckets move one step older (implemented by shifting after waste removed) 
    Returns: (new_buckets, waste, Ipos_end) 
    """ 
    m = len(age_buckets)   
    age_buckets[0] += max(0, int(receipt))   
    rem = max(0, int(demand)) 
    
    for a in range(m - 1, -1, -1): 
        if rem <= 0: 
            break 

        take = min(age_buckets[a], rem) 
        age_buckets[a] -= take 
        rem -= take  

    waste = age_buckets[m - 1] 
    age_buckets[m - 1] = 0  

    for a in range(m - 1, 0, -1): 
        age_buckets[a] = age_buckets[a - 1] 
    age_buckets[0] = 0   

    Ipos = sum(age_buckets) 
    return age_buckets, int(waste), int(Ipos) 

  

  

def greedy_route_selection_and_loading_int(
    inst: Instance,
    t: int,
    qmin: Dict[str, float],
) -> Tuple[
    Dict[Tuple[str, int], int],
    Dict[Tuple[str, str, int], int],
    Dict[Tuple[str, str, int], int]
]:
    """
    Route selection (parametresiz iyileştirme):
      - Route score = expected_load / fixed_cost  (maximize)
        expected_load = min(route_capacity, sum_{i in cover} qmin[i])
      - Then load integer Z by descending remaining need.
      - Induce Q as usual.
    """

    # depots with remaining need
    U = {i for i, v in qmin.items() if v > 1e-9}

    X_t: Dict[Tuple[str, int], int] = {}
    Z_t: Dict[Tuple[str, str, int], int] = {}
    Q_t: Dict[Tuple[str, str, int], int] = {(b, i, t): 0 for b in inst.B for i in inst.D}

    r_deps = {r.r_id: set(r.depots) for r in inst.routes}
    r_cost = {r.r_id: float(r.fixed_cost) for r in inst.routes}
    r_cap  = {r.r_id: int_floor(r.capacity) for r in inst.routes}
    r_hub  = {r.r_id: r.hub for r in inst.routes}

    # 1) greedy selection
    selected: List[str] = []
    while U:
        best_r = None
        best_score = -1.0

        for r in inst.routes:
            cover = r_deps[r.r_id] & U
            if not cover:
                continue

            cover_sum = sum(float(qmin[i]) for i in cover)
            if cover_sum <= 1e-12:
                continue

            exp_load = min(float(r_cap[r.r_id]), cover_sum)  # no new param
            score = exp_load / max(1e-12, r_cost[r.r_id])    # load per euro

            if score > best_score:
                best_score = score
                best_r = r.r_id

        if best_r is None:
            break

        selected.append(best_r)
        X_t[(best_r, t)] = 1

        # remove covered depots (set-cover style)
        U = U - r_deps[best_r]

    # 2) load selected routes (int), prioritizing highest remaining need
    rem = {i: float(qmin.get(i, 0.0)) for i in inst.D}

    for r_id in selected:
        cap = int(r_cap[r_id])
        deps = [i for i in r_deps[r_id] if rem[i] > 1e-9]
        deps.sort(key=lambda i: rem[i], reverse=True)

        for i in deps:
            if cap <= 0:
                break
            x = min(rem[i], cap)
            x_int = int_floor(x)
            if x_int > 0:
                Z_t[(i, r_id, t)] = Z_t.get((i, r_id, t), 0) + x_int
                rem[i] -= x_int
                cap -= x_int

    # 3) induce Q
    for (i, r_id, tt), z in Z_t.items():
        b = r_hub[r_id]
        Q_t[(b, i, tt)] += int(z)

    return X_t, Z_t, Q_t

  

  
def inbound_assign_greedy_one_scenario_int(
    inst: Instance,
    s: str,
    t: int,
    OUT_bt: Dict[str, int],
) -> Tuple[
    Dict[Tuple[str, str, str, str, int], int],
    Dict[Tuple[str, str, str, str, int], int],
    Dict[str, int]
]:
    """
    Parametresiz iyileştirme hedefleri:
      - Y sayısını azalt (assignmentcost ↓): büyük gönderimi ucuza yapan kombinasyonları seç.
      - Seçim metriği: unit_cost = (beta + gamma*ship) / ship
      - Aynı hub b için önce zaten kapasitesi kalan "aktif" araçları doldurmaya yatkın seçim.
      - Supplier yine sadece 1 kez kullanılacak (used_f).
    """

    cap_left = {k: int_floor(inst.theta[k]) for k in inst.K}
    used_f: Set[str] = set()

    Y_s: Dict[Tuple[str, str, str, str, int], int] = {}
    L_s: Dict[Tuple[str, str, str, str, int], int] = {}
    achieved = {b: 0 for b in inst.B}

    # hubs ordered by descending requirement
    hubs_sorted = sorted(inst.B, key=lambda b: OUT_bt.get(b, 0), reverse=True)

    for b in hubs_sorted:
        remaining = int(OUT_bt.get(b, 0))
        if remaining <= 0:
            continue

        # Build candidate suppliers for this hub
        cand_f = []
        for f in inst.F:
            if f in used_f:
                continue

            v = int_floor(inst.supply.get((f, s, t), 0.0))
            if v <= 0:
                continue

            if (f, b) not in inst.allowed_f_b:
                continue

            # We don't sort purely by gamma anymore; we'll evaluate by unit_cost in (f,k).
            cand_f.append((f, v))

        # Optional: try larger suppliers first to naturally reduce number of Y (no new param)
        cand_f.sort(key=lambda x: x[1], reverse=True)

        # Track "active vehicles" within this (b,s,t): vehicles already used for this hub
        active_k_for_b: Set[str] = set()

        for f, v in cand_f:
            if remaining <= 0:
                break
            if f in used_f:
                continue

            need_from_f = min(v, remaining)
            if need_from_f <= 0:
                continue

            best_k = None
            best_ship = 0
            best_unit = float("inf")
            best_total = float("inf")

            # Prefer filling already-used vehicles first (no new param):
            # - we evaluate active vehicles first, then all others
            k_order = list(active_k_for_b) + [k for k in inst.K if k not in active_k_for_b]

            for k in k_order:
                if cap_left[k] <= 0:
                    continue
                if (f, k) not in inst.beta:
                    continue
                if (f, b, k) not in inst.gamma:
                    continue

                ship = min(need_from_f, cap_left[k])
                if ship <= 0:
                    continue

                total_cost = float(inst.beta[(f, k)]) + float(inst.gamma[(f, b, k)]) * float(ship)
                unit_cost = total_cost / float(ship)

                # Primary: minimize unit_cost
                # Tie-break 1: maximize ship (fewer Y)
                # Tie-break 2: minimize total_cost
                if (unit_cost < best_unit - 1e-12 or
                    (abs(unit_cost - best_unit) <= 1e-12 and ship > best_ship) or
                    (abs(unit_cost - best_unit) <= 1e-12 and ship == best_ship and total_cost < best_total)):
                    best_unit = unit_cost
                    best_k = k
                    best_ship = int(ship)
                    best_total = total_cost

            if best_k is None or best_ship <= 0:
                continue

            # commit
            Y_s[(f, b, best_k, s, t)] = 1
            L_s[(f, b, best_k, s, t)] = int(best_ship)

            cap_left[best_k] -= int(best_ship)
            used_f.add(f)
            active_k_for_b.add(best_k)

            achieved[b] += int(best_ship)
            remaining -= int(best_ship)

    return Y_s, L_s, achieved


  

def scale_Q_int_to_target(sol: FFHSolution, inst: Instance, b: str, t: int, target_out: int) -> None: 
    """ 
    Reduce Q_{b,i,t} so that sum_i Q = target_out, keeping all Q int >=0. 
    No increase allowed. 
    """ 
    cur_vals = [(i, int(sol.Q.get((b, i, t), 0))) for i in inst.D] 
    cur_total = sum(v for _, v in cur_vals) 

    if cur_total <= target_out: 
        return 
    if target_out <= 0: 
        for i, _ in cur_vals: 
            sol.Q[(b, i, t)] = 0 
        return 

    ratio = target_out / cur_total 

    base = [] 
    rema = [] 
    base_sum = 0 


    for i, v in cur_vals: 
        x = v * ratio 
        bi = int_floor(x) 
        base.append((i, bi)) 
        base_sum += bi 
        rema.append((x - bi, i)) 

    leftover = target_out - base_sum 
    rema.sort(reverse=True) 
  
    add = {i: 0 for i in inst.D} 
    for k in range(max(0, leftover)): 
        _, i = rema[k] 
        add[i] += 1 


    for i, bi in base: 

        sol.Q[(b, i, t)] = int(bi + add[i]) 

  

  

def sync_Z_from_Q_int(sol: FFHSolution, inst: Instance, t: int) -> None: 

    """ 

    Make Z consistent with Q: 

      For each hub b and depot i, sum_{r:hub=b} Z_{i,r,t} == Q_{b,i,t}. 

    We scale existing Z proportionally within each (b,i). 

    """ 

    routes_by_hub: Dict[str, List[Route]] = {} 

    for r in inst.routes: 

        routes_by_hub.setdefault(r.hub, []).append(r) 

  

    for b, routes in routes_by_hub.items(): 

        for i in inst.D: 

            q_target = int(sol.Q.get((b, i, t), 0)) 

  

            z_keys = [] 

            cur_total = 0 

            for r in routes: 

                k = (i, r.r_id, t) 

                if k in sol.Z: 

                    z_keys.append(k) 

                    cur_total += int(sol.Z.get(k, 0)) 

  

            if cur_total == q_target: 

                continue 

  

            if cur_total == 0: 

                # nothing to distribute 

                continue 

  

            if q_target <= 0: 

                for k in z_keys: 

                    sol.Z[k] = 0 

                continue 

  

            ratio = q_target / cur_total 

  

            base = [] 

            rema = [] 

            base_sum = 0 

  

            for k in z_keys: 

                v = int(sol.Z.get(k, 0)) 

                x = v * ratio 

                bk = int_floor(x) 

                base.append((k, bk)) 

                base_sum += bk 

                rema.append((x - bk, k)) 

  

            leftover = q_target - base_sum 

            rema.sort(reverse=True) 

            add = {k: 0 for k in z_keys} 

  

            for kk in range(max(0, leftover)): 

                _, k = rema[kk] 

                add[k] += 1 

  

            for k, bk in base: 

                sol.Z[k] = int(bk + add[k]) 

  

  

# ============================================================ 

# 3) Cost functions (float costs on int decisions) 

# ============================================================ 

  

def downstream_cost(inst: Instance, sol: FFHSolution) -> Tuple[float, float, float, float]: 

    """ 

    FFH-style downstream cost (you can compare with OPL, but note: 

      OPL wastecost uses t >= m (as in your OPL code). 

    Here we keep FFH-style also aligned with that (t >= m). 

    """ 

    RC = 0.0 

    for t in inst.T: 

        for r in inst.routes: 

            RC += r.fixed_cost * float(sol.X.get((r.r_id, t), 0)) 

  

    HC = 0.0 

    for i in inst.D: 

        hi = inst.holding_cost[i] 

        for t in inst.T: 

            HC += hi * float(sol.Ipos.get((i, t), 0)) 

  

    WC = 0.0 

    for i in inst.D: 

        for t in inst.T: 

            if t >= inst.shelf_life:  # OPL: t >= m 

                WC += inst.waste_cost * float(sol.W.get((i, t), 0)) 

  

    return RC, HC, WC, (RC + HC + WC) 

  

  

def upstream_cost_expected_OPL(inst: Instance, sol: FFHSolution) -> Tuple[float, float, float]: 

    """ 

    EXACTLY OPL objective definitions: 

  

    assignmentcost = sum(t,s) pr[s] * sum(f,k,b) beta[f][k] * Y[f][b][k][s][t] 

    loadcost       = sum(t,s) pr[s] * sum(f,k,b) gamma[f,b,k] * L[f][b][k][s][t] 

  

    IMPORTANT: NO deduplication (your previous version undercounted AC). 

    """ 

    AC = 0.0 

    TC = 0.0 

  

    # AC from Y 

    for (f, b, k, s, t), y in sol.Y.items(): 

        if int(y) != 1: 

            continue 

        pr = inst.scenario_probs.get(s, 0.0) 

        if pr <= 0: 

            continue 

        AC += pr * float(inst.beta[(f, k)]) * 1.0 

  

    # TC from L 

    for (f, b, k, s, t), qty in sol.L.items(): 

        if int(qty) <= 0: 

            continue 

        pr = inst.scenario_probs.get(s, 0.0) 

        if pr <= 0: 

            continue 

        TC += pr * float(inst.gamma[(f, b, k)]) * float(qty) 

  

    return AC, TC, (AC + TC) 

  

  

def objective_cost_OPL(inst: Instance, sol: FFHSolution) -> Dict[str, float]: 

    """ 

    Returns OPL-style components: 

      invcost, wastecost, routecost, assignmentcost, loadcost, total 

    """ 

    invcost = 0.0 

    for i in inst.D: 

        hi = inst.holding_cost[i] 

        for t in inst.T: 

            invcost += float(sol.Ipos.get((i, t), 0)) * hi 

  

    wastecost = 0.0 

    for i in inst.D: 

        for t in inst.T: 

            if t >= inst.shelf_life:  # OPL: t >= m 

                wastecost += float(sol.W.get((i, t), 0)) * float(inst.waste_cost) 

  

    routecost = 0.0 

    for r in inst.routes: 

        for t in inst.T: 

            routecost += float(sol.X.get((r.r_id, t), 0)) * float(r.fixed_cost) 

  

    assignmentcost, loadcost, _ = upstream_cost_expected_OPL(inst, sol) 

    total = invcost + wastecost + routecost + assignmentcost + loadcost 

  

    return { 

        "invcost": invcost, 

        "wastecost": wastecost, 

        "routecost": routecost, 

        "assignmentcost": assignmentcost, 

        "loadcost": loadcost, 

        "total": total 

    } 

  

  

# ============================================================ 

# 4) Checks (REQ + flow balance) 

# ============================================================ 

  

def check_service_level_REQ(inst: Instance, sol: FFHSolution, REQ: Dict[Tuple[str, int], float]) -> pd.DataFrame: 

    """ 

    Check cumulative feasibility: 

      Q_cum(i,t) - W_cum(i,t) >= REQ(i,t)   (deterministic surrogate) 

    """ 

    rows = [] 

    for i in inst.D: 

        for t in inst.T: 

            qc = float(sol.Q_cum.get((i, t), 0)) 

            wc = float(sol.W_cum.get((i, t), 0)) 

            lhs = qc - wc 

            req = float(REQ[(i, t)]) 

            slack = lhs - req 

            rows.append({ 

                "i": i, "t": t, 

                "Q_cum": int(sol.Q_cum.get((i, t), 0)), 

                "W_cum": int(sol.W_cum.get((i, t), 0)), 

                "QminusW": lhs, 

                "REQ": req, 

                "slack": slack, 

                "feasible": int(slack >= -1e-6) 

            }) 

    df = pd.DataFrame(rows).sort_values(["i", "t"]) 

    n_bad = int((df["feasible"] == 0).sum()) 

    if n_bad == 0: 

        print("[CHECK-A] Service-level cumulative constraint: OK (all depots, all periods).") 

    else: 

        print(f"[CHECK-A] Service-level cumulative constraint: VIOLATIONS = {n_bad} rows.") 

    return df 

  

  

def check_flow_balance(inst: Instance, sol: FFHSolution) -> pd.DataFrame: 

    """ 

    Check flow balance per (b,s,t): 

      sum_{f,k} L[f,b,k,s,t] == sum_i Q[b,i,t] 

    """ 

    rows = [] 

    ok = True 

    for s in inst.S: 

        for t in inst.T: 

            for b in inst.B: 

                out_bt = sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D) 

                in_bst = sum( 

                    int(v) for (f, bb, k, ss, tt), v in sol.L.items() 

                    if ss == s and tt == t and bb == b 

                ) 

                gap = in_bst - out_bt 

                if gap != 0: 

                    ok = False 

                rows.append({"b": b, "s": s, "t": t, "IN": in_bst, "OUT": out_bt, "gap(IN-OUT)": gap}) 

    df = pd.DataFrame(rows).sort_values(["s", "t", "b"]) 

    if ok: 

        print("[CHECK-B] Flow balance: OK (all hubs, scenarios, periods).") 

    else: 

        bad = int((df["gap(IN-OUT)"] != 0).sum()) 

        print(f"[CHECK-B] Flow balance: VIOLATIONS = {bad} rows.") 

    return df 

  

  

# ============================================================ 

# 5) FFH main solve 

# ============================================================ 

  

def solve_ffh(inst: Instance) -> Tuple[FFHSolution, Dict[Tuple[str, int], float]]: 

    sol = FFHSolution() 

  

    # init trackers: t=0 baseline 

    for i in inst.D: 

        sol.Q_cum[(i, 0)] = 0 

        sol.W_cum[(i, 0)] = 0 

  

    # inventory age buckets (int) 

    age = {i: [0 for _ in range(inst.shelf_life)] for i in inst.D} 

  

    with PhaseTimer("Precompute REQ_{i,t} (chance surrogate)"): 

        REQ = precompute_REQ(inst) 

  

    # period loop 

    for t in inst.T: 

        # -------- Phase 1 

        with PhaseTimer(f"Phase 1 (t={t}) - sizing + route selection/loading"): 

            qmin = {} 

            for i in inst.D: 

                Qprev = sol.Q_cum[(i, t - 1)] 

                Wprev = sol.W_cum[(i, t - 1)] 

                qmin[i] = max(0.0, REQ[(i, t)] + float(Wprev) - float(Qprev)) 

  

            X_t, Z_t, Q_t = greedy_route_selection_and_loading_int(inst, t, qmin) 

  

            # store 

            for (r_id, tt), v in X_t.items(): 

                sol.X[(r_id, tt)] = int(v) 

  

            for (i, r_id, tt), v in Z_t.items(): 

                if v > 0: 

                    sol.Z[(i, r_id, tt)] = int(v) 

  

            # ensure full keys exist for Q at (b,i,t) 

            for b in inst.B: 

                for i in inst.D: 

                    sol.Q[(b, i, t)] = int(Q_t.get((b, i, t), 0)) 

  

            selected_count = sum(sol.X.get((r.r_id, t), 0) for r in inst.routes) 

            print(f"Phase 1 completed (t={t}). Selected routes: {selected_count}") 

  

        # -------- Phase 2 

        with PhaseTimer(f"Phase 2 (t={t}) - robust inbound feasibility + scaling"): 

            OUT = {b: sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D) for b in inst.B} 

            # ✅ 1) OUT (before) burada yazdır
            print("t=", t, "OUT (before)=", OUT)

            # scenario-wise achievable amounts 

            achieved_s = {(b, s): 0 for b in inst.B for s in inst.S} 

            for s in inst.S: 

                _, _, ach = inbound_assign_greedy_one_scenario_int(inst, s, t, OUT) 

                for b in inst.B: 

                    achieved_s[(b, s)] = int(ach[b]) 

  

            # robust target per hub: min across scenarios 

            OUT_target = {b: min(achieved_s[(b, s)] for s in inst.S) for b in inst.B} 

            # ✅ 2) OUT_target burada yazdır
            print("t=", t, "OUT_target (after robust min)=", OUT_target)

  

            # enforce Q exactly to OUT_target (int) 

            for b in inst.B: 

                scale_Q_int_to_target(sol, inst, b, t, int(OUT_target[b])) 

  

            # make Z consistent with new Q (int) 

            sync_Z_from_Q_int(sol, inst, t) 

  

            # clear all (s,t) upstream entries before writing 

            keysY_del = [k for k in list(sol.Y.keys()) if k[4] == t]  # (f,b,k,s,t) 

            for k in keysY_del: 

                del sol.Y[k] 

            keysL_del = [k for k in list(sol.L.keys()) if k[4] == t] 

            for k in keysL_del: 

                del sol.L[k] 

  

            # rebuild inbound for each scenario to match OUT_target EXACTLY (if possible) 

            OUT_final = {b: int(OUT_target[b]) for b in inst.B} 

  

            # generate (and if still fails in any scenario, tighten robustly and redo once) 

            min_ach = {b: OUT_final[b] for b in inst.B} 

            temp_store = {} 

  

            for s in inst.S: 

                Y_s, L_s, ach2 = inbound_assign_greedy_one_scenario_int(inst, s, t, OUT_final) 

                temp_store[s] = (Y_s, L_s, ach2) 

                for b in inst.B: 

                    min_ach[b] = min(min_ach[b], int(ach2[b])) 

  

            # If any scenario couldn't meet OUT_final, tighten and enforce once 

            if any(min_ach[b] < OUT_final[b] for b in inst.B): 

                OUT_final = {b: int(min_ach[b]) for b in inst.B} 

                for b in inst.B: 

                    scale_Q_int_to_target(sol, inst, b, t, OUT_final[b]) 

                sync_Z_from_Q_int(sol, inst, t) 

  

                # rebuild again for exact OUT_final 

                temp_store = {} 

                for s in inst.S: 

                    Y_s, L_s, ach2 = inbound_assign_greedy_one_scenario_int(inst, s, t, OUT_final) 

                    temp_store[s] = (Y_s, L_s, ach2) 

  

            # write final Y/L 

            for s in inst.S: 

                Y_s, L_s, _ = temp_store[s] 

                for k, v in Y_s.items(): 

                    sol.Y[k] = int(v) 

                for k, v in L_s.items(): 

                    sol.L[k] = int(v) 

  

            # report robust ratio 

            ratios = [] 

            for b in inst.B: 

                denom = OUT[b] 

                if denom <= 0: 

                    ratios.append(1.0) 

                else: 

                    ratios.append(OUT_final[b] / denom) 

            print(f"Phase 2 completed (t={t}). Min robust OUT ratio: {round(min(ratios), 4)}") 

  

        # -------- Phase 3 

        with PhaseTimer(f"Phase 3 (t={t}) - inventory aging & waste update (FIFO)"): 

            for i in inst.D: 

                receipt = sum(int(sol.Q.get((b, i, t), 0)) for b in inst.B) 
                if i == "19" and t == 1:
                    print("DEBUG mu[19,1] =", inst.mu[(i, t)])

                demand_int = int_round(inst.mu[(i, t)]) 

  

                age[i], waste, Ipos = fifo_update_int(age[i], receipt, demand_int) 

  

                sol.W[(i, t)] = int(waste) 

                sol.Ipos[(i, t)] = int(Ipos) 

  

                sol.Q_cum[(i, t)] = int(sol.Q_cum[(i, t - 1)] + receipt) 

                sol.W_cum[(i, t)] = int(sol.W_cum[(i, t - 1)] + waste) 

  

            print(f"Phase 3 completed (t={t}).") 

  

    # -------- Cost breakdowns 

    with PhaseTimer("Cost evaluation (breakdown)"): 

        # FFH-aligned (still OPL-like for waste t>=m) 

        RC, HC, WC, DOWN = downstream_cost(inst, sol) 

        AC_opl, TC_opl, UP_opl = upstream_cost_expected_OPL(inst, sol) 

        TOT = DOWN + UP_opl 

  

        sol.cost_breakdown = { 

            "RC": RC, "HC": HC, "WC": WC, "DOWN": DOWN, 

            "AC": AC_opl, "TC": TC_opl, "UP": UP_opl, 

            "TOTAL": TOT 

        } 

  

        opl = objective_cost_OPL(inst, sol) 

        sol.cost_breakdown_opl = opl 

  

        print("\n" + "=" * 75) 

        print("FFH COST BREAKDOWN (using OPL-style upstream, OPL-style waste t>=m)") 

        print("=" * 75) 

        print(f"Route cost (RC)      : {RC:.2f}") 

        print(f"Holding cost (HC)    : {HC:.2f}") 

        print(f"Waste cost (WC)      : {WC:.2f}") 

        print(f"Downstream total     : {DOWN:.2f}") 

        print("-" * 75) 

        print(f"Assignment cost (AC) : {AC_opl:.2f}") 

        print(f"Transp. cost (TC)    : {TC_opl:.2f}") 

        print(f"Upstream total       : {UP_opl:.2f}") 

        print("-" * 75) 

        print(f"TOTAL COST           : {TOT:.2f}") 

        print("=" * 75) 

  

        print("\n" + "=" * 75) 

        print("OPL-STYLE OBJECTIVE COMPONENTS (for 1-to-1 comparison)") 

        print("=" * 75) 

        print(f"invcost        : {opl['invcost']:.2f}") 

        print(f"wastecost      : {opl['wastecost']:.2f}") 

        print(f"routecost      : {opl['routecost']:.2f}") 

        print(f"assignmentcost : {opl['assignmentcost']:.2f}") 

        print(f"loadcost       : {opl['loadcost']:.2f}") 

        print(f"TOTAL          : {opl['total']:.2f}") 

        print("=" * 75) 

  

    return sol, REQ 

  

  

# ============================================================ 

# 6) Export solution to Excel (int decisions + checks) 

# ============================================================ 

  

def export_solution_to_excel( 

    inst: Instance, 

    sol: FFHSolution, 

    REQ: Dict[Tuple[str, int], float], 

    out_path: str = "ffh_solution_export.xlsx" 

) -> None: 

    """ 

    Exports decision variables and validation summaries to Excel: 

      - X, Z, Q, Y, L, Ipos, W 

      - OUT (hub,t) 

      - IN  (hub,s,t) 

      - REQ/service-level check 

      - Flow-balance check 

      - Cost breakdown (FFH) + OPL-style objective components 

    """ 

    # X 

    X_rows = [{"r": r, "t": t, "X": int(v)} for (r, t), v in sol.X.items()] 

    df_X = pd.DataFrame(X_rows).sort_values(["t", "r"]) if X_rows else pd.DataFrame(columns=["r", "t", "X"]) 

  

    # Z 

    Z_rows = [{"i": i, "r": r, "t": t, "Z": int(v)} for (i, r, t), v in sol.Z.items() if int(v) != 0] 

    df_Z = pd.DataFrame(Z_rows).sort_values(["t", "r", "i"]) if Z_rows else pd.DataFrame(columns=["i", "r", "t", "Z"]) 

  

    # Q 

    Q_rows = [] 

    for t in inst.T: 

        for b in inst.B: 

            for i in inst.D: 

                v = int(sol.Q.get((b, i, t), 0)) 

                if v != 0: 

                    Q_rows.append({"b": b, "i": i, "t": t, "Q": v}) 

    df_Q = pd.DataFrame(Q_rows).sort_values(["t", "b", "i"]) if Q_rows else pd.DataFrame(columns=["b", "i", "t", "Q"]) 

  

    # Y 

    Y_rows = [{"f": f, "b": b, "k": k, "s": s, "t": t, "Y": int(v)} 

              for (f, b, k, s, t), v in sol.Y.items() if int(v) != 0] 

    df_Y = pd.DataFrame(Y_rows).sort_values(["s", "t", "b", "k", "f"]) if Y_rows else pd.DataFrame(columns=["f", "b", "k", "s", "t", "Y"]) 

  

    # L 

    L_rows = [{"f": f, "b": b, "k": k, "s": s, "t": t, "L": int(v)} 

              for (f, b, k, s, t), v in sol.L.items() if int(v) != 0] 

    df_L = pd.DataFrame(L_rows).sort_values(["s", "t", "b", "k", "f"]) if L_rows else pd.DataFrame(columns=["f", "b", "k", "s", "t", "L"]) 

  

    # Inventory / Waste 

    I_rows, W_rows = [], [] 

    for t in inst.T: 

        for i in inst.D: 

            I_rows.append({"i": i, "t": t, "Ipos": int(sol.Ipos.get((i, t), 0))}) 

            W_rows.append({"i": i, "t": t, "W": int(sol.W.get((i, t), 0))}) 

    df_I = pd.DataFrame(I_rows).sort_values(["t", "i"]) 

    df_W = pd.DataFrame(W_rows).sort_values(["t", "i"]) 

  

    # OUT (hub,t) 

    OUT_rows = [] 

    for t in inst.T: 

        for b in inst.B: 

            out_bt = sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D) 

            OUT_rows.append({"b": b, "t": t, "OUT": int(out_bt)}) 

    df_OUT = pd.DataFrame(OUT_rows).sort_values(["t", "b"]) 

  

    # IN (hub,s,t) 

    IN_rows = [] 

    for s in inst.S: 

        for t in inst.T: 

            for b in inst.B: 

                in_bst = sum( 

                    int(val) for (f, bb, k, ss, tt), val in sol.L.items() 

                    if ss == s and tt == t and bb == b 

                ) 

                IN_rows.append({"b": b, "s": s, "t": t, "IN": int(in_bst)}) 

    df_IN = pd.DataFrame(IN_rows).sort_values(["s", "t", "b"]) 

  

    # checks 

    df_req = check_service_level_REQ(inst, sol, REQ) 

    df_flow = check_flow_balance(inst, sol) 

  

    # costs 

    df_cost_ffh = pd.DataFrame([sol.cost_breakdown]) if sol.cost_breakdown else pd.DataFrame() 

    df_cost_opl = pd.DataFrame([sol.cost_breakdown_opl]) if sol.cost_breakdown_opl else pd.DataFrame() 

  

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer: 

        df_X.to_excel(writer, sheet_name="X_route_activation", index=False) 

        df_Z.to_excel(writer, sheet_name="Z_route_loads", index=False) 

        df_Q.to_excel(writer, sheet_name="Q_hub_to_depot", index=False) 

        df_Y.to_excel(writer, sheet_name="Y_assignments", index=False) 

        df_L.to_excel(writer, sheet_name="L_inbound_qty", index=False) 

        df_I.to_excel(writer, sheet_name="Ipos_inventory", index=False) 

        df_W.to_excel(writer, sheet_name="W_waste", index=False) 

        df_OUT.to_excel(writer, sheet_name="OUT_hub_period", index=False) 

        df_IN.to_excel(writer, sheet_name="IN_hub_scenario_period", index=False) 

        df_req.to_excel(writer, sheet_name="CHECK_REQ_service", index=False) 

        df_flow.to_excel(writer, sheet_name="CHECK_flow_balance", index=False) 

        df_cost_ffh.to_excel(writer, sheet_name="Cost_breakdown_FFH", index=False) 

        df_cost_opl.to_excel(writer, sheet_name="Cost_breakdown_OPL", index=False) 

  

    print(f"[EXPORT] Solution written to: {out_path}") 

  

  

# ============================================================ 

# 7) RUN 

# ============================================================ 

  

if __name__ == "__main__": 

    file_path = "step1.xlsx" 

  

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

  

    route_to_hub = { 

        1: 2, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 2, 8: 3, 9: 2, 10: 1, 

        11: 2, 12: 3, 13: 2, 14: 1, 15: 3, 16: 2, 17: 2, 18: 2, 19: 1, 

        20: 3, 21: 2, 22: 3, 23: 3, 24: 2, 25: 1, 26: 1, 27: 3, 28: 1, 

        29: 1, 30: 1 

    } 

  

    # parameters 

    alpha = 0.995 

    cv = 0.10 

    m = 2 

    waste_cost = 27.55696873 

    scenario_probs = {1: 0.3, 2: 0.5, 3: 0.2}  # MUST match your Excel S IDs 

  

    inst = load_instance_from_excel( 

        file_path=file_path, 

        route_to_depots=route_to_depots, 

        route_to_hub=route_to_hub, 

        alpha=alpha, 

        cv=cv, 

        shelf_life=m, 

        waste_cost=waste_cost, 

        scenario_probs=scenario_probs, 

    ) 

  

    sol, REQ = solve_ffh(inst) 

    print("\nAll phases completed.") 

# =========================================================
# QUICK DIAGNOSTIC PACK (BUG var mı?)
# =========================================================

import math

def quick_diagnostics(inst, sol, REQ):
    print("\n" + "="*80)
    print("QUICK DIAGNOSTICS")
    print("="*80)

    # -----------------------------
    # 1) Q_cum ve W_cum tutarlı mı? (yeniden hesapla ve karşılaştır)
    # -----------------------------
    bad_cum = 0
    for i in inst.D:
        cumQ = 0
        cumW = 0
        for t in inst.T:
            receipt_t = sum(int(sol.Q.get((b,i,t),0)) for b in inst.B)
            cumQ += receipt_t
            cumW += int(sol.W.get((i,t),0))

            if int(sol.Q_cum.get((i,t), -999999)) != cumQ:
                bad_cum += 1
                print(f"[BAD CUM] depot={i} t={t} sol.Q_cum={sol.Q_cum.get((i,t))} recomputed={cumQ}")
                break

            if int(sol.W_cum.get((i,t), -999999)) != cumW:
                bad_cum += 1
                print(f"[BAD CUM] depot={i} t={t} sol.W_cum={sol.W_cum.get((i,t))} recomputed={cumW}")
                break
    print("[1] Cumulative trackers (Q_cum/W_cum) OK?" , "YES" if bad_cum==0 else f"NO -> {bad_cum} depots mismatch")

    # -----------------------------
    # 2) FIFO fiziksel denge kontrolü: implied inventory negatif mi?
    #    I_implied(t) = cumQ - cumDem - cumW  (backlog yok varsayımıyla)
    #    Bu negatifse, FIFO/receipt/demand akışında bug/uyumsuzluk vardır.
    # -----------------------------
    bad_implied = 0
    for i in inst.D:
        cumQ = 0
        cumW = 0
        cumDem = 0
        for t in inst.T:
            cumQ += sum(int(sol.Q.get((b,i,t),0)) for b in inst.B)
            cumW += int(sol.W.get((i,t),0))
            cumDem += int(round(inst.mu[(i,t)]))

            I_implied = cumQ - cumDem - cumW
            if I_implied < 0:
                bad_implied += 1
                print(f"[BAD IMPLIED] depot={i} t={t} cumQ={cumQ} cumDem={cumDem} cumW={cumW} I_implied={I_implied}")
                break
    print("[2] Implied inventory never negative?" , "YES" if bad_implied==0 else f"NO -> {bad_implied} depots violated")

    # -----------------------------
    # 3) Ipos ile implied inventory uyumlu mu?
    #    Ipos(t) FIFO bucket toplamı. Bu, implied ile aynı olmalı (backlog yoksa).
    #    Küçük farklar rounding kaynaklı olabilir ama büyük fark bug işareti.
    # -----------------------------
    bad_I = 0
    for i in inst.D:
        cumQ = 0
        cumW = 0
        cumDem = 0
        for t in inst.T:
            cumQ += sum(int(sol.Q.get((b,i,t),0)) for b in inst.B)
            cumW += int(sol.W.get((i,t),0))
            cumDem += int(round(inst.mu[(i,t)]))

            I_implied = cumQ - cumDem - cumW
            Ipos = int(sol.Ipos.get((i,t),0))

            if abs(Ipos - I_implied) > 1:  # tolerans 1 birim
                bad_I += 1
                print(f"[BAD I] depot={i} t={t} Ipos={Ipos} I_implied={I_implied} diff={Ipos-I_implied}")
                break
    print("[3] Ipos matches implied inventory (±1)?" , "YES" if bad_I==0 else f"NO -> {bad_I} depots mismatch")

    # -----------------------------
    # 4) Flow balance gerçekten sağlanıyor mu? (CHECK-B gibi ama hızlı)
    #    IN(b,s,t) == OUT(b,t) her (b,s,t) için olmalı
    # -----------------------------
    bad_flow = 0
    for s in inst.S:
        for t in inst.T:
            for b in inst.B:
                OUT = sum(int(sol.Q.get((b,i,t),0)) for i in inst.D)
                IN = sum(int(v) for (f,bb,k,ss,tt),v in sol.L.items() if ss==s and tt==t and bb==b)
                if IN != OUT:
                    bad_flow += 1
                    print(f"[BAD FLOW] s={s} t={t} b={b} IN={IN} OUT={OUT} gap={IN-OUT}")
    print("[4] Flow balance IN==OUT everywhere?" , "YES" if bad_flow==0 else f"NO -> {bad_flow} rows violated")

    # -----------------------------
    # 5) REQ yeniden üretildiğinde aynı mı? (REQ hesabında bug var mı)
    # -----------------------------
    def recompute_REQ(inst):
        z = z_value(inst.alpha)
        out = {}
        for i in inst.D:
            cum_mu=0.0
            cum_var=0.0
            for t in inst.T:
                m = inst.mu[(i,t)]
                s = inst.cv*m
                cum_mu += m
                cum_var += s*s
                out[(i,t)] = math.ceil(cum_mu + z*math.sqrt(cum_var))
        return out

    REQ2 = recompute_REQ(inst)
    bad_req = 0
    for key, v in REQ.items():
        if int(v) != int(REQ2[key]):
            bad_req += 1
            if bad_req <= 10:
                print("[BAD REQ]", key, "REQ=", v, "REQ2=", REQ2[key])
    print("[5] REQ reproducible exactly?" , "YES" if bad_req==0 else f"NO -> {bad_req} keys differ")

    # -----------------------------
    # 6) Son olarak: CHECK-A neden fail? -> “fail sadece waste yüzünden mi?”
    #    Eğer fail satırlarında slack ≈ - (W_cum artışı) gibi gidiyorsa bug değil, tasarım etkisi.
    # -----------------------------
    fail_rows = []
    for i in inst.D:
        prev = None
        for t in inst.T:
            lhs = int(sol.Q_cum[(i,t)]) - int(sol.W_cum[(i,t)])
            rhs = int(REQ[(i,t)])
            slack = lhs - rhs
            if slack < 0:
                w_now = int(sol.W_cum[(i,t)])
                w_prev = int(sol.W_cum[(i,t-1)]) if (i,t-1) in sol.W_cum else 0
                dW = w_now - w_prev
                fail_rows.append((i,t,slack,dW))
                break
    print(f"[6] First-fail count (CHECK-A): {len(fail_rows)}")
    if fail_rows:
        print("    Sample first 10 fails: (depot, t, slack, dW_at_t)")
        for r in fail_rows[:10]:
            print("   ", r)

    print("="*80)
    return {
        "bad_cum": bad_cum,
        "bad_implied": bad_implied,
        "bad_I": bad_I,
        "bad_flow": bad_flow,
        "bad_req": bad_req,
        "fails": len(fail_rows),
    }

diag = quick_diagnostics(inst, sol, REQ)
print("DIAG SUMMARY:", diag)


# =========================
# DEBUG: implied inventory feasibility check
# =========================
bad = []
for i in inst.D:
    cumQ = 0
    cumW = 0
    for t in inst.T:
        cumQ += sum(sol.Q.get((b,i,t),0) for b in inst.B)
        cumW += sol.W.get((i,t),0)
        cumDem = sum(int_round(inst.mu[(i,a)]) for a in inst.T if a <= t)

        I_implied = cumQ - (cumDem + cumW)
        if I_implied < 0:
            bad.append((i,t,cumQ,cumDem,cumW,I_implied))
            break   # ilk patlayan t yeterli

print("\n[DEBUG] Negative implied inventory (first violations):")
for row in bad[:10]:
    print("Depot", row[0], "t=", row[1],
          "| cumQ=", row[2],
          "| cumDem=", row[3],
          "| cumW=", row[4],
          "| I_implied=", row[5])

print("Total depots with violation:", len(bad))

# =========================
# Export
# =========================


    # Export (includes checks + both cost breakdowns) 

export_solution_to_excel(inst, sol, REQ, out_path="ffh_solution_export.xlsx") 

  

viol = 0 

for s in inst.S: 

    for t in inst.T: 

        for f in inst.F: 

            cnt = sum(sol.Y.get((f,b,k,s,t),0) for b in inst.B for k in inst.K) 

            if cnt > 1: 

                viol += 1 

print("[CHECK] multi-assign (sum_{b,k} Y > 1) count =", viol) 

  

# FFH main loop bittikten sonra (örneğin solve_ffh() fonksiyonunun sonunda)

t = 1
i = "19"

qsum = sum(sol.Q.get((b, i, t), 0) for b in inst.B)

print("=== DEBUG CHECK (Depot 19, t=1) ===")
print("Sum_b Q[b,19,1] =", qsum)
print("mu[19,1] =", inst.mu[(i, t)])
print("Ipos[19,1] =", sol.Ipos.get((i, t), 0),
      "W[19,1] =", sol.W.get((i, t), 0))
print("=================================")
  

bad = 0 

for (f,b,k,s,t), y in sol.Y.items(): 

    if y==1 and sol.L.get((f,b,k,s,t),0)==0: 

        bad += 1 

print("[CHECK] Y=1 but L=0 count =", bad) 

  

AC = 0.0 

for s in inst.S: 

    pr = inst.scenario_probs[s] 

    for t in inst.T: 

        for f in inst.F: 

            for b in inst.B: 

                for k in inst.K: 

                    y = sol.Y.get((f,b,k,s,t),0) 

                    if y==1: 

                        AC += pr * inst.beta[(f,k)] 

print("AC_check(OPL) =", AC) 



'''
def safe_build_R_and_cap(inst, cap_override=None):
    """
    Returns:
      R_list: list[str] of route ids
      cap_r:  dict[str, float] route capacities (may be empty if not found)

    cap_override: if you already have a dict like {route_id: capacity}, pass it here.
    """
    # If you already have capacities as a dict, use it
    if isinstance(cap_override, dict) and len(cap_override) > 0:
        R_list = sorted([str(rid) for rid in cap_override.keys()], key=lambda x: (len(x), x))
        cap_r = {str(rid): float(cap) for rid, cap in cap_override.items()}
        return R_list, cap_r

    R_list = []
    cap_r = {}

    if not hasattr(inst, "routes") or inst.routes is None:
        return R_list, cap_r

    for r in inst.routes:
        # robust route id extraction
        rid = None
        for attr in ["id", "rid", "r_id", "route_id", "name", "code"]:
            if hasattr(r, attr):
                rid = getattr(r, attr)
                if rid is not None:
                    rid = str(rid)
                    break
        if rid is None:
            # fallback: try dict-like
            try:
                rid = str(r["id"])
            except Exception:
                raise AttributeError(
                    "Could not determine route id. Please tell me your Route fields (print(r))."
                )

        R_list.append(rid)

        # robust capacity extraction
        cap = None
        for attr in ["capacity", "cap", "c", "c_r", "lambda_r", "route_cap"]:
            if hasattr(r, attr):
                cap = getattr(r, attr)
                if cap is not None:
                    break
        if cap is not None:
            try:
                cap_r[rid] = float(cap)
            except Exception:
                pass  # leave it out if not numeric

    return sorted(set(R_list), key=lambda x: (len(x), x)), cap_r
# ============================================================
# RUN THE DIAGNOSTICS (FFHSolution-compatible)
# Place this at the very end, after you have `inst` and `sol`.
# ============================================================

from collections import defaultdict
import math

# ----------------------------
# 0) Build route id list + route capacity dict
# ----------------------------
def build_R_and_cap(inst):
    """
    Returns:
      R: list of route IDs (strings)
      cap_r: dict {route_id: capacity}
    Adjust attribute names below if your Route dataclass differs.
    """
    R = []
    cap_r = {}

    for r in inst.routes:
        # Route id attribute (common: r.id or r.rid or r.name)
        rid = getattr(r, "id", None)
        if rid is None:
            rid = getattr(r, "rid", None)
        if rid is None:
            rid = getattr(r, "name", None)
        if rid is None:
            raise AttributeError("Route object has no id/rid/name attribute. Please adjust build_R_and_cap().")

        # capacity attribute (common: r.capacity or r.cap or r.c)
        cap = getattr(r, "capacity", None)
        if cap is None:
            cap = getattr(r, "cap", None)
        if cap is None:
            cap = getattr(r, "c", None)
        if cap is None:
            # If you don't have capacity in Route, keep None -> util prints NA
            cap = None

        R.append(str(rid))
        if cap is not None:
            cap_r[str(rid)] = float(cap)

    return R, cap_r


# ----------------------------
# 1) Y count and avg shipment per Y
# ----------------------------
def print_assignment_stats(inst, sol):
    count_Y = 0
    sum_L = 0.0

    countY_st = defaultdict(int)   # (s,t)
    sumL_st = defaultdict(float)
    countY_k = defaultdict(int)    # k
    sumL_k = defaultdict(float)

    # sol.Y and sol.L are tuple-key dicts: (f,b,k,s,t)
    for (f,b,k,s,t), yval in sol.Y.items():
        if yval and yval > 0.5:
            count_Y += 1
            countY_st[(s,t)] += 1
            countY_k[k] += 1

            lval = float(sol.L.get((f,b,k,s,t), 0.0))
            sum_L += lval
            sumL_st[(s,t)] += lval
            sumL_k[k] += lval

    avg_ship = (sum_L / count_Y) if count_Y > 0 else 0.0

    print("\n" + "="*70)
    print("[DEBUG] Assignment activation & shipment stats (from sol.Y / sol.L)")
    print(f"  #Y (activated arcs)          = {count_Y}")
    print(f"  Sum L (total shipped)        = {sum_L:.2f}")
    print(f"  Avg shipped per activated Y  = {avg_ship:.2f}")
    print("-"*70)

    print("  Breakdown by (s,t):  countY | sumL | avgLperY")
    for s in sorted(inst.S):
        for t in sorted(inst.T):
            cy = countY_st[(s,t)]
            sl = sumL_st[(s,t)]
            al = (sl/cy) if cy > 0 else 0.0
            print(f"    (s={s}, t={t}) : {cy:5d} | {sl:10.2f} | {al:8.2f}")

    print("-"*70)
    print("  Breakdown by vehicle k:  countY | sumL | avgLperY")
    for k in sorted(inst.K):
        cy = countY_k[k]
        sl = sumL_k[k]
        al = (sl/cy) if cy > 0 else 0.0
        print(f"    k={k} : {cy:5d} | {sl:10.2f} | {al:8.2f}")

    return count_Y, sum_L, avg_ship


# ----------------------------
# 2) Route utilization for X=1 routes
# ----------------------------
def print_route_utilization(inst, sol, cap_r=None, low_util_threshold=0.30):
    """
    sol.X: (r,t) -> 0/1
    sol.Z: (i,r,t) -> load
    """
    print("\n" + "="*70)
    print("[DEBUG] Route utilization (only X=1 routes)")

    util_rt = {}
    low_list = []

    print("  r,t : X | load | cap | util")
    print("-"*70)

    # Pre-aggregate loads per (r,t) for speed
    load_rt = defaultdict(float)
    for (i,r,t), zval in sol.Z.items():
        load_rt[(r,t)] += float(zval)

    for (r,t), xval in sol.X.items():
        if not (xval and xval > 0.5):
            continue

        load = load_rt.get((r,t), 0.0)
        cap = None
        if cap_r is not None:
            cap = cap_r.get(str(r), None)

        if cap is None or cap <= 0:
            print(f"  r={r}, t={t}: {int(xval)} | {load:8.2f} |  NA   |   NA")
            util = float("nan")
        else:
            util = load / cap
            print(f"  r={r}, t={t}: {int(xval)} | {load:8.2f} | {cap:5.1f} | {util:6.3f}")

        util_rt[(r,t)] = util
        if (not math.isnan(util)) and util < low_util_threshold:
            low_list.append(((r,t), util))

    if low_list:
        print("-"*70)
        print(f"  Low-util routes (<{low_util_threshold:.2f}):")
        for (r,t), u in sorted(low_list, key=lambda x: x[1]):
            print(f"    r={r}, t={t} util={u:.3f}")

    return util_rt


# ----------------------------
# 3) Depot receipt vs demand (+ ending inv/waste from sol.Ipos/sol.W)
# ----------------------------
def print_depot_flow_balance(inst, sol, mu):
    """
    sol.Q: (b,i,t) -> receipt to depot i from hub b at time t
    mu can be:
      - dict keyed by (i,t)
      - nested dict mu[i][t]
      - or object with mu[(i,t)] style
    """
    def get_mu(i,t):
        if isinstance(mu, dict):
            if (i,t) in mu:
                return float(mu[(i,t)])
            # nested dict
            if i in mu and isinstance(mu[i], dict) and t in mu[i]:
                return float(mu[i][t])
        # fallback
        return 0.0

    print("\n" + "="*70)
    print("[DEBUG] Depot receipts vs demand + InvEnd/Waste (from sol.Q, sol.Ipos, sol.W)")
    print("-"*70)

    # Pre-aggregate receipt per (i,t)
    rec_it = defaultdict(float)
    for (b,i,t), qval in sol.Q.items():
        rec_it[(i,t)] += float(qval)

    for i in sorted(inst.D):
        print(f"Depot i={i}")
        for t in sorted(inst.T):
            receipt = rec_it.get((i,t), 0.0)
            demand = get_mu(i,t)
            inv_end = float(sol.Ipos.get((i,t), 0.0))
            waste = float(sol.W.get((i,t), 0.0))
            print(f"  t={t}: receipt={receipt:8.2f} | demand={demand:8.2f} | InvEnd={inv_end:8.2f} | Waste={waste:8.2f}")
        print("-"*70)


# ----------------------------
# 4) Print cost breakdown (already in sol)
# ----------------------------
def print_costs(sol):
    print("\n" + "="*70)
    print("[DEBUG] Cost breakdown (sol.cost_breakdown_opl)")
    if sol.cost_breakdown_opl:
        for k, v in sol.cost_breakdown_opl.items():
            print(f"  {k:15s}: {v:.2f}")
    else:
        print("  (empty) sol.cost_breakdown_opl is empty.")
    print("="*70)


# ===========================
# RUN
# ===========================

# (A) Route list + capacities from inst.routes
try:
    R_list, cap_r = build_R_and_cap(inst)
except Exception as e:
    print("\n[DEBUG] Could not build route capacities from inst.routes:", repr(e))
    R_list, cap_r = None, None

# (B) Costs
print_costs(sol)

# (C) Assignment stats
count_Y, sum_L, avg_ship = print_assignment_stats(inst, sol)

# (D) Route utilization
util_rt = print_route_utilization(inst, sol, cap_r=cap_r, low_util_threshold=0.30)

# (E) Depot receipts vs demand (+ inv/waste)
# IMPORTANT: set `mu` below correctly (your demand mean dict)
# Example options:
#   mu = mu_dict
#   mu = inst.mu
#   mu = demand_mu
try:
    mu=inst.mu  # just to see if it exists
except NameError:
    mu = {}  # fallback to avoid crash

print_depot_flow_balance(inst, sol, mu)

print("\n[DEBUG] Diagnostics completed.")
print("="*70)

# ============================================================
# FINAL RUN THE DIAGNOSTICS (Instance.mu + FFHSolution)
# ============================================================

# 1) Demand mean alias
mu = inst.mu   # <-- THIS FIXES demand=0 problem in debug

# 2) (Optional) Sanity check: make sure keys match inst.D/inst.T
print("\n[DEBUG] mu sanity check")
print("  mu len:", len(mu))
print("  sample mu keys:", list(mu.keys())[:5])
i0, t0 = inst.D[0], inst.T[0]
print(f"  test mu[(i0,t0)] for i0={i0}, t0={t0} ->", mu.get((i0, t0), "MISSING"))

# Example: if you have a dict already, pass it
# R_list, cap_r = safe_build_R_and_cap(inst, cap_override=route_capacity_dict)

# If you don't have it, try reading from inst.routes safely:
R_list, cap_r = safe_build_R_and_cap(inst)
print("[DEBUG] routes:", len(R_list), "| capacity entries:", len(cap_r))

# 4) Costs
print_costs(sol)

# 5) Assignment stats
count_Y, sum_L, avg_ship = print_assignment_stats(inst, sol)

# 6) Route utilization
util_rt = print_route_utilization(inst, sol, cap_r=cap_r, low_util_threshold=0.30)

# 7) Depot receipts vs demand (+ inv/waste)
print_depot_flow_balance(inst, sol, mu)

print("\n[DEBUG] Diagnostics completed.")
print("="*70)

'''