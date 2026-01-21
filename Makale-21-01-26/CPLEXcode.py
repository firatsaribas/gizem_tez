import pandas as pd
import numpy as np
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from pyomo.environ import *
import time 




# 1. Python zamanlayıcıyı başlat
start_time = time.time()
# ============================================================
# 1. VERİ YAPILARI (Dataclasses)
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
    mu: Dict[Tuple[str, int], float]
    alpha: float; cv: float; shelf_life: int
    holding_cost: Dict[str, float]
    waste_cost: float
    supply: Dict[Tuple[str, str, int], float]
    theta: Dict[str, float]
    beta: Dict[Tuple[str, str], float]
    gamma: Dict[Tuple[str, str, str], float]
    scenario_probs: Dict[str, float]

# ============================================================
# 2. YARDIMCI FONKSİYONLAR
# ============================================================
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def to_str_id(x) -> str:
    if pd.isna(x): return "0"
    if isinstance(x, float) and float(x).is_integer(): x = int(x)
    return str(x)

def z_lookup(alpha: float) -> float:
    lookup = {0.90: 1.281, 0.95: 1.644, 0.975: 1.959, 0.99: 2.326, 0.995: 2.575}
    return lookup.get(alpha, 2.575)

# ============================================================
# 3. VERİ YÜKLEME (Excel -> Instance)
# ============================================================
def load_instance_from_excel(file_path, r_to_d, r_to_h, alpha, cv, shelf_life, waste_cost, scenario_probs) -> Instance:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} dosyası bulunamadı!")
    
    xl = pd.ExcelFile(file_path)
    
    # Sayfaları oku
    beta_df = norm_cols(pd.read_excel(xl, sheet_name="beta")).astype({"f": int, "k": int})
    gamma_df = norm_cols(pd.read_excel(xl, sheet_name="gamma")).astype({"f": int, "b": int, "k": int})
    theta_df = norm_cols(pd.read_excel(xl, sheet_name="theta")).astype({"k": int})
    stock_costs_df = norm_cols(pd.read_excel(xl, sheet_name="stock_costs")).astype({"d": int})
    supply_df = norm_cols(pd.read_excel(xl, sheet_name="supply")).astype({"f": int, "s": int, "t": int})
    demand_df = norm_cols(pd.read_excel(xl, sheet_name="demand")).astype({"d": int, "t": int})
    route_costs_df = norm_cols(pd.read_excel(xl, sheet_name="route_costs")).astype({"r": int})
    route_capacity_df = norm_cols(pd.read_excel(xl, sheet_name="route_capacity")).astype({"r": int})

    # hi sayfası verileri (Maliyetler ve Olasılıklar)
    df_hi = pd.read_excel(xl, sheet_name="stock_costs", header=None)

    # Instance Nesnesini Oluştur
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

    # Rota Listesini Manuel Sözlüklerden Doldur
    r_costs = {int(r.r): float(r.cost) for r in route_costs_df.itertuples(index=False)}
    r_caps = {int(r.r): float(r.capacity) for r in route_capacity_df.itertuples(index=False)}
    
    for r_id_int in r_costs.keys():
        r_id_str = str(r_id_int)
        if r_id_str not in r_to_h or r_id_str not in r_to_d:
            raise KeyError(f"Missing route mapping for r_id={r_id_str}")
        inst.routes.append(Route(
            r_id=r_id_str,
            hub=str(r_to_h[r_id_str]),
            depots=tuple(str(d) for d in r_to_d[r_id_str]),
            capacity=r_caps[r_id_int],
            fixed_cost=r_costs[r_id_int]
        ))
    
    return inst

# ============================================================
# 4. PYOMO MODEL TANIMI
# ============================================================
def build_pyomo_model(inst: Instance):
    model = ConcreteModel()

    model.F = Set(initialize=inst.F); model.S = Set(initialize=inst.S)
    model.T = Set(initialize=inst.T); model.B = Set(initialize=inst.B)
    model.D = Set(initialize=inst.D); model.K = Set(initialize=inst.K)
    model.R = Set(initialize=[r.r_id for r in inst.routes])

    model.Y = Var(model.F, model.B, model.K, model.S, model.T, domain=Binary, initialize=0)
    model.L = Var(model.F, model.B, model.K, model.S, model.T, domain=NonNegativeIntegers, initialize=0)
    model.Q = Var(model.B, model.D, model.T, domain=NonNegativeIntegers, initialize=0)
    model.Z = Var(model.D, model.R, model.T, domain=NonNegativeIntegers, initialize=0)
    model.X = Var(model.R, model.T, domain=Binary, initialize=0)
    model.I = Var(model.D, model.T, domain=Integers, initialize=0)
    model.Ipo = Var(model.D, model.T, domain=NonNegativeIntegers, initialize=0)
    model.W = Var(model.D, model.T, domain=NonNegativeIntegers, initialize=0)

    # Objective
    inv_cost = sum(model.Ipo[i, t] * inst.holding_cost[i] for i in model.D for t in model.T)
    waste_cost = sum(model.W[i, t] * inst.waste_cost for i in model.D for t in model.T if t >= inst.shelf_life)
    route_cost = sum(r.fixed_cost * model.X[r.r_id, t] for r in inst.routes for t in model.T)
    
    assign_cost = sum(inst.scenario_probs[s] * inst.beta[f, k] * model.Y[f, b, k, s, t]
                      for f, k in inst.beta for b in model.B for s in model.S for t in model.T)
    
    load_cost = sum(inst.scenario_probs[s] * inst.gamma[f, b, k] * model.L[f, b, k, s, t]
                    for f, b, k in inst.gamma for s in model.S for t in model.T)

    model.obj = Objective(expr=inv_cost + waste_cost + route_cost + assign_cost + load_cost, sense=minimize)

    # Bunları daha sonra ekranda görmek için model objesine kaydedelim
    model.inv_total = inv_cost
    model.waste_total = waste_cost
    model.route_total = route_cost
    model.assign_total = assign_cost
    model.load_total = load_cost

    # Constraints *//
    def supply_rule(model, f, s, t):
        return sum(model.L[f, b, k, s, t] for b in model.B for k in model.K if (f, b, k) in inst.gamma) <= inst.supply.get((f, s, t), 0)
    model.c_supply = Constraint(model.F, model.S, model.T, rule=supply_rule)
 
    def veh_cap_rule(model, k, s, t):
        return sum(model.L[f, b, k, s, t] for f in model.F for b in model.B if (f, b, k) in inst.gamma) <= inst.theta[k]
    model.c_veh_cap = Constraint(model.K, model.S, model.T, rule=veh_cap_rule)
   
    # forall(f in F, b in B, s in S, t in T, k in K)
    def assignment_coupling_rule(model, f, b, k, s, t):
        # L[f][b][k][s][t] <= Y[f][b][k][s][t] * theta[k]
        # Sadece geçerli (f, b, k) kombinasyonları için tanımlıyoruz
        if (f, b, k) in inst.gamma:
            return model.L[f, b, k, s, t] <= model.Y[f, b, k, s, t] * inst.theta[k]
        return Constraint.Skip
    model.c_assign_coupling = Constraint(model.F, model.B, model.K, model.S, model.T, rule=assignment_coupling_rule)

    # forall(f in F, s in S, t in T, k in K)
    def one_hub_assignment_rule(model, f, s, t, k):
        # sum(b in B) Y[f][b][k][s][t] <= 1
        return sum(model.Y[f, b, k, s, t] for b in model.B) <= 1

    model.c_one_hub = Constraint(model.F, model.S, model.T, model.K, rule=one_hub_assignment_rule)

    # forall(b in B, s in S, t in T)
    def hub_balance_rule(model, b, s, t):
        # sum(f in F, k in K) L[f,b,k,s,t] == sum(i in D) Q[b,i,t]
        incoming = sum(model.L[f, b, k, s, t] for f in model.F for k in model.K if (f, b, k) in inst.gamma)
        outgoing = sum(model.Q[b, i, t] for i in model.D)
        
        return incoming == outgoing

    model.c_hub_balance = Constraint(model.B, model.S, model.T, rule=hub_balance_rule)

    # forall(i in D, t in T)
    def inventory_balance_rule(model, i, t):
        # sum (a in 1..t, b in B) Q[b][i][a]
        total_inflow = sum(model.Q[b, i, a] for a in range(1, t + 1) for b in model.B)
        
        # sum(a in 1..t) (de[i][a] + W[i][a])
        total_outflow = sum(inst.mu[i, a] + model.W[i, a] for a in range(1, t + 1))
        
        # I[i][t] == Inflow - Outflow
        return model.I[i, t] == total_inflow - total_outflow

    model.c_inventory_balance = Constraint(model.D, model.T, rule=inventory_balance_rule)

    # forall(i in D, t in T)
    def inventory_positive_rule(model, i, t):
        # Ipo[i, t] >= I[i, t]
        return model.Ipo[i, t] >= model.I[i, t]

    model.c_inventory_positive = Constraint(model.D, model.T, rule=inventory_positive_rule)


    # forall(i in D, t in T: t >= m)
    def waste_calculation_rule(model, i, t):
        # OPL'deki t >= m şartını kontrol ediyoruz
        if t < inst.shelf_life:
            return Constraint.Skip
        # I[i][t-m+1]
        inventory_term = model.I[i, t - inst.shelf_life + 1]
        # sum (a in t-m+2..t) de[i][a]
        demand_sum = sum(inst.mu[i, a] for a in range(t - inst.shelf_life + 2, t + 1))
        # sum (a in t-m+2..t-1) W[i][a]
        waste_sum = sum(model.W[i, a] for a in range(t - inst.shelf_life + 2, t))
        # W[i][t] >= I - DemandSum - WasteSum
        return model.W[i, t] >= inventory_term - demand_sum - waste_sum

    model.c_waste_calc = Constraint(model.D, model.T, rule=waste_calculation_rule)


    # forall(i in D, t in T: t < m)
    def no_waste_before_shelf_life_rule(model, i, t):
        # OPL'deki t < m şartı
        if t < inst.shelf_life:
            return model.W[i, t] == 0
        return Constraint.Skip

    model.c_no_waste = Constraint(model.D, model.T, rule=no_waste_before_shelf_life_rule)

    # forall(i in D, t in T)
    def service_level_rule(model, i, t):
        # sum (a in 1..t, b in B) Q[b][i][a]
        total_inflow = sum(model.Q[b, i, a] for a in range(1, t + 1) for b in model.B)
        
        # sum (a in 1..t-1) W[i][a]
        total_waste_past = sum(model.W[i, a] for a in range(1, t))
        
        # Karekök içindeki kısım: sum (a in 1..t) de[i][a]*de[i][a]
        # de[i][a] senin modelinde inst.mu[i,a] olarak geçiyor
        variance_sum = sum((inst.mu[i, a] ** 2) for a in range(1, t + 1))
        
        # Sağ taraf: Total Demand + Safety Stock
        # co = inst.cv, Zalpha = inst.z_alpha (bunları inst içinden alıyoruz)
        right_side = sum(inst.mu[i, a] for a in range(1, t + 1)) + \
                     (variance_sum ** 0.5) * inst.cv * inst.alpha
        
        return total_inflow - total_waste_past >= right_side
    
    model.c_service_level = Constraint(model.D, model.T, rule=service_level_rule)
    
    
    # forall(r in R, t in T, i in D)
    def route_visit_restriction_rule(model, r_id, t, i):
        # 1. r_to_d_manual içindeki geçerli depoları alıyoruz
        # r_id anahtar olarak kullanılır, eğer r_id yoksa boş liste döner
        allowed_depots = r_to_d_manual.get(r_id, [])
        
        # 2. Eğer mevcut depo (i), bu rotanın uğradığı depolar listesinde DEĞİLSE
        if i not in allowed_depots:
            # Z[i][r][t] == 0 kısıtını uygula
            return model.Z[i, r_id, t] == 0
        
        # 3. Eğer depo listede varsa, bu kısıtı bu i için geç (herhangi bir engel koyma)
        return Constraint.Skip

    model.c_route_visit_restriction = Constraint(model.R, model.T, model.D, rule=route_visit_restriction_rule)
   

    def route_capacity_rule(model, r_id, t):
        # 1. Mevcut rota nesnesine (kapasite bilgisi için) ve durak listesine erişelim
        # r_to_d_manual içindeki duraklar a.NR kümesini temsil eder
        allowed_depots = r_to_d_manual.get(r_id, [])
        
        # 2. sum(i in D, i in a.NR) Z[i][r][t]
        # Sadece o rotanın uğradığı depolar üzerindeki yükleri topluyoruz
        total_load = sum(model.Z[i, r_id, t] for i in allowed_depots if i in model.D)
        
        # 3. c[r] * X[r][t]
        # inst.routes içindeki kapasite (capacity) bilgisini kullanıyoruz
        curr_route_obj = next(r for r in inst.routes if r.r_id == r_id)
        capacity_limit = curr_route_obj.capacity * model.X[r_id, t]
        
        return total_load <= capacity_limit

    model.c_route_capacity = Constraint(model.R, model.T, rule=route_capacity_rule)


    # forall(i in D, t in T, b in B)
    def route_hub_assignment_rule(model, i, t, b):
        # sum (r in R) Z[i,r,t] * delta[b,r]
        # Sadece b hub'ına bağlı olan rotaları (r) topluyoruz
        
        # inst.r_to_h_manual sözlüğünü kullanarak delta[b][r] == 1 olanları filtreliyoruz
        relevant_routes = [r_id for r_id, hub_id in r_to_h_manual.items() if hub_id == b]
        
        incoming_from_routes = sum(model.Z[i, r_id, t] for r_id in relevant_routes)
        
        # == Q[b,i,t]
        return incoming_from_routes == model.Q[b, i, t]

    model.c_route_hub_assignment = Constraint(model.D, model.T, model.B, rule=route_hub_assignment_rule)
   
    return model

# ============================================================
# 5. DEBUG VE ÇALIŞTIRMA BLOĞU
# ============================================================

def print_debug_info(name, data_dict):
    """Sözlük verisinin ilk 5 ve son 5 kaydını konsola yazdırır."""
    print(f"\n>>> [DEBUG] {name} (Toplam Kayıt: {len(data_dict)})")
    if not data_dict:
        print("    DIKKAT: Bu veri tablosu bos!")
        return
    
    keys = list(data_dict.keys())
    # İlk 5 Kayıt
    print("    Ilk 5 Kayit:")
    for k in keys[:5]:
        print(f"      {k}: {data_dict[k]}")
    
    if len(keys) > 10:
        print("      ...")
        # Son 5 Kayıt
        print("    Son 5 Kayit:")
        for k in keys[-5:]:
            print(f"      {k}: {data_dict[k]}")

# ============================================================
# 5. EXECUTION
# ============================================================
if __name__ == "__main__":
    # 1. Excel Dosya Adı
    file_name = "step1.xlsx"
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

    # Normalize manual maps to string IDs to match model sets.
    r_to_d_manual = {str(k): [str(d) for d in v] for k, v in r_to_d_manual.items()}
    r_to_h_manual = {str(k): str(v) for k, v in r_to_h_manual.items()}

    

    # 3. Veri Yükleme
    print("\n--- Veri Yukleme Islemi Basliyor ---")
    t0 = time.perf_counter()
    inst = load_instance_from_excel(
        file_name, r_to_d_manual, r_to_h_manual,
        alpha=0.995, cv=0.1, shelf_life=2, waste_cost=27.55696873 ,
        scenario_probs={1: 0.3, 2: 0.5, 3: 0.2}
    )
    t1 = time.perf_counter()
    print(f"[TIMER] load_instance_from_excel: {t1 - t0:.2f}s")

    # 4. Veri Kontrolü (Baş-Son 5 kaydı yazdırır)
    t2 = time.perf_counter()
    print_debug_info("BETA (Assignment Cost)", inst.beta)
    print_debug_info("GAMA (Loading Cost)", inst.gamma)
    print_debug_info("SUPPLY (Tedarik)", inst.supply)
    print_debug_info("DEMAND (Talep)", inst.mu)
    print_debug_info("HOLDING COST", inst.holding_cost)
    
    # THETA Yazdırma
    print_debug_info("THETA (Vehicle Capacity)", inst.theta)
    # Debug bloğuna bunu da ekle
    print(f"\n>>> [DEBUG] WASTE COST (p): {inst.waste_cost}")

    # ROUTE COSTS ve CAPACITY Yazdırma (Liste içinde olduğu için manuel döngü)
    print(f"\n>>> [DEBUG] ROUTE DATA (Toplam Rota: {len(inst.routes)})")
    print("    Ilk 5 Rota:")
    for r in inst.routes[:5]:
        print(f"      ID: {r.r_id} | Hub: {r.hub} | Cap: {r.capacity} | Cost: {r.fixed_cost}")
    
    print("      ...")
    
    print("    Son 5 Rota:")
    for r in inst.routes[-5:]:
        print(f"      ID: {r.r_id} | Hub: {r.hub} | Cap: {r.capacity} | Cost: {r.fixed_cost}")
    t3 = time.perf_counter()
    print(f"[TIMER] debug_prints: {t3 - t2:.2f}s")

    input("\nKonsoldaki veriler Excel ile ayni mi? Dogruysa Enter'a basarak cozumu baslatin...")
    # 5. Model Kurma ve Çözme
    print("\nPyomo modeli kuruluyor...")
    t4 = time.perf_counter()
    model = build_pyomo_model(inst)
    t5 = time.perf_counter()
    print(f"[TIMER] build_pyomo_model: {t5 - t4:.2f}s")
    
    print("Solver (CPLEX) baslatiliyor...")
    opt = SolverFactory('cplex')
    t6 = time.perf_counter()
    results = opt.solve(model, tee=True)
    t7 = time.perf_counter()
    print(f"[TIMER] solve: {t7 - t6:.2f}s")

    # %1 Gap Ayarı (0.01 = %1)
    # mip.tolerances.mipgap parametresi CPLEX için standarttır
    opt.options['mip_tolerances_mipgap'] = 0.05
    opt.options['timelimit'] = 120
    # 2. Python zamanlayıcıyı durdur
    end_time = time.time()
    elapsed_time = end_time - start_time
    
# --- BURADAN İTİBAREN GÜNCELLE ---
    
    # Değişkenlerin değerini güvenli bir şekilde okumak için yardımcı fonksiyon
    # Bu fonksiyon def build_pyomo_model'in DIŞINDA da olabilir, İÇİNDE de.
    def safe_value(expr):
        try:
            from pyomo.environ import value
            return value(expr)
        except:
            return 0.0

    if results.solver.termination_condition == TerminationCondition.optimal:
        print("\n" + "="*40)
        print("BAŞARILI: OPTİMAL ÇÖZÜM BULUNDU")
        print("="*40)
        
        print(f"Çözüm Durumu   : OPTİMAL/FEASIBLE")
        print(f"Toplam Süre    : {end_time - start_time:.2f} saniye")
        print(f"Gap Toleransı  : %1")
        print("-" * 45)

        # Maliyetleri yazdırırken safe_value kullanıyoruz
        print(f"1. Envanter Tutma : {safe_value(model.inv_total):.2f}")
        print(f"2. Atik Maliyeti  : {safe_value(model.waste_total):.2f}")
        print(f"3. Rota Maliyeti  : {safe_value(model.route_total):.2f}")
        print(f"4. Atama (Beta)   : {safe_value(model.assign_total):.2f}")
        print(f"5. Yukleme (Gama) : {safe_value(model.load_total):.2f}")
        print("-" * 40)
        print(f"TOPLAM OBJECTIVE  : {safe_value(model.obj):.2f}")
        print("="*40)
    else:
        print("\n" + "!"*40)
        print(f"HATA: Çözüm bulunamadı! Durum: {results.solver.termination_condition}")
        print("!"*40)

# Sonuçları ekrana basmak için
#model.pprint()

if (results.solver.status == SolverStatus.ok) or (results.solver.termination_condition == TerminationCondition.optimal):
    print(f"Çözüm Başarılı! Objective: {value(model.obj)}")
        
        # Excel'e yazmak için bir "Writer" oluşturuyoruz
    with pd.ExcelWriter("Model_Sonuclari.xlsx") as writer:
            
            # 1. AMAÇ FONKSİYONU DEĞERİ
        df_obj = pd.DataFrame({"Metric": ["Objective Value"], "Value": [value(model.obj)]})
        df_obj.to_excel(writer, sheet_name="Summary", index=False)

        # 2. L[f, b, k, s, t] - Yükleme Miktarları (Sadece 0'dan büyük olanlar)
        l_data = []
        for f, b, k, s, t in model.L:
            val = value(model.L[f, b, k, s, t])
            if val > 0.01:
                    l_data.append({"Supplier": f, "Hub": b, "Vehicle": k, "Scenario": s, "Time": t, "Amount": val})
        pd.DataFrame(l_data).to_excel(writer, sheet_name="L_Shipments", index=False)

        # 3. Q[b, i, t] - Hub'dan Depoya Giden Miktar
        q_data = []
        for b, i, t in model.Q:
            val = value(model.Q[b, i, t])
            if val > 0.01:
                q_data.append({"Hub": b, "Depot": i, "Time": t, "Amount": val})
        pd.DataFrame(q_data).to_excel(writer, sheet_name="Q_HubToDepot", index=False)

        # 4. X[r, t] - Rota Seçimleri (1 olanlar)
        x_data = []
        for r, t in model.X:
            val = value(model.X[r, t])
            if val > 0.5:
                x_data.append({"Route": r, "Time": t, "Selected": 1})
        pd.DataFrame(x_data).to_excel(writer, sheet_name="X_Routes", index=False)

        # 5. I[i, t] - Envanter Durumu
        i_data = []
        for i, t in model.I:
            i_data.append({"Depot": i, "Time": t, "InventoryLevel": value(model.I[i, t])})
        pd.DataFrame(i_data).to_excel(writer, sheet_name="I_Inventory", index=False)

        # 6. W[i, t] - Atık Miktarı
        w_data = []
        for i, t in model.W:
            val = value(model.W[i, t])
            if val > 0.01:
                w_data.append({"Depot": i, "Time": t, "Waste": val})
        pd.DataFrame(w_data).to_excel(writer, sheet_name="W_Waste", index=False)

    print("Sonuçlar 'Model_Sonuclari.xlsx' dosyasına kaydedildi.")
else:
    print("Model çözülemediği için Excel raporu oluşturulamadı.")
