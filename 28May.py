# -- coding: utf-8 --
"""
Created on Wed May 28 10:33:01 2025

@author: gizem.celik
"""
import pip
pip.main(["install", "openpyxl"])
import numpy as np
import pandas as pd
import random  # random.choice için gerekli
from collections import defaultdict

# Excel'den veri çekimi
file_path = "step1.xlsx"
beta_df = pd.read_excel(file_path, sheet_name="beta")
gamma_df = pd.read_excel(file_path, sheet_name="gamma")
theta_df = pd.read_excel(file_path, sheet_name="theta")
vehicle_owners_df = pd.read_excel(file_path, sheet_name="vehicles")
stock_costs_df = pd.read_excel(file_path, sheet_name="stock_costs")
supply_df = pd.read_excel(file_path, sheet_name="supply")
demand_df = pd.read_excel(file_path, sheet_name="demand")
route_costs_df = pd.read_excel(file_path, sheet_name="route_costs")
route_capacity_df = pd.read_excel(file_path, sheet_name="route_capacity")

# Format düzenleme
beta_df = beta_df.astype({"f": int, "k": int})
gamma_df = gamma_df.astype({"f": int, "b": int, "k": int})
theta_df = theta_df.astype({"k": int})
vehicle_owners_df = vehicle_owners_df.astype({"k": int, "f": int})
stock_costs_df = stock_costs_df.astype({"d": int})
supply_df = supply_df.astype({"f": int, "s": int, "t": int})
demand_df = demand_df.astype({"d": int, "t": int})
route_capacity_df = route_capacity_df.astype({"r": int})
route_costs_df.iloc[:, 0] = route_costs_df.iloc[:, 0].astype(int)
print(route_costs_df.dtypes)
# route_costs_df['r'] = route_costs_df['r'].astype(int)

#dataframeleri dict formatına çevirerek kodun geri kalanında erişimini kolaylaştırmış oluyoruz
beta_dict = {(row["f"], row["k"]): row["beta"] for _, row in beta_df.iterrows()}# Beta (atama maliyetleri): f, k, s, t -> cost
gamma_dict = {(row["f"], row["b"], row["k"]): row["gamma"] for _, row in gamma_df.iterrows()}# Gamma (gönderim maliyetleri): f, b, k, s, t -> cost
theta_dict = {row["k"]: row["theta"] for _, row in theta_df.iterrows()}# Araç kapasitesi: k -> capacity
vehicle_owners_dict = {row["k"]: row["f"] for _, row in vehicle_owners_df.iterrows()}# Araç sahipliği: f -> k
stock_cost_dict = {row["d"]: row["stock_cost"] for _, row in stock_costs_df.iterrows()}# Envanter maliyeti: d -> cost
supply_dict = {(row["f"], row["s"], row["t"]): row["supply"] for _, row in supply_df.iterrows()}# Arz miktarı: f, s, t -> amount
demand_dict = {(row["d"], row["t"]): row["demand"] for _, row in demand_df.iterrows()}# Talep miktarı: d, t -> demand
route_capacity_dict = {row["r"]: row["capacity"] for _, row in route_capacity_df.iterrows()}# Rota kapasitesi: r -> capacity
route_costs_dict = {row["r"]: row["cost"] for _, row in route_costs_df.iterrows()}# Rota sabit maliyeti: r -> cost


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

#Setler

F_set = sorted(beta_df["f"].unique())# Tedarikçiler
K_set = sorted(beta_df["k"].unique())# Araçlar
S_set = sorted(supply_df["s"].unique())# Senaryolar
T_set = sorted(supply_df["t"].unique())# Zaman periyotları
B_set = sorted(gamma_df["b"].unique())# Aktarma merkezleri (b)
D_set = sorted(stock_costs_df["d"].unique())# Depolar (demand veya stock_costs datasına göre)
R_set = sorted(route_to_depots.keys())  # Tüm rota ID’leri


# Parametreler
iterations = 10  # Toplam iterasyon sayısı
z_value = 2.575    # Güven aralığı katsayısı (örn. %99)
variation_rate = 0.10  # Standart sapma oranı
waste_cost = 27.55696873 #atık maliyeti
scenario_probs={1: 0.3, 2: 0.5, 3: 0.2}
shelf_life=2

#dictionaries
# mean değerlerini {(d, t): mean} sözlüğüne çevir
mean_dict = {(row["d"], row["t"]): row["demand"] for _, row in demand_df.iterrows()}
sigma_dict = {key: val * variation_rate for key, val in mean_dict.items()}



    
def generate_target_demand(demand_df, z_value, variation_rate, multi_period=False):
    #rng = np.random.default_rng(42) #şuanda bunu deterministik alacağım çünkü yaptığım değişikliklerin koda etkisini görmem gerek
                                    #Model oturunca bunu kaldırmayı unutma
    target_demand=[]
    
    for i in range(iterations):  # Her iterasyon için döngü başlat
        iter_name = f"iteration_{i+1}"  # İterasyon adı belirle
        for d in D_set:
            use_two = False
            for t in T_set:
                if use_two is True: 
                    use_two = False
                    continue
                if t < max(T_set):  # Eğer son zaman değilse çift dönem kararı verilebilir
                    use_two = random.choice([True, False])  # Modelden emin olunca bunu aç, Rastgele çift/tek karar ver
                    #use_two = rng.choice([True, False])  # 50/50 chance, şimdilik ekledin kaldır
                else:
                    use_two = False  # Son zamanda çift dönemli kullanım olmaz

                if use_two:  # Çift dönemli hesap
                    mu = mean_dict.get((d, t), 0) + mean_dict.get((d, t+1), 0)
                    std = np.sqrt(sigma_dict.get((d, t), 0)**2 + sigma_dict.get((d, t+1), 0)**2)
                else:  # Tek dönemli hesap
                    mu = mean_dict.get((d, t), 0)
                    std = sigma_dict.get((d, t), 0)

                val = max(0, int(round(mu + z_value * std)))  # Hedef talep hesapla

                # Sonuçlara ekle
                target_demand.append({
                    "iteration": iter_name,
                    "d": d,
                    "t": t,
                    "target": val,
                    "two_period": use_two
                })
    return pd.DataFrame(target_demand)



def select_routes_based_on_target(target_demand, route_to_depots):
    """
    Hedef talepleri karşılamak için rotaları seçer (hub target olmadan).
    İyileştirmeler:
      - Kapasite-duyarlı skor: score = route_cost / min(covered, capacity)
      - Dinamik greedy: her seçimden sonra skorlar yeniden hesaplanır
      - Rota içi depo önceliği: kalan talebi büyük olana öncelik
      - (r,t) başına tek kullanım: aynı rota aynı t'de yalnızca 1 kez çalıştırılır
    Tüm tahsisler tamsayıdır.
    """
    selected_routes = []

    for iteration_name in target_demand["iteration"].unique():
        tdf = target_demand[target_demand["iteration"] == iteration_name]

        for t in sorted(tdf["t"].unique()):
            # Bu t dönemi için depo bazlı kalan hedef (int)
            remaining = {
                int(row["d"]): int(row["target"])
                for _, row in tdf[tdf["t"] == t].iterrows()
                if int(row["target"]) > 0
            }
            if not remaining:
                continue  # bu t'de hedef yok

            used_routes = set()  # (r,t) başına tek kullanım

            # Kalan talep var oldukça, her adımda en iyi rotayı seçip tahsis et
            while True:
                # Tüm depolar doydu mu?
                if all(v <= 0 for v in remaining.values()):
                    break

                # Kullanılabilir rotaları kapasite-duyarlı skorla
                route_scores = []
                for r, depots in route_to_depots.items():
                    if r in used_routes:
                        continue  # aynı t içinde ikinci kez kullanma

                    cap = int(route_capacity_dict.get(r, 0))
                    if cap <= 0:
                        continue

                    # O anda bu rotanın kapsadığı toplam kalan talep
                    covered = sum(remaining.get(d, 0) for d in depots)
                    if covered <= 0:
                        continue

                    deliverable = min(covered, cap)                 # gerçekten taşınabilecek miktar (int)
                    cost = float(route_costs_dict.get(r, 10**12))   # skor için float bölme normal
                    score = cost / max(deliverable, 1)              # 0’a bölmeyi önle

                    route_scores.append((score, r, cap))

                # Seçilecek faydalı rota kalmadıysa dur
                if not route_scores:
                    break

                # En iyi skorlu rotayı seç
                route_scores.sort()
                _, r_best, cap_left = route_scores[0]
                depots = route_to_depots[r_best]

                # Depoları kalan talebe göre (büyükten küçüğe) sırala
                depots_ordered = sorted(depots, key=lambda d: remaining.get(d, 0), reverse=True)

                route_allocation = []
                for d in depots_ordered:
                    if cap_left <= 0:
                        break
                    need = remaining.get(d, 0)
                    if need <= 0:
                        continue
                    alloc = min(need, cap_left)   # int
                    if alloc > 0:
                        route_allocation.append((d, alloc))
                        remaining[d] = need - alloc
                        cap_left -= alloc

                # Bu rotayla gerçekten sevkiyat yapılabildiyse kayıt et ve rotayı kilitle
                used_routes.add(r_best)  # (r,t) tek kullanım kuralı
                if route_allocation:
                    for d, amount in route_allocation:
                        selected_routes.append({
                            "iteration": iteration_name,
                            "r": r_best,
                            "d": d,
                            "t": t,
                            "amount": int(amount)
                        })
                # route_allocation boşsa, rota bu t'de iş göremez; sonraki en iyi rotaya geçilecek

    return pd.DataFrame(selected_routes)





def calculate_hub_targets_from_selected_routes(selected_routes, route_to_hub):
    """
    Hub target = seçilmiş rotaların bağlı olduğu hub'ların (b) her t döneminde çekeceği
    PLANLANAN miktardır. (iteration,b,t) bazında amount toplamı döner.

    Beklenen kolonlar: selected_routes[['iteration','r','t','amount']]
    Dönen: DataFrame(['iteration','b','t','target_amount'])
    """

    # Boşsa boş tablo döndür
    if selected_routes is None or len(selected_routes) == 0:
        return pd.DataFrame(columns=["iteration", "b", "t", "target_amount"])

    # Sadece gereken kolonları al (fazla kolonlar varsa sorun etmez)
    sr = selected_routes[["iteration", "r", "t", "amount"]].copy()

    # r -> b eşlemesi: her rotanın bağlı olduğu hub
    sr["b"] = sr["r"].map(route_to_hub)

    # (iteration,b,t) bazında amount toplamı = hub target
    hub_targets_df = (
        sr.groupby(["iteration", "b", "t"], as_index=False)["amount"]
          .sum()
          .rename(columns={"amount": "target_amount"})
    )

    # İsteğe bağlı: int'e döndür (görüntü için)
    hub_targets_df["t"] = hub_targets_df["t"].astype(int)
    hub_targets_df["b"] = hub_targets_df["b"].astype(int)
    if pd.api.types.is_integer_dtype(sr["amount"].dtype):
        hub_targets_df["target_amount"] = hub_targets_df["target_amount"].astype(int)

    return hub_targets_df


# 2. Rota seçimi

def assign_suppliers(supply_dict, beta_dict, gamma_dict, theta_dict, prob_dict, vehicle_owners_df,
                     hub_targets, F_set, K_set, B_set, S_set, T_set):
    """
    BASİT HUB-ODAKLI ATAMA (greedy)
    - Her (iteration, b, t) hedefini, beklenen birim maliyeti en düşük (f,k) ile doldurur.
    - Kısıtlar: (f,s,t) arz, (k,t) araç kapasitesi, (k,t) tek hub, hedef ≤ hub_targets.
    - Seçim metriği (sade):  expected_gamma + beta/θ_k
        expected_gamma = sum_s p_s * gamma_{f,b,k}
        beta/θ_k       = aktivasyon sabitinin kapasiteye yayılmış (amortize) hali (sadece seçim rehberi)
    Not: β gerçek maliyette ilk aktivasyonda 1 kez sayılmalı; burada sadece seçim için kullanılır.
    Döner: DataFrame(['iteration','s','t','f','k','b','amount'])
    """



    # --- 0) Yardımcı: tedarikçinin kullanabileceği araç listesi (önce sahip oldukları, yoksa tüm K_set) ---
    owner_map = vehicle_owners_df.groupby("f")["k"].apply(list).to_dict()
    def vehicles_of(f):
        owned = owner_map.get(f, [])
        return owned if owned else list(K_set)

    # --- 1) Beklenen birim maliyet tablosu: c[(f,k,b)] ---
    #     c = sum_s p_s*gamma_{f,b,k} + (beta_{f,k}/theta_k)
    #     (sum_s p_s = 1 varsayımıyla, beta/θ_k zaten senaryo-bağımsızdır)
    c = {}
    for f in F_set:
        for k in K_set:
            theta_k = max(int(theta_dict.get(k, 0)), 1)
            beta_fk = float(beta_dict.get((f, k), 1e6))
            beta_term = beta_fk / theta_k
            for b in B_set:
                gamma_exp = 0.0
                for s in S_set:
                    pr = float(prob_dict.get(s, 0.0))
                    gamma_exp += pr * float(gamma_dict.get((f, b, k), 1e6))
                c[(f, k, b)] = gamma_exp + beta_term

    # --- 2) Ana döngü: iteration -> scenario -> time ---
    assignments= []
    for it in hub_targets["iteration"].unique():
        ht_it = hub_targets[hub_targets["iteration"] == it]

        for s in S_set:
            for t in T_set:
                # (k,t) kapasite; (f,s,t) arz; (k,t) tek hub kilidi
                rem_cap  = {int(k): int(theta_dict.get(k, 0)) for k in K_set}
                rem_sup  = {int(f): int(supply_dict.get((f, s, t), 0)) for f in F_set}
                hub_of_k = {}  # k -> b (bu t'de k sadece bir hub'a hizmet eder)

                # Bu t'deki hub hedefleri (büyükten küçüğe) — daha hızlı dolum
                ht_t = ht_it[ht_it["t"] == t]
                bt_list = [(int(r["b"]), int(r["target_amount"])) for _, r in ht_t.iterrows()]
                bt_list.sort(key=lambda x: x[1], reverse=True)

                for b, target in bt_list:
                    remaining = target
                    if remaining <= 0:
                        continue

                    # Hedef bitene kadar en ucuz (f,k) ile doldur
                    while remaining > 0:
                        best_f, best_k, best_cost = None, None, float("inf")

                        # adayları tara: arzı olan f ve kapasitesi/hub uygun k
                        for f in F_set:
                            if rem_sup[f] <= 0:
                                continue
                            for k in vehicles_of(f):
                                k = int(k)
                                if rem_cap.get(k, 0) <= 0:
                                    continue
                                if k in hub_of_k and hub_of_k[k] != b:
                                    continue  # bu t'de k başka hub'a kilitli

                                unit_cost = c[(f, k, b)]
                                if unit_cost < best_cost:
                                    best_f, best_k, best_cost = f, k, unit_cost

                        if best_f is None:   # aday kalmadı, hedefin bir kısmı karşılanamayabilir
                            break

                        qty = min(remaining, rem_sup[best_f], rem_cap[best_k])
                        if qty <= 0:
                            break

                        # aracı bu t'de bu hub'a kilitle
                        if best_k not in hub_of_k:
                            hub_of_k[best_k] = b

                        # kayıt
                        assignments.append({
                            "iteration": it,
                            "s": s,
                            "t": t,
                            "f": best_f,
                            "k": best_k,
                            "b": b,
                            "amount": int(qty)
                        })

                        # state güncelle
                        remaining         -= qty
                        rem_sup[best_f]   -= qty
                        rem_cap[best_k]   -= qty

    return pd.DataFrame(assignments)



def build_depot_deliveries(selected_routes, D_set, T_set):
    """
    selected_routes: DataFrame ['iteration','r','d','t','amount']
    D_set, T_set   : depolar ve dönemler (liste)
    Dönen          : DataFrame ['iteration','d','t','delivered']  (tüm (it,d,t) kombinasyonları dolu, boşlar 0)
    """
    if selected_routes is None or selected_routes.empty:
        return pd.DataFrame(columns=['iteration','d','t','delivered'])

    # (iteration,d,t) bazında toplam teslimatı hesapla
    agg = (selected_routes
           .groupby(['iteration','d','t'], as_index=False)['amount']
           .sum()
           .rename(columns={'amount':'delivered'}))

    # Tüm (iteration,d,t) kombinasyonlarını üret ve 0’larla doldur
    iters = agg['iteration'].unique().tolist()
    grid = (pd.MultiIndex.from_product([iters, D_set, T_set],
                                       names=['iteration','d','t'])
            .to_frame(index=False))

    deliveries = grid.merge(agg, how='left', on=['iteration','d','t'])
    deliveries['delivered'] = deliveries['delivered'].fillna(0)

    # Görsel/uyum için tipleri toparla
    deliveries['d'] = deliveries['d'].astype(int)
    deliveries['t'] = deliveries['t'].astype(int)
    deliveries['delivered'] = deliveries['delivered'].astype(int)

    return deliveries




def fifo_inventory_and_waste(selected_routes, demand_dict, T_set):
    """
    Her (iteration, d, t) için:
      - Talep önce geçen dönemden devreden envanterden (carry) karşılanır.
      - Kalan talep bu dönemin teslimatından karşılanır.
      - Carry'den arta kalan miktar ATIKTIR (bu dönemde kullanılamadı).
      - Teslimattan arta kalan miktar BİR SONRAKİ döneme ENVANTER olur.
      - Son dönemde de (t = time_periods[-1]) teslimattan artan envanterde kalır (atık değil).
    Döner: waste_df, remaining_inventory_df
    """
    if selected_routes is None or selected_routes.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Depo kümesi: rotalarda veya talepte geçen tüm depolar
    depots_from_routes = set(selected_routes['d'].unique())
    depots_from_demand = {d for (d, t) in demand_dict.keys() if t in set(T_set)}
    depots = sorted(depots_from_routes | depots_from_demand)

    all_waste, all_inv = [], []

    for it in selected_routes['iteration'].unique():
        sr_it = selected_routes[selected_routes['iteration'] == it]

        # (d,t) teslimat matrisi (eksikler 0)
        deliveries = (
            sr_it.groupby(['d','t'], as_index=False)['amount'].sum()
                 .pivot(index='d', columns='t', values='amount')
                 .reindex(index=depots, columns=T_set, fill_value=0)
                 .fillna(0) 
        )

        waste_it = pd.DataFrame(0, index=depots, columns=T_set)
        inv_it   = pd.DataFrame(0, index=depots, columns=T_set)

        for d in depots:
            carry = 0  # sadece bir dönem devreder (t -> t+1)

            for t in T_set:
                demand    = int(demand_dict.get((d, t), 0))
                delivered = int(deliveries.loc[d, t])

                # 1) Önce eldeki envanterden (carry) tüket
                use_from_carry   = min(carry, demand)
                carry_leftover   = carry - use_from_carry      # bu dönem kullanılamayan eski envanter
                demand_remaining = demand - use_from_carry

                # 2) Sonra bu dönemin teslimatından tüket
                use_from_delivery    = min(delivered, demand_remaining)
                delivered_leftover   = delivered - use_from_delivery
                # demand_remaining   -= use_from_delivery  # backorder izlenmiyorsa gerek yok

                # 3) Dönem sonu: carry'den kalan = ATIK, teslimattan kalan = ENVANTER
                waste_it.loc[d, t] = carry_leftover
                inv_it.loc[d, t]   = delivered_leftover

                # 4) Bir sonraki döneme devreden envanteri güncelle
                carry = delivered_leftover

        waste_it['iteration'] = it
        inv_it['iteration']   = it
        all_waste.append(waste_it)
        all_inv.append(inv_it)

    waste_df = pd.concat(all_waste, ignore_index=False)
    remaining_inventory_df = pd.concat(all_inv, ignore_index=False)
    return waste_df, remaining_inventory_df




                        


def calculate_comprehensive_costs(selected_routes, suppliers_assignments, waste_df, remaining_inventory_df, 

                                route_costs_dict, stock_cost_dict, waste_cost, scenario_probs):
    """
    Her iterasyon için kapsamlı maliyet hesaplaması yapar:
    - Rota maliyetleri
    - Tedarikçi atama maliyetleri (beta + gamma)
    - Envanter maliyetleri  
    - Atık maliyetleri
    """
    all_costs = []

    for iteration in selected_routes['iteration'].unique():
        iteration_costs = {
            'iteration': iteration,
            'route_costs': 0,
            'assignment_costs': 0,
            'transportation_costs': 0,
            'inventory_costs': 0,
            'waste_costs': 0,
            'total_cost': 0
        }

        # 1. Rota maliyetleri hesaplama
    
        # 1. Rota maliyetleri hesaplama - her (r, t) kombinasyonu için sadece bir kez maliyet eklenir
        selected_routes_iter = selected_routes[selected_routes['iteration'] == iteration]
        unique_r_t = selected_routes_iter[['r', 't']].drop_duplicates()
        route_costs = 0
        for _, row in unique_r_t.iterrows():
            r = row['r']
            lambda_r = route_costs_dict.get(r, 0)
            route_costs += lambda_r

        iteration_costs['route_costs'] = route_costs

       # 2. Atama maliyetleri (β) ve Taşıma maliyetleri (γ)
        suppliers_iter = suppliers_assignments[suppliers_assignments['iteration'] == iteration]

        # --- β kısmı ---
        g_beta = suppliers_iter.groupby(['s','t','f','k'], as_index=False)['amount'].sum()
        g_beta['Y']    = (g_beta['amount'] > 0).astype(int)
        g_beta['beta'] = g_beta.apply(lambda r: beta_dict.get((int(r['f']), int(r['k'])), 0.0), axis=1)
        g_beta['pr']   = g_beta['s'].map(lambda s: float(scenario_probs.get(s, 1.0)))
        assignment_costs = float((g_beta['pr'] * g_beta['beta'] * g_beta['Y']).sum())

        # --- γ kısmı ---
        g_gamma = suppliers_iter.groupby(['s','t','f','k','b'], as_index=False)['amount'].sum()
        g_gamma['gamma'] = g_gamma.apply(lambda r: gamma_dict.get((int(r['f']), int(r['b']), int(r['k'])), 0.0), axis=1)
        g_gamma['pr']    = g_gamma['s'].map(lambda s: float(scenario_probs.get(s, 1.0)))
        transportation_costs = float((g_gamma['pr'] * g_gamma['gamma'] * g_gamma['amount']).sum())

        iteration_costs['assignment_costs']     = assignment_costs
        iteration_costs['transportation_costs'] = transportation_costs

        # 3. Envanter maliyetleri
        inventory_iter = remaining_inventory_df[remaining_inventory_df['iteration'] == iteration] if 'iteration' in remaining_inventory_df.columns else remaining_inventory_df
        inventory_costs = 0
        for d in inventory_iter.index:
            if d in stock_cost_dict:
                for t in T_set:
                    if t in inventory_iter.columns:
                        inventory_amount = inventory_iter.loc[d, t]
                        inventory_costs += inventory_amount * stock_cost_dict[d]
        iteration_costs['inventory_costs'] = inventory_costs

        # 4. Atık maliyetleri
        waste_iter = waste_df[waste_df['iteration'] == iteration] if 'iteration' in waste_df.columns else waste_df
        waste_costs = 0
        for d in waste_iter.index:
            for t in T_set:
                if t in waste_iter.columns:
                    waste_amount = waste_iter.loc[d, t]
                    waste_costs += waste_amount * waste_cost
        iteration_costs['waste_costs'] = waste_costs
        
        # 5. Toplam maliyet
        iteration_costs['total_cost'] = (iteration_costs['route_costs'] + 
                                       iteration_costs['assignment_costs'] + 
                                       iteration_costs['transportation_costs']+
                                       iteration_costs['inventory_costs'] + 
                                       iteration_costs['waste_costs'])     

        all_costs.append(iteration_costs)

    return pd.DataFrame(all_costs)

def print_cost_summary(comprehensive_costs_df):
    """
    Kapsamlı maliyet analizi, en iyi iterasyonu bulma ve özetleri yazdırır.
    """
    print("=" * 80)
    print("KAPSAMLI MALİYET ANALİZİ - TÜM İTERASYONLAR")
    print("=" * 80)

    # Toplam maliyet özeti
    total_costs = comprehensive_costs_df.groupby('iteration').agg({
        'route_costs': 'sum',
        'assignment_costs': 'sum',
        'transportation_costs': 'sum',
        'inventory_costs': 'sum',
        'waste_costs': 'sum',
        'total_cost': 'sum'
    })

    # 🔍 En düşük maliyetli iterasyonu bul
    best_iter = total_costs['total_cost'].idxmin()
    best_row = total_costs.loc[best_iter]

    print(f"\n🏆 En Düşük Maliyetli İterasyon: {best_iter.upper()}")
    print(f"   • TOPLAM MALİYET: {best_row['total_cost']:,.2f} TL")
    print("   • Maliyet Dağılımı:")
    print(f"     - Rota:     {best_row['route_costs']:,.2f} TL")
    print(f"     - Atama:    {best_row['assignment_costs']:,.2f} TL")
    print(f"     - Taşıma:   {best_row['transportation_costs']:,.2f} TL")
    print(f"     - Envanter: {best_row['inventory_costs']:,.2f} TL")
    print(f"     - Atık:     {best_row['waste_costs']:,.2f} TL")
    
    # Sadece best iter için yüzdesel dağılım
    total = best_row['total_cost']
    if total > 0:
        print(f"\n📈 Maliyet Dağılımı (%):")
        print(f"     - Rota:     {(best_row['route_costs']/total)*100:>6.1f}%")
        print(f"     - Atama:    {(best_row['assignment_costs']/total)*100:>6.1f}%")
        print(f"     - Taşıma:   {(best_row['transportation_costs']/total)*100:>6.1f}%")
        print(f"     - Envanter: {(best_row['inventory_costs']/total)*100:>6.1f}%")
        print(f"     - Atık:     {(best_row['waste_costs']/total)*100:>6.1f}%")
    
    print("\n" + "=" * 80)
    print("ANALİZ TAMAMLANDI - En iyi iterasyon sonuçları kaydedildi.")
    print("=" * 80)

    return best_iter

# 1. Hedef talep oluştur
target_demand = generate_target_demand(demand_df, z_value, variation_rate, multi_period=False)

# 2. Rota seçimi

selected_routes = select_routes_based_on_target(target_demand, route_to_depots)


# 3. Hub hedefleri hesapla
hub_targets_df = calculate_hub_targets_from_selected_routes(selected_routes, route_to_hub)



# Save hub_targets_df to CSV
hub_targets_df.to_csv("hub_targets_all_iterations.csv", index=False)

### ✅ KONTROL 1: Aynı depo-zaman birden fazla rotada mı?
selected_routes_depots = selected_routes.copy()
selected_routes_depots["r_depots"] = selected_routes_depots["r"].map(route_to_depots)

from collections import defaultdict
dup_check = defaultdict(int)

for _, row in selected_routes_depots.iterrows():
    iteration = row["iteration"]
    t = row["t"]
    r = row["r"]
    depots = route_to_depots.get(r, [])
    for d in depots:
        dup_check[(iteration, t, d)] += 1

dup_check_df = pd.DataFrame([
    {"iteration": it, "t": t, "d": d, "num_routes": count}
    for (it, t, d), count in dup_check.items() if count > 1
])

print("[KONTROL 1] Aynı (d,t) için birden fazla rota:")
print(dup_check_df.sort_values(by=["iteration", "t", "d"]))


### ✅ KONTROL 2: Toplam talep ile hub targets eşleşiyor mu?
total_demand = target_demand.groupby("iteration")["target"].sum().reset_index(name="total_demand")
hub_totals = hub_targets_df.groupby("iteration")["target_amount"].sum().reset_index(name="total_from_hub_targets")

merged = pd.merge(total_demand, hub_totals, on="iteration")
merged["difference"] = merged["total_from_hub_targets"] - merged["total_demand"]

print("[KONTROL 2] Toplam hedef farkı:")
print(merged)
# 4. Tedarikçi atamaları
suppliers_assignments = assign_suppliers(
    supply_dict, beta_dict, gamma_dict, theta_dict, scenario_probs,
    vehicle_owners_df, hub_targets_df, F_set, K_set, B_set, S_set, T_set)

deliveries = build_depot_deliveries(selected_routes, D_set, T_set)



# 5. FIFO bazlı atık ve stok takibi
waste_df, remaining_inventory_df = fifo_inventory_and_waste(selected_routes, demand_dict, T_set)

# 6. Maliyet hesaplamaları
comprehensive_costs_df = calculate_comprehensive_costs(
    selected_routes, suppliers_assignments, waste_df, remaining_inventory_df,
    route_costs_dict, stock_cost_dict, waste_cost, scenario_probs)

# 7. En iyi iterasyonu bul
best_iteration_row = comprehensive_costs_df.loc[comprehensive_costs_df['total_cost'].idxmin()]
best_iteration = best_iteration_row['iteration']
print(f"\n✅ En düşük maliyetli iterasyon: {best_iteration}")
print(f"💰 Toplam maliyet: {best_iteration_row['total_cost']:,.2f} TL")

# 8. Sadece en iyi iterasyonun sonuçlarını CSV olarak kaydet
target_demand[target_demand['iteration'] == best_iteration].to_csv("best_target_demand.csv", index=False)
selected_routes[selected_routes['iteration'] == best_iteration].to_csv("best_selected_routes.csv", index=False)
suppliers_assignments[suppliers_assignments['iteration'] == best_iteration].to_csv("best_suppliers_assignments.csv", index=False)
deliveries[deliveries['iteration']== best_iteration].to_csv("best_depot_deliveries.csv", index=False)
waste_df[waste_df['iteration'] == best_iteration].to_csv("best_waste_df.csv", index=False)
remaining_inventory_df[remaining_inventory_df['iteration'] == best_iteration].to_csv("best_remaining_inventory_df.csv", index=False)

# 9. Sadece tüm iterasyonların maliyet analizini kaydet
comprehensive_costs_df.to_csv("comprehensive_costs_all_iterations.csv", index=False)

# 10. Raporlama
best_iter = print_cost_summary(comprehensive_costs_df)

# --- [EK KONTROL] Best Iterasyon için Hub Giriş/Çıkış Miktarları Karşılaştırması ---

# 1. Hub'a gelen toplam miktar (best_iteration için)
incoming = hub_targets_df[hub_targets_df['iteration'] == best_iteration] \
    .groupby(['b', 't'])['target_amount'].sum().reset_index(name='incoming_to_hub')

# 2. Hub'dan çıkan toplam miktar (selected_routes üzerinden)
selected_routes_best = selected_routes[selected_routes['iteration'] == best_iteration].copy()
selected_routes_best['b'] = selected_routes_best['r'].map(route_to_hub)
outgoing = selected_routes_best.groupby(['b', 't'])['amount'].sum().reset_index(name='outgoing_from_hub')

# 3. Karşılaştırma
comparison = pd.merge(incoming, outgoing, on=['b', 't'], how='outer')
comparison['difference'] = comparison['incoming_to_hub'] - comparison['outgoing_from_hub']

# 5. CSV olarak dışa aktar
comparison.to_csv(f"hub_flow_comparison_{best_iteration}.csv", index=False)




