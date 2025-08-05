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
iterations = 3  # Toplam iterasyon sayısı
z_value = 2.575    # Güven aralığı katsayısı (örn. %99)
variation_rate = 0.10  # Standart sapma oranı
waste_cost = 27.55696873 #atık maliyeti
scenario_probs={1: 0.3, 2: 0.5, 3: 0.2}
shelf_life=2

#dictionaries
# mean değerlerini {(d, t): mean} sözlüğüne çevir
mean_dict = {(row["d"], row["t"]): row["demand"] for _, row in demand_df.iterrows()}
sigma_dict = {key: val * variation_rate for key, val in mean_dict.items()}

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
        selected_routes_iter = selected_routes[selected_routes['iteration'] == iteration]
        route_costs = 0
        for r in selected_routes_iter['r'].unique():
            route_costs += route_costs_dict.get(r, 0)
        iteration_costs['route_costs'] = route_costs

        # 2. Tedarikçi atama maliyetleri (beta + gamma) - senaryo olasılıkları ile ağırlıklandırılmış
        suppliers_iter = suppliers_assignments[suppliers_assignments['iteration'] == iteration] if 'iteration' in suppliers_assignments.columns else suppliers_assignments
        assignment_costs = 0
        transportation_costs = 0

        seen_assignments = set() # Track unique Y variables to avoid double counting in assignment cost

        for _, row in suppliers_iter.iterrows():

            s, f, k, b, amount = row['s'], row['f'], row['k'], row['b'], row['amount']
            prob = scenario_probs.get(s, 1)

            # --- (1) Assignment Cost: pr_s * β_{f,k} * Y_{f,b,k,s,t}
            # We assume Y=1 if this row exists; count only once per (f,k,s,t)
            assignment_key = (f, k, s, row['t'])  # add t if needed
            if assignment_key not in seen_assignments:
                beta_cost = beta_dict.get((f, k), 0)
                assignment_costs += prob * beta_cost
                seen_assignments.add(assignment_key)

            # --- (2) Transportation Cost: pr_s * γ_{f,b,k} * L_{f,b,k,s,t}
            gamma_cost = gamma_dict.get((f, b, k), 0)
            transportation_costs += prob * amount * gamma_cost

        iteration_costs['assignment_costs'] = assignment_costs
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

    
def generate_target_demand(demand_df, z_value, variation_rate, multi_period=False):
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
                    use_two = random.choice([True, False])  # Rastgele çift/tek karar ver
                else:
                    use_two = False  # Son zamanda çift dönemli kullanım olmaz

                if use_two:  # Çift dönemli hesap
                    mu = mean_dict.get((d, t), 0) + mean_dict.get((d, t+1), 0)
                    std = np.sqrt(sigma_dict.get((d, t), 0)*2 + sigma_dict.get((d, t+1), 0)*2)
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
    Hedef talepleri karşılamak için en uygun rotaları seçer.
    Greedy yaklaşımla çalışır. Senaryolardan bağımsızdır.
    """
    selected_routes = []  # Seçilen rotaların dağıtım planını tutacak liste
    for iteration_name in target_demand["iteration"].unique():
        target_demand_for_iteration = target_demand[target_demand["iteration"] == iteration_name]
        for t in sorted(target_demand_for_iteration["t"].unique()):  # Her zaman periyodu için sırayla çalış
            # Bu zaman dilimindeki her deponun talebini al
            remaining_demand = {
                row["d"]: row["target"]
                for _, row in target_demand_for_iteration[target_demand_for_iteration["t"] == t].iterrows()
            }

            # covered_depots_at_t = list(set(range(1, 31)) - set(remaining_demand.keys()))

            # route_to_depots_temp = route_to_depots
            # for route, depots in route_to_depots_temp:


            route_scores = []  # Rotaların skorlarını tut (maliyet / kapsanan talep)

            for r, depots in route_to_depots.items():  # Her rota için
                capacity = route_capacity_dict.get(r, 0)  # Rota kapasitesini al
                cost = route_costs_dict.get(r, 1e6)        # Rota sabit maliyetini al
                # Bu rotanın kapsadığı depolardaki toplam hedef talep
                covered = sum(remaining_demand.get(d, 0) for d in depots)

                if covered > 0:
                    score = cost / covered  # Skor: maliyet / kapsadığı hedef talep
                    route_scores.append((score, r))  # Listeye ekle

            route_scores.sort()  # Skoru düşük olanlar daha avantajlı → sırala

            for _, r in route_scores:  # En iyi skorlu rotadan başlayarak sırayla
                depots = route_to_depots[r]  # Rota üzerindeki depolar
                capacity = route_capacity_dict.get(r, 0)  # Bu rotanın kapasitesi
                route_allocation = []  # Bu rotada hangi depoya ne kadar gönderildi

                for d in depots:  # Rota üzerindeki her depo için
                    if remaining_demand.get(d, 0) <= 0:
                        continue  # Bu deponun ihtiyacı kalmadıysa geç

                    allocate = min(remaining_demand[d], capacity)  # Ne kadar karşılanabilir?
                    if allocate > 0:
                        route_allocation.append((d, allocate))  # Atamayı listeye ekle
                        remaining_demand[d] -= allocate  # Depo ihtiyacından düş
                        capacity -= allocate  # Rotanın kalan kapasitesini azalt

                        if capacity <= 0:  # Kapasite bittiğinde çık
                            break

                # Bu rotada en az bir depo için ürün gönderildiyse
                if route_allocation:
                    for d, amount in route_allocation:
                        selected_routes.append({
                            "iteration": iteration_name,
                            "r": r,       # Rota numarası
                            "d": d,       # Depo numarası
                            "t": t,       # Zaman periyodu
                            "amount": amount  # Gönderilen miktar
                        })

    # Sonuçları DataFrame olarak döndür
    return pd.DataFrame(selected_routes)


def calculate_hub_targets_from_selected_routes(selected_routes, target_demand_df, route_to_depots, route_to_hub):
    """
    Seçilen rotalara göre hub'lara zaman bazlı gönderilecek ürün miktarını hesaplar.
    selected_routes: Seçilen rota numaraları
    target_demand_df: pd.DataFrame - iteration, d, t, target bilgisi
    route_to_depots: dict - r → [d1, d2, ...]
    route_to_am: dict - r → b
    """
    first_column = selected_routes.values[:, 1]        # get first column (e.g., 22, 30, 24...)
    unique_values = np.unique(first_column)            # get unique values
    unique_list = unique_values.tolist()    

    # Define hub_targets_df outside iteration loop to accumulate all values
    hub_targets_df = pd.DataFrame(columns=["iteration", "b", "t", "target_amount"])

    # TODO 
    # t for loopu olacak şekilde dışarıda olacak refactoring yapmamız lazım 
    # r loopu t den sonra olacak 
    # 
    # TODO 
    for iter in target_demand_df['iteration'].unique():
        hub_targets = defaultdict(float)  # (b, t) → toplam ihtiyaç
        target_demand_for_iter=target_demand_df[target_demand_df['iteration']==iter]
        selected_routes_for_iter=selected_routes[selected_routes['iteration']==iter]
        for t in sorted(target_demand_for_iter['t'].unique()):
            
            selected_routes_for_t=selected_routes_for_iter[selected_routes_for_iter['t']==t]
            for r in selected_routes_for_t['r'].unique():#seçilen her bir rota için döngü başlat
                depots = route_to_depots.get(r, []) #bu rota hangi depoları kapsıyor
                b = route_to_hub[r] #rota hangi hubtan başlıyor

            #aşağıdaki isin methodu Depo d kolonundaki değerlerin depots listesinde olup olmadığını kontrol eder.Örneğin: depots = [25, 24, 14, 13] ise, d kolonu içinde bu 4 depoyu filtreler.
                total = target_demand_for_iter[
                    (target_demand_for_iter["d"].isin(depots)) & (target_demand_for_iter["t"] == t)
                ]["target"].sum()
                hub_targets[(b, t)] += total #oplam talebi hub b için, zaman t'de gönderilmesi gereken miktar olarak hub_targets'a ekler. Böylece aynı (b,t) için farklı rotalarla gelen değerler toplanır.


        # İsteğe bağlı: DataFrame formatında da dönebiliriz - append to existing dataframe
        iter_hub_targets_df = pd.DataFrame([
            {"iteration":iter,"b": b, "t": t, "target_amount": amt}
            for (b, t), amt in hub_targets.items()
        ])
        hub_targets_df = pd.concat([hub_targets_df, iter_hub_targets_df], ignore_index=True)
        # Her iteration, b ve t için bir satır olacak şekilde gruplandır ve topla
        #hub_targets_df = hub_targets_df.groupby(["iteration", "b", "t"], as_index=False)["target_amount"].sum()

    return hub_targets_df



def get_target_amount(remaining_targets, b, t, default_value=0):
    """
    Get the target amount for a specific hub (b) and time period (t).
    
    Parameters:
    - remaining_targets: DataFrame with columns 'b', 't', 'target_amount'
    - b: hub identifier
    - t: time period
    
    Returns:
    - target_amount: float, the target amount if found, 0 if not found or empty
    """
    """""
    targets_row = remaining_targets[(remaining_targets['b'] == b) & (remaining_targets['t'] == t)]
    if targets_row.empty:
        return default_value
    return targets_row.iloc[0, remaining_targets.columns.get_loc('target_amount')]
    """

    filtered = remaining_targets[(remaining_targets['b'] == b) & (remaining_targets['t'] == t)]
    return filtered["target_amount"].sum() if not filtered.empty else default_value

def assign_suppliers(supply_dict, beta_dict, gamma_dict, theta_dict, prob_dict, vehicle_owners_df,
                     hub_targets, F_set, K_set, B_set, S_set, T_set):
    """
    Tedarikçilerin araçlara ve hub’lara atanmasını yapar.
    Her senaryoda ve her zaman periyodunda aynı hub_targets’a göre hedef karşılanır.
    """

    assignments = []  # Tüm atamaları tutacak liste

  
    """
    # ADIM 1: Toplam arzı yazdır (s=1, t=1)
    total_supply_s1t1 = sum(
        v for (f_, s_, t_), v in supply_dict.items() if s_ == 1 and t_ == 1
    )
    print(f"[KONTROL] Senaryo 1, Zaman 1 için toplam tedarikçi arzı: {total_supply_s1t1}")
    """

    for iter in hub_targets['iteration'].unique():
        hub_targets_for_iter = hub_targets[hub_targets['iteration'] == iter]
        print(f"[İZLEME] {iter} için atama işlemi başlatıldı.")

        for s in S_set:  # Senaryo döngüsü
            for t in T_set:
                for b in B_set:
               
                # Bu senaryo için hedefleri yeniden başlat (senaryosuz hedef)
                    remaining_targets = hub_targets_for_iter[(hub_targets_for_iter["t"] == t) & (hub_targets_for_iter["b"] == b)].copy()
                
                    vehicle_cap = {k: theta_dict.get(k, 0) for k in K_set}

                    for f in F_set:
                        supply = supply_dict.get((f, s, t), 0)
                        if supply <= 0:
                            continue
                        
                        # Eğer tedarikçinin kendi aracı varsa sadece onu kullanır
                        owned_vehicles = vehicle_owners_df[vehicle_owners_df["f"] == f]["k"].tolist()
                        candidate_vehicles = owned_vehicles if owned_vehicles else K_set

                        retry = True  # En az bir atama yapılana kadar dön
                        while supply > 0:
                            retry = False  # Eğer bu turda atama yapılmazsa çıkılacak

                            best_cost = float("inf")
                            best_choice = None

                            for k in candidate_vehicles:
                                if vehicle_cap[k] <= 0:
                                    continue

                                for _, row in remaining_targets.iterrows():
                                    b = row['b']
                                    remaining = row['target_amount']
                                    if remaining <= 0:
                                        continue

                                    gamma_cost = gamma_dict.get((f, b, k), 1e6)
                                    beta_cost = beta_dict.get((f, k), 1e6)
                                    total_cost = prob_dict[s] * (beta_cost + gamma_cost)
                                   
                                    if total_cost < best_cost:
                                        best_cost = total_cost
                                        best_choice = (k, b)

                            # Eğer uygun eşleşme bulunduysa
                            if best_choice:
                                k, b = best_choice
                                target_amt = remaining_targets[
                                    (remaining_targets['b'] == b) & (remaining_targets['t'] == t)
                                    ]['target_amount'].values[0]

                                #target_amt = remaining_targets.loc[remaining_targets['b'] == b, 'target_amount'].values[0]
                                assign_qty = min(supply, vehicle_cap[k], target_amt)

                                vehicle_cap[k] -= assign_qty
                                supply -= assign_qty
                                # Hedeften düş
                                remaining_targets.loc[
                                    (remaining_targets['b'] == b) & (remaining_targets['t'] == t),
                                    'target_amount'
                                    ] -= assign_qty

                                #remaining_targets.loc[remaining_targets['b'] == b, 'target_amount'] -= assign_qty
                            else:
                                break # Uygun eşleşme kalmadıysa çık
                                
                            assignments.append({
                                    "iteration": iter,
                                    "s": s,
                                    "t": t,
                                    "f": f,
                                    "k": k,
                                    "b": b,
                                    "amount": assign_qty
                                })


    return pd.DataFrame(assignments)



def fifo_inventory_and_waste(selected_routes, demand_dict, time_periods, shelf_life=2):
    """
    FIFO mantığı ile her depo ve zaman periyodu için envanter ve atık takibi yapar.

    Parametreler:
    - depot_delivery: DataFrame (index=depots, columns=time_periods), her zaman ve depo için gelen ürün miktarı
    - demand_dict: Dictionary {(d, t): demand}, talep değerleri
    - time_periods: list, örn. [1, 2, 3]
    - shelf_life: int, raf ömrü (örneğin 2 → 2 dönem sonra ürün bozulur)

    Dönüş:
    - waste_df: DataFrame, her zaman ve depo için oluşan atık miktarı
    - remaining_inventory_df: DataFrame, her zaman ve depo için dönem sonunda kalan toplam envanter
    """

    # Lists to store DataFrames from all iterations
    all_waste_dfs = []
    all_remaining_inventory_dfs = []

    # Her zaman periyodu için işlem yap
    for iter in selected_routes['iteration'].unique():
        selected_routes_for_iter = selected_routes[selected_routes['iteration'] == iter]
        depot_delivery_for_iter = selected_routes_for_iter.groupby(["d", "t"])["amount"].sum().unstack(fill_value=0)

        # Her depo için yaş bazlı envanteri tutan dict (örn. {d1: {0: 10, 1: 5}})
        inventory_for_iter = {d: {age: 0 for age in range(shelf_life)} for d in depot_delivery_for_iter.index}

        # Çıktılar: atık ve kalan envanter tabloları
        waste_df_for_iter = pd.DataFrame(0, index=depot_delivery_for_iter.index, columns=time_periods)
        remaining_inventory_df_for_iter = pd.DataFrame(0, index=depot_delivery_for_iter.index, columns=time_periods)

        for t in time_periods:
            for d in depot_delivery_for_iter.index:
                # 1. Envanteri yaşlandır
                updated_inventory = {age: 0 for age in range(shelf_life)}
                for age in range(shelf_life - 1):
                    updated_inventory[age + 1] = inventory_for_iter[d][age]

                # 2. Raf ömrünü aşan ürünleri atık olarak yaz
                waste_df_for_iter.loc[d, t] += inventory_for_iter[d][shelf_life - 1]

                # 3. Talep değerini al
                demand = demand_dict.get((d, t), 0)

                # 4. FIFO ile talebi yaşlandırılmış stoktan karşıla
                for age in sorted(updated_inventory.keys()):
                    if demand <= 0:
                        break
                    usable = min(demand, updated_inventory[age])
                    updated_inventory[age] -= usable
                    demand -= usable

                # 5. Yeni teslimat al
                delivered = depot_delivery_for_iter.loc[d, t]

                # 6. Talep hala kaldıysa teslimattan karşıla
                used_from_delivery = min(demand, delivered)
                delivered -= used_from_delivery
                demand -= used_from_delivery

                # 7. Kalan teslimatı yaş 0 olarak stoğa ekle
                updated_inventory[0] += delivered

                # 8. Envanteri güncelle
                inventory_for_iter[d] = updated_inventory

                # 9. Dönem sonu kalan stok toplamını kaydet
                remaining_inventory_df_for_iter.loc[d, t] = sum(updated_inventory.values())

        # Add iteration identifier and store the DataFrames
        waste_df_for_iter['iteration'] = iter
        remaining_inventory_df_for_iter['iteration'] = iter

        all_waste_dfs.append(waste_df_for_iter)
        all_remaining_inventory_dfs.append(remaining_inventory_df_for_iter)

    # Concatenate all iterations into single DataFrames
    if all_waste_dfs:
        waste_df = pd.concat(all_waste_dfs, ignore_index=False)
        remaining_inventory_df = pd.concat(all_remaining_inventory_dfs, ignore_index=False)
    else:
        waste_df = pd.DataFrame()
        remaining_inventory_df = pd.DataFrame()

    return waste_df, remaining_inventory_df

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
    """
    # Tüm iterasyonlar için detaylı maliyet dağılımı
    print("\n📊 İTERASYON BAZLI MALİYET DAĞILIMI:")
    for iteration in total_costs.index:
        costs = total_costs.loc[iteration]
        print(f"\n🔍 {iteration.upper()}:")
        print(f"   • Rota Maliyetleri:         {costs['route_costs']:>12,.2f} TL")
        print(f"   • Tedarikçi Atama Maliyetleri: {costs['assignment_costs']:>8,.2f} TL")
        print(f"   • Taşıma Maliyetleri:       {costs['transportation_costs']:>8,.2f} TL")
        print(f"   • Envanter Maliyetleri:     {costs['inventory_costs']:>12,.2f} TL")
        print(f"   • Atık Maliyetleri:         {costs['waste_costs']:>12,.2f} TL")
        print(f"   • TOPLAM MALİYET:           {costs['total_cost']:>12,.2f} TL")

        # Yüzdesel dağılım
        total = costs['total_cost']
        if total > 0:
            print(f"   📈 Maliyet Dağılımı (%):")
            print(f"     - Rota:         {(costs['route_costs']/total)*100:>6.1f}%")
            print(f"     - Atama:        {(costs['assignment_costs']/total)*100:>6.1f}%")
            print(f"     - Taşıma:       {(costs['transportation_costs']/total)*100:>6.1f}%")
            print(f"     - Envanter:     {(costs['inventory_costs']/total)*100:>6.1f}%")
            print(f"     - Atık:         {(costs['waste_costs']/total)*100:>6.1f}%")

    print("\n📈 GENEL İSTATİSTİKLER:")
    print(f"   • Ortalama Toplam Maliyet:     {comprehensive_costs_df['total_cost'].mean():>12,.2f} TL")
    print(f"   • Minimum Toplam Maliyet:      {comprehensive_costs_df['total_cost'].min():>12,.2f} TL")
    print(f"   • Maksimum Toplam Maliyet:     {comprehensive_costs_df['total_cost'].max():>12,.2f} TL")
    print(f"   • Standart Sapma:              {comprehensive_costs_df['total_cost'].std():>12,.2f} TL")
    """
    print("\n" + "=" * 80)
    print("ANALİZ TAMAMLANDI - En iyi iterasyon sonuçları kaydedildi.")
    print("=" * 80)

    return best_iter

# 1. Hedef talep oluştur
target_demand = generate_target_demand(demand_df, z_value, variation_rate, multi_period=False)

# 2. Rota seçimi
selected_routes = select_routes_based_on_target(target_demand, route_to_depots)

# 3. Hub hedefleri hesapla
hub_targets_df = calculate_hub_targets_from_selected_routes(
    selected_routes, target_demand, route_to_depots, route_to_hub)
# Save hub_targets_df to CSV
hub_targets_df.to_csv("hub_targets_all_iterations.csv", index=False)

# 4. Tedarikçi atamaları
suppliers_assignments = assign_suppliers(
    supply_dict, beta_dict, gamma_dict, theta_dict, scenario_probs,
    vehicle_owners_df, hub_targets_df, F_set, K_set, B_set, S_set, T_set)

# 5. FIFO bazlı atık ve stok takibi
waste_df, remaining_inventory_df = fifo_inventory_and_waste(
    selected_routes, demand_dict, T_set, shelf_life=2)

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




"""
target_demand = generate_target_demand(demand_df, z_value, variation_rate, multi_period=False)

# CSV'ye kaydet (opsiyonel)
target_demand.to_csv("target_demand_all_iterations.csv", index=False)

selected_routes = select_routes_based_on_target(target_demand, route_to_depots)

# Save selected_routes to CSV
selected_routes.to_csv("selected_routes_all_iterations.csv", index=False)

hub_targets_df = calculate_hub_targets_from_selected_routes(selected_routes, target_demand, route_to_depots, route_to_hub)

# Save hub_targets_df to CSV
hub_targets_df.to_csv("hub_targets_all_iterations.csv", index=False)


suppliers_assignments = assign_suppliers(supply_dict, beta_dict, gamma_dict, theta_dict, scenario_probs, vehicle_owners_df,
                     hub_targets_df, F_set, K_set, B_set, S_set, T_set)

# Save suppliers_assignments to CSV
suppliers_assignments.to_csv("suppliers_assignments_all_iterations.csv", index=False)
print("Kaç iterasyonluk veri geldi:", suppliers_assignments['iteration'].nunique())



waste_df, remaining_inventory_df = fifo_inventory_and_waste(selected_routes, demand_dict, T_set, shelf_life=2)

# Save the results to CSV files
waste_df.to_csv("waste_df_all_iterations.csv", index=True)
remaining_inventory_df.to_csv("remaining_inventory_df_all_iterations.csv",index=True)

comprehensive_costs_df = calculate_comprehensive_costs(selected_routes, suppliers_assignments, waste_df, remaining_inventory_df, 
                                route_costs_dict, stock_cost_dict, waste_cost, scenario_probs)


# Save comprehensive_costs_df to CSV
comprehensive_costs_df.to_csv("comprehensive_costs_all_iterations.csv", index=False)

best_iteration_row = comprehensive_costs_df.loc[comprehensive_costs_df['total_cost'].idxmin()]
best_iteration = best_iteration_row['iteration']
print(f"En düşük maliyetli iterasyon: {best_iteration}")
print(f"Toplam maliyet: {best_iteration_row['total_cost']}")

best_routes = selected_routes[selected_routes['iteration'] == best_iteration]
best_assignments = suppliers_assignments[suppliers_assignments['iteration'] == best_iteration]
best_inventory = remaining_inventory_df[remaining_inventory_df['iteration'] == best_iteration]
best_waste = waste_df[waste_df['iteration'] == best_iteration]


best_routes.to_csv(f"best_routes_{best_iteration}.csv", index=False)
best_assignments.to_csv(f"best_assignments_{best_iteration}.csv", index=False)
best_inventory.to_csv(f"best_inventory_{best_iteration}.csv")
best_waste.to_csv(f"best_waste_{best_iteration}.csv")


def print_cost_summary(comprehensive_costs_df, selected_routes):
 
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

    # İterasyon bazlı detaylar
    print("\n📊 İTERASYON BAZLI MALİYET DAĞILIMI:")
    for iteration in total_costs.index:
        costs = total_costs.loc[iteration]
        print(f"\n🔍 {iteration.upper()}:")
        print(f"   • Rota Maliyetleri:        {costs['route_costs']:>12,.2f} TL")
        print(f"   • Tedarikçi Atama Maliyetleri: {costs['assignment_costs']:>8,.2f} TL")
        print(f"   • Taşıma Maliyetleri: {costs['transportation_costs']:>8,.2f} TL")
        print(f"   • Envanter Maliyetleri:    {costs['inventory_costs']:>12,.2f} TL")
        print(f"   • Atık Maliyetleri:        {costs['waste_costs']:>12,.2f} TL")
        print(f"   • TOPLAM MALİYET:          {costs['total_cost']:>12,.2f} TL")

        # Maliyet dağılımı yüzdesi
        total = costs['total_cost']
        if total > 0:
            print(f"   📈 Maliyet Dağılımı:")
            print(f"     - Rota:         {(costs['route_costs']/total)*100:>6.1f}%")
            print(f"     - Atama:        {(costs['assignment_costs']/total)*100:>6.1f}%")
            print(f"     - Taşıma:       {(costs['transportation_costs']/total)*100:>6.1f}%")
            print(f"     - Envanter:     {(costs['inventory_costs']/total)*100:>6.1f}%")
            print(f"     - Atık:         {(costs['waste_costs']/total)*100:>6.1f}%")

    # Genel istatistikler
    print("\n📈 GENEL İSTATİSTİKLER:")
    print(f"   • Ortalama Toplam Maliyet:     {comprehensive_costs_df['total_cost'].mean():>12,.2f} TL")
    print(f"   • Minimum Toplam Maliyet:      {comprehensive_costs_df['total_cost'].min():>12,.2f} TL")
    print(f"   • Maksimum Toplam Maliyet:     {comprehensive_costs_df['total_cost'].max():>12,.2f} TL")
    print(f"   • Standart Sapma:              {comprehensive_costs_df['total_cost'].std():>12,.2f} TL")

    print("\n" + "=" * 80)
    print("ANALİZ TAMAMLANDI - Detaylar CSV dosyalarında kaydedildi")
    print("=" * 80)

    return best_iter  # en düşük maliyetli iterasyonu dışarı aktar
"""