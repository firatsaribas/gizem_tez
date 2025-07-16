# -*- coding: utf-8 -*-
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
scenario_probs={"S1": 0.3, "S2": 0.5, "S3": 0.2}
shelf_life=2

#dictionaries
# mean değerlerini {(d, t): mean} sözlüğüne çevir
mean_dict = {(row["d"], row["t"]): row["demand"] for _, row in demand_df.iterrows()}
sigma_dict = {key: val * variation_rate for key, val in mean_dict.items()}
    
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



def select_routes_based_on_target(target_demand, route_to_depots, iteration_name):
    """
    Hedef talepleri karşılamak için en uygun rotaları seçer.
    Greedy yaklaşımla çalışır. Senaryolardan bağımsızdır.
    """
    selected_routes = []  # Seçilen rotaların dağıtım planını tutacak liste
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
    hub_targets = defaultdict(float)  # (b, t) → toplam ihtiyaç

    first_column = selected_routes.values[:, 0]        # get first column (e.g., 22, 30, 24...)
    unique_values = np.unique(first_column)            # get unique values
    unique_list = unique_values.tolist()    

    # TODO 
    # t for loopu olacak şekilde dışarıda olacak refactoring yapmamız lazım 
    # r loopu t den sonra olacak 
    # 
    # TODO 
    for t in sorted(target_demand_df['t'].unique()):
        selected_routes_for_t=selected_routes[selected_routes['t']==t]
        for r in selected_routes_for_t['r'].unique():#seçilen her bir rota için döngü başlat
            depots = route_to_depots.get(r, []) #bu rota hangi depoları kapsıyor
            b = route_to_hub[r] #rota hangi hubtan başlıyor

        #aşağıdaki isin methodu Depo d kolonundaki değerlerin depots listesinde olup olmadığını kontrol eder.Örneğin: depots = [25, 24, 14, 13] ise, d kolonu içinde bu 4 depoyu filtreler.
            total = target_demand_df[
                (target_demand_df["d"].isin(depots)) & (target_demand_df["t"] == t)
            ]["target"].sum()
            hub_targets[(b, t)] += total #oplam talebi hub b için, zaman t’de gönderilmesi gereken miktar olarak hub_targets’a ekler. Böylece aynı (b,t) için farklı rotalarla gelen değerler toplanır.

    # İsteğe bağlı: DataFrame formatında da dönebiliriz
        hub_targets_df = pd.DataFrame([
            {"b": b, "t": t, "target_amount": amt}
            for (b, t), amt in hub_targets.items()
        ])
    return hub_targets_df

def assign_suppliers(supply_dict, beta_dict, gamma_dict, theta_dict, prob_dict, vehicle_owners_df,
                     hub_targets, F_set, K_set, B_set, S_set, T_set):
    """
    Tedarikçilerin araçlara ve aktarma merkezlerine atanmasını yapar.
    Her senaryoda ve her zaman periyodunda,
    tedarikçi sadece kendi aracını veya uygun boş araçları kullanarak,
    belirlenen hub_targets'a göre ürün çeker.

    """
    assignments = []  # Tüm atamaları tutacak liste

    for s in S_set:
        for t in T_set:
            # Her senaryo-zaman için hedefleri kopyala (değiştirmemek için)
            remaining_targets = hub_targets.copy()

            # Araç kapasitelerini başlat
            vehicle_cap = {k: theta_dict.get(k, 0) for k in K_set}

            for f in F_set:
                supply = supply_dict.get((f, s, t), 0)
                if supply <= 0:
                    continue  # Tedarikçinin arzı yoksa geç
                
                
                # Aracı varsa sadece kendi aracını kullanabilir
                owned_vehicles = vehicle_owners_df[vehicle_owners_df["f"] == f]["k"].tolist()
                if owned_vehicles:
                    candidate_vehicles = owned_vehicles
                else:
                    candidate_vehicles = K_set  # Aracı yoksa tüm araçlar aday
                
                best_cost = float("inf")
                best_choice = None
                ##TO DO
                # Burda vehicle hub eşleştirmesi yaparken total durumdaki tüm eşleştirmelerin total costuna değil 
                # kalan araçlar ve o aracı özelinde arasından en düşük costa bakıyor. Huba araç-üretici 
                # kombinasyonları atamak daha mantıklı gibi
                ##TO DO

                for k in candidate_vehicles:# Her bir aday araç ve hub kombinasyonunu dene
                    for b in B_set: 
                        if vehicle_cap[k] <= 0:
                            continue  # Kapasitesi kalmamış araçları geç
                        if remaining_targets.get((b, t), 0) <= 0:
                            continue  # Hub için ihtiyaç yoksa geç   


                        gamma_cost = gamma_dict.get((f, b, k), 1e6)
                        beta_cost = beta_dict.get((f, k), 1e6)
                        total_cost = scenario_probs[s]  * (beta_cost + gamma_cost)

                        if total_cost < best_cost and remaining_targets.get((b, t), 0) > 0:
                            best_cost = total_cost
                            best_choice = (k, b)

                # En iyi eşleşmeye göre atama yap
                if best_choice:
                    k, b = best_choice
                    assign_qty = min(supply, vehicle_cap[k], remaining_targets.get((b, t), 0))

                    assignments.append({
                        "s": s, "t": t, "f": f, "k": k, "b": b, "amount": assign_qty
                    })

                    # Güncellemeler
                    vehicle_cap[k] -= assign_qty #Bu satır, araca atanan miktar kadar kapasitesini azaltır.
                    remaining_targets[(b, t)] -= assign_qty #Bu satır, ilgili aktarma merkezi (hub) ve zaman dilimi için kalan talebi azaltır.
                    supply -= assign_qty #Bu da tedarikçinin elindeki mevcut arzı azaltır.

                    if supply <= 0:
                        continue  # Tedarikçinin arzı bittiyse çık

    return pd.DataFrame(assignments)  # Atamaları DataFrame olarak döndür



def fifo_inventory_and_waste(selected_routes, target_demand_df, time_periods, shelf_life=2):
    """
    FIFO mantığı ile her depo ve zaman periyodu için envanter ve atık takibi yapar.
    
    Parametreler:
    - depot_delivery: DataFrame (index=depots, columns=time_periods), her zaman ve depo için gelen ürün miktarı
    - target_demand_df: DataFrame (index=depots, columns=time_periods), hedef talep değerleri
    - time_periods: list, örn. [1, 2, 3]
    - shelf_life: int, raf ömrü (örneğin 2 → 2 dönem sonra ürün bozulur)

    Dönüş:
    - waste_df: DataFrame, her zaman ve depo için oluşan atık miktarı
    - remaining_inventory_df: DataFrame, her zaman ve depo için dönem sonunda kalan toplam envanter
    """
    depot_delivery = selected_routes.groupby(["d", "t"])["amount"].sum().unstack(fill_value=0)
    
    # Her depo için yaş bazlı envanteri tutan dict (örn. {d1: {0: 10, 1: 5}})
    inventory = {d: {age: 0 for age in range(shelf_life)} for d in depot_delivery.index}

    # Çıktılar: atık ve kalan envanter tabloları
    waste_df = pd.DataFrame(0, index=depot_delivery.index, columns=time_periods)
    remaining_inventory_df = pd.DataFrame(0, index=depot_delivery.index, columns=time_periods)

    # Her zaman periyodu için işlem yap
    for t in time_periods:
        for d in depot_delivery.index:
            # Yeni gelen ürünleri 0 yaşındaki stoğa ekle
            delivered = depot_delivery.loc[d, t]
            inventory[d][0] += delivered

            # Talep değeri
            demand = target_demand_df.loc[d, t]

            # FIFO: en eski üründen başlayarak talebi karşıla
            for age in sorted(inventory[d].keys()):
                if demand <= 0:
                    break
                usable = min(demand, inventory[d][age])
                inventory[d][age] -= usable
                demand -= usable

            # Ürün yaşlandırma ve shelf life kontrolü
            updated_inventory = {age: 0 for age in range(shelf_life)}
            for age in range(shelf_life):
                if age + 1 < shelf_life:
                    updated_inventory[age + 1] = inventory[d][age]
                else:
                    # Raf ömrünü aşan ürünler atık olur
                    waste_df.loc[d, t] += inventory[d][age]
            inventory[d] = updated_inventory

            # Dönem sonu kalan stok toplamı
            remaining_inventory_df.loc[d, t] = sum(inventory[d].values())

    return waste_df, remaining_inventory_df


target_demand = generate_target_demand(demand_df, z_value, variation_rate, multi_period=False)

# CSV'ye kaydet (opsiyonel)
target_demand.to_csv("target_demand_all_iterations.csv", index=False)

selected_routes = select_routes_based_on_target(target_demand, route_to_depots, "iteration_1")

calculate_hub_targets_from_selected_routes(selected_routes, target_demand, route_to_depots, route_to_hub)




