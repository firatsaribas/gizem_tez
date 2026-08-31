"""
PHASE I - CONSTRUCTIVE INITIAL SOLUTION
=======================================

Bu dosya yalnızca başlangıç çözümünü oluşturur.

Akış:

1. Excel'den problem verilerini oku.
2. Mesafe ve seyahat maliyetlerini hesapla.
3. Dönem dönem ziyaret programını oluştur.
4. Ziyaret programına göre cohort ve pickup miktarlarını hesapla.
5. Ziyaret edilen farmer'ları regret-2 insertion ile rotalara yerleştir.
6. Revenue, travel cost, service cost ve profit değerlerini yazdır.

Bu dosyada MILP çözümü, base-case sonucu veya CALNS kullanılmaz.
Phase II kapsamında route-internal 2-opt, inter-route relocation ve inter-route swap
hareketleri dönüşümlü olarak uygulanır. Üç hareket de iyileştirme üretmediğinde
local search sona erer.
"""

import math
import os
import openpyxl


EPS = 1e-9
PARAMS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "heuristic_parameters_input.xlsx",
)


# =============================================================================
# 1. EXCEL'DEN VERİLERİ OKU
# =============================================================================

def load_parameters(filepath):
    """Base-case kodundaki Excel yapısını kullanarak tüm verileri okur."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")

    wb = openpyxl.load_workbook(filepath, data_only=True)

    dimensions = {}
    for row in wb["Dimensions"].iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            dimensions[str(row[0]).strip()] = int(row[1])

    nF = dimensions["nF"]
    nK = dimensions["nK"]
    nT = dimensions["nT"]
    nS = dimensions["nS"]

    N = list(range(1, nF + 1))
    K = list(range(1, nK + 1))
    T = list(range(1, nT + 1))
    S = list(range(1, nS + 1))

    d = {}
    rho = {}
    for row in wb["Scenarios"].iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            s = int(row[0])
            d[s] = float(row[1])
            rho[s] = float(row[2])

    q = {}
    for row in wb["Vehicles"].iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            q[int(row[0])] = float(row[1])

    r_price = {}
    alpha = {}
    a0 = {}
    for row in wb["Farmers"].iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            i = int(row[0])
            r_price[i] = float(row[1])
            alpha[i] = float(row[2])
            a0[i] = float(row[3])

    g = {}
    for row in wb["Harvest"].iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            i = int(row[0])
            t = int(row[1])
            g[i, t] = float(row[2])

    coords = {}
    for row in wb["Coordinates"].iter_rows(min_row=2, values_only=True):
        if row[0] is not None:
            coords[int(row[0])] = (float(row[1]), float(row[2]))

    if abs(sum(rho.values()) - 1.0) > 1e-6:
        raise ValueError("Scenario probabilities must sum to 1.")
    if len(q) != nK:
        raise ValueError("Vehicle rows do not match nK.")
    if len(r_price) != nF:
        raise ValueError("Farmer rows do not match nF.")
    if len(g) != nF * nT:
        raise ValueError("Harvest sheet must contain nF x nT rows.")

    origin_of = {k: nF + k for k in K}
    hub = 0
    expected_nodes = {hub, *N, *origin_of.values()}
    if set(coords) != expected_nodes:
        raise ValueError("Coordinates must contain hub, all farmers, and all vehicle origins.")

    return {
        "N": N,
        "K": K,
        "T": T,
        "S": S,
        "d": d,
        "rho": rho,
        "q": q,
        "r_price": r_price,
        "alpha": alpha,
        "a0": a0,
        "g": g,
        "coords": coords,
        "origin_of": origin_of,
        "hub": hub,
    }


# =============================================================================
# 2. MESAFE VE SEYAHAT MALİYETLERİNİ HESAPLA
# =============================================================================

def calculate_travel_costs(data):
    """Base-case ile aynı mesafe ve maliyet hesabını kullanır."""

    cost_per_km = (12 / 100) * 1.29
    nodes = [data["hub"]] + data["N"] + list(data["origin_of"].values())
    c = {}

    for u in nodes:
        for v in nodes:
            if u == v:
                km = 0.0
            else:
                lat_u, lon_u = data["coords"][u]
                lat_v, lon_v = data["coords"][v]
                dlat = (lat_v - lat_u) * 111.0
                dlon = (lon_v - lon_u) * 111.0 * math.cos(math.radians(38.0))
                km = math.sqrt(dlat ** 2 + dlon ** 2)

            for k in data["K"]:
                c[u, v, k] = round(km * cost_per_km, 4)

    data["c"] = c


# =============================================================================
# 3. ZİYARETLERİ VE ROTALARI BİRLİKTE OLUŞTUR
# =============================================================================

def construct_solution(data):
    """
    Her dönemde ziyaret kararını ve rota yerleştirmesini birlikte yapar.

    Bir farmer için net katkı:
        ertelemeyle kaybedilecek gelir
        - gerçek rota ekleme maliyeti
        - gerekli araç atamalarının service cost'u

    Pozitif net katkısı en yüksek farmer seçilir ve doğrudan geçici rotaya
    eklenir. Daha sonra kalan farmer'lar yeniden değerlendirilir.
    """

    N, K, T, S = data["N"], data["K"], data["T"], data["S"]
    d, rho = data["d"], data["rho"]
    q = data["q"]
    r_price, alpha = data["r_price"], data["alpha"]
    a0, g, c = data["a0"], data["g"], data["c"]
    hub, origin_of = data["hub"], data["origin_of"]

    # En düşük deterioration oranı en yüksek pickup miktarını verir.
    capacity_scenario = min(S, key=lambda s: d[s])

    visits = {i: [] for i in N}
    last_visit = {i: 0 for i in N}
    routes = {t: {k: [] for k in K} for t in T}
    assigned_load = {}
    period_details = []
    errors = []

    # Bir rotanın seyahat maliyeti.
    def route_cost(k, route):
        if not route:
            return 0.0
        value = c[origin_of[k], route[0], k]
        for u, v in zip(route, route[1:]):
            value += c[u, v, k]
        value += c[route[-1], hub, k]
        return value

    # Önceki ziyaret ile yeni ziyaret arasındaki pickup ve revenue değerleri.
    quantity = {}
    revenue = {}

    for i in N:
        for previous_visit in [0] + T:
            for visit_period in T:
                if previous_visit >= visit_period:
                    continue

                expected_revenue = 0.0

                for s in S:
                    scenario_quantity = 0.0
                    scenario_revenue = 0.0

                    # Initial inventory cohort, tau = 0
                    if previous_visit == 0:
                        available = a0[i] * (1 - d[s]) ** (visit_period - 1)
                        unit_price = r_price[i] * (1 - d[s]) ** visit_period
                        scenario_quantity += available
                        scenario_revenue += available * unit_price

                    # Harvest cohorts, tau >= 1
                    for tau in T:
                        if previous_visit < tau <= visit_period:
                            available = g[i, tau] * (1 - d[s]) ** (visit_period - tau)
                            unit_price = r_price[i] * (1 - d[s]) ** (visit_period - tau)
                            scenario_quantity += available
                            scenario_revenue += available * unit_price

                    quantity[i, previous_visit, visit_period, s] = scenario_quantity
                    expected_revenue += rho[s] * scenario_revenue

                revenue[i, previous_visit, visit_period] = expected_revenue

    for t in T:
        vehicle_load = {k: 0.0 for k in K}
        remaining_farmers = set(N)
        selected_this_period = []
        evaluated_candidates = []

        while remaining_farmers:
            best_candidate = None

            for i in sorted(remaining_farmers):
                previous_visit = last_visit[i]

                # Şimdi ziyaret etmenin, bir dönem ertelemeye göre gelir avantajı.
                if t < T[-1]:
                    visit_now = revenue[i, previous_visit, t]
                    postpone = revenue[i, previous_visit, t + 1]
                    next_period_new_harvest = revenue[i, t, t + 1]
                    postponement_loss = visit_now + next_period_new_harvest - postpone
                else:
                    # Horizon sonrasında toplanmayan ürün objective'e katkı yapmaz.
                    postponement_loss = revenue[i, previous_visit, t]

                pickup = quantity[i, previous_visit, t, capacity_scenario]
                if pickup <= EPS:
                    continue

                # Önce farmer'ın tek bir araca tam olarak yerleşebileceği seçenekler.
                insertion_options = []

                for k in K:
                    if vehicle_load[k] + pickup > q[k] + EPS:
                        continue

                    old_cost = route_cost(k, routes[t][k])
                    for position in range(len(routes[t][k]) + 1):
                        trial_route = routes[t][k].copy()
                        trial_route.insert(position, i)
                        extra_travel = route_cost(k, trial_route) - old_cost
                        total_extra_cost = extra_travel + alpha[i]
                        insertion_options.append({
                            "vehicles": [(k, position, pickup)],
                            "extra_travel": extra_travel,
                            "extra_service": alpha[i],
                            "total_extra_cost": total_extra_cost,
                        })

                # Tek araca sığmıyorsa, toplam kalan kapasite yeterliyse split denenir.
                if not insertion_options:
                    residual_capacity = {
                        k: max(0.0, q[k] - vehicle_load[k]) for k in K
                    }

                    if sum(residual_capacity.values()) + EPS >= pickup:
                        split_parts = []
                        remaining_pickup = pickup
                        extra_travel = 0.0
                        used_vehicles = 0

                        split_choices = []
                        for k in K:
                            if residual_capacity[k] <= EPS:
                                continue

                            old_cost = route_cost(k, routes[t][k])
                            best_extra = float("inf")
                            best_position = 0

                            for position in range(len(routes[t][k]) + 1):
                                trial_route = routes[t][k].copy()
                                trial_route.insert(position, i)
                                increase = route_cost(k, trial_route) - old_cost
                                if increase < best_extra:
                                    best_extra = increase
                                    best_position = position

                            split_choices.append(
                                (best_extra, k, best_position, residual_capacity[k])
                            )

                        split_choices.sort()

                        for increase, k, position, residual in split_choices:
                            if remaining_pickup <= EPS:
                                break
                            allocation = min(remaining_pickup, residual)
                            split_parts.append((k, position, allocation))
                            extra_travel += increase
                            used_vehicles += 1
                            remaining_pickup -= allocation

                        if remaining_pickup <= EPS:
                            extra_service = used_vehicles * alpha[i]
                            insertion_options.append({
                                "vehicles": split_parts,
                                "extra_travel": extra_travel,
                                "extra_service": extra_service,
                                "total_extra_cost": extra_travel + extra_service,
                            })

                if not insertion_options:
                    continue

                insertion_options.sort(key=lambda option: option["total_extra_cost"])
                best_insertion = insertion_options[0]

                if len(insertion_options) >= 2:
                    regret = (
                        insertion_options[1]["total_extra_cost"]
                        - best_insertion["total_extra_cost"]
                    )
                else:
                    regret = float("inf")

                marginal_gain = postponement_loss - best_insertion["total_extra_cost"]

                evaluated_candidates.append((i, marginal_gain))

                if marginal_gain > EPS:
                    candidate = {
                        "farmer": i,
                        "gain": marginal_gain,
                        "loss": postponement_loss,
                        "pickup": pickup,
                        "regret": regret,
                        "insertion": best_insertion,
                    }

                    # Ziyaret için gain pozitif olmalıdır. Pozitif adaylar arasından
                    # regret-2 değeri en yüksek farmer önce yerleştirilir.
                    if best_candidate is None or (
                        candidate["regret"],
                        candidate["gain"],
                        candidate["loss"],
                        candidate["pickup"],
                        -candidate["farmer"],
                    ) > (
                        best_candidate["regret"],
                        best_candidate["gain"],
                        best_candidate["loss"],
                        best_candidate["pickup"],
                        -best_candidate["farmer"],
                    ):
                        best_candidate = candidate

            # Pozitif katkılı ve uygulanabilir farmer kalmadıysa dönem tamamlanır.
            if best_candidate is None:
                break

            i = best_candidate["farmer"]
            insertion = best_candidate["insertion"]

            # Split durumunda aynı farmer farklı araç rotalarına eklenebilir.
            # Pozisyonlar rota değiştikçe kaymaması için araç bazında uygulanır.
            for k, position, allocation in insertion["vehicles"]:
                routes[t][k].insert(position, i)
                vehicle_load[k] += allocation
                assigned_load[i, k, t] = allocation

            visits[i].append(t)
            last_visit[i] = t
            selected_this_period.append({
                "farmer": i,
                "gain": best_candidate["gain"],
                "pickup": best_candidate["pickup"],
            })
            remaining_farmers.remove(i)

            # Aynı dönemde farmer yalnızca bir kez ziyaret kararı alabilir.

        period_details.append({
            "period": t,
            "evaluated": sorted({i for i, _ in evaluated_candidates}),
            "selected": [row["farmer"] for row in selected_this_period],
            "pickup": sum(row["pickup"] for row in selected_this_period),
            "capacity": sum(q.values()),
            "gains": {row["farmer"]: row["gain"] for row in selected_this_period},
        })

    # Gerçek seyahat ve service maliyetleri, oluşturulan rotalardan hesaplanır.
    travel_cost = sum(route_cost(k, routes[t][k]) for t in T for k in K)
    service_cost = sum(
        alpha[i]
        for (i, k, t), load in assigned_load.items()
        if load > EPS
    )

    return (
        visits,
        routes,
        assigned_load,
        period_details,
        capacity_scenario,
        travel_cost,
        service_cost,
        errors,
    )


# =============================================================================
# 4. COHORT, PICKUP VE EXPECTED REVENUE HESABI
# =============================================================================

def simulate_cohorts(data, visits):
    """Seçilen ziyaret programı altında matematiksel modeldeki cohort akışını uygular."""

    N, T, S = data["N"], data["T"], data["S"]
    d, rho = data["d"], data["rho"]
    r_price, a0, g = data["r_price"], data["a0"], data["g"]

    Q = {}
    expected_revenue = 0.0

    for i in N:
        for s in S:
            previous_inventory = {tau: 0.0 for tau in [0] + T}

            for t in T:
                collected_in_period = 0.0

                for tau in [0] + T:
                    if tau > t:
                        continue

                    if tau == 0:
                        available = a0[i] if t == 1 else (1 - d[s]) * previous_inventory[0]
                    elif tau == t:
                        available = g[i, t]
                    else:
                        available = (1 - d[s]) * previous_inventory[tau]

                    if t in visits[i]:
                        picked = available
                        remaining = 0.0
                    else:
                        picked = 0.0
                        remaining = available

                    previous_inventory[tau] = remaining
                    collected_in_period += picked

                    unit_price = r_price[i] * (1 - d[s]) ** (t - tau)
                    expected_revenue += rho[s] * unit_price * picked

                Q[i, t, s] = collected_in_period

    return Q, expected_revenue


# =============================================================================
# 5. TÜM SENARYOLARDA KAPASİTEYİ DOĞRULA
# =============================================================================

def verify_capacity(data, Q, assigned_load, capacity_scenario):
    """Dominant senaryoda yapılan yük dağılımını tüm senaryolarda kontrol eder."""

    errors = []

    for t in data["T"]:
        for k in data["K"]:
            for s in data["S"]:
                scenario_load = 0.0

                for i in data["N"]:
                    dominant_alloc = assigned_load.get((i, k, t), 0.0)
                    dominant_total = Q[i, t, capacity_scenario]

                    if dominant_alloc > EPS and dominant_total > EPS:
                        scenario_load += (
                            dominant_alloc * Q[i, t, s] / dominant_total
                        )

                if scenario_load > data["q"][k] + 1e-6:
                    errors.append(
                        f"Period {t}, vehicle {k}, scenario {s}: capacity exceeded."
                    )

    return errors


# =============================================================================
# 6. PHASE II - ROUTE-INTERNAL FIRST-IMPROVEMENT 2-OPT
# =============================================================================

def apply_two_opt(data, routes):
    """
    Aynı araç rotası içindeki farmer sırasını iyileştirir.

    Ziyaret dönemleri, araç atamaları, pickup miktarları ve service cost
    değişmez. Yalnızca rota sırası ve travel cost değişebilir.
    """

    c = data["c"]
    origin_of = data["origin_of"]
    hub = data["hub"]

    def route_cost(k, route):
        if not route:
            return 0.0
        value = c[origin_of[k], route[0], k]
        for u, v in zip(route, route[1:]):
            value += c[u, v, k]
        value += c[route[-1], hub, k]
        return value

    improved_routes = {t: {k: routes[t][k].copy() for k in data["K"]} for t in data["T"]}
    moves = []

    improved = True
    while improved:
        improved = False

        for t in data["T"]:
            for k in data["K"]:
                route = improved_routes[t][k]

                # En az üç farmer yoksa anlamlı bir 2-opt hareketi oluşmaz.
                if len(route) < 3:
                    continue

                old_cost = route_cost(k, route)

                # First-improvement: ilk iyileştirici ters çevirme hemen kabul edilir.
                for start in range(len(route) - 1):
                    for end in range(start + 1, len(route)):
                        trial_route = (
                            route[:start]
                            + list(reversed(route[start:end + 1]))
                            + route[end + 1:]
                        )
                        new_cost = route_cost(k, trial_route)

                        if new_cost < old_cost - EPS:
                            improved_routes[t][k] = trial_route
                            moves.append({
                                "period": t,
                                "vehicle": k,
                                "old_route": route.copy(),
                                "new_route": trial_route.copy(),
                                "saving": old_cost - new_cost,
                            })
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    old_travel = sum(
        route_cost(k, routes[t][k])
        for t in data["T"]
        for k in data["K"]
    )
    new_travel = sum(
        route_cost(k, improved_routes[t][k])
        for t in data["T"]
        for k in data["K"]
    )

    return improved_routes, old_travel, new_travel, moves




# =============================================================================
# 7. PHASE II - INTER-ROUTE FIRST-IMPROVEMENT RELOCATION
# =============================================================================

def apply_relocation(data, routes, assigned_load):
    """
    Tek bir araca atanmış farmer'ı aynı dönemde diğer aracın rotasına taşır.

    Ziyaret dönemi ve pickup miktarı değişmez. Hedef araçta yeterli kapasite
    varsa ve toplam travel cost azalırsa ilk iyileştirici hareket kabul edilir.
    Split pickup kullanılan farmer'lar bu ilk sürümde değerlendirilmez.
    """

    c = data["c"]
    origin_of = data["origin_of"]
    hub = data["hub"]
    q = data["q"]

    def route_cost(k, route):
        if not route:
            return 0.0
        value = c[origin_of[k], route[0], k]
        for u, v in zip(route, route[1:]):
            value += c[u, v, k]
        value += c[route[-1], hub, k]
        return value

    improved_routes = {
        t: {k: routes[t][k].copy() for k in data["K"]}
        for t in data["T"]
    }
    improved_loads = assigned_load.copy()
    moves = []

    improved = True
    while improved:
        improved = False

        for t in data["T"]:
            for source_k in data["K"]:
                source_route = improved_routes[t][source_k]

                for source_pos, farmer in enumerate(source_route):
                    # Farmer bu dönemde birden fazla araca atanmışsa split vardır.
                    active_vehicles = [
                        k for k in data["K"]
                        if improved_loads.get((farmer, k, t), 0.0) > EPS
                    ]
                    if len(active_vehicles) != 1:
                        continue

                    pickup = improved_loads.get((farmer, source_k, t), 0.0)
                    if pickup <= EPS:
                        continue

                    for target_k in data["K"]:
                        if target_k == source_k:
                            continue

                        target_load = sum(
                            improved_loads.get((i, target_k, t), 0.0)
                            for i in data["N"]
                        )
                        if target_load + pickup > q[target_k] + EPS:
                            continue

                        old_cost = (
                            route_cost(source_k, source_route)
                            + route_cost(target_k, improved_routes[t][target_k])
                        )

                        new_source_route = source_route.copy()
                        new_source_route.pop(source_pos)

                        for target_pos in range(len(improved_routes[t][target_k]) + 1):
                            new_target_route = improved_routes[t][target_k].copy()
                            new_target_route.insert(target_pos, farmer)

                            new_cost = (
                                route_cost(source_k, new_source_route)
                                + route_cost(target_k, new_target_route)
                            )

                            if new_cost < old_cost - EPS:
                                improved_routes[t][source_k] = new_source_route
                                improved_routes[t][target_k] = new_target_route
                                improved_loads[farmer, source_k, t] = 0.0
                                improved_loads[farmer, target_k, t] = pickup
                                moves.append({
                                    "period": t,
                                    "farmer": farmer,
                                    "source_vehicle": source_k,
                                    "target_vehicle": target_k,
                                    "old_source_route": source_route.copy(),
                                    "new_source_route": new_source_route.copy(),
                                    "old_target_route": improved_routes[t][target_k].copy() if False else None,
                                    "new_target_route": new_target_route.copy(),
                                    "saving": old_cost - new_cost,
                                })
                                improved = True
                                break

                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    old_travel = sum(
        route_cost(k, routes[t][k])
        for t in data["T"]
        for k in data["K"]
    )
    new_travel = sum(
        route_cost(k, improved_routes[t][k])
        for t in data["T"]
        for k in data["K"]
    )

    return improved_routes, improved_loads, old_travel, new_travel, moves



# =============================================================================
# 8. PHASE II - INTER-ROUTE FIRST-IMPROVEMENT SWAP
# =============================================================================

def apply_swap(data, routes, assigned_load):
    """
    Aynı dönemde farklı araçlarda bulunan iki farmer'ın araçlarını değiştirir.

    Her iki farmer da yalnızca bir araca atanmış olmalıdır. Swap sonrasında
    iki aracın kapasitesi sağlanıyor ve toplam travel cost azalıyorsa ilk
    iyileştirici hareket kabul edilir.
    """

    c = data["c"]
    origin_of = data["origin_of"]
    hub = data["hub"]
    q = data["q"]

    def route_cost(k, route):
        if not route:
            return 0.0
        value = c[origin_of[k], route[0], k]
        for u, v in zip(route, route[1:]):
            value += c[u, v, k]
        value += c[route[-1], hub, k]
        return value

    improved_routes = {
        t: {k: routes[t][k].copy() for k in data["K"]}
        for t in data["T"]
    }
    improved_loads = assigned_load.copy()
    moves = []

    improved = True
    while improved:
        improved = False

        for t in data["T"]:
            for index_a, vehicle_a in enumerate(data["K"]):
                for vehicle_b in data["K"][index_a + 1:]:
                    route_a = improved_routes[t][vehicle_a]
                    route_b = improved_routes[t][vehicle_b]

                    if not route_a or not route_b:
                        continue

                    load_a = sum(
                        improved_loads.get((i, vehicle_a, t), 0.0)
                        for i in data["N"]
                    )
                    load_b = sum(
                        improved_loads.get((i, vehicle_b, t), 0.0)
                        for i in data["N"]
                    )

                    for pos_a, farmer_a in enumerate(route_a):
                        active_a = [
                            k for k in data["K"]
                            if improved_loads.get((farmer_a, k, t), 0.0) > EPS
                        ]
                        if len(active_a) != 1:
                            continue

                        pickup_a = improved_loads.get((farmer_a, vehicle_a, t), 0.0)
                        if pickup_a <= EPS:
                            continue

                        for pos_b, farmer_b in enumerate(route_b):
                            active_b = [
                                k for k in data["K"]
                                if improved_loads.get((farmer_b, k, t), 0.0) > EPS
                            ]
                            if len(active_b) != 1:
                                continue

                            pickup_b = improved_loads.get((farmer_b, vehicle_b, t), 0.0)
                            if pickup_b <= EPS:
                                continue

                            new_load_a = load_a - pickup_a + pickup_b
                            new_load_b = load_b - pickup_b + pickup_a

                            if new_load_a > q[vehicle_a] + EPS:
                                continue
                            if new_load_b > q[vehicle_b] + EPS:
                                continue

                            old_cost = route_cost(vehicle_a, route_a) + route_cost(vehicle_b, route_b)

                            reduced_a = route_a.copy()
                            reduced_a.pop(pos_a)
                            reduced_b = route_b.copy()
                            reduced_b.pop(pos_b)

                            for insert_a in range(len(reduced_a) + 1):
                                trial_a = reduced_a.copy()
                                trial_a.insert(insert_a, farmer_b)

                                for insert_b in range(len(reduced_b) + 1):
                                    trial_b = reduced_b.copy()
                                    trial_b.insert(insert_b, farmer_a)

                                    new_cost = (
                                        route_cost(vehicle_a, trial_a)
                                        + route_cost(vehicle_b, trial_b)
                                    )

                                    if new_cost < old_cost - EPS:
                                        improved_routes[t][vehicle_a] = trial_a
                                        improved_routes[t][vehicle_b] = trial_b

                                        improved_loads[farmer_a, vehicle_a, t] = 0.0
                                        improved_loads[farmer_a, vehicle_b, t] = pickup_a
                                        improved_loads[farmer_b, vehicle_b, t] = 0.0
                                        improved_loads[farmer_b, vehicle_a, t] = pickup_b

                                        moves.append({
                                            "period": t,
                                            "farmer_a": farmer_a,
                                            "vehicle_a": vehicle_a,
                                            "farmer_b": farmer_b,
                                            "vehicle_b": vehicle_b,
                                            "old_route_a": route_a.copy(),
                                            "new_route_a": trial_a.copy(),
                                            "old_route_b": route_b.copy(),
                                            "new_route_b": trial_b.copy(),
                                            "saving": old_cost - new_cost,
                                        })
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    old_travel = sum(
        route_cost(k, routes[t][k])
        for t in data["T"]
        for k in data["K"]
    )
    new_travel = sum(
        route_cost(k, improved_routes[t][k])
        for t in data["T"]
        for k in data["K"]
    )

    return improved_routes, improved_loads, old_travel, new_travel, moves



# =============================================================================
# 9. PHASE II - TÜM LOCAL SEARCH HAREKETLERİNİ BİRLİKTE UYGULA
# =============================================================================

def run_local_search(data, routes, assigned_load):
    """
    2-opt, relocation ve swap hareketlerini sırayla uygular.

    Bir hareket rota yapısını değiştirdiğinde diğer hareketler yeniden iyileştirme
    bulabilir. Bu nedenle üç operatör de hiç iyileştirme üretmeyene kadar aynı sıra
    tekrar edilir.
    """

    current_routes = {
        t: {k: routes[t][k].copy() for k in data["K"]}
        for t in data["T"]
    }
    current_loads = assigned_load.copy()

    all_two_opt_moves = []
    all_relocation_moves = []
    all_swap_moves = []
    cycle_results = []

    initial_travel = None
    after_first_two_opt = None
    after_first_relocation = None
    cycle = 0

    while True:
        cycle += 1
        cycle_start_routes = {
            t: {k: current_routes[t][k].copy() for k in data["K"]}
            for t in data["T"]
        }

        (
            current_routes,
            start_travel,
            two_opt_travel,
            two_opt_moves,
        ) = apply_two_opt(data, current_routes)

        if initial_travel is None:
            initial_travel = start_travel
            after_first_two_opt = two_opt_travel

        (
            current_routes,
            current_loads,
            _,
            relocation_travel,
            relocation_moves,
        ) = apply_relocation(data, current_routes, current_loads)

        if after_first_relocation is None:
            after_first_relocation = relocation_travel

        (
            current_routes,
            current_loads,
            _,
            swap_travel,
            swap_moves,
        ) = apply_swap(data, current_routes, current_loads)

        all_two_opt_moves.extend(two_opt_moves)
        all_relocation_moves.extend(relocation_moves)
        all_swap_moves.extend(swap_moves)

        cycle_results.append({
            "cycle": cycle,
            "start_travel": start_travel,
            "after_two_opt": two_opt_travel,
            "after_relocation": relocation_travel,
            "after_swap": swap_travel,
        })

        # Bu turda hiçbir operatör iyileştirme yapmadıysa local search tamamlanır.
        if not two_opt_moves and not relocation_moves and not swap_moves:
            break

        # Güvenlik kontrolü: rota yapısı değişmediyse gereksiz döngüyü durdur.
        if current_routes == cycle_start_routes:
            break

    final_travel = cycle_results[-1]["after_swap"]

    return (
        current_routes,
        current_loads,
        initial_travel,
        final_travel,
        all_two_opt_moves,
        all_relocation_moves,
        all_swap_moves,
        cycle_results,
        after_first_two_opt,
        after_first_relocation,
    )


# =============================================================================
# 10. SONUÇLARI YAZDIR
# =============================================================================

def print_results(data, visits, period_details, routes, assigned_load,
                  capacity_scenario, revenue, travel, service, errors,
                  phase1_travel=None, two_opt_travel=None,
                  two_opt_moves=None, relocation_moves=None, swap_moves=None,
                  relocation_travel=None):
    """Constructive çözümün temel sonuçlarını anlaşılır biçimde gösterir."""

    profit = revenue - travel - service

    print("\n" + "=" * 72)
    print("PHASE I - CONSTRUCTIVE INITIAL SOLUTION")
    print("=" * 72)
    print(f"Capacity screening scenario : {capacity_scenario}")
    print(f"Expected revenue            : {revenue:.4f} EUR")
    if phase1_travel is None:
        print(f"Travel cost                 : {travel:.4f} EUR")
    else:
        print(f"Phase I travel cost         : {phase1_travel:.4f} EUR")
        print(f"Phase II travel cost        : {travel:.4f} EUR")
        print(f"Total travel saving         : {phase1_travel - travel:.4f} EUR")
    print(f"Service cost                : {service:.4f} EUR")
    print(f"Expected profit             : {profit:.4f} EUR")
    print(f"Status                      : {'FEASIBLE' if not errors else 'CHECK REQUIRED'}")

    print("\nVISIT SCHEDULE")
    for i in data["N"]:
        print(f"  Farmer {i}: {visits[i]}")

    print("\nPERIOD DECISIONS")
    for row in period_details:
        print(
            f"  Period {row['period']}: evaluated={row['evaluated']}, "
            f"selected={row['selected']}, "
            f"pickup={row['pickup']:.2f}/{row['capacity']:.2f} kg"
        )

    print("\nROUTES")
    for t in data["T"]:
        for k in data["K"]:
            if routes[t][k]:
                full_route = [data["origin_of"][k]] + routes[t][k] + [data["hub"]]
                load = sum(
                    assigned_load.get((i, k, t), 0.0)
                    for i in data["N"]
                )
                print(f"  Period {t}, Vehicle {k}: {full_route}, load={load:.2f} kg")

    if two_opt_moves is not None:
        print("\n2-OPT MOVES")
        if not two_opt_moves:
            print("  No improving 2-opt move was found.")
        else:
            for move in two_opt_moves:
                print(
                    f"  Period {move['period']}, Vehicle {move['vehicle']}: "
                    f"{move['old_route']} -> {move['new_route']}, "
                    f"saving={move['saving']:.4f} EUR"
                )

    if relocation_moves is not None:
        print("\nRELOCATION MOVES")
        if not relocation_moves:
            print("  No improving relocation move was found.")
        else:
            for move in relocation_moves:
                print(
                    f"  Period {move['period']}, Farmer {move['farmer']}: "
                    f"Vehicle {move['source_vehicle']} -> Vehicle {move['target_vehicle']}, "
                    f"saving={move['saving']:.4f} EUR"
                )

    if swap_moves is not None:
        print("\nSWAP MOVES")
        if not swap_moves:
            print("  No improving swap move was found.")
        else:
            for move in swap_moves:
                print(
                    f"  Period {move['period']}: "
                    f"Farmer {move['farmer_a']} (Vehicle {move['vehicle_a']}) <-> "
                    f"Farmer {move['farmer_b']} (Vehicle {move['vehicle_b']}), "
                    f"saving={move['saving']:.4f} EUR"
                )

    if errors:
        print("\nWARNINGS")
        for error in errors:
            print(" -", error)


# =============================================================================
# 11. ANA PROGRAM
# =============================================================================

if __name__ == "__main__":

    print(f"Reading parameters from: {PARAMS_FILE}")

    data = load_parameters(PARAMS_FILE)
    calculate_travel_costs(data)

    (
        visits,
        routes,
        assigned_load,
        period_details,
        capacity_scenario,
        travel,
        service,
        errors,
    ) = construct_solution(data)

    Q, revenue = simulate_cohorts(data, visits)
    errors.extend(verify_capacity(data, Q, assigned_load, capacity_scenario))

    (
        local_routes,
        local_assigned_load,
        phase1_travel,
        local_travel,
        two_opt_moves,
        relocation_moves,
        swap_moves,
        cycle_results,
        two_opt_travel,
        relocation_travel,
    ) = run_local_search(data, routes, assigned_load)

    errors.extend(verify_capacity(
        data, Q, local_assigned_load, capacity_scenario
    ))

    print_results(
        data,
        visits,
        period_details,
        local_routes,
        local_assigned_load,
        capacity_scenario,
        revenue,
        local_travel,
        service,
        errors,
        phase1_travel=phase1_travel,
        two_opt_travel=two_opt_travel,
        two_opt_moves=two_opt_moves,
        relocation_moves=relocation_moves,
        swap_moves=swap_moves,
        relocation_travel=relocation_travel,
    )

    print("\nLOCAL SEARCH CYCLES")
    for row in cycle_results:
        print(
            f"  Cycle {row['cycle']}: "
            f"start={row['start_travel']:.4f}, "
            f"after 2-opt={row['after_two_opt']:.4f}, "
            f"after relocation={row['after_relocation']:.4f}, "
            f"after swap={row['after_swap']:.4f} EUR"
        )
