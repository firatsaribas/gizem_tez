import random
import pandas as pd

# =========================
# 0) INPUT: mevcut 30 rota
# =========================
'''
route_to_am_30 = {
    1:2, 2:1, 3:1, 4:2, 5:3, 6:3, 7:2, 8:3, 9:2, 10:1,
    11:2, 12:3, 13:2, 14:1, 15:3, 16:2, 17:2, 18:2, 19:1, 20:3,
    21:2, 22:3, 23:3, 24:2, 25:1, 26:1, 27:3, 28:1, 29:1, 30:1
}

route_to_depots_30 = {
    1:[25,24,14,13],
    2:[16,17,23,20],
    3:[10,22,27,30],
    4:[28,26,19],
    5:[11,12,29],
    6:[8,18,21,15,9],
    7:[19,20,14,7,2],
    8:[17,15,21,25],
    9:[22,8,6,5,1],
    10:[23,13,10,11],
    11:[29,26,24,16,4,3],
    12:[9,18,30,28,27],
    13:[30,23,5,4,2],
    14:[12,15,21,25,28],
    15:[7,9,10,16,22],
    16:[18,19,17,14,12],
    17:[29,24,8,1,3],
    18:[27,20,7,6,4,2],
    19:[7,6,5,3,1],
    20:[20,21,22,24,26,28],
    21:[23,8,16,13,11],
    22:[6,5,4,3,2,1],
    23:[17,30,14,9,10,12],
    24:[18,19,25,26,27,29],
    25:[11,13,15,20,22,26],
    26:[10,16,15,27],
    27:[7,13,14,17,28],
    28:[9,12,11],
    29:[8,19,23,25,24,21],
    30:[18,30,29]
}

ALL_DEPOTS = list(range(1, 31))

# =========================
# 1) yardımcı fonksiyonlar (rota üretimi)
# =========================
def normalize_route(depots):
    """Tekrarları at, sırayı koru."""
    seen = set()
    out = []
    for d in depots:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out

def mutate_route(depots, hub, min_len=3, max_len=6):
    depots = depots[:]  # copy
    op = random.choice(["swap", "add", "remove", "replace"])

    if op == "swap" and len(depots) >= 2:
        i, j = random.sample(range(len(depots)), 2)
        depots[i], depots[j] = depots[j], depots[i]

    elif op == "add" and len(depots) < max_len:
        cand = [d for d in ALL_DEPOTS if d not in depots]
        if cand:
            depots.insert(random.randrange(len(depots)+1), random.choice(cand))

    elif op == "remove" and len(depots) > min_len:
        depots.pop(random.randrange(len(depots)))

    elif op == "replace":
        cand = [d for d in ALL_DEPOTS if d not in depots]
        if cand and depots:
            depots[random.randrange(len(depots))] = random.choice(cand)

    depots = normalize_route(depots)

    # uzunluğu min/max aralığına çek
    if len(depots) < min_len:
        cand = [d for d in ALL_DEPOTS if d not in depots]
        random.shuffle(cand)
        depots += cand[:(min_len - len(depots))]
    if len(depots) > max_len:
        depots = depots[:max_len]

    return depots

def random_route(min_len=3, max_len=6):
    L = random.randint(min_len, max_len)
    return random.sample(ALL_DEPOTS, L)

def merge_routes(a, b, min_len=3, max_len=6):
    merged = normalize_route(a + b)
    if len(merged) < min_len:
        cand = [d for d in ALL_DEPOTS if d not in merged]
        random.shuffle(cand)
        merged += cand[:(min_len - len(merged))]
    if len(merged) > max_len:
        merged = merged[:max_len]
    return merged

def signature(hub, depots):
    """Aynı rotayı tekrar üretmemek için imza (hub + sıra)."""
    return (hub, tuple(depots))

def expand_routes_to_100(route_to_am, route_to_depots, target=100, seed=42,
                         min_len=3, max_len=6,
                         n_mutate=30, n_random=25, n_merge=15):
    random.seed(seed)

    new_am = dict(route_to_am)
    new_dep = {r: normalize_route(ds) for r, ds in route_to_depots.items()}
    used = set(signature(new_am[r], new_dep[r]) for r in new_dep)

    next_id = max(new_dep.keys()) + 1

    # (A) mutation
    base_routes = list(new_dep.keys())
    for _ in range(n_mutate):
        if next_id > target:
            break
        r0 = random.choice(base_routes)
        hub = new_am[r0]
        depots = mutate_route(new_dep[r0], hub, min_len, max_len)
        sig = signature(hub, depots)
        if sig in used:
            continue
        new_am[next_id] = hub
        new_dep[next_id] = depots
        used.add(sig)
        next_id += 1

    # (B) random
    for _ in range(n_random):
        if next_id > target:
            break
        hub = random.choice([1,2,3])
        depots = random_route(min_len, max_len)
        sig = signature(hub, depots)
        if sig in used:
            continue
        new_am[next_id] = hub
        new_dep[next_id] = depots
        used.add(sig)
        next_id += 1

    # (C) merge (same hub)
    by_hub = {1:[], 2:[], 3:[]}
    for r in new_dep:
        by_hub[new_am[r]].append(r)

    for _ in range(n_merge):
        if next_id > target:
            break
        hub = random.choice([1,2,3])
        if len(by_hub[hub]) < 2:
            continue
        r1, r2 = random.sample(by_hub[hub], 2)
        depots = merge_routes(new_dep[r1], new_dep[r2], min_len, max_len)
        sig = signature(hub, depots)
        if sig in used:
            continue
        new_am[next_id] = hub
        new_dep[next_id] = depots
        used.add(sig)
        by_hub[hub].append(next_id)
        next_id += 1

    # Fill remaining
    while next_id <= target:
        if random.random() < 0.5:
            r0 = random.choice(list(new_dep.keys()))
            hub = new_am[r0]
            depots = mutate_route(new_dep[r0], hub, min_len, max_len)
        else:
            hub = random.choice([1,2,3])
            depots = random_route(min_len, max_len)

        sig = signature(hub, depots)
        if sig in used:
            continue
        new_am[next_id] = hub
        new_dep[next_id] = depots
        used.add(sig)
        next_id += 1

    return new_am, new_dep

route_to_am_100, route_to_depots_100 = expand_routes_to_100(
    route_to_am_30, route_to_depots_30,
    target=100, seed=7,
    min_len=3, max_len=6,
    n_mutate=30, n_random=25, n_merge=15
)


# =========================
# 2) rotalarr.csv formatına çevir (birebir)
# =========================
RETURN_TO_HUB = False  # dönüş istersen True yap (en sona yine "Bölge birliği X" ekler)

def build_route_points(hub_id: int, depots: list[int]) -> list[str]:
    # İSTENEN FORMAT:
    # "Bölge birliği 2", "Depo 25", ...
    pts = [f"Bölge birliği {hub_id}"] + [f"Depo {d}" for d in depots]
    if RETURN_TO_HUB:
        pts.append(f"Bölge birliği {hub_id}")  # H2 değil! Noktalar.csv ile aynı isim
    return pts

rotalarr_lines = []
for rid in range(1, 101):
    hub_id = route_to_am_100[rid]
    depots = route_to_depots_100[rid]
    points = build_route_points(hub_id, depots)

    # CSV satırını birebir senin formatında yaz (header yok)
    line = ",".join([f"Rota {rid}"] + points)
    rotalarr_lines.append(line)

rotalarr_out = "rotalarr_generated.csv"
with open(rotalarr_out, "w", encoding="utf-8-sig") as f:
    f.write("\n".join(rotalarr_lines))

print("Yazıldı:", rotalarr_out)
print("Örnek ilk 4 satır:")
for x in rotalarr_lines[:4]:
    print(x)

'''

# =========================
# 3) Noktalar.csv oku (birebir) + geodesic
# =========================
from geopy.distance import geodesic
import pandas as pd
import os

# ---- DOSYA YOLLARI ----
noktalar_file = r"C:\Users\gizem\OneDrive\Belgeler\GitHub\gizem_tez\Makale-21-01-26\Noktalar.csv"
rotalar_file  = r"C:\Users\gizem\OneDrive\Belgeler\GitHub\gizem_tez\Makale-21-01-26\rotalarr.csv"

def clean_name(x):
    return str(x).replace("\u00a0", " ").strip()

# =========================
# 1) NOKTALAR VERİSİNİ HAZIRLA
# =========================
df_noktalar = pd.read_csv(noktalar_file, encoding="utf-8-sig")
df_noktalar["Ad"] = df_noktalar["Ad"].apply(clean_name)

# Koordinat temizliği
df_noktalar["Enlem"]  = pd.to_numeric(df_noktalar["Enlem"].astype(str).str.replace(",", "."), errors="coerce")
df_noktalar["Boylam"] = pd.to_numeric(df_noktalar["Boylam"].astype(str).str.replace(",", "."), errors="coerce")

if df_noktalar[["Enlem", "Boylam"]].isna().any().any():
    print("Uyarı: Bazı koordinatlar sayıya dönüştürülemedi!")

noktalar_dict = df_noktalar.set_index("Ad")[["Enlem", "Boylam"]].to_dict(orient="index")

# =========================
# 2) ROTALAR VERİSİNİ OKU
# =========================
with open(rotalar_file, "r", encoding="utf-8-sig") as f:
    max_columns = max(len(line.strip().split(",")) for line in f)

column_names = ["Route"] + [f"Point_{i}" for i in range(1, max_columns)]
df_rotalar = pd.read_csv(rotalar_file, encoding="utf-8-sig", names=column_names, header=None, engine="python")

# Veri temizleme
for c in df_rotalar.columns:
    df_rotalar[c] = df_rotalar[c].apply(clean_name)
    df_rotalar.loc[df_rotalar[c].isin(["", "nan", "None", "None"]), c] = None

# =========================
# 3) HESAPLAMA FONKSİYONLARI
# =========================
fuel_price_per_litre = 1.29
capacity_by_num_depots = {3: 750, 4: 900, 5: 1200, 6: 1400}
# Ziyaret edilen depo sayısına göre 100km'deki yakıt tüketimi (Litre)
fuel_consumption_map = {
    3: 25,
    4: 30,
    5: 38,
    6: 38
}

def calculate_route_distance(route_points):
    total = 0.0
    for i in range(len(route_points) - 1):
        p1, p2 = route_points[i], route_points[i+1]
        a, b = noktalar_dict.get(p1), noktalar_dict.get(p2)
        if not a or not b:
            missing = p1 if not a else p2
            raise KeyError(f"Nokta bulunamadı: {missing}")
        total += geodesic((a["Enlem"], a["Boylam"]), (b["Enlem"], b["Boylam"])).km
    return total

def route_cost(distance_km, num_depots):
    # Tablodan o depo sayısına denk gelen tüketim oranını al (Bulamazsa varsayılan 30 al)
    consumption = fuel_consumption_map.get(num_depots, 40)
    
    # Hesaplama: (Mesafe * (Tüketim / 100)) * Yakıt Fiyatı
    return (distance_km * (consumption / 100.0)) * fuel_price_per_litre

def get_capacity(route_points):
    # 'Depo' veya 'D' ile başlayan noktaları say (Hub hariç)
    # Senin formatına göre 'Depo' kelimesini aratıyoruz
    n_depots = sum(1 for p in route_points if "DEPO" in str(p).upper())
    return capacity_by_num_depots.get(n_depots, None)

# =========================
# 4) ANA DÖNGÜ VE RAPORLAMA
# =========================
rows_out = []

for _, row in df_rotalar.iterrows():
    route_name = row["Route"]
    # Satırdaki boş olmayan tüm noktaları listeye al (Rota adı hariç)
    all_points = row.drop(labels=["Route"]).dropna().tolist()
    
    try:
        dist_km = calculate_route_distance(all_points)
        
        # DÜZELTME: Maliyet hesabı için önce depo sayısını buluyoruz
        depots_only_count = sum(1 for p in all_points if "DEPO" in str(p).upper())
        cost_eur = route_cost(dist_km, depots_only_count) # Fonksiyona ikinci parametre eklendi
        
        cap = get_capacity(all_points)
        err = ""
    except Exception as e:
        dist_km, cost_eur, cap = None, None, None
        err = str(e)

    # Hub ve Depo bilgisini görselleştirmek için ayıkla
    hub_point = all_points[0] if all_points else "Bilinmiyor"
    depots_only = [p for p in all_points if "DEPO" in str(p).upper()]

    rows_out.append({
        "route_name": route_name,
        "hub_point": hub_point,
        "depots": " - ".join(depots_only),
        "num_depots": len(depots_only),
        "distance_km": dist_km,
        "route_cost_eur": cost_eur,
        "route_capacity": cap,
        "error": err
    })

df_out = pd.DataFrame(rows_out)

# =========================
# 5) ÇIKTI
# =========================
output_excel = "routes_analysis_results.xlsx"
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    df_out.to_excel(writer, sheet_name="Analiz Sonuclari", index=False)
    df_rotalar.to_excel(writer, sheet_name="Okunan Rotalar", index=False)

print(f"✅ Analiz tamamlandı ve '{output_excel}' dosyasına kaydedildi.")
print(df_out[["route_name", "distance_km", "route_cost_eur"]].head())