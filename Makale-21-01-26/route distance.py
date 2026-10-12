import pandas as pd
from geopy.distance import geodesic

noktalar_file = r"C:\Users\gizem\OneDrive\Belgeler\GitHub\gizem_tez\Makale-21-01-26\Noktalar.csv"
rotalar_file  = r"C:\Users\gizem\OneDrive\Belgeler\GitHub\gizem_tez\Makale-21-01-26\rotalarr.csv"

def clean_name(x):
    return str(x).replace("\u00a0", " ").strip()

# ---- ROTALAR: değişken kolon sayısı için dinamik okuma ----
with open(rotalar_file, "r", encoding="utf-8-sig") as f:
    max_columns = max(len(line.strip().split(",")) for line in f)

column_names = ["Route"] + [f"Point_{i}" for i in range(1, max_columns)]
df_rotalar = pd.read_csv(
    rotalar_file,
    encoding="utf-8-sig",
    names=column_names,
    header=None,
    engine="python"
)

# temizle
for c in df_rotalar.columns:
    df_rotalar[c] = df_rotalar[c].apply(clean_name)
    df_rotalar.loc[df_rotalar[c].isin(["", "nan", "None"]), c] = None

# ---- NOKTALAR ----
df_noktalar = pd.read_csv(noktalar_file, encoding="utf-8-sig")
df_noktalar["Ad"] = df_noktalar["Ad"].apply(clean_name)

# koordinatları kesin sayıya çevir (boşluk/virgül sorunlarını gider)
df_noktalar["Enlem"]  = pd.to_numeric(df_noktalar["Enlem"].astype(str).str.replace(",", ".").str.strip(), errors="coerce")
df_noktalar["Boylam"] = pd.to_numeric(df_noktalar["Boylam"].astype(str).str.replace(",", ".").str.strip(), errors="coerce")

bad = df_noktalar[df_noktalar[["Enlem","Boylam"]].isna().any(axis=1)]
if len(bad) > 0:
    raise ValueError(f"Noktalar.csv koordinat hatası olan satırlar:\n{bad.head(20)}")

noktalar_dict = df_noktalar.set_index("Ad")[["Enlem", "Boylam"]].to_dict(orient="index")

def calculate_route_distance(route_points):
    total = 0.0
    for i in range(len(route_points) - 1):
        p1, p2 = route_points[i], route_points[i+1]
        a = noktalar_dict.get(p1)
        b = noktalar_dict.get(p2)
        if not a or not b:
            # hangi isim eksik net görelim:
            missing = []
            if not a: missing.append(p1)
            if not b: missing.append(p2)
            raise KeyError(f"Nokta bulunamadı: {missing}. Noktalar.csv'de yok veya yazımı farklı.")
        total += geodesic((a["Enlem"], a["Boylam"]), (b["Enlem"], b["Boylam"])).km
    return total

route_distances = {}
for _, row in df_rotalar.iterrows():
    route_name = row["Route"]
    points = row.drop(labels=["Route"]).dropna().tolist()
    route_distances[route_name] = calculate_route_distance(points)

df_out = pd.DataFrame(route_distances.items(), columns=["Rota", "Toplam Mesafe (km)"])
df_out.to_excel("rota_mesafeleri.xlsx", index=False)
print("✅ rota_mesafeleri.xlsx yazıldı")