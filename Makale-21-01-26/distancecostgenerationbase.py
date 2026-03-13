import pandas as pd
from geopy.distance import geodesic

# CSV dosyasını oku (Dosya adını değiştirin)
csv_file = "150suppliers.csv"  # Dosya yolunu kendi dosyanıza göre değiştirin
df_suppliers = pd.read_csv(csv_file)

# Mesafe matrisini oluştur
num_suppliers = len(df_suppliers)
distance_matrix = pd.DataFrame(index=df_suppliers["Ad"], columns=df_suppliers["Ad"])

# Mesafeleri hesapla
for i in range(num_suppliers):
    for j in range(num_suppliers):
        if i != j:
            point1 = (df_suppliers.iloc[i]["Enlem"], df_suppliers.iloc[i]["Boylam"])
            point2 = (df_suppliers.iloc[j]["Enlem"], df_suppliers.iloc[j]["Boylam"])
            distance_matrix.iloc[i, j] = geodesic(point1, point2).km
        else:
            distance_matrix.iloc[i, j] = 0  # Aynı noktaya olan mesafe sıfırdır

# Mesafe matrisini Excel dosyasına kaydetme
output_file = "150suppliersdistance.xlsx"
distance_matrix.to_excel(output_file)

# Sabit yakıt tüketimi ve yakıt fiyatı
fuel_consumption_per_100km = 12  # litre/100 km (tam yüklü araç)
fuel_price_per_litre = 1.29  # Euro/litre

# Mesafe matrisini bir DataFrame olarak alın (dist_matrix)
# dist_matrix: tüm noktalar arasındaki mesafeleri içeren DataFrame

# Yakıt maliyeti matrisini oluşturma
cost_matrix = distance_matrix.copy()

# Yakıt maliyetini hesaplama
for i in range(len(cost_matrix)):
    for j in range(len(cost_matrix)):
        if i != j:
            distance = float(cost_matrix.iloc[i, j])  # Mesafe (km)
            fuel_consumption = distance * (fuel_consumption_per_100km / 100)  # Tüketim (litre)
            cost = fuel_consumption * fuel_price_per_litre  # Yakıt maliyeti (TL)
            cost_matrix.iloc[i, j] = cost
        else:
            cost_matrix.iloc[i, j] = 0  # Aynı noktadaki maliyet sıfırdır

# Yakıt maliyet matrisini kaydetme
output_file_fuel_cost = "noktalar_arasi_yakit_maliyetleri3150supplierscost.xlsx"
cost_matrix.to_excel(output_file_fuel_cost)

print("Yakıt maliyet matris dosyası oluşturuldu:", output_file_fuel_cost)