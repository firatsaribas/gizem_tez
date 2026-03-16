import pandas as pd
from geopy.distance import geodesic

# ── Load Data ──────────────────────────────────────────────────────────────────
csv_file = "150suppliers.csv"
df_suppliers = pd.read_csv(csv_file)

num_suppliers = len(df_suppliers)
names = df_suppliers["Ad"].tolist()

# Separate suppliers (T...) and hubs (BB...)
supplier_names = [n for n in names if str(n).startswith("T")]
hub_names      = [n for n in names if str(n).startswith("BB")]

# Suppliers WITH vehicle (k index) — update this list as needed
vehicle_supplier_numbers = [
    2, 13, 32, 33, 37, 39, 46, 48, 58, 62, 65, 67,
    77, 82, 87, 89, 97, 98, 105, 106, 107,
    110, 112, 119, 122, 127, 135, 139, 140, 143
]
# Map numbers to zero-padded supplier names (e.g. 2 → "T002", 13 → "T013")
vehicle_supplier_names = [f"T{n:03d}" for n in vehicle_supplier_numbers]

# ── Display Label Mappings ─────────────────────────────────────────────────────
# T001 → 1, T002 → 2, ... (strip T and leading zeros)
supplier_label = {n: int(str(n)[1:]) for n in supplier_names}
# BB01 → 1, BB02 → 2, ... (strip BB and leading zeros)
hub_label = {n: int(str(n)[2:]) for n in hub_names}
# Vehicle suppliers re-indexed in given order: T002 → 1, T013 → 2, T037 → 3, ...
vehicle_label = {n: idx + 1 for idx, n in enumerate(vehicle_supplier_names)}

# ── Distance Matrix ────────────────────────────────────────────────────────────
distance_matrix = pd.DataFrame(index=names, columns=names, dtype=float)

for i in range(num_suppliers):
    for j in range(num_suppliers):
        if i != j:
            point1 = (df_suppliers.iloc[i]["Enlem"], df_suppliers.iloc[i]["Boylam"])
            point2 = (df_suppliers.iloc[j]["Enlem"], df_suppliers.iloc[j]["Boylam"])
            distance_matrix.iloc[i, j] = geodesic(point1, point2).km
        else:
            distance_matrix.iloc[i, j] = 0

distance_matrix.to_excel("150suppliersdistance.xlsx")
print("Distance matrix saved.")

# ── Fuel Cost Constants ────────────────────────────────────────────────────────
fuel_consumption_per_100km = 12    # litres/100 km
fuel_price_per_litre       = 1.29  # Euro/litre

# ── Full Fuel Cost Matrix (all points × all points) ───────────────────────────
cost_matrix = distance_matrix.copy()

for i in range(num_suppliers):
    for j in range(num_suppliers):
        if i != j:
            dist = float(distance_matrix.iloc[i, j])
            cost_matrix.iloc[i, j] = dist * (fuel_consumption_per_100km / 100) * fuel_price_per_litre
        else:
            cost_matrix.iloc[i, j] = 0

cost_matrix.to_excel("noktalar_arasi_yakit_maliyetleri150suppliers.xlsx")
print("Fuel cost matrix saved.")

# ── Beta Matrix ────────────────────────────────────────────────────────────────
# Indices: f (all suppliers T...) × k (vehicle suppliers)
# beta[f][k] = fuel_cost(f, k) + 50   if f != k
#            = 5                       if f == k
# Labels: rows → 1,2,3... (supplier number), cols → 1,2,3... (vehicle re-index)

beta_row_labels = [supplier_label[f] for f in supplier_names]
beta_col_labels = [vehicle_label[k] for k in vehicle_supplier_names]
beta_matrix = pd.DataFrame(index=beta_row_labels, columns=beta_col_labels, dtype=float)

for f in supplier_names:
    for k in vehicle_supplier_names:
        fl = supplier_label[f]
        kl = vehicle_label[k]
        if f != k:
            beta_matrix.loc[fl, kl] = cost_matrix.loc[f, k] + 50
        else:
            beta_matrix.loc[fl, kl] = 5

beta_matrix.index.name   = "f"
beta_matrix.columns.name = "k"

# ── Gamma Matrix ───────────────────────────────────────────────────────────────
# Indices: f (all suppliers T...) × b (hubs BB...) × k (vehicle suppliers)
# gamma[f][b][k] = fuel_cost(f, k) / 245 + fuel_cost(k, b)
# Labels: f → supplier number, b → hub number, k → vehicle re-index

gamma_rows = []

for f in supplier_names:
    for b in hub_names:
        for k in vehicle_supplier_names:
            fuel_fk   = cost_matrix.loc[f, k]
            fuel_kb   = cost_matrix.loc[k, b]
            gamma_val = fuel_fk / 245 + fuel_kb/245
            gamma_rows.append({
                "f": supplier_label[f],
                "b": hub_label[b],
                "k": vehicle_label[k],
                "gamma": gamma_val
            })

gamma_df = pd.DataFrame(gamma_rows)

# ── Write All Sheets to One Workbook ──────────────────────────────────────────
output_combined = "150suppliers_all_matrices.xlsx"

with pd.ExcelWriter(output_combined, engine="openpyxl") as writer:
    distance_matrix.to_excel(writer, sheet_name="distance")
    cost_matrix.to_excel(writer, sheet_name="fuel_cost")
    beta_matrix.to_excel(writer, sheet_name="beta")
    gamma_df.to_excel(writer, sheet_name="gamma", index=False)

print("Combined workbook saved:", output_combined)
print(f"  - beta  : {len(supplier_names)} suppliers x {len(vehicle_supplier_names)} vehicle suppliers")
print(f"  - gamma : {len(gamma_rows)} rows ({len(supplier_names)} f x {len(hub_names)} b x {len(vehicle_supplier_names)} k)")