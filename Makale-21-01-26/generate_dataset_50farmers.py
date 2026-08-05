"""
generate_dataset_50farmers.py
=============================
Creates a 50-farmer data set for the multi-period collaborative pickup open VRP,
using exactly the data-generation logic of the 10-farmer base case
(basecase.tex / parameters_input_V10.xlsx).

OUTPUT
    parameters_input_50farmers.xlsx   <- the data set, in the same sheet schema
                                         that VRP_kgV2.py / VRP_kgV3.py read
    farmers_50.csv                    <- same table, flat
    farmers_50_table.tex              <- longtable for the thesis

WHAT IS CARRIED OVER UNCHANGED FROM THE BASE CASE
    - six collection periods, three deterioration scenarios
      (d = 0.02 / 0.045 / 0.085 with rho = 0.25 / 0.55 / 0.20)
    - vehicle capacity 3,500 kg; fuel basis 12 L/100 km at 1.29 EUR/L
    - the five harvest profiles and the rule g_it = round5(supply_i * share_t)
    - the initial-inventory rule, the price range and the service-cost logic
    - producer sizes in the base case's 950-5,800 kg band

WHAT CHANGES, AND WHY
    1. STUDY REGION is extended from Manisa + Izmir to the wider Aegean fresh
       tomato belt: Manisa + Izmir + Aydin + Denizli.  Both added provinces meet
       the selection criteria already stated in the Base Case Description
       (open-field fresh tomato; not predominantly protected cultivation as in
       Antalya, not processing tomato as in Bursa).
       Mugla is deliberately NOT included: its tomato production is
       predominantly under protected cultivation, so adding it would contradict
       the paper's own reason for excluding Antalya.

    2. HUB moves from Turgutlu to Odemis, the centroid of the extended region
       and a real Kucuk Menderes fresh-produce packing centre, so the "centrally
       positioned within the production corridor" justification still holds.

    3. ONE PRODUCER PER DISTRICT (two only in the largest districts, placed at
       opposite ends), with a minimum separation of 12 km enforced between any
       two producers.  In the base case no two farmers were within 10 km and the
       median nearest-neighbour distance was 26.9 km; packing several producers
       into the same district instead produces near-identical points, which
       makes the routing decisions degenerate.

    The published 10-farmer base case is NOT affected by any of this.

Run:  python generate_dataset_50farmers.py
"""

import csv
import math
import os
import random

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 2026
OUT_XLSX = os.path.join(HERE, "parameters_input_50farmers.xlsx")
OUT_CSV = os.path.join(HERE, "farmers_50.csv")
OUT_TEX = os.path.join(HERE, "farmers_50_table.tex")


# =============================================================================
# 1.  BASE-CASE GENERATION RULES  (unchanged)
# =============================================================================

nT = 6                                    # collection periods
nS = 3                                    # deterioration scenarios
VEHICLE_CAPACITY = 3500.0                 # kg
PEAK_LOAD_TARGET = 0.75                   # base case: 5,275 / 7,000 = 75.4%

SCENARIOS = {                             # s: (d_s, rho_s)
    1: (0.020, 0.25),                     # favourable storage environment
    2: (0.045, 0.55),                     # moderate baseline
    3: (0.085, 0.20),                     # unfavourable conditions
}

HUB_COORD = (38.2292, 27.9694)
HUB_NAME = "Odemis"

# Period shares recovered exactly from the base-case harvest table:
#   every base-case g_it equals round5( total_supply_i * share[profile_i][t] )
PROFILE_SHARES = {
    "early":     (0.24, 0.22, 0.19, 0.15, 0.12, 0.08),
    "early-mid": (0.18, 0.21, 0.22, 0.17, 0.13, 0.09),
    "mid":       (0.10, 0.16, 0.22, 0.23, 0.18, 0.11),
    "mid-late":  (0.06, 0.11, 0.18, 0.24, 0.23, 0.18),
    "late":      (0.03, 0.06, 0.12, 0.20, 0.27, 0.32),
}

# Initial inventory: a small proportion of the first-period harvest, larger for
# early profiles and smaller for late ones.  Recovered factors below reproduce
# every a_i^0 in the base case; late producers carry only a token stock
# (base case: g_i1 of 30 -> 0 kg, and 40/55/55/65 -> 5 kg).
A0_FACTOR = {"early": 0.295, "early-mid": 0.25, "mid": 0.19, "mid-late": 0.16}
A0_LATE_KG = 5.0
A0_LATE_MIN_HARVEST = 35.0

# Producer size: base-case band, scaled down for later profiles because in the
# base case the late/remote producers are the small ones.
SUPPLY_BOUNDS = (950, 5800)
SIZE_BAND = {"large": (2500, 5800), "medium": (1500, 3600), "small": (950, 2000)}
PROFILE_SIZE_FACTOR = {
    "early": 1.00, "early-mid": 1.00, "mid": 1.00, "mid-late": 0.75, "late": 0.50,
}

# Price rises with producer scale; service cost falls with scale and rises with
# distance from the hub (field accessibility, loading effort).  Same ranges as
# the base case.
R_MIN, R_MAX = 0.47, 0.55
ALPHA_MIN, ALPHA_MAX = 8.0, 13.0
ALPHA_SCALE_WEIGHT = 0.65

# Spatial separation, matching the base case's spacing
MIN_SEP_KM = 12.0
MAX_DRIFT_DEG = 0.07                      # producers stay inside their district
SECOND_PRODUCER_OFFSET = (0.085, 0.105)   # ~19 km apart in a two-producer district
JITTER = 0.022                            # ~2.5 km


# =============================================================================
# 2.  STUDY REGION -- open-field fresh tomato districts of the Aegean belt
# =============================================================================
# (district, province, lat, lon, production weight, [profile of each producer])
# Profiles follow agronomy: coastal and lowland plains harvest earlier, inland
# valleys mid, upland eastern and southern districts late.
#
# 39 districts, 50 producers:  Manisa 11/14, Izmir 12/16, Aydin 11/14, Denizli 5/6

DISTRICTS = [
    # ---- Manisa ----
    ("Turgutlu",    "Manisa",  38.4954, 27.7037, 100, ["early", "mid"]),
    ("Salihli",     "Manisa",  38.4815, 28.1414,  95, ["mid", "mid-late"]),
    ("Akhisar",     "Manisa",  38.9152, 27.8483,  85, ["early-mid", "mid"]),
    ("Saruhanli",   "Manisa",  38.7378, 27.5561,  70, ["early"]),
    ("Alasehir",    "Manisa",  38.3506, 28.5183,  55, ["late"]),
    ("Ahmetli",     "Manisa",  38.5223, 27.9450,  45, ["mid"]),
    ("Kirkagac",    "Manisa",  39.1017, 27.6717,  40, ["mid-late"]),
    ("Golmarmara",  "Manisa",  38.7189, 27.9931,  35, ["mid"]),
    ("Sarigol",     "Manisa",  38.2434, 28.6822,  30, ["late"]),
    ("Kula",        "Manisa",  38.5464, 28.6489,  25, ["late"]),
    ("Soma",        "Manisa",  39.1858, 27.6089,  22, ["mid-late"]),
    # ---- Izmir ----
    ("Menemen",     "Izmir",   38.6050, 27.0675,  98, ["early", "early-mid"]),
    ("Odemis",      "Izmir",   38.2292, 27.9694,  92, ["mid", "mid-late"]),
    ("Torbali",     "Izmir",   38.1611, 27.3611,  88, ["mid", "mid"]),
    ("Bergama",     "Izmir",   39.1211, 27.1806,  80, ["mid", "mid-late"]),
    ("Tire",        "Izmir",   38.0894, 27.7350,  60, ["mid"]),
    ("Bayindir",    "Izmir",   38.2214, 27.6497,  50, ["mid-late"]),
    ("Kemalpasa",   "Izmir",   38.4256, 27.4181,  48, ["early-mid"]),
    ("Menderes",    "Izmir",   38.2500, 27.1330,  38, ["early-mid"]),
    ("Selcuk",      "Izmir",   37.9500, 27.3667,  32, ["early"]),
    ("Kiraz",       "Izmir",   38.2300, 28.2050,  28, ["late"]),
    ("Kinik",       "Izmir",   39.0864, 27.3831,  26, ["late"]),
    ("Beydag",      "Izmir",   38.0850, 28.2100,  20, ["late"]),
    # ---- Aydin ----
    ("Soke",        "Aydin",   37.7500, 27.4100,  90, ["early", "mid"]),
    ("Nazilli",     "Aydin",   37.9130, 28.3200,  82, ["mid", "mid-late"]),
    ("Germencik",   "Aydin",   37.8700, 27.6000,  72, ["early-mid", "mid"]),
    ("Kocarli",     "Aydin",   37.7600, 27.7100,  52, ["mid"]),
    ("Sultanhisar", "Aydin",   37.8900, 28.1400,  44, ["mid"]),
    ("Incirliova",  "Aydin",   37.8500, 27.7200,  42, ["early-mid"]),
    ("Kuyucak",     "Aydin",   37.9100, 28.4300,  36, ["mid-late"]),
    ("Cine",        "Aydin",   37.6100, 28.0600,  34, ["mid-late"]),
    ("Bozdogan",    "Aydin",   37.6700, 28.3100,  27, ["late"]),
    ("Yenipazar",   "Aydin",   37.8100, 28.1900,  24, ["mid"]),
    ("Karacasu",    "Aydin",   37.7300, 28.6100,  18, ["late"]),
    # ---- Denizli ----
    ("Saraykoy",    "Denizli", 37.9200, 28.9200,  65, ["mid", "mid-late"]),
    ("Honaz",       "Denizli", 37.7600, 29.2600,  46, ["mid-late"]),
    ("Tavas",       "Denizli", 37.5700, 29.0700,  33, ["late"]),
    ("Cal",         "Denizli", 38.0800, 29.4000,  29, ["late"]),
    ("Bekilli",     "Denizli", 38.2500, 29.4300,  16, ["late"]),
]

N_FARMERS = sum(len(d[5]) for d in DISTRICTS)     # 50


def size_class(weight):
    """District size class, driving the producer size band -- as in the base
    case, where the largest-production districts hosted the large producers."""
    if weight >= 70:
        return "large"
    if weight >= 34:
        return "medium"
    return "small"


# =============================================================================
# 3.  HELPERS
# =============================================================================

def round5(x):
    """Round to the nearest 5 kg -- the rounding used throughout the base case."""
    return int(math.floor(x / 5.0 + 0.5) * 5)


def round_half(x):
    """Round to the nearest 0.5 -- service-cost granularity of the base case."""
    return math.floor(x * 2.0 + 0.5) / 2.0


def dist_km(a, b):
    """Planar approximation around latitude 38 deg, as in VRP_kgV2.py."""
    dlat = (b[0] - a[0]) * 111.0
    dlon = (b[1] - a[1]) * 111.0 * math.cos(math.radians(38.0))
    return math.hypot(dlat, dlon)


def normalise(values):
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def kmeans(points, k, rng, iters=300):
    """Depots are placed at production-cluster centres, which is what the base
    case did by hand with its two origin nodes."""
    centres = [list(p) for p in rng.sample(points, k)]
    for _ in range(iters):
        groups = [[] for _ in range(k)]
        for p in points:
            groups[min(range(k), key=lambda j: dist_km(p, centres[j]))].append(p)
        moved = 0.0
        for j, grp in enumerate(groups):
            if not grp:
                continue
            new = [sum(c[0] for c in grp) / len(grp), sum(c[1] for c in grp) / len(grp)]
            moved = max(moved, dist_km(centres[j], new))
            centres[j] = new
        if moved < 1e-4:
            break
    return [(round(c[0], 4), round(c[1], 4)) for c in centres]


def enforce_separation(farmers, iters=400):
    """Push apart any pair closer than MIN_SEP_KM, keeping every producer within
    MAX_DRIFT_DEG of its own district centre."""
    kx = 111.0 * math.cos(math.radians(38.0))
    for _ in range(iters):
        worst = 1e9
        for a in range(len(farmers)):
            for b in range(a + 1, len(farmers)):
                fa, fb = farmers[a], farmers[b]
                d = dist_km(fa["coord"], fb["coord"])
                worst = min(worst, d)
                if d >= MIN_SEP_KM or d < 1e-9:
                    continue
                push = (MIN_SEP_KM - d) / 2.0 * 1.05
                ux = (fb["coord"][0] - fa["coord"][0]) * 111.0 / d
                uy = (fb["coord"][1] - fa["coord"][1]) * kx / d
                for f, sgn in ((fa, -1), (fb, +1)):
                    lat = f["coord"][0] + sgn * push * ux / 111.0
                    lon = f["coord"][1] + sgn * push * uy / kx
                    lat = min(max(lat, f["centre"][0] - MAX_DRIFT_DEG),
                              f["centre"][0] + MAX_DRIFT_DEG)
                    lon = min(max(lon, f["centre"][1] - MAX_DRIFT_DEG),
                              f["centre"][1] + MAX_DRIFT_DEG)
                    f["coord"] = (lat, lon)
        if worst >= MIN_SEP_KM:
            break
    for f in farmers:
        f["coord"] = (round(f["coord"][0], 4), round(f["coord"][1], 4))


# =============================================================================
# 4.  BUILD THE 50 FARMERS
# =============================================================================

rng = random.Random(SEED)
farmers = []

for name, province, lat, lon, weight, profiles in DISTRICTS:
    for idx, profile in enumerate(profiles):
        if len(profiles) == 1:
            off_lat = off_lon = 0.0
        else:                                     # opposite ends of the district
            sgn = 1 if idx == 0 else -1
            off_lat = sgn * SECOND_PRODUCER_OFFSET[0]
            off_lon = sgn * SECOND_PRODUCER_OFFSET[1]
        cls = size_class(weight)
        lo, hi = SIZE_BAND[cls]
        supply = rng.uniform(lo, hi) * PROFILE_SIZE_FACTOR[profile]
        farmers.append({
            "district": name, "province": province, "profile": profile,
            "weight": weight, "size_class": cls,
            "supply_raw": min(max(supply, SUPPLY_BOUNDS[0]), SUPPLY_BOUNDS[1]),
            "centre": (lat, lon),
            "coord": (lat + off_lat + rng.uniform(-JITTER, JITTER),
                      lon + off_lon + rng.uniform(-JITTER, JITTER)),
        })

assert len(farmers) == N_FARMERS == 50, f"district table yields {len(farmers)} producers"
enforce_separation(farmers)

# order the table by province then district, so the Excel sheet reads naturally
farmers.sort(key=lambda f: (f["province"], f["district"], -f["supply_raw"]))
for i, f in enumerate(farmers, start=1):
    f["i"] = i

# ---- harvest g_it and initial inventory a_i^0 -------------------------------
for f in farmers:
    f["g"] = [round5(f["supply_raw"] * sh) for sh in PROFILE_SHARES[f["profile"]]]
    f["supply"] = sum(f["g"])
    if f["profile"] == "late":
        f["a0"] = A0_LATE_KG if f["g"][0] >= A0_LATE_MIN_HARVEST else 0.0
    else:
        f["a0"] = float(round5(A0_FACTOR[f["profile"]] * f["g"][0]))

# ---- unit price r_i and service cost alpha_i --------------------------------
scale_norm = normalise([f["supply"] for f in farmers])
remote_norm = normalise([dist_km(HUB_COORD, f["coord"]) for f in farmers])
for f, sn, rn in zip(farmers, scale_norm, remote_norm):
    f["dist_hub_km"] = round(dist_km(HUB_COORD, f["coord"]), 1)
    f["r_price"] = round(R_MIN + (R_MAX - R_MIN) * sn, 2)
    f["alpha_score"] = ALPHA_SCALE_WEIGHT * (1 - sn) + (1 - ALPHA_SCALE_WEIGHT) * rn
for f, an in zip(farmers, normalise([f["alpha_score"] for f in farmers])):
    f["alpha"] = round_half(ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * an)

# ---- fleet: keep the base case's peak-period capacity tightness -------------
period_harvest = [sum(f["g"][t] for f in farmers) for t in range(nT)]
peak = max(period_harvest)
nK = max(2, math.ceil(peak / (PEAK_LOAD_TARGET * VEHICLE_CAPACITY)))

# ---- depots: one per vehicle, at production-cluster centres ----------------
origins = kmeans([f["coord"] for f in farmers], nK, random.Random(SEED))
origins.sort(key=lambda c: (-c[0], c[1]))       # numbered north to south


# =============================================================================
# 5.  WRITE THE WORKBOOK  (schema read by VRP_kgV2.py / VRP_kgV3.py)
# =============================================================================

HEADER_FONT = Font(bold=True, color="FFFFFF")
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
CENTER = Alignment(horizontal="center")


def style_header(ws, headers):
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font, cell.fill, cell.alignment = HEADER_FONT, HEADER_FILL, CENTER


wb = openpyxl.Workbook()
fleet = nK * VEHICLE_CAPACITY

# ---- Instructions -----------------------------------------------------------
ws = wb.active
ws.title = "Instructions"
for r, text in enumerate([
    "HOW TO USE THIS FILE",
    "1. DIMENSIONS: set nF, nK, nT, nS.",
    "2. SCENARIOS: enter d_s and rho_s (probabilities must sum to 1).",
    "3. VEHICLES: one row per vehicle k with capacity q_k (kg). Match nK.",
    "4. FARMERS: one row per farmer -- r_price (EUR/kg), alpha (EUR/visit), a0 (kg).",
    "5. HARVEST: one row per farmer-period pair -- g_it (kg). Must have nF x nT rows.",
    "6. COORDINATES: lat/lon for all nodes (hub=0, farmers=1..nF, origins=nF+1..nF+nK).",
    "",
    f"THIS INSTANCE (nF={N_FARMERS}, nK={nK}, nT={nT}, nS={nS}):",
    "  Large test instance generated with the SAME data-generation logic as the",
    "  10-farmer base case (basecase.tex / parameters_input_V10.xlsx). Horizon,",
    "  scenarios, vehicle capacity, cost basis, harvest profiles, initial-inventory",
    "  rule, price range and service-cost logic are all unchanged.",
    "",
    "  Region : extended Aegean fresh tomato belt -- Manisa, Izmir, Aydin, Denizli.",
    "           Aydin and Denizli meet the same criteria as the base case provinces",
    "           (open-field fresh tomato). Mugla is excluded for the same reason as",
    "           Antalya: production there is predominantly protected cultivation.",
    f"  Hub    : {HUB_NAME} ({HUB_COORD[0]}, {HUB_COORD[1]}), centroid of the extended region.",
    "  Spread : 39 districts, one producer each (two in the largest districts, at",
    f"           opposite ends), minimum separation {MIN_SEP_KM:.0f} km between any two",
    "           producers -- the base case had no two farmers within 10 km.",
    "  Harvest: g_it = round5(supply_i x profile share_t), profiles early /",
    "           early-mid / mid / mid-late / late.",
    "  a0     : early .295, early-mid .25, mid .19, mid-late .16 of g_i1; late",
    "           producers carry a token 5 kg (0 kg if g_i1 < 35 kg).",
    "  r_price: 0.47-0.55 EUR/kg, increasing in producer scale.",
    "  alpha  : 8.0-13.0 EUR/visit, falling with scale, rising with hub distance.",
    f"  Fleet  : {nK} x {VEHICLE_CAPACITY:.0f} kg = {fleet:,.0f} kg/period; peak-period harvest",
    f"           {peak:,.0f} kg = {peak / fleet * 100:.1f}% of fleet (base case 75.4%).",
    "",
    f"  Reproducible: generate_dataset_50farmers.py, random seed {SEED}.",
], start=1):
    ws.cell(row=r, column=1, value=text)
ws.column_dimensions["A"].width = 84

# ---- Dimensions -------------------------------------------------------------
ws = wb.create_sheet("Dimensions")
style_header(ws, ["Parameter", "Value", "Description"])
for r, (p, v, desc) in enumerate([
    ("nF", N_FARMERS, "Number of farmers"), ("nK", nK, "Number of vehicles"),
    ("nT", nT, "Number of periods"), ("nS", nS, "Number of scenarios"),
], start=2):
    ws.cell(r, 1, p); ws.cell(r, 2, v); ws.cell(r, 3, desc)

# ---- Scenarios --------------------------------------------------------------
ws = wb.create_sheet("Scenarios")
style_header(ws, ["s", "d_s (deterioration rate)", "rho_s (probability)"])
for r, (s, (d_s, rho_s)) in enumerate(sorted(SCENARIOS.items()), start=2):
    ws.cell(r, 1, s); ws.cell(r, 2, d_s); ws.cell(r, 3, rho_s)

# ---- Vehicles ---------------------------------------------------------------
ws = wb.create_sheet("Vehicles")
style_header(ws, ["k", "q_k (kg)", "Description"])
for k in range(1, nK + 1):
    lat, lon = origins[k - 1]
    ws.cell(k + 1, 1, k)
    ws.cell(k + 1, 2, VEHICLE_CAPACITY)
    ws.cell(k + 1, 3, f"Medium collection truck - origin: depot {k} ({lat}, {lon})")

# ---- Farmers ----------------------------------------------------------------
ws = wb.create_sheet("Farmers")
style_header(ws, ["i", "r_price (EUR/kg)", "alpha (EUR/visit)", "a0 (kg)",
                  "District", "Province", "Profile", "Horizon supply (kg)",
                  "Distance to hub (km)"])
for f in farmers:
    r = f["i"] + 1
    ws.cell(r, 1, f["i"]); ws.cell(r, 2, f["r_price"]); ws.cell(r, 3, f["alpha"])
    ws.cell(r, 4, f["a0"]); ws.cell(r, 5, f["district"]); ws.cell(r, 6, f["province"])
    ws.cell(r, 7, f["profile"]); ws.cell(r, 8, f["supply"]); ws.cell(r, 9, f["dist_hub_km"])

# ---- Harvest ----------------------------------------------------------------
ws = wb.create_sheet("Harvest")
style_header(ws, ["i (farmer)", "t (period)", "g_it (kg)"])
r = 2
for f in farmers:
    for t in range(1, nT + 1):
        ws.cell(r, 1, f["i"]); ws.cell(r, 2, t); ws.cell(r, 3, f["g"][t - 1])
        r += 1

# ---- Coordinates ------------------------------------------------------------
ws = wb.create_sheet("Coordinates")
style_header(ws, ["node", "lat", "lon", "type"])
ws.cell(2, 1, 0); ws.cell(2, 2, HUB_COORD[0]); ws.cell(2, 3, HUB_COORD[1])
ws.cell(2, 4, f"hub ({HUB_NAME} packing centre)")
r = 3
for f in farmers:
    ws.cell(r, 1, f["i"]); ws.cell(r, 2, f["coord"][0]); ws.cell(r, 3, f["coord"][1])
    ws.cell(r, 4, f"farmer {f['i']} ({f['district']}, {f['province']})")
    r += 1
for k in range(1, nK + 1):
    lat, lon = origins[k - 1]
    ws.cell(r, 1, N_FARMERS + k); ws.cell(r, 2, lat); ws.cell(r, 3, lon)
    ws.cell(r, 4, f"origin {k} (depot, production-cluster centre)")
    r += 1

wb.save(OUT_XLSX)


# =============================================================================
# 6.  CSV AND LATEX TABLE
# =============================================================================

with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["i", "district", "province", "profile", "r_price", "alpha", "a0"]
               + [f"g_t{t}" for t in range(1, nT + 1)]
               + ["horizon_supply_kg", "dist_hub_km", "lat", "lon"])
    for f in farmers:
        w.writerow([f["i"], f["district"], f["province"], f["profile"], f["r_price"],
                    f["alpha"], f["a0"]] + f["g"]
                   + [f["supply"], f["dist_hub_km"], f["coord"][0], f["coord"][1]])

with open(OUT_TEX, "w", encoding="utf-8") as fh:
    fh.write("% Auto-generated by generate_dataset_50farmers.py -- do not edit by hand.\n")
    fh.write("\\begin{longtable}{ccccccccccc}\n")
    fh.write("\\caption{Farmer-specific parameters for the 50-farmer instance.}\n")
    fh.write("\\label{tab:farmer_params_50}\\\\\n\\toprule\n")
    head = ("$i$ & District & $r_i$ & $\\alpha_i$ & $a_i^0$ & $t\\!=\\!1$ & $t\\!=\\!2$ & "
            "$t\\!=\\!3$ & $t\\!=\\!4$ & $t\\!=\\!5$ & $t\\!=\\!6$ \\\\\n")
    fh.write(head + " & & (\\euro/kg) & (\\euro/visit) & (kg) & "
             "\\multicolumn{6}{c}{$g_{it}$ (kg)} \\\\\n\\midrule\n\\endfirsthead\n")
    fh.write("\\toprule\n" + head + "\\midrule\n\\endhead\n")
    for f in farmers:
        fh.write(f"{f['i']} & {f['district']} & {f['r_price']:.2f} & {f['alpha']:.1f} & "
                 f"{f['a0']:.0f} & " + " & ".join(str(v) for v in f["g"]) + " \\\\\n")
    fh.write("\\bottomrule\n\\end{longtable}\n")


# =============================================================================
# 7.  SUMMARY
# =============================================================================

nn = sorted(min(dist_km(a["coord"], b["coord"]) for b in farmers if b is not a)
            for a in farmers)
prof = {}
prov = {}
for f in farmers:
    prof[f["profile"]] = prof.get(f["profile"], 0) + 1
    prov[f["province"]] = prov.get(f["province"], 0) + 1

print(f"\nData set written -> {os.path.basename(OUT_XLSX)}")
print(f"  also: {os.path.basename(OUT_CSV)}, {os.path.basename(OUT_TEX)}")
print(f"\n  nF={N_FARMERS}  nK={nK}  nT={nT}  nS={nS}   (seed {SEED}, hub {HUB_NAME})")
print(f"  nodes |V|             : {1 + N_FARMERS + nK}  (hub + {N_FARMERS} farmers + {nK} origins)")
print(f"  districts             : {len(DISTRICTS)}  |  "
      + ", ".join(f"{k} {v}" for k, v in sorted(prov.items())))
print(f"  harvest profiles      : " + ", ".join(f"{k} {prof.get(k,0)}" for k in PROFILE_SHARES))
print(f"  harvest per period    : " + ", ".join(f"t{t+1}={period_harvest[t]:,}" for t in range(nT)))
print(f"  peak period load      : {peak:,} kg = {peak/fleet*100:.1f}% of fleet  (base case 75.4%)")
print(f"  horizon supply        : {sum(f['supply'] for f in farmers):,} kg harvest "
      f"+ {sum(f['a0'] for f in farmers):,.0f} kg initial")
print(f"  per-farmer supply     : {min(f['supply'] for f in farmers):,} - "
      f"{max(f['supply'] for f in farmers):,} kg  (base case 975 - 5,780)")
print(f"  r_price               : {min(f['r_price'] for f in farmers):.2f} - "
      f"{max(f['r_price'] for f in farmers):.2f} EUR/kg  (base case 0.47 - 0.54)")
print(f"  alpha                 : {min(f['alpha'] for f in farmers):.1f} - "
      f"{max(f['alpha'] for f in farmers):.1f} EUR/visit  (base case 8.0 - 13.0)")
print(f"  distance to hub       : {min(f['dist_hub_km'] for f in farmers):.1f} - "
      f"{max(f['dist_hub_km'] for f in farmers):.1f} km")
print(f"  nearest neighbour     : min {nn[0]:.1f} km, median {nn[len(nn)//2]:.1f} km"
      f"   (base case min 12.5, median 26.9)")
print()
