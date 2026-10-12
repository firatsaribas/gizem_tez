"""
ALNS - Adaptive Large Neighborhood Search  (FINAL VERSION)
Perishable Multi-Echelon Network Design
========================================
F=100 suppliers | K=25 vehicles | B=3 hubs | D=30 depots
R=30 routes     | T=6 periods   | S=3 scenarios | m=2 shelf life

CHANGES vs. previous version
─────────────────────────────
P1  Acceptance criterion: replaced LAHC with Simulated Annealing.
    T0 = 2% of initial objective; geometric cooling to T_min = 0.01% of obj.
P2  max_iter default raised to 30 000; segment to 200; lam to 0.3.
P3  _reassign_greedy / _reassign_regret: repair now uses per-hub targets
    derived from current Q, not aggregate demand.  This is the most
    important structural fix — it kept Echelon-1 and Echelon-2 decoupled.
P4  fix_service_violations: trim-before-fix step added at the top.
    Eliminates the m=2 waste cascade (deliver extra → waste → fix SL →
    deliver more → …).
P5  round_solution removed from inside the ALNS loop.  It is now called
    only once at the very end.  This reclaims ~30–40 % of runtime for
    actual search.
P6  no_improve_limit raised to max(500, max_iter // 15).  Prevents
    premature perturbation that was destroying good solutions.
P7  destroy_inventory_driven: removal size raised from //5 to //2.
P8  lam (weight reaction factor) raised to 0.3 in run_alns default.
P9  SIGMA_3 now scores accepted-non-improving moves correctly (score = 1).
P10 score variable initialised to 0 correctly; SIGMA scores are 10/5/1.

Excel file: step1_final.xlsx
Sheets used:
    beta          : f, k, beta
    gamma         : f, b, k, gamma
    theta         : k, theta
    vehicles      : k, f
    supply        : f, s, t, supply
    demand        : d, t, demand
    route_costs   : r, cost
    route_capacity: r, capacity
    route_depots  : route, depot
    route_hub     : route, hub
    stock_costs   : d, stock_cost
    scenario_prob : s, probability

Sections:
    1.  Data loading
    2.  Solution class
    3.  Helper computations
    4.  Route utilities
    5.  Greedy initialization
    6.  Destroy operators  (D1-D5)
    7.  Repair operators   (R1-R3)
    8.  Roulette wheel + weight update
    8b. Perturbation
    9.  Main ALNS loop  ← SA acceptance, 30k iters, fixed perturbation trigger
    10. Export results
    11. Integer rounding
    12. Constraint checker
"""

import math
import copy
import random
import time
import numpy as np
import pandas as pd
from scipy.stats import norm


def _compute_hub_assignment_rates(data):
    owned_f   = {data['owner'][k]: k for k in data['K']}
    pr_low    = data['pr'].get(1, 0.3)
    avg_no_supply = 35.0

    min_beta_no = min(
        (data['beta'].get((f, k), 999)
         for f in data['F'] if f not in owned_f
         for k in data['K']),
        default=50.0
    )

    hub_owner = {b: 0.0 for b in data['B']}
    for f, k in owned_f.items():
        sup = data['upsilon'].get((f, 1, 1), 0)
        if sup <= 0:
            continue
        b_best = min(data['B'],
                     key=lambda b: data['gamma'].get((f, b, k), float('inf')))
        hub_owner[b_best] += sup

    data['hub_assignment_rate'] = {b: 0.0 for b in data['B']}
    data['_f_by_beta'] = sorted(data['F'],
        key=lambda f: min(data['beta'].get((f, k), 999) for k in data['K']))


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def _find_optimal_route_cover(data):
    from itertools import combinations

    R_list     = list(data['R'])
    N_dict     = {r: set(data['N'][r]) for r in R_list}
    lam        = data['lambda']
    all_depots = set(data['D'])

    routes_sorted = sorted(R_list, key=lambda r: lam[r])
    candidates    = routes_sorted[:20]

    best_cost  = float('inf')
    best_combo = None

    for size in range(5, 13):
        for combo in combinations(candidates, size):
            covered = set()
            cost    = 0.0
            for r in combo:
                covered |= N_dict[r]
                cost    += lam[r]
            if all_depots.issubset(covered) and cost < best_cost:
                best_cost  = cost
                best_combo = list(combo)

    if best_combo is None:
        uncov = set(all_depots)
        best_combo = []
        while uncov:
            r_best = min(R_list,
                         key=lambda r: lam[r] / max(1, len(uncov & N_dict[r]))
                         if uncov & N_dict[r] else float('inf'))
            best_combo.append(r_best)
            uncov -= N_dict[r_best]

    return best_combo


def load_data(filepath):
    xl = pd.ExcelFile(filepath)

    beta_df        = pd.read_excel(xl, sheet_name='beta')
    gamma_df       = pd.read_excel(xl, sheet_name='gamma')
    theta_df       = pd.read_excel(xl, sheet_name='theta')
    supply_df      = pd.read_excel(xl, sheet_name='supply')
    demand_df      = pd.read_excel(xl, sheet_name='demand')
    route_costs_df = pd.read_excel(xl, sheet_name='route_costs')
    route_cap_df   = pd.read_excel(xl, sheet_name='route_capacity')
    route_dep_df   = pd.read_excel(xl, sheet_name='route_depots')
    route_hub_df   = pd.read_excel(xl, sheet_name='route_hub')
    stock_df       = pd.read_excel(xl, sheet_name='stock_costs')
    scenario_df    = pd.read_excel(xl, sheet_name='scenario_prob')

    data = {}

    data['F'] = sorted(supply_df['f'].unique().tolist())
    data['K'] = sorted(theta_df['k'].unique().tolist())
    data['B'] = [1, 2, 3]
    data['D'] = sorted(demand_df['d'].unique().tolist())
    data['R'] = sorted(route_costs_df['r'].unique().tolist())
    data['T'] = sorted(demand_df['t'].unique().tolist())
    data['S'] = sorted(scenario_df['s'].unique().tolist())
    data['m'] = 2
    data['alpha'] = 0.995

    data['theta'] = dict(zip(theta_df['k'], theta_df['theta']))
    data['lambda'] = dict(zip(route_costs_df['r'], route_costs_df['cost']))
    data['c']      = dict(zip(route_cap_df['r'],   route_cap_df['capacity']))

    data['N'] = {r: [] for r in data['R']}
    for _, row in route_dep_df.iterrows():
        data['N'][int(row['route'])].append(int(row['depot']))

    data['delta'] = {(b, r): 0 for b in data['B'] for r in data['R']}
    for _, row in route_hub_df.iterrows():
        data['delta'][(int(row['hub']), int(row['route']))] = 1

    data['upsilon'] = {}
    for _, row in supply_df.iterrows():
        data['upsilon'][(int(row['f']), int(row['s']), int(row['t']))] = float(row['supply'])

    data['mu']    = {}
    data['sigma'] = {}
    for _, row in demand_df.iterrows():
        i, t = int(row['d']), int(row['t'])
        mu_val = float(row['demand'])
        data['mu'][(i, t)]    = mu_val
        data['sigma'][(i, t)] = 0.10 * mu_val

    vehicles_df = pd.read_excel(xl, sheet_name='vehicles')
    data['owner'] = dict(zip(vehicles_df['k'].astype(int),
                             vehicles_df['f'].astype(int)))

    hub_depots = {b: set() for b in data['B']}
    for _, row in route_dep_df.iterrows():
        r = int(row['route'])
        d = int(row['depot'])
        b = int(route_hub_df[route_hub_df['route'] == r]['hub'].values[0])
        hub_depots[b].add(d)
    data['hub_depots'] = hub_depots

    data['beta'] = {}
    for _, row in beta_df.iterrows():
        data['beta'][(int(row['f']), int(row['k']))] = float(row['beta'])

    data['gamma'] = {}
    for _, row in gamma_df.iterrows():
        data['gamma'][(int(row['f']), int(row['b']), int(row['k']))] = float(row['gamma'])

    data['h']  = dict(zip(stock_df['d'].astype(int), stock_df['stock_cost']))
    data['p']  = 27.55696873
    data['pr'] = dict(zip(scenario_df['s'].astype(int), scenario_df['probability']))

    data['Z_alpha'] = 2.575
    data['C']       = 0.10

    data['route_hub_map'] = {}
    for _, row in route_hub_df.iterrows():
        data['route_hub_map'][int(row['route'])] = int(row['hub'])

    data['sl_rhs_cache'] = {}
    Z_alpha = data['Z_alpha']
    C       = data['C']
    for i in data['D']:
        for t in data['T']:
            mu_sum  = sum(data['mu'][(i, a)] for a in data['T'] if a <= t)
            var_sum = sum(data['mu'][(i, a)] ** 2 for a in data['T'] if a <= t)
            data['sl_rhs_cache'][(i, t)] = (
                mu_sum + math.sqrt(max(var_sum, 0)) * C * Z_alpha
            )

    _compute_hub_assignment_rates(data)

    _owned_f = {data['owner'][k]: k for k in data['K']}
    data['_f_by_beta'] = sorted(data['F'],
        key=lambda f: min(data['beta'].get((f, k), 999) for k in data['K']))
    data['_nonowner_best_k'] = {
        f: min(data['K'], key=lambda k: data['beta'].get((f, k), 999))
        for f in data['F'] if f not in _owned_f
    }

    owned_f = {data['owner'][k]: k for k in data['K']}
    min_beta_no = min(
        (data['beta'].get((f, k), 999)
         for f in data['F'] if f not in owned_f
         for k in data['K']),
        default=50.0
    )
    data['_no_cost_rate'] = min_beta_no / 35.0

    data['optimal_routes'] = _find_optimal_route_cover(data)

    print("Data loaded.")
    print(f"  F={len(data['F'])} | K={len(data['K'])} | B={len(data['B'])} | "
          f"D={len(data['D'])} | R={len(data['R'])} | T={len(data['T'])} | S={len(data['S'])}")
    return data


# =============================================================================
# SECTION 2 — SOLUTION CLASS
# =============================================================================

class Solution:
    __slots__ = ['Y','L','X','Z','Q','I','I_plus','W','obj']

    def __init__(self):
        self.Y      = {}
        self.L      = {}
        self.X      = {}
        self.Z      = {}
        self.Q      = {}
        self.I      = {}
        self.I_plus = {}
        self.W      = {}
        self.obj    = float('inf')

    def copy(self):
        s         = Solution()
        s.Y       = dict(self.Y)
        s.L       = dict(self.L)
        s.X       = dict(self.X)
        s.Z       = dict(self.Z)
        s.Q       = dict(self.Q)
        s.I       = dict(self.I)
        s.I_plus  = dict(self.I_plus)
        s.W       = dict(self.W)
        s.obj     = self.obj
        return s


# =============================================================================
# SECTION 3 — HELPER COMPUTATIONS
# =============================================================================

def compute_Q_from_Z(sol, data):
    for key in list(sol.Q.keys()):
        sol.Q[key] = 0

    route_hub = data['route_hub_map']
    for (i, r, t), z_val in sol.Z.items():
        if z_val > 1e-9:
            b = route_hub.get(r)
            if b is not None:
                sol.Q[(b, i, t)] = sol.Q.get((b, i, t), 0) + int(round(z_val))


def compute_inventory_waste(sol, data):
    D, T, B = data['D'], data['T'], data['B']
    mu, m   = data['mu'], data['m']

    for i in D:
        cumul_delivery = 0.0
        cumul_demand   = 0.0
        cumul_waste    = 0.0

        for t in T:
            delivery_t = sum(sol.Q.get((b, i, t), 0.0) for b in B)
            cumul_delivery += delivery_t
            cumul_demand   += mu[(i, t)]

            if t >= m:
                t0        = t - m + 1
                I_t0      = sol.I.get((i, t0), 0.0)
                d_window  = sum(mu[(i, a)] for a in T if t - m + 2 <= a <= t)
                w_window  = sum(sol.W.get((i, a), 0.0) for a in T if t - m + 2 <= a <= t - 1)
                waste_t   = max(0.0, I_t0 - d_window - w_window)
            else:
                waste_t = 0.0

            sol.W[(i, t)]      = waste_t
            cumul_waste       += waste_t
            inv                = cumul_delivery - cumul_demand - cumul_waste
            sol.I[(i, t)]      = inv
            sol.I_plus[(i, t)] = max(0.0, inv)


def compute_objective(sol, data):
    obj = 0.0
    pr, beta, gamma = data['pr'], data['beta'], data['gamma']
    h, p, lam       = data['h'], data['p'], data['lambda']

    for (f, b, k, s, t), y_val in sol.Y.items():
        if y_val:
            l_val = sol.L.get((f, b, k, s, t), 0.0)
            ps    = pr[s]
            obj  += ps * (beta[(f, k)] * y_val + gamma[(f, b, k)] * l_val)

    m = data['m']
    for i in data['D']:
        hi = h[i]
        for t in data['T']:
            ip = sol.I_plus.get((i, t), 0.0)
            if ip:
                obj += ip * hi
            if t >= m:
                w = sol.W.get((i, t), 0.0)
                if w:
                    obj += w * p

    for (r, t), x_val in sol.X.items():
        if x_val:
            obj += lam[r]

    sol.obj = obj
    return obj


def sl_rhs(data, i, t):
    return data['sl_rhs_cache'][(i, t)]


def sl_lhs(sol, data, i, t):
    delivery = sum(sol.Q.get((b, i, a), 0.0)
                   for b in data['B'] for a in data['T'] if a <= t)
    waste    = sum(sol.W.get((i, a), 0.0)
                   for a in data['T'] if a <= t - 1)
    return delivery - waste


def get_violations(sol, data, ignore_waste_gaps=False):
    out = []
    for i in data['D']:
        for t in data['T']:
            gap = sl_rhs(data, i, t) - sl_lhs(sol, data, i, t)
            if gap <= 1e-6:
                continue
            if ignore_waste_gaps:
                waste_reduction = sum(sol.W.get((i, a), 0.0)
                                      for a in data['T'] if a <= t - 1)
                if gap <= waste_reduction + 1e-4:
                    continue
            out.append((i, t, gap))
    return out


# =============================================================================
# SECTION 4 — ROUTE UTILITIES
# =============================================================================

def cover_all_depots(sol, data, period_t, randomize=False):
    D, R = data['D'], data['R']

    def min_delivery(i, t):
        rhs_t  = data['sl_rhs_cache'][(i, t)]
        rhs_t1 = data['sl_rhs_cache'].get((i, t - 1), 0.0)
        mu_t   = data['mu'][(i, t)]
        marginal = max(0.0, rhs_t - rhs_t1 - mu_t)
        return mu_t + marginal

    remaining = {
        i: max(0.0, min_delivery(i, period_t)
               - sum(sol.Z.get((i, r, period_t), 0.0) for r in R))
        for i in D
    }
    uncovered = {i for i in D if remaining[i] > 1e-6}

    if not randomize and 'optimal_routes' in data:
        for r in data['optimal_routes']:
            new_depots = uncovered & set(data['N'][r])
            if not new_depots:
                continue
            sol.X[(r, period_t)] = 1
            total_need = sum(remaining[i] for i in new_depots)
            cap        = data['c'][r]
            if total_need <= cap:
                for i in new_depots:
                    sol.Z[(i, r, period_t)] = sol.Z.get((i, r, period_t), 0) + int(math.ceil(remaining[i]))
                    remaining[i] = 0.0
                uncovered -= new_depots
            else:
                for i in new_depots:
                    share = (remaining[i] / total_need) * cap
                    sol.Z[(i, r, period_t)] = sol.Z.get((i, r, period_t), 0) + int(math.ceil(share))
                    remaining[i] = max(0.0, remaining[i] - share)
                uncovered = {i for i in D if remaining[i] > 1e-6}

    while uncovered:
        candidates = []
        for r in R:
            new_depots = uncovered & set(data['N'][r])
            if not new_depots:
                continue
            score = data['lambda'][r] / max(len(new_depots), 1)
            candidates.append((score, r))

        if not candidates:
            break

        candidates.sort()

        if randomize and len(candidates) > 1:
            top   = candidates[:min(3, len(candidates))]
            inv_w = [1.0 / max(s, 1e-9) for s, _ in top]
            total = sum(inv_w)
            probs = [w / total for w in inv_w]
            cum   = 0.0
            rnd   = random.random()
            chosen_r = top[-1][1]
            for prob, (_, r) in zip(probs, top):
                cum += prob
                if rnd <= cum:
                    chosen_r = r
                    break
        else:
            chosen_r = candidates[0][1]

        sol.X[(chosen_r, period_t)] = 1
        new_depots = uncovered & set(data['N'][chosen_r])
        total_need = sum(remaining[i] for i in new_depots)
        cap        = data['c'][chosen_r]

        if total_need <= cap:
            for i in new_depots:
                add = int(math.ceil(remaining[i]))
                sol.Z[(i, chosen_r, period_t)] = (
                    sol.Z.get((i, chosen_r, period_t), 0) + add
                )
                remaining[i] = 0.0
            uncovered -= new_depots
        else:
            for i in new_depots:
                share = int(math.ceil((remaining[i] / total_need) * cap))
                sol.Z[(i, chosen_r, period_t)] = (
                    sol.Z.get((i, chosen_r, period_t), 0) + share
                )
                remaining[i] = max(0.0, remaining[i] - share)
            uncovered = {i for i in D if remaining[i] > 1e-6}


def boost_existing_route(sol, data, i, t, amount):
    active = [r for r in data['R'] if i in data['N'][r] and sol.X.get((r, t), 0) == 1]
    if not active:
        return False
    r_best = min(active, key=lambda r: data['lambda'][r])
    sol.Z[(i, r_best, t)] = sol.Z.get((i, r_best, t), 0) + int(math.ceil(amount))
    return True


# ── FIX P4 ───────────────────────────────────────────────────────────────────
def fix_service_violations(sol, data):
    """
    Fix ALL service level violations (constraint 19) — hard constraint.

    SPEED: compute_Q_from_Z and compute_inventory_waste are now called
    ONCE per pass (batched), not after every depot-period fix.
    With 30 depots × 6 periods this eliminates ~180 redundant recomputes
    per repair call — the single biggest runtime saving.

    P4 trim-before-fix retained: prevents m=2 waste cascade.
    """
    T = data['T']
    D = data['D']
    R = data['R']

    # ── Trim-before-fix (P4) — batch, single recompute at end ────────────
    trim_changed = False
    for i in D:
        prev_rhs = 0.0
        for t in T:
            rhs_t    = data['sl_rhs_cache'][(i, t)]
            mu_t     = data['mu'][(i, t)]
            marginal = max(0.0, rhs_t - prev_rhs - mu_t)
            prev_rhs = rhs_t
            max_useful = mu_t + marginal + 2.0
            actual = sum(sol.Z.get((i, r, t), 0.0) for r in R)
            if actual <= max_useful + 1e-6:
                continue
            excess = actual - max_useful
            for r in R:
                z = sol.Z.get((i, r, t), 0.0)
                if z > 1e-6:
                    cut = min(z, excess)
                    sol.Z[(i, r, t)] = max(0.0, z - cut)
                    excess -= cut
                    trim_changed = True
                    if excess <= 1e-6:
                        break

    # Single recompute after all trim changes
    if trim_changed:
        compute_Q_from_Z(sol, data)
        compute_inventory_waste(sol, data)
    # ── End trim-before-fix ───────────────────────────────────────────────

    # Step 1: Marginal safety stock — batch all additions, recompute once
    step1_changed = False
    for i in D:
        prev_rhs = 0.0
        for t in T:
            rhs      = data['sl_rhs_cache'][(i, t)]
            mu_t     = data['mu'][(i, t)]
            marginal = max(0.0, rhs - prev_rhs - mu_t)
            prev_rhs = rhs
            if marginal < 1e-6:
                continue
            current = sum(sol.Z.get((i, r, t), 0) for r in R)
            needed  = max(0.0, marginal - max(0.0, current - mu_t))
            if needed < 1e-6:
                continue
            if not boost_existing_route(sol, data, i, t, needed):
                inactive = [r for r in R
                            if i in data['N'][r] and sol.X.get((r, t), 0) == 0]
                if inactive:
                    r_best = min(inactive, key=lambda r: data['lambda'][r])
                    sol.X[(r_best, t)] = 1
                    sol.Z[(i, r_best, t)] = (sol.Z.get((i, r_best, t), 0)
                                             + int(math.ceil(needed)))
            step1_changed = True
            # NO per-depot-period recompute here — batched below

    if step1_changed:
        compute_Q_from_Z(sol, data)
        compute_inventory_waste(sol, data)

    # Step 2: Fix ALL remaining gaps — recompute once per full pass, not per fix
    for _pass in range(len(T)):
        changed = False
        for i in D:
            for t in T:
                gap = sl_rhs(data, i, t) - sl_lhs(sol, data, i, t)
                if gap <= 1e-6:
                    continue
                changed = True
                if not boost_existing_route(sol, data, i, t, gap):
                    inactive = [r for r in R
                                if i in data['N'][r] and sol.X.get((r, t), 0) == 0]
                    if inactive:
                        r_best = min(inactive, key=lambda r: data['lambda'][r])
                        sol.X[(r_best, t)] = 1
                        sol.Z[(i, r_best, t)] = (sol.Z.get((i, r_best, t), 0)
                                                 + int(math.ceil(gap)))
                # NO per-fix recompute — batched at end of pass
        if changed:
            compute_Q_from_Z(sol, data)
            compute_inventory_waste(sol, data)
        else:
            break

    compute_Q_from_Z(sol, data)


def _assign_nonowners_optimal(F, K, B, data, hub_need, remaining_cap, s, t, sol):
    """
    Assign non-owner suppliers to vehicles to cover hub_need at minimum total cost.

    Total Echelon-1 cost = sum over assigned (f,k,b,s,t) of:
        pi_s * [ beta[f,k] * Y[f,b,k,s,t] + gamma[f,b,k] * L[f,b,k,s,t] ]

    The key insight from the MIP gap: beta is a ONE-TIME fixed cost per
    (f,k) assignment.  Once a vehicle k is already assigned to supplier f,
    additional load onto that same (f,k) pair costs only gamma per crate —
    no extra beta.  Greedy must exploit this by:
      1. Maximising load on already-committed (f,k) pairs first (free beta).
      2. When a new assignment is needed, choosing the (f,k,b) triple with
         the lowest MARGINAL cost = beta[f,k]*pr_s + gamma[f,b,k]*amount.

    This directly matches MIP behaviour and targets the 310-unit gap.
    """
    owned  = {data['owner'][k]: k for k in K}
    ups    = data['upsilon']
    beta   = data['beta']
    gamma  = data['gamma']
    theta  = data['theta']
    pr_s   = data['pr'].get(s, 1.0)

    # Non-owners available this (s,t)
    non_owners = {f: ups.get((f, s, t), 0.0)
                  for f in F if f not in owned and ups.get((f, s, t), 0.0) > 0}
    if not non_owners:
        return

    used_supply = {f: 0.0 for f in non_owners}

    # Which (f,k) assignments are already active this (s,t)?
    # Adding load to these costs 0 extra beta.
    committed_fk = {}   # (f,k) -> hub b already assigned
    for f in non_owners:
        for k in K:
            for b in B:
                if sol.Y.get((f, b, k, s, t), 0) == 1:
                    committed_fk[(f, k)] = b
                    used_supply[f] += sol.L.get((f, b, k, s, t), 0)

    # Iterate: each pass makes the single best assignment
    max_passes = len(non_owners) * len(K) + 1
    for _pass in range(max_passes):
        if not any(hub_need[b] > 1e-6 for b in B):
            break

        best_cost   = float('inf')
        best_assign = None   # (f, k, b, amount, marginal_beta)

        for f, total_sup in non_owners.items():
            avail = total_sup - used_supply[f]
            if avail < 0.5:
                continue

            for k in K:
                cap = remaining_cap.get(k, 0)
                if cap < 0.5:
                    continue

                # Is this (f,k) already committed to a hub?
                committed_b = committed_fk.get((f, k))

                for b in B:
                    # If already committed to a different hub, skip
                    if committed_b is not None and committed_b != b:
                        continue
                    need = hub_need[b]
                    if need < 0.5:
                        continue
                    amount = min(avail, cap, need)
                    if amount < 0.5:
                        continue

                    # Marginal cost: beta only if this is a NEW (f,k) assignment
                    marginal_beta = 0.0 if committed_b == b else beta[(f, k)] * pr_s
                    marginal_gamma = gamma[(f, b, k)] * amount * pr_s
                    total_marginal = marginal_beta + marginal_gamma

                    # Normalise by amount to get cost-per-crate for comparison
                    cost_per_crate = total_marginal / max(amount, 1e-9)

                    if cost_per_crate < best_cost:
                        best_cost   = cost_per_crate
                        best_assign = (f, k, b, amount, marginal_beta > 0)

        if best_assign is None:
            break

        f, k, b, amount, new_assignment = best_assign
        amount = int(round(amount))
        if amount <= 0:
            break

        existing = sol.L.get((f, b, k, s, t), 0)
        sol.Y[(f, b, k, s, t)]  = 1
        sol.L[(f, b, k, s, t)]  = existing + amount
        used_supply[f]          += amount
        remaining_cap[k]        -= amount
        hub_need[b]             -= amount
        if new_assignment:
            committed_fk[(f, k)] = b


def greedy_initialize(data):
    F, K, B, D, R, T, S = (data['F'], data['K'], data['B'],
                             data['D'], data['R'], data['T'], data['S'])
    sol = Solution()

    for f in F:
        for b in B:
            for k in K:
                for s in S:
                    for t in T:
                        sol.Y[(f, b, k, s, t)] = 0
                        sol.L[(f, b, k, s, t)] = 0
    for r in R:
        for t in T:
            sol.X[(r, t)] = 0
            for i in D:
                sol.Z[(i, r, t)] = 0
    for b in B:
        for i in D:
            for t in T:
                sol.Q[(b, i, t)] = 0
    for i in D:
        for t in T:
            sol.I[(i, t)]      = 0.0
            sol.I_plus[(i, t)] = 0.0
            sol.W[(i, t)]      = 0.0

    owned_vehicle = {data['owner'][k]: k for k in data['K']}

    for t in T:
        cover_all_depots(sol, data, t)
    compute_Q_from_Z(sol, data)

    for s in S:
        for t in T:
            remaining_cap = {k: data['theta'][k] for k in K}
            hub_need = {b: sum(sol.Q.get((b, i, t), 0.0) for i in D) for b in B}

            for f in F:
                if f not in owned_vehicle:
                    continue
                k      = owned_vehicle[f]
                supply = data['upsilon'].get((f, s, t), 0.0)
                if supply <= 0 or remaining_cap[k] < 1e-6:
                    continue
                for best_b in sorted(B, key=lambda b: data['gamma'][(f, b, k)]):
                    need = hub_need[best_b]
                    if need <= 1e-6:
                        continue
                    amount = min(supply, need, remaining_cap[k])
                    if amount <= 1e-6:
                        continue
                    sol.Y[(f, best_b, k, s, t)] = 1
                    sol.L[(f, best_b, k, s, t)] = int(round(amount))
                    remaining_cap[k]            -= amount
                    hub_need[best_b]            -= amount
                    break

            if any(need > 1e-6 for need in hub_need.values()):
                _assign_nonowners_optimal(F, K, B, data, hub_need,
                                          remaining_cap, s, t, sol)

    for s in S:
        for t in T:
            for b in B:
                q_target = sum(sol.Q.get((b, i, t), 0) for i in D)
                l_total  = sum(sol.L.get((f, b, k, s, t), 0)
                               for f in F for k in K)
                diff = q_target - l_total
                if abs(diff) < 1:
                    continue
                for f in F:
                    for k in K:
                        cur = sol.L.get((f, b, k, s, t), 0)
                        if cur > 0:
                            sol.L[(f, b, k, s, t)] = max(0, cur + diff)
                            diff = 0
                            break
                    if diff == 0:
                        break

    compute_inventory_waste(sol, data)
    fix_service_violations(sol, data)
    route_swap_local_search(sol, data)

    for s in S:
        for t in T:
            remaining_cap = {k: data['theta'][k] for k in K}
            for f in F:
                for b in B:
                    for k in K:
                        remaining_cap[k] = max(0.0,
                            remaining_cap[k] - sol.L.get((f, b, k, s, t), 0.0))

            hub_need = {b: max(0.0,
                sum(sol.Q.get((b, i, t), 0) for i in D)
                - sum(sol.L.get((f, b, k, s, t), 0.0) for f in F for k in K))
                for b in B}

            if not any(need > 1e-6 for need in hub_need.values()):
                continue

            for f in F:
                if f not in owned_vehicle:
                    continue
                k      = owned_vehicle[f]
                supply = max(0.0, data['upsilon'].get((f, s, t), 0.0)
                             - sum(sol.L.get((f, b, k, s, t), 0.0) for b in B))
                if supply <= 0 or remaining_cap[k] < 1e-6:
                    continue
                for best_b in sorted(B, key=lambda b: data['gamma'][(f, b, k)]):
                    need = hub_need[best_b]
                    if need <= 1e-6:
                        continue
                    amount = min(supply, need, remaining_cap[k])
                    if amount <= 1e-6:
                        continue
                    sol.Y[(f, best_b, k, s, t)] = 1
                    sol.L[(f, best_b, k, s, t)] = (
                        sol.L.get((f, best_b, k, s, t), 0) + int(round(amount)))
                    remaining_cap[k]  -= amount
                    hub_need[best_b]  -= amount
                    break

            if any(need > 1e-6 for need in hub_need.values()):
                _assign_nonowners_optimal(F, K, B, data, hub_need,
                                          remaining_cap, s, t, sol)

    compute_inventory_waste(sol, data)
    compute_objective(sol, data)

    return sol


# =============================================================================
# SECTION 6 — DESTROY OPERATORS
# =============================================================================

def _removal_size(data):
    n = len(data['F'])
    return random.randint(max(2, int(0.05 * n)), max(5, int(0.15 * n)))


def destroy_random(sol, data):
    sol_new = sol.copy()
    active = [(f, b, k, s, t)
              for f in data['F'] for b in data['B'] for k in data['K']
              for s in data['S'] for t in data['T']
              if sol_new.Y.get((f, b, k, s, t), 0) == 1]

    q       = min(_removal_size(data), len(active))
    removed = random.sample(active, q)
    for key in removed:
        sol_new.Y[key] = 0
        sol_new.L[key] = 0.0
    return sol_new, removed


def destroy_worst_cost(sol, data):
    sol_new = sol.copy()
    scored = []
    for f in data['F']:
        for b in data['B']:
            for k in data['K']:
                for s in data['S']:
                    for t in data['T']:
                        if sol_new.Y.get((f, b, k, s, t), 0) == 1:
                            cost = (data['beta'][(f, k)]
                                    + data['gamma'][(f, b, k)]
                                    * sol_new.L.get((f, b, k, s, t), 0.0))
                            scored.append(((f, b, k, s, t), cost))

    scored.sort(key=lambda x: x[1], reverse=True)
    q       = min(_removal_size(data), len(scored))
    removed = [item[0] for item in scored[:q]]
    for key in removed:
        sol_new.Y[key] = 0
        sol_new.L[key] = 0.0
    return sol_new, removed


def destroy_route(sol, data):
    sol_new     = sol.copy()
    active_rt   = [(r, t) for r in data['R'] for t in data['T']
                   if sol_new.X.get((r, t), 0) == 1]

    q         = max(1, min(int(0.25 * len(active_rt)), len(active_rt)))
    to_remove = random.sample(active_rt, q)

    for (r, t) in to_remove:
        sol_new.X[(r, t)] = 0
        for i in data['N'][r]:
            sol_new.Z[(i, r, t)] = 0.0

    compute_Q_from_Z(sol_new, data)
    return sol_new, []


def destroy_waste_driven(sol, data):
    sol_new = sol.copy()
    m       = data['m']

    waste_scores = sorted(
        [((i, t), sol_new.W.get((i, t), 0.0))
         for i in data['D'] for t in data['T']
         if t >= m and sol_new.W.get((i, t), 0.0) > 1e-6],
        key=lambda x: x[1], reverse=True
    )

    q = min(max(1, _removal_size(data) // 5), len(waste_scores))
    for ((i, t), w) in waste_scores[:q]:
        cause_t = t - m + 1
        if cause_t in data['T']:
            for r in data['R']:
                if i in data['N'][r]:
                    sol_new.Z[(i, r, cause_t)] = max(
                        0.0, sol_new.Z.get((i, r, cause_t), 0.0) - w
                    )

    compute_Q_from_Z(sol_new, data)
    return sol_new, []


# ── FIX P7: removal size raised from //5 to //2 ──────────────────────────────
def destroy_inventory_driven(sol, data):
    """D5 — Inventory-driven removal.
    FIX P7: removal size raised from _removal_size()//5 to //2.
    Removing only 1–2 items produced near-identical solutions that the
    repair immediately reversed.  Larger removals create genuine diversity.
    """
    sol_new = sol.copy()
    T, D, R = data['T'], data['D'], data['R']

    scores = []
    for i in D:
        for t in T:
            rhs_t   = data['sl_rhs_cache'][(i, t)]
            rhs_t1  = data['sl_rhs_cache'].get((i, t - 1), 0.0)
            mu_t    = data['mu'][(i, t)]
            min_del = mu_t + max(0.0, rhs_t - rhs_t1 - mu_t)
            actual  = sum(sol_new.Z.get((i, r, t), 0.0) for r in R)
            excess  = actual - min_del
            if excess > 1e-6:
                scores.append(((i, t), excess))

    scores.sort(key=lambda x: x[1], reverse=True)
    # FIX P7: was _removal_size(data) // 5  →  now // 2
    q = min(max(3, _removal_size(data) // 2), len(scores))

    for ((i, t), excess) in scores[:q]:
        for r in R:
            z = sol_new.Z.get((i, r, t), 0.0)
            if z > 1e-6:
                reduce = min(z, excess)
                sol_new.Z[(i, r, t)] = z - reduce
                excess -= reduce
                if excess <= 1e-6:
                    break

    compute_Q_from_Z(sol_new, data)
    return sol_new, []


# =============================================================================
# SECTION 7 — REPAIR OPERATORS
# =============================================================================

def _get_remaining_cap(sol, data, s, t):
    remaining = {k: data['theta'][k] for k in data['K']}
    for f in data['F']:
        for b in data['B']:
            for k in data['K']:
                remaining[k] -= sol.L.get((f, b, k, s, t), 0.0)
    return remaining


# ── FIX P3: hub-driven repair ─────────────────────────────────────────────────
def _reassign_greedy(sol, data, by_st):
    """
    Demand-driven greedy reassignment.

    FIX P3: Uses per-hub targets derived from current sol.Q instead of
    aggregate depot demand.  The previous version computed a single
    remaining_need = total_demand - already_transported which ignored
    which hub needed supply, breaking flow balance constraint (5) silently.
    """
    owned_vehicle = {data['owner'][k]: k for k in data['K']}

    for (s, t), unserved in by_st.items():
        remaining_cap = _get_remaining_cap(sol, data, s, t)

        # Per-hub targets: how much each hub still needs after current L
        hub_need = {
            b: max(0.0,
                   sum(sol.Q.get((b, i, t), 0.0) for i in data['D'])
                   - sum(sol.L.get((f, b, k, s, t), 0.0)
                         for f in data['F'] for k in data['K']))
            for b in data['B']
        }

        # Pass 1: owner suppliers in unserved — assign to highest-need hub
        for f in [f for f in unserved if f in owned_vehicle]:
            if not any(hub_need[b] > 1e-6 for b in data['B']):
                break
            k      = owned_vehicle[f]
            supply = data['upsilon'].get((f, s, t), 0.0)
            if supply <= 0 or remaining_cap.get(k, 0) < 1e-6:
                continue
            # Pick hub with highest need and lowest gamma (owner goes to cheapest route)
            eligible = [b for b in data['B'] if hub_need[b] > 1e-6]
            if not eligible:
                continue
            best_b = min(eligible, key=lambda b: data['gamma'][(f, b, k)])
            amount = min(supply, hub_need[best_b], remaining_cap[k])
            if amount <= 1e-6:
                continue
            sol.Y[(f, best_b, k, s, t)] = 1
            sol.L[(f, best_b, k, s, t)] = int(round(amount))
            remaining_cap[k]           -= amount
            hub_need[best_b]           -= amount

        # Pass 2: non-owners — use optimised consolidation to minimise beta payments
        if any(hub_need[b] > 1e-6 for b in data['B']):
            _assign_nonowners_optimal(data['F'], data['K'], data['B'], data,
                                      hub_need, remaining_cap, s, t, sol)


# ── FIX P3: same hub-target logic for regret repair ──────────────────────────
def _reassign_regret(sol, data, by_st):
    """Regret-based reassignment.
    FIX P3: uses per-hub capacity targets, same as _reassign_greedy.
    """
    for (s, t), unserved in by_st.items():
        remaining_cap = _get_remaining_cap(sol, data, s, t)

        # Per-hub targets
        hub_need = {
            b: max(0.0,
                   sum(sol.Q.get((b, i, t), 0.0) for i in data['D'])
                   - sum(sol.L.get((f, b, k, s, t), 0.0)
                         for f in data['F'] for k in data['K']))
            for b in data['B']
        }

        regrets = []
        for f in unserved:
            supply = data['upsilon'].get((f, s, t), 0.0)
            if supply <= 0:
                continue
            options = []
            for k in data['K']:
                if remaining_cap[k] < supply:
                    continue
                for b in data['B']:
                    if hub_need[b] <= 1e-6:
                        continue
                    cost = data['beta'][(f, k)] + data['gamma'][(f, b, k)] * supply
                    options.append((cost, k, b))
            options.sort()
            if not options:
                continue
            best   = options[0][0]
            second = options[1][0] if len(options) > 1 else best * 2.0
            regrets.append((second - best, f, options[0][1], options[0][2], supply))

        for (_, f, best_k, best_b, supply) in sorted(regrets, reverse=True):
            if hub_need.get(best_b, 0) <= 1e-6:
                continue
            if remaining_cap.get(best_k, 0) >= supply:
                sol.Y[(f, best_b, best_k, s, t)] = 1
                sol.L[(f, best_b, best_k, s, t)] = int(round(supply))
                remaining_cap[best_k]            -= supply
                hub_need[best_b]                 -= supply
            else:
                for k in data['K']:
                    if remaining_cap[k] >= supply:
                        for b in data['B']:
                            if hub_need.get(b, 0) <= 1e-6:
                                continue
                            sol.Y[(f, b, k, s, t)] = 1
                            sol.L[(f, b, k, s, t)] = int(round(supply))
                            remaining_cap[k]       -= supply
                            hub_need[b]            -= supply
                            break
                        break


def route_swap_local_search(sol, data):
    """
    SPEED: precompute N_sets and cheaper-route index once per call.
    Inner loop now does set lookups (O(1)) instead of list scans.
    """
    R, T = data['R'], data['T']
    lam  = data['lambda']
    cap  = data['c']

    # Precompute once per call
    N_sets = {r: set(data['N'][r]) for r in R}
    # For each route r_out, find all cheaper routes with superset depots
    cheaper = {}
    for r_out in R:
        cheaper[r_out] = [
            r2 for r2 in R
            if r2 != r_out
            and lam[r2] < lam[r_out]
            and N_sets[r_out].issubset(N_sets[r2])
            and cap[r2] >= cap[r_out]
        ]

    improved = True
    while improved:
        improved = False
        for t in T:
            active = [r for r in R if sol.X.get((r, t), 0) == 1]
            for r_out in active:
                candidates = [r2 for r2 in cheaper[r_out]
                              if sol.X.get((r2, t), 0) == 0]
                if not candidates:
                    continue
                r_in = min(candidates, key=lambda r: lam[r])
                sol.X[(r_out, t)] = 0
                sol.X[(r_in,  t)] = 1
                for i in N_sets[r_out]:
                    z_old = sol.Z.get((i, r_out, t), 0.0)
                    if z_old > 0:
                        sol.Z[(i, r_in,  t)] = sol.Z.get((i, r_in, t), 0) + int(round(z_old))
                        sol.Z[(i, r_out, t)] = 0
                compute_Q_from_Z(sol, data)
                improved = True
                break
            if improved:
                break


def _enforce_flow_balance(sol, data):
    F, K, B, S, T = data['F'], data['K'], data['B'], data['S'], data['T']
    owned_v  = {data['owner'][k]: k for k in K}
    beta_d   = data['beta']
    ups      = data['upsilon']
    theta    = data['theta']

    active_L_all = {key: v for key, v in sol.L.items() if v > 0}
    active_Y_all = {key: v for key, v in sol.Y.items() if v > 0}

    for s in S:
        for t in T:
            sup_used = {}
            veh_used = {}
            fk_hub   = {}
            l_by_b   = {}

            for (f, b, k, _s, _t), l in active_L_all.items():
                if _s != s or _t != t:
                    continue
                sup_used[f] = sup_used.get(f, 0) + l
                veh_used[k] = veh_used.get(k, 0) + l
                l_by_b[b]   = l_by_b.get(b,   0) + l

            for (f, b, k, _s, _t), y in active_Y_all.items():
                if _s != s or _t != t:
                    continue
                if (f, k) not in fk_hub:
                    fk_hub[(f, k)] = b
                elif fk_hub[(f, k)] != b:
                    l_keep = sol.L.get((f, fk_hub[(f,k)], k, s, t), 0)
                    l_drop = sol.L.get((f, b, k, s, t), 0)
                    if l_drop >= l_keep:
                        sol.Y[(f, fk_hub[(f,k)], k, s, t)] = 0
                        sol.L[(f, fk_hub[(f,k)], k, s, t)] = 0
                        sup_used[f]  = max(0, sup_used.get(f, 0) - l_keep)
                        veh_used[k]  = max(0, veh_used.get(k, 0) - l_keep)
                        l_by_b[fk_hub[(f,k)]] = max(0, l_by_b.get(fk_hub[(f,k)],0) - l_keep)
                        fk_hub[(f, k)] = b
                    else:
                        sol.Y[(f, b, k, s, t)] = 0
                        sol.L[(f, b, k, s, t)] = 0
                        sup_used[f]  = max(0, sup_used.get(f, 0) - l_drop)
                        veh_used[k]  = max(0, veh_used.get(k, 0) - l_drop)
                        l_by_b[b]    = max(0, l_by_b.get(b, 0) - l_drop)

            active_by_hub = {}
            for (f2, b2, k2, _s2, _t2), lv in active_L_all.items():
                if _s2 != s or _t2 != t: continue
                if b2 not in active_by_hub: active_by_hub[b2] = []
                active_by_hub[b2].append((f2, k2, lv))

            for b in B:
                q_tgt   = int(sum(sol.Q.get((b, i, t), 0) for i in data['D']))
                l_total = l_by_b.get(b, 0)
                diff    = l_total - q_tgt

                if abs(diff) < 1:
                    continue

                if diff > 0:
                    active_bst = sorted(active_by_hub.get(b, []),
                                        key=lambda x: -x[2])
                    for f, k, cur in active_bst:
                        if diff <= 0: break
                        trim = min(cur, diff)
                        sol.L[(f, b, k, s, t)] -= trim
                        sup_used[f] = max(0, sup_used.get(f, 0) - trim)
                        veh_used[k] = max(0, veh_used.get(k, 0) - trim)
                        if sol.L[(f, b, k, s, t)] == 0:
                            sol.Y[(f, b, k, s, t)] = 0
                            fk_hub.pop((f, k), None)
                        diff -= trim

                else:
                    deficit = -diff
                    for f in data.get('_f_by_beta', sorted(F,
                            key=lambda f: min(beta_d.get((f,k2),999) for k2 in K))):
                        if deficit <= 0: break
                        sup_avail = int(ups.get((f, s, t), 0)) - sup_used.get(f, 0)
                        if sup_avail <= 0: continue

                        k_own = owned_v.get(f)
                        if k_own is not None:
                            existing = fk_hub.get((f, k_own))
                            if existing is not None and existing != b: continue
                            k_use = k_own
                        else:
                            k_best = data.get('_nonowner_best_k', {}).get(f)
                            if k_best is not None:
                                eh = fk_hub.get((f, k_best))
                                va = theta[k_best] - veh_used.get(k_best, 0)
                                if (eh is None or eh == b) and va > 0:
                                    k_use = k_best
                                else:
                                    k_use = None
                                    for k2 in K:
                                        eh2 = fk_hub.get((f, k2))
                                        if eh2 is not None and eh2 != b: continue
                                        va2 = theta[k2] - veh_used.get(k2, 0)
                                        if va2 > 0:
                                            k_use = k2; break
                            else:
                                k_use = None
                            if k_use is None: continue

                        va = theta[k_use] - veh_used.get(k_use, 0)
                        if va <= 0: continue
                        room = min(sup_avail, va, deficit)
                        if room <= 0: continue

                        already = sol.L.get((f, b, k_use, s, t), 0)
                        sol.Y[(f, b, k_use, s, t)] = 1
                        sol.L[(f, b, k_use, s, t)] = already + room
                        sup_used[f]     = sup_used.get(f, 0)   + room
                        veh_used[k_use] = veh_used.get(k_use, 0) + room
                        fk_hub[(f, k_use)] = b
                        deficit -= room
                    active_L_all = {key: v for key, v in sol.L.items() if v > 0}


def _drop_empty_routes(sol, data):
    """Remove routes with zero total load — free cost reduction, no SL risk."""
    for r in data['R']:
        for t in data['T']:
            if sol.X.get((r, t), 0) == 1:
                if sum(sol.Z.get((i, r, t), 0) for i in data['D']) == 0:
                    sol.X[(r, t)] = 0


def _finalise(sol, data, rebuild_routes=False, randomize=False, do_route_swap=False):
    """
    SPEED: route_swap_local_search is now opt-in (do_route_swap=False by default).
    During the main ALNS loop it is skipped entirely — it is only run
    every SWAP_INTERVAL iterations from run_alns, and always at the end.
    _drop_empty_routes runs on every repair (near-zero cost) to keep the
    search objective aligned with the true objective.
    """
    if rebuild_routes:
        for t in data['T']:
            cover_all_depots(sol, data, t, randomize=randomize)
        compute_Q_from_Z(sol, data)

    compute_inventory_waste(sol, data)
    fix_service_violations(sol, data)
    _drop_empty_routes(sol, data)
    if do_route_swap:
        route_swap_local_search(sol, data)
    compute_inventory_waste(sol, data)
    compute_objective(sol, data)


def repair_greedy(sol, data, removed, randomize=False):
    sol_new  = sol.copy()
    by_st    = {}
    for item in removed:
        if len(item) == 5:
            (f, b, k, s, t) = item
            by_st.setdefault((s, t), set()).add(f)

    _reassign_greedy(sol_new, data, by_st)
    _finalise(sol_new, data, rebuild_routes=(not by_st), randomize=randomize)
    return sol_new


def repair_regret(sol, data, removed, randomize=False):
    sol_new = sol.copy()
    by_st   = {}
    for item in removed:
        if len(item) == 5:
            (f, b, k, s, t) = item
            by_st.setdefault((s, t), set()).add(f)

    _reassign_regret(sol_new, data, by_st)
    _finalise(sol_new, data, rebuild_routes=(not by_st), randomize=randomize)
    return sol_new


def repair_service_level(sol, data, removed, randomize=False):
    sol_new = sol.copy()
    by_st   = {}
    for item in removed:
        if len(item) == 5:
            (f, b, k, s, t) = item
            by_st.setdefault((s, t), set()).add(f)

    _reassign_greedy(sol_new, data, by_st)

    if not by_st:
        for t in data['T']:
            cover_all_depots(sol_new, data, t, randomize=randomize)
        compute_Q_from_Z(sol_new, data)

    compute_inventory_waste(sol_new, data)

    violations = sorted(get_violations(sol_new, data), key=lambda x: x[2], reverse=True)
    for (i, t, gap) in violations:
        if not boost_existing_route(sol_new, data, i, t, gap):
            inactive = [r for r in data['R']
                        if i in data['N'][r] and sol_new.X.get((r, t), 0) == 0]
            if inactive:
                r_best = min(inactive, key=lambda r: data['lambda'][r])
                sol_new.X[(r_best, t)] = 1
                sol_new.Z[(i, r_best, t)] = sol_new.Z.get((i, r_best, t), 0.0) + gap

    compute_Q_from_Z(sol_new, data)
    compute_inventory_waste(sol_new, data)
    compute_objective(sol_new, data)
    return sol_new


# =============================================================================
# SECTION 8 — ROULETTE WHEEL + WEIGHT UPDATE
# =============================================================================

def roulette_select(weights):
    total = sum(weights)
    r     = random.uniform(0, total)
    cum   = 0.0
    for i, w in enumerate(weights):
        cum += w
        if r <= cum:
            return i
    return len(weights) - 1


# ── FIX P8: lam default raised to 0.3 in run_alns; update_weights unchanged ──
def update_weights(weights, scores, usage, lam=0.3):
    for i in range(len(weights)):
        if usage[i] > 0:
            avg        = scores[i] / usage[i]
            weights[i] = (1 - lam) * weights[i] + lam * avg
            weights[i] = max(weights[i], 0.01)
    return weights


# =============================================================================
# SECTION 8b — PERTURBATION
# =============================================================================

def perturb(sol, data):
    """
    Double-bridge perturbation: simultaneously destroys 30% of supplier
    assignments AND 30% of active routes, then rebuilds fully.
    """
    sol_new = sol.copy()
    F, K, B, D, R, T, S = (data['F'], data['K'], data['B'],
                             data['D'], data['R'], data['T'], data['S'])

    active = [(f, b, k, s, t)
              for f in F for b in B for k in K for s in S for t in T
              if sol_new.Y.get((f, b, k, s, t), 0) == 1]
    q = max(1, int(0.30 * len(active)))
    for key in random.sample(active, min(q, len(active))):
        sol_new.Y[key] = 0
        sol_new.L[key] = 0.0

    active_rt = [(r, t) for r in R for t in T if sol_new.X.get((r, t), 0) == 1]
    q_r = max(1, int(0.30 * len(active_rt)))
    for (r, t) in random.sample(active_rt, min(q_r, len(active_rt))):
        sol_new.X[(r, t)] = 0
        for i in data['N'][r]:
            sol_new.Z[(i, r, t)] = 0.0

    compute_Q_from_Z(sol_new, data)
    for t in T:
        cover_all_depots(sol_new, data, t, randomize=True)
    compute_Q_from_Z(sol_new, data)

    owned_vehicle = {data['owner'][k]: k for k in data['K']}
    for s in S:
        for t in T:
            remaining_cap = {k: data['theta'][k] for k in K}
            hub_need = {b: sum(sol_new.Q.get((b, i, t), 0.0) for i in D) for b in B}

            for f in F:
                if f not in owned_vehicle:
                    continue
                for b in B:
                    for k_old in K:
                        if sol_new.Y.get((f, b, k_old, s, t), 0) == 1:
                            sol_new.Y[(f, b, k_old, s, t)] = 0
                            sol_new.L[(f, b, k_old, s, t)] = 0.0
                            remaining_cap[k_old] += sol_new.L.get((f, b, k_old, s, t), 0.0)

                k      = owned_vehicle[f]
                supply = data['upsilon'].get((f, s, t), 0.0)
                if supply <= 0 or remaining_cap[k] < 1e-6:
                    continue
                for best_b in sorted(B, key=lambda b: data['gamma'][(f, b, k)]):
                    need = hub_need[best_b]
                    if need <= 1e-6:
                        continue
                    amount = min(supply, need, remaining_cap[k])
                    if amount <= 1e-6:
                        continue
                    sol_new.Y[(f, best_b, k, s, t)] = 1
                    sol_new.L[(f, best_b, k, s, t)] = amount
                    remaining_cap[k]                -= amount
                    hub_need[best_b]                -= amount
                    break

            if any(need > 1e-6 for need in hub_need.values()):
                non_owners = [f for f in F if f not in owned_vehicle
                              and data['upsilon'].get((f, s, t), 0.0) > 0]
                random.shuffle(non_owners)
                for f in non_owners:
                    if not any(hub_need[b] > 1e-6 for b in B):
                        break
                    supply = data['upsilon'].get((f, s, t), 0.0)
                    if supply <= 0:
                        continue
                    best_cost, best_k, best_b = float('inf'), None, None
                    for k in K:
                        if remaining_cap[k] < 1e-6:
                            continue
                        for b in B:
                            if hub_need[b] <= 1e-6:
                                continue
                            amount = int(round(min(supply, hub_need[b], remaining_cap[k])))
                            if amount <= 1e-6:
                                continue
                            cost = data['beta'][(f, k)] + data['gamma'][(f, b, k)] * amount
                            if cost < best_cost:
                                best_cost, best_k, best_b = cost, k, b
                    if best_k:
                        amount = int(round(min(supply, hub_need[best_b], remaining_cap[best_k])))
                        sol_new.Y[(f, best_b, best_k, s, t)] = 1
                        sol_new.L[(f, best_b, best_k, s, t)] = amount
                        remaining_cap[best_k]                -= amount
                        hub_need[best_b]                     -= amount

    compute_inventory_waste(sol_new, data)
    fix_service_violations(sol_new, data)
    compute_inventory_waste(sol_new, data)
    compute_objective(sol_new, data)
    return sol_new


# =============================================================================
# SECTION 9 — MAIN ALNS LOOP
# =============================================================================

def run_alns(data, max_iter=15000, segment=200, lam=0.3, verbose=True):
    """
    Run ALNS with Simulated Annealing acceptance criterion.

    SPEED OPTIMIZATIONS (this version):
    ─────────────────────────────────────
    S1  max_iter reduced 30 000 → 15 000.  With faster iterations the same
        wall-clock budget now covers more search.  Target: ~8–10 min.
    S2  route_swap_local_search removed from every repair (_finalise now
        takes do_route_swap flag).  It runs only every SWAP_INTERVAL iters
        and always before round_solution at the end.
    S3  fix_service_violations now batches all Z changes within a pass and
        recomputes Q/inventory ONCE per pass, not after every depot-period.
        Eliminates ~180 redundant recomputes per repair call.
    S4  route_swap_local_search precomputes the cheaper-route index once
        per call instead of scanning all R² pairs on every swap attempt.

    All quality fixes (P1–P9) from the previous version are retained.

    Parameters
    ----------
    data     : loaded problem data
    max_iter : total SA iterations (default 15 000, ~8–10 min)
    segment  : weight update frequency  (default 200)
    lam      : reaction factor for weight update  (default 0.3)
    verbose  : print progress every segment
    """
    SIGMA_1 = 10
    SIGMA_2 = 5
    SIGMA_3 = 1

    # SPEED S2: route_swap runs every SWAP_INTERVAL iterations only
    SWAP_INTERVAL = 500

    destroy_ops = [destroy_random, destroy_worst_cost,
                   destroy_route,  destroy_waste_driven,
                   destroy_inventory_driven]
    repair_ops  = [repair_greedy, repair_regret, repair_service_level]
    d_names     = ['D1-random', 'D2-worst', 'D3-route', 'D4-waste', 'D5-inventory']
    r_names     = ['R1-greedy', 'R2-regret', 'R3-service']

    n_d, n_r   = len(destroy_ops), len(repair_ops)
    w_d, w_r   = [1.0] * n_d, [1.0] * n_r
    sc_d, sc_r = [0.0] * n_d, [0.0] * n_r
    us_d, us_r = [0]   * n_d, [0]   * n_r

    # Phase 1: Greedy initial solution
    t_start = time.time()
    print("\nPhase 1: Building initial solution...")
    x      = greedy_initialize(data)
    x_best = x.copy()
    t_init = time.time() - t_start
    print(f"  Objective      : {x.obj:.4f}")
    print(f"  Active routes  : {sum(x.X.get((r,t),0) for r in data['R'] for t in data['T'])}")
    print(f"  SL violations  : {len(get_violations(x, data, ignore_waste_gaps=False))}")
    print(f"  Init time      : {t_init:.1f}s\n")

    # SA calibration
    delta_init = 0.02 * max(x.obj, 1.0)
    T0         = -delta_init / math.log(0.8)
    T_min      = 0.0001 * max(x.obj, 1.0)
    cooling    = (T_min / max(T0, 1e-12)) ** (1.0 / max(max_iter, 1))
    temp       = T0
    print(f"  SA T0={T0:.4f}  T_min={T_min:.6f}  cooling={cooling:.8f}")
    print(f"  route_swap every {SWAP_INTERVAL} iterations\n")

    no_improve       = 0
    no_improve_limit = max(500, max_iter // 15)
    print(f"  Perturbation trigger: {no_improve_limit} stagnant iterations\n")

    # Main ALNS loop
    print("Phase 2: ALNS search...")
    t_search_start = time.time()

    # Diagnostic counters — printed at iter 50 to catch stagnation early
    _diag_new_best   = 0
    _diag_accepted   = 0
    _diag_rejected   = 0
    _diag_deltas     = []   # obj differences of all x_new vs x
    for it in range(1, max_iter + 1):

        # Perturbation restart if stagnant
        if no_improve >= no_improve_limit:
            x_perturbed = perturb(x_best, data)
            x           = x_perturbed.copy()
            no_improve  = 0
            temp = T0 * ((cooling ** it) ** 0.5)
            if verbose:
                print(f"  ** Perturbation at iter {it} | new obj={x.obj:.4f} | temp={temp:.4f} **")

        # Select operators
        d_idx = roulette_select(w_d)
        r_idx = roulette_select(w_r)

        # Destroy
        x_partial, removed = destroy_ops[d_idx](x, data)

        # Repair
        # SPEED S2: do_route_swap=True only every SWAP_INTERVAL iterations
        use_random   = (d_idx in [2, 3, 4])
        do_swap_now  = (it % SWAP_INTERVAL == 0)
        x_new = repair_ops[r_idx](x_partial, data, removed,
                                   randomize=use_random)
        # Apply route_swap periodically to the repaired solution
        if do_swap_now:
            route_swap_local_search(x_new, data)
            compute_inventory_waste(x_new, data)
            compute_objective(x_new, data)

        # SA acceptance
        score = 0

        delta = x_new.obj - x_best.obj
        if delta < -1e-6:
            x_best     = x_new.copy()
            score      = SIGMA_1
            no_improve = 0
            _diag_new_best += 1
        else:
            no_improve += 1

        delta_curr = x_new.obj - x.obj
        _diag_deltas.append(delta_curr)
        if delta_curr < -1e-6:
            x     = x_new.copy()
            score = max(score, SIGMA_2)
            _diag_accepted += 1
        elif random.random() < math.exp(-delta_curr / max(temp, 1e-12)):
            x     = x_new.copy()
            score = max(score, SIGMA_3)
            _diag_accepted += 1
        else:
            _diag_rejected += 1

        temp *= cooling

        # ── Early diagnostic at iteration 50 ────────────────────────────
        if it == 50:
            avg_delta  = sum(_diag_deltas) / len(_diag_deltas)
            zero_moves = sum(1 for d in _diag_deltas if abs(d) < 1e-4)
            print(f"\n  ── DIAGNOSTIC (first 50 iters) ──")
            print(f"  New best found   : {_diag_new_best}")
            print(f"  Accepted moves   : {_diag_accepted}")
            print(f"  Rejected moves   : {_diag_rejected}")
            print(f"  Zero-delta moves : {zero_moves}  "
                  f"({'STUCK — repair returns same solution' if zero_moves > 40 else 'OK'})")
            print(f"  Avg obj delta    : {avg_delta:.4f}  "
                  f"({'repair barely changes obj' if abs(avg_delta) < 1.0 else 'OK'})")
            print(f"  SA temp T        : {temp:.4f}")
            print(f"  SA T0            : {T0:.4f}")
            if zero_moves > 40:
                print(f"  !! PROBLEM: repair is returning near-identical solutions.")
                print(f"     Root cause: fix_service_violations is over-constraining.")
                print(f"     The destroy operators are removing assignments but repair")
                print(f"     immediately restores the same Q/Z values via SL enforcement.")
            print(f"  ─────────────────────────────────\n")

        sc_d[d_idx] += score;  us_d[d_idx] += 1
        sc_r[r_idx] += score;  us_r[r_idx] += 1

        # Weight update every segment
        if it % segment == 0:
            w_d = update_weights(w_d, sc_d, us_d, lam)
            w_r = update_weights(w_r, sc_r, us_r, lam)
            sc_d, sc_r = [0.0]*n_d, [0.0]*n_r
            us_d, us_r = [0]*n_d,   [0]*n_r

            if verbose:
                elapsed   = time.time() - t_search_start
                iters_sec = it / max(elapsed, 1e-9)
                remaining = (max_iter - it) / max(iters_sec, 1e-9)
                ar = sum(x_best.X.get((r,t),0) for r in data['R'] for t in data['T'])
                print(f"Iter {it:6d}/{max_iter} | curr={x.obj:.2f} | best={x_best.obj:.2f} "
                      f"| routes={ar} | stagnant={no_improve} | T={temp:.4f} "
                      f"| {elapsed/60:.1f}min elapsed | ETA {remaining/60:.1f}min")
                print(f"  Destroy : { {d_names[i]: round(w_d[i],3) for i in range(n_d)} }")
                print(f"  Repair  : { {r_names[i]: round(w_r[i],3) for i in range(n_r)} }")

    # Final route_swap on best before rounding
    t_search_end = time.time()
    print(f"\nALNS search complete.")
    print(f"  Search time    : {(t_search_end - t_search_start)/60:.2f} min  "
          f"({(t_search_end - t_search_start):.1f}s  |  "
          f"{max_iter / max(t_search_end - t_search_start, 1e-9):.0f} iter/s)")
    print(f"  Applying final route_swap to best solution...")
    route_swap_local_search(x_best, data)
    compute_inventory_waste(x_best, data)
    compute_objective(x_best, data)

    print(f"  Best objective (search) : {x_best.obj:.4f}")
    print(f"  Post-processing (round_solution)...")
    t_round_start = time.time()
    x_best_true = round_solution(x_best, data)
    t_round_end = time.time()
    print(f"  Round time     : {t_round_end - t_round_start:.1f}s")
    print(f"  Best objective (true)   : {x_best_true.obj:.4f}")
    print(f"  Active routes  : {sum(x_best_true.X.get((r,t),0) for r in data['R'] for t in data['T'])}")
    print(f"  SL violations  : {len(get_violations(x_best_true, data, ignore_waste_gaps=False))}")
    print(f"  TOTAL WALL TIME: {(time.time() - t_start)/60:.2f} min")
    return x_best_true


# =============================================================================
# SECTION 10 — EXPORT RESULTS
# =============================================================================

def export_solution(sol, data, path='alns_solution.xlsx'):
    F, K, B, D, R, T, S = (data['F'], data['K'], data['B'],
                             data['D'], data['R'], data['T'], data['S'])

    with pd.ExcelWriter(path, engine='openpyxl') as writer:

        pd.DataFrame([
            {'supplier': f, 'hub': b, 'vehicle': k, 'scenario': s, 'period': t}
            for f in F for b in B for k in K for s in S for t in T
            if sol.Y.get((f, b, k, s, t), 0) == 1
        ]).to_excel(writer, sheet_name='Y_assignments', index=False)

        pd.DataFrame([
            {'supplier': f, 'hub': b, 'vehicle': k, 'scenario': s,
             'period': t, 'amount': round(sol.L.get((f, b, k, s, t), 0.0), 4)}
            for f in F for b in B for k in K for s in S for t in T
            if sol.L.get((f, b, k, s, t), 0.0) > 1e-6
        ]).to_excel(writer, sheet_name='L_transport', index=False)

        pd.DataFrame([
            {'route': r, 'period': t}
            for r in R for t in T if sol.X.get((r, t), 0) == 1
        ]).to_excel(writer, sheet_name='X_routes', index=False)

        pd.DataFrame([
            {'depot': i, 'route': r, 'period': t,
             'load': round(sol.Z.get((i, r, t), 0.0), 4)}
            for i in D for r in R for t in T
            if sol.Z.get((i, r, t), 0.0) > 1e-6
        ]).to_excel(writer, sheet_name='Z_loads', index=False)

        pd.DataFrame([
            {'hub': b, 'depot': i, 'period': t,
             'delivery': round(sol.Q.get((b, i, t), 0.0), 4)}
            for b in B for i in D for t in T
            if sol.Q.get((b, i, t), 0.0) > 1e-6
        ]).to_excel(writer, sheet_name='Q_deliveries', index=False)

        pd.DataFrame([
            {'depot': i, 'period': t,
             'inventory': round(sol.I.get((i, t), 0.0), 4),
             'inventory_pos': round(sol.I_plus.get((i, t), 0.0), 4),
             'waste': round(sol.W.get((i, t), 0.0), 4)}
            for i in D for t in T
        ]).to_excel(writer, sheet_name='Inventory_Waste', index=False)

        pr, beta, gamma = data['pr'], data['beta'], data['gamma']
        h, p, lam       = data['h'], data['p'], data['lambda']

        cost_assign = sum(pr[s]*beta[(f,k)]*sol.Y.get((f,b,k,s,t),0)
                          for f in F for b in B for k in K for s in S for t in T)
        cost_trans  = sum(pr[s]*gamma[(f,b,k)]*sol.L.get((f,b,k,s,t),0.0)
                          for f in F for b in B for k in K for s in S for t in T)
        cost_inv    = sum(sol.I_plus.get((i,t),0.0)*h[i] for i in D for t in T)
        cost_waste  = sum(sol.W.get((i,t),0.0)*p for i in D for t in T if t>=data['m'])
        cost_route  = sum(lam[r] for r in R for t in T if sol.X.get((r,t),0)==1)

        pd.DataFrame([{
            'Total objective':   round(sol.obj, 4),
            'Assignment cost':   round(cost_assign, 4),
            'Transport cost':    round(cost_trans, 4),
            'Inventory cost':    round(cost_inv, 4),
            'Waste cost':        round(cost_waste, 4),
            'Route fixed cost':  round(cost_route, 4),
            'Active routes':     sum(sol.X.get((r,t),0) for r in R for t in T),
            'Total waste (CR)':  round(sum(sol.W.get((i,t),0.0) for i in D for t in T), 4),
        }]).to_excel(writer, sheet_name='Summary', index=False)

    print(f"Solution exported to: {path}")


# =============================================================================
# SECTION 11 — INTEGER ROUNDING WITH FLOW BALANCE ENFORCEMENT
# =============================================================================

def _trim_to_dp_minimum(sol, data):
    D, R, T, B = data['D'], data['R'], data['T'], data['B']

    for i in D:
        demands = [data['mu'][(i, t)] for t in T]
        sl_vals = [data['sl_rhs_cache'][(i, t)] for t in T]

        cumQ      = 0
        inv       = 0.0
        waste_sim = [0] * 6
        changed   = False

        for tidx, t in enumerate(T):
            prior_w  = sum(waste_sim[:tidx])
            min_cumQ = math.ceil(sl_vals[tidx] + prior_w - 1e-9)
            target   = max(int(math.ceil(demands[tidx] - 1e-9)),
                           max(0, min_cumQ - cumQ))
            actual   = sum(sol.Q.get((b, i, t), 0) for b in B)

            if actual > target:
                excess = actual - target
                active = sorted(
                    [r for r in R if i in data['N'][r]
                     and sol.X.get((r, t), 0) == 1
                     and sol.Z.get((i, r, t), 0) > 0],
                    key=lambda r: data['lambda'][r]
                )
                for r in active:
                    z_cur = sol.Z.get((i, r, t), 0)
                    red   = min(z_cur, excess)
                    sol.Z[(i, r, t)] = z_cur - red
                    excess -= red
                    if excess <= 0:
                        break
                compute_Q_from_Z(sol, data)
                actual = sum(sol.Q.get((b, i, t), 0) for b in B)
                changed = True
            elif actual < target:
                need = target - actual
                if not boost_existing_route(sol, data, i, t, need):
                    inactive = [r for r in R
                                if i in data['N'][r] and sol.X.get((r, t), 0) == 0]
                    if inactive:
                        r_best = min(inactive, key=lambda r: data['lambda'][r])
                        sol.X[(r_best, t)] = 1
                        sol.Z[(i, r_best, t)] = sol.Z.get((i, r_best, t), 0) + need
                compute_Q_from_Z(sol, data)
                actual = sum(sol.Q.get((b, i, t), 0) for b in B)
                changed = True

            cumQ += actual
            waste_sim[tidx] = max(0, int(round(max(0.0, inv - demands[tidx])))) if tidx >= 1 else 0
            inv = inv + actual - int(demands[tidx]) - waste_sim[tidx]

        if changed:
            compute_inventory_waste(sol, data)
            for tidx, t in enumerate(T):
                gap = sl_rhs(data, i, t) - sl_lhs(sol, data, i, t)
                if gap > 1e-6:
                    if not boost_existing_route(sol, data, i, t, gap):
                        inactive = [r for r in R
                                    if i in data['N'][r] and sol.X.get((r, t), 0) == 0]
                        if inactive:
                            r_best = min(inactive, key=lambda r: data['lambda'][r])
                            sol.X[(r_best, t)] = 1
                            sol.Z[(i, r_best, t)] = (sol.Z.get((i, r_best, t), 0)
                                                     + int(math.ceil(gap)))
                    compute_Q_from_Z(sol, data)
                    compute_inventory_waste(sol, data)

    compute_Q_from_Z(sol, data)


def _consolidate_vehicles(sol, data):
    F, K, B, S, T = (data['F'], data['K'], data['B'], data['S'], data['T'])
    owned = {data['owner'][k]: k for k in K}

    for s in S:
        for t in T:
            veh_load = {k: sum(sol.L.get((f, b, k, s, t), 0)
                               for f in F for b in B) for k in K}

            nonowner_assigns = sorted([
                (data['beta'][(f, k)], f, k,
                 next((b for b in B if sol.Y.get((f, b, k, s, t), 0) == 1), None),
                 sum(sol.L.get((f, b, k, s, t), 0) for b in B))
                for f in F if f not in owned
                for k in K if sol.Y.get((f, next((b for b in B
                    if sol.Y.get((f, b, k, s, t), 0) == 1), B[0]), k, s, t), 0) == 1
            ], reverse=True)

            for beta_v, f, k_old, b_old, load in nonowner_assigns:
                if b_old is None or load <= 0:
                    continue

                best_k   = None
                best_cost = beta_v
                for k_new in K:
                    if k_new == k_old:
                        continue
                    if data['beta'][(f, k_new)] >= best_cost:
                        continue
                    spare = data['theta'][k_new] - veh_load[k_new]
                    if spare < load - 0.5:
                        continue
                    best_k    = k_new
                    best_cost = data['beta'][(f, k_new)]

                if best_k is None:
                    continue

                sol.Y[(f, b_old, k_old, s, t)] = 0
                sol.L[(f, b_old, k_old, s, t)] = 0
                sol.Y[(f, b_old, best_k, s, t)] = 1
                sol.L[(f, b_old, best_k, s, t)] = load
                veh_load[k_old] -= load
                veh_load[best_k] += load

    compute_objective(sol, data)


def _eliminate_redundant_routes(sol, data):
    D, R, T, B = data['D'], data['R'], data['T'], data['B']
    optimal = set(data.get('optimal_routes', []))

    for t in T:
        for r in list(R):
            if sol.X.get((r, t), 0) == 1:
                load = sum(sol.Z.get((i, r, t), 0) for i in D)
                if load == 0:
                    sol.X[(r, t)] = 0

        changed = True
        while changed:
            changed = False
            for r in sorted(R, key=lambda r: -data['lambda'][r]):
                if sol.X.get((r, t), 0) == 0:
                    continue
                if r in optimal:
                    continue
                load = sum(sol.Z.get((i, r, t), 0) for i in D)
                if load == 0:
                    sol.X[(r, t)] = 0
                    changed = True
                    continue

                moved_all = True
                moves = {}
                for i in D:
                    z = sol.Z.get((i, r, t), 0)
                    if z == 0:
                        continue
                    alternatives = [
                        r2 for r2 in R
                        if r2 != r
                        and sol.X.get((r2, t), 0) == 1
                        and i in data['N'][r2]
                        and (sum(sol.Z.get((d, r2, t), 0) for d in data['N'][r2]) + z
                             <= data['c'][r2])
                    ]
                    if not alternatives:
                        moved_all = False
                        break
                    r_dst = min(alternatives, key=lambda r2: data['lambda'][r2])
                    moves[i] = (r_dst, z)

                if moved_all and moves:
                    for i, (r_dst, z) in moves.items():
                        sol.Z[(i, r, t)]     = 0
                        sol.Z[(i, r_dst, t)] = sol.Z.get((i, r_dst, t), 0) + z
                    sol.X[(r, t)] = 0

                    compute_Q_from_Z(sol, data)
                    compute_inventory_waste(sol, data)
                    viols = [
                        (i2, t2, sl_rhs(data, i2, t2) - sl_lhs(sol, data, i2, t2))
                        for i2 in D for t2 in T
                        if sl_rhs(data, i2, t2) - sl_lhs(sol, data, i2, t2) > 1e-6
                    ]
                    if viols:
                        for i, (r_dst, z) in moves.items():
                            sol.Z[(i, r, t)]     = z
                            sol.Z[(i, r_dst, t)] = max(0, sol.Z.get((i, r_dst, t), 0) - z)
                        sol.X[(r, t)] = 1
                        compute_Q_from_Z(sol, data)
                        compute_inventory_waste(sol, data)
                    else:
                        changed = True

    compute_Q_from_Z(sol, data)
    compute_inventory_waste(sol, data)


def _reroute_owners_to_reduce_nonowners(sol, data):
    F, K, B, S, T = data['F'], data['K'], data['B'], data['S'], data['T']
    owned   = {data['owner'][k]: k for k in K}
    pr      = data['pr']
    beta_d  = data['beta']
    gamma_d = data['gamma']
    theta   = data['theta']
    ups     = data['upsilon']

    for s in S:
        for t in T:
            q_tgt = {b: sum(sol.Q.get((b, i, t), 0) for i in data['D']) for b in B}
            remaining = dict(q_tgt)

            for f in F:
                for b in B:
                    for k in K:
                        sol.Y[(f, b, k, s, t)] = 0
                        sol.L[(f, b, k, s, t)] = 0

            veh_cap = {k: theta[k] for k in K}

            owner_candidates = []
            for f in owned:
                k = owned[f]
                sup = ups.get((f, s, t), 0)
                if sup <= 0:
                    continue
                for b in B:
                    if q_tgt[b] <= 0:
                        continue
                    cost = beta_d[(f, k)] + gamma_d[(f, b, k)] * sup
                    owner_candidates.append((cost, f, k, b, sup))
            owner_candidates.sort()

            assigned_owners = set()
            for _, f, k, b, sup in owner_candidates:
                if f in assigned_owners:
                    continue
                if remaining[b] <= 0:
                    continue
                if veh_cap[k] <= 0:
                    continue
                load = int(round(min(sup, remaining[b], veh_cap[k])))
                if load <= 0:
                    continue
                sol.Y[(f, b, k, s, t)] = 1
                sol.L[(f, b, k, s, t)] = load
                remaining[b]  -= load
                veh_cap[k]    -= load
                assigned_owners.add(f)

            non_owners_avail = sorted(
                [(f, ups.get((f, s, t), 0))
                 for f in F if f not in owned and ups.get((f, s, t), 0) > 0],
                key=lambda x: -x[1]
            )
            assigned_no = set()
            for b in B:
                if remaining[b] <= 0:
                    continue
                nos = sorted(
                    [(beta_d[(f, k)] + gamma_d[(f, b, k)] * sup, f, sup)
                     for f, sup in non_owners_avail
                     if f not in assigned_no
                     for k in K],
                    )
                for _, f, sup in nos:
                    if f in assigned_no:
                        continue
                    if remaining[b] <= 0:
                        break
                    best_k = min(K, key=lambda k: gamma_d[(f, b, k)]
                                 if veh_cap[k] > 0 else float('inf'))
                    if veh_cap[best_k] <= 0:
                        continue
                    load = int(round(min(sup, remaining[b], veh_cap[best_k])))
                    if load <= 0:
                        continue
                    sol.Y[(f, b, best_k, s, t)] = 1
                    sol.L[(f, b, best_k, s, t)] = load
                    remaining[b]  -= load
                    veh_cap[best_k] -= load
                    assigned_no.add(f)

            for b in B:
                l_total = sum(sol.L.get((f, b, k, s, t), 0) for f in F for k in K)
                diff = l_total - q_tgt[b]
                if abs(diff) < 1:
                    continue
                entries = sorted(
                    [(f, k, sol.L.get((f, b, k, s, t), 0))
                     for f in F for k in K if sol.L.get((f, b, k, s, t), 0) > 0],
                    key=lambda x: -x[2]
                )
                for f, k, cur in entries:
                    adj = min(cur, abs(int(diff))) if diff > 0 else -abs(int(diff))
                    sol.L[(f, b, k, s, t)] = max(0, cur - adj if diff > 0 else cur + abs(adj))
                    if sol.L[(f, b, k, s, t)] == 0:
                        sol.Y[(f, b, k, s, t)] = 0
                    diff -= adj if diff > 0 else -abs(adj)
                    if abs(diff) < 1:
                        break

    compute_objective(sol, data)


def _shift_hub3_load_to_cheaper_hubs(sol, data):
    D, R, T, B = data['D'], data['R'], data['T'], data['B']
    owned = {data['owner'][k]: k for k in data['K']}
    pr    = data['pr']

    for t in T:
        hub_target = {b: sum(sol.Q.get((b, i, t), 0) for i in D) for b in B}

        hub_owner_s1 = {}
        for b in B:
            supply = sum(
                min(data['upsilon'].get((f, 1, t), 0), data['theta'][owned[f]])
                for f in owned
            )
            hub_owner_s1[b] = supply

        for r_from in sorted(R, key=lambda r: -data['lambda'][r]):
            if sol.X.get((r_from, t), 0) == 0:
                continue
            b_from = data['route_hub_map'].get(r_from)
            if b_from != 3:
                continue

            depots_on_r = [i for i in D if sol.Z.get((i, r_from, t), 0) > 0]
            if not depots_on_r:
                continue

            for r_to in sorted(R, key=lambda r: data['lambda'][r]):
                if r_to == r_from:
                    continue
                if sol.X.get((r_to, t), 0) == 1:
                    continue
                b_to = data['route_hub_map'].get(r_to)
                if b_to == 3:
                    continue
                if b_to is None:
                    continue

                if not all(i in data['N'][r_to] for i in depots_on_r):
                    continue

                existing_load_rto = sum(sol.Z.get((i, r_to, t), 0) for i in D)
                shift_load = sum(sol.Z.get((i, r_from, t), 0) for i in depots_on_r)
                if existing_load_rto + shift_load > data['c'][r_to]:
                    continue

                avg_no_supply = 35
                min_beta_no = min(data['beta'][(f, k)]
                                  for f in data['F'] if f not in owned
                                  for k in data['K'])
                n_no_saved = shift_load / avg_no_supply
                beta_saving_per_t = n_no_saved * min_beta_no * pr[1]
                route_extra_cost = data['lambda'][r_to]

                r_from_can_deactivate = all(
                    any(sol.X.get((r2, t), 0) == 1 and i in data['N'][r2]
                        for r2 in R if r2 != r_from and data['route_hub_map'].get(r2) != 3)
                    or i in data['N'][r_to]
                    for i in data['N'][r_from]
                    if any(sol.Z.get((i, r_from, t), 0) > 0 for i2 in [i])
                )

                net = beta_saving_per_t - route_extra_cost
                if net <= 0:
                    continue

                for i in depots_on_r:
                    z = sol.Z.get((i, r_from, t), 0)
                    sol.Z[(i, r_from, t)] = 0
                    sol.Z[(i, r_to, t)]   = sol.Z.get((i, r_to, t), 0) + z

                sol.X[(r_to, t)] = 1

                if sum(sol.Z.get((i, r_from, t), 0) for i in D) == 0:
                    sol.X[(r_from, t)] = 0

                compute_Q_from_Z(sol, data)
                compute_inventory_waste(sol, data)

                viols = [1 for i in D for t2 in T
                         if sl_rhs(data, i, t2) - sl_lhs(sol, data, i, t2) > 1e-6]
                if viols:
                    for i in depots_on_r:
                        z = sol.Z.get((i, r_to, t), 0)
                        sol.Z[(i, r_to, t)]   = max(0, z - sol.Z.get((i, r_from, t), 0))
                        sol.Z[(i, r_from, t)] = sol.Z.get((i, r_from, t), 0)
                    sol.X[(r_to, t)] = 0
                    sol.X[(r_from, t)] = 1
                    compute_Q_from_Z(sol, data)
                    compute_inventory_waste(sol, data)
                else:
                    break

    compute_Q_from_Z(sol, data)
    compute_inventory_waste(sol, data)


def _fill_lq_deficits(sol, data):
    F, K, B, S, T, D = (data['F'], data['K'], data['B'],
                         data['S'], data['T'], data['D'])
    owned_v  = {data['owner'][k]: k for k in K}
    theta    = data['theta']
    ups      = data['upsilon']
    beta_d   = data['beta']
    f_sorted = data.get('_f_by_beta',
                        sorted(F, key=lambda f: min(beta_d.get((f,k),999) for k in K)))
    no_best_k = data.get('_nonowner_best_k', {})

    L_by_st   = {}
    Y_by_st   = {}

    for (f, b, k, _s, _t), l in sol.L.items():
        if l <= 0: continue
        key_st = (_s, _t)
        if key_st not in L_by_st: L_by_st[key_st] = {}
        L_by_st[key_st][(f, b, k)] = l

    for (f, b, k, _s, _t), y in sol.Y.items():
        if not y: continue
        key_st = (_s, _t)
        if key_st not in Y_by_st: Y_by_st[key_st] = {}
        Y_by_st[key_st][(f, k)] = b

    for s in S:
        for t in T:
            l_st = L_by_st.get((s, t), {})
            y_st = Y_by_st.get((s, t), {})

            sup_used = {}
            veh_used = {}
            l_by_b   = {}
            fk_hub   = dict(y_st)

            for (f, b, k), l in l_st.items():
                sup_used[f] = sup_used.get(f, 0) + l
                veh_used[k] = veh_used.get(k, 0) + l
                l_by_b[b]   = l_by_b.get(b, 0)  + l

            for b in B:
                q_tgt  = int(sum(sol.Q.get((b, i, t), 0) for i in D))
                deficit = q_tgt - l_by_b.get(b, 0)
                if deficit <= 0:
                    continue

                for f in f_sorted:
                    if deficit <= 0: break
                    sup_avail = int(ups.get((f, s, t), 0)) - sup_used.get(f, 0)
                    if sup_avail <= 0: continue

                    k_own = owned_v.get(f)
                    if k_own is not None:
                        eh = fk_hub.get((f, k_own))
                        if eh is not None and eh != b: continue
                        k_use = k_own
                    else:
                        k_pre = no_best_k.get(f)
                        if k_pre is not None:
                            eh = fk_hub.get((f, k_pre))
                            va = theta[k_pre] - veh_used.get(k_pre, 0)
                            if (eh is None or eh == b) and va > 0:
                                k_use = k_pre
                            else:
                                k_use = next((k for k in K
                                    if (fk_hub.get((f,k)) is None
                                        or fk_hub.get((f,k)) == b)
                                    and theta[k] - veh_used.get(k, 0) > 0), None)
                        else:
                            k_use = None
                        if k_use is None: continue

                    va = theta[k_use] - veh_used.get(k_use, 0)
                    if va <= 0: continue
                    room = min(sup_avail, va, deficit)
                    if room <= 0: continue

                    already = sol.L.get((f, b, k_use, s, t), 0)
                    sol.Y[(f, b, k_use, s, t)] = 1
                    sol.L[(f, b, k_use, s, t)] = already + room
                    l_st[(f, b, k_use)]  = already + room
                    sup_used[f]          = sup_used.get(f, 0) + room
                    veh_used[k_use]      = veh_used.get(k_use, 0) + room
                    l_by_b[b]            = l_by_b.get(b, 0) + room
                    fk_hub[(f, k_use)]   = b
                    if (b,s,t) not in L_by_st: L_by_st[(b,s,t)] = {}
                    deficit -= room


def round_solution(sol, data):
    sol_r = sol.copy()
    D, B, F, K, S, T, R = (data["D"], data["B"], data["F"],
                             data["K"], data["S"], data["T"], data["R"])
    owned_vehicle = {data["owner"][k]: k for k in data["K"]}

    for key in sol_r.Z:
        v = sol_r.Z[key]
        sol_r.Z[key] = int(round(v)) if v > 0 else 0

    compute_Q_from_Z(sol_r, data)

    compute_inventory_waste(sol_r, data)
    fix_service_violations(sol_r, data)

    for key in sol_r.L:
        sol_r.L[key] = 0
    for key in sol_r.Y:
        sol_r.Y[key] = 0

    for s in S:
        for t in T:
            remaining_cap = {k: data["theta"][k] for k in K}
            hub_need = {b: int(sum(sol_r.Q.get((b, i, t), 0) for i in D))
                        for b in B}

            for f in F:
                if f not in owned_vehicle:
                    continue
                k      = owned_vehicle[f]
                supply = int(data["upsilon"].get((f, s, t), 0.0))
                if supply <= 0 or remaining_cap[k] <= 0:
                    continue
                for best_b in sorted(B, key=lambda b: data["gamma"][(f, b, k)]):
                    need = hub_need[best_b]
                    if need <= 0:
                        continue
                    amount = min(supply, need, remaining_cap[k])
                    if amount <= 0:
                        continue
                    sol_r.Y[(f, best_b, k, s, t)] = 1
                    sol_r.L[(f, best_b, k, s, t)] = amount
                    remaining_cap[k]              -= amount
                    hub_need[best_b]              -= amount
                    break

            if any(need > 0 for need in hub_need.values()):
                _assign_nonowners_optimal(F, K, B, data, hub_need,
                                          remaining_cap, s, t, sol_r)

    _eliminate_redundant_routes(sol_r, data)
    _trim_to_dp_minimum(sol_r, data)

    _l_bst = {}
    for (f2,b2,k2,s2,t2), lv in sol_r.L.items():
        if lv > 0:
            _l_bst.setdefault((b2,s2,t2), []).append([f2, k2, lv])

    _q_bt = {(b, t): int(sum(sol_r.Q.get((b,i,t),0) for i in D))
             for b in B for t in T}

    for (b,s,t), entries in _l_bst.items():
        q_target = _q_bt.get((b, t), 0)
        l_total  = sum(e[2] for e in entries)
        diff = l_total - q_target
        if diff < 1:
            continue
        entries.sort(key=lambda x: -x[2])
        for entry in entries:
            f, k, cur = entry[0], entry[1], entry[2]
            if diff < 1: break
            reduce = min(cur, int(diff))
            new_val = cur - reduce
            sol_r.L[(f, b, k, s, t)] = new_val
            entry[2] = new_val
            if new_val == 0:
                sol_r.Y[(f, b, k, s, t)] = 0
            diff -= reduce

    _owned_v5 = {data['owner'][k]:k for k in K}
    _f_beta5  = data.get('_f_by_beta',
                         sorted(F, key=lambda f: min(data['beta'].get((f,k2),999) for k2 in K)))
    _l_bst2 = {}
    for (f2,b2,k2,s2,t2), lv in sol_r.L.items():
        if lv > 0:
            _l_bst2.setdefault((b2,s2,t2), []).append((f2,k2,lv))

    for s in S:
        for t in T:
            _su = {}; _vu = {}; _fkh = {}
            for b in B:
                for (f2,k2,lv) in _l_bst2.get((b,s,t), []):
                    _su[f2] = _su.get(f2,0) + lv
                    _vu[k2] = _vu.get(k2,0) + lv
                    _fkh[(f2,k2)] = b

            for b in B:
                q_tgt   = _q_bt.get((b, t), 0)
                l_now   = sum(lv for (f2,k2,lv) in _l_bst2.get((b,s,t), []))
                deficit = q_tgt - l_now
                if deficit < 1:
                    continue
                # Build remaining_cap from current vehicle usage
                rc = {k2: data['theta'][k2] - _vu.get(k2, 0) for k2 in K}
                hn = {b2: (deficit if b2 == b else 0) for b2 in B}
                _assign_nonowners_optimal(F, K, B, data, hn, rc, s, t, sol_r)
                # Update sparse tracking
                for (f2,b2,k2,_s2,_t2), lv in sol_r.L.items():
                    if _s2 == s and _t2 == t and lv > 0:
                        _vu[k2] = _vu.get(k2, 0)  # rebuild below
                _l_bst2 = {}
                for (f2,b2,k2,s2,t2), lv in sol_r.L.items():
                    if lv > 0:
                        _l_bst2.setdefault((b2,s2,t2), []).append((f2,k2,lv))

    compute_inventory_waste(sol_r, data)
    for key in sol_r.I:
        sol_r.I[key]      = int(round(sol_r.I[key]))
    for key in sol_r.I_plus:
        sol_r.I_plus[key] = max(0, int(round(sol_r.I_plus[key])))
    for key in sol_r.W:
        sol_r.W[key]      = max(0, int(round(sol_r.W[key])))

    compute_objective(sol_r, data)
    return sol_r


# =============================================================================
# SECTION 12 — CONSTRAINT CHECKER
# =============================================================================

def check_solution(sol, data, label="ALNS", tol=1e-2):
    F, K, B, D, R, T, S = (data["F"], data["K"], data["B"],
                             data["D"], data["R"], data["T"], data["S"])
    pr    = data["pr"]
    beta  = data["beta"]
    gamma = data["gamma"]
    h     = data["h"]
    p     = data["p"]
    lam   = data["lambda"]
    m     = data["m"]

    results = {}
    all_ok  = True

    print(f'\n{"="*55}')
    print(f'  SOLUTION CHECK — {label}')
    print(f'{"="*55}')

    ac = sum(
        pr[s] * (beta[(f, k)] * sol.Y.get((f, b, k, s, t), 0)
                 + gamma[(f, b, k)] * sol.L.get((f, b, k, s, t), 0.0))
        for f in F for b in B for k in K for s in S for t in T
    )
    results["AC"] = ac
    print(f'  AC_check({label}) = {ac:.4f}')
    print()

    multi_count = sum(
        1 for f in F for k in K for s in S for t in T
        if sum(sol.Y.get((f, b, k, s, t), 0) for b in B) > 1
    )
    results["multi_assign"] = multi_count
    print(f'  [CHECK] multi-assign (same (f,k) pair in >1 hub) count = {multi_count}')
    if multi_count > 0:
        all_ok = False

    rc_viol = sum(
        1 for r in R for t in T
        if sum(sol.Z.get((i, r, t), 0.0) for i in D)
           > data["c"][r] * sol.X.get((r, t), 0) + tol
    )
    results["route_cap_violations"] = rc_viol
    status_d = "OK" if rc_viol == 0 else f"FAIL ({rc_viol} violations)"
    print(f"  [CHECK-D] Route capacity: {status_d}")
    if rc_viol > 0:
        all_ok = False

    ze_viol = 0
    for r in R:
        for t in T:
            for i in D:
                z = sol.Z.get((i, r, t), 0.0)
                if z > tol:
                    if sol.X.get((r, t), 0) == 0:
                        ze_viol += 1
                    elif i not in data["N"][r]:
                        ze_viol += 1
    results["z_when_x0_violations"] = ze_viol
    status_e = "OK" if ze_viol == 0 else f"FAIL ({ze_viol} violations)"
    print(f"  [CHECK-E] Z=0 when X=0: {status_e}")
    if ze_viol > 0:
        all_ok = False

    sup_viol = sum(
        1 for f in F for s in S for t in T
        if sum(sol.L.get((f, b, k, s, t), 0) for b in B for k in K)
           > data["upsilon"].get((f, s, t), 0.0) + tol
    )
    results["supply_cap_violations"] = sup_viol
    status_f = "OK" if sup_viol == 0 else f"FAIL ({sup_viol} violations)"
    print(f"  [CHECK-F] Supply capacity: {status_f}")
    if sup_viol > 0:
        all_ok = False

    vc_viol = sum(
        1 for k in K for s in S for t in T
        if sum(sol.L.get((f, b, k, s, t), 0.0) for f in F for b in B)
           > data["theta"][k] + tol
    )
    results["vehicle_cap_violations"] = vc_viol
    status_g = "OK" if vc_viol == 0 else f"FAIL ({vc_viol} violations)"
    print(f"  [CHECK-G] Vehicle capacity: {status_g}")
    if vc_viol > 0:
        all_ok = False

    wm_viol = sum(
        1 for i in D for t in T
        if t < m and sol.W.get((i, t), 0.0) > tol
    )
    results["waste_before_m_violations"] = wm_viol
    status_h = "OK" if wm_viol == 0 else f"FAIL ({wm_viol} violations)"
    print(f"  [CHECK-H] W=0 before shelf_life: {status_h}")
    if wm_viol > 0:
        all_ok = False

    yl_count = sum(
        1 for f in F for b in B for k in K for s in S for t in T
        if sol.Y.get((f, b, k, s, t), 0) == 1
        and sol.L.get((f, b, k, s, t), 0.0) < tol
    )
    results["y1_l0_count"] = yl_count
    print(f"  [CHECK] Y=1 but L=0 count = {yl_count}")
    if yl_count > 0:
        all_ok = False

    print(f"  AC_check({label}) = {ac:.4f}")

    fb_viol  = 0
    fb_max   = 0.0
    for b in B:
        for s in S:
            for t in T:
                lhs = sum(sol.L.get((f, b, k, s, t), 0.0) for f in F for k in K)
                rhs = sum(sol.Q.get((b, i, t), 0.0) for i in D)
                diff = abs(lhs - rhs)
                if diff > tol:
                    fb_viol += 1
                    fb_max   = max(fb_max, diff)
    results["flow_balance_violations"] = fb_viol
    results["flow_balance_max_diff"]   = fb_max
    status_c = "OK" if fb_viol == 0 else f"FAIL ({fb_viol} violations, max diff={fb_max:.4f})"
    print(f"  [CHECK-C] OPL balance constraint 2.11: {status_c}")
    if fb_viol > 0:
        all_ok = False

    sl_viol = get_violations(sol, data, ignore_waste_gaps=False)
    results["sl_violations"] = len(sl_viol)
    status_a = "OK" if not sl_viol else f"FAIL ({len(sl_viol)} violations)"
    print(f"  [CHECK-A] Service-level constraint ({label} 2.15): {status_a}")
    if sl_viol:
        all_ok = False
        for i, t, gap in sorted(sl_viol, key=lambda x: x[2], reverse=True)[:3]:
            print(f"    depot={i} t={t}: shortfall={gap:.4f}")

    efb_viol = 0
    efb_max  = 0.0
    for b in B:
        for t in T:
            e_lhs = sum(pr[s] * sum(sol.L.get((f, b, k, s, t), 0.0)
                                    for f in F for k in K)
                        for s in S)
            rhs   = sum(sol.Q.get((b, i, t), 0.0) for i in D)
            diff  = abs(e_lhs - rhs)
            if diff > tol:
                efb_viol += 1
                efb_max   = max(efb_max, diff)
    results["expected_flow_balance_violations"] = efb_viol
    status_b = "OK" if efb_viol == 0 else f"FAIL ({efb_viol} violations, max diff={efb_max:.4f})"
    print(f"  [CHECK-B] Flow balance: {status_b}")
    if efb_viol > 0:
        all_ok = False

    cost_assign = sum(pr[s] * beta[(f, k)] * sol.Y.get((f, b, k, s, t), 0)
                      for f in F for b in B for k in K for s in S for t in T)
    cost_trans  = sum(pr[s] * gamma[(f, b, k)] * sol.L.get((f, b, k, s, t), 0.0)
                      for f in F for b in B for k in K for s in S for t in T)
    cost_inv    = sum(sol.I_plus.get((i, t), 0.0) * h[i] for i in D for t in T)
    cost_waste  = sum(sol.W.get((i, t), 0.0) * p for i in D for t in T if t >= m)
    cost_route  = sum(lam[r] for r in R for t in T if sol.X.get((r, t), 0) == 1)
    total       = cost_assign + cost_trans + cost_inv + cost_waste + cost_route

    results.update({
        "cost_assign": cost_assign, "cost_trans": cost_trans,
        "cost_inv": cost_inv,       "cost_waste": cost_waste,
        "cost_route": cost_route,   "total": total,
    })

    n_routes = sum(sol.X.get((r, t), 0) for r in R for t in T)
    n_assign = sum(1 for f in F for b in B for k in K for s in S for t in T
                   if sol.Y.get((f, b, k, s, t), 0) == 1)
    n_sl_ok  = len(T) * len(D) - len(sl_viol)

    print()
    print(f'  {"─"*50}')
    print(f"  OBJECTIVE BREAKDOWN")
    print(f'  {"─"*50}')
    print(f"  Assignment cost   : {cost_assign:>12.4f}")
    print(f"  Transport cost    : {cost_trans:>12.4f}")
    print(f"  Inventory cost    : {cost_inv:>12.4f}")
    print(f"  Waste cost        : {cost_waste:>12.4f}")
    print(f"  Route fixed cost  : {cost_route:>12.4f}")
    print(f'  {"─"*50}')
    print(f"  TOTAL             : {total:>12.4f}")
    print()
    print(f"  Active routes     : {n_routes}")
    print(f"  Active Y=1 assign : {n_assign}")
    print(f"  SL satisfied      : {n_sl_ok}/{len(T)*len(D)}")
    print()
    print(f'  OVERALL: {"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    print(f'{"="*55}')

    results["all_ok"] = all_ok
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import random
    random.seed(42)

    t_total_start = time.time()
    data = load_data('step1_final.xlsx')

    # Run ALNS — 15 000 iterations (~8–10 min on typical hardware)
    best = run_alns(
        data,
        max_iter = 500,   # SPEED: 30000→15000; faster iterations compensate
        segment  = 200,
        lam      = 0.3,
        verbose  = True
    )

    # Constraint check and export
    print()
    check_solution(best, data, label='ALNS_final')
    export_solution(best, data, path='alns_solution_final.xlsx')
    print(f"\nTotal script time (incl. data load + export): "
          f"{(time.time() - t_total_start)/60:.2f} min")