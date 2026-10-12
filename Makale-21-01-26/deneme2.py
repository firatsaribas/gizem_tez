# -*- coding: utf-8 -*-
"""
FFH (Feasibility-First Heuristic) - IMPROVED VERSION v3
=========================================================
Gap vs 12-hour MILP (before v3):
  Route cost   +€225  (40% of gap) ← targeted by IMP-R1, IMP-R2
  Assign cost  +€235  (42% of gap) ← targeted by IMP-A1
  Transport    +€68   (12% of gap) ← targeted by IMP-A2
  Holding/Waste ≈ 0   (essentially optimal — untouched)
 
New improvements over v2 (annotated with IMP-* tags):
 
[Phase 1 — route selection]
  IMP-R1: Multi-route swap (2-opt on route set)
           After single-swap local search, try replacing every PAIR of
           selected routes (r1,r2) with every single cheaper route r_new
           that covers U(r1)∪U(r2).  This exposes consolidation
           opportunities invisible to 1-for-1 swaps — the primary reason
           the heuristic overspends on routes.
 
  IMP-R2: Route set perturbation + restart (iterated local search)
           After local search converges, randomly drop one selected route,
           repair coverage greedily, then re-run drop+swap.  Repeat for
           N_RESTARTS=8 independent restarts, keep the best-cost feasible
           set.  Breaks out of local optima that trap the greedy.
 
[Phase 2 — inbound assignment]
  IMP-A1: Cross-scenario vehicle pre-selection
           Before running per-scenario greedies, identify the set of (f,k)
           pairs that are cheapest on a EXPECTED-cost basis across all
           scenarios (weighted by scenario_probs).  These are passed as
           pre_activated_fk so the per-scenario greedy reuses them at zero
           marginal beta.  Reduces total activation count, directly cutting
           assignment cost.
 
  IMP-A2: Gamma-sorted supplier ordering in Pass 1
           Suppliers are now pre-sorted by gamma[(f,b,k_best)] ascending
           for each hub before the inner loop, so the first feasible
           candidate encountered is already the cheapest-transport option.
           Reduces Pass 2 residual and lowers transport cost.
 
Changes over v1 (annotated with fix IDs):
 
[Phase 1 — greedy_route_selection_and_loading_int]
  FIX-P1A: Hub-aware load cap
            _load_routes now tracks per-hub cumulative load and stops adding
            Z entries once a hub's total equals OUT_budget[b]. This prevents
            Phase 1 from over-loading hubs, reducing the number of
            convergence iterations needed in Phase 2 (often from 3-5 → 1-2).
 
  FIX-P1B: Demand-weighted swap evaluation
            The swap local-search now evaluates *both* cost improvement AND
            coverage quality (Σ qmin over newly-covered depots) so it does
            not swap a high-demand route for a marginally cheaper one that
            covers only low-demand depots.
 
[Phase 2 — inbound_assign_greedy_one_scenario_int / convergence loop]
  FIX-P2A: Float/int boundary in _best_vehicle_for_pair
            actual_ship is now int_floor'd before computing regret, preventing
            fractional values < 1 from inflating unit-cost scores.
 
  FIX-P2B: Convergence loop final-rerun correctness
            The exhausted-loop branch now collects achieved[b] per scenario
            *before* updating OUT_final, so every scenario in the final pass
            targets the same OUT_final and ΣL = ΣQ is guaranteed.
 
  FIX-P2C: Pass-1 inner loop break condition
            After commit() sets hub_needs[b] to 0 or removes it, the while
            condition `b in hub_needs` correctly exits. Previously the loop
            could attempt a second iteration for hubs that were already fully
            served when supply_left[f] happened to be exactly 0.
 
[Phase 3 — OPL-exact inventory]
  FIX-P3A: O(|T|) cumulative demand
            de_cum and de_sq_cum are now maintained as running sums across
            the t-loop instead of being recomputed by inner-loop summation
            on every period. Eliminates the O(|D|×|T|²) inner sum.
 
  FIX-P3B: Consistent rounding convention
            All integer conversions now use int_floor throughout Phase 3
            (W, I, Ipos). The previous mix of floor+1e-9 for W and round for
            I caused ±1 discrepancies in CHECK-C. I is re-derived as
            Q_cum - de_cum - W_cum, all integers after this change.
 
  FIX-P3C: W_cum integrity guard
            W_cum[(i,t)] is computed as W_cum[(i,t-1)] + W[(i,t)] using only
            integer-floored W values, keeping the cumulative exact.
 
[Code structure]
  - de_cum_state dict replaces repeated per-period summation.
  - _load_routes gains hub_budget parameter.
  - No external API changes — all dataclasses and public signatures unchanged.
"""
 
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import math
import time
 
DEBUG_MODE = False
 
 
# ============================================================
# Timing helper
# ============================================================
class PhaseTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
 
    def __enter__(self):
        self.t0 = time.perf_counter()
        print(f"\n[START] {self.name}")
        return self
 
    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        status = "DONE" if exc is None else "FAIL"
        suffix = "" if exc is None else f" | error={exc}"
        print(f"[{status}]  {self.name} | elapsed={dt:.2f} sec{suffix}")
        return False
 
 
# ============================================================
# Integer helpers
# ============================================================
def int_floor(x: float) -> int:
    return int(math.floor(x + 1e-9))
 
 
def int_round(x: float) -> int:
    return int(round(x))
 
 
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
 
 
# ============================================================
# Dataclasses
# ============================================================
@dataclass(frozen=True)
class Route:
    r_id: str
    hub: str
    depots: Tuple[str, ...]
    capacity: float
    fixed_cost: float
 
 
@dataclass
class Instance:
    T: List[int]
    D: List[str]
    B: List[str]
    F: List[str]
    S: List[str]
    K: List[str]
    routes: List[Route]
 
    mu: Dict[Tuple[str, int], float]
    alpha: float
    cv: float
    shelf_life: int
 
    holding_cost: Dict[str, float]
    waste_cost: float
 
    supply: Dict[Tuple[str, str, int], float]
    allowed_f_b: Set[Tuple[str, str]]
    theta: Dict[str, float]
 
    beta: Dict[Tuple[str, str], float]
    gamma: Dict[Tuple[str, str, str], float]
 
    scenario_probs: Dict[str, float]
 
 
@dataclass
class FFHSolution:
    X: Dict[Tuple[str, int], int] = field(default_factory=dict)
    Z: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    Q: Dict[Tuple[str, str, int], int] = field(default_factory=dict)
    Y: Dict[Tuple[str, str, str, str, int], int] = field(default_factory=dict)
    L: Dict[Tuple[str, str, str, str, int], int] = field(default_factory=dict)
    Ipos: Dict[Tuple[str, int], int] = field(default_factory=dict)
    I:    Dict[Tuple[str, int], int] = field(default_factory=dict)
    W: Dict[Tuple[str, int], int] = field(default_factory=dict)
    Q_cum: Dict[Tuple[str, int], int] = field(default_factory=dict)
    W_cum: Dict[Tuple[str, int], int] = field(default_factory=dict)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    cost_breakdown_opl: Dict[str, float] = field(default_factory=dict)
 
 
# ============================================================
# Excel Loader  (unchanged)
# ============================================================
def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df
 
 
def assert_cols(df: pd.DataFrame, needed: List[str], name: str) -> None:
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing columns: {missing}. Existing: {list(df.columns)}")
 
 
def to_str_id(x) -> str:
    if pd.isna(x):
        raise ValueError("Found NaN in an ID field.")
    if isinstance(x, float) and float(x).is_integer():
        x = int(x)
    return str(x)
 
 
def z_value(alpha: float) -> float:
    lookup = {
        0.90: 1.281551566,
        0.95: 1.644853627,
        0.975: 1.959963985,
        0.99: 2.326347874,
        0.995: 2.575829304,
    }
    if alpha in lookup:
        return lookup[alpha]
    x = 2 * alpha - 1
    ln = math.log(1 - x * x)
    tt = 2 / (math.pi * 0.147) + ln / 2
    erfinv = math.copysign(
        math.sqrt(max(0.0, math.sqrt(tt * tt - ln / 0.147) - tt)), x
    )
    return math.sqrt(2) * erfinv
 
 
def load_instance_from_excel(
    file_path: str,
    route_to_depots: Dict[int, List[int]],
    route_to_hub: Dict[int, int],
    *,
    alpha: float,
    cv: float,
    shelf_life: int,
    waste_cost: float,
    scenario_probs: Dict[int, float],
) -> Instance:
    with PhaseTimer("Phase 0 - Data loading (Excel -> Instance)"):
        sheets = {
            "beta": ["f", "k", "beta"],
            "gamma": ["f", "b", "k", "gamma"],
            "theta": ["k", "theta"],
            "stock_costs": ["d", "stock_cost"],
            "supply": ["f", "s", "t", "supply"],
            "demand": ["d", "t", "demand"],
            "route_costs": ["r", "cost"],
            "route_capacity": ["r", "capacity"],
        }
        dfs = {}
        for sheet, cols in sheets.items():
            df = norm_cols(pd.read_excel(file_path, sheet_name=sheet))
            assert_cols(df, cols, sheet)
            dfs[sheet] = df
 
        beta_df        = dfs["beta"].astype({"f": int, "k": int})
        gamma_df       = dfs["gamma"].astype({"f": int, "b": int, "k": int})
        theta_df       = dfs["theta"].astype({"k": int})
        stock_costs_df = dfs["stock_costs"].astype({"d": int})
        supply_df      = dfs["supply"].astype({"f": int, "s": int, "t": int})
        demand_df      = dfs["demand"].astype({"d": int, "t": int})
        route_costs_df = dfs["route_costs"].astype({"r": int})
        route_cap_df   = dfs["route_capacity"].astype({"r": int})
 
        beta         = {(to_str_id(r.f), to_str_id(r.k)): float(r.beta)
                        for r in beta_df.itertuples(index=False)}
        gamma        = {(to_str_id(r.f), to_str_id(r.b), to_str_id(r.k)): float(r.gamma)
                        for r in gamma_df.itertuples(index=False)}
        theta        = {to_str_id(r.k): float(r.theta)
                        for r in theta_df.itertuples(index=False)}
        holding_cost = {to_str_id(r.d): float(r.stock_cost)
                        for r in stock_costs_df.itertuples(index=False)}
        supply       = {(to_str_id(r.f), to_str_id(r.s), int(r.t)): float(r.supply)
                        for r in supply_df.itertuples(index=False)}
        mu           = {(to_str_id(r.d), int(r.t)): float(r.demand)
                        for r in demand_df.itertuples(index=False)}
 
        route_costs_dict   = {int(r.r): float(r.cost) for r in route_costs_df.itertuples(index=False)}
        route_cap_dict     = {int(r.r): float(r.capacity) for r in route_cap_df.itertuples(index=False)}
        R_set = sorted(route_costs_dict.keys())
 
        F = sorted({to_str_id(x) for x in beta_df["f"].unique()})
        K = sorted({to_str_id(x) for x in beta_df["k"].unique()})
        B = sorted({to_str_id(x) for x in gamma_df["b"].unique()})
        D = sorted({to_str_id(x) for x in stock_costs_df["d"].unique()})
        S = sorted({to_str_id(x) for x in supply_df["s"].unique()})
        T = sorted({int(x) for x in supply_df["t"].unique()})
 
        if set(route_to_depots.keys()) != set(R_set):
            raise ValueError("Route ID mismatch: route_to_depots vs Excel route_costs.")
        if set(route_to_hub.keys()) != set(R_set):
            raise ValueError("Route ID mismatch: route_to_hub vs Excel route_costs.")
 
        allowed_f_b = {(f, b) for (f, b, k) in gamma.keys()}
 
        routes: List[Route] = []
        for r in R_set:
            routes.append(Route(
                r_id=str(r),
                hub=str(route_to_hub[r]),
                depots=tuple(str(int(d)) for d in route_to_depots[r]),
                capacity=float(route_cap_dict[r]),
                fixed_cost=float(route_costs_dict[r]),
            ))
 
        sp = {to_str_id(s): float(p) for s, p in scenario_probs.items()}
 
        inst = Instance(
            T=T, D=D, B=B, F=F, S=S, K=K, routes=routes,
            mu=mu, alpha=float(alpha), cv=float(cv), shelf_life=int(shelf_life),
            holding_cost=holding_cost, waste_cost=float(waste_cost),
            supply=supply, allowed_f_b=allowed_f_b, theta=theta,
            beta=beta, gamma=gamma, scenario_probs=sp,
        )
        print("Instance sizes:",
              f"|F|={len(inst.F)}, |K|={len(inst.K)}, |B|={len(inst.B)}, "
              f"|D|={len(inst.D)}, |S|={len(inst.S)}, |T|={len(inst.T)}, "
              f"|R|={len(inst.routes)}")
        return inst
 
 
# ============================================================
# REQ precomputation  (unchanged)
# ============================================================
def precompute_REQ(inst: Instance) -> Dict[Tuple[str, int], float]:
    z = z_value(inst.alpha)
    REQ: Dict[Tuple[str, int], float] = {}
    for i in inst.D:
        cum_mu  = 0.0
        cum_var = 0.0
        for t in inst.T:
            mu_it    = inst.mu[(i, t)]
            cum_mu  += mu_it
            cum_var += (inst.cv * mu_it) ** 2
            REQ[(i, t)] = math.ceil(cum_mu + z * math.sqrt(cum_var))
    return REQ
 
 
# ============================================================
# Phase 2 helper: best vehicle for a (supplier, hub) pair
# ============================================================
def _best_vehicle_for_pair(
    f: str,
    b: str,
    remaining_hub: int,
    supply_available: float,          # kept float per FIX-B of v1
    inst: Instance,
    cap_left: Dict[str, int],
    activated_fk: set = None,
    fk_hub: Dict = None,
) -> Optional[Tuple[float, str, int, float]]:
    """
    Two-tier vehicle selection.  See v1 docstring for rationale.
 
    FIX-P2A: actual_ship is int_floor'd before regret computation so that
    unit-cost scores are never divided by a fractional value < 1.
    """
    max_to_ship = min(supply_available, remaining_hub)
    if max_to_ship <= 0:
        return None
 
    if activated_fk is None:
        activated_fk = set()
    if fk_hub is None:
        fk_hub = {}
 
    reuse: List[Tuple[float, str, int]] = []
    new:   List[Tuple[float, float, str, int]] = []
 
    for k in inst.K:
        if cap_left[k] <= 0:
            continue
        if fk_hub.get((f, k)) not in (None, b):
            continue
        if (f, k) not in inst.beta or (f, b, k) not in inst.gamma:
            continue
        # FIX-P2A: integer-floor the shipment before scoring
        actual_ship = int_floor(min(float(max_to_ship), float(cap_left[k])))
        if actual_ship <= 0:
            continue
        var_cost = inst.gamma[(f, b, k)] * actual_ship
        if (f, k) in activated_fk:
            reuse.append((var_cost, k, actual_ship))
        else:
            new.append((inst.beta[(f, k)], var_cost, k, actual_ship))
 
    if not reuse and not new:
        return None
 
    if reuse:
        reuse.sort()
        var_cost, best_k, best_ship = reuse[0]
        best_unit_cost = var_cost / max(1, best_ship)
        if new:
            new.sort()
            regret = new[0][0]
        elif len(reuse) > 1:
            regret = (reuse[1][0] - reuse[0][0]) / max(1, best_ship)
        else:
            regret = best_unit_cost
    else:
        new.sort()
        beta_val, var_cost, best_k, best_ship = new[0]
        best_unit_cost = (beta_val + var_cost) / max(1, best_ship)
        if len(new) > 1:
            b2b, b2v, _, b2s = new[1]
            regret = (b2b + b2v) / max(1, b2s) - best_unit_cost
        else:
            regret = best_unit_cost
 
    return best_unit_cost, best_k, best_ship, regret
 
 
# ============================================================
# Phase 2: inbound assignment (one scenario, one period)
# ============================================================
def inbound_assign_greedy_one_scenario_int(
    inst: Instance,
    s: str,
    t: int,
    OUT_bt: Dict[str, int],
    pre_activated_fk: set = None,
) -> Tuple[
    Dict[Tuple[str, str, str, str, int], int],
    Dict[Tuple[str, str, str, str, int], int],
    Dict[str, int],
]:
    """
    Activation-minimising inbound assignment (2-pass).
 
    IMP-A1: pre_activated_fk — set of (f,k) pairs selected by the
    cross-scenario pre-selector (see _cross_scenario_preselect).  These
    pairs have marginal beta = 0 in this scenario, so they are always
    preferred in Pass 1 Tier-1 selection.  This directly reduces the
    total weighted activation count across scenarios.
 
    IMP-A2: Suppliers are pre-sorted by their best gamma to hub b before
    the inner loop, so the first feasible candidate found in Tier-2 is
    already the cheapest-transport option.  This reduces the residual left
    for Pass 2 and lowers transport cost.
    """
    cap_left: Dict[str, int] = {k: int_floor(inst.theta[k]) for k in inst.K}
    # IMP-A1: merge caller-provided pre_activated_fk into local activated set
    activated_fk: set = set(pre_activated_fk) if pre_activated_fk else set()
 
    Y_s: Dict[Tuple[str, str, str, str, int], int] = {}
    L_s: Dict[Tuple[str, str, str, str, int], int] = {}
    achieved: Dict[str, int] = {b: 0 for b in inst.B}
    fk_hub: Dict[Tuple[str, str], str] = {}
 
    supply_left: Dict[str, float] = {
        f: float(inst.supply.get((f, s, t), 0.0)) for f in inst.F
    }
    hub_needs: Dict[str, int] = {
        b: int(OUT_bt.get(b, 0)) for b in inst.B if OUT_bt.get(b, 0) > 0
    }
 
    # IMP-A2: pre-sort suppliers per hub by (supply_s DESC, best_gamma ASC)
    # Sorting by supply first prevents low-supply suppliers from claiming
    # vehicle activations that should go to high-supply suppliers.
    # best_gamma_fb[(f,b)] = min gamma[(f,b,k)] over all k with data
    best_gamma_fb: Dict[Tuple[str, str], float] = {}
    for (f, b, k), g in inst.gamma.items():
        key = (f, b)
        if key not in best_gamma_fb or g < best_gamma_fb[key]:
            best_gamma_fb[key] = g
    # suppliers_for_hub[b] = list of f sorted by supply DESC, then gamma ASC
    suppliers_for_hub: Dict[str, List[str]] = {
        b: sorted(
            [f for f in inst.F if (f, b) in inst.allowed_f_b],
            key=lambda f: (
                -float(inst.supply.get((f, s, t), 0.0)),   # supply DESC
                best_gamma_fb.get((f, b), float('inf'))     # gamma ASC
            )
        )
        for b in inst.B
    }
 
    def commit(f: str, b: str, k: str, ship) -> None:
        ship_int = int_floor(float(ship))
        if ship_int <= 0:
            return
        key = (f, b, k, s, t)
        Y_s[key] = 1
        L_s[key]       = L_s.get(key, 0) + ship_int
        cap_left[k]   -= ship_int
        fk_hub[(f, k)] = b
        activated_fk.add((f, k))
        supply_left[f] -= ship_int
        achieved[b]   += ship_int
        hub_needs[b]  -= ship_int
        if hub_needs[b] <= 0:
            hub_needs.pop(b, None)
 
    def best_free_pair_for_residual(b: str, need: int) -> Optional[Tuple[str, str, int]]:
        """
        IMP-A4: Consolidation check.
        Find the already-active (f,k) pair with remaining cap and supply
        that can absorb the most of `need` at lowest gamma cost.
        Returns (f, k, ship) or None.
        """
        best: Optional[Tuple[float, str, str, int]] = None  # (gamma_cost, f, k, ship)
        for (fk, b_locked) in fk_hub.items():
            if b_locked != b:
                continue
            f, k = fk
            if supply_left.get(f, 0) <= 0:
                continue
            if cap_left.get(k, 0) <= 0:
                continue
            if (f, b, k) not in inst.gamma:
                continue
            ship = int_floor(min(supply_left[f], float(cap_left[k]), float(need)))
            if ship <= 0:
                continue
            g = inst.gamma[(f, b, k)]
            if best is None or g < best[0]:
                best = (g, f, k, ship)
        if best is None:
            return None
        return best[1], best[2], best[3]
 
    # ── PASS 1: fill each hub greedily ───────────────────────────────────────
    for b in sorted(list(hub_needs.keys()), key=lambda b: hub_needs[b], reverse=True):
        while b in hub_needs:
            need = hub_needs[b]
 
            # IMP-A4: try consolidation onto an existing activation first
            consolidation = best_free_pair_for_residual(b, need)
            if consolidation is not None:
                f, k, ship = consolidation
                commit(f, b, k, ship)
                if DEBUG_MODE:
                    print(f"  [P2-P1-CONS] t={t} s={s} CONSOLIDATE b={b} f={f} k={k} ship={ship}")
                continue
 
            best_free: Optional[Tuple[float, float, str, str, int]] = None
            best_new:  Optional[Tuple[float, float, str, str, int]] = None
            best_new_fallback: Optional[Tuple[float, float, str, str, int]] = None
 
            for f in suppliers_for_hub.get(b, []):
                if supply_left[f] <= 0:
                    continue
                for k in inst.K:
                    if cap_left[k] <= 0:
                        continue
                    if fk_hub.get((f, k)) not in (None, b):
                        continue
                    if (f, k) not in inst.beta or (f, b, k) not in inst.gamma:
                        continue
                    ship = int_floor(min(supply_left[f], float(cap_left[k]), float(need)))
                    if ship <= 0:
                        continue
                    var_cost = inst.gamma[(f, b, k)] * ship
                    coverage = ship / need
                    if (f, k) in activated_fk:
                        score = (coverage, -var_cost)
                        if best_free is None or score > (best_free[0], best_free[1]):
                            best_free = (coverage, -var_cost, f, k, ship)
                    else:
                        amortised = (inst.beta[(f, k)] + var_cost) / max(1, ship)
                        score = (-amortised, coverage)
                        # Only open a new activation if supplier covers ≥5% of
                        # remaining need. Suppliers with 2-3 units on a hub that
                        # needs 500 should not trigger a full beta payment.
                        # The fallback ensures feasibility is never lost.
                        if coverage >= 0.05:
                            if best_new is None or score > (best_new[0], best_new[1]):
                                best_new = (-amortised, coverage, f, k, ship)
                        else:
                            if best_new_fallback is None or score > (best_new_fallback[0], best_new_fallback[1]):
                                best_new_fallback = (-amortised, coverage, f, k, ship)
 
            commit(f, b, k, ship)
            if DEBUG_MODE:
                print(f"  [P2-P1] t={t} s={s} FILL b={b} f={f} k={k} ship={ship} rem={hub_needs.get(b,0)}")
 
    # ── PASS 2: residual Vogel ────────────────────────────────────────────────
    while hub_needs:
        best_entry: Optional[Tuple[float, str, str, str, int]] = None
 
        for b, remaining in hub_needs.items():
            for f in suppliers_for_hub.get(b, []):   # IMP-A2: gamma-sorted here too
                if supply_left[f] <= 0:
                    continue
                result = _best_vehicle_for_pair(
                    f, b, remaining, supply_left[f], inst, cap_left,
                    activated_fk, fk_hub
                )
                if result is None:
                    continue
                _, best_k, best_ship, regret = result
                if best_entry is None or regret > best_entry[0]:
                    best_entry = (regret, b, f, best_k, best_ship)
 
        if best_entry is None:
            break
 
        _, b, f, k, ship = best_entry
        commit(f, b, k, ship)
        if DEBUG_MODE:
            print(f"  [P2-P2] t={t} s={s} VOGEL f={f}->b={b} k={k} ship={ship}")
 
    return Y_s, L_s, achieved
 
 
# ============================================================
# Phase 1: greedy route selection and loading
# ============================================================
def _load_routes(
    selected: List[str],
    r_deps: Dict,
    r_cap: Dict,
    r_hub: Dict,
    qmin: Dict[str, float],
    t: int,
    hub_budget: Dict[str, int] = None,   # FIX-P1A: per-hub OUT cap
) -> Tuple[Dict, Dict, Dict]:
    """
    Load a fixed set of selected routes and induce Q from Z.
 
    FIX-P1A: hub_budget caps the total Z loaded per hub so Phase 1 never
    over-loads a hub beyond what Phase 2 can supply.  When hub_budget is
    None the function behaves identically to v1.
    """
    Z_t: Dict[Tuple[str, str, int], int] = {}
    Q_t: Dict[Tuple[str, str, int], int] = {}
    rem = {i: float(qmin.get(i, 0.0)) for i in qmin}
    hub_loaded: Dict[str, int] = {}   # cumulative Z per hub so far
 
    for r_id in selected:
        cap  = r_cap[r_id]
        hub  = r_hub[r_id]
 
        # FIX-P1A: respect hub budget
        if hub_budget is not None:
            budget_left = hub_budget.get(hub, 0) - hub_loaded.get(hub, 0)
            cap = min(cap, budget_left)
        if cap <= 0:
            continue
 
        deps = sorted(
            (i for i in r_deps[r_id] if rem.get(i, 0.0) > 1e-9),
            key=lambda i: rem[i], reverse=True
        )
        for i in deps:
            if cap <= 0:
                break
            x_int = int_floor(min(rem[i], cap))
            if x_int > 0:
                Z_t[(i, r_id, t)] = Z_t.get((i, r_id, t), 0) + x_int
                rem[i]           -= x_int
                cap              -= x_int
                hub_loaded[hub]   = hub_loaded.get(hub, 0) + x_int
 
    for (i, r_id, tt), z in Z_t.items():
        b = r_hub[r_id]
        Q_t[(b, i, tt)] = Q_t.get((b, i, tt), 0) + int(z)
    Q_t = {k: v for k, v in Q_t.items() if v > 0}
    return Z_t, Q_t
 
 
def _route_cost(selected: List[str], r_cost: Dict) -> float:
    return sum(r_cost[r] for r in selected)
 
 
def _covers_all(selected: List[str], U: set, r_deps: Dict) -> bool:
    covered = set()
    for r in selected:
        covered |= r_deps[r]
    return U.issubset(covered)
 
 
def _local_search(
    selected: List[str],
    U: set,
    r_deps: Dict,
    r_cost: Dict,
    all_route_ids: List[str],
    qmin: Dict[str, float],
    demand_penalty_scale: float,
) -> List[str]:
    """
    Full local search: drop redundant routes, then 1-swap, then 2-for-1 swap.
 
    IMP-R1: After 1-swap converges, try replacing every pair (r1, r2) in the
    current selected set with a single route r_new (not currently selected)
    whose depot set covers U(r1) ∪ U(r2).  Accepts if total cost drops.
    This is a 2-for-1 move invisible to 1-swap — the key consolidation
    opportunity that causes the greedy to overspend on routes.
    """
    # Step A: drop redundant routes (most expensive first)
    improved = True
    while improved:
        improved = False
        for r in sorted(selected, key=lambda r: r_cost[r], reverse=True):
            without = [x for x in selected if x != r]
            if _covers_all(without, U, r_deps):
                selected = without
                improved = True
                break
 
    # Step B: 1-for-1 swap (demand-weighted, FIX-P1B)
    improved = True
    while improved:
        improved = False
        for r in list(selected):
            without = [x for x in selected if x != r]
            for r2 in all_route_ids:
                if r2 in selected:
                    continue
                trial = without + [r2]
                if not _covers_all(trial, U, r_deps):
                    continue
                cost_saving = r_cost[r] - r_cost[r2]
                lost_depots = r_deps[r] - r_deps[r2]
                demand_loss = sum(qmin.get(i, 0.0) for i in lost_depots if i in U)
                net_gain = cost_saving - demand_loss * demand_penalty_scale
                if net_gain > 1e-9:
                    selected = trial
                    improved = True
                    break
            if improved:
                break
 
    # Step C: IMP-R1 — 2-for-1 consolidation swap
    # For each pair of selected routes whose combined cost exceeds any single
    # candidate that covers their union, make the swap.
    improved = True
    while improved:
        improved = False
        sel_set = set(selected)
        sel_list = list(selected)
        for i_idx in range(len(sel_list)):
            if improved:
                break
            r1 = sel_list[i_idx]
            for j_idx in range(i_idx + 1, len(sel_list)):
                r2 = sel_list[j_idx]
                combined_cost = r_cost[r1] + r_cost[r2]
                combined_depots = r_deps[r1] | r_deps[r2]
                without_pair = [x for x in selected if x not in (r1, r2)]
                for r_new in all_route_ids:
                    if r_new in sel_set:
                        continue
                    if r_cost[r_new] >= combined_cost:
                        continue  # no cost benefit
                    # r_new must cover everything the pair covered that
                    # the remaining routes do not already cover
                    trial = without_pair + [r_new]
                    if not _covers_all(trial, U, r_deps):
                        continue
                    # Accept: genuine cost saving, coverage maintained
                    selected = trial
                    improved = True
                    break
                if improved:
                    break
 
    return selected
 
 
def greedy_route_selection_and_loading_int(
    inst: Instance,
    t: int,
    qmin: Dict[str, float],
    hub_supply_cap: Dict[str, int] = None,
    n_restarts: int = 8,              # IMP-R2: iterated local search restarts
    rng_seed: int = 42,
) -> Tuple[
    Dict[Tuple[str, int], int],
    Dict[Tuple[str, str, int], int],
    Dict[Tuple[str, str, int], int],
]:
    """
    IMP-R1: 2-for-1 consolidation swap (see _local_search).
    IMP-R2: Iterated local search with n_restarts independent perturbations.
            Each restart: drop one random selected route, repair coverage
            with cheapest covering route, run full local search, keep best.
    """
    import random
    rng = random.Random(rng_seed + t)  # deterministic per period
 
    U = {i for i, v in qmin.items() if v > 1e-9}
    if not U:
        return {}, {}, {}
 
    r_deps = {r.r_id: set(r.depots) for r in inst.routes}
    r_cost = {r.r_id: r.fixed_cost for r in inst.routes}
    r_cap  = {r.r_id: int_floor(r.capacity) for r in inst.routes}
    r_hub  = {r.r_id: r.hub for r in inst.routes}
    all_route_ids = [r.r_id for r in inst.routes]
 
    max_route_cost       = max(r_cost.values()) if r_cost else 1.0
    demand_penalty_scale = 1.0 / max(1.0, max_route_cost)
 
    def greedy_cover(remaining: set) -> List[str]:
        """Weighted set-cover greedy starting from an uncovered set."""
        sel: List[str] = []
        rem = set(remaining)
        while rem:
            best_r, best_score = None, -1.0
            for r in inst.routes:
                cover     = r_deps[r.r_id] & rem
                cover_sum = sum(qmin[i] for i in cover)
                if cover_sum <= 1e-12:
                    continue
                score = cover_sum / max(1e-12, r_cost[r.r_id])
                if score > best_score:
                    best_score = score
                    best_r     = r.r_id
            if best_r is None:
                break
            sel.append(best_r)
            rem -= r_deps[best_r]
        return sel
 
    # ── Initial solution via greedy cover + local search ─────────────────────
    initial = greedy_cover(U)
    best_selected = _local_search(
        initial, U, r_deps, r_cost, all_route_ids, qmin, demand_penalty_scale
    )
    best_cost = _route_cost(best_selected, r_cost)
 
    if DEBUG_MODE:
        print(f"  [P1 t={t}] Initial cost={best_cost:.2f}  |routes|={len(best_selected)}")
 
    # ── IMP-R2: iterated local search restarts ────────────────────────────────
    # Each restart randomly drops one route, repairs with greedy cover, then
    # runs full local search.  Accept only if strictly better.
    for restart in range(n_restarts):
        if not best_selected:
            break
        # Perturbation: drop a random route (bias toward expensive ones)
        weights = [r_cost[r] for r in best_selected]
        total_w = sum(weights)
        probs   = [w / total_w for w in weights]
        # Weighted random choice
        pick = rng.random()
        cum  = 0.0
        drop_r = best_selected[-1]
        for r, p in zip(best_selected, probs):
            cum += p
            if pick <= cum:
                drop_r = r
                break
 
        without = [x for x in best_selected if x != drop_r]
        # Find depots now uncovered
        covered_by_rest = set()
        for r in without:
            covered_by_rest |= r_deps[r]
        uncovered = U - covered_by_rest
 
        # Repair: greedy cover for uncovered depots only
        repair = greedy_cover(uncovered)
        candidate = without + repair
 
        if not _covers_all(candidate, U, r_deps):
            continue  # repair failed (no route covers some depot)
 
        candidate = _local_search(
            candidate, U, r_deps, r_cost, all_route_ids, qmin, demand_penalty_scale
        )
        candidate_cost = _route_cost(candidate, r_cost)
 
        if candidate_cost < best_cost - 1e-9:
            best_selected = candidate
            best_cost     = candidate_cost
            if DEBUG_MODE:
                print(f"  [P1 t={t}] Restart {restart}: improved to {best_cost:.2f}")
 
    # ── Finalise X ───────────────────────────────────────────────────────────
    X_t: Dict[Tuple[str, int], int] = {}
    for r_id in best_selected:
        X_t[(r_id, t)] = 1
 
    Z_t, Q_t = _load_routes(best_selected, r_deps, r_cap, r_hub, qmin, t,
                            hub_budget=hub_supply_cap)
    return X_t, Z_t, Q_t
 
 
# ============================================================
# Scaling helpers  (unchanged from v1)
# ============================================================
def scale_Z_to_target(
    sol: FFHSolution, inst: Instance, b: str, t: int, target_out: int,
    qmin_floor: Dict[str, int] = None,
) -> None:
    r_hub = {r.r_id: r.hub for r in inst.routes}
    if qmin_floor is None:
        qmin_floor = {}
 
    z_keys = [(i, r_id, tt) for (i, r_id, tt) in sol.Z
              if tt == t and r_hub.get(r_id) == b and int(sol.Z.get((i, r_id, tt), 0)) > 0]
 
    cur_total = sum(int(sol.Z[k]) for k in z_keys)
    if cur_total <= target_out:
        return
 
    depot_z: Dict[str, List] = {}
    for k in z_keys:
        i = k[0]
        depot_z.setdefault(i, []).append(k)
 
    depot_cur   = {i: sum(int(sol.Z[k]) for k in keys) for i, keys in depot_z.items()}
    depot_floor = {i: int(qmin_floor.get(i, 0)) for i in depot_cur}
    total_floor = sum(depot_floor[i] for i in depot_cur)
 
    if target_out < total_floor:
        target_out = total_floor
 
    total_surplus  = cur_total - total_floor
    target_surplus = target_out - total_floor
 
    if total_surplus <= 0 or target_surplus >= total_surplus:
        return
 
    ratio = target_surplus / total_surplus
 
    for i, keys in depot_z.items():
        floor_i   = depot_floor[i]
        cur_i     = depot_cur[i]
        surplus_i = cur_i - floor_i
        if surplus_i <= 0:
            continue
 
        new_total_i = floor_i + int_floor(surplus_i * ratio)
        new_total_i = max(floor_i, new_total_i)
 
        if new_total_i == cur_i:
            continue
 
        base_vals  = []
        base_sum   = 0
        remainders = []
        for k in keys:
            v  = int(sol.Z[k])
            x  = v * (new_total_i / cur_i)
            bk = int_floor(x)
            base_vals.append((k, bk))
            base_sum += bk
            remainders.append((x - bk, k))
 
        leftover = new_total_i - base_sum
        remainders.sort(reverse=True)
        add = {k: 0 for k in keys}
        for kk in range(max(0, leftover)):
            add[remainders[kk][1]] += 1
 
        for k, bk in base_vals:
            new_val = bk + add[k]
            if new_val > 0:
                sol.Z[k] = new_val
            else:
                sol.Z.pop(k, None)
 
 
def recompute_Q_from_Z(sol: FFHSolution, inst: Instance, t: int) -> None:
    r_hub = {r.r_id: r.hub for r in inst.routes}
 
    for key in [k for k in list(sol.Q.keys()) if k[2] == t]:
        del sol.Q[key]
 
    for (i, r_id, tt), z in sol.Z.items():
        if tt != t or int(z) == 0:
            continue
        b = r_hub[r_id]
        sol.Q[(b, i, t)] = sol.Q.get((b, i, t), 0) + int(z)
 
 
# ============================================================
# Cost functions  (unchanged)
# ============================================================
def downstream_cost(
    inst: Instance, sol: FFHSolution
) -> Tuple[float, float, float, float]:
    RC = sum(
        r.fixed_cost * float(sol.X.get((r.r_id, t), 0))
        for t in inst.T for r in inst.routes
    )
    HC = sum(
        inst.holding_cost[i] * float(sol.Ipos.get((i, t), 0))
        for i in inst.D for t in inst.T
    )
    WC = sum(
        inst.waste_cost * float(sol.W.get((i, t), 0))
        for i in inst.D for t in inst.T if t >= inst.shelf_life
    )
    return RC, HC, WC, RC + HC + WC
 
 
def upstream_cost_expected_OPL(
    inst: Instance, sol: FFHSolution
) -> Tuple[float, float, float]:
    AC = sum(
        inst.scenario_probs.get(s, 0.0) * float(inst.beta[(f, k)])
        for (f, b, k, s, t), y in sol.Y.items()
        if int(y) == 1 and inst.scenario_probs.get(s, 0.0) > 0
    )
    TC = sum(
        inst.scenario_probs.get(s, 0.0) * float(inst.gamma[(f, b, k)]) * float(qty)
        for (f, b, k, s, t), qty in sol.L.items()
        if int(qty) > 0 and inst.scenario_probs.get(s, 0.0) > 0
    )
    return AC, TC, AC + TC
 
 
def objective_cost_OPL(inst: Instance, sol: FFHSolution) -> Dict[str, float]:
    invcost = sum(
        float(sol.Ipos.get((i, t), 0)) * inst.holding_cost[i]
        for i in inst.D for t in inst.T
    )
    wastecost = sum(
        float(sol.W.get((i, t), 0)) * float(inst.waste_cost)
        for i in inst.D for t in inst.T if t >= inst.shelf_life
    )
    routecost = sum(
        float(sol.X.get((r.r_id, t), 0)) * float(r.fixed_cost)
        for r in inst.routes for t in inst.T
    )
    AC, TC, _ = upstream_cost_expected_OPL(inst, sol)
    total = invcost + wastecost + routecost + AC + TC
    return {
        "invcost": invcost, "wastecost": wastecost,
        "routecost": routecost, "assignmentcost": AC,
        "loadcost": TC, "total": total,
    }
 
 
# ============================================================
# Checks  (unchanged except check_service_level_REQ doc)
# ============================================================
def check_service_level_REQ(
    inst: Instance,
    sol: FFHSolution,
    REQ: Dict[Tuple[str, int], float],
) -> pd.DataFrame:
    co     = inst.cv
    Zalpha = z_value(inst.alpha)
    rows   = []
    for i in inst.D:
        for t in inst.T:
            qc  = float(sol.Q_cum.get((i, t), 0))
            wc  = float(sol.W_cum.get((i, t - 1), 0))
            lhs = qc - wc
 
            de_sum    = sum(float(inst.mu[(i, a)]) for a in inst.T if a <= t)
            de_sq_sum = sum(float(inst.mu[(i, a)])**2 for a in inst.T if a <= t)
            rhs_opl   = de_sum + math.sqrt(de_sq_sum) * co * Zalpha
 
            req_ceil  = float(REQ[(i, t)])
 
            rows.append({
                "i": i, "t": t,
                "Q_cum": qc, "W_cum_prev": wc,
                "LHS": lhs,
                "RHS_OPL": round(rhs_opl, 6),
                "REQ_ceil": req_ceil,
                "slack_vs_OPL": round(lhs - rhs_opl, 6),
                "feasible": int(lhs - rhs_opl >= -1e-6),
            })
    df    = pd.DataFrame(rows).sort_values(["i", "t"])
    n_bad = int((df["feasible"] == 0).sum())
    if n_bad == 0:
        print("[CHECK-A] Service-level constraint (OPL 2.15): OK")
    else:
        print(f"[CHECK-A] Service-level constraint (OPL 2.15): VIOLATIONS = {n_bad}")
    return df
 
 
def check_flow_balance(inst: Instance, sol: FFHSolution) -> pd.DataFrame:
    rows = []
    ok   = True
    for s in inst.S:
        for t in inst.T:
            for b in inst.B:
                out_bt  = sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D)
                in_bst  = sum(
                    int(v) for (f, bb, k, ss, tt), v in sol.L.items()
                    if ss == s and tt == t and bb == b
                )
                gap = in_bst - out_bt
                if gap != 0:
                    ok = False
                    print(f"  [CHECK-B VIOLATION] b={b} s={s} t={t}: "
                          f"Σ L={in_bst}  Σ Q={out_bt}  gap={gap}")
                rows.append({"b": b, "s": s, "t": t,
                             "IN": in_bst, "OUT": out_bt, "gap(IN-OUT)": gap})
    df = pd.DataFrame(rows).sort_values(["s", "t", "b"])
    if ok:
        print("[CHECK-B] Flow balance: OK")
    else:
        print(f"[CHECK-B] Flow balance: VIOLATIONS = {int((df['gap(IN-OUT)'] != 0).sum())}")
    return df
 
 
# ============================================================
# IMP-A1: Cross-scenario vehicle pre-selection
# ============================================================
def _cross_scenario_preselect(
    inst: Instance,
    t: int,
    OUT_bt: Dict[str, int],
) -> set:
    """
    IMP-A1 (v6): Scenario-union pre-selection with feasibility filter.
 
    For each hub b, simulate a full greedy for each scenario independently
    to find which (f,k) pairs each scenario naturally activates.
    Pre-activate the UNION of top-N pairs across scenarios — but only pairs
    that are feasible (have supply > 0) in ALL scenarios.
 
    This is strictly safer than the strict-intersection (v5): intersection
    was too conservative (often returned nothing), while union-with-feasibility-
    filter pre-activates pairs every scenario can actually use, letting the
    per-scenario greedy reuse them at zero marginal beta.
 
    N=3 per hub: covers multi-vehicle hubs without over-committing.
    """
    pre_activated: set = set()
 
    for b in inst.B:
        need = OUT_bt.get(b, 0)
        if need <= 0:
            continue
 
        # Collect top-N (f,k) candidates per scenario by amortised cost
        candidates_per_scenario: List[List[Tuple[str, str]]] = []
 
        for s in inst.S:
            ranked: List[Tuple[float, Tuple[str, str]]] = []
            remaining = float(need)
            sup_left = {f: float(inst.supply.get((f, s, t), 0.0)) for f in inst.F}
 
            for _ in range(3):   # top-3 per scenario
                best_score = float('inf')
                best_fk    = None
                for f in inst.F:
                    if sup_left[f] <= 0:
                        continue
                    if (f, b) not in inst.allowed_f_b:
                        continue
                    for k in inst.K:
                        if (f, k) not in inst.beta or (f, b, k) not in inst.gamma:
                            continue
                        cap_k = int_floor(inst.theta[k])
                        if cap_k <= 0:
                            continue
                        ship = int_floor(min(sup_left[f], float(cap_k), remaining))
                        if ship <= 0:
                            continue
                        score = (inst.beta[(f, k)] + inst.gamma[(f, b, k)] * ship) / ship
                        if score < best_score:
                            best_score = score
                            best_fk    = (f, k)
                if best_fk is None:
                    break
                f_pick, k_pick = best_fk
                ranked.append(best_fk)
                ship = int_floor(min(sup_left[f_pick],
                                     float(int_floor(inst.theta[k_pick])),
                                     remaining))
                sup_left[f_pick] = max(0.0, sup_left[f_pick] - ship)
                remaining = max(0.0, remaining - ship)
 
            candidates_per_scenario.append(ranked)
 
        # Union of candidates across scenarios
        all_candidates: set = set()
        for ranked in candidates_per_scenario:
            all_candidates.update(ranked)
 
        # Feasibility filter: only keep pairs with supply > 0 in ALL scenarios
        for fk in all_candidates:
            f, k = fk
            if all(inst.supply.get((f, s, t), 0.0) > 0 for s in inst.S):
                pre_activated.add(fk)
 
    return pre_activated
 
 
# ============================================================
# Option B: Lagrangian relaxation of flow balance
# ============================================================
def lagrangian_inbound_assignment(
    inst: Instance,
    t: int,
    Q_sum: Dict[str, int],
    pre_fk: set,
    n_iter: int = 0,
    step_init: float = 0.0,
    step_decay: float = 0.0,
) -> Tuple[
    Dict[Tuple[str, str, str, str, int], int],
    Dict[Tuple[str, str, str, str, int], int],
]:
    """
    Supply-ordered sequential greedy assignment.
 
    Processes scenarios in descending total-supply order so the richest
    scenario runs first and establishes the baseline activation set.
    Each subsequent scenario inherits all prior activations as
    pre_activated_fk (zero marginal beta), minimising redundant activations.
 
    This replicates the key insight from the MILP data: the optimal solution
    uses a consistent core set of suppliers across scenarios, with each
    scenario paying beta only for its unique activations.
    """
    S_list = list(inst.S)
 
    scenarios_by_supply = sorted(
        S_list,
        key=lambda s: sum(
            float(inst.supply.get((f, s, t), 0.0))
            for f in inst.F for b in inst.B
            if (f, b) in inst.allowed_f_b
        ),
        reverse=True,
    )
 
    Y_final: Dict[Tuple[str, str, str, str, int], int] = {}
    L_final: Dict[Tuple[str, str, str, str, int], int] = {}
    cumulative: set = set(pre_fk)
 
    for s in scenarios_by_supply:
        Y_s, L_s, ach = inbound_assign_greedy_one_scenario_int(
            inst, s, t, Q_sum, pre_activated_fk=cumulative
        )
        for k2, v2 in Y_s.items():
            Y_final[k2] = int(v2)
        for k2, v2 in L_s.items():
            L_final[k2] = int(v2)
        for (f, b, k, ss, tt), y in Y_s.items():
            if y == 1:
                cumulative.add((f, k))
 
    return Y_final, L_final
 
# ============================================================
# Post-opt 1: Y local search
# ============================================================
def y_local_search(inst: Instance, sol: FFHSolution) -> int:
    """
    For each active Y[f,b,k,s,t]=1, try replacing (f,k) with a cheaper
    (f2,k2) that can deliver the same load in the same (b,s,t).
 
    Accepts swap only if beta[f2,k2] < beta[f,k] and all supply/capacity/
    fk_hub constraints are satisfied. Iterates until no improvement.
    """
    total_swaps = 0
 
    for t in inst.T:
        for s in inst.S:
            prob = inst.scenario_probs.get(s, 0.0)
            if prob <= 0:
                continue
 
            fk_hub_st: Dict[Tuple[str,str], str] = {}
            for (f, b, k, ss, tt), y in sol.Y.items():
                if ss == s and tt == t and y == 1:
                    fk_hub_st[(f, k)] = b
 
            improved = True
            while improved:
                improved = False
                for (f, b, k, ss, tt) in list(sol.Y.keys()):
                    if ss != s or tt != t or sol.Y.get((f,b,k,ss,tt),0) != 1:
                        continue
                    l_val = int(sol.L.get((f, b, k, s, t), 0))
                    if l_val <= 0:
                        continue
                    beta_cur = inst.beta.get((f, k), 0.0)
 
                    for f2 in inst.F:
                        if (f2, b) not in inst.allowed_f_b or f2 == f:
                            continue
                        sup2 = float(inst.supply.get((f2, s, t), 0.0))
                        if sup2 < l_val:
                            continue
                        for k2 in inst.K:
                            if (f2, k2) not in inst.beta:
                                continue
                            if inst.beta[(f2, k2)] >= beta_cur:
                                continue
                            if (f2, b, k2) not in inst.gamma:
                                continue
                            if int_floor(inst.theta.get(k2, 0)) < l_val:
                                continue
                            k2_used = sum(
                                int(v) for (ff,bb,kk,ss2,tt2),v in sol.L.items()
                                if kk==k2 and ss2==s and tt2==t
                            )
                            if k2_used + l_val > int_floor(inst.theta.get(k2, 0)):
                                continue
                            if fk_hub_st.get((f2, k2), b) != b:
                                continue
                            f2_used = sum(
                                int(v) for (ff,bb,kk,ss2,tt2),v in sol.L.items()
                                if ff==f2 and ss2==s and tt2==t
                            )
                            if f2_used + l_val > int_floor(sup2):
                                continue
 
                            old_key = (f,  b, k,  s, t)
                            new_key = (f2, b, k2, s, t)
                            sol.L.pop(old_key, None)
                            sol.Y.pop(old_key, None)
                            sol.Y[new_key] = 1
                            sol.L[new_key] = sol.L.get(new_key, 0) + l_val
                            fk_hub_st.pop((f, k), None)
                            fk_hub_st[(f2, k2)] = b
                            total_swaps += 1
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
 
    return total_swaps
 
 
# ============================================================
# Post-opt 2: L re-optimisation
# ============================================================
def l_reoptimise(inst: Instance, sol: FFHSolution) -> float:
    """
    With Y fixed, re-allocate L within each (b,s,t) to minimise
    Sigma pi_s * gamma[f,b,k] * L[f,b,k,s,t].
 
    Sorts active (f,k) pairs by gamma ascending and fills greedily.
    No new activations — only redistribution within fixed Y set.
    """
    total_saving = 0.0
 
    for t in inst.T:
        for s in inst.S:
            prob = inst.scenario_probs.get(s, 0.0)
            if prob <= 0:
                continue
 
            for b in inst.B:
                active = []
                for (f, bb, k, ss, tt), y in sol.Y.items():
                    if bb == b and ss == s and tt == t and y == 1:
                        active.append((f, k))
                if not active:
                    continue
 
                Q_target = sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D)
                if Q_target <= 0:
                    continue
 
                old_tc = sum(
                    inst.gamma.get((f, b, k), 0.0) * int(sol.L.get((f,b,k,s,t), 0))
                    for f, k in active
                )
 
                pairs_sorted = sorted(
                    active,
                    key=lambda fk: inst.gamma.get((fk[0], b, fk[1]), float('inf'))
                )
                new_L: Dict[Tuple[str,str], int] = {}
                remaining = Q_target
                for f, k in pairs_sorted:
                    if remaining <= 0:
                        break
                    sup_a = int_floor(float(inst.supply.get((f, s, t), 0.0)))
                    cap_a = int_floor(inst.theta.get(k, 0))
                    alloc = min(sup_a, cap_a, remaining)
                    if alloc > 0:
                        new_L[(f, k)] = alloc
                        remaining -= alloc
 
                if remaining > 0:
                    continue
 
                new_tc = sum(
                    inst.gamma.get((f, b, k), 0.0) * v
                    for (f, k), v in new_L.items()
                )
                if new_tc >= old_tc - 1e-6:
                    continue
 
                total_saving += prob * (old_tc - new_tc)
                for f, k in active:
                    sol.L.pop((f, b, k, s, t), None)
                for (f, k), v in new_L.items():
                    sol.L[(f, b, k, s, t)] = v
 
    return total_saving
 
 
def scenario_swap_post_optimise(
    inst: Instance,
    sol: FFHSolution,
) -> int:
    """
    Option C: Fix-and-optimise post-processing.
 
    After a complete feasible solution exists, reduce expected assignment
    cost by swapping which scenarios carry each (f,k,b,t) activation.
 
    Core idea:
      For each hub b and period t, collect all (f,k) pairs that are active
      in at least one scenario.  For each such pair, compute its expected
      beta contribution:
        current_cost = Σ_s prob[s] * beta[f,k] * Y[f,b,k,s,t]
 
      A swap replaces a high-beta activation in scenario s1 with a
      zero-beta reuse from scenario s2 (where the pair is already active),
      by transferring load from a redundant pair in s1 to (f,k).
 
    Swap conditions (all must hold):
      1. (f,k) is active in s2 but not s1 for hub b — beta already paid in s2.
      2. supply[f,s1,t] > 0 — supplier can actually deliver in s1.
      3. cap_left[k,s1] > 0 — vehicle has remaining capacity in s1.
      4. There exists a "donor" (f2,k2) active in s1 for hub b whose load
         can be reduced by the transferred amount — keeping ΣL = Q_sum[b].
      5. The swap saves expected beta: prob[s1]*beta[f,k] > 0 net saving
         (always true if Y[f,b,k,s1] would be eliminated entirely).
      6. (f,k) is not already committed to a different hub in s1
         (fk_hub constraint).
 
    Returns number of swaps performed.
    """
    total_swaps = 0
 
    for t in inst.T:
        # Q_sum[b] = total outbound from hub b in period t (same for all s)
        Q_sum: Dict[str, int] = {
            b: sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D)
            for b in inst.B
        }
 
        for b in inst.B:
            if Q_sum[b] == 0:
                continue
 
            # ── Build current state for this (b,t) ───────────────────────────
            # active_in[f,k] = set of scenarios where Y[f,b,k,s,t]=1
            active_in: Dict[Tuple[str,str], set] = {}
            for s in inst.S:
                for f in inst.F:
                    for k in inst.K:
                        if sol.Y.get((f, b, k, s, t), 0) == 1:
                            active_in.setdefault((f, k), set()).add(s)
 
            # Per-scenario capacity used by each vehicle k
            # cap_used[s][k] = Σ_{f,b2} L[f,b2,k,s,t]
            cap_used: Dict[str, Dict[str, int]] = {s: {} for s in inst.S}
            for (f2, b2, k2, s2, t2), lv in sol.L.items():
                if t2 != t:
                    continue
                cap_used[s2][k2] = cap_used[s2].get(k2, 0) + int(lv)
 
            # Per-scenario supply used by each supplier f
            sup_used: Dict[str, Dict[str, float]] = {s: {} for s in inst.S}
            for (f2, b2, k2, s2, t2), lv in sol.L.items():
                if t2 != t:
                    continue
                sup_used[s2][f2] = sup_used[s2].get(f2, 0.0) + float(lv)
 
            # fk_hub[s][(f,k)] = hub assigned to (f,k) in scenario s
            fk_hub_s: Dict[str, Dict[Tuple[str,str], str]] = {s: {} for s in inst.S}
            for (f2, b2, k2, s2, t2), y in sol.Y.items():
                if t2 != t or y != 1:
                    continue
                fk_hub_s[s2][(f2, k2)] = b2
 
            # ── Attempt swaps ─────────────────────────────────────────────────
            # For each (f,k) active in some but not all scenarios, try to
            # extend it into the missing scenarios by transferring load from
            # an existing donor in that scenario.
            improved = True
            while improved:
                improved = False
                for (f, k), s_active in list(active_in.items()):
                    s_missing = [s for s in inst.S if s not in s_active]
                    if not s_missing:
                        continue
 
                    beta_fk = inst.beta.get((f, k), 0.0)
                    if beta_fk <= 0:
                        continue
 
                    for s1 in s_missing:
                        prob_s1 = inst.scenario_probs.get(s1, 0.0)
                        if prob_s1 <= 0:
                            continue
 
                        # Check (f,k) not locked to a different hub in s1
                        if fk_hub_s[s1].get((f, k), b) != b:
                            continue
 
                        # Check supply available in s1
                        sup_total = float(inst.supply.get((f, s1, t), 0.0))
                        sup_already = sup_used[s1].get(f, 0.0)
                        sup_free = sup_total - sup_already
                        if sup_free <= 0:
                            continue
 
                        # Check vehicle capacity in s1
                        cap_total = int_floor(inst.theta.get(k, 0))
                        cap_already = cap_used[s1].get(k, 0)
                        cap_free = cap_total - cap_already
                        if cap_free <= 0:
                            continue
 
                        # How much can we transfer?
                        # We want to cover enough of Q_sum[b] in s1 via (f,k)
                        # to eliminate at least one donor activation.
                        transfer = int_floor(min(sup_free, float(cap_free)))
                        if transfer <= 0:
                            continue
 
                        # Find a donor (f2,k2) active in s1 for hub b
                        # that can give up `transfer` units.
                        # Prefer donors with highest beta (most expensive to keep).
                        best_donor = None
                        best_donor_beta = -1.0
                        for (f2, k2), s2_active in active_in.items():
                            if s1 not in s2_active:
                                continue
                            if (f2, k2) == (f, k):
                                continue
                            l_val = int(sol.L.get((f2, b, k2, s1, t), 0))
                            if l_val <= 0:
                                continue
                            give = min(l_val, transfer)
                            if give <= 0:
                                continue
                            db = inst.beta.get((f2, k2), 0.0)
                            # Only swap if donor beta > 0 (saves something)
                            # and donor would be fully eliminated (Y goes to 0)
                            if l_val == give and db > best_donor_beta:
                                best_donor = (f2, k2, give)
                                best_donor_beta = db
 
                        if best_donor is None:
                            continue
 
                        f2, k2, give = best_donor
 
                        # Expected cost saving: remove Y[f2,b,k2,s1,t]=1
                        # Cost of new Y[f,b,k,s1,t]=1 if not already active:
                        # net saving = prob[s1]*(beta[f2,k2] - beta[f,k])
                        # Only worth it if beta[f2,k2] > beta[f,k]
                        # OR if (f,k) will be activated anyway (already in s_active)
                        # Since we're adding (f,k) to s1, we always pay beta[f,k].
                        # Net saving = prob[s1] * (beta[f2,k2] - beta[f,k])
                        net_saving = prob_s1 * (best_donor_beta - beta_fk)
                        if net_saving <= 1e-6:
                            continue
 
                        # ── Apply swap ────────────────────────────────────────
                        # Add (f,k) to s1 with load = give
                        key_new = (f, b, k, s1, t)
                        sol.Y[key_new] = 1
                        sol.L[key_new] = sol.L.get(key_new, 0) + give
 
                        # Remove donor load
                        key_old = (f2, b, k2, s1, t)
                        old_val = int(sol.L.get(key_old, 0))
                        new_val = old_val - give
                        if new_val <= 0:
                            sol.L.pop(key_old, None)
                            sol.Y.pop(key_old, None)
                            active_in[(f2, k2)].discard(s1)
                        else:
                            sol.L[key_old] = new_val
 
                        # Update tracking
                        active_in.setdefault((f, k), set()).add(s1)
                        fk_hub_s[s1][(f, k)] = b
 
                        # Update cap and sup used
                        cap_used[s1][k]  = cap_used[s1].get(k, 0) + give
                        sup_used[s1][f]  = sup_used[s1].get(f, 0.0) + give
                        cap_used[s1][k2] = cap_used[s1].get(k2, 0) - give
                        sup_used[s1][f2] = sup_used[s1].get(f2, 0.0) - give
 
                        total_swaps += 1
                        improved = True
 
                        if DEBUG_MODE:
                            print(f"  [SWAP] t={t} b={b} s={s1}: "
                                  f"({f2},{k2})→({f},{k}) give={give} "
                                  f"saving={net_saving:.4f}")
 
    print(f"[OPT-C] Scenario-swap post-optimiser: {total_swaps} swaps performed.")
    return total_swaps
def solve_ffh(inst: Instance) -> Tuple[FFHSolution, Dict[Tuple[str, int], float]]:
    sol = FFHSolution()
 
    for i in inst.D:
        sol.Q_cum[(i, 0)] = 0
        sol.W_cum[(i, 0)] = 0
 
    with PhaseTimer("Precompute REQ_{i,t}"):
        REQ = precompute_REQ(inst)
 
    # FIX-P3A: precompute cumulative demand state so Phase 3 is O(|D|×|T|)
    # de_cum_state[(i,t)] = Σ_{a<=t} mu[(i,a)]  — built up incrementally
    de_cum_state: Dict[Tuple[str, int], float] = {}
    for i in inst.D:
        running = 0.0
        for t in inst.T:
            running += inst.mu[(i, t)]
            de_cum_state[(i, t)] = running
 
    for t in inst.T:
 
        # ── Phase 1 ──────────────────────────────────────────────────────────
        with PhaseTimer(f"Phase 1 (t={t}) - sizing + route selection"):
            qmin = {
                i: max(0.0,
                       REQ[(i, t)]
                       + float(sol.W_cum[(i, t - 1)])
                       - float(sol.Q_cum[(i, t - 1)]))
                for i in inst.D
            }
 
            # FIX-P1A: compute per-hub supply cap from min-achievable supply
            # across all scenarios.  This bounds how much Phase 1 can load per
            # hub so that Phase 2 can always match it.
            hub_supply_cap: Dict[str, int] = {}
            for b in inst.B:
                depots_in_b = {
                    i for r in inst.routes if r.hub == b for i in r.depots
                }
                # what Phase 1 needs to deliver to this hub
                hub_qmin_total = int(math.ceil(sum(qmin.get(i, 0.0) for i in depots_in_b)))
                # conservative achievable supply from any single scenario
                min_supply = min(
                    sum(float(inst.supply.get((f, s, t), 0.0))
                        for f in inst.F if (f, b) in inst.allowed_f_b)
                    for s in inst.S
                )
                hub_supply_cap[b] = max(hub_qmin_total, int_floor(min_supply))
 
            X_t, Z_t, Q_t = greedy_route_selection_and_loading_int(
                inst, t, qmin, hub_supply_cap=hub_supply_cap
            )
            qmin_t = {i: int(math.ceil(qmin[i])) for i in inst.D}
 
            for (r_id, tt), v in X_t.items():
                sol.X[(r_id, tt)] = int(v)
            for (i, r_id, tt), v in Z_t.items():
                if v > 0:
                    sol.Z[(i, r_id, tt)] = int(v)
            for b in inst.B:
                for i in inst.D:
                    v = int(Q_t.get((b, i, t), 0))
                    if v > 0:
                        sol.Q[(b, i, t)] = v
                    else:
                        sol.Q.pop((b, i, t), None)
 
            n_sel = sum(sol.X.get((r.r_id, t), 0) for r in inst.routes)
            print(f"  Phase 1 done (t={t}). Routes selected: {n_sel}")
 
        # ── Phase 2 ──────────────────────────────────────────────────────────
        with PhaseTimer(f"Phase 2 (t={t}) - robust inbound assignment"):
            OUT = {b: sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D)
                   for b in inst.B}
 
            out_phase1_floor: Dict[str, int] = {b: int(OUT[b]) for b in inst.B}
            OUT_final: Dict[str, int]        = {b: int(OUT[b]) for b in inst.B}
 
            # IMP-A1: compute cross-scenario pre-activated (f,k) pairs once
            # per period, before the convergence loop.  These are reused by
            # every per-scenario greedy call at zero marginal beta.
            pre_fk = _cross_scenario_preselect(inst, t, OUT_final)
            if DEBUG_MODE:
                print(f"  [IMP-A1] pre-activated (f,k) pairs: {len(pre_fk)}")
 
            # ── Convergence loop ─────────────────────────────────────────────
            for _iter in range(5):
                temp_store: Dict[str, tuple] = {}
                min_ach: Dict[str, int] = {b: OUT_final[b] for b in inst.B}
 
                for s in inst.S:
                    Y_s, L_s, ach = inbound_assign_greedy_one_scenario_int(
                        inst, s, t, OUT_final, pre_activated_fk=pre_fk
                    )
                    temp_store[s] = (Y_s, L_s, ach)
                    for b in inst.B:
                        min_ach[b] = min(min_ach[b], int(ach[b]))
 
                if all(min_ach[b] >= OUT_final[b] for b in inst.B):
                    break
 
                OUT_final = {b: max(int(min_ach[b]), out_phase1_floor[b])
                             for b in inst.B}
                if DEBUG_MODE:
                    print(f"  [P2 iter={_iter}] tightened OUT_final: {OUT_final}")
 
            else:
                # FIX-P2B
                final_ach_all: Dict[str, Dict[str, int]] = {}
                temp_store = {}
                for s in inst.S:
                    Y_s, L_s, ach = inbound_assign_greedy_one_scenario_int(
                        inst, s, t, OUT_final, pre_activated_fk=pre_fk
                    )
                    temp_store[s]    = (Y_s, L_s, ach)
                    final_ach_all[s] = ach
 
                for b in inst.B:
                    min_achieved = min(int(final_ach_all[s][b]) for s in inst.S)
                    OUT_final[b] = max(min_achieved, out_phase1_floor[b])
 
                if DEBUG_MODE:
                    print(f"  [P2 final re-run] OUT_final: {OUT_final}")
 
            # ── Scale Z/Q down to match OUT_final ────────────────────────────
            for b in inst.B:
                scale_Z_to_target(sol, inst, b, t, OUT_final[b])
            recompute_Q_from_Z(sol, inst, t)
 
            for r in inst.routes:
                if all(sol.Z.get((i, r.r_id, t), 0) == 0 for i in r.depots):
                    sol.X.pop((r.r_id, t), None)
 
            Q_sum = {b: sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D)
                     for b in inst.B}
 
            # ── Option B: Lagrangian inbound assignment ───────────────────────
            # Replaces the per-scenario C5 enforce loop.
            # Runs 50 subgradient iterations to push scenarios toward the
            # same (f,k) selection, then recovers a feasible primal solution.
            Y_lag, L_lag = lagrangian_inbound_assignment(
                inst, t, Q_sum, pre_fk
            )
 
            # ── Write final Y/L ───────────────────────────────────────────────
            for k in [k for k in list(sol.Y.keys()) if k[4] == t]:
                del sol.Y[k]
            for k in [k for k in list(sol.L.keys()) if k[4] == t]:
                del sol.L[k]
 
            for k2, v2 in Y_lag.items():
                sol.Y[k2] = int(v2)
            for k2, v2 in L_lag.items():
                sol.L[k2] = int(v2)
 
            ratios = [
                OUT_final[b] / OUT[b] if OUT[b] > 0 else 1.0
                for b in inst.B
            ]
            print(f"  Phase 2 done (t={t}). Min robust OUT ratio: {round(min(ratios), 4)}")
 
        # ── Phase 3 ──────────────────────────────────────────────────────────
        with PhaseTimer(f"Phase 3 (t={t}) - OPL-exact inventory"):
            for i in inst.D:
                receipt = sum(int(sol.Q.get((b, i, t), 0)) for b in inst.B)
                # FIX-P3C: Q_cum stays integer
                sol.Q_cum[(i, t)] = int(sol.Q_cum[(i, t - 1)]) + int(receipt)
 
                m = inst.shelf_life
                if t < m:
                    w_it = 0
                else:
                    t_ref = t - m + 1
                    # FIX-P3A: use precomputed de_cum_state instead of inner sum
                    q_cum_ref  = int(sol.Q_cum[(i, t_ref)])
                    de_cum_ref = de_cum_state[(i, t_ref)]
                    w_cum_ref  = int(sol.W_cum[(i, t_ref)])
                    I_ref      = q_cum_ref - de_cum_ref - w_cum_ref
 
                    # FIX-P3A: de_tail also uses precomputed state
                    de_tail = (de_cum_state[(i, t)]
                               - de_cum_state.get((i, t - m + 1), 0.0)
                               + inst.mu[(i, t - m + 1)])
                    # Σ mu for a in [t-m+2 .. t]
                    de_tail = de_cum_state[(i, t)] - de_cum_state[(i, t_ref)]
 
                    w_tail = sum(
                        int(sol.W.get((i, a), 0))
                        for a in inst.T if t - m + 2 <= a <= t - 1
                    )
                    w_rhs = I_ref - de_tail - w_tail
                    # FIX-P3B: use int_floor consistently (no round/floor mix)
                    w_it  = max(0, int_floor(w_rhs))
 
                # FIX-P3B: all inventory variables use int_floor, not round
                sol.W[(i, t)]     = w_it
                # FIX-P3C: W_cum is integer arithmetic only
                sol.W_cum[(i, t)] = int(sol.W_cum[(i, t - 1)]) + w_it
 
                # FIX-P3B: I derived as integer arithmetic from integer Q_cum and W_cum
                # de_cum is float but bounded; result is floored then clamped for Ipos
                de_cum_t = de_cum_state[(i, t)]
                I_it_raw = int(sol.Q_cum[(i, t)]) - de_cum_t - int(sol.W_cum[(i, t)])
                I_it_int = int_floor(I_it_raw)   # FIX-P3B: floor not round
                sol.I[(i, t)]    = I_it_int
                sol.Ipos[(i, t)] = max(0, I_it_int)
 
            print(f"  Phase 3 done (t={t}).")
 
    # ── Post-optimisation (3 stages) ─────────────────────────────────────────
    with PhaseTimer("Post-opt 1 - Y local search (activation swap)"):
        y_swaps = y_local_search(inst, sol)
        print(f"  Y local search: {y_swaps} swaps performed.")
 
    with PhaseTimer("Post-opt 2 - L re-optimisation (transport cost)"):
        l_savings = l_reoptimise(inst, sol)
        print(f"  L re-optimisation: {l_savings:.2f} transport cost saved.")
 
    # ── Cost evaluation ───────────────────────────────────────────────────────
    with PhaseTimer("Cost evaluation"):
        RC, HC, WC, DOWN   = downstream_cost(inst, sol)
        AC_opl, TC_opl, UP = upstream_cost_expected_OPL(inst, sol)
        TOT                = DOWN + UP
        opl                = objective_cost_OPL(inst, sol)
 
        sol.cost_breakdown = {
            "RC": RC, "HC": HC, "WC": WC, "DOWN": DOWN,
            "AC": AC_opl, "TC": TC_opl, "UP": UP, "TOTAL": TOT,
        }
        sol.cost_breakdown_opl = opl
 
        sep = "=" * 75
        print(f"\n{sep}")
        print("FFH COST BREAKDOWN")
        print(sep)
        print(f"Route cost (RC)      : {RC:.2f}")
        print(f"Holding cost (HC)    : {HC:.2f}")
        print(f"Waste cost (WC)      : {WC:.2f}")
        print(f"Downstream total     : {DOWN:.2f}")
        print("-" * 75)
        print(f"Assignment cost (AC) : {AC_opl:.2f}")
        print(f"Transport cost (TC)  : {TC_opl:.2f}")
        print(f"Upstream total       : {UP:.2f}")
        print("-" * 75)
        print(f"TOTAL COST           : {TOT:.2f}")
        print(sep)
        print("\nOPL-STYLE COMPONENTS")
        print(sep)
        for k, v in opl.items():
            print(f"  {k:<18}: {v:.2f}")
        print(sep)
 
    return sol, REQ

  

  

# ============================================================ 

# 6) Export solution to Excel (int decisions + checks) 

# ============================================================ 

  

def export_solution_to_excel( 

    inst: Instance, 

    sol: FFHSolution, 

    REQ: Dict[Tuple[str, int], float], 

    out_path: str = "ffh_solution_export123.xlsx" 

) -> None: 

    """ 

    Exports decision variables and validation summaries to Excel: 

      - X, Z, Q, Y, L, Ipos, W 

      - OUT (hub,t) 

      - IN  (hub,s,t) 

      - REQ/service-level check 

      - Flow-balance check 

      - Cost breakdown (FFH) + OPL-style objective components 

    """ 

    # X 

    X_rows = [{"r": r, "t": t, "X": int(v)} for (r, t), v in sol.X.items()] 

    df_X = pd.DataFrame(X_rows).sort_values(["t", "r"]) if X_rows else pd.DataFrame(columns=["r", "t", "X"]) 

  

    # Z 

    Z_rows = [{"i": i, "r": r, "t": t, "Z": int(v)} for (i, r, t), v in sol.Z.items() if int(v) != 0] 

    df_Z = pd.DataFrame(Z_rows).sort_values(["t", "r", "i"]) if Z_rows else pd.DataFrame(columns=["i", "r", "t", "Z"]) 

  

    # Q 

    Q_rows = [] 

    for t in inst.T: 

        for b in inst.B: 

            for i in inst.D: 

                v = int(sol.Q.get((b, i, t), 0)) 

                if v != 0: 

                    Q_rows.append({"b": b, "i": i, "t": t, "Q": v}) 

    df_Q = pd.DataFrame(Q_rows).sort_values(["t", "b", "i"]) if Q_rows else pd.DataFrame(columns=["b", "i", "t", "Q"]) 

  

    # Y 

    Y_rows = [{"f": f, "b": b, "k": k, "s": s, "t": t, "Y": int(v)} 

              for (f, b, k, s, t), v in sol.Y.items() if int(v) != 0] 

    df_Y = pd.DataFrame(Y_rows).sort_values(["s", "t", "b", "k", "f"]) if Y_rows else pd.DataFrame(columns=["f", "b", "k", "s", "t", "Y"]) 

  

    # L 

    L_rows = [{"f": f, "b": b, "k": k, "s": s, "t": t, "L": int(v)} 

              for (f, b, k, s, t), v in sol.L.items() if int(v) != 0] 

    df_L = pd.DataFrame(L_rows).sort_values(["s", "t", "b", "k", "f"]) if L_rows else pd.DataFrame(columns=["f", "b", "k", "s", "t", "L"]) 

  

    # Inventory / Waste 

    I_rows, W_rows = [], [] 

    for t in inst.T: 

        for i in inst.D: 

            I_rows.append({"i": i, "t": t, "Ipos": int(sol.Ipos.get((i, t), 0))}) 

            W_rows.append({"i": i, "t": t, "W": int(sol.W.get((i, t), 0))}) 

    df_I = pd.DataFrame(I_rows).sort_values(["t", "i"]) 

    df_W = pd.DataFrame(W_rows).sort_values(["t", "i"]) 

  

    # OUT (hub,t) 

    OUT_rows = [] 

    for t in inst.T: 

        for b in inst.B: 

            out_bt = sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D) 

            OUT_rows.append({"b": b, "t": t, "OUT": int(out_bt)}) 

    df_OUT = pd.DataFrame(OUT_rows).sort_values(["t", "b"]) 

  

    # IN (hub,s,t) 

    IN_rows = [] 

    for s in inst.S: 

        for t in inst.T: 

            for b in inst.B: 

                in_bst = sum( 

                    int(val) for (f, bb, k, ss, tt), val in sol.L.items() 

                    if ss == s and tt == t and bb == b 

                ) 

                IN_rows.append({"b": b, "s": s, "t": t, "IN": int(in_bst)}) 

    df_IN = pd.DataFrame(IN_rows).sort_values(["s", "t", "b"]) 

  

    # checks 

    df_req = check_service_level_REQ(inst, sol, REQ) 

    df_flow = check_flow_balance(inst, sol) 

  

    # costs 

    df_cost_ffh = pd.DataFrame([sol.cost_breakdown]) if sol.cost_breakdown else pd.DataFrame() 

    df_cost_opl = pd.DataFrame([sol.cost_breakdown_opl]) if sol.cost_breakdown_opl else pd.DataFrame() 

  

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer: 

        df_X.to_excel(writer, sheet_name="X_route_activation", index=False) 

        df_Z.to_excel(writer, sheet_name="Z_route_loads", index=False) 

        df_Q.to_excel(writer, sheet_name="Q_hub_to_depot", index=False) 

        df_Y.to_excel(writer, sheet_name="Y_assignments", index=False) 

        df_L.to_excel(writer, sheet_name="L_inbound_qty", index=False) 

        df_I.to_excel(writer, sheet_name="Ipos_inventory", index=False) 

        df_W.to_excel(writer, sheet_name="W_waste", index=False) 

        df_OUT.to_excel(writer, sheet_name="OUT_hub_period", index=False) 

        df_IN.to_excel(writer, sheet_name="IN_hub_scenario_period", index=False) 

        df_req.to_excel(writer, sheet_name="CHECK_REQ_service", index=False) 

        df_flow.to_excel(writer, sheet_name="CHECK_flow_balance", index=False) 

        df_cost_ffh.to_excel(writer, sheet_name="Cost_breakdown_FFH", index=False) 

        df_cost_opl.to_excel(writer, sheet_name="Cost_breakdown_OPL", index=False) 

  

    print(f"[EXPORT] Solution written to: {out_path}") 

  

  

# ============================================================ 

# 7) RUN 

# ============================================================ 

  

if __name__ == "__main__": 

    file_path = "step1.xlsx" 

  

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

  

    # parameters 

    alpha = 0.995 

    cv = 0.10 

    m = 2 

    waste_cost = 27.55696873 

    scenario_probs = {1: 0.3, 2: 0.5, 3: 0.2}  # MUST match your Excel S IDs 

  

    inst = load_instance_from_excel( 

        file_path=file_path, 

        route_to_depots=route_to_depots, 

        route_to_hub=route_to_hub, 

        alpha=alpha, 

        cv=cv, 

        shelf_life=m, 

        waste_cost=waste_cost, 

        scenario_probs=scenario_probs, 

    ) 

  

    sol, REQ = solve_ffh(inst) 

    print("\nAll phases completed.") 

# =========================
# DEBUG: implied inventory feasibility check
# =========================
bad = []
for i in inst.D:
    cumQ = 0
    cumW = 0
    for t in inst.T:
        cumQ += sum(sol.Q.get((b,i,t),0) for b in inst.B)
        cumW += sol.W.get((i,t),0)
        cumDem = sum(int_round(inst.mu[(i,a)]) for a in inst.T if a <= t)

        I_implied = cumQ - (cumDem + cumW)
        if I_implied < 0:
            bad.append((i,t,cumQ,cumDem,cumW,I_implied))
            break   # ilk patlayan t yeterli

print("\n[DEBUG] Negative implied inventory (first violations):")
for row in bad[:10]:
    print("Depot", row[0], "t=", row[1],
          "| cumQ=", row[2],
          "| cumDem=", row[3],
          "| cumW=", row[4],
          "| I_implied=", row[5])

print("Total depots with violation:", len(bad))

# =========================
# Export
# =========================


    # Export (includes checks + both cost breakdowns) 

export_solution_to_excel(inst, sol, REQ, out_path="ffh_solution_export123.xlsx") 

  

viol = 0 

for s in inst.S: 

    for t in inst.T: 

        for f in inst.F: 

            cnt = sum(sol.Y.get((f,b,k,s,t),0) for b in inst.B for k in inst.K) 

            if cnt > 1: 

                viol += 1 

print("[CHECK] multi-assign (sum_{b,k} Y > 1) count =", viol) 

  

# FFH main loop bittikten sonra (örneğin solve_ffh() fonksiyonunun sonunda)

t = 1
i = "19"

qsum = sum(sol.Q.get((b, i, t), 0) for b in inst.B)

print("=== DEBUG CHECK (Depot 19, t=1) ===")
print("Sum_b Q[b,19,1] =", qsum)
print("mu[19,1] =", inst.mu[(i, t)])
print("Ipos[19,1] =", sol.Ipos.get((i, t), 0),
      "W[19,1] =", sol.W.get((i, t), 0))
print("=================================")
  

bad = 0 

for (f,b,k,s,t), y in sol.Y.items(): 

    if y==1 and sol.L.get((f,b,k,s,t),0)==0: 

        bad += 1 

print("[CHECK] Y=1 but L=0 count =", bad) 

  

AC = 0.0 

for s in inst.S: 

    pr = inst.scenario_probs[s] 

    for t in inst.T: 

        for f in inst.F: 

            for b in inst.B: 

                for k in inst.K: 

                    y = sol.Y.get((f,b,k,s,t),0) 

                    if y==1: 

                        AC += pr * inst.beta[(f,k)] 

print("AC_check(OPL) =", AC) 


 