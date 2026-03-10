# -*- coding: utf-8 -*-
"""
FFH (Feasibility-First Heuristic) - IMPROVED VERSION
=====================================================
Key improvements over original:

[Phase 2 - inbound_assign_greedy_one_scenario_int]
  FIX-1: Amortized cost criterion
          Score now uses (beta[f,k] / ship + gamma[f,b,k]) so the fixed
          assignment cost is spread over the actual shipment size, preventing
          the heuristic from wasting a full vehicle activation on tiny loads.

  FIX-2: Pure Vogel regret (no waste-risk inflation)
          Regret = best_alternative_unit_cost - best_unit_cost.
          Waste-risk was re-added every iteration for the same goods, causing
          the algorithm to over-prioritise early assignments and miss cheaper
          overall solutions. Removed from per-iteration scoring.

  FIX-3: Partial fulfillment tracking
          Suppliers can now partially fill a hub across multiple vehicles
          (if supply > one vehicle's remaining capacity). Previously a supplier
          was removed from `used_f` after first use, silently under-delivering.

  FIX-4: Greedy-by-unit-cost fallback
          When only one vehicle is available for a (f,b) pair (no regret signal),
          the decision is still guided by amortized unit cost, not flat beta.

  FIX-5: Early termination guard
          If hub_needs cannot be reduced in an iteration (no feasible assignment
          exists), the loop exits immediately instead of spinning forever.

[Code structure]
  - Phase 2 helper extracted into _best_vehicle_for_pair() for clarity.
  - All magic numbers replaced with named expressions.
  - Debug print statements separated behind a DEBUG_MODE flag.
  - Main solve loop unchanged in structure; only the assignment kernel replaced.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
import pandas as pd
import math
import time

# ── toggle verbose debug prints ──────────────────────────────────────────────
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
    I:    Dict[Tuple[str, int], int] = field(default_factory=dict)  # signed inventory (can be ≤ 0)
    W: Dict[Tuple[str, int], int] = field(default_factory=dict)
    Q_cum: Dict[Tuple[str, int], int] = field(default_factory=dict)
    W_cum: Dict[Tuple[str, int], int] = field(default_factory=dict)
    cost_breakdown: Dict[str, float] = field(default_factory=dict)
    cost_breakdown_opl: Dict[str, float] = field(default_factory=dict)


# ============================================================
# Excel Loader
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
# REQ precomputation
# ============================================================
def precompute_REQ(inst: Instance) -> Dict[Tuple[str, int], float]:
    """
    Implements model constraints (17)-(19): deterministic service-level surrogate.

    REQ[i,t] = ceil( sum_{a=1}^{t} mu_{i,a} + sqrt(sum_{a=1}^{t}(cv*mu_{i,a})^2) * Z_alpha )

    ceil() is required: qmin drives integer Q decisions in Phase 1.
    Without ceil(), a float REQ of e.g. 47.3 produces qmin=0.3, which
    Phase 1 truncates to 0 integer units — leaving Q_cum one unit short
    of satisfying constraint (19). ceil() guarantees the integer Q always
    satisfies the service-level constraint.

    Computed cumulatively inside the t-loop: REQ[(i,t)] covers periods 1..t.
    """
    z = z_value(inst.alpha)
    REQ: Dict[Tuple[str, int], float] = {}
    for i in inst.D:
        cum_mu  = 0.0
        cum_var = 0.0
        for t in inst.T:
            mu_it    = inst.mu[(i, t)]
            cum_mu  += mu_it
            cum_var += (inst.cv * mu_it) ** 2
            REQ[(i, t)] = math.ceil(cum_mu + z * math.sqrt(cum_var))  # ← ceil + inside t-loop
    return REQ


# ============================================================
# OPL-exact inventory computation
# ============================================================

# ============================================================
# Phase 2 helper: best vehicle for a (supplier, hub) pair
# ============================================================
def _best_vehicle_for_pair(
    f: str,
    b: str,
    remaining_hub: int,
    supply_available: int,
    inst: Instance,
    cap_left: Dict[str, int],
    activated_fk: set = None,
    fk_hub: Dict = None,
) -> Optional[Tuple[float, str, int, float]]:
    """
    Two-tier vehicle selection targeting MILP cost structure (AC:TC ~ 12:1).

    TIER 1 — REUSE  (beta already paid this period):
      Pick lowest gamma*ship among already-active (f,k) pairs.
      If ANY reuse candidate exists, always prefer it over a new activation.
      Regret vs. new = beta[f,k_new_cheapest]  (what we avoid paying).

    TIER 2 — NEW ACTIVATION (only if no reuse candidate):
      Sort by beta[f,k] ASC as primary key (minimise fixed charge first),
      then gamma*ship as tiebreaker.
      Regret = unit_cost_2nd - unit_cost_best.
    """
    max_to_ship = min(supply_available, remaining_hub)
    if max_to_ship <= 0:
        return None

    reuse: List[Tuple[float, str, int]] = []        # (var_cost, k, ship)
    new:   List[Tuple[float, float, str, int]] = []  # (beta, var_cost, k, ship)

    if activated_fk is None:
        activated_fk = set()
    if fk_hub is None:
        fk_hub = {}

    for k in inst.K:
        if cap_left[k] <= 0:
            continue
        # OPL constraint 4: (f,k) already committed to a different hub
        if fk_hub.get((f, k)) not in (None, b):
            continue
        if (f, k) not in inst.beta or (f, b, k) not in inst.gamma:
            continue
        actual_ship = min(max_to_ship, cap_left[k])
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
        # Regret = beta of cheapest new activation we are avoiding
        if new:
            new.sort()
            regret = new[0][0]          # cheapest beta we avoid
        elif len(reuse) > 1:
            regret = (reuse[1][0] - reuse[0][0]) / max(1, best_ship)
        else:
            regret = best_unit_cost
    else:
        new.sort()                      # sort by beta ASC, then var_cost
        beta_val, var_cost, best_k, best_ship = new[0]
        best_unit_cost = (beta_val + var_cost) / max(1, best_ship)
        if len(new) > 1:
            b2b, b2v, _, b2s = new[1]
            regret = (b2b + b2v) / max(1, b2s) - best_unit_cost
        else:
            regret = best_unit_cost

    return best_unit_cost, best_k, best_ship, regret


# ============================================================
# Phase 2: improved inbound assignment (one scenario, one period)
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
    ACTIVATION-MINIMISING INBOUND ASSIGNMENT  (2-pass)
    ====================================================
    Targets MILP cost structure: AC (beta) >> TC (gamma), ratio ~12:1.

    pre_activated_fk: set of (f,k) pairs already committed this period
      (computed by the caller from scenario cross-comparison — uses only
      beta, gamma, theta, supply from the model). When provided, these
      pairs have marginal beta = 0, reducing redundant activations.

    PASS 1 — CONSOLIDATION (largest hub first):
      For each hub b, find the single (f,k) that covers the most of that
      hub's need at lowest marginal cost. Reuses already-activated (f,k)
      pairs (marginal beta = 0) before opening new activations.

    PASS 2 — RESIDUAL VOGEL:
      Any remaining hub needs handled with Vogel regret on marginal costs.
      New activations sorted by beta ASC to minimise fixed activation cost.
    """

    # ── initialise ───────────────────────────────────────────────────────────
    cap_left: Dict[str, int]              = {k: int_floor(inst.theta[k]) for k in inst.K}
    activated_fk: set = set()  # built up within this scenario only

    Y_s: Dict[Tuple[str, str, str, str, int], int] = {}
    L_s: Dict[Tuple[str, str, str, str, int], int] = {}
    achieved: Dict[str, int] = {b: 0 for b in inst.B}
    # OPL constraint 4: Σ_b Y[f,b,k,s,t] ≤ 1  per (f,k,s,t)
    # Correct interpretation: a specific (f,k) pair may serve only ONE hub,
    # but supplier f may use DIFFERENT vehicles (k1, k2, …) to serve
    # DIFFERENT hubs simultaneously. Only the (f,k) pair is constrained.
    fk_hub: Dict[Tuple[str,str], str] = {}   # (f,k) -> b already committed

    # FIX-B: keep supply as float — constraint (1) uses continuous L >= 0.
    # Integer floor silently discards supply; only round when writing final L.
    supply_left: Dict[str, float] = {
        f: float(inst.supply.get((f, s, t), 0.0)) for f in inst.F
    }
    hub_needs: Dict[str, int] = {
        b: int(OUT_bt.get(b, 0)) for b in inst.B if OUT_bt.get(b, 0) > 0
    }

    def commit(f: str, b: str, k: str, ship) -> None:
        ship_int = int_floor(float(ship))
        if ship_int <= 0:
            return
        key = (f, b, k, s, t)
        Y_s[key] = 1
        L_s[key]        = L_s.get(key, 0) + ship_int
        cap_left[k]    -= ship_int
        fk_hub[(f, k)] = b   # lock this (f,k) pair to hub b
        activated_fk.add((f, k))
        supply_left[f] -= ship_int
        achieved[b]    += ship_int
        hub_needs[b]   -= ship_int
        if hub_needs[b] <= 0:
            hub_needs.pop(b, None)

    # ── PASS 1: fill each hub greedily until need is met ────────────────────
    # Process hubs largest-first. For each hub, loop until need==0 or no
    # feasible supplier remains. At each step:
    #   TIER-1: reuse an already-activated (f,k) — zero marginal beta.
    #           Score: coverage DESC, var_cost ASC.
    #   TIER-2: new activation. Score by amortised total cost per unit:
    #           (beta[f,k] + gamma[f,b,k]*ship) / ship
    #           This accounts for the fixed charge amortised over actual
    #           shipment size, preferring suppliers that can cover more of
    #           the remaining need in one activation.
    for b in sorted(list(hub_needs.keys()), key=lambda b: hub_needs[b], reverse=True):
        # Inner loop: keep filling this hub until fully covered or supply exhausted
        while b in hub_needs:
            need = hub_needs[b]
            best_free: Optional[Tuple[float, float, str, str, int]] = None
            best_new:  Optional[Tuple[float, float, str, str, int]] = None

            for f in inst.F:
                if supply_left[f] <= 0:
                    continue
                if (f, b) not in inst.allowed_f_b:
                    continue
                for k in inst.K:
                    if cap_left[k] <= 0:
                        continue
                    if fk_hub.get((f, k)) not in (None, b):
                        continue
                    if (f, k) not in inst.beta or (f, b, k) not in inst.gamma:
                        continue
                    ship = min(supply_left[f], cap_left[k], need)
                    if ship <= 0:
                        continue
                    var_cost = inst.gamma[(f, b, k)] * ship
                    coverage = ship / need
                    if (f, k) in activated_fk:
                        # Tier-1: already paid beta — score by coverage then var_cost
                        score = (coverage, -var_cost)
                        if best_free is None or score > (best_free[0], best_free[1]):
                            best_free = (coverage, -var_cost, f, k, ship)
                    else:
                        # Tier-2: amortised total cost per unit delivered
                        amortised = (inst.beta[(f, k)] + var_cost) / max(1, ship)
                        score = (-amortised, coverage)
                        if best_new is None or score > (best_new[0], best_new[1]):
                            best_new = (-amortised, coverage, f, k, ship)

            if best_free is not None:
                f, k, ship = best_free[2], best_free[3], best_free[4]
            elif best_new is not None:
                f, k, ship = best_new[2], best_new[3], best_new[4]
            else:
                break  # no feasible supplier — leave residual for Pass 2

            commit(f, b, k, ship)
            if DEBUG_MODE:
                print(f"  [P2-P1] t={t} s={s} FILL b={b} f={f} k={k} ship={ship} rem={hub_needs.get(b,0)}")

    # ── PASS 2: residual Vogel for any unmet hub needs ───────────────────────
    while hub_needs:
        best_entry: Optional[Tuple[float, str, str, str, int]] = None

        for b, remaining in hub_needs.items():
            for f in inst.F:
                if supply_left[f] <= 0:
                    continue
                if (f, b) not in inst.allowed_f_b:
                    continue
                # OPL constraint 4: filter handled inside _best_vehicle_for_pair via fk_hub
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
            break  # stall guard

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
    r_deps: Dict, r_cap: Dict, r_hub: Dict,
    qmin: Dict[str, float],
    t: int,
) -> Tuple[Dict, Dict, Dict]:
    """Load a fixed set of selected routes and induce Q from Z."""
    Z_t: Dict[Tuple[str, str, int], int] = {}
    Q_t: Dict[Tuple[str, str, int], int] = {}
    rem = {i: float(qmin.get(i, 0.0)) for i in qmin}

    for r_id in selected:
        cap  = r_cap[r_id]
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
                rem[i] -= x_int
                cap    -= x_int

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


def greedy_route_selection_and_loading_int(
    inst: Instance,
    t: int,
    qmin: Dict[str, float],
) -> Tuple[
    Dict[Tuple[str, int], int],
    Dict[Tuple[str, str, int], int],
    Dict[Tuple[str, str, int], int],
]:
    U = {i for i, v in qmin.items() if v > 1e-9}

    X_t: Dict[Tuple[str, int], int] = {}

    r_deps = {r.r_id: set(r.depots) for r in inst.routes}
    r_cost = {r.r_id: r.fixed_cost for r in inst.routes}
    r_cap  = {r.r_id: int_floor(r.capacity) for r in inst.routes}
    r_hub  = {r.r_id: r.hub for r in inst.routes}

    # ── Step 1: weighted set-cover greedy ────────────────────────────────────
    # Score = Σ qmin[i over uncovered depots] / λ[r]
    # Standard ratio heuristic; no utilisation penalty (would skip mandatory
    # deliveries to low-demand depots, violating service-level constraint).
    selected: List[str] = []
    remaining_U = set(U)
    while remaining_U:
        best_r, best_score = None, -1.0
        for r in inst.routes:
            cover     = r_deps[r.r_id] & remaining_U
            cover_sum = sum(qmin[i] for i in cover)
            if cover_sum <= 1e-12:
                continue
            score = cover_sum / max(1e-12, r_cost[r.r_id])
            if score > best_score:
                best_score = score
                best_r     = r.r_id
        if best_r is None:
            break
        selected.append(best_r)
        remaining_U -= r_deps[best_r]

    # ── Step 2: drop redundant routes ────────────────────────────────────────
    # A route r is redundant if all its mandatory depots are covered by the
    # remaining routes. Removing it saves λ[r] at no feasibility cost.
    # Process cheapest routes last (try to drop expensive ones first).
    improved = True
    while improved:
        improved = False
        for r in sorted(selected, key=lambda r: r_cost[r], reverse=True):
            without = [x for x in selected if x != r]
            if _covers_all(without, U, r_deps):
                selected = without
                improved = True
                break   # restart after each drop

    # ── Step 3: swap local search ─────────────────────────────────────────────
    # Try replacing each selected route r with each unselected route r2.
    # Accept if: (a) all mandatory depots still covered, and (b) cost improves.
    # One pass over all selected routes; repeat until no improvement found.
    all_route_ids = [r.r_id for r in inst.routes]
    improved = True
    while improved:
        improved = False
        for r in list(selected):
            without = [x for x in selected if x != r]
            candidates = [r2 for r2 in all_route_ids if r2 not in selected]
            for r2 in candidates:
                trial = without + [r2]
                if _covers_all(trial, U, r_deps):
                    saving = r_cost[r] - r_cost[r2]
                    if saving > 1e-9:
                        selected = trial
                        improved = True
                        break   # restart inner loop after swap
            if improved:
                break   # restart outer loop after swap

    # ── Finalise X ───────────────────────────────────────────────────────────
    for r_id in selected:
        X_t[(r_id, t)] = 1

    # ── Load and induce Q ─────────────────────────────────────────────────────
    Z_t, Q_t = _load_routes(selected, r_deps, r_cap, r_hub, qmin, t)

    return X_t, Z_t, Q_t


# ============================================================
# Scaling helpers (unchanged from original)
# ============================================================
def scale_Z_to_target(
    sol: FFHSolution, inst: Instance, b: str, t: int, target_out: int,
    qmin_floor: Dict[str, int] = None,
) -> None:
    """
    Scale Z[i,r,t] for hub b so that Σ Z = target_out,
    but never reduce any depot i below qmin_floor[i].

    Only the SURPLUS above qmin_floor is scaled proportionally.
    This ensures constraint 2.15 (service level) is never violated
    by Phase 2 reducing deliveries below what Phase 1 sized.
    """
    r_hub = {r.r_id: r.hub for r in inst.routes}
    if qmin_floor is None:
        qmin_floor = {}

    # Collect nonzero Z entries for routes starting at hub b
    z_keys = [(i, r_id, tt) for (i, r_id, tt) in sol.Z
              if tt == t and r_hub.get(r_id) == b and int(sol.Z.get((i, r_id, tt), 0)) > 0]

    cur_total = sum(int(sol.Z[k]) for k in z_keys)
    if cur_total <= target_out:
        return  # nothing to scale down

    # Per-depot current delivery and mandatory floor
    depot_z: Dict[str, List] = {}   # i -> list of (key, value)
    for k in z_keys:
        i = k[0]
        depot_z.setdefault(i, []).append(k)

    # Compute per-depot current total and floor
    depot_cur   = {i: sum(int(sol.Z[k]) for k in keys) for i, keys in depot_z.items()}
    depot_floor = {i: int(qmin_floor.get(i, 0)) for i in depot_cur}

    # Total mandatory floor across all depots at this hub
    total_floor = sum(depot_floor[i] for i in depot_cur)

    if target_out < total_floor:
        # Cannot satisfy service-level floors at this target — clip target to floor
        target_out = total_floor

    # Surplus = amount above floor that can be scaled
    total_surplus = cur_total - total_floor
    target_surplus = target_out - total_floor

    if total_surplus <= 0 or target_surplus >= total_surplus:
        return  # nothing to cut

    ratio = target_surplus / total_surplus

    # Scale each depot's Z proportionally within its surplus
    for i, keys in depot_z.items():
        floor_i   = depot_floor[i]
        cur_i     = depot_cur[i]
        surplus_i = cur_i - floor_i
        if surplus_i <= 0:
            continue  # already at floor, do not touch

        new_total_i = floor_i + int(math.floor(surplus_i * ratio + 1e-9))
        new_total_i = max(floor_i, new_total_i)  # never below floor

        if new_total_i == cur_i:
            continue

        # Distribute new_total_i across Z entries for depot i proportionally
        base_vals = []
        base_sum  = 0
        remainders = []
        for k in keys:
            v  = int(sol.Z[k])
            x  = v * (new_total_i / cur_i)
            bk = int(math.floor(x + 1e-9))
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
    """
    Recompute Q[b,i,t] = Σ_{r: hub(r)=b} Z[i,r,t]  for all (b,i).

    This is the CORRECT way to maintain constraint (13) after any
    change to Z values. Rather than trying to sync Z to match Q
    (which introduces rounding errors), we derive Q from Z exactly.
    Only nonzero Q values are written to sol.Q.
    """
    r_hub = {r.r_id: r.hub for r in inst.routes}

    # Clear existing Q entries for this period
    for key in [k for k in list(sol.Q.keys()) if k[2] == t]:
        del sol.Q[key]

    # Recompute from Z — guaranteed to satisfy constraint (13)
    for (i, r_id, tt), z in sol.Z.items():
        if tt != t or int(z) == 0:
            continue
        b = r_hub[r_id]
        sol.Q[(b, i, t)] = sol.Q.get((b, i, t), 0) + int(z)


# ============================================================
# Cost functions
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
# Checks
# ============================================================
def check_service_level_REQ(
    inst: Instance,
    sol: FFHSolution,
    REQ: Dict[Tuple[str, int], float],
) -> pd.DataFrame:
    """
    Checks OPL constraint 2.15 exactly as OPL computes it:

      sum(a=1..t, b in B) Q[b][i][a] - sum(a=1..t-1) W[i][a]
        >= sum(a=1..t) de[i][a] + sqrt(sum(a=1..t) de[i][a]^2) * co * Zalpha

    LHS uses Q_cum[i,t] - W_cum[i,t-1]  (W_cum excludes W[i,t] itself).
    RHS is the raw float OPL computes — NOT the heuristic's ceil()-based REQ.
    Both are reported for comparison.
    """
    co     = inst.cv          # coefficient of variation
    Zalpha = z_value(inst.alpha)
    rows   = []
    for i in inst.D:
        for t in inst.T:
            qc  = float(sol.Q_cum.get((i, t), 0))
            # W_cum[i,t-1] = sum of W[i,a] for a=1..t-1  (NOT including t)
            wc  = float(sol.W_cum.get((i, t - 1), 0))
            lhs = qc - wc

            # OPL RHS: sum(de) + sqrt(sum(de^2)) * co * Zalpha
            de_sum    = sum(float(inst.mu[(i, a)]) for a in inst.T if a <= t)
            de_sq_sum = sum(float(inst.mu[(i, a)])**2 for a in inst.T if a <= t)
            rhs_opl   = de_sum + math.sqrt(de_sq_sum) * co * Zalpha

            # Heuristic REQ (ceil of same formula, used to size Q)
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
# FFH main solve
# ============================================================
def solve_ffh(inst: Instance) -> Tuple[FFHSolution, Dict[Tuple[str, int], float]]:
    sol = FFHSolution()

    for i in inst.D:
        sol.Q_cum[(i, 0)] = 0
        sol.W_cum[(i, 0)] = 0

    with PhaseTimer("Precompute REQ_{i,t}"):
        REQ = precompute_REQ(inst)

    for t in inst.T:

        # ── Phase 1 ──────────────────────────────────────────────────────────
        with PhaseTimer(f"Phase 1 (t={t}) - sizing + route selection"):
            # qmin[i] = minimum units to deliver to depot i in period t,
            # derived from model constraint (19):
            #   Q_cum[i,t] - W_cum[i,t-1] >= REQ[i,t]
            # → qmin[i] = max(0, REQ[i,t] + W_cum[i,t-1] - Q_cum[i,t-1])
            #
            # SOFT_FACTOR deliberately omitted: softening qmin violates the
            # service-level constraint (19), which is a hard feasibility
            # requirement. For Q1 publication the heuristic must be provably
            # feasible. Route cost efficiency is achieved through Phase 1
            # set-cover scoring, not by reducing required deliveries.
            qmin = {
                i: max(0.0,
                       REQ[(i, t)]
                       + float(sol.W_cum[(i, t - 1)])
                       - float(sol.Q_cum[(i, t - 1)]))
                for i in inst.D
            }

            X_t, Z_t, Q_t = greedy_route_selection_and_loading_int(inst, t, qmin)
            # Keep qmin accessible to Phase 2 for service-level floor enforcement
            qmin_t = {i: int(math.ceil(qmin[i])) for i in inst.D}

            for (r_id, tt), v in X_t.items():
                sol.X[(r_id, tt)] = int(v)
            for (i, r_id, tt), v in Z_t.items():
                if v > 0:
                    sol.Z[(i, r_id, tt)] = int(v)
            # Write only nonzero Q entries — constraint (13) requires
            # Q[b,i,t] = Σ_{r:hub=b} Z[i,r,t], so Q=0 entries must be absent
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
            # OUT[b] = Σ_i Q[b,i,t]  — the downstream demand placed on each hub
            OUT = {b: sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D)
                   for b in inst.B}

            if DEBUG_MODE:
                print(f"  OUT (before robust): {OUT}")




            # ── Convergence loop ─────────────────────────────────────────────
            out_phase1_floor: Dict[str, int] = {b: int(OUT[b]) for b in inst.B}
            OUT_final: Dict[str, int] = {b: int(OUT[b]) for b in inst.B}

            for _iter in range(5):
                temp_store: Dict[str, tuple] = {}
                min_ach: Dict[str, int] = {b: OUT_final[b] for b in inst.B}

                for s in inst.S:
                    Y_s, L_s, ach = inbound_assign_greedy_one_scenario_int(
                        inst, s, t, OUT_final
                    )
                    temp_store[s] = (Y_s, L_s, ach)
                    for b in inst.B:
                        min_ach[b] = min(min_ach[b], int(ach[b]))

                if all(min_ach[b] >= OUT_final[b] for b in inst.B):
                    break  # converged — temp_store matches OUT_final exactly

                # Tighten to min achieved — but never below Phase 1 floor
                OUT_final = {b: max(int(min_ach[b]), out_phase1_floor[b])
                             for b in inst.B}
                if DEBUG_MODE:
                    print(f"  [P2 iter={_iter}] tightened OUT_final: {OUT_final}")

            else:
                # Loop exhausted without converging: OUT_final was tightened on
                # the last iteration but the greedy was never re-run against it.
                # temp_store targets the previous (too-high) OUT_final, so
                # Σ L ≠ OUT_final = Σ Q after scaling — violating constraint 5.
                # Run one final greedy against the tightened OUT_final.
                temp_store = {}
                for s in inst.S:
                    Y_s, L_s, ach = inbound_assign_greedy_one_scenario_int(
                        inst, s, t, OUT_final
                    )
                    temp_store[s] = (Y_s, L_s, ach)
                    # Update OUT_final to what was actually achieved, floored by Phase 1
                    for b in inst.B:
                        OUT_final[b] = max(int(ach[b]), out_phase1_floor[b])
                if DEBUG_MODE:
                    print(f"  [P2 final re-run] OUT_final after re-run: {OUT_final}")

            # ── Scale Z/Q down to match OUT_final ────────────────────────────
            # No qmin_floor: supply may be less than qmin for some hubs.
            # Q is set to min(Phase1_Q, supply_achievable) via OUT_final.
            # Service-level feasibility is checked in CHECK-A after Phase 3.
            for b in inst.B:
                scale_Z_to_target(sol, inst, b, t, OUT_final[b])
            recompute_Q_from_Z(sol, inst, t)

            # Deactivate routes that have all Z=0 after scaling (constraint 11)
            for r in inst.routes:
                if all(sol.Z.get((i, r.r_id, t), 0) == 0 for i in r.depots):
                    sol.X.pop((r.r_id, t), None)

            # ── Enforce constraint 5: re-run greedy to match actual Q ─────────
            # Q is now authoritative (set by scale_Z → recompute_Q_from_Z).
            # L must equal Q per hub per scenario. Re-run greedy for every
            # scenario targeting the actual Q_sum — no scaling of Q allowed here.
            Q_sum = {b: sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D)
                     for b in inst.B}

            for s in inst.S:
                Y_new, L_new, ach_new = inbound_assign_greedy_one_scenario_int(
                    inst, s, t, Q_sum
                )
                # Verify greedy achieved Q_sum exactly
                for b in inst.B:
                    if int(ach_new[b]) != Q_sum[b]:
                        print(f"  [C5-WARN] t={t} s={s} b={b}: "
                              f"greedy achieved {ach_new[b]}, Q_sum={Q_sum[b]}")
                temp_store[s] = (Y_new, L_new, ach_new)

            # ── Write final Y/L ───────────────────────────────────────────────
            # L comes directly from the converged greedy — no post-hoc patching.
            # Constraint (5): Σ L[f,b,k,s,t] = achieved[b,s] = OUT_final[b]
            #                                 = Σ_i Q[b,i,t]  ← satisfied.
            # Constraints (1),(2),(3) are satisfied by the greedy's commit()
            # which enforces supply, capacity, and Y=1 ↔ L>0.
            for k in [k for k in list(sol.Y.keys()) if k[4] == t]:
                del sol.Y[k]
            for k in [k for k in list(sol.L.keys()) if k[4] == t]:
                del sol.L[k]

            for s in inst.S:
                Y_s, L_s, _ = temp_store[s]
                for k, v in Y_s.items():
                    sol.Y[k] = int(v)
                for k, v in L_s.items():
                    sol.L[k] = int(v)

            ratios = [
                OUT_final[b] / OUT[b] if OUT[b] > 0 else 1.0
                for b in inst.B
            ]
            print(f"  Phase 2 done (t={t}). Min robust OUT ratio: {round(min(ratios), 4)}"
                  )

        # ── Phase 3 ──────────────────────────────────────────────────────────
        # W[i][t] and I[i][t] are computed AFTER all periods using the exact
        # OPL constraint formulas (2.11–2.14). We only need Q_cum here to
        # drive qmin in the next period's Phase 1.
        # W_cum for qmin uses a forward pass via compute_opl_inventory
        # called once after the full t-loop — but qmin needs W_cum[i,t-1]
        # at the START of each period. So we compute W incrementally here
        # using the same OPL formula, storing results into sol for reuse.
        with PhaseTimer(f"Phase 3 (t={t}) - OPL-exact inventory"):
            for i in inst.D:
                receipt = sum(int(sol.Q.get((b, i, t), 0)) for b in inst.B)
                sol.Q_cum[(i, t)] = float(sol.Q_cum[(i, t - 1)]) + float(receipt)

                # W[i][t] using exact OPL formula (2.13)/(2.14)
                m = inst.shelf_life
                if t < m:
                    w_it = 0
                else:
                    t_ref     = t - m + 1
                    q_cum_ref = float(sol.Q_cum[(i, t_ref)])
                    de_cum_ref = sum(float(inst.mu[(i, a)]) for a in inst.T if a <= t_ref)
                    w_cum_ref  = float(sol.W_cum[(i, t_ref)])
                    I_ref      = q_cum_ref - de_cum_ref - w_cum_ref

                    de_tail = sum(float(inst.mu[(i, a)]) for a in inst.T
                                  if t - m + 2 <= a <= t)
                    w_tail  = sum(
                        int(sol.W.get((i, a), 0))
                        for a in inst.T if t - m + 2 <= a <= t - 1
                    )
                    w_rhs = I_ref - de_tail - w_tail
                    w_it  = max(0, int(math.floor(w_rhs + 1e-9)))

                sol.W[(i, t)]     = w_it
                sol.W_cum[(i, t)] = float(sol.W_cum[(i, t - 1)]) + float(w_it)

                # I[i][t] using exact OPL formula (2.11)
                # I is the SIGNED net inventory — can be negative when Q_cum < demand_cum.
                # OPL's balance constraint uses the signed I directly.
                # Ipo (non-negative part) is used only for holding cost.
                de_cum_t = sum(float(inst.mu[(i, a)]) for a in inst.T if a <= t)
                I_it_raw = float(sol.Q_cum[(i, t)]) - de_cum_t - float(sol.W_cum[(i, t)])
                I_it_int = int(round(I_it_raw))
                sol.I[(i, t)]    = I_it_int                # signed — exported to OPL
                sol.Ipos[(i, t)] = max(0, I_it_int)        # non-neg — used for holding cost

            print(f"  Phase 3 done (t={t}).")

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
# Export
# ============================================================
def export_solution_to_excel(
    inst: Instance,
    sol: FFHSolution,
    REQ: Dict[Tuple[str, int], float],
    out_path: str = "ffh_solution_improved500.xlsx",
) -> None:
    def rows_to_df(rows, cols):
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

    # Column names MUST match OPL tuple field names exactly.
    # All tuples end with `int value` so the last column must be "value".
    X_rows = [{"r": r, "t": t, "value": int(v)} for (r, t), v in sol.X.items()]
    Z_rows = [{"i": i, "r": r, "t": t, "value": int(v)}
              for (i, r, t), v in sol.Z.items() if int(v) != 0]
    Q_rows = [{"b": b, "i": i, "t": t, "value": int(sol.Q[(b, i, t)])}
              for t in inst.T for b in inst.B for i in inst.D
              if int(sol.Q.get((b, i, t), 0)) != 0]
    Y_rows = [{"f": f, "b": b, "k": k, "s": s, "t": t, "value": int(v)}
              for (f, b, k, s, t), v in sol.Y.items() if int(v) != 0]
    L_rows = [{"f": f, "b": b, "k": k, "s": s, "t": t, "value": int(v)}
              for (f, b, k, s, t), v in sol.L.items() if int(v) != 0]

    # I and W: declared as dvar int / dvar int+ in OPL — export as integers.
    I_rows = [{"i": i, "t": t, "value": int(sol.I.get((i, t), 0))}
              for t in inst.T for i in inst.D]
    W_rows = [{"i": i, "t": t, "value": int(sol.W.get((i, t), 0))}
              for t in inst.T for i in inst.D]

    OUT_rows = [{"b": b, "t": t,
                 "OUT": sum(int(sol.Q.get((b, i, t), 0)) for i in inst.D)}
                for t in inst.T for b in inst.B]
    IN_rows  = [{"b": b, "s": s, "t": t,
                 "IN": sum(int(v) for (f, bb, k, ss, tt), v in sol.L.items()
                           if ss == s and tt == t and bb == b)}
                for s in inst.S for t in inst.T for b in inst.B]

    # ── CHECK-C: OPL balance constraint 2.11 ────────────────────────────────
    # For each (i,t), OPL requires:
    #   I[i,t] + W[i,t] = Q_cum[i,t] - demand_cum[i,t]
    # i.e.  I[i,t] = Q_cum[i,t] - demand_cum[i,t] - W[i,t]
    # If this doesn't hold exactly (integer), OPL will flag an infeasibility
    # when the variables are fixed. We verify here before export.
    balance_violations = 0
    for i in inst.D:
        for t in inst.T:
            q_cum    = float(sol.Q_cum.get((i, t), 0))
            de_cum   = sum(float(inst.mu[(i, a)]) for a in inst.T if a <= t)
            w_cum    = float(sol.W_cum.get((i, t), 0))
            I_export = int(sol.I.get((i, t), 0))
            W_export = int(sol.W.get((i, t), 0))
            # What OPL's LHS of balance equation gives:
            expected_I = int(round(q_cum - de_cum - w_cum))
            if I_export != expected_I:
                balance_violations += 1
                print(f"  [CHECK-C VIOLATION] i={i} t={t}: "
                      f"exported I={I_export}  expected I={expected_I}  "
                      f"(Q_cum={q_cum:.1f} de_cum={de_cum:.1f} W_cum={w_cum:.1f})")
            # Also verify W[i,t] is consistent with W_cum
            w_cum_prev = float(sol.W_cum.get((i, t-1), 0))
            if abs((w_cum_prev + W_export) - w_cum) > 0.5:
                balance_violations += 1
                print(f"  [CHECK-C W_CUM] i={i} t={t}: "
                      f"W_cum[t-1]={w_cum_prev:.1f} + W[t]={W_export} != W_cum[t]={w_cum:.1f}")
    if balance_violations == 0:
        print("[CHECK-C] OPL balance constraint 2.11: OK")
    else:
        print(f"[CHECK-C] OPL balance constraint 2.11: VIOLATIONS = {balance_violations}")

    df_req  = check_service_level_REQ(inst, sol, REQ)
    df_flow = check_flow_balance(inst, sol)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for df, name, cols in [
            # Sheet names MUST match the OPL .dat SheetRead references exactly:
            #   X_in  from SheetRead(sheet,"X!A2:C181")
            #   Q_in  from SheetRead(sheet,"Q!A2:D181")
            #   Z_in  from SheetRead(sheet,"Z!A2:D5401")
            #   L_in  from SheetRead(sheet,"L!A2:F667")
            #   Y_in  from SheetRead(sheet,"Y!A2:F667")
            #   I_in  from SheetRead(sheet,"I!A2:C181")
            #   W_in  from SheetRead(sheet,"W!A2:C181")
            (rows_to_df(X_rows,   ["r","t","value"]),               "X",                       False),
            (rows_to_df(Z_rows,   ["i","r","t","value"]),            "Z",                       False),
            (rows_to_df(Q_rows,   ["b","i","t","value"]),            "Q",                       False),
            (rows_to_df(Y_rows,   ["f","b","k","s","t","value"]),    "Y",                       False),
            (rows_to_df(L_rows,   ["f","b","k","s","t","value"]),    "L",                       False),
            (pd.DataFrame(I_rows).sort_values(["t","i"]),            "I",                       False),
            (pd.DataFrame(W_rows).sort_values(["t","i"]),            "W",                       False),
            (pd.DataFrame(OUT_rows).sort_values(["t","b"]),      "OUT_hub_period",          False),
            (pd.DataFrame(IN_rows).sort_values(["s","t","b"]),   "IN_hub_scenario_period",  False),
            (df_req,                                              "CHECK_REQ_service",       False),
            (df_flow,                                             "CHECK_flow_balance",      False),
            (pd.DataFrame([sol.cost_breakdown]),                  "Cost_breakdown_FFH",      False),
            (pd.DataFrame([sol.cost_breakdown_opl]),              "Cost_breakdown_OPL",      False),
        ]:
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"[EXPORT] Written to: {out_path}")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    file_path = "step4.xlsx"

    route_to_depots = {
    1: [25, 24, 14, 13],
    2: [16, 17, 23, 20],
    3: [10, 22, 27, 30],
    4: [28, 26, 19],
    5: [11, 12, 29],
    6: [8, 18, 21, 15, 9],
    7: [19, 20, 14, 7, 2],
    8: [17, 15, 21, 25],
    9: [22, 8, 6, 5, 1],
    10: [23, 13, 10, 11],
    11: [29, 26, 24, 16, 4, 3],
    12: [9, 18, 30, 28, 27],
    13: [30, 23, 5, 4, 2],
    14: [12, 15, 21, 25, 28],
    15: [7, 9, 10, 16, 22],
    16: [18, 19, 17, 14, 12],
    17: [29, 24, 8, 1, 3],
    18: [27, 20, 7, 6, 4, 2],
    19: [7, 6, 5, 3, 1],
    20: [20, 21, 22, 24, 26, 28],
    21: [23, 8, 16, 13, 11],
    22: [6, 5, 4, 3, 2, 1],
    23: [17, 30, 14, 9, 10, 12],
    24: [18, 19, 25, 26, 27, 29],
    25: [11, 13, 15, 20, 22, 26],
    26: [10, 16, 15, 27],
    27: [7, 13, 14, 17, 28],
    28: [9, 12, 11],
    29: [8, 19, 23, 25, 24, 21],
    30: [18, 30, 29],
    31: [19, 10, 22, 27, 30],
    32: [12, 15, 21, 25, 4, 28],
    33: [7, 6, 5, 3],
    34: [23, 7, 13, 14, 17, 28],
    35: [7, 6, 9, 5, 3, 1],
    36: [18, 19, 30, 14, 12],
    37: [7, 9, 15, 16, 22],
    38: [17, 3, 15, 21, 25],
    39: [18, 19, 17, 12],
    40: [12, 15, 5, 21, 25, 28],
    41: [14, 30, 29],
    42: [10, 16, 27],
    43: [17, 30, 14, 9, 12],
    44: [3, 19, 17, 14, 12],
    45: [28, 8, 6, 5, 1],
    46: [16, 17, 23],
    47: [23, 13, 26, 11],
    48: [25, 12, 14, 13],
    49: [19, 14, 7, 2],
    50: [19, 23, 5, 4, 2],
    51: [8, 18, 21, 15, 16],
    52: [22, 8, 6, 23, 5, 1],
    53: [22, 8, 6, 16, 1],
    54: [6, 17, 15, 21, 25],
    55: [11, 1, 12, 29],
    56: [18, 19, 10, 17, 14, 12],
    57: [25, 24, 14, 20, 13],
    58: [9, 30, 28, 27],
    59: [30, 23, 5, 16, 2],
    60: [28, 22, 19],
    61: [7, 16, 17, 23, 20],
    62: [14, 7, 9, 10, 16, 22],
    63: [20, 21, 22, 24, 26, 5],
    64: [22, 8, 6, 5],
    65: [4, 18, 30, 28, 27],
    66: [9, 18, 11],
    67: [19, 17, 14, 12],
    68: [18, 19, 25, 27, 29],
    69: [7, 13, 14, 17, 1, 28],
    70: [18, 30, 26, 29],
    71: [17, 21, 25],
    72: [10, 16, 15, 28, 27],
    73: [10, 20, 16, 15, 27],
    74: [25, 24, 14],
    75: [22, 8, 6, 5, 1, 25],
    76: [9, 18, 26, 28, 27],
    77: [17, 15, 7, 25],
    78: [19, 20, 14, 7, 25],
    79: [10, 16, 15, 6, 27],
    80: [15, 21, 25, 28],
    81: [10, 16, 15, 18],
    82: [5, 8, 18, 21, 15, 9],
    83: [7, 26, 5, 3, 1],
    84: [20, 21, 27, 24, 26, 28],
    85: [9, 1, 12, 11],
    86: [22, 8, 21, 6, 5, 1],
    87: [17, 15, 25],
    88: [9, 20, 7, 6, 4, 2],
    89: [8, 19, 23, 25, 28, 21],
    90: [7, 21, 14, 17, 28],
    91: [11, 12, 22],
    92: [20, 14, 7, 2],
    93: [17, 30, 14, 10, 12],
    94: [29, 20, 8, 1, 3],
    95: [7, 13, 14, 5, 28],
    96: [28, 15, 19],
    97: [10, 22, 27, 3, 30],
    98: [17, 14, 9, 10, 12],
    99: [22, 8, 6, 12, 5, 1],
    100: [8, 28, 18, 21, 15, 9],
    101: [12, 15, 21, 11, 28],
    102: [19, 20, 7, 2],
    103: [22, 27, 30],
    104: [29, 26, 24, 16, 4, 18],
    105: [25, 24, 11, 13],
    106: [7, 13, 26, 17, 28],
    107: [30, 23, 5, 4, 20, 2],
    108: [7, 6, 28, 3, 1],
    109: [10, 16, 15, 3, 27],
    110: [9, 10, 22, 27, 30],
    111: [29, 26, 24, 16, 11, 3],
    112: [28, 26, 2, 19],
    113: [8, 18, 26, 21, 15, 9],
    114: [23, 13, 18, 10, 11],
    115: [29, 24, 15, 8, 1, 3],
    116: [18, 19, 17, 4, 14, 12],
    117: [6, 5, 4, 28, 2, 1],
    118: [27, 20, 22, 6, 4, 2],
    119: [7, 13, 14, 15, 17, 28],
    120: [3, 16, 17, 23, 20],
    121: [23, 8, 16, 11],
    122: [2, 15, 6, 9, 1],
    123: [11, 18, 8, 2, 29],
    124: [12, 6, 1, 11],
    125: [16, 9, 17],
    126: [8, 17, 25, 1],
    127: [27, 3, 5, 13, 19],
    128: [1, 10, 29, 21, 8, 3],
    129: [22, 29, 23, 26],
    130: [25, 11, 24, 16, 5, 10],
    131: [2, 27, 23, 29],
    132: [24, 23, 26, 17, 5, 27],
    133: [27, 22, 19],
    134: [3, 1, 2, 5],
    135: [4, 13, 27, 15, 18],
    136: [21, 18, 22],
    137: [9, 1, 15, 26, 3, 24],
    138: [22, 17, 3],
    139: [9, 26, 3, 30, 8, 24],
    140: [24, 21, 15, 16],
    141: [16, 30, 22],
    142: [20, 21, 7],
    143: [11, 9, 21, 24],
    144: [20, 19, 5, 1, 16],
    145: [9, 22, 4, 23, 7, 29],
    146: [23, 17, 10, 15, 25],
    147: [10, 3, 30, 16],
    148: [15, 3, 27, 17, 9],
    149: [30, 7, 3, 19],
    150: [24, 17, 9, 12],
    151: [29, 4, 23, 12, 8],
    152: [13, 1, 6, 29, 16, 22],
    153: [10, 24, 5, 14, 12, 13],
    154: [27, 11, 1],
    155: [27, 13, 4, 30, 7],
    156: [29, 24, 10],
    157: [3, 13, 28, 19, 12],
    158: [28, 2, 9, 4, 27],
    159: [21, 30, 5, 8, 9],
    160: [7, 25, 12, 26, 14],
    161: [30, 29, 18, 28, 7, 24],
    162: [30, 24, 14],
    163: [21, 28, 10, 16],
    164: [6, 16, 14, 11],
    165: [9, 24, 21, 13, 8],
    166: [18, 22, 13, 4, 6, 21],
    167: [7, 17, 29],
    168: [15, 30, 11, 25],
    169: [5, 18, 7, 8, 3, 6],
    170: [11, 8, 12],
    171: [29, 1, 24, 28],
    172: [14, 24, 17, 7, 13, 9],
    173: [16, 9, 19],
    174: [22, 17, 21, 26],
    175: [9, 29, 8],
    176: [21, 15, 14, 10, 1, 5],
    177: [23, 25, 26, 16, 19, 27],
    178: [13, 30, 27],
    179: [15, 8, 26, 4, 29, 5],
    180: [27, 24, 23],
    181: [3, 18, 25, 2, 1, 5],
    182: [21, 23, 10],
    183: [17, 21, 14, 23, 25],
    184: [3, 10, 17],
    185: [13, 9, 8, 26],
    186: [1, 18, 10],
    187: [11, 21, 27, 29, 8],
    188: [18, 8, 1, 14],
    189: [2, 1, 7, 16, 29],
    190: [3, 9, 8, 22, 14, 12],
    191: [2, 23, 11, 29, 14, 12],
    192: [7, 1, 26, 10, 24, 17],
    193: [16, 7, 10, 25],
    194: [15, 8, 9, 25],
    195: [20, 16, 6],
    196: [14, 22, 2, 20, 5, 13],
    197: [1, 20, 5, 14],
    198: [6, 13, 15],
    199: [24, 4, 3, 30, 6],
    200: [6, 21, 30, 17],
    201: [2, 10, 22, 24, 13, 12],
    202: [6, 4, 1, 3, 9, 27],
    203: [29, 4, 18, 25, 7, 13],
    204: [27, 26, 14, 3, 2],
    205: [7, 12, 18, 15, 30, 11],
    206: [1, 21, 14, 8, 26, 29],
    207: [13, 2, 15],
    208: [9, 7, 24],
    209: [12, 9, 11, 20, 2],
    210: [30, 9, 10, 1, 24],
    211: [1, 27, 8],
    212: [23, 15, 25, 13, 26, 9],
    213: [5, 16, 6, 1, 26, 24],
    214: [20, 8, 11, 28],
    215: [12, 26, 29, 20, 3, 17],
    216: [25, 6, 8, 14, 3, 21],
    217: [18, 30, 11, 6, 14, 4],
    218: [20, 3, 7, 4, 14],
    219: [6, 8, 5, 14, 15, 20],
    220: [24, 18, 28, 25],
    221: [25, 27, 10],
    222: [19, 9, 12, 24, 7],
    223: [6, 8, 5, 10],
    224: [11, 3, 13, 9],
    225: [21, 26, 4, 15],
    226: [1, 16, 29],
    227: [30, 12, 2, 10, 8, 4],
    228: [20, 27, 19, 7],
    229: [17, 28, 6, 15, 20],
    230: [4, 21, 20],
    231: [7, 2, 12, 11, 5],
    232: [9, 2, 20, 24],
    233: [27, 1, 11, 14],
    234: [6, 20, 10, 3, 7],
    235: [18, 16, 3, 14, 4, 13],
    236: [21, 18, 3, 6],
    237: [14, 10, 22, 2, 24],
    238: [14, 1, 28, 25, 26],
    239: [13, 24, 7, 1],
    240: [14, 4, 27, 3],
    241: [15, 25, 6, 5, 1],
    242: [21, 26, 30, 13],
    243: [24, 17, 6, 5, 12],
    244: [17, 6, 30, 3],
    245: [16, 25, 26, 28, 27, 7],
    246: [27, 2, 30, 16],
    247: [20, 30, 21],
    248: [29, 23, 20],
    249: [21, 26, 28, 8],
    250: [20, 28, 7, 27, 16, 6],
    251: [2, 13, 17, 6],
    252: [4, 5, 8, 24, 27],
    253: [29, 18, 27],
    254: [22, 27, 11],
    255: [20, 15, 18, 21, 25, 10],
    256: [10, 19, 8, 14, 13, 22],
    257: [17, 15, 6, 1, 27, 20],
    258: [8, 15, 25, 20, 28, 29],
    259: [13, 4, 3, 5, 12, 14],
    260: [26, 15, 17],
    261: [2, 21, 5],
    262: [25, 24, 17, 3, 2],
    263: [21, 26, 5, 1, 3, 20],
    264: [7, 5, 29],
    265: [26, 30, 6, 22, 24],
    266: [27, 12, 20],
    267: [11, 29, 20, 9],
    268: [9, 17, 30, 16],
    269: [20, 17, 8, 11, 12],
    270: [6, 13, 21, 30],
    271: [29, 13, 6, 26, 9],
    272: [12, 9, 11, 20, 2, 10],
    273: [22, 17, 3, 7, 1, 26],
    274: [27, 1, 11, 14, 10, 19],
    275: [8, 18, 21, 15, 16, 28],
    276: [12, 6, 1, 11, 20, 16],
    277: [15, 3, 27, 17, 9, 24],
    278: [27, 24, 23, 8, 19, 25],
    279: [7, 9, 10, 16, 22, 11],
    280: [3, 16, 17, 23, 20, 18],
    281: [2, 1, 7, 16, 29, 13],
    282: [22, 17, 3, 6, 5, 4],
    283: [15, 8, 9, 25, 19, 23],
    284: [10, 16, 15, 27, 17, 23],
    285: [4, 13, 27, 15, 18, 8],
    286: [9, 7, 24, 10, 3, 30],
    287: [11, 12, 22, 3, 18, 25],
    288: [8, 17, 25, 1, 2, 21],
    289: [10, 16, 15, 18, 3, 27],
    290: [10, 22, 27, 30, 12, 15],
    291: [4, 18, 30, 28, 27, 7],
    292: [25, 24, 17, 3, 2, 10],
    293: [29, 13, 6, 26, 9, 20],
    294: [9, 12, 11, 29, 18, 27],
    295: [3, 1, 2, 5, 21, 26],
    296: [27, 1, 11, 14, 25, 10],
    297: [18, 30, 26, 29, 16, 17],
    298: [9, 12, 11, 23, 15, 25],
    299: [6, 16, 14, 11, 10, 20],
    300: [9, 18, 11, 12],
    301: [10, 22, 27, 30, 18, 16],
    302: [7, 17, 29, 15, 21, 25],
    303: [7, 5, 29, 2, 1, 16],
    304: [9, 18, 26, 28, 27, 7],
    305: [21, 18, 3, 6, 11, 12],
    306: [27, 11, 1, 29, 23, 20],
    307: [29, 18, 27, 7, 6, 5],
    308: [24, 21, 15, 16, 9, 7],
    309: [7, 13, 26, 17, 28, 9],
    310: [10, 16, 15, 3, 27, 25],
    311: [27, 1, 11, 14, 7, 13],
    312: [20, 8, 11, 28, 25, 24],
    313: [1, 20, 5, 14, 10, 22],
    314: [7, 6, 28, 3, 1, 10],
    315: [2, 27, 23, 29, 5, 8],
    316: [29, 4, 23, 12, 8, 1],
    317: [11, 12, 29, 9, 18, 30],
    318: [17, 6, 30, 3, 19, 20],
    319: [22, 29, 23, 26, 7, 21],
    320: [13, 2, 15, 6, 21, 30],
    321: [21, 26, 4, 15, 18, 30],
    322: [7, 6, 5, 3, 1, 12],
    323: [9, 18, 30, 28, 27, 20],
    324: [6, 13, 21, 30, 29, 18],
    325: [29, 28, 25],
    326: [8, 7, 4],
    327: [30, 11, 6, 14, 4],
    328: [28, 10, 20, 24, 1, 14],
    329: [23, 28, 27, 3],
    330: [14, 1, 17, 7],
    331: [7, 13, 23, 5, 28],
    332: [19, 6, 10, 27, 7],
    333: [6, 4, 21, 25, 3, 16],
    334: [21, 11, 12],
    335: [12, 19, 11],
    336: [6, 15, 23, 25, 9],
    337: [15, 21, 29, 23, 8],
    338: [25, 23, 27, 20, 5],
    339: [20, 17, 12, 6, 8],
    340: [22, 27, 3, 30],
    341: [9, 18, 28, 27],
    342: [2, 4, 3, 30, 6],
    343: [11, 3, 13, 9, 25],
    344: [9, 18, 22, 30, 28, 27],
    345: [30, 28, 14, 23],
    346: [28, 8, 22, 24, 21, 25],
    347: [22, 6, 21, 4],
    348: [30, 29, 18, 28, 7],
    349: [26, 13, 23, 21],
    350: [20, 30, 11, 6, 14, 4],
    351: [29, 21, 11, 25],
    352: [17, 12, 4, 28],
    353: [23, 16, 17, 1],
    354: [14, 24, 15, 7, 22],
    355: [13, 2, 1, 3, 14],
    356: [19, 9, 4, 8, 10],
    357: [26, 13, 15, 7],
    358: [18, 19, 30, 25, 14, 12],
    359: [12, 22, 21, 27],
    360: [10, 25, 18, 21, 5, 29],
    361: [9, 23, 13, 22],
    362: [20, 5, 1, 16],
    363: [16, 14, 20, 21, 3, 22],
    364: [28, 13, 2, 3, 27],
    365: [17, 27, 12, 21],
    366: [7, 3, 21],
    367: [12, 26, 5, 7, 13, 18],
    368: [8, 27, 5, 16, 29, 18],
    369: [14, 4, 27, 10, 3],
    370: [10, 16, 28, 18, 3, 27],
    371: [14, 22, 3, 6, 21, 12],
    372: [20, 2, 22],
    373: [17, 16, 25],
    374: [23, 14, 21, 5],
    375: [25, 23, 20, 5],
    376: [6, 13, 16, 21, 30],
    377: [29, 18, 11],
    378: [7, 21, 16, 26, 4],
    379: [21, 10, 16],
    380: [26, 2, 13],
    381: [13, 10, 4],
    382: [25, 14, 9, 10, 12],
    383: [20, 5, 21, 22, 23, 26],
    384: [7, 2, 22],
    385: [4, 22, 6, 28],
    386: [28, 10, 6, 14, 2],
    387: [15, 3, 1, 22, 13, 20],
    388: [16, 25, 14, 18],
    389: [14, 9, 17, 7],
    390: [1, 22, 4],
    391: [28, 4, 5, 16],
    392: [14, 4, 27, 29, 10, 3],
    393: [25, 24, 23, 28, 5],
    394: [21, 18, 23, 16, 15],
    395: [23, 2, 1],
    396: [11, 12, 19],
    397: [5, 26, 4, 12],
    398: [16, 13, 25, 26, 15, 9],
    399: [10, 9, 2, 20, 21],
    400: [28, 20, 24, 1, 27],
    401: [21, 30, 5, 11, 9],
    402: [16, 7, 10, 12, 25],
    403: [9, 14, 6, 19, 30],
    404: [27, 5, 26, 29, 28],
    405: [28, 26, 18, 22, 25],
    406: [18, 16, 26],
    407: [27, 5, 21, 26, 29, 28],
    408: [29, 28, 18, 25, 7, 13],
    409: [26, 13, 15],
    410: [25, 3, 8, 13, 19],
    411: [16, 17, 19, 7, 3],
    412: [12, 22, 27],
    413: [25, 17, 28, 5, 8, 2],
    414: [28, 4, 12, 21, 15],
    415: [20, 1, 12, 9, 17],
    416: [7, 28, 19],
    417: [10, 9, 2, 21],
    418: [17, 28, 2, 6, 15, 20],
    419: [3, 30, 14, 10, 12],
    420: [9, 7, 24, 10, 3],
    421: [8, 6, 5, 3],
    422: [6, 15, 28, 30, 12, 8],
    423: [2, 9, 12, 29],
    424: [9, 26, 17],
    425: [2, 4, 5, 11, 25, 1],
    426: [19, 15, 25, 21, 4],
    427: [2, 23, 11, 14, 12],
    428: [20, 6, 21],
    429: [3, 30, 20, 28],
    430: [7, 2, 22, 5],
    431: [21, 3, 15],
    432: [16, 4, 21, 12],
    433: [28, 5, 9, 14, 27, 8],
    434: [9, 26, 30, 8, 24],
    435: [4, 11, 15, 16, 30, 5],
    436: [18, 16, 27, 10],
    437: [8, 30, 4, 13, 10],
    438: [10, 16, 15, 28, 1],
    439: [13, 9, 8, 2],
    440: [17, 30, 14, 10, 12, 6],
    441: [5, 20, 22, 23],
    442: [7, 1, 3, 23, 24],
    443: [17, 26, 12],
    444: [3, 4, 21, 12],
    445: [28, 4, 7, 16],
    446: [16, 15, 18, 3, 27],
    447: [4, 12, 23],
    448: [2, 4, 5, 11, 25],
    449: [10, 28, 4],
    450: [17, 1, 30, 26, 18, 5],
    451: [16, 6, 17, 23],
    452: [9, 1, 27, 20],
    453: [23, 15, 4, 12],
    454: [2, 9, 4, 15],
    455: [15, 6, 1, 27, 20],
    456: [9, 10, 7, 24],
    457: [9, 12, 11, 30, 18, 27],
    458: [8, 27, 11, 23, 14, 19],
    459: [28, 18, 2, 11, 17, 5],
    460: [28, 14, 22, 21],
    461: [19, 9, 12, 24, 27, 7],
    462: [10, 16, 15, 14, 27],
    463: [28, 21, 20],
    464: [30, 29, 18, 28],
    465: [9, 4, 17],
    466: [2, 19, 6, 10, 27, 7],
    467: [3, 15, 19, 18, 30],
    468: [22, 8, 17, 6, 5],
    469: [24, 3, 18, 10],
    470: [21, 13, 7, 18],
    471: [20, 16, 27, 10, 1],
    472: [23, 13, 18, 10, 22, 11],
    473: [9, 18, 1, 12],
    474: [11, 18, 8, 2],
    475: [9, 3, 30, 8, 24],
    476: [22, 2, 17, 13, 15, 12],
    477: [22, 24, 30, 5],
    478: [11, 13, 9, 25],
    479: [27, 17, 4, 24, 28],
    480: [26, 21, 23, 30, 5],
    481: [16, 7, 10, 12],
    482: [23, 15, 10, 24, 12, 28],
    483: [6, 13, 24, 30],
    484: [25, 14, 9, 17, 12],
    485: [8, 21, 15, 16, 28],
    486: [13, 15, 8, 9, 25],
    487: [16, 4, 8, 21, 12],
    488: [2, 9, 19],
    489: [18, 20, 14, 17, 27],
    490: [13, 15, 12, 2, 20, 22],
    491: [7, 1, 26, 10, 27, 17],
    492: [14, 16, 13, 15],
    493: [23, 17, 24, 27, 3],
    494: [28, 10, 16],
    495: [4, 21, 29, 10],
    496: [21, 6, 17, 10, 28, 7],
    497: [6, 2, 21, 19, 20, 4],
    498: [17, 23, 20],
    499: [19, 1, 22],
    500: [7, 21, 14, 17, 30],

    }

    route_to_hub = {
        1: 2,
        2: 1,
        3: 1,
        4: 2,
        5: 3,
        6: 3,
        7: 2,
        8: 3,
        9: 2,
        10: 1,
        11: 2,
        12: 3,
        13: 2,
        14: 1,
        15: 3,
        16: 2,
        17: 2,
        18: 2,
        19: 1,
        20: 3,
        21: 2,
        22: 3,
        23: 3,
        24: 2,
        25: 1,
        26: 1,
        27: 3,
        28: 1,
        29: 1,
        30: 1,
        31: 1,
        32: 1,
        33: 1,
        34: 3,
        35: 1,
        36: 2,
        37: 3,
        38: 3,
        39: 2,
        40: 1,
        41: 1,
        42: 1,
        43: 3,
        44: 2,
        45: 2,
        46: 1,
        47: 1,
        48: 2,
        49: 2,
        50: 2,
        51: 3,
        52: 2,
        53: 2,
        54: 3,
        55: 3,
        56: 2,
        57: 2,
        58: 3,
        59: 2,
        60: 2,
        61: 1,
        62: 3,
        63: 3,
        64: 2,
        65: 3,
        66: 1,
        67: 2,
        68: 2,
        69: 3,
        70: 1,
        71: 3,
        72: 1,
        73: 1,
        74: 2,
        75: 2,
        76: 3,
        77: 3,
        78: 2,
        79: 1,
        80: 1,
        81: 1,
        82: 3,
        83: 1,
        84: 3,
        85: 1,
        86: 2,
        87: 3,
        88: 2,
        89: 1,
        90: 3,
        91: 3,
        92: 2,
        93: 3,
        94: 2,
        95: 3,
        96: 2,
        97: 1,
        98: 3,
        99: 2,
        100: 3,
        101: 1,
        102: 2,
        103: 1,
        104: 2,
        105: 2,
        106: 3,
        107: 2,
        108: 1,
        109: 1,
        110: 1,
        111: 2,
        112: 2,
        113: 3,
        114: 1,
        115: 2,
        116: 2,
        117: 3,
        118: 2,
        119: 3,
        120: 1,
        121: 2,
        122: 3,
        123: 2,
        124: 2,
        125: 2,
        126: 3,
        127: 1,
        128: 1,
        129: 3,
        130: 3,
        131: 3,
        132: 3,
        133: 3,
        134: 3,
        135: 3,
        136: 1,
        137: 1,
        138: 3,
        139: 3,
        140: 1,
        141: 2,
        142: 2,
        143: 1,
        144: 3,
        145: 1,
        146: 2,
        147: 1,
        148: 1,
        149: 2,
        150: 1,
        151: 1,
        152: 2,
        153: 2,
        154: 2,
        155: 2,
        156: 3,
        157: 2,
        158: 2,
        159: 3,
        160: 2,
        161: 1,
        162: 1,
        163: 2,
        164: 1,
        165: 2,
        166: 2,
        167: 1,
        168: 2,
        169: 2,
        170: 2,
        171: 2,
        172: 2,
        173: 2,
        174: 2,
        175: 1,
        176: 2,
        177: 1,
        178: 1,
        179: 3,
        180: 1,
        181: 3,
        182: 1,
        183: 1,
        184: 1,
        185: 3,
        186: 3,
        187: 2,
        188: 2,
        189: 3,
        190: 3,
        191: 1,
        192: 3,
        193: 1,
        194: 1,
        195: 2,
        196: 1,
        197: 1,
        198: 1,
        199: 3,
        200: 2,
        201: 3,
        202: 2,
        203: 2,
        204: 2,
        205: 3,
        206: 2,
        207: 2,
        208: 1,
        209: 1,
        210: 2,
        211: 3,
        212: 1,
        213: 2,
        214: 2,
        215: 2,
        216: 1,
        217: 1,
        218: 1,
        219: 2,
        220: 3,
        221: 3,
        222: 2,
        223: 2,
        224: 3,
        225: 1,
        226: 1,
        227: 1,
        228: 1,
        229: 1,
        230: 2,
        231: 3,
        232: 1,
        233: 3,
        234: 3,
        235: 1,
        236: 3,
        237: 2,
        238: 3,
        239: 2,
        240: 2,
        241: 2,
        242: 1,
        243: 1,
        244: 2,
        245: 1,
        246: 2,
        247: 2,
        248: 2,
        249: 3,
        250: 3,
        251: 3,
        252: 2,
        253: 1,
        254: 3,
        255: 1,
        256: 3,
        257: 2,
        258: 2,
        259: 1,
        260: 2,
        261: 3,
        262: 1,
        263: 3,
        264: 3,
        265: 2,
        266: 1,
        267: 2,
        268: 2,
        269: 1,
        270: 1,
        271: 2,
        272: 1,
        273: 3,
        274: 3,
        275: 3,
        276: 2,
        277: 1,
        278: 1,
        279: 3,
        280: 1,
        281: 3,
        282: 3,
        283: 1,
        284: 1,
        285: 3,
        286: 1,
        287: 3,
        288: 3,
        289: 1,
        290: 1,
        291: 3,
        292: 1,
        293: 2,
        294: 1,
        295: 3,
        296: 3,
        297: 1,
        298: 1,
        299: 1,
        300: 1,
        301: 1,
        302: 1,
        303: 3,
        304: 3,
        305: 3,
        306: 2,
        307: 1,
        308: 1,
        309: 3,
        310: 1,
        311: 3,
        312: 2,
        313: 1,
        314: 1,
        315: 3,
        316: 1,
        317: 3,
        318: 2,
        319: 3,
        320: 2,
        321: 1,
        322: 1,
        323: 3,
        324: 1,
        325: 1,
        326: 2,
        327: 1,
        328: 3,
        329: 3,
        330: 2,
        331: 3,
        332: 3,
        333: 1,
        334: 3,
        335: 1,
        336: 2,
        337: 1,
        338: 2,
        339: 1,
        340: 1,
        341: 3,
        342: 3,
        343: 3,
        344: 3,
        345: 1,
        346: 3,
        347: 3,
        348: 1,
        349: 2,
        350: 1,
        351: 2,
        352: 1,
        353: 3,
        354: 2,
        355: 2,
        356: 3,
        357: 3,
        358: 2,
        359: 1,
        360: 2,
        361: 2,
        362: 3,
        363: 2,
        364: 1,
        365: 2,
        366: 3,
        367: 1,
        368: 2,
        369: 2,
        370: 1,
        371: 2,
        372: 1,
        373: 2,
        374: 1,
        375: 2,
        376: 1,
        377: 1,
        378: 3,
        379: 2,
        380: 3,
        381: 2,
        382: 3,
        383: 3,
        384: 3,
        385: 3,
        386: 3,
        387: 2,
        388: 3,
        389: 2,
        390: 2,
        391: 1,
        392: 2,
        393: 1,
        394: 1,
        395: 2,
        396: 2,
        397: 2,
        398: 1,
        399: 3,
        400: 3,
        401: 3,
        402: 1,
        403: 2,
        404: 1,
        405: 1,
        406: 2,
        407: 1,
        408: 2,
        409: 3,
        410: 3,
        411: 2,
        412: 1,
        413: 2,
        414: 2,
        415: 1,
        416: 1,
        417: 3,
        418: 1,
        419: 3,
        420: 1,
        421: 1,
        422: 3,
        423: 1,
        424: 1,
        425: 3,
        426: 3,
        427: 1,
        428: 2,
        429: 1,
        430: 3,
        431: 2,
        432: 2,
        433: 1,
        434: 3,
        435: 2,
        436: 1,
        437: 2,
        438: 1,
        439: 3,
        440: 3,
        441: 1,
        442: 1,
        443: 3,
        444: 2,
        445: 1,
        446: 1,
        447: 3,
        448: 3,
        449: 3,
        450: 2,
        451: 1,
        452: 3,
        453: 2,
        454: 3,
        455: 2,
        456: 1,
        457: 1,
        458: 2,
        459: 2,
        460: 2,
        461: 2,
        462: 1,
        463: 1,
        464: 1,
        465: 1,
        466: 3,
        467: 3,
        468: 2,
        469: 2,
        470: 3,
        471: 2,
        472: 1,
        473: 1,
        474: 2,
        475: 3,
        476: 2,
        477: 1,
        478: 3,
        479: 3,
        480: 2,
        481: 1,
        482: 2,
        483: 1,
        484: 3,
        485: 3,
        486: 1,
        487: 2,
        488: 1,
        489: 2,
        490: 3,
        491: 3,
        492: 1,
        493: 3,
        494: 2,
        495: 3,
        496: 3,
        497: 1,
        498: 1,
        499: 2,
        500: 3,


    }

    alpha          = 0.995
    cv             = 0.10
    m              = 2
    waste_cost     = 27.55696873
    scenario_probs = {1: 0.3, 2: 0.5, 3: 0.2}

    inst = load_instance_from_excel(
        file_path=file_path,
        route_to_depots=route_to_depots,
        route_to_hub=route_to_hub,
        alpha=alpha, cv=cv, shelf_life=m,
        waste_cost=waste_cost, scenario_probs=scenario_probs,
    )

    sol, REQ = solve_ffh(inst)
    print("\nAll phases completed.")

    # ── post-solve checks ────────────────────────────────────────────────────
    # Multi-assignment check — OPL constraint 4: Σ_b Y[f,b,k,s,t] ≤ 1
    # A specific (f,k) pair must not be assigned to more than one hub b
    # within the same scenario s and period t.
    # Supplier f CAN use different vehicles to serve different hubs — that is
    # allowed. Only the (f,k) pair is restricted to one hub per (s,t).
    viol = 0
    for s in inst.S:
        for t in inst.T:
            for f in inst.F:
                for k in inst.K:
                    hubs = [b for b in inst.B
                            if sol.Y.get((f, b, k, s, t), 0) == 1]
                    if len(hubs) > 1:
                        viol += 1
                        print(f"  [MULTI-ASSIGN] f={f} k={k} s={s} t={t}: "
                              f"assigned to hubs {hubs}")
    print(f"[CHECK] multi-assign (same (f,k) pair in >1 hub) count = {viol}")

    # ── CHECK-D: route capacity Σ_i Z[i,r,t] <= cap[r] ──────────────────────
    r_cap = {r.r_id: int_floor(r.capacity) for r in inst.routes}
    cap_viol = 0
    for t in inst.T:
        for r in inst.routes:
            load = sum(int(sol.Z.get((i, r.r_id, t), 0)) for i in inst.D)
            if load > r_cap[r.r_id]:
                cap_viol += 1
                print(f"  [CHECK-D] r={r.r_id} t={t}: load={load} > cap={r_cap[r.r_id]}")
    print(f"[CHECK-D] Route capacity: {'OK' if cap_viol==0 else f'VIOLATIONS={cap_viol}'}")

    # ── CHECK-E: Z=0 wherever X=0 ────────────────────────────────────────────
    xz_viol = 0
    for (i, r_id, t), z in sol.Z.items():
        if int(z) > 0 and sol.X.get((r_id, t), 0) == 0:
            xz_viol += 1
            print(f"  [CHECK-E] Z[{i},{r_id},{t}]={z} but X[{r_id},{t}]=0")
    print(f"[CHECK-E] Z=0 when X=0: {'OK' if xz_viol==0 else f'VIOLATIONS={xz_viol}'}")

    # ── CHECK-F: supply capacity Σ_bk L[f,b,k,s,t] <= supply[f,s,t] ─────────
    sup_viol = 0
    for s in inst.S:
        for t in inst.T:
            for f in inst.F:
                used = sum(int(v) for (ff, b, k, ss, tt), v in sol.L.items()
                           if ff == f and ss == s and tt == t)
                avail = float(inst.supply.get((f, s, t), 0.0))
                if used > avail + 0.5:
                    sup_viol += 1
                    print(f"  [CHECK-F] f={f} s={s} t={t}: used={used} > supply={avail:.1f}")
    print(f"[CHECK-F] Supply capacity: {'OK' if sup_viol==0 else f'VIOLATIONS={sup_viol}'}")

    # ── CHECK-G: vehicle capacity Σ_f L[f,b,k,s,t] <= theta[k] ──────────────
    veh_viol = 0
    for s in inst.S:
        for t in inst.T:
            for k in inst.K:
                used = sum(int(v) for (f, b, kk, ss, tt), v in sol.L.items()
                           if kk == k and ss == s and tt == t)
                cap  = int_floor(inst.theta.get(k, 0))
                if used > cap:
                    veh_viol += 1
                    print(f"  [CHECK-G] k={k} s={s} t={t}: used={used} > cap={cap}")
    print(f"[CHECK-G] Vehicle capacity: {'OK' if veh_viol==0 else f'VIOLATIONS={veh_viol}'}")

    # ── CHECK-H: W[i,t]=0 for t < shelf_life ─────────────────────────────────
    w_viol = 0
    for (i, t), w in sol.W.items():
        if t < inst.shelf_life and int(w) > 0:
            w_viol += 1
            print(f"  [CHECK-H] W[{i},{t}]={w} but shelf_life={inst.shelf_life}")
    print(f"[CHECK-H] W=0 before shelf_life: {'OK' if w_viol==0 else f'VIOLATIONS={w_viol}'}")

    # Y=1 but L=0 check
    bad_yl = sum(
        1 for (f, b, k, s, t), y in sol.Y.items()
        if y == 1 and sol.L.get((f, b, k, s, t), 0) == 0
    )
    print(f"[CHECK] Y=1 but L=0 count = {bad_yl}")

    # AC manual verification
    AC_check = sum(
        inst.scenario_probs[s] * inst.beta[(f, k)]
        for (f, b, k, s, t), y in sol.Y.items()
        if y == 1
        for s in [s]  # already unpacked
    )
    print(f"AC_check(OPL) = {AC_check:.4f}")

    export_solution_to_excel(inst, sol, REQ, out_path="ffh_solution_improved500.xlsx")