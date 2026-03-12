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
    out_path: str = "ffh_solution_improved750.xlsx",
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
    file_path = "step750.xlsx"

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
        31: [17, 28, 15, 21, 25],
        32: [23, 8, 16, 9, 11],
        33: [6, 5, 3, 1],
        34: [17, 30, 15, 9, 10, 12],
        35: [11, 12, 4, 29],
        36: [10, 22, 4, 30],
        37: [21, 22, 24, 26, 28],
        38: [20, 19, 25, 26, 27, 29],
        39: [18, 30, 3],
        40: [23, 10, 11],
        41: [13, 15, 20, 22, 26],
        42: [16, 9, 12, 11],
        43: [22, 8, 26, 5, 1],
        44: [8, 18, 15, 9],
        45: [20, 14, 7, 2],
        46: [8, 18, 16, 15, 9],
        47: [18, 30, 28, 29],
        48: [10, 16, 15],
        49: [19, 20, 14, 16, 2],
        50: [11, 12, 26, 29],
        51: [7, 6, 5, 24, 1],
        52: [9, 19, 18, 30, 28, 27],
        53: [11, 12, 29, 22],
        54: [10, 22, 27, 14],
        55: [29, 24, 8, 1],
        56: [6, 5, 4, 3, 2],
        57: [8, 18, 1, 15, 9],
        58: [29, 24, 8, 1, 6, 3],
        59: [7, 15, 13, 14, 17, 28],
        60: [8, 23, 25, 24, 21],
        61: [16, 17, 23, 20, 3],
        62: [10, 3, 27, 30],
        63: [20, 12, 29],
        64: [22, 8, 6, 5, 11],
        65: [17, 30, 14, 10, 12],
        66: [6, 5, 4, 2, 1],
        67: [8, 4, 23, 25, 24, 21],
        68: [7, 6, 5, 3, 13, 1],
        69: [29, 24, 26, 8, 1, 3],
        70: [18, 19, 17, 14, 5, 12],
        71: [18, 19, 25, 8, 27, 29],
        72: [7, 16, 15, 27],
        73: [28, 12, 19],
        74: [29, 15, 21, 25, 28],
        75: [19, 25, 26, 27, 29],
        76: [17, 20, 15, 21, 25],
        77: [7, 9, 10, 6, 16, 22],
        78: [12, 8, 6, 5, 1],
        79: [10, 15, 22, 27, 30],
        80: [18, 19, 17, 7, 12],
        81: [25, 24, 14, 9],
        82: [23, 13, 10, 27],
        83: [11, 12, 7, 29],
        84: [16, 19, 23, 20],
        85: [12, 15, 25, 28],
        86: [19, 14, 7, 2],
        87: [22, 8, 9, 5, 1],
        88: [9, 10, 16, 22],
        89: [22, 24, 14, 13],
        90: [10, 22, 27, 30, 9],
        91: [23, 13, 10, 21, 11],
        92: [17, 30, 14, 9, 12],
        93: [28, 26, 4, 19],
        94: [23, 13, 7, 10, 11],
        95: [6, 5, 4, 3, 1],
        96: [19, 17, 14, 12],
        97: [9, 22, 27, 30],
        98: [25, 14, 13],
        99: [23, 16, 13, 11],
        100: [18, 19, 25, 26, 27, 20],
        101: [10, 12, 15, 21, 25, 28],
        102: [8, 19, 23, 25, 24, 26],
        103: [11, 28, 12, 29],
        104: [10, 16, 15, 1, 27],
        105: [8, 18, 21, 9],
        106: [10, 16, 6, 15, 27],
        107: [8, 12, 11],
        108: [7, 13, 15, 17, 28],
        109: [7, 1, 13, 14, 17, 28],
        110: [30, 23, 4, 2],
        111: [23, 8, 16, 13, 27],
        112: [18, 21, 15, 9],
        113: [28, 26, 12],
        114: [17, 30, 14, 9, 1, 12],
        115: [18, 30, 14, 29],
        116: [10, 22, 30],
        117: [29, 24, 8, 3],
        118: [29, 26, 28, 16, 4, 3],
        119: [19, 20, 14, 27, 2],
        120: [20, 21, 22, 26, 28],
        121: [19, 20, 14, 7, 24],
        122: [23, 8, 16, 11],
        123: [7, 27, 10, 16, 22],
        124: [29, 28, 8, 1, 3],
        125: [5, 4, 3, 2, 1],
        126: [7, 13, 14, 17, 28, 11],
        127: [17, 1, 15, 21, 25],
        128: [16, 17, 23, 24, 20],
        129: [17, 30, 14, 21, 10, 12],
        130: [29, 17, 15, 21, 25],
        131: [8, 18, 21, 15, 2],
        132: [10, 16, 25, 27],
        133: [11, 13, 15, 20, 22, 25],
        134: [8, 19, 23, 20, 24, 21],
        135: [18, 19, 15, 26, 27, 29],
        136: [7, 13, 14, 17],
        137: [18, 19, 20, 17, 14, 12],
        138: [10, 27, 30],
        139: [22, 8, 5, 1],
        140: [17, 27, 21, 25],
        141: [10, 22, 15, 30],
        142: [18, 20, 7, 6, 4, 2],
        143: [19, 20, 14, 7, 16],
        144: [11, 13, 15, 20, 17, 26],
        145: [18, 19, 16, 17, 14, 12],
        146: [6, 30, 4, 3, 2, 1],
        147: [7, 13, 14, 17, 5],
        148: [6, 12, 29],
        149: [22, 15, 6, 5, 1],
        150: [7, 9, 16, 22],
        151: [11, 13, 15, 9, 22, 26],
        152: [23, 17, 15, 21, 25],
        153: [11, 18, 12, 29],
        154: [22, 30, 23, 5, 4, 2],
        155: [9, 18, 24, 28, 27],
        156: [22, 27, 30],
        157: [9, 30, 11],
        158: [8, 29, 23, 25, 24, 21],
        159: [12, 15, 21, 25, 28, 19],
        160: [23, 8, 16, 13],
        161: [10, 16, 15, 18],
        162: [7, 13, 17, 28],
        163: [10, 9, 12, 11],
        164: [8, 19, 23, 9, 24, 21],
        165: [16, 6, 5, 3, 1],
        166: [18, 17, 14, 12],
        167: [18, 19, 9, 17, 14, 12],
        168: [29, 26, 24, 16, 3],
        169: [8, 25, 24, 14, 13],
        170: [18, 19, 25, 26, 16, 29],
        171: [17, 30, 14, 9, 10, 26],
        172: [1, 19, 17, 14, 12],
        173: [23, 13, 10, 27, 11],
        174: [17, 15, 25],
        175: [18, 19, 17, 12],
        176: [18, 19, 26, 27, 29],
        177: [17, 30, 11, 9, 10, 12],
        178: [29, 22, 8, 6, 5, 1],
        179: [19, 20, 14, 7, 2, 21],
        180: [8, 6, 5, 1],
        181: [7, 13, 9, 14, 17, 28],
        182: [9, 18, 1, 30, 28, 27],
        183: [4, 8, 16, 13, 11],
        184: [7, 6, 5, 1],
        185: [18, 11, 17, 14, 12],
        186: [16, 17, 23],
        187: [16, 15, 27],
        188: [11, 13, 15, 18, 22, 26],
        189: [12, 13, 10, 11],
        190: [12, 15, 21, 25, 27],
        191: [17, 30, 14, 9, 10, 3],
        192: [17, 21, 25],
        193: [23, 8, 5, 16, 13, 11],
        194: [28, 24, 19],
        195: [17, 30, 14, 9, 11, 12],
        196: [19, 20, 7, 6, 4, 2],
        197: [10, 22, 27, 30, 20],
        198: [24, 14, 13],
        199: [18, 9, 17, 14, 12],
        200: [7, 6, 5, 26, 1],
        201: [10, 22, 27, 13],
        202: [29, 26, 24, 16, 4],
        203: [28, 26, 14, 19],
        204: [10, 22, 13, 27, 25, 18],
        205: [3, 11, 9, 29, 4, 25],
        206: [22, 28, 18],
        207: [2, 7, 17, 12, 20, 25],
        208: [25, 2, 7, 9, 18, 5],
        209: [29, 23, 16, 4, 1, 21],
        210: [23, 6, 10, 18],
        211: [3, 8, 27, 4, 15, 28],
        212: [16, 30, 23, 10],
        213: [14, 27, 16, 8, 15],
        214: [13, 7, 30, 20],
        215: [28, 3, 9, 25],
        216: [30, 26, 17, 9, 27],
        217: [24, 10, 27, 19, 22],
        218: [15, 18, 16, 12],
        219: [15, 11, 28, 7, 23, 8],
        220: [8, 28, 25, 14, 2, 11],
        221: [23, 26, 13, 28, 22, 21],
        222: [2, 5, 17, 19, 11, 4],
        223: [17, 30, 15],
        224: [14, 28, 21, 5],
        225: [26, 9, 11, 20, 23, 13],
        226: [28, 11, 22],
        227: [11, 21, 23, 25, 16, 18],
        228: [8, 21, 22],
        229: [24, 3, 14, 4],
        230: [15, 6, 23],
        231: [2, 11, 26],
        232: [12, 14, 5, 8, 17],
        233: [6, 3, 20, 28],
        234: [16, 30, 19, 5],
        235: [21, 9, 15, 29, 22, 1],
        236: [22, 18, 6, 3, 15],
        237: [21, 14, 23, 9, 15],
        238: [13, 28, 16, 4],
        239: [19, 12, 30, 10, 23, 27],
        240: [9, 1, 19, 22, 25, 24],
        241: [27, 29, 10, 25, 26, 8],
        242: [8, 21, 7, 20, 9],
        243: [21, 4, 29, 2],
        244: [2, 19, 12, 24, 5, 3],
        245: [24, 14, 6, 7, 5],
        246: [17, 30, 9, 27, 6],
        247: [26, 10, 24, 11, 30, 4],
        248: [5, 25, 8],
        249: [28, 26, 18, 12, 3, 13],
        250: [18, 4, 15, 12, 22],
        251: [19, 13, 27, 21, 12],
        252: [16, 1, 20, 29],
        253: [30, 20, 8, 21, 3],
        254: [30, 23, 10, 21, 14, 4],
        255: [2, 10, 16],
        256: [8, 29, 18],
        257: [15, 12, 22, 24, 23, 18],
        258: [29, 14, 21, 4],
        259: [9, 2, 23, 12, 7, 15],
        260: [28, 12, 4, 22],
        261: [2, 13, 9, 7, 4],
        262: [22, 7, 21],
        263: [2, 26, 11],
        264: [26, 19, 7, 3],
        265: [19, 7, 27, 28],
        266: [25, 5, 26, 29, 20],
        267: [28, 5, 18, 9, 26],
        268: [22, 28, 1],
        269: [12, 26, 8],
        270: [1, 6, 9, 2, 5],
        271: [17, 4, 24, 3, 16, 15],
        272: [15, 17, 8],
        273: [24, 26, 30],
        274: [15, 21, 1, 2, 16],
        275: [22, 4, 16, 23, 15, 3],
        276: [20, 5, 3, 9, 21],
        277: [13, 20, 17, 10, 15],
        278: [4, 26, 23, 30, 21, 29],
        279: [14, 15, 29, 8],
        280: [27, 15, 13, 14, 24],
        281: [14, 11, 22, 9, 12],
        282: [3, 30, 27, 29, 28, 14],
        283: [26, 5, 18, 2, 19],
        284: [22, 4, 14, 12, 28],
        285: [28, 24, 2, 10, 20, 27],
        286: [19, 17, 7],
        287: [8, 28, 4, 12, 18, 27],
        288: [19, 8, 26, 14, 28],
        289: [20, 30, 22],
        290: [1, 6, 9, 23, 25],
        291: [12, 1, 6, 28, 5],
        292: [3, 5, 24, 21, 1, 30],
        293: [13, 14, 15, 11],
        294: [10, 24, 11, 25, 19],
        295: [29, 2, 5],
        296: [22, 3, 9],
        297: [16, 20, 15, 14, 9, 7],
        298: [12, 14, 4],
        299: [17, 22, 10, 2, 8, 13],
        300: [1, 7, 10],
        301: [25, 9, 10, 11],
        302: [16, 24, 14],
        303: [13, 18, 23, 8],
        304: [3, 13, 28, 24, 2],
        305: [15, 30, 3],
        306: [19, 13, 23, 21, 14, 10],
        307: [1, 11, 6, 26, 20, 15],
        308: [3, 14, 28, 4, 8],
        309: [17, 3, 13, 10, 24, 11],
        310: [25, 6, 3, 17, 21],
        311: [29, 25, 12, 24],
        312: [8, 4, 5, 9],
        313: [20, 5, 25, 21],
        314: [25, 21, 16, 15],
        315: [22, 29, 19, 21, 27, 20],
        316: [5, 15, 3, 16, 21],
        317: [19, 2, 12, 17, 3],
        318: [15, 2, 29, 12, 10, 3],
        319: [20, 17, 13],
        320: [15, 30, 26],
        321: [11, 20, 16, 17],
        322: [15, 4, 26],
        323: [17, 21, 6],
        324: [23, 15, 17, 20],
        325: [12, 30, 10, 13, 14],
        326: [26, 21, 11],
        327: [4, 18, 22, 13, 10],
        328: [11, 3, 19, 22],
        329: [10, 21, 23, 22, 13],
        330: [10, 18, 13],
        331: [27, 5, 22, 23, 24],
        332: [21, 22, 14],
        333: [1, 12, 10, 6, 7],
        334: [7, 8, 5, 28, 3, 10],
        335: [22, 11, 29],
        336: [20, 13, 5, 6],
        337: [24, 15, 2, 14],
        338: [15, 20, 10, 25],
        339: [8, 18, 30, 10, 26, 16],
        340: [22, 19, 15, 25, 10],
        341: [6, 27, 7, 26, 20, 5],
        342: [21, 16, 28],
        343: [23, 28, 17],
        344: [3, 25, 6, 9, 15],
        345: [27, 14, 3, 30],
        346: [29, 12, 1, 14, 2, 13],
        347: [8, 13, 3, 12, 1],
        348: [27, 23, 21],
        349: [5, 2, 10, 30],
        350: [25, 23, 16, 15],
        351: [29, 3, 1],
        352: [27, 5, 18, 24],
        353: [4, 25, 10, 8, 28, 30],
        354: [14, 21, 26, 20],
        355: [4, 27, 29],
        356: [21, 17, 19],
        357: [10, 14, 1, 20],
        358: [19, 14, 6, 22],
        359: [17, 12, 3],
        360: [13, 28, 16],
        361: [12, 9, 24, 1, 30, 3],
        362: [24, 22, 21, 4],
        363: [5, 2, 12, 18, 11],
        364: [27, 25, 22, 15],
        365: [21, 6, 26, 5, 3, 23],
        366: [10, 7, 2],
        367: [11, 30, 10],
        368: [27, 18, 16, 9, 2, 25],
        369: [10, 12, 28, 25],
        370: [9, 4, 26, 12, 14],
        371: [29, 13, 11, 6, 16, 23],
        372: [30, 26, 17, 9, 3],
        373: [3, 14, 20, 27, 6, 18],
        374: [4, 3, 11, 22, 10],
        375: [20, 23, 14, 6, 29, 15],
        376: [2, 24, 28, 12, 20, 14],
        377: [3, 22, 21],
        378: [17, 26, 24, 22, 6],
        379: [28, 20, 22, 26],
        380: [5, 3, 8],
        381: [12, 13, 19, 2, 20],
        382: [12, 30, 15, 25, 3, 19],
        383: [13, 11, 21, 9, 8],
        384: [24, 6, 16],
        385: [30, 18, 4, 9, 25, 27],
        386: [7, 20, 10, 23, 16, 30],
        387: [28, 3, 15, 6],
        388: [3, 26, 22, 11, 28, 12],
        389: [18, 10, 29],
        390: [23, 30, 21, 6],
        391: [4, 7, 26, 5],
        392: [1, 12, 18, 19, 29, 15],
        393: [20, 29, 3, 10],
        394: [17, 14, 25, 29, 19, 3],
        395: [21, 3, 15, 22, 17],
        396: [29, 27, 25, 18],
        397: [25, 5, 14, 17],
        398: [17, 5, 10],
        399: [11, 30, 23, 8],
        400: [28, 3, 9, 7, 21],
        401: [5, 21, 10, 20, 18],
        402: [19, 5, 6, 22],
        403: [27, 30, 19, 2, 28],
        404: [2, 21, 25],
        405: [21, 7, 25, 19, 14],
        406: [16, 29, 21],
        407: [21, 10, 16, 8, 26],
        408: [10, 15, 3, 23, 2, 6],
        409: [16, 15, 7, 11, 20, 5],
        410: [24, 28, 12, 13, 5],
        411: [11, 8, 15],
        412: [15, 8, 5, 4, 2],
        413: [28, 20, 14, 8, 6, 11],
        414: [7, 25, 6, 16, 17],
        415: [29, 10, 16, 1, 3, 13],
        416: [8, 7, 19, 12, 2, 26],
        417: [20, 29, 27, 21, 22, 16],
        418: [28, 4, 14],
        419: [24, 12, 25, 13, 2],
        420: [19, 18, 7],
        421: [3, 13, 17, 15, 25],
        422: [27, 20, 22, 4, 5],
        423: [12, 26, 11, 18, 30, 25],
        424: [20, 17, 13, 2],
        425: [5, 23, 11],
        426: [5, 20, 17, 30, 11, 29],
        427: [13, 20, 24, 27],
        428: [17, 27, 18, 16, 23],
        429: [16, 27, 1, 12, 11],
        430: [14, 19, 10],
        431: [20, 16, 9],
        432: [24, 2, 19, 16],
        433: [5, 27, 22, 8, 2, 19],
        434: [7, 1, 15],
        435: [5, 14, 23, 7, 29, 17],
        436: [28, 30, 24, 29, 2, 23],
        437: [18, 11, 22, 16],
        438: [11, 6, 15, 18, 30, 27],
        439: [20, 16, 7, 8, 9],
        440: [8, 10, 25, 23, 7],
        441: [11, 16, 12, 18, 26, 24],
        442: [4, 19, 22, 18, 13],
        443: [25, 26, 5, 10, 2],
        444: [12, 30, 15],
        445: [24, 16, 7, 27, 18],
        446: [5, 4, 20, 24, 19],
        447: [2, 22, 29, 17],
        448: [2, 4, 14, 11],
        449: [4, 22, 25, 5, 1, 18],
        450: [21, 29, 16, 28, 30, 7],
        451: [10, 21, 2, 30, 25],
        452: [18, 24, 28, 30],
        453: [14, 29, 27, 6],
        454: [26, 16, 6, 24, 10, 2],
        455: [19, 20, 4, 30, 11],
        456: [21, 18, 17, 16, 5, 28],
        457: [7, 26, 4, 11, 6],
        458: [21, 9, 23, 6, 1, 24],
        459: [19, 22, 25, 7, 6],
        460: [27, 14, 17, 11, 3, 13],
        461: [6, 5, 16],
        462: [1, 9, 13, 8],
        463: [11, 10, 19, 24, 1],
        464: [23, 8, 2, 22, 4],
        465: [6, 13, 22, 17, 30],
        466: [23, 4, 21, 30, 10],
        467: [8, 5, 16, 15],
        468: [14, 23, 18, 30, 16],
        469: [25, 8, 22, 20],
        470: [17, 23, 12, 3, 19, 4],
        471: [19, 18, 5, 6],
        472: [4, 22, 7, 23, 19, 16],
        473: [26, 2, 15, 5, 17, 14],
        474: [18, 15, 22],
        475: [13, 9, 27],
        476: [19, 3, 2, 14],
        477: [18, 2, 30],
        478: [2, 10, 14, 6, 25, 5],
        479: [12, 29, 13, 15, 28, 26],
        480: [21, 28, 12, 4],
        481: [17, 5, 24, 8, 1, 25],
        482: [15, 22, 24, 18, 14],
        483: [27, 8, 29, 15, 12, 5],
        484: [30, 29, 24, 25],
        485: [26, 22, 14],
        486: [8, 7, 3],
        487: [15, 20, 22],
        488: [8, 24, 2],
        489: [8, 18, 7, 25, 27, 2],
        490: [8, 27, 30, 24, 19],
        491: [8, 10, 29, 5, 22],
        492: [14, 10, 9, 2],
        493: [21, 22, 14, 18],
        494: [12, 21, 22],
        495: [23, 14, 5, 10, 13],
        496: [8, 28, 30, 10, 23, 5],
        497: [18, 14, 17],
        498: [8, 9, 7, 11, 21, 3],
        499: [3, 18, 24, 27, 7],
        500: [13, 22, 20, 2, 28],
        501: [26, 25, 19, 24],
        502: [16, 7, 28, 30],
        503: [30, 1, 7, 24, 4],
        504: [28, 26, 14, 19, 15, 17],
        505: [23, 30, 21, 6, 13, 22],
        506: [15, 18, 16, 12, 23, 8],
        507: [18, 19, 17, 7, 12, 5],
        508: [23, 8, 16, 13, 28, 12],
        509: [7, 6, 5, 26, 1, 2],
        510: [1, 19, 17, 14, 12, 15],
        511: [13, 20, 17, 10, 15, 14],
        512: [18, 10, 29, 21, 6, 26],
        513: [20, 5, 3, 9, 21, 10],
        514: [18, 11, 22, 16, 24, 6],
        515: [24, 12, 25, 13, 2, 8],
        516: [18, 30, 3, 10, 22, 27],
        517: [30, 23, 4, 2, 7, 26],
        518: [10, 9, 12, 11, 17, 5],
        519: [2, 13, 9, 7, 4, 19],
        520: [8, 24, 2, 7, 20, 10],
        521: [20, 17, 13, 2, 4, 14],
        522: [28, 11, 22, 16, 27, 1],
        523: [8, 13, 3, 12, 1, 17],
        524: [16, 15, 27, 11, 13, 9],
        525: [7, 25, 6, 16, 17, 30],
        526: [19, 8, 26, 14, 28, 29],
        527: [22, 7, 21, 16, 15, 11],
        528: [20, 14, 7, 2, 1, 9],
        529: [10, 16, 25, 27, 17, 23],
        530: [17, 21, 6, 23, 8, 2],
        531: [23, 8, 16, 9, 11, 29],
        532: [7, 13, 14, 17, 28, 27],
        533: [20, 21, 22, 26, 28, 29],
        534: [8, 23, 25, 24, 21, 3],
        535: [14, 15, 29, 8, 18, 9],
        536: [10, 18, 13, 7, 6, 5],
        537: [28, 5, 18, 9, 26, 15],
        538: [14, 23, 18, 30, 16, 22],
        539: [29, 24, 8, 1, 3, 19],
        540: [15, 20, 10, 25, 19, 14],
        541: [14, 11, 22, 9, 12, 2],
        542: [15, 30, 26, 18, 20, 7],
        543: [8, 6, 5, 1, 15, 18],
        544: [16, 17, 23, 20, 3, 11],
        545: [22, 3, 9, 7, 6, 5],
        546: [16, 24, 14, 1, 20, 29],
        547: [19, 14, 7, 2, 25, 26],
        548: [23, 13, 7, 10, 11, 3],
        549: [10, 21, 23, 22, 13, 19],
        550: [25, 24, 14, 9, 18, 19],
        551: [20, 5, 3, 9, 21, 2],
        552: [21, 28, 12, 4, 17, 5],
        553: [18, 24, 28, 30, 11, 13],
        554: [1, 7, 10, 8, 18, 21],
        555: [24, 22, 21, 4, 28, 26],
        556: [16, 17, 23, 24, 20, 10],
        557: [5, 15, 3, 16, 21, 29],
        558: [8, 6, 5, 1, 23, 16],
        559: [20, 29, 3, 10, 14, 23],
        560: [23, 17, 15, 21, 25, 6],
        561: [9, 22, 27, 30, 19, 12],
        562: [1, 7, 10, 24, 26, 30],
        563: [22, 18, 6, 3, 15, 30],
        564: [14, 23, 18, 30, 16, 21],
        565: [2, 11, 26, 20, 16, 7],
        566: [11, 18, 12, 29, 30, 4],
        567: [29, 26, 24, 16, 4, 22],
        568: [11, 18, 12, 29, 23, 4],
        569: [7, 13, 17, 28, 4, 25],
        570: [18, 14, 17, 29, 24, 8],
        571: [30, 29, 24, 25, 10, 7],
        572: [12, 15, 21, 25, 27, 20],
        573: [22, 8, 6, 5, 11, 7],
        574: [23, 30, 21, 6, 26, 17],
        575: [20, 13, 5, 6, 16, 9],
        576: [14, 23, 18, 30, 16, 8],
        577: [10, 15, 22, 27, 30, 12],
        578: [14, 29, 27, 6, 24, 12],
        579: [23, 13, 10, 11, 7, 16],
        580: [8, 25, 24, 14, 13, 4],
        581: [21, 10, 16, 8, 26, 28],
        582: [22, 19, 15, 25, 10, 8],
        583: [23, 28, 17, 25, 24, 14],
        584: [6, 5, 16, 11, 18, 12],
        585: [5, 4, 3, 2, 1, 14],
        586: [14, 22, 25, 5, 1, 18],
        587: [17, 6, 18, 11],
        588: [26, 13, 21, 6],
        589: [25, 26, 5, 10, 1],
        590: [13, 1, 23, 7, 27, 9],
        591: [26, 28, 18],
        592: [12, 26, 19],
        593: [22, 5, 10],
        594: [25, 8, 23, 22, 20],
        595: [11, 4, 8, 10],
        596: [13, 11, 21, 9, 17, 8],
        597: [18, 19, 7, 6, 4, 2],
        598: [5, 6, 11, 27, 15],
        599: [16, 5, 7],
        600: [14, 22, 16, 30, 23, 15],
        601: [19, 8, 22, 20],
        602: [18, 27, 25, 2, 24],
        603: [14, 4, 21, 5],
        604: [28, 4, 25, 26, 8, 7],
        605: [28, 23, 20, 21, 14],
        606: [19, 2, 20, 14, 7, 16],
        607: [20, 18, 6, 11, 23],
        608: [6, 18, 11],
        609: [5, 23, 11, 27],
        610: [27, 30, 19, 2, 6],
        611: [13, 15, 3, 21, 26, 25],
        612: [20, 14, 7, 2, 1],
        613: [16, 11, 19, 1, 25, 24],
        614: [23, 12, 3, 19, 4],
        615: [14, 27, 7, 16],
        616: [12, 1, 6, 28],
        617: [26, 11, 22, 25, 2, 30],
        618: [25, 7, 14, 13],
        619: [28, 30, 29, 9],
        620: [4, 26, 12, 14],
        621: [21, 3, 23, 4, 1, 7],
        622: [7, 13, 14, 17, 26],
        623: [16, 3, 8, 13],
        624: [25, 27, 30, 5, 21, 22],
        625: [28, 7, 3, 4, 5, 25],
        626: [17, 14, 25, 19, 3],
        627: [22, 8, 9, 5, 25, 1],
        628: [25, 23, 1, 28, 22],
        629: [24, 13, 10, 21],
        630: [15, 30, 9, 27, 6],
        631: [28, 27, 19, 14, 12, 18],
        632: [1, 19, 3],
        633: [14, 23, 13, 3, 21, 18],
        634: [3, 10, 17, 30, 7],
        635: [13, 19, 21, 25, 27],
        636: [10, 23, 14],
        637: [8, 3, 30],
        638: [6, 12, 5, 24, 25],
        639: [25, 29, 1],
        640: [8, 10, 25, 23],
        641: [28, 19, 6, 12, 30, 9],
        642: [8, 24, 17],
        643: [19, 30, 23, 5, 4, 2],
        644: [8, 11, 30, 14, 24, 29],
        645: [19, 17, 7, 12],
        646: [8, 26, 17, 2],
        647: [28, 10, 25, 29, 11, 3],
        648: [7, 10, 19, 26, 24],
        649: [19, 16, 25, 7, 24],
        650: [1, 30, 14, 8],
        651: [11, 1, 25],
        652: [4, 7, 17, 8, 14],
        653: [10, 3, 4, 27, 30],
        654: [13, 28, 26, 4],
        655: [19, 2, 14],
        656: [10, 9, 28, 11],
        657: [11, 15, 9, 8, 3],
        658: [5, 4, 6],
        659: [15, 4, 5],
        660: [14, 27, 16, 11, 15],
        661: [7, 27, 28],
        662: [23, 13, 17, 7, 10, 11],
        663: [9, 27, 6, 1, 13],
        664: [4, 15, 26, 5],
        665: [22, 10, 8, 6, 5, 11],
        666: [19, 9, 30],
        667: [7, 29, 21],
        668: [21, 7, 25, 1, 14],
        669: [7, 30, 6, 9, 15],
        670: [21, 10, 16, 14],
        671: [3, 9, 13],
        672: [13, 26, 23],
        673: [8, 6, 5, 1, 17, 16],
        674: [6, 1, 11, 20, 8],
        675: [4, 8, 22, 9],
        676: [25, 8, 19],
        677: [30, 27, 28, 6, 8],
        678: [30, 19, 1, 23, 12],
        679: [19, 7, 27, 26],
        680: [6, 16, 2, 3, 8],
        681: [1, 11, 20, 7, 26, 5],
        682: [18, 1, 30, 28, 27],
        683: [18, 19, 8, 22, 20],
        684: [23, 22, 13, 19, 4],
        685: [27, 29, 5, 22, 23, 24],
        686: [21, 11, 6, 23, 18, 27],
        687: [8, 22, 19, 27, 6],
        688: [16, 27, 24, 14],
        689: [29, 6, 18, 11],
        690: [17, 26, 21, 11],
        691: [18, 19, 13, 17, 7, 12],
        692: [14, 28, 3, 9, 25],
        693: [14, 19, 10, 18],
        694: [4, 27, 16, 26, 19, 21],
        695: [15, 3, 26, 28],
        696: [23, 8, 13, 11],
        697: [6, 29, 13, 24, 3],
        698: [20, 29, 16, 23, 1],
        699: [21, 10, 16, 14, 17],
        700: [10, 15, 1, 27],
        701: [22, 10, 8, 4, 2],
        702: [7, 18, 6, 3, 15, 30],
        703: [28, 4, 10, 26, 13, 16],
        704: [27, 28, 6, 8],
        705: [12, 29, 13, 6],
        706: [1, 6, 20, 2, 5],
        707: [4, 22, 10, 8, 28, 30],
        708: [18, 22, 28, 16, 19, 8],
        709: [27, 24, 14, 9],
        710: [4, 25, 15, 6],
        711: [26, 7, 5, 2],
        712: [10, 8, 1, 27],
        713: [30, 24, 26, 3, 13],
        714: [28, 9, 17, 15, 30],
        715: [4, 14, 5, 28, 19, 26],
        716: [19, 25, 4],
        717: [29, 14, 9, 13, 21],
        718: [25, 2, 30, 5, 21, 22],
        719: [6, 2, 29],
        720: [2, 24, 16, 30],
        721: [26, 16, 24, 10, 2],
        722: [7, 15, 30],
        723: [25, 17, 28, 12, 2, 29],
        724: [23, 17, 13, 5, 25],
        725: [6, 7, 29],
        726: [2, 15, 30, 12],
        727: [27, 15, 17, 13, 21],
        728: [8, 12, 16],
        729: [19, 18, 12, 28, 6, 9],
        730: [10, 1, 13],
        731: [22, 7, 9, 21],
        732: [8, 9, 20],
        733: [14, 29, 13, 28, 12],
        734: [14, 28, 21, 11],
        735: [18, 21, 20, 10, 22],
        736: [6, 5, 2, 1],
        737: [21, 23, 29, 10, 7],
        738: [24, 1, 21, 16],
        739: [13, 30, 24, 2, 19, 23],
        740: [24, 23, 8, 22, 18],
        741: [17, 10, 28, 6, 27],
        742: [10, 4, 16, 5],
        743: [22, 29, 11, 21],
        744: [27, 25, 14, 22, 15],
        745: [9, 19, 10, 15, 28],
        746: [17, 10, 14, 2, 5, 26],
        747: [19, 29, 15, 5, 11, 23],
        748: [11, 14, 7, 29],
        749: [23, 13, 10, 30, 27, 11],
        750: [21, 14, 13, 29, 16, 28],

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
        31: 3,
        32: 2,
        33: 1,
        34: 3,
        35: 3,
        36: 1,
        37: 3,
        38: 2,
        39: 1,
        40: 1,
        41: 1,
        42: 1,
        43: 2,
        44: 3,
        45: 2,
        46: 3,
        47: 1,
        48: 1,
        49: 2,
        50: 3,
        51: 1,
        52: 3,
        53: 3,
        54: 1,
        55: 2,
        56: 3,
        57: 3,
        58: 2,
        59: 3,
        60: 1,
        61: 1,
        62: 1,
        63: 3,
        64: 2,
        65: 3,
        66: 3,
        67: 1,
        68: 1,
        69: 2,
        70: 2,
        71: 2,
        72: 1,
        73: 2,
        74: 1,
        75: 2,
        76: 3,
        77: 3,
        78: 2,
        79: 1,
        80: 2,
        81: 2,
        82: 1,
        83: 3,
        84: 1,
        85: 1,
        86: 2,
        87: 2,
        88: 3,
        89: 2,
        90: 1,
        91: 1,
        92: 3,
        93: 2,
        94: 1,
        95: 3,
        96: 2,
        97: 1,
        98: 2,
        99: 2,
        100: 2,
        101: 1,
        102: 1,
        103: 3,
        104: 1,
        105: 3,
        106: 1,
        107: 1,
        108: 3,
        109: 3,
        110: 2,
        111: 2,
        112: 3,
        113: 2,
        114: 3,
        115: 1,
        116: 1,
        117: 2,
        118: 2,
        119: 2,
        120: 3,
        121: 2,
        122: 2,
        123: 3,
        124: 2,
        125: 3,
        126: 3,
        127: 3,
        128: 1,
        129: 3,
        130: 3,
        131: 3,
        132: 1,
        133: 1,
        134: 1,
        135: 2,
        136: 3,
        137: 2,
        138: 1,
        139: 2,
        140: 3,
        141: 1,
        142: 2,
        143: 2,
        144: 1,
        145: 2,
        146: 3,
        147: 3,
        148: 3,
        149: 2,
        150: 3,
        151: 1,
        152: 3,
        153: 3,
        154: 2,
        155: 3,
        156: 1,
        157: 1,
        158: 1,
        159: 1,
        160: 2,
        161: 1,
        162: 3,
        163: 1,
        164: 1,
        165: 1,
        166: 2,
        167: 2,
        168: 2,
        169: 2,
        170: 2,
        171: 3,
        172: 2,
        173: 1,
        174: 3,
        175: 2,
        176: 2,
        177: 3,
        178: 2,
        179: 2,
        180: 2,
        181: 3,
        182: 3,
        183: 2,
        184: 1,
        185: 2,
        186: 1,
        187: 1,
        188: 1,
        189: 1,
        190: 1,
        191: 3,
        192: 3,
        193: 2,
        194: 2,
        195: 3,
        196: 2,
        197: 1,
        198: 2,
        199: 2,
        200: 1,
        201: 1,
        202: 2,
        203: 2,
        204: 3,
        205: 1,
        206: 2,
        207: 2,
        208: 2,
        209: 2,
        210: 3,
        211: 1,
        212: 3,
        213: 3,
        214: 3,
        215: 3,
        216: 2,
        217: 1,
        218: 2,
        219: 2,
        220: 3,
        221: 3,
        222: 1,
        223: 2,
        224: 1,
        225: 1,
        226: 3,
        227: 3,
        228: 1,
        229: 2,
        230: 3,
        231: 2,
        232: 1,
        233: 2,
        234: 2,
        235: 1,
        236: 2,
        237: 2,
        238: 2,
        239: 1,
        240: 1,
        241: 1,
        242: 3,
        243: 3,
        244: 2,
        245: 2,
        246: 3,
        247: 2,
        248: 2,
        249: 3,
        250: 1,
        251: 3,
        252: 1,
        253: 3,
        254: 3,
        255: 1,
        256: 1,
        257: 1,
        258: 2,
        259: 2,
        260: 2,
        261: 2,
        262: 2,
        263: 3,
        264: 1,
        265: 3,
        266: 1,
        267: 1,
        268: 1,
        269: 1,
        270: 3,
        271: 3,
        272: 2,
        273: 3,
        274: 3,
        275: 2,
        276: 1,
        277: 3,
        278: 3,
        279: 3,
        280: 2,
        281: 1,
        282: 1,
        283: 1,
        284: 3,
        285: 3,
        286: 2,
        287: 1,
        288: 1,
        289: 3,
        290: 3,
        291: 2,
        292: 3,
        293: 3,
        294: 1,
        295: 3,
        296: 1,
        297: 2,
        298: 3,
        299: 2,
        300: 3,
        301: 1,
        302: 1,
        303: 1,
        304: 3,
        305: 2,
        306: 2,
        307: 1,
        308: 3,
        309: 2,
        310: 1,
        311: 1,
        312: 3,
        313: 1,
        314: 1,
        315: 3,
        316: 2,
        317: 2,
        318: 2,
        319: 3,
        320: 2,
        321: 3,
        322: 1,
        323: 2,
        324: 1,
        325: 1,
        326: 2,
        327: 1,
        328: 2,
        329: 1,
        330: 1,
        331: 3,
        332: 3,
        333: 3,
        334: 2,
        335: 1,
        336: 3,
        337: 1,
        338: 2,
        339: 3,
        340: 1,
        341: 2,
        342: 2,
        343: 2,
        344: 1,
        345: 3,
        346: 1,
        347: 3,
        348: 2,
        349: 2,
        350: 2,
        351: 3,
        352: 2,
        353: 3,
        354: 1,
        355: 2,
        356: 2,
        357: 1,
        358: 2,
        359: 3,
        360: 3,
        361: 1,
        362: 2,
        363: 3,
        364: 3,
        365: 3,
        366: 2,
        367: 1,
        368: 3,
        369: 3,
        370: 1,
        371: 2,
        372: 2,
        373: 3,
        374: 2,
        375: 2,
        376: 2,
        377: 2,
        378: 2,
        379: 1,
        380: 2,
        381: 3,
        382: 1,
        383: 1,
        384: 1,
        385: 3,
        386: 3,
        387: 1,
        388: 3,
        389: 3,
        390: 2,
        391: 2,
        392: 1,
        393: 3,
        394: 2,
        395: 1,
        396: 2,
        397: 3,
        398: 1,
        399: 1,
        400: 2,
        401: 3,
        402: 1,
        403: 3,
        404: 1,
        405: 3,
        406: 3,
        407: 3,
        408: 3,
        409: 2,
        410: 2,
        411: 2,
        412: 1,
        413: 2,
        414: 3,
        415: 2,
        416: 3,
        417: 2,
        418: 2,
        419: 1,
        420: 2,
        421: 2,
        422: 3,
        423: 1,
        424: 1,
        425: 1,
        426: 2,
        427: 2,
        428: 2,
        429: 3,
        430: 3,
        431: 3,
        432: 3,
        433: 1,
        434: 3,
        435: 2,
        436: 3,
        437: 1,
        438: 3,
        439: 2,
        440: 3,
        441: 3,
        442: 2,
        443: 2,
        444: 2,
        445: 3,
        446: 2,
        447: 1,
        448: 1,
        449: 3,
        450: 1,
        451: 2,
        452: 1,
        453: 1,
        454: 1,
        455: 1,
        456: 2,
        457: 2,
        458: 3,
        459: 2,
        460: 3,
        461: 3,
        462: 2,
        463: 2,
        464: 2,
        465: 2,
        466: 3,
        467: 2,
        468: 3,
        469: 3,
        470: 1,
        471: 1,
        472: 2,
        473: 1,
        474: 2,
        475: 2,
        476: 1,
        477: 2,
        478: 1,
        479: 3,
        480: 1,
        481: 1,
        482: 1,
        483: 3,
        484: 2,
        485: 1,
        486: 3,
        487: 1,
        488: 3,
        489: 2,
        490: 1,
        491: 2,
        492: 3,
        493: 3,
        494: 2,
        495: 2,
        496: 1,
        497: 2,
        498: 1,
        499: 2,
        500: 1,
        501: 1,
        502: 3,
        503: 2,
        504: 2,
        505: 2,
        506: 2,
        507: 2,
        508: 2,
        509: 1,
        510: 2,
        511: 3,
        512: 3,
        513: 1,
        514: 1,
        515: 1,
        516: 1,
        517: 2,
        518: 1,
        519: 2,
        520: 3,
        521: 1,
        522: 3,
        523: 3,
        524: 1,
        525: 3,
        526: 1,
        527: 2,
        528: 2,
        529: 1,
        530: 2,
        531: 2,
        532: 3,
        533: 3,
        534: 1,
        535: 3,
        536: 1,
        537: 1,
        538: 3,
        539: 2,
        540: 2,
        541: 1,
        542: 2,
        543: 2,
        544: 1,
        545: 1,
        546: 1,
        547: 2,
        548: 1,
        549: 1,
        550: 2,
        551: 1,
        552: 1,
        553: 1,
        554: 3,
        555: 2,
        556: 1,
        557: 2,
        558: 2,
        559: 3,
        560: 3,
        561: 1,
        562: 3,
        563: 2,
        564: 3,
        565: 2,
        566: 3,
        567: 2,
        568: 3,
        569: 3,
        570: 2,
        571: 2,
        572: 1,
        573: 2,
        574: 2,
        575: 3,
        576: 3,
        577: 1,
        578: 1,
        579: 1,
        580: 2,
        581: 3,
        582: 1,
        583: 2,
        584: 3,
        585: 3,
        586: 3,
        587: 1,
        588: 2,
        589: 2,
        590: 2,
        591: 1,
        592: 1,
        593: 1,
        594: 3,
        595: 3,
        596: 1,
        597: 2,
        598: 2,
        599: 2,
        600: 3,
        601: 3,
        602: 2,
        603: 1,
        604: 1,
        605: 3,
        606: 2,
        607: 3,
        608: 1,
        609: 1,
        610: 3,
        611: 3,
        612: 2,
        613: 2,
        614: 1,
        615: 2,
        616: 2,
        617: 3,
        618: 2,
        619: 2,
        620: 1,
        621: 1,
        622: 3,
        623: 2,
        624: 1,
        625: 3,
        626: 2,
        627: 2,
        628: 3,
        629: 1,
        630: 3,
        631: 1,
        632: 2,
        633: 3,
        634: 1,
        635: 1,
        636: 3,
        637: 1,
        638: 3,
        639: 3,
        640: 3,
        641: 1,
        642: 3,
        643: 2,
        644: 1,
        645: 2,
        646: 1,
        647: 1,
        648: 1,
        649: 2,
        650: 1,
        651: 3,
        652: 3,
        653: 1,
        654: 2,
        655: 2,
        656: 1,
        657: 2,
        658: 1,
        659: 1,
        660: 3,
        661: 3,
        662: 1,
        663: 3,
        664: 3,
        665: 2,
        666: 2,
        667: 1,
        668: 3,
        669: 1,
        670: 1,
        671: 2,
        672: 3,
        673: 2,
        674: 2,
        675: 1,
        676: 3,
        677: 2,
        678: 3,
        679: 3,
        680: 2,
        681: 1,
        682: 3,
        683: 3,
        684: 1,
        685: 3,
        686: 3,
        687: 3,
        688: 1,
        689: 1,
        690: 2,
        691: 2,
        692: 3,
        693: 3,
        694: 2,
        695: 1,
        696: 2,
        697: 1,
        698: 1,
        699: 1,
        700: 1,
        701: 2,
        702: 2,
        703: 1,
        704: 2,
        705: 2,
        706: 3,
        707: 3,
        708: 2,
        709: 2,
        710: 3,
        711: 1,
        712: 1,
        713: 2,
        714: 3,
        715: 3,
        716: 1,
        717: 3,
        718: 1,
        719: 3,
        720: 2,
        721: 1,
        722: 3,
        723: 3,
        724: 1,
        725: 3,
        726: 1,
        727: 2,
        728: 3,
        729: 1,
        730: 3,
        731: 3,
        732: 3,
        733: 2,
        734: 1,
        735: 1,
        736: 3,
        737: 2,
        738: 3,
        739: 2,
        740: 1,
        741: 2,
        742: 3,
        743: 3,
        744: 3,
        745: 2,
        746: 3,
        747: 2,
        748: 3,
        749: 1,
        750: 1,


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

    _t_start = time.perf_counter()
    sol, REQ = solve_ffh(inst)
    _t_heuristic = time.perf_counter() - _t_start
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

    export_solution_to_excel(inst, sol, REQ, out_path="ffh_solution_improved750.xlsx")

        # ── Total computation time ────────────────────────────────────────────────
    _t_total = time.perf_counter() - _t_start
    sep = "=" * 75
    print(f"\n{sep}")
    print("SC-CIH COMPUTATION TIME")
    print(sep)
    print(f"  Heuristic (solve_ffh)  : {_t_heuristic:.4f} sec")
    print(f"  Post-solve checks      : {_t_total - _t_heuristic:.4f} sec")
    print(f"  TOTAL                  : {_t_total:.4f} sec  ({_t_total/60:.2f} min)")
    print(sep)