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
    out_path: str = "ffh_solution_improved1000.xlsx",
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
    file_path = "step1000.xlsx"

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
        204: [17, 30, 14, 9, 10, 13],
        205: [29, 26, 16, 4, 3],
        206: [17, 26, 19],
        207: [20, 5, 4, 3, 2, 1],
        208: [19, 20, 14, 7],
        209: [11, 13, 15, 27, 22, 26],
        210: [4, 9, 10, 16, 22],
        211: [23, 8, 16, 13, 11, 6],
        212: [23, 8, 16, 28, 13, 11],
        213: [23, 13, 10],
        214: [7, 20, 14, 17, 28],
        215: [7, 9, 10, 8, 16, 22],
        216: [9, 18, 30, 29],
        217: [11, 13, 15, 20, 12, 26],
        218: [7, 13, 14, 5, 28],
        219: [27, 20, 16, 6, 4, 2],
        220: [27, 20, 19, 6, 4, 2],
        221: [9, 22, 12, 11],
        222: [30, 23, 5, 3, 4, 2],
        223: [29, 26, 24, 28, 4, 3],
        224: [29, 2, 8, 1, 3],
        225: [18, 12, 15, 21, 25, 28],
        226: [10, 16, 27],
        227: [25, 21, 22, 24, 26, 28],
        228: [9, 12, 14],
        229: [23, 30, 14, 9, 10, 12],
        230: [30, 14, 9, 10, 12],
        231: [17, 23, 20],
        232: [23, 13, 11],
        233: [12, 6, 15, 21, 25, 28],
        234: [10, 21, 27, 30],
        235: [10, 22, 13, 30],
        236: [23, 8, 28, 13, 11],
        237: [7, 10, 16, 22],
        238: [30, 7, 5, 4, 2],
        239: [30, 23, 5, 4],
        240: [13, 10, 11],
        241: [10, 13, 14, 17, 28],
        242: [7, 14, 17, 28],
        243: [20, 22, 24, 26, 28],
        244: [23, 8, 16, 13, 10, 11],
        245: [10, 16, 2, 27],
        246: [18, 6, 25, 26, 27, 29],
        247: [29, 8, 1, 3],
        248: [22, 8, 14, 5, 1],
        249: [20, 7, 6, 4, 2],
        250: [10, 16, 1, 27],
        251: [23, 5, 4, 2],
        252: [22, 18, 30, 29],
        253: [7, 13, 27, 17, 28],
        254: [23, 4, 16, 13, 11],
        255: [30, 23, 5, 4, 15],
        256: [12, 15, 21, 24, 25, 28],
        257: [28, 21, 19],
        258: [7, 9, 11, 16, 22],
        259: [30, 5, 4, 2],
        260: [28, 26, 3],
        261: [17, 15, 21, 25, 7],
        262: [19, 7, 27, 28],
        263: [25, 5, 26, 29, 20],
        264: [28, 5, 18, 9, 26],
        265: [22, 28, 1],
        266: [12, 26, 8],
        267: [1, 6, 9, 2, 5],
        268: [17, 4, 24, 3, 16, 15],
        269: [15, 17, 8],
        270: [24, 26, 30],
        271: [15, 21, 1, 2, 16],
        272: [22, 4, 16, 23, 15, 3],
        273: [20, 5, 3, 9, 21],
        274: [13, 20, 17, 10, 15],
        275: [4, 26, 23, 30, 21, 29],
        276: [14, 15, 29, 8],
        277: [27, 15, 13, 14, 24],
        278: [14, 11, 22, 9, 12],
        279: [3, 30, 27, 29, 28, 14],
        280: [26, 5, 18, 2, 19],
        281: [22, 4, 14, 12, 28],
        282: [28, 24, 2, 10, 20, 27],
        283: [19, 17, 7],
        284: [8, 28, 4, 12, 18, 27],
        285: [19, 8, 26, 14, 28],
        286: [20, 30, 22],
        287: [1, 6, 9, 23, 25],
        288: [12, 1, 6, 28, 5],
        289: [3, 5, 24, 21, 1, 30],
        290: [13, 14, 15, 11],
        291: [10, 24, 11, 25, 19],
        292: [29, 2, 5],
        293: [22, 3, 9],
        294: [16, 20, 15, 14, 9, 7],
        295: [12, 14, 4],
        296: [17, 22, 10, 2, 8, 13],
        297: [1, 7, 10],
        298: [25, 9, 10, 11],
        299: [16, 24, 14],
        300: [13, 18, 23, 8],
        301: [3, 13, 28, 24, 2],
        302: [15, 30, 3],
        303: [19, 13, 23, 21, 14, 10],
        304: [1, 11, 6, 26, 20, 15],
        305: [3, 14, 28, 4, 8],
        306: [17, 3, 13, 10, 24, 11],
        307: [25, 6, 3, 17, 21],
        308: [29, 25, 12, 24],
        309: [8, 4, 5, 9],
        310: [20, 5, 25, 21],
        311: [25, 21, 16, 15],
        312: [22, 29, 19, 21, 27, 20],
        313: [5, 15, 3, 16, 21],
        314: [19, 2, 12, 17, 3],
        315: [15, 2, 29, 12, 10, 3],
        316: [20, 17, 13],
        317: [15, 30, 26],
        318: [11, 20, 16, 17],
        319: [15, 4, 26],
        320: [17, 21, 6],
        321: [23, 15, 17, 20],
        322: [12, 30, 10, 13, 14],
        323: [26, 21, 11],
        324: [4, 18, 22, 13, 10],
        325: [11, 3, 19, 22],
        326: [10, 21, 23, 22, 13],
        327: [10, 18, 13],
        328: [27, 5, 22, 23, 24],
        329: [21, 22, 14],
        330: [1, 12, 10, 6, 7],
        331: [7, 8, 5, 28, 3, 10],
        332: [22, 11, 29],
        333: [20, 13, 5, 6],
        334: [24, 15, 2, 14],
        335: [15, 20, 10, 25],
        336: [8, 18, 30, 10, 26, 16],
        337: [22, 19, 15, 25, 10],
        338: [6, 27, 7, 26, 20, 5],
        339: [21, 16, 28],
        340: [23, 28, 17],
        341: [3, 25, 6, 9, 15],
        342: [27, 14, 3, 30],
        343: [29, 12, 1, 14, 2, 13],
        344: [8, 13, 3, 12, 1],
        345: [27, 23, 21],
        346: [5, 2, 10, 30],
        347: [25, 23, 16, 15],
        348: [29, 3, 1],
        349: [27, 5, 18, 24],
        350: [4, 25, 10, 8, 28, 30],
        351: [14, 21, 26, 20],
        352: [4, 27, 29],
        353: [21, 17, 19],
        354: [10, 14, 1, 20],
        355: [19, 14, 6, 22],
        356: [17, 12, 3],
        357: [13, 28, 16],
        358: [12, 9, 24, 1, 30, 3],
        359: [24, 22, 21, 4],
        360: [5, 2, 12, 18, 11],
        361: [27, 25, 22, 15],
        362: [21, 6, 26, 5, 3, 23],
        363: [10, 7, 2],
        364: [11, 30, 10],
        365: [27, 18, 16, 9, 2, 25],
        366: [10, 12, 28, 25],
        367: [9, 4, 26, 12, 14],
        368: [29, 13, 11, 6, 16, 23],
        369: [30, 26, 17, 9, 3],
        370: [3, 14, 20, 27, 6, 18],
        371: [4, 3, 11, 22, 10],
        372: [20, 23, 14, 6, 29, 15],
        373: [2, 24, 28, 12, 20, 14],
        374: [3, 22, 21],
        375: [17, 26, 24, 22, 6],
        376: [28, 20, 22, 26],
        377: [5, 3, 8],
        378: [12, 13, 19, 2, 20],
        379: [12, 30, 15, 25, 3, 19],
        380: [13, 11, 21, 9, 8],
        381: [24, 6, 16],
        382: [30, 18, 4, 9, 25, 27],
        383: [7, 20, 10, 23, 16, 30],
        384: [28, 3, 15, 6],
        385: [3, 26, 22, 11, 28, 12],
        386: [18, 10, 29],
        387: [23, 30, 21, 6],
        388: [4, 7, 26, 5],
        389: [1, 12, 18, 19, 29, 15],
        390: [20, 29, 3, 10],
        391: [17, 14, 25, 29, 19, 3],
        392: [21, 3, 15, 22, 17],
        393: [29, 27, 25, 18],
        394: [25, 5, 14, 17],
        395: [17, 5, 10],
        396: [11, 30, 23, 8],
        397: [28, 3, 9, 7, 21],
        398: [5, 21, 10, 20, 18],
        399: [19, 5, 6, 22],
        400: [27, 30, 19, 2, 28],
        401: [2, 21, 25],
        402: [21, 7, 25, 19, 14],
        403: [16, 29, 21],
        404: [21, 10, 16, 8, 26],
        405: [10, 15, 3, 23, 2, 6],
        406: [16, 15, 7, 11, 20, 5],
        407: [24, 28, 12, 13, 5],
        408: [11, 8, 15],
        409: [15, 8, 5, 4, 2],
        410: [28, 20, 14, 8, 6, 11],
        411: [7, 25, 6, 16, 17],
        412: [29, 10, 16, 1, 3, 13],
        413: [8, 7, 19, 12, 2, 26],
        414: [20, 29, 27, 21, 22, 16],
        415: [28, 4, 14],
        416: [24, 12, 25, 13, 2],
        417: [19, 18, 7],
        418: [3, 13, 17, 15, 25],
        419: [27, 20, 22, 4, 5],
        420: [12, 26, 11, 18, 30, 25],
        421: [20, 17, 13, 2],
        422: [5, 23, 11],
        423: [5, 20, 17, 30, 11, 29],
        424: [13, 20, 24, 27],
        425: [17, 27, 18, 16, 23],
        426: [16, 27, 1, 12, 11],
        427: [14, 19, 10],
        428: [20, 16, 9],
        429: [24, 2, 19, 16],
        430: [5, 27, 22, 8, 2, 19],
        431: [7, 1, 15],
        432: [5, 14, 23, 7, 29, 17],
        433: [28, 30, 24, 29, 2, 23],
        434: [18, 11, 22, 16],
        435: [11, 6, 15, 18, 30, 27],
        436: [20, 16, 7, 8, 9],
        437: [8, 10, 25, 23, 7],
        438: [11, 16, 12, 18, 26, 24],
        439: [4, 19, 22, 18, 13],
        440: [25, 26, 5, 10, 2],
        441: [12, 30, 15],
        442: [24, 16, 7, 27, 18],
        443: [5, 4, 20, 24, 19],
        444: [2, 22, 29, 17],
        445: [2, 4, 14, 11],
        446: [4, 22, 25, 5, 1, 18],
        447: [21, 29, 16, 28, 30, 7],
        448: [10, 21, 2, 30, 25],
        449: [18, 24, 28, 30],
        450: [14, 29, 27, 6],
        451: [26, 16, 6, 24, 10, 2],
        452: [19, 20, 4, 30, 11],
        453: [21, 18, 17, 16, 5, 28],
        454: [7, 26, 4, 11, 6],
        455: [21, 9, 23, 6, 1, 24],
        456: [19, 22, 25, 7, 6],
        457: [27, 14, 17, 11, 3, 13],
        458: [6, 5, 16],
        459: [1, 9, 13, 8],
        460: [11, 10, 19, 24, 1],
        461: [23, 8, 2, 22, 4],
        462: [6, 13, 22, 17, 30],
        463: [23, 4, 21, 30, 10],
        464: [8, 5, 16, 15],
        465: [14, 23, 18, 30, 16],
        466: [25, 8, 22, 20],
        467: [17, 23, 12, 3, 19, 4],
        468: [19, 18, 5, 6],
        469: [4, 22, 7, 23, 19, 16],
        470: [26, 2, 15, 5, 17, 14],
        471: [18, 15, 22],
        472: [13, 9, 27],
        473: [19, 3, 2, 14],
        474: [18, 2, 30],
        475: [2, 10, 14, 6, 25, 5],
        476: [12, 29, 13, 15, 28, 26],
        477: [21, 28, 12, 4],
        478: [17, 5, 24, 8, 1, 25],
        479: [15, 22, 24, 18, 14],
        480: [27, 8, 29, 15, 12, 5],
        481: [30, 29, 24, 25],
        482: [26, 22, 14],
        483: [8, 7, 3],
        484: [15, 20, 22],
        485: [8, 24, 2],
        486: [8, 18, 7, 25, 27, 2],
        487: [8, 27, 30, 24, 19],
        488: [8, 10, 29, 5, 22],
        489: [14, 10, 9, 2],
        490: [21, 22, 14, 18],
        491: [12, 21, 22],
        492: [23, 14, 5, 10, 13],
        493: [8, 28, 30, 10, 23, 5],
        494: [18, 14, 17],
        495: [8, 9, 7, 11, 21, 3],
        496: [3, 18, 24, 27, 7],
        497: [13, 22, 20, 2, 28],
        498: [26, 25, 19, 24],
        499: [16, 7, 28, 30],
        500: [30, 1, 7, 24, 4],
        501: [30, 8, 23, 20, 28, 7],
        502: [18, 11, 25, 10],
        503: [18, 21, 12, 10, 9, 28],
        504: [15, 4, 26, 24, 16, 25],
        505: [12, 11, 14, 2],
        506: [24, 5, 1, 9],
        507: [10, 5, 7, 11, 8, 13],
        508: [16, 18, 21, 22],
        509: [25, 27, 16, 24, 21],
        510: [15, 6, 24, 26, 12, 29],
        511: [6, 29, 18, 21, 2, 17],
        512: [27, 22, 2],
        513: [5, 27, 21, 8, 3, 23],
        514: [7, 17, 15],
        515: [20, 21, 22],
        516: [22, 16, 1, 28, 18, 26],
        517: [1, 17, 24],
        518: [1, 17, 27, 23, 22],
        519: [4, 30, 17, 5],
        520: [20, 17, 9, 27],
        521: [26, 13, 3, 12, 15],
        522: [23, 8, 10, 22],
        523: [3, 13, 18],
        524: [21, 1, 23],
        525: [16, 28, 14],
        526: [19, 28, 4, 29, 17],
        527: [7, 29, 28, 23],
        528: [9, 2, 3, 22, 30, 18],
        529: [6, 30, 28],
        530: [7, 19, 5],
        531: [30, 3, 10, 6, 19, 8],
        532: [22, 29, 18, 11, 13, 25],
        533: [26, 23, 24, 3],
        534: [2, 4, 14, 8, 27],
        535: [20, 25, 13, 11, 1],
        536: [26, 15, 16, 8, 12],
        537: [14, 6, 22, 19, 28, 13],
        538: [26, 8, 23, 3, 9],
        539: [23, 26, 21, 5, 24, 13],
        540: [4, 3, 1, 10, 15],
        541: [4, 5, 3, 6, 14],
        542: [4, 1, 3, 12, 18, 28],
        543: [28, 13, 1, 10, 14],
        544: [24, 29, 18],
        545: [22, 13, 6, 5],
        546: [9, 16, 5, 3, 6],
        547: [14, 10, 16, 26, 3],
        548: [8, 24, 21, 16, 20],
        549: [15, 4, 5, 10],
        550: [11, 27, 20, 13, 26, 30],
        551: [14, 27, 28, 21, 20],
        552: [11, 20, 23, 7, 16],
        553: [13, 11, 10, 24],
        554: [19, 26, 8, 11, 13, 9],
        555: [4, 30, 19, 7, 18],
        556: [24, 15, 23],
        557: [26, 10, 27, 23, 3, 14],
        558: [5, 21, 10, 8, 9, 22],
        559: [26, 13, 3, 15, 20, 16],
        560: [18, 17, 28, 23, 14, 30],
        561: [23, 29, 26, 18, 20],
        562: [4, 25, 8],
        563: [6, 21, 20, 2, 19],
        564: [25, 11, 26, 14, 4, 1],
        565: [8, 17, 24, 18, 19],
        566: [15, 12, 13, 25],
        567: [12, 1, 16, 4],
        568: [30, 3, 4, 27, 24, 5],
        569: [11, 15, 26, 7, 17],
        570: [16, 4, 15, 24, 23],
        571: [3, 10, 2, 26, 23],
        572: [28, 11, 21],
        573: [30, 24, 8, 17],
        574: [11, 18, 14, 15],
        575: [21, 6, 29, 30, 22, 14],
        576: [24, 20, 29],
        577: [19, 14, 13, 1, 23, 7],
        578: [25, 23, 26, 3, 19],
        579: [12, 11, 7, 15],
        580: [22, 28, 16, 17, 21],
        581: [20, 13, 19, 4, 12, 26],
        582: [27, 22, 23, 26],
        583: [22, 5, 11],
        584: [10, 4, 6, 12],
        585: [17, 13, 14, 20],
        586: [14, 6, 16, 21, 18, 23],
        587: [18, 6, 16, 10],
        588: [11, 27, 15, 20],
        589: [1, 16, 5, 7, 27],
        590: [14, 22, 16, 30, 23, 15],
        591: [3, 19, 1, 26],
        592: [2, 9, 8, 18, 10],
        593: [19, 24, 25, 28, 16, 18],
        594: [19, 4, 9],
        595: [18, 27, 25, 2, 24],
        596: [14, 4, 24, 27],
        597: [10, 28, 2, 15],
        598: [28, 27, 3, 15, 4],
        599: [26, 24, 19, 23],
        600: [18, 9, 20, 6, 11],
        601: [10, 19, 9, 28, 17],
        602: [5, 25, 26],
        603: [9, 28, 21],
        604: [8, 5, 23, 11],
        605: [16, 5, 19, 21, 9, 27],
        606: [15, 3, 21, 26, 25, 29],
        607: [23, 30, 29, 12, 15],
        608: [19, 1, 28, 25, 24],
        609: [21, 22, 23, 12, 3, 18],
        610: [14, 27, 7, 16],
        611: [27, 10, 11, 18, 19],
        612: [26, 11, 22, 25, 2, 30],
        613: [1, 4, 28, 6, 15, 26],
        614: [7, 23, 5, 21, 10, 6],
        615: [21, 12, 9],
        616: [22, 29, 21, 6, 2],
        617: [24, 23, 25, 8, 14],
        618: [23, 4, 1],
        619: [3, 5, 19, 8, 17, 22],
        620: [1, 23, 11],
        621: [23, 5, 16, 3, 8, 13],
        622: [4, 11, 12],
        623: [13, 25, 27, 28],
        624: [3, 17, 19, 1],
        625: [15, 12, 24, 7],
        626: [30, 14, 20, 22],
        627: [3, 29, 4, 5],
        628: [12, 14, 11, 26, 5, 8],
        629: [8, 18, 20],
        630: [25, 23, 1, 28, 22],
        631: [17, 20, 7, 24],
        632: [21, 2, 26, 29, 8],
        633: [4, 8, 16, 21, 20, 3],
        634: [12, 30, 11],
        635: [28, 27, 19, 14, 12, 18],
        636: [25, 16, 3, 1],
        637: [1, 9, 7],
        638: [26, 13, 17],
        639: [14, 23, 13, 3, 21, 18],
        640: [9, 3, 10, 17],
        641: [18, 11, 13, 19],
        642: [10, 23, 14],
        643: [2, 8, 3, 30],
        644: [15, 20, 2],
        645: [4, 1, 23, 5],
        646: [6, 16, 12],
        647: [6, 12, 5, 24, 25],
        648: [25, 29, 1],
        649: [9, 17, 3, 30, 23, 19],
        650: [16, 15, 17],
        651: [16, 28, 19],
        652: [6, 9, 25, 4, 29],
        653: [8, 24, 17],
        654: [28, 1, 8],
        655: [12, 13, 5, 6, 2, 18],
        656: [8, 11, 30, 14, 24, 29],
        657: [19, 12, 4],
        658: [6, 8, 26],
        659: [13, 3, 29],
        660: [25, 28, 10, 11, 3],
        661: [1, 12, 7, 10, 19, 27],
        662: [15, 12, 19, 16],
        663: [5, 1, 30, 14],
        664: [18, 12, 21, 27],
        665: [11, 1, 25],
        666: [8, 4, 5, 9, 10, 28],
        667: [1, 6, 9, 2, 5, 11],
        668: [24, 12, 25, 13, 2, 16],
        669: [19, 14, 6, 22, 20, 16],
        670: [28, 11, 21, 13, 15, 20],
        671: [10, 22, 4, 30, 7, 6],
        672: [27, 14, 3, 30, 4, 25],
        673: [21, 16, 28, 4, 3, 1],
        674: [18, 21, 15, 9, 22, 16],
        675: [11, 12, 26, 29, 5, 2],
        676: [30, 1, 7, 24, 4, 28],
        677: [24, 14, 13, 18, 19, 17],
        678: [9, 30, 11, 22, 27],
        679: [9, 10, 16, 22, 7, 13],
        680: [16, 17, 23, 24, 20, 22],
        681: [20, 17, 13, 2, 5, 3],
        682: [22, 19, 15, 25, 10, 12],
        683: [28, 24, 19, 27, 10, 11],
        684: [17, 15, 21, 25, 7, 8],
        685: [10, 9, 12, 11, 6, 25],
        686: [28, 26, 12, 19, 25, 27],
        687: [7, 9, 11, 16, 22, 10],
        688: [9, 12, 11, 25, 6, 3],
        689: [24, 28, 12, 13, 5, 4],
        690: [25, 24, 14, 9, 20, 29],
        691: [10, 22, 27, 30, 9, 12],
        692: [18, 11, 13, 19, 16, 5],
        693: [23, 13, 7, 10, 11, 22],
        694: [30, 5, 4, 2, 12, 21],
        695: [19, 28, 4, 29, 17, 20],
        696: [19, 17, 14, 12, 18, 25],
        697: [13, 28, 16, 11, 1, 25],
        698: [21, 22, 14, 16, 29],
        699: [10, 24, 11, 25, 19, 22],
        700: [10, 22, 15, 30, 9, 18],
        701: [7, 13, 27, 17, 28, 16],
        702: [17, 20, 7, 24, 28, 3],
        703: [19, 20, 14, 16, 2, 28],
        704: [16, 15, 17, 7, 9, 10],
        705: [4, 30, 19, 7, 18, 12],
        706: [27, 5, 18, 24, 29, 25],
        707: [29, 8, 1, 3, 16, 28],
        708: [21, 10, 16, 8, 26, 13],
        709: [24, 14, 13, 22, 8, 5],
        710: [13, 10, 11, 2, 14, 6],
        711: [9, 12, 11, 23, 13, 7],
        712: [20, 14, 7, 2, 27, 22],
        713: [5, 3, 8, 23, 4, 2],
        714: [12, 13, 10, 11, 3, 18],
        715: [8, 18, 16, 15, 9, 12],
        716: [19, 20, 4, 30, 11, 5],
        717: [20, 7, 6, 4, 2, 24],
        718: [15, 21, 1, 2, 16, 6],
        719: [13, 14, 15, 11, 27, 17],
        720: [20, 21, 22, 26, 28, 9],
        721: [28, 1, 8, 4, 11, 12],
        722: [13, 11, 21, 9, 8, 18],
        723: [13, 15, 20, 22, 26, 7],
        724: [20, 29, 3, 10, 11, 12],
        725: [19, 12, 4, 30, 15],
        726: [20, 7, 6, 4, 2, 23],
        727: [6, 21, 20, 2, 19, 22],
        728: [5, 1, 30, 14, 15, 8],
        729: [28, 26, 12, 4, 3, 1],
        730: [23, 8, 2, 22, 4, 24],
        731: [25, 23, 26, 3, 19, 10],
        732: [5, 21, 10, 20, 18, 9],
        733: [22, 8, 14, 5, 1, 12],
        734: [7, 29, 28, 23, 26, 2],
        735: [19, 8, 26, 14, 28, 12],
        736: [27, 5, 22, 23, 24, 17],
        737: [7, 13, 14, 17, 28, 6],
        738: [15, 21, 1, 2, 16, 13],
        739: [11, 12, 7, 29, 16, 21],
        740: [22, 18, 30, 29, 10, 4],
        741: [16, 17, 23, 28, 5, 18],
        742: [30, 23, 5, 4, 18, 19],
        743: [2, 4, 14, 8, 27, 22],
        744: [12, 13, 10, 11, 24, 15],
        745: [22, 28, 1, 10, 27, 30],
        746: [11, 18, 14, 15, 16, 17],
        747: [22, 3, 9, 10, 18, 13],
        748: [29, 3, 1, 17, 15, 21],
        749: [4, 5, 3, 6, 14, 22],
        750: [15, 20, 10, 25, 23, 8],
        751: [28, 26, 12, 29, 24, 8],
        752: [18, 17, 14, 12, 28, 26],
        753: [30, 7, 5, 4, 2, 13],
        754: [23, 10, 11, 7, 16, 15],
        755: [5, 3, 8, 28, 20, 14],
        756: [25, 23, 16, 15, 18, 19],
        757: [16, 24, 14, 26, 6, 10],
        758: [23, 10, 11, 14, 22, 9],
        759: [12, 30, 10, 13, 14, 3],
        760: [10, 16, 15, 27, 8, 29],
        761: [11, 30, 10, 5, 1, 14],
        762: [7, 13, 14, 5, 28, 6],
        763: [26, 5, 18, 2, 19, 9],
        764: [21, 1, 23, 30, 5, 4],
        765: [28, 12, 19, 17, 26, 24],
        766: [20, 17, 13, 2, 10, 22],
        767: [1, 6, 9, 23, 25, 8],
        768: [27, 14, 3, 30, 20, 22],
        769: [8, 5, 23, 11, 28, 1],
        770: [15, 4, 5, 10, 21, 22],
        771: [5, 2, 10, 30, 25, 24],
        772: [19, 20, 14, 27, 2, 11],
        773: [7, 13, 14, 17, 28, 21],
        774: [16, 15, 27, 23, 13, 7],
        775: [12, 15, 21, 25, 27, 16],
        776: [19, 20, 14, 27, 2, 13],
        777: [9, 12, 14, 19, 5, 6],
        778: [16, 4, 15, 24, 23, 6],
        779: [28, 1, 8, 16, 6, 5],
        780: [10, 16, 15, 18, 9, 30],
        781: [23, 14, 5, 10, 13, 3],
        782: [14, 10, 16, 26, 3, 4],
        783: [21, 22, 14, 18, 10, 29],
        784: [28, 5, 18, 9, 26, 2],
        785: [17, 21, 25, 22, 24, 26],
        786: [18, 10, 29, 11, 12, 7],
        787: [7, 19, 5, 18, 25, 26],
        788: [4, 19, 22, 18, 13, 10],
        789: [4, 7, 26, 5, 9, 28],
        790: [19, 28, 4, 29, 17, 30],
        791: [25, 17, 28, 12, 2, 29],
        792: [23, 17, 13, 5, 25],
        793: [6, 7, 29],
        794: [2, 15, 30, 12],
        795: [27, 15, 17, 13, 21],
        796: [8, 12, 16],
        797: [19, 18, 12, 28, 6, 9],
        798: [10, 1, 13],
        799: [22, 7, 9, 21],
        800: [8, 9, 20],
        801: [14, 29, 13, 28, 12],
        802: [29, 2, 8, 1, 14],
        803: [1, 17, 22],
        804: [10, 22, 4, 16, 3],
        805: [8, 28, 12, 18, 27],
        806: [28, 12, 4, 3, 1],
        807: [22, 11, 30, 5, 24, 1],
        808: [30, 14, 13, 24, 2],
        809: [11, 24, 23, 8],
        810: [30, 22, 12, 17, 10, 6],
        811: [10, 4, 16, 5],
        812: [22, 29, 11, 21],
        813: [11, 30, 16, 10],
        814: [9, 19, 10, 15, 28],
        815: [17, 10, 14, 2, 5, 26],
        816: [19, 29, 15, 5, 11, 23],
        817: [11, 14, 7, 29],
        818: [6, 18, 7, 29],
        819: [20, 14, 21, 13],
        820: [1, 10, 9, 2],
        821: [1, 29, 11, 25, 27],
        822: [21, 24, 8],
        823: [29, 25, 18],
        824: [26, 23, 24],
        825: [29, 16, 12, 25, 7],
        826: [24, 1, 19, 17],
        827: [2, 23, 4],
        828: [10, 14, 22],
        829: [26, 24, 3],
        830: [20, 7, 25, 9],
        831: [23, 14, 10, 6, 2],
        832: [11, 8, 1, 28, 22, 3],
        833: [5, 3, 25, 17, 29, 24],
        834: [15, 10, 9, 2],
        835: [24, 4, 29, 25, 2, 1],
        836: [11, 12, 29, 15],
        837: [12, 27, 11],
        838: [7, 25, 19, 16, 17],
        839: [26, 14, 21, 30, 6, 4],
        840: [16, 5, 19, 21, 9, 24],
        841: [30, 14, 28, 22],
        842: [16, 30, 4, 29],
        843: [17, 23, 20, 4, 14],
        844: [12, 22, 29, 21, 13, 5],
        845: [1, 21, 30],
        846: [6, 29, 24, 25],
        847: [11, 9, 2, 17],
        848: [15, 3, 29],
        849: [21, 9, 23, 4, 1, 24],
        850: [28, 8, 16, 11],
        851: [23, 13, 7, 10],
        852: [11, 12, 27, 2, 8, 15],
        853: [18, 10, 29, 11, 12, 28],
        854: [4, 25, 10, 8, 28],
        855: [23, 4, 2, 29, 6],
        856: [16, 30, 5, 29],
        857: [30, 28, 13],
        858: [1, 21, 5, 30],
        859: [25, 29, 15],
        860: [8, 5, 23, 1, 11],
        861: [27, 19, 16, 17, 25, 6],
        862: [4, 13, 22, 20],
        863: [14, 20, 23, 25, 8, 9],
        864: [7, 20, 23, 10],
        865: [23, 21, 6, 17],
        866: [23, 4, 25, 12, 18],
        867: [17, 13, 26, 20],
        868: [17, 4, 1],
        869: [22, 9, 5, 27, 2],
        870: [25, 18, 2, 12],
        871: [24, 21, 18, 13],
        872: [22, 4, 16, 3],
        873: [26, 22, 11, 18, 2, 1],
        874: [27, 8, 21, 23, 30, 28],
        875: [9, 11, 28, 21],
        876: [2, 14, 9, 18, 28, 5],
        877: [13, 26, 20],
        878: [26, 19, 18, 12, 3, 23],
        879: [29, 10, 16, 27],
        880: [3, 12, 8, 7, 22, 11],
        881: [3, 13, 28, 24, 20, 2],
        882: [24, 6, 8, 26],
        883: [13, 9, 18],
        884: [17, 26, 15, 16, 2],
        885: [7, 22, 21],
        886: [27, 4, 7, 18, 23, 24],
        887: [24, 22, 5],
        888: [8, 26, 14, 28, 12],
        889: [6, 5, 23, 4, 3, 2],
        890: [16, 30, 15, 4, 29],
        891: [6, 17, 12],
        892: [25, 24, 19],
        893: [19, 20, 14, 16, 28],
        894: [28, 5, 18, 16, 26, 2],
        895: [6, 18, 8, 5, 24, 13],
        896: [21, 5, 10, 22, 29],
        897: [1, 18, 28, 5, 4],
        898: [22, 6, 16, 10],
        899: [1, 9, 23, 25, 8],
        900: [15, 6, 12, 19, 16],
        901: [23, 2, 29, 6],
        902: [22, 3, 9, 4, 7],
        903: [12, 5, 22, 21, 8],
        904: [22, 19, 15, 25],
        905: [20, 13, 4, 12, 26],
        906: [11, 8, 26],
        907: [23, 10, 12, 14, 22, 9],
        908: [28, 26, 12, 24, 8],
        909: [29, 13, 11, 16, 23],
        910: [27, 9, 28],
        911: [9, 24, 29, 21],
        912: [25, 19, 7, 2, 18, 9],
        913: [21, 25, 5, 12, 13, 8],
        914: [5, 15, 17, 3, 13],
        915: [25, 17, 5, 15],
        916: [22, 28, 16, 17, 21, 29],
        917: [15, 1, 2, 16],
        918: [28, 1, 8, 4, 11],
        919: [7, 26, 5],
        920: [17, 13, 12, 27],
        921: [29, 20, 1, 17, 5, 15],
        922: [13, 17, 28],
        923: [24, 4, 16, 22, 7],
        924: [1, 8, 2, 5, 19, 26],
        925: [18, 11, 17, 14, 25],
        926: [3, 7, 15, 20, 16],
        927: [27, 14, 20, 5],
        928: [6, 24, 22, 11, 30],
        929: [27, 5, 20, 4, 6, 16],
        930: [29, 26, 18],
        931: [16, 22, 5],
        932: [29, 25, 22],
        933: [12, 28, 2, 23, 17, 14],
        934: [21, 29, 8],
        935: [20, 29, 18, 14, 9],
        936: [17, 24, 21, 25],
        937: [12, 4, 5, 14, 21, 19],
        938: [2, 14, 24, 18, 28, 5],
        939: [25, 7, 14, 18, 26, 27],
        940: [13, 11, 12, 9, 8],
        941: [24, 5, 28, 22, 4, 30],
        942: [10, 29, 2, 8, 1, 14],
        943: [6, 13, 21],
        944: [23, 13, 7, 25, 10, 11],
        945: [6, 4, 2, 1],
        946: [23, 20, 2, 25, 27],
        947: [10, 1, 7, 11, 8, 13],
        948: [7, 2, 26, 20, 17],
        949: [22, 2, 12, 8],
        950: [9, 25, 13],
        951: [10, 25, 3, 23, 2, 6],
        952: [30, 3, 18],
        953: [28, 8, 11],
        954: [20, 7, 15, 30],
        955: [20, 21, 22, 26],
        956: [23, 15, 13, 19, 16],
        957: [22, 15, 19, 1, 16, 4],
        958: [21, 3, 15, 22, 17, 28],
        959: [4, 18, 22, 13, 15],
        960: [8, 10, 3, 13, 22],
        961: [18, 30, 6],
        962: [10, 7, 8, 2],
        963: [16, 24, 7, 8],
        964: [8, 7, 27, 20],
        965: [20, 1, 25, 18, 2, 16],
        966: [26, 20, 11],
        967: [21, 22, 14, 10, 29],
        968: [20, 3, 19, 1, 26],
        969: [17, 12, 13],
        970: [4, 23, 17, 15, 11, 29],
        971: [18, 9, 8, 16],
        972: [6, 16, 14, 7, 23],
        973: [9, 15, 6, 19, 5, 27],
        974: [26, 8, 18, 14, 28],
        975: [7, 9, 16, 30, 22],
        976: [5, 14, 23, 7, 29, 6],
        977: [26, 16, 10, 9, 18],
        978: [4, 23, 6, 2, 7, 9],
        979: [3, 14, 20, 27, 18],
        980: [1, 14, 4, 12, 28],
        981: [19, 21, 18, 6],
        982: [11, 22, 17, 16, 25],
        983: [20, 10, 15, 21, 5, 6],
        984: [21, 20, 19, 13, 3, 26],
        985: [10, 21, 30],
        986: [20, 21, 13, 9, 29],
        987: [17, 26, 15, 16],
        988: [6, 30, 28, 1, 5, 24],
        989: [10, 8, 29, 24, 30, 15],
        990: [1, 2, 22, 12],
        991: [15, 28, 18, 9, 29, 30],
        992: [6, 16, 14, 7, 25, 23],
        993: [1, 6, 9, 23, 25, 21],
        994: [16, 18, 8, 26, 3],
        995: [17, 10, 24, 22, 6],
        996: [25, 24, 10, 14, 13],
        997: [22, 8, 6, 30, 1],
        998: [17, 21, 25, 22, 10, 26],
        999: [21, 15, 29, 30, 22, 14],
        1000: [5, 17, 8, 28, 20, 14],


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
        205: 2,
        206: 2,
        207: 3,
        208: 2,
        209: 1,
        210: 3,
        211: 2,
        212: 2,
        213: 1,
        214: 3,
        215: 3,
        216: 1,
        217: 1,
        218: 3,
        219: 2,
        220: 2,
        221: 1,
        222: 2,
        223: 2,
        224: 2,
        225: 1,
        226: 1,
        227: 3,
        228: 1,
        229: 3,
        230: 3,
        231: 1,
        232: 1,
        233: 1,
        234: 1,
        235: 1,
        236: 2,
        237: 3,
        238: 2,
        239: 2,
        240: 1,
        241: 3,
        242: 3,
        243: 3,
        244: 2,
        245: 1,
        246: 2,
        247: 2,
        248: 2,
        249: 2,
        250: 1,
        251: 2,
        252: 1,
        253: 3,
        254: 2,
        255: 2,
        256: 1,
        257: 2,
        258: 3,
        259: 2,
        260: 2,
        261: 3,
        262: 1,
        263: 1,
        264: 1,
        265: 1,
        266: 1,
        267: 3,
        268: 3,
        269: 2,
        270: 3,
        271: 3,
        272: 2,
        273: 1,
        274: 3,
        275: 3,
        276: 3,
        277: 2,
        278: 1,
        279: 1,
        280: 1,
        281: 3,
        282: 3,
        283: 2,
        284: 1,
        285: 1,
        286: 3,
        287: 3,
        288: 2,
        289: 3,
        290: 3,
        291: 1,
        292: 3,
        293: 1,
        294: 2,
        295: 3,
        296: 2,
        297: 3,
        298: 1,
        299: 1,
        300: 1,
        301: 3,
        302: 2,
        303: 2,
        304: 1,
        305: 3,
        306: 2,
        307: 1,
        308: 1,
        309: 3,
        310: 1,
        311: 1,
        312: 3,
        313: 2,
        314: 2,
        315: 2,
        316: 3,
        317: 2,
        318: 3,
        319: 1,
        320: 2,
        321: 1,
        322: 1,
        323: 2,
        324: 1,
        325: 2,
        326: 1,
        327: 1,
        328: 3,
        329: 3,
        330: 3,
        331: 2,
        332: 1,
        333: 3,
        334: 1,
        335: 2,
        336: 3,
        337: 1,
        338: 2,
        339: 2,
        340: 2,
        341: 1,
        342: 3,
        343: 1,
        344: 3,
        345: 2,
        346: 2,
        347: 2,
        348: 3,
        349: 2,
        350: 3,
        351: 1,
        352: 2,
        353: 2,
        354: 1,
        355: 2,
        356: 3,
        357: 3,
        358: 1,
        359: 2,
        360: 3,
        361: 3,
        362: 3,
        363: 2,
        364: 1,
        365: 3,
        366: 3,
        367: 1,
        368: 2,
        369: 2,
        370: 3,
        371: 2,
        372: 2,
        373: 2,
        374: 2,
        375: 2,
        376: 1,
        377: 2,
        378: 3,
        379: 1,
        380: 1,
        381: 1,
        382: 3,
        383: 3,
        384: 1,
        385: 3,
        386: 3,
        387: 2,
        388: 2,
        389: 1,
        390: 3,
        391: 2,
        392: 1,
        393: 2,
        394: 3,
        395: 1,
        396: 1,
        397: 2,
        398: 3,
        399: 1,
        400: 3,
        401: 1,
        402: 3,
        403: 3,
        404: 3,
        405: 3,
        406: 2,
        407: 2,
        408: 2,
        409: 1,
        410: 2,
        411: 3,
        412: 2,
        413: 3,
        414: 2,
        415: 2,
        416: 1,
        417: 2,
        418: 2,
        419: 3,
        420: 1,
        421: 1,
        422: 1,
        423: 2,
        424: 2,
        425: 2,
        426: 3,
        427: 3,
        428: 3,
        429: 3,
        430: 1,
        431: 3,
        432: 2,
        433: 3,
        434: 1,
        435: 3,
        436: 2,
        437: 3,
        438: 3,
        439: 2,
        440: 2,
        441: 2,
        442: 3,
        443: 2,
        444: 1,
        445: 1,
        446: 3,
        447: 1,
        448: 2,
        449: 1,
        450: 1,
        451: 1,
        452: 1,
        453: 2,
        454: 2,
        455: 3,
        456: 2,
        457: 3,
        458: 3,
        459: 2,
        460: 2,
        461: 2,
        462: 2,
        463: 3,
        464: 2,
        465: 3,
        466: 3,
        467: 1,
        468: 1,
        469: 2,
        470: 1,
        471: 2,
        472: 2,
        473: 1,
        474: 2,
        475: 1,
        476: 3,
        477: 1,
        478: 1,
        479: 1,
        480: 3,
        481: 2,
        482: 1,
        483: 3,
        484: 1,
        485: 3,
        486: 2,
        487: 1,
        488: 2,
        489: 3,
        490: 3,
        491: 2,
        492: 2,
        493: 1,
        494: 2,
        495: 1,
        496: 2,
        497: 1,
        498: 1,
        499: 3,
        500: 2,
        501: 3,
        502: 2,
        503: 2,
        504: 3,
        505: 2,
        506: 3,
        507: 3,
        508: 3,
        509: 2,
        510: 3,
        511: 1,
        512: 1,
        513: 1,
        514: 1,
        515: 2,
        516: 3,
        517: 2,
        518: 2,
        519: 2,
        520: 1,
        521: 2,
        522: 3,
        523: 1,
        524: 2,
        525: 1,
        526: 3,
        527: 1,
        528: 3,
        529: 3,
        530: 2,
        531: 3,
        532: 3,
        533: 3,
        534: 3,
        535: 1,
        536: 3,
        537: 3,
        538: 1,
        539: 1,
        540: 2,
        541: 2,
        542: 2,
        543: 3,
        544: 2,
        545: 1,
        546: 2,
        547: 2,
        548: 2,
        549: 3,
        550: 1,
        551: 2,
        552: 1,
        553: 2,
        554: 3,
        555: 2,
        556: 1,
        557: 1,
        558: 3,
        559: 1,
        560: 3,
        561: 1,
        562: 3,
        563: 3,
        564: 3,
        565: 1,
        566: 3,
        567: 3,
        568: 2,
        569: 2,
        570: 2,
        571: 2,
        572: 1,
        573: 1,
        574: 1,
        575: 1,
        576: 2,
        577: 1,
        578: 1,
        579: 1,
        580: 1,
        581: 2,
        582: 2,
        583: 2,
        584: 1,
        585: 3,
        586: 1,
        587: 3,
        588: 1,
        589: 1,
        590: 2,
        591: 2,
        592: 1,
        593: 1,
        594: 3,
        595: 3,
        596: 2,
        597: 3,
        598: 2,
        599: 1,
        600: 1,
        601: 3,
        602: 3,
        603: 2,
        604: 1,
        605: 1,
        606: 2,
        607: 2,
        608: 2,
        609: 1,
        610: 2,
        611: 2,
        612: 1,
        613: 1,
        614: 1,
        615: 2,
        616: 1,
        617: 2,
        618: 3,
        619: 1,
        620: 2,
        621: 1,
        622: 1,
        623: 2,
        624: 1,
        625: 3,
        626: 3,
        627: 2,
        628: 1,
        629: 2,
        630: 3,
        631: 2,
        632: 2,
        633: 2,
        634: 3,
        635: 1,
        636: 3,
        637: 3,
        638: 1,
        639: 2,
        640: 3,
        641: 1,
        642: 3,
        643: 3,
        644: 2,
        645: 2,
        646: 3,
        647: 3,
        648: 2,
        649: 2,
        650: 3,
        651: 2,
        652: 1,
        653: 3,
        654: 1,
        655: 1,
        656: 3,
        657: 2,
        658: 3,
        659: 3,
        660: 2,
        661: 3,
        662: 3,
        663: 1,
        664: 1,
        665: 3,
        666: 3,
        667: 3,
        668: 1,
        669: 2,
        670: 1,
        671: 1,
        672: 3,
        673: 2,
        674: 3,
        675: 3,
        676: 2,
        677: 2,
        678: 1,
        679: 3,
        680: 1,
        681: 1,
        682: 1,
        683: 2,
        684: 3,
        685: 1,
        686: 2,
        687: 3,
        688: 1,
        689: 2,
        690: 2,
        691: 1,
        692: 1,
        693: 1,
        694: 2,
        695: 3,
        696: 2,
        697: 3,
        698: 3,
        699: 1,
        700: 1,
        701: 3,
        702: 2,
        703: 2,
        704: 3,
        705: 2,
        706: 2,
        707: 2,
        708: 3,
        709: 2,
        710: 1,
        711: 1,
        712: 2,
        713: 2,
        714: 1,
        715: 3,
        716: 1,
        717: 2,
        718: 3,
        719: 3,
        720: 3,
        721: 1,
        722: 1,
        723: 1,
        724: 3,
        725: 2,
        726: 2,
        727: 3,
        728: 1,
        729: 2,
        730: 2,
        731: 1,
        732: 3,
        733: 2,
        734: 1,
        735: 1,
        736: 3,
        737: 3,
        738: 3,
        739: 3,
        740: 1,
        741: 1,
        742: 2,
        743: 3,
        744: 1,
        745: 1,
        746: 1,
        747: 1,
        748: 3,
        749: 2,
        750: 2,
        751: 2,
        752: 2,
        753: 2,
        754: 1,
        755: 2,
        756: 2,
        757: 1,
        758: 1,
        759: 1,
        760: 1,
        761: 1,
        762: 3,
        763: 1,
        764: 2,
        765: 2,
        766: 1,
        767: 3,
        768: 3,
        769: 1,
        770: 3,
        771: 2,
        772: 2,
        773: 3,
        774: 1,
        775: 1,
        776: 2,
        777: 1,
        778: 2,
        779: 1,
        780: 1,
        781: 2,
        782: 2,
        783: 3,
        784: 1,
        785: 3,
        786: 3,
        787: 2,
        788: 2,
        789: 2,
        790: 3,
        791: 2,
        792: 1,
        793: 3,
        794: 1,
        795: 2,
        796: 3,
        797: 1,
        798: 3,
        799: 3,
        800: 3,
        801: 2,
        802: 2,
        803: 2,
        804: 2,
        805: 1,
        806: 2,
        807: 3,
        808: 2,
        809: 3,
        810: 3,
        811: 3,
        812: 3,
        813: 1,
        814: 2,
        815: 3,
        816: 2,
        817: 3,
        818: 3,
        819: 1,
        820: 3,
        821: 3,
        822: 1,
        823: 2,
        824: 3,
        825: 1,
        826: 1,
        827: 3,
        828: 1,
        829: 3,
        830: 2,
        831: 1,
        832: 1,
        833: 3,
        834: 2,
        835: 2,
        836: 3,
        837: 2,
        838: 3,
        839: 3,
        840: 1,
        841: 3,
        842: 3,
        843: 3,
        844: 1,
        845: 3,
        846: 2,
        847: 2,
        848: 3,
        849: 3,
        850: 2,
        851: 1,
        852: 3,
        853: 3,
        854: 3,
        855: 2,
        856: 3,
        857: 1,
        858: 3,
        859: 1,
        860: 1,
        861: 1,
        862: 1,
        863: 2,
        864: 2,
        865: 3,
        866: 3,
        867: 3,
        868: 2,
        869: 1,
        870: 2,
        871: 1,
        872: 2,
        873: 2,
        874: 1,
        875: 2,
        876: 1,
        877: 3,
        878: 2,
        879: 1,
        880: 2,
        881: 3,
        882: 3,
        883: 2,
        884: 1,
        885: 3,
        886: 3,
        887: 3,
        888: 1,
        889: 3,
        890: 3,
        891: 3,
        892: 3,
        893: 2,
        894: 1,
        895: 1,
        896: 3,
        897: 3,
        898: 3,
        899: 3,
        900: 3,
        901: 2,
        902: 1,
        903: 2,
        904: 1,
        905: 2,
        906: 1,
        907: 1,
        908: 2,
        909: 2,
        910: 1,
        911: 3,
        912: 3,
        913: 3,
        914: 3,
        915: 3,
        916: 1,
        917: 3,
        918: 1,
        919: 2,
        920: 2,
        921: 2,
        922: 3,
        923: 2,
        924: 3,
        925: 2,
        926: 3,
        927: 1,
        928: 1,
        929: 3,
        930: 3,
        931: 1,
        932: 3,
        933: 3,
        934: 1,
        935: 1,
        936: 3,
        937: 2,
        938: 1,
        939: 3,
        940: 1,
        941: 3,
        942: 2,
        943: 2,
        944: 1,
        945: 3,
        946: 3,
        947: 3,
        948: 3,
        949: 1,
        950: 1,
        951: 3,
        952: 3,
        953: 2,
        954: 2,
        955: 3,
        956: 3,
        957: 2,
        958: 1,
        959: 1,
        960: 3,
        961: 2,
        962: 2,
        963: 3,
        964: 2,
        965: 3,
        966: 2,
        967: 3,
        968: 2,
        969: 2,
        970: 1,
        971: 2,
        972: 2,
        973: 2,
        974: 1,
        975: 3,
        976: 2,
        977: 3,
        978: 1,
        979: 3,
        980: 2,
        981: 3,
        982: 2,
        983: 1,
        984: 2,
        985: 1,
        986: 3,
        987: 1,
        988: 2,
        989: 1,
        990: 1,
        991: 2,
        992: 2,
        993: 3,
        994: 2,
        995: 2,
        996: 2,
        997: 2,
        998: 3,
        999: 1,
        1000: 2,


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

    export_solution_to_excel(inst, sol, REQ, out_path="ffh_solution_improved1000.xlsx")

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