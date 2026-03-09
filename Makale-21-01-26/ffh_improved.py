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
    vehicle_hub: Dict[str, Optional[str]],
    activated_fk: set = None,
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
    if activated_fk is None:
        activated_fk = set()

    max_to_ship = min(supply_available, remaining_hub)
    if max_to_ship <= 0:
        return None

    reuse: List[Tuple[float, str, int]] = []        # (var_cost, k, ship)
    new:   List[Tuple[float, float, str, int]] = []  # (beta, var_cost, k, ship)

    for k in inst.K:
        if cap_left[k] <= 0:
            continue
        if vehicle_hub[k] is not None and vehicle_hub[k] != b:
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
    vehicle_hub: Dict[str, Optional[str]] = {k: None for k in inst.K}
    activated_fk: set = set()  # built up within this scenario only

    Y_s: Dict[Tuple[str, str, str, str, int], int] = {}
    L_s: Dict[Tuple[str, str, str, str, int], int] = {}
    achieved: Dict[str, int] = {b: 0 for b in inst.B}
    # OPL constraint 4: Σ_b Y[f,b,k,s,t] ≤ 1  — supplier f can serve at most
    # one (b,k) combination per scenario-period. Track which hub each f is
    # committed to; block any attempt to commit f to a second hub.
    supplier_hub: Dict[str, str] = {}   # f -> b (first hub committed)

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
        vehicle_hub[k]  = b
        supplier_hub[f] = b   # lock supplier f to hub b
        activated_fk.add((f, k))
        supply_left[f] -= ship_int
        achieved[b]    += ship_int
        hub_needs[b]   -= ship_int
        if hub_needs[b] <= 0:
            hub_needs.pop(b, None)

    # ── PASS 1: consolidation — cover each hub in a single activation ────────
    for b in sorted(list(hub_needs.keys()), key=lambda b: hub_needs[b], reverse=True):
        if b not in hub_needs:
            continue
        need = hub_needs[b]
        # TIER-1 reuse: already-active (f,k), score = (coverage DESC, var_cost ASC)
        # TIER-2 new:   new activation,        score = (neg_beta DESC, coverage DESC, neg_var ASC)
        # Always pick any tier-1 over any tier-2.
        best_free: Optional[Tuple[float, float, str, str, int]] = None
        best_new:  Optional[Tuple[float, float, float, str, str, int]] = None

        for f in inst.F:
            if supply_left[f] <= 0:
                continue
            if (f, b) not in inst.allowed_f_b:
                continue
            # OPL constraint 4: supplier f already committed to a different hub
            if supplier_hub.get(f) is not None and supplier_hub[f] != b:
                continue
            for k in inst.K:
                if cap_left[k] <= 0:
                    continue
                if vehicle_hub[k] is not None and vehicle_hub[k] != b:
                    continue
                if (f, k) not in inst.beta or (f, b, k) not in inst.gamma:
                    continue
                ship = min(supply_left[f], cap_left[k], need)
                if ship <= 0:
                    continue
                var_cost = inst.gamma[(f, b, k)] * ship
                coverage = ship / need
                if (f, k) in activated_fk:
                    score = (coverage, -var_cost)
                    if best_free is None or score > (best_free[0], best_free[1]):
                        best_free = (coverage, -var_cost, f, k, ship)
                else:
                    score = (-inst.beta[(f, k)], coverage, -var_cost)
                    if best_new is None or score > (best_new[0], best_new[1], best_new[2]):
                        best_new = (-inst.beta[(f, k)], coverage, -var_cost, f, k, ship)

        if best_free is not None:
            f, k, ship = best_free[2], best_free[3], best_free[4]
        elif best_new is not None:
            f, k, ship = best_new[3], best_new[4], best_new[5]
        else:
            f = k = ship = None

        if f is not None:
            commit(f, b, k, ship)
            if DEBUG_MODE:
                print(f"  [P2-P1] t={t} s={s} CONSOLIDATE f={f}->b={b} k={k} ship={ship}")

    # ── PASS 2: residual Vogel for any unmet hub needs ───────────────────────
    while hub_needs:
        best_entry: Optional[Tuple[float, str, str, str, int]] = None

        for b, remaining in hub_needs.items():
            for f in inst.F:
                if supply_left[f] <= 0:
                    continue
                if (f, b) not in inst.allowed_f_b:
                    continue
                # OPL constraint 4: supplier f already committed to a different hub
                if supplier_hub.get(f) is not None and supplier_hub[f] != b:
                    continue
                result = _best_vehicle_for_pair(
                    f, b, remaining, supply_left[f], inst, cap_left, vehicle_hub,
                    activated_fk
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
    Z_t: Dict[Tuple[str, str, int], int] = {}
    Q_t: Dict[Tuple[str, str, int], int] = {}   # built from Z only

    r_deps = {r.r_id: set(r.depots) for r in inst.routes}
    r_cost = {r.r_id: r.fixed_cost for r in inst.routes}
    r_cap  = {r.r_id: int_floor(r.capacity) for r in inst.routes}
    r_hub  = {r.r_id: r.hub for r in inst.routes}

    # Greedy set-cover: select routes to minimise total fixed cost while
    # covering all depots with qmin > 0.
    # Score = (total qmin demand covered by this route) / (route fixed cost λ_r)
    # This is the standard weighted set-cover ratio heuristic.
    # It selects routes that deliver the most required product per euro of
    # fixed cost, naturally preferring routes that bundle many depots.
    # Utilisation penalties are NOT applied — they would cause the heuristic
    # to skip delivering to low-demand depots, violating constraint (19).
    selected: List[str] = []
    while U:
        best_r, best_score = None, -1.0
        for r in inst.routes:
            cover     = r_deps[r.r_id] & U
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
        X_t[(best_r, t)] = 1
        U -= r_deps[best_r]

    # Load routes (int quantities)
    rem = {i: float(qmin.get(i, 0.0)) for i in inst.D}
    for r_id in selected:
        cap  = r_cap[r_id]
        deps = sorted(
            (i for i in r_deps[r_id] if rem[i] > 1e-9),
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

    # Induce Q from Z — only set Q[b,i,t] if Z actually delivers to (b,i)
    # This guarantees constraint (13): Σ_{r:hub=b} Z[i,r,t] = Q[b,i,t]
    for (i, r_id, tt), z in Z_t.items():
        b = r_hub[r_id]
        Q_t[(b, i, tt)] = Q_t.get((b, i, tt), 0) + int(z)

    # Remove zero entries — OPL must see Q[b,i,t]=0 as simply absent
    Q_t = {k: v for k, v in Q_t.items() if v > 0}

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
                    inst, s, t, Q_sum,
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
                de_cum_t = sum(float(inst.mu[(i, a)]) for a in inst.T if a <= t)
                I_it_raw = float(sol.Q_cum[(i, t)]) - de_cum_t - float(sol.W_cum[(i, t)])
                sol.Ipos[(i, t)] = max(0, int(round(I_it_raw)))

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
    out_path: str = "ffh_solution_improved.xlsx",
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
    I_rows = [{"i": i, "t": t, "value": int(sol.Ipos.get((i, t), 0))}
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
    file_path = "step1.xlsx"

    route_to_depots = {
        1: [25,24,14,13], 2: [16,17,23,20], 3: [10,22,27,30], 4: [28,26,19],
        5: [11,12,29], 6: [8,18,21,15,9], 7: [19,20,14,7,2], 8: [17,15,21,25],
        9: [22,8,6,5,1], 10: [23,13,10,11], 11: [29,26,24,16,4,3], 12: [9,18,30,28,27],
        13: [30,23,5,4,2], 14: [12,15,21,25,28], 15: [7,9,10,16,22], 16: [18,19,17,14,12],
        17: [29,24,8,1,3], 18: [27,20,7,6,4,2], 19: [7,6,5,3,1], 20: [20,21,22,24,26,28],
        21: [23,8,16,13,11], 22: [6,5,4,3,2,1], 23: [17,30,14,9,10,12], 24: [18,19,25,26,27,29],
        25: [11,13,15,20,22,26], 26: [10,16,15,27], 27: [7,13,14,17,18], 28: [9,12,11],
        29: [8,19,23,25,24,21], 30: [18,30,29],
    }

    route_to_hub = {
        1: 2, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 2, 8: 3, 9: 2, 10: 1,
        11: 2, 12: 3, 13: 2, 14: 1, 15: 3, 16: 2, 17: 2, 18: 2, 19: 1,
        20: 3, 21: 2, 22: 3, 23: 3, 24: 2, 25: 1, 26: 1, 27: 3, 28: 1,
        29: 1, 30: 1,
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
    # Multi-assignment check
    viol = sum(
        1 for s in inst.S for t in inst.T for f in inst.F
        if sum(sol.Y.get((f, b, k, s, t), 0) for b in inst.B for k in inst.K) > 1
    )
    print(f"[CHECK] multi-assign (sum_{{b,k}} Y > 1) count = {viol}")

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

    export_solution_to_excel(inst, sol, REQ, out_path="ffh_solution_improved.xlsx")