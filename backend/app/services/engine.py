"""The JourneyMind engine: one request in, one recommendation out.

    request
      -> data layer          (bundled study-area bundle)
      -> multimodal graph    (road + transit + transfer + ride + access)
      -> travel-time model   (GraphSAGE, or whichever model is loaded)
      -> candidate journeys  (Yen's k-shortest under several weightings)
      -> constraint filter   (budget, deadline)
      -> Pareto frontier     (drop dominated journeys)
      -> personalised rank   (normalised weighted score)
      -> explanation         (deterministic, from the journey's attributes)

Every stage records what it did into a `pipeline` trace that is returned with
the response, so the recommendation can be audited rather than trusted.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from ..config import get_settings
from ..data.geo import haversine_km
from ..graph.builder import DESTINATION_ID, ORIGIN_ID, RequestGraph, get_graph
from ..graph.features import TimeContext
from ..models.fares import FareEstimator
from ..models.loader import active_model_info, get_predictor
from ..optimisation import constraints as C
from ..optimisation.pareto import dominated_by, frontier
from ..optimisation.scoring import pick_alternatives, score_all, weights_for
from ..routing.costs import build_cost_table
from ..routing.journey import Journey, build_journey, deduplicate
from ..routing.kshortest import BLENDS, generate_candidates
from ..routing.validate import partition_valid, rejection_summary
from .explain import explain, explain_alternative, no_feasible_message

log = logging.getLogger("journeymind.engine")


class RoutingError(Exception):
    """Raised for a request the engine cannot serve. Carries a user-safe message."""

    def __init__(self, message: str, code: str = "routing_error"):
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass
class JourneyRequest:
    origin_lat: float
    origin_lon: float
    origin_label: str
    dest_lat: float
    dest_lon: float
    dest_label: str
    departure: datetime
    budget: float
    max_time_min: float
    preference: str = "balanced"
    manual_weights: dict | None = None
    max_transfers: int = 3
    allowed_modes: set[str] | None = None
    rain: bool = False


@dataclass
class Recommendation:
    request: JourneyRequest
    recommended: Journey | None
    recommended_status: C.ConstraintStatus | None
    explanation: object | None
    alternatives: list[dict] = field(default_factory=list)
    fallbacks: list[dict] = field(default_factory=list)
    mode_comparison: list[dict] = field(default_factory=list)
    feasible: bool = True
    message: str | None = None
    pipeline: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)
    preset: str = "balanced"
    #: Every candidate that survived validation, cheapest first. The comparison
    #: builds its "travel in stages" list from these rather than from the two
    #: alternatives it happens to show, so a genuinely cheap itinerary is not
    #: lost just because something else was ranked above it.
    candidates: list[Journey] = field(default_factory=list)
    #: The walk-the-whole-way journey, whether or not it was recommended.
    #: Walking is always physically available and is the true cost floor, so
    #: the comparison must be able to price it even when nobody would choose
    #: it. Harvested from the candidate set rather than from the presentation
    #: pool, which only ever contained it by accident.
    walk_reference: Journey | None = None


# The single-vehicle options a rider would otherwise have had to price by hand,
# one app at a time. v1 section 3 shows exactly this table as the worked example;
# section 17 counts it as baseline 5, "does multi-modal planning beat what the
# apps do today?". The search already generates one reference journey per mode
# (routing/kshortest.REFERENCE_MODES) -- until now they were computed, filtered
# out for being dominated or over budget, and never shown to anyone.
COMPARISON_MODES = ("metro", "bus", "bike_taxi", "auto", "cab")


def walk_only(journeys: list[Journey]) -> Journey | None:
    """The fastest journey that uses nothing but feet."""
    walks = [j for j in journeys if set(j.modes) == {"walk"}]
    return min(walks, key=lambda j: j.total_min) if walks else None


#: Modes that give a journey its identity. A trip built around the metro is a
#: metro trip even when a bike taxi covers the first kilometre.
TRANSIT_MODES = ("metro", "bus")


def single_vehicle_mode(j: Journey) -> str | None:
    """The one vehicle this journey uses, or None if it mixes or only walks."""
    vehicles = {m for m in j.modes if m != "walk"}
    return next(iter(vehicles)) if len(vehicles) == 1 else None


def primary_mode(j: Journey) -> str | None:
    """What a rider would call this journey.

    The transit spine wins when there is exactly one: "bike taxi, metro, bike
    taxi" is how you take the metro, and pricing it as anything else is how the
    Metro card came to advertise 25 rupees for a journey that also needed two
    bike taxis. Otherwise it is a single-vehicle trip and names itself.
    """
    transit = [m for m in j.modes if m in TRANSIT_MODES]
    if len(set(transit)) == 1:
        return transit[0]
    return single_vehicle_mode(j)


def access_legs(j: Journey, spine: str) -> list:
    """The hailed legs at either end of a transit journey."""
    return [lg for lg in j.legs if lg.mode != spine and lg.kind == "ride"]


def build_mode_comparison(journeys: list[Journey], best: Journey | None,
                          budget: float, max_time_min: float) -> list[dict]:
    """One row per vehicle mode: what that option alone would have cost you.

    Every mode is reported whether or not it survived filtering, because "you
    cannot afford it" is information the rider wants and is the whole point of
    the comparison. Rows are never presented as recommendations -- each carries
    the limit it breaks.
    """
    by_mode: dict[str, Journey] = {}
    for j in journeys:
        m = primary_mode(j)
        if m is None:
            continue
        # cheapest wins the row, then fastest; the rider is comparing options,
        # not variants of one option
        cur = by_mode.get(m)
        if cur is None or (j.cost, j.total_min) < (cur.cost, cur.total_min):
            by_mode[m] = j

    rows = []
    for mode in COMPARISON_MODES:
        j = by_mode.get(mode)
        if j is None:
            continue
        st = C.evaluate(j, budget, max_time_min)
        if st.feasible:
            verdict = "Fits both your limits"
        elif not st.within_budget and not st.within_time:
            verdict = (f"{-st.budget_headroom:.0f} over budget and "
                       f"{-st.time_headroom:.0f} min too slow")
        elif not st.within_budget:
            verdict = f"You cannot afford it — {-st.budget_headroom:.0f} over budget"
        else:
            verdict = f"You would be {-st.time_headroom:.0f} minutes late"
        acc = access_legs(j, mode)
        rows.append({
            "mode": mode,
            "journey_id": j.journey_id,
            # The hailed legs at either end. The fare and the minutes are
            # already inside this row's totals -- a card that quoted the metro
            # ticket alone described a two-hour journey as a 25-rupee one -- so
            # these are here to be shown and to be reasoned about, not added.
            "access": {
                "rides": len(acc),
                "mode": acc[0].mode if acc else None,
                "minutes": round(sum(lg.total_min for lg in acc), 1),
                "distance_km": round(sum(lg.total_km for lg in acc), 2),
                "fare": round(sum(lg.fare.amount for lg in acc if lg.fare), 2),
                "fare_low": round(sum(lg.fare.low for lg in acc if lg.fare), 2),
                "fare_high": round(sum(lg.fare.high for lg in acc if lg.fare), 2),
            },
            "cost": round(j.cost, 2),
            "total_cost": j.total_cost,
            "total_min": round(j.total_min, 1),
            "distance_km": round(j.distance_km, 3),
            "transfers": j.transfers,
            "feasible": st.feasible,
            "verdict": verdict,
            "beaten_by_recommendation": bool(
                best is not None and best.journey_id != j.journey_id
                and (best.cost < j.cost - 1 or best.total_min < j.total_min - 1)),
        })
    return rows


class JourneyMindEngine:
    def __init__(self, city_id: str | None = None):
        s = get_settings()
        self.graph = get_graph(city_id or s.city_id)
        self.fares = FareEstimator(self.graph.fares)
        self.city = self.graph.city

    # -- helpers -----------------------------------------------------------
    def _check_inside_area(self, lat: float, lon: float, what: str) -> None:
        bbox = self.city.bbox
        pad = 0.06  # ~6.5 km of slack outside the study bbox
        if not (bbox["min_lat"] - pad <= lat <= bbox["max_lat"] + pad
                and bbox["min_lon"] - pad <= lon <= bbox["max_lon"] + pad):
            raise RoutingError(
                f"{what} is outside the {self.city.display_name} study area. "
                f"This MVP covers one bounded corridor only.",
                code="outside_study_area",
            )

    # -- the pipeline ------------------------------------------------------
    def recommend(self, req: JourneyRequest) -> Recommendation:
        t0 = time.perf_counter()
        s = get_settings()
        trace: dict = {}

        self._check_inside_area(req.origin_lat, req.origin_lon, "Your starting point")
        self._check_inside_area(req.dest_lat, req.dest_lon, "Your destination")

        straight = haversine_km(req.origin_lat, req.origin_lon, req.dest_lat, req.dest_lon)
        if straight < 0.05:
            raise RoutingError("Your start and destination are the same place.",
                               code="same_endpoints")

        # 1. graph -------------------------------------------------------
        rg = RequestGraph(self.graph, (req.origin_lat, req.origin_lon),
                          (req.dest_lat, req.dest_lon), req.allowed_modes)
        trace["graph"] = {
            "nodes": len(rg.nodes), "edges": len(rg.edges),
            "request_edges_added": len(rg.request_edges),
            "straight_line_km": round(straight, 3),
        }

        # 2. travel-time model ------------------------------------------
        predictor = get_predictor()
        ctx = TimeContext.from_datetime(req.departure, rain=req.rain)
        costs = build_cost_table(rg, self.graph, predictor, ctx)
        trace["prediction"] = costs.diagnostics
        model_info = active_model_info(predictor)

        # 3. candidates --------------------------------------------------
        # Pooling k paths across five weightings and de-duplicating yields
        # roughly `k_candidates` genuinely distinct journeys.
        k_per_blend = max(3, round(s.k_candidates / len(BLENDS)))
        paths = generate_candidates(rg, costs, k_per_blend, req.max_transfers)
        if not paths:
            raise RoutingError(
                "No route could be found between those two points inside the "
                "study area. Try points closer to the corridor.",
                code="no_path",
            )
        journeys = [
            build_journey(rg, costs, p.edges, self.fares, f"J{i + 1}", p.origin_blend)
            for i, p in enumerate(paths)
        ]
        journeys = deduplicate(journeys)

        # 3b. logic gate --------------------------------------------------
        # A path through a graph is not automatically a journey a person could
        # take. Anything physically or arithmetically impossible dies here,
        # BEFORE ranking -- so a nonsense candidate can never win, and the
        # interface is never the thing hiding it.
        before_validation = len(journeys)
        journeys, rejected = partition_valid(journeys, straight)
        trace["validation"] = rejection_summary(rejected)
        trace["validation"]["checked"] = before_validation
        if not journeys:
            raise RoutingError(
                "No usable route could be built between those two points. "
                "Every candidate broke a physical or arithmetic check.",
                code="no_valid_journey")

        trace["candidates"] = {
            "paths_found": len(paths), "after_deduplication": before_validation,
            "after_validation": len(journeys),
            "k_per_weighting": k_per_blend,
            "note": ("Yen's k-shortest paths pooled across five time/money "
                     "weightings. This approximates the resource-constrained "
                     "shortest-path problem, which is NP-hard."),
        }

        # 4. constraints -------------------------------------------------
        feasible, infeasible = C.partition(journeys, req.budget, req.max_time_min)
        trace["constraints"] = {
            "budget": req.budget, "max_time_min": req.max_time_min,
            "kept": len(feasible), "removed_over_budget":
                sum(1 for _, st in infeasible if not st.within_budget),
            "removed_over_time":
                sum(1 for _, st in infeasible if not st.within_time),
        }

        weights, preset = weights_for(req.preference, req.manual_weights)

        # 4b. nothing fits: say so, then offer labelled near-misses -------
        if not feasible:
            fallbacks = C.near_miss_alternatives(infeasible)
            ranked_fb = score_all([f["journey"] for f in fallbacks], weights)
            order = {j.journey_id: i for i, j in enumerate(ranked_fb)}
            fallbacks.sort(key=lambda f: order.get(f["journey"].journey_id, 99))
            trace["pareto"] = {"skipped": "no feasible journey to filter"}
            trace["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return Recommendation(
                request=req, recommended=None, recommended_status=None,
                explanation=None, alternatives=[],
                fallbacks=[{
                    "label": f["label"], "why": f["why"], "journey": f["journey"],
                    "status": f["status"],
                    "reason": explain_alternative(f["journey"], f["status"], f["journey"]),
                } for f in fallbacks],
                feasible=False,
                message=no_feasible_message(req.budget, req.max_time_min),
                mode_comparison=build_mode_comparison(
                    journeys, None, req.budget, req.max_time_min),
                pipeline=trace, model_info=model_info,
                weights=weights.as_dict(), preset=preset,
                walk_reference=walk_only(journeys),
                candidates=sorted(journeys, key=lambda j: (j.cost, j.total_min)),
            )

        # 5. Pareto ------------------------------------------------------
        feasible_js = [j for j, _ in feasible]
        killed = dominated_by(feasible_js)
        front = frontier(feasible_js)
        trace["pareto"] = {
            "in": len(feasible_js), "on_frontier": len(front),
            "dominated_removed": len(feasible_js) - len(front),
            "dominated_by": killed,
            "note": ("A journey no better than another on cost, time, transfers "
                     "or comfort is removed. All four are used, so an option that "
                     "loses on price and speed can still survive on comfort."),
        }

        # 6. rank --------------------------------------------------------
        ranked = score_all(front, weights)
        best = ranked[0]
        best_status = C.evaluate(best, req.budget, req.max_time_min)
        alts = pick_alternatives(ranked, s.max_alternatives)
        trace["ranking"] = {
            "preset": preset, "weights": weights.as_dict(),
            "scored": len(ranked),
            "order": [{"id": j.journey_id, "score": j.score,
                       "cost": j.cost, "time": round(j.total_min, 1)} for j in ranked],
        }

        expl = explain(best, best_status, journeys, req.budget,
                       req.max_time_min, preset)

        # If the scheduled network is shut at this hour, say so on the answer
        # itself rather than letting the absence of a metro look like a routing
        # preference.
        closed = costs.diagnostics.get("routes_out_of_service") or []
        if closed and len(closed) == len(self.graph.routes):
            expl.caveats.append(
                "No metro or bus service is running at this hour, so only road "
                "options are offered. Service hours are approximate.")
        elif closed:
            expl.caveats.append(
                f"Not running at this hour: {', '.join(closed[:3])}"
                + (f" and {len(closed) - 3} more." if len(closed) > 3 else "."))

        alternatives = []
        for a in alts:
            st = C.evaluate(a, req.budget, req.max_time_min)
            alternatives.append({
                "journey": a, "status": st, "kind": "feasible",
                "reason": explain_alternative(a, st, best),
            })

        # Fewer than two options actually fit? Show the near misses rather than
        # a lonely single result -- clearly labelled with the limit they break,
        # never presented as if they were valid answers.
        if len(alternatives) < s.max_alternatives and infeasible:
            shown = {best.signature} | {a["journey"].signature for a in alternatives}
            def overshoot(pair):
                _, st = pair
                return (max(0.0, -st.budget_headroom) / max(req.budget, 1.0)
                        + max(0.0, -st.time_headroom) / max(req.max_time_min, 1.0))

            def mode_set(j):
                return frozenset(m for m in j.modes if m != "walk")

            near = sorted((p for p in infeasible if p[0].signature not in shown),
                          key=overshoot)
            used_modes = {mode_set(best)} | {mode_set(a["journey"]) for a in alternatives}
            # two passes: first the closest miss with a mode mix nobody has seen,
            # then simply the closest misses
            for require_new_modes in (True, False):
                for j, st in near:
                    if len(alternatives) >= s.max_alternatives:
                        break
                    if j.signature in shown:
                        continue
                    if require_new_modes and mode_set(j) in used_modes:
                        continue
                    shown.add(j.signature)
                    used_modes.add(mode_set(j))
                    alternatives.append({
                        "journey": j, "status": st, "kind": "near_miss",
                        "reason": explain_alternative(j, st, best),
                    })
            trace["ranking"]["near_misses_added"] = sum(
                1 for a in alternatives if a["kind"] == "near_miss")

        comparison = build_mode_comparison(journeys, best, req.budget, req.max_time_min)
        trace["mode_comparison"] = {
            "modes_priced": [r["mode"] for r in comparison],
            "affordable": [r["mode"] for r in comparison if r["feasible"]],
            "note": ("One single-vehicle reference journey per mode, priced whether "
                     "or not it survived filtering. These are baseline 5 from the "
                     "documentation, and they are what the rider would otherwise "
                     "have had to check one app at a time."),
        }

        trace["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return Recommendation(
            request=req, recommended=best, recommended_status=best_status,
            explanation=expl, alternatives=alternatives, feasible=True,
            mode_comparison=comparison,
            pipeline=trace, model_info=model_info,
            weights=weights.as_dict(), preset=preset,
            walk_reference=walk_only(journeys),
            candidates=sorted(journeys, key=lambda j: (j.cost, j.total_min)),
        )


_engine: JourneyMindEngine | None = None


def get_engine() -> JourneyMindEngine:
    global _engine
    if _engine is None:
        _engine = JourneyMindEngine()
    return _engine


def warm_up() -> dict:
    """Called at startup so the first user request is not the one that pays
    for building the graph and loading the model."""
    eng = get_engine()
    predictor = get_predictor()
    ctx = TimeContext.from_datetime(datetime(2025, 1, 6, 9, 0))
    predictor.predict_static(eng.graph, ctx)
    # Warm the reliability heads and the booking table too. Both are lazy and
    # cached, so whoever touches them first pays for them -- and on a cold
    # instance that is a rider mid-demo. Better the boot pays it.
    bookings = 0
    try:
        from ..reliability.model import get_reliability_model
        get_reliability_model()
        from ..enterprise.analytics import load_bookings
        table = load_bookings()
        bookings = table.n if table is not None else 0
    except Exception:
        log.warning("secondary warm-up failed; those layers will load on demand",
                    exc_info=True)

    return {
        "city": eng.city.display_name,
        "nodes": len(eng.graph.nodes),
        "edges": len(eng.graph.edges),
        "model": predictor.info.display_name,
        "bookings": bookings,
    }
