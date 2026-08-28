"""Travel-time prediction interface, shared by every model and baseline.

Section 18 of the project documentation asks for six comparable models behind
one interface so that "does the graph help?" is an experiment rather than an
assumption. That interface is `TravelTimePredictor`. Everything -- free-flow
physics, a lookup table, gradient-boosted trees, a graph-free MLP, GraphSAGE,
GAT -- implements the same two methods and is evaluated by the same script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..graph.features import TimeContext


@dataclass(frozen=True)
class ModelInfo:
    """What the API reports about whatever produced the numbers on screen."""

    name: str            # graphsage | gat | mlp | gbt | historical | freeflow
    display_name: str
    family: str          # gnn | neural | tree | lookup | analytic
    predicts: str
    status: str          # prototype | baseline
    trained_on: str
    notes: str
    available: bool = True

    def as_dict(self) -> dict:
        return {
            "model": self.display_name, "key": self.name, "family": self.family,
            "prediction": self.predicts, "status": self.status,
            "trained_on": self.trained_on, "notes": self.notes,
        }


class TravelTimePredictor(ABC):
    """Predicts minutes to traverse each *static* graph edge at a given time.

    Static edges are road, transit and transfer edges -- the ones with a fixed
    identity that can carry historical observations. Ride edges are priced
    separately (see `apply_predictions`) because a hailed vehicle is not a
    fixed piece of infrastructure.
    """

    info: ModelInfo

    @abstractmethod
    def predict_static(self, graph, ctx: TimeContext) -> np.ndarray:
        """Minutes per static edge, aligned with `graph.static_edge_idx`."""

    def predict_rows(self, node_feats: np.ndarray, edge_feats: np.ndarray,
                     src: np.ndarray, dst: np.ndarray, edge_uv: np.ndarray,
                     time_feats: np.ndarray) -> np.ndarray:
        """Batch interface used by the training/evaluation harness, where each
        row may carry its own time context. Optional for simple baselines."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the batch interface")


# --------------------------------------------------------------------------
# turning per-edge predictions into per-edge minutes on a request graph
# --------------------------------------------------------------------------
CONGESTION_FLOOR = 0.75
CONGESTION_CEILING = 3.5


#: How many points along a ride edge are sampled for congestion. Interior
#: fractions only: the request's own origin and destination nodes carry no road
#: edges, so sampling them contributes nothing but the global mean.
RIDE_SAMPLE_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _ride_congestion_samples(request_graph, base_graph) -> dict[int, np.ndarray]:
    """For each ride edge, the node rows whose congestion describes its route.

    Averaging only the two ENDPOINTS made travel time non-additive: a long
    door-to-door ride was scored on its two ends while the same ground split
    across a hub picked up that hub's reading, so splitting a trip could make
    it *faster*. The router then preferred two hailed vehicles in a row over
    one direct ride -- twice the base fare, twice the pickup wait, and a
    journey no rider would take. Sampling along the line prices the corridor
    the vehicle actually drives through, and the whole is once again roughly
    the sum of its parts.

    Geometry does not change between elapsed-time buckets, so this is computed
    once per request and cached on the request graph.
    """
    cached = getattr(request_graph, "_ride_samples", None)
    if cached is not None:
        return cached

    order = base_graph.node_order
    lat = np.fromiter((base_graph.nodes[n].lat for n in order), dtype=np.float64,
                      count=len(order))
    lon = np.fromiter((base_graph.nodes[n].lon for n in order), dtype=np.float64,
                      count=len(order))

    ride_idx = [i for i, e in enumerate(request_graph.edges) if e.kind == "ride"]
    samples: dict[int, np.ndarray] = {}
    if not ride_idx:
        request_graph._ride_samples = samples
        return samples

    pts = []
    for i in ride_idx:
        e = request_graph.edges[i]
        nu, nv = request_graph.nodes.get(e.u), request_graph.nodes.get(e.v)
        if nu is None or nv is None:
            pts.append([(0.0, 0.0)] * len(RIDE_SAMPLE_FRACTIONS))
            continue
        pts.append([(nu.lat + (nv.lat - nu.lat) * f,
                     nu.lon + (nv.lon - nu.lon) * f) for f in RIDE_SAMPLE_FRACTIONS])

    flat = np.asarray(pts, dtype=np.float64).reshape(-1, 2)      # (n_edge*k, 2)
    # Planar nearest neighbour: over a single city corridor the error against a
    # great-circle distance is far below the spacing between graph nodes.
    dlat = flat[:, 0][:, None] - lat[None, :]
    dlon = (flat[:, 1][:, None] - lon[None, :]) * np.cos(np.radians(lat.mean()))
    nearest = np.argmin(dlat * dlat + dlon * dlon, axis=1)
    nearest = nearest.reshape(len(ride_idx), len(RIDE_SAMPLE_FRACTIONS))
    for row, i in enumerate(ride_idx):
        samples[i] = nearest[row]

    request_graph._ride_samples = samples
    return samples


def predict_edge_minutes(request_graph, base_graph, predictor: TravelTimePredictor,
                         ctx: TimeContext) -> tuple[np.ndarray, dict]:
    """Minutes to traverse every edge of a request graph at time `ctx`.

    Static edges (road / transit / transfer) get the model's prediction
    directly. Ride edges get a free-flow time scaled by the congestion the
    model predicts along the corridor they drive through -- so a bike-taxi
    slows down when the model thinks those streets are slow. Walking is not
    scaled: pedestrians do not sit in traffic.

    Returns an array indexed by `request_graph.edges[i].idx` and a small
    diagnostics dict. Nothing is written onto the shared city graph.
    """
    minutes = predictor.predict_static(base_graph, ctx)
    row_of = base_graph.edge_id_to_static_row()

    # congestion ratio per road edge, aggregated onto its endpoints
    ratio_sum: dict[str, float] = {}
    ratio_n: dict[str, int] = {}
    for idx, row in row_of.items():
        e = base_graph.edges[idx]
        if e.kind != "road" or e.base_min <= 1e-6:
            continue
        ratio = float(np.clip(minutes[row] / e.base_min, CONGESTION_FLOOR, CONGESTION_CEILING))
        for n in (e.u, e.v):
            ratio_sum[n] = ratio_sum.get(n, 0.0) + ratio
            ratio_n[n] = ratio_n.get(n, 0) + 1

    total_n = sum(ratio_n.values())
    global_ratio = (sum(ratio_sum.values()) / total_n) if total_n else 1.0

    def node_congestion(node_id: str) -> float:
        n = ratio_n.get(node_id, 0)
        return ratio_sum[node_id] / n if n else global_ratio

    # congestion per node row, for the sampled ride corridors
    node_ratio = np.fromiter(
        (node_congestion(n) for n in base_graph.node_order),
        dtype=np.float64, count=len(base_graph.node_order))
    samples = _ride_congestion_samples(request_graph, base_graph)

    out = np.empty(len(request_graph.edges), dtype=np.float32)
    n_model = n_ride = n_walk = 0
    for i, e in enumerate(request_graph.edges):
        if e.mode == "walk":
            # Pedestrians do not sit in traffic. Walking time is distance over
            # pace, computed analytically -- the model is never asked for it.
            out[i] = e.walk_min or e.base_min
            n_walk += 1
        elif e.kind == "ride":
            rows = samples.get(i)
            if rows is None or not len(rows):
                factor = (node_congestion(e.u) + node_congestion(e.v)) / 2.0
            else:
                factor = float(node_ratio[rows].mean())
            out[i] = e.base_min * factor
            n_ride += 1
        elif e.is_static and e.idx in row_of:
            out[i] = minutes[row_of[e.idx]]
            n_model += 1
        else:
            out[i] = e.base_min
    return out, {
        "transit_edges_from_model": n_model,
        "ride_edges_scaled_by_predicted_congestion": n_ride,
        "walk_edges_analytic": n_walk,
        "road_edges_scored_for_congestion": len(ratio_n),
        "mean_congestion_ratio": round(float(global_ratio), 4),
        "ride_congestion_samples_per_edge": len(RIDE_SAMPLE_FRACTIONS),
        "note": ("The model predicts vehicle time on road and transit edges. "
                 "Transit legs use it directly; ride legs are scaled by the "
                 "congestion it implies along the corridor they drive through, "
                 "sampled at several points so that splitting a trip cannot "
                 "change its predicted duration; walking is analytic."),
    }
