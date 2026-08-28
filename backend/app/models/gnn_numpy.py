"""The serving-side forward pass: GraphSAGE / GAT / MLP in pure NumPy.

Why this exists. The deployed service runs on a small CPU instance. Installing
PyTorch there would add hundreds of megabytes and a large resident footprint to
run a two-layer network over a 160-node graph. So the model is trained offline
with PyTorch, its weights are exported to a `.npz`, and this module replays the
identical arithmetic with NumPy alone.

This is real inference on real learned weights -- not a lookup of cached
answers. `tests/test_model_parity.py` asserts that this implementation and the
PyTorch one agree to within 1e-4 on the same inputs, so the claim is checked
rather than asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..graph.features import (
    EDGE_FEATURE_DIM, NODE_FEATURE_DIM, TIME_FEATURE_DIM, TimeContext,
)
from .base import ModelInfo, TravelTimePredictor


# --------------------------------------------------------------------------
# primitive ops, mirroring the PyTorch versions exactly
# --------------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def leaky_relu(x: np.ndarray, slope: float = 0.2) -> np.ndarray:
    return np.where(x >= 0, x, x * slope)


def scatter_mean(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, values.shape[-1]), dtype=np.float64)
    np.add.at(out, index, values)
    cnt = np.zeros((n, 1), dtype=np.float64)
    np.add.at(cnt, index, 1.0)
    return out / np.maximum(cnt, 1.0)


def scatter_softmax(logits: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    mx = np.full((n, logits.shape[-1]), -1e30, dtype=np.float64)
    np.maximum.at(mx, index, logits)
    ex = np.exp(logits - mx[index])
    denom = np.zeros((n, logits.shape[-1]), dtype=np.float64)
    np.add.at(denom, index, ex)
    return ex / np.maximum(denom[index], 1e-16)


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    """torch.nn.Linear stores weight as [out, in]."""
    y = x @ w.T
    return y + b if b is not None else y


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class NumpyEdgeTravelTimeModel:
    """Loads an exported checkpoint and reproduces its forward pass."""

    def __init__(self, params: dict[str, np.ndarray], meta: dict):
        self.p = {k: v.astype(np.float64) for k, v in params.items()
                  if not k.startswith("__")}
        self.meta = meta
        self.encoder_kind: str = meta["encoder"]
        self.config: dict = meta["config"]
        self.n_layers: int = int(self.config["layers"])
        self.heads: int = int(self.config.get("heads", 2))

        sig = meta.get("features", {})
        for key, expect in (("node_dim", NODE_FEATURE_DIM),
                            ("edge_dim", EDGE_FEATURE_DIM),
                            ("time_dim", TIME_FEATURE_DIM)):
            got = sig.get(key)
            if got is not None and int(got) != expect:
                raise ValueError(
                    f"Checkpoint was trained with {key}={got} but this build "
                    f"encodes {expect}. Retrain with scripts/train.py."
                )

        self.node_mean = self.p.get("norm.node_mean", np.zeros(NODE_FEATURE_DIM))
        self.node_std = self.p.get("norm.node_std", np.ones(NODE_FEATURE_DIM))
        self.edge_mean = self.p.get("norm.edge_mean", np.zeros(EDGE_FEATURE_DIM))
        self.edge_std = self.p.get("norm.edge_std", np.ones(EDGE_FEATURE_DIM))

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "NumpyEdgeTravelTimeModel":
        with np.load(str(path), allow_pickle=False) as z:
            params = {k: z[k] for k in z.files}
        raw = params.pop("__meta__", None)
        if raw is None:
            raise ValueError(f"{path} has no __meta__ block; re-export it.")
        meta = json.loads(bytes(raw.astype(np.uint8)).decode("utf-8"))
        return cls(params, meta)

    # -- encoder -----------------------------------------------------------
    def _encode(self, x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        h = x
        for i in range(self.n_layers):
            if self.encoder_kind == "graphsage":
                pre = f"encoder.layers.{i}."
                h = (linear(h, self.p[pre + "lin_self.weight"], self.p[pre + "lin_self.bias"])
                     + linear(scatter_mean(h[src], dst, n),
                              self.p[pre + "lin_neigh.weight"], None))
            elif self.encoder_kind == "gat":
                pre = f"encoder.layers.{i}."
                w = self.p[pre + "lin.weight"]
                heads, dim_out = self.heads, w.shape[0] // self.heads
                wh = linear(h, w, None).reshape(n, heads, dim_out)
                a_src = (wh * self.p[pre + "att_src"]).sum(-1)
                a_dst = (wh * self.p[pre + "att_dst"]).sum(-1)
                logits = leaky_relu(a_src[src] + a_dst[dst], 0.2)
                alpha = scatter_softmax(logits, dst, n)
                msg = wh[src] * alpha[:, :, None]
                out = np.zeros((n, heads, dim_out), dtype=np.float64)
                np.add.at(out, dst, msg)
                h = out.reshape(n, heads * dim_out) + self.p[pre + "bias"]
            elif self.encoder_kind == "mlp":
                pre = f"encoder.layers.{i}."
                h = linear(h, self.p[pre + "lin.weight"], self.p[pre + "lin.bias"])
            else:
                raise ValueError(f"unknown encoder '{self.encoder_kind}'")
            if i < self.n_layers - 1:
                h = relu(h)   # dropout is inference-time identity
        return h

    # -- head --------------------------------------------------------------
    def _head(self, z: np.ndarray) -> np.ndarray:
        h = relu(linear(z, self.p["head.0.weight"], self.p["head.0.bias"]))
        h = relu(linear(h, self.p["head.3.weight"], self.p["head.3.bias"]))
        return linear(h, self.p["head.5.weight"], self.p["head.5.bias"])[:, 0]

    # -- public ------------------------------------------------------------
    def normalise_nodes(self, x: np.ndarray) -> np.ndarray:
        return (x - self.node_mean) / np.maximum(self.node_std, 1e-6)

    def normalise_edges(self, x: np.ndarray) -> np.ndarray:
        return (x - self.edge_mean) / np.maximum(self.edge_std, 1e-6)

    def embed(self, node_feats: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        return self._encode(self.normalise_nodes(node_feats.astype(np.float64)), src, dst)

    def forward_log(self, node_feats, src, dst, edge_uv, edge_feats, time_feats
                    ) -> np.ndarray:
        emb = self.embed(node_feats, src, dst)
        z = np.hstack([
            emb[edge_uv[:, 0]], emb[edge_uv[:, 1]],
            self.normalise_edges(edge_feats.astype(np.float64)),
            time_feats.astype(np.float64),
        ])
        return self._head(z)

    def predict_minutes(self, node_feats, src, dst, edge_uv, edge_feats, time_feats
                        ) -> np.ndarray:
        log = self.forward_log(node_feats, src, dst, edge_uv, edge_feats, time_feats)
        return np.maximum(np.expm1(log), 0.05).astype(np.float32)


# --------------------------------------------------------------------------
# predictor wrapper
# --------------------------------------------------------------------------
# How far a prediction may stray from an edge's free-flow time before it is
# treated as extrapolation gone wrong. Wide on purpose: real congestion in this
# corridor lands well inside these bounds.
COLD_START_LOW = 0.40
COLD_START_HIGH = 6.0

DISPLAY = {
    "graphsage": ("GraphSAGE", "gnn",
                  "Two rounds of message passing: each edge's prediction is "
                  "informed by everything within two hops of it."),
    "gat": ("GAT", "gnn",
            "Graph attention. Learns how much to weight each neighbour rather "
            "than averaging them equally."),
    "mlp": ("MLP (graph removed)", "neural",
            "Baseline 4 — the critical ablation. Same features and capacity as "
            "GraphSAGE with message passing deleted."),
}


class NeuralEdgePredictor(TravelTimePredictor):
    """Serves an exported GraphSAGE / GAT / MLP checkpoint."""

    def __init__(self, model: NumpyEdgeTravelTimeModel, weights_path: str | None = None):
        self.model = model
        self.weights_path = weights_path
        kind = model.encoder_kind
        name, family, note = DISPLAY.get(kind, (kind.upper(), "neural", ""))
        metrics = model.meta.get("metrics", {})
        self.info = ModelInfo(
            name=kind, display_name=name, family=family,
            predicts="estimated edge travel time",
            status="prototype",
            trained_on=model.meta.get("trained_on", "bundled travel-time observations"),
            notes=note,
        )
        self.metrics = metrics
        self._embed_cache: np.ndarray | None = None
        self._embed_graph = None          # strong ref keeps the cache key valid
        self._uv_cache: np.ndarray | None = None
        self._base_cache: np.ndarray | None = None
        self._base_graph = None
        self.last_clamped = 0

    @classmethod
    def load(cls, path: str | Path) -> "NeuralEdgePredictor":
        return cls(NumpyEdgeTravelTimeModel.load(path), str(path))

    def _embeddings(self, graph) -> np.ndarray:
        """Node embeddings depend only on the graph, never on the clock, so
        they are computed once per process and reused on every request."""
        if self._embed_cache is None or self._embed_graph is not graph:
            src, dst = graph.adj_index
            self._embed_cache = self.model.embed(graph.node_features, src, dst)
            self._uv_cache = np.asarray(
                [[graph.node_pos[graph.edges[i].u], graph.node_pos[graph.edges[i].v]]
                 for i in graph.static_edge_idx], dtype=np.int64)
            self._embed_graph = graph
        return self._embed_cache

    def predict_static(self, graph, ctx: TimeContext) -> np.ndarray:
        emb = self._embeddings(graph)
        uv = self._uv_cache
        ef = self.model.normalise_edges(graph.edge_features.astype(np.float64))
        tf = np.tile(ctx.vector().astype(np.float64), (ef.shape[0], 1))
        z = np.hstack([emb[uv[:, 0]], emb[uv[:, 1]], ef, tf])
        pred = np.maximum(np.expm1(self.model._head(z)), 0.05).astype(np.float32)
        return self._guard(graph, pred)

    def _guard(self, graph, pred: np.ndarray) -> np.ndarray:
        """Cold-start safeguard.

        Some edges -- walking transfers, in this bundle -- carry no travel-time
        observations at all, so the model is extrapolating on them. The
        documentation is explicit about this case: fall back towards a free-flow
        estimate rather than failing silently. Predictions are clamped to a wide
        but finite band around each edge's own free-flow time. A correct
        prediction is nowhere near these bounds; a broken one cannot escape
        them, and the count is reported so the clamping is visible rather than
        hidden.
        """
        if self._base_cache is None or self._base_graph is not graph:
            self._base_cache = np.asarray(
                [max(graph.edges[i].base_min, 1e-3) for i in graph.static_edge_idx],
                dtype=np.float32)
            self._base_graph = graph
        base = self._base_cache
        lo, hi = base * COLD_START_LOW, base * COLD_START_HIGH
        clamped = np.clip(pred, lo, hi)
        self.last_clamped = int(np.count_nonzero(clamped != pred))
        return clamped

    def predict_rows(self, node_feats, edge_feats, src, dst, edge_uv, time_feats, **kw):
        return self.model.predict_minutes(node_feats, src, dst, edge_uv,
                                          edge_feats, time_feats)

    def metadata(self) -> dict:
        d = self.info.as_dict()
        if self.metrics:
            d["validation_metrics"] = self.metrics
        return d
