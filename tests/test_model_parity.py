"""The serving model must be the model that was trained.

The deployed service runs the GNN through a NumPy forward pass so that PyTorch
is not needed in the image. That is only legitimate if the two implementations
compute the same thing, so this asserts it rather than assuming it. The PyTorch
half is skipped automatically where torch is not installed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.graph.builder import RequestGraph, get_graph                    # noqa: E402
from app.graph.features import TimeContext                               # noqa: E402
from app.models.base import predict_edge_minutes                         # noqa: E402
from app.models.gnn_numpy import (                                       # noqa: E402
    COLD_START_HIGH, COLD_START_LOW, NeuralEdgePredictor, NumpyEdgeTravelTimeModel,
)
from app.models.loader import WEIGHT_FILENAMES, get_predictor, registry  # noqa: E402

ENCODERS = ["graphsage", "gat", "mlp"]
MODELS_DIR = os.path.join(ROOT, "models")


def weights(enc):
    return os.path.join(MODELS_DIR, WEIGHT_FILENAMES[enc])


@pytest.fixture(scope="module")
def graph():
    return get_graph()


@pytest.mark.parametrize("enc", ENCODERS)
def test_checkpoint_exists_and_loads(enc):
    path = weights(enc)
    if not os.path.exists(path):
        pytest.skip(f"{enc} not trained; run scripts/train.py --encoder {enc}")
    m = NumpyEdgeTravelTimeModel.load(path)
    assert m.encoder_kind == enc
    assert m.meta["features"]["node_dim"] > 0


@pytest.mark.parametrize("enc", ENCODERS)
def test_numpy_matches_pytorch(enc, graph):
    path = weights(enc)
    if not os.path.exists(path):
        pytest.skip(f"{enc} not trained")
    torch = pytest.importorskip("torch", reason="PyTorch is a training-only dependency")
    from app.models.gnn_torch import EdgeTravelTimeModel

    npm = NumpyEdgeTravelTimeModel.load(path)
    cfg = npm.config
    tm = EdgeTravelTimeModel(encoder=cfg["encoder"], hidden=cfg["hidden"],
                             layers=cfg["layers"], heads=cfg["heads"],
                             head_hidden=cfg["head_hidden"], dropout=cfg["dropout"])
    tm.load_state_dict({k: torch.tensor(v) for k, v in npm.p.items()
                        if not k.startswith("norm.")})
    tm.eval()

    src, dst = graph.adj_index
    uv = np.asarray([[graph.node_pos[graph.edges[i].u], graph.node_pos[graph.edges[i].v]]
                     for i in graph.static_edge_idx], dtype=np.int64)
    ef = graph.edge_features
    tf = np.tile(TimeContext(hour=9.25, dow=1).vector(), (ef.shape[0], 1))

    nx_ = (graph.node_features - npm.node_mean) / npm.node_std
    ex_ = (ef - npm.edge_mean) / npm.edge_std
    with torch.no_grad():
        t_out = tm(torch.tensor(nx_, dtype=torch.float32), torch.tensor(src),
                   torch.tensor(dst), torch.tensor(uv),
                   torch.tensor(ex_, dtype=torch.float32), torch.tensor(tf)).numpy()
    n_out = npm.forward_log(graph.node_features, src, dst, uv, ef, tf)

    assert np.abs(t_out - n_out).max() < 1e-4, \
        f"{enc}: serving path disagrees with the trained model"


@pytest.mark.parametrize("enc", ENCODERS)
def test_predictions_are_physically_sane(enc, graph):
    if not os.path.exists(weights(enc)):
        pytest.skip(f"{enc} not trained")
    p = NeuralEdgePredictor.load(weights(enc))
    pred = p.predict_static(graph, TimeContext(hour=9.0, dow=1))
    assert len(pred) == len(graph.static_edge_idx)
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)

    # Clamping is measured on the edges the model was actually trained on.
    # Walking transfers carry no travel-time observations by construction, so
    # the model extrapolates on them by definition -- that is exactly the case
    # the cold-start band exists for, and the serving path never uses those
    # predictions anyway (see test_walking_time_never_comes_from_the_model).
    base = np.asarray([max(graph.edges[i].base_min, 1e-3)
                       for i in graph.static_edge_idx], dtype=np.float64)
    observed = np.asarray([graph.edges[i].kind in ("road", "transit")
                           for i in graph.static_edge_idx])
    at_bound = ((pred <= base * COLD_START_LOW * 1.0001)
                | (pred >= base * COLD_START_HIGH * 0.9999))
    n_clamped = int(np.count_nonzero(at_bound & observed))
    assert n_clamped < 0.05 * int(observed.sum()), (
        f"{enc}: {n_clamped} of {int(observed.sum())} observed-edge predictions "
        f"needed clamping — the model is extrapolating badly")


def test_walking_time_never_comes_from_the_model(graph):
    """Pedestrians do not sit in traffic, and they do not wait on a GNN either.

    Walking edges -- road segments traversed on foot, and the transfer edges
    the model has no observations for -- must be priced analytically. This is
    what keeps a badly extrapolated transfer prediction out of a journey.
    """
    predictor = get_predictor()
    rg = RequestGraph(graph, (12.9185, 77.6880), (12.9346, 77.5353))
    minutes, diag = predict_edge_minutes(rg, graph, predictor,
                                         TimeContext(hour=9.0, dow=1))
    walk = [i for i, e in enumerate(rg.edges) if e.mode == "walk"]
    assert walk, "expected walking edges in a cross-city request graph"
    for i in walk:
        e = rg.edges[i]
        assert abs(float(minutes[i]) - (e.walk_min or e.base_min)) < 1e-4, \
            f"walking edge {e.edge_id} was priced by the model, not by pace"
    assert diag["walk_edges_analytic"] == len(walk)


def test_model_learned_that_peak_hours_are_slower(graph):
    if not os.path.exists(weights("graphsage")):
        pytest.skip("graphsage not trained")
    p = NeuralEdgePredictor.load(weights("graphsage"))
    road = np.asarray([graph.edges[i].kind == "road" for i in graph.static_edge_idx])
    base = np.asarray([graph.edges[i].base_min for i in graph.static_edge_idx])

    def ratio(ctx):
        return float(np.mean(p.predict_static(graph, ctx)[road] / base[road]))

    peak = ratio(TimeContext(hour=9.0, dow=1))
    midday = ratio(TimeContext(hour=13.0, dow=1))
    weekend = ratio(TimeContext(hour=9.0, dow=5))
    assert peak > midday, "weekday peak must be slower than the middle of the day"
    assert peak > weekend, "weekday peak must be slower than the same hour at the weekend"
    assert 0.8 < midday < 3.0, "congestion multipliers must stay physically plausible"


def test_registry_reports_availability_honestly():
    rows = registry()
    for r in rows:
        if r["key"] in WEIGHT_FILENAMES:
            assert r["available"] == os.path.exists(weights(r["key"]))
        assert r["available"] or r["reason"]


def test_service_falls_back_rather_than_crashing(monkeypatch):
    """A missing checkpoint must degrade to a baseline, not take the app down."""
    from app.config import get_settings
    import app.models.loader as loader

    # Clear the cache belonging to the module this test actually CALLS.
    # `tests/test_deployment.py` deletes every `app.*` module from sys.modules,
    # so a later import yields a fresh module object with a fresh lru_cache --
    # and the `get_predictor` bound at the top of this file is then a different
    # function from `loader.get_predictor`. Clearing the wrong one left a
    # warm cache and the fallback was never exercised.
    loader.get_predictor.cache_clear()
    monkeypatch.setattr(get_settings(), "travel_time_model", "nonexistent_model")
    monkeypatch.setattr(get_settings(), "model_fallback", "freeflow")
    p = loader.get_predictor()
    assert p.info.name == "freeflow"
    loader.get_predictor.cache_clear()
