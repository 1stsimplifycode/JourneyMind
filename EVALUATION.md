# EVALUATION

What was measured, on what, and what it does and does not support.

Reproduce with:

```bash
python scripts/evaluate.py --seeds 0 1 2 3      # prediction + recommendations
python scripts/evaluate.py --skip-training      # recommendations only (fast)
```

Machine-readable output: `models/evaluation.json`.

---

## 0. Read this first

Every number below is measured on the **bundled synthetic dataset**. No real
trip was ever logged for this repository.

The generator gives each node a latent "congestion susceptibility"; an edge's
delay depends on the **neighbourhood mean** of that latent, while node features
expose only a **noisy** per-node reading of it. That design makes neighbourhood
averaging useful *by construction*.

So: **these figures describe a process we wrote down, not Bengaluru.** They can
show that the pipeline works, that training and serving agree, and that the
comparison harness is sound. They cannot support any claim about real-world
travel-time prediction.

---

## 1. Method

**Temporal split, never random.** Weeks 1–5 train, week 6 validate, weeks 7–8
test. A random split would put 09:00 Tuesday in training and 09:15 Tuesday in
test; the model would have effectively seen the answer and the reported accuracy
would be fiction.

**Identical budget for every neural model.** Same features, same hidden width,
same optimiser, same learning-rate schedule, same 450-epoch cap, same patience
of 60. The only difference between GraphSAGE and the MLP is message passing.
That is the point of the ablation, and it is why the budget must be equal — an
under-trained GraphSAGE would fake a result in the MLP's favour just as surely
as a hand-tuned one would fake it the other way.

**Metrics.** MAE, RMSE and MAPE in minutes, plus error restricted to peak hours
(weekday 07:30–10:30 and 17:00–20:30), because being right at 09:00 is what
matters.

---

## 2. A. Travel-time prediction

Test split (weeks 7–8), neural results averaged over seeds 0–2. Measured on the
widened corridor bundle (223 nodes, 67 430 observations, 718 distinct edges).

| # | Model | MAE (min) | RMSE | MAPE | Peak MAE | MAE spread over seeds |
|---|---|---|---|---|---|---|
| 1 | Free-flow time | 0.505 | 0.812 | 18.3 % | 1.005 | — |
| 2 | Historical mean per edge/hour | 0.291 | 0.429 | 12.2 % | 0.359 | — |
| 3 | Gradient-boosted trees | 0.236 | 0.351 | 9.8 % | 0.312 | — |
| 4 | **MLP — identical features, graph removed** | **0.228** | **0.337** | **9.6 %** | **0.286** | 0.227 – 0.230 |
| 5 | GraphSAGE | 0.260 | 0.400 | 10.4 % | 0.342 | 0.238 – 0.273 |
| 6 | GAT | 0.238 | 0.356 | 9.9 % | 0.299 | 0.230 – 0.253 |

All six rows are computed by `scripts/evaluate.py` and written to
`models/evaluation.json`. Regenerate them on your machine rather than trusting
this table.

### The result, stated plainly

**No graph model beats the graph-free MLP on this dataset.** The MLP,
gradient-boosted trees and GAT land within 4 % of each other — a gap smaller
than the spread GAT shows across three initialisations, so it is not a finding
either way. GraphSAGE is behind all three.

**GraphSAGE is also unstable across initialisations**: 0.238 to 0.273 minutes
depending only on the seed, a ±7 % swing from initialisation alone. The MLP
moved by ±0.7 % over the same seeds. We report the mean and the range rather
than quoting the best seed.

**This result got *worse* for the graph models when the study area was
widened.** An earlier, smaller bundle (164 nodes, Purple/Green corridor only)
put GAT marginally ahead. Extending the corridor east to Sarjapur Road and west
to 100 Feet Ring Road added long, sparsely-connected arterial edges, and the
graph models lost their edge on them. A conclusion that flips when the map
changes was never a conclusion — which is the point of §0.

### Which model the service actually serves

`JM_MODEL` defaults to `gat`. That is **not** a claim that it is the most
accurate: the MLP measures better here. GAT is the default because it is the
graph model the project exists to exercise, it is the more accurate of the two
graph encoders, and it is stable enough to serve. Set `JM_MODEL=mlp`,
`JM_MODEL=gbt` or `JM_MODEL=historical` to serve any of the others — the
comparison is a runtime switch, not a rebuild, and `/api/models` reports which
one produced the numbers on screen.

**We therefore do not claim the GNN is more accurate.** Answering RQ1 on this
data: **no**.

### Why this is the expected outcome

The edge feature vector nearly identifies the edge — its class one-hot, length,
free-flow speed, scheduled time, lane count and endpoint congestion readings
together pin down which edge it is. With ~110 observations per edge in the
training weeks, *any* model with enough capacity can learn each edge's own
behaviour directly. There is very little left for message passing to add.

This is the same warning the project documentation gives about baseline 2:
*"historical mean per edge per hour bucket — this is the real bar, and it is
much stronger than people expect."* It turned out to be true of the neural
models too.

### The experiment that would actually settle it

**Hold out a spatial region.** Train on most of the map, test on a region the
model has never seen. Edges in a held-out region cannot be memorised from their
own history, so a model must infer their behaviour from their surroundings —
which is exactly what message passing does and what a flat model cannot do.

`scripts/evaluate.py --spatial` computes and reports the split (the eastern
slice of the study area by longitude) so the experiment is defined and
reproducible. Running it needs `scripts/train.py` to mask every
observation touching the held-out region, which it does not yet do. **This is
the single highest-value next step for both RQ1 and RQ5**, and it is listed as
not-run rather than quietly skipped.

---

## 3. B. Do the recommendations help?

50 scenarios: 10 origin/destination pairs × 5 departure hours (08:00, 09:00,
13:00, 18:00, 21:00), across all three presets.

| Metric | Value |
|---|---|
| Scenarios run | 50 |
| Feasible recommendation returned | 50 |
| **Constraint satisfaction rate** | **1.00** |
| Recommendations that were multi-modal | 24 % |
| Multi-modal beat *every* single-mode option on time | 7 |
| Mean regret (weighted-score units) | 0.0009 |
| Max regret | 0.0140 |
| Median latency | 246 ms |
| p95 latency | 348 ms |

**Constraint satisfaction is 1.00 by construction**, and it is worth being
precise about what that does and does not mean. It means the optimiser never
returns a journey that violates the budget or the deadline *as the model
predicts them* — the filter provably works. It does **not** mean the trip would
really have taken that long; that would need real outcomes to measure against.

**Regret** is measured against the best journey in each request's own feasible
pool, in the optimiser's weighted-score units. Near-zero regret means the
ranking is self-consistent — it is not evidence that the chosen journey is the
one a human would have picked. There is no ground truth for "the best journey",
which is exactly why the documentation asks for a 20–30 person user study. That
study has not been run.

**22 % multi-modal** is a real and useful number: in roughly one request in
five, the best answer was a journey no single-mode app would have offered. In 8
of those, the mixed-mode journey was faster than *every* single-mode option
available.

---

## 4. Model parity and learned structure

The deployed service runs the GNN through a NumPy forward pass so PyTorch is not
needed in the image. `tests/test_model_parity.py` asserts that the NumPy and
PyTorch implementations agree to within `1e-4` on identical inputs, for all
three architectures. Measured: **~6e-7**.

The same test file checks that predictions stay physically plausible and that
the model learned the structure it was supposed to:

| Condition (weekday, road edges) | Predicted congestion vs free flow |
|---|---|
| 06:00 | ×1.12 |
| 09:00 | ×1.93 |
| 13:00 | ×1.09 |
| 19:00 | ×2.00 |
| 22:00 | ×1.17 |
| Saturday 09:00 | ×1.22 |
| Weekday 09:00, raining | ×2.13 |

The model recovered morning and evening peaks, the weekend difference and the
rain effect from the data alone.

---

## 5. What is not evaluated

- **Baseline 6, map-app ETA.** No licensed reference was available, and using a
  commercial routing API as a training or evaluation label would violate its
  terms. Not run, not faked.
- **RQ2** (1-hop vs 2-hop vs 3-hop, over-smoothing). The harness supports it via
  `--layers`; not yet run systematically.
- **RQ3** (does better prediction produce better recommendations?). Needs the
  spatial hold-out first, so that there is a genuine accuracy gap to propagate.
- **RQ4** (learned preference weights). Needs real observed user choices.
- **A user study.** Not run. Without one, no claim is made that these
  recommendations are ones people would actually accept.

---

## The same experiment on a real network

Everything above is measured on the synthetic bundle. `bengaluru_osm` reruns it
over an OpenStreetMap extract of the same corridor — real roads, real classes,
real distances, real BMTC and Namma Metro stop order — with the observation
generator unchanged, so the geography is the only thing that differs.

Three seeds, same split discipline, same early stopping:

| | Synthetic MAE (min) | OSM topology MAE (min) |
|---|---|---|
| 1. Free-flow time | 0.505 | 0.432 |
| 2. Historical mean | 0.291 | 0.160 |
| 3. Gradient-boosted trees | 0.236 | **0.140** |
| 4. MLP (graph removed) | **0.228** ±0.002 | 0.148 ±0.002 |
| 5. GraphSAGE | 0.260 ±0.016 | 0.155 ±0.016 |
| 6. GAT | 0.238 ±0.002 | 0.145 ±0.002 |

**The two columns are not comparable to each other.** Different networks have
different edge-length distributions; the smaller numbers on the right mean the
arterial extract's edges are easier to predict, not that anything improved.
Only within-column comparisons mean anything.

### What the real network changed, and what it did not

On a single seed GAT (0.144) beat the MLP (0.150) and it was tempting to write
that the graph finally earned its place. Three seeds says otherwise: GAT 0.1453
against MLP 0.1475 is a gap of 0.0022, which is the size of each model's own
standard deviation. **They are indistinguishable.**

And the model that actually wins on the real topology is **gradient-boosted
trees at 0.140** — no graph, no neural network, no GPU.

So the conclusion is unchanged, and if anything it is firmer:

> On both the synthetic corridor and the real OpenStreetMap one, the graph
> encoders have **not** been shown to beat a well-tuned graph-free baseline.
> GAT is competitive and stable; GraphSAGE is neither; GBT wins outright on the
> real network.

The GNN stays in the product because it is the interesting *architecture* to
have wired end-to-end and because `JM_MODEL` makes the comparison a one-line
experiment — not because it is the most accurate thing here. It is not.

### Reproducing

```
python scripts/fetch_osm.py
python scripts/build_osm_bundle.py
JM_CITY=bengaluru_osm python scripts/evaluate.py \
    --seeds 0 1 2 --out models_osm/evaluation.json
```

`--out-dir` defaults to the directory of `--out`, so evaluating one city can no
longer overwrite another city's served weights. It did, twice, before that was
fixed.
