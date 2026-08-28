# ARCHITECTURE

How the system is put together, what each model is for, and why.

---

## 1. The one idea

Everything below exists to compute one number honestly:

> **Expected cost** — what a trip will actually cost you, once the probability
> that the booking falls through, the time you lose finding out, and the price
> of the replacement ride are all priced in.

Every fare-comparison product stops at the advertised fare. The stages up to
`providers` in the pipeline below are table stakes. The `lifecycle` stage is
the product.

---

## 2. The pipeline

```
  Journey request  (from, to, priority, optional budget and deadline)
        |
  [1] POLICY / AUTH .................. rate limits, RBAC on enterprise routes
        |
  [2] DATA LAYER ..................... bundled study-area graph, fares, service hours
        |
  [3] MULTIMODAL GRAPH ............... road + transit + transfer + ride + access edges
        |
  [4] TRAVEL-TIME MODEL .............. GNN (GraphSAGE / GAT) or a baseline, per edge, per hour
        |
  [5] ROUTING ........................ Yen's k-shortest, time-dependent, 5 weightings
        |                              -> one single-mode reference journey per mode
        |
  [6] PROVIDERS ...................... nine adapters behind one interface
        |                              fare, ETA, availability, route, cancellation
        |
  [7] RELIABILITY .................... P(match), P(accept), P(cancel)  <- calibrated
        |
  [8] LIFECYCLE ...................... absorbing Markov chain -> EXPECTED COST
        |                              exact distribution, p10/p50/p90, P(success)
        |
  [9] CONSTRAINTS .................... filter on EXPECTED cost and time, not advertised
        |
 [10] RANKING ....................... cheapest | fastest | reliable | balanced
        |
 [11] EXPLANATION ................... why this one, and why not the cheaper one
        |
 [12] AUDIT ......................... decision + model versions + confidence, append-only
        |
  Recommendation
        |
 [13] BOOK NOW ...................... a real attempt, sampled from the SAME
        |                             probabilities that priced the option
 [14] RETRY ......................... at an escalated fare, up to a budget
        |
 [15] REVEAL ........................ advertised vs expected, and why
```

### The demonstration loop

Stages 13-15 are what make the product legible. A prediction shown before the
rider has felt the thing being predicted is a dashboard; the same prediction
shown *after* is an explanation.

    BOOK NOW  ->  the server samples one trajectory through the state machine
                  and returns narrated steps with dwell times. The interface
                  animates a result; it never decides one.

    TRY AGAIN ->  a fresh attempt at fare x (1 + surge_per_retry)^(n-1), the
                  same escalation the expected-cost solver assumes, so the
                  lived sequence and the predicted average cannot disagree.

    REVEAL    ->  every probability quoted was computed BEFORE the booking ran.
                  Probabilities are frozen at session start, so the explanation
                  describes the booking that actually failed rather than a
                  fresh calculation. A test asserts this.

    ESCALATE  ->  once the retry budget is spent (default 4 attempts) the rider
                  is not shown another button. The projection runs on the
                  RIDER'S clock -- departure plus the minutes already lost, not
                  the server's wall clock -- and answers three questions: will
                  they miss what they were travelling to, what should they
                  switch to, and does anyone need to be told.

                  The switch is the cheapest option that still arrives in time,
                  not the fastest. Fastest-wins recommended a Rs 543 cab over a
                  Rs 113 option eight minutes slower; being on time is the
                  constraint, cost is the objective. When nothing arrives in
                  time, the fastest is all that is left.

    NOTIFY    ->  offered, never automatic, and composed rather than sent. The
                  message and an anonymous incident record are returned and
                  written to the audit log; no mail or chat transport is wired
                  in, and reporting a message as sent when it was not would be
                  a false claim about an action outside this system.

Demo mode fixes the random seed so a live demonstration is reproducible. **It
fixes the dice, not the outcome** -- the probabilities remain the model's, and
two tests assert both halves: seeded runs are identical, unseeded runs vary.

Stages 2–5 are the original JourneyMind engine and are unchanged. Stages 6–8
and 12 are the mobility-intelligence layer; stages 13–15 are the demonstration
loop, all built *around* the original engine rather than replacing it.

---

## 3. Module map

```
backend/app/
  config.py              env-driven settings; nothing secret, starts with all unset
  main.py                app factory, CORS, rate limiting, static serving
  security.py            API keys, roles, audit log            [NEW]
  schemas.py             request/response contracts

  data/                  study-area bundle: nodes, edges, routes, fares, service hours
  graph/                 the multimodal graph + feature encoding
  models/                travel-time models: freeflow, historical, gbt, mlp, graphsage, gat
  routing/               k-shortest search, time-dependent costs, journey assembly
  optimisation/          constraint filter, Pareto frontier, weighted scoring

  providers/             the mobility-provider abstraction            [NEW]
    base.py                MobilityProvider ABC + quote composition
    simulated.py           6 modes across 5 providers: bike taxi (Rapido),
                           auto (metered / Namma Yatri), cab, metro, bus.
                           A MODE is a vehicle; a PROVIDER is who you book it
                           through, which is why one auto has two providers.
  reliability/           P(match) / P(accept) / P(cancel)             [NEW]
    features.py            shared encoding: train and serve cannot diverge
    model.py               NumPy serving + honest fallback
  lifecycle/             the booking state machine                    [NEW]
    states.py              legal transitions, trajectory simulation
    expected_cost.py       the exact Markov solve
  booking/               live BOOK NOW sessions                       [NEW]
    session.py             seeded sessions, narrated attempts, retry budget
  enterprise/            population-level analytics                   [NEW]
    store.py               columnar booking table (see PERFORMANCE below)
    analytics.py           spend, failure cost, provider scorecards, insights
  services/
    engine.py              the journey pipeline
    compare.py             the expected-cost comparison                [NEW]
    explain.py             deterministic explanations
    clock.py               study-area local time
  api/
    routes.py              journey planning, city, places, models
    mobility.py            compare, providers, lifecycle, enterprise    [NEW]
    booking.py             book, retry, reveal, insights                [NEW]
```

---

## 4. Where each model is used, and why

The brief asked for a GNN only where it earns its place. Here is the reasoning
for each prediction, including the two places a GNN was rejected.

### 4.1 Travel time per edge — **GNN** ✔

**Model:** GraphSAGE / GAT / MLP, trained in PyTorch, exported to `.npz`,
served in NumPy.

**Why a graph model:** congestion is not a property of a road, it is a property
of a *neighbourhood*. A jammed arterial slows the streets feeding it. A flat
model sees one road's own history and has no way to represent "my neighbour is
blocked" without hand-crafting a column per neighbour, then per
neighbour-of-neighbour. Message passing gets that for free because the graph
*is* the neighbourhood structure.

**And the honest result:** on the bundled data the GNN **does not beat** the
graph-free MLP — MLP 0.228 MAE, GAT 0.238, GraphSAGE 0.260. That is reported in
`EVALUATION.md` and in the README rather than buried. The ablation was run to
be answered either way.

### 4.2 Cancellation, rejection, no-supply — **calibrated logistic regression** ✔ (not a GNN)

**Why not a GNN.** Cancellation is not neighbourhood-shaped. It is driven by
properties of the individual request: how short the fare is, how far the pickup
is, what hour it is, whether it is raining. The one genuinely spatial input —
neighbourhood congestion — is *already computed by the graph* and arrives as a
scalar feature. Message passing would add parameters, training time and opacity
to buy nothing measurable.

**Why logistic regression specifically.** Four reasons, in order:

1. **The output is multiplied into money.** It must be a calibrated
   probability, not a score. Log loss optimises exactly that.
2. **The coefficients are the explanation.** The product promises to say *why*
   an option was rejected. A weight on `short_trip_penalty` is that sentence.
3. **It serves without scikit-learn**, keeping the deployed image small.
4. **It is not assumed to win.** It is run against a global rate, a
   per-provider rate, a provider×hour lookup table and gradient-boosted trees.

**Measured, on the simulated bundle** (`models/reliability_evaluation.json`):

| Head | Baseline (lookup) Brier | Logistic Brier | AUC | ECE |
|---|---|---|---|---|
| match | 0.134 | **0.130** | 0.749 | 0.006 |
| accept | 0.148 | **0.143** | 0.657 | 0.008 |
| cancel | 0.129 | **0.126** | 0.659 | 0.006 |

GBT edges logistic on Brier for two heads by ≤0.0005 — a tie — while logistic
is better calibrated on both. **The selection rule is stated before the numbers
are read: Brier ranks, ECE decides**, because the output is multiplied into a
rupee figure.

### 4.3 Expected cost — **exact Markov solve** ✔ (no ML at all)

Not learned, because it does not need to be. The lifecycle is small enough to
enumerate exactly, and an exact answer beats an approximated one. Learning it
would replace a closed form with a black box and lose the outcome distribution.

### 4.4 Demand forecasting — **not built**

Would be a time-series problem, not a graph one. Named in the roadmap rather
than implemented, because there is no real demand data to fit it to.

### 4.5 Driver supply spillover — **not built, and this is the honest GNN case**

If there were real driver-location telemetry, supply in a zone would genuinely
depend on supply in neighbouring zones — drivers move — and *that* would be a
defensible second GNN over a zone graph. There is no such data, so it is not
built. This is recorded because it is the one place a GNN could later be added
for a reason rather than for decoration.

---

## 5. How cancellation and rebooking are modelled

### The state machine (`lifecycle/states.py`)

```
SEARCHING -> REQUESTED -> DRIVER_MATCHED -> DRIVER_ACCEPTED -> RIDE_STARTED -> RIDE_COMPLETED
                |               |                  |
                v               v                  v
     NO_DRIVER_AVAILABLE  DRIVER_REJECTED   DRIVER_CANCELLED
                |               |                  |
                +---------------+------------------+
                                |
                          REBOOKING -> REQUESTED   (or ABANDONED)
```

Transitions are **enforced**, not documented: an event stream claiming
`REQUESTED -> RIDE_COMPLETED` raises `IllegalTransition`, so a corrupt stream
cannot silently poison the analytics downstream.

The three failure edges are kept separate because they cost different amounts:

| Failure | Cost to the rider |
|---|---|
| `NO_DRIVER_AVAILABLE` | search timeout — cheapest, you learn quickly |
| `DRIVER_REJECTED` | seconds of matching |
| `DRIVER_CANCELLED` | **expensive** — you waited most of a pickup for nothing |

A single "cancellation rate" percentage throws that distinction away.

### The solve (`lifecycle/expected_cost.py`)

One attempt succeeds with `q = P(match) × P(accept) × (1 − P(cancel))`, so the
outcome space is short and exact:

| Outcome | Probability | You pay |
|---|---|---|
| ride on attempt *k* | `(1−q)^(k−1) · q` | `fare × (1+surge)^(k−1)` |
| gave up after *K* | `(1−q)^K` | the fallback |

From that enumeration: expected cost, expected time, expected wasted minutes,
P(success), and exact p10/p50/p90 — not a point estimate with a confidence
adjective attached.

**The fallback term matters.** If every attempt fails you do not teleport home;
you take the most reliable alternative available. Omitting that costs failure at
zero and makes unreliable options look cheap.

**And it is flagged when it dominates.** If P(abandon) ≥ 10% the expectation
blends two different journeys, `is_blended` is set, and the UI says
*"below the fare only because 53% of the time you end up on the Bus"* — because
otherwise an option that fails into a cheap bus reads as a discount.

**Verified three ways** (`tests/test_mobility.py`): against the closed form,
against a 20,000-run Monte Carlo of the same state machine, and for the
degenerate case where a scheduled service must return exactly its fare.

---

## 6. Data architecture

```
  OSM + Namma Metro facts + published fare tables      REAL / PUBLISHED
        |
  scripts/generate_dataset.py                          -> data/city/
        |   road graph, bus stops, headways, travel-time observations   SIMULATED
        |
  scripts/train.py            -> models/{graphsage,gat,mlp}_model.npz   PREDICTED
        |
  scripts/generate_mobility_data.py                    -> data/mobility/
        |   60,000 booking events with campus / team / cost centre      SIMULATED
        |
  scripts/train_reliability.py -> models/reliability_model.npz          PREDICTED
        |
  serving: NumPy only. No torch, no sklearn in the image.
```

### The three-way labelling

Every number that leaves the API carries its class, and the classes are never
blended:

| Class | Meaning | Example |
|---|---|---|
| `published` | Transcribed from an operator's table | Metro fare |
| `simulated` | From a documented generator in this repo | Ride-hailing availability |
| `predicted` | Output of a model in this repo | Travel time, P(cancel) |

**No adapter in this repository contacts a live commercial ride-hailing API,
because no such API is open.** The ride adapters are simulated, they say so on
every response, and the interface is shaped so a real adapter replaces one
without any other code changing.

---

## 7. API

Base URL is the deployment root; the same service serves the UI.

### Open

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness — reports what is actually loaded |
| `GET` | `/api/city` | Study area, bounds, transit lines, live clock, service hours |
| `GET` | `/api/places` | Named places for the pickers |
| `GET` | `/api/models` | All six travel-time models and availability |
| `GET` | `/api/providers` | The provider registry and the reliability model version |
| `GET` | `/api/lifecycle` | The booking state machine, as data |
| `GET` | `/api/demo` | The bundled scenario, computed live |
| `POST` | `/api/recommend` | Multi-modal journey planning |
| `POST` | **`/api/compare`** | **Expected cost across every provider** |
| `POST` | **`/api/book`** | **Press BOOK NOW; runs attempt 1** |
| `POST` | `/api/book/{id}/retry` | TRY AGAIN, at the escalated fare |
| `GET` | `/api/book/{id}` | The session as it stands |
| `GET` | `/api/book/{id}/reveal` | What actually happened, and what it cost |
| `GET` | `/api/book/{id}/escalation` | Arrival risk once the retry budget is spent, and what to switch to |
| `POST` | `/api/book/{id}/notify` | Compose the manager notification and open an incident — composed, never transmitted |
| `GET` | `/api/insights` | Supply-demand relationships behind all of it |

### Gated — requires `X-API-Key` with role `analyst` or above

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/enterprise/facets` | Filter options |
| `GET` | `/api/enterprise/overview` | KPIs, breakdowns, provider scorecards, insights |
| `GET` | `/api/enterprise/audit` | Recorded AI decisions |

### `POST /api/compare`

```bash
curl -s localhost:8000/api/compare -H 'Content-Type: application/json' -d '{
  "origin": "Wipro Campus",
  "destination": "PES University",
  "priority": "balanced",
  "budget": 300,
  "max_time": 90
}'
```

`origin` / `destination` accept a `place_id`, a free-text name, or
`{"lat": …, "lon": …}`. `budget` and `max_time` are **optional** — comparing is
something you do before you know what you can afford. `priority` is
`cheapest` | `fastest` | `reliable` | `balanced`; **`cheapest` ranks on
expected cost, not the advertised fare.**

Each option returns:

```jsonc
{
  "provider_id": "bike_taxi",
  "service_class": "hailed",
  "data_class": "simulated",
  "fare":        { "amount": 67, "display": "₹53–₹80", "surge_multiplier": 1.16 },
  "reliability": { "p_match": 0.86, "p_accept": 0.80, "p_cancel": 0.20,
                   "basis": "calibrated logistic heads, 42,702 simulated bookings" },
  "expected":    { "expected_cost": 68.4, "surcharge": 1.4,
                   "expected_minutes": 27.6, "expected_wasted_min": 1.8,
                   "p_success": 0.91, "expected_attempts": 1.12,
                   "cost_p10": 67, "cost_p90": 78,
                   "is_blended": false,
                   "outcomes": [ { "probability": 0.71, "cost": 67,
                                   "label": "ride on the first request" } ] }
}
```

### Enterprise

```bash
curl -s "localhost:8000/api/enterprise/overview?campus=cmp_sarjapur&hour_from=17&hour_to=21" \
     -H "X-API-Key: demo-analyst-key"
```

Filters: `campus`, `provider`, `employee_group`, `mode`, `date_from`,
`date_to`, `hour_from`, `hour_to`, `minute_cost`.

### Errors

`422` with a human sentence and a stable `code` for anything the caller can
fix; `429` when rate-limited; `401` / `403` / `503` on the enterprise routes.
Never a stack trace.

---

## 8. Security

| Control | Implementation |
|---|---|
| **Authentication** | API keys, SHA-256 digested, compared with `hmac.compare_digest` |
| **Authorisation** | Ordered roles: `rider` < `analyst` < `admin` |
| **Fail closed** | No keys + `DEMO_MODE=false` ⇒ enterprise routes refuse **every** request. Never falls back to open |
| **Demo mode** | One clearly-named demo key, announced in logs and in every response it authorises, so a demo cannot be mistaken for a configured deployment |
| **Input validation** | Typed Pydantic models; bounds on every numeric field |
| **Rate limiting** | Per-IP, per-minute, on the compute-heavy routes |
| **Algorithmic DoS** | Search expansion caps — candidate generation approximates an NP-hard problem |
| **Audit** | Append-only decision log with model versions and confidence |
| **Privacy** | No employee identifier anywhere in the pipeline. Cohorts below 25 trips are **suppressed, not rounded** |
| **Provenance** | Every number carries its data class |

**Not implemented, and named as such:** SSO, tenant isolation, encryption at
rest, secret rotation, WAF. This is API-key auth suitable for a pilot behind a
gateway.

---

## 9. Deployment

Single service: FastAPI serves the JSON API and the built React bundle from one
origin. No cross-origin configuration, one thing to deploy.

### Render

`render.yaml` is committed and complete — Docker runtime, health check on
`/health`, `JM_API_KEYS` generated as a secret at first deploy. Connect the
repository and Render reads it; no dashboard configuration is required.

```bash
docker build -t journeymind .      # local equivalent
docker run -p 8000:8000 journeymind
```

### Why the image is small

Two-stage build: Node builds the bundle, a slim Python runtime serves it.
**PyTorch and scikit-learn are not in the runtime image.** Models are trained
offline and exported to `.npz`, then replayed in NumPy — which keeps the image
inside a free-tier instance and means a training-only CVE cannot reach
production.

That is a fragile property: one convenient `import pandas` in a serving module
and the deploy dies on boot while every local test still passes. So
`tests/test_deployment.py` **blocks those imports at the meta-path and drives
every endpoint through the result.** It also asserts that the trained model
still loads without torch — if the NumPy path ever breaks, the test fails
rather than the deploy.

### One worker, deliberately

The container runs `--workers 1`, and a test enforces it. Booking sessions and
the audit log live in process memory, so with two workers a rider could press
TRY AGAIN and land on a process that has never heard of their booking. **This
is a correctness constraint, not a performance choice**, and it is the first
thing to change if this ever needs to scale horizontally: move sessions to
Redis, then raise the worker count.

### What ships in the image

| Path | Size | Why the runtime needs it |
|---|---|---|
| `data/city/…` | 5 MB | The graph, fares, routes, service hours |
| `data/city/…/travel_times.csv` | 4 MB | The historical-mean fallback model |
| `data/mobility/bookings.csv` | 10 MB | Enterprise analytics and `/api/insights` |
| `models/*.npz` | 190 KB | All six travel-time models plus the reliability heads |

### Cold-start behaviour

Everything lazy is warmed at boot instead: the graph, the travel-time model,
the reliability heads and the booking table. Boot takes about 1.7 s, and the
first enterprise request then costs **63 ms instead of 2.3 s**. On a free
instance that spins down when idle, the boot cost is paid once on wake rather
than by whoever clicks first.

Measured, on 60,000 bookings:

| | Before | After |
|---|---|---|
| Booking history in memory | 89 MB | **3.3 MB** |
| First enterprise request | 2,314 ms | **63 ms** |
| Six filter clicks | 3,498 ms | **58 ms** |

The store is columnar NumPy rather than a list of dicts (`enterprise/store.py`).
Filtering is a boolean mask; aggregation is a sum over it.

### Free-tier caveats, stated plainly

- **Instances sleep when idle.** The first request after a sleep pays the boot.
- **Booking sessions are in process memory.** A restart loses them; a rider
  mid-booking sees "that booking has expired" and starts again. Acceptable for
  a demo, not for production.
- **The audit log is a ring buffer** unless `JM_AUDIT_LOG` points at a file, and
  a free instance has no persistent disk.
- **Demo auth is on** while `DEMO_MODE=true`. Set `DEMO_MODE=false` and
  `JM_API_KEYS` for anything real; the enterprise routes then fail closed.

---

## 10. Known limitations

1. **No live provider data.** Ride-hailing fares, availability and cancellation
   are simulated. Labelled everywhere.
2. **The reliability model describes a generator**, not any real operator.
3. **One city, one corridor.** Nothing claims to generalise.
4. **No booking or payment.** The system recommends; it does not reserve or pay.
5. **The GNN is not shown to be better** than a graph-free MLP on this data.
6. **Surge and retry escalation are assumptions**, exposed as parameters.
7. **Cycling is derived from the walking path** at cycling speed — no cycle
   network is modelled.
8. **Single tenant, no SSO, no encryption at rest.**
9. **The agentic layer is specified, not built.**
