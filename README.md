# JourneyMind

**The cheapest fare is not the cheapest journey.**

A mobility app you can actually book a ride in — wrapped around an engine that
predicts whether that booking will survive contact with reality, and what it
will really cost when it doesn't.

---

## Try it in three minutes

```bash
run-demo.bat                 # Windows: sets everything up, opens the browser
```

The app opens on **Book a ride**. Nothing on that screen mentions a model.

### 1 — It looks like a normal ride app

Default trip: **Wipro Campus, Doddakannelli → PES University, RR Campus**, 09:00.

```
   Bus        · BMTC             ₹153   90 min   Moderate   [ START TRIP ]
   Metro      · Namma Metro      ₹167   56 min   Moderate   [ START TRIP ]
   Bike taxi  · Rapido           ₹215   61 min   High       [ BOOK NOW ]
   Auto       · Namma Yatri      ₹320   75 min   High       [ BOOK NOW ]
   Auto       · Metered auto     ₹371   75 min   High       [ BOOK NOW ]
   Cab        · Cab aggregator   ₹482   70 min   High       [ BOOK NOW ]

   Or travel in stages
   Bike taxi → Bus → Metro → Bike taxi   ₹106–₹141   82 min   [ VIEW JOURNEY ]
```

Every price is door to door: the Metro row includes the bike taxi to the
station and the one at the far end, because a station is not a doorstep.

The left column is the VEHICLE and the right is the OPERATOR. One auto, two
providers — that separation is what stops the engine becoming "recommend
Rapido" instead of "recommend a bike taxi".

You have a 10:00 meeting, so the 90-minute bus is out. Press **Book now** on
the bike taxi.

### 2 — The booking is real, and it can fail

```
   Searching for a driver…
   ✗ No driver available            [ TRY AGAIN ]
   Searching for a driver…   ₹35    ← the fare moved
   ✓ Driver found · Ganesh M.
   ✓ Journey completed — you paid ₹35
```

The outcome is **sampled from the same probabilities that priced the option**,
not from scripted UI text. Turn on Demo mode and the sequence is reproducible
on stage; turn it off and it varies exactly as the model says it should.

### 3 — Only now, the reveal

|  | Bike taxi *(you chose)* | Auto *(the engine picked)* |
|---|---|---|
| Advertised fare | **₹29** | ₹36 |
| Booking succeeds | 21% | 82% |
| Cancelled after accepting | 46% | 21% |
| Expected attempts | 3.56 | 2.07 |
| **Expected cost** | **₹45** | **₹43** |

> *"₹29 was the advertised fare; ₹45 is what this option is expected to cost
> once the chance of it falling through is priced in — 54% more. Like for like,
> Auto advertises ₹36 — more than ₹29 — but completes 82% of the time, so its
> expected cost is ₹43."*

**The cheapest sticker price is the most expensive journey.** That crossover is
the product, and it is asserted by a test so it cannot quietly stop being true.

### 4 — Then the rest unlocks

**Insights** (why this is a market phenomenon, not a quirk) → **Intelligence**
(expected cost across every option) → **Journey planner** (the full multi-modal
engine) → **Enterprise** (the same problem across 60,000 bookings).

---

## What is real and what is not

No open API publishes ride-hailing supply or cancellation data, so **every
hailed-vehicle quote here is modelled, not live**. Metro and bus fares are
transcribed from published tables; travel times are model predictions; routes
come from a real graph of the study corridor. Every number carries a
machine-readable `data_class` on the API.

**On causation:** the engine never claims a low fare *causes* a cancellation.
Fare, demand, supply, acceptance and cancellation all respond to the same
market conditions. What the data shows — and what the model uses — is that
**short fares are declined more often when demand is high**. That is an
association, strong enough to predict, and the UI says so.

---

## Documentation

| Document | Contents |
|---|---|
| **`ARCHITECTURE.md`** | Pipeline, module map, **why each model is used**, API reference, deployment |
| **`SOURCES.md`** | Every data element and its provenance |
| **`EVALUATION.md`** | Measured performance — including where the GNN loses |
| **`V2_TRUST_SECURITY_GOVERNANCE.md`** | Trust, verification, security, agentic design, governance |
| **`OPEN_SOURCE_ATTRIBUTIONS.md`** | What influenced what, under which licence |
| **`WIPRO_PRODUCT_PITCH.md`** | Commercial positioning and a worked ROI example |

<details><summary>Manual setup</summary>

```bash
python -m venv .venv && .venv/Scripts/pip install -r backend/requirements.txt
python scripts/generate_dataset.py            # study-area graph
python scripts/generate_mobility_data.py      # booking history
python scripts/train_reliability.py           # cancellation models
cd frontend && npm install && npm run build && cd ..
cd backend && python -m uvicorn app.main:app --port 8000
```

Enterprise dashboard demo key: `demo-analyst-key`. Run the tests with
`python -m pytest tests -q` (129 tests).
</details>

---

## 1. Overview

You have ₹100 in your pocket and 30 minutes before class. Four apps will each
tell you what *their* ride costs. None of them will tell you that walking four
minutes to the metro, riding it for seventeen, and taking a bike-taxi for the
last mile gets you there for ₹50 with two minutes to spare.

JourneyMind answers the question the apps don't: **what is the best complete
journey for me, right now, with the money and the time I actually have?**

Ask it once, and it returns one recommendation plus two alternatives, each with
a cost, a time, a transfer count, and a plain-English reason.

## 2. The problem

Today's tools answer *"which app is cheapest?"* Nobody answers *"what is the
best complete journey?"*

| What exists | What it does | What it misses |
|---|---|---|
| Ride apps | Price one ride, one mode | Never suggest mixing with bus or metro |
| Map apps | Show routes, some transit | Don't price ride-hailing; don't respect a budget |
| Aggregators | Several ride prices side by side | Still one mode; still "which app is cheaper" |
| Transit apps | Metro and bus timings | Ignore the first and last kilometre |

A single-mode system says *"Rapido is ₹20 cheaper."* JourneyMind says *"Take the
metro for ₹25, then a Rapido for about ₹25 — ₹45 to ₹55 in total, about 28
minutes, and it fits both your limits."*

## 3. The solution

```
User request
     ↓
Data layer            bundled study-area graph, fares, observations
     ↓
Multimodal graph      road + transit + transfer + ride + access edges
     ↓
Travel-time model     GraphSAGE / GAT / MLP — predicted minutes per edge
     ↓
Candidate journeys    Yen's k-shortest under five time/money weightings
     ↓
Constraint filter     drop anything over budget or over the deadline
     ↓
Pareto frontier       drop anything both dearer and slower than something else
     ↓
Personalised ranking  normalised weighted score — cheapest / balanced / fastest
     ↓
Explanation           deterministic, generated from the journey's own attributes
     ↓
Web interface         map, route timeline, cost, time, transfers, and the reason
```

## 4. Why multi-modal routing

Because "Metro + Rapido" is not a clever special case somebody hard-coded. Once
the city is a graph in which walking, metro, bus and hailed rides are all just
edges, **every mixed-mode journey is already in there, waiting to be found**. The
search doesn't know what "multi-modal" means; it only knows about paths.

That is the whole architectural argument, and it is why the graph is built the
way it is.

## 5. Why a GNN

Places are not independent. A jam on one road spills onto the next. A delayed
metro triples the queue at the bus stop outside it. Rain makes everyone switch
to autos and auto waiting times rise across the whole area.

A model that looks at one road at a time cannot see any of that — "my
neighbour is jammed" isn't a column you can put in a flat table without
hand-crafting it, and then hand-crafting the neighbour's neighbour, for every
road in the city. A GNN gets it for free, because the graph *is* the
neighbourhood structure.

**That is the hypothesis, and this project treats it as a hypothesis.** See
§16 and `EVALUATION.md` — including the result where the graph did **not** help.

## 6. Architecture

```
backend/app/
├── main.py                 FastAPI app; serves the API and the built UI
├── config.py               environment-driven settings, all with defaults
├── schemas.py              request/response validation
├── api/
│   ├── routes.py           the six endpoints
│   └── serialise.py        engine objects → API payloads, provenance attached
├── data/
│   ├── provider.py         TransportDataProvider — the swap point for real data
│   ├── static_provider.py  the bundled study-area implementation
│   └── geo.py              haversine and detour factors
├── graph/
│   ├── builder.py          MultimodalGraph + per-request RequestGraph
│   └── features.py         node / edge / time-context encoders
├── models/
│   ├── base.py             TravelTimePredictor interface
│   ├── baselines.py        free-flow, historical mean, gradient-boosted trees
│   ├── gnn_torch.py        GraphSAGE, GAT, MLP ablation — training
│   ├── gnn_numpy.py        the same forward pass in NumPy — serving
│   ├── fares.py            published vs estimated fare rules
│   └── loader.py           model registry, fallback, availability reporting
├── routing/
│   ├── costs.py            time-dependent cost planes
│   ├── index.py            integer-indexed graph for the search hot loop
│   ├── kshortest.py        Yen's k-shortest + single-mode reference journeys
│   └── journey.py          paths → priced, readable journeys
├── optimisation/
│   ├── constraints.py      budget and deadline filtering
│   ├── pareto.py           dominated-journey removal
│   └── scoring.py          normalised weighted ranking
└── services/
    ├── engine.py           the pipeline, with a full audit trace
    └── explain.py          deterministic natural-language explanations
```

**Layer discipline:** nothing above the data layer knows what a file is.
Everything talks to `TransportDataProvider`.

## 7. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Frontend | React 18 + Vite | Small, fast, no framework ceremony |
| Map | Leaflet + OpenStreetMap | Open, free, no API key, no paid dependency |
| Backend | Python 3.12 + FastAPI | Typed request validation, automatic OpenAPI |
| Graph & search | NumPy + hand-written Yen's | The search needs state-augmented Dijkstra, which no off-the-shelf routine provides |
| ML training | PyTorch (CPU) | GraphSAGE/GAT written directly — no PyTorch Geometric dependency to install |
| ML serving | **NumPy only** | See below |
| Deployment | Docker → Render, one web service | Fewest moving parts |

**Why NumPy at serving time.** PyTorch would add hundreds of megabytes and a
large resident footprint to run a two-layer network over a 223-node graph. So the
model trains offline in PyTorch, its weights export to a ~50 KB `.npz`, and
`gnn_numpy.py` replays the identical arithmetic. This is real inference on real
learned weights — and `tests/test_model_parity.py` asserts the two
implementations agree to within 1e-4, so it is checked rather than claimed.

## 8. Data strategy

The study area is **one bounded corridor** of Bengaluru — the Namma Metro
Purple, Green and Yellow lines and the streets around them, stretching from
Doddakannelli / Sarjapur Road in the east to 100 Feet Ring Road (Banashankari
3rd Stage) in the west. 223 nodes, 526 road links, 87 transit links over 10
routes, ~67,000 travel-time observations.

**Read `SOURCES.md` before believing any number.** In summary:

- **Real:** metro station names, lines and approximate positions; approximate
  first/last service times; metro, bus and auto fare structures (transcribed).
- **Synthetic:** road junctions, bus stops, bus routes, headways, and **every
  travel-time observation**.
- **Estimated:** all ride-hailing fares, via a transparent
  `base + per-km + per-min` model with an uncertainty band.
- **Not modelled at all:** surge pricing, live traffic, live vehicle positions,
  published timetables (headways are averaged, not scheduled).

The application labels this as **"Demo / estimated data"**, and tags each
individual number `published`, `estimated` or `predicted`.

Regenerate the dataset deterministically:

```bash
python scripts/generate_dataset.py --seed 20250827
```

## 9. How routing works

**The graph has five kinds of edge.**

| Kind | Meaning |
|---|---|
| `road` | Walking along a street segment |
| `transit` | One stop to the next on a metro or bus route |
| `transfer` | Walking between two nearby transit nodes |
| `ride` | A hailed vehicle between two hubs (added per request) |
| `access` | Walking from your actual coordinates to the network |

Rides are hailed **hub to hub**, not segment by segment, because nobody books a
bike-taxi for one block and another for the next.

**The search is Yen's k-shortest paths**, with three departures from the textbook:

1. **State, not just node.** The search state is `(node, route you are sitting
   on, boardings used)`. Waiting is charged when you **board**, not at every
   stop — otherwise a metro trip would pay a fresh wait at every station. The
   boarding count is in the state because a transfer cap is not a property of a
   node.
2. **Time-dependent weights.** An edge entered 20 minutes in is priced from the
   15–30 minute bucket, not from the departure-time prediction. The model runs
   once per bucket (0/15/30/45/60/90 min). This is piecewise-constant in elapsed
   time — an approximation, and described as one.
3. **Five weightings, not one.** k-shortest by time returns k variations on the
   fastest trip; a Pareto frontier built from those collapses to a point. So the
   search runs under a family of time/money blends and pools the results, plus
   one **single-mode reference journey per mode** (metro-only, bus-only,
   Rapido-only, auto-only, cab-only, walk-only). Those references do double duty:
   they guarantee the user sees the options they'd have compared by hand, and
   they are baseline 5 from the project documentation.

**This is an approximation and the code says so.** "Cheapest path that also fits
a time limit" is the Resource Constrained Shortest Path Problem, which is
NP-hard.

### 9.1 Answering for *now*

The recommendation is for a moment, and the moment is the current one by
default. Three things follow from that, and all three are visible in the
interface.

**The clock is the study area's, not the browser's.** Every time-of-day
judgement the service makes — the congestion peak shape, headways, whether the
last train has gone — is about local wall-clock time in Bengaluru. `departure_time`
defaults to `now` on that clock; an offset-aware timestamp from a browser is
converted to it. Reading `09:00 IST` as `09:00 UTC` would have priced the
morning peak at 03:30, so the conversion is explicit
(`backend/app/services/clock.py`).

**Routes have service hours.** Each route carries an approximate first and last
service time. Outside them, boarding does not silently become impossible —
it costs the **real wait until the first departure**, which the constraint
filter then almost always rejects. Ask for a journey at 01:00 and you get
bike-taxis, plus a caveat saying the network is shut. Ask at 09:00 and the
metro wins the middle of the trip. Same pipeline, different answer, because it
is a different time.

**"Live" means the clock, not a feed.** With live mode on, the answer is
recomputed every minute against the current time and stamped with when it was
produced, so a result that has been on screen for ten minutes cannot pass
itself off as current. There is no live traffic source, no GTFS-Realtime feed
and no vehicle-availability data in this project, and the interface never
implies otherwise.

## 10. How optimisation works

Three stages, in order.

**Stage 1 — remove the impossible.** Over budget: deleted. Over the deadline:
deleted. No weighting can rescue a journey you cannot pay for.

**Stage 2 — remove the dominated.** If journey B is *both* cheaper and faster
than journey A, then A is removed regardless of anyone's preferences. This is
what stops the system ever recommending something silly, and it runs *before*
scoring.

**Stage 3 — rank what is left.**

```
score(J) = w_cost      · normalise(cost)
         + w_time      · normalise(time)
         + w_transfers · normalise(transfers)
         + w_comfort   · normalise(discomfort)
```

Two rules matter more than the formula:

- **Weights sum to 1**, so the presets are genuinely comparable.
- **Every objective is min–max normalised across the current candidate set.**
  Rupees and minutes are never added together directly.

## 11. How personalisation works

Three presets, plus manual sliders:

| Preset | cost | time | transfers | comfort |
|---|---|---|---|---|
| Cheapest | 0.78 | 0.10 | 0.06 | 0.06 |
| Balanced | 0.38 | 0.38 | 0.14 | 0.10 |
| Fastest | 0.10 | 0.74 | 0.10 | 0.06 |

Personalisation means the *same* system gives the student with a deadline the
metro and the student with a free afternoon the bus. A system that always says
"cheapest" is not personalised; it is one hard-coded rule.

Learning weights from observed choices with a discrete-choice model is v2 and is
deliberately **not** implemented.

## 12. Installation

Requires Python 3.12+ and Node 20+.

```bash
git clone <your-repo-url> journeymind && cd journeymind

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt

cd frontend && npm install && cd ..
```

`backend/requirements.txt` is the **serving** set (no PyTorch).
`requirements-train.txt` adds PyTorch and scikit-learn.
`requirements-dev.txt` adds pytest.

## 13. Local development

**One process (production shape):**

```bash
cd frontend && npm run build && cd ..          # builds into backend/app/static
cd backend && uvicorn app.main:app --port 8000
# → http://127.0.0.1:8000
```

**Two processes (hot reload):**

```bash
cd backend && uvicorn app.main:app --port 8011 --reload
cd frontend && npm run dev                      # → http://127.0.0.1:5173
```

API docs: `/api/docs`.

## 14. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness — reports what is actually loaded, not what was configured |
| `GET` | `/api/city` | Study area, bounds, transit lines, data-honesty notice |
| `GET` | `/api/places` | Named places for the From/To pickers |
| `GET` | `/api/models` | All six models and whether each can run right now |
| `GET` | `/api/demo` | The demo scenario, computed live by the real pipeline |
| `POST` | `/api/recommend` | The product |

```bash
curl -s localhost:8000/api/recommend -H 'Content-Type: application/json' -d '{
  "origin": "pl_wipro_sarjapur",
  "destination": "pl_pes_university",
  "budget": 250,
  "max_time": 120,
  "preference": "balanced"
}'
```

`departure_time` is optional and **defaults to right now**, on the study area's
clock. Send an offset-aware ISO timestamp (`2026-08-28T03:30:00Z`) and it is
converted to that clock before the model sees it; send a bare one
(`2026-08-28T09:00:00`) and it is read as local wall-clock time. Getting this
wrong would price a Bengaluru morning peak at 03:30, so it is converted rather
than assumed.

Returns the recommended journey, two alternatives, per-leg fares with
provenance, constraint status, the explanation, model metadata, and a full
`pipeline` trace of what each stage did.

**When nothing fits**, the API returns `200` with `feasible: false`, a message
naming both limits, and up to three labelled near-misses ("Closest under
budget", "Fastest available", "Cheapest available") — each carrying the
constraint it breaks. It never silently returns an invalid route.

## 15. ML training

```bash
python scripts/train.py --all --epochs 450 --patience 60
```

- **Target:** expected travel time for one edge, at one hour, on one day.
- **Loss:** Huber over `log(1 + travel_time)`. Log space stops 40-minute bus
  rides drowning out 4-minute walks; Huber stops one GPS glitch dominating.
- **Split: temporal, not random.** Weeks 1–5 train, week 6 validate, weeks 7–8
  test. A random split would put 09:00 Tuesday in training and 09:15 Tuesday in
  test — the model would have effectively seen the answer.
- **Features:** node kind, degree, interchange flag, adjoining road speed, coarse
  position, noisy congestion reading; edge class, length, speed, scheduled time,
  lanes, headway; hour and day-of-week as cyclic sine/cosine pairs, weekend flag,
  rain flag.
- **Output:** `models/{graphsage,gat,mlp}_model.npz` (~50 KB each), loaded by the
  NumPy serving path.

Architecture, as specified:

```
node features ─┐
               ├→ [encoder layer 1] → [encoder layer 2] → node embeddings
graph edges ───┘                                              │
                                                              ▼
edge (u,v) prediction = MLP([emb_u ‖ emb_v ‖ edge_features ‖ time_context])
                                                              │
                                                              ▼
                                              predicted travel time (minutes)
```

Swapping `graphsage` for `mlp` deletes message passing and changes nothing else
— same features, same capacity, same optimiser. That is baseline 4.

## 16. Evaluation and baselines

```bash
python scripts/evaluate.py
```

Six models behind one interface, as the documentation requires:

| # | Baseline | What it tests |
|---|---|---|
| 1 | Free-flow time | Does the model beat naive physics? |
| 2 | Historical mean per edge per hour | Does it beat a lookup table? *The real bar.* |
| 3 | Gradient-boosted trees, no graph | Does it beat good classical ML? |
| 4 | **MLP, identical features, graph removed** | **The critical ablation.** |
| 5 | Single-mode-only recommendation | Does multi-modal planning beat what apps do today? |
| 6 | Map-app ETA | Not run — no licensed reference available. |

**The measured result, stated plainly.** On the bundled synthetic dataset, with
an identical training budget, **no graph model beats the graph-free MLP**:

| Model | Test MAE (min) | RMSE | MAPE |
|---|---|---|---|
| GraphSAGE | 0.260 | 0.400 | 10.4 % |
| GAT | 0.238 | 0.356 | 9.9 % |
| MLP (graph removed) | **0.228** | **0.337** | **9.6 %** |

GraphSAGE is also **sensitive to initialisation** here — across seeds 0–2 its
test MAE ranged from 0.238 to 0.273, while GAT and the MLP were stable. We
report that rather than quoting the best seed.

**We therefore do not claim the GNN is more accurate.** On this data it is not.
See `EVALUATION.md` for why that is the expected outcome given how the bundled
data was generated, and what experiment would actually settle the question.

## 17. Testing

```bash
python -m pytest tests/ -q
```

64 tests covering: multimodal graph construction; access and ride overlay;
over-budget removal; over-time removal; Pareto domination; preset divergence;
alternative distinctness; metro charged once across an interchange; fare
provenance; explanation determinism; the no-feasible-journey path; invalid
input; out-of-area rejection; peak-vs-off-peak time dependence; and **NumPy ↔
PyTorch model parity**.

## 18. Deployment

Single Render web service. FastAPI serves both the API and the built React
bundle from the same origin, which removes cross-origin configuration and halves
what can break at deploy time.

**Deploy:**

1. Push this repository to GitHub (or GitLab).
2. In Render: **New → Blueprint**, point at the repo. `render.yaml` is detected.
3. Confirm. Render builds the Dockerfile and deploys.

`render.yaml` sets `DEMO_MODE=true`, `JM_MODEL=gat`, the health-check path
`/health`, and the region. The container binds `$PORT` — nothing is hard-coded
to localhost.

**Or manually:** New → Web Service → Docker runtime → health check `/health`.
No environment variables are required; every setting has a working default and
there are no secrets.

**Locally:**

```bash
docker build -t journeymind .
docker run -p 8000:8000 journeymind
```

Design constraints the deployment respects: no GPU, CPU-only inference, no
model download at boot, no external data fetch, no persistent storage, ~1 s cold
graph build, ~50 KB model weights, and a service that starts even if the model
weights are missing (it falls back to the historical-mean baseline and says so
in `/health` and in every response).

## 19. Limitations

Stated plainly, because a project that names its own limits is more credible
than one that doesn't.

- **One city, one bounded corridor.** Nothing here claims to generalise.
- **The training data is synthetic.** No real trip was ever logged for this
  repository. Accuracy figures describe a generator we wrote, not Bengaluru.
- **Ride-hailing fares are estimates, not quotes.** Surge pricing is proprietary
  and is deliberately not modelled.
- **"Live" means the clock, not a live feed.** The service recomputes against
  the current minute, and models the current hour's congestion and service
  hours. There is no live traffic source, no live vehicle positions and no
  availability data. Recommending a bike-taxi does not mean one is nearby.
- **Waiting time is `headway ÷ 2`** — a scheduling assumption, not an
  observation. No individual departure is scheduled, so "the 08:42" does not
  exist here.
- **Service hours are approximate**, rounded to the half hour, and do not model
  holidays, disruptions or the last train actually leaving early.
- **No booking.** The system recommends; it does not reserve or pay.
- **Comfort and reliability are crude proxies.** Both are subjective.
- **Accessibility is not modelled.** Step-free routing is a real need and a real
  gap.
- **The GNN is not shown to be better.** See §16.
- **The k-shortest search is an approximation** of an NP-hard problem.

## 20. Future work

> **A designed extension exists.** `V2_TRUST_SECURITY_GOVERNANCE.md` continues this
> documentation at §46 and works through a rider decision layer, a trust score with
> abstention, a threat model, responsible-AI objectives, a future agent permission
> matrix, and a governance framework — all wrapped *around* the graph, GNN and
> optimiser described above, which it leaves unchanged. Its §90 is an honest ledger of
> what is built today versus what is only proposed.

- Plug in live traffic and live transit positions as extra edge features — the
  architecture already accommodates them.
- Learn preference weights from observed choices (multinomial logit).
- Uncertainty-aware routing: prefer a journey that is *reliably* 30 minutes over
  one that averages 26 but sometimes takes 50.
- Accessibility routing: step-free paths, seat likelihood.
- A second city, to measure transfer performance — a genuinely strong result if
  it works.
- Group travel, where four people splitting an auto changes the answer.
- The accessibility-atlas extension (isochrones and cumulative-opportunity
  scores) on the same graph.

## 21. Research questions

| ID | Question | Status |
|---|---|---|
| RQ1 | Does message passing over a multi-modal transport graph improve travel-time prediction over an identical model without graph structure? | **Answered on synthetic data: no.** See §16. |
| RQ2 | How far does the benefit extend — does 2-hop help more than 1-hop, and does 3-hop hurt through over-smoothing? | Harness exists (`--layers`); not yet run systematically. |
| RQ3 | Does more accurate travel-time prediction translate into measurably better recommendations, or does the optimiser wash the improvement out? | Open. `scripts/evaluate.py` includes the recommendation-level metrics. |
| RQ4 | How many observed choices are needed to recover a user's preference weights well enough to beat a fixed "balanced" default? | Not started — needs real user choices. |
| RQ5 | Does the model generalise to a held-out spatial region, or has it memorised specific roads? | **The most interesting open question.** This is where graph structure should matter, because a held-out region's edges cannot be memorised. |

## 22. Data and licensing

Full detail in **`SOURCES.md`**. Headlines:

- Map data © **OpenStreetMap** contributors, ODbL — attribution is displayed on
  every map in the application, as the licence requires.
- Fare tables are transcribed from published operator tariffs and labelled
  `published`, with a note to verify before citing.
- **No private app endpoint is queried, scraped, or reverse-engineered.** Uber,
  Rapido, Namma Yatri and Ola are products, not data sources.
- **No paid API is laundered** — no commercial routing result is stored,
  redistributed, or used as a training label anywhere in this repository.
- **No personal data is collected.** No accounts, no cookies, no tracking, no
  trip history. The only retained state is a 60-second per-IP request counter
  for rate limiting.

---

*JourneyMind is a student research prototype. It is not a production
journey planner, and it should not be used to make a journey you cannot afford
to get wrong.*
