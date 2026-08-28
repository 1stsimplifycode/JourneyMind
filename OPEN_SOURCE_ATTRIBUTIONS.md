# OPEN SOURCE ATTRIBUTIONS

What influenced what, and under which licence.

---

## 0. Read this first

**No source code from any project listed here has been copied into this
repository.** Every component named below was written from scratch. What was
taken is *architectural*: a data format, a decomposition, a state vocabulary,
an idea about where a boundary belongs.

That distinction is not a formality. Two of the most relevant projects in this
space — **OpenTripPlanner (LGPL-3.0)** and **LibreTaxi (AGPL-3.0)** — carry
copyleft licences that would impose obligations on this entire codebase if
their code were incorporated. Reimplementation from a published design is the
correct route, and it is the route taken.

### Verification status

Licences and project descriptions were checked against each project's own
repository or site while writing this file. Entries are tagged:

| Tag | Meaning |
|---|---|
| **VERIFIED** | Licence and purpose confirmed against the project's own repository/site |
| **UNVERIFIED** | Named in the brief; a single canonical project could not be confidently identified. Treated as conceptual background only |

**If you intend to vendor code from anything here, re-check its licence
yourself at that moment.** Licences change, forks diverge, and this file
records a point in time.

---

## 1. Projects that genuinely shaped this codebase

### OpenTripPlanner — VERIFIED
`github.com/opentripplanner/OpenTripPlanner` · **LGPL-3.0**

The most influential reference in the project, and the reason the routing layer
looks the way it does.

| What was taken | Where it lives here |
|---|---|
| **GTFS + OpenStreetMap as the two substrates of a multimodal graph** | `backend/app/data/`, `scripts/generate_dataset.py` |
| **Modelling transit, street and transfer edges in *one* graph rather than routing per-mode and stitching** | `backend/app/graph/builder.py` — road / transit / transfer / ride / access edges |
| **Access and egress legs as first-class parts of the journey** | `RequestGraph._add_access_edges` |
| **Time-dependent search: an edge entered later is a different edge** | `backend/app/routing/costs.py` bucketed cost planes |
| **Service calendars — a route runs or it does not** | `backend/app/data/provider.py` service hours |

**Not taken:** any Java, any of OTP's RAPTOR/Range-RAPTOR implementation, its
GraphQL schema, or its build pipeline. The search here is Yen's *k*-shortest
over a state-augmented graph, which is a different algorithm solving a
different problem (diverse candidates for multi-objective ranking, not a single
fastest itinerary).

**Licence note:** LGPL-3.0. Linking obligations would apply to distributed
binaries containing OTP. Nothing is linked; nothing is copied.

---

### LibreTaxi — VERIFIED
`github.com/ro31337/libretaxi` · **AGPL-3.0**

| What was taken | Where it lives here |
|---|---|
| **The rider↔driver negotiation is a state machine, and the interesting states are the failure ones** | `backend/app/lifecycle/states.py` |
| **A ride is not confirmed when it is requested** — the gap between request and pickup is where the product lives | The whole expected-cost model |

**Not taken:** any JavaScript, the Telegram bot layer, the Firebase data model.

**Licence note:** AGPL-3.0 is the strongest copyleft in common use and reaches
network-delivered services. Copying LibreTaxi code into this repository would
require releasing this entire service under AGPL-3.0. It was read, not copied.

---

### Uber H3 — VERIFIED
`github.com/uber/h3` · **Apache-2.0**

| What was taken | Where it lives here |
|---|---|
| **Hexagonal spatial indexing for zone aggregates** — equal-area cells, uniform neighbour distance | Specified for the accessibility atlas (v1 Part Three, §41) and for the privacy coarsening rule in `V2_TRUST_SECURITY_GOVERNANCE.md` §72 |

Currently a **design dependency, not a code dependency**: the enterprise layer
aggregates to graph nodes rather than H3 cells, because the study area is one
corridor and node-level zones are already fine-grained. H3 is the documented
path when the area widens. Apache-2.0 would permit direct use.

---

### GTFS / MobilityData — VERIFIED
`gtfs.org`, `github.com/MobilityData` · specification, **CC-BY**-style terms

The transit data model — `stops`, `routes`, `trips`, `stop_times`, service
calendars — is the vocabulary this project's transit layer speaks, and it is
why `transit_routes.json` has the shape it has. A specification, not code.

---

### Routing engines: OSRM, Valhalla, GraphHopper — VERIFIED
**BSD-2-Clause**, **MIT**, **Apache-2.0** respectively

Influenced two decisions rather than any implementation:

- **Free-flow routing as a baseline you must beat.** Baseline 1 in
  `EVALUATION.md` is distance ÷ speed limit, exactly what these engines give
  you without traffic data.
- **Contraction//preprocessing separates build time from query time**, which is
  why the city graph is built once and cached (`get_graph`) while per-request
  work is confined to `RequestGraph`.

All three carry permissive licences and could be adopted directly if this
project ever needs real street geometry rather than a study-area graph.

---

### Dispatch-platform patterns (surveyed, not adopted)
Several open ride-hailing backends share a common architecture — Go or Node
microservices, PostgreSQL + PostGIS for geospatial queries, Redis for
driver-location matching, separate passenger/driver/admin services. Licences
across this group range from **MIT** to **GPL**; individual projects were not
verified in detail because none was adopted.

| What was taken | Where it lives here |
|---|---|
| **The booking lifecycle vocabulary** — requested → matched → accepted → cancelled → completed, with rejection and no-supply as distinct terminal reasons | `backend/app/lifecycle/states.py` |
| **Driver-side and rider-side cancellation are different events with different causes** | The three separate probabilities `p_match`, `p_accept`, `p_cancel` |
| **Provider adapters behind one interface** so a new operator is a plugin | `backend/app/providers/base.py` |

**Deliberately not adopted: the microservice split.** This system is one
process. A dispatch platform is split because it has independent scaling axes
(millions of driver location pings vs thousands of bookings). This has no
driver fleet, no location stream, and one workload. Splitting it would add
deployment surface and network failure modes to buy nothing. That is a
considered rejection of the reference architecture, not an oversight.

---

## 2. Projects named in the brief that could not be verified

Each of these was searched for. In each case either no single canonical
open-source project could be identified, several unrelated projects share the
name, or the result appeared to be a commercial template rather than an
open-source project.

| Project | Status | Note |
|---|---|---|
| **OpenRide** | UNVERIFIED | At least three distinct things carry this name: an inactive SourceForge dynamic-ridesharing project, a GPLv3 ride-hailing app, and a newer platform. No single canonical repository. |
| **Locomotion** | UNVERIFIED | Appears to relate to co-operative car-sharing. Could not confirm a repository or licence. |
| **OpenTaxi** | UNVERIFIED | Name used by several unrelated projects. |
| **24Ryde** | UNVERIFIED | No open-source repository identified. |
| **Ocar** | UNVERIFIED | No open-source repository identified. |
| **Rydr** | UNVERIFIED | No open-source repository identified. |
| **RideSmart** | UNVERIFIED | Name used by several unrelated products; no canonical open-source fare-comparison project identified. |

**Nothing was taken from any of these**, and no licence claim is made about
them. They are listed so the omission is visible rather than silent. The
architectural patterns the brief expected from them — provider abstraction,
fare comparison, booking lifecycle, availability modelling — are all present,
derived from the verified sources above and from first principles.

> **Why this section exists rather than being quietly dropped.** Writing a
> confident architectural analysis of a project I could not verify would be
> exactly the "unsupported claim" failure this project is built to detect
> (`V2_TRUST_SECURITY_GOVERNANCE.md` §59). An honest gap is worth more than a
> plausible fabrication.

---

## 3. Runtime and build dependencies

Actual code dependencies, with their licences.

### Serving (`backend/requirements.txt`)

| Package | Licence | Why |
|---|---|---|
| FastAPI | MIT | API framework |
| Uvicorn | BSD-3-Clause | ASGI server |
| Pydantic | MIT | Request/response validation |
| NumPy | BSD-3-Clause | All model inference at serving time |
| NetworkX | BSD-3-Clause | Graph utilities |
| tzdata | Apache-2.0 / public-domain data | IANA zones on hosts without a system database |

**Not in the serving image, deliberately:** PyTorch and scikit-learn. Both
models are trained offline and exported to `.npz`; serving replays them in
NumPy. This keeps the image small enough for a free-tier instance and means a
training-only CVE cannot reach production.

### Training (`backend/requirements-train.txt`)

| Package | Licence | Why |
|---|---|---|
| PyTorch | BSD-3-Clause | GraphSAGE / GAT / MLP training |
| scikit-learn | BSD-3-Clause | Gradient-boosted baselines; logistic regression for the reliability heads |

### Frontend

| Package | Licence | Why |
|---|---|---|
| React | MIT | UI |
| Vite | MIT | Build |
| Leaflet | BSD-2-Clause | Map rendering |

### Data

| Source | Licence | Obligation |
|---|---|---|
| OpenStreetMap | **ODbL-1.0** | Attribution required; share-alike applies to derived *databases*. Attributed in the UI map footer and in `SOURCES.md`. |
| Namma Metro network facts | Public knowledge | Station names, line assignments and approximate positions. |
| BMRCL / BMTC fare tables | Published tariffs | Transcribed; verify before quoting. |

Full provenance for every data element, including what is synthetic, is in
**`SOURCES.md`**.

---

## 4. What is original to this project

Listed so the boundary is clear.

| Component | Description |
|---|---|
| **Expected-cost solver** | `backend/app/lifecycle/expected_cost.py`. The booking lifecycle as an absorbing Markov chain, solved exactly for expected cost, expected time and the full cost distribution, including the fallback-substitution term. No reference project computes this. |
| **Reliability heads** | Three calibrated probability models (match / accept / cancel), selected on calibration rather than discrimination because the output is multiplied into money. |
| **Multi-objective journey optimiser** | Yen's *k*-shortest pooled across five time/money weightings, constraint filter, four-objective Pareto frontier, personalised ranking. |
| **GNN travel-time model** | GraphSAGE / GAT / MLP written directly in PyTorch — no PyTorch Geometric — and exported for NumPy serving with a parity test asserting agreement to 1e-4. |
| **Provenance labelling** | Every number that leaves the API carries `real` / `published` / `estimated` / `predicted` / `simulated`. |
| **Enterprise analytics** | Reliability-adjusted cost per km, cost-of-failure accounting, cohort suppression. |
| **Trust, security and governance design** | `V2_TRUST_SECURITY_GOVERNANCE.md`. |

---

## 5. If you vendor anything

1. Re-check the licence **at that moment** — this file is a snapshot.
2. Add the project to §3 with its licence.
3. Preserve its `LICENSE` and `NOTICE` files verbatim in the tree.
4. If it is copyleft (LGPL, GPL, AGPL), work out the obligation on the
   *distributed artifact* before writing the first line, not after.
5. For ODbL data specifically: attribution is required and share-alike attaches
   to derived databases, which includes a graph built from an OSM extract.
