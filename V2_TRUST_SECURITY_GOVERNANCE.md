# JourneyMind v2 — Trust, Security and Governance

**Parts Four to Ten of the JourneyMind project documentation.**
Continues the section numbering of `JourneyMind_OSINT_Isopolis.pdf` v1.0, which ends at §45.

---

## How to read this document

This is an **extension**, not a replacement. The definition in v1 §23 stands unchanged:

> JourneyMind is a multi-modal journey recommendation system that represents a city as a graph,
> uses a graph neural network to predict travel time on each segment of that graph under current
> time-of-day conditions, and applies constrained multi-objective optimisation to recommend the
> best complete journey — including journeys that combine public transport and ride-hailing —
> subject to a user's stated budget, deadline and personal preferences.

Everything in Parts Four to Ten sits **around** that core. The graph, the GNN, the candidate
generator, the Pareto filter and the weighted ranking are untouched. Nothing here turns JourneyMind
into a chatbot, a generic AI-security project, or an autonomous-agent demo. The mobility problem
stays the problem.

Three status tags are used throughout, and they are load-bearing:

| Tag | Meaning |
|---|---|
| **BUILT** | Exists in the current repository today. File paths are given. |
| **PARTIAL** | A hook exists; the mechanism described here is not finished. |
| **NEW** | Proposed. Not built. Not claimed to work. |

§90 collects every claim in this document into one table under those three tags. If you read
nothing else, read §90 — it is the honest summary of what is real.

---

# PART FOUR — The Rider Decision Layer

## 46. Why extend, and what must not change

JourneyMind is not a text generator with a plausible-sounding output. It is a decision system whose
output costs a person money and time, and can strand them. A wrong travel-time prediction at 22:40
does not produce an awkward sentence; it produces someone at a bus stop after the last bus.

That is the whole justification for this extension. **Consequence is what makes trust, verification,
security and governance load-bearing rather than decorative.** A recommendation engine that can be
wrong in ways that cost real money is exactly the setting where "how does the system know it might be
wrong?" is a research question rather than an afterthought.

Three boundaries from v1 must survive the extension, and they are treated here as invariants:

1. **v1 §19 — "No booking. The system recommends; it does not reserve or pay."**
   This is re-read in Part Eight not as a limitation but as a *safety property*, and it becomes the
   hardest constraint on any future agent.
2. **v1 §20 HARD RULE — "Nothing in the MVP may depend on a data source we have not already confirmed
   we can access."** Extended in Part Six: nothing may depend on a source whose *integrity* has not
   been confirmed either.
3. **v1 §33 — "No personal data … aggregate at the segment or zone level, never store an individual's
   identifiable trip history."** The rider decision layer directly stresses this rule, and §72 and
   §73 say how, rather than quietly walking past it.

## 47. What a rider actually does

The v1 pipeline ends at the recommendation. In reality that is the halfway point. A rider does
things afterwards, and those things are the only outcome evidence the system will ever get.

The observable event vocabulary:

| Event | Meaning |
|---|---|
| `SHOWN` | A recommendation set (1 recommended + 2 alternatives) was displayed |
| `ACCEPTED` | The rider selected the recommended journey |
| `ALT_CHOSEN` | The rider selected one of the alternatives instead |
| `REJECTED` | The rider dismissed the set without selecting anything |
| `RE_REQUESTED` | The rider changed inputs and asked again within a short window |
| `BOOK_ATTEMPTED` | The rider tried to hail the ride leg (outside the MVP; a link-out today) |
| `BOOK_FAILED` | The attempt did not produce a vehicle |
| `CANCELLED` | A confirmed booking was cancelled |
| `MODE_SWITCHED` | The rider began the journey but changed mode mid-way |
| `COMPLETED` | The rider arrived, having followed the recommendation |
| `ABANDONED` | The rider began and did not complete |
| `PROVIDER_REPEAT` | The rider chooses the same provider or mode across many requests, against advice |

Every one of these is a fact about a person's behaviour. **None of them is a fact about whether the
recommendation was good.** The gap between those two statements is the entire subject of Part Four.

## 48. Cancellation is not a verdict

> **THE MISTAKE THIS SECTION EXISTS TO PREVENT**
> Treating `CANCELLED` as a negative label and feeding it to the model. Do that and the system
> learns "riders dislike this route" when the truth may be "the metro was closed", "the fare
> estimator drifted", or "someone poisoned the feed".

The confusion has a precise shape. What the system wants to know is a **counterfactual**: would the
rider have been better off following this recommendation than the alternative? What the system
observes is a **decision**: what the rider did, given everything they knew and the system did not.

Between those sits a confounder. The rider had information the system lacked (a friend offering a
lift, a message that the meeting moved, a look at the actual queue at the auto stand), and the world
moved between recommendation and action (the fare changed, the driver cancelled, it started raining).

So the operating rule is:

> **Rejection is information about preferences. Error is a mismatch between prediction and reality.
> They are different signals, they have different owners, and they must never be pooled.**

A rider who rejects a metro recommendation for the fifth time in favour of an auto is not producing
evidence that the metro prediction is wrong. They are producing evidence that their comfort weight is
higher than the balanced preset assumes — which is *exactly* the signal v1 §15 already wants for
"v2: learn weights from observed choices using a multinomial logit / discrete choice model", and
exactly what original **RQ4** asks about. That is a feature request being answered, not a fault
report.

## 49. Ten causes, and the evidence that separates them

The cause of a cancellation cannot be read off the cancellation. It can sometimes be *inferred* from
what changed between the moment the recommendation was issued and the moment it was abandoned.

That is the mechanism: **snapshot the decision context at issue time, snapshot it again at the
abandonment event, and treat the delta as the evidence.** The system already builds most of this
snapshot — v1 §6's pipeline trace, implemented today as the `pipeline` object returned with every
recommendation (`backend/app/api/serialise.py`). Persisting it is what makes attribution possible.

| # | Candidate cause | Fingerprint that supports it | Is it a system fault? |
|---|---|---|---|
| 1 | **Recommendation error** | Realised travel time on completed trips over the same edges and hour diverges systematically from prediction; the residual is not a one-off | Yes — model |
| 2 | **Fare change** | Fare snapshot at t₁ falls outside the band quoted at t₀ | Sometimes — estimator |
| 3 | **Availability failure** | `BOOK_FAILED`; no vehicle returned; concentrated on one provider or one pickup area | No — world; but recommending an unavailable mode is a design fault (§68) |
| 4 | **ETA change** | Predicted arrival revised upward between t₀ and t₁ beyond the stated interval | Sometimes — model or world |
| 5 | **Rider preference** | The chosen alternative was on the Pareto frontier; the choice is consistent across requests | **No — this is taste** |
| 6 | **Changed their mind** | Cancellation with no observable delta in any system signal, no re-request, no pattern | No |
| 7 | **External event** | The same behaviour appears across many unrelated riders in one time-and-place window; correlates with weather, a match, a bandh, a festival | No |
| 8 | **Provider cancellation** | The cancellation event originates from the provider side, not the rider | No |
| 9 | **Data staleness** | Feed age at t₀ exceeded its freshness budget; the recommendation was built on old inputs | **Yes — data** |
| 10 | **Safety / accessibility** | Rider-stated reason, or the route scored poorly on the §70 proxies for a rider who declared a need | **Yes — objective design** |

> **HONESTY CHECK**
> These are hypotheses with likelihoods, not labels. Many events will be genuinely unattributable.
> The system must therefore keep an explicit `UNATTRIBUTED` bucket and **report its size as a
> headline metric**. If 60% of abandonments cannot be attributed, the feedback loop is weak, and the
> right thing to do is say so rather than force every event into a bin.

## 50. Three learning channels

Once causes are separated, the events route to three different places. Pooling them is the failure
mode; separating them is the design.

| Channel | Fed by | Learns | Owner |
|---|---|---|---|
| **Prediction** | `COMPLETED` trips with realised leg times | Travel-time model — a genuine supervised label, the scarcest asset in the project (v1 §30) | GNN |
| **Preference** | `ACCEPTED` vs `ALT_CHOSEN` over the *offered set* | The rider's weights, via discrete choice — original **RQ4** | Optimiser |
| **Integrity** | `BOOK_FAILED`, fare surprises, staleness flags, cohort anomalies | Whether the inputs and the world still match — Parts Five and Six | Data / security |

The third channel is the new one, and it is the bridge to everything downstream. A cancellation that
attributes to cause 9 (staleness) is not training data at all. It is an **incident**.

## 51. Set quality is not rank quality

JourneyMind returns one recommendation plus two alternatives (v1 §20). That structure makes an
evaluation distinction available for free, and it matters:

- **Rank quality** — was the *recommended* journey the one the rider took? (`ACCEPTED`)
- **Set quality** — was the journey the rider took *anywhere in the set we offered*?
  (`ACCEPTED` or `ALT_CHOSEN`)

A rider taking alternative 2 is a **partial success**: the candidate generator and the Pareto filter
did their job, and only the personalised weighting was mis-tuned for this person. That is a far
smaller failure than offering three journeys none of which the rider would consider — and treating
both as "rejected" throws the distinction away.

Both numbers should be reported. A persistent gap between them is the precise signal that
personalisation — not routing, not prediction — is what needs work.

## 52. What the feedback loop cannot see

Two limits, stated before any result is claimed from behavioural data.

**Selection bias.** Outcomes are only ever observed for journeys that were offered and taken. The
journey the system never recommended produces no evidence, forever. A loop trained naively on its own
outputs can only narrow: it grows more confident about the corridor it already recommends, and
blinder everywhere else.

The standard remedies both cost something, and both should be stated rather than assumed:

- **Deliberate exploration** — occasionally promoting a Pareto-equivalent alternative to rank 1 so
  that outcomes are observed for it. Ethically this requires disclosure; a rider must not be silently
  experimented on, and with a small cohort informed consent is the honest route.
- **Off-policy estimation** — reweighting observed outcomes by the probability the old policy would
  have shown them. This needs logged propensities and far more traffic than the 20–30 person user
  study of v1 §16 will produce.

> **HONESTY CHECK**
> With a user study of the size v1 plans, neither remedy will be statistically satisfying. The
> defensible claim from this layer is **"we built an attribution mechanism and characterised how
> often it can and cannot attribute"** — not "we learned rider preferences from behaviour at scale".

---

# PART FIVE — Trust, Verification and Abstention

## 53. Should JourneyMind trust its own recommendation?

Every stage of the v1 pipeline already produces evidence about its own shakiness, and every stage
currently throws that evidence away at the point of display. The GNN knows which edges it has never
observed. The fare estimator knows which numbers came from a published table and which from a fitted
model. The constraint checker knows whether a journey clears the budget by ₹80 or by ₹2. The data
layer knows how old the feed is.

A trust layer is the decision to **keep that evidence, combine it, and let it change what the rider is
told** — including letting the system decline to answer.

The target behaviour, in the product's own voice:

> Recommended: Rapido, then the Yellow Line. **Confidence: high.**

> Recommendation available, but **confidence is low**: the transit feed for this corridor was last
> refreshed 47 minutes ago, and the ride-hailing fare for the last leg is a fitted estimate with a
> ±₹60 spread against a ₹250 budget. Here is a slower option I am more sure of.

## 54. The signals, and why each one earns its place

No formula yet. First, what should contribute, and why.

### A. Model-side — how sure is the predictor?

| Signal | Why it belongs | Status |
|---|---|---|
| **Per-edge prediction interval** | A point estimate cannot distinguish "22 minutes, reliably" from "22 minutes on average, 12 to 55 in practice". The single highest-value addition in this extension, and v1 §21 already asks for it | **NEW** |
| **Cold-start / out-of-support flag** | v1 §18 already commits to falling back to free-flow with a widened band on never-observed hops. The serving path implements the band and counts every clamp (`COLD_START_LOW/HIGH`, `last_clamped` in `backend/app/models/gnn_numpy.py`). That count is a ready-made trust input that is currently discarded | **PARTIAL** |
| **Distance from the training distribution** | An edge whose features sit far from anything seen in training is being extrapolated on, whether or not it clamps | **NEW** |
| **Cross-model disagreement** | v1 §17 mandates six comparable models behind one interface for *evaluation*. All six are loadable at runtime today (`backend/app/models/loader.py`: `freeflow`, `historical`, `gbt`, `mlp`, `graphsage`, `gat`). Where free-flow physics, a historical lookup, trees and two GNNs agree, the number is probably fine; where they scatter, something is off. **The baseline suite is already a free runtime ensemble — it has simply never been used as one** | **PARTIAL** |

### B. Data-side — where did the inputs come from?

| Signal | Why it belongs | Status |
|---|---|---|
| **Feed age against a per-source freshness budget** | A GTFS feed refreshed hourly and one refreshed in March are not the same input, and only one supports a confident claim about tonight's last train | **NEW** |
| **Source status and provenance class** | v1 §25's `OPEN` / `CONDITIONAL` / `CLOSED` tags and v1 §33's "label uncertainty honestly" already exist as a vocabulary. `SOURCES.md` records URL, date accessed, licence and hash per dataset | **PARTIAL** |
| **Integrity: hash and schema match** | The recorded hash still matching is the difference between "our data" and "whatever is at that URL today" | **NEW** |
| **Coverage of the underlying map** | Isopolis §43 already names the trap: OSM footway coverage in Indian cities is patchy, so a walking leg through a thin-footpath area is a weaker claim than the same leg elsewhere | **NEW** |

### C. Fare-side — how firm is the price?

| Signal | Why it belongs | Status |
|---|---|---|
| **Fare provenance** | A BMRCL slab fare is `published`. A Rapido fare is a fitted `base + per-km + per-min` estimate (v1 §10, §31). The repo already tags every fare `exact` / `published` / `estimated` and propagates the *worst* label to the journey total (`backend/app/models/fares.py`) | **BUILT** |
| **Fare-model residual spread and sample age** | An estimator fitted on samples collected six months ago is drifting whether or not anyone has noticed | **NEW** |

### D. Service-side — does this exist in the world tonight?

| Signal | Why it belongs | Status |
|---|---|---|
| **Service hours** | Route-level first and last service, so a 01:00 request is not offered a train sitting in the depot. Implemented (`backend/app/data/provider.py`, `backend/app/routing/costs.py`): out-of-service boarding costs the real wait until first departure, and the caveat appears on the answer | **BUILT** |
| **Headway assumption confidence** | v1 §18 charges `headway ÷ 2` and calls it a known error source. Scheduled headways in Indian cities can differ from observed by a factor of two (Isopolis §43) | **PARTIAL** |
| **Availability evidence** | There is none. v1 §19 says so plainly: "We may recommend a bike-taxi when none is nearby." An honest zero is still a trust signal — it caps how confident a ride-leg recommendation may ever be | **NEW** |

### E. Historical performance — the empirical prior

| Signal | Why it belongs | Status |
|---|---|---|
| **Corridor × hour historical error** | If this corridor at 18:00 has historically shown 40% MAPE, that is the best available forecast of how wrong we are about to be. Peak-hour error is already an evaluation metric (v1 §16); measuring it *per corridor* turns it into a runtime signal | **NEW** |
| **Outcome rate for similar recommendations** | From Part Four, confounder-adjusted | **NEW** |

### F. Constraint-side — how much room is there?

| Signal | Why it belongs | Status |
|---|---|---|
| **Headroom against budget and deadline** | Already computed and returned: `budget_headroom`, `time_headroom`, and a `cost_at_risk` flag for when the point estimate fits but the upper fare band does not (`backend/app/optimisation/constraints.py`) | **BUILT** |
| **Headroom measured in units of uncertainty** | This is the real move. ₹80 of headroom against a ±₹15 fare band is comfortable; ₹20 against a ±₹60 band is not. **Feasibility should become a probability, not a boolean** | **NEW** |
| **Rank stability under uncertainty** | Resample the uncertain quantities across the existing candidate set and ask how often the recommended journey stays rank 1. If a small perturbation flips the winner, the *ranking* is untrustworthy even when every prediction is. Nearly free once sampling exists | **NEW** |

### G. Rider context — what does being wrong cost *this* person?

| Signal | Why it belongs | Status |
|---|---|---|
| **Consequence asymmetry** | Being 15 minutes late to a film and to an exam are not the same error. The confidence bar should scale with what the rider says is at stake | **NEW** |
| **Declared accessibility or safety need** | Raises the bar, and converts some soft preferences into hard constraints (§69) | **NEW** |

## 55. Two stages: hard gates, then a calibrated number

A single weighted sum over §54 would be wrong by construction, because it lets a very fresh feed
numerically compensate for "this train does not run tonight". Some conditions are not evidence to be
weighed; they are evidence of impossibility.

### Stage 0 — hard gates (boolean, non-negotiable)

| Gate | Fails when | Consequence |
|---|---|---|
| **G1 Source integrity** | A source used has no matching hash / schema / licence entry in the ledger | Block |
| **G2 Service existence** | A scheduled leg runs on a route not in service at that time | Block — **BUILT** |
| **G3 Constraint satisfaction** | The point estimate breaks budget or deadline | Not recommended; shown only as a labelled near-miss — **BUILT** |
| **G4 Physical plausibility** | Implied speed sits outside the mode's envelope | Block |
| **G5 Internal consistency** | Legs do not connect in space or time; leg times do not sum to the total; a transfer is not physically walkable in the time allowed | Block |
| **G6 Declared access need** | A stated step-free or accessibility requirement is not met | Block for that rider |

Gates produce a **reason**, never a silent removal. That is continuous with v1 behaviour, where an
infeasible journey is shown as a labelled near-miss carrying the limit it breaks rather than deleted.

### Stage 1 — a number that means something

The headline confidence should be a **predicted probability of a defined, observable event**, because
a probability can be checked against reality and a hand-tuned index cannot.

> **THE TARGET EVENT**
> **S** = *the rider who follows this recommendation arrives within the stated deadline and pays no
> more than the stated budget.*

Why this target and not another:

1. It is **observable** — `COMPLETED` trips with realised time and paid fare settle it.
2. It is **exactly what the rider was promised**. v1's whole premise is "inside the money and time
   you actually have"; the confidence number should be about that promise and nothing else.
3. It **couples time and money**, which no single-axis uncertainty measure does.

**Estimator.** Start with regularised logistic regression over the §54 signal vector. The
justification is not that it is the most powerful model available — it is that its coefficients are
readable, an auditor can see which signal drove a downgrade, and it behaves sanely on a few hundred
observations, which is the realistic data volume. Move to gradient-boosted trees only if the
reliability diagram demands it, and keep calibration as the acceptance test either way.

**Propagating uncertainty to get there.** Three concrete steps, each small:

1. Give the edge model a **quantile head** — the same architecture with three outputs trained under
   pinball loss at τ = 0.1 / 0.5 / 0.9. No new data required.
2. **Monte Carlo over the existing candidate set** — a few hundred draws of per-leg times and fares.
   Crucially, draw a **shared corridor-hour factor** plus independent per-leg noise: legs on the same
   arterial at the same hour are correlated, and sampling them independently understates journey
   variance badly.
3. Read off **P(within deadline)**, **P(within budget)** and **rank stability** from the same draws.

> **HONESTY CHECK — the cold start of the trust model itself**
> The calibrated model cannot be fitted before completed-trip outcomes exist. Version 1 of the trust
> layer must therefore be a **transparent rule table**, explicitly labelled *uncalibrated* in both the
> interface and the audit record, whose only job is to be replaced. A confidence score that has never
> been checked against outcomes is a design intention, not a measurement, and must not be presented
> as one.

## 56. What the number must mean

A trust score is worth nothing unless "high confidence" empirically means "usually right". So the
score is accepted or rejected on **calibration**, not on plausibility:

- **Reliability diagram** — bucket recommendations by predicted confidence, plot observed success
  rate. The diagonal is the target.
- **Expected calibration error (ECE)** — the headline scalar.
- **Selective-prediction metrics** — coverage (what fraction we answer), selective risk (error rate
  among answered), and the **risk–coverage curve** with its area. This is the standard frame for a
  system allowed to abstain, and it is the right frame here.

## 57. Confidence bands and what they have to earn

Four bands, each of which must eventually cash out as an empirical claim:

| Band | The claim it makes | Behaviour |
|---|---|---|
| **HIGH** | ≥ 85% of these complete within both limits | Recommend normally |
| **MEDIUM** | 60–85% | Recommend, name the weakest signal |
| **LOW** | < 60% | Recommend only alongside a more robust option, warning first |
| **ABSTAIN** | A gate failed, or evidence is insufficient to make any claim | See §58 |

Until measured, the thresholds are provisional and must be labelled as such. **A band is a promise;
an unmeasured band is an unkept one.**

The score must also **decompose**. v1 §5 defines the recommender's job as explaining the choice, and
the repo already generates deterministic reasons and caveats (`backend/app/services/explain.py`).
Trust output follows the same rule: never a bare number, always the one or two signals that set it.

## 58. Abstention as an answer

Letting the system decline is the point of the whole layer. But abstention must be **actionable** — a
dead end is not safety, it is a different failure.

Four permitted forms:

1. **Recommend with an explicit warning**, naming the weak signal.
2. **Recommend the robust option instead** — the journey with the highest P(success), which is often
   not the one with the highest weighted score. This is v1 §21's "prefer a journey that is reliably
   30 minutes over one that averages 26 but sometimes takes 50", made operational.
3. **Widen the promise** — "₹138–₹197, and I am not confident of the upper end."
4. **Defer** — "the feed for this corridor is 47 minutes stale; ask again shortly."

Never permitted: silence, or a confident answer the evidence does not support.

> **THE METRIC THAT KEEPS THIS HONEST**
> Abstention rate is monitored like any other. A system that abstains on 40% of requests is not
> cautious, it is broken — and is probably hiding a data problem behind politeness.

## 59. Unsupported recommendations, not hallucinations

> **TERMINOLOGY, DELIBERATELY**
> A GNN travel-time error is **not** an LLM hallucination, and this document will not call it one.
> The general class is **AI-generated recommendation error**, and within it the interesting subclass
> is the **unsupported recommendation**.

**Definition.** An *unsupported recommendation* is one whose claims are not adequately backed by
evidence available to the system at the time of issue — regardless of whether it happens to be right.

Three distinct failure types, which demand three different fixes:

| Type | What went wrong | Example | Fix lives in |
|---|---|---|---|
| **Prediction error** | The number is wrong; the evidence chain is fine | Predicted 22 min, realised 40 | Model |
| **Unsupported claim** | The number may be right, but nothing entitles the system to assert it | A fitted fare shown with the confidence of a published one; a leg resting on a three-week-old feed | Verification + provenance |
| **Infeasible recommendation** | The journey cannot be executed as described | A metro leg after the last train; a transfer that needs four minutes and is given one | Gates |

The connection to LLM hallucination is **structural, not literal** — and worth stating precisely,
because it is what makes this project legible to a trustworthy-AI audience: in both cases a system
emits a fluent, internally plausible output whose *grounding* was never checked. The remedy belongs to
the same family — **verify against independent evidence before surfacing** — even though the
mechanisms producing the error are entirely different.

## 60. The verification layer

Nine checks, run after prediction and before display.

| # | Check | Evidence it uses | Catches |
|---|---|---|---|
| **V1** | Physical plausibility | Mode kinematics | A 90 km/h city bus |
| **V2** | Historical consistency | The raw observation table — baseline 2 used as a *verifier* rather than a competitor | A prediction outside the 1st–99th percentile ever seen on that edge at that hour |
| **V3** | Cross-model agreement | The other five models in the registry | Single-model failure; silent weight corruption |
| **V4** | Freshness and provenance | `SOURCES.md` ledger: age, licence, hash | Stale or unverifiable inputs |
| **V5** | Service existence and hours | Published first/last service — **BUILT** | Journeys through a shut network |
| **V6** | Constraint re-check under uncertainty | The Monte Carlo draws of §55 | A "fits" that only fits at the median |
| **V7** | Fare-claim strength | Fare provenance labels — **BUILT** | An estimate asserted as a price |
| **V8** | Internal consistency | Graph geometry and the journey's own arithmetic | Legs that do not connect; impossible transfers |
| **V9** | Peer anomaly | Recent recommendations for the same origin–destination pair | A sudden unexplained change in what is being recommended |

Flow, replacing the tail of v1 §6:

```
Prediction → Verification → Confidence estimation → Policy checks → Recommendation | Abstention
```

## 61. The independence rule

> **THE DESIGN PRINCIPLE OF THE WHOLE VERIFICATION LAYER**
> A verifier that shares its evidence with the predictor is a rubber stamp. Verification is only
> worth running where the evidence is independent of the thing being checked.

Applied honestly to the nine checks:

| Check | Independent of the GNN? |
|---|---|
| V1 physics, V4 ledger, V5 service hours, V8 geometry | **Fully** — none of them consults the model at all |
| V2 historical observations | **Largely** — the model trained on this data, but the check compares against the raw distribution rather than the fit |
| V3 cross-model | **Partially, and this must be admitted.** The six models share features and training data, so they can be wrong together. Free-flow and historical-mean are the genuinely independent arms; MLP–GraphSAGE–GAT agreement is weaker evidence than it looks |
| V6, V7, V9 | Derived — they check the *use* of the prediction, not the prediction |

Naming V3's partial dependence is the difference between a verification layer and a comfort blanket.

---

# PART SIX — Security

## 62. What is worth attacking

Before threats, assets. Someone attacks JourneyMind to achieve one of:

| Asset | Why an attacker cares |
|---|---|
| **The recommendation itself** | Steer riders toward or away from a provider, a corridor, a mall, a competitor |
| **Rider trust** | Degrade the system until it is abandoned |
| **Rider location history** | Home and workplace, inferable from repeated journeys (§72) |
| **The fare model** | Extract a competitor's pricing structure by enumeration |
| **Compute** | Force expensive searches; deny service |
| **The training labels** | Persistent, hard-to-detect corruption of the model itself |

## 63. Threat model by entry point

Organised by *where the attack enters the pipeline*, so each threat lands on an architectural layer
that can own it.

### Entry point 1 — upstream open data

| Threat | Plausibility here | What breaks | Detection | Mitigation |
|---|---|---|---|---|
| **Poisoned OSM edits** | Real. OSM is world-editable and targeted vandalism happens | Graph structure: a footpath that does not exist, an altered speed limit, a removed barrier | Diff successive extracts; flag high-impact edits (new fast road, deleted barrier) from recent, low-history editors | **Pin an extract version.** Never auto-update into production. Review the diff |
| **Manipulated or spoofed GTFS** | Plausible for unofficial community feeds — v1 §27 already flags one as unreliable | Schedules, service hours, stop positions | GTFS validator; schema diff; stop-position jump detection | Prefer official feeds; treat unofficial as `CONDITIONAL` and cap trust; hash-pin |
| **Feed substitution** | Portal moves, DNS or CDN compromise, typosquatted mirror | Everything downstream | Certificate and hash mismatch against `SOURCES.md` | **Hash-pin every source. A fetch failure must be visible, never a silent fall back to cache** |
| **Stale-data injection (replay)** | Serving an old-but-valid feed to hide a disruption | Trust in freshness | Monotonic feed-version check; age against budget | Reject non-advancing versions; surface age as a trust signal |
| **A fake source entering the catalogue** | Most likely as an accident: a plausible URL added under deadline | Everything | Human review of any catalogue addition | Allow-list. Additions land in quarantine, never production (§77) |

### Entry point 2 — the training pipeline

| Threat | Plausibility here | What breaks | Detection | Mitigation |
|---|---|---|---|---|
| **Probe-data poisoning** | **This is the project-specific one.** v1 §30 makes "your own logged trips, collected by your own team" the primary label source. One compromised phone or one careless collector corrupts the scarcest asset in the project | Travel-time labels | Per-collector residual analysis; leave-one-collector-out validation | Robust statistics per edge-hour (trimmed mean or median, never mean); per-collector identity on every trace |
| **Adversarial graph features / node injection** | Research-plausible. GNNs are specifically vulnerable to *structure* perturbation, because aggregation propagates it to neighbours | Predictions across a whole neighbourhood, not one edge | Degree and feature-distribution monitoring; embedding drift on unchanged nodes | Bound the influence of any single new node; review structural diffs |
| **Backdoor / trigger pattern** | Low for a student project, high value to demonstrate | A specific feature combination makes an edge look artificially fast | Held-out spatial testing (original **RQ5**); activation clustering | Provenance on every training row; reproducible builds |

### Entry point 3 — inference and requests

| Threat | Plausibility here | What breaks | Detection | Mitigation |
|---|---|---|---|---|
| **Algorithmic denial of service** | Real. Candidate generation approximates an NP-hard problem (v1 §15); a large budget, a long deadline and six transfers is an expensive request | Availability | Latency and expansion-count monitoring | Search caps and per-minute rate limiting exist today (`MAX_EXPANSIONS`, `MAX_SPUR_POSITIONS`, `rate_limit_per_min`) — **PARTIAL**; add cost-aware admission |
| **Malicious input** | Routine | Errors, resource use | Typed request models — **BUILT** | Bounds on every numeric field — **BUILT** |
| **Fare-model extraction** | Plausible | Commercial value of the estimator | Query-pattern anomaly | Rate limits; return bands, never point quotes — **BUILT** |
| **Inference on rider history** | Serious once §47 events are stored | Privacy | — | §72: aggregation, coarsening, retention limits |

### Entry point 4 — the feedback loop

| Threat | Plausibility here | What breaks | Detection | Mitigation |
|---|---|---|---|---|
| **Sybil feedback** | The novel one. Coordinated fake accounts accepting-then-cancelling a rival's route until the system demotes it | The learning loop, and through it every future recommendation | Cohort diversity checks; account-age weighting; geographic and timing correlation | §67: **no behavioural signal auto-modifies anything.** Human review sits between signal and change |

### Entry point 5 — the future agentic layer

| Threat | Plausibility here | What breaks | Detection | Mitigation |
|---|---|---|---|---|
| **Prompt injection via external data** | High, and concrete. `stop_name`, `route_long_name`, `agency_name` in GTFS and `name`/`description` tags in OSM are **attacker-writable free text** that would flow straight into an agent's context | Agent behaviour | Content scanning at ingestion; provenance-tagged context | §78: external content is data, never instruction. Strip and structure before any model sees it |
| **Compromised tool or API** | Plausible | Anything the tool can reach | Egress monitoring; response schema validation | Allow-listed tools and egress; no dynamic tool discovery |
| **Confused deputy** | Plausible | Privilege escalation across agents | Audit-log correlation | Per-agent credentials; no shared write paths (§77) |

### Entry point 6 — model artefacts

| Threat | Plausibility here | What breaks | Detection | Mitigation |
|---|---|---|---|---|
| **Weight-file tampering** | The service loads `.npz` weights from disk at startup | Every prediction, silently | Hash verification at load | Sign and verify checkpoints; record the model hash in every audit entry (§81) |

## 64. Data quality vs AI reliability vs security attack

This distinction is the conceptual centre of Part Six.

| | **Data quality problem** | **AI reliability problem** | **Security attack** |
|---|---|---|---|
| **Origin** | The world, or the collection process | The model | An adversary |
| **Intent** | None | None | **Yes** |
| **Typical pattern** | Broad, correlated with a source or a region; often noisy | Systematic, correlated with model family or with distance from training data | Targeted, often low-volume, timed — and someone benefits |
| **Detected by** | Validators, schema checks, freshness budgets, coverage statistics | Calibration, drift monitoring, residual analysis | Anomaly detection **plus** provenance, integrity and benefit analysis |
| **Correct response** | Fix the pipe | Retrain, recalibrate, or widen intervals | **Contain, revoke, preserve evidence — do not retrain** |
| **Owner** | Data engineering | ML | Security |

## 65. Same symptom, three responses

The reason §64 is not academic:

> **Symptom: "predicted travel times on the Sarjapur Road corridor are badly wrong this week."**
>
> - If it is a **data problem** — the feed changed format and stop positions shifted — the fix is in
>   the pipeline. Retraining on the bad data makes it permanent.
> - If it is a **reliability problem** — a flyover opened and the world moved (v1 §18's distribution
>   shift) — retraining is exactly right.
> - If it is an **attack** — someone is poisoning the observations — **retraining is the worst
>   possible response**, because it bakes the attack into the weights and destroys the evidence.

Same symptom. Opposite correct actions. That is why the distinction must be made *before* the
response, and why a triage order is a design artifact rather than an operational detail:

1. **Exogenous?** Weather, festival, match, bandh, holiday. Cheapest to check, most often the answer.
2. **Data?** Feed age, validator output, schema diff, hash match.
3. **Provider?** Failures concentrated on one operator.
4. **Model?** Residual drift on completed trips for that corridor.
5. **Fare estimator?** Fare residuals against realised payments.
6. **Only then: manipulation.** Requires positive evidence — coordination, timing, benefit — not
   merely the absence of another explanation.

Step 6 needing *positive* evidence matters. "We could not explain it, therefore attack" is how
security teams generate noise and lose credibility.

## 66. Behaviour as a security sensor

Part Four's events, aggregated, are one of the cheapest anomaly detectors available — because riders
encounter the real world and the system does not.

Cohort-level monitors, **never individual**:

- cancel rate by route / provider / corridor / hour, against its own baseline
- booking-attempt failure rate
- fare-surprise rate: realised payment outside the quoted band
- ETA-revision magnitude between issue and departure
- top-1 versus in-set acceptance (§51)
- provider switch rate

Each breach opens the §65 triage cascade rather than a model update. Sybil resistance is part of the
design, not an afterthought: rate-limit per identity, weight by account history, and require the
cohort behind a signal to be **diverse in identity, geography and time** before it can trigger
anything at all.

Privacy is a hard precondition here, not a footnote: a minimum cohort size before any statistic is
computed, and H3-cell and hour-bucket resolution rather than points and timestamps (§72).

## 67. Any signal that changes behaviour becomes an attack surface

> **THE PRINCIPLE**
> The moment a rider action automatically changes what the system recommends, that action becomes a
> lever an attacker can pull. Auto-demoting a route on cancellations turns cancellations into a
> weapon.

Hence the standing rule, which flows directly into the agent permissions of Part Eight:

**Behavioural signals raise tickets. Humans authorise changes.**

The loop in §84's architecture therefore closes into **governance review**, not into the model. That
is a deliberate architectural choice, and the diagram is drawn to make it visible.

---

# PART SEVEN — Responsible AI

## 68. Objectives beyond cost and time

v1 §15 scores journeys on cost, time, transfers and comfort. v1 §19 admits accessibility is not
modelled and that comfort and safety are crude. This part extends the objective set and, more
importantly, says **how** each new objective enters — because the mechanism matters more than the
list.

| Objective | How it enters | Why not otherwise |
|---|---|---|
| **Accessibility** | **Hard constraint** when declared | §69 |
| **Reliability** | Optimise an arrival **quantile**, not a mean | §70 |
| **Safety proxies** | Rider-selectable soft weight, infrastructure-based only | §70 |
| **Uncertainty** | Surfaced in the answer; feasibility as a probability | Part Five |
| **Fairness** | Audit dimension, not an objective term | §71 |
| **Transparency** | Already the product's spine; extended to trust and abstention reasons | v1 §5 |
| **User control** | Presets, sliders, override, history control, opt-out of personalisation | v1 §12 |

## 69. Hard constraints, soft weights

> **THE RULE**
> Access needs are hard constraints. Preferences are soft weights. Putting an access need into the
> weighted sum means it can be traded away for ₹15 — and a route that is "mostly step-free" is not a
> discount, it is a failure.

This answers the case named directly in the brief: a system must not silently optimise for cost when
doing so produces a poor accessibility outcome. Two mechanisms:

1. **When a need is declared** — step-free, seat likelihood, maximum walk distance — it becomes a
   filter applied *before* the Pareto stage, exactly where the budget and deadline filters already
   sit. A journey that fails it is never ranked.
2. **When nothing is declared** — the default preset carries an **accessibility floor**, and the
   interface must *disclose* when the cheapest option is materially worse on access:
   *"₹15 cheaper, but two flights of stairs and a 700 m walk."* Silence is the failure mode here; the
   rider cannot choose what they are not shown.

## 70. Reliability, safety, and the limits of proxies

**Reliability.** v1 §21 already states the goal: prefer a journey that is *reliably* 30 minutes over
one that averages 26 but sometimes takes 50. Once the quantile head of §55 exists this is a small
change — optimise the 80th-percentile arrival, or mean + λ·σ, rather than the mean. It also changes
the *product*, not just the metric: the promise becomes "you will almost certainly be there by 09:40"
instead of "about 34 minutes".

**Safety proxies, handled carefully.** Route safety is a real rider need, especially at night and
especially for women — and it is the most ethically dangerous objective in this document.

> **THE FAILURE MODE, NAMED**
> A "safety score" fitted on incident data or on aggregate demographics becomes algorithmic
> redlining: the system routes people around neighbourhoods, encodes a prejudice as a number, and
> presents it as an engineering fact.

Three rules, or the objective does not ship:

1. **Infrastructure only** — street lighting, footpath presence, staffed stations, active frontage,
   pedestrian volume by hour. Never the demographic composition of an area, and never crime
   statistics aggregated to residential geographies.
2. **Rider-selectable and off by default**, so nobody is routed by an assumption about them.
3. **Transparent** — the interface names the factor ("this route uses an unlit 600 m stretch after
   21:00"), so the rider judges, not the model.

## 71. Fairness, with Isopolis as the auditor

Two senses, and both are measurable.

**Distributional fairness — is service quality even?** Stratify JourneyMind's own quality metrics by
area type: confidence-band distribution, abstention rate, feasibility rate, top-1 acceptance. If the
system abstains twice as often in peripheral neighbourhoods — because feeds are thinner, footpaths
unmapped and informal modes invisible (Isopolis §43) — then it is systematically less useful to the
people who most need it. That is a finding, not a bug report.

> **WHY THIS IS CHEAPER THAN IT LOOKS**
> Part Three of v1 already builds the equity instrument. Isopolis computes a **Gini coefficient of
> access** across the city (v1 §42). The same machinery, pointed at *recommendation quality* instead
> of *travel time*, produces a Gini coefficient of **service quality** — an audit of JourneyMind by
> Isopolis. v1 §38 says the two systems are "the same machine pointed in two directions"; this is a
> third direction, and it costs almost nothing.

**Procedural fairness.** Identical inputs produce identical outputs. No price steering, no
differential ranking by inferred willingness to pay, no personalisation the rider cannot see and
switch off.

## 72. Behavioural data and sensitive inference

Journey traces are among the most revealing data a person generates, and in the Indian context the
revelations are specific:

- **Repeated origins and destinations give home and workplace** — and residence can proxy caste,
  religion, income and community.
- **A journey to a hospital or clinic** implies health information.
- **Journey timing** can reveal religious observance.
- **Destinations** can reveal political affiliation or sexual identity.

None of this has to be inferred deliberately to become a risk. It falls out of the raw traces.

Rules, binding on the whole rider decision layer:

1. **Never infer, derive or store sensitive attributes.** No rider segments that could act as proxies
   for them. This is not a hard problem to avoid — it is a decision not to.
2. **Aggregate before use.** A minimum cohort size (of the order of 50 riders across at least 5
   distinct days) before any behavioural statistic is computed or acted on.
3. **Coarsen space and time.** H3 cells and hour buckets, never coordinates and timestamps, in
   anything retained beyond the request.
4. **Minimise retention.** Raw traces for the shortest useful window; only coarsened, derived signals
   persist.
5. **Keep preference weights local.** A rider's learned weights are the most personal artifact the
   system produces, and have no reason to leave their device.
6. **Consent that is real** — specific, revocable and legible. Especially for any exploration (§52),
   where the rider is being experimented on.

Anchors: India's **DPDP Act 2023**, already cited in v1 §33, plus purpose limitation and data
minimisation as design defaults rather than compliance paperwork.

## 73. The tension this extension creates

> **HONESTY CHECK — this extension argues with v1**
> v1 §33 says: *"No personal data. Aggregate at the segment or zone level, never store an
> individual's identifiable trip history."*
>
> Part Four wants per-rider event sequences. **These are in genuine tension**, and pretending
> otherwise would be exactly the dishonesty this documentation is written against.

The resolution is not to weaken the rule but to design around it:

- **Session-scoped, not identity-scoped.** Attribution in §49 needs a *recommendation* and its
  outcome, not a person's life history. Most of the value is available from a pseudonymous
  per-request identifier that is never joined across sessions.
- **On-device sequence, off-device aggregate.** Preference learning (original RQ4) runs locally; only
  aggregated, k-anonymous statistics leave the device.
- **Separate the consented study from the product.** The 20–30 person user study of v1 §16 is an
  explicitly consented research context and can hold richer per-participant data under ethics
  approval. The deployed product cannot, and must not inherit that assumption.

---

# PART EIGHT — Agentic JourneyMind (future)

## 74. What agentic means here

> **NOT NOW**
> The current MVP is **not agentic**, and this document does not claim it is. It is a deterministic
> pipeline: a request enters, fixed stages run in a fixed order, an answer leaves. Nothing chooses
> its own next action. Everything in Part Eight is design for a **future** system and is tagged
> **NEW** throughout.

Agentic here means a narrow, useful thing: components that can **decide what to do next within a
bounded task**, call tools, and hand results to each other — not a conversational assistant, and not
an autonomous system that acts on a rider's behalf.

The tasks worth delegating are the ones that are currently manual, tedious and done badly under
deadline: checking whether a feed is still live, noticing a schema change, spotting that two sources
disagree, finding what is missing.

## 75. Six agents

| Agent | Its one job |
|---|---|
| **Data Agent** | Discover and monitor approved sources; check freshness, integrity and schema; flag conflicts between sources |
| **Prediction Agent** | Run the travel-time models and produce predictions with intervals |
| **Route Agent** | Generate candidate journeys, apply constraints and the Pareto filter, rank, and assemble the explanation |
| **Verification Agent** | Run the §60 checks; issue a verdict; downgrade confidence or veto |
| **Monitoring Agent** | Watch aggregated outcomes and cohort anomalies; run the §65 triage; raise tickets |
| **Policy / Governance Agent** | Enforce policy, gate releases, and write the audit record |

Explanation is not a seventh agent. It is a *capability* of the Route and Verification agents, because
an explanation generated by a component that did not make the decision is a plausible story rather
than a reason — which is precisely the failure mode Part Five exists to prevent.

## 76. Why none gets unrestricted autonomy

Seven reasons, in descending order of how much they should worry you:

1. **Consequence.** Money, time, and someone's physical position in a city at night.
2. **The v1 boundary is a safety property.** "Recommends but does not reserve or pay" is the single
   most important invariant in the system. Agency must not erode it by convenience.
3. **Untrusted input.** Agents that read open feeds are reading attacker-writable text (§63).
   Autonomy plus injection is an exploit path, not a feature.
4. **Compounding error.** Chained steps multiply error rates, and an unverified step feeds the next.
5. **Auditability.** An agent that can change its own inputs makes post-hoc explanation impossible —
   and the explanation *is* the product (v1 §5).
6. **Separation of duties.** The component that checks must not be the component that acts.
7. **Reversibility.** Reading a feed is reversible. Booking a cab is not.

> **THE ORGANISING PRINCIPLE**
> **Autonomy is granted in proportion to reversibility and blast radius.** Read-only, request-scoped,
> easily undone → broad autonomy. Persistent, cross-rider, or irreversible → human approval.

## 77. The permission matrix

Least privilege, per agent. Every row is a design commitment.

### Data Agent

| | |
|---|---|
| **May read** | Sources on the approved allow-list; the `SOURCES.md` ledger; its own quarantine store |
| **May write** | Quarantine registry; freshness and integrity reports; conflict flags |
| **Tools** | HTTP fetch to allow-listed hosts only; GTFS validator; schema differ; hasher |
| **May decide** | Whether a source is fresh, valid and self-consistent; whether two sources disagree |
| **Needs human approval** | **Adding any source to the production graph.** Changing a freshness budget. Changing the allow-list |
| **Forbidden** | Writing to the production graph. Executing anything found inside fetched content. Following URLs discovered in feed data. Fetching outside the allow-list |
| **Logged** | Every fetch: URL, timestamp, bytes, hash, validator verdict, decision |

### Prediction Agent

| | |
|---|---|
| **May read** | The graph, features, model weights, time context |
| **May write** | Predictions and intervals into a **request-scoped** buffer |
| **Tools** | Model inference only |
| **May decide** | Which registered model to use; when to widen intervals; when to declare out-of-support |
| **Needs human approval** | Promoting a retrained model. Changing the default model |
| **Forbidden** | Writing training data. Modifying weights. Persisting anything beyond the request |
| **Logged** | Model id and weight hash, feature snapshot hash, interval widths, clamp count |

### Route Agent

| | |
|---|---|
| **May read** | Predictions, fares, constraints, rider preferences for this request |
| **May write** | The candidate set, rankings and the explanation, request-scoped |
| **Tools** | Search, Pareto filter, scoring |
| **May decide** | Which candidates to generate; how to rank under the given weights |
| **Needs human approval** | — (nothing it does persists) |
| **Forbidden** | **Relaxing a rider's stated budget, deadline or access constraint.** Contacting a provider. Booking. Spending. Overriding a Verification Agent verdict |
| **Logged** | Candidate count, filter decisions, frontier, final ranking, weights used |

The forbidden row here is the important one. An agent that "helpfully" stretches a ₹250 budget to
₹280 because the journey is much better has broken the product's core promise — and would do so
invisibly.

### Verification Agent

| | |
|---|---|
| **May read** | Everything relevant to the recommendation, **plus the independent evidence of §61** |
| **May write** | Verdicts, confidence bands, flags, abstention decisions |
| **Tools** | The nine checks; the calibration store |
| **May decide** | **Downgrade confidence. Attach a warning. Veto the recommendation. Abstain.** |
| **Needs human approval** | Changing a check's threshold. Disabling a check |
| **Forbidden** | Modifying the model, the data, or the *content* of the recommendation. **It may veto; it may not edit.** Being overridden by the Route Agent |
| **Logged** | Every check, its inputs, its verdict, and the final band with its driving signals |

### Monitoring Agent

| | |
|---|---|
| **May read** | **Aggregated, k-anonymous** outcome statistics; drift metrics; feed health |
| **May write** | Alerts, tickets, triage findings |
| **Tools** | Statistics, anomaly detection, the §65 triage checklist |
| **May decide** | That an anomaly exists, and which triage branch it falls in |
| **Needs human approval** | **Every action that follows.** Retraining. Demoting a route. Disabling a source |
| **Forbidden** | Reading individual rider histories. Triggering retraining. Changing rankings. Acting on a cohort below the minimum size |
| **Logged** | Every alert with its evidence, cohort size and triage outcome |

This is §67 made structural: the agent that detects has no power to change.

### Policy / Governance Agent

| | |
|---|---|
| **May read** | Policy configuration; the audit log; release evidence |
| **May write** | Policy decisions; **append-only** audit entries |
| **Tools** | Policy evaluation; audit writer |
| **May decide** | Block a release. Block a class of recommendation. Require human sign-off |
| **Needs human approval** | **Any change to policy itself** — it enforces policy, it does not author it |
| **Forbidden** | Approving its own policy changes. Deleting or editing audit records. Granting itself permissions |
| **Logged** | Itself, immutably |

## 78. Untrusted content is data, never instruction

The concrete version of the abstract rule — because in this system the attack surface has names:

- GTFS `stop_name`, `route_long_name`, `agency_name`, `trip_headsign`
- OSM `name`, `description`, `note`, and arbitrary tag values
- Anything in a portal's HTML, changelog or README

Every one of those is a string written by someone else, and in an agentic pipeline it would flow into
a model's context.

Controls:

1. **Parse to schema first.** External data becomes typed fields before any model sees it; free text
   is carried as a display string with a provenance tag, never as prose in a prompt.
2. **Neutralise free-text fields** used in any model context — length caps, stripped control and
   directive-shaped content, and clear delimiting that marks the region as untrusted data.
3. **No URL following from content.** Links discovered inside fetched data are recorded, never
   fetched.
4. **Egress allow-list**, so that a successful injection still cannot reach anywhere new.
5. **Provenance-tagged context** — every span in an agent's context carries its origin, and origin
   determines authority.

## 79. Degrade to deterministic

> **THE KILL SWITCH THAT MATTERS**
> The entire agentic layer must be removable at runtime, falling back to the deterministic v1
> pipeline. **The MVP is the safe mode.**

This is the strongest argument for building the extension in this order. Because v1 is a fixed
pipeline with no autonomy, it is always available as the thing to fall back *to*. A system designed
agentic-first has no such floor.

---

# PART NINE — Governance

## 80. The framework

Five domains plus audit. For each: what policy asserts, what control enforces it, what evidence
proves it, and how often it is checked.

### Data

| Policy | Control | Evidence | Cadence |
|---|---|---|---|
| Every source is identified, licensed and permitted | `SOURCES.md` ledger: URL, date accessed, licence, hash, purpose — v1 §25 | Ledger entry | On add; quarterly review |
| Licence terms are honoured, including ODbL share-alike and non-commercial clauses | Licence field gates use; attribution in README, footer and report — v1 §33 | Ledger + published attribution | On add; on publication |
| Sources meet a freshness budget | Per-source budget; age surfaced as a trust signal | Fetch log | Per request |
| Source integrity is verified | Hash and schema check at fetch | Fetch log | Per fetch |
| Uncertainty is labelled, never implied | Provenance tags on every number — **BUILT** | Response payload | Per response |
| No personal data beyond the minimum | §72 rules; k-anonymity floor | Retention config; cohort sizes | Per aggregate |

### Model

| Policy | Control | Evidence | Cadence |
|---|---|---|---|
| Accuracy is measured on a temporal split, never a random one | v1 §16 — **BUILT** (`scripts/evaluate.py`, `EVALUATION.md`) | Evaluation report | Per training run |
| The no-graph ablation is always reported, whichever way it goes | Baseline 4 — **BUILT**, and currently reporting *against* the GNN | `EVALUATION.md` | Per training run |
| Confidence is calibrated, not asserted | Reliability diagram, ECE, risk–coverage | Calibration report | Per model version |
| Drift is monitored | Residual monitoring by corridor and hour | Drift dashboard | Continuous |
| Every prediction is attributable to a model version | Weight hash in every record | Audit entry | Per request |
| Serving matches training | Parity test — **BUILT** (`tests/test_model_parity.py`, agreement asserted to 1e-4) | Test run | Per build |

### Recommendation

| Policy | Control | Evidence | Cadence |
|---|---|---|---|
| Stated constraints are never violated silently | Hard filter; near-misses labelled with the limit they break — **BUILT** | Response payload | Per response |
| Every recommendation carries a confidence and its reasons | §57 decomposition | Response payload | Per response |
| Low-confidence recommendations abstain or warn | §58 | Abstention log | Per response |
| Responsible-AI checks run before display | §69–§71 | Check log | Per response |
| Every number carries provenance | **BUILT** | Response payload | Per response |

### Agent

| Policy | Control | Evidence | Cadence |
|---|---|---|---|
| Least privilege per agent | §77 matrix, enforced by credentials not convention | Permission config | Per release |
| Irreversible actions need human approval | Approval gates | Approval log | Per action |
| Spending authority is zero by default | Hard-coded; no payment credential anywhere in the agent path | Config | Per release |
| Tool and egress access is allow-listed | Allow-list; no dynamic discovery | Config | Per release |
| External content never carries instruction authority | §78 | Ingestion log | Per fetch |
| The agentic layer can be disabled | §79 | Kill-switch test | Per release |

### User

| Policy | Control | Evidence | Cadence |
|---|---|---|---|
| Consent is specific, informed and revocable | Consent record | Consent log | Per rider |
| The rider can see, export and delete their data | Self-service | Request log | On request |
| Personalisation can be switched off | Preference control | Setting | Per rider |
| The rider is told when confidence is low | §58 | Response payload | Per response |
| The rider is never silently experimented on | Disclosed exploration only | Study protocol | Per study |

### Audit

Every recommendation produces one immutable record. §81 defines it.

## 81. The Recommendation Record

The concrete artifact that makes everything above auditable. It is not a new invention — it is the v1
§6 pipeline trace, which the service **already builds and returns today**, promoted from a debugging
aid to a governance object.

| Field group | Contents |
|---|---|
| **Request** | Request id, timestamp, origin and destination at cell resolution, budget, deadline, preset or weights, declared needs |
| **Data** | Every source used: id, version, hash, fetch time, age at use, licence |
| **Model** | Model key, weight hash, feature snapshot hash, clamp count, interval widths |
| **Candidates** | Number generated, filter decisions, Pareto frontier, dominated set |
| **Recommendation** | Chosen journey, alternatives, cost with provenance, time with intervals, constraint status with headroom |
| **Trust** | Gate results, all nine check verdicts, confidence band, driving signals, abstention decision and reason |
| **Responsible AI** | Which checks ran, what they found, what was disclosed to the rider |
| **Agent** *(future)* | Each agent action: agent id, inputs, tools called, decision, approvals obtained |
| **Outcome** | The rider event sequence, the attributed cause and its confidence, or `UNATTRIBUTED` |

Properties: **append-only**, hash-chained, retained under §72's limits, and queryable by request id,
so that *"why did the system say that, on what evidence, on what day?"* has an answer.

Roughly 60% of these fields already appear in the returned `pipeline` object. The governance work is
persistence, immutability, and the trust and outcome groups — not new instrumentation.

## 82. Release gates and incidents

**A model ships only if** parity holds; the temporal split is respected; the no-graph ablation is
reported; calibration meets its threshold; the spatial hold-out (original RQ5) has been run; the model
card is updated; and a rollback path exists.

**Incident severity:**

| Sev | Definition | Response |
|---|---|---|
| **1** | Riders are being given infeasible or unsafe recommendations | Abstain globally for the affected class; investigate; **do not retrain** until the cause is known |
| **2** | Confidence is materially miscalibrated, or a source is compromised | Widen intervals; downgrade bands; quarantine the source |
| **3** | Drift beyond threshold, or a persistent unexplained behavioural anomaly | Triage per §65; schedule retraining if and only if the cause is drift |
| **4** | Data-quality degradation with no rider impact yet | Ticket; fix the pipe |

The rule that binds them: **evidence before retraining.** §65 explains why.

## 83. Who holds which hat

> **REALISM**
> This is a student research project, not an enterprise with a compliance function. Naming six roles
> does not conjure six people. The point of naming them is that when one person wears four hats,
> everyone can see which hat they had on when they made a call — and that a decision made wearing the
> "ship it" hat was not also the review of that decision.

| Hat | Owns | Cannot also |
|---|---|---|
| **Data steward** | The ledger, licences, freshness | Approve their own source additions |
| **Model owner** | Training, evaluation, calibration | Sign off their own release |
| **Verification owner** | The checks, thresholds, abstention policy | Be overruled by the model owner |
| **Security owner** | Threat model, integrity, incident response | Be the sole reviewer of an incident they caused |
| **Responsible-AI reviewer** | §68–§73 checks, fairness audit | Report to the person shipping the feature |
| **Rider advocate** | Consent, control, clarity, the abstention experience | — |

---

# PART TEN — The whole thing

## 84. Unified architecture

```
                         RIDER
                           |
                    Journey request
                           |
        +------------------v------------------+
        |      POLICY / GOVERNANCE GATE       |   consent - declared needs
        |                                     |   policy - rate limits
        +------------------+------------------+
                           |
    #======================v======================#
    #   JOURNEYMIND CORE - UNCHANGED FROM v1      #
    #                                             #
    #   Multi-modal graph  ->  GNN  ->  Optimiser #
    #   v1 §7 nodes/edges     §14      §15        #
    #                                 Pareto+rank #
    #======================+======================#
                           |  candidate journeys + predictions
        +------------------v------------------+
        |   VERIFICATION  §60                 |   nine checks against
        |   gates -> checks -> verdict        |   INDEPENDENT evidence §61
        +------------------+------------------+
                           |
        +------------------v------------------+
        |   TRUST / CONFIDENCE  §55           |   P(success), bands,
        |                                     |   rank stability
        +------------------+------------------+
                           |
        +------------------v------------------+
        |   SECURITY CONTROLS  §63            |   integrity - freshness
        |                                     |   anomaly - rate limits
        +------------------+------------------+
                           |
        +------------------v------------------+
        |   RESPONSIBLE-AI CHECK  §68-71      |   access - reliability
        |                                     |   disclosure - fairness
        +------------------+------------------+
                           |
              +------------v-------------+
              |  RECOMMENDATION + REASON |   or  ABSTENTION + REASON
              |  + confidence + evidence |
              +------------+-------------+
                           |
        +------------------v------------------+
        |   RIDER ACTION  §47                 |   accept - alt - reject
        |                                     |   book - cancel - switch
        |                                     |   complete - abandon
        +------------------+------------------+
                           |
        +------------------v------------------+
        |   OUTCOME MONITORING  §49           |   attribute the cause
        |                                     |   (or mark UNATTRIBUTED)
        +------------------+------------------+
                           |
        +------------------v------------------+
        |   FEEDBACK + ANOMALY SIGNAL  §66    |   cohort-level only
        +------------------+------------------+
                           |
        +------------------v------------------+
        |   GOVERNANCE / REVIEW  §80          |   <-- the loop closes HERE,
        |   a human decides what changes      |       not at the model
        +------+-----------+-----------+------+
               |           |           |
    authorised | authorised| authorised|       every arrow is a
      data fix |  retrain  |  policy   |       HUMAN decision (§67)
               v           v           v
```

Two features of this drawing carry the argument:

1. **The core is boxed and marked unchanged.** Everything new wraps it. Remove every wrapper and v1
   still runs — which is §79's kill switch drawn rather than described.
2. **The loop closes at governance, not at the model.** There is no arrow from rider behaviour back
   into training. That absence is the single most important line in the diagram (§67).

### Where the agents sit

Agents do not replace layers; they **operate** them. Each agent attaches to the layer it already
corresponds to, with the permissions of §77:

| Layer | Agent | Autonomy |
|---|---|---|
| Data ingestion (before the core) | **Data Agent** | Read and propose; never writes production |
| Core: prediction | **Prediction Agent** | Request-scoped only |
| Core: routing | **Route Agent** | Request-scoped; cannot relax constraints |
| Verification | **Verification Agent** | May veto, may not edit |
| Outcome monitoring | **Monitoring Agent** | Raises tickets only |
| Policy gate and governance | **Policy Agent** | Enforces; does not author |

The architecture is unchanged by their arrival. That is the test: if adding agents required redrawing
the pipeline, the pipeline was wrong.

## 85. Three tiers

| | **CURRENT MVP** | **RESEARCH EXTENSION** | **FUTURE AGENTIC SYSTEM** |
|---|---|---|---|
| **Status** | Built and running | One further project cycle | Design only |
| **Autonomy** | None — fixed pipeline | None — fixed pipeline | Bounded, per §77 |
| **Contains** | Graph, GNN, optimiser, explanations, provenance labels, service hours, constraint headroom | Uncertainty head, verification layer, trust score, abstention, rider decision layer, anomaly monitoring, responsible-AI objectives, Recommendation Record | Six agents, permission enforcement, tool allow-lists, approval gates, injection defence |
| **Answers** | RQ1–RQ5 | RQ6–RQ9, RQ12 | RQ10, RQ11 |
| **Risk if rushed** | — | Claiming calibration without outcome data | Claiming autonomy is safe without the permission layer |

The extension is deliberately buildable **without any agents at all**. Trust, verification, abstention
and outcome attribution are all deterministic mechanisms. Agents are an implementation strategy for
later, not a prerequisite — and saying so protects the research from depending on the least mature
part of it.

## 86. Research progression

> **A NOTE ON NUMBERING**
> v1 §22 already defines RQ1–RQ5, and they are not abandoned. New questions therefore start at RQ6.
> The brief's seven-item progression maps onto them in the right-hand column below.

### Carried forward, unchanged

| ID | Question | Status |
|---|---|---|
| **RQ1** | Does message passing over a multi-modal transport graph improve travel-time prediction over an identical model without graph structure? | **Answered on the bundled data: no.** MLP 0.228 MAE vs GAT 0.238 vs GraphSAGE 0.260, reported in `EVALUATION.md` |
| **RQ2** | How far does the benefit extend — 2-hop vs 1-hop, and does 3-hop hurt through over-smoothing? | Open |
| **RQ3** | Does more accurate prediction translate into better recommendations, or does the optimiser wash it out? | Open — **and now measurable end-to-end** via §51 and §56 |
| **RQ4** | How many observed choices are needed to recover a user's preference weights? | Open — Part Four supplies the data |
| **RQ5** | Does the model generalise to a held-out spatial region? | Open — **and now doubles as a poisoning-detection test** (§63) |

### New

| ID | Question | Hypothesis | Method | Metric | A negative result would mean | Brief |
|---|---|---|---|---|---|---|
| **RQ6** | Can JourneyMind detect when its own predictions are unreliable? | Interval width, cold-start flags, cross-model disagreement and corridor-hour history predict error better than chance | Quantile head plus ensemble spread; correlate against held-out error | AUROC of error prediction; ECE | The signals are uninformative, and any confidence display would be theatre — a genuinely useful finding | RQ2 |
| **RQ7** | Does uncertainty plus provenance plus verification reduce unsupported recommendations? | Yes, measurably, at a cost in coverage | Ablate each verification check; measure both sides | Unsupported-recommendation rate; risk–coverage area | Verification only costs coverage, meaning the pipeline was already sound | RQ3 |
| **RQ8** | Can behavioural feedback distinguish recommendation failure from ordinary rider choice? | Context deltas (§49) attribute a useful fraction of abandonments | Instrumented study with a held-out labelled subset from rider-reported reasons | Attribution accuracy; **size of the UNATTRIBUTED bucket** | Behaviour is too confounded to serve as a quality signal — important, and rarely reported | RQ4 |
| **RQ9** | Can abnormal behaviour and data patterns be separated into data quality, drift and manipulation? | Yes, given provenance and integrity signals alongside behavioural ones | Controlled injection: stale feeds, simulated drift, synthetic poisoning, Sybil cancellations | Per-class detection rate; false-positive rate; **time to the correct triage branch** | The classes are empirically inseparable and the §64 taxonomy is conceptual only | RQ5 |
| **RQ10** | How should agent permissions and human approval be designed for an agentic mobility system? | Reversibility and blast radius predict where approval must sit | Design study; red-team the matrix with injection and confused-deputy scenarios | Escalations blocked; unauthorised actions still reachable | The matrix is either too tight to be useful or too loose to be safe | RQ6 |
| **RQ11** | Can a governance layer make an agentic JourneyMind auditable, controllable and trustworthy? | The Recommendation Record supports post-hoc reconstruction of any decision | Adversarial audit: reconstruct 50 decisions from records alone | Reconstruction completeness; audit time; gaps found | Records are insufficient — and what is missing is the contribution | RQ7 |
| **RQ12** | Does trust-aware abstention improve **rider outcomes**, not just model metrics? | Abstaining and offering the robust option raises completion rate | A/B within the consented study | Completion rate; constraint satisfaction on realised trips; rider-reported trust | Abstention annoys riders more than errors do — the most useful negative result available here | *(added)* |

RQ12 is the one that closes the loop. Everything else measures the machinery; RQ12 asks whether the
machinery helped a person get somewhere. Without it, the extension is defensible as engineering and
unproven as a contribution.

## 87. Roadmap

v1 §45 sequences weeks 1–16. This continues into a second cycle.

| Phase | Focus | Deliverable |
|---|---|---|
| **Weeks 17–19** | Uncertainty | Quantile head trained; per-leg intervals; Monte Carlo journey sampling with corridor correlation; rank stability computed |
| **Weeks 19–22** | Verification | Nine checks implemented; gates enforced; independence audited (§61); RQ7 ablation running |
| **Weeks 21–24** | Trust | Rule-based band v1, labelled uncalibrated; decomposed reasons in the interface; abstention flow; risk–coverage baseline |
| **Weeks 23–26** | Rider decision layer | Event schema; context snapshots; attribution engine; UNATTRIBUTED measured; RQ8 study instrumented |
| **Weeks 25–28** | Monitoring and security | Cohort monitors; §65 triage; controlled injection experiments for RQ9; integrity pinning and hash verification |
| **Weeks 27–30** | Responsible AI | Accessibility as a hard constraint; reliability-quantile objective; disclosure copy; Isopolis-based fairness audit |
| **Weeks 29–32** | Governance | Recommendation Record persisted and hash-chained; model cards; release gates; RQ11 adversarial audit |
| **Weeks 31–34** | Calibration and write-up | Refit trust on collected outcomes; reliability diagrams; RQ12 A/B; honest limitations |
| **Design only** | Agentic | §75–§79 as specification and red-team exercise; **no autonomous execution in this cycle** |

> **THE DEPENDENCY THAT WILL BITE YOU, AGAIN**
> v1 warned that everything depends on probe collection starting in week one. The same shape recurs:
> **the trust score cannot be calibrated without completed-trip outcomes, and outcomes cannot be
> collected until the rider decision layer is instrumented.** If instrumentation slips past week 26,
> the cycle ends with an uncalibrated confidence score — which is to say, with the exact thing this
> document argues against.

## 88. Project titles

| # | Title | What it foregrounds | Choose it if |
|---|---|---|---|
| **1** | **Knowing When Not to Recommend: Trust, Verification and Governance for Graph-Based Mobility AI** | Abstention as the contribution | You want the intellectual core in the title. The strongest option — it states a finding, not a topic |
| **2** | **From Prediction to Permission: A Trust and Governance Architecture for Agentic Mobility Recommendation** | The full arc, GNN through agents | The agentic and governance work is the emphasis |
| **3** | **Grounded Mobility AI: Provenance, Uncertainty and Abstention in Multi-Modal Journey Recommendation** | Evidence and grounding | You want the OSINT and provenance heritage of v1 Part Two visible |
| **4** | **Cancelled: Separating Recommendation Failure from Rider Choice in a Multi-Modal AI Advisor** | The behavioural contribution | The rider decision layer is the novel part and you want a memorable title |
| **5** | **JourneyMind: A Trust-Aware and Secure Framework for Multi-Modal Journey Recommendation under Adversarial and Uncertain Data** | Everything, formally | You need a conventional, scope-complete thesis title |

Recommendation: **title 1** for a paper or dissertation; **title 5** for a formal project registration
where the title must enumerate scope.

## 89. Explain the extension to a five-year-old

Following v1 §24 — everyday words only.

**What is JourneyMind today?**
It is a helper that knows all the ways to get across the city — walking, buses, trains, bikes you can
call — and adds them up for you. You tell it how much money you have and how long you have got, and
it says: *do this one, and here is why*.

**What problem does trust solve?**
Sometimes the helper is guessing. Maybe nobody has ever timed that road. Maybe the train timetable it
read is old. Right now it says every answer in the same confident voice. Trust means the helper learns
to say *"I am very sure"* or *"I am not very sure, and here is why"* — and sometimes to say *"I would
rather not guess; take this safer one instead."* A helper that admits when it does not know is a
better helper than one that is always sure.

**Why do cancellations matter?**
If you ask for a way to go and then change your mind, that could mean lots of different things. Maybe
the helper was wrong. Maybe the price went up. Maybe no bike came. Maybe your friend just offered you
a lift. If the helper thinks *every* time you change your mind means it was wrong, it will start
believing silly things. So it has to look at what actually changed before it decides whose fault it
was — and quite often the honest answer is *"I cannot tell."*

**Why does security matter?**
The helper reads maps and timetables that other people write. Someone could write something untrue on
purpose — a road that does not exist, a train that is not running — to send everybody the wrong way.
So the helper has to check that the things it reads are really the things it read yesterday, and
notice when they change strangely.

**Why does agentic AI create new risks?**
Right now the helper only *tells* you things. If we let it *do* things — book your ride, spend your
money, change its own maps — then a mistake stops being a wrong sentence and starts being a wrong
charge on your card, or you standing somewhere alone at night. So the helper is allowed to look things
up on its own, but it must ask a grown-up before doing anything that costs money or cannot be undone.

**Why do we need governance?**
Because someone has to be able to ask, next week, *"why did you tell that person to do that?"* — and
get a real answer. Governance is keeping the receipt for every single answer: what the helper read,
how sure it was, what it checked, what it decided, and what happened next.

## 90. What already exists and what is new

The honest ledger. Anything marked **NEW** is a proposal, and has not been built or tested.

### Already in the v1 documentation and in the running system

| Capability | Where | Status |
|---|---|---|
| Multi-modal graph: road, transit, transfer, ride, access edges | v1 §14; `backend/app/graph/builder.py` | **BUILT** |
| GNN travel-time prediction, GraphSAGE and GAT | v1 §14; `backend/app/models/gnn_*.py` | **BUILT** |
| Six models behind one interface | v1 §17; `backend/app/models/loader.py` | **BUILT** |
| Yen's k-shortest, time-dependent, five weightings | v1 §15; `backend/app/routing/kshortest.py` | **BUILT** |
| Constraint filter and Pareto frontier | v1 §15; `backend/app/optimisation/` | **BUILT** |
| Deterministic explanations | v1 §5; `backend/app/services/explain.py` | **BUILT** |
| Fare provenance labels, worst-label propagation | v1 §33; `backend/app/models/fares.py` | **BUILT** |
| Constraint headroom and `cost_at_risk` | `backend/app/optimisation/constraints.py` | **BUILT** |
| Labelled near-misses instead of silent failure | v1 §20; `backend/app/optimisation/constraints.py` | **BUILT** |
| Cold-start clamping with a reported clamp count | v1 §18; `backend/app/models/gnn_numpy.py` | **BUILT** |
| Per-leg reliability score | `backend/app/routing/journey.py` | **BUILT** |
| Service hours; out-of-service caveat on the answer | `backend/app/data/provider.py`, `routing/costs.py` | **BUILT** |
| Pipeline trace returned with every response | v1 §6; `backend/app/api/serialise.py` | **BUILT** |
| Source ledger with licence, date and hash | v1 §25; `SOURCES.md` | **BUILT** |
| `OPEN` / `CONDITIONAL` / `CLOSED` source tags | v1 §25 | **BUILT** |
| Temporal split; no-graph ablation reported | v1 §16–17; `EVALUATION.md` | **BUILT** |
| Train/serve parity assertion | `tests/test_model_parity.py` | **BUILT** |
| Request validation and rate limiting | `backend/app/schemas.py`, `config.py` | **BUILT** |
| Search caps against runaway queries | `backend/app/routing/kshortest.py` | **BUILT** |
| "Recommends but does not reserve or pay" | v1 §19–20 | **BUILT** (as policy) |
| DPDP-aligned no-personal-data stance | v1 §33 | **BUILT** (as policy) |
| Uncertainty-aware routing named as future scope | v1 §21 | **PARTIAL** — named, not built |
| Preference learning by discrete choice named as v2 | v1 §15 | **PARTIAL** — named, not built |
| Isopolis Gini coefficient of access | v1 §42 | **PARTIAL** — specified in v1 Part Three |

### New in this document

| Capability | Section | Status |
|---|---|---|
| Rider event vocabulary and decision records | §47 | **NEW** |
| Ten-cause attribution with context-delta evidence | §49 | **NEW** |
| Three separated learning channels | §50 | **NEW** |
| Set quality vs rank quality as distinct metrics | §51 | **NEW** |
| Selection-bias treatment and exploration policy | §52 | **NEW** |
| Trust signal taxonomy across seven families | §54 | **NEW** |
| Hard gates separated from scored confidence | §55 | **NEW** (G2, G3 exist as mechanisms) |
| P(success) as the calibrated target | §55 | **NEW** |
| Quantile head and correlated Monte Carlo | §55 | **NEW** |
| Rank stability under uncertainty | §54 | **NEW** |
| Calibration acceptance criteria; risk–coverage | §56 | **NEW** |
| Confidence bands tied to empirical claims | §57 | **NEW** |
| Abstention with four permitted forms | §58 | **NEW** |
| "Unsupported recommendation" definition and taxonomy | §59 | **NEW** |
| Nine-check verification layer | §60 | **NEW** (V5, V7 exist as mechanisms) |
| The independence rule | §61 | **NEW** |
| Threat model by entry point | §63 | **NEW** |
| Probe-data poisoning as the project-specific threat | §63 | **NEW** |
| Data quality vs reliability vs attack taxonomy | §64 | **NEW** |
| Triage order requiring positive evidence for attack | §65 | **NEW** |
| Behaviour as a security sensor | §66 | **NEW** |
| "Any signal that changes behaviour is an attack surface" | §67 | **NEW** |
| Accessibility as a hard constraint, not a weight | §69 | **NEW** |
| Reliability as an arrival quantile | §70 | **NEW** (named in v1 §21) |
| Infrastructure-only safety proxies with anti-redlining rules | §70 | **NEW** |
| Isopolis Gini repurposed to audit service quality | §71 | **NEW** |
| Sensitive-inference rules for behavioural data | §72 | **NEW** |
| The v1 §33 tension, named and resolved | §73 | **NEW** |
| Six agents with a full permission matrix | §75, §77 | **NEW** |
| Autonomy proportional to reversibility | §76 | **NEW** |
| Injection defence for GTFS and OSM free text | §78 | **NEW** |
| Degrade-to-deterministic kill switch | §79 | **NEW** |
| Five-domain governance framework | §80 | **NEW** |
| The Recommendation Record | §81 | **NEW** (~60% of fields already emitted) |
| Release gates and incident severities | §82 | **NEW** |
| RQ6–RQ12 | §86 | **NEW** |

---

> **CLOSING NOTE**
> v1 ended by saying its test was whether a reader who knows nothing about graphs or neural networks
> finishes it saying *"I understand the problem now."* This document's test is different: whether a
> reader finishes it able to say **what JourneyMind would refuse to answer, and why.** A system that
> cannot answer that question does not have a trust layer — it has a confidence display.
