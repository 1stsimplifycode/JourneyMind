# Expected-Cost Mobility Intelligence

**A commercial positioning document for JourneyMind as an enterprise platform.**

> **Status of this document.** Wipro is presented here as a *potential*
> customer, implementation partner or distribution channel. Wipro does not use,
> endorse, sponsor or have any relationship with this project. Every financial
> figure below is a worked example built on stated assumptions, not a
> measurement of any organisation. Assumptions are labelled inline.

---

## 1. The problem

An enterprise with employees, campuses and a transport budget buys mobility in
one of three ways: a contracted fleet, reimbursed ride-hailing, or a
combination. In all three, the spend is managed against **advertised prices and
published schedules**.

That is the wrong denominator, and the gap is structural rather than
occasional:

- A booking that is accepted and then cancelled costs the employee 5–8 minutes
  and produces a re-request into a market that has just demonstrated it is
  tight — so the replacement ride is dearer.
- A booking that never matches costs the wait before the app gives up.
- Neither event appears in a spend report, because **no money moved**. The cost
  landed on the employee's calendar and on the next fare, and conventional
  transport reporting is blind to both.

The consequence: an organisation optimising on advertised fare will
systematically over-select the providers that fail most often, because failing
is what makes them cheap to advertise.

> **The one-line version.** Enterprises manage transportation on the price that
> is quoted. They pay the price that is realised. Nothing in their stack
> measures the difference.

---

## 2. The solution

A platform that predicts the **expected cost and reliability** of every
transport option for every trip, and manages an organisation's mobility on
those numbers instead.

Concretely, for one trip it produces this — and the recommendation is not the
cheapest row:

| Option | Advertised | Cancellation risk | Completes | **Expected cost** |
|---|---|---|---|---|
| Bike taxi | ₹67 | 20% | 91% | **₹68** |
| Auto | ₹99 | 12% | 94% | **₹102** |
| Cab | ₹158 | 10% | 98% | **₹164** |
| Bus | ₹11 | — | 100% | **₹11** |
| Carpool | ₹57 | 30% | 47% | ₹37 *(blended — 53% of the time you end up on the bus)* |

*Illustrative output from the running system on simulated provider data.*

The differentiator is not the comparison. It is that the platform can say:

> *"Provider A was not recommended despite a lower advertised fare because its
> predicted cancellation probability is 30% and its chance of completing at all
> is 47%, which raises the expected cost of actually arriving."*

That sentence — generated, explainable, auditable — is the product.

### Two layers

**Layer 1 — Mobility Intelligence Engine.** Multimodal routing over a city
graph, GNN travel-time prediction, calibrated reliability prediction,
expected-cost computation over the booking lifecycle, constrained
multi-objective optimisation, explainable recommendations.

**Layer 2 — Enterprise Mobility Platform.** The same engine pointed at a
population: spend, cost of failure, provider scorecards, SLA breaches, demand
by campus and hour, and a governance log of every AI decision.

The consumer comparison view is the **demonstration layer**. The product is the
engine and the enterprise platform above it.

---

## 3. Why this is a Wipro-shaped opportunity

The claim is not "Wipro needs a taxi app". It is that this is a **compact,
demonstrable instance of the hardest problem in enterprise AI**: putting a
model in the path of a decision that spends real money, and being able to
defend it afterwards.

| Wipro capability | What this platform exercises |
|---|---|
| **Enterprise AI / AI services** | A production decision system where a wrong prediction has a rupee cost, not a wrong sentence |
| **Data & analytics** | An OSINT-sourced data catalogue with per-field provenance and licence tracking (`SOURCES.md`) |
| **Cloud & engineering** | Containerised, single-service, deployable to a free tier; models exported for CPU-only serving |
| **Responsible AI** | Every recommendation carries confidence, reasons and data provenance; the system can abstain |
| **Cybersecurity** | A concrete threat model for poisoned transport data, manipulated feeds and adversarial graph features |
| **AI governance** | An append-only decision record: what was recommended, on what evidence, by which model version, with what confidence |
| **Workplace / employee experience** | Employee transportation is a real line item at every campus |

### The progression this demonstrates

The mobility problem is the vehicle; the capability ladder is the point.

```
Trustworthy enterprise AI      a decision system with money at stake
   -> AI reliability           calibrated confidence, measured not asserted
   -> error detection          verification before a recommendation is shown
   -> AI security              poisoned feeds, adversarial graph features
   -> responsible AI           explainability, accessibility, fairness
   -> agentic AI               agents that investigate, within permissions
   -> agentic AI security      least privilege, human approval, no spending
   -> AI governance            audit trail, model versioning, human override
```

Each rung is implemented or specified in this repository. The full design is
`V2_TRUST_SECURITY_GOVERNANCE.md`; the running code covers reliability,
verification, explainability, security controls and the audit trail. The
agentic layer is **specified and deliberately not built** — see §7.

---

## 4. Where the money is

Four mechanisms, in descending order of confidence.

### 4.1 Provider selection on realised cost
Rank on expected rather than advertised cost. The saving is the gap between
what an organisation currently selects and what it would select.
**Confidence: high** — it is arithmetic once the reliability model is
calibrated on the organisation's own booking data.

### 4.2 Recovering the cost of failure
Failed bookings cost paid time that no report currently attributes. Making it
visible per campus, per provider and per hour turns an invisible loss into a
managed one. **Confidence: high for measurement, medium for reduction** —
measuring it is certain; reducing it depends on having alternatives.

### 4.3 Modal shift
Where a reliable scheduled service is competitive on door-to-door time, shifting
trips to it removes both fare and failure cost. **Confidence: medium** —
depends entirely on local transit coverage.

### 4.4 Provider negotiation
A reliability-adjusted cost-per-km scorecard is a commercial instrument. It
converts "we think they cancel a lot" into a number in a contract review.
**Confidence: medium-high.**

---

## 5. A worked ROI example

> **EVERY NUMBER IN THIS SECTION IS AN ASSUMPTION.** They are illustrative
> arithmetic for a mid-sized campus, not a measurement, not a benchmark, and
> not a forecast. The purpose is to show the *shape* of the calculation and
> which inputs a pilot would have to establish.

**Assumed scenario:** one campus, 4,000 employees, 30% using company-supported
ride-hailing, 18 trips per person per month.

| Input | Assumed value | Basis |
|---|---|---|
| Trips per month | 21,600 | 1,200 employees × 18 |
| Mean fare | ₹95 | assumption |
| **Annual fare spend** | **₹2.46 crore** | 21,600 × 95 × 12 |
| Booking failure rate | 22% | assumption; order-of-magnitude from the simulated bundle |
| Minutes lost per failure | 6 | assumption |
| Loaded cost per employee-minute | ₹6 | assumption |
| **Annual cost of failure** | **₹20.5 lakh** | 21,600 × 0.22 × 6 × ₹6 × 12 |
| **Total annual mobility cost** | **₹2.67 crore** | |

**Assumed effect of the platform:**

| Lever | Assumed improvement | Annual value |
|---|---|---|
| Provider selection on expected cost | 6% of fare spend | ₹14.8 lakh |
| Fewer failures through better selection | 22% → 17% failure rate | ₹4.7 lakh |
| Modal shift on eligible trips | 3% of fare spend | ₹7.4 lakh |
| **Gross annual benefit** | | **₹26.9 lakh** |

**Assumed platform cost:** ₹40/employee/month across 4,000 employees =
**₹19.2 lakh/year**, plus a one-off implementation of ₹8 lakh.

| | Year 1 | Year 2+ |
|---|---|---|
| Benefit | ₹26.9 L | ₹26.9 L |
| Platform + implementation | ₹27.2 L | ₹19.2 L |
| **Net** | **−₹0.3 L** | **+₹7.7 L** |
| **ROI** | ~0% | ~40% |

> **Read the shape, not the digits.** On these assumptions the platform is
> roughly break-even in year one at 4,000 employees and clears 40% thereafter —
> which means **the model is sensitive to scale and to the failure rate**. If
> the real failure rate is 10% rather than 22%, the case weakens materially. A
> pilot exists to measure that number before anyone signs anything. Presenting
> this as a projected saving would be exactly the overconfidence the product is
> designed to detect.

---

## 6. Commercial models

| Model | Shape | Best fit |
|---|---|---|
| **SaaS per employee** | ₹30–60/employee/month | Large campuses, predictable |
| **SaaS per campus** | Annual, banded by trip volume | Multi-site, uneven adoption |
| **Enterprise licence** | Annual, unlimited sites, self-hosted | Data-residency requirements |
| **API metered** | Per 1,000 comparisons | Embedding in an existing travel tool |
| **Managed service** | Licence + operations retainer | Where reliability models need continuous recalibration — **the most defensible, because the model degrades without it** |
| **Implementation** | One-off integration | Always, alongside one of the above |

The managed-service model deserves emphasis: a calibrated reliability model is
not a one-time deliverable. Provider behaviour drifts, and an uncalibrated
confidence score is worse than none. Recalibration is a recurring service, and
that is a feature of the business model rather than a defect of the product.

---

## 7. What is honestly not built

Stated here rather than discovered in diligence.

| Not built | Status |
|---|---|
| **Live provider APIs** | No open API publishes ride-hailing supply or cancellation data. All hailed-vehicle quotes are SIMULATED and labelled as such in every response. A real adapter implements five methods; nothing else changes. |
| **Real booking data** | The reliability models are trained on a documented generative process. They demonstrate the pipeline; they are not evidence about any operator. |
| **Booking and payment** | The system recommends. It does not reserve or pay. This is a deliberate safety boundary, not a gap. |
| **Agentic execution** | Six agents are specified with a full permission matrix; none is implemented. Deliberate — see below. |
| **Multi-tenancy** | Single-tenant. Role-based API keys, not SSO, not tenant isolation. |
| **Encryption at rest** | Not implemented. |
| **More than one city** | One bounded corridor of Bengaluru. |

> **Why the agentic layer is specified and not shipped.** An agent that can
> change data sources or spend money is exactly the thing that needs
> governance *before* it exists. The permission matrix, the human-approval
> gates and the "veto but never edit" separation are written down and testable;
> building the autonomy before the controls would be the wrong order, and an
> enterprise buyer should treat a vendor who does it the other way round with
> suspicion.

---

## 8. A pilot that would actually settle it

**Scope:** one campus · one metropolitan region · 3–5 providers · 90 days.

### Phase 1 — Measure (weeks 1–4)
Instrument existing bookings. Establish the organisation's **real** failure
rate, real cost of failure, real provider reliability. This phase alone has
value: nobody currently has these numbers, and it is what tests §5's central
assumption.

### Phase 2 — Calibrate (weeks 5–8)
Fit the reliability heads on that data. Accept or reject on calibration —
reliability diagram and expected calibration error — not on plausibility. **A
model that is not calibrated on real data does not proceed.**

### Phase 3 — Recommend (weeks 9–12)
A/B: expected-cost recommendation against current practice. Measure realised
cost, not predicted cost.

### KPIs

| KPI | Measured how | Success |
|---|---|---|
| Reduction in realised cost per trip | A/B against control | ≥ 5% |
| Booking success rate | Completed ÷ requested | +3pp |
| Cost of failure | Minutes lost × loaded rate | −20% |
| ETA accuracy | MAE against realised | < 15% MAPE |
| Cancellation prediction | AUROC + **ECE** | AUROC > 0.68, ECE < 0.03 |
| Recommendation acceptance | Accepted ÷ shown | > 60% |
| Decisions with an explanation | Audit log | **100%** |
| Model drift | Rolling residual | Flagged within 7 days |

Two of these deserve emphasis. **ECE is a first-class KPI** because a
probability that gets multiplied into a rupee figure has to mean what it says.
And **100% of decisions carrying an explanation** is a governance requirement,
not a stretch goal — it is already true of the running system.

### The honest exit criterion

If Phase 1 shows the organisation's real booking failure rate is low, the
business case is weak and **the pilot should stop there**. A vendor who cannot
name the condition under which their product is not worth buying has not
measured anything.

---

## 9. Positioning

**Not:** "an app that compares Rapido, Ola and Namma Yatri."

**But:**

> An enterprise-grade AI mobility intelligence platform that helps
> organisations optimise transportation cost, reliability and operations while
> making AI-powered decisions explainable, secure and governable.

The stack that makes that true:

```
Mobility data (OSINT, provenance-tracked)
  + Graph intelligence (multimodal city graph, GNN travel time)
  + Predictive AI (calibrated reliability, expected cost)
  + Optimisation (constrained multi-objective)
  + Trustworthy AI (confidence, verification, abstention)
  + Security (threat model, RBAC, integrity)
  + Governance (audit trail, versioning, human override)
  + Enterprise analytics (spend, failure cost, provider scorecards)
```

Every line is implemented or specified in this repository, and §7 says which is
which.

---

## 10. The demo, in three minutes

The order matters more than any individual screen. The evaluator must
*experience* the problem before being shown the intelligence — a prediction
displayed first is a dashboard; the same prediction displayed after is an
explanation.

**1 · It looks like a ride app.** College (Shanthinagar) → M.G. Road. Four
options, a fare, a time, an availability badge, a Book now button. Nothing on
this screen mentions a model.

```
   Carpool       ₹29    33 min   Low availability        [ BOOK NOW ]
   Bike taxi     ₹31    13 min   High availability       [ BOOK NOW ]
   Auto          ₹36    18 min   Moderate availability   [ BOOK NOW ]
   Cab           ₹81    18 min   High availability       [ BOOK NOW ]
```

**2 · Book the cheapest.** *Searching for a driver… → No driver available.*
Try again — the fare has moved to ₹35. A driver accepts. The ride completes.
The outcome was sampled from the same probabilities that priced the option, and
in demo mode the sequence is reproducible on stage.

**3 · Ask why.**

|  | Carpool *(chosen)* | Auto *(engine's pick)* |
|---|---|---|
| Advertised | **₹29** | ₹36 |
| Completes | 21% | 82% |
| Cancelled after accepting | 46% | 21% |
| **Expected cost** | **₹45** | **₹43** |

The cheapest sticker price is the most expensive journey. That is the whole
pitch, and the evaluator arrived at it by pressing a button rather than reading
a claim.

**4 · Show it is a market, not a quirk.** *Insights* — trip length against
acceptance, demand against supply, provider reliability against effective cost
per km. Each panel is labelled an association, never a cause.

**5 · Scale it.** *Enterprise* — the same engine over 60,000 bookings: spend,
the cost of failure in paid minutes, provider scorecards ranked by
reliability-adjusted cost per km, insights tagged *observation* or *prediction*.

**6 · Show it is governable.** The audit trail: every recommendation with its
model version and confidence, append-only.

Step 5 is where a viewer stops seeing a ride-booking app.

---

### Related documents

| Document | Contents |
|---|---|
| `README.md` | Architecture, setup, API |
| `SOURCES.md` | Every data element and its provenance |
| `EVALUATION.md` | Measured model performance, including where the GNN loses |
| `V2_TRUST_SECURITY_GOVERNANCE.md` | Trust, verification, security, agentic design, governance |
| `OPEN_SOURCE_ATTRIBUTIONS.md` | What influenced what, under which licence |
