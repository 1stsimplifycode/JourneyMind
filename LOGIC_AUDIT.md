# Logic audit

A full pass over the recommendation pipeline, from graph construction to what
the browser paints, looking for outputs that are arithmetically fine and
physically absurd.

The method was not reading code and guessing. Three sweeps drove it:

1. **An invariant sweep** over all 210 ordered place pairs × 3 departure hours,
   checking every candidate journey for continuity, additivity, fare-band
   sanity and repeated modes.
2. **A state-machine sweep** — 438 sampled booking attempts across every
   provider, every transition checked against `LEGAL_TRANSITIONS`.
3. **An edge-case sweep** through the public API: same origin and destination,
   outside the study area, 100 m apart, 02:00, rain, ₹1 budget, 1-minute limit.

Then the browser, driven by Playwright, because two of the defects below were
only visible on a rendered page.

---

## Bugs found

### 1. The direct ride was deleted on long trips
`JM_MAX_RIDE_KM` (16 km) was written to bound a **first/last-mile hop to a
hub** — past ~16 km a "quick hop to the metro" is not a quick hop. It was being
applied to the **door-to-door ride as well**, so on any trip over 16 km the
option a rider always has simply did not exist in the graph.

**12 of 210 place pairs**, including the demonstration route (Wipro →
PES, 21.3 km).

### 2. Travel time was not additive, so splitting a trip made it faster
A ride edge's duration was free-flow time scaled by the congestion at its **two
endpoints**. The request's own origin and destination carry no road edges, so
both fell back to the global mean; a two-hop path through a real hub picked up
that hub's local reading instead.

Measured on the demo route: the direct 21.31 km ride was predicted at
**21.2 km/h**, while the same ground split at HSR Layout averaged **22.9 km/h**.
Two vehicles beat one for no reason but a modelling artefact.

### 3. …so the router recommended two of the same vehicle in a row
The consequence of 1 and 2 together. The invariant sweep found **166 candidate
journeys** containing consecutive identical hailed legs — 102 `rapido→rapido`,
plus `cab→cab`, `auto→auto`, `namma_yatri→namma_yatri`. On the demo route the
**recommended** journey was `rapido → rapido` via an office park.

### 4. Every long-trip ride card was secretly a two-hop chain
`HailedProvider.get_route` reads the single-mode reference journey. Because of
1, that reference was the two-hop chain — **two base fares and two pickup
waits** — presented on screen as one direct ride: "Bike taxi · 66 min · 4 min
pickup · 21.4 km".

### 5. "Metro → Metro" and "Bus → Bus" in journey summaries
A Yellow-to-Green interchange at Rashtreeya Vidyalaya Road is a real journey
thousands of people make. Describing it as two separate trains is not.

### 6. Walking was systematically unavailable
Walking only reached the comparison if a walk-only journey happened to be
recommended. On a 2 km trip the Walk row said **"no walk route between these
points in the study area"**, which is false — the graph is fully connected.

### 7. Unroutable options published fabricated numbers
The lifecycle solver returns a figure for an option with no route: it prices
the **fallback** you would take instead. That was published as the option's own
cost, so the payload carried **"Walk: ₹25, 0 min"**.

### 8. A privately-owned bicycle won every priority
Once 6 was fixed, Cycle — free, quick, never cancels — became the
recommendation for **all four priorities** on any trip it could reach. It is
also a machine the rider may not own, on a graph with no cycle network and no
bike-share to hire one from.

### 9. Walking as the fallback made unreliable options look free
`_fallback_for` picked the cheapest available non-hailed option. Once walking
worked, that was walking, at ₹0 — so the failure mass of every unreliable
option was costed at zero. **Carpool advertised ₹29 and reported ₹7 expected**,
because seven times in ten the model had the rider walk for half an hour and
charged nothing for it.

### 10. The budget gate used the average, not the price
An option was "within budget" if its **expected** cost fitted. Expected cost
sits *below* the fare precisely when an option often fails, so a ₹300 ride
passed a ₹250 budget on the grounds that you probably would not get it.

### 11. Offered journeys ignored the rider's stated limits
The planner is run unconstrained inside `compare` so the ride cards can price
options the rider cannot afford. The multimodal journeys harvested from that
run were never measured against the budget or time the caller actually sent.

### 12. The headline contradicted the list underneath it
`compare(budget=200, max_time=65)` printed **"Nothing fits ₹200"** directly
above a **₹137–₹197, 56-minute** journey that fitted both limits. The sentence
was written from the provider cards; the list was filtered somewhere else.

### 13. Two providers quoting one identical number
Auto and Namma Yatri share a fare table (base ₹30, ₹15/km) and a reliability
class, so every figure was identical — one option printed twice. The code
claimed a "different fare structure" that did not exist.

### 14. A typed coordinate was treated as a place name
`PointInput`'s docstring promised coordinates and `resolve_point` could use
them, but a typed pair only ever arrived as `label`, so `"12.9345, 77.6100"`
came back **"Could not find '12.9345, 77.6100' in this study area"**.

### 15. Three screens, three different demo trips
The booking page hard-coded `College (Shanthinagar) → M.G. Road` while the
planner used the server's `Wipro → PES`, and the escalation invented a meeting
"an hour after departure" that appeared nowhere else.

### 16. Booking a metro searched for its driver
> Searching for driver… → Driver found → **Kiran K.** is on the way → Driver
> accepted your request → Ride started

A train has nobody to match and nobody to accept your request.

### 17. "Book now" on a service with no ticketing integration
Metro and Bus carried the same BOOK NOW button as a hailed ride.

### 18. A slow, cheap winner never admitted what it cost
In rain at 18:30 the answer was **"Best value: Bus"** — ₹54, 183 minutes — with
a 98-minute cab available. The ranking is defensible (ten times the price for
twice the speed); stating it without the three hours attached is not.

### 19. Two defects the tests could not see
- **A frontend crash.** Publishing `expected: null` (fix 7) made
  `Compare.jsx` dereference it and unmount the entire Intelligence view. The
  capture script only reported a 40-second selector timeout.
- **A test-isolation bug.** `tests/test_deployment.py` deletes every `app.*`
  module from `sys.modules`, so a later import yields a fresh module with a
  fresh `lru_cache`. `test_service_falls_back_rather_than_crashing` was
  clearing one `get_predictor` cache and calling a different one; it passed
  only because the second cache happened to be cold.

---

## Bugs fixed

| # | Change | Where |
|---|---|---|
| 1 | Split the ride cap in two: `JM_MAX_RIDE_KM` (16 km) bounds a hub hop, `JM_MAX_DIRECT_RIDE_KM` (60 km) bounds the door-to-door ride | `graph/builder.py`, `config.py`, `.env.example` |
| 2 | Ride congestion sampled at 5 points **along the corridor**, cached per request; travel time is additive again | `models/base.py` |
| 3 | Consecutive identical hailed legs rejected outright | `routing/validate.py` |
| 4 | Follows from 1 — every hailed reference journey is now `transfers=0` | measured, not asserted |
| 5 | `Journey.shape()` collapses consecutive same-mode legs; legs keep their route names and carry `interchange: true` | `routing/journey.py`, `services/compare.py`, `Book.jsx` |
| 6 | `Recommendation.walk_reference` harvested from the candidate set, not the presentation pool | `services/engine.py`, `services/compare.py` |
| 7 | Unrouted options publish `null` for expected cost, times and distance, plus their reason | `api/mobility.py`, guarded in `Compare.jsx` |
| 8 | Cycle is unavailable with an honest reason — *"needs a bicycle of your own"*. Walking stays recommendable: everybody has feet | `providers/simulated.py` |
| 9 | The fallback must be a **scheduled** service — the thing still running after four failed bookings | `services/compare.py` |
| 10 | Hard limits are the **fare you are charged** and the **trip's own duration**; the expectation is reported separately as `budget_at_risk` / `time_at_risk` | `services/compare.py`, `api/mobility.py`, `Compare.jsx` |
| 11 | Journeys carry `within_budget` / `within_time` against the caller's limits | `services/compare.py` |
| 12 | One `offerable()` rule, used by the headline **and** the list; the headline now names the journey that fits | `services/compare.py`, `api/mobility.py` |
| 13 | Namma Yatri modelled without demand pricing — stated as an assumption about platform economics, not a measurement | `providers/simulated.py` |
| 14 | `PointInput` parses `"lat, lon"`, `"lat,lon"` and `"lat lon"` before the place-name branch | `schemas.py` |
| 15 | One `demo_scenario.py`, carrying the meeting as well; read by the booking page, planner, `/api/city`, `/api/demo` and the escalation | new module |
| 16 | Scheduled services get their own sequence: **Checking the service → On board → Journey completed**, no driver anywhere. `REQUESTED → RIDE_STARTED` added to the state machine for it | `booking/session.py`, `lifecycle/states.py` |
| 17 | The button reads **"Start trip"** for a scheduled service | `Book.jsx` |
| 18 | When a faster option exists, the reasoning names it and the minutes and money involved — *"worth it or not is your call, not the model's"* | `services/compare.py` |
| 19 | Console errors and failed requests fail the capture run; the parity test clears the cache it actually calls | `scripts/capture_demo.py`, `tests/test_model_parity.py` |
| — | `carpool` and `cycle` added to the UI mode table (they were falling back to a grey default dot) | `modes.js` |

---

## Rules added

`backend/app/routing/validate.py` runs between assembly and ranking, so an
invalid candidate can never win and the interface is never the thing hiding it.
Two severities: **reject** (dropped) and **warn** (kept, and repeated to the
rider).

| Rule | Severity | What it catches |
|---|---|---|
| `discontinuous` | reject | Leg *n* ends somewhere leg *n+1* does not start |
| `consecutive_hailed` | reject | Two of the same hailed vehicle in a row |
| `consecutive_walk` | reject | Two walking legs that were never merged |
| `split_same_route` | reject | One service split across two legs — an assembly bug, unlike a genuine interchange |
| `zero_length_vehicle` | reject | A vehicle leg under 50 m |
| `negative_leg` | reject | Negative time or distance |
| `decorative_transfer` | reject | One leg carries ≥85% of the distance; the other transfers earn nothing |
| `time_mismatch` | reject | Legs do not sum to the journey duration |
| `distance_mismatch` | reject | Legs do not sum to the journey distance |
| `zero_duration` | reject | A journey that takes no time |
| `negative_fare` / `fare_band` | reject | A negative fare, or a point estimate outside its own band |
| `transfer_count` | reject | Boardings and transfers disagree |
| `long_walk` | **warn** | Over 35 minutes on foot — possible, worth saying |

A **transit** interchange between two *different* services is explicitly legal:
the validator distinguishes a line change from a split service. What was wrong
was never the routing, only the summary that called it "Metro → Metro".

Rejections are counted into `pipeline.validation` (`rejected`, `by_rule`,
`examples`) rather than dropped silently.

---

## Tests added

**44 tests** in `tests/test_logic_audit.py`, grouped by pipeline stage. Each one
corresponds to a defect found above, not to a rule invented in the abstract.

- **Graph** — a direct ride exists for all 210 pairs; splitting a ride at any
  hub never makes it faster.
- **Validator** — no consecutive hailed legs, continuity, arithmetic closure,
  interchange-versus-split-service, collapsed summaries, no decorative
  transfers, the gate actually rejects, rejections are recorded.
- **Direct vs multimodal** — a direct ride can win; multimodal can win; walking
  wins at 100 m; walking is priced at 2 km; a private bicycle is never offered.
- **Hard constraints** — time and budget are not negotiable, at-risk is flagged
  rather than hidden, a cheap slow option cannot beat a time limit, offered
  journeys respect the limits, the headline cannot contradict them.
- **Preferences** — each preference changes the plan; all of them respect the
  hard limits.
- **Alternatives** — never the recommendation again; two providers of the same
  vehicle do not quote identically.
- **Service hours** — a shut network is said out loud; no boarding without
  paying the wait.
- **Scheduled services** — no driver is searched for; every narrated transition
  is legal across all six providers.
- **Edge cases** — same endpoints, unknown place, outside the area, typed
  coordinates in three formats, nothing affordable, one failure never becomes
  an enterprise pattern.
- **Demo scenario** — exactly one definition, and the escalation names it.

Suite total: **192 passed, 2 skipped**.

---

## Remaining limitations

Stated plainly, because several of these look like results and are not.

- **Provider behaviour is simulated.** Fares, availability, pickup waits and
  cancellation rates come from the models in `providers/simulated.py` and the
  reliability heads. No commercial API is contacted. Metro and bus fares are
  transcribed from published tables; ride-hailing fares are a documented
  base + per-km + per-min model with a band.
- **Namma Yatri's price advantage is an assumption**, not an observation. The
  fare table is identical to a metered auto; the difference modelled is the
  absence of demand pricing.
- **Service hours are approximate**, and the graph carries one corridor —
  Purple/Green/Yellow lines and a handful of bus routes. "No metro route
  between these points" often means "not in this dataset".
- **Congestion sampling is 5 points along a straight line.** It fixed
  additivity; it is not a route-following traffic model, and a ride edge still
  does not know which streets it uses.
- **The 85% dominance threshold and the 35-minute walking warning are
  judgement calls**, tuned against this corridor.
- **`decorative_transfer` cannot see the alternatives.** It rejects a journey
  where one leg is the whole trip, but "is this transfer worth it *compared to
  the direct ride*" is answered by the Pareto filter and the ranking, not the
  validator.
- **Cycling is priced on the walking path** at 13.5 km/h. It is a floor
  estimate, and it is not offered at all.
- **Enterprise analytics run on 60,000 simulated bookings.** Insights are
  labelled `observation` or `prediction`, and cohorts under 25 are suppressed
  rather than rounded. A single rider's failed booking is an incident and never
  enters that table — verified by test.
- **Nothing is ever sent.** The manager notification is composed, recorded in
  the audit log and returned for display.

---

## Demo scenario

One definition, in `backend/app/demo_scenario.py`, read by every screen.

| | |
|---|---|
| Origin | Wipro Campus, Doddakannelli (Sarjapur Road) |
| Destination | PES University, RR Campus (100 Feet Ring Road) |
| Departure | next weekday 09:00 |
| Meeting | the 10:00 project review |
| Budget / limit | ₹250 / 120 min (journey planner) |
| Demo mode | fixes the seed, not the probabilities |

**The flow**

1. **Book a ride** opens on that trip. Seven options, cheapest first; anything
   over twice the quickest journey drops into *"Cheaper, a lot slower"* —
   Metro at ₹25 and 217 minutes is real, and it is not the top of the list.
2. **Or travel in stages** — the planner's multimodal journeys, on the same
   screen. *Bike taxi → Metro → Bike taxi*, ₹137–₹197, 56 min, expanding to
   real legs (Central Silk Board, Yellow Line, Rashtreeya Vidyalaya Road).
   **View journey**, never BOOK NOW.
3. **Book now** on the cheapest ride: Searching for driver → Driver found →
   Driver accepted your request → **Driver cancelled your ride**.
4. **Try again**, at a higher fare, up to `JM_MAX_BOOKING_ATTEMPTS` (4).
5. **Escalation** — arrival risk against the 10:00 review on the *rider's*
   clock, the cheapest alternative that still arrives in time, and
   **Notify manager**: composed, recorded, never transmitted.
6. **Why did that happen?** — advertised versus expected, side by side, with
   the time each option costs; every probability was computed before BOOK NOW
   was pressed.
7. **Insights → Intelligence → Enterprise** — the same failure at market and
   organisation scale, with the audit log underneath it.

Verified end to end by `python scripts/capture_demo.py`: 18 screenshots, each
asserted before it was photographed, **no console errors and no failed
requests**.
