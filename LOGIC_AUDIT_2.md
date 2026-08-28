# Final logic + product audit

The second pass. Where [LOGIC_AUDIT.md](LOGIC_AUDIT.md) fixed defects one at a
time, this one changed the things underneath them: what a *mode* is, what
walking is for, and what a price on a card actually covers.

Method, as before: sweep first, then fix. All 210 ordered place pairs across
three departure hours, every candidate journey checked for continuity,
additivity, mode legality and walking load; a booking-state sweep; an
edge-case pass over budgets from ₹5 to ₹400; then the browser.

---

## 1. Bugs discovered

### A brand was a mode
The graph's ride modes were `rapido`, `auto`, `namma_yatri`, `cab`. Two of
those are companies. Namma Yatri had its own fare table identical to the auto
meter, so **one vehicle appeared twice in every comparison** with the same
number beside it, and the router was brand-locked at the graph level.

### A metro ticket was quoted as a journey
The Metro card said **₹25, 217 minutes**. Two hundred of those minutes were
walking. A station is not a doorstep, and no card said so.

### Walking was a commute
Journeys were routinely `walk → bus → walk → metro → walk`, with **58 minutes
on foot** in one case and **202 in another**. The search actively preferred
this: walking costs ₹0, so the cheapest blend would take any amount of it. One
candidate paired a bike taxi to the metro with a **77-minute walk** to the door.

### Carpool steered the whole comparison
A mode nobody could book was the cheapest card on almost every trip — a
fraction of a cab fare into a thin market — and it **completed 11% of the
time**. It supplied the product's headline crossover by being absurd.

### Cheap journeys were impossible by construction
A hailed vehicle could only reach a *hub*: a metro station, a named place, or
a stop served by two routes. **The nearest hub to the Wipro gate is 6.7 km
away**, so every cheap itinerary had to start on foot and was then correctly
thrown out. There are bus stops **500 m** from that gate.

### An option was its own fallback
`_fallback_for` picked the cheapest scheduled service and handed it to *every*
option, including itself. So the cost of the bus failing was the cost of taking
the bus: a bus that failed **72% of the time** priced out at barely more than
its own fare, and the advertised-vs-expected crossover vanished from every
scenario.

### A metro journey claimed certainty it did not have
Scheduled quotes reported `p_success = 100%`. Once a journey needed a hailed
first and last mile, that was false: the weak link is the booking, not the
timetable.

### A 78 m trip was answered with a 1 km bike taxi
The only ride edge in reach ran out to a bus stop and back, so the router used
it — **15 minutes and ₹29 to cover 78 metres**.

### Two identical vehicles, still
`bus → bus` and `metro → metro` survived as adjacent legs whenever a walk
between them was removed, because the merge rule required the same route id.

### Smaller ones
- A comparison row's `total_min` and the leg sum diverged once access distance
  moved fields.
- The escalation could only suggest another single ride — never "travel in
  stages" — after four hailed bookings had just failed.
- A three-transfer itinerary beat a direct ride on the fourth decimal place.
- Enterprise analytics reported a scorecard row for carpool: **15,124 of
  60,000** bundled bookings, for a mode no employee can book.
- `test_deployment` deletes every `app.*` module, so a later `import` yields a
  fresh `lru_cache`; the parity test cleared one cache and called another.

---

## 2. Bugs fixed

| # | Fix | Where |
|---|---|---|
| 1 | Modes are vehicles: `bike_taxi`, `auto`, `cab`, `metro`, `bus`. Providers are Rapido, Metered auto, Namma Yatri, Cab aggregator, Namma Metro, BMTC | `graph/builder.py`, `providers/*`, `fares.json` |
| 2 | Namma Yatri is an `auto` provider modelled without demand pricing — an assumption about platform economics, stated on every quote | `providers/simulated.py` |
| 3 | Scheduled quotes are door to door and declare their hailed access legs | `providers/base.py`, `services/engine.py` |
| 4 | A scheduled journey reached by hailed legs inherits their failure probability | `providers/simulated.py` |
| 5 | Walking is folded into the leg it serves (`access_min` / `access_km`), never shown, never priced | `routing/journey.py` |
| 6 | A journey leaning on more than 12 minutes on foot is rejected | `routing/validate.py` |
| 7 | Walking carries a **search-only** shadow price of ₹3/min, so the optimiser reaches for a last-mile ride instead of a 77-minute walk | `routing/costs.py` |
| 8 | Carpool removed from providers, fares, the mode table, the UI and the enterprise history (excluded at load, counted, logged) | 8 files |
| 9 | Hailed vehicles can reach any stop or station within 3 km of either end, not only a hub | `graph/builder.py`, `config.py` |
| 10 | Two targeted searches per transit mode guarantee the ride-transit-ride family is always in the candidate set | `routing/kshortest.py` |
| 11 | An option is never its own fallback | `services/compare.py` |
| 12 | The fallback is the most *reliable* scheduled service, not the cheapest | `services/compare.py` |
| 13 | Consecutive same-mode transit legs merge into one leg carrying `segments`, so per-boarding fares survive and no summary says "Metro → Metro" | `routing/journey.py` |
| 14 | A journey travelling more than 3× the straight line (floor 0.6 km) is a detour, not a route | `routing/validate.py` |
| 15 | Ties within ₹10 and 5 minutes go to the journey with fewer changes | `optimisation/scoring.py` |
| 16 | The escalation reconsiders itineraries, not just another hailed ride | `api/booking.py`, `booking/escalation.py` |
| 17 | Legs name the point you actually board — a metro leg no longer claims to start at a bus stop | `routing/journey.py`, `services/compare.py` |
| 18 | The parity test clears the cache it calls | `tests/test_model_parity.py` |

---

## 3. Carpool removed

`CarpoolProvider`, its reliability class in the product path, its fare
assumptions, its per-km enterprise rates, its UI colour, and its 15,124 rows of
bundled history — which are excluded at load with the count logged and asserted
by a test. Searched for `carpool`, `pool`, `ride_share`, `shared_ride`: what
remains are unrelated uses (a connection pool, `numpy` pooling) and the
`provider_carpool` **column in the trained reliability checkpoint**, which stays
so the weights file remains valid and is documented as never firing.

Verified by `test_a_removed_mode_is_gone_from_every_surface`, parameterised over
carpool, walk and cycle, across the compare payload, the journeys, the provider
registry and `/api/providers`.

---

## 4. Walking removed from the commute

Walking stays in the graph — there is no other way onto a platform — and is
**absorbed** into the leg it serves rather than deleted, so not one second goes
missing from the total. Three rules hold the line:

- no leg a rider is shown has mode `walk` (`ALLOWED_MODES`)
- a journey over **12 minutes** of total walking is rejected
- walking costs ₹3/min *inside the search*, so the optimiser prefers a
  first-mile ride — the fare is still zero, and the constant says so

The 202-minute-walk "Metro" journey is now `Bike taxi → Metro → Bike taxi`.

---

## 5. Routing fixes

Additivity (from the first pass) holds: ride time is sampled along the
corridor, so splitting a trip cannot make it faster. On top of that: first/last
mile rides to any nearby stop; two targeted ride-plus-transit searches per
transit mode; the detour rule; and merged same-mode transit legs.

---

## 6. Multimodal fixes

Neither shape is forced. On the demo corridor, measured:

| Budget | Recommendation |
|---|---|
| ₹400 | **Bike taxi**, direct — ₹185, 57 min |
| ₹250 | **Bike taxi → Metro → Bike taxi** — ₹167, 56 min |
| ₹150 | **Bike taxi → Bus → Metro → Bike taxi** — ₹124, 82 min |
| ₹100 and below | nothing fits, said with the cheapest that does |

The direct ride wins when it deserves to and loses when it does not.

---

## 7. Booking fixes

The demo sequence is unchanged and asserted: *Searching for driver… → Driver
found → Driver accepted your request → Driver cancelled your ride → TRY AGAIN*,
four attempts, then escalation. Scheduled services keep their own honest
sequence (*Checking the service → On board → Journey completed*) and a **Start
trip** button, because there is no ticketing integration behind a BOOK NOW.
Attempts are counted only when a booking is actually attempted.

---

## 8. Availability fixes

Reliability still varies by provider, hour, distance, rain and congestion —
nothing is hard-coded per mode. What changed is that a **scheduled service is
no longer automatically certain**: when its journey needs hailed access, it
inherits `p_match ** n`, `p_accept ** n` and `1-(1-p_cancel)**n` over those *n*
bookings, and the basis string says so.

---

## 9. Expected-cost fixes

An option is never its own fallback, and the fallback is the most reliable
scheduled service rather than the cheapest. The crossover the product exists to
show is back and honest — wet peak on the demo corridor:

> **Bus** advertises **₹155** and is expected to cost **₹174** (28% complete).
> **Metro** advertises **more** at ₹170 and is expected to cost **less** at ₹168.

In dry peak there is no crossover, and none is manufactured.

---

## 10. Budget fixes

Budget is a hard constraint on the fare you are charged; the expectation is
reported separately as `budget_at_risk`. Below the floor the answer is a
sentence and the numbers, never an empty screen —
`test_every_budget_gets_an_answer_or_a_reason` runs ₹5 / ₹10 / ₹20 / ₹50 /
₹100 / ₹150 / ₹250 / ₹400 and requires either a journey within budget or a
message naming what the cheapest one actually costs.

---

## 11. Time and deadline fixes

`within_time` is the trip's own duration; `time_at_risk` is the expectation
once retries are counted. Arrival risk runs on the **rider's** clock —
departure plus minutes already lost — not the server's.

---

## 12. Manager escalation fixes

Offered only when the projected arrival actually misses the meeting
(`can_notify` requires `at_risk` or `late`). The message is composed from live
numbers, addressed in the rider's own voice, and states `composed_not_sent`.
The alternative it names is the **cheapest option that still arrives in time**,
and it now includes travelling in stages — after four hailed bookings have
failed, "try another hailed ride" is the one answer already disproved.

---

## 13. Tests added

**32 new tests** in `tests/test_modes_and_journeys.py`, plus the earlier 44 in
`tests/test_logic_audit.py`:

- **Vocabulary** — exactly six modes; a brand is never a mode; two providers
  share the auto; each removed mode is gone from every surface; carpool history
  is excluded from the enterprise view.
- **Walking** — never a leg anywhere; what remains is a station approach;
  absorbed minutes still counted; a fare is never charged for walking.
- **Shape** — a direct ride wins when it deserves to; multimodal wins when it
  deserves to; ties go to the simpler journey while real trade-offs do not; a
  journey is named by its transit spine.
- **Pricing** — a metro card prices the ride to the station; a metro reached by
  bike taxi does not claim certainty; an option is never its own fallback; the
  crossover still happens.
- **Budgets** — eight budgets from ₹5 to ₹400, each answered or explained; the
  planner reaches for transit as money tightens; a hailed vehicle can reach a
  stop, not only a hub.
- **Validator** — a detour is not a route; no duplicate journeys; the booking
  screen never repeats a card as an itinerary.

Suite total: **226 passed, 1 skipped**.

---

## 14. Remaining limitations

- **Provider behaviour is simulated.** No commercial API is contacted. Metro
  and bus fares are published tables; hailed fares are a documented
  base + per-km + per-min model with a band.
- **Namma Yatri's price advantage is an assumption** about platform economics,
  not an observed rate. The fare table is identical to any metered auto.
- **The reliability checkpoint still carries a `provider_carpool` column.**
  Retraining to drop it would change every number in the evaluation; it is
  documented and never fires.
- **The bundled history lost 25% of its rows** to the carpool exclusion.
  44,876 bookings remain.
- **The 12-minute walking budget, the ₹3/min shadow price, the 3× detour
  ratio, and the ₹10 / 5-minute simplicity band are judgement calls**, each
  calibrated against the alternative it competes with and stated in the code.
- **Access rides are modelled as independent bookings.** Two of them failing
  are treated as independent events; in reality the same scarcity drives both,
  so the combined failure probability is likely a little pessimistic.
- **One corridor.** "No metro route between these points" often means "not in
  this dataset".
- **Nothing is ever sent.** The manager notification is composed, recorded and
  displayed.

---

## 15. The demo flow

**Wipro Campus, Doddakannelli → PES University, RR Campus**, next weekday
09:00, meeting at 10:00.

1. **Book a ride** — six options, cheapest first, each door to door, each
   naming its vehicle and its operator:

   | | | | |
   |---|---|---|---|
   | Bus · BMTC | ₹153 | 90 min | Start trip |
   | Metro · Namma Metro | ₹167 | 56 min | Start trip |
   | Bike taxi · Rapido | ₹215 | 61 min | Book now |
   | Auto · Namma Yatri | ₹320 | 75 min | Book now |
   | Auto · Metered auto | ₹371 | 75 min | Book now |
   | Cab · Cab aggregator | ₹482 | 70 min | Book now |

2. **Or travel in stages** — `Bike taxi → Bus → Metro → Bike taxi`, ₹106–₹141,
   82 min, expanding to real legs: *bike taxi to Sarjapur Road stop 3, route
   356 to stop 10, Yellow Line to Rashtreeya Vidyalaya Road, bike taxi to the
   door*. **View journey**, never Book now.

3. You have a 10:00 meeting, so the 90-minute bus is out. **Book now** on the
   bike taxi → *Searching for driver… → Driver found → Driver accepted your
   request → **Driver cancelled your ride***.

4. **Try again**, at a higher fare, up to four attempts.

5. **Arrival risk** against the 10:00 review, on the rider's clock. The
   alternative offered is the cheapest thing that still arrives in time —
   another vehicle, or travelling in stages. **Notify manager**: composed,
   recorded, never transmitted.

6. **Why did that happen?** — advertised versus expected, side by side, with
   the time each option costs. Every probability was computed before Book now
   was pressed.

7. **Insights → Intelligence → Enterprise** — the same failure at market and
   organisation scale, with the audit log underneath.

Verified end to end by `python scripts/capture_demo.py`: 18 screenshots, each
asserted before it was photographed, no console errors and no failed requests.
