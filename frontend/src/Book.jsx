import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, bookRide, compare, getEscalation, notifyManager,
         retryBooking, revealBooking } from './api.js'
import { modeInfo } from './modes.js'

/* ===========================================================================
   The product-first view.

   This screen deliberately contains NO model output: a fare, a time, a
   supply badge, a button. That is what a rider sees in any mobility app, and
   it is what makes the failure land — the prediction is only revealed after
   the rider has lived the thing being predicted.

   Everything below the fold (the reveal, the comparison, the explanation) is
   gated behind an actual booking attempt.
   =========================================================================== */

/** The trip the booking screen opens on.
 *
 *  The server declares the demonstration route (`/api/city` -> demo_scenario),
 *  so the booking screen, the planner and the pitch all open on the same pair
 *  instead of drifting apart. The literal below is only the fallback for a
 *  city whose data ships no scenario. */
const DEMO_TRIP = { from: 'Wipro Campus, Doddakannelli (Sarjapur Road)',
                    to: 'PES University, RR Campus (100 Feet Ring Road)' }

function demoTrip(city, places) {
  const scen = city?.demo_scenario
  if (!scen || !places?.length) return DEMO_TRIP
  const nameOf = (id) => places.find(p => p.place_id === id)?.name
  const from = nameOf(scen.origin), to = nameOf(scen.destination)
  return (from && to) ? { from, to } : DEMO_TRIP
}

/** Demo mode books the morning commute rather than whatever time it happens to
 *  be. Not to rig the outcome — the probabilities are the model's either way —
 *  but because the crossover this product exists to show (a dearer sticker
 *  price with a lower expected cost) is a peak-hour phenomenon. At 23:00 the
 *  cheapest option really is the cheapest, and there is nothing to explain. */
function nextWeekdayMorning() {
  const d = new Date()
  d.setHours(9, 0, 0, 0)
  if (Date.now() > d.getTime()) d.setDate(d.getDate() + 1)
  while (d.getDay() === 0 || d.getDay() === 6) d.setDate(d.getDate() + 1)
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T09:00:00`
}

/** Supply, in the words a consumer app would use. From P(a vehicle responds). */
function supplyBand(p) {
  if (p >= 0.90) return { key: 'vhigh', label: 'Very high availability' }
  if (p >= 0.78) return { key: 'high', label: 'High availability' }
  if (p >= 0.60) return { key: 'mid', label: 'Moderate availability' }
  return { key: 'low', label: 'Low availability' }
}

const money = (x) => `₹${Math.round(x).toLocaleString('en-IN')}`
const pct = (x) => `${Math.round((x ?? 0) * 100)}%`

export default function Book({ places, city, onExplore }) {
  const [origin, setOrigin] = useState(DEMO_TRIP.from)
  const [destination, setDestination] = useState(DEMO_TRIP.to)
  const [demo, setDemo] = useState(true)
  const [options, setOptions] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const [booking, setBooking] = useState(null)   // { session, attempt }
  const [reveal, setReveal] = useState(null)
  const [escalation, setEscalation] = useState(null)
  const revealRef = useRef(null)

  const search = useCallback(async () => {
    if (!origin.trim() || !destination.trim()) {
      setError(new ApiError('Enter where you are starting and where you are going.', 'missing'))
      return
    }
    setBusy(true); setError(null); setOptions(null); setBooking(null); setReveal(null); setEscalation(null)
    try {
      const r = await compare({
        origin: origin.trim(), destination: destination.trim(),
        departure_time: demo ? nextWeekdayMorning() : null,
      })
      setOptions(r)
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError('Could not fetch rides.', 'unknown'))
    } finally { setBusy(false) }
  }, [origin, destination, demo])

  // Adopt the server's demo route once /api/city lands, then search once. A
  // state flag, not a ref: the first search has to run in the render *after*
  // the route is in state, or it searches the fallback pair.
  const [ready, setReady] = useState(false)
  useEffect(() => {
    if (ready || !city) return
    const t = demoTrip(city, places)
    setOrigin(t.from); setDestination(t.to); setReady(true)
  }, [city, places, ready])
  useEffect(() => { if (ready) search() }, [ready])   // eslint-disable-line

  const startBooking = async (providerId) => {
    setReveal(null); setEscalation(null)
    try {
      const r = await bookRide({
        origin: origin.trim(), destination: destination.trim(),
        provider_id: providerId, demo,
        departure_time: demo ? nextWeekdayMorning() : null,
      })
      setBooking(r)
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError('Booking failed to start.', 'unknown'))
    }
  }

  const tryAgain = async () => {
    const r = await retryBooking(booking.session.session_id)
    setBooking(r)
    // Once the retry budget is spent WITHOUT a ride, the rider needs more than
    // another button. `can_retry` is also false when a booking finally works,
    // and firing the escalation on that showed "You may be late — switch to
    // Cab" directly under "Journey completed, you paid ₹254". The rider was
    // already in the vehicle.
    if (r.session.exhausted && !r.session.settled) {
      try { setEscalation(await getEscalation(r.session.session_id)) } catch { /* optional */ }
    }
  }

  const showReveal = async () => {
    const r = await revealBooking(booking.session.session_id)
    setReveal(r)
    requestAnimationFrame(() =>
      revealRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }))
  }

  // Cheapest first, unavailable last — what every ride app does, and what
  // makes the rider's eye land on the cheap option the demo turns on.
  const allRides = (options?.options ?? [])
    .filter(o => o.service_class !== 'self')
    .slice()
    .sort((a, b) => (a.available === b.available)
      ? a.fare.amount - b.fare.amount
      : (a.available ? -1 : 1))

  // ...but "cheapest" alone puts a ₹25 metro that takes three and a half hours
  // above a ₹190 ride that takes ninety minutes, and no rider treats those as
  // the same list. Nothing is hidden: options far slower than the quickest way
  // there move into a second, labelled group, still cheapest-first inside it.
  const SLOW_FACTOR = 2.0
  const quickest = Math.min(
    ...allRides.filter(o => o.available).map(o => o.door_to_door_min ?? Infinity),
    Infinity)
  const isSlow = (o) => o.available && Number.isFinite(quickest)
    && (o.door_to_door_min ?? 0) > quickest * SLOW_FACTOR
  const rides = allRides.filter(o => !isSlow(o))
  const slowRides = allRides.filter(isSlow)

  return (
    <div className="bk">
      <div className="bk-search">
        <datalist id="bk-places">
          {places?.map(p => <option key={p.place_id} value={p.name} />)}
        </datalist>
        <div className="bk-fields">
          <div className="field">
            <label htmlFor="bfrom">From</label>
            <input id="bfrom" list="bk-places" value={origin} autoComplete="off"
                   placeholder="Pickup" onChange={e => setOrigin(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="bto">To</label>
            <input id="bto" list="bk-places" value={destination} autoComplete="off"
                   placeholder="Drop" onChange={e => setDestination(e.target.value)} />
          </div>
          <button className="go bk-go" type="button" disabled={busy}
                  onClick={search}>{busy ? 'Finding rides…' : 'Find rides'}</button>
        </div>
        <label className="demotoggle" title="Fixes the random seed so a live demo is reproducible.">
          <input type="checkbox" checked={demo} onChange={e => setDemo(e.target.checked)} />
          <span>Demo mode — reproducible booking sequence</span>
        </label>
      </div>

      {error && <div className="errbox" role="alert">
        <b>{error.message}</b>{error.detail && <div>{error.detail}</div>}</div>}

      {busy && <div className="bk-skel">
        {[0, 1, 2, 3].map(i => <div className="ridecard skel" key={i}>
          <div className="skeleton" style={{ width: '40%', height: 18 }} />
          <div className="skeleton" style={{ width: '70%' }} /></div>)}
      </div>}

      {!busy && rides.length > 0 && (
        <>
          {(options.origin?.typed || options.destination?.typed) && (
            <p className="bk-matched">
              Matched
              {options.origin?.typed && <> your start to <b>{options.origin.label}</b></>}
              {options.origin?.typed && options.destination?.typed && ' and'}
              {options.destination?.typed && <> your destination to <b>{options.destination.label}</b></>}
              {' '}inside the study corridor.
            </p>
          )}
          <div className="bk-head">
            <h2>Rides to {options.destination.label}</h2>
            <span>{allRides.filter(r => r.available).length} available now</span>
          </div>
          <div className="ridelist">
            {rides.map(o => (
              <RideCard key={o.provider_id} o={o}
                        onBook={() => startBooking(o.provider_id)}
                        busy={!!booking && !booking.session.settled
                              && booking.session.can_retry} />
            ))}
          </div>
          {slowRides.length > 0 && (
            <>
              <div className="bk-head bk-head-sub">
                <h2>Cheaper, a lot slower</h2>
                <span>over {SLOW_FACTOR}× the quickest way there</span>
              </div>
              <div className="ridelist">
                {slowRides.map(o => (
                  <RideCard key={o.provider_id} o={o}
                            onBook={() => startBooking(o.provider_id)}
                            busy={!!booking && !booking.session.settled
                                  && booking.session.can_retry} />
                ))}
              </div>
            </>
          )}
          {options?.journeys?.length > 0 && (
            <>
              <div className="bk-head bk-head-sub">
                <h2>Or travel in stages</h2>
                <span>combinations the route planner found</span>
              </div>
              <div className="ridelist">
                {options.journeys.map(j => <JourneyCard key={j.journey_id} j={j} />)}
              </div>
            </>
          )}
          <p className="bk-foot">
            Fares and waiting times are estimates for this demonstration, not live
            operator quotes.
          </p>
        </>
      )}

      {booking && (
        <BookingPanel booking={booking} onRetry={tryAgain} onReveal={showReveal}
                      onPickAnother={() => { setBooking(null); setReveal(null) }}
                      revealed={!!reveal} />
      )}

      {escalation && <Escalation esc={escalation} sessionId={booking?.session?.session_id} />}

      {reveal && <div ref={revealRef}><Reveal reveal={reveal} onExplore={onExplore} /></div>}
    </div>
  )
}

/* --------------------------------------------------------------------- */
/** A multi-stage journey from the route planner: bike taxi → metro → bike taxi
 *  and the like. There is no BOOK NOW here — you do not book a journey, you
 *  book the rides inside it. */
function JourneyCard({ j }) {
  const [open, setOpen] = useState(false)
  // The server's shape already collapses consecutive legs of one mode: a
  // change between two metro lines is an interchange, and "Metro → Metro"
  // describes it as two separate trains. The legs below keep both, with the
  // second marked as the interchange it is.
  const steps = (j.shape || '').split(' → ').filter(Boolean)
  return (
    <article className="ridecard journeycard">
      <div className="ridecard-main">
        <span className="journeydots">
          {steps.map((m, i) => (
            <i key={i} style={{ background: modeInfo(m).colour }} />
          ))}
        </span>
        <div>
          <h3>{steps.map(m => modeInfo(m).label).join(' → ')}</h3>
          <div className="ridecard-sub">
            {Math.round(j.total_min)} min · {j.transfers} change{j.transfers === 1 ? '' : 's'}
            {j.walk_min >= 1 ? ` · ${Math.round(j.walk_min)} min walking` : ''}
          </div>
          {j.warnings?.length > 0 && (
            <div className="journeywarn">{j.warnings.join(' ')}</div>
          )}
          {open && (
            <ol className="journeylegs">
              {j.legs.map((lg, i) => (
                <li key={i}>
                  <span className="jl-mode" style={{ color: modeInfo(lg.mode).colour }}>
                    {lg.interchange ? 'change' : modeInfo(lg.mode).label}
                  </span>
                  <span className="jl-txt">
                    {lg.from} → {lg.to}
                    {lg.route ? ` · ${lg.route}` : ''} · {Math.round(lg.minutes)} min
                    {lg.access_min >= 1 &&
                      ` · ${Math.round(lg.access_min)} min on foot to reach it`}
                  </span>
                </li>
              ))}
            </ol>
          )}
        </div>
      </div>
      <div className="ridecard-cta">
        <div className="ridecard-fare">{j.fare_display}</div>
        <button className="viewjourney" type="button" onClick={() => setOpen(v => !v)}>
          {open ? 'Hide journey' : 'View journey'}
        </button>
      </div>
    </article>
  )
}

/** What an enterprise product does when a consumer app would show a spinner. */
function Escalation({ esc, sessionId }) {
  const [sent, setSent] = useState(null)
  const [busy, setBusy] = useState(false)
  const risk = esc.risk

  const send = async () => {
    setBusy(true)
    try { setSent(await notifyManager(sessionId, {})) }
    finally { setBusy(false) }
  }

  return (
    <section className={`escbox esc-${risk.level}`}>
      <div className="eyebrow">After {esc.attempts} attempts</div>
      <h3>{risk.headline}</h3>
      <p>{risk.detail}</p>

      {esc.alternative && (
        <div className="escalt">
          <b>Switch to {esc.alternative.display_name}</b> —{' '}
          {/* An itinerary has no single completion rate: it is several
              bookings and a timetable. Rendering the missing value printed
              "completes 0% of the time", which is worse than saying nothing. */}
          {esc.alternative.p_success != null && (
            <>completes {pct(esc.alternative.p_success)} of the time, </>
          )}
          about {Math.round(esc.alternative.expected_minutes)} min, expected{' '}
          {money(esc.alternative.expected_cost)}.
        </div>
      )}

      {!sent && esc.can_notify && (
        <div className="escactions">
          <button className="booknow" type="button" disabled={busy} onClick={send}>
            {busy ? 'Composing…' : 'Notify manager'}
          </button>
          <span className="escnote">
            Nothing is sent until you press this, and nothing leaves this demo.
          </span>
        </div>
      )}

      {sent && (
        <div className="escsent">
          <div className="escsent-head">
            To {sent.message.to} · <code>{sent.message.delivery}</code>
          </div>
          <pre>{sent.message.body}</pre>
          <p className="escnote">{sent.message.delivery_note}</p>
          <div className="escincident">
            Incident <code>{sent.incident.incident_id}</code> ·{' '}
            {sent.incident.severity} · {sent.incident.attempts} attempts ·{' '}
            {Math.round(sent.incident.minutes_lost)} min lost ·{' '}
            {money(sent.incident.productivity_cost)} of paid time
          </div>
        </div>
      )}
    </section>
  )
}

function RideCard({ o, onBook, busy }) {
  const info = modeInfo(o.mode)
  const band = supplyBand(o.reliability.p_match)
  if (!o.available) {
    return (
      <article className="ridecard out">
        <div className="ridecard-main">
          <span className="ridecard-dot" style={{ background: info.colour }} />
          <div>
            <h3>{o.display_name}</h3>
            <div className="ridecard-sub">{o.unavailable_reason}</div>
          </div>
        </div>
        <div className="ridecard-cta"><span className="ridecard-na">Unavailable</span></div>
      </article>
    )
  }
  return (
    <article className="ridecard">
      <div className="ridecard-main">
        <span className="ridecard-dot" style={{ background: info.colour }} />
        <div>
          <h3>
            {o.display_name}
            {o.provider_name && <span className="via">{' · '}{o.provider_name}</span>}
          </h3>
          <div className="ridecard-sub">
            {Math.round(o.door_to_door_min)} min
            {o.pickup_min >= 1 ? ` · ${Math.round(o.pickup_min)} min pickup` : ''}
            {' · '}{Math.round(o.distance_km * 10) / 10} km
          </div>
          <div className={`avail avail-${band.key}`}>{band.label}</div>
        </div>
      </div>
      <div className="ridecard-cta">
        <div className="ridecard-fare">{money(o.fare.amount)}</div>
        <button className="booknow" type="button" onClick={onBook} disabled={busy}>
          {o.service_class === 'scheduled' ? 'Start trip' : 'Book now'}
        </button>
      </div>
    </article>
  )
}

/* --------------------------------------------------------------------- */
/** Plays the attempt the server sampled, one step at a time.
 *  The steps and their dwell times come from the API — the interface animates
 *  a result, it never decides one. */
function BookingPanel({ booking, onRetry, onReveal, onPickAnother, revealed }) {
  const { session, attempt } = booking
  const [shown, setShown] = useState(0)
  const timers = useRef([])

  useEffect(() => {
    timers.current.forEach(clearTimeout)
    timers.current = []
    setShown(0)
    let t = 0
    attempt.steps.forEach((s, i) => {
      timers.current.push(setTimeout(() => setShown(i + 1), t))
      t += s.dwell_ms
    })
    return () => timers.current.forEach(clearTimeout)
  }, [attempt])

  const done = shown >= attempt.steps.length
  const failed = done && !attempt.succeeded

  return (
    <div className="bookpanel">
      <div className="bookpanel-head">
        <div>
          <div className="eyebrow">Booking · attempt {attempt.number} of {session.max_attempts}</div>
          <h3>{session.display_name} · {money(attempt.fare)}</h3>
        </div>
        {attempt.number > 1 && attempt.fare > session.advertised_fare + 0.5 && (
          <span className="farebump">
            was {money(session.advertised_fare)} → now {money(attempt.fare)}
          </span>
        )}
      </div>

      <ol className="steps">
        {attempt.steps.slice(0, shown).map((s, i) => (
          <li key={i} className={`step step-${s.tone}${i === shown - 1 ? ' live' : ''}`}>
            <span className="step-mark" />
            <div>
              <b>{s.label}</b>
              <div className="step-detail">{s.detail}</div>
            </div>
          </li>
        ))}
        {!done && <li className="step step-pending"><span className="step-mark" />
          <div className="step-detail">…</div></li>}
      </ol>

      {done && attempt.succeeded && (
        <div className="bookdone good">
          <b>Journey completed.</b> You paid {money(attempt.fare)}
          {session.attempt_count > 1 && ` after ${session.attempt_count} attempts, having advertised ${money(session.advertised_fare)}`}.
        </div>
      )}

      {failed && (
        <div className="bookdone bad">
          <b>Your ride could not be completed.</b>
          {session.can_retry
            ? ' You can try again — the fare may have moved.'
            : ' No attempts left on this booking.'}
        </div>
      )}

      {done && (
        <div className="bookactions">
          {failed && session.can_retry &&
            <button className="booknow" type="button" onClick={onRetry}>Try again</button>}
          <button className="linkish inline" type="button" onClick={onPickAnother}>
            Choose another ride
          </button>
          {!revealed && (
            <button className="reveal-cta" type="button" onClick={onReveal}>
              {failed ? 'Why did that happen?' : 'What did that actually cost?'}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

/* --------------------------------------------------------------------- */
function Reveal({ reveal, onExplore }) {
  const { chosen, better, better_same_class: same, lived } = reveal

  // Prefer an alternative that is genuinely CHEAPER in expectation — that is
  // the crossover the product exists to show. Falling back to "the most
  // reliable option" is honest, but it must not be dressed up as a cost win:
  // crowning a ₹88 option green next to a ₹36 one tells the viewer the engine
  // recommends paying more, which is the opposite of the point.
  const cheaper = [same, better].find(
    a => a && a.expected_cost < chosen.expected_cost - 0.5)
  const alt = cheaper || same || better
  const altBeatsOnCost = !!cheaper
  const rows = [
    ['Advertised fare', 'fare', v => money(v)],
    ['Expected total time', 'expected_minutes', v => `${Math.round(v)} min`],
    ['Availability', 'p_match', v => supplyBand(v).label.replace(' availability', '')],
    ['Booking succeeds', 'p_success', pct],
    ['Cancelled after accepting', 'p_cancel', pct],
    ['Expected waiting lost', 'expected_wasted_min', v => `${Math.round(v)} min`],
    ['Expected attempts', 'expected_attempts', v => v.toFixed(2)],
    ['Expected cost', 'expected_cost', v => money(v)],
  ]

  return (
    <section className="revealbox">
      <div className="eyebrow">What actually happened</div>
      <h2 className="reveal-title">
        The advertised fare was never the price of the journey.
      </h2>

      <div className="livedstrip">
        <div><span>{lived.attempts}</span> attempt{lived.attempts > 1 ? 's' : ''}</div>
        <div><span>{money(lived.advertised)}</span> advertised</div>
        <div><span>{lived.settled ? money(lived.paid) : '—'}</span> actually paid</div>
        <div><span>{Math.round(lived.wasted_min)} min</span> lost</div>
      </div>

      <div className="cmp2">
        <table>
          <thead>
            <tr>
              <th></th>
              <th>{chosen.display_name}<small>what you chose</small></th>
              {alt && (
                <th className={altBeatsOnCost ? 'win' : ''}>
                  {alt.display_name}
                  <small>{altBeatsOnCost
                    ? 'what the engine picked'
                    : 'the more reliable option'}</small>
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, key, fmt]) => (
              <tr key={key} className={key === 'expected_cost' ? 'big' : ''}>
                <td>{label}</td>
                <td>{fmt(chosen[key])}</td>
                {alt && <td className={altBeatsOnCost ? 'win' : ''}>{fmt(alt[key])}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <ul className="reveal-why">
        {reveal.narrative.map((n, i) => <li key={i}>{n}</li>)}
        {alt && !altBeatsOnCost && (
          <li>
            This time the option you picked <b>was</b> the cheapest in expectation —
            {' '}{alt.display_name} completes {pct(alt.p_success)} of the time against
            {' '}{pct(chosen.p_success)}, but at {money(alt.expected_cost)} against
            {' '}{money(chosen.expected_cost)} you would be buying reliability, not
            saving money. The engine says so rather than inventing a saving.
          </li>
        )}
      </ul>

      <div className="reveal-notes">
        <p><b>How this was known.</b> {reveal.method_note}</p>
        <p><b>On causation.</b> {reveal.causality_note}</p>
      </div>

      <div className="reveal-next">
        <button className="reveal-cta" type="button" onClick={() => onExplore('insights')}>
          View mobility insights
        </button>
        <button className="linkish inline" type="button" onClick={() => onExplore('intelligence')}>
          See the engine behind it
        </button>
      </div>
    </section>
  )
}
