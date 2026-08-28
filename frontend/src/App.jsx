import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, getCity, getDemo, getModels, getPlaces, recommend } from './api.js'
import { minutes, modeInfo, provWord } from './modes.js'
import JourneyMap from './JourneyMap.jsx'
import Book from './Book.jsx'
import Compare from './Compare.jsx'
import Enterprise from './Enterprise.jsx'
import Insights from './Insights.jsx'
import { AlternativeCard, Checks, Metrics, Timeline } from './Journey.jsx'

const PRESETS = [
  { key: 'cheapest', label: 'Cheapest' },
  { key: 'balanced', label: 'Balanced' },
  { key: 'fastest', label: 'Fastest' },
]

const toLocalInput = (iso) => {
  const d = iso ? new Date(iso) : new Date()
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}T${p(d.getHours())}:${p(d.getMinutes())}`
}

// How often a live answer is recomputed. Long enough not to hammer the engine,
// short enough that "leaving now" stays true.
const LIVE_REFRESH_MS = 60_000

// The backend answers on the study area's clock and sends naive local
// timestamps, so they are read as wall-clock text rather than as instants --
// parsing them as UTC would shift every displayed time by the viewer's offset.
const cityClock = (naiveIso) => {
  if (!naiveIso) return null
  const m = String(naiveIso).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/)
  return m ? `${m[4]}:${m[5]}` : null
}

// "12.9185, 77.6880" / "12.9185 77.6880" — a pasted coordinate pair.
const COORDS = /^\s*(-?\d{1,3}(?:\.\d+)?)\s*[,\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*$/

/** Turn whatever the user typed into something the API understands.
 *
 * The API accepts a place_id, a lat/lon pair, or a free-text label it matches
 * itself. So an unrecognised string is not an error here — it is passed through
 * as a label and the backend gets to say whether it knows the place, in one
 * voice instead of two. */
export function resolvePoint(text, places) {
  const t = (text || '').trim()
  if (!t) return null

  const c = t.match(COORDS)
  if (c) {
    const lat = Number(c[1]), lon = Number(c[2])
    if (Math.abs(lat) <= 90 && Math.abs(lon) <= 180) return { lat, lon, label: t }
  }

  const lower = t.toLowerCase()
  const exact = places.find(p => p.name.toLowerCase() === lower)
  if (exact) return { place_id: exact.place_id }

  const partial = places.filter(p => p.name.toLowerCase().includes(lower))
  if (partial.length === 1) return { place_id: partial[0].place_id }

  return { label: t }
}

/** What the resolver made of what you typed, shown while you type. */
function PointHint({ text, places }) {
  const t = (text || '').trim()
  if (!t) return null
  const r = resolvePoint(t, places)
  if (r.lat != null) {
    return <div className="hint">Coordinates: {r.lat.toFixed(4)}, {r.lon.toFixed(4)}</div>
  }
  if (r.place_id) {
    const p = places.find(x => x.place_id === r.place_id)
    return <div className="hint">Using <b>{p?.name}</b></div>
  }
  const n = places.filter(p => p.name.toLowerCase().includes(t.toLowerCase())).length
  return (
    <div className="hint">
      {n > 1
        ? `${n} places match — keep typing, or pick one from the list.`
        : 'Not a place in this study area. Paste coordinates instead, e.g. 12.9346, 77.5353.'}
    </div>
  )
}

const secondsAgo = (naiveIso, cityNowMs) => {
  if (!naiveIso || !cityNowMs) return null
  const t = Date.parse(`${naiveIso}Z`)
  return Number.isNaN(t) ? null : Math.max(0, Math.round((cityNowMs - t) / 1000))
}

export default function App() {
  const [city, setCity] = useState(null)
  const [places, setPlaces] = useState([])
  const [models, setModels] = useState(null)
  const [bootError, setBootError] = useState(null)
  //  compare  = what will this trip really cost?      (the product)
  //  plan     = the full multi-modal journey planner  (the engine, visible)
  //  enterprise = how should an organisation manage mobility?
  // The story is ordered: a rider books, hits a real failure, and only then
  // is the intelligence revealed. So the front door is the product, and the
  // analytical views stay out of the way until the reveal opens them.
  const [view, setView] = useState('book')

  const [origin, setOrigin] = useState('')
  const [destination, setDestination] = useState('')
  const [departure, setDeparture] = useState(toLocalInput())
  // Live mode is the default: the answer is for right now, and it keeps being
  // for right now. Turning it off pins the departure to whatever is in the box.
  const [live, setLive] = useState(true)
  const [clockOffset, setClockOffset] = useState(0)   // city clock - this device
  const [nowMs, setNowMs] = useState(() => Date.now())
  const [budget, setBudget] = useState(100)
  const [maxTime, setMaxTime] = useState(30)
  const [preference, setPreference] = useState('balanced')
  const [useSliders, setUseSliders] = useState(false)
  const [w, setW] = useState({ cost: 40, time: 40, transfers: 12, comfort: 8 })

  const [result, setResult] = useState(null)
  const [scenario, setScenario] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const resultsRef = useRef(null)
  const lastRequest = useRef(null)

  const symbol = city?.currency_symbol || '₹'

  // --- boot -------------------------------------------------------------
  useEffect(() => {
    let alive = true
    Promise.all([getCity(), getPlaces(), getModels().catch(() => null)])
      .then(([c, p, m]) => {
        if (!alive) return
        setCity(c)
        setPlaces(p.places)
        setModels(m)
        setDeparture(toLocalInput(c.default_departure))
        // The server answers on the study area's clock. Remember the gap so a
        // laptop in another timezone still shows "now" as the city sees it.
        if (c.now) setClockOffset(Date.parse(`${c.now}Z`) - Date.now())
        const nameOf = (id) => p.places.find(x => x.place_id === id)?.name || ''
        const scen = c.demo_scenario
        if (scen && nameOf(scen.origin) && nameOf(scen.destination)) {
          setOrigin(nameOf(scen.origin))
          setDestination(nameOf(scen.destination))
          setBudget(scen.budget)
          setMaxTime(scen.max_time)
          setPreference(scen.preference)
        } else if (p.places.length > 1) {
          setOrigin(p.places[0].name)
          setDestination(p.places[1].name)
        }
      })
      .catch((e) => alive && setBootError(e))
    return () => { alive = false }
  }, [])

  // --- the clock --------------------------------------------------------
  // One second is what a live badge needs; nothing here re-fetches.
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 1000)
    return () => clearInterval(id)
  }, [])

  // `cityNowMs` is the study area's wall clock expressed as if it were UTC, so
  // the getUTC* accessors below read out Bengaluru time on any machine.
  const cityNowMs = nowMs + clockOffset
  const cityMinute = Math.floor(cityNowMs / 60000)
  const cityClockText = useMemo(() => {
    const d = new Date(cityNowMs)
    const p = (n) => String(n).padStart(2, '0')
    return `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}`
  }, [cityNowMs])

  // While live mode is on, the departure box mirrors the clock instead of
  // pretending a timestamp chosen ten minutes ago is still "now".
  useEffect(() => {
    if (!live) return
    setDeparture(toLocalInput(new Date(cityNowMs).toISOString().replace('Z', '')))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, cityMinute])

  // --- submit -----------------------------------------------------------
  const run = useCallback(async (payload, scen, opts = {}) => {
    if (!opts.silent) setBusy(true)
    setError(null)
    try {
      const r = await recommend(payload)
      setResult(r)
      setScenario(scen || null)
      lastRequest.current = { payload, scen }
      if (!opts.silent) {
        requestAnimationFrame(() => resultsRef.current?.scrollTo({ top: 0, behavior: 'smooth' }))
      }
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError('Something went wrong.', 'unknown'))
      if (!opts.silent) setResult(null)
    } finally {
      if (!opts.silent) setBusy(false)
    }
  }, [])

  // --- keep a live answer live -------------------------------------------
  // Re-runs the same question against the current minute. Traffic, headways
  // and whether the metro is still running all move on without us.
  useEffect(() => {
    if (!live || !result || busy) return undefined
    const id = setInterval(() => {
      const prev = lastRequest.current
      if (!prev) return
      run({ ...prev.payload, departure_time: new Date().toISOString() },
          prev.scen, { silent: true })
    }, LIVE_REFRESH_MS)
    return () => clearInterval(id)
  }, [live, result, busy, run])

  const onSubmit = (e) => {
    e.preventDefault()
    const from = resolvePoint(origin, places)
    const to = resolvePoint(destination, places)
    if (!from || !to) {
      setError(new ApiError('Type where you are starting and where you are going.', 'missing_points',
        'Use a place name from the list, or paste coordinates like 12.9346, 77.5353.'))
      return
    }
    if (JSON.stringify(from) === JSON.stringify(to)) {
      setError(new ApiError('Your start and destination are the same place.', 'same_endpoints',
        'Pick two different points.'))
      return
    }
    run({
      origin: from, destination: to,
      // Live mode sends the actual instant. A pinned departure is sent as a
      // bare local timestamp, which the backend reads on the study area's
      // clock -- so "09:00" means 09:00 in Bengaluru wherever the browser is.
      departure_time: live ? new Date().toISOString() : departure,
      budget: Number(budget),
      max_time: Number(maxTime),
      preference,
      weights: useSliders
        ? { cost: w.cost / 100, time: w.time / 100, transfers: w.transfers / 100, comfort: w.comfort / 100 }
        : null,
    })
  }

  const runDemo = async () => {
    setBusy(true); setError(null)
    try {
      const d = await getDemo()
      setResult(d)
      setScenario(d.scenario)
      // reflect the demo back into the form so the user can tweak it
      const s = d.scenario.request
      setOrigin(d.origin?.label || s.origin)
      setDestination(d.destination?.label || s.destination)
      setBudget(s.budget); setMaxTime(s.max_time)
      setPreference(s.preference); setUseSliders(false)
      setDeparture(toLocalInput(s.departure_time))
      lastRequest.current = {
        payload: {
          origin: s.origin, destination: s.destination,
          budget: s.budget, max_time: s.max_time, preference: s.preference,
        },
        scen: d.scenario,
      }
      requestAnimationFrame(() => resultsRef.current?.scrollTo({ top: 0, behavior: 'smooth' }))
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError('Could not load the demo.', 'unknown'))
    } finally {
      setBusy(false)
    }
  }

  const activeModel = models?.models?.find(m => m.active)
  const computedAgo = secondsAgo(result?.computed_at, cityNowMs)

  if (bootError) {
    return (
      <div className="app">
        <Topbar city={null} model={null} />
        <div className="results" style={{ maxWidth: 640, margin: '40px auto' }}>
          <div className="errbox">
            <b>JourneyMind could not start up.</b>
            {bootError.message}
            {bootError.detail && <div style={{ marginTop: 6 }}>{bootError.detail}</div>}
          </div>
        </div>
      </div>
    )
  }

  const explore = (next) => setView(next)

  if (view !== 'plan') {
    return (
      <div className="app">
        <Topbar city={city} model={result?.model_info || activeModel}
                view={view} setView={setView} />
        <div className="wide">
          {view === 'book' && <Book places={places} city={city} onExplore={explore} />}
          {view === 'insights' && <Insights onExplore={explore} />}
          {view === 'intelligence' && <Compare places={places} symbol={symbol} />}
          {view === 'enterprise' && <Enterprise symbol={symbol} />}
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      <Topbar city={city} model={result?.model_info || activeModel}
              view={view} setView={setView} />
      <div className="main">
        <form className="panel" onSubmit={onSubmit}>
          <div className="lede">Where are you going?</div>
          <div className="sub">
            One question, one answer — the best complete journey inside the money and
            time you actually have.
          </div>

          <datalist id="place-options">
            {places.map(p => <option key={p.place_id} value={p.name} />)}
          </datalist>

          <div className="field">
            <label htmlFor="from">From</label>
            <input id="from" list="place-options" value={origin} autoComplete="off"
                   placeholder="Type a place, or paste 12.9185, 77.6880"
                   onChange={e => setOrigin(e.target.value)} />
            <PointHint text={origin} places={places} />
          </div>

          <div className="field">
            <label htmlFor="to">To</label>
            <input id="to" list="place-options" value={destination} autoComplete="off"
                   placeholder="Type a place, or paste 12.9346, 77.5353"
                   onChange={e => setDestination(e.target.value)} />
            <PointHint text={destination} places={places} />
          </div>

          <div className="field">
            <label htmlFor="dep">Departure</label>
            <div className="livebar">
              <button type="button" className={`livetoggle${live ? ' on' : ''}`}
                      aria-pressed={live} onClick={() => setLive(v => !v)}>
                <span className="livedot" />
                {live ? 'Leaving now' : 'Leave now'}
              </button>
              <span className="liveclock" title={city?.timezone || 'Asia/Kolkata'}>
                {cityClockText} {city?.timezone ? city.timezone.split('/')[1] : ''}
              </span>
            </div>
            {/* Never disabled. Editing a departure IS the intent to pin one, so
                typing here switches live mode off rather than refusing input. */}
            <input id="dep" type="datetime-local" value={departure}
                   onChange={e => { setLive(false); setDeparture(e.target.value) }} />
            <div className="hint">
              {live
                ? 'Live: priced for this minute on the Bengaluru clock, and refreshed every minute.'
                : 'Time of day changes the prediction — try 09:00 against 14:00, or 01:00 when nothing is running.'}
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label htmlFor="budget">Budget ({symbol})</label>
              {/* step="any": with min="1" a step of 5 makes the only valid
                  values 1, 6, 11... so 250 and 100 are rejected by HTML5
                  validation and the form silently refuses to submit. */}
              <input id="budget" type="number" min="1" max="100000" step="any"
                     inputMode="decimal"
                     value={budget} onChange={e => setBudget(e.target.value)} />
            </div>
            <div className="field">
              <label htmlFor="maxt">Max time (min)</label>
              <input id="maxt" type="number" min="1" max="1440" step="any"
                     inputMode="decimal"
                     value={maxTime} onChange={e => setMaxTime(e.target.value)} />
            </div>
          </div>

          <div className="field">
            <label>Preference</label>
            <div className="segmented" role="group" aria-label="Preference">
              {PRESETS.map(p => (
                <button key={p.key} type="button"
                        aria-pressed={!useSliders && preference === p.key}
                        onClick={() => { setPreference(p.key); setUseSliders(false) }}>
                  {p.label}
                </button>
              ))}
            </div>
            <button type="button" className="linkish" style={{ marginTop: 8 }}
                    onClick={() => setUseSliders(v => !v)}>
              {useSliders ? 'Use a preset instead' : 'Set your own priorities'}
            </button>
            {useSliders && (
              <div className="sliders">
                {[['cost', 'Cost'], ['time', 'Time'], ['transfers', 'Transfers'], ['comfort', 'Comfort']]
                  .map(([k, lbl]) => (
                    <div className="slider-row" key={k}>
                      <span>{lbl}</span>
                      <input type="range" min="0" max="100" value={w[k]}
                             onChange={e => setW({ ...w, [k]: Number(e.target.value) })} />
                      <output>{w[k]}</output>
                    </div>
                  ))}
                <div className="hint">Weights are normalised, so only the balance between them matters.</div>
              </div>
            )}
          </div>

          <button className="go" type="submit" disabled={busy}>
            {busy ? 'Finding your journey…' : 'Find My Best Journey'}
          </button>
          <button className="linkish" type="button" onClick={runDemo} disabled={busy}>
            Try the demo: Wipro Sarjapur Rd → PES University
          </button>

          {city && <DataNotice city={city} />}
        </form>

        <div className="results" ref={resultsRef}>
          {error && (
            <div className="errbox" role="alert">
              <b>{error.message}</b>
              {error.detail && <div>{error.detail}</div>}
            </div>
          )}

          {busy && <LoadingSkeleton />}

          {!busy && !result && !error && <Welcome onDemo={runDemo} city={city} />}

          {!busy && result && (
            <Results result={result} scenario={scenario} symbol={symbol}
                     budget={Number(budget)} maxTime={Number(maxTime)}
                     routes={city?.routes} live={live} computedAgo={computedAgo} />
          )}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// The storytelling order, and nothing more. Booking is the landing view because
// the demo works best when the problem is felt before it is explained -- but
// every view stays reachable from the first second. Hiding navigation to make a
// narrative land takes the product away from anyone who came to see the rest of
// it, which is a worse failure than a demo that opens on the wrong tab.
const VIEWS = [
  { key: 'book', label: 'Book a ride', hint: 'Find and book a ride' },
  { key: 'insights', label: 'Insights', hint: 'Why the cheapest quote is not the cheapest journey' },
  { key: 'intelligence', label: 'Intelligence', hint: 'Expected cost across every option' },
  { key: 'plan', label: 'Journey planner', hint: 'The full multi-modal planner' },
  { key: 'enterprise', label: 'Enterprise', hint: 'The same problem at organisation scale' },
]

function Topbar({ city, model, view, setView }) {
  return (
    <header className="topbar">
      <div className="brand">
        <div className="mark">Journey<span>Mind</span></div>
        <div className="tag">Book the ride that actually gets you there.</div>
      </div>
      {setView && (
        <nav className="viewnav" aria-label="View">
          {VIEWS.map(v => (
            <button key={v.key} type="button" title={v.hint}
                    className={view === v.key ? 'on' : ''}
                    aria-pressed={view === v.key}
                    onClick={() => setView(v.key)}>{v.label}</button>
          ))}
        </nav>
      )}
      <div className="spacer" />
      {/* The booking screen must read as a ride app, not an AI demo. The model
          badge appears once the rider has been through the reveal. */}
      {model && view !== 'book' && (
        <span className="pill pill-model" title={model.notes || ''}>
          {(model.model || model.display || 'model')} · prototype
        </span>
      )}
      {city && <span className="pill pill-demo" title={city.data_notice?.notes || ''}>
        {city.data_notice?.label || 'Demo data'}
      </span>}
    </header>
  )
}

function DataNotice({ city }) {
  const fp = city.data_notice?.fare_provenance || {}
  return (
    <details className="disclose" style={{ marginTop: 22 }}>
      <summary>Where these numbers come from</summary>
      <p style={{ fontSize: 12.5, color: 'var(--ink-3)', marginTop: 8 }}>
        Study area: <b>{city.display_name}</b> — {city.counts?.nodes} nodes,{' '}
        {city.counts?.road_edges} road links, {city.counts?.transit_edges} transit links
        over {city.counts?.routes} routes.
      </p>
      <table className="provtable">
        <tbody>
          {Object.entries(fp).map(([mode, prov]) => (
            <tr key={mode}>
              <td>{modeInfo(mode).label}</td>
              <td>{provWord(prov)} fare</td>
            </tr>
          ))}
          <tr><td>Travel times</td><td>predicted by the model</td></tr>
        </tbody>
      </table>
      <p style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 8 }}>{city.data_notice?.notes}</p>
    </details>
  )
}

function Welcome({ onDemo, city }) {
  return (
    <div className="empty">
      <h2 style={{ fontSize: 20, color: 'var(--ink)', marginBottom: 8 }}>
        Nobody tells you the best <i>combination</i> of rides.
      </h2>
      <p style={{ maxWidth: 520, margin: '0 auto 18px' }}>
        Ride apps price one ride. Map apps ignore your budget. JourneyMind looks at the
        whole trip — bike taxi, auto, cab, metro, bus — and finds the best complete journey
        inside the money and the time you actually have.
      </p>
      <button className="linkish" style={{ maxWidth: 380, margin: '0 auto' }} onClick={onDemo}>
        Run the demo: Wipro Sarjapur Rd → PES University
      </button>
      {city && (
        <p style={{ marginTop: 20, fontSize: 12 }}>
          Study area: {city.display_name}
        </p>
      )}
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="card card-pad">
      <div className="eyebrow">Working it out</div>
      <div className="skeleton" style={{ width: '70%', height: 22 }} />
      <div className="skeleton" style={{ width: '45%' }} />
      <div className="skeleton" style={{ width: '85%' }} />
      <div className="skeleton" style={{ width: '60%' }} />
      <p style={{ color: 'var(--ink-3)', fontSize: 12.5, marginTop: 14 }}>
        Building the graph, predicting edge travel times, generating candidate
        journeys, filtering and ranking.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
function Results({ result, scenario, symbol, budget, maxTime, routes, live, computedAgo }) {
  const r = result
  const mapJourney = r.recommended || r.fallbacks?.[0]?.journey || null
  const usedBudget = scenario?.request?.budget ?? budget
  const usedTime = scenario?.request?.max_time ?? maxTime

  return (
    <>
      {scenario && (
        <div className="card card-pad" style={{ background: 'var(--paper-2)' }}>
          <div className="eyebrow">Demo scenario · {scenario.title}</div>
          <p style={{ margin: 0, fontSize: 13.5, color: 'var(--ink-2)' }}>{scenario.description}</p>
        </div>
      )}

      {!r.feasible && (
        <div className="nofit">
          <h3>{r.message}</h3>
          <p>
            Here is what we found instead. Each option below breaks one of your limits,
            and says which one.
          </p>
        </div>
      )}

      {r.recommended && (
        <div className="card card-hero card-pad">
          <div className="eyebrow eyebrow-row">
            <span>Your best journey</span>
            <LiveStamp live={live} computedAgo={computedAgo}
                       departure={r.departure_time} computedAt={r.computed_at} />
          </div>
          <div className="headline">{r.explanation?.headline}</div>
          <Metrics journey={r.recommended} symbol={symbol} />
          <Checks constraints={r.recommended.constraints} budget={usedBudget}
                  maxTime={usedTime} symbol={symbol} />

          {(r.explanation?.reasons?.length || r.explanation?.comparisons?.length) > 0 && (
            <div className="why">
              <h4>Why this route?</h4>
              <ul>
                {r.explanation.reasons.map((x, i) => <li key={`r${i}`}>{x}</li>)}
                {r.explanation.comparisons.map((x, i) => <li key={`c${i}`}>{x}</li>)}
              </ul>
            </div>
          )}

          {r.explanation?.caveats?.length > 0 && (
            <div className="caveats">
              <h4>What we are not certain about</h4>
              <ul>{r.explanation.caveats.map((x, i) => <li key={i}>{x}</li>)}</ul>
            </div>
          )}

          <Timeline journey={r.recommended}
                    originLabel={r.origin?.label} destLabel={r.destination?.label} />
        </div>
      )}

      <ModeComparison rows={r.mode_comparison} best={r.recommended} symbol={symbol} />

      <div className="card" style={{ padding: 12 }}>
        <JourneyMap journey={mapJourney} origin={r.origin} destination={r.destination}
                    routes={routes} />
        <p style={{ fontSize: 11.5, color: 'var(--ink-3)', margin: '8px 4px 2px' }}>
          Map data © OpenStreetMap contributors. Route lines follow the study-area
          graph, not turn-by-turn street geometry.
        </p>
      </div>

      {r.alternatives?.map((alt, i) => (
        <AlternativeCard key={alt.journey.journey_id} alt={alt} index={i}
                         budget={usedBudget} maxTime={usedTime} symbol={symbol} />
      ))}

      {r.fallbacks?.map((f, i) => (
        <div className="card card-pad" key={f.journey.journey_id}>
          <div className="eyebrow eyebrow-muted">{f.label}</div>
          <div className="alt-head">
            <div>
              <div className="alt-title">
                {f.journey.modes.filter(m => m !== 'walk').map(m => modeInfo(m).label).join(' → ')}
              </div>
              <div style={{ marginTop: 6 }}>
                <span className="badge badge-miss">Breaks a limit</span>
              </div>
            </div>
            <div className="alt-figs">
              <div className="f"><b>{f.journey.total_cost.display}</b><i>{provWord(f.journey.total_cost.provenance)}</i></div>
              <div className="f"><b>{minutes(f.journey.total_min)}</b><i>predicted</i></div>
              <div className="f"><b>{f.journey.transfers}</b><i>transfers</i></div>
            </div>
          </div>
          <div className="alt-reason">{f.why}</div>
          <Checks constraints={f.journey.constraints} budget={usedBudget}
                  maxTime={usedTime} symbol={symbol} />
          <details className="disclose">
            <summary>Show the route</summary>
            <Timeline journey={f.journey} />
          </details>
        </div>
      ))}

      <HowItDecided result={r} />
    </>
  )
}

/** The comparison the product exists to make: what every single-mode option
 *  would have cost, including the ones you cannot afford. This is the table
 *  from the project documentation's worked example, computed live. */
function ModeComparison({ rows, best, symbol }) {
  if (!rows?.length) return null
  return (
    <div className="card card-pad">
      <div className="eyebrow">What each app would have told you</div>
      <p style={{ fontSize: 13, color: 'var(--ink-3)', margin: '2px 0 12px' }}>
        One ride, one mode — priced whether or not you can afford it. None of these
        is a recommendation; each says which of your limits it breaks.
      </p>
      <table className="comparetable">
        <tbody>
          {rows.map(row => {
            const info = modeInfo(row.mode)
            return (
              <tr key={row.mode} className={row.feasible ? '' : 'over'}>
                <td className="m">
                  <span className="swatch" style={{ background: info.colour }} />
                  {info.label}
                </td>
                <td className="c">{row.total_cost.display}</td>
                <td className="t">{minutes(row.total_min)}</td>
                <td className="v">{row.verdict}</td>
              </tr>
            )
          })}
          {best && (
            <tr className="winner">
              <td className="m">
                {best.modes.filter(m => m !== 'walk').map(m => modeInfo(m).label).join(' + ')}
              </td>
              <td className="c">{best.total_cost.display}</td>
              <td className="t">{minutes(best.total_min)}</td>
              <td className="v">JourneyMind — the combination no single app offers</td>
            </tr>
          )}
        </tbody>
      </table>
      <p style={{ fontSize: 11.5, color: 'var(--ink-3)', margin: '10px 2px 0' }}>
        Ride-hailing figures are {symbol} ranges from a transparent fare model, not quotes.
        Availability is not modelled — see the data notice.
      </p>
    </div>
  )
}

function LiveStamp({ live, computedAgo, departure, computedAt }) {
  const dep = cityClock(departure)
  const at = cityClock(computedAt)
  if (!at) return null
  const age = computedAgo == null
    ? null
    : computedAgo < 75 ? 'just now' : `${Math.round(computedAgo / 60)} min ago`
  return (
    <span className={`livestamp${live ? ' on' : ''}`}
          title={`Departure ${dep}, computed at ${at}, study-area clock`}>
      {live ? <span className="livedot" /> : null}
      {live ? `Live · leaving ${dep}` : `Departure ${dep}`}
      {age ? ` · computed ${age}` : ''}
    </span>
  )
}

function HowItDecided({ result }) {
  const p = result.pipeline || {}
  const m = result.model_info || {}
  return (
    <details className="disclose card card-pad" style={{ marginTop: 4 }}>
      <summary>How JourneyMind decided</summary>
      <table className="provtable">
        <tbody>
          <tr><td>Graph</td><td>{p.graph?.nodes} nodes, {p.graph?.edges} edges ({p.graph?.request_edges_added} added for this request)</td></tr>
          <tr><td>Travel-time model</td><td>{m.model} — {m.status}{m.fell_back ? ` (fell back from ${m.requested})` : ''}</td></tr>
          <tr><td>Predicted congestion</td><td>×{p.prediction?.mean_congestion_ratio} against free flow at {p.prediction?.hour_local != null ? `${String(Math.floor(p.prediction.hour_local)).padStart(2, '0')}:${String(Math.round((p.prediction.hour_local % 1) * 60)).padStart(2, '0')}` : 'this hour'}</td></tr>
          <tr><td>Not running now</td><td>{p.prediction?.routes_out_of_service?.length ? p.prediction.routes_out_of_service.join(', ') : 'every route is in service at this hour'}</td></tr>
          <tr><td>Candidates</td><td>{p.candidates?.paths_found} paths → {p.candidates?.after_deduplication} distinct journeys</td></tr>
          <tr><td>Constraint filter</td><td>{p.constraints?.kept} kept · {p.constraints?.removed_over_budget} over budget · {p.constraints?.removed_over_time} over time</td></tr>
          <tr><td>Pareto frontier</td><td>{p.pareto?.on_frontier != null ? `${p.pareto.on_frontier} non-dominated, ${p.pareto.dominated_removed} dominated removed` : 'not run — nothing was feasible'}</td></tr>
          <tr><td>Weights used</td><td>{Object.entries(result.weights || {}).map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(' · ')}</td></tr>
          <tr><td>Computed in</td><td>{p.elapsed_ms} ms</td></tr>
        </tbody>
      </table>
      {m.validation_metrics?.test && (
        <p style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 10 }}>
          Model test-split error on the bundled dataset: MAE {m.validation_metrics.test.MAE_min} min,
          MAPE {m.validation_metrics.test.MAPE_pct}%. This describes the bundled synthetic
          data, not a real city.
        </p>
      )}
    </details>
  )
}
