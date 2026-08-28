import { useCallback, useEffect, useState } from 'react'
import { ApiError, compare } from './api.js'
import { minutes, modeInfo } from './modes.js'

const PRIORITIES = [
  { key: 'cheapest', label: 'Cheapest', hint: 'Lowest expected cost — not the lowest sticker price' },
  { key: 'balanced', label: 'Balanced', hint: 'Cost and time together, with a reliability floor' },
  { key: 'fastest', label: 'Fastest', hint: 'Including the time lost to failed bookings' },
  { key: 'reliable', label: 'Most reliable', hint: 'Highest chance of completing first time' },
]

const pct = (x) => `${Math.round((x ?? 0) * 100)}%`

/** Risk band from the chance a booking falls through entirely. */
function riskBand(o) {
  if (o.service_class !== 'hailed') return { key: 'none', label: 'No booking risk' }
  const c = o.reliability.p_cancel
  if (c < 0.12) return { key: 'low', label: `${pct(c)} cancellation risk` }
  if (c < 0.22) return { key: 'mid', label: `${pct(c)} cancellation risk` }
  return { key: 'high', label: `${pct(c)} cancellation risk` }
}

export default function Compare({ places, symbol = '₹' }) {
  const [origin, setOrigin] = useState('')
  const [destination, setDestination] = useState('')
  const [priority, setPriority] = useState('balanced')
  const [budget, setBudget] = useState('')
  const [maxTime, setMaxTime] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!places?.length || origin) return
    const wipro = places.find(p => p.place_id === 'pl_wipro_sarjapur')
    const pes = places.find(p => p.place_id === 'pl_pes_university')
    setOrigin((wipro || places[0]).name)
    setDestination((pes || places[1] || places[0]).name)
  }, [places, origin])

  const run = useCallback(async (nextPriority) => {
    if (!origin.trim() || !destination.trim()) {
      setError(new ApiError('Type where you are starting and where you are going.', 'missing'))
      return
    }
    setBusy(true); setError(null)
    try {
      setResult(await compare({
        origin: origin.trim(),
        destination: destination.trim(),
        priority: nextPriority || priority,
        budget: budget === '' ? null : Number(budget),
        max_time: maxTime === '' ? null : Number(maxTime),
      }))
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError('Something went wrong.', 'unknown'))
      setResult(null)
    } finally { setBusy(false) }
  }, [origin, destination, priority, budget, maxTime])

  // Changing priority re-ranks immediately — that responsiveness is the point:
  // the same options, a different answer, because you asked a different question.
  const choosePriority = (k) => { setPriority(k); if (result) run(k) }

  return (
    <div className="cmp">
      <form className="cmp-form" onSubmit={(e) => { e.preventDefault(); run() }}>
        <datalist id="cmp-places">
          {places?.map(p => <option key={p.place_id} value={p.name} />)}
        </datalist>
        <div className="cmp-row">
          <div className="field">
            <label htmlFor="cfrom">From</label>
            <input id="cfrom" list="cmp-places" value={origin} autoComplete="off"
                   placeholder="Type a place or paste coordinates"
                   onChange={e => setOrigin(e.target.value)} />
          </div>
          <div className="field">
            <label htmlFor="cto">To</label>
            <input id="cto" list="cmp-places" value={destination} autoComplete="off"
                   placeholder="Type a place or paste coordinates"
                   onChange={e => setDestination(e.target.value)} />
          </div>
          <div className="field narrow">
            <label htmlFor="cbud">Budget ({symbol})</label>
            <input id="cbud" type="number" min="1" step="any" inputMode="decimal"
                   placeholder="any"
                   value={budget} onChange={e => setBudget(e.target.value)} />
          </div>
          <div className="field narrow">
            <label htmlFor="ctime">Max time</label>
            <input id="ctime" type="number" min="1" step="any" inputMode="decimal"
                   placeholder="any"
                   value={maxTime} onChange={e => setMaxTime(e.target.value)} />
          </div>
          <button className="go cmp-go" type="submit" disabled={busy}>
            {busy ? 'Pricing…' : 'Compare'}
          </button>
        </div>

        <div className="prio">
          <span className="prio-label">Priority</span>
          {PRIORITIES.map(p => (
            <button key={p.key} type="button" title={p.hint}
                    className={`prio-btn${priority === p.key ? ' on' : ''}`}
                    aria-pressed={priority === p.key}
                    onClick={() => choosePriority(p.key)}>{p.label}</button>
          ))}
          <span className="prio-hint">{PRIORITIES.find(p => p.key === priority)?.hint}</span>
        </div>
      </form>

      {error && (
        <div className="errbox" role="alert">
          <b>{error.message}</b>{error.detail && <div>{error.detail}</div>}
        </div>
      )}

      {busy && <div className="card card-pad"><div className="skeleton" style={{ width: '60%', height: 20 }} />
        <div className="skeleton" style={{ width: '85%' }} /><div className="skeleton" style={{ width: '40%' }} />
        <p style={{ color: 'var(--ink-3)', fontSize: 12.5, marginTop: 12 }}>
          Routing every mode, predicting travel times, then pricing each option through the booking lifecycle.
        </p></div>}

      {!busy && result && <Results result={result} symbol={symbol} />}

      {!busy && !result && !error && (
        <div className="empty">
          <h2 style={{ fontSize: 20, color: 'var(--ink)', marginBottom: 8 }}>
            The cheapest fare is not the cheapest trip.
          </h2>
          <p style={{ maxWidth: 540, margin: '0 auto' }}>
            Every app shows you an advertised fare. None of them tells you that a third of those
            bookings fall through, that you lose six minutes finding out, and that the replacement
            ride costs more. Enter a trip and see what each option is really expected to cost.
          </p>
        </div>
      )}
    </div>
  )
}

function Results({ result, symbol }) {
  const notice = result.data_notice
  return (
    <>
      <div className="verdict">
        <div className="verdict-badge">{result.headline}</div>
        <ul className="verdict-why">
          {result.reasoning.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      </div>

      <div className="cards">
        {result.options.map(o => <OptionCard key={o.provider_id} o={o} symbol={symbol} />)}
      </div>

      <details className="disclose card card-pad">
        <summary>How the expected cost is calculated</summary>
        <p style={{ fontSize: 13.5, color: 'var(--ink-2)', marginTop: 10 }}>
          Each option is run through the booking lifecycle as an absorbing Markov chain. One attempt
          succeeds with <code>P(match) × P(accept) × (1 − P(cancel))</code>; a failed attempt costs
          time and the retry is priced higher. The outcome space is small enough to enumerate
          exactly, so the expected cost, the spread and the chance of giving up are computed rather
          than sampled.
        </p>
        <table className="provtable">
          <tbody>
            <tr><td>Options priced</td><td>{result.pipeline?.quotes_returned} of {result.pipeline?.providers_queried} providers</td></tr>
            <tr><td>Fallback if all attempts fail</td><td>{result.pipeline?.fallback
              ? `${result.pipeline.fallback.label} at ${symbol}${result.pipeline.fallback.cost}` : 'none available'}</td></tr>
            <tr><td>Computed in</td><td>{result.pipeline?.elapsed_ms} ms</td></tr>
          </tbody>
        </table>
        <p style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 10 }}>
          <b>{notice?.label}.</b> {notice?.detail}
        </p>
      </details>
    </>
  )
}

function OptionCard({ o, symbol }) {
  const info = modeInfo(o.mode)
  const e = o.expected
  const risk = riskBand(o)
  const dimmed = !o.available || !o.feasible

  // An option with no route has no numbers to show. The API stopped publishing
  // a fabricated expected cost for it -- that figure was the price of the
  // FALLBACK, and printing it here read as "Walk: ₹25, 0 min".
  if (!e) {
    return (
      <article className="ocard dim">
        <header className="ocard-head">
          <span className="ocard-dot" style={{ background: info.colour }} />
          <h3>{o.display_name}</h3>
          {o.provider_name && <span className="ocard-via">{o.provider_name}</span>}
        </header>
        <div className="ocard-block">{o.unavailable_reason || 'Not available for this trip'}</div>
      </article>
    )
  }

  return (
    <article className={`ocard${o.recommended ? ' best' : ''}${dimmed ? ' dim' : ''}`}>
      {o.recommended && <div className="ocard-flag">Recommended</div>}
      <header className="ocard-head">
        <span className="ocard-dot" style={{ background: info.colour }} />
        <h3>{o.display_name}</h3>
        {o.provider_name && <span className="ocard-via">{o.provider_name}</span>}
      </header>

      <div className="ocard-fare">{o.fare.display}</div>
      <div className="ocard-sub">advertised{o.fare.surge_multiplier > 1.001
        ? ` · includes ${Math.round((o.fare.surge_multiplier - 1) * 100)}% surge` : ''}</div>

      <dl className="ocard-stats">
        <div><dt>Door to door</dt><dd>{minutes(e.expected_minutes)}</dd></div>
        <div><dt>Pickup</dt><dd>{o.pickup_min > 0 ? `${Math.round(o.pickup_min)} min` : '—'}</dd></div>
        <div><dt>Completes</dt><dd>{pct(e.p_success)}</dd></div>
      </dl>

      <div className={`ocard-risk risk-${risk.key}`}>{risk.label}</div>

      <div className="ocard-expected">
        <div className="lbl">Expected cost</div>
        <div className="val">{e.expected_cost_display}</div>
        <div className="delta">
          {e.surcharge > 0.5
            ? `+${symbol}${Math.round(e.surcharge)} over the advertised fare`
            : e.is_blended
              ? `below the fare only because ${pct(e.substitution_share)} of the time you end up on ${e.fallback_label}`
              : 'matches the advertised fare'}
        </div>
      </div>

      {e.expected_wasted_min >= 1 && (
        <div className="ocard-note">
          ~{Math.round(e.expected_wasted_min)} min typically lost to failed requests
          {e.expected_attempts > 1.05 ? ` · ${e.expected_attempts.toFixed(1)} attempts on average` : ''}
        </div>
      )}

      {!o.available && <div className="ocard-block">{o.unavailable_reason}</div>}
      {o.available && !o.within_budget && <div className="ocard-block">The fare is over your budget</div>}
      {o.available && !o.within_time && <div className="ocard-block">Slower than your time limit</div>}
      {o.available && o.budget_at_risk && <div className="ocard-block warn">Fits your budget, but not once failed attempts are paid for</div>}
      {o.available && o.time_at_risk && <div className="ocard-block warn">Fits your time limit, but not once failed attempts are counted</div>}

      <details className="ocard-more">
        <summary>What could happen</summary>
        <table className="outcomes">
          <tbody>
            {e.outcomes.map((x, i) => (
              <tr key={i}>
                <td className="p">{pct(x.probability)}</td>
                <td className="c">{symbol}{Math.round(x.cost)}</td>
                <td className="l">{x.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {o.service_class === 'hailed' && (
          <p className="basis">
            No vehicle {pct(1 - o.reliability.p_match)} · declines {pct(1 - o.reliability.p_accept)} ·
            cancels after accepting {pct(o.reliability.p_cancel)}.<br />
            <span>{o.reliability.basis}</span>
          </p>
        )}
        {o.notes?.map((n, i) => <p className="basis" key={i}>{n}</p>)}
      </details>
    </article>
  )
}
