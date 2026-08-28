import { useEffect, useState } from 'react'
import { ApiError, getInsights } from './api.js'

/* ===========================================================================
   Visual evidence that this is a market phenomenon, not a quirk of one
   booking. Every panel is a relationship between observed quantities; the
   copy is careful to call them associations, because that is what they are.
   =========================================================================== */

const pct = (x) => `${Math.round((x ?? 0) * 100)}%`

export default function Insights({ onExplore }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getInsights()
      .then(setData)
      .catch(e => setError(e instanceof ApiError ? e
        : new ApiError('Could not load insights.', 'unknown')))
  }, [])

  if (error) return <div className="errbox" role="alert"><b>{error.message}</b></div>
  if (!data) return <div className="card card-pad"><div className="skeleton" style={{ width: '50%', height: 20 }} /></div>

  return (
    <div className="ins">
      <div className="ins-head">
        <div className="eyebrow">Mobility insights</div>
        <h2>Why the cheapest quote is not the cheapest journey</h2>
        <p>
          Every panel below is drawn from the booking history behind this demo. They show how
          fare, demand, supply, acceptance and cancellation move together — which is the market
          condition the prediction engine exists to read.
        </p>
      </div>

      <div className="ins-caution">
        <b>These are associations, not causes.</b> {data.causality_note}
      </div>

      <div className="ins-grid">
        {data.panels.map(p => <Panel key={p.key} p={p} />)}
      </div>

      <div className="ins-next">
        <button className="reveal-cta" type="button" onClick={() => onExplore('intelligence')}>
          See the engine behind it
        </button>
        <button className="linkish inline" type="button" onClick={() => onExplore('enterprise')}>
          The same problem at enterprise scale
        </button>
      </div>
    </div>
  )
}

function Panel({ p }) {
  const rows = p.rows || []
  if (!rows.length) return null

  // Which series a panel shows depends on what it is about. Provider panels
  // compare cost; the rest compare the stages a booking can fail at.
  const series = p.key === 'provider'
    ? [['success', 'Completes', 'good'], ['cancellation', 'Cancelled', 'bad']]
    : p.key === 'hour'
      ? [['success', 'Completes', 'good'], ['cancellation', 'Cancelled', 'bad']]
      : [['supply', 'Vehicle found', 'mid'],
         ['acceptance', 'Driver accepts', 'good'],
         ['cancellation', 'Cancelled after accepting', 'bad']]

  return (
    <section className="panel-card">
      <h3>{p.title}</h3>
      <div className="panel-legend">
        {series.map(([k, label, tone]) => (
          <span key={k}><i className={`sw sw-${tone}`} />{label}</span>
        ))}
      </div>

      <div className="panel-rows">
        {rows.map(r => (
          <div className="prow" key={r.label}>
            <div className="prow-lab" title={`${r.n.toLocaleString('en-IN')} bookings`}>
              {r.label}
            </div>
            <div className="prow-bars">
              {series.map(([k, , tone]) => (
                <div className="pbar" key={k}>
                  <div className={`pbar-fill pbar-${tone}`}
                       style={{ width: `${Math.min(100, (r[k] ?? 0) * 100)}%` }} />
                  <span className="pbar-val">{pct(r[k])}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {p.key === 'provider' && (
        <div className="panel-extra">
          <table className="enttable">
            <thead><tr><th>Provider</th><th>Billed ₹/km</th><th>Effective ₹/km</th></tr></thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.label}>
                  <td className="strong">{r.label}</td>
                  <td>{r.cost_per_km?.toFixed(2) ?? '—'}</td>
                  <td className="strong">{r.effective_cost_per_km?.toFixed(2) ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="panel-reading">{p.reading}</p>
    </section>
  )
}
