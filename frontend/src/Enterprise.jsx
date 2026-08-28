import { useCallback, useEffect, useState } from 'react'

// The facet list is ids, because that is what the filter sends back. What a
// person reads is not: `bike_taxi` in a dropdown is a database key on screen.
const PROVIDER_LABEL = {
  bike_taxi: 'Bike taxi', auto: 'Auto', cab: 'Cab', metro: 'Metro', bus: 'Bus',
}
import { ApiError, enterpriseAudit, enterpriseFacets, enterpriseOverview } from './api.js'

const DEMO_KEY = 'demo-analyst-key'
const money = (x, s = '₹') => x == null ? '—' : `${s}${Math.round(x).toLocaleString('en-IN')}`
const pct = (x) => x == null ? '—' : `${(x * 100).toFixed(1)}%`

export default function Enterprise({ symbol = '₹' }) {
  const [apiKey, setApiKey] = useState(DEMO_KEY)
  const [facets, setFacets] = useState(null)
  const [data, setData] = useState(null)
  const [audit, setAudit] = useState(null)
  const [filters, setFilters] = useState({ campus: '', provider: '', employee_group: '', mode: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const load = useCallback(async (key, f) => {
    setBusy(true); setError(null)
    try {
      const [fc, ov, ad] = await Promise.all([
        enterpriseFacets(key),
        enterpriseOverview(key, f),
        enterpriseAudit(key).catch(() => null),
      ])
      setFacets(fc.facets); setData(ov); setAudit(ad)
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError('Could not load the dashboard.', 'unknown'))
      setData(null)
    } finally { setBusy(false) }
  }, [])

  useEffect(() => { load(apiKey, filters) }, [])          // eslint-disable-line

  const setFilter = (k, v) => {
    const next = { ...filters, [k]: v }
    setFilters(next); load(apiKey, next)
  }

  if (error) {
    return (
      <div className="ent">
        <div className="errbox" role="alert">
          <b>{error.message}</b>
          {error.detail && <div>{error.detail}</div>}
        </div>
        <div className="card card-pad">
          <div className="eyebrow">Access</div>
          <p style={{ fontSize: 13.5, color: 'var(--ink-2)' }}>
            Enterprise endpoints expose population-level data, so they are never open. This
            deployment accepts a demo key; a real one sets <code>JM_API_KEYS</code>.
          </p>
          <div className="field" style={{ maxWidth: 340 }}>
            <label htmlFor="k">X-API-Key</label>
            <input id="k" value={apiKey} onChange={e => setApiKey(e.target.value)} />
          </div>
          <button className="go" style={{ maxWidth: 200 }}
                  onClick={() => load(apiKey, filters)}>Retry</button>
        </div>
      </div>
    )
  }

  const o = data?.overview
  return (
    <div className="ent">
      <div className="ent-head">
        <div>
          <div className="eyebrow">Enterprise mobility intelligence</div>
          <h2 className="ent-title">How should this organisation manage mobility?</h2>
        </div>
        {data?.principal && (
          <span className="chip chip-role">
            {data.principal.role}{data.principal.demo ? ' · demo key' : ''}
          </span>
        )}
      </div>

      {data && (
        <div className="ent-notice">
          <b>Demo dataset.</b> {data.data_note} Groups smaller than {data.cohort_floor} trips
          are hidden rather than rounded, so no individual can be identified from a cell.
        </div>
      )}

      {facets && (
        <div className="ent-filters">
          <Sel label="Campus" value={filters.campus} onChange={v => setFilter('campus', v)}
               options={facets.campuses.map(c => [c.id, c.name])} />
          <Sel label="Provider" value={filters.provider} onChange={v => setFilter('provider', v)}
               options={facets.providers.map(p => [p, PROVIDER_LABEL[p] || p])} />
          <Sel label="Team" value={filters.employee_group} onChange={v => setFilter('employee_group', v)}
               options={facets.employee_groups.map(g => [g, g])} />
          <Sel label="Mode" value={filters.mode} onChange={v => setFilter('mode', v)}
               options={facets.modes.map(m => [m, PROVIDER_LABEL[m] || m])} />
          {facets.date_range && (
            <div className="ent-range">{facets.date_range.from} → {facets.date_range.to}</div>
          )}
        </div>
      )}

      {busy && <div className="card card-pad"><div className="skeleton" style={{ width: '50%', height: 22 }} />
        <div className="skeleton" style={{ width: '80%' }} /></div>}

      {!busy && o && o.bookings > 0 && (
        <>
          <div className="kpis">
            <Kpi label="Transportation spend" value={money(o.total_spend, symbol)}
                 sub={`${o.completed_trips.toLocaleString('en-IN')} completed trips`} />
            <Kpi label="Booking success" value={pct(o.booking_success_rate)}
                 sub={`${pct(o.no_supply_rate)} found no vehicle`} tone={o.booking_success_rate < 0.7 ? 'bad' : 'ok'} />
            <Kpi label="Cancellation rate" value={pct(o.cancellation_rate)}
                 sub="of bookings a driver had accepted" tone={o.cancellation_rate > 0.15 ? 'bad' : 'ok'} />
            <Kpi label="Cost of failure" value={money(o.wasted_minutes_cost, symbol)}
                 sub={`${Math.round(o.wasted_minutes).toLocaleString('en-IN')} minutes lost to failed bookings`} tone="warn" />
            <Kpi label="Mean trip cost" value={money(o.mean_trip_cost, symbol)}
                 sub={`${o.mean_distance_km} km average`} />
            <Kpi label="SLA breaches" value={o.sla_breaches.toLocaleString('en-IN')}
                 sub={`over ${o.sla_minutes} min door to door · ${pct(o.sla_breach_rate)}`}
                 tone={o.sla_breach_rate > 0.05 ? 'bad' : 'ok'} />
          </div>

          {data.insights?.length > 0 && (
            <div className="card card-pad">
              <div className="eyebrow">AI insights</div>
              <div className="insights">
                {data.insights.map((i, n) => (
                  <div className={`insight sev-${i.severity}`} key={n}>
                    <div className="insight-head">
                      <span className={`chip chip-${i.kind}`}>{i.kind}</span>
                      <b>{i.title}</b>
                    </div>
                    <p>{i.detail}</p>
                  </div>
                ))}
              </div>
              <p style={{ fontSize: 11.5, color: 'var(--ink-3)', margin: '10px 2px 0' }}>
                <b>observation</b> = arithmetic over the booking history.
                <b> prediction</b> = the model extrapolating. They are never blended.
              </p>
            </div>
          )}

          <div className="card card-pad">
            <div className="eyebrow">Provider scorecard</div>
            <p className="ent-sub">
              Ranked by what a kilometre <i>actually</i> costs — the billed rate divided by the share
              of bookings that complete. A provider can be cheapest per km and worst on this.
            </p>
            <div className="tblwrap">
              <table className="enttable">
                <thead><tr>
                  <th>Provider</th><th>Bookings</th><th>Success</th><th>Cancels</th>
                  <th>{symbol}/km billed</th><th>{symbol}/km adjusted</th><th>Spend</th>
                </tr></thead>
                <tbody>
                  {data.providers.filter(p => !p.suppressed).map(p => (
                    <tr key={p.display_name || p.provider_id}>
                      <td className="strong">{p.display_name || p.provider_id}</td>
                      <td>{p.bookings.toLocaleString('en-IN')}</td>
                      <td>{pct(p.success_rate)}</td>
                      <td>{pct(p.cancellation_rate)}</td>
                      <td>{p.cost_per_km?.toFixed(2)}</td>
                      <td className="strong">{p.reliability_adjusted_cost_per_km?.toFixed(2)}</td>
                      <td>{money(p.spend, symbol)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="ent-two">
            <Breakdown title="By campus" rows={data.by_campus} keyName="campus" symbol={symbol} />
            <Breakdown title="By team" rows={data.by_employee_group} keyName="employee_group" symbol={symbol} />
          </div>

          <div className="card card-pad">
            <div className="eyebrow">Demand and reliability by hour</div>
            <Hourly rows={data.hourly} />
          </div>

          {audit?.entries?.length > 0 && (
            <div className="card card-pad auditcard">
              <div className="eyebrow">Governance — recorded AI decisions</div>
              <p className="ent-sub">
                Every recommendation this instance produced, with the model version and the
                confidence behind it. {audit.durable ? 'Appended durably.' : 'In-memory ring buffer.'}
              </p>
              <div className="tblwrap">
                <table className="enttable">
                  <thead><tr>
                    <th>When</th><th>Kind</th><th>Actor</th><th>Decision</th>
                    <th>Confidence</th><th>Models</th>
                  </tr></thead>
                  <tbody>
                    {audit.entries.slice(0, 12).map((e, i) => (
                      <tr key={i}>
                        <td className="mono">{e.at.slice(11, 19)}</td>
                        <td>{e.kind}</td>
                        <td>{e.actor}</td>
                        <td>{e.decision?.recommended || e.decision?.headline
                          || `${e.decision?.bookings_in_scope ?? '—'} bookings`}</td>
                        <td>{e.confidence != null ? pct(e.confidence) : '—'}</td>
                        <td className="mono">{Object.values(e.model_versions || {}).join(' · ') || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {!busy && o && o.bookings === 0 && (
        <div className="card card-pad">
          <b>No bookings match these filters.</b>
          <p style={{ color: 'var(--ink-3)', fontSize: 13.5 }}>
            Widen the selection, or run <code>python scripts/generate_mobility_data.py</code> if the
            history has not been generated yet.
          </p>
        </div>
      )}
    </div>
  )
}

/** A typed filter. Suggestions are offered, not imposed: you can type a value
 *  or pick one, and clearing the box means "all". */
function Sel({ label, value, onChange, options }) {
  const id = `flt-${label.toLowerCase().replace(/\s+/g, '-')}`
  const byLabel = new Map(options.map(([v, l]) => [String(l).toLowerCase(), v]))
  const byValue = new Map(options.map(([v]) => [String(v).toLowerCase(), v]))
  const shown = options.find(([v]) => v === value)?.[1] ?? value

  const commit = (text) => {
    const t = text.trim()
    if (!t) return onChange('')
    const hit = byLabel.get(t.toLowerCase()) ?? byValue.get(t.toLowerCase())
    onChange(hit ?? t)
  }

  return (
    <div className="field narrow">
      <label htmlFor={id}>{label}</label>
      <input id={id} list={`${id}-opts`} defaultValue={shown} autoComplete="off"
             placeholder="All"
             onBlur={e => commit(e.target.value)}
             onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); commit(e.target.value) } }} />
      <datalist id={`${id}-opts`}>
        {options.map(([v, l]) => <option key={v} value={l} />)}
      </datalist>
    </div>
  )
}

function Kpi({ label, value, sub, tone }) {
  return (
    <div className={`kpi${tone ? ` kpi-${tone}` : ''}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      <div className="kpi-sub">{sub}</div>
    </div>
  )
}

function Breakdown({ title, rows, keyName, symbol }) {
  const shown = rows.filter(r => !r.suppressed)
  const suppressed = rows.length - shown.length
  return (
    <div className="card card-pad">
      <div className="eyebrow">{title}</div>
      <div className="tblwrap">
        <table className="enttable">
          <thead><tr><th>{title.replace('By ', '')}</th><th>Trips</th><th>Success</th><th>Spend</th><th>Lost time</th></tr></thead>
          <tbody>
            {shown.map(r => (
              <tr key={r[keyName]}>
                <td className="strong">{r[keyName]}</td>
                <td>{r.bookings.toLocaleString('en-IN')}</td>
                <td>{pct(r.success_rate)}</td>
                <td>{money(r.spend, symbol)}</td>
                <td>{money(r.wasted_cost, symbol)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {suppressed > 0 && (
        <p style={{ fontSize: 11.5, color: 'var(--ink-3)', marginTop: 8 }}>
          {suppressed} group{suppressed > 1 ? 's' : ''} suppressed — too few trips to report without
          risking re-identification.
        </p>
      )}
    </div>
  )
}

function Hourly({ rows }) {
  const max = Math.max(...rows.map(r => r.bookings), 1)
  return (
    <div className="hourly">
      {rows.map(r => {
        const h = Math.round((r.bookings / max) * 100)
        const cx = r.cancellation_rate ?? 0
        return (
          <div className="hbar" key={r.hour}
               title={`${String(r.hour).padStart(2, '0')}:00 — ${r.bookings} bookings, ${pct(r.cancellation_rate)} cancelled`}>
            <div className="hbar-track">
              <div className="hbar-fill" style={{ height: `${h}%`, opacity: 0.35 + cx * 2.2 }} />
            </div>
            <div className="hbar-lab">{r.hour % 6 === 0 ? String(r.hour).padStart(2, '0') : ''}</div>
          </div>
        )
      })}
      <div className="hourly-key">Bar height = trips · shading = cancellation rate</div>
    </div>
  )
}
