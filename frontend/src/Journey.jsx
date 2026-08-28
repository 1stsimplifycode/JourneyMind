import { modeInfo, minutes, provWord } from './modes.js'

/** The route timeline: places as dots, hops as coloured bars between them. */
export function Timeline({ journey, originLabel, destLabel }) {
  const legs = journey.legs || []
  if (!legs.length) return null
  return (
    <div className="timeline">
      <div className="stop">
        <div className="dotcol"><span className="dot start" /></div>
        <div className="name">{originLabel || legs[0].from_name}</div>
      </div>
      {legs.map((leg, i) => {
        const info = modeInfo(leg.mode)
        const last = i === legs.length - 1
        return (
          <div key={leg.index}>
            <div className="hop">
              <div className="barcol">
                <span className="bar" style={{
                  background: info.dash
                    ? `repeating-linear-gradient(180deg, ${info.colour} 0 5px, transparent 5px 10px)`
                    : info.colour,
                }} />
              </div>
              <div className="body">
                <div className="line1">
                  <span className="modechip" style={{ background: info.colour }}>{info.label}</span>
                  <span className="dur">{minutes(leg.total_min)}</span>
                  {leg.fare && leg.fare.amount > 0 && (
                    <span className="fare">{leg.fare.display}</span>
                  )}
                </div>
                <div className="meta">
                  {leg.route_name ? `${leg.route_name}` : null}
                  {leg.stops > 1 ? `${leg.route_name ? ' · ' : ''}${leg.stops} stops` : null}
                  {(leg.route_name || leg.stops > 1) ? ' · ' : ''}
                  {leg.distance_km.toFixed(1)} km
                  {leg.wait_min >= 0.5 ? ` · ${Math.round(leg.wait_min)} min wait` : ''}
                  {leg.fare && leg.fare.amount > 0 ? ` · fare ${provWord(leg.fare.provenance)}` : ''}
                  {` · time ${provWord(leg.time_provenance)}`}
                </div>
              </div>
            </div>
            <div className="stop">
              <div className="dotcol"><span className={`dot${last ? ' end' : ''}`} /></div>
              <div className="name">{last ? (destLabel || leg.to_name) : leg.to_name}</div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export function Metrics({ journey, symbol = '₹' }) {
  const c = journey.total_cost
  return (
    <div className="metrics">
      <div className="metric">
        <div className="v">{c.display}</div>
        <div className="k">Total</div>
        <div className="prov">{provWord(c.provenance)}</div>
      </div>
      <div className="metric">
        <div className="v">{minutes(journey.total_min)}</div>
        <div className="k">Door to door</div>
        <div className="prov">predicted</div>
      </div>
      <div className="metric">
        <div className="v">{journey.transfers}</div>
        <div className="k">{journey.transfers === 1 ? 'Transfer' : 'Transfers'}</div>
        <div className="prov">
          {journey.modes.filter(m => m !== 'walk').length || 1}{' '}
          {journey.modes.filter(m => m !== 'walk').length === 1 ? 'mode' : 'modes'}
        </div>
      </div>
      {journey.walk_min >= 1 && (
        <div className="metric">
          <div className="v">{Math.round(journey.walk_min)}</div>
          <div className="k">Min walking</div>
          <div className="prov">{journey.distance_km.toFixed(1)} km total</div>
        </div>
      )}
    </div>
  )
}

export function Checks({ constraints, budget, maxTime, symbol = '₹' }) {
  const c = constraints
  return (
    <div className="checks">
      <div className={`check ${c.within_budget ? 'ok' : 'no'}`}>
        <span className="ic">{c.within_budget ? '✓' : '✕'}</span>
        <span>
          {c.within_budget
            ? `Within your ${symbol}${Math.round(budget)} budget`
            : `${symbol}${Math.abs(Math.round(c.budget_headroom))} over your ${symbol}${Math.round(budget)} budget`}
        </span>
      </div>
      <div className={`check ${c.within_time ? 'ok' : 'no'}`}>
        <span className="ic">{c.within_time ? '✓' : '✕'}</span>
        <span>
          {c.within_time
            ? `Within your ${Math.round(maxTime)} minute limit`
            : `${Math.abs(Math.round(c.time_headroom))} min over your ${Math.round(maxTime)} minute limit`}
        </span>
      </div>
      {c.cost_at_risk && (
        <div className="check no">
          <span className="ic">!</span>
          <span>At the top of the estimated fare range this would go over budget.</span>
        </div>
      )}
    </div>
  )
}

/** One alternative, or one labelled near-miss. */
export function AlternativeCard({ alt, index, budget, maxTime, symbol = '₹' }) {
  const j = alt.journey
  const near = alt.kind === 'near_miss'
  return (
    <div className="card card-pad">
      <div className="eyebrow eyebrow-muted">Alternative {index + 1}</div>
      <div className="alt-head">
        <div>
          <div className="alt-title">
            {j.modes.filter(m => m !== 'walk').map(m => modeInfo(m).label).join(' → ') || 'Walk'}
          </div>
          <div style={{ marginTop: 6 }}>
            <span className={`badge ${near ? 'badge-miss' : 'badge-ok'}`}>
              {near ? 'Outside your limits' : 'Fits your limits'}
            </span>
          </div>
        </div>
        <div className="alt-figs">
          <div className="f"><b>{j.total_cost.display}</b><i>{provWord(j.total_cost.provenance)}</i></div>
          <div className="f"><b>{minutes(j.total_min)}</b><i>predicted</i></div>
          <div className="f"><b>{j.transfers}</b><i>{j.transfers === 1 ? 'transfer' : 'transfers'}</i></div>
        </div>
      </div>
      <div className="alt-reason">{alt.reason}</div>
      <Checks constraints={j.constraints} budget={budget} maxTime={maxTime} symbol={symbol} />
      <details className="disclose">
        <summary>Show the route</summary>
        <Timeline journey={j} />
      </details>
    </div>
  )
}
