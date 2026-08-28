import { useEffect, useRef } from 'react'
import L from 'leaflet'
import { modeInfo } from './modes.js'

// Leaflet's default marker icons are loaded from a CDN path that a bundler
// rewrites incorrectly. Small circular DivIcons avoid the problem entirely and
// suit the design better anyway.
const pin = (fill, ring, size = 14) => L.divIcon({
  className: '',
  html: `<div style="width:${size}px;height:${size}px;border-radius:50%;
    background:${fill};border:3px solid ${ring};box-shadow:0 1px 4px rgba(0,0,0,.35)"></div>`,
  iconSize: [size, size],
  iconAnchor: [size / 2, size / 2],
})

/**
 * Draws one journey: each leg as its own polyline in that mode's colour, with
 * walking dashed, plus a marker at every boarding and alighting point.
 */
export default function JourneyMap({ journey, origin, destination, routes }) {
  const holder = useRef(null)
  const map = useRef(null)
  const layer = useRef(null)

  useEffect(() => {
    if (map.current || !holder.current) return
    map.current = L.map(holder.current, { scrollWheelZoom: false, zoomControl: true })
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map.current)
    layer.current = L.layerGroup().addTo(map.current)
    map.current.setView([12.975, 77.605], 12)
    return () => { map.current?.remove(); map.current = null }
  }, [])

  // faint context: the transit lines of the study area
  useEffect(() => {
    if (!map.current || !routes?.length) return
    const ctx = L.layerGroup().addTo(map.current)
    routes.filter(r => r.mode === 'metro').forEach(r => {
      const pts = r.stops.map(s => [s.lat, s.lon])
      if (pts.length > 1) {
        L.polyline(pts, { color: r.colour, weight: 2.5, opacity: 0.22 }).addTo(ctx)
      }
    })
    return () => { ctx.remove() }
  }, [routes])

  useEffect(() => {
    if (!map.current || !layer.current) return
    layer.current.clearLayers()
    const g = layer.current
    const bounds = []

    if (journey?.legs?.length) {
      journey.legs.forEach((leg) => {
        const info = modeInfo(leg.mode)
        const pts = (leg.geometry || []).filter(p => p && p.length === 2 && (p[0] || p[1]))
        if (pts.length > 1) {
          // a soft casing under each line keeps colours legible over map tiles
          L.polyline(pts, { color: '#ffffff', weight: 8, opacity: 0.75 }).addTo(g)
          L.polyline(pts, {
            color: info.colour,
            weight: 4.5,
            opacity: 0.95,
            dashArray: info.dash || undefined,
            lineCap: 'round',
            lineJoin: 'round',
          }).addTo(g)
          pts.forEach(p => bounds.push(p))
        }
        if (info.vehicle && pts.length) {
          L.marker(pts[0], { icon: pin(info.colour, '#fff', 12) })
            .bindTooltip(`${info.label} from ${leg.from_name}`, { direction: 'top' })
            .addTo(g)
          L.marker(pts[pts.length - 1], { icon: pin('#fff', info.colour, 12) })
            .bindTooltip(`${info.label} to ${leg.to_name}`, { direction: 'top' })
            .addTo(g)
        }
      })
    }

    if (origin) {
      L.marker([origin.lat, origin.lon], { icon: pin('#101820', '#fff', 16) })
        .bindTooltip(origin.label || 'Start', { direction: 'top' }).addTo(g)
      bounds.push([origin.lat, origin.lon])
    }
    if (destination) {
      L.marker([destination.lat, destination.lon], { icon: pin('#b03636', '#fff', 16) })
        .bindTooltip(destination.label || 'Destination', { direction: 'top' }).addTo(g)
      bounds.push([destination.lat, destination.lon])
    }

    if (bounds.length > 1) {
      map.current.fitBounds(L.latLngBounds(bounds), { padding: [42, 42], maxZoom: 15 })
    } else if (bounds.length === 1) {
      map.current.setView(bounds[0], 14)
    }
  }, [journey, origin, destination])

  const used = [...new Set((journey?.legs || []).map(l => l.mode))]

  return (
    <div className="mapwrap">
      <div ref={holder} aria-label="Map of the recommended journey" />
      {used.length > 0 && (
        <div className="maplegend">
          {used.map(m => {
            const i = modeInfo(m)
            return (
              <div key={m}>
                <i style={{
                  background: i.dash
                    ? `repeating-linear-gradient(90deg, ${i.colour} 0 4px, transparent 4px 8px)`
                    : i.colour,
                }} />
                {i.label}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
