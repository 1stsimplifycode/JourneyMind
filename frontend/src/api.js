// One place that knows how to talk to the backend.
//
// The base URL is relative by default, because the FastAPI service serves this
// bundle itself. VITE_API_BASE exists only for running the Vite dev server
// against a backend on another host; nothing is ever hard-coded to localhost.
const BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')

async function request(path, options) {
  let res
  try {
    res = await fetch(BASE + path, options)
  } catch {
    throw new ApiError('Could not reach the JourneyMind server.', 'network',
      'Check that the backend is running and try again.')
  }
  let body = null
  try { body = await res.json() } catch { /* non-JSON error page */ }

  if (!res.ok) {
    const d = body?.detail ?? body ?? {}
    throw new ApiError(
      d.error || body?.error || `Request failed (${res.status})`,
      d.code || body?.code || String(res.status),
      d.detail || null,
    )
  }
  return body
}

export class ApiError extends Error {
  constructor(message, code, detail) {
    super(message)
    this.code = code
    this.detail = detail
  }
}

export const getCity = () => request('/api/city')
export const getPlaces = () => request('/api/places')
export const getModels = () => request('/api/models')
export const getDemo = () => request('/api/demo')

// --- mobility intelligence -------------------------------------------------
export const compare = (payload) => request('/api/compare', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
})

export const getProviders = () => request('/api/providers')

// --- booking: BOOK NOW actually books -------------------------------------
export const bookRide = (payload) => request('/api/book', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
})

export const retryBooking = (id) => request(`/api/book/${id}/retry`, { method: 'POST' })

export const revealBooking = (id) => request(`/api/book/${id}/reveal`)

export const getEscalation = (id, meeting) => {
  const q = new URLSearchParams(
    Object.entries(meeting || {}).filter(([, v]) => v)).toString()
  return request(`/api/book/${id}/escalation${q ? `?${q}` : ''}`)
}

export const notifyManager = (id, meeting) => request(`/api/book/${id}/notify`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(meeting || {}),
})

export const getInsights = () => request('/api/insights')

// Enterprise endpoints are gated: population-level data is never open.
const withKey = (key) => ({ headers: key ? { 'X-API-Key': key } : {} })

export const enterpriseFacets = (key) => request('/api/enterprise/facets', withKey(key))

export const enterpriseOverview = (key, filters = {}) => {
  const q = new URLSearchParams(
    Object.entries(filters).filter(([, v]) => v !== '' && v != null)).toString()
  return request(`/api/enterprise/overview${q ? `?${q}` : ''}`, withKey(key))
}

export const enterpriseAudit = (key, limit = 25) =>
  request(`/api/enterprise/audit?limit=${limit}`, withKey(key))

export const recommend = (payload) => request('/api/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
})
