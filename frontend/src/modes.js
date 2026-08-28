// Mode presentation: one colour, one label, one line style per MODE, shared by
// the timeline and the map so they never disagree.
//
// A mode is a vehicle, never a brand. Rapido is a provider of a bike taxi and
// Namma Yatri is a provider of an auto; both live on the provider, so the
// product can gain or lose an operator without gaining or losing a mode.
//
// `walk` is here because the map still draws the odd approach path. It is not
// a commute this product offers and never appears as a leg a rider is shown.
export const MODES = {
  walk:      { label: 'Walk',      colour: '#6b7a89', dash: '2 7',  vehicle: false },
  metro:     { label: 'Metro',     colour: '#7b3fa0', dash: null,   vehicle: true },
  bus:       { label: 'Bus',       colour: '#c2571a', dash: null,   vehicle: true },
  bike_taxi: { label: 'Bike taxi', colour: '#1a7f52', dash: null,   vehicle: true },
  auto:      { label: 'Auto',      colour: '#d0a215', dash: null,   vehicle: true },
  cab:       { label: 'Cab',       colour: '#12507e', dash: null,   vehicle: true },
}

export const modeInfo = (m) => MODES[m] || { label: m, colour: '#6b7a89', dash: null, vehicle: true }

export const minutes = (m) => {
  const v = Math.round(m)
  if (v < 60) return `${v} min`
  const h = Math.floor(v / 60)
  const r = v % 60
  return r ? `${h} h ${r} min` : `${h} h`
}

// Honesty labels. These are the words the documentation asks for, and they are
// attached to numbers rather than to the page as a whole.
export const PROVENANCE = {
  exact:     { word: 'exact',     note: 'A fixed, known value.' },
  published: { word: 'published', note: 'From an operator fare table.' },
  estimated: { word: 'estimated', note: 'From a transparent fare model. Not a quote.' },
  predicted: { word: 'predicted', note: 'Predicted by the travel-time model.' },
  demo:      { word: 'demo',      note: 'Bundled demonstration data.' },
}
export const provWord = (p) => (PROVENANCE[p]?.word) || p
