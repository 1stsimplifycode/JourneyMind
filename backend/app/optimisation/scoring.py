"""Stage 3: rank what survived, using this user's priorities.

    score(J) = w_cost      · normalise(cost)
             + w_time      · normalise(time)
             + w_transfers · normalise(transfers)
             + w_comfort   · normalise(discomfort)

Lower is better. Two rules matter more than the formula:

  * **Weights sum to 1.** Otherwise "cheapest" and "fastest" are not
    comparable presets, they are differently-scaled ones.
  * **Every objective is min-max normalised across the current candidate
    set.** Rupees and minutes are never added together directly. Normalising
    within the candidate set means the question is always "how does this
    journey compare with the other options you actually have", which is the
    only comparison that means anything.

Presets are the MVP personalisation (v1). Learning weights from observed
choices with a discrete-choice model is v2 and is deliberately not here.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..routing.journey import Journey

OBJECTIVES = ("cost", "time", "transfers", "comfort")

#: When two journeys are this close in BOTH money and time, a rider would call
#: them the same answer -- and between two same answers the one with fewer
#: changes wins. Stated in rupees and minutes rather than as a score band: a
#: band on the normalised score sounds equivalent and is not, because its width
#: in real terms depends on the spread of whatever else was found. Tried that
#: way first, and under "fastest" it promoted a journey four minutes slower.
SIMPLICITY_COST_BAND = 10.0
SIMPLICITY_TIME_BAND = 5.0


@dataclass(frozen=True)
class Weights:
    cost: float
    time: float
    transfers: float
    comfort: float

    def normalised(self) -> "Weights":
        total = self.cost + self.time + self.transfers + self.comfort
        if total <= 0:
            return Weights(0.25, 0.25, 0.25, 0.25)
        return Weights(self.cost / total, self.time / total,
                       self.transfers / total, self.comfort / total)

    def as_dict(self) -> dict:
        return {"cost": round(self.cost, 4), "time": round(self.time, 4),
                "transfers": round(self.transfers, 4), "comfort": round(self.comfort, 4)}


PRESETS: dict[str, Weights] = {
    "cheapest": Weights(cost=0.78, time=0.10, transfers=0.06, comfort=0.06),
    "balanced": Weights(cost=0.38, time=0.38, transfers=0.14, comfort=0.10),
    "fastest": Weights(cost=0.10, time=0.74, transfers=0.10, comfort=0.06),
}
DEFAULT_PRESET = "balanced"


def weights_for(preset: str | None, manual: dict | None = None) -> tuple[Weights, str]:
    """Manual sliders win over the preset when supplied."""
    if manual:
        w = Weights(
            cost=max(0.0, float(manual.get("cost", 0.25))),
            time=max(0.0, float(manual.get("time", 0.25))),
            transfers=max(0.0, float(manual.get("transfers", 0.25))),
            comfort=max(0.0, float(manual.get("comfort", 0.25))),
        ).normalised()
        return w, "custom"
    key = (preset or DEFAULT_PRESET).lower()
    if key not in PRESETS:
        key = DEFAULT_PRESET
    return PRESETS[key].normalised(), key


def _minmax(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    return (lo, hi) if hi - lo > 1e-9 else (lo, lo + 1.0)


def score_all(journeys: list[Journey], weights: Weights) -> list[Journey]:
    """Attach `score` and `score_parts` to each journey. Lower score wins."""
    if not journeys:
        return []
    w = weights.normalised()
    raw = {
        "cost": [j.cost for j in journeys],
        "time": [j.total_min for j in journeys],
        "transfers": [float(j.transfers) for j in journeys],
        "comfort": [j.discomfort for j in journeys],
    }
    bounds = {k: _minmax(v) for k, v in raw.items()}
    wmap = {"cost": w.cost, "time": w.time, "transfers": w.transfers, "comfort": w.comfort}

    for i, j in enumerate(journeys):
        parts, total = {}, 0.0
        for obj in OBJECTIVES:
            lo, hi = bounds[obj]
            norm = (raw[obj][i] - lo) / (hi - lo)
            contribution = wmap[obj] * norm
            parts[obj] = {"raw": round(raw[obj][i], 3), "normalised": round(norm, 4),
                          "weight": round(wmap[obj], 4),
                          "contribution": round(contribution, 4)}
            total += contribution
        j.score = round(total, 6)
        j.score_parts = parts

    ranked = sorted(journeys, key=lambda j: (j.score, j.total_min, j.cost))
    ranked = _prefer_the_simpler_winner(ranked)
    return _reject_a_dominated_winner(ranked)


def _reject_a_dominated_winner(ranked: list[Journey]) -> list[Journey]:
    """Nothing wins while something else is cheaper AND quicker.

    The other half of the simplicity rule, and the half that was missing.
    `_prefer_the_simpler_winner` promotes a simpler journey when the difference
    is small enough that a rider would not feel it. But the score's own
    transfer term promotes simplicity with no bound at all, and the two
    together let a direct ride win at ₹144 and 44 minutes over a ₹129, 43-minute
    option -- beaten on both axes, ahead on transfers alone.

    Fewer changes is worth something. It is not worth arbitrary amounts of
    money and time, and the amount it IS worth is already written down as
    SIMPLICITY_COST_BAND / SIMPLICITY_TIME_BAND. Past those, a rider who asked
    for a balance between cost and time gets one.
    """
    if len(ranked) < 2:
        return ranked
    head = ranked[0]
    beats = [j for j in ranked[1:]
             if j.cost < head.cost - SIMPLICITY_COST_BAND
             or j.total_min < head.total_min - SIMPLICITY_TIME_BAND]
    # dominated on BOTH, and by more than the band on at least one
    dominating = [j for j in beats
                  if j.cost < head.cost - 0.5 and j.total_min < head.total_min - 0.5]
    if not dominating:
        return ranked
    winner = min(dominating, key=lambda j: (j.score, j.cost))
    return [winner] + [j for j in ranked if j is not winner]


def _prefer_the_simpler_winner(ranked: list[Journey]) -> list[Journey]:
    """Between two answers a rider cannot tell apart, take the simpler one.

    Only the head is reconsidered, and only against journeys within a few
    rupees and a few minutes of it. A three-transfer itinerary winning by the
    fourth decimal place is not a difference anybody can feel; it just reads as
    the planner showing off. Anything outside those bands is a real trade-off
    and the preset's own weights decide it.
    """
    if len(ranked) < 2:
        return ranked
    head = ranked[0]
    rivals = [j for j in ranked[1:]
              if j.transfers < head.transfers
              and abs(j.cost - head.cost) <= SIMPLICITY_COST_BAND
              and abs(j.total_min - head.total_min) <= SIMPLICITY_TIME_BAND]
    if not rivals:
        return ranked
    simplest = min(rivals, key=lambda j: (j.transfers, j.score))
    return [simplest] + [j for j in ranked if j is not simplest]


def pick_alternatives(ranked: list[Journey], n: int = 2) -> list[Journey]:
    """Alternatives must be genuinely different from the winner and from each
    other -- otherwise the user is shown the same trip three times.

    Preference order: a different mode mix first, then the best remaining
    scores. Falls back to score order only if nothing differs.
    """
    if len(ranked) <= 1:
        return []
    best = ranked[0]
    rest = ranked[1:]

    def mode_set(j: Journey) -> frozenset[str]:
        return frozenset(m for m in j.modes if m != "walk")

    chosen: list[Journey] = []
    used_modes = {mode_set(best)}
    for j in rest:
        if len(chosen) >= n:
            break
        ms = mode_set(j)
        if ms not in used_modes:
            chosen.append(j)
            used_modes.add(ms)
    for j in rest:                       # top up from score order if needed
        if len(chosen) >= n:
            break
        if j not in chosen:
            chosen.append(j)
    return chosen[:n]
