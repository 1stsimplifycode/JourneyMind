"""Human names for the things this product moves people in.

One table, because there were two and they disagreed. `bike_taxi` is a key: it
belongs in a join, an audit row and a URL. It does not belong in "bike_taxi
then bus then metro then bike_taxi", which is what the manager notification
said the day the escalation learned to suggest an itinerary.

Modes only. Provider names live on the provider (`providers/simulated.py`),
because who you book through is a different fact from what you travel in.
"""

from __future__ import annotations

MODE_LABEL: dict[str, str] = {
    "bike_taxi": "Bike taxi",
    "auto": "Auto",
    "cab": "Cab",
    "metro": "Metro",
    "bus": "Bus",
    "walk": "Walk",
}


def label_for(mode: str) -> str:
    """A mode's display name. Unknown keys are tidied rather than hidden, so a
    new mode reads oddly instead of vanishing."""
    return MODE_LABEL.get(mode, mode.replace("_", " ").capitalize())


def journey_phrase(modes, joiner: str = " then ") -> str:
    """An itinerary as a sentence: "Bike taxi then Bus then Metro"."""
    return joiner.join(label_for(m) for m in modes)
