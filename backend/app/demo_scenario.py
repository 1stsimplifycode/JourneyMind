"""The one demonstration scenario, defined once.

Origin, destination, budget, time limit, preference AND the commitment the
rider is travelling to. Every screen reads this: the booking page, the journey
planner, the demo endpoint and the escalation that decides whether somebody is
late. Before this existed the booking page hard-coded a different pair of
places from the planner, and the manager notification invented a meeting "an
hour after departure" that appeared nowhere else -- three screens describing
three different trips.
"""

from __future__ import annotations

DEMO_SCENARIO = {
    "origin": "pl_wipro_sarjapur",
    "destination": "pl_pes_university",
    "budget": 250.0,
    "max_time": 120.0,
    "preference": "balanced",
    # What the rider is travelling TO. The escalation needs a real commitment
    # to be late for, and inventing one per request meant the manager
    # notification described a meeting the rest of the demo had never heard of.
    "meeting_title": "the 10:00 project review",
    "meeting_hour": 10.0,
    "title": "Wipro, Sarjapur Road → PES University, Banashankari",
    "description": (
        "Doddakannelli, Sarjapur Road (560035) to the PES University RR campus "
        "on 100 Feet Ring Road (560085) — 16.6 km straight-line, right across "
        "the city. ₹250 in your pocket, two hours on the clock, leaving now. "
        "The answer changes with the clock: at 09:00 the roads are jammed and "
        "the metro wins the middle of the trip; after the last train it falls "
        "back to a bike-taxi and says so. Computed live, at this minute, by the "
        "same pipeline the form uses."
    ),
}
