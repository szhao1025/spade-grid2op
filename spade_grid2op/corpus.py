"""A small grounding corpus of power-system operating conditions.

SPADE grounds the Environment Designer on documents sampled from a pretraining corpus, which
keeps generated environments diverse. Here each iteration samples one short domain document.
"""
import random

DOCUMENTS = [
    ("N-1 contingency",
     "Transmission operators must keep the grid secure after the loss of any single element (N-1). "
     "A single line trip redistributes flow onto parallel paths; if a neighbouring line was already "
     "near its thermal limit, it overloads and protection relays disconnect it after a delay."),
    ("Cascading failure",
     "Large blackouts usually start with one or two outages that overload neighbouring lines, which "
     "then trip in sequence within minutes. Outages clustered in time and in the same area are far "
     "more dangerous than the same number of outages spread across the day."),
    ("Heat wave",
     "During heat waves, air-conditioning pushes demand to seasonal peaks in the late afternoon, "
     "while high ambient temperature reduces how much current overhead lines can carry (dynamic "
     "line rating), so thermal limits effectively shrink exactly when loading is highest."),
    ("Storm damage",
     "Wind storms cause correlated faults: several lines in the same corridor can be knocked out "
     "within an hour, and crews may need hours before lines can be reconnected."),
    ("Planned maintenance",
     "Lines are taken out for planned maintenance during low-demand periods. If demand rises "
     "unexpectedly while a line is out, the remaining network has less margin to absorb it."),
    ("Evening ramp",
     "Demand ramps quickly in the early evening as people return home, creating a steep peak a few "
     "hours long; operators must pre-position the network before the ramp arrives."),
    ("Wildfire de-energization",
     "Utilities may proactively switch off lines in wildfire-risk zones for public safety, removing "
     "several lines at once for many hours and forcing power through fewer, longer paths."),
    ("Protection misoperation",
     "A faulty relay can trip a healthy line shortly after a real fault elsewhere, turning a single "
     "contingency into a double contingency that the operator did not plan for."),
]


def sample_document(rng: random.Random) -> tuple[str, str]:
    return rng.choice(DOCUMENTS)
