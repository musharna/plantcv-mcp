"""Score one benchmark answer against its question's truth.

Every run ends its reply with ``ANSWER: <value>`` (or ``ANSWER: UNKNOWN``);
the verdict is decided from that line alone, so the same reply always scores
the same way. Verdicts:

- ``right``: the value is within the category's tolerance of the truth.
- ``wrong``: a value that is not.
- ``declined``: ``ANSWER: UNKNOWN``.
- ``no_answer``: no ``ANSWER:`` line at all (a format failure, kept apart so it
  is never silently counted as either of the others).

Tolerances, per category (README: Scoring has the reasons):

- ``area`` and tray areas: within 1% of the pixel count.
- ``extent``: within 1 pixel.
- ``area_mm``: within 2.5%.
- counts (``leaf_count``, tray counts): exact.
- hue: within 3 degrees, around the circle.
- yellow share: within 1 percentage point.
- field set (``field_2026-10-24/``, hand-annotated trays): ``tray_plant_area``
  takes two numbers, pixels then mm², and is right only if both fall in the
  key's windows (``px_lo``..``px_hi``, ``mm2_lo``..``mm2_hi``: the hand mask
  eroded and dilated by 1 px, the mm² window also widened by the lens-model
  and scale uncertainty); ``tray_leaf_count`` is exact, and ``dic`` (answer
  minus truth) is kept for the |DiC| <= 1 secondary.

A wrong answer is also given a ``slip`` when it matches a specific other
quantity, checked against the number, never guessed from the wording:
``contour_area`` (within 1% of OpenCV's contour area for the drawn mask, which
is smaller than the pixel count), ``pixels`` and ``linear_scale`` (an mm²
answer equal to the pixel area, or to it divided by px/mm once instead of
twice), ``other_axis`` (the width when the height was asked, or the reverse),
``opencv_half_degrees`` (hue on OpenCV's 0-179 scale), ``green_share`` (the
green percentage instead of the yellow). ``None`` otherwise. A slip is named
only where its window and the right answer's window do not overlap: on a large
plant the contour area is under 2% below the pixel count, so an answer near it
may as well be a loose threshold, and the number alone cannot say which.
"""

from __future__ import annotations

import re
from typing import Any

# An opening <answer> tag may sit on the same line as the ANSWER label.
_ANSWER_RE = re.compile(
    r"^[\s>*_`#-]*(?:<answer>[\s*_`]*)?answer[*_`]*\s*:\s*(.*?)\s*$", re.IGNORECASE
)
_UNKNOWN = {"UNKNOWN", "UNKNOWN.", "N/A", "NONE"}
# A thousands separator is a comma followed by exactly three digits; any other
# comma ends the number.
_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_CM2 = re.compile(r"\bcm\s*(?:²|\^?2)|\bsq(?:uare)?\.?\s*cm\b", re.IGNORECASE)

TOL = {
    "area_rel": 0.01,
    "extent_px": 1,
    "area_mm_rel": 0.025,
    "hue_deg": 3.0,
    "pct_points": 1.0,
}


def extract(reply: str) -> str | None:
    """The value on the reply's last ``ANSWER:`` line, unwrapped; None if none."""
    found = None
    for line in reply.splitlines():
        m = _ANSWER_RE.match(line)
        if m:
            found = m.group(1)
    if found is None:
        return None
    # Some replies wrap the line in <answer>...</answer>; the tags are not the value.
    return re.sub(r"</?answer>", "", found, flags=re.IGNORECASE).strip(" \t*_`")


def number(value: str) -> float | None:
    """The first number in the value: '≈12,345 px' -> 12345.0."""
    m = _NUMBER.search(value)
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


def _rel(got: float, want: float, tol: float) -> bool:
    return abs(got - want) <= tol * abs(want)


def _apart(a: float, b: float, tol: float) -> bool:
    """The two relative-tolerance windows around a and b do not overlap."""
    return abs(a - b) > tol * (abs(a) + abs(b))


def _hue_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def truth_value(question: dict[str, Any]) -> float:
    """The key's own answer to a question, in the unit the question asks for."""
    t = question["truth"]
    field = {
        "area": "area_px",
        "extent": "px",
        "area_mm": "area_mm2",
        "leaf_count": "count",
        "tray_leaf_count": "count",
    }.get(question["category"])
    if field is None:  # tray and color: the field depends on the kind
        field = {
            "count": "count",
            "area": "area_px",
            "hue": "hue_deg",
            "yellow_pct": "pct",
        }[t["kind"]]
    return t[field]


# A number and the unit word right after it, if any.
_PAIR_ITEM = re.compile(
    r"(?<![\w^.])(" + _NUMBER.pattern + r")\s*(px|pixels?|mm|cm|sq)?", re.IGNORECASE
)


def area_pair(value: str) -> tuple[float | None, float | None]:
    """(pixels, mm²) from a two-number answer. A number followed by px or
    pixel(s) is the pixel count; one followed by mm, cm or sq is the area,
    multiplied by 100 when its unit is cm². Unlabelled numbers fill what is
    left in order, pixels first. A number glued to a letter or a caret (the 2
    of mm2 or mm^2) is part of a unit, not a number."""
    px = area = None
    loose: list[float] = []
    for m in _PAIR_ITEM.finditer(value):
        x = float(m.group(1).replace(",", ""))
        unit = (m.group(2) or "").lower()
        if unit.startswith("p") and px is None:
            px = x
        elif unit in ("mm", "cm", "sq") and area is None:
            area = x * 100.0 if _CM2.match(value, m.start(2)) else x
        else:
            loose.append(x)
    for x in loose:
        if px is None:
            px = x
        elif area is None:
            area = x
    return px, area


def truth_answer(question: dict[str, Any]) -> str:
    """The key's own answer written as a reply value (the oracle's input)."""
    if question["category"] == "tray_plant_area":
        t = question["truth"]
        return f"{t['area_px']} px, {t['area_mm2']} mm²"
    return str(truth_value(question))


def score(question: dict[str, Any], reply: str) -> dict[str, Any]:
    value = extract(reply)
    if value is None:
        return {"verdict": "no_answer", "value": None, "number": None, "slip": None}
    if value.strip().upper() in _UNKNOWN:
        return {"verdict": "declined", "value": value, "number": None, "slip": None}
    truth = question["truth"]
    category = question["category"]
    if category == "tray_plant_area":
        px, area = area_pair(value)
        px_ok = px is not None and truth["px_lo"] <= px <= truth["px_hi"]
        mm_ok = area is not None and truth["mm2_lo"] <= area <= truth["mm2_hi"]
        return {
            "verdict": "right" if px_ok and mm_ok else "wrong",
            "value": value, "number": px, "number_mm2": area, "slip": None,
            "px_right": px_ok, "mm2_right": mm_ok,
        }  # fmt: skip
    x = number(value)
    if category == "tray_leaf_count":
        return {
            "verdict": "right" if x is not None and x == truth["count"] else "wrong",
            "value": value, "number": x, "slip": None,
            "dic": None if x is None else x - truth["count"],
        }  # fmt: skip
    kind = truth.get("kind", category)
    slip = None
    if x is None:
        right = False
    elif category == "area" or (category == "tray" and kind == "area"):
        right = _rel(x, truth["area_px"], TOL["area_rel"])
        c = truth["contour_area"]
        if (
            not right
            and _apart(truth["area_px"], c, TOL["area_rel"])
            and _rel(x, c, TOL["area_rel"])
        ):
            slip = "contour_area"
    elif category == "extent":
        right = abs(x - truth["px"]) <= TOL["extent_px"]
        if not right and abs(x - truth["other_px"]) <= TOL["extent_px"]:
            slip = "other_axis"
    elif category == "area_mm":
        # An answer that names cm² is converted; any other unit is read as mm².
        if _CM2.search(value):
            x *= 100.0
        right = _rel(x, truth["area_mm2"], TOL["area_mm_rel"])
        if not right:
            if _rel(x, truth["area_px"], TOL["area_mm_rel"]):
                slip = "pixels"
            elif _rel(x, truth["area_px"] / truth["px_per_mm"], TOL["area_mm_rel"]):
                slip = "linear_scale"
    elif kind in ("count", "leaf_count"):
        right = x == truth["count"]
    elif kind == "hue":
        right = _hue_diff(x, truth["hue_deg"]) <= TOL["hue_deg"]
        if not right and _hue_diff(x, truth["hue_deg"] / 2) <= TOL["hue_deg"] / 2:
            slip = "opencv_half_degrees"
    elif kind == "yellow_pct":
        # Every truth here is between 10% and 60%, so a value of at most 1 with
        # no % sign can only be a fraction, and is read as one.
        if "%" not in value and 0 < x <= 1:
            x *= 100.0
        right = abs(x - truth["pct"]) <= TOL["pct_points"]
        if not right and abs(x - (100.0 - truth["pct"])) <= TOL["pct_points"]:
            slip = "green_share"
    else:
        raise ValueError(f"unknown category {category!r} / kind {kind!r}")
    return {
        "verdict": "right" if right else "wrong",
        "value": value,
        "number": x,
        "slip": slip,
    }
