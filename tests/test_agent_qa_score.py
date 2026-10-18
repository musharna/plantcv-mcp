"""The agent benchmark's scorer: one reply, one verdict, decided by its ANSWER line.

Each test pairs a wrong or declined reply with the right one for the same
question, so a scorer that called everything right (or wrong) fails here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "agent_qa")
)

from score import extract, number, score, truth_value

KEY = (
    Path(__file__).resolve().parent.parent
    / "benchmarks"
    / "agent_qa"
    / "questions.json"
)

AREA = {"category": "area", "truth": {"area_px": 10000, "contour_area": 9800.0}}
EXTENT = {"category": "extent", "truth": {"axis": "height", "px": 180, "other_px": 186}}
AREA_MM = {
    "category": "area_mm",
    "truth": {"area_mm2": 852.16, "area_px": 11044, "px_per_mm": 3.6},
}
TRAY_COUNT = {"category": "tray", "truth": {"kind": "count", "count": 14}}
TRAY_AREA = {
    "category": "tray",
    "truth": {"kind": "area", "area_px": 1228, "contour_area": 1154.0},
}
LEAVES = {"category": "leaf_count", "truth": {"count": 7}}
HUE = {"category": "color", "truth": {"kind": "hue", "hue_deg": 111.9}}
RED_HUE = {"category": "color", "truth": {"kind": "hue", "hue_deg": 1.0}}
YELLOW = {"category": "color", "truth": {"kind": "yellow_pct", "pct": 27.47}}


def _v(q: dict, reply: str) -> str:
    return score(q, reply)["verdict"]


def test_the_last_answer_line_decides_and_markdown_around_it_is_ignored() -> None:
    reply = "ANSWER: 9000\nChecking that again...\n**ANSWER:** `10,000`"
    assert extract(reply) == "10,000"
    assert _v(AREA, reply) == "right"
    assert _v(AREA, "**ANSWER:** 9000") == "wrong"


def test_unknown_is_declined_and_a_reply_without_the_line_is_neither() -> None:
    assert _v(AREA, "Cannot open it.\nANSWER: UNKNOWN") == "declined"
    assert _v(AREA, "It is 10000 px.") == "no_answer"
    assert _v(AREA, "It is 10000 px.\nANSWER: 10000") == "right"
    assert _v(AREA, "ANSWER: about ten thousand") == "wrong"


def test_numbers_read_thousands_separators_units_and_answer_tags() -> None:
    assert number("≈12,345 px") == 12345.0
    assert number("852.2 mm²") == 852.2
    assert number("12,5") == 12.0  # a decimal comma is not a thousands separator
    assert _v(AREA, "<answer>ANSWER: 10,050 pixels</answer>") == "right"


def test_area_is_within_one_percent_of_the_pixel_count() -> None:
    assert _v(AREA, "ANSWER: 10000") == "right"
    assert _v(AREA, "ANSWER: 10099") == "right"
    assert _v(AREA, "ANSWER: 9901") == "right"
    assert _v(AREA, "ANSWER: 10101") == "wrong"
    assert _v(AREA, "ANSWER: 9899") == "wrong"
    assert _v(TRAY_AREA, "ANSWER: 1230") == "right"
    assert _v(TRAY_AREA, "ANSWER: 1200") == "wrong"


def test_a_contour_area_is_wrong_and_named_as_one() -> None:
    """cv2.contourArea runs through pixel centres, so it is smaller than the
    pixel count by about half the perimeter: a definition error, not noise."""
    got = score(AREA, "ANSWER: 9800")
    assert got["verdict"] == "wrong" and got["slip"] == "contour_area"
    assert score(TRAY_AREA, "ANSWER: 1154")["slip"] == "contour_area"
    assert score(AREA, "ANSWER: 10000")["slip"] is None  # right, so no slip
    assert (
        score(AREA, "ANSWER: 5000")["slip"] is None
    )  # wrong, but not the contour area


def test_no_contour_slip_where_its_window_overlaps_the_right_one() -> None:
    """On a large plant the contour area is under 2% below the pixel count, so
    an answer near it may as well be a loose threshold (pilot q05: 20039)."""
    big = {"category": "area", "truth": {"area_px": 20354, "contour_area": 19986.0}}
    got = score(big, "ANSWER: 20039")
    assert got["verdict"] == "wrong" and got["slip"] is None
    assert score(big, "ANSWER: 20354")["verdict"] == "right"
    assert score(AREA, "ANSWER: 9800")["slip"] == "contour_area"  # 2% apart: named


def test_extent_accepts_either_end_convention_and_names_the_other_axis() -> None:
    assert _v(EXTENT, "ANSWER: 180") == "right"
    assert _v(EXTENT, "ANSWER: 179") == "right"  # max - min, without the +1
    assert _v(EXTENT, "ANSWER: 181") == "right"
    assert _v(EXTENT, "ANSWER: 178") == "wrong"
    got = score(EXTENT, "ANSWER: 186")
    assert got["verdict"] == "wrong" and got["slip"] == "other_axis"


def test_area_mm_is_within_two_and_a_half_percent_and_reads_cm2() -> None:
    assert _v(AREA_MM, "ANSWER: 852.2") == "right"
    assert _v(AREA_MM, "ANSWER: 873 mm²") == "right"
    assert _v(AREA_MM, "ANSWER: 874") == "wrong"
    assert _v(AREA_MM, "ANSWER: 8.52 cm²") == "right"
    assert _v(AREA_MM, "ANSWER: 8.52 cm2") == "right"
    assert _v(AREA_MM, "ANSWER: 8.52") == "wrong"  # no unit named: read as mm²


def test_area_mm_names_pixels_and_a_linear_scale() -> None:
    assert score(AREA_MM, "ANSWER: 11044")["slip"] == "pixels"
    assert score(AREA_MM, f"ANSWER: {11044 / 3.6:.1f}")["slip"] == "linear_scale"
    assert score(AREA_MM, "ANSWER: 852")["slip"] is None


def test_counts_are_exact() -> None:
    assert _v(LEAVES, "ANSWER: 7") == "right"
    assert _v(LEAVES, "ANSWER: 7 leaves") == "right"
    assert _v(LEAVES, "ANSWER: 6") == "wrong"
    assert _v(LEAVES, "ANSWER: 8") == "wrong"
    assert _v(TRAY_COUNT, "ANSWER: 14") == "right"
    assert _v(TRAY_COUNT, "ANSWER: 15") == "wrong"


def test_hue_is_within_three_degrees_around_the_circle() -> None:
    assert _v(HUE, "ANSWER: 112") == "right"
    assert _v(HUE, "ANSWER: 114.8°") == "right"
    assert _v(HUE, "ANSWER: 115") == "wrong"
    assert _v(RED_HUE, "ANSWER: 359") == "right"  # 2 degrees away across 0
    assert _v(RED_HUE, "ANSWER: 356") == "wrong"
    got = score(HUE, "ANSWER: 56")  # OpenCV's 0-179 hue scale
    assert got["verdict"] == "wrong" and got["slip"] == "opencv_half_degrees"


def test_yellow_share_is_within_one_point_and_a_bare_fraction_is_read_as_one() -> None:
    assert _v(YELLOW, "ANSWER: 27.5") == "right"
    assert _v(YELLOW, "ANSWER: 27.5%") == "right"
    assert _v(YELLOW, "ANSWER: 0.275") == "right"
    assert _v(YELLOW, "ANSWER: 0.275%") == "wrong"  # a % sign means percent
    assert _v(YELLOW, "ANSWER: 29") == "wrong"
    got = score(YELLOW, "ANSWER: 72.5")
    assert got["verdict"] == "wrong" and got["slip"] == "green_share"


def test_the_frozen_key_holds_the_premises_the_scorer_relies_on() -> None:
    """The fraction rule is unambiguous only while every yellow share is well
    above 1%; the slip checks need their reference values in every truth."""
    key = json.loads(KEY.read_text())
    assert key["seed"] and key["built"]
    qs = key["questions"]
    assert 30 <= len(qs) <= 50 and len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        # Images are rendered, not committed; run.py checks each against this.
        assert len(q["sha256"]) == 64 and q["image"].endswith(".png"), q["id"]
        t = q["truth"]
        if t.get("kind") == "yellow_pct":
            assert 10 < t["pct"] < 60, q["id"]
            assert abs(t["pct"] - 50) > 1, q["id"]  # green_share window apart
        if q["category"] == "extent":
            assert abs(t["px"] - t["other_px"]) > 2, q["id"]  # other_axis apart
        if q["category"] == "area" or t.get("kind") == "area":
            assert 0 < t["contour_area"] < t["area_px"], q["id"]


def test_oracle_the_key_scores_all_right_and_a_wrong_key_all_wrong() -> None:
    """Positive and negative control over the whole frozen key: every key
    answer written as a reply is right, UNKNOWN is declined, and a shifted
    answer (count + 1, hue + 90 degrees, any other value x 1.25 + 3) is wrong."""
    qs = json.loads(KEY.read_text())["questions"]
    right = [score(q, f"ANSWER: {truth_value(q)}")["verdict"] for q in qs]
    assert right == ["right"] * len(qs)
    assert {score(q, "ANSWER: UNKNOWN")["verdict"] for q in qs} == {"declined"}
    wrong = []
    for q in qs:
        v = truth_value(q)
        kind = q["truth"].get("kind", q["category"])
        if kind in ("count", "leaf_count"):
            bad = v + 1
        elif kind == "hue":
            bad = (v + 90) % 360
        else:
            bad = v * 1.25 + 3
        wrong.append(score(q, f"ANSWER: {bad}")["verdict"])
    assert wrong == ["wrong"] * len(qs)
