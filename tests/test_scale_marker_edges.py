"""calibrate_scale at its edges: which object is the marker, where its warnings
switch on, and what its refusals say.

Written against mutmut survivors in calibrate_scale (2026-11-02). Every scene
is a light field with dark rectangles, read on `v` with Otsu, `dark`, so the
mask is exactly the rectangles drawn.
"""

import math

import numpy as np
import pytest

from plantcv_mcp.scale import MarkerNotFoundError, calibrate_scale


def _scene(h, w, rects):
    """Light field with dark rectangles, each (x, y, width, height) in pixels."""
    img = np.full((h, w, 3), 240, np.uint8)
    for x, y, rw, rh in rects:
        img[y : y + rh, x : x + rw] = 30
    return img


def _codes(est):
    return [w.code for w in est.warnings]


# --- which object is the marker ------------------------------------------------


def test_the_largest_object_is_the_marker_not_the_first_one_scanned():
    """A speck above and left of the marker is labelled first; the marker is
    the larger object."""
    img = _scene(120, 120, [(5, 5, 3, 3), (40, 40, 40, 40)])
    est = calibrate_scale(img, 0, 0, 100, 100, 20.0)
    assert est.marker_length_px == 40
    assert est.marker_area_px == 1600


def test_a_one_pixel_marker_is_measured_not_refused():
    est = calibrate_scale(_scene(40, 40, [(20, 20, 1, 1)]), 10, 10, 20, 20, 0.5)
    assert est.marker_length_px == 1
    assert est.px_per_mm == pytest.approx(2.0)


# --- marker_touches_crop_edge: each edge on its own -------------------------------

# The crop starts at (50, 50), so crop width and right edge differ, as do crop
# height and bottom edge. A 40 px marker, placed in crop coordinates.
CROP = (50, 50, 100, 100)


def _touch_codes(local_x, local_y):
    img = _scene(200, 200, [(50 + local_x, 50 + local_y, 40, 40)])
    return _codes(calibrate_scale(img, *CROP, 20.0))


@pytest.mark.parametrize(
    ("edge", "local_x", "local_y"),
    [("left", 0, 30), ("top", 30, 0), ("right", 60, 30), ("bottom", 30, 60)],
)
def test_touching_one_edge_of_the_crop_is_enough_to_warn(edge, local_x, local_y):
    assert "marker_touches_crop_edge" in _touch_codes(local_x, local_y), edge


@pytest.mark.parametrize(
    ("where", "local_x", "local_y"),
    [("1 px from left and top", 1, 1), ("1 px from right and bottom", 59, 59)],
)
def test_a_marker_one_pixel_clear_of_the_edges_does_not_warn(where, local_x, local_y):
    assert "marker_touches_crop_edge" not in _touch_codes(local_x, local_y), where


# --- marker_fills_crop: above 90%, not at it --------------------------------------


def test_a_marker_covering_exactly_90_percent_of_its_crop_does_not_warn_fills():
    img = _scene(20, 20, [(5, 5, 10, 9)])  # 90 of the crop's 100 px
    est = calibrate_scale(img, 5, 5, 10, 10, 5.0)
    assert est.crop_fraction == pytest.approx(0.9)
    assert "marker_fills_crop" not in _codes(est)


def test_the_fills_warning_says_how_much_of_the_crop_it_covers():
    img = _scene(20, 20, [(5, 5, 10, 10)])
    img[13:15, 13:15] = 240  # a light corner, so the crop has two levels: 96 px dark
    est = calibrate_scale(img, 5, 5, 10, 10, 5.0)
    (fills,) = [w for w in est.warnings if w.code == "marker_fills_crop"]
    assert "96% of the crop" in fills.message


# --- marker_not_round: above 1.5:1, not at it ---------------------------------------


def _not_round(rw, rh):
    img = _scene(60, 60, [(20, 25, rw, rh)])
    est = calibrate_scale(img, 10, 10, 40, 40, 5.0)
    return [w for w in est.warnings if w.code == "marker_not_round"]


def test_a_marker_at_exactly_one_and_a_half_to_one_is_not_flagged():
    assert _not_round(15, 10) == []


def test_a_marker_just_past_one_and_a_half_to_one_is_flagged_with_its_sides():
    (w,) = _not_round(16, 10)
    assert "16x10 px" in w.message


def test_a_one_pixel_wide_line_is_flagged_not_round():
    (w,) = _not_round(20, 1)
    assert "20x1 px" in w.message


# --- what the refusals say ------------------------------------------------------


def test_an_infinite_marker_length_is_refused_as_such():
    with pytest.raises(ValueError, match="marker_length_mm must be a positive finite"):
        calibrate_scale(_scene(60, 60, [(20, 20, 10, 10)]), 10, 10, 30, 30, math.inf)


@pytest.mark.parametrize(("w", "h"), [(0, 10), (10, 0)])
def test_a_crop_with_no_width_or_no_height_is_refused_as_such(w, h):
    with pytest.raises(ValueError, match=f"positive size, got w={w} h={h}"):
        calibrate_scale(_scene(60, 60, [(20, 20, 10, 10)]), 10, 10, w, h, 5.0)


@pytest.mark.parametrize(("w", "h"), [(1, 20), (20, 1)])
def test_a_one_pixel_thin_crop_is_still_a_crop(w, h):
    """One pixel across is a positive size: the crop is measured, half of it
    on the marker."""
    img = _scene(60, 60, [(20, 20, 10, 10)])
    est = calibrate_scale(img, 25 - (w > 1) * 15, 25 - (h > 1) * 15, w, h, 5.0)
    assert est.marker_length_px == 10


def test_no_object_names_the_settings_that_found_nothing():
    blank = np.full((50, 50, 3), 240, np.uint8)
    with pytest.raises(MarkerNotFoundError, match="channel='v' method='otsu'"):
        calibrate_scale(blank, 0, 0, 50, 50, 5.0)
