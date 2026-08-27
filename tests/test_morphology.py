"""Skeleton-based morphology traits.

Everything here is measured against a synthetic plant whose geometry is KNOWN:
a vertical stem with leaves drawn at chosen angles and lengths. PlantCV 4.11.3
has systematic behaviours that the tolerances below absorb deliberately (all
measured 2026-08-27, see docs/superpowers/specs/2026-08-27-morphology-design.md):

* leaf path lengths come out ~10 px short (junction trimming), ordering kept;
* insertion angles come out 5-9 degrees high, ordering kept;
* the stem tip above the last junction is classed as a "leaf";
* a perfectly vertical stem yields stem_angle = -14373 degrees (slope blow-up).

The last one is the reason this module exists as more than a pass-through: a
number that is not an angle must not be returned as one.
"""

import math

import cv2
import numpy as np
import pytest

from plantcv_mcp.diagnostics import DegenerateMaskError
from plantcv_mcp.morphology import (
    MorphologyRefusedError,
    measure_morphology,
)

SIZE = 400
STEM_TOP, STEM_BASE = 120, 380  # stem height 260 px


def _plant(leaves=((30, 90), (45, 80), (60, 70)), stem_tilt_deg=0.0):
    """Vertical (or tilted) stem with leaves at (angle_from_stem, length)."""
    img = np.full((SIZE, SIZE, 3), 200, np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    dx = math.tan(math.radians(stem_tilt_deg)) * (STEM_BASE - STEM_TOP)
    cv2.line(mask, (200, STEM_BASE), (int(200 + dx), STEM_TOP), 255, 9)
    bases = [320, 250, 180][: len(leaves)]
    for i, ((ang, length), by) in enumerate(zip(leaves, bases, strict=True)):
        bx = int(200 + dx * (STEM_BASE - by) / (STEM_BASE - STEM_TOP))
        sgn = 1 if i % 2 == 0 else -1
        ex = int(bx + sgn * length * math.sin(math.radians(ang)))
        ey = int(by - length * math.cos(math.radians(ang)))
        cv2.line(mask, (bx, by), (ex, ey), 255, 7)
    img[mask > 0] = (40, 150, 40)
    return img, mask


def _leafiest(segments, n):
    """The n segments with the largest insertion angles = the drawn leaves; the
    stem tip PlantCV also calls a leaf has insertion angle ~0."""
    return sorted(segments, key=lambda s: s["insertion_angle"], reverse=True)[:n]


def test_recovers_known_leaf_angles_and_lengths_in_order():
    truth = [(30, 90), (45, 80), (60, 70)]
    img, mask = _plant(truth)
    res = measure_morphology(img, mask)

    assert res.plant["num_cycles"] == 0
    assert res.plant["leaf_count"] >= 3
    # PlantCV's stem is the skeleton BELOW the topmost junction (the stem tip
    # above it is sorted as a leaf), so stem_height runs base -> last junction:
    # 380 - 180 = 200 px, not the 260 px drawn. Measured: 202.
    assert abs(res.plant["stem_height"] - (STEM_BASE - 180)) / 200 < 0.10

    leaves = _leafiest(res.segments, 3)
    # Ordering: the leaf with the largest insertion angle is the shortest one.
    angles = [s["insertion_angle"] for s in leaves]
    lengths = [s["path_length"] for s in leaves]
    assert angles == sorted(angles, reverse=True)
    assert lengths == sorted(lengths), (angles, lengths)
    for (true_ang, true_len), seg in zip(
        sorted(truth, reverse=True), leaves, strict=True
    ):
        assert abs(seg["insertion_angle"] - true_ang) <= 12, (seg, true_ang)
        assert abs(seg["path_length"] - true_len) / true_len <= 0.15, (seg, true_len)

    # Positive control: a different plant gives a different answer.
    two = measure_morphology(*_plant(((30, 90), (60, 70))))
    assert len(two.segments) < len(res.segments)

    # The overlay is the numbered-segment picture, same size as the input.
    assert res.overlay.shape == img.shape
    assert not np.array_equal(res.overlay, img)


def test_vertical_stem_angle_is_null_with_a_named_warning():
    res = measure_morphology(*_plant(stem_tilt_deg=0.0))
    assert res.plant["stem_angle"] is None
    assert "stem_angle_undefined" in [w.code for w in res.warnings]

    # Positive control: a tilted stem has a finite angle and no such warning.
    tilted = measure_morphology(*_plant(stem_tilt_deg=20.0))
    assert tilted.plant["stem_angle"] is not None
    assert -180 <= tilted.plant["stem_angle"] <= 180
    assert "stem_angle_undefined" not in [w.code for w in tilted.warnings]


def test_ring_shaped_mask_reports_skeleton_cycles():
    img = np.full((SIZE, SIZE, 3), 200, np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    cv2.circle(mask, (200, 200), 80, 255, 12)  # a ring: one cycle
    img[mask > 0] = (40, 150, 40)
    res = measure_morphology(img, mask)
    assert res.plant["num_cycles"] >= 1
    assert "skeleton_has_cycles" in [w.code for w in res.warnings]

    # Positive control: the plant has no cycles and no such warning.
    plain = measure_morphology(*_plant())
    assert "skeleton_has_cycles" not in [w.code for w in plain.warnings]


def test_multi_specimen_and_empty_masks_are_refused_by_name():
    img, mask = _plant()
    two_plants = mask.copy()
    two_plants[:, :] = np.roll(mask, 150, axis=1) | np.roll(mask, -150, axis=1)
    with pytest.raises(MorphologyRefusedError, match="measure_regions"):
        measure_morphology(img, two_plants)
    with pytest.raises(DegenerateMaskError):
        measure_morphology(img, np.zeros_like(mask))
    # Positive control: the single plant measures.
    assert measure_morphology(img, mask).segments


def _hairy(mask, spur_len):
    """The plant with short spurs drawn off it every 40th mask pixel."""
    rng = np.random.default_rng(3)
    hairy = mask.copy()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys[::40], xs[::40], strict=True):
        ang = rng.uniform(0, 2 * np.pi)
        cv2.line(
            hairy,
            (int(x), int(y)),
            (int(x + spur_len * np.cos(ang)), int(y + spur_len * np.sin(ang))),
            255,
            2,
        )
    return hairy


def test_prune_size_sensitivity_is_reported_when_segment_count_is_unstable():
    """Measured segment counts: the clean plant is 7 at every prune size (PlantCV
    never prunes a tip-terminated leaf); 12-px spurs give 70 at prune 15 and 46
    at 30, a 34% change. Both directions are asserted so the advisory can
    discriminate."""
    img, mask = _plant()
    stable = measure_morphology(img, mask, prune_size=15)
    assert "prune_size_sensitive" not in [w.code for w in stable.warnings]

    res = measure_morphology(img, _hairy(mask, 12), prune_size=15)
    assert "prune_size_sensitive" in [w.code for w in res.warnings]


def test_a_skeleton_plantcv_cannot_analyse_is_refused_with_the_prune_counts():
    """On a badly fragmented skeleton PlantCV's segment_tangent_angle calls
    fatal_error ("Too many tips found per segment, try pruning again"). That
    must arrive as a refusal that says what to do, carrying the counts."""
    img, mask = _plant()
    with pytest.raises(MorphologyRefusedError, match="prune_size=8 leaves"):
        measure_morphology(img, _hairy(mask, 6), prune_size=8)
    # Positive control: pruned harder, the same mask analyses.
    assert measure_morphology(img, _hairy(mask, 6), prune_size=20).plant


def test_px_per_mm_scales_lengths_but_never_angles():
    img, mask = _plant()
    px = measure_morphology(img, mask)
    mm = measure_morphology(img, mask, px_per_mm=10.0)
    s_px, s_mm = _leafiest(px.segments, 1)[0], _leafiest(mm.segments, 1)[0]
    assert s_mm["path_length"] == pytest.approx(s_px["path_length"] / 10.0)
    assert s_mm["insertion_angle"] == s_px["insertion_angle"]
    assert mm.plant["stem_height"] == pytest.approx(px.plant["stem_height"] / 10.0)
    assert mm.units["path_length"] == "mm" and px.units["path_length"] == "pixels"
    assert mm.units["insertion_angle"] == "degrees"


def test_tangent_window_longer_than_a_leaf_is_flagged_not_reported_as_zero():
    """Measured: insertion-angle bias shrinks as tangent_size grows (24.5 deg at
    10, 5.9 at 25, 4.2 at 30) until the window exceeds a leaf's length, when the
    angle silently collapses to 0.0 (size 40 on a 70 px leaf). A zero that is an
    artefact of the window must not read as a measurement."""
    img, mask = _plant()
    res = measure_morphology(img, mask, tangent_size=40)
    assert "tangent_window_exceeds_segment" in [w.code for w in res.warnings]
    # Positive control: the default window is shorter than every leaf.
    ok = measure_morphology(img, mask)
    assert "tangent_window_exceeds_segment" not in [w.code for w in ok.warnings]
