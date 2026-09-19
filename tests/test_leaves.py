"""Leaf instances by watershed: known-answer geometry, then two real rosettes.

The synthetic scenes settle what the code must get exactly right (N separate
shapes are N instances; two touching shapes are two). The real rosettes settle
nothing in the tool's favour: they pin what it measures today against a
hand-annotated leaf count, error included, so a change in either direction is
seen. Figures and how the fixtures were chosen: docs/EVAL.md, "Leaf instances".
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from plantcv import plantcv as pcv

from plantcv_mcp.diagnostics import DegenerateMaskError
from plantcv_mcp.imaging import render_overlay
from plantcv_mcp.leaves import (
    LeafCountRefusedError,
    _instance_color,
    count_leaves,
)

FIXTURES = Path(__file__).parent / "fixtures" / "aberystwyth"
H, W = 600, 900
# Mildly elongated on purpose. A long ellipse has a nearly flat distance-transform
# ridge, and pixel rounding puts two maxima on it further apart than
# min_distance: that over-segmentation is real, and is pinned separately in
# test_a_long_ellipse_is_cut_in_two_at_the_default_distance rather than hidden
# by choosing shapes that avoid it.
AXES = (34, 26)


def _scene(centres, axes=AXES, angles=None, shape=(H, W)):
    mask = np.zeros(shape, np.uint8)
    angles = angles or [0] * len(centres)
    for centre, angle in zip(centres, angles, strict=True):
        cv2.ellipse(mask, centre, axes, angle, 0, 360, 255, -1)
    img = np.full((*shape, 3), 200, np.uint8)
    img[mask > 0] = (40, 150, 40)
    return img, mask


def _separated(n, seed):
    """n ellipses at seeded random spots and angles, no two closer than 2.5 major
    axes centre to centre, so they cannot touch whatever their angles."""
    rng = np.random.default_rng(seed)
    centres: list[tuple[int, int]] = []
    while len(centres) < n:
        c = (int(rng.integers(70, W - 70)), int(rng.integers(70, H - 70)))
        if all(math.dist(c, o) > 2.5 * AXES[0] for o in centres):
            centres.append(c)
    angles = [int(a) for a in rng.integers(0, 180, n)]
    return centres, angles


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_n_separated_ellipses_are_exactly_n_instances(n):
    centres, angles = _separated(n, seed=20260918 + n)
    img, mask = _scene(centres, angles=angles)
    # The scene is what it claims to be: n components, before any counting.
    assert cv2.connectedComponents(mask)[0] - 1 == n

    res = count_leaves(img, mask)

    assert res.leaf_count == n
    assert [i["id"] for i in res.instances] == list(range(1, n + 1))
    # Each instance IS one ellipse: its centroid sits on a drawn centre and no
    # two instances claim the same one.
    claimed = set()
    for inst in res.instances:
        nearest = min(centres, key=lambda c: math.dist(c, inst["centroid"]))
        assert math.dist(nearest, inst["centroid"]) < 1.5
        claimed.add(nearest)
        # cv2.ellipse rasterises ~3% fat of pi*a*b; the bound is loose enough
        # for that and far too tight for a merged or halved ellipse.
        assert inst["area_px"] == pytest.approx(math.pi * AXES[0] * AXES[1], rel=0.05)
    assert claimed == set(centres)
    # The instances partition the mask: nothing dropped, nothing double-counted.
    assert sum(i["area_px"] for i in res.instances) == int((mask > 0).sum())
    assert int((res.labels > 0).sum()) == int((mask > 0).sum())


def test_two_touching_ellipses_are_split_into_two_and_one_ellipse_is_not():
    centres = [(400, 300), (400 + 2 * AXES[1] - 6, 300)]  # side by side, overlapping
    img, mask = _scene(centres, angles=[90, 90])
    # One connected blob: counting components would say 1.
    assert cv2.connectedComponents(mask)[0] - 1 == 1

    res = count_leaves(img, mask)

    assert res.leaf_count == 2
    xs = sorted(i["centroid"][0] for i in res.instances)
    assert xs[0] == pytest.approx(centres[0][0], abs=4)
    assert xs[1] == pytest.approx(centres[1][0], abs=4)
    a, b = (i["area_px"] for i in res.instances)
    assert a == pytest.approx(b, rel=0.05)  # split down the middle, not 90/10
    assert a + b == int((mask > 0).sum())

    # Control: the split comes from the shape, not from a habit of splitting.
    alone = count_leaves(*_scene([centres[0]], angles=[90]))
    assert alone.leaf_count == 1


def test_a_long_ellipse_is_cut_in_two_at_the_default_distance():
    """The documented weakness, as a number: ONE 70x24 ellipse is two instances
    at min_distance 10 and one at 30. If PlantCV's peak finding ever changes,
    this is where it shows."""
    img, mask = _scene([(450, 300)], axes=(70, 24))
    res = count_leaves(img, mask)
    assert res.leaf_count == 2
    assert res.sensitivity["20"] == 1
    assert count_leaves(img, mask, min_distance=30).leaf_count == 1


def test_a_shape_at_the_frame_edge_keeps_its_peak():
    """peak_local_max drops peaks within min_distance of the array border. A
    disc of radius 20 whose centre is 22 px from the corner has its only peak
    inside that band at distance 30."""
    img, mask = _scene([(22, 22)], axes=(20, 20))
    assert count_leaves(img, mask, min_distance=30).leaf_count == 1
    # And away from every border, the same disc and distance: the same answer.
    assert count_leaves(*_scene([(450, 300)], axes=(20, 20)), 30).leaf_count == 1


def test_empty_mask_is_a_named_refusal_not_a_zero_count():
    img, mask = _scene([(300, 300)])
    with pytest.raises(DegenerateMaskError, match="mask is empty"):
        count_leaves(img, np.zeros_like(mask))
    # Same image, same call, a mask with a plant in it: counted.
    assert count_leaves(img, mask).leaf_count == 1


def test_inverted_mask_is_refused_by_name():
    img, mask = _scene([(300, 300)])
    with pytest.raises(LeafCountRefusedError, match="implausible_coverage"):
        count_leaves(img, cv2.bitwise_not(mask))
    assert count_leaves(img, mask).leaf_count == 1


def test_min_distance_and_scale_are_validated():
    img, mask = _scene([(300, 300)])
    with pytest.raises(ValueError, match="min_distance must be >= 1"):
        count_leaves(img, mask, min_distance=0)
    with pytest.raises(TypeError, match="min_distance must be an integer"):
        count_leaves(img, mask, min_distance=True)
    with pytest.raises(TypeError, match="min_distance must be an integer"):
        count_leaves(img, mask, min_distance=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="px_per_mm must be a positive finite"):
        count_leaves(img, mask, px_per_mm=float("nan"))
    assert count_leaves(img, mask, min_distance=10, px_per_mm=2.0).leaf_count == 1


def test_min_distance_beyond_the_mask_is_refused_before_anything_is_allocated():
    """The ring of zeros is 2 x min_distance wide on every side, so an unbounded
    min_distance is an unbounded allocation from one tool argument (200000 on
    this 96 x 52 px ellipse asks for ~480 GB). A distance longer than the mask
    is also meaningless: nothing in it can be that far apart."""
    img, mask = _scene([(450, 300)], axes=(48, 26))
    with pytest.raises(ValueError, match=r"min_distance=200000 exceeds .* 97 px"):
        count_leaves(img, mask, min_distance=200000)
    with pytest.raises(ValueError, match=r"min_distance=98 exceeds"):
        count_leaves(img, mask, min_distance=98)
    # The largest legal value is the mask's own extent. Its "twice" pass is
    # capped there too, and says so by its key.
    res = count_leaves(img, mask, min_distance=97)
    assert res.leaf_count == 1
    assert res.sensitivity == {"48": 1, "97": 1}


def test_px_per_mm_scales_area_only():
    centres, angles = _separated(3, seed=7)
    img, mask = _scene(centres, angles=angles)
    px = count_leaves(img, mask)
    mm = count_leaves(img, mask, px_per_mm=4.0)
    assert px.units["area"] == "pixels" and mm.units["area"] == "mm2"
    for a, b in zip(px.instances, mm.instances, strict=True):
        assert a["area"] == a["area_px"]
        assert b["area"] == pytest.approx(b["area_px"] / 16.0)
        assert b["area_px"] == a["area_px"]
        assert b["centroid"] == a["centroid"] and b["bbox"] == a["bbox"]


def test_centroid_and_bbox_are_in_full_frame_coordinates():
    """The work runs on a crop; a forgotten offset would put every instance in
    the top-left corner. The ellipse is far from the origin on both axes."""
    centre = (700, 450)
    res = count_leaves(*_scene([centre]))
    (inst,) = res.instances
    assert inst["centroid"] == pytest.approx(list(centre), abs=1.0)
    x, y, w, h = inst["bbox"]
    assert (x, y) == (centre[0] - AXES[0], centre[1] - AXES[1])
    assert (w, h) == (2 * AXES[0] + 1, 2 * AXES[1] + 1)
    assert res.labels[centre[1], centre[0]] == 1 and res.labels[10, 10] == 0


def test_sensitivity_is_reported_and_warned_only_when_the_count_moves():
    # Two ellipses overlapping end to end: one elongated blob whose count
    # depends on min_distance (measured: 4 at 5 px, 2 at 10 and 20).
    moving = count_leaves(*_scene([(400, 300), (460, 300)], axes=(45, 25)))
    assert moving.sensitivity == {"5": 4, "20": 2}
    assert moving.leaf_count == 2
    assert [w.code for w in moving.warnings] == ["min_distance_sensitive"]

    # Two overlapping DISCS have one peak each at every distance tried.
    steady = count_leaves(*_scene([(400, 300), (470, 300)], axes=(40, 40)))
    assert steady.leaf_count == 2
    assert steady.sensitivity == {"5": 2, "20": 2}
    assert steady.warnings == []


def test_holes_in_the_mask_are_named_because_they_inflate_the_count():
    img, mask = _scene([(450, 300)], axes=(120, 120))
    solid = count_leaves(img, mask)
    assert solid.leaf_count == 1
    assert "mask_has_holes" not in [w.code for w in solid.warnings]

    holed = mask.copy()
    for cx in (400, 450, 500):
        cv2.circle(holed, (cx, 300), 4, 0, -1)
    res = count_leaves(img, holed)
    (warning,) = [w for w in res.warnings if w.code == "mask_has_holes"]
    assert "3 enclosed hole(s)" in warning.message
    assert res.leaf_count > 1  # the reason the warning exists


def test_several_objects_are_counted_together_and_flagged():
    centres, angles = _separated(3, seed=11)
    res = count_leaves(*_scene(centres, angles=angles))
    assert res.leaf_count == 3
    assert [w.code for w in res.warnings] == ["multi_object_mask"]
    one = count_leaves(*_scene(centres[:1], angles=angles[:1]))
    assert one.warnings == []


def test_overlay_is_deterministic_numbered_and_drawn_in_instance_colours():
    centres = [(300, 300), (600, 300)]
    img, mask = _scene(centres)
    first = count_leaves(img, mask).overlay
    second = count_leaves(img, mask).overlay
    assert np.array_equal(first, second)  # PlantCV's own watershed palette is random
    assert first.shape == img.shape
    plain = render_overlay(img, mask)
    assert not np.array_equal(first, plain)
    for idx, (cx, cy) in enumerate(centres, start=1):
        colour = np.array(_instance_color(idx), np.uint8)
        # The outline passes through the ellipse's leftmost point.
        edge = first[cy - 3 : cy + 4, cx - AXES[0] - 2 : cx - AXES[0] + 3]
        assert (edge == colour).all(axis=2).any()
        # A digit is drawn at the centroid: white on black, neither in `plain`.
        patch = first[cy - 12 : cy + 12, cx - 12 : cx + 12]
        assert (patch == 255).all(axis=2).any() and (patch == 0).all(axis=2).any()


def test_host_pcv_outputs_survive_a_count():
    pcv.outputs.clear()
    pcv.outputs.add_observation(
        sample="host",
        variable="kept",
        trait="host value",
        method="test",
        scale="none",
        datatype=int,
        value=7,
        label="none",
    )
    try:
        assert count_leaves(*_scene([(300, 300)])).leaf_count == 1
        assert pcv.outputs.observations["host"]["kept"]["value"] == 7
        assert list(pcv.outputs.observations) == ["host"]
    finally:
        pcv.outputs.clear()


# --- real rosettes ---------------------------------------------------------

REAL = [
    # name, annotated leaves, count through segment()+fill_holes+keep_largest at
    # min_distance 10 (the default) and at 6, measured 2026-09-18.
    ("tray032_2015-12-21_plant0", 11, 7, 8),
    ("tray032_2016-01-05_plant0", 25, 13, 19),
]


def _annotated_leaves(name: str) -> int:
    gt = cv2.imread(str(FIXTURES / f"{name}_gt.png"), cv2.IMREAD_UNCHANGED)[..., :3]
    colours = np.unique(gt.reshape(-1, 3), axis=0)
    return int(colours.any(axis=1).sum())  # every colour but black


@pytest.mark.parametrize(("name", "truth", "at_10", "at_6"), REAL)
def test_real_rosette_counts_are_pinned_with_their_error(name, truth, at_10, at_6):
    from plantcv_mcp.server import (
        _count_leaves_impl,
        _refine_impl,
        _segment_impl,
        _store,
    )

    # The truth is recounted from the annotation, not copied from a table.
    assert _annotated_leaves(name) == truth

    seg = _segment_impl(str(FIXTURES / f"{name}.png"), "a", "otsu")
    ops = [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}]
    sid = _refine_impl(seg["session_id"], ops)["session_id"]

    default = _count_leaves_impl(sid)
    assert default["min_distance"] == 10
    assert default["leaf_count"] == at_10
    assert default["lineage"] == ops
    assert _count_leaves_impl(sid, min_distance=6)["leaf_count"] == at_6
    # The honest part: on real overlapping leaves the watershed UNDERCOUNTS, and
    # the default is further off than a distance fitted to this camera height.
    assert at_10 < at_6 < truth
    # Every masked pixel belongs to exactly one instance.
    mask_px = int((_store.get(sid).mask > 0).sum())
    assert sum(i["area_px"] for i in default["instances"]) == mask_px
    assert len(default["instances"]) == at_10


def test_real_rosette_pinholes_inflate_the_count_and_are_flagged():
    """The thresholded mask of the grown rosette has pinholes; counted as is, it
    reads 30 leaves against 25 annotated — close by accident, and 13 once the
    holes are filled. The advisory is what tells the two apart."""
    from plantcv_mcp.server import _count_leaves_impl, _refine_impl, _segment_impl

    name = "tray032_2016-01-05_plant0"
    seg = _segment_impl(str(FIXTURES / f"{name}.png"), "a", "otsu")
    raw = _refine_impl(seg["session_id"], [{"op": "keep_largest", "n": 1}])
    holed = _count_leaves_impl(raw["session_id"])
    assert holed["leaf_count"] == 30
    assert "mask_has_holes" in [w["code"] for w in holed["warnings"]]

    filled = _refine_impl(
        seg["session_id"], [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}]
    )
    clean = _count_leaves_impl(filled["session_id"])
    assert clean["leaf_count"] == 13
    assert "mask_has_holes" not in [w["code"] for w in clean["warnings"]]
