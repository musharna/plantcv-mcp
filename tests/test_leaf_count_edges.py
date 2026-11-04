"""The leaf counters at their edges: the boundaries of each check, the parts of
the result nothing else reads, and the masks the usual scenes never draw.

Written from the mutants that survived the nightly mutation run (#97) once
mutmut could see the tests that reach these functions through the server: each
test here fails on at least one of them. Mutants left alive on purpose are
listed in the PR that added this file.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from plantcv_mcp.leaves import count_leaves, split_by_seeds


def _scene(shape, *shapes):
    """Draw ("circle", centre, r) / ("ellipse", centre, axes) / ("rect", p0, p1)
    as plant-green on grey. The image matters only to the coverage checks."""
    mask = np.zeros(shape, np.uint8)
    for kind, a, b in shapes:
        if kind == "circle":
            cv2.circle(mask, a, b, 255, -1)
        elif kind == "ellipse":
            cv2.ellipse(mask, a, b, 0, 0, 360, 255, -1)
        else:
            cv2.rectangle(mask, a, b, 255, -1)
    img = np.full((*shape, 3), 200, np.uint8)
    img[mask > 0] = (40, 150, 40)
    return img, mask


ELLIPSE = ("ellipse", (150, 100), (34, 26))


# --- count_leaves ----------------------------------------------------------------


def test_min_distance_of_one_is_accepted_and_labels_the_mask():
    """1 px is the smallest legal distance. Its half is floored at 1, its double
    is 2, and the ring around the crop is therefore 4 px: the labels come back
    whole, not as the empty interior of a ring 0 px wide."""
    img, mask = _scene((200, 300), ELLIPSE)
    res = count_leaves(img, mask, min_distance=1)
    assert res.leaf_count == 1
    assert res.sensitivity == {"1": 1, "2": 2}
    assert res.labels.dtype == np.int32
    assert np.array_equal(res.labels > 0, mask > 0)
    with pytest.raises(ValueError, match="min_distance must be >= 1 pixel, got 0"):
        count_leaves(img, mask, min_distance=0)


def test_px_per_mm_below_one_is_a_scale_and_zero_is_refused():
    img, mask = _scene((200, 300), ELLIPSE)
    res = count_leaves(img, mask, px_per_mm=0.5)
    (inst,) = res.instances
    assert inst["area"] == pytest.approx(inst["area_px"] * 4)
    with pytest.raises(ValueError, match="px_per_mm must be a positive finite"):
        count_leaves(img, mask, px_per_mm=0.0)


def test_extent_is_the_taller_side_for_a_tall_mask():
    """The extent check, for a mask taller than wide: 97 rows is the largest
    legal min_distance, 98 is refused."""
    img, mask = _scene((200, 300), ("ellipse", (150, 100), (26, 48)))
    assert count_leaves(img, mask, min_distance=97).leaf_count == 1
    with pytest.raises(
        ValueError, match=r"min_distance=98 exceeds the mask's own extent, 97 px"
    ):
        count_leaves(img, mask, min_distance=98)


def test_a_zero_one_mask_counts_like_a_zero_255_mask():
    img, mask = _scene((200, 300), ELLIPSE)
    a = count_leaves(img, mask)
    b = count_leaves(img, (mask > 0).astype(np.uint8))
    assert b.leaf_count == a.leaf_count == 1
    assert [i["area_px"] for i in b.instances] == [i["area_px"] for i in a.instances]


def test_two_comparable_objects_are_already_flagged():
    img, mask = _scene(
        (200, 400), ("circle", (100, 100), 40), ("circle", (300, 100), 40)
    )
    res = count_leaves(img, mask)
    assert res.leaf_count == 2
    (warning,) = res.warnings
    assert warning.code == "multi_object_mask"
    assert warning.message.startswith("The mask holds 2 comparably sized objects")
    one = count_leaves(*_scene((200, 400), ("circle", (100, 100), 40)))
    assert one.warnings == []


def _discs_and_blobs(n_discs, n_blobs):
    """Discs keep one peak at every distance tried; each blob of two overlapping
    ellipses is 2 at min_distance 10 and 20, and 4 at 5 (measured)."""
    shapes = []
    for b in range(n_blobs):
        shapes += [
            ("ellipse", (150 + 300 * b, 120), (45, 25)),
            ("ellipse", (210 + 300 * b, 120), (45, 25)),
        ]
    shapes += [
        ("circle", (100 + 150 * (i % 9), 300 + 150 * (i // 9)), 40)
        for i in range(n_discs)
    ]
    return _scene((900, 1400), *shapes)


def test_sensitivity_warns_above_thirty_percent_and_not_at_it():
    """The warning compares the count's move to the count: 2 of 7 (29%) and
    6 of 20 (exactly 30%) are below "more than 30%"; 2 of 2 is above it."""
    small = count_leaves(*_discs_and_blobs(5, 1))
    assert (small.leaf_count, small.sensitivity) == (7, {"5": 9, "20": 7})
    at = count_leaves(*_discs_and_blobs(14, 3))
    assert (at.leaf_count, at.sensitivity) == (20, {"5": 26, "20": 20})
    for res in (small, at):
        assert [w.code for w in res.warnings] == ["multi_object_mask"]

    moving = count_leaves(*_discs_and_blobs(0, 1))
    (warning,) = [w for w in moving.warnings if w.code == "min_distance_sensitive"]
    assert warning.message.startswith(
        "min_distance=10 yields 2 instances, but 5 yields 4 and 20 yields 2."
    )


# --- split_by_seeds --------------------------------------------------------------


def test_two_seeds_on_one_object_make_two_regions():
    """The regions come from the caller's seeds, not from the mask's own minima
    (an ellipse has one): two seeds, two halves."""
    img, mask = _scene((200, 300), ("ellipse", (150, 100), (80, 30)))
    res = split_by_seeds(img, mask, [[90, 100], [210, 100]])
    assert res.leaf_count == 2
    areas = [i["area_px"] for i in res.instances]
    assert len(areas) == 2 and sum(areas) == int((mask > 0).sum())
    assert min(areas) > 0.45 * sum(areas)


def test_the_seeded_result_carries_its_full_shape():
    img, mask = _scene((200, 300), ("ellipse", (150, 100), (80, 30)))
    res = split_by_seeds(img, mask, [[90, 100], [210, 100]])
    assert res.min_distance == 0 and res.sensitivity == {}
    assert res.units == {
        "area": "pixels",
        "area_px": "pixels",
        "centroid": "pixels",
        "bbox": "pixels",
    }
    assert res.labels.dtype == np.int32
    assert sorted(np.unique(res.labels).tolist()) == [0, 1, 2]
    assert np.array_equal(res.labels > 0, mask > 0)
    (warning,) = res.warnings
    assert warning.code == "seeded_count"
    assert warning.message.startswith(
        "The count is the number of seeds given (2), not a"
    )


def test_seeded_px_per_mm_scales_area_and_bad_scales_are_refused():
    img, mask = _scene((200, 300), ("ellipse", (150, 100), (80, 30)))
    seeds = [[90, 100], [210, 100]]
    px = split_by_seeds(img, mask, seeds)
    mm = split_by_seeds(img, mask, seeds, px_per_mm=0.5)
    assert mm.units["area"] == "mm2" and px.units["area"] == "pixels"
    for a, b in zip(px.instances, mm.instances, strict=True):
        assert b["area_px"] == a["area_px"] == a["area"]
        assert b["area"] == pytest.approx(a["area_px"] * 4)
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(
            ValueError, match=rf"px_per_mm must be a positive finite number, got {bad}"
        ):
            split_by_seeds(img, mask, seeds, px_per_mm=bad)


def test_objects_on_the_frame_edges_keep_every_pixel():
    """A tall, narrow frame with one object in the top-left corner and one on
    the bottom edge: the crop must not clip a row or a column of either."""
    img, mask = _scene(
        (120, 60), ("rect", (0, 0), (20, 20)), ("rect", (30, 90), (50, 119))
    )
    for m in (mask, (mask > 0).astype(np.uint8)):
        res = split_by_seeds(img, m, [[5, 5], [40, 110]])
        assert [i["area_px"] for i in res.instances] == [441, 630]
        assert np.array_equal(res.labels > 0, mask > 0)


def test_a_component_without_a_seed_is_reported_with_its_share():
    img, mask = _scene(
        (60, 80), ("rect", (10, 10), (19, 19)), ("rect", (40, 40), (42, 44))
    )
    res = split_by_seeds(img, mask, [[12, 12]])
    assert [w.code for w in res.warnings] == ["seeded_count", "unseeded_mask"]
    assert res.warnings[1].message == (
        "15 mask px (13.0%) lie in components with no seed and belong to no region."
    )
    both = split_by_seeds(img, mask, [[12, 12], [40, 40]])
    assert [w.code for w in both.warnings] == ["seeded_count"]


def test_a_region_under_fifteen_percent_of_the_median_is_named():
    """Two regions, so the median is the larger (index len // 2). 15 px is
    exactly 15% of 100 and is not "under"; 14 px is."""

    def split(w, h):
        img, mask = _scene(
            (60, 80), ("rect", (10, 10), (19, 19)), ("rect", (40, 40), (39 + w, 39 + h))
        )
        return split_by_seeds(img, mask, [[12, 12], [40, 40]])

    at = split(3, 5)
    assert [i["area_px"] for i in at.instances] == [100, 15]
    assert [w.code for w in at.warnings] == ["seeded_count"]

    under = split(2, 7)
    assert [i["area_px"] for i in under.instances] == [100, 14]
    assert [w.code for w in under.warnings] == ["seeded_count", "small_seeded_region"]
    assert under.warnings[1].message.startswith(
        "Region(s) [2] are under 15% of the median region (100 px)"
    )
