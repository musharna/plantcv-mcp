"""Mask refinement.

The property under test is not that ops run — PlantCV runs them — but that a
refinement can never produce a session that then measures nonsense: an op list
that empties the mask is refused up front, every op is validated before any is
applied (PlantCV itself silently no-ops on fill(size=-1), erode(i=0) and even an
even median kernel), and the trait table that comes out of a refined session says
how its mask was made.
"""

import cv2
import numpy as np
import pytest

from plantcv_mcp.diagnostics import analyze_mask
from plantcv_mcp.measurement import measure_traits
from plantcv_mcp.refine import (
    REFINE_OPS,
    RefinementErasedMaskError,
    RefineSpecError,
    apply_refinements,
    apply_refinements_traced,
    refinement_warnings,
    validate_ops,
)

R = 40  # disc radius; area = pi * R**2 = 5026.5 px


def _clean_disc(size: int = 200) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), R, 255, -1)
    return mask


def _noisy_disc(size: int = 200) -> np.ndarray:
    """A disc with an interior hole, salt specks outside, and pepper inside."""
    mask = _clean_disc(size)
    cv2.circle(mask, (size // 2, size // 2), 6, 0, -1)  # hole
    rng = np.random.default_rng(7)
    for _ in range(40):  # salt: isolated 1-px specks on the background
        y, x = rng.integers(0, size, 2)
        if (x - size // 2) ** 2 + (y - size // 2) ** 2 > (R + 10) ** 2:
            mask[y, x] = 255
    return mask


# --- each op does what it says ---


def test_fill_holes_closes_an_interior_hole():
    mask = _clean_disc()
    cv2.circle(mask, (100, 100), 6, 0, -1)
    out = apply_refinements(mask, [{"op": "fill_holes"}])
    assert np.array_equal(out, _clean_disc())


def test_keep_largest_drops_everything_but_the_n_biggest_components():
    mask = _clean_disc()
    mask[10, 10] = 255
    mask[20:24, 20:24] = 255
    out = apply_refinements(mask, [{"op": "keep_largest", "n": 1}])
    assert np.array_equal(out, _clean_disc())
    two = apply_refinements(mask, [{"op": "keep_largest", "n": 2}])
    assert analyze_mask(two).component_count == 2


def test_erode_shrinks_and_dilate_grows():
    mask = _clean_disc()
    before = int((mask > 0).sum())
    eroded = apply_refinements(mask, [{"op": "erode", "ksize": 3, "iterations": 2}])
    dilated = apply_refinements(mask, [{"op": "dilate", "ksize": 3}])
    assert int((eroded > 0).sum()) < before < int((dilated > 0).sum())


def test_fill_removes_components_smaller_than_size():
    mask = _clean_disc()
    mask[20:23, 20:23] = 255  # 9 px
    out = apply_refinements(mask, [{"op": "fill", "size": 10}])
    assert np.array_equal(out, _clean_disc())


def test_ops_apply_in_the_order_given():
    """erode-then-dilate (opening) and dilate-then-erode (closing) differ on a
    thin bridge; if order were ignored they would agree."""
    mask = np.zeros((100, 100), np.uint8)
    mask[40:60, 10:45] = 255
    mask[40:60, 55:90] = 255
    mask[49:51, 45:55] = 255  # a 2-px bridge
    a = apply_refinements(
        mask, [{"op": "erode", "ksize": 3}, {"op": "dilate", "ksize": 3}]
    )
    b = apply_refinements(
        mask, [{"op": "dilate", "ksize": 3}, {"op": "erode", "ksize": 3}]
    )
    assert analyze_mask(a).component_count == 2  # bridge gone
    assert analyze_mask(b).component_count == 1  # bridge kept


# --- the eval: refinement recovers a known geometry ---


def test_fill_holes_plus_keep_largest_recovers_the_known_disc_area():
    truth = np.pi * R**2
    img = np.full((200, 200, 3), 200, np.uint8)
    noisy = _noisy_disc()

    # Positive control: the noisy mask is measurably WRONG first — more components
    # and a hole — or the refinement below would be proving nothing.
    assert analyze_mask(noisy).component_count > 1
    noisy_area = measure_traits(img, noisy)["area"]["value"]
    assert abs(noisy_area - truth) / truth > 0.01

    refined = apply_refinements(
        noisy, [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}]
    )
    area = measure_traits(img, refined)["area"]["value"]
    assert analyze_mask(refined).component_count == 1
    assert abs(area - truth) / truth < 0.01, (area, truth)


# --- validation is all-or-nothing and stricter than PlantCV's ---


@pytest.mark.parametrize(
    "ops, needle",
    [
        ([], "empty"),
        ([{"op": "sharpen"}], "sharpen"),
        ([{"op": "fill"}], "size"),  # missing param
        ([{"op": "fill", "size": -1}], "size"),  # PlantCV silently accepts this
        ([{"op": "fill", "size": 0}], "size"),  # PlantCV no-op; lineage would lie
        ([{"op": "erode", "ksize": 3, "iterations": 0}], "iterations"),
        ([{"op": "erode", "ksize": 1}], "ksize"),
        ([{"op": "median_blur", "ksize": 4}], "odd"),
        ([{"op": "keep_largest", "n": 0}], "n"),
        ([{"op": "fill_holes", "size": 3}], "unknown parameter"),
        ([{"op": "dilate", "ksize": "3"}], "ksize"),  # wrong type
        ([{"op": "fill_holes"}, {"op": "erode", "ksize": 0}], "op 1"),  # index named
    ],
)
def test_invalid_op_lists_are_refused_before_anything_runs(ops, needle):
    mask = _clean_disc()
    with pytest.raises(RefineSpecError, match=needle):
        validate_ops(ops)
    with pytest.raises(RefineSpecError):
        apply_refinements(mask, ops)
    # Positive control: a valid list on the same mask is accepted.
    assert validate_ops([{"op": "fill_holes"}]) == [{"op": "fill_holes"}]


def test_validate_ops_fills_in_defaults_so_lineage_records_what_ran():
    assert validate_ops([{"op": "erode", "ksize": 3}]) == [
        {"op": "erode", "ksize": 3, "iterations": 1}
    ]


def test_every_documented_op_is_runnable():
    """REFINE_OPS is what list_methods publishes; each entry must actually apply."""
    mask = _clean_disc()
    for name, spec in REFINE_OPS.items():
        op = {"op": name, **{p: c["example"] for p, c in spec["params"].items()}}
        out = apply_refinements(mask, [op])
        assert out.dtype == np.uint8 and out.shape == mask.shape, name


# --- an op list that deletes the plant is refused, not measured ---


def test_erosion_that_empties_the_mask_is_refused_with_before_and_after():
    mask = _clean_disc()
    with pytest.raises(RefinementErasedMaskError) as exc:
        apply_refinements(mask, [{"op": "erode", "ksize": 5, "iterations": 40}])
    msg = str(exc.value)
    assert "before" in msg and "after" in msg
    # Positive control: a mild erosion on the same mask is fine.
    assert (apply_refinements(mask, [{"op": "erode", "ksize": 3}]) > 0).any()


# --- a refinement that discards a major object says so (real-photo finding #5) ---


def _plant_with_bridged_leaf(size: int = 300) -> np.ndarray:
    """A big disc (the plant) joined by a 3-px bridge to a smaller disc (a leaf).

    opening(7) cuts the bridge; keep_largest(1) then throws the leaf away. The
    leaf is ~25% of the plant's area — a change small enough (<25% of the total
    mask) that refine_large_change stays silent, which is exactly the gap: a
    17% change on a real sorghum photo was a whole leaf and only the overlay
    showed it.
    """
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (100, 150), 50, 255, -1)  # plant, area ~7854
    cv2.circle(mask, (230, 150), 25, 255, -1)  # leaf, area ~1963
    mask[149:152, 150:210] = 255  # 3-px bridge
    return mask


def test_dropping_a_major_object_is_named_with_the_op_that_split_it_off():

    mask = _plant_with_bridged_leaf()
    ops = [{"op": "opening", "ksize": 7}, {"op": "keep_largest", "n": 1}]
    out, dropped = apply_refinements_traced(mask, ops)
    assert analyze_mask(out).component_count == 1
    assert len(dropped) == 1
    d = dropped[0]
    assert d.op_index == 1 and d.op_name == "keep_largest"
    assert d.split_by_op_index == 0
    assert 1700 < d.area < 2100  # the leaf, not the bridge
    warnings = refinement_warnings(mask, analyze_mask(mask), analyze_mask(out), dropped)
    codes = [w.code for w in warnings]
    assert "refine_dropped_object" in codes
    assert "refine_large_change" not in codes  # below 25% -- the old alarm is silent
    msg = next(w.message for w in warnings if w.code == "refine_dropped_object")
    assert "op 1 (keep_largest)" in msg
    assert "op 0 (opening)" in msg
    assert "overlay" in msg

    # Positive control: keep_largest on a plant plus a genuine speck drops
    # nothing major and stays silent; the mask is otherwise identical.
    speck = _clean_disc()
    speck[10:13, 10:13] = 255
    out2, dropped2 = apply_refinements_traced(speck, [{"op": "keep_largest", "n": 1}])
    assert dropped2 == []
    assert analyze_mask(out2).component_count == 1
    assert (
        refinement_warnings(speck, analyze_mask(speck), analyze_mask(out2), dropped2)
        == []
    )


def test_apply_refinements_traced_returns_the_same_mask_as_apply_refinements():
    mask = _plant_with_bridged_leaf()
    ops = [{"op": "opening", "ksize": 7}, {"op": "keep_largest", "n": 1}]
    out, _ = apply_refinements_traced(mask, ops)
    np.testing.assert_array_equal(out, apply_refinements(mask, ops))


def test_dropped_object_warning_lists_the_largest_three_and_counts_the_rest():
    from plantcv_mcp.refine import DroppedObject, dropped_object_warning

    drops = [
        DroppedObject(1, "keep_largest", area, 10_000, 0, "opening")
        for area in (1200, 5000, 1500, 3000, 2000)
    ]
    msg = dropped_object_warning(drops).message
    assert msg.index("5000-px") < msg.index("3000-px") < msg.index("2000-px")
    assert "1500-px" not in msg and "1200-px" not in msg
    assert "2 more" in msg
    assert "2 more" not in dropped_object_warning(drops[:3]).message


# --- panel audit of 1.5.1 (2026-08-29) ---


def test_refine_large_change_fires_above_the_threshold():
    """Only the silent side was asserted; a deleted guard passed the suite."""
    mask = _clean_disc()
    out, dropped = apply_refinements_traced(
        mask, [{"op": "dilate", "ksize": 15, "iterations": 3}]
    )
    codes = [
        w.code
        for w in refinement_warnings(
            out, analyze_mask(mask), analyze_mask(out), dropped
        )
    ]
    assert "refine_large_change" in codes


def test_dropped_object_attribution_is_honest_about_which_split_it_names():
    """Two openings split two leaves; keep_largest drops both. A single
    'split_by' slot names the LAST op that raised the component count for
    both, which is wrong for the first leaf. The message must say what it
    actually knows."""
    from plantcv_mcp.refine import dropped_object_warning

    mask = np.zeros((300, 400), np.uint8)
    cv2.circle(mask, (100, 150), 50, 255, -1)  # plant
    cv2.circle(mask, (230, 150), 25, 255, -1)  # leaf A, thin bridge
    mask[149:152, 150:210] = 255
    cv2.circle(mask, (100, 40), 22, 255, -1)  # leaf B, thicker bridge
    mask[60:110, 96:105] = 255
    ops = [
        {"op": "opening", "ksize": 5},  # cuts the 3-px bridge (leaf A)
        {"op": "opening", "ksize": 11},  # cuts the 9-px bridge (leaf B)
        {"op": "keep_largest", "n": 1},
    ]
    _out, dropped = apply_refinements_traced(mask, ops)
    assert len(dropped) == 2
    msg = dropped_object_warning(dropped).message
    assert "last op that raised the component count" in msg
    assert "op 1 (opening)" in msg
