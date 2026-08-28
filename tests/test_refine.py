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
