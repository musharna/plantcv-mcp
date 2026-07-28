import numpy as np
import pytest

from plantcv_mcp.diagnostics import (
    DegenerateMaskError,
    analyze_mask,
    assert_not_degenerate,
    component_areas,
)


def _mask_with_squares(shape, squares):
    """squares = [(row, col, size), ...] -> uint8 mask with disjoint filled squares."""
    m = np.zeros(shape, dtype=np.uint8)
    for r, c, s in squares:
        m[r : r + s, c : c + s] = 255
    return m


def test_component_areas_descending_excludes_background():
    mask = _mask_with_squares((100, 100), [(0, 0, 10), (50, 50, 5)])
    assert component_areas(mask) == [100, 25]


def test_analyze_mask_reports_fraction_and_counts():
    mask = _mask_with_squares((100, 100), [(0, 0, 10)])
    diag = analyze_mask(mask)
    assert diag.component_count == 1
    assert diag.largest_area == 100
    assert diag.mask_fraction == pytest.approx(100 / 10000)


def test_degenerate_empty_mask_raises_and_valid_mask_does_not():
    # NEGATIVE CASE plus its POSITIVE CONTROL, in the same test, so an
    # always-raises bug cannot masquerade as working detection.
    empty = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(DegenerateMaskError):
        assert_not_degenerate(analyze_mask(empty))

    valid = _mask_with_squares((100, 100), [(0, 0, 30)])
    assert_not_degenerate(analyze_mask(valid))  # must NOT raise


def test_degenerate_below_min_fraction_raises_and_just_above_does_not():
    # 0.1% of 100x100 = 10 px. A 3x3 square (9 px) is below; 4x4 (16 px) is above.
    below = _mask_with_squares((100, 100), [(0, 0, 3)])
    with pytest.raises(DegenerateMaskError):
        assert_not_degenerate(analyze_mask(below))

    above = _mask_with_squares((100, 100), [(0, 0, 4)])
    assert_not_degenerate(analyze_mask(above))  # positive control
