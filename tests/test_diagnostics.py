import numpy as np
import pytest

from plantcv_mcp.diagnostics import (
    DegenerateMaskError,
    analyze_mask,
    assert_not_degenerate,
    component_areas,
    multi_specimen_warning,
)


def _mask_with_squares(shape, squares):
    """squares = [(row, col, size), ...] -> uint8 mask with disjoint filled squares."""
    m = np.zeros(shape, dtype=np.uint8)
    for r, c, s in squares:
        m[r : r + s, c : c + s] = 255
    return m


def _mask_from_areas(shape, areas):
    """Build a mask with disjoint square blobs of the given pixel areas.

    Raises ValueError if the blobs do not fit within the canvas height or width.
    """
    height, width = shape
    sides = [int(np.sqrt(a)) for a in areas]

    # Validate height: blobs are written to m[0:side, ...], so max(side) must fit
    max_height = max(sides) if sides else 0
    if max_height > height:
        raise ValueError(
            f"Canvas height {height} is insufficient for blobs requiring {max_height}px "
            f"(max side: {max_height})"
        )

    # Validate width: sum of sides + gaps (5 pixels between each blob)
    total_width = sum(sides) + 5 * (len(areas) - 1)
    if total_width > width:
        raise ValueError(
            f"Canvas width {width} is insufficient for blobs requiring {total_width}px "
            f"(sides: {sides}, gaps: 5px each)"
        )

    m = np.zeros(shape, dtype=np.uint8)
    col = 0
    for side in sides:
        m[0:side, col : col + side] = 255
        col += side + 5  # gap keeps components disjoint
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


def test_major_object_count_four_comparable_and_tail():
    """Four comparable objects plus small tail — must count exactly 4 major.

    Target areas 8628, 7981, 7106, 6748, 570, 454.
    Using int(sqrt(area)) per side gives actual areas:
    8464, 7921, 7056, 6724, 529, 441.
    The four large are all ≥78% of largest (8464); the tail is ≤6.6%,
    well under the 0.25 threshold.
    """
    target_areas = [8628, 7981, 7106, 6748, 570, 454]
    sides = [int(np.sqrt(a)) for a in target_areas]  # [92, 89, 84, 82, 23, 21]
    # Actual areas: [8464, 7921, 7056, 6724, 529, 441]

    # Place squares horizontally with 2-pixel gaps to ensure disjoint components
    height = max(sides)
    squares = []
    col = 0
    for side in sides:
        squares.append((0, col, side))
        col += side + 2

    mask = _mask_with_squares((height, col), squares)

    # Verify components are disjoint (sanity check)
    areas = component_areas(mask)
    assert len(areas) == 6, f"Expected 6 components, got {len(areas)}"

    # Test major_object_count
    diag = analyze_mask(mask)
    assert diag.major_object_count == 4


def test_major_object_count_one_large_plus_fragments():
    """One large object plus small fragments — must count exactly 1 major.

    Target areas 8628, 570, 454, 274.
    Using int(sqrt(area)) per side gives actual areas:
    8464, 529, 441, 256.
    The large is ≥2116 (threshold); fragments are all <256.
    Positive control: proves threshold actually discriminates.
    """
    target_areas = [8628, 570, 454, 274]
    sides = [int(np.sqrt(a)) for a in target_areas]  # [92, 23, 21, 16]
    # Actual areas: [8464, 529, 441, 256]

    # Place squares horizontally with 2-pixel gaps
    height = max(sides)
    squares = []
    col = 0
    for side in sides:
        squares.append((0, col, side))
        col += side + 2

    mask = _mask_with_squares((height, col), squares)

    # Verify components are disjoint (sanity check)
    areas = component_areas(mask)
    assert len(areas) == 4, f"Expected 4 components, got {len(areas)}"

    # Test major_object_count: only the largest should count as major
    diag = analyze_mask(mask)
    assert diag.major_object_count == 1


def test_mask_from_areas_guard_rejects_insufficient_canvas_and_accepts_sufficient():
    """Validates the fail-loud guard on _mask_from_areas. The positive control lives
    in the SAME test so an always-raises or never-raises bug cannot pass silently."""
    # NEGATIVE CASE: canvas too small in width
    # Sizes [92, 89, 84, 82, 23, 21] need 391 + 25 gaps = 416px, but only 400 provided
    with pytest.raises(ValueError, match="Canvas width 400 is insufficient"):
        _mask_from_areas((400, 400), [8628, 7981, 7106, 6748, 570, 454])

    # POSITIVE CONTROL: same areas but canvas wide enough must NOT raise
    mask = _mask_from_areas((400, 450), [8628, 7981, 7106, 6748, 570, 454])
    assert mask is not None

    # NEGATIVE CASE: canvas too small in height
    # Largest side is 92, but only 50px height provided
    with pytest.raises(ValueError, match="Canvas height 50 is insufficient"):
        _mask_from_areas((50, 500), [8628, 7981, 7106, 6748, 570, 454])

    # POSITIVE CONTROL: same areas but canvas tall enough must NOT raise
    mask = _mask_from_areas((100, 500), [8628, 7981, 7106, 6748, 570, 454])
    assert mask is not None


def test_multi_specimen_fires_on_measured_failure_and_not_on_single_plant():
    """Calibrated on the real mode-1 failure. The positive control lives in the
    SAME test so an always-fires bug cannot masquerade as detection."""

    # Real measured areas from bio3d-arena/.../736_multi4.png
    # Sizes: [92, 89, 84, 82, 23, 21], total width needed: 391 + 25 (gaps) = 416px
    four_plants = _mask_from_areas((400, 450), [8628, 7981, 7106, 6748, 570, 454])
    # Verify the constructed areas match intentions (fail-loud: we measure what we built)
    expected_areas = [8464, 7921, 7056, 6724, 529, 441]  # int(sqrt(x))^2 for each area
    actual_areas = component_areas(four_plants)
    assert actual_areas == expected_areas, (
        f"Expected areas {expected_areas}, got {actual_areas}"
    )

    warn = multi_specimen_warning(analyze_mask(four_plants))
    assert warn is not None
    assert warn.code == "multi_specimen"
    assert "auto_grid" in warn.message

    # POSITIVE CONTROL: one plant + disconnected leaf tips -> must NOT fire
    # Sizes: [92, 23, 21, 16], total width needed: 152 + 15 (gaps) = 167px
    one_plant = _mask_from_areas((400, 200), [8628, 570, 454, 274])
    # Verify areas match intentions
    expected_areas_control = [8464, 529, 441, 256]
    actual_areas_control = component_areas(one_plant)
    assert actual_areas_control == expected_areas_control, (
        f"Expected areas {expected_areas_control}, got {actual_areas_control}"
    )

    assert multi_specimen_warning(analyze_mask(one_plant)) is None
