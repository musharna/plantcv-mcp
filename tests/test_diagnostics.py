import cv2
import numpy as np
import pytest

from plantcv_mcp.diagnostics import (
    DegenerateMaskError,
    analyze_mask,
    assert_not_degenerate,
    component_areas,
    frame_clipping_warning,
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
    # Names the remedy that EXISTS. It used to point at "roi.auto_grid (phase 2)";
    # phase 2 shipped as measure_regions(), and a warning pointing at unbuilt
    # work is worse than no pointer.
    assert "measure_regions" in warn.message

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


def test_frame_clipping_fires_when_touching_edge_and_not_when_interior():
    clipped = np.zeros((100, 100), dtype=np.uint8)
    clipped[0:40, 0:40] = 255  # touches top and left edges
    warn = frame_clipping_warning(clipped)
    assert warn is not None
    assert warn.code == "frame_clipping"
    assert "lower bound" in warn.message.lower()

    # POSITIVE CONTROL in the same test: interior object must NOT fire
    interior = np.zeros((100, 100), dtype=np.uint8)
    interior[20:60, 20:60] = 255
    assert frame_clipping_warning(interior) is None


def test_frame_clipping_fires_on_bottom_only_contact():
    """The top+left test above cannot catch a regression that drops or
    duplicates the `binary[-1, :]` (bottom) disjunct — exercise it alone."""
    bottom_only = np.zeros((100, 100), dtype=np.uint8)
    bottom_only[60:100, 20:60] = 255  # touches bottom edge only
    warn = frame_clipping_warning(bottom_only)
    assert warn is not None
    assert warn.code == "frame_clipping"

    # POSITIVE CONTROL, same test: interior object must NOT fire
    interior = np.zeros((100, 100), dtype=np.uint8)
    interior[20:60, 20:60] = 255
    assert frame_clipping_warning(interior) is None


def test_frame_clipping_fires_on_right_only_contact():
    """Independent coverage for the `binary[:, -1]` (right) disjunct."""
    right_only = np.zeros((100, 100), dtype=np.uint8)
    right_only[20:60, 60:100] = 255  # touches right edge only
    warn = frame_clipping_warning(right_only)
    assert warn is not None
    assert warn.code == "frame_clipping"

    # POSITIVE CONTROL, same test: interior object must NOT fire
    interior = np.zeros((100, 100), dtype=np.uint8)
    interior[20:60, 20:60] = 255
    assert frame_clipping_warning(interior) is None


def test_is_noisy_needs_both_many_specks_and_no_dominant_object():
    """Both halves of the rule carry weight. Sixty specks around a plant that
    holds 70% of the mask is a plant with a dirty background, not texture; the
    same sixty specks around a plant holding 30% is texture. Dropping the
    largest-fraction clause turns every speckled real tray into a refusal."""
    from plantcv_mcp.diagnostics import is_noisy

    assert is_noisy(component_count=61, largest_frac=0.30, major_object_count=1)
    assert not is_noisy(component_count=61, largest_frac=0.70, major_object_count=1)
    assert not is_noisy(component_count=30, largest_frac=0.30, major_object_count=1)


def test_a_dominant_plant_with_many_specks_is_not_noisy_end_to_end():
    """analyze_mask -> largest_fraction -> is_noisy on real pixels: one 200x200
    plant (40,000 px) and 60 18x18 specks (19,440 px). The plant is 67% of
    the mask, so it is measurable; shrink it to 90x90 and it is not."""
    from plantcv_mcp.diagnostics import (
        RGB_REMEDIES,
        analyze_mask,
        is_noisy,
        largest_fraction,
        noisy_segmentation_warning,
    )

    def scene(plant_px: int) -> np.ndarray:
        rng = np.random.default_rng(3)
        mask = np.zeros((600, 600), np.uint8)
        mask[100 : 100 + plant_px, 100 : 100 + plant_px] = 255
        placed = 0
        while placed < 60:
            y, x = rng.integers(0, 600 - 18, 2)
            if 80 <= y <= 100 + plant_px + 2 and 80 <= x <= 100 + plant_px + 2:
                continue
            if mask[y - 2 : y + 20, x - 2 : x + 20].any():
                continue
            mask[y : y + 18, x : x + 18] = 255
            placed += 1
        return mask

    big = analyze_mask(scene(200))
    assert big.component_count == 61
    assert largest_fraction(big) > 0.6
    assert not is_noisy(
        big.component_count, largest_fraction(big), big.major_object_count
    )
    assert noisy_segmentation_warning(big, RGB_REMEDIES.noisy) is None

    small = analyze_mask(scene(90))
    assert small.component_count == 61
    assert is_noisy(
        small.component_count, largest_fraction(small), small.major_object_count
    )
    assert noisy_segmentation_warning(small, RGB_REMEDIES.noisy) is not None


def test_frame_clipping_ignores_a_minor_sliver_at_the_edge():
    """The real beans photo: every specimen interior, but two 5-px-wide
    background slivers on the right edge declared the group 'cut off'.
    Clipping is a claim about the SPECIMEN, so it needs a major object at
    the edge, not any stray pixel."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, -1)  # the plant, interior
    mask[60:140, 198:200] = 255  # a 160-px background sliver on the edge
    assert frame_clipping_warning(mask) is None

    # Positive control, same test: the plant itself at the edge still fires.
    clipped = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(clipped, (196, 100), 40, 255, -1)
    warn = frame_clipping_warning(clipped)
    assert warn is not None and warn.code == "frame_clipping"

    # And a plant that is MOSTLY out of frame is itself the largest object,
    # so its sliver still counts as clipping.
    mostly_out = np.zeros((200, 200), dtype=np.uint8)
    mostly_out[80:120, 197:200] = 255
    warn = frame_clipping_warning(mostly_out)
    assert warn is not None and warn.code == "frame_clipping"


# --- fisheye dogfood 2026-08-31: minor components inflating extent traits ---


def test_a_far_minor_component_inflates_extent_and_is_named():
    """The fisheye photo's failure shape: a 458k-px plant plus a 1,886-px sliver
    at the opposite corner made measure() report width 2040 px (true 940) with
    no warning at all. The union-vs-major-extent gap needs its own advisory."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((200, 300), np.uint8)
    mask[80:140, 40:100] = 255  # the plant
    mask[0:8, 292:300] = 255  # far corner sliver, well below major threshold
    w = minor_extent_inflation_warning(mask, analyze_mask(mask))
    assert w is not None
    assert w.code == "minor_components_inflate_extent"
    assert "keep_largest" in w.message  # the remedy
    assert "64" in w.message  # the offender's area is named
    # Positive control: the same plant alone earns nothing.
    clean = np.zeros((200, 300), np.uint8)
    clean[80:140, 40:100] = 255
    assert minor_extent_inflation_warning(clean, analyze_mask(clean)) is None


def test_a_speck_beside_the_plant_does_not_warn():
    """Specks adjacent to the plant barely move the union extent; warning on
    them would fire on nearly every real segmentation."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((200, 300), np.uint8)
    mask[80:140, 40:100] = 255
    mask[100:106, 104:110] = 255  # touching-distance speck
    assert minor_extent_inflation_warning(mask, analyze_mask(mask)) is None


def test_two_majors_apart_are_multi_specimen_territory_not_extent_inflation():
    """Two comparably-sized objects far apart are a multi_specimen case: both
    belong to the measurement question, so the union extent is not 'inflated'
    by noise and this advisory must stay silent."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((200, 300), np.uint8)
    mask[80:140, 20:80] = 255
    mask[80:140, 220:280] = 255
    diag = analyze_mask(mask)
    assert diag.major_object_count == 2
    assert minor_extent_inflation_warning(mask, diag) is None


def test_mask_warnings_carries_extent_inflation_to_segment_and_measure():
    """mask_warnings is the shared segment-time/measure-time reporter; the
    advisory must ride it so the overlay and the trait table agree."""
    from plantcv_mcp.diagnostics import mask_warnings

    mask = np.zeros((200, 300), np.uint8)
    mask[80:140, 40:100] = 255
    mask[0:8, 292:300] = 255
    codes = [w.code for w in mask_warnings(mask, analyze_mask(mask))]
    assert "minor_components_inflate_extent" in codes


# --- panel audit of 1.8.0 (2026-08-31): extent-inflation corrections ---


def test_a_speck_off_the_short_axis_of_a_tall_plant_warns():
    """Panel 7 (codex, reproduced): a 10x1000 plant plus a 1-px speck 190 px
    off the short axis inflates width 20x while the union DIAGONAL moves only
    1.02x — the diagonal metric silently missed the exact bug class this
    advisory exists for. Width and height must be compared separately."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((1200, 500), np.uint8)
    mask[100:1100, 100:110] = 255
    mask[600:601, 300:301] = 255
    w = minor_extent_inflation_warning(mask, analyze_mask(mask))
    assert w is not None
    assert w.code == "minor_components_inflate_extent"


def test_two_majors_plus_a_far_speck_never_recommends_keep_largest():
    """Panel 10: with two real plants and one far speck the warning fired and
    told the user to keep_largest — which discards a real plant. The remedy
    must defer to measure_regions when several majors exist."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((200, 400), np.uint8)
    mask[80:140, 20:80] = 255
    mask[80:140, 160:220] = 255
    mask[0:5, 392:397] = 255
    diag = analyze_mask(mask)
    assert diag.major_object_count == 2
    w = minor_extent_inflation_warning(mask, diag)
    assert w is not None
    assert "keep_largest" not in w.message
    assert "measure_regions" in w.message


def test_the_named_offender_is_the_extent_driver_not_the_biggest():
    """Panel 11: the message named the largest-area outsider; a big blob just
    over the major bbox edge outranked the far speck that actually stretched
    the extent. Name the component with the largest overhang."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((300, 600), np.uint8)
    mask[100:200, 100:200] = 255  # the plant (10,000 px)
    mask[120:140, 202:222] = 255  # 400-px blob barely past the bbox edge
    mask[150:158, 560:568] = 255  # 64-px speck far away — the extent driver
    w = minor_extent_inflation_warning(mask, analyze_mask(mask))
    assert w is not None
    assert "64 px" in w.message
    assert "400 px" not in w.message


# --- mutation round 11 (2026-09-01): pinning the per-axis advisory's calibration ---


def test_material_half_a_plant_width_away_still_inflates_the_extent():
    """Round 11: every firing fixture inflated an axis by 2.3x or more, so the
    1.25 threshold could drift to 2.0 unnoticed — past the real fisheye
    photo's 2.17x. A sliver one third of a plant-width past the plant
    stretches width 1.47x: that is a corrupted width and must warn."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((200, 300), np.uint8)
    mask[80:140, 40:100] = 255  # the plant, 60 px wide
    mask[100:108, 120:128] = 255  # 64-px sliver 20 px past its edge (union 88/60)
    w = minor_extent_inflation_warning(mask, analyze_mask(mask))
    assert w is not None
    assert "64" in w.message


def test_a_crumb_beside_the_second_plant_is_measured_against_both_plants():
    """Round 11: the baseline extent is the union of the MAJOR components, not
    the largest one. With two plants far apart, a crumb touching the second
    plant barely moves the major union (silent), while measured against the
    largest plant alone it would read as a 4x width inflation."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((200, 400), np.uint8)
    mask[80:140, 20:80] = 255  # plant A
    mask[80:140, 260:320] = 255  # plant B
    mask[100:106, 322:328] = 255  # crumb at B's edge
    diag = analyze_mask(mask)
    assert diag.major_object_count == 2
    assert minor_extent_inflation_warning(mask, diag) is None
    # Positive control: the same crumb far past BOTH plants does inflate.
    far = mask.copy()
    far[100:106, 322:328] = 0
    far[0:6, 392:398] = 255
    assert minor_extent_inflation_warning(far, analyze_mask(far)) is not None


def test_the_named_offender_is_on_the_axis_that_crossed_the_threshold():
    """Panel audit of 1.8.2 (codex; reproduced): a 1000x10 plant, a 1-px speck
    stretching WIDTH to 1.20x (below threshold) and a 4x4 blob stretching
    HEIGHT to 1.7x (the trigger). Ranking by raw overhang named the 1-px
    speck, and its remedy 'fill_size above 1' could not clear the warning. The
    offender must be the component driving an axis that actually crossed."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((60, 1300), np.uint8)
    mask[20:30, 100:1100] = 255  # the plant, 1000 x 10
    mask[25:26, 1299:1300] = 255  # 1 px, 199 px past the right edge: width 1.20x
    mask[33:37, 500:504] = 255  # 16 px, 7 px below: height 17/10 = 1.7x
    w = minor_extent_inflation_warning(mask, analyze_mask(mask))
    assert w is not None
    assert "16 px" in w.message
    assert "fill_size above 16" in w.message
    assert "1 px" not in w.message


def test_offender_overhang_is_relative_to_each_axis_extent():
    """Round 12: with BOTH axes over threshold, a 300-px overhang on a
    1000-px width (0.3x) must not outrank an 8-px overhang on a 10-px
    height (0.8x) — the second is the larger corruption. Both blobs are 16
    px, so only the position tells them apart."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((60, 1500), np.uint8)
    mask[20:30, 100:1100] = 255  # the plant, 1000 x 10
    mask[24:28, 1396:1400] = 255  # width to 1.30x: overhang 300 px = 0.30 of 1000
    mask[34:38, 600:604] = 255  # height to 2.2x: overhang 8 px = 0.80 of 10
    w = minor_extent_inflation_warning(mask, analyze_mask(mask))
    assert w is not None
    assert "(600, 34)" in w.message
    assert "(1396" not in w.message


def test_the_offender_is_on_the_triggering_axis_even_when_another_overhangs_more():
    """Round 12: relative scoring alone makes the axis gate redundant for a
    single extender (a non-triggering axis's extender is always under 0.25).
    It is NOT redundant when the trigger comes from two sides: two 150-px
    width extenders (0.15 each, union 1.30x) trigger while a 2-px height blob
    (0.20, union 1.2x) does not — the blob must not be named."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((60, 1500), np.uint8)
    mask[20:30, 250:1250] = 255  # the plant, 1000 x 10
    mask[24:28, 96:100] = 255  # 150 px past the left edge
    mask[24:28, 1400:1404] = 255  # 150 px past the right edge
    mask[31:32, 700:704] = 255  # 1 px below a 1-px gap: height 12/10, silent
    w = minor_extent_inflation_warning(mask, analyze_mask(mask))
    assert w is not None
    assert "(700, 31)" not in w.message
    assert "(96, 24)" in w.message or "(1400, 24)" in w.message
    # Transposed: two height extenders trigger, one width blob stays silent.
    tall = np.zeros((1500, 60), np.uint8)
    tall[250:1250, 20:30] = 255
    tall[96:100, 24:28] = 255
    tall[1400:1404, 24:28] = 255
    tall[700:704, 31:32] = 255
    w2 = minor_extent_inflation_warning(tall, analyze_mask(tall))
    assert w2 is not None
    assert "(31, 700)" not in w2.message


def test_the_fill_size_remedy_clears_every_extender_not_just_the_named_one():
    """Panel of 1.9.0 (four judges; reproduced): the remedy's fill_size came
    from the ONE named offender. A 100-px speck 50 px left of a 100-px-wide
    plant outranks a 400-px blob 30 px right of it (relative overhang 0.5 vs
    0.3); 'fill_size above 100' removed the speck and the warning fired again
    on the blob. The threshold must clear every component overhanging a
    triggering axis, and say so."""
    from plantcv_mcp.diagnostics import minor_extent_inflation_warning

    mask = np.zeros((300, 800), np.uint8)
    mask[100:200, 500:600] = 255  # the plant, 100 x 100
    mask[120:130, 440:450] = 255  # 100 px, 50 px left: overhang 0.5
    mask[140:160, 630:650] = 255  # 400 px, 30 px right: overhang 0.3
    mask[170:200, 510:540] = 0  # a hole, so the next blob is a separate component
    mask[175:195, 515:535] = 255  # 400-px bystander INSIDE the extent: not far
    mask[105:145, 500:530] = 0  # a notch at the plant's left edge, and in it
    mask[110:140, 500:525] = 255  # a 750-px blob FLUSH with that edge: overhang 0
    w = minor_extent_inflation_warning(mask, analyze_mask(mask))
    assert w is not None
    assert "(440, 120)" in w.message  # the named offender is still the speck
    assert "fill_size above 400" in w.message
    # Panel of 1.10.1 (codex): fill_size removes EVERY component under it,
    # the interior 400-px bystander included — the count says what the
    # remedy will actually delete, and points at the overlay.
    assert "2 other component(s) under that size" in w.message
    assert "check the overlay" in w.message
    assert "1 other far component" not in w.message
    assert "fill_size above 100" not in w.message
    # Following the remedy clears the warning.
    cleared = mask.copy()
    cleared[120:130, 440:450] = 0
    cleared[140:160, 630:650] = 0
    assert minor_extent_inflation_warning(cleared, analyze_mask(cleared)) is None
    # And a threshold that removes only the speck does not — the old remedy.
    half = mask.copy()
    half[120:130, 440:450] = 0
    assert minor_extent_inflation_warning(half, analyze_mask(half)) is not None
