"""Per-region measurement.

Two failure modes dominate here, and both look like success:

1. **An empty cell reports zeros.** Measured on PlantCV 4.11.3: `create_labels`
   returns n=4 for a 2x2 grid with one empty cell, and the analysis emits a
   complete trait set with `area = 0.0` for that cell. A zero that looks like a
   measurement is worse than no answer, so the empty region must be refused by
   name.
2. **Traits get attributed to the wrong plant.** If region indices and label
   indices ever drift apart, every row after a gap describes its neighbour —
   silently, with entirely plausible numbers. Every test here therefore uses
   plants of DELIBERATELY DIFFERENT SIZES, so a shift is detectable at all. A
   fixture of identical plants could not fail this test.
"""

import asyncio
import json

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from plantcv_mcp.imaging import render_region_overlay
from plantcv_mcp.regions import (
    MAX_REGIONS,
    RegionSpecError,
    build_regions,
    measure_regions,
)
from plantcv_mcp.server import build_server

# Radii chosen so areas are far apart: pi*r^2 = 1257 / 5027 / 11310 px.
# Cell 2 is deliberately EMPTY.
PLANTS = {0: (100, 100, 20), 1: (300, 100, 40), 3: (300, 300, 60)}
SIZE = 400


def _tray() -> tuple[np.ndarray, np.ndarray]:
    """A 2x2 tray: three plants of different sizes, one empty cell."""
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    for cx, cy, r in PLANTS.values():
        sel = (xx - cx) ** 2 + (yy - cy) ** 2 <= r**2
        mask[sel] = 255
        img[sel] = (40, 200, 40)
    return img, mask


def _rect_grid(img, mask):
    return build_regions(
        img,
        mask,
        mode="rect_grid",
        nrows=2,
        ncols=2,
        coord=(10, 10),
        height=180,
        width=180,
        spacing=(200, 200),
    )


def test_empty_region_is_refused_by_name_not_reported_as_zero():
    """The core guard. PlantCV WILL hand back area=0.0 for the empty cell."""
    img, mask = _tray()
    results = measure_regions(img, mask, _rect_grid(img, mask))

    assert len(results) == 4
    empty = results[2]
    assert empty["measured"] is False, "the empty cell must not be reported as measured"
    assert empty["traits"] is None, "an empty region must carry NO traits, not zeros"
    assert "no plant material" in empty["reason"].lower()

    # Positive control in the same test: the other three ARE measured, so a
    # blanket refusal cannot pass this by refusing everything.
    for i in (0, 1, 3):
        assert results[i]["measured"] is True
        assert results[i]["traits"]["area"]["value"] > 0


def test_traits_are_attributed_to_the_right_region():
    """Sizes are deliberately unequal so a one-off index shift is visible.

    This is the test the empty cell exists to threaten: if empties were dropped
    from the label sequence, region 3's traits would land on region 2.
    """
    img, mask = _tray()
    results = measure_regions(img, mask, _rect_grid(img, mask))

    areas = {i: results[i]["traits"]["area"]["value"] for i in (0, 1, 3)}
    assert areas[0] < areas[1] < areas[3], f"regions mis-attributed: {areas}"

    # Areas must match the discs actually drawn, within rasterisation slack.
    for i, (_, _, r) in PLANTS.items():
        assert abs(areas[i] - np.pi * r**2) / (np.pi * r**2) < 0.05

    # Geometry is reported per region, and rows/cols follow the grid.
    assert [r["row"] for r in results] == [0, 0, 1, 1]
    assert [r["col"] for r in results] == [0, 1, 0, 1]


def test_auto_grid_infers_the_layout_from_the_mask():
    img, mask = _tray()
    regions = build_regions(img, mask, mode="auto_grid", nrows=2, ncols=2)
    results = measure_regions(img, mask, regions)

    measured = [r for r in results if r["measured"]]
    assert len(measured) == 3, (
        "three plants were drawn; auto_grid found a different set"
    )
    # The three measured areas must still be the three drawn discs.
    got = sorted(round(r["traits"]["area"]["value"]) for r in measured)
    want = sorted(round(np.pi * r**2) for _, _, r in PLANTS.values())
    for g, w in zip(got, want, strict=True):
        assert abs(g - w) / w < 0.05


def test_auto_grid_refuses_an_empty_mask_instead_of_inventing_a_grid():
    img, _ = _tray()
    with pytest.raises(RegionSpecError, match="mask is empty"):
        build_regions(
            img, np.zeros((SIZE, SIZE), np.uint8), mode="auto_grid", nrows=2, ncols=2
        )


def test_rect_grid_refuses_incomplete_geometry_rather_than_guessing():
    """A guessed cell origin silently measures the neighbouring plant."""
    img, mask = _tray()
    with pytest.raises(RegionSpecError, match="coord"):
        build_regions(
            img, mask, mode="rect_grid", nrows=2, ncols=2, height=180, width=180
        )

    # Positive control: complete geometry is accepted, so the check above is not
    # simply rejecting rect_grid outright.
    assert len(_rect_grid(img, mask)) == 4


def test_region_count_is_capped():
    img, mask = _tray()
    with pytest.raises(RegionSpecError, match="exceeds"):
        build_regions(
            img,
            mask,
            mode="rect_grid",
            nrows=MAX_REGIONS,
            ncols=2,
            coord=(0, 0),
            height=2,
            width=2,
            spacing=(2, 2),
        )


def test_overlay_marks_measured_and_refused_regions_differently():
    """The numbers are per-region, so the picture has to say which is which."""
    img, mask = _tray()
    regions = _rect_grid(img, mask)
    results = measure_regions(img, mask, regions)
    measured = [bool(r["measured"]) for r in results]

    overlay = render_region_overlay(img, mask, regions.bboxes, measured)
    assert overlay.shape == img.shape
    assert not np.array_equal(overlay, img), "overlay is identical to the input"

    # The refused region must be drawn in the refused colour, and a measured one
    # in the measured colour -- otherwise an empty cell and an uncovered cell
    # look the same, and they call for opposite fixes.
    from plantcv_mcp.imaging import EMPTY_BGR, MEASURED_BGR

    flat = overlay.reshape(-1, 3)
    assert any((flat == np.array(EMPTY_BGR)).all(axis=1)), (
        "no refused-region colour drawn"
    )
    assert any((flat == np.array(MEASURED_BGR)).all(axis=1)), (
        "no measured-region colour drawn"
    )


def test_units_convert_per_region():
    img, mask = _tray()
    px_per_mm = 10.0
    plain = measure_regions(img, mask, _rect_grid(img, mask))
    mm = measure_regions(img, mask, _rect_grid(img, mask), px_per_mm=px_per_mm)

    a_px = plain[1]["traits"]["area"]["value"]
    a_mm = mm[1]["traits"]["area"]["value"]
    assert mm[1]["traits"]["area"]["unit"] == "mm2"
    assert abs(a_mm - a_px / px_per_mm**2) < 1e-6


def test_measure_regions_over_the_real_mcp_layer(tmp_path):
    """The tool must return BOTH the rows and the overlay image.

    Per-region numbers without the labelled picture would break the rule the
    whole server is built around.
    """
    img, _ = _tray()
    path = tmp_path / "tray.png"
    PILImage.fromarray(img[:, :, ::-1]).save(path)

    server = build_server()
    seg = asyncio.run(
        server.call_tool(
            "segment",
            {
                "image_path": str(path),
                "channel": "a",
                "method": "otsu",
                "object_type": "dark",
                "fill_size": 10,
            },
        )
    )
    session_id = json.loads(seg.content[0].text)["session_id"]

    out = asyncio.run(
        server.call_tool(
            "measure_regions",
            {
                "session_id": session_id,
                "mode": "rect_grid",
                "nrows": 2,
                "ncols": 2,
                "coord": [10, 10],
                "height": 180,
                "width": 180,
                "spacing": [200, 200],
            },
        )
    )
    text_block, image_block = out.content
    payload = json.loads(text_block.text)

    assert payload["regions_total"] == 4
    assert payload["regions_measured"] == 3
    assert payload["regions_empty"] == 1
    assert image_block.type == "image", "per-region traits must come with the overlay"


def test_coord_of_the_wrong_length_is_refused(tmp_path):
    """Silently taking the first two would lay the grid down in the wrong place."""
    from plantcv_mcp.server import _as_xy

    assert _as_xy(None, "coord") is None
    assert _as_xy([1, 2], "coord") == (1, 2)
    with pytest.raises(RegionSpecError, match="exactly"):
        _as_xy([1, 2, 3], "spacing")


# --- geometry validation: malformed rect_grid input must be refused up front ---
#
# `pcv.roi.multi_rect` accepts any integers, and a rectangle that starts off the
# image reached native OpenCV/PlantCV code inside `measure_regions` and killed
# the whole process with SIGSEGV (exit 139) — measured on PlantCV 4.11.3 with the
# exact values in test_rect_grid_refuses_cells_outside_the_image below. A crash
# in a stdio server takes every session with it, so the wire schema's leniency
# has to be closed here, before anything native is called.


def _small_scene():
    img = np.zeros((100, 100, 3), np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[20:40, 20:40] = 255
    return img, mask


@pytest.mark.parametrize(
    "bad",
    [
        {"height": 0},
        {"height": -5},
        {"width": 0},
        {"width": -50},
    ],
)
def test_rect_grid_refuses_nonpositive_cell_dimensions(bad):
    img, mask = _small_scene()
    spec = {
        "mode": "rect_grid",
        "nrows": 1,
        "ncols": 1,
        "coord": (10, 10),
        "height": 50,
        "width": 50,
        **bad,
    }
    with pytest.raises(RegionSpecError, match="positive"):
        build_regions(img, mask, spacing=(50, 50), **spec)

    # Positive control: the same spec with sane dimensions is accepted.
    good = build_regions(
        img,
        mask,
        mode="rect_grid",
        nrows=1,
        ncols=1,
        coord=(10, 10),
        height=50,
        width=50,
        spacing=(50, 50),
    )
    assert good.bboxes == [(10, 10, 50, 50)]


def test_rect_grid_requires_spacing_even_for_a_single_cell():
    """Pinned PlantCV rejects spacing=None outright, so allowing it for a 1x1
    grid only moved the error somewhere with a worse message."""
    img, mask = _small_scene()
    with pytest.raises(RegionSpecError, match="spacing"):
        build_regions(
            img,
            mask,
            mode="rect_grid",
            nrows=1,
            ncols=1,
            coord=(10, 10),
            height=50,
            width=50,
            spacing=None,
        )


@pytest.mark.parametrize(
    "coord, spacing, nrows, ncols",
    [
        ((-10, -10), (0, 0), 1, 1),  # the SIGSEGV case
        ((500, 500), (50, 50), 1, 1),  # entirely off a 100px image
        ((10, 10), (60, 60), 2, 2),  # second column runs past the right edge
        ((10, 10), (-60, -60), 2, 2),  # negative spacing walks off the top-left
    ],
)
def test_rect_grid_refuses_cells_outside_the_image(coord, spacing, nrows, ncols):
    img, mask = _small_scene()
    with pytest.raises(RegionSpecError, match="outside"):
        build_regions(
            img,
            mask,
            mode="rect_grid",
            nrows=nrows,
            ncols=ncols,
            coord=coord,
            height=50,
            width=50,
            spacing=spacing,
        )


def test_in_bounds_rect_grid_still_measures():
    """Positive control for the bounds guard: a valid grid on the same scene
    both builds and measures, so the refusals above are not rejecting
    rect_grid wholesale."""
    img, mask = _small_scene()
    regions = build_regions(
        img,
        mask,
        mode="rect_grid",
        nrows=1,
        ncols=1,
        coord=(10, 10),
        height=50,
        width=50,
        spacing=(50, 50),
    )
    out = measure_regions(img, mask, regions)
    assert out[0]["measured"] is True
    assert out[0]["traits"]["area"]["value"] == 400.0


def test_auto_grid_refuses_nonpositive_radius():
    img, mask = _tray()
    with pytest.raises(RegionSpecError, match="radius"):
        build_regions(img, mask, mode="auto_grid", nrows=2, ncols=2, radius=0)


def test_a_few_stray_pixels_in_a_cell_are_refused_not_measured():
    """A 2x2 speck in a cell produced a full, plausible trait row (area 4,
    convex hull, ...) indistinguishable from a germinating seedling. Each cell
    is held to the same degeneracy floor measure() applies to a whole frame."""
    img, mask = _tray()
    mask2, img2 = mask.copy(), img.copy()
    mask2[300:302, 100:102] = 255  # 4 px inside the otherwise-empty cell 2
    img2[300:302, 100:102] = (40, 200, 40)
    results = measure_regions(img2, mask2, _rect_grid(img2, mask2))

    speck = results[2]
    assert speck["measured"] is False, "4 stray px must not be a measured plant"
    assert speck["traits"] is None
    assert "below" in speck["reason"], "the reason must name the floor"

    # Positive control: the three real plants still measure.
    for i in (0, 1, 3):
        assert results[i]["measured"] is True
        assert results[i]["traits"]["area"]["value"] > 0


def test_nan_px_per_mm_is_refused_on_the_regions_path_too():
    """measure() refuses a non-finite px_per_mm; the regions path used to let
    NaN through (NaN <= 0 is False) and every converted trait came back NaN."""
    img, mask = _tray()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            measure_regions(img, mask, _rect_grid(img, mask), px_per_mm=bad)
    # Positive control: a finite scale converts to mm2.
    res = measure_regions(img, mask, _rect_grid(img, mask), px_per_mm=2.0)
    assert res[0]["traits"]["area"]["unit"] == "mm2"


def test_implausible_longest_path_is_flagged_per_region():
    """Live dogfood repro: on the four-view fixture, PlantCV reports region 3's
    longest_path as ~7 px against a ~343 px tall object — an artefact that read
    like a measurement. It must carry a warning naming itself."""
    from pathlib import Path

    from plantcv_mcp.server import _measure_regions_impl, _segment_impl

    fixture = str(Path(__file__).parent / "fixtures" / "multi_specimen.png")
    seg = _segment_impl(fixture, "a", "otsu")  # the exact recipe observed live
    res = _measure_regions_impl(seg["session_id"], nrows=2, ncols=2)
    r3 = res["regions"][3]
    assert r3["traits"]["longest_path"]["value"] < 0.1 * r3["traits"]["height"]["value"]
    assert "implausible_longest_path" in [w["code"] for w in r3["warnings"]]
    # Positive control: region 1's longest_path is honest and stays unflagged.
    assert "implausible_longest_path" not in [
        w["code"] for w in res["regions"][1]["warnings"]
    ]


def test_an_object_spilling_out_of_its_cell_is_flagged_and_a_contained_one_is_not():
    """Found on a real X-Rite tray photo: create_labels(roi_type="partial")
    measures any object that OVERLAPS a cell whole, so a misaligned grid
    reported objects 785 px wide inside 369 px cells -- two plants per row --
    with `regions_measured: 17` and no warning. The cell must say when the
    object it reports is not inside it."""
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    # Two plants merged into one 281 px object spanning cells 0 and 1 (each
    # 180 px wide: 1.56x the cell), and one plant well inside cell 2. Leaf-tip
    # overhang on a tight cell measures up to 1.02x on real trays and must NOT
    # fire, so the object here is unambiguously larger than a cell.
    sel = ((xx - 200) / 140.0) ** 2 + ((yy - 100) / 40.0) ** 2 <= 1.0
    mask[sel] = 255
    img[sel] = (40, 200, 40)
    sel = (xx - 100) ** 2 + (yy - 300) ** 2 <= 40**2
    mask[sel] = 255
    img[sel] = (40, 200, 40)
    results = measure_regions(img, mask, _rect_grid(img, mask))

    codes = [[w["code"] for w in r["warnings"]] for r in results]
    # PlantCV hands the straddling disc WHOLE to one cell (here cell 1) and
    # reports the other as empty. Both halves of that must be named.
    assert "object_exceeds_region" in codes[1]
    msg = next(
        w["message"]
        for w in results[1]["warnings"]
        if w["code"] == "object_exceeds_region"
    )
    assert "1.6x" in msg and "outside" in msg
    assert results[0]["measured"] is False
    assert "object_claimed_by_neighbour" in codes[0]
    assert "region 1" in results[0]["reason"]
    # Positive control: the contained plant carries no such warning.
    assert results[2]["measured"] is True
    assert "object_exceeds_region" not in codes[2]


def test_auto_grid_with_empty_cells_and_spilling_objects_is_called_misaligned():
    """The set-level signal for the same photo: a whole column of empty cells
    next to cells whose objects spill out means the inferred grid does not sit
    on the tray. rect_grid geometry is the user's, so it is not second-guessed."""
    from plantcv_mcp.regions import grid_misalignment_warning

    spill = {
        "measured": True,
        "warnings": [{"code": "object_exceeds_region", "message": ""}],
    }
    empty = {"measured": False, "warnings": []}
    fine = {"measured": True, "warnings": []}
    assert (
        grid_misalignment_warning("auto_grid", [spill, empty, fine]).code
        == "grid_misaligned"
    )
    # Positive controls: neither ingredient alone, and never for rect_grid.
    assert grid_misalignment_warning("auto_grid", [spill, fine, fine]) is None
    assert grid_misalignment_warning("auto_grid", [empty, fine, fine]) is None
    assert grid_misalignment_warning("rect_grid", [spill, empty, fine]) is None


# --- panel audit of 1.5.1 (2026-08-29) ---


def _tight_two_cells():
    """Two discs each filling ~72% of a tight rect_grid cell: the happy path a
    tray's cells are calibrated for, which the per-cell coverage check (with
    the whole-FRAME threshold and the RGB polarity remedy) called inverted."""
    import cv2

    img = np.zeros((100, 200, 3), np.uint8)
    mask = np.zeros((100, 200), np.uint8)
    cv2.circle(mask, (50, 50), 48, 255, -1)
    cv2.circle(mask, (150, 50), 48, 255, -1)
    regions = build_regions(
        img,
        mask,
        mode="rect_grid",
        nrows=1,
        ncols=2,
        coord=(0, 0),
        height=100,
        width=100,
        spacing=(100, 0),
    )
    return img, mask, regions


def test_a_plant_filling_its_cell_is_not_called_inverted():
    img, mask, regions = _tight_two_cells()
    rows = measure_regions(img, mask, regions)
    assert [r["measured"] for r in rows] == [True, True]
    assert all(r["region_coverage"] > 0.7 for r in rows)
    assert all(
        "implausible_coverage" not in [w["code"] for w in r["warnings"]] for r in rows
    )
    assert all(r["warnings"] == [] for r in rows)


def test_a_cell_holding_several_plants_says_multi_specimen():
    """A 2x2 grid over a 4x4 tray: four seedlings share a cell, their combined
    bbox fits the cell (no object_exceeds_region), and the row's area is four
    plants. The cell must say so."""
    import cv2

    img = np.zeros((200, 200, 3), np.uint8)
    mask = np.zeros((200, 200), np.uint8)
    for cx, cy in ((25, 25), (75, 25), (25, 75), (75, 75)):  # four in cell 0
        cv2.circle(mask, (cx, cy), 12, 255, -1)
    cv2.circle(mask, (150, 150), 30, 255, -1)  # one in cell 3
    regions = build_regions(
        img,
        mask,
        mode="rect_grid",
        nrows=2,
        ncols=2,
        coord=(0, 0),
        height=100,
        width=100,
        spacing=(100, 100),
    )
    rows = measure_regions(img, mask, regions)
    codes = [[w["code"] for w in r["warnings"]] for r in rows]
    assert "multi_specimen" in codes[0]
    assert "object_exceeds_region" not in codes[0]
    msg = next(
        w["message"] for w in rows[0]["warnings"] if w["code"] == "multi_specimen"
    )
    assert "4" in msg and "cell" in msg
    assert codes[3] == []  # positive control: one plant, no advisory


def test_leaf_tip_overhang_does_not_trip_object_exceeds_region():
    """The 1.25 gate has a measured margin: clean-tray overhang 1.02x, merged
    neighbours 1.68x. Pin both sides so the constant cannot drift to 1.01."""
    import cv2

    img = np.zeros((100, 400, 3), np.uint8)
    mask = np.zeros((100, 400), np.uint8)
    cv2.rectangle(
        mask, (5, 30), (114, 70), 255, -1
    )  # 110 px wide in a 100 px cell: 1.10x
    cv2.rectangle(mask, (205, 30), (344, 70), 255, -1)  # 140 px wide: 1.40x
    regions = build_regions(
        img,
        mask,
        mode="rect_grid",
        nrows=1,
        ncols=4,
        coord=(0, 0),
        height=100,
        width=100,
        spacing=(100, 0),
    )
    rows = measure_regions(img, mask, regions)
    codes = {r["index"]: [w["code"] for w in r["warnings"]] for r in rows}
    measured = {r["index"]: r["measured"] for r in rows}
    # The 1.10x object is measured by whichever cell PlantCV hands it to, and
    # that cell must not call it a merge.
    owner = next(i for i in (0, 1) if measured[i])
    assert "object_exceeds_region" not in codes[owner]
    owner_b = next(i for i in (2, 3) if measured[i])
    assert "object_exceeds_region" in codes[owner_b]  # 1.40x: past the gate


def test_a_claimed_cell_reports_the_coverage_its_reason_describes():
    """region_coverage was 0.0 on a cell whose reason said 'N px of plant
    material lie in this cell'. Two fields of one row must agree."""
    img = np.full((SIZE, SIZE, 3), 30, np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    yy, xx = np.ogrid[:SIZE, :SIZE]
    mask[((xx - 200) / 140.0) ** 2 + ((yy - 100) / 40.0) ** 2 <= 1.0] = 255
    rows = measure_regions(img, mask, _rect_grid(img, mask))
    claimed = next(r for r in rows if r["reason"] and "Not empty" in r["reason"])
    assert claimed["region_coverage"] > 0.0
    px = int(claimed["reason"].split("Not empty: ")[1].split(" px")[0])
    _x, _y, w, h = claimed["bbox"]
    assert claimed["region_coverage"] == pytest.approx(px / (w * h))


def test_region_count_mismatch_is_reported_when_fewer_regions_are_built(monkeypatch):
    import plantcv_mcp.regions as regions_mod

    img, mask = _tray()
    real = regions_mod._bboxes_from
    monkeypatch.setattr(regions_mod, "_bboxes_from", lambda rois: real(rois)[:-1])
    region_set = build_regions(img, mask, mode="auto_grid", nrows=2, ncols=2)
    assert len(region_set) == 3
    codes = [w.code for w in region_set.warnings]
    assert "region_count_mismatch" in codes
    msg = next(
        w.message for w in region_set.warnings if w.code == "region_count_mismatch"
    )
    assert "4" in msg and "3" in msg


def test_grid_misalignment_counts_a_claimed_cell_once_on_each_side():
    """A cell refused as claimed-by-neighbour is both an empty cell and a spill
    signal; on its own it is a straddle, which under auto_grid IS the
    misalignment signature. Pinned so the double count is a decision, not an
    accident."""
    from plantcv_mcp.regions import grid_misalignment_warning

    claimed = {
        "measured": False,
        "warnings": [{"code": "object_claimed_by_neighbour", "message": ""}],
    }
    fine = {"measured": True, "warnings": []}
    assert (
        grid_misalignment_warning("auto_grid", [claimed, fine]).code
        == "grid_misaligned"
    )
    assert grid_misalignment_warning("rect_grid", [claimed, fine]) is None


def test_a_leaf_that_leaves_and_reenters_its_cell_is_one_plant():
    """Judged on the cell CROP, a single plant whose leaf loops outside the cell
    is two comparably-sized pieces and read as two plants (a 20,533-px
    arabidopsis did). The object is one component; judge the object.

    Geometry is checked, not assumed: the body and a second lobe sit in cell
    1, joined only by a stalk that loops through cell 0. PlantCV's partial
    ROI hands the whole object to cell 1, whose crop holds TWO comparable
    pieces while the frame holds ONE component. (The first version of this
    test put the body in cell 0 and the loop in cell 1; PlantCV then gave the
    object to cell 1, whose crop was a single piece, and the test could not
    fail — a mutation check caught it.)"""
    import cv2

    from plantcv_mcp.diagnostics import analyze_mask

    img = np.zeros((100, 200, 3), np.uint8)
    mask = np.zeros((100, 200), np.uint8)
    cv2.circle(mask, (159, 30), 22, 255, -1)  # body in cell 1
    cv2.rectangle(mask, (59, 26), (139, 34), 255, -1)  # stalk out into cell 0
    cv2.rectangle(mask, (59, 34), (67, 70), 255, -1)  # down, inside cell 0
    cv2.rectangle(mask, (59, 62), (139, 70), 255, -1)  # and back
    cv2.circle(mask, (154, 78), 18, 255, -1)  # second lobe, inside cell 1
    crop = mask[:, 100:]
    assert cv2.connectedComponents(crop)[0] - 1 == 2, "must split in the crop"
    assert cv2.connectedComponents(mask)[0] - 1 == 1, "one object in the frame"
    assert analyze_mask(crop).major_object_count == 2  # comparable pieces
    geometry = {
        "mode": "rect_grid",
        "nrows": 1,
        "ncols": 2,
        "coord": (0, 0),
        "height": 100,
        "width": 100,
    }
    rows = measure_regions(
        img, mask, build_regions(img, mask, spacing=(100, 0), **geometry)
    )
    owner = next(r for r in rows if r["measured"])
    assert owner["index"] == 1
    assert "multi_specimen" not in [w["code"] for w in owner["warnings"]]
    # Positive control: two genuinely separate plants in a cell DO fire.
    mask2 = np.zeros((100, 200), np.uint8)
    cv2.circle(mask2, (130, 30), 18, 255, -1)
    cv2.circle(mask2, (170, 70), 18, 255, -1)
    rows2 = measure_regions(
        img, mask2, build_regions(img, mask2, spacing=(100, 0), **geometry)
    )
    owner2 = next(r for r in rows2 if r["measured"])
    assert "multi_specimen" in [w["code"] for w in owner2["warnings"]]


# --- panel audit of 1.5.4 (2026-08-30) -------------------------------------


def _row_of_discs(n: int) -> tuple[np.ndarray, np.ndarray]:
    img = np.full((300, 600, 3), 200, np.uint8)
    mask = np.zeros((300, 600), np.uint8)
    for i in range(n):
        cv2.circle(mask, (60 + i * 110, 150), 20, 255, -1)
    return img, mask


@pytest.mark.parametrize(
    ("n", "nrows", "ncols"),
    [
        (1, 1, 1),  # sklearn: 1 sample, GaussianMixture needs 2
        (1, 1, 2),
        (2, 1, 3),  # n_samples < n_components
        (2, 2, 2),  # one row of objects, two rows asked: cv2.error drawing NaN
        (4, 2, 2),
    ],
)
def test_auto_grid_that_cannot_infer_the_layout_is_refused_by_name(n, nrows, ncols):
    """PlantCV's auto_grid fits a mixture per axis; with fewer objects than
    components it raises sklearn's ValueError, and with objects that do not
    spread into the rows asked it draws NaN geometry and OpenCV raises. Both
    leaked raw (batch quoted 'Found array with 1 sample(s)' as the reason)."""
    img, mask = _row_of_discs(n)
    with pytest.raises(RegionSpecError, match="auto_grid could not infer"):
        build_regions(img, mask, nrows=nrows, ncols=ncols)
    # Positive control: a layout the objects support (two objects at least —
    # PlantCV's mixture needs two samples even for one column).
    m = max(n, 2)
    assert len(build_regions(*_row_of_discs(m), nrows=1, ncols=m).bboxes) == m


def test_a_fragment_of_a_neighbours_object_is_not_a_plant():
    """Two discs filling their 1x2 cells, mask inverted: PlantCV hands cell 1
    the whole 400x200 background (exceeds its cell) and cell 0 a 544-px
    outline of the same object, 195x195 — inside the exceeds ratio, above the
    floor, and it measured as a plant with area 544."""
    img = np.full((200, 400, 3), 200, np.uint8)
    mask = np.full((200, 400), 255, np.uint8)
    cv2.circle(mask, (100, 100), 96, 0, -1)
    cv2.circle(mask, (300, 100), 96, 0, -1)
    grid = {
        "mode": "rect_grid",
        "nrows": 1,
        "ncols": 2,
        "coord": (0, 0),
        "height": 200,
        "width": 200,
        "spacing": (200, 0),
    }
    rows = measure_regions(img, mask, build_regions(img, mask, **grid))
    # Interactive measure_regions keeps the exceeding row beside the numbered
    # overlay (batch withholds it); the fragment is refused outright.
    assert [r["measured"] for r in rows] == [False, True]
    assert "object_exceeds_region" in [w["code"] for w in rows[1]["warnings"]]
    fragment = rows[0]
    assert "object_claimed_by_neighbour" in fragment["reason"]
    assert "544 of the" in fragment["reason"]
    assert "region 1" in fragment["reason"]
    # Positive control: the discs themselves, each filling its cell.
    rows = measure_regions(img, 255 - mask, build_regions(img, 255 - mask, **grid))
    assert [r["measured"] for r in rows] == [True, True]
    assert all(r["traits"]["area"]["value"] > 28000 for r in rows)


def test_an_intruded_upon_cell_keeps_its_own_plant():
    """The owned-material guard's other half: the misaligned X-Rite tray's
    intruded-upon cells own 0.35-0.39 of the material in them and their own
    object IS their plant. A cell in that band must stay measured, and its
    traits must be its own object, not the intruder."""
    img = np.full((200, 400, 3), 200, np.uint8)
    mask = np.zeros((200, 400), np.uint8)
    cv2.circle(mask, (60, 100), 30, 255, -1)  # cell 0's own plant
    cv2.circle(mask, (300, 100), 60, 255, -1)  # cell 1's plant...
    mask[40:160, 150:240] = 255  # ...with a lobe reaching into cell 0
    own = int((mask[:, :150] > 0).sum())
    in_cell0 = int((mask[:, :200] > 0).sum())
    # Fixture honesty: between the shipped fraction and half, the band the
    # real trays sit in — a stricter fraction would take this cell.
    assert 0.2 < own / in_cell0 < 0.5
    grid = {
        "mode": "rect_grid",
        "nrows": 1,
        "ncols": 2,
        "coord": (0, 0),
        "height": 200,
        "width": 200,
        "spacing": (200, 0),
    }
    rows = measure_regions(img, mask, build_regions(img, mask, **grid))
    assert [r["measured"] for r in rows] == [True, True]
    assert "object_claimed_by_neighbour" not in (rows[0]["reason"] or "")
    area = rows[0]["traits"]["area"]["value"]
    assert own * 0.95 < area < own * 1.05  # its own disc, not the intruder
    assert rows[1]["traits"]["area"]["value"] > 20000
