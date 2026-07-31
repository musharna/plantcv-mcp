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
