"""Per-region measurement for thermal and hyperspectral sessions.

Found on a real FLIR tray (20 plants): `multi_specimen` told the caller to use
measure_regions(), which refused the thermal session. The per-region
partition (grid, labels, empty/claimed/degenerate reasons) is the same for
every modality; only the per-region statistic differs.
"""

import json

import numpy as np
import pytest
from plantcv import plantcv as pcv
from plantcv.plantcv.classes import Spectral_data

from plantcv_mcp.regions import (
    build_regions,
    measure_regions_spectral,
    measure_regions_thermal,
)

H, W = 120, 160
WL = [600.0, 670.0, 700.0, 800.0, 900.0]


def _two_discs():
    yy, xx = np.ogrid[:H, :W]
    # Inside cells 0 and 3 of a 4-column grid (40-px cells) and cells 0 and 1
    # of a 2-column grid: a disc straddling a cell edge is handed whole to one
    # cell by roi_type="partial", which is its own (tested) behaviour.
    left = (xx - 20) ** 2 + (yy - 60) ** 2 <= 15**2
    right = (xx - 140) ** 2 + (yy - 60) ** 2 <= 15**2
    return left, right


def _thermal_scene():
    left, right = _two_discs()
    frame = np.full((H, W), 20.0, np.float64)
    frame[left] = 31.5
    frame[right] = 28.0
    mask = np.where(left | right, 255, 0).astype(np.uint8)
    return frame, mask


def _grid(mask, ncols):
    img = np.zeros((H, W, 3), np.uint8)
    return build_regions(
        img,
        mask,
        mode="rect_grid",
        nrows=1,
        ncols=ncols,
        coord=(0, 0),
        height=H,
        width=W // ncols,
        spacing=(W // ncols, 0),
    )


def test_thermal_regions_report_each_plants_temperature_and_refuse_empties():
    frame, mask = _thermal_scene()
    rows = measure_regions_thermal(frame, mask, _grid(mask, 2))
    assert [r["measured"] for r in rows] == [True, True]
    assert rows[0]["temperature"]["mean"] == pytest.approx(31.5, abs=0.01)
    assert rows[1]["temperature"]["mean"] == pytest.approx(28.0, abs=0.01)
    assert rows[0]["temperature"]["unit"] == "celsius"
    assert rows[0]["pixel_count"] == int(_two_discs()[0].sum())
    assert "traits" not in rows[0]

    # Four cells: the two outer ones are empty and say so; nothing is zero.
    rows4 = measure_regions_thermal(frame, mask, _grid(mask, 4))
    assert [r["measured"] for r in rows4] == [True, False, False, True]
    assert "No plant material" in rows4[1]["reason"]
    assert rows4[1].get("temperature") is None


def _cube_with_two_discs():
    left, right = _two_discs()
    cube = np.zeros((H, W, len(WL)), np.float32)
    cube[..., 0] = 0.3
    cube[..., 1] = np.where(left, 0.2, np.where(right, 0.35, 0.6))  # 670
    cube[..., 2] = 0.5
    cube[..., 3] = np.where(left, 0.8, np.where(right, 0.65, 0.4))  # 800
    cube[..., 4] = 0.7
    # NDVI: left 0.6, right 0.3, background -0.2
    return Spectral_data(
        array_data=cube,
        max_wavelength=max(WL),
        min_wavelength=min(WL),
        max_value=float(cube.max()),
        min_value=float(cube.min()),
        d_type=np.float32,
        wavelength_dict={w: i for i, w in enumerate(WL)},
        samples=W,
        lines=H,
        interleave="bil",
        wavelength_units="nm",
        array_type="datacube",
        pseudo_rgb=np.zeros((H, W, 3), np.uint8),
        filename="two",
        default_bands=None,
    )


def test_spectral_regions_report_each_plants_index_stats():
    left, right = _two_discs()
    mask = np.where(left | right, 255, 0).astype(np.uint8)
    rows = measure_regions_spectral(
        _cube_with_two_discs(), mask, _grid(mask, 2), ("ndvi",)
    )
    assert [r["measured"] for r in rows] == [True, True]
    assert rows[0]["indices"]["ndvi"]["mean"] == pytest.approx(0.6, abs=0.01)
    assert rows[1]["indices"]["ndvi"]["mean"] == pytest.approx(0.3, abs=0.01)
    assert rows[0]["pixel_count"] == int(left.sum())


@pytest.mark.anyio
async def test_measure_regions_accepts_thermal_and_hsi_sessions_over_the_wire(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    from plantcv_mcp.server import build_server

    frame, _ = _thermal_scene()
    csv = tmp_path / "tray.csv"
    np.savetxt(csv, frame, delimiter=",", fmt="%.4f")
    cube = _cube_with_two_discs()
    pcv.hyperspectral.write_data(filename=str(tmp_path / "two"), spectral_data=cube)

    mcp = build_server()
    th = json.loads(
        (await mcp.call_tool("segment_thermal", {"path": str(csv), "min_c": 25.0}))
        .content[0]
        .text
    )
    assert "multi_specimen" in [w["code"] for w in th["warnings"]]
    geometry = {
        "mode": "rect_grid",
        "nrows": 1,
        "ncols": 2,
        "coord": [0, 0],
        "height": H,
        "width": W // 2,
        "spacing": [W // 2, 0],
    }
    r = await mcp.call_tool(
        "measure_regions", {"session_id": th["session_id"], **geometry}
    )
    payload = json.loads(r.content[0].text)
    assert payload["kind"] == "thermal"
    assert payload["regions_measured"] == 2
    assert payload["regions"][0]["temperature"]["mean"] == pytest.approx(31.5, abs=0.01)
    assert payload["regions"][1]["temperature"]["mean"] == pytest.approx(28.0, abs=0.01)
    assert r.content[1].type == "image"

    hs = json.loads(
        (
            await mcp.call_tool(
                "segment_hyperspectral",
                {"envi_path": str(tmp_path / "two.raw"), "threshold": 0.1},
            )
        )
        .content[0]
        .text
    )
    r2 = await mcp.call_tool(
        "measure_regions",
        {"session_id": hs["session_id"], **geometry, "indices": ["ndvi", "savi"]},
    )
    p2 = json.loads(r2.content[0].text)
    assert p2["kind"] == "hsi"
    assert p2["regions"][0]["indices"]["ndvi"]["mean"] == pytest.approx(0.6, abs=0.01)
    assert p2["regions"][1]["indices"]["ndvi"]["mean"] == pytest.approx(0.3, abs=0.01)
    assert "savi" in p2["regions"][1]["indices"]

    # RGB-only arguments on a typed session are refused by name, not ignored.
    with pytest.raises(ToolError, match="px_per_mm"):
        await mcp.call_tool(
            "measure_regions",
            {"session_id": th["session_id"], **geometry, "px_per_mm": 5.0},
        )
    with pytest.raises(ToolError, match="analyses"):
        await mcp.call_tool(
            "measure_regions",
            {"session_id": th["session_id"], **geometry, "analyses": ["color"]},
        )
