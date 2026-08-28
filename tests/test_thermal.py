"""Thermal sessions: FLIR radiometric JPEGs, CSV and .npz temperature arrays.

Evidence: a synthetic degrees-Celsius array with a warm disc of KNOWN mean, and
PlantCV's real data — a FLIR JPEG (19-24 C via flyr) and the 480x640 array plus
hand-made mask its own test suite uses, against which PlantCV reports mean_temp
33.5095 (measured 2026-08-27).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from plantcv_mcp.server import build_server
from plantcv_mcp.thermal import load_thermal, measure_thermal, segment_thermal

FIXTURES = Path(__file__).parent / "fixtures" / "plantcv"


def _scene():
    frame = np.full((120, 160), 20.0, np.float64)
    yy, xx = np.ogrid[:120, :160]
    disc = (xx - 80) ** 2 + (yy - 60) ** 2 <= 25**2
    frame[disc] = 31.5
    return frame, disc


def _write_csv(tmp_path, frame, name="frame.csv"):
    p = tmp_path / name
    np.savetxt(p, frame, delimiter=",", fmt="%.4f")
    return str(p)


def _write_npz(tmp_path, frame, name="frame.npz"):
    p = tmp_path / name
    np.savez(p, frame)
    return str(p)


def test_load_thermal_reads_csv_npz_and_flir_with_a_digest(tmp_path):
    frame, _ = _scene()
    csv = load_thermal(_write_csv(tmp_path, frame))
    npz = load_thermal(_write_npz(tmp_path, frame))
    assert csv.celsius.shape == npz.celsius.shape == (120, 160)
    assert np.allclose(csv.celsius, frame, atol=1e-3) and np.allclose(
        npz.celsius, frame
    )
    assert csv.digest != npz.digest and len(csv.digest) == 64
    flir = load_thermal(str(FIXTURES / "FLIR_test.jpg"))
    assert flir.celsius.shape == (480, 640)
    assert 19.0 < float(flir.celsius.min()) < float(flir.celsius.max()) < 25.0


def test_segment_by_temperature_and_measure_a_known_warm_disc(tmp_path):
    frame, disc = _scene()
    path = _write_csv(tmp_path, frame)
    seg = segment_thermal(path, min_c=25.0)
    assert seg.diagnostics.mask_fraction == pytest.approx(disc.mean(), abs=0.005)
    assert seg.frame_range == pytest.approx((20.0, 31.5), abs=1e-3)
    assert seg.overlay.shape == (120, 160, 3)
    res = measure_thermal(path, seg.mask)
    assert res.temperature["mean"] == pytest.approx(31.5, abs=0.01)
    assert res.temperature["max"] == pytest.approx(31.5, abs=0.01)
    assert res.pixel_count == int(disc.sum())
    assert "histogram" not in res.as_dict()
    # Positive control: the background measures cold.
    cold = measure_thermal(path, np.where(seg.mask > 0, 0, 255).astype(np.uint8))
    assert cold.temperature["mean"] == pytest.approx(20.0, abs=0.01)


def test_nan_pixels_are_reported_not_silently_dropped(tmp_path):
    """Non-finite pixels are excluded from segmentation correctly — but
    silently. The result must say how many pixels the band never saw."""
    frame, _ = _scene()
    frame[:10, :10] = np.nan  # 100 dead pixels, away from the disc
    seg = segment_thermal(_write_npz(tmp_path, frame), min_c=25.0)
    nan_warnings = [w for w in seg.warnings if w.code == "nan_pixels"]
    assert len(nan_warnings) == 1
    assert "100" in nan_warnings[0].message
    # Positive control: a finite frame carries no such advisory.
    clean, _ = _scene()
    seg2 = segment_thermal(_write_npz(tmp_path, clean, name="clean.npz"), min_c=25.0)
    assert not [w for w in seg2.warnings if w.code == "nan_pixels"]


def test_a_band_that_selects_nothing_is_refused(tmp_path):
    from plantcv_mcp.diagnostics import DegenerateMaskError

    frame, _ = _scene()
    path = _write_csv(tmp_path, frame)
    with pytest.raises(DegenerateMaskError):
        segment_thermal(path, min_c=40.0)
    with pytest.raises(ValueError, match="min_c"):
        segment_thermal(path, min_c=30.0, max_c=25.0)
    with pytest.raises(ValueError, match="min_c"):
        segment_thermal(path)
    # Positive control.
    assert segment_thermal(path, max_c=25.0).diagnostics.mask_fraction > 0.5


def test_real_plantcv_thermal_array_matches_plantcvs_own_number():
    """PlantCV's test data + its mask: analyze.thermal reports mean 33.5095."""
    import cv2

    npz = str(FIXTURES / "thermal_img.npz")
    mask = cv2.imread(str(FIXTURES / "thermal_img_mask.png"), cv2.IMREAD_GRAYSCALE)
    res = measure_thermal(npz, mask, include_histograms=True)
    assert res.temperature["mean"] == pytest.approx(33.509481646483025, abs=1e-6)
    assert res.temperature["max"] == pytest.approx(35.29486, abs=1e-4)
    assert len(res.histogram["bins"]) == len(res.histogram["counts"]) > 10


def test_real_flir_jpeg_segments_by_temperature():
    seg = segment_thermal(str(FIXTURES / "FLIR_test.jpg"), min_c=22.0)
    assert 0.0 < seg.diagnostics.mask_fraction < 1.0
    assert seg.frame_range[0] > 19.0 and seg.frame_range[1] < 25.0


@pytest.mark.anyio
async def test_thermal_tools_over_the_real_mcp_layer(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    frame, _ = _scene()
    path = _write_npz(tmp_path, frame)
    mcp = build_server()
    result = await mcp.call_tool("segment_thermal", {"path": path, "min_c": 25.0})
    text_block, image_block = result.content
    seg = json.loads(text_block.text)
    assert image_block.type == "image"
    assert seg["kind"] == "thermal"
    assert seg["frame_range"] == pytest.approx([20.0, 31.5], abs=1e-3)

    measured = await mcp.call_tool("measure_thermal", {"session_id": seg["session_id"]})
    out = measured.structured_content
    assert out["temperature"]["mean"] == pytest.approx(31.5, abs=0.01)
    assert out["temperature"]["unit"] == "celsius"
    assert out["engine"]["name"] == "PlantCV"

    with pytest.raises(ToolError, match="measure_thermal"):
        await mcp.call_tool("measure", {"session_id": seg["session_id"]})
    with pytest.raises(ToolError, match="Unknown session"):
        await mcp.call_tool("measure_thermal", {"session_id": "nope"})
