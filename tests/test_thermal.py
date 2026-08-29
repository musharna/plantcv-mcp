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
    with pytest.raises(ValueError, match="entirely"):
        segment_thermal(path, min_c=40.0)  # outside the frame: refused by range
    with pytest.raises(DegenerateMaskError):
        segment_thermal(path, min_c=22.0, max_c=24.0)  # inside, selects nothing
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


def test_a_multi_array_npz_is_refused_naming_its_arrays(tmp_path):
    """An .npz with several arrays used to yield whichever numpy listed first —
    a silent choice between candidate frames. Refuse and name them instead."""
    frame, _ = _scene()
    p = tmp_path / "two.npz"
    np.savez(p, celsius=frame, kelvin=frame + 273.15)
    with pytest.raises(ValueError, match="celsius"):
        load_thermal(str(p))
    # Positive control: a single-array archive still loads.
    assert load_thermal(_write_npz(tmp_path, frame)).source == "npz"


def test_thermal_implausible_coverage_advises_the_band_not_object_type(tmp_path):
    """The advisory used to give RGB advice ('opposite object_type') on a tool
    that has no object_type; a thermal whole-frame mask needs a narrower band."""
    frame, _ = _scene()
    p = _write_csv(tmp_path, frame)
    seg = segment_thermal(p, min_c=-100.0, max_c=500.0)
    msgs = {w.code: w.message for w in seg.warnings}
    assert "implausible_coverage" in msgs
    assert "min_c" in msgs["implausible_coverage"]
    assert "object_type" not in msgs["implausible_coverage"]


# --- modality dogfood 2026-08-28: remedies, blind band refusal, out-of-range bands ---


def test_band_less_call_refuses_with_the_frame_range_and_percentiles(tmp_path):
    """The old refusal said only "give min_c and/or max_c": the first call was
    always wasted because nothing told the caller the frame spans 20-31.5."""
    frame, _ = _scene()
    path = _write_csv(tmp_path, frame)
    with pytest.raises(ValueError) as exc:
        segment_thermal(path)
    msg = str(exc.value)
    assert "min_c" in msg and "20.0" in msg and "31.5" in msg
    assert "p50" in msg and "p95" in msg
    assert "segment_thermal" in msg


def test_band_outside_the_frame_is_refused_naming_the_frame_range(tmp_path):
    frame, _ = _scene()
    path = _write_csv(tmp_path, frame)
    with pytest.raises(ValueError) as exc:
        segment_thermal(path, min_c=40.0)
    msg = str(exc.value)
    assert "40.0" in msg and "31.5" in msg and "entirely" in msg
    assert "channel" not in msg  # not the RGB remedy
    with pytest.raises(ValueError, match="entirely"):
        segment_thermal(path, max_c=10.0)
    # Positive control: an in-range band that selects the disc is fine, no
    # range advisory.
    seg = segment_thermal(path, min_c=25.0)
    assert "threshold_outside_range" not in [w.code for w in seg.warnings]


def test_band_enclosing_the_whole_frame_gets_a_range_advisory(tmp_path):
    frame, _ = _scene()
    path = _write_csv(tmp_path, frame)
    seg = segment_thermal(path, min_c=10.0, max_c=40.0)
    codes = [w.code for w in seg.warnings]
    assert "threshold_outside_range" in codes
    msg = next(w.message for w in seg.warnings if w.code == "threshold_outside_range")
    assert "20 to 31.5" in msg and "every" in msg


def test_thermal_refusals_and_advisories_name_thermal_tools_not_rgb(tmp_path):
    from plantcv_mcp.diagnostics import DegenerateMaskError

    frame, _ = _scene()
    path = _write_csv(tmp_path, frame)
    # measure on an empty mask: the degenerate refusal must send the caller to
    # segment_thermal(), not to "a different channel or method".
    with pytest.raises(DegenerateMaskError) as exc:
        measure_thermal(path, np.zeros(frame.shape, np.uint8))
    assert "segment_thermal" in str(exc.value)
    assert "channel" not in str(exc.value)
    # a band inside the frame that selects nothing (a gap) is degenerate too
    frame2 = frame.copy()
    with pytest.raises(DegenerateMaskError) as exc2:
        segment_thermal(_write_csv(tmp_path, frame2, "f2.csv"), min_c=22.0, max_c=24.0)
    assert "segment_thermal" in str(exc2.value) and "channel" not in str(exc2.value)


# --- panel audit of 1.5.1 (2026-08-29) ---


def test_a_small_thermal_plant_erased_by_fill_size_is_told_the_fill_size(tmp_path):
    """assert_not_degenerate ran before segmentation_warnings, so the
    fill_erased_mask sentence ('fill_size below N') was unreachable and a
    150-px plant was blamed on the band."""
    from plantcv_mcp.diagnostics import DegenerateMaskError

    frame = np.full((120, 160), 20.0)
    yy, xx = np.ogrid[:120, :160]
    frame[(xx - 80) ** 2 + (yy - 60) ** 2 <= 7**2] = 31.5  # ~150 px
    path = _write_csv(tmp_path, frame)
    with pytest.raises(DegenerateMaskError) as exc:
        segment_thermal(path, min_c=25.0)
    msg = str(exc.value)
    assert "fill_size" in msg and "below" in msg
    assert "band" not in msg.split("fill_size")[0]  # the band is not blamed first
    # Positive control: a fill_size under the plant measures it.
    assert (
        segment_thermal(path, min_c=25.0, fill_size=50).diagnostics.component_count == 1
    )


def test_a_noisy_thermal_mask_is_advised_in_thermal_terms(tmp_path):
    frame = np.full((400, 400), 20.0)
    frame[180:220, 180:220] = 31.5
    rng = np.random.default_rng(6)
    placed = 0
    while placed < 60:
        y, x = rng.integers(0, 382, 2)
        if 150 <= y <= 240 and 150 <= x <= 240:
            continue
        if (frame[y : y + 18, x : x + 18] > 30).any():
            continue
        frame[y : y + 18, x : x + 18] = 31.5
        placed += 1
    seg = segment_thermal(_write_csv(tmp_path, frame), min_c=25.0)
    msg = next(w.message for w in seg.warnings if w.code == "noisy_segmentation")
    assert "segment_thermal" in msg and "band" in msg
    assert "colourspace" not in msg and "segment()" not in msg
