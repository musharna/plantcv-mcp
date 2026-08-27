"""Hyperspectral sessions: ENVI cubes, spectral indices, reflectance.

Two kinds of evidence. A synthetic float cube whose NDVI is KNOWN per pixel
(0.6 inside a disc, -0.2 outside), and PlantCV's real corn-kernel cube (uint16
counts, 580 bands). The guard this module exists for, measured on 4.11.3: an
index computed on integer counts wraps around silently — NDVI on the uint16
version of the synthetic cube reads 65.3 on the background, on a [-1, 1] index.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from plantcv import plantcv as pcv
from plantcv.plantcv import Spectral_data

from plantcv_mcp.hyperspectral import (
    CalibrationDegenerateError,
    IndexUnavailableError,
    load_cube,
    measure_spectral,
    segment_hyperspectral,
)
from plantcv_mcp.server import build_server

FIXTURES = Path(__file__).parent / "fixtures" / "plantcv"
CORN = str(FIXTURES / "corn-kernel-hyperspectral")

WL = [600.0, 670.0, 700.0, 800.0, 900.0]
H, W = 60, 80


def _disc():
    yy, xx = np.ogrid[:H, :W]
    return (xx - 40) ** 2 + (yy - 30) ** 2 <= 15**2


def _spectral(array, name="synthetic"):
    return Spectral_data(
        array_data=array,
        max_wavelength=max(WL),
        min_wavelength=min(WL),
        max_value=float(array.max()),
        min_value=float(array.min()),
        d_type=array.dtype.type,
        wavelength_dict={w: i for i, w in enumerate(WL)},
        samples=W,
        lines=array.shape[0],
        interleave="bil",
        wavelength_units="nm",
        array_type="datacube",
        pseudo_rgb=np.zeros((array.shape[0], W, 3), np.uint8),
        filename=name,
        default_bands=None,
    )


def _write_cube(tmp_path, name, array):
    """Write an ENVI pair; return the .raw path (PlantCV derives <name>.hdr)."""
    pcv.hyperspectral.write_data(
        filename=str(tmp_path / name), spectral_data=_spectral(array)
    )
    return str(tmp_path / f"{name}.raw")


def _known_cube(dtype=np.float32, scale=1.0):
    """NDVI = (r800 - r670)/(r800 + r670): disc 0.6, background -0.2."""
    disc = _disc()
    cube = np.zeros((H, W, len(WL)), np.float64)
    cube[..., 0] = 0.3
    cube[..., 1] = np.where(disc, 0.2, 0.6)
    cube[..., 2] = 0.5
    cube[..., 3] = np.where(disc, 0.8, 0.4)
    cube[..., 4] = 0.7
    return (cube * scale).astype(dtype)


def _refs(tmp_path):
    white = np.full((1, W, len(WL)), 1000, np.uint16)
    dark = np.zeros((1, W, len(WL)), np.uint16)
    return _write_cube(tmp_path, "white", white), _write_cube(tmp_path, "dark", dark)


# --- reading ---


def test_load_cube_accepts_raw_hdr_or_bare_name_and_hashes_both_files(tmp_path):
    raw = _write_cube(tmp_path, "synth", _known_cube())
    a = load_cube(raw)
    b = load_cube(raw[:-4] + ".hdr")  # the header path
    c = load_cube(raw[:-4])  # bare name, PlantCV's own convention
    assert a.cube.array_data.shape == (H, W, 5)
    assert a.digest == b.digest == c.digest
    # The digest covers the header too: change a wavelength, digest changes.
    hdr = Path(raw[:-4] + ".hdr")
    hdr.write_text(hdr.read_text().replace("900.0", "901.0"))
    assert load_cube(raw).digest != a.digest


def test_missing_header_is_named(tmp_path):
    raw = _write_cube(tmp_path, "synth", _known_cube())
    os.remove(raw[:-4] + ".hdr")
    with pytest.raises(FileNotFoundError, match="hdr"):
        load_cube(raw)


# --- the guard: integer counts are cast before any index ---


def test_index_on_integer_counts_is_cast_not_wrapped(tmp_path):
    """Positive control first: PlantCV's ndvi on the uint16 cube WITHOUT a cast
    wraps around (background 65.3). Through our path the same cube measures
    the true index, and says the cube was uncalibrated."""
    counts = _known_cube(np.uint16, scale=1000)
    raw_pcv = pcv.spectral_index.ndvi(hsi=_spectral(counts), distance=20)
    assert float(raw_pcv.array_data[~_disc()].mean()) > 10, "wraparound not reproduced"

    path = _write_cube(tmp_path, "counts", counts)
    seg = segment_hyperspectral(path, index="ndvi", threshold=0.2, object_type="light")
    assert seg.calibration == "none"
    assert "uncalibrated_cube" in [w.code for w in seg.warnings]
    res = measure_spectral(
        path, seg.mask, indices=["ndvi"], calibration=seg.calibration_args
    )
    assert res.indices["ndvi"]["mean"] == pytest.approx(0.6, abs=0.01)


def test_known_ndvi_is_recovered_on_the_float_cube(tmp_path):
    path = _write_cube(tmp_path, "synth", _known_cube())
    seg = segment_hyperspectral(path, index="ndvi", threshold=0.2, object_type="light")
    assert seg.diagnostics.mask_fraction == pytest.approx(_disc().mean(), abs=0.01)
    assert seg.index_range == pytest.approx((-0.2, 0.6), abs=1e-3)
    res = measure_spectral(
        path, seg.mask, indices=["ndvi", "savi"], calibration=seg.calibration_args
    )
    assert res.indices["ndvi"]["mean"] == pytest.approx(0.6, abs=0.01)
    assert res.indices["ndvi"]["std"] == pytest.approx(0.0, abs=0.01)
    assert res.band_count == 5 and res.wavelength_range == (600.0, 900.0)
    assert "spectrum" not in res.as_dict()  # opt-in, the large-result rule
    full = measure_spectral(
        path,
        seg.mask,
        indices=["ndvi"],
        calibration=seg.calibration_args,
        include_spectrum=True,
    )
    assert len(full.spectrum["wavelength_means"]) == 5
    # Positive control: the background is a different number.
    bg = measure_spectral(
        path,
        np.where(seg.mask > 0, 0, 255).astype(np.uint8),
        indices=["ndvi"],
        calibration=seg.calibration_args,
    )
    assert bg.indices["ndvi"]["mean"] == pytest.approx(-0.2, abs=0.01)


# --- calibration ---


def test_white_and_dark_references_calibrate_counts_to_reflectance(tmp_path):
    counts = _known_cube(np.uint16, scale=1000)
    path = _write_cube(tmp_path, "counts", counts)
    white, dark = _refs(tmp_path)
    seg = segment_hyperspectral(
        path, index="ndvi", threshold=0.2, white_reference=white, dark_reference=dark
    )
    assert seg.calibration == "white/dark"
    assert "uncalibrated_cube" not in [w.code for w in seg.warnings]
    res = measure_spectral(
        path,
        seg.mask,
        indices=["ndvi"],
        calibration=seg.calibration_args,
        include_spectrum=True,
    )
    assert res.indices["ndvi"]["mean"] == pytest.approx(0.6, abs=0.01)
    # Reflectance, not counts: the 800 nm mean inside the disc is 0.8.
    means = dict(
        zip(res.spectrum["wavelengths"], res.spectrum["wavelength_means"], strict=True)
    )
    assert means[800.0] == pytest.approx(0.8, abs=0.01)


def test_degenerate_references_are_refused_not_clipped(tmp_path):
    """Measured: PlantCV's calibrate() with white == dark clips to 1.0 instead of
    producing NaN, so a bad reference pair would silently yield a flat cube."""
    path = _write_cube(tmp_path, "counts", _known_cube(np.uint16, scale=1000))
    _, dark = _refs(tmp_path)
    with pytest.raises(CalibrationDegenerateError, match="white"):
        segment_hyperspectral(
            path, index="ndvi", threshold=0.2, white_reference=dark, dark_reference=dark
        )
    # Positive control: a proper pair calibrates.
    white, dark = _refs(tmp_path)
    assert (
        segment_hyperspectral(
            path,
            index="ndvi",
            threshold=0.2,
            white_reference=white,
            dark_reference=dark,
        ).calibration
        == "white/dark"
    )


# --- index availability ---


def test_index_outside_the_wavelength_range_is_refused_by_name(tmp_path):
    path = _write_cube(tmp_path, "synth", _known_cube())
    with pytest.raises(IndexUnavailableError, match="wi"):  # water index needs 900/970
        segment_hyperspectral(path, index="wi", threshold=0.2)
    with pytest.raises(IndexUnavailableError, match="Valid"):
        segment_hyperspectral(path, index="not_an_index", threshold=0.2)


# --- real data: PlantCV's corn kernel cube ---


def test_real_corn_cube_segments_and_measures_uncalibrated():
    """Thresholds measured on the cube itself: uncalibrated NDVI on the corn
    kernel is mostly NEGATIVE (counts, and a kernel is not green vegetation) —
    5th/95th percentiles -0.171/+0.066 — and the whole frame is 1333 px, so the
    default fill_size=200 would erase everything."""
    seg = segment_hyperspectral(
        CORN, index="ndvi", threshold=-0.1, object_type="light", fill_size=5
    )
    assert seg.calibration == "none"
    assert "uncalibrated_cube" in [w.code for w in seg.warnings]
    assert -1.0 <= seg.index_range[0] < seg.index_range[1] <= 1.0, seg.index_range
    assert 0.05 < seg.diagnostics.mask_fraction < 0.95
    assert seg.overlay.shape == (31, 43, 3)
    res = measure_spectral(
        CORN, seg.mask, indices=["ndvi"], calibration=seg.calibration_args
    )
    assert res.band_count == 580
    assert -1.0 <= res.indices["ndvi"]["mean"] <= 1.0


# --- MCP layer ---


@pytest.mark.anyio
async def test_hyperspectral_tools_over_the_real_mcp_layer(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    path = _write_cube(tmp_path, "synth", _known_cube())
    mcp = build_server()
    result = await mcp.call_tool(
        "segment_hyperspectral",
        {"envi_path": path, "index": "ndvi", "threshold": 0.2, "object_type": "light"},
    )
    text_block, image_block = result.content
    seg = json.loads(text_block.text)
    assert image_block.type == "image"
    assert seg["kind"] == "hsi" and seg["calibration"] == "none"
    assert "traits" not in seg

    measured = await mcp.call_tool(
        "measure_spectral", {"session_id": seg["session_id"], "indices": ["ndvi"]}
    )
    out = measured.structured_content
    assert out["indices"]["ndvi"]["mean"] == pytest.approx(0.6, abs=0.01)
    assert out["engine"]["name"] == "PlantCV"
    assert out["band_count"] == 5

    # Typed sessions: an RGB tool refuses an HSI session by name, and vice versa.
    with pytest.raises(ToolError, match="measure_spectral"):
        await mcp.call_tool("measure", {"session_id": seg["session_id"]})
