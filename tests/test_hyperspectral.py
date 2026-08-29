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


def test_nan_references_are_refused_naming_nan(tmp_path):
    """A NaN in a reference correctly trips the span check (NaN > 0 is False),
    but the message must say the reference is not finite, not just 'not
    positive' — the user's fix is different."""
    white = np.full((1, W, len(WL)), 1000.0, np.float64)
    white[0, 3, 2] = np.nan
    dark = np.zeros((1, W, len(WL)), np.float64)
    wp = _write_cube(tmp_path, "white_nan", white)
    dp = _write_cube(tmp_path, "dark_zero", dark)
    path = _write_cube(tmp_path, "counts", _known_cube(np.uint16, scale=1000))
    with pytest.raises(CalibrationDegenerateError, match="not finite"):
        segment_hyperspectral(path, white_reference=wp, dark_reference=dp)


def test_span_check_is_plantcvs_own_denominator_per_column(tmp_path):
    """The check must match what PlantCV's calibrate() actually divides by:
    mean(white, axis=0) - mean(dark, axis=0), per column x band — NOT per
    pixel. A dead detector column refuses; a single dead pixel whose column
    mean stays positive does not."""
    dark = np.zeros((4, W, len(WL)), np.float64)
    dp = _write_cube(tmp_path, "dark4", dark)
    path = _write_cube(tmp_path, "counts", _known_cube(np.uint16, scale=1000))

    white = np.full((4, W, len(WL)), 1000.0, np.float64)
    white[:, 7, :] = 0.0  # dead column: its mean equals dark's
    with pytest.raises(CalibrationDegenerateError, match="not positive"):
        segment_hyperspectral(
            path,
            white_reference=_write_cube(tmp_path, "white_deadcol", white),
            dark_reference=dp,
        )

    # Positive control, and the discriminator: one dead PIXEL leaves the
    # column mean at 750 > 0, so PlantCV's denominator is fine and a
    # per-pixel check would be wrong to refuse.
    white = np.full((4, W, len(WL)), 1000.0, np.float64)
    white[0, 5, 1] = 0.0
    seg = segment_hyperspectral(
        path,
        white_reference=_write_cube(tmp_path, "white_deadpx", white),
        dark_reference=dp,
    )
    assert seg.calibration == "white/dark"


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


def test_a_symlinked_raw_sibling_cannot_escape_the_read_roots(tmp_path):
    """_resolve_pair derives the .raw sibling of a .hdr the caller named; the
    sibling's bytes must pass containment too, or a symlink smuggles them out."""
    from plantcv_mcp import paths
    from plantcv_mcp.paths import PathOutsideRootsError

    allowed = tmp_path / "allowed"
    other = tmp_path / "other"
    allowed.mkdir()
    other.mkdir()
    out_raw = _write_cube(other, "cube", _known_cube())
    in_raw = _write_cube(allowed, "cube", _known_cube())
    # The .hdr the caller names is inside the root; the raw bytes are not.
    os.remove(in_raw)
    os.symlink(out_raw, in_raw)
    paths.set_roots([str(allowed)])
    try:
        with pytest.raises(PathOutsideRootsError):
            load_cube(str(allowed / "cube.hdr"))
        # Positive control: a fully in-root pair loads under the same roots.
        legit = _write_cube(allowed, "legit", _known_cube())
        assert load_cube(legit).cube.array_data.shape == (H, W, 5)
    finally:
        paths.set_roots(None)


def test_a_swapped_calibration_reference_is_refused_at_measure(tmp_path):
    """The refs are part of the measurement's identity: every calibrated number
    depends on their bytes. They are digest-pinned at segmentation like the cube
    itself, so a swap between segment and measure is a refusal, not a silently
    different measurement."""
    from plantcv_mcp.hyperspectral import CalibrationReferencesChangedError

    path = _write_cube(tmp_path, "counts", _known_cube(np.uint16, scale=1000))
    white, dark = _refs(tmp_path)
    seg = segment_hyperspectral(
        path, index="ndvi", threshold=0.2, white_reference=white, dark_reference=dark
    )
    # Positive control: with the refs untouched, measurement runs.
    res = measure_spectral(
        path, seg.mask, indices=["ndvi"], calibration=seg.calibration_args
    )
    assert res.indices["ndvi"]["mean"] == pytest.approx(0.6, abs=0.01)
    # Swap the white reference for a plausible but different file.
    _write_cube(tmp_path, "white", np.full((1, W, len(WL)), 900, np.uint16))
    with pytest.raises(CalibrationReferencesChangedError, match="white"):
        measure_spectral(
            path, seg.mask, indices=["ndvi"], calibration=seg.calibration_args
        )


def test_the_server_pins_calibration_refs_in_the_session(tmp_path):
    """Same property through the server layer: the session must carry the ref
    digests, and measure_spectral() must check them."""
    from plantcv_mcp.hyperspectral import CalibrationReferencesChangedError
    from plantcv_mcp.server import _measure_spectral_impl, _segment_hsi_impl

    path = _write_cube(tmp_path, "counts", _known_cube(np.uint16, scale=1000))
    white, dark = _refs(tmp_path)
    seg = _segment_hsi_impl(path, white_reference=white, dark_reference=dark)
    assert seg["calibration"] == "white/dark"
    # Positive control first.
    res = _measure_spectral_impl(seg["session_id"], indices=["ndvi"])
    assert res["indices"]["ndvi"]["mean"] == pytest.approx(0.6, abs=0.01)
    _write_cube(tmp_path, "dark", np.full((1, W, len(WL)), 7, np.uint16))
    with pytest.raises(CalibrationReferencesChangedError, match="dark"):
        _measure_spectral_impl(seg["session_id"], indices=["ndvi"])


def test_nonfinite_index_values_are_reported_not_silently_dropped(tmp_path):
    """min/max already skip NaN pixels quietly while pixel_count claims the
    whole mask. The dropped evidence must be named: a nan_pixels advisory and a
    finite_pixel_count per index."""
    cube = _known_cube()
    cube[28:32, 38:42, 1] = np.nan  # 16 in-disc pixels lose the 670 nm band
    path = _write_cube(tmp_path, "nan_cube", cube)
    mask = np.where(_disc(), 255, 0).astype(np.uint8)

    res = measure_spectral(path, mask, indices=["ndvi"])
    assert "nan_pixels" in [w.code for w in res.warnings]
    assert res.pixel_count == int(_disc().sum())
    assert res.indices["ndvi"]["finite_pixel_count"] == int(_disc().sum()) - 16

    # Positive control: a clean cube reports full finite evidence, no advisory.
    clean = measure_spectral(
        _write_cube(tmp_path, "clean_cube", _known_cube()), mask, indices=["ndvi"]
    )
    assert "nan_pixels" not in [w.code for w in clean.warnings]
    assert clean.indices["ndvi"]["finite_pixel_count"] == clean.pixel_count


# --- modality dogfood 2026-08-28: remedies and threshold-vs-index-range ---


def test_threshold_above_the_index_maximum_is_named(tmp_path):
    """On a real leaf cube (NDVI 0.19-0.89) threshold=0.95 returned an empty
    mask blaming 'channel or method'; threshold=0.2 (below the minimum)
    selected 100% with only the inverted-mask advisory."""
    path = _write_cube(tmp_path, "synth", _known_cube())
    seg = segment_hyperspectral(path, index="ndvi", threshold=0.95, object_type="light")
    codes = [w.code for w in seg.warnings]
    assert "threshold_outside_range" in codes and "empty_mask" in codes
    rng = next(w.message for w in seg.warnings if w.code == "threshold_outside_range")
    assert "0.95" in rng and "0.6" in rng and "nothing" in rng
    empty = next(w.message for w in seg.warnings if w.code == "empty_mask")
    assert "segment_hyperspectral" in empty and "threshold" in empty
    assert "channel or method" not in empty

    low = segment_hyperspectral(path, index="ndvi", threshold=-0.5, object_type="light")
    codes = [w.code for w in low.warnings]
    assert "threshold_outside_range" in codes and "implausible_coverage" in codes
    rng = next(w.message for w in low.warnings if w.code == "threshold_outside_range")
    assert "-0.5" in rng and "-0.2" in rng and "every" in rng
    cov = next(w.message for w in low.warnings if w.code == "implausible_coverage")
    assert "segment_hyperspectral" in cov and "segment()" not in cov

    # Positive control: a threshold inside the range carries no range advisory,
    # and dark polarity flips which side is "nothing".
    ok = segment_hyperspectral(path, index="ndvi", threshold=0.2, object_type="light")
    assert "threshold_outside_range" not in [w.code for w in ok.warnings]
    dark_none = segment_hyperspectral(
        path, index="ndvi", threshold=-0.5, object_type="dark"
    )
    rng = next(
        w.message for w in dark_none.warnings if w.code == "threshold_outside_range"
    )
    assert "nothing" in rng


def test_spectral_degenerate_refusal_names_segment_hyperspectral(tmp_path):
    from plantcv_mcp.diagnostics import DegenerateMaskError

    path = _write_cube(tmp_path, "synth", _known_cube())
    with pytest.raises(DegenerateMaskError) as exc:
        measure_spectral(path, np.zeros((H, W), np.uint8), indices=["ndvi"])
    assert "segment_hyperspectral" in str(exc.value)
    assert "channel" not in str(exc.value)
