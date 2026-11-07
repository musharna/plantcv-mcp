"""thermal.py at its edges: the band's exact ends, non-finite pixels, every
refusal's full text, and the small helpers (mutation burn-down, #97).

Each test pins behavior a surviving mutant changed without any test noticing
(nightly 2026-11-06: 148 survivors in thermal.py). The scene is the one
tests/test_thermal.py uses: 20.0 C background, a 31.5 C disc of radius 25.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from plantcv_mcp.diagnostics import DegenerateMaskError
from plantcv_mcp.thermal import (
    _undecodable,
    _value,
    grey_frame,
    load_thermal,
    measure_thermal,
    segment_thermal,
)

FIXTURES = Path(__file__).parent / "fixtures" / "plantcv"
FLIR = FIXTURES / "FLIR_test.jpg"


def _scene():
    frame = np.full((120, 160), 20.0, np.float64)
    yy, xx = np.ogrid[:120, :160]
    disc = (xx - 80) ** 2 + (yy - 60) ** 2 <= 25**2
    frame[disc] = 31.5
    return frame, disc


def _csv(tmp_path, frame, name="frame.csv"):
    p = tmp_path / name
    np.savetxt(p, frame, delimiter=",", fmt="%.4f")
    return str(p)


def _npz(tmp_path, name="frame.npz", **arrays):
    p = tmp_path / name
    np.savez(p, **arrays)
    return str(p)


def _selected(seg):
    return seg.mask > 0


# --- the band at the frame's exact ends ---


def test_a_band_starting_or_ending_exactly_at_the_frame_range_selects_those_pixels(
    tmp_path,
):
    """The band is inclusive (c >= min_c, c <= max_c), so a band that touches
    the frame's maximum or minimum selects the pixels AT it; it is neither
    "entirely outside" the frame nor empty."""
    frame, disc = _scene()
    path = _csv(tmp_path, frame)
    assert (_selected(segment_thermal(path, min_c=31.5)) == disc).all()
    assert (_selected(segment_thermal(path, max_c=20.0)) == ~disc).all()
    # Control: just past either end is refused as outside the frame.
    with pytest.raises(ValueError, match="entirely outside"):
        segment_thermal(path, min_c=31.6)
    with pytest.raises(ValueError, match="entirely outside"):
        segment_thermal(path, max_c=19.9)


def test_the_whole_frame_advisory_agrees_with_the_mask_at_the_exact_ends(tmp_path):
    frame, _ = _scene()
    path = _csv(tmp_path, frame)
    expected = (
        "The band 20.0-31.5 C selects every pixel: the data span 20 to 31.5 C, so "
        "no pixel is on the other side of the rule. The mask says nothing about "
        "the plant. Choose a value inside that range."
    )
    for band in ({"min_c": 20.0, "max_c": 31.5}, {"min_c": 20.0}, {"max_c": 31.5}):
        seg = segment_thermal(path, **band)
        assert _selected(seg).all(), band  # the mask really is the whole frame
        # First, ahead of the coverage advisory the same mask also raises.
        assert [w.code for w in seg.warnings] == [
            "threshold_outside_range",
            "implausible_coverage",
        ], band
        assert seg.warnings[0].message == expected, band
    # Control: a band one step inside the top end does not select everything,
    # and says nothing about the range.
    seg = segment_thermal(path, min_c=20.0, max_c=31.4)
    assert not _selected(seg).all()
    assert "threshold_outside_range" not in [w.code for w in seg.warnings]


def test_an_infinite_pixel_is_never_selected_by_a_band(tmp_path):
    """A +Inf pixel passes `c >= min_c`; only the finiteness mask keeps it out."""
    frame, disc = _scene()
    frame[:20, :20] = np.inf  # 400 px, away from the disc
    seg = segment_thermal(_npz(tmp_path, a=frame), min_c=25.0)
    assert (_selected(seg) == disc).all()
    nan = [w.message for w in seg.warnings if w.code == "nan_pixels"]
    assert nan == [
        (
            "400 of 19200 pixels are not finite (NaN/Inf) and could never be "
            "selected by the temperature band. If they cluster on the plant, the "
            "mask has a hole the diagnostics cannot see; check the sensor export."
        )
    ]


def test_an_empty_band_is_refused_and_a_one_step_band_is_not(tmp_path):
    frame, disc = _scene()
    path = _csv(tmp_path, frame)
    with pytest.raises(ValueError) as exc:
        segment_thermal(path, min_c=31.5, max_c=31.5)
    assert str(exc.value) == "min_c must be below max_c, got 31.5 >= 31.5"
    assert (_selected(segment_thermal(path, min_c=31.5, max_c=31.6)) == disc).all()


def test_the_result_records_the_band_and_the_source(tmp_path):
    frame, _ = _scene()
    seg = segment_thermal(_csv(tmp_path, frame), min_c=25.0, max_c=40.0)
    assert (seg.min_c, seg.max_c, seg.source) == (25.0, 40.0, "csv")
    seg = segment_thermal(_csv(tmp_path, frame, "b.csv"), max_c=25.0)
    assert (seg.min_c, seg.max_c) == (None, 25.0)


def test_the_default_fill_size_keeps_a_201_px_plant_and_drops_a_200_px_one(tmp_path):
    """PlantCV's fill removes objects of area <= size (skimage max_size)."""
    frame = np.full((60, 60), 20.0)
    frame[10:20, 10:30] = 31.5  # 200 px
    with pytest.raises(DegenerateMaskError) as exc:
        segment_thermal(_csv(tmp_path, frame), min_c=25.0)
    assert str(exc.value) == (
        "Thresholding found 1 object(s), the largest 200 px, and then "
        "fill_size=200 removed every one of them. This is a fill_size problem, "
        "not a channel or method problem — the specimen is smaller than the "
        "speckle filter. Re-run with fill_size below 200."
    )
    frame[20, 10] = 31.5  # 201 px
    seg = segment_thermal(_csv(tmp_path, frame, "b.csv"), min_c=25.0)
    assert int(_selected(seg).sum()) == 201


# --- refusals, full text ---


def test_a_frame_with_no_finite_temperature_is_refused_by_name(tmp_path):
    path = _npz(tmp_path, a=np.full((30, 30), np.nan))
    with pytest.raises(ValueError) as exc:
        segment_thermal(path, min_c=1.0)
    assert str(exc.value) == f"No finite temperatures in {path!r}; nothing to segment."
    # Control: one finite pixel is enough to get past this refusal.
    frame = np.full((30, 30), np.nan)
    frame[0, 0] = 5.0
    with pytest.raises(ValueError, match="entirely outside"):
        segment_thermal(_npz(tmp_path, "one.npz", a=frame), min_c=10.0)


def test_the_band_less_refusal_quotes_the_frame_range_and_its_percentiles(tmp_path):
    path = _npz(tmp_path, a=np.arange(101, dtype=np.float64).reshape(1, 101))
    with pytest.raises(ValueError) as exc:
        segment_thermal(path)
    assert str(exc.value) == (
        "Give min_c and/or max_c: the band of temperatures that is the plant. "
        "This frame spans 0.0 to 100.0 C (p5=5.0, p25=25.0, p50=50.0, p75=75.0, "
        "p95=95.0). Re-run segment_thermal() with a band inside that range, e.g. "
        "max_c near the cool tail for a transpiring plant on a warmer background."
    )


def test_a_band_outside_the_frame_is_refused_in_full(tmp_path):
    frame, _ = _scene()
    with pytest.raises(ValueError) as exc:
        segment_thermal(_csv(tmp_path, frame), min_c=40.0)
    assert str(exc.value) == (
        "The band 40.0-31.5 C lies entirely outside this frame's 20.0-31.5 C, so "
        "it can select nothing. Re-run segment_thermal() with a band inside the "
        "frame range."
    )


# --- loading ---


def test_jpeg_spellings_load_as_flir(tmp_path):
    for name in ("frame.jpeg", "FRAME.JPG", "frame.JPEG"):
        copy = tmp_path / name
        shutil.copyfile(FLIR, copy)
        assert load_thermal(str(copy)).source == "flir", name
    assert load_thermal(_csv(tmp_path, _scene()[0])).source == "csv"


def test_decoding_a_flir_jpeg_leaves_nothing_in_the_working_directory(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    assert load_thermal(str(FLIR.resolve())).celsius.shape == (480, 640)
    assert list(tmp_path.iterdir()) == []


def test_undecodable_jpeg_and_csv_are_refused_naming_the_file_and_the_format(
    tmp_path,
):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(ValueError) as exc:
        load_thermal(str(bad))
    assert str(exc.value) == (
        f"Could not decode {str(bad)!r} as a FLIR radiometric JPEG: "
        "ValueError: Invalid FLIR: no metadata encountered"
    )
    assert isinstance(exc.value.__cause__, ValueError)

    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"1,2\nx,y\n")
    with pytest.raises(ValueError) as exc:
        load_thermal(str(bad))
    assert str(exc.value) == (
        f"Could not decode {str(bad)!r} as a .csv of Celsius values: "
        "ValueError: could not convert string 'x' to float64 at row 1, column 1."
    )
    assert isinstance(exc.value.__cause__, ValueError)


def test_an_unsupported_file_and_a_1d_frame_are_refused_in_full(tmp_path):
    tif = tmp_path / "frame.tif"
    tif.write_bytes(b"x")
    with pytest.raises(ValueError) as exc:
        load_thermal(str(tif))
    assert str(exc.value) == (
        f"Unsupported thermal file {str(tif)!r}: expected a FLIR radiometric .jpg, "
        "a .csv of temperatures, or a .npz array."
    )
    row = tmp_path / "row.csv"
    row.write_bytes(b"1,2,3\n")
    with pytest.raises(ValueError) as exc:
        load_thermal(str(row))
    assert str(exc.value) == "A thermal frame must be 2-D degrees Celsius; got (3,)"


def test_npz_refusals_in_full(tmp_path):
    empty = _npz(tmp_path, "none.npz")
    with pytest.raises(ValueError) as exc:
        load_thermal(empty)
    assert str(exc.value) == f"{empty!r} holds no arrays"

    two = _npz(tmp_path, "two.npz", celsius=np.zeros((2, 2)), kelvin=np.ones((2, 2)))
    with pytest.raises(ValueError) as exc:
        load_thermal(two)
    assert str(exc.value) == (
        f"{two!r} holds 2 arrays (['celsius', 'kelvin']); a thermal .npz must hold "
        "exactly one 2-D Celsius array. Re-save just the temperature frame."
    )

    zero = tmp_path / "zero.npz"
    zero.write_bytes(b"")
    with pytest.raises(ValueError) as exc:
        load_thermal(str(zero))
    assert str(exc.value) == (
        f"Could not decode {str(zero)!r} as a .npz holding one 2-D Celsius array: "
        "EOFError: No data left in file"
    )

    obj = _npz(tmp_path, "obj.npz", frame=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError) as exc:
        load_thermal(obj)
    assert str(exc.value) == (
        f"Could not decode {obj!r} as a .npz whose array 'frame' is a numeric "
        "frame: ValueError: Object arrays cannot be loaded when allow_pickle=False"
    )
    assert isinstance(exc.value.__cause__, ValueError)


def test_an_integer_npz_loads_as_float_celsius(tmp_path):
    frame = np.arange(16, dtype=np.int16).reshape(4, 4)
    load = load_thermal(_npz(tmp_path, a=frame))
    assert load.celsius.dtype == np.float64
    assert (load.celsius == frame).all()


def test_undecodable_keeps_the_first_line_of_the_reason_up_to_120_characters():
    long = _undecodable("f.npz", "x", ValueError("a" * 200))
    assert str(long) == f"Could not decode 'f.npz' as x: ValueError: {'a' * 120}"
    lines = _undecodable("f.npz", "x", OSError("first\nsecond"))
    assert str(lines) == "Could not decode 'f.npz' as x: OSError: first"
    empty = _undecodable("f.npz", "x", EOFError())
    assert str(empty) == "Could not decode 'f.npz' as x: EOFError: "


# --- the overlay frame ---


def test_grey_frame_scales_the_finite_range_and_clips_non_finite_pixels():
    frame = np.array([[0.0, 5.0, 10.0, np.inf], [-np.inf, np.nan, 10.0, 0.0]])
    grey = grey_frame(frame)
    assert grey.shape == (2, 4, 3)
    assert grey[..., 0].tolist() == [[0, 127, 255, 255], [0, 0, 255, 0]]
    assert (grey[..., 0] == grey[..., 1]).all() and (grey[..., 1] == grey[..., 2]).all()
    # A flat frame has no range to scale over: black, not an error.
    assert grey_frame(np.full((2, 3), 7.0)).tolist() == [[[0, 0, 0]] * 3] * 2
    with pytest.raises(ValueError) as exc:
        grey_frame(np.full((2, 2), np.nan))
    assert str(exc.value) == "No finite temperature values; nothing to segment."


# --- measuring ---


def test_measure_reads_a_0_1_mask_like_a_0_255_mask(tmp_path):
    frame, disc = _scene()
    path = _csv(tmp_path, frame)
    full = measure_thermal(path, np.where(disc, 255, 0).astype(np.uint8))
    ones = measure_thermal(path, disc.astype(np.uint8))
    assert ones.pixel_count == full.pixel_count == int(disc.sum())
    assert ones.temperature == full.temperature


def test_measure_reports_every_statistic_and_a_100_bin_histogram(tmp_path):
    frame, disc = _scene()
    frame[disc] = np.linspace(30.0, 33.0, int(disc.sum()))
    path = _npz(tmp_path, a=frame)
    res = measure_thermal(path, np.where(disc, 255, 0).astype(np.uint8), True)
    t = res.temperature
    assert set(t) == {"max", "min", "mean", "median"}
    assert t["min"] == pytest.approx(30.0) and t["max"] == pytest.approx(33.0)
    assert t["mean"] == pytest.approx(31.5) and t["median"] == pytest.approx(31.5)
    assert len(res.histogram["bins"]) == len(res.histogram["counts"]) == 100
    out = res.as_dict()
    assert out["histogram"] == res.histogram
    assert out["temperature"] == {**t, "unit": "celsius"}
    assert out["frame_range"] == [20.0, 33.0]


def test_a_mask_of_the_wrong_shape_is_refused_in_full(tmp_path):
    frame, _ = _scene()
    mask = np.full((10, 10), 255, np.uint8)
    with pytest.raises(ValueError) as exc:
        measure_thermal(_csv(tmp_path, frame), mask)
    assert str(exc.value) == "mask (10, 10) does not match the frame (120, 160)"


def test_value_reads_only_this_modules_label():
    obs = {
        "other": {"max_temp": {"value": 1.0}},
        "thermal_1": {"max_temp": {"value": 2.0}},
    }
    assert _value(obs, "max_temp") == {"value": 2.0}
    assert _value({"thermal": {"max_temp": 3}}, "max_temp") == 3
    assert _value({"thermal_1": {"min_temp": 1}}, "max_temp") is None
