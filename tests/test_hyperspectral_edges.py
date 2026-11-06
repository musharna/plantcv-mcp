"""Hyperspectral at its edges: the boundaries of each check, the result fields
nothing else reads, and the cubes the usual scenes never write.

Written from the mutants of hyperspectral.py that survived the nightly mutation
run (#97): each test here fails on at least one of them. Mutants left alive on
purpose are listed in the PR that added this file. The fixtures are
test_hyperspectral's: a cube whose NDVI is 0.6 inside a disc, -0.2 outside.
"""

import hashlib
import os
import tempfile

import numpy as np
import pytest
from plantcv import plantcv as pcv
from plantcv.plantcv import Spectral_data

from plantcv_mcp import hyperspectral as hs
from plantcv_mcp.diagnostics import Advisory
from plantcv_mcp.hyperspectral import (
    CalibrationDegenerateError,
    CalibrationReferencesChangedError,
    CubeLoad,
    IndexUnavailableError,
    SpectralResult,
    available_indices,
    check_reference_digests,
    compute_index,
    load_cube,
    measure_spectral,
    prepare_cube,
    segment_hyperspectral,
)

WL = [600.0, 670.0, 700.0, 800.0, 900.0]
H, W = 60, 80


def _disc():
    yy, xx = np.ogrid[:H, :W]
    return (xx - 40) ** 2 + (yy - 30) ** 2 <= 15**2


def _write_cube(tmp_path, name, array):
    """Write an ENVI pair; return the .raw path (PlantCV derives <name>.hdr)."""
    pcv.hyperspectral.write_data(
        filename=str(tmp_path / name),
        spectral_data=Spectral_data(
            array_data=array,
            max_wavelength=max(WL),
            min_wavelength=min(WL),
            max_value=float(np.nanmax(array)),
            min_value=float(np.nanmin(array)),
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
        ),
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


FULL = np.full((H, W), 255, np.uint8)
DISC = np.where(_disc(), 255, 0).astype(np.uint8)


def _ref(tmp_path, name, value, rows=1, dtype=np.float64):
    return load_cube(
        _write_cube(tmp_path, name, np.full((rows, W, len(WL)), value, dtype))
    )


# --- discovery and reading ---


def test_available_indices_are_the_public_index_functions_only():
    names = available_indices()
    assert "ndvi" in names and "savi" in names
    assert not [n for n in names if n.startswith("_")]


def test_a_raw_file_with_its_own_extension_loads_beside_its_header(tmp_path):
    """ENVI raws are often .img or .dat; the path the caller names is the raw."""
    raw = _write_cube(tmp_path, "cube", _known_cube())
    img = str(tmp_path / "cube.img")
    os.rename(raw, img)
    load = load_cube(img)
    assert load.raw_path == img and load.hdr_path == str(tmp_path / "cube.hdr")
    assert load.cube.array_data.shape == (H, W, len(WL))


def test_a_bare_name_with_a_dot_finds_name_raw_and_name_hdr(tmp_path):
    """'plot.v1' names plot.v1.raw + plot.v1.hdr, not plot.raw + plot.hdr."""
    raw = _write_cube(tmp_path, "a", _known_cube())
    os.rename(raw, str(tmp_path / "plot.v1.raw"))
    os.rename(raw[:-4] + ".hdr", str(tmp_path / "plot.v1.hdr"))
    load = load_cube(str(tmp_path / "plot.v1"))
    assert load.raw_path == str(tmp_path / "plot.v1.raw")
    assert load.hdr_path == str(tmp_path / "plot.v1.hdr")


def test_a_missing_raw_is_named_not_a_stopiteration(tmp_path):
    raw = _write_cube(tmp_path, "cube", _known_cube())
    os.remove(raw)
    hdr = raw[:-4] + ".hdr"
    with pytest.raises(FileNotFoundError) as info:
        load_cube(hdr)
    assert str(info.value) == (
        f"An ENVI cube needs a raw file and its .hdr header side by side; for "
        f"{hdr!r} found raw=None, hdr={hdr!r}. PlantCV derives the header "
        "name by stripping the raw file's extension."
    )


def test_an_uppercase_header_is_never_read_as_the_pixels(tmp_path):
    """cube.HDR is a header whatever its case: naming it must not make it the
    raw. (The lookup wants a lowercase .hdr, so the pair is refused, naming the
    real raw.)"""
    raw = _write_cube(tmp_path, "cube", _known_cube())
    upper = raw[:-4] + ".HDR"
    os.rename(raw[:-4] + ".hdr", upper)
    with pytest.raises(FileNotFoundError) as info:
        load_cube(upper)
    assert f"raw={raw!r}, hdr=None" in str(info.value)


def test_load_cube_records_its_paths_and_digest_and_leaves_no_copy(
    tmp_path, monkeypatch
):
    raw = _write_cube(tmp_path, "cube", _known_cube())
    hdr = raw[:-4] + ".hdr"
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    load = load_cube(raw)
    assert load.raw_path == raw and load.hdr_path == hdr
    assert load.cube.filename == raw  # the private copy's name never leaks out
    with open(hdr, "rb") as fh:
        hdr_bytes = fh.read()
    with open(raw, "rb") as fh:
        raw_bytes = fh.read()
    assert load.digest == hashlib.sha256(hdr_bytes + b"\0" + raw_bytes).hexdigest()
    assert os.listdir(scratch) == []  # the decoded copy is gone


def test_a_decode_failure_quotes_the_first_line_of_the_reason(tmp_path, monkeypatch):
    hdr = tmp_path / "nowl.hdr"
    hdr.write_bytes(
        b"ENVI\nsamples = 2\nlines = 2\nbands = 1\nheader offset = 0\n"
        b"data type = 1\ninterleave = bsq\nbyte order = 0\n"
    )
    raw = tmp_path / "nowl.raw"
    raw.write_bytes(b"\x00\x00\x00\x00")
    tail = (
        ". The header must define samples, lines, bands, data type, "
        "interleave, byte order and wavelength, and the .raw must hold exactly "
        "samples*lines*bands values of that data type."
    )
    head = f"Could not decode {str(raw)!r} + {str(hdr)!r} as an ENVI cube: "
    with pytest.raises(ValueError) as info:
        load_cube(str(raw))
    assert str(info.value) == head + "KeyError: 'wavelength'" + tail

    def long_reason(*_a, **_k):
        raise ValueError("x" * 200 + "\nsecond line")

    monkeypatch.setattr(pcv, "readimage", long_reason)
    with pytest.raises(ValueError) as info:
        load_cube(str(raw))
    assert str(info.value) == head + "ValueError: " + "x" * 120 + tail

    def no_reason(*_a, **_k):
        raise ValueError()

    monkeypatch.setattr(pcv, "readimage", no_reason)
    with pytest.raises(ValueError) as info:
        load_cube(str(raw))
    assert str(info.value) == head + "ValueError: " + tail


# --- preparing the cube ---


def test_an_uncalibrated_count_cube_is_cast_keeping_every_field(tmp_path):
    counts = _known_cube(np.uint16, scale=1000)
    pseudo = np.full((H, W, 3), 7, np.uint8)
    cube = Spectral_data(
        array_data=counts,
        max_wavelength=900.0,
        min_wavelength=600.0,
        max_value=800,
        min_value=200,
        d_type=np.uint16,
        wavelength_dict={w: i for i, w in enumerate(WL)},
        samples=W,
        lines=H,
        interleave="bsq",
        wavelength_units="nm",
        array_type="datacube",
        pseudo_rgb=pseudo,
        filename="plot.raw",
        default_bands=[1, 2, 3],
        metadata={"sensor": "x"},
    )
    prepared, label, warnings = prepare_cube(cube, None, None)
    assert label == "none"
    assert prepared.array_data.dtype == np.float64
    np.testing.assert_array_equal(prepared.array_data, counts)
    assert prepared.d_type is np.float64
    assert (prepared.max_value, prepared.min_value) == (800.0, 200.0)
    assert (prepared.samples, prepared.lines) == (W, H)
    assert (prepared.interleave, prepared.wavelength_units) == ("bsq", "nm")
    assert prepared.array_type == "datacube"
    assert prepared.pseudo_rgb is pseudo
    assert prepared.filename == "plot.raw"
    assert prepared.default_bands == [1, 2, 3]
    assert prepared.metadata == {"sensor": "x"}
    assert [(w.code, w.message) for w in warnings] == [
        (
            "uncalibrated_cube",
            (
                "The cube holds integer counts (uint16, max 800), not reflectance. "
                "Indices are computed on a float cast of the counts, so they are "
                "RELATIVE (comparable within this cube, not across sensors or "
                "sessions). Pass white_reference and dark_reference to calibrate."
            ),
        )
    ]


def test_a_float_cube_without_references_passes_through_untouched(tmp_path):
    load = load_cube(_write_cube(tmp_path, "float", _known_cube()))
    prepared, label, warnings = prepare_cube(load.cube, None, None)
    assert prepared is load.cube and label == "none" and warnings == []


def test_one_reference_alone_is_refused(tmp_path):
    white = _ref(tmp_path, "white", 1000.0)
    cube = load_cube(_write_cube(tmp_path, "c", _known_cube(np.uint16, 1000))).cube
    message = (
        "white_reference and dark_reference must be given together; a "
        "single reference cannot calibrate."
    )
    for pair in ((white, None), (None, white)):
        with pytest.raises(ValueError) as info:
            prepare_cube(cube, *pair)
        assert str(info.value) == message


def test_the_span_is_white_minus_dark_per_column(tmp_path):
    """mean(white, axis=0) - mean(dark, axis=0), per column x band: white == dark
    at a nonzero level is degenerate, a hot dark column is degenerate there
    even though the whole-frame dark mean is small, and a span below 1 is
    still a span."""
    cube = load_cube(_write_cube(tmp_path, "c", _known_cube(np.uint16, 1000))).cube
    level = _ref(tmp_path, "level", 500.0)
    with pytest.raises(CalibrationDegenerateError) as info:
        prepare_cube(cube, level, level)
    assert str(info.value) == (
        "white_reference - dark_reference is not positive at every band and "
        "column (min 0.0), so reflectance is undefined there. PlantCV would "
        "clip this to 1.0 silently; check that the white and dark references "
        "are the right files and not the same one."
    )

    hot = np.zeros((4, W, len(WL)), np.float64)
    hot[:, 7, :] = 1000.0  # a hot dark column, as bright as the white
    hot_load = load_cube(_write_cube(tmp_path, "hot", hot))
    with pytest.raises(CalibrationDegenerateError, match="not positive"):
        prepare_cube(cube, _ref(tmp_path, "white", 1000.0, rows=4), hot_load)

    # Positive controls: a span of 0.5 everywhere calibrates; so does a dark
    # reference that is hot everywhere but still below the white.
    half = prepare_cube(cube, _ref(tmp_path, "w05", 0.5), _ref(tmp_path, "d0", 0.0))
    assert half[1] == "white/dark"
    warm = prepare_cube(
        cube, _ref(tmp_path, "w1000", 1000.0), _ref(tmp_path, "d500", 500.0)
    )
    assert warm[1] == "white/dark"


def test_a_nan_reference_names_how_many_positions(tmp_path):
    cube = load_cube(_write_cube(tmp_path, "c", _known_cube(np.uint16, 1000))).cube
    white = np.full((1, W, len(WL)), 1000.0, np.float64)
    white[0, 3, 2] = np.nan
    with pytest.raises(CalibrationDegenerateError) as info:
        prepare_cube(
            cube,
            load_cube(_write_cube(tmp_path, "wnan", white)),
            _ref(tmp_path, "d0", 0.0),
        )
    assert str(info.value).startswith(
        "white_reference - dark_reference is not finite (NaN/Inf) at 1 "
        "band-column position(s) — a reference file contains invalid pixels, "
        "so reflectance is undefined there."
    )


def test_calibration_keeps_the_cubes_pseudo_rgb(tmp_path):
    load = load_cube(_write_cube(tmp_path, "c", _known_cube(np.uint16, 1000)))
    white, dark = (load_cube(p) for p in _refs(tmp_path))
    prepared, label, _ = prepare_cube(load.cube, white, dark)
    assert label == "white/dark"
    assert prepared.pseudo_rgb is load.cube.pseudo_rgb


def test_an_integer_calibration_result_is_cast(tmp_path, monkeypatch):
    """PlantCV 4.11.3's calibrate() returns float64; a version that returned
    the raw dtype would hand integer data to the index, which wraps."""
    load = load_cube(_write_cube(tmp_path, "c", _known_cube(np.uint16, 1000)))
    white, dark = (load_cube(p) for p in _refs(tmp_path))
    real = pcv.hyperspectral.calibrate

    def integer_calibrate(raw_data, white_reference, dark_reference):
        out = real(
            raw_data=raw_data,
            white_reference=white_reference,
            dark_reference=dark_reference,
        )
        out.array_data = (out.array_data * 1000).astype(np.uint16)
        return out

    monkeypatch.setattr(pcv.hyperspectral, "calibrate", integer_calibrate)
    prepared, label, _ = prepare_cube(load.cube, white, dark)
    assert label == "white/dark"
    assert prepared.array_data.dtype == np.float64
    assert float(prepared.array_data.max()) == 800.0
    assert prepared.pseudo_rgb is load.cube.pseudo_rgb


# --- indices ---


def test_compute_index_refuses_integer_data_itself(tmp_path):
    counts = load_cube(_write_cube(tmp_path, "c", _known_cube(np.uint16, 1000)))
    with pytest.raises(IndexUnavailableError) as info:
        compute_index(counts.cube, "ndvi")
    assert str(info.value) == (
        "Refusing to compute an index on integer counts (unsigned wraparound); "
        "this is a bug in the caller, prepare_cube() must run first."
    )
    # Positive control: the same cube, prepared, computes.
    prepared, _, _ = prepare_cube(counts.cube, None, None)
    assert compute_index(prepared, "ndvi") is not None


def test_an_index_outside_the_range_names_the_range(tmp_path):
    load = load_cube(_write_cube(tmp_path, "f", _known_cube()))
    with pytest.raises(IndexUnavailableError) as info:
        compute_index(load.cube, "wi")
    assert str(info.value) == (
        "Index 'wi' cannot be computed from wavelengths 600-900 nm (within 20 "
        "nm of the bands it needs). Choose an index the cube's range supports."
    )


# --- segmentation ---


def test_segment_defaults_and_the_result_fields(tmp_path):
    seg = segment_hyperspectral(_write_cube(tmp_path, "f", _known_cube()))
    assert (seg.index, seg.threshold, seg.object_type) == ("ndvi", 0.2, "light")
    assert seg.band_count == 5
    assert seg.wavelength_range == (600.0, 900.0)
    np.testing.assert_array_equal(seg.mask, DISC)


def test_the_default_fill_size_is_200(tmp_path):
    """A 7x7 = 49 px plant: the default fill erases it and says fill_size=200."""
    cube = _known_cube()
    small = np.zeros((H, W), bool)
    small[10:17, 10:17] = True
    cube[..., 1] = np.where(small, 0.2, 0.6)
    cube[..., 3] = np.where(small, 0.8, 0.4)
    path = _write_cube(tmp_path, "small", cube)
    seg = segment_hyperspectral(path)
    erased = next(w.message for w in seg.warnings if w.code == "fill_erased_mask")
    assert "the largest 49 px, and then fill_size=200 removed" in erased
    # Positive control: a fill below the plant keeps it.
    assert int((segment_hyperspectral(path, fill_size=40).mask > 0).sum()) == 49


def test_an_unknown_object_type_is_named(tmp_path):
    with pytest.raises(ValueError) as info:
        segment_hyperspectral(
            _write_cube(tmp_path, "f", _known_cube()), object_type="bright"
        )
    assert str(info.value) == "object_type must be 'light' or 'dark', got 'bright'"


def test_dark_polarity_selects_the_low_side(tmp_path):
    seg = segment_hyperspectral(
        _write_cube(tmp_path, "f", _known_cube()), threshold=0.2, object_type="dark"
    )
    np.testing.assert_array_equal(seg.mask, 255 - DISC)


def test_an_index_with_no_finite_value_is_refused_by_name(tmp_path):
    cube = _known_cube()
    cube[..., 1] = np.nan
    with pytest.raises(ValueError) as info:
        segment_hyperspectral(_write_cube(tmp_path, "nan", cube))
    assert str(info.value) == "No finite ndvi index values; nothing to segment."


def _range_codes(seg):
    return [w.message for w in seg.warnings if w.code == "threshold_outside_range"]


def test_dark_above_the_maximum_selects_every_pixel_and_says_so(tmp_path):
    seg = segment_hyperspectral(
        _write_cube(tmp_path, "f", _known_cube()), threshold=0.95, object_type="dark"
    )
    lo, hi = seg.index_range
    assert _range_codes(seg) == [
        (
            f"The threshold 0.95 with object_type='dark' on ndvi selects every "
            f"pixel: the data span {lo:.4g} to {hi:.4g}, so no pixel is on the "
            "other side of the rule. The mask says nothing about the plant. Choose "
            "a value inside that range."
        )
    ]


def test_calibration_args_name_preloaded_references(tmp_path):
    """The server passes LOADED refs and no paths; the session must still
    record where they came from, or measure_spectral has no refs to reload."""
    path = _write_cube(tmp_path, "c", _known_cube(np.uint16, 1000))
    white_path, dark_path = _refs(tmp_path)
    white, dark = load_cube(white_path), load_cube(dark_path)
    seg = segment_hyperspectral(path, white_load=white, dark_load=dark)
    assert seg.calibration_args == {
        "white_reference": white_path,
        "dark_reference": dark_path,
        "white_digest": white.digest,
        "dark_digest": dark.digest,
    }
    # Positive control: paths given by the caller are recorded as given.
    hdr_named = segment_hyperspectral(
        path, white_reference=white_path[:-4] + ".hdr", dark_reference=dark_path
    )
    assert hdr_named.calibration_args["white_reference"] == white_path[:-4] + ".hdr"


# --- results ---


def test_as_dict_carries_warnings_spectrum_and_histograms():
    res = SpectralResult(
        indices={"ndvi": {"mean": 0.6}},
        band_count=5,
        wavelength_range=(600.0, 900.0),
        calibration="none",
        warnings=[Advisory(code="nan_pixels", message="m")],
        pixel_count=3,
        spectrum={"wavelengths": [600.0]},
        histograms={"ndvi": {"bins": [0.6], "frequencies": [100.0]}},
    )
    assert res.as_dict() == {
        "indices": {"ndvi": {"mean": 0.6}},
        "band_count": 5,
        "wavelength_range": [600.0, 900.0],
        "calibration": "none",
        "pixel_count": 3,
        "warnings": [{"code": "nan_pixels", "message": "m"}],
        "spectrum": {"wavelengths": [600.0]},
        "histograms": {"ndvi": {"bins": [0.6], "frequencies": [100.0]}},
    }


def test_values_are_read_from_our_label_only():
    other = {"k": {"value": "theirs", "label": "theirs"}}
    ours = {"k": {"value": "ours", "label": "ours"}}
    for name in ("hsi", "hsi_1"):
        obs = {"default": other, name: ours}
        assert hs._value(obs, "k") == "ours"
        assert hs._label_of(obs, "k") == "ours"
    assert hs._value({"default": other}, "k") is None
    assert hs._label_of({"default": other}, "k") == []


def test_a_changed_reference_is_named_by_role():
    def load(digest):
        return CubeLoad(cube=None, digest=digest, raw_path="r", hdr_path="h")

    pinned = {"white_digest": "w", "dark_digest": "d"}
    check_reference_digests(pinned, load("w"), load("d"))  # positive control
    for role, white, dark in (
        ("white_reference", load("x"), load("d")),
        ("dark_reference", load("w"), load("x")),
    ):
        with pytest.raises(CalibrationReferencesChangedError) as info:
            check_reference_digests(pinned, white, dark)
        assert str(info.value) == (
            f"The {role} file changed since segment_hyperspectral() pinned it "
            "(SHA-256 mismatch). The calibrated values would silently differ "
            "from the segmentation's; re-run segment_hyperspectral() with the "
            "current reference files."
        )


# --- measurement ---


def test_measure_defaults_carry_no_histograms(tmp_path):
    res = measure_spectral(_write_cube(tmp_path, "f", _known_cube()), DISC)
    assert res.histograms == {} and "histograms" not in res.as_dict()
    assert list(res.indices) == ["ndvi"]


def test_index_statistics_over_both_populations(tmp_path):
    """The whole frame holds 0.6 (disc) and -0.2 (background): min, max and
    median are distinct, and each is reported under its own key."""
    res = measure_spectral(_write_cube(tmp_path, "f", _known_cube()), FULL)
    stats = res.indices["ndvi"]
    assert set(stats) == {"mean", "median", "std", "min", "max", "finite_pixel_count"}
    assert stats["min"] == pytest.approx(-0.2, abs=1e-6)
    assert stats["max"] == pytest.approx(0.6, abs=1e-6)
    assert stats["median"] == pytest.approx(-0.2, abs=1e-6)  # the disc is < half
    assert stats["finite_pixel_count"] == H * W


def test_histograms_span_the_masked_values_in_100_bins(tmp_path):
    res = measure_spectral(
        _write_cube(tmp_path, "f", _known_cube()), FULL, include_histograms=True
    )
    hist = res.histograms["ndvi"]
    assert set(hist) == {"bins", "frequencies"}
    assert len(hist["bins"]) == len(hist["frequencies"]) == 100
    assert all(isinstance(b, float) for b in hist["bins"])
    # "auto" bins run from the masked minimum to the masked maximum (not to
    # PlantCV's default of 1); the labels are bin centers, 0.8 / 100 apart.
    assert hist["bins"][0] == pytest.approx(-0.196, abs=1e-6)
    assert hist["bins"][-1] == pytest.approx(0.596, abs=1e-6)
    assert sum(hist["frequencies"]) == pytest.approx(100.0)
    assert res.as_dict()["histograms"] == res.histograms


def test_a_mask_of_the_wrong_shape_names_both_shapes(tmp_path):
    with pytest.raises(ValueError) as info:
        measure_spectral(
            _write_cube(tmp_path, "f", _known_cube()), np.full((10, 10), 255, np.uint8)
        )
    assert (
        str(info.value) == f"mask (10, 10) does not match the cube's frame ({H}, {W})"
    )


def test_a_zero_one_mask_measures_like_a_zero_255_mask(tmp_path):
    path = _write_cube(tmp_path, "f", _known_cube())
    ones = measure_spectral(path, (DISC > 0).astype(np.uint8))
    assert ones.pixel_count == int(_disc().sum())
    assert ones.indices == measure_spectral(path, DISC).indices


def test_a_mask_over_only_nonfinite_pixels_reports_no_min_or_max(tmp_path):
    cube = _known_cube()
    cube[_disc(), 1] = np.nan
    res = measure_spectral(_write_cube(tmp_path, "nan", cube), DISC)
    n = int(_disc().sum())
    stats = res.indices["ndvi"]
    assert stats["min"] is None and stats["max"] is None
    assert stats["finite_pixel_count"] == 0
    assert [(w.code, w.message) for w in res.warnings] == [
        (
            "nan_pixels",
            (
                f"{n} of {n} masked pixels have a non-finite ndvi value (NaN/Inf) "
                "and are excluded from min/max; PlantCV's mean/median/std may also "
                "be affected. pixel_count counts the full mask; finite_pixel_count "
                "counts the pixels the statistics rest on."
            ),
        )
    ]


def test_the_spectrum_reports_every_field(tmp_path):
    res = measure_spectral(
        _write_cube(tmp_path, "f", _known_cube()), FULL, include_spectrum=True
    )
    spec = res.spectrum
    assert list(spec) == [
        "wavelengths",
        "wavelength_means",
        "max_reflectance",
        "min_reflectance",
        "spectral_std",
        "global_mean_reflectance",
        "global_median_reflectance",
        "global_spectral_std",
    ]
    assert spec["wavelengths"] == WL
    by_wl = {k: dict(zip(WL, spec[k], strict=True)) for k in list(spec)[1:5]}
    # 670 nm: 0.2 in the disc, 0.6 outside; 600 nm is 0.3 everywhere.
    assert by_wl["max_reflectance"][670.0] == pytest.approx(0.6, abs=1e-6)
    assert by_wl["min_reflectance"][670.0] == pytest.approx(0.2, abs=1e-6)
    assert by_wl["spectral_std"][600.0] == pytest.approx(0.0, abs=1e-4)  # float32
    assert by_wl["spectral_std"][670.0] > 0.1
    assert isinstance(spec["global_mean_reflectance"], float)
    assert isinstance(spec["global_median_reflectance"], float)
    assert isinstance(spec["global_spectral_std"], float)
