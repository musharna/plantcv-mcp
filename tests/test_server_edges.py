"""The server layer at its edges: what each tool hands its analysis, what it
stores in the session, and the exact envelope it returns (mutation burn-down,
#97).

Most of these were mutants that no test could see. Two shapes recur:

- The parent loads a file and passes the load to the analysis so the isolation
  worker never touches the filesystem. Every analysis also reloads whatever it
  is not given, so dropping the parent's load changed no number. A recording
  `dispatch` (calling through to the real one) asserts the handoff itself.
- Envelope keys, refusal texts and help strings were matched loosely or not
  at all; here they are compared whole.
"""

import itertools
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from plantcv_mcp import plantcv_version, sam_leaves, server, workers
from plantcv_mcp.diagnostics import analyze_mask, mask_warnings
from plantcv_mcp.hyperspectral import load_cube
from plantcv_mcp.imaging import downscale, encode_png, load_image
from plantcv_mcp.limits import ParameterRangeError
from plantcv_mcp.server import (
    ANALYSES,
    OBJECT_TYPES,
    REGION_MODES,
    ImageChangedSinceSegmentationError,
    _clip_box,
    _crop_impl,
    _measure_morphology_impl,
    _measure_spectral_impl,
    _measure_thermal_impl,
    _refine_impl,
    _require_box,
    _segment_hsi_impl,
    _segment_impl,
    _segment_leaves_sam_impl,
    _segment_thermal_impl,
    _store,
    _zoom_impl,
    list_methods_impl,
)
from plantcv_mcp.thermal import load_thermal

ENGINE = {"name": "PlantCV", "version": plantcv_version()}
SUMMARY_KEYS = {
    "mask_fraction",
    "component_count",
    "major_object_count",
    "largest_area",
}

# --- fixtures and helpers ---------------------------------------------------


@pytest.fixture
def calls(monkeypatch):
    """Every dispatch the server makes, in-process, recorded and then run for
    real: (name, args, kwargs, result)."""
    seen: list = []
    real = workers.dispatch

    def recording(name, *args, **kwargs):
        result = real(name, *args, **kwargs)
        seen.append(SimpleNamespace(name=name, args=args, kwargs=kwargs, result=result))
        return result

    workers.set_isolation(False)
    monkeypatch.setattr(server, "dispatch", recording)
    yield seen
    workers.set_isolation(None)


def _hsi_helpers():
    sys.path.insert(0, os.path.dirname(__file__))
    import test_hyperspectral as th

    return th


def _counts_cube(tmp_path, name="counts", scale=1000):
    th = _hsi_helpers()
    return th._write_cube(tmp_path, name, th._known_cube(np.uint16, scale=scale))


def _thermal_npz(tmp_path, name="frame.npz", nan=True):
    frame = np.full((120, 160), 20.0, np.float64)
    yy, xx = np.ogrid[:120, :160]
    frame[(xx - 80) ** 2 + (yy - 60) ** 2 <= 25**2] = 31.5
    if nan:
        frame[0, 0] = np.nan  # one non-finite pixel: a nan_pixels warning
    path = tmp_path / name
    np.savez(path, frame)
    return str(path)


def _png(tmp_path, img, name="img.png"):
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


def _decoded_shape(png: bytes) -> tuple[int, int]:
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED).shape[:2]


def _warning_dicts(items) -> list[dict]:
    return [{"code": w.code, "message": w.message} for w in items]


# --- list_methods() ---------------------------------------------------------

GUIDANCE = (
    "Pick a channel AND the object_type that goes with it. object_type "
    "says which side of the threshold is the plant: 'dark' selects "
    "pixels below it, 'light' above it. Choosing the wrong one returns "
    "the BACKGROUND as the plant, with traits that still look "
    "plausible. Measured on a green-plant render: 'a' of LAB needs "
    "object_type='dark' (the usual starting point for green tissue on a "
    "light background), while 's' and 'b' need object_type='light' — "
    "with 'dark' they select 96% of the frame. 'l' and 'v' were "
    "ambiguous on that image. These are starting points, not rules: "
    "call suggest_segmentation(), which reports what BOTH polarities "
    "actually yield on your image."
)


def test_list_methods_returns_every_key_and_the_guidance_word_for_word():
    info = list_methods_impl()
    assert set(info) == {
        "server_version",
        "plantcv_version",
        "channels",
        "methods",
        "object_types",
        "analyses",
        "region_modes",
        "refine_ops",
        "read_roots",
        "guidance",
    }
    assert info["object_types"] == list(OBJECT_TYPES)
    assert info["analyses"] == list(ANALYSES)
    assert info["region_modes"] == list(REGION_MODES)
    assert info["guidance"] == GUIDANCE


# --- main(): the command line -----------------------------------------------


def _run_main(monkeypatch, argv):
    """main() with the server, isolation and roots replaced by recorders."""
    rec = SimpleNamespace(isolation=[], roots=[], runs=0)

    class FakeServer:
        def run(self):
            rec.runs += 1

    monkeypatch.setattr(server, "build_server", lambda: FakeServer())
    monkeypatch.setattr(server, "set_isolation", rec.isolation.append)
    monkeypatch.setattr(server, "set_roots", rec.roots.append)
    # argv[0] is NOT the program name, so prog= is what names it in --help.
    monkeypatch.setattr(sys, "argv", ["not-the-prog", *argv])
    server.main()
    return rec


def test_main_help_names_the_program_and_says_what_each_flag_does(monkeypatch, capsys):
    monkeypatch.setenv("COLUMNS", "400")
    with pytest.raises(SystemExit) as done:
        _run_main(monkeypatch, ["--help"])
    assert done.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert text.startswith(
        "usage: plantcv-mcp [-h] [--isolate | --no-isolate] [--root DIR] "
        "PlantCV as an MCP measurement instrument (stdio). "
    )
    assert (
        "--isolate, --no-isolate run every PlantCV analysis in a worker "
        "subprocess so a native crash becomes a tool error instead of killing "
        "the server. ON by default (measured +7.7% wall time); --no-isolate or "
        "PLANTCV_MCP_ISOLATE=0 runs analyses in-process."
    ) in text
    assert text.endswith(
        "--root DIR only read images — and write correct_lens_distortion's "
        "output — under DIR (repeatable; also PLANTCV_MCP_ROOTS, "
        "os.pathsep-separated). Unset: read anything the host user can read."
    )


def test_main_wires_isolation_and_roots_only_when_given(monkeypatch):
    rec = _run_main(monkeypatch, [])
    assert (rec.isolation, rec.roots, rec.runs) == ([], [], 1)
    rec = _run_main(monkeypatch, ["--no-isolate", "--root", "a", "--root", "b"])
    assert (rec.isolation, rec.roots, rec.runs) == ([False], [["a", "b"]], 1)
    rec = _run_main(monkeypatch, ["--isolate"])
    assert (rec.isolation, rec.roots, rec.runs) == ([True], [], 1)


# --- boxes: _require_box, _clip_box ------------------------------------------


@pytest.mark.parametrize(
    ("box", "message"),
    [
        ((-1, 0, 1, 1), "x must be >= 0, got -1."),
        ((0, -1, 1, 1), "y must be >= 0, got -1."),
        ((0, 0, 0, 1), "w must be >= 1, got 0."),
        ((0, 0, 1, 0), "h must be >= 1, got 0."),
    ],
)
def test_a_bad_box_is_refused_naming_the_field(box, message):
    with pytest.raises(ParameterRangeError) as refused:
        _require_box(*box)
    assert str(refused.value) == message


def test_the_smallest_box_is_accepted():
    assert _require_box(0, 0, 1, 1) is None


def test_a_box_starting_on_the_far_edge_is_outside_the_image():
    img = np.zeros((10, 20, 3), np.uint8)
    assert _clip_box(img, 19, 9, 5, 5) == (20, 10)  # the last pixel is inside
    for x, y in ((20, 0), (0, 10)):
        with pytest.raises(ParameterRangeError) as refused:
            _clip_box(img, x, y, 1, 1)
        assert str(refused.value) == (
            f"the box starts at ({x}, {y}), outside the 20x10 image."
        )


# --- zoom_image() -----------------------------------------------------------

ZOOM_NOTE = (
    "A picture to look at, not a measurement: image pixel = "
    "(zoomed pixel / scale) + box origin. To segment, measure or count "
    "leaves in this box, crop_image() it and work on that file."
)


def test_zoom_enlarges_with_cubic_interpolation_and_says_how_to_map_back(tmp_path):
    rng = np.random.default_rng(0)
    path = _png(tmp_path, rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    crop = load_image(path)
    out = _zoom_impl(path, 0, 0, 64, 64, scale=2)
    cubic = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    area = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    linear = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    assert out["_png"] == encode_png(cubic)
    # Positive control: the interpolations really differ on this image.
    assert encode_png(cubic) != encode_png(area) != encode_png(linear)
    assert out == {
        "box": [0, 0, 64, 64],
        "clipped": False,
        "scale": 2.0,
        "note": ZOOM_NOTE,
        "_png": encode_png(cubic),
    }


def test_zoom_scale_one_and_a_one_pixel_box_stay_at_one(tmp_path):
    path = _png(tmp_path, np.full((8, 8, 3), 90, np.uint8))
    out = _zoom_impl(path, 3, 3, 1, 1, scale=1)
    assert out["scale"] == 1.0 and _decoded_shape(out["_png"]) == (1, 1)


def test_zoom_refuses_a_scale_past_the_cap_by_name(tmp_path):
    path = _png(tmp_path, np.full((8, 8, 3), 90, np.uint8))
    with pytest.raises(ParameterRangeError) as refused:
        _zoom_impl(path, 0, 0, 4, 4, scale=9)
    assert str(refused.value) == "scale must be between 1 and 8, got 9."


def test_zoom_never_enlarges_a_box_past_the_output_edge(tmp_path):
    # 600 px: 1024 // 600 is 1, so the box is shown at 1:1, not stretched.
    path = _png(tmp_path, np.full((30, 600, 3), 90, np.uint8), "wide.png")
    out = _zoom_impl(path, 0, 0, 600, 30)
    assert out["scale"] == 1.0 and _decoded_shape(out["_png"]) == (30, 600)
    # A requested scale above what fits is held to the fit: 300 px fits 3x.
    path = _png(tmp_path, np.full((20, 300, 3), 90, np.uint8), "mid.png")
    out = _zoom_impl(path, 0, 0, 300, 20, scale=8)
    assert out["scale"] == 3.0 and _decoded_shape(out["_png"]) == (60, 900)


def test_zoom_of_a_box_wider_than_the_output_reports_the_shrink(tmp_path):
    path = _png(tmp_path, np.full((40, 1100, 3), 90, np.uint8))
    out = _zoom_impl(path, 0, 0, 1100, 40)
    assert out["scale"] == 0.9309  # 1024 / 1100, to 4 places
    assert _decoded_shape(out["_png"]) == (37, 1024)


# --- crop_image() -----------------------------------------------------------

CROP_NOTE = (
    "Same pixels, same scale: px_per_mm calibrated on the full "
    "image applies unchanged. Positions in results on the crop are crop "
    "pixels; image pixel = crop pixel + (box x, box y)."
)


def test_crop_envelope_is_exact(tmp_path):
    img = np.zeros((40, 60, 3), np.uint8)
    img[10:30, 20:50] = (60, 180, 60)
    path = _png(tmp_path, img)
    out = _crop_impl(path, 20, 10, 30, 20)
    crop = img[10:30, 20:50]
    preview, scale = downscale(crop)
    assert out == {
        "cropped_image_path": str(tmp_path / "img_crop_x20_y10_w30_h20.png"),
        "box": [20, 10, 30, 20],
        "clipped": False,
        "note": CROP_NOTE,
        "overlay_scale": scale,
        "_png": encode_png(preview),
    }
    assert scale > 1  # the preview of a small crop is enlarged, and says so


def test_crop_refuses_a_lossy_or_existing_output_path_in_full(tmp_path):
    path = _png(tmp_path, np.full((20, 20, 3), 90, np.uint8))
    jpg = str(tmp_path / "out.jpg")
    with pytest.raises(ValueError) as refused:
        _crop_impl(path, 0, 0, 5, 5, output_path=jpg)
    assert str(refused.value) == (
        f"output_path {jpg!r} must end in .png: the crop is written lossless, "
        "so the pixels measured are the image's own."
    )
    taken = tmp_path / "taken.png"
    taken.write_bytes(b"someone else's file")
    with pytest.raises(FileExistsError) as refused:
        _crop_impl(path, 0, 0, 5, 5, output_path=str(taken))
    assert str(refused.value) == (
        f"output_path {str(taken)!r} already exists. Pass a new path, or omit "
        "output_path to write (and overwrite) the derived "
        "<image>_crop_x<x>_y<y>_w<w>_h<h>.png next to the input."
    )
    assert taken.read_bytes() == b"someone else's file"


# --- segment_hyperspectral() / measure_spectral() ---------------------------


def test_segment_hsi_hands_the_worker_the_parents_loads(tmp_path, calls):
    th = _hsi_helpers()
    path = _counts_cube(tmp_path)
    white, dark = th._refs(tmp_path)
    out = _segment_hsi_impl(
        path,
        index="sr",
        threshold=1.5,
        object_type="light",
        white_reference=white,
        dark_reference=dark,
    )
    (call,) = calls
    assert call.name == "hsi_segment"
    (load,) = call.args
    assert load.raw_path == path and load.digest == load_cube(path).digest
    kw = dict(call.kwargs)
    white_load, dark_load = kw.pop("white_load"), kw.pop("dark_load")
    assert kw == {
        "index": "sr",
        "threshold": 1.5,
        "object_type": "light",
        "white_reference": white,
        "dark_reference": dark,
        "fill_size": 200,
    }
    # Loaded HERE, in the parent: the worker must not read the refs itself.
    assert white_load.raw_path == white and dark_load.raw_path == dark
    assert white_load.digest == load_cube(white).digest
    assert dark_load.digest == load_cube(dark).digest
    session = _store.get(out["session_id"])
    assert session.channel == "index:sr" and session.method == "threshold:1.5"
    assert session.extra == {
        "index": "sr",
        "threshold": 1.5,
        "object_type": "light",
        "calibration": "white/dark",
        "calibration_args": call.result.calibration_args,
    }


def test_segment_hsi_envelope_is_exact(tmp_path, calls):
    path = _counts_cube(tmp_path)
    out = _segment_hsi_impl(path, threshold=0, object_type="light")
    (call,) = calls
    seg = call.result
    assert seg.warnings, "premise: an uncalibrated counts cube is flagged"
    small, scale = downscale(seg.overlay)
    d = seg.diagnostics
    assert out == {
        "session_id": out["session_id"],
        "kind": "hsi",
        "envi_path": path,
        "index": "ndvi",
        "threshold": 0.0,
        "object_type": "light",
        "calibration": "none",
        "index_range": list(seg.index_range),
        "band_count": 5,
        "wavelength_range": [600.0, 900.0],
        "mask_fraction": d.mask_fraction,
        "component_count": d.component_count,
        "major_object_count": d.major_object_count,
        "largest_area": d.largest_area,
        "overlay_scale": scale,
        "overlay_png_bytes": len(encode_png(small)),
        "warnings": _warning_dicts(seg.warnings),
        "engine": ENGINE,
        "_png": encode_png(small),
    }
    assert type(out["threshold"]) is float


def test_measure_spectral_rereads_the_refs_in_the_parent(tmp_path, calls):
    th = _hsi_helpers()
    path = _counts_cube(tmp_path)
    white, dark = th._refs(tmp_path)
    seg = _segment_hsi_impl(path, white_reference=white, dark_reference=dark)
    calls.clear()
    out = _measure_spectral_impl(seg["session_id"])
    (call,) = calls
    assert call.name == "hsi_measure"
    load, mask = call.args
    assert load.digest == load_cube(path).digest
    assert mask is _store.get(seg["session_id"]).mask
    kw = dict(call.kwargs)
    white_load, dark_load = kw.pop("white_load"), kw.pop("dark_load")
    assert white_load.raw_path == white and dark_load.raw_path == dark
    assert kw == {
        "indices": ("ndvi",),
        "calibration": _store.get(seg["session_id"]).extra["calibration_args"],
        "include_spectrum": False,
        "include_histograms": False,
    }
    assert out == {
        "session_id": seg["session_id"],
        "kind": "hsi",
        **call.result.as_dict(),
        "engine": ENGINE,
    }
    # Both extras are passed through when asked for.
    calls.clear()
    _measure_spectral_impl(
        seg["session_id"],
        indices=["sr"],
        include_spectrum=True,
        include_histograms=True,
    )
    (call,) = calls
    assert call.kwargs["indices"] == ("sr",)
    assert call.kwargs["include_spectrum"] is True
    assert call.kwargs["include_histograms"] is True


def test_measure_spectral_defaults_to_the_sessions_index_then_ndvi(tmp_path, calls):
    path = _counts_cube(tmp_path)
    seg = _segment_hsi_impl(path, index="sr", threshold=1.5)
    calls.clear()
    _measure_spectral_impl(seg["session_id"])
    assert calls[-1].kwargs["indices"] == ("sr",)
    # A session that recorded no index falls back to ndvi.
    s = _store.get(seg["session_id"])
    bare = _store.create(
        s.image_path, s.mask, s.channel, s.method, digest=s.digest, kind="hsi"
    )
    calls.clear()
    _measure_spectral_impl(bare.session_id)
    assert calls[-1].kwargs["indices"] == ("ndvi",)


def test_a_changed_cube_is_refused_with_the_full_reason(tmp_path):
    path = _counts_cube(tmp_path)
    seg = _segment_hsi_impl(path)
    _measure_spectral_impl(seg["session_id"])  # positive control: unchanged
    _counts_cube(tmp_path, scale=500)  # same name, new bytes
    with pytest.raises(ImageChangedSinceSegmentationError) as refused:
        _measure_spectral_impl(seg["session_id"])
    assert str(refused.value) == (
        f"The ENVI cube at {path!r} (or its .hdr) changed since "
        "segment_hyperspectral() produced this session's mask (SHA-256 "
        "mismatch). Re-run segment_hyperspectral() on the current files."
    )


# --- segment_thermal() / measure_thermal() ----------------------------------


def test_segment_thermal_hands_over_the_load_and_records_the_band(tmp_path, calls):
    path = _thermal_npz(tmp_path)
    out = _segment_thermal_impl(path, min_c=25.0, max_c=35.0)
    (call,) = calls
    assert call.name == "thermal_segment"
    load, given_path = call.args
    assert given_path == path and load.digest == load_thermal(path).digest
    assert call.kwargs == {"min_c": 25.0, "max_c": 35.0, "fill_size": 200}
    seg = call.result
    assert seg.warnings, "premise: the NaN pixel is flagged"
    session = _store.get(out["session_id"])
    assert session.channel == "celsius" and session.method == "band:25.0-35.0"
    assert session.extra == {"min_c": 25.0, "max_c": 35.0, "source": "npz"}
    small, scale = downscale(seg.overlay)
    d = seg.diagnostics
    assert out == {
        "session_id": out["session_id"],
        "kind": "thermal",
        "path": path,
        "source": "npz",
        "min_c": 25.0,
        "max_c": 35.0,
        "frame_range": [20.0, 31.5],
        "mask_fraction": d.mask_fraction,
        "component_count": d.component_count,
        "major_object_count": d.major_object_count,
        "largest_area": d.largest_area,
        "overlay_scale": scale,
        "overlay_png_bytes": len(encode_png(small)),
        "warnings": _warning_dicts(seg.warnings),
        "engine": ENGINE,
        "_png": encode_png(small),
    }
    assert set(out) >= SUMMARY_KEYS


def test_measure_thermal_hands_over_the_checked_load(tmp_path, calls):
    path = _thermal_npz(tmp_path)
    seg = _segment_thermal_impl(path, min_c=25.0, max_c=35.0)
    calls.clear()
    out = _measure_thermal_impl(seg["session_id"])
    (call,) = calls
    assert call.name == "thermal_measure"
    load, given_path, mask = call.args
    session = _store.get(seg["session_id"])
    assert load.digest == session.digest and given_path == path
    assert mask is session.mask
    assert call.kwargs == {"include_histograms": False}
    assert out == {
        "session_id": seg["session_id"],
        "kind": "thermal",
        **call.result.as_dict(),
        "engine": ENGINE,
    }
    calls.clear()
    _measure_thermal_impl(seg["session_id"], include_histograms=True)
    assert calls[-1].kwargs == {"include_histograms": True}


def test_a_changed_thermal_file_is_refused_with_the_full_reason(tmp_path):
    path = _thermal_npz(tmp_path)
    seg = _segment_thermal_impl(path, min_c=25.0, max_c=35.0)
    _measure_thermal_impl(seg["session_id"])  # positive control: unchanged
    _thermal_npz(tmp_path, nan=False)  # same name, new bytes
    with pytest.raises(ImageChangedSinceSegmentationError) as refused:
        _measure_thermal_impl(seg["session_id"])
    assert str(refused.value) == (
        f"The thermal file at {path!r} changed since "
        "segment_thermal() produced this session's mask (SHA-256 mismatch). "
        "Re-run segment_thermal() on the current file."
    )


def test_a_thermal_session_of_another_shape_is_refused_with_both_shapes(tmp_path):
    """Same bytes, different shape: reachable only through a session whose mask
    does not match its frame, which the digest check cannot catch."""
    path = _thermal_npz(tmp_path)
    digest = load_thermal(path).digest
    session = _store.create(
        path,
        np.zeros((60, 80), np.uint8),
        "celsius",
        "band:None-None",
        digest=digest,
        kind="thermal",
    )
    with pytest.raises(ImageChangedSinceSegmentationError) as refused:
        _measure_thermal_impl(session.session_id)
    assert str(refused.value) == (
        f"The thermal frame at {path!r} is now (120, 160) but was (60, 80) at "
        "segmentation."
    )


# --- measure_morphology() ---------------------------------------------------


def _cross_png(tmp_path):
    img = np.full((200, 200, 3), 128, np.uint8)
    cv2.rectangle(img, (95, 20), (105, 180), (60, 180, 60), -1)
    cv2.rectangle(img, (60, 90), (140, 100), (60, 180, 60), -1)
    return _png(tmp_path, img, "cross.png")


def test_measure_morphology_defaults_and_envelope(tmp_path, calls):
    seg = _segment_impl(_cross_png(tmp_path), "a", "otsu")
    refined = _refine_impl(seg["session_id"], [{"op": "fill_holes"}])["session_id"]
    calls.clear()
    out = _measure_morphology_impl(refined, px_per_mm=2.0)
    (call,) = calls
    assert call.name == "morphology"
    _img, mask, prune, tangent, px_per_mm = call.args
    assert (prune, tangent, px_per_mm) == (15, 25, 2.0)
    assert mask is _store.get(refined).mask
    res = call.result
    assert res.warnings, "premise: the cross has no leaf segments to measure"
    small, scale = downscale(res.overlay)
    assert out == {
        "session_id": refined,
        "lineage": [{"op": "fill_holes"}],
        "prune_size": 15,
        "tangent_size": 25,
        "px_per_mm": 2.0,
        "plant": res.plant,
        "segments": res.segments,
        "units": res.units,
        "warnings": _warning_dicts(res.warnings),
        "overlay_scale": scale,
        "engine": ENGINE,
        "_png": encode_png(small),
    }


# --- segment_leaves_sam(), without the model --------------------------------


@pytest.fixture
def sam_stubs(monkeypatch, tmp_path):
    """The extra 'present', the model never loaded, candidates the plant's
    three vertical thirds. Returns the recorded load_model calls."""
    loads: list = []
    monkeypatch.setattr(server, "require_sam", lambda: (None, None))
    monkeypatch.setattr(
        sam_leaves, "load_model", lambda ckpt, device: loads.append(ckpt) or object()
    )

    def thirds(model, rgb, plant):
        xs = np.nonzero(plant.any(axis=0))[0]
        cuts = np.linspace(xs.min(), xs.max() + 1, 4).astype(int)
        cols = np.arange(plant.shape[1])
        return [
            (plant & ((cols >= a) & (cols < b))[None, :], 0.95, 0.95)
            for a, b in itertools.pairwise(cuts)
        ]

    monkeypatch.setattr(sam_leaves, "_candidates", thirds)
    monkeypatch.setenv(sam_leaves.CACHE_ENV, str(tmp_path / "empty-cache"))
    return loads


def _edge_plant_png(tmp_path):
    img = np.full((200, 300, 3), 128, np.uint8)
    img[0:120, 60:240] = (60, 180, 60)  # cut by the top edge: frame_clipping
    return _png(tmp_path, img, "edge.png")


def test_segment_leaves_sam_passes_the_resolved_checkpoint_and_exact_envelope(
    tmp_path, calls, sam_stubs
):
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"never opened")
    seg = _segment_impl(_edge_plant_png(tmp_path), "a", "otsu")
    refined = _refine_impl(seg["session_id"], [{"op": "fill_holes"}])["session_id"]
    calls.clear()
    out = _segment_leaves_sam_impl(
        refined, checkpoint_path=str(checkpoint), px_per_mm=2.0
    )
    (call,) = calls
    assert call.name == "leaves_sam"
    _img, mask, ckpt, device, px_per_mm, observed = call.args
    session = _store.get(refined)
    assert (ckpt, device, px_per_mm) == (str(checkpoint), "cpu", 2.0)
    assert mask is session.mask and observed is session.observed
    assert sam_stubs == [str(checkpoint)]
    res = call.result
    codes = [w["code"] for w in out["warnings"]]
    assert "frame_clipping" in codes, "premise: the mask-level advisory is carried"
    carried = [
        {"code": w.code, "message": w.message}
        for w in mask_warnings(session.mask, analyze_mask(session.mask))
        if w.code != "multi_specimen"
    ]
    small, scale = downscale(res.overlay)
    assert out == {
        "session_id": refined,
        "lineage": [{"op": "fill_holes"}],
        "method": "segment_anything_point_grid",
        "model": {
            "name": "Segment Anything",
            "type": sam_leaves.MODEL_TYPE,
            "checkpoint": sam_leaves.CHECKPOINT_NAME,
            "sha256": sam_leaves.CHECKPOINT_SHA256,
            "license": "Apache-2.0",
        },
        "device": "cpu",
        "px_per_mm": 2.0,
        "leaf_count": res.leaf_count,
        "candidate_masks": res.candidates,
        "mask_coverage": res.mask_coverage,
        "instances": res.instances,
        "units": res.units,
        "warnings": _warning_dicts(res.warnings) + carried,
        "overlay_scale": scale,
        "engine": ENGINE,
        "_png": encode_png(small),
    }


def test_segment_leaves_sam_never_downloads_unless_asked(
    tmp_path, monkeypatch, sam_stubs
):
    fetched: list = []
    monkeypatch.setattr(sam_leaves, "_download", fetched.append)
    seg = _segment_impl(_edge_plant_png(tmp_path), "a", "otsu")
    with pytest.raises(
        sam_leaves.CheckpointMissingError, match="Nothing was downloaded"
    ):
        _segment_leaves_sam_impl(seg["session_id"])
    assert fetched == [] and sam_stubs == []
