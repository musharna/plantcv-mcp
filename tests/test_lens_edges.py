"""The lens tools at their edges: every advisory's exact text, each threshold
on both sides of its line, the calibration cache, the checkerboard digest, the
write refusals, and the correction's whole envelope (mutation burn-down, #97).

The calibrations here are built by hand: the advisories, the cache and the
routing in _correct_lens_impl depend only on a LensCalibration's fields, and a
real fit (tests/test_lens.py) cannot place rms, coverage or the focal
uncertainty exactly on a threshold.
"""

import hashlib
import os
from collections import OrderedDict

import cv2
import numpy as np
import pytest

import plantcv_mcp.server as S
from plantcv_mcp import paths
from plantcv_mcp.imaging import load_image
from plantcv_mcp.lens import LensCalibration, undistort_image

QUIET = {"roi_degenerate": False, "residual_void_px": 0, "crop_fraction": 0.0}
TAIL = (
    " Segment and measure {out!r} from here on — and take any px_per_mm from "
    "calibrate_scale_from_marker on the CORRECTED image. A scale calibrated on "
    "the original mixes two geometries: distortion is anisotropic, and no "
    "single number can compensate for it."
)


def _calib(**kw):
    """Ten views, a tight fit, full coverage: no advisory but lens_corrected."""
    base = {
        "mtx": np.eye(3),
        "dist": np.zeros(5),
        "rms": 0.5,
        "frames_used": [f"v{i}.png" for i in range(10)],
        "frames_skipped": [],
        "shape": (600, 800),
    }
    base.update(kw)
    return LensCalibration(**base)


def _codes(calib, info=QUIET):
    return [w["code"] for w in S._lens_advisories(calib, info, "o.png")]


# --- the advisories, word for word ---------------------------------------------


def test_every_advisory_at_once_word_for_word():
    calib = _calib(
        rms=3.0,
        frames_used=["a.png", "b.png", "c.png"],
        frames_skipped=["bad.png", "notes.txt"],
        coverage=0.25,
        # two of each, so the separators that join them are in the text
        frames_outliers=(
            ("o.png", 37.04, "37.0 px against a median of 4.0 px"),
            ("p.png", 21.5, "21.5 px against a median of 4.0 px"),
        ),
        median_view_rms=4.04,
        frames_duplicates=(("copy.png", "a.png"), ("copy2.png", "b.png")),
        focal_uncertainty=0.04,
    )
    info = {"roi_degenerate": False, "residual_void_px": 7, "crop_fraction": 0.12}
    out = "/data/scene_undistorted.png"
    assert S._lens_advisories(calib, info, out) == [
        {
            "code": "lens_corrected",
            "message": "Lens distortion corrected from 3 checkerboard frame(s), "
            "rms reprojection error 3.00 px. No usable fully-valid crop exists "
            "for this calibration; 7 fabricated void pixels remain in the output."
            + TAIL.format(out=out),
        },
        {
            "code": "outlier_frames_dropped",
            "message": "2 checkerboard frame(s) were left out of the calibration, "
            "each judged against the views around it at the time: o.png 37.0 px "
            "against a median of 4.0 px; p.png 21.5 px against a median of 4.0 "
            "px. The 3 views kept fit at a median of 4.0 "
            "px. A frame like that is usually blurred, mis-detected, shot at a "
            "different zoom, or of a board that was not flat; the camera was "
            "refitted without it (measured: one such frame moved the focal length "
            "13% on the PlantCV tutorial set). Check those frames, and re-shoot "
            "or remove them.",
        },
        {
            "code": "frames_skipped",
            "message": "2 file(s) in the checkerboard directory were left out "
            "before any camera was fitted: bad.png, notes.txt. No board of the "
            "given corner counts was detected there, the file did not decode, or "
            "its size differs from the other frames. The calibration used the "
            "remaining 3. If you expected those frames to count, check the corner "
            "counts (INNER corners, one less per side than squares) and that "
            "every photo is of the same board at the same resolution.",
        },
        {
            "code": "duplicate_frames_ignored",
            "message": "2 checkerboard file(s) are byte-for-byte copies of "
            "another and were counted once: copy.png (same bytes as a.png), "
            "copy2.png (same bytes as b.png). "
            "Copies add no geometry; they only make every uncertainty the fit "
            "reports look smaller than it is.",
        },
        {
            "code": "focal_length_uncertain",
            "message": "The 3 views determine the focal length only to about "
            "±4.0% (the fit's own uncertainty on it), so the correction is "
            "correspondingly loose toward the frame corners (measured: a set at "
            "3.6% left the corners 54 px off). Judged per view, because that "
            "ratio falls as frames are added whether or not the geometry "
            "improves: these views are worth ±6.9% each. The board was tilted "
            "too little, or only about one axis; more strongly and more "
            "variously tilted views tighten it — more frames of the SAME view "
            "will not.",
        },
        {
            "code": "thin_calibration",
            "message": "Only 3 checkerboard frame(s) went into this calibration "
            "(2 skipped: bad.png, notes.txt). Ten or more views, tilted and "
            "spread across the frame, make a materially better model — the "
            "correction here may be partial.",
        },
        {
            "code": "high_reprojection_error",
            "message": "The calibration fits its own checkerboards only to 3.0 "
            "px rms — 0.30% of the frame diagonal (good calibrations sit under "
            "0.15%). That number is the fit at the detected corners, not a bound "
            "on the residual distortion elsewhere: the correction is only as "
            "good as the fit. Sharper, more varied board views tighten it.",
        },
        {
            "code": "low_calibration_coverage",
            "message": "The detected checkerboards covered only 25% of the frame "
            "(under 40%). The model is fitted where boards were and extrapolated "
            "everywhere else, so the correction is least trustworthy toward the "
            "frame edges and corners. Add views with the board near each edge "
            "and corner.",
        },
        {
            "code": "distortion_voids_remain",
            "message": "The corrected image still contains fabricated pixels at "
            "its edges — black voids and the void-blended border around them "
            "(residual_void_px counts both). A value/darkness threshold will "
            "select them as objects; segment on a channel where black is neutral "
            "(such as 'a'), and check the overlay.",
        },
    ]


def test_thin_calibration_with_nothing_skipped_says_no_skips():
    (thin,) = [
        w
        for w in S._lens_advisories(_calib(frames_used=["a.png"]), QUIET, "o.png")
        if w["code"] == "thin_calibration"
    ]
    assert thin["message"] == (
        "Only 1 checkerboard frame(s) went into this calibration. Ten or more "
        "views, tilted and spread across the frame, make a materially better "
        "model — the correction here may be partial."
    )


@pytest.mark.parametrize(
    ("info", "note", "voids"),
    [
        # either half of the void test speaks on its own
        (
            {"roi_degenerate": True, "residual_void_px": 0, "crop_fraction": 0.3},
            (
                "No usable fully-valid crop exists for this calibration; 0 fabricated "
                "void pixels remain in the output."
            ),
            True,
        ),
        (
            {"roi_degenerate": False, "residual_void_px": 1, "crop_fraction": 0.3},
            (
                "No usable fully-valid crop exists for this calibration; 1 fabricated "
                "void pixels remain in the output."
            ),
            True,
        ),
        (
            {"roi_degenerate": False, "residual_void_px": 0, "crop_fraction": 0.12},
            (
                "The corrected frame was cropped to the largest fully-valid rectangle "
                "— 12% of the frame (remap voids and the contaminated border around "
                "them) was removed."
            ),
            False,
        ),
        (QUIET, "No void crop was needed.", False),
    ],
)
def test_the_crop_note_and_the_void_warning(info, note, voids):
    warnings = S._lens_advisories(_calib(), info, "o.png")
    assert warnings[0] == {
        "code": "lens_corrected",
        "message": "Lens distortion corrected from 10 checkerboard frame(s), rms "
        f"reprojection error 0.50 px. {note}" + TAIL.format(out="o.png"),
    }
    assert [w["code"] for w in warnings[1:]] == (
        ["distortion_voids_remain"] if voids else []
    )


# --- each threshold on both sides of its line ------------------------------------


def test_focal_uncertainty_is_judged_per_view_and_strictly_above_the_line():
    line = S.FOCAL_UNCERTAINTY_ADVISORY
    one = ["a.png"]
    # One view: per-view uncertainty is the ratio itself (sqrt(max(1, 1)) = 1).
    assert "focal_length_uncertain" not in _codes(
        _calib(frames_used=one, focal_uncertainty=line)
    )
    assert "focal_length_uncertain" in _codes(
        _calib(frames_used=one, focal_uncertainty=line * 1.01)
    )
    # 0.8 of the line on one view stays quiet; on two views it is 0.8 * sqrt(2)
    # = 1.13 of the line and speaks.
    assert "focal_length_uncertain" not in _codes(
        _calib(frames_used=one, focal_uncertainty=line * 0.8)
    )
    assert "focal_length_uncertain" in _codes(
        _calib(frames_used=["a.png", "b.png"], focal_uncertainty=line * 0.8)
    )


def test_five_views_are_not_thin_four_are():
    assert "thin_calibration" not in _codes(_calib(frames_used=list("abcde")))
    assert "thin_calibration" in _codes(_calib(frames_used=list("abcd")))


def test_reprojection_exactly_on_the_line_is_not_high():
    # hypot(600, 800) is exactly 1000, so rms 1.5 is exactly the line.
    assert S.rms_fraction(1.5, (600, 800)) == S.HIGH_REPROJECTION_FRACTION
    assert "high_reprojection_error" not in _codes(_calib(rms=1.5))
    assert "high_reprojection_error" in _codes(_calib(rms=1.51))


def test_coverage_exactly_on_the_line_is_not_low():
    line = S.MIN_CALIBRATION_COVERAGE
    assert "low_calibration_coverage" not in _codes(_calib(coverage=line))
    assert "low_calibration_coverage" in _codes(_calib(coverage=line - 0.01))


# --- the checkerboard digest ---------------------------------------------------


def _member(name: str, data: bytes) -> bytes:
    enc = name.encode()
    return len(enc).to_bytes(8, "little") + enc + len(data).to_bytes(8, "little") + data


def test_the_digest_is_the_documented_length_prefixed_serialization(tmp_path):
    boards = tmp_path / "boards"
    boards.mkdir()
    (boards / "a.png").write_bytes(b"\x89PNG-first")
    (boards / "b.txt").write_bytes(b"notes")
    (boards / "sub").mkdir()  # a directory is not a member at all
    os.mkfifo(boards / "pipe")  # a FIFO is a named member with no bytes
    frames, digest = S._load_checkerboard_frames(str(boards))
    assert frames == [("a.png", b"\x89PNG-first"), ("b.txt", b"notes"), ("pipe", b"")]
    enc = b"pipe"
    expected = hashlib.sha256(
        _member("a.png", b"\x89PNG-first")
        + _member("b.txt", b"notes")
        + len(enc).to_bytes(8, "little")
        + enc
        + (0).to_bytes(8, "little")
    ).hexdigest()
    assert digest == expected
    assert S._checkerboard_digest(str(boards)) == expected


def test_a_member_opened_outside_the_roots_is_refused_by_its_own_name(
    tmp_path, monkeypatch
):
    inside = tmp_path / "inside"
    boards = inside / "boards"
    boards.mkdir(parents=True)
    member = boards / "v.png"
    member.write_bytes(b"inside")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    paths.set_roots([str(inside)])
    try:
        # Positive control: the member reads normally.
        assert S._load_checkerboard_frames(str(boards))[0] == [("v.png", b"inside")]
        # The window check_readable cannot close: its verdict names one file and
        # the open lands on another. The descriptor check refuses, naming the
        # member as the directory spells it.
        monkeypatch.setattr(S, "check_readable", lambda p: str(outside))
        with pytest.raises(paths.PathOutsideRootsError) as exc:
            S._load_checkerboard_frames(str(boards))
        assert str(exc.value).startswith(f"{str(member)!r} was opened as ")
    finally:
        paths.set_roots(None)


# --- the calibration cache ----------------------------------------------------


def test_the_cache_reuses_a_fit_and_holds_exactly_its_bound(tmp_path, monkeypatch):
    calls = []

    def fit(frames, row_corners, col_corners):
        calls.append([n for n, _ in frames])
        return _calib(frames_used=[n for n, _ in frames])

    monkeypatch.setattr(S, "_lens_cache", OrderedDict())
    monkeypatch.setattr(S, "LENS_CACHE_ENTRIES", 2)
    monkeypatch.setattr(S, "calibrate_lens_from_frames", fit)
    dirs = []
    for i in range(3):
        d = tmp_path / f"d{i}"
        d.mkdir()
        (d / f"f{i}.png").write_bytes(bytes([i]))
        dirs.append(str(d))

    first = S._lens_calibration(dirs[0], 6, 9)
    again = S._lens_calibration(dirs[0], 6, 9)
    assert again is first and calls == [["f0.png"]]
    S._lens_calibration(dirs[1], 6, 9)
    assert len(S._lens_cache) == 2  # at the bound, nothing evicted yet
    S._lens_calibration(dirs[2], 6, 9)
    assert len(S._lens_cache) == 2
    assert calls == [["f0.png"], ["f1.png"], ["f2.png"]]
    # d0 was the least recently used, so it went first: asking again refits it.
    S._lens_calibration(dirs[1], 6, 9)
    S._lens_calibration(dirs[0], 6, 9)
    assert calls == [["f0.png"], ["f1.png"], ["f2.png"], ["f0.png"]]


# --- the write refusals -------------------------------------------------------


def _boards_refusal(named: str) -> str:
    return (
        f"The corrected image would be written into the checkerboard directory "
        f"{named!r}, which is read as calibration input: every file there is a "
        "candidate frame and the cached calibration is keyed on the directory's "
        "contents, so the correction would be offered back to the next "
        "calibration and each image would refit from scratch. Nothing was "
        "written. Pass output_path pointing outside that directory, or move the "
        "image you are correcting out of it."
    )


def test_writing_into_the_boards_is_refused_word_for_word(tmp_path):
    boards = tmp_path / "boards"
    boards.mkdir()
    with pytest.raises(ValueError) as exc:
        S._refuse_writing_into_the_boards(
            str(boards / "x.png"), str(boards), "my boards"
        )
    assert str(exc.value) == _boards_refusal("my boards")
    # Positive control: a sibling directory is fine.
    S._refuse_writing_into_the_boards(str(tmp_path / "x.png"), str(boards), "b")


def test_write_new_image_refusals_word_for_word(tmp_path):
    img = np.zeros((4, 6, 3), np.uint8)
    link = tmp_path / "link.png"
    link.symlink_to(tmp_path / "nowhere.png")
    with pytest.raises(OSError) as exc:
        S._write_new_image(
            img, derived="d.png", output_path=str(link), derived_label="L"
        )
    assert str(exc.value) == (
        f"output_path {str(link)!r} is a symlink; images are written only to "
        "real files, so nothing was written. Remove the link or pass a "
        "different output_path."
    )
    with pytest.raises(OSError) as exc:
        S._write_new_image(
            img, derived="d.png", output_path=str(tmp_path), derived_label="L"
        )
    assert str(exc.value) == (
        f"output_path {str(tmp_path)!r} is a directory; pass the path of the "
        "file to create (it must not exist yet)."
    )
    # Positive control: a new path is written and returned.
    new = tmp_path / "new.png"
    assert S._write_new_image(
        img, derived="d.png", output_path=str(new), derived_label="L"
    ) == str(new)
    assert load_image(str(new)).shape == (4, 6, 3)


# --- the correction: routing, refusals, envelope ------------------------------

H, W = 60, 80  # not square, so the size's two axes cannot be swapped unseen


def _scene(path):
    rng = np.random.default_rng(3)
    cv2.imwrite(str(path), rng.integers(0, 255, (H, W, 3), dtype=np.uint8))
    return str(path)


def _pinhole(**kw):
    mtx = np.array([[100.0, 0.0, W / 2], [0.0, 100.0, H / 2], [0.0, 0.0, 1.0]])
    return _calib(mtx=mtx, shape=(H, W), **kw)


def _unreached(*a, **k):
    raise AssertionError("calibration ran: the early refusal should have stopped it")


def test_an_image_inside_the_boards_is_refused_before_any_calibration(
    tmp_path, monkeypatch
):
    boards = tmp_path / "boards"
    boards.mkdir()
    scene = _scene(boards / "scene.png")
    monkeypatch.setattr(S, "_lens_calibration", _unreached)
    named = str(boards) + "/"  # the caller's spelling is the one named
    with pytest.raises(ValueError) as exc:
        S._correct_lens_impl(scene, named, 6, 9)
    assert str(exc.value) == _boards_refusal(named)


def test_an_output_path_inside_the_boards_is_refused_before_any_calibration(
    tmp_path, monkeypatch
):
    boards = tmp_path / "boards"
    boards.mkdir()
    scene = _scene(tmp_path / "scene.png")
    monkeypatch.setattr(S, "_lens_calibration", _unreached)
    with pytest.raises(ValueError) as exc:
        S._correct_lens_impl(
            scene, str(boards), 6, 9, output_path=str(boards / "out.png")
        )
    assert str(exc.value) == _boards_refusal(str(boards))


def test_the_binding_guard_names_the_directory_as_the_caller_spelled_it(
    tmp_path, monkeypatch
):
    boards = tmp_path / "boards"
    boards.mkdir()
    safe = tmp_path / "safe"
    safe.mkdir()
    link = tmp_path / "out"
    link.symlink_to(safe)
    scene = _scene(tmp_path / "scene.png")

    def retarget(*a, **k):  # after the early refusal, before the write
        link.unlink()
        link.symlink_to(boards)
        return _pinhole()

    monkeypatch.setattr(S, "_lens_calibration", retarget)
    with pytest.raises(ValueError) as exc:
        S._correct_lens_impl(
            scene, str(boards), 6, 9, output_path=str(link / "corrected.png")
        )
    assert str(exc.value) == _boards_refusal(str(boards))
    assert list(boards.iterdir()) == []


def test_an_existing_output_path_is_refused_naming_the_derived_file(
    tmp_path, monkeypatch
):
    scene = _scene(tmp_path / "scene.png")
    taken = tmp_path / "taken.png"
    taken.write_bytes(b"mine")
    boards = tmp_path / "boards"
    boards.mkdir()
    monkeypatch.setattr(S, "_lens_calibration", lambda *a, **k: _pinhole())
    with pytest.raises(FileExistsError) as exc:
        S._correct_lens_impl(scene, str(boards), 6, 9, output_path=str(taken))
    assert str(exc.value) == (
        f"output_path {str(taken)!r} already exists. Pass a new path, or omit "
        "output_path to write (and overwrite) the derived "
        "<image>_undistorted.png next to the input."
    )
    assert taken.read_bytes() == b"mine"


def test_the_correction_envelope_whole(tmp_path, monkeypatch):
    scene = _scene(tmp_path / "scene.png")
    boards = tmp_path / "boards"
    boards.mkdir()
    calib = _pinhole(
        frames_outliers=(("o.png", 3.14159, "far above the others"),),
        median_view_rms=1.0,
        frames_duplicates=(("copy.png", "v0.png"),),
        focal_uncertainty=0.01,
        coverage=0.9,
    )
    monkeypatch.setattr(S, "_lens_calibration", lambda *a, **k: calib)
    res = S._correct_lens_impl(scene, str(boards), 6, 9)
    out = str(tmp_path / "scene_undistorted.png")
    corrected, info = undistort_image(load_image(scene), calib)
    png = res.pop("_png")
    assert res == {
        "corrected_image_path": out,
        "frames_used": 10,
        "frames_skipped": [],
        "frames_outliers": [
            {"name": "o.png", "rms_px": 3.14, "reason": "far above the others"}
        ],
        "frames_duplicates": [{"name": "copy.png", "same_as": "v0.png"}],
        "rms_reprojection_error": 0.5,
        "focal_uncertainty": 0.01,
        "valid_roi": info["valid_roi"],
        "crop_fraction": info["crop_fraction"],
        "residual_void_px": info["residual_void_px"],
        "board_coverage": 0.9,
        "corrected_size": [H, W],
        "overlay_scale": S.downscale(corrected)[1],
        "warnings": S._lens_advisories(calib, info, out),
    }
    assert list(corrected.shape[:2]) == [H, W]
    assert np.array_equal(load_image(out), corrected)
    assert png.startswith(b"\x89PNG")
