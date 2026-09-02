"""Lens-distortion calibration and correction.

Ground truth is synthetic: checkerboard views are rendered flat, warped to
fixed poses, then distorted through a KNOWN camera model (the same remap
direction cv2.undistortPoints defines), so "corrected" has a measurable
meaning — chessboard corner rows become collinear again — rather than merely
"different". Measured on the real fisheye tutorial photo, uncorrected
distortion inflated plant area 2.13x and was anisotropic (the same pot's rim,
face and base scaled 1.73x/1.31x/1.59x), which no px_per_mm can cancel; this
module is the reason a correction tool exists at all.
"""

import math
import os
import stat

import cv2
import numpy as np
import pytest

# A deliberately barrel-distorted camera. k1 is mild enough that
# findChessboardCorners still detects every view, strong enough that a corner
# row bends by several pixels — measurable straightness.
MTX = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.50, 0.20, 0.0, 0.0, 0.0])
ROWS, COLS = 6, 9  # inner corners


def _board(square=48):
    rows_sq, cols_sq = ROWS + 1, COLS + 1
    board = np.full((rows_sq * square, cols_sq * square), 255, np.uint8)
    for r in range(rows_sq):
        for c in range(cols_sq):
            if (r + c) % 2 == 0:
                board[r * square : (r + 1) * square, c * square : (c + 1) * square] = 0
    return board


def _distort(img, mtx=MTX, dist=DIST):
    """Apply the camera's distortion to an ideal pinhole image (remap via
    undistortPoints)."""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, mtx, dist, P=mtx).reshape(h, w, 2)
    return cv2.remap(
        img,
        und[..., 0],
        und[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


# Fixed RIGID poses of the board (Rodrigues rotation, translation in square
# units), projected through MTX. Earlier fixtures warped the board by arbitrary
# 2-D homographies, and a set of homographies implies its own intrinsics: that
# fixture calibrated to fx=217, cx=386, so every "against a true 400" claim was
# measured against a camera that did not exist. Deterministic on purpose — a
# flaky calibration test would teach nothing. The first pose is the most
# tilted of the centre views (its corner rows bend 4.6 px under DIST); it is
# the scene and the thumbnail source too. The set is spread to the frame's
# corners on purpose: the 1.8.2 poses covered 24% of the frame and the
# recovered model, exact where boards had been, was off by up to 659 px at
# the corners (panel audit of 1.8.2). These cover 48%.
POSES = [
    ((0.00, 0.35, 0.10), (-5.5, -2.5, 12.5)),
    ((0.20, 0.15, 0.00), (-8.6, -6.3, 11.0)),
    ((0.20, -0.15, 0.00), (-1.6, -6.3, 11.0)),
    ((-0.20, 0.15, 0.00), (-8.4, -1.2, 11.0)),
    ((0.15, 0.10, 0.00), (-11.0, -8.2, 14.0)),
    ((0.15, -0.10, 0.00), (0.8, -8.2, 14.0)),
    ((-0.15, 0.10, 0.00), (-11.0, 1.0, 14.0)),
    ((-0.15, -0.10, 0.00), (0.8, 1.0, 14.0)),
    ((-0.20, -0.15, 0.10), (-1.6, -0.7, 11.0)),
    ((0.35, 0.25, 0.00), (-4.5, -3.0, 10.0)),
    ((-0.35, -0.20, -0.15), (-4.0, -3.5, 10.5)),
    ((0.00, 0.00, 0.00), (-5.0, -3.5, 11.0)),
    ((0.10, 0.40, 0.20), (-7.0, -3.5, 12.0)),
    ((-0.15, -0.40, -0.20), (-2.5, -3.0, 12.0)),
]


def _view(rvec, tvec, size=(640, 480), mtx=MTX):
    """Render the flat board at a rigid pose through the PINHOLE part of MTX.

    The board plane is z=0 with one unit per square; its four corners are
    projected with cv2.projectPoints (no distortion — _distort applies DIST
    afterwards through the same MTX), and the flat render is warped onto
    those image points.
    """
    board = _board()
    bh, bw = board.shape
    square = bh / (ROWS + 1)
    base = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    world = np.float32([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]]) / square
    pts, _ = cv2.projectPoints(world, np.float32(rvec), np.float32(tvec), mtx, None)
    hom = cv2.getPerspectiveTransform(base, pts.reshape(4, 2).astype(np.float32))
    return cv2.warpPerspective(
        board,
        hom,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def _write_frames(directory, n=None):
    directory.mkdir(parents=True, exist_ok=True)
    for i, (rvec, tvec) in enumerate(POSES if n is None else POSES[:n]):
        cv2.imwrite(str(directory / f"view{i}.png"), _distort(_view(rvec, tvec)))


def _row_residual(gray):
    """Worst deviation of the outer corner rows from their own fitted lines."""
    found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS))
    assert found, "checkerboard not detected — the fixture itself is broken"
    grid = corners.reshape(ROWS, COLS, 2)
    residuals = []
    for row in (grid[0], grid[-1]):
        coeffs = np.polyfit(row[:, 0], row[:, 1], 1)
        residuals.append(
            float(np.max(np.abs(np.polyval(coeffs, row[:, 0]) - row[:, 1])))
        )
    return max(residuals)


@pytest.fixture(scope="module")
def calib_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("checkerboards") / "frames"
    _write_frames(d)
    return d


@pytest.fixture(scope="module")
def calibration(calib_dir):
    from plantcv_mcp.lens import calibrate_lens

    return calibrate_lens(str(calib_dir), row_corners=ROWS, col_corners=COLS)


def test_calibration_reports_every_frame_and_a_finite_rms(calibration):
    assert len(calibration.frames_used) == len(POSES)
    assert calibration.frames_skipped == []
    assert 0.0 < calibration.rms < 2.0
    assert calibration.mtx.shape == (3, 3)


def test_calibration_skips_undetectable_frames_by_name(tmp_path):
    from plantcv_mcp.lens import calibrate_lens

    d = tmp_path / "frames"
    _write_frames(d)
    cv2.imwrite(str(d / "blank.png"), np.full((480, 640), 255, np.uint8))
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    assert calib.frames_skipped == ["blank.png"]
    assert len(calib.frames_used) == len(POSES)


def test_calibration_refuses_too_few_detected_frames(tmp_path):
    """PlantCV's own checkerboard_calib crashes with a raw cv2 error when no
    frame matches; a calibration built from one or two frames is silently
    garbage. Both become one typed refusal that names the counts."""
    from plantcv_mcp.lens import (
        MIN_CALIBRATION_FRAMES,
        LensCalibrationError,
        calibrate_lens,
    )

    d = tmp_path / "frames"
    _write_frames(d, n=MIN_CALIBRATION_FRAMES - 1)
    cv2.imwrite(str(d / "blank.png"), np.full((480, 640), 255, np.uint8))
    with pytest.raises(LensCalibrationError, match=str(MIN_CALIBRATION_FRAMES)):
        calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    # Positive control: the full pose set calibrates through the same path.
    d2 = tmp_path / "enough"
    _write_frames(d2)
    calib = calibrate_lens(str(d2), row_corners=ROWS, col_corners=COLS)
    assert len(calib.frames_used) == len(POSES)


def test_undistortion_straightens_a_bent_corner_row(calibration):
    from plantcv_mcp.lens import undistort_image

    distorted = _distort(_view(*POSES[0]))
    before = _row_residual(distorted)
    corrected, info = undistort_image(
        cv2.cvtColor(distorted, cv2.COLOR_GRAY2BGR), calibration
    )
    after = _row_residual(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY))
    assert before > 3.0  # the fixture really is bent
    assert after < 0.5 * before  # and the correction really straightens it
    assert 0.0 <= info["crop_fraction"] < 1.0


def test_correct_lens_distortion_writes_next_to_the_image(tmp_path, calib_dir):
    from plantcv_mcp.server import _correct_lens_impl

    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    res = _correct_lens_impl(
        str(scene), str(calib_dir), row_corners=ROWS, col_corners=COLS
    )
    out = res["corrected_image_path"]
    assert out == str(tmp_path / "scene_undistorted.png")
    assert os.path.exists(out)
    assert res["frames_used"] == len(POSES)
    assert res["frames_skipped"] == []
    assert 0.0 < res["rms_reprojection_error"] < 2.0
    codes = [w["code"] for w in res["warnings"]]
    # The advisory that closes the dogfood loop: a px_per_mm calibrated on the
    # ORIGINAL image is invalid for the corrected one.
    assert "lens_corrected" in codes
    corrected = cv2.imread(out)
    assert corrected is not None


def test_a_thin_calibration_is_flagged(tmp_path):
    """Three or four frames calibrate — and deserve a caveat, not silence."""
    from plantcv_mcp.server import _correct_lens_impl

    d = tmp_path / "frames"
    _write_frames(d, n=4)
    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    res = _correct_lens_impl(str(scene), str(d), row_corners=ROWS, col_corners=COLS)
    assert "thin_calibration" in [w["code"] for w in res["warnings"]]


def test_an_explicit_output_path_refuses_to_overwrite(tmp_path, calib_dir):
    """The derived default name is the tool's own to overwrite; a user-supplied
    output_path pointing at an existing file is not."""
    from plantcv_mcp.server import _correct_lens_impl

    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    target = tmp_path / "precious.png"
    target.write_bytes(b"not an image, and not ours to replace")
    with pytest.raises(FileExistsError):
        _correct_lens_impl(
            str(scene),
            str(calib_dir),
            row_corners=ROWS,
            col_corners=COLS,
            output_path=str(target),
        )
    assert target.read_bytes() == b"not an image, and not ours to replace"


# --- panel audit of 1.8.0 (2026-08-31): pinning the confirmed findings ---


def test_a_wrong_resolution_image_is_refused(calibration):
    """Panel 1 (4/5 judges): a 1280x960 image through the 640x480 calibration
    came back 49x127 with no warning — intrinsics are in calibration-frame
    pixels. The mismatch must be a typed refusal, not silent garbage."""
    from plantcv_mcp.lens import CalibrationResolutionMismatchError, undistort_image

    big = cv2.resize(_distort(_view(*POSES[0])), (1280, 960))
    with pytest.raises(CalibrationResolutionMismatchError) as err:
        undistort_image(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR), calibration)
    assert "1280" in str(err.value) and "640" in str(err.value)
    # Positive control: the matching resolution still corrects.
    ok, _ = undistort_image(
        cv2.cvtColor(_distort(_view(*POSES[0])), cv2.COLOR_GRAY2BGR), calibration
    )
    assert ok.size > 0


def test_the_crop_contains_no_fabricated_pixels():
    """Panel 2 (codex, reproduced): with the exact fixture model and an all-127
    source, OpenCV's alpha=1 ROI [13,24,613,430] kept 566 exactly-black
    fabricated pixels. The crop must be computed from the TRUE valid mask."""
    from plantcv_mcp.lens import LensCalibration, undistort_image

    exact = LensCalibration(
        mtx=MTX,
        dist=DIST,
        rms=0.5,
        frames_used=["a"] * 8,
        frames_skipped=[],
        shape=(480, 640),
    )
    corrected, info = undistort_image(np.full((480, 640, 3), 127, np.uint8), exact)
    black = int((corrected == 0).all(axis=2).sum())
    assert black == 0
    # Positive control: something was genuinely cropped away.
    assert info["crop_fraction"] > 0.0
    assert corrected.shape[0] < 480 or corrected.shape[1] < 640


def test_ten_copies_of_one_pose_are_refused(tmp_path):
    """Panel 3: ten duplicates 'calibrate' to a LOW rms with wrong intrinsics
    — measured with the gate disabled on the true-camera fixture: rms 0.19,
    fx=252 and k1=-0.24 against 400 and -0.50. Copies are ONE board
    orientation, the degenerate end of the orientation-count refusal."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    d = tmp_path / "dups"
    d.mkdir()
    one = _distort(_view(*POSES[0]))
    for i in range(10):
        cv2.imwrite(str(d / f"dup{i}.png"), one)
    # Byte-identical copies are one observation (panel of 1.10.1) and are
    # named as such; the same board re-photographed from one pose is one
    # orientation, the degenerate end of the orientation-count refusal.
    with pytest.raises(LensCalibrationError, match="copies of another"):
        calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    d1 = tmp_path / "onepose"
    d1.mkdir()
    for i in range(10):
        cv2.imwrite(
            str(d1 / f"same{i}.png"),
            _distort(_view(POSES[0][0], POSES[0][1]))
            if i == 0
            else _distort(
                _view(
                    POSES[0][0],
                    (POSES[0][1][0] + 0.05 * i, POSES[0][1][1], POSES[0][1][2]),
                )
            ),
        )
    with pytest.raises(LensCalibrationError, match="orientation"):
        calibrate_lens(str(d1), row_corners=ROWS, col_corners=COLS)
    # Positive control: the distinct-pose set calibrates through the same path.
    d2 = tmp_path / "distinct"
    _write_frames(d2)
    calib = calibrate_lens(str(d2), row_corners=ROWS, col_corners=COLS)
    assert len(calib.frames_used) == len(POSES)


def test_checkerboard_symlinks_outside_roots_are_refused(tmp_path):
    """Panel 4 (security, reproduced): only the DIRECTORY was containment-
    checked; member symlinks to outside the roots were opened and calibrated.
    Members must pass the same check_readable contract as every other read."""
    from plantcv_mcp import paths
    from plantcv_mcp.paths import PathOutsideRootsError
    from plantcv_mcp.server import _correct_lens_impl

    outside = tmp_path / "outside"
    _write_frames(outside)
    allowed = tmp_path / "allowed"
    linkdir = allowed / "calib"
    linkdir.mkdir(parents=True)
    for i in range(len(POSES)):
        os.symlink(str(outside / f"view{i}.png"), str(linkdir / f"view{i}.png"))
    scene = allowed / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    paths.set_roots([str(allowed)])
    try:
        with pytest.raises(PathOutsideRootsError):
            _correct_lens_impl(
                str(scene), str(linkdir), row_corners=ROWS, col_corners=COLS
            )
        # Positive control: real files inside the root calibrate normally.
        realdir = allowed / "real"
        _write_frames(realdir)
        res = _correct_lens_impl(
            str(scene), str(realdir), row_corners=ROWS, col_corners=COLS
        )
        assert res["frames_used"] == len(POSES)
    finally:
        paths.set_roots(None)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_an_unreadable_member_file_is_skipped_not_fatal(tmp_path):
    """Panel 4b: one unreadable regular file crashed the digest before
    calibrate_lens could skip it. It must be named and skipped instead."""
    from plantcv_mcp.server import _correct_lens_impl

    d = tmp_path / "frames"
    _write_frames(d)
    locked = d / "locked.bin"
    locked.write_bytes(b"sealed")
    locked.chmod(0)
    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    try:
        res = _correct_lens_impl(str(scene), str(d), row_corners=ROWS, col_corners=COLS)
    finally:
        locked.chmod(0o600)
    assert res["frames_used"] == len(POSES)
    assert "locked.bin" in res["frames_skipped"]


def test_a_symlink_at_the_derived_output_is_refused(tmp_path, calib_dir):
    """Panel 5 (destructive, reproduced): a pre-existing scene_undistorted.png
    symlink was followed by the write and the victim file clobbered."""
    from plantcv_mcp.server import _correct_lens_impl

    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    victim = tmp_path / "victim.png"
    victim.write_bytes(b"PRECIOUS")
    os.symlink(str(victim), str(tmp_path / "scene_undistorted.png"))
    with pytest.raises(OSError, match="is a symlink"):  # the message, not the tmp path
        _correct_lens_impl(
            str(scene), str(calib_dir), row_corners=ROWS, col_corners=COLS
        )
    assert victim.read_bytes() == b"PRECIOUS"


def test_the_digest_cannot_be_collided_by_concatenation(tmp_path, calib_dir):
    """Panel 6 (codex's construction, reproduced): name||bytes concatenation
    made {0.png: A||"1.png"||B, 2.png: C} hash equal to {0.png: A, 1.png: B,
    2.png: C}. Length-prefixed serialization must separate them."""
    from plantcv_mcp.server import _checkerboard_digest

    a = (calib_dir / "view0.png").read_bytes()
    b = (calib_dir / "view1.png").read_bytes()
    c = (calib_dir / "view2.png").read_bytes()
    d1 = tmp_path / "dir1"
    d2 = tmp_path / "dir2"
    d1.mkdir()
    d2.mkdir()
    (d1 / "0.png").write_bytes(a + b"1.png" + b)
    (d1 / "2.png").write_bytes(c)
    (d2 / "0.png").write_bytes(a)
    (d2 / "1.png").write_bytes(b)
    (d2 / "2.png").write_bytes(c)
    assert _checkerboard_digest(str(d1)) != _checkerboard_digest(str(d2))


def test_thumbnails_sorting_first_do_not_hijack_the_calibration(tmp_path):
    """Panel 12: the lexically first DETECTED frame fixed the working size, so
    three thumbnails sorting first calibrated while eight full-resolution
    views were skipped. The majority shape must win."""
    from plantcv_mcp.lens import calibrate_lens

    d = tmp_path / "mixed"
    d.mkdir()
    small = cv2.resize(_distort(_view(*POSES[0])), (320, 240))
    found, _ = cv2.findChessboardCorners(small, (COLS, ROWS))
    assert found, "fixture thumbnail must itself be detectable"
    for i in range(3):
        cv2.imwrite(str(d / f"a_thumb{i}.png"), small)
    for i, (off, jig) in enumerate(POSES):
        cv2.imwrite(str(d / f"view{i}.png"), _distort(_view(off, jig)))
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    assert calib.shape == (480, 640)
    assert sorted(calib.frames_used) == sorted(
        f"view{i}.png" for i in range(len(POSES))
    )
    # The first thumbnail is skipped for its size; the other two are its
    # byte-for-byte copies and are named as such (panel of 1.10.1).
    assert set(calib.frames_skipped) == {"a_thumb0.png"}
    assert calib.frames_duplicates == (
        ("a_thumb1.png", "a_thumb0.png"),
        ("a_thumb2.png", "a_thumb0.png"),
    )


def test_high_reprojection_error_earns_an_advisory():
    """Panel 3b: the real tutorial calibration ran at rms 13 px and said
    nothing. The advisory assembly is pure, so it is tested directly."""
    from plantcv_mcp.lens import LensCalibration
    from plantcv_mcp.server import _lens_advisories

    noisy = LensCalibration(
        mtx=MTX,
        dist=DIST,
        rms=13.0,
        frames_used=["a"] * 9,
        frames_skipped=[],
        shape=(480, 640),
    )
    info = {
        "valid_roi": [0, 0, 600, 440],
        "crop_fraction": 0.2,
        "roi_degenerate": False,
        "residual_void_px": 0,
    }
    codes = [w["code"] for w in _lens_advisories(noisy, info, "/tmp/x.png")]
    assert "high_reprojection_error" in codes
    # Positive control: a tight calibration earns no such advisory.
    tight = LensCalibration(
        mtx=MTX,
        dist=DIST,
        rms=0.4,
        frames_used=["a"] * 9,
        frames_skipped=[],
        shape=(480, 640),
    )
    codes2 = [w["code"] for w in _lens_advisories(tight, info, "/tmp/x.png")]
    assert "high_reprojection_error" not in codes2


def test_the_degenerate_path_never_claims_no_crop_was_needed():
    """Panel 9: crop_fraction 0.0 on the degenerate path produced 'No void
    crop was needed' beside distortion_voids_remain — a self-contradiction."""
    from plantcv_mcp.lens import LensCalibration
    from plantcv_mcp.server import _lens_advisories

    calib = LensCalibration(
        mtx=MTX,
        dist=DIST,
        rms=0.5,
        frames_used=["a"] * 9,
        frames_skipped=[],
        shape=(480, 640),
    )
    info = {
        "valid_roi": [0, 0, 0, 0],
        "crop_fraction": 0.0,
        "roi_degenerate": True,
        "residual_void_px": 293000,
    }
    warnings = _lens_advisories(calib, info, "/tmp/x.png")
    codes = [w["code"] for w in warnings]
    assert "distortion_voids_remain" in codes
    lens_msg = next(w["message"] for w in warnings if w["code"] == "lens_corrected")
    assert "No void crop was needed" not in lens_msg


# --- mutation round 11 (2026-09-01): pinning the 1.8.1 guards the suite missed ---


def test_a_size_tie_goes_to_the_larger_frames(tmp_path):
    """Round 11: `shape` breaks a majority tie to the LARGER frame — the
    comment said so, and no fixture had a tie. Three thumbnails against
    three full-resolution views must calibrate at full resolution."""
    from plantcv_mcp.lens import calibrate_lens

    d = tmp_path / "tied"
    d.mkdir()
    small = cv2.resize(_distort(_view(*POSES[0])), (320, 240))
    for i in range(3):
        cv2.imwrite(str(d / f"a_thumb{i}.png"), small)
    _write_frames(d, n=3)
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    assert calib.shape == (480, 640)
    assert sorted(calib.frames_used) == ["view0.png", "view1.png", "view2.png"]


def test_two_views_are_refused_by_the_literal_minimum(tmp_path):
    """Round 11: the existing too-few test derives its fixture from
    MIN_CALIBRATION_FRAMES and survived the constant being set to 1 (it
    wrote zero frames and matched the '1' in 'Only 0 of 1'). Pin the
    number: two distinct views are refused, and the message says three."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    d = tmp_path / "two"
    _write_frames(d, n=2)
    with pytest.raises(LensCalibrationError, match="at least 3"):
        calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    # Positive control: three views is the floor and calibrates.
    d3 = tmp_path / "three"
    _write_frames(d3, n=3)
    assert (
        len(calibrate_lens(str(d3), row_corners=ROWS, col_corners=COLS).frames_used)
        == 3
    )


@pytest.mark.parametrize("bad_rms", [float("inf"), float("nan"), 1e12])
def test_a_meaningless_fit_is_refused(tmp_path, monkeypatch, bad_rms):
    """Round 11: no frame set this suite can build reaches the rms gate
    (mirrored views calibrate cleanly; duplicates are refused upstream), so
    the optimiser is stubbed to return the garbage the gate exists for —
    codex measured 5.95e10 on one duplicate-pose reproduction. The gate must
    refuse non-finite and absurd fits by name."""
    from plantcv_mcp import lens
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    d = tmp_path / "frames"
    _write_frames(d)
    # Positive control first, on the real optimiser: this set calibrates.
    assert 0.0 < calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS).rms < 2.0
    real = lens.cv2.calibrateCameraExtended

    def garbage(*args, **kwargs):
        _, *rest = real(*args, **kwargs)
        return (bad_rms, *rest)

    monkeypatch.setattr(lens.cv2, "calibrateCameraExtended", garbage)
    with pytest.raises(LensCalibrationError, match="meaningless"):
        calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)


def test_every_pixel_of_the_crop_is_real_source_data():
    """Round 11: the no-fabricated-pixels test only forbids exactly-black
    pixels; a validity mask of `> 0` instead of `== 255` keeps the
    void-blended border (measured: minimum pixel 8 on an all-127 source, and
    zero black pixels). Every pixel in the crop must be the source value."""
    from plantcv_mcp.lens import LensCalibration, undistort_image

    exact = LensCalibration(
        mtx=MTX,
        dist=DIST,
        rms=0.5,
        frames_used=["a"] * 8,
        frames_skipped=[],
        shape=(480, 640),
    )
    corrected, info = undistort_image(np.full((480, 640, 3), 127, np.uint8), exact)
    assert int(corrected.min()) == 127 and int(corrected.max()) == 127
    assert info["residual_void_px"] == 0
    assert info["crop_fraction"] > 0.0


def test_a_frame_too_small_to_crop_is_returned_whole_with_its_voids_counted():
    """Round 11: the degenerate branch (no usable all-valid rectangle) was
    exercised only through a hand-made info dict. A 30x30 frame under a
    strong model has no 16-px valid rectangle: it must come back uncropped,
    flagged degenerate, with residual_void_px equal to the black pixels
    actually in it."""
    from plantcv_mcp.lens import LensCalibration, undistort_image

    n = 30
    strong = LensCalibration(
        mtx=np.array([[25.0, 0.0, 15.0], [0.0, 25.0, 15.0], [0.0, 0.0, 1.0]]),
        dist=np.array([-0.8, 0.0, 0.0, 0.0, 0.0]),
        rms=0.5,
        frames_used=["a"] * 8,
        frames_skipped=[],
        shape=(n, n),
    )
    out, info = undistort_image(np.full((n, n, 3), 127, np.uint8), strong)
    assert info["roi_degenerate"] is True
    assert out.shape[:2] == (n, n)
    black = int((out == 0).all(axis=2).sum())
    assert black > 0
    assert info["residual_void_px"] == black
    assert info["crop_fraction"] == 0.0


def test_an_explicit_output_path_outside_the_roots_is_refused(tmp_path):
    """Round 11: output_path went through check_readable, and nothing pinned
    it — dropping the call lets a caller write outside the configured roots.
    Inside the roots the same call writes normally."""
    from plantcv_mcp import paths
    from plantcv_mcp.paths import PathOutsideRootsError
    from plantcv_mcp.server import _correct_lens_impl

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    scene = allowed / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    calib_copy = allowed / "calib"
    _write_frames(calib_copy)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    paths.set_roots([str(allowed)])
    try:
        with pytest.raises(PathOutsideRootsError):
            _correct_lens_impl(
                str(scene),
                str(calib_copy),
                row_corners=ROWS,
                col_corners=COLS,
                output_path=str(elsewhere / "out.png"),
            )
        assert not (elsewhere / "out.png").exists()
        # Positive control: an explicit path inside the roots is written.
        inside = allowed / "out.png"
        res = _correct_lens_impl(
            str(scene),
            str(calib_copy),
            row_corners=ROWS,
            col_corners=COLS,
            output_path=str(inside),
        )
        assert res["corrected_image_path"] == str(inside)
        assert inside.exists()
    finally:
        paths.set_roots(None)


def test_a_cropped_frame_says_so():
    """Round 11: the three-way crop note's middle branch had no assertion; a
    cropped frame must not be described as needing no crop."""
    from plantcv_mcp.lens import LensCalibration
    from plantcv_mcp.server import _lens_advisories

    calib = LensCalibration(
        mtx=MTX,
        dist=DIST,
        rms=0.5,
        frames_used=["a"] * 9,
        frames_skipped=[],
        shape=(480, 640),
    )
    cropped = {
        "valid_roi": [23, 24, 594, 431],
        "crop_fraction": 0.17,
        "roi_degenerate": False,
        "residual_void_px": 0,
    }
    msg = next(
        w
        for w in _lens_advisories(calib, cropped, "/tmp/x.png")
        if w["code"] == "lens_corrected"
    )["message"]
    assert "cropped" in msg and "17%" in msg
    assert "No void crop was needed" not in msg
    # Positive control: an uncropped, void-free frame says exactly that.
    whole = dict(cropped, valid_roi=[0, 0, 640, 480], crop_fraction=0.0)
    msg2 = next(
        w
        for w in _lens_advisories(calib, whole, "/tmp/x.png")
        if w["code"] == "lens_corrected"
    )["message"]
    assert "No void crop was needed" in msg2


# --- 2026-09-01: the fixture must have a camera for "true" to mean anything ---


def test_calibration_recovers_the_synthetic_camera(calibration):
    """Every earlier 'fx=224 against a true 400' claim compared against a
    truth that did not exist: views were built from arbitrary homographies,
    which imply their own intrinsics (the pristine set calibrated to fx=217,
    cx=386). Views are now rigid poses projected through MTX, so the
    calibration must RECOVER MTX and DIST — the assertion that makes the
    duplicate-pose and mirrored-view measurements meaningful."""
    fx, fy = calibration.mtx[0, 0], calibration.mtx[1, 1]
    cx, cy = calibration.mtx[0, 2], calibration.mtx[1, 2]
    assert abs(fx - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert abs(fy - MTX[1, 1]) < 0.02 * MTX[1, 1]
    assert abs(cx - MTX[0, 2]) < 3.0 and abs(cy - MTX[1, 2]) < 3.0
    d = calibration.dist.ravel()
    assert abs(float(d[0]) - DIST[0]) < 0.1 * abs(DIST[0])
    assert abs(float(d[1]) - DIST[1]) < 0.1  # measured 0.200 vs 0.20
    assert calibration.rms < 1.0  # measured 0.48; asserted, not pinned


def _forward_map_error(calib):
    """Pixel error of the correction FIELD the tool applies (initUndistort-
    RectifyMap, the same map cv2.undistort builds) against the declared
    camera, over the whole frame — not just where boards were.

    Both maps are built at the RECOVERED calibration's output camera, which
    is the matrix undistort_image hands to cv2.undistort (panel of 1.9.0:
    building both at the truth's output camera compared the two distortion
    models under a destination the tool never uses — on the weakly-
    conditioned ±7° set that oracle read 139 px where the applied field was
    391 px, and a constructed calibration passed it at 20 px with the applied
    field 25 px off)."""
    h, w = calib.shape
    newm, _ = cv2.getOptimalNewCameraMatrix(calib.mtx, calib.dist, (w, h), 1, (w, h))
    tx, ty = cv2.initUndistortRectifyMap(MTX, DIST, None, newm, (w, h), cv2.CV_32FC1)
    rx, ry = cv2.initUndistortRectifyMap(
        calib.mtx, calib.dist, None, newm, (w, h), cv2.CV_32FC1
    )
    return np.hypot(tx - rx, ty - ry)


def test_the_recovered_correction_field_holds_across_the_whole_frame(calibration):
    """Panel of 1.8.2 (codex, reproduced): the recovery test checked k1 and
    the intrinsics, and the model was exact where the boards had been — and
    off by up to 659 px at the frame corners, because a free k3 (-0.16) folded
    the polynomial over outside the 24% of the frame the boards covered. On
    the spread fixture the free fit is worse still (fx 655, k3 -5.4, map error
    31,000 px); fixing k3 recovers k1/k2 to the second decimal with the map
    within 10 px at the 95th percentile and 25 px at the worst pixel (measured
    9.3 max, 4.6 at the 95th percentile, at the recovered output camera).
    The correction is a FIELD; the oracle must look at the whole of it, and
    at the output camera the tool actually uses (panel of 1.9.0)."""
    err = _forward_map_error(calibration)
    assert float(np.percentile(err, 95)) < 10.0
    assert float(err.max()) < 25.0
    assert float(calibration.dist.ravel()[4]) == 0.0  # k3 is not fitted
    # The recovered output camera must itself be the true one's: the map
    # comparison above is blind to a shared destination that is wrong.
    h, w = calibration.shape
    true_newm, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 1, (w, h))
    rec_newm, _ = cv2.getOptimalNewCameraMatrix(
        calibration.mtx, calibration.dist, (w, h), 1, (w, h)
    )
    assert abs(rec_newm[0, 0] - true_newm[0, 0]) < 0.02 * true_newm[0, 0]
    assert abs(rec_newm[1, 1] - true_newm[1, 1]) < 0.02 * true_newm[1, 1]
    assert abs(rec_newm[0, 2] - true_newm[0, 2]) < 5.0
    assert abs(rec_newm[1, 2] - true_newm[1, 2]) < 5.0


def test_mirrored_frames_are_a_documented_limit_not_a_symmetry(tmp_path):
    """1.8.2 claimed mirrored sets recover the same camera; the panel showed
    that held only because the fixture is centred with no tangential term.
    For a camera with cx=285 and p2=0.012: all-mirrored frames recover the
    REFLECTED camera (cx ≈ 639-285, p2 sign flipped — consistent, so a
    mirrored scene corrects correctly), while a set with SOME frames mirrored
    returns a camera that is neither, at a low rms, and nothing in the
    calibration can tell. Pinned as the limit it is."""
    from plantcv_mcp.lens import calibrate_lens

    mtx = np.array([[400.0, 0.0, 285.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.array([-0.5, 0.2, 0.0, 0.012, 0.0])
    views = [_distort(_view(r, t, mtx=mtx), mtx, dist) for r, t in POSES]
    results = {}
    for label, pick in (
        ("plain", lambda i: False),
        ("all", lambda i: True),
        ("half", lambda i: i % 2 == 1),
    ):
        d = tmp_path / label
        d.mkdir()
        for i, img in enumerate(views):
            if pick(i):
                img = np.ascontiguousarray(img[:, ::-1])
            cv2.imwrite(str(d / f"view{i}.png"), img)
        results[label] = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    plain, allm, half = results["plain"], results["all"], results["half"]
    assert abs(plain.mtx[0, 2] - 285.0) < 3.0
    assert abs(allm.mtx[0, 2] - (639.0 - 285.0)) < 3.0
    assert abs(float(allm.dist.ravel()[3]) + 0.012) < 0.003
    # The mixed set: a principal point belonging to neither camera, low rms.
    assert abs(half.mtx[0, 2] - 285.0) > 20.0 and abs(half.mtx[0, 2] - 354.0) > 20.0
    assert half.rms < 1.0


# --- panel audit of 1.8.2 (2026-09-01): pinning the confirmed findings ---


def _write_pose_set(directory, poses):
    directory.mkdir(parents=True, exist_ok=True)
    for i, (r, t) in enumerate(poses):
        cv2.imwrite(str(directory / f"view{i}.png"), _distort(_view(r, t)))


def test_frontal_views_at_different_positions_are_refused(tmp_path):
    """Panel 1 (3/3 judges; reproduced): eight translations of a fronto-
    parallel board pass a corner-displacement gate and calibrate to fx=228
    (true 400) at rms 0.19; depth changes to 333; in-plane rotations to 314 at
    rms 0.074. Zhang's method needs distinct board ORIENTATIONS; translation
    supplies none. Refuse, naming the orientation count."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    frontal = [
        ((0.0, 0.0, 0.0), (-4.5 + dx, -3.2 + dy, 12.0))
        for dx, dy in [
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, -1),
            (1.5, -0.5),
        ]
    ]
    _write_pose_set(tmp_path / "frontal", frontal)
    with pytest.raises(LensCalibrationError, match="orientation") as err:
        calibrate_lens(str(tmp_path / "frontal"), row_corners=ROWS, col_corners=COLS)
    assert "1 distinct" in str(err.value)
    # Positive control: the same eight positions with the board tilted
    # differently at each calibrate to the true camera.
    _write_pose_set(tmp_path / "tilted", POSES[:8])
    calib = calibrate_lens(str(tmp_path / "tilted"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]


def test_two_orientations_however_many_frames_are_refused(tmp_path):
    """Panel 1 (codex's reproduction): four copies each of two poses pass the
    displacement gate and 'calibrate' to fx=883 with k1=-3.6 at rms 0.59.
    Two orientations leave the intrinsics undetermined; three is the floor."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    # Four positions each (byte-identical copies would count once).
    def slid(pose, k):
        (r, (x, y, z)) = pose
        return (r, (x + 0.6 * k, y + 0.4 * k, z))

    two = [slid(POSES[1], k) for k in range(4)] + [slid(POSES[9], k) for k in range(4)]
    _write_pose_set(tmp_path / "two", two)
    with pytest.raises(LensCalibrationError, match="2 distinct"):
        calibrate_lens(str(tmp_path / "two"), row_corners=ROWS, col_corners=COLS)
    # Positive control: three orientations, three positions each, calibrate.
    _write_pose_set(
        tmp_path / "three",
        [slid(p, k) for k in range(3) for p in (POSES[0], POSES[1], POSES[9])],
    )
    calib = calibrate_lens(str(tmp_path / "three"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.03 * MTX[0, 0]


def test_exif_rotated_frames_and_scenes_decode_alike(tmp_path):
    """Panel 4 (codex; reproduced): identical EXIF-orientation-6 JPEG bytes
    decoded to 640x480 through the calibration path (IMREAD_GRAYSCALE honours
    EXIF) and 480x640 through the scene path (IMREAD_UNCHANGED ignores it), so
    a camera's own frames were refused as a resolution mismatch. Both paths
    must read the stored raster the same way."""
    import io

    from PIL import Image

    from plantcv_mcp.imaging import decode_image
    from plantcv_mcp.lens import calibrate_lens_from_frames

    frames = []
    for i, (r, t) in enumerate(POSES):
        gray = _distort(_view(r, t))
        im = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        exif = Image.Exif()
        exif[0x0112] = 6
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=95, exif=exif.tobytes())
        frames.append((f"view{i}.jpg", buf.getvalue()))
    scene = decode_image(frames[0][1], "view0.jpg")
    calib = calibrate_lens_from_frames(frames, row_corners=ROWS, col_corners=COLS)
    assert calib.shape == scene.shape[:2]
    assert len(calib.frames_used) == len(POSES)


def test_fit_thresholds_are_fractions_of_the_frame_not_pixels():
    """Panel 5 (3/3 judges): the same geometric fit scaled 8x crossed the 5-px
    advisory and the 100-px refusal on resolution alone; the real 2880-px set
    fits at 13 px (0.45% of width). Thresholds are fractions of the frame
    diagonal, so one fit means one thing at any resolution."""
    from plantcv_mcp.lens import LensCalibration
    from plantcv_mcp.server import _lens_advisories

    info = {
        "valid_roi": [0, 0, 1, 1],
        "crop_fraction": 0.1,
        "roi_degenerate": False,
        "residual_void_px": 0,
    }

    def codes(rms, shape):
        calib = LensCalibration(
            mtx=MTX,
            dist=DIST,
            rms=rms,
            frames_used=["a"] * 9,
            frames_skipped=[],
            shape=shape,
            coverage=0.7,
        )
        return [w["code"] for w in _lens_advisories(calib, info, "/tmp/x.png")]

    assert "high_reprojection_error" in codes(3.0, (480, 640))
    assert "high_reprojection_error" not in codes(3.0, (1920, 2880))
    assert "high_reprojection_error" in codes(13.0, (1920, 2880))  # the real set


def test_low_board_coverage_earns_an_advisory(tmp_path, calibration):
    """Panel 3 (codex; reproduced): a calibration is exact where boards were
    and extrapolated everywhere else — 24% coverage left the model 31 px off
    at the corners even with k3 fixed. The tool reports the covered fraction
    and says so below 40%."""
    from plantcv_mcp.lens import calibrate_lens
    from plantcv_mcp.server import _lens_advisories

    info = {
        "valid_roi": [0, 0, 1, 1],
        "crop_fraction": 0.1,
        "roi_degenerate": False,
        "residual_void_px": 0,
    }
    assert calibration.coverage >= 0.4
    assert "low_calibration_coverage" not in [
        w["code"] for w in _lens_advisories(calibration, info, "/tmp/x.png")
    ]
    _write_pose_set(tmp_path / "centre", POSES[:1] + POSES[9:12])
    centre = calibrate_lens(
        str(tmp_path / "centre"), row_corners=ROWS, col_corners=COLS
    )
    assert centre.coverage < 0.4
    warnings = _lens_advisories(centre, info, "/tmp/x.png")
    low = next(w for w in warnings if w["code"] == "low_calibration_coverage")
    assert f"{centre.coverage:.0%}" in low["message"]


def test_a_member_swapped_for_a_symlink_after_the_check_is_not_read(
    tmp_path, monkeypatch
):
    """Panel 6 (codex; security): check_readable's resolved path was discarded
    and the original spelling reopened, so a member replaced by an outside
    symlink between the check and the open was read and calibrated. The open
    must bind to the checked path and never follow a link."""
    from plantcv_mcp import server
    from plantcv_mcp.server import _load_checkerboard_frames

    outside = tmp_path / "outside"
    _write_frames(outside)
    calib = tmp_path / "calib"
    _write_frames(calib)
    target = calib / "view0.png"
    real_check = server.check_readable

    def swap_after_check(path):
        real = real_check(path)
        if os.path.basename(path) == "view0.png":
            os.unlink(target)
            os.symlink(str(outside / "view3.png"), str(target))
        return real

    monkeypatch.setattr(server, "check_readable", swap_after_check)
    frames, _ = _load_checkerboard_frames(str(calib))
    by_name = dict(frames)
    assert by_name["view0.png"] != (outside / "view3.png").read_bytes()
    assert by_name["view0.png"] == b""  # skipped, not followed
    # Positive control: the untouched members were read normally.
    assert by_name["view1.png"] == (calib / "view1.png").read_bytes()


def test_a_hard_link_at_the_derived_output_leaves_the_linked_file_intact(
    tmp_path, calib_dir
):
    """Panel 7 (codex; destructive): O_NOFOLLOW guards a symlink at the name,
    not a hard link — a hard-linked scene_undistorted.png was opened and
    truncated (800 -> 79 bytes). The write must never open an existing inode:
    it goes to a fresh sibling and replaces the directory entry."""
    from plantcv_mcp.server import _correct_lens_impl

    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    victim = tmp_path / "victim.bin"
    victim.write_bytes(b"precious" * 100)
    os.link(str(victim), str(tmp_path / "scene_undistorted.png"))
    with pytest.raises(OSError, match="hard link"):
        _correct_lens_impl(
            str(scene), str(calib_dir), row_corners=ROWS, col_corners=COLS
        )
    assert victim.read_bytes() == b"precious" * 100
    # Positive control: with the squatter gone, the derived name is written
    # and a re-run overwrites it (the tool's own file) without complaint.
    os.unlink(str(tmp_path / "scene_undistorted.png"))
    for _ in range(2):
        res = _correct_lens_impl(
            str(scene), str(calib_dir), row_corners=ROWS, col_corners=COLS
        )
    assert cv2.imread(res["corrected_image_path"]) is not None
    assert victim.read_bytes() == b"precious" * 100


# --- mutation round 12 (2026-09-01): pinning the 1.9.0 guards the suite missed ---


def test_board_coverage_is_reported_in_the_result(tmp_path, calib_dir, calibration):
    """Round 12: the advisory read calib.coverage but nothing pinned the
    result field a caller would use to decide whether to reshoot."""
    from plantcv_mcp.server import _correct_lens_impl

    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    res = _correct_lens_impl(
        str(scene), str(calib_dir), row_corners=ROWS, col_corners=COLS
    )
    assert abs(res["board_coverage"] - calibration.coverage) < 1e-9
    assert 0.4 <= res["board_coverage"] <= 1.0


def test_no_partial_file_is_left_beside_the_output(tmp_path, calib_dir):
    """Round 12: write_image stages the bytes in a sibling `.partial` file;
    a leftover would be a stray file in the user's image directory on every
    call. Both the derived and the explicit path must leave exactly the
    output behind."""
    from plantcv_mcp.server import _correct_lens_impl

    d = tmp_path / "shots"
    d.mkdir()
    scene = d / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    _correct_lens_impl(str(scene), str(calib_dir), row_corners=ROWS, col_corners=COLS)
    _correct_lens_impl(
        str(scene),
        str(calib_dir),
        row_corners=ROWS,
        col_corners=COLS,
        output_path=str(d / "explicit.png"),
    )
    assert sorted(p.name for p in d.iterdir()) == [
        "explicit.png",
        "scene.png",
        "scene_undistorted.png",
    ]


def test_one_tilt_slid_around_the_frame_is_one_orientation(tmp_path):
    """Round 12: the 5° cluster angle's lower side was unpinned — at 0.05° a
    board held at ONE tilt and slid to eight positions reads as six
    orientations (its recovered normals jitter by a few tenths of a degree)
    and calibrates to fx=354. It is one orientation and must be refused."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    slid = [
        ((0.3, 0.0, 0.0), (-4.5 + dx, -3.2 + dy, 12.5))
        for dx, dy in [
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (-1, -1),
            (1.5, -0.5),
        ]
    ]
    _write_pose_set(tmp_path / "slid", slid)
    with pytest.raises(LensCalibrationError, match="1 distinct"):
        calibrate_lens(str(tmp_path / "slid"), row_corners=ROWS, col_corners=COLS)
    # Positive control: the same board tilted through eight angles about one
    # axis is eight orientations and calibrates to the true camera.
    tilted = [
        ((a, 0.0, 0.0), (-4.5, -3.2, 12.5))
        for a in (-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4)
    ]
    _write_pose_set(tmp_path / "tilted", tilted)
    calib = calibrate_lens(str(tmp_path / "tilted"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]


def test_a_symlinked_member_inside_the_roots_is_read_not_skipped(tmp_path):
    """Round 12: the member open binds to the path check_readable resolved.
    Opening the member's own name with O_NOFOLLOW instead would turn a
    legitimate symlink to a frame INSIDE the roots into a silent skip. The
    swap test could not see that (its member starts as a regular file)."""
    from plantcv_mcp import paths
    from plantcv_mcp.server import _load_checkerboard_frames

    allowed = tmp_path / "allowed"
    real = allowed / "real"
    _write_frames(real)
    linked = allowed / "linked"
    linked.mkdir()
    for i in range(len(POSES)):
        os.symlink(str(real / f"view{i}.png"), str(linked / f"view{i}.png"))
    paths.set_roots([str(allowed)])
    try:
        frames, _ = _load_checkerboard_frames(str(linked))
    finally:
        paths.set_roots(None)
    assert len(frames) == len(POSES)
    assert all(data == (real / name).read_bytes() for name, data in frames)


# --- panel audit of 1.9.0 (2026-09-01): pinning the confirmed findings ---


def _applied_field_error(calib, mtx=MTX, dist=DIST):
    """Error of the map the TOOL applies: both fields built at the recovered
    calibration's own output camera, exactly as undistort_image does."""
    h, w = calib.shape
    newm, _ = cv2.getOptimalNewCameraMatrix(calib.mtx, calib.dist, (w, h), 1, (w, h))
    tx, ty = cv2.initUndistortRectifyMap(mtx, dist, None, newm, (w, h), cv2.CV_32FC1)
    rx, ry = cv2.initUndistortRectifyMap(
        calib.mtx, calib.dist, None, newm, (w, h), cv2.CV_32FC1
    )
    return np.hypot(tx - rx, ty - ry)


def test_small_symmetric_tilts_are_loose_not_wrong(tmp_path):
    """Panel of 1.10.1 (reproduced): 1.10.0 refused a board tilted -7/0/+7
    degrees about ONE axis as 'weakly conditioned' on the strength of a fit
    that was a wrong START, not weak data — from a start at the frame width
    the same views calibrate to fx 392 with the correction 16 px at worst.
    The fit's own uncertainty on the focal length is what can honestly be
    said: the ±7 degree two-axis set (3.6%, correction 54 px — the documented
    soft spot) is accepted and the looseness named; the ±3 degree set
    genuinely does not determine the camera and is refused."""
    from plantcv_mcp.lens import (
        FOCAL_UNCERTAINTY_ADVISORY,
        LensCalibrationError,
        calibrate_lens,
    )
    from plantcv_mcp.server import _lens_advisories

    info = {"roi_degenerate": False, "residual_void_px": 0, "crop_fraction": 0.1}
    weak = [((0.12 * ((i % 3) - 1), 0.0, 0.0), t) for i, (_, t) in enumerate(POSES)]
    _write_pose_set(tmp_path / "weak", weak)
    calib = calibrate_lens(str(tmp_path / "weak"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.03 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 25.0
    # The soft spot: ±7 degrees about two axes, accepted and said.
    tilts = [(-0.12, 0.0, 0.0), (0.0, 0.12, 0.0), (0.12, 0.0, 0.0), (0.0, -0.12, 0.0)]
    soft = [(tilts[i % 4], t) for i, (_, t) in enumerate(POSES)]
    _write_pose_set(tmp_path / "soft", soft)
    loose = calibrate_lens(str(tmp_path / "soft"), row_corners=ROWS, col_corners=COLS)
    assert loose.focal_uncertainty > FOCAL_UNCERTAINTY_ADVISORY
    codes = {w["code"] for w in _lens_advisories(loose, info, "x.png")}
    assert "focal_length_uncertain" in codes
    # Positive control: the fixture is tight, and says nothing.
    _write_pose_set(tmp_path / "five", POSES[:5])
    five = calibrate_lens(str(tmp_path / "five"), row_corners=ROWS, col_corners=COLS)
    assert five.focal_uncertainty < FOCAL_UNCERTAINTY_ADVISORY
    assert "focal_length_uncertain" not in {
        w["code"] for w in _lens_advisories(five, info, "x.png")
    }
    # Negative control: ±3 degrees about one axis is one orientation.
    flat = [((0.052 * ((i % 3) - 1), 0.0, 0.0), t) for i, (_, t) in enumerate(POSES)]
    _write_pose_set(tmp_path / "flat", flat)
    with pytest.raises(LensCalibrationError, match="orientation"):
        calibrate_lens(str(tmp_path / "flat"), row_corners=ROWS, col_corners=COLS)


def test_outlier_frames_are_dropped_named_and_the_camera_recovered(
    tmp_path, calibration
):
    """Found while reproducing the panel of 1.9.0: per-view residuals were
    never examined. PlantCV's own tutorial set carries one frame at 37 px rms
    against a median of 4, which moves fx by 13%; this synthetic one-sided
    20/26/40 degree set carries three frames at four times the median and
    'calibrated' to fx 612 with the field 2814 px wrong at rms 1.6 (only the
    generic advisory). Frames above 3x the median AND 0.25% of the diagonal
    are dropped by name and the camera refitted."""
    from plantcv_mcp.lens import calibrate_lens

    tilts = [20, 26, 40]
    poses = [
        ((math.radians(tilts[i % 3]), 0.0, 0.0), t) for i, (_, t) in enumerate(POSES)
    ]
    _write_pose_set(tmp_path / "onesided", poses)
    calib = calibrate_lens(
        str(tmp_path / "onesided"), row_corners=ROWS, col_corners=COLS
    )
    dropped = [name for name, _, _ in calib.frames_outliers]
    # Judged one at a time against the views around it, from a fit that is
    # no longer a wrong start, two of the three stand out (the third fits
    # once the camera is right); the set is a subset of the three either way.
    assert dropped and set(dropped) <= {"view4.png", "view6.png", "view7.png"}
    assert all(px > 2.0 for _, px, _ in calib.frames_outliers)
    assert all("median" in why for _, _, why in calib.frames_outliers)
    assert not set(dropped) & set(calib.frames_used)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 15.0
    # The tool says so, naming the frames and their residuals.
    from plantcv_mcp.server import _lens_advisories

    info = {"roi_degenerate": False, "residual_void_px": 0, "crop_fraction": 0.1}
    codes = {w["code"]: w["message"] for w in _lens_advisories(calib, info, "x.png")}
    assert "outlier_frames_dropped" in codes
    assert all(name in codes["outlier_frames_dropped"] for name in dropped)
    # Positive control: the fixture's own residual spread (0.08-1.14 px) is
    # rendering, not a bad frame; nothing is dropped from it and nothing said.
    assert calibration.frames_outliers == ()
    assert len(calibration.frames_used) == len(POSES)
    quiet = {w["code"] for w in _lens_advisories(calibration, info, "x.png")}
    assert "outlier_frames_dropped" not in quiet


def test_orientation_count_is_independent_of_frame_order_and_not_a_chain():
    """Panel of 1.9.0 (three judges): greedy first-fit packing counted the
    tilts 0/4/8/12/16 degrees as three orientations in one filename order and
    two in another. Panel of 1.10.1 (every judge; reproduced): the transitive
    fix made that chain ONE orientation and refused a board swept smoothly
    through 40 degrees that calibrated to 1 px. Pack in a canonical order:
    the count is a property of the set, and a sweep has spread."""
    from plantcv_mcp.lens import _distinct_orientations

    rvecs = {
        a: cv2.Rodrigues(cv2.Rodrigues(np.float32((math.radians(a), 0.0, 0.0)))[0])[0]
        for a in (0, 4, 8, 12, 16)
    }
    chain_a = _distinct_orientations([rvecs[a] for a in (0, 8, 16, 4, 12)])
    chain_b = _distinct_orientations([rvecs[a] for a in (4, 12, 0, 8, 16)])
    assert chain_a == chain_b == 3
    # Positive control: three tilts 8 degrees apart are three in any order,
    # and copies of one tilt are one.
    three = [rvecs[a] for a in (0, 8, 16)]
    assert _distinct_orientations(three) == _distinct_orientations(three[::-1]) == 3
    assert _distinct_orientations([rvecs[8]] * 4) == 1


def test_a_smooth_sweep_is_accepted(tmp_path):
    """Panel of 1.10.1 (every judge; reproduced): a board tilted 0, 4, 8 ...
    40 degrees about one axis — a phone video of a nodding board — was refused
    as one orientation. It determines the camera to about a pixel."""
    from plantcv_mcp.lens import calibrate_lens

    sweep = [
        ((math.radians(4 * i), 0.0, 0.0), t) for i, (_, t) in enumerate(POSES[:11])
    ]
    _write_pose_set(tmp_path / "sweep", sweep)
    calib = calibrate_lens(str(tmp_path / "sweep"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 25.0


def test_a_wrong_start_no_longer_yields_a_wrong_camera(tmp_path):
    """Panel of 1.10.1 (found while reproducing): thirteen of the fixture's
    own fourteen honest views — view 12 left out — calibrated to fx 780
    against 400 (k1 -1.96, rms 1.8 px, no outlier, every gate quiet) with the
    correction 272 px wrong: OpenCV's distortion-blind initial focal length
    (982 for this subset) led the optimiser into a wrong basin, and 41 of the
    fixture's 91 twelve-view subsets did the same. The fit now starts from
    several focal lengths and keeps the lowest residual."""
    from plantcv_mcp.lens import calibrate_lens

    poses = POSES[:12] + POSES[13:]
    _write_pose_set(tmp_path / "sub", poses)
    calib = calibrate_lens(str(tmp_path / "sub"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 25.0
    assert calib.frames_outliers == ()
    # The mechanism, pinned: OpenCV's own start on these corners is the wrong
    # basin (if this ever passes, the multi-start is no longer load-bearing
    # on this fixture and a harder one is needed).
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    obj, img = [], []
    for r, t in poses:
        gray = _distort(_view(r, t))
        found, corners = cv2.findChessboardCorners(gray, (COLS, ROWS))
        assert found
        obj.append(objp)
        img.append(
            cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
        )
    cold_rms, cold_mtx, *_ = cv2.calibrateCamera(
        obj, img, (640, 480), None, None, flags=cv2.CALIB_FIX_K3
    )
    assert abs(cold_mtx[0, 0] - MTX[0, 0]) > 0.5 * MTX[0, 0]
    assert cold_rms > 3 * calib.rms


def test_the_documented_open_limit_of_1_10_0_was_a_wrong_start(tmp_path):
    """1.10.0 documented a one-sided 10/20/30 degree set that 'still fits fx
    670 under the rms advisory' as an open limit no statistic separated.
    Panel of 1.10.1 (reproduced): from a start at the frame width the same
    views fit fx 393 at a LOWER residual (0.71 vs 1.13 px). Closed."""
    from plantcv_mcp.lens import calibrate_lens

    tilts = [10, 20, 30]
    poses = [
        ((math.radians(tilts[i % 3]), 0.0, 0.0), t) for i, (_, t) in enumerate(POSES)
    ]
    _write_pose_set(tmp_path / "onesided", poses)
    calib = calibrate_lens(
        str(tmp_path / "onesided"), row_corners=ROWS, col_corners=COLS
    )
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.03 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 25.0


def _shear(img, amount):
    """An image sheared as a rolling shutter shears a panning frame."""
    warp = np.float32([[1.0, amount, 0.0], [0.0, 1.0, 0.0]])
    return cv2.warpAffine(img, warp, (img.shape[1], img.shape[0]), borderValue=255)


def _ripple(img, amplitude, phase):
    """A smooth non-projective warp: a board that was not flat."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = xx + amplitude * np.sin(yy / 37 + phase) * np.sin(xx / 53 + phase)
    my = yy + amplitude * np.sin(xx / 41 - phase) * np.sin(yy / 47 + phase)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=255)


def test_a_view_that_bends_the_camera_is_dropped_for_its_influence(tmp_path):
    """Panel of 1.10.1 (reproduced): one view of eight sheared by 5% — a
    rolling-shutter frame, a bent board — is consistent with SOME camera, so
    the optimiser compromised: fx 459 against 400, the correction 111 px
    wrong, the view's own residual 1.7x the median, every gate quiet. Refit
    without it and the focal length moves 16% where no other view moves it
    more than 2%: dropped by name, for that reason."""
    from plantcv_mcp.lens import calibrate_lens

    d = tmp_path / "sheared"
    d.mkdir()
    for i, (r, t) in enumerate(POSES[:10]):
        img = _view(r, t)
        if i == 0:
            img = _shear(img, 0.05)
        cv2.imwrite(str(d / f"view{i}.png"), _distort(img))
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    dropped = {n: why for n, _, why in calib.frames_outliers}
    assert set(dropped) == {"view0.png"}
    assert "moves the focal length" in dropped["view0.png"]
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.03 * MTX[0, 0]
    # Positive control: the same ten views unsheared drop nothing.
    _write_pose_set(tmp_path / "clean", POSES[:10])
    clean = calibrate_lens(str(tmp_path / "clean"), row_corners=ROWS, col_corners=COLS)
    assert clean.frames_outliers == ()


def test_bad_views_are_dropped_one_at_a_time_not_behind_a_median(tmp_path):
    """Panel of 1.10.1 (codex; reproduced): three of six views rippled by
    4-6 px inflated the whole-set median so that none stood 3x above it, and
    the set 'calibrated' to fx 450 with the correction 92 px wrong. Each view
    is judged against the OTHERS' median and the worst dropped, refit,
    repeated: all three are named and the camera recovered."""
    from plantcv_mcp.lens import calibrate_lens

    d = tmp_path / "rippled"
    d.mkdir()
    bad = {3: (4.0, 0.3), 4: (4.0, 1.1), 5: (6.0, 2.0)}
    for i, (r, t) in enumerate(POSES[:6]):
        img = _distort(_view(r, t))
        if i in bad:
            img = _ripple(img, *bad[i])
        cv2.imwrite(str(d / f"view{i}.png"), img)
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    assert {n for n, _, _ in calib.frames_outliers} == {
        "view3.png",
        "view4.png",
        "view5.png",
    }
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.03 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 25.0


def test_two_bad_views_of_four_are_refused_not_averaged(tmp_path):
    """Panel of 1.10.1 (codex; reproduced): two of four views rippled by 4 px
    calibrated to fx 359 with the correction 1462 px wrong, silently. Half a
    set cannot be told from the other half; what the fit CAN say is that its
    focal length is uncertain by 4.9%, and that is refused."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    d = tmp_path / "half"
    d.mkdir()
    bad = {2: (4.0, 0.3), 3: (4.0, 1.1)}
    for i, (r, t) in enumerate(POSES[:4]):
        img = _distort(_view(r, t))
        if i in bad:
            img = _ripple(img, *bad[i])
        cv2.imwrite(str(d / f"view{i}.png"), img)
    with pytest.raises(LensCalibrationError, match="undetermined"):
        calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    # Positive control: the same four views clean calibrate, nothing dropped.
    _write_pose_set(tmp_path / "four", POSES[:4])
    four = calibrate_lens(str(tmp_path / "four"), row_corners=ROWS, col_corners=COLS)
    assert four.frames_outliers == ()
    assert float(_applied_field_error(four).max()) < 25.0


def test_a_sound_three_view_set_is_accepted(tmp_path):
    """Panel of 1.10.1 (codex; reproduced): views 0, 1 and the frontal 11 of
    the fixture recover the camera to 0.1% with the correction 2.5 px at
    worst — and 1.10.0 refused them at 'conditioning 23'. The statistic did
    not measure what its name said; the fit's relative uncertainty on the
    focal length (0.6% here) does."""
    from plantcv_mcp.lens import FOCAL_UNCERTAINTY_ADVISORY, calibrate_lens

    _write_pose_set(tmp_path / "three", [POSES[0], POSES[1], POSES[11]])
    calib = calibrate_lens(str(tmp_path / "three"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.01 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 10.0
    assert calib.focal_uncertainty < FOCAL_UNCERTAINTY_ADVISORY


def test_duplicate_frames_are_counted_once_and_named(tmp_path):
    """Panel of 1.10.1 (codex; reproduced): eight byte-for-byte copies of a
    set passed a gate the set itself failed — copies add no geometry but
    shrink every reported uncertainty by the square root of their number.
    Identical bytes are one observation: the calibration is the one the
    distinct frames give, and the copies are named."""
    import shutil

    from plantcv_mcp.lens import calibrate_lens
    from plantcv_mcp.server import _lens_advisories

    _write_pose_set(tmp_path / "plain", POSES[:6])
    plain = calibrate_lens(str(tmp_path / "plain"), row_corners=ROWS, col_corners=COLS)
    _write_pose_set(tmp_path / "copied", POSES[:6])
    for k in range(3):
        shutil.copyfile(
            tmp_path / "copied" / "view0.png", tmp_path / "copied" / f"z{k}.png"
        )
    calib = calibrate_lens(str(tmp_path / "copied"), row_corners=ROWS, col_corners=COLS)
    assert calib.frames_duplicates == (
        ("z0.png", "view0.png"),
        ("z1.png", "view0.png"),
        ("z2.png", "view0.png"),
    )
    assert calib.frames_used == plain.frames_used
    assert calib.focal_uncertainty == pytest.approx(plain.focal_uncertainty)
    info = {"roi_degenerate": False, "residual_void_px": 0, "crop_fraction": 0.1}
    codes = {w["code"]: w["message"] for w in _lens_advisories(calib, info, "x.png")}
    assert "z2.png" in codes["duplicate_frames_ignored"]
    assert "duplicate_frames_ignored" not in {
        w["code"] for w in _lens_advisories(plain, info, "x.png")
    }


def test_a_non_finite_focal_uncertainty_is_refused(tmp_path, monkeypatch):
    """Panel of 1.10.1 (three judges): `nan > threshold` is False, so a fit
    whose covariance came back NaN (a singular Jacobian at finite rms) sailed
    through the gate that exists to catch undetermined focal lengths. A
    standard deviation that is not a number is refused, not compared."""
    from plantcv_mcp import lens
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    real_fit = lens._fit

    def nan_fit(*args, **kwargs):
        fit = real_fit(*args, **kwargs)
        sd = fit.sd.copy()
        sd[0] = sd[1] = float("nan")
        return fit._replace(sd=sd)

    monkeypatch.setattr(lens, "_fit", nan_fit)
    _write_pose_set(tmp_path / "six", POSES[:6])
    with pytest.raises(LensCalibrationError, match="not finite"):
        calibrate_lens(str(tmp_path / "six"), row_corners=ROWS, col_corners=COLS)
    # Positive control: the unpatched fit on the same frames is accepted.
    monkeypatch.setattr(lens, "_fit", real_fit)
    calibrate_lens(str(tmp_path / "six"), row_corners=ROWS, col_corners=COLS)


def test_a_true_k3_camera_is_a_documented_limit(tmp_path):
    """Panel of 1.9.0 (codex; reproduced): k3 is fixed at zero, and the
    fixture's truth has k3 = 0, so the oracle could never see a sixth-order
    lens. A camera with k3 = 0.1 calibrates at rms 0.34 px (no advisory) with
    the applied correction 77 px wrong at the corners, and the residuals carry
    no radial signature to detect it by. Pinned as the limit it is."""
    from plantcv_mcp.lens import calibrate_lens, rms_fraction
    from plantcv_mcp.server import HIGH_REPROJECTION_FRACTION

    dist = np.array([-0.5, 0.2, 0.0, 0.0, 0.1])
    d = tmp_path / "k3"
    d.mkdir()
    for i, (r, t) in enumerate(POSES):
        cv2.imwrite(str(d / f"view{i}.png"), _distort(_view(r, t), MTX, dist))
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    assert float(calib.dist.ravel()[4]) == 0.0
    assert rms_fraction(calib.rms, calib.shape) < HIGH_REPROJECTION_FRACTION
    assert calib.frames_outliers == ()
    err = _applied_field_error(calib, MTX, dist)
    assert float(err.max()) > 40.0  # the limit: silent, and large


def test_a_member_swapped_through_its_parent_directory_is_refused(
    tmp_path, monkeypatch
):
    """Panel of 1.9.0 (5/5 judges; reproduced): O_NOFOLLOW guards the LAST
    path component. Renaming the calibration directory and planting an
    outside symlink at its name between the check and the open read outside
    bytes into the digest and the calibration. The read must be checked on
    the OPENED file, not on a name."""
    from plantcv_mcp import paths, server
    from plantcv_mcp.paths import PathOutsideRootsError
    from plantcv_mcp.server import _load_checkerboard_frames

    root = tmp_path / "root"
    calib = root / "calib"
    _write_frames(calib)
    outside = tmp_path / "outside"
    _write_frames(outside)
    real_check = server.check_readable

    # The LAST member in listing order (view9 sorts after view13): swapping an
    # earlier one is caught by the NEXT member's own name check, which is how
    # this test passed on the unfixed code until it was aimed here.
    def swap_parent_after_check(path):
        real = real_check(path)
        if os.path.basename(path) == "view9.png" and not os.path.islink(calib):
            os.rename(calib, root / "calib.moved")
            os.symlink(str(outside), str(calib))
        return real

    paths.set_roots([str(root)])
    try:
        monkeypatch.setattr(server, "check_readable", swap_parent_after_check)
        with pytest.raises(PathOutsideRootsError):
            _load_checkerboard_frames(str(calib))
        # Positive control: with no swap the same directory reads normally.
        monkeypatch.setattr(server, "check_readable", real_check)
        frames, _ = _load_checkerboard_frames(str(root / "calib.moved"))
    finally:
        paths.set_roots(None)
    assert len(frames) == len(POSES)
    assert all(data == (root / "calib.moved" / n).read_bytes() for n, data in frames)


def test_a_member_swapped_for_a_fifo_is_skipped_not_hung(tmp_path, monkeypatch):
    """Panel of 1.9.0 (two judges; reproduced): a regular member replaced by a
    FIFO between the check and the open blocked os.open(O_RDONLY) for good —
    the 'named skip' never happened. Open without blocking, and only a regular
    file is read."""
    import threading

    from plantcv_mcp import server
    from plantcv_mcp.server import _load_checkerboard_frames

    calib = tmp_path / "calib"
    _write_frames(calib)
    target = calib / "view0.png"
    real_check = server.check_readable

    def fifo_after_check(path):
        real = real_check(path)
        if os.path.basename(path) == "view0.png" and not stat.S_ISFIFO(
            os.lstat(target).st_mode
        ):
            os.unlink(target)
            os.mkfifo(target)
        return real

    monkeypatch.setattr(server, "check_readable", fifo_after_check)
    result = {}

    def run():
        result["frames"] = _load_checkerboard_frames(str(calib))[0]

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(15.0)
    assert not worker.is_alive(), "the member open blocked on the FIFO"
    by_name = dict(result["frames"])
    assert by_name["view0.png"] == b""
    assert by_name["view1.png"] == (calib / "view1.png").read_bytes()


def test_a_dangling_symlink_at_output_path_is_refused_by_name(tmp_path, calib_dir):
    """Panel of 1.9.0 (or-gpt): output_path was resolved BEFORE write_image
    saw it, so a dangling symlink there created its target and reported the
    target's path. The name the caller gave is the one that must be judged."""
    from plantcv_mcp.server import _correct_lens_impl

    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    link = tmp_path / "out.png"
    os.symlink(str(tmp_path / "unrequested.png"), str(link))
    with pytest.raises(OSError, match="is a symlink"):
        _correct_lens_impl(str(scene), str(calib_dir), ROWS, COLS, str(link))
    assert not (tmp_path / "unrequested.png").exists()
    # Positive control: a plain new path is written.
    res = _correct_lens_impl(
        str(scene), str(calib_dir), ROWS, COLS, str(tmp_path / "ok.png")
    )
    assert res["corrected_image_path"] == str(tmp_path / "ok.png")
    assert (tmp_path / "ok.png").exists()


def test_environment_roots_are_a_snapshot_not_re_resolved_per_call(
    tmp_path, monkeypatch
):
    """Panel of 1.10.1 (codex; reproduced): PLANTCV_MCP_ROOTS was realpath'd
    on every call, so renaming the root directory and planting a symlink to
    outside at its name between the check on the name and the check on the
    opened descriptor made BOTH checks resolve to outside, and outside bytes
    were read. `--root` always snapshotted; the environment form does too."""
    from plantcv_mcp import paths
    from plantcv_mcp.imaging import open_regular_file
    from plantcv_mcp.paths import PathOutsideRootsError, check_readable

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "image.png").write_bytes(b"INSIDE")
    (outside / "image.png").write_bytes(b"OUTSIDE")
    monkeypatch.setenv("PLANTCV_MCP_ROOTS", str(root))
    monkeypatch.setattr(paths, "_env_snapshot", None, raising=False)
    path = str(root / "image.png")
    real = check_readable(path)
    os.rename(root, tmp_path / "root.old")
    os.symlink(outside, root)
    with pytest.raises(PathOutsideRootsError):
        fd = open_regular_file(real, path)
        os.close(fd)
    # Positive control: with the directory back in place the read succeeds.
    os.unlink(root)
    os.rename(tmp_path / "root.old", root)
    fd = open_regular_file(check_readable(path), path)
    try:
        assert os.read(fd, 16) == b"INSIDE"
    finally:
        os.close(fd)


def test_a_fifo_present_from_the_start_is_a_named_skip(tmp_path):
    """Panel of 1.10.1 (two judges): only a member SWAPPED for a FIFO after
    the file check was named; one present at listing time failed isfile()
    and vanished from the accounting. Anything at a member's name that is
    not a regular file is a named skip; a subdirectory is still not a frame."""
    from plantcv_mcp.server import _lens_calibration, _load_checkerboard_frames

    calib = tmp_path / "calib"
    _write_frames(calib)
    os.mkfifo(calib / "pipe.png")
    (calib / "subdir").mkdir()
    frames, _ = _load_checkerboard_frames(str(calib))
    names = [n for n, _ in frames]
    assert "pipe.png" in names
    assert dict(frames)["pipe.png"] == b""
    assert "subdir" not in names
    result = _lens_calibration(str(calib), ROWS, COLS)
    assert "pipe.png" in result.frames_skipped
    assert len(result.frames_used) == len(POSES)


def test_an_output_path_that_is_a_directory_is_said_to_be_one(tmp_path, calib_dir):
    """Panel of 1.10.1 (probe): a directory at output_path was refused as
    'already exists. Pass a new path' — true, and useless."""
    from plantcv_mcp.server import _correct_lens_impl

    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), cv2.cvtColor(_distort(_view(*POSES[0])), cv2.COLOR_GRAY2BGR)
    )
    (tmp_path / "outdir").mkdir()
    with pytest.raises(OSError, match="is a directory"):
        _correct_lens_impl(
            str(scene), str(calib_dir), ROWS, COLS, str(tmp_path / "outdir")
        )
    # Positive control: a fresh file name is written.
    out = _correct_lens_impl(
        str(scene), str(calib_dir), ROWS, COLS, str(tmp_path / "out.png")
    )
    assert os.path.exists(out["corrected_image_path"])


# --- Mutation round 14: pins for the greens of the 1.11.0 guards ---


def test_a_start_that_fails_inside_opencv_is_skipped_not_fatal(tmp_path, monkeypatch):
    """Round 14 (L4): one of the four starting focal lengths raising inside
    calibrateCamera must not abort a calibration the other starts recover;
    that start is skipped and the best remaining fit kept."""
    from plantcv_mcp import lens
    from plantcv_mcp.lens import calibrate_lens

    real_fit = lens._fit

    def failing_start(objp, imgp, size, guess=None):
        if guess is not None and abs(float(guess[0, 0]) - 2.0 * size[0]) < 1e-6:
            raise cv2.error("simulated: one start does not converge")
        return real_fit(objp, imgp, size, guess)

    monkeypatch.setattr(lens, "_fit", failing_start)
    _write_pose_set(tmp_path / "six", POSES[:6])
    calib = calibrate_lens(str(tmp_path / "six"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 25.0


def test_a_start_with_a_non_finite_residual_never_wins(tmp_path, monkeypatch):
    """Round 14 (L5): `min` over the starts' residuals with a NaN among them
    is order luck — nothing compares less than NaN, so a NaN first start
    would be kept and the set refused as meaningless. Non-finite residuals
    sort last, and a finite start wins."""
    from plantcv_mcp import lens
    from plantcv_mcp.lens import calibrate_lens

    real_fit = lens._fit

    def nan_cold_start(objp, imgp, size, guess=None):
        fit = real_fit(objp, imgp, size, guess)
        return fit._replace(rms=float("nan")) if guess is None else fit

    monkeypatch.setattr(lens, "_fit", nan_cold_start)
    _write_pose_set(tmp_path / "six", POSES[:6])
    calib = calibrate_lens(str(tmp_path / "six"), row_corners=ROWS, col_corners=COLS)
    assert math.isfinite(calib.rms)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]


def test_four_views_are_not_judged_by_a_three_view_leave_one_out(tmp_path):
    """Round 14 (L20): views 2, 8, 9 and 10 of the fixture calibrate to 0.5%
    with the correction 17 px at worst. Judged by the fit WITHOUT each view
    — three views — one of them shifts the focal length 7% at 4.7σ, is
    dropped, and the three that remain 'calibrate' to fx 427 with the
    correction 111 px wrong, every gate quiet (measured under the mutant).
    Influence is judged from five views, where the fit without one still
    has a say."""
    from plantcv_mcp.lens import calibrate_lens

    _write_pose_set(tmp_path / "four", [POSES[i] for i in (2, 8, 9, 10)])
    calib = calibrate_lens(str(tmp_path / "four"), row_corners=ROWS, col_corners=COLS)
    assert calib.frames_outliers == ()
    assert len(calib.frames_used) == 4
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 25.0


def test_the_worst_view_is_dropped_first_not_the_first_flagged(tmp_path):
    """Round 14 (L24): with view 6 of eight rippled by 6 px, the camera bent
    to fit it makes the honest view 0 look influential — 24% at 4σ, flagged
    only because of view 6 (view 1 rippled by 2.5 px keeps the remainder
    loose enough for that). Dropping the first flagged view throws the honest
    one away and then view 6; dropping the WORST removes view 6, after which
    view 0 is quiet and kept."""
    from plantcv_mcp.lens import calibrate_lens

    d = tmp_path / "cascade"
    d.mkdir()
    for i, (r, t) in enumerate(POSES[:8]):
        img = _distort(_view(r, t))
        if i == 1:
            img = _ripple(img, 2.5, 0.7)
        if i == 6:
            img = _ripple(img, 6.0, 2.0)
        cv2.imwrite(str(d / f"view{i}.png"), img)
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    assert {n for n, _, _ in calib.frames_outliers} == {"view6.png"}
    assert "view0.png" in calib.frames_used
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]


def test_the_refit_after_a_drop_starts_from_several_focal_lengths_too(tmp_path):
    """Round 14 (L25): with view 12 spoiled by a 6-px ripple and dropped, the
    thirteen honest views that remain are exactly the subset a cold start
    fits to fx 780 (test_a_wrong_start_no_longer_yields_a_wrong_camera). A
    cold refit landed there and, from that wrong camera, dropped two more
    honest views (6 and 9) before recovering (measured under the mutant).
    The refit starts from several focal lengths like the first fit does."""
    from plantcv_mcp.lens import calibrate_lens

    d = tmp_path / "spoiled"
    d.mkdir()
    for i, (r, t) in enumerate(POSES):
        img = _distort(_view(r, t))
        if i == 12:
            img = _ripple(img, 6.0, 1.0)
        cv2.imwrite(str(d / f"view{i}.png"), img)
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    assert [n for n, _, _ in calib.frames_outliers] == ["view12.png"]
    assert len(calib.frames_used) == 13
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 25.0


def test_dropped_views_that_supplied_the_orientations_are_named(tmp_path):
    """Round 14 (L26): four views at one tilt plus two at other tilts, both
    rippled by 6 px: three orientations at the start, the two dropped for
    their residuals, and the four that remain ONE orientation. That is the
    orientation refusal naming the dropped views — not 'undetermined', which
    is what the four views' uncertainty (16%) says when the count is not
    taken again after the drops."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    def write(directory, rippled):
        directory.mkdir()
        for i, (_, t) in enumerate(POSES[:4]):
            cv2.imwrite(
                str(directory / f"view{i}.png"), _distort(_view((0.12, 0.0, 0.0), t))
            )
        for i, ((r, t), phase) in enumerate(((POSES[9], 0.3), (POSES[10], 1.1))):
            img = _distort(_view(r, t))
            if rippled:
                img = _ripple(img, 6.0, phase)
            cv2.imwrite(str(directory / f"view{4 + i}.png"), img)

    write(tmp_path / "supplied", rippled=True)
    with pytest.raises(LensCalibrationError, match="orientation") as excinfo:
        calibrate_lens(str(tmp_path / "supplied"), row_corners=ROWS, col_corners=COLS)
    assert "view4.png" in str(excinfo.value) and "view5.png" in str(excinfo.value)
    # Positive control: the same two views clean are the missing orientations.
    write(tmp_path / "clean", rippled=False)
    calib = calibrate_lens(str(tmp_path / "clean"), row_corners=ROWS, col_corners=COLS)
    assert calib.frames_outliers == ()
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]


def test_the_response_carries_each_reason_and_the_focal_uncertainty(
    tmp_path, calib_dir
):
    """Round 14 (S3, S4, S8): why a frame was dropped reaches the caller
    twice — in the advisory text and as `reason` on each frames_outliers
    entry — and the focal uncertainty the gate judged is reported."""
    from plantcv_mcp.lens import FOCAL_UNCERTAINTY_ADVISORY
    from plantcv_mcp.server import _correct_lens_impl

    tilts = [20, 26, 40]
    poses = [
        ((math.radians(tilts[i % 3]), 0.0, 0.0), t) for i, (_, t) in enumerate(POSES)
    ]
    _write_pose_set(tmp_path / "onesided", poses)
    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), cv2.cvtColor(_distort(_view(*POSES[0])), cv2.COLOR_GRAY2BGR)
    )
    res = _correct_lens_impl(
        str(scene), str(tmp_path / "onesided"), ROWS, COLS, str(tmp_path / "out.png")
    )
    assert res["frames_outliers"]
    assert all(entry["reason"] for entry in res["frames_outliers"])
    message = next(
        w["message"] for w in res["warnings"] if w["code"] == "outlier_frames_dropped"
    )
    assert all(entry["reason"] in message for entry in res["frames_outliers"])
    assert 0.0 < res["focal_uncertainty"] < FOCAL_UNCERTAINTY_ADVISORY
    # Positive control: the clean fixture reports its uncertainty and no drops.
    quiet = _correct_lens_impl(
        str(scene), str(calib_dir), ROWS, COLS, str(tmp_path / "quiet.png")
    )
    assert quiet["frames_outliers"] == []
    assert 0.0 < quiet["focal_uncertainty"] < FOCAL_UNCERTAINTY_ADVISORY


def test_a_new_environment_roots_value_is_resolved(tmp_path, monkeypatch):
    """Round 14 (P2): the roots are a snapshot per VALUE of the variable, not
    for the life of the process — a server handed a new PLANTCV_MCP_ROOTS
    reads under the new roots, and the same value asked again is the same
    snapshot."""
    from plantcv_mcp import paths

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    paths.set_roots(None)
    monkeypatch.setattr(paths, "_env_snapshot", None, raising=False)
    monkeypatch.setenv("PLANTCV_MCP_ROOTS", str(a))
    assert paths.configured_roots() == [os.path.realpath(a)]
    assert paths.configured_roots() == [os.path.realpath(a)]
    monkeypatch.setenv("PLANTCV_MCP_ROOTS", str(b))
    assert paths.configured_roots() == [os.path.realpath(b)]


# --- the L18 design question of mutation round 14, decided (1.11.1) ---


def test_a_bad_view_is_not_excused_by_the_size_of_the_set(tmp_path):
    """Round 14 left the influence rule's absolute shift floor open: it was
    ANDed with the sigma test at 3% of the focal length. A leave-one-out shift
    shrinks as views are added while its sigma barely does — one view sheared
    5% moves fx 15.9% at 6.7 sigma among eight views and 2.5% at 4.8 sigma
    among fourteen — so the floor excused the bad view in the larger set and
    nowhere else: kept at fx 407 with the correction 28.7 px wrong, against
    fx 397 and 8.1 px once dropped. Swept over 52 sets at seven floors, that
    was the only outcome the floor decided, and over 203 views of 22 sound
    sets it protected none of them (not one reached 4 sigma at a shift under
    3%). The floor now sits at the half percent below which a shift cannot
    move the applied correction by 2 px."""
    from plantcv_mcp.lens import calibrate_lens

    d = tmp_path / "sheared"
    d.mkdir()
    for i, (r, t) in enumerate(POSES):
        img = _view(r, t)
        if i == 0:
            img = _shear(img, 0.05)
        cv2.imwrite(str(d / f"view{i}.png"), _distort(img))
    calib = calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    dropped = {n: why for n, _, why in calib.frames_outliers}
    assert set(dropped) == {"view0.png"}
    assert "moves the focal length" in dropped["view0.png"]
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.03 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 15.0
    # Positive control: the same fourteen views unsheared keep every one of
    # them — the lower floor does not start dropping sound views.
    _write_pose_set(tmp_path / "clean", POSES)
    clean = calibrate_lens(str(tmp_path / "clean"), row_corners=ROWS, col_corners=COLS)
    assert clean.frames_outliers == ()
    assert len(clean.frames_used) == len(POSES)


# --- the real-camera dogfood of 1.11.1 (1.12.0) ---


def test_the_correction_is_not_written_into_the_checkerboard_directory(tmp_path):
    """Real-camera dogfood: correcting an image that lives in the checkerboard
    directory put the tool's own <image>_undistorted.png back among the
    calibration frames -- it came back in frames_skipped on the next call,
    caught only by the majority-size rule because the correction happened to
    crop. The directory is calibration INPUT: everything in it is a candidate
    frame and the cache is keyed on its contents, so writing there also
    refits every image of a batch from scratch. Refused, both for the derived
    name and for an explicit output_path."""
    from plantcv_mcp.server import _correct_lens_impl

    d = tmp_path / "frames"
    _write_frames(d)
    # A plant photo that happens to be filed with the calibration shots: not
    # a board, so it never counts as a frame and the view total below is the
    # pose set alone.
    inside = d / "scene.png"
    cv2.imwrite(str(inside), np.full((480, 640, 3), 127, np.uint8))

    with pytest.raises(ValueError) as excinfo:
        _correct_lens_impl(str(inside), str(d), row_corners=ROWS, col_corners=COLS)
    assert "checkerboard" in str(excinfo.value)
    assert "Nothing was written" in str(excinfo.value)
    assert not (d / "scene_undistorted.png").exists()

    # The same refusal when the caller names the path explicitly, including
    # by a route that only resolves to the directory (".." through a sibling).
    outside = tmp_path / "scene.png"
    cv2.imwrite(
        str(outside), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )
    with pytest.raises(ValueError):
        _correct_lens_impl(
            str(outside),
            str(d),
            row_corners=ROWS,
            col_corners=COLS,
            output_path=str(d / "elsewhere.png"),
        )
    with pytest.raises(ValueError):
        _correct_lens_impl(
            str(outside),
            str(d),
            row_corners=ROWS,
            col_corners=COLS,
            output_path=str(tmp_path / "frames" / ".." / "frames" / "sneaky.png"),
        )
    assert sorted(p.name for p in d.glob("*.png")) == sorted(
        [f"view{i}.png" for i in range(len(POSES))] + ["scene.png"]
    )

    # Positive control: the identical correction to a path OUTSIDE the
    # checkerboard directory still succeeds, so the guard refuses the sink and
    # not the tool.
    res = _correct_lens_impl(str(outside), str(d), row_corners=ROWS, col_corners=COLS)
    assert res["corrected_image_path"] == str(tmp_path / "scene_undistorted.png")
    assert os.path.exists(res["corrected_image_path"])
    assert res["frames_used"] == len(POSES)


def test_a_skipped_frame_is_named_even_when_the_calibration_is_thick(tmp_path):
    """Real-camera dogfood: bad_checkerboard.png was left out of a calibration
    with eight usable views and NOTHING said so -- skips were named only
    inside thin_calibration, which fires below five used frames, while every
    outlier drop was announced. The skip is the likelier user error of the
    two (wrong corner counts, a blurred frame, a file that is not the board)."""
    from plantcv_mcp.server import _correct_lens_impl

    d = tmp_path / "frames"
    _write_frames(d)
    cv2.imwrite(str(d / "not_a_board.png"), np.full((480, 640, 3), 127, np.uint8))
    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), _distort(cv2.cvtColor(_view(*POSES[0]), cv2.COLOR_GRAY2BGR))
    )

    res = _correct_lens_impl(str(scene), str(d), row_corners=ROWS, col_corners=COLS)
    assert res["frames_used"] >= 5  # thick: thin_calibration cannot fire
    assert "thin_calibration" not in [w["code"] for w in res["warnings"]]
    skipped = [w for w in res["warnings"] if w["code"] == "frames_skipped"]
    assert len(skipped) == 1
    assert "not_a_board.png" in skipped[0]["message"]
    assert "INNER corners" in skipped[0]["message"]

    # Positive control: with nothing skipped the warning does not appear, so
    # it reports a real skip rather than firing on every call.
    clean = tmp_path / "clean"
    _write_frames(clean)
    res2 = _correct_lens_impl(
        str(scene),
        str(clean),
        row_corners=ROWS,
        col_corners=COLS,
        output_path=str(tmp_path / "clean_out.png"),
    )
    assert res2["frames_skipped"] == []
    assert "frames_skipped" not in [w["code"] for w in res2["warnings"]]


def test_the_wrong_corner_counts_are_echoed_the_way_they_were_given(tmp_path):
    """Real-camera dogfood: asked for row_corners=9, col_corners=6 the refusal
    read '6x9-inner-corner', transposing the caller's own arguments back at
    them while telling them to check those arguments."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    d = tmp_path / "frames"
    _write_frames(d)
    with pytest.raises(LensCalibrationError) as excinfo:
        calibrate_lens(str(d), row_corners=COLS + 2, col_corners=ROWS + 2)
    message = str(excinfo.value)
    assert f"{COLS + 2} x {ROWS + 2} inner corners" in message
    assert "row_corners x col_corners" in message


def test_dropping_a_view_earns_its_place_even_when_it_picks_the_wrong_one(
    tmp_path, monkeypatch
):
    """The small-set wrong-view drop, investigated 2026-09-02 and deliberately
    left alone. Six views with one sheared 5%: the honest view scores 11.0
    sigma against the sheared view's 7.8, so the loop drops the honest one.
    That is the symptom -- and it is not the thing that matters. Measured over
    32 faulted sets against the same sets with the influence rule off, 28 of
    32 drops improve the correction and 6 of the 7 that removed an honest view
    still improved it. Four sharper rules were measured and none beat this one
    (see the comment on the drop loop). What is pinned here is the invariant
    that survived: dropping earns its place. A future rule that picks better
    passes this too; one that makes the answer worse than not dropping does
    not."""
    from plantcv_mcp import lens as L

    d = tmp_path / "sheared"
    d.mkdir()
    for i, (r, t) in enumerate(POSES[:6]):
        img = _view(r, t)
        if i == 0:
            img = _shear(img, 0.05)
        cv2.imwrite(str(d / f"view{i}.png"), _distort(img))

    with_rule = L.calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    field_with = float(_applied_field_error(with_rule).max())

    # The same set judged with the influence rule switched off -- the honest
    # alternative the original report never measured against.
    monkeypatch.setattr(L, "INFLUENCE_SHIFT", 1e9)
    without_rule = L.calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
    field_without = float(_applied_field_error(without_rule).max())
    monkeypatch.undo()

    assert with_rule.frames_outliers, "the rule is expected to fire on this set"
    assert not without_rule.frames_outliers
    # It drops the wrong view here, and is still worth having: 72.5 px against
    # 118.8 px when nothing is dropped, a little under two thirds. The margin
    # is what makes this assertion bite -- a rule with its thresholds removed
    # throws two honest views away and still lands at 115.1 px, which beats
    # 118.8 by enough to pass a bare inequality and nothing more.
    assert field_with < 0.8 * field_without

    # Positive control: on the same six poses with nothing sheared the rule
    # fires on no one, so the gain above is a response to the fault and not a
    # rule that always throws a view away.
    clean = tmp_path / "clean"
    _write_pose_set(clean, POSES[:6])
    sound = L.calibrate_lens(str(clean), row_corners=ROWS, col_corners=COLS)
    assert sound.frames_outliers == ()
    assert len(sound.frames_used) == 6
