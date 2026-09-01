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
    with pytest.raises(LensCalibrationError, match="orientation"):
        calibrate_lens(str(d), row_corners=ROWS, col_corners=COLS)
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
    assert set(calib.frames_skipped) == {f"a_thumb{i}.png" for i in range(3)}


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

    _write_pose_set(tmp_path / "two", [POSES[1]] * 4 + [POSES[9]] * 4)
    with pytest.raises(LensCalibrationError, match="2 distinct"):
        calibrate_lens(str(tmp_path / "two"), row_corners=ROWS, col_corners=COLS)
    # Positive control: three orientations, three copies each, calibrate.
    _write_pose_set(tmp_path / "three", [POSES[0], POSES[1], POSES[9]] * 3)
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


def test_symmetric_small_tilts_are_refused_as_weakly_conditioned(tmp_path):
    """Panel of 1.9.0 (codex; reproduced): a board tilted -7/0/+7 degrees about
    ONE axis at fourteen spread positions passes the orientation count (3),
    coverage (48%) and rms (0.03% of the diagonal) — and calibrates to fx 334
    against 400 with the applied correction 391 px wrong at the corners. The
    count is a proxy; the calibration's own focal-length uncertainty per pixel
    of corner error (52 here, under 22 on every set that recovered the
    camera, 3.5 on the real tutorial set) is the statistic. Refuse on it."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    weak = [((0.12 * ((i % 3) - 1), 0.0, 0.0), t) for i, (_, t) in enumerate(POSES)]
    _write_pose_set(tmp_path / "weak", weak)
    with pytest.raises(LensCalibrationError, match="focal length"):
        calibrate_lens(str(tmp_path / "weak"), row_corners=ROWS, col_corners=COLS)
    # Positive control: five well-tilted views pass the same gate and recover
    # the camera (conditioning 10.5 measured).
    _write_pose_set(tmp_path / "five", POSES[:5])
    calib = calibrate_lens(str(tmp_path / "five"), row_corners=ROWS, col_corners=COLS)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert calib.focal_conditioning < 15.0


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
    dropped = [name for name, _ in calib.frames_outliers]
    assert dropped == ["view4.png", "view6.png", "view7.png"]
    assert all(px > 2.0 for _, px in calib.frames_outliers)
    assert not set(dropped) & set(calib.frames_used)
    assert abs(calib.mtx[0, 0] - MTX[0, 0]) < 0.02 * MTX[0, 0]
    assert float(_applied_field_error(calib).max()) < 15.0
    # The tool says so, naming the frames and their residuals.
    from plantcv_mcp.server import _lens_advisories

    info = {"roi_degenerate": False, "residual_void_px": 0, "crop_fraction": 0.1}
    codes = {w["code"]: w["message"] for w in _lens_advisories(calib, info, "x.png")}
    assert "outlier_frames_dropped" in codes
    assert "view4.png" in codes["outlier_frames_dropped"]
    # Positive control: the fixture's own residual spread (0.08-1.14 px) is
    # rendering, not a bad frame; nothing is dropped from it and nothing said.
    assert calibration.frames_outliers == ()
    assert len(calibration.frames_used) == len(POSES)
    quiet = {w["code"] for w in _lens_advisories(calibration, info, "x.png")}
    assert "outlier_frames_dropped" not in quiet


def test_orientation_count_is_independent_of_frame_order():
    """Panel of 1.9.0 (three judges): greedy first-fit packing counted the
    tilts 0/4/8/12/16 degrees as three orientations in one filename order and
    two in another, so renaming identical photos flipped the verdict. 'Within
    5 degrees count as one' is a transitive relation; count its classes."""
    from plantcv_mcp.lens import _distinct_orientations

    rvecs = {
        a: cv2.Rodrigues(cv2.Rodrigues(np.float32((math.radians(a), 0.0, 0.0)))[0])[0]
        for a in (0, 4, 8, 12, 16)
    }
    chain_a = _distinct_orientations([rvecs[a] for a in (0, 8, 16, 4, 12)])
    chain_b = _distinct_orientations([rvecs[a] for a in (4, 12, 0, 8, 16)])
    assert chain_a == chain_b == 1
    # Positive control: three tilts 8 degrees apart are three in any order.
    three = [rvecs[a] for a in (0, 8, 16)]
    assert _distinct_orientations(three) == _distinct_orientations(three[::-1]) == 3


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
