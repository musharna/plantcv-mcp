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

import os

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


def _distort(img):
    """Apply MTX/DIST to an ideal pinhole image (remap via undistortPoints)."""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, MTX, DIST, P=MTX).reshape(h, w, 2)
    return cv2.remap(
        img,
        und[..., 0],
        und[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


# Fixed poses: offsets move the board around the frame, jiggles tilt it.
# Deterministic on purpose — a flaky calibration test would teach nothing.
POSES = [
    ((30, 30), ((0, 0), (0, 0), (0, 0), (0, 0))),
    ((150, 40), ((0, 10), (0, -10), (0, 10), (0, -10))),
    ((40, 120), ((12, 0), (-12, 0), (-12, 0), (12, 0))),
    ((120, 100), ((0, 0), (0, 14), (0, -14), (0, 0))),
    ((60, 60), ((8, 8), (-8, -8), (8, 8), (-8, -8))),
    ((100, 30), ((0, 0), (10, 0), (0, 10), (-10, 0))),
    ((20, 80), ((-6, 0), (6, 4), (0, -8), (4, 6))),
    ((90, 90), ((0, -10), (0, 10), (-10, 0), (10, 0))),
]


def _view(offset, jiggle, size=(640, 480)):
    board = _board()
    bh, bw = board.shape
    base = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])
    dst = base * 0.72 + np.float32(offset) + np.float32(jiggle)
    hom = cv2.getPerspectiveTransform(base, dst)
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
    for i, (offset, jiggle) in enumerate(POSES if n is None else POSES[:n]):
        cv2.imwrite(str(directory / f"view{i}.png"), _distort(_view(offset, jiggle)))


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
    """Panel 3: ten duplicates calibrated to rms 0.125 with fx=224 (true 400)
    — a LOW-rms wrong calibration, so only a pose-diversity gate can catch
    it."""
    from plantcv_mcp.lens import LensCalibrationError, calibrate_lens

    d = tmp_path / "dups"
    d.mkdir()
    one = _distort(_view(*POSES[0]))
    for i in range(10):
        cv2.imwrite(str(d / f"dup{i}.png"), one)
    with pytest.raises(LensCalibrationError, match="pose"):
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
    with pytest.raises(OSError, match="symlink"):
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
