"""Lens-distortion calibration from checkerboards, and image correction.

**Why the corner loop does not call pcv.transform.checkerboard_calib.** That
function print()s per-frame detection failures to stdout and then calls
cv2.calibrateCamera unconditionally: a directory where zero frames detect
crashes with a raw cv2 error, and one where a single frame detects "succeeds"
with a garbage calibration and no warning. The loop here is the same algorithm
(findChessboardCorners + cornerSubPix with the same criteria); what changes is
the accounting and the refusals — every frame is reported as used or skipped
by name, fewer than MIN_CALIBRATION_FRAMES detections refuses, and so do a set
of duplicate poses (measured with that refusal disabled, on a synthetic camera
whose intrinsics are known: ten copies of one view "calibrated" to rms 0.19
with fx=252 and k1=-0.24 against a true 400 and -0.50 — a LOW-rms wrong answer
no error gate could catch) and a non-finite or absurd fit. Mirrored views, by
contrast, are NOT a defect: a mirrored board is the same board seen from
behind, and all- or half-mirrored sets recover the same camera.

**Why the crop comes from a real validity mask, not OpenCV's ROI.**
getOptimalNewCameraMatrix(alpha=1) fabricates black void pixels wherever the
corrected frame has no source data, and the ROI it returns is approximate:
measured on this module's own test camera model, the ROI crop retained 566
fabricated black pixels. So a white frame is remapped through the identical
transform; a pixel is valid only where that comes back exactly 255 (partially
interpolated border pixels are contaminated and excluded too), and the output
is cropped to the largest fully-valid rectangle of that mask. When no usable
rectangle exists the frame is returned uncropped with the void count reported.

**Why resolution is part of the calibration's identity.** fx/fy/cx/cy are in
pixels at the calibration frames' resolution; applying them to a different
size is silently wrong geometry (measured: a 1280x960 image through a 640x480
calibration came back as a 49x127 crop). The frame shape is stored and a
mismatched image is refused by name.

Why correction matters at all, measured on the real fisheye tutorial photo:
uncorrected distortion inflated the centre-frame plant's area 2.13x and was
ANISOTROPIC — the same pot's rim, face height and base scaled
1.728x/1.313x/1.591x — so no px_per_mm, however calibrated, can compensate.
Correct first, then calibrate scale on the corrected image.
"""

import math
import os
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV's own guidance is ~10 views; the tutorial camera ships 9. Below three
# the optimiser will still return numbers, and they are numbers about nothing.
MIN_CALIBRATION_FRAMES = 3

# Detected corner sets from genuinely different poses differ by tens to
# hundreds of pixels; re-detections of the same bytes differ by ~0. Duplicate
# exports and stationary-camera bursts sit at the latter.
POSE_DIVERSITY_MIN_PX = 1.0

# A fit this bad is not a camera model. The real (poor) tutorial calibration
# runs at rms 13 px and still visibly straightens; the duplicate-pose failure
# that motivated an rms gate measured 5.95e10 in one reproduction. 100 px
# separates "imperfect" from "meaningless" with orders of magnitude to spare.
MAX_CALIBRATION_RMS = 100.0

# The degenerate fallback: a valid rectangle smaller than this is a postage
# stamp, not a corrected photo — return the full frame and count the voids.
_MIN_CROP_EDGE = 16

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class LensCalibrationError(Exception):
    """Raised when a usable calibration cannot be built from the frames."""


class CalibrationResolutionMismatchError(Exception):
    """Raised when an image's size differs from the calibration frames'."""


@dataclass(frozen=True)
class LensCalibration:
    mtx: np.ndarray
    dist: np.ndarray
    rms: float
    frames_used: list[str]
    frames_skipped: list[str]
    # (height, width) of the calibration frames. The intrinsics are in pixels
    # at THIS resolution; undistort_image refuses any other.
    shape: tuple[int, int]


def calibrate_lens_from_frames(
    frames: list[tuple[str, bytes]], row_corners: int, col_corners: int
) -> LensCalibration:
    """Build a camera calibration from named image byte strings.

    row_corners / col_corners count INNER corners (a standard 10x7-square
    board has 9x6). Frames that do not decode, whose size differs from the
    majority, or where no board is detected are skipped and named. The
    majority size wins deliberately: the first-detected-frame rule let three
    thumbnails that sorted first hijack the calibration away from eight
    full-resolution views.
    """
    if row_corners < 2 or col_corners < 2:
        raise LensCalibrationError(
            f"row_corners={row_corners} col_corners={col_corners}: both must be "
            "at least 2, counting INNER corners of the checkerboard."
        )

    decoded: list[tuple[str, np.ndarray]] = []
    skipped: list[str] = []
    for name, data in frames:
        img = None
        if data:
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append(name)
            continue
        decoded.append((name, img))

    shape: tuple[int, int] | None = None
    if decoded:
        counts = Counter(img.shape for _, img in decoded)
        # Majority size; ties break to the larger frame (a thumbnail never
        # outranks the full-resolution captures it was made from).
        shape = max(counts, key=lambda s: (counts[s], s[0] * s[1]))

    objp = np.zeros((row_corners * col_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col_corners, 0:row_corners].T.reshape(-1, 2)

    used: list[str] = []
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    for name, img in decoded:
        if img.shape != shape:
            skipped.append(name)
            continue
        found, corners = cv2.findChessboardCorners(img, (col_corners, row_corners))
        if not found:
            skipped.append(name)
            continue
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
        used.append(name)
        objpoints.append(objp)
        imgpoints.append(corners)

    if len(used) < MIN_CALIBRATION_FRAMES:
        raise LensCalibrationError(
            f"Only {len(used)} of {len(frames)} file(s) contain a detectable "
            f"{col_corners}x{row_corners}-inner-corner checkerboard"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
            + f". A calibration needs at least {MIN_CALIBRATION_FRAMES} views "
            "— and the corner counts must be the INNER corners of the board, "
            "one less per side than its squares."
        )

    # Duplicate poses fit perfectly and mean nothing: the optimiser has one
    # view of the board, however many copies of it there are.
    spread = max(float(np.mean(np.abs(pts - imgpoints[0]))) for pts in imgpoints[1:])
    if spread < POSE_DIVERSITY_MIN_PX:
        raise LensCalibrationError(
            f"All {len(used)} detected views show the checkerboard in the SAME "
            f"pose (corner sets differ by {spread:.2f} px). Copies of one "
            "photo cannot constrain a camera model — the numbers come back "
            "plausible and wrong. Photograph the board tilted and moved "
            "around the frame, several distinct views."
        )

    assert shape is not None
    rms, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, shape[::-1], None, None
    )
    rms = float(rms)
    if not math.isfinite(rms) or rms > MAX_CALIBRATION_RMS:
        raise LensCalibrationError(
            f"The calibration fit is meaningless (rms reprojection error "
            f"{rms:.3g} px over {len(used)} views). This happens when the "
            "views do not constrain the model — near-identical poses, motion "
            "blur, or wrong corner counts. Re-shoot varied, sharp views."
        )
    return LensCalibration(
        mtx=mtx,
        dist=dist,
        rms=rms,
        frames_used=used,
        frames_skipped=skipped,
        shape=(int(shape[0]), int(shape[1])),
    )


def calibrate_lens(
    checkerboard_dir: str, row_corners: int, col_corners: int
) -> LensCalibration:
    """Build a calibration from a directory of checkerboard photos.

    Library-level entry: reads plainly, with an unreadable file counted as a
    skipped frame. The MCP server builds the frame list itself so every member
    passes the read-root contract and the cache digest covers the same bytes.
    """
    try:
        entries = sorted(os.listdir(checkerboard_dir))
    except OSError as exc:
        raise LensCalibrationError(
            f"Cannot list checkerboard directory {checkerboard_dir!r}: {exc}"
        ) from exc
    frames: list[tuple[str, bytes]] = []
    for name in entries:
        path = os.path.join(checkerboard_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                frames.append((name, fh.read()))
        except OSError:
            frames.append((name, b""))
    return calibrate_lens_from_frames(
        frames, row_corners=row_corners, col_corners=col_corners
    )


def _largest_valid_rectangle(valid: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest axis-aligned all-True rectangle as (x, y, w, h), or None.

    One histogram-stack pass per row — the classic
    largest-rectangle-in-histogram sweep over cumulative column heights.
    """
    h, w = valid.shape
    heights = np.zeros(w, dtype=np.int64)
    best = (0, 0, 0, 0)
    best_area = 0
    for y in range(h):
        heights = np.where(valid[y], heights + 1, 0)
        stack: list[int] = []
        x = 0
        while x <= w:
            cur = int(heights[x]) if x < w else 0
            if not stack or int(heights[stack[-1]]) <= cur:
                stack.append(x)
                x += 1
                continue
            top = stack.pop()
            left = stack[-1] + 1 if stack else 0
            height = int(heights[top])
            area = height * (x - left)
            if area > best_area:
                best_area = area
                best = (left, y - height + 1, x - left, height)
    if best_area == 0:
        return None
    return best


def undistort_image(img: np.ndarray, calib: LensCalibration) -> tuple[np.ndarray, dict]:
    """Correct an image; crop to the largest rectangle of REAL pixels.

    Returns (corrected, info): info carries the chosen rectangle as
    (x, y, w, h) in the uncropped corrected frame, the fraction of the frame
    the crop removed, roi_degenerate=True when no usable rectangle exists
    (the frame is returned uncropped), and residual_void_px — fabricated
    pixels REMAINING in the returned image (0 whenever a crop was applied).
    """
    h, w = img.shape[:2]
    if (h, w) != calib.shape:
        ch, cw = calib.shape
        raise CalibrationResolutionMismatchError(
            f"This image is {w}x{h} but the calibration was built from "
            f"{cw}x{ch} checkerboard frames. Camera intrinsics are in pixels "
            "at the calibration resolution, so applying them here would be "
            "silently wrong geometry. Calibrate with checkerboards shot at "
            "this resolution, or export the image at the calibrated one."
        )
    newmtx, _ = cv2.getOptimalNewCameraMatrix(calib.mtx, calib.dist, (w, h), 1, (w, h))
    corrected = cv2.undistort(img, calib.mtx, calib.dist, None, newmtx)
    # The identical remap applied to a white frame: only pixels that come back
    # exactly 255 have purely-real source data (void-blended borders do not).
    white = np.full((h, w), 255, np.uint8)
    valid = cv2.undistort(white, calib.mtx, calib.dist, None, newmtx) == 255

    rect = _largest_valid_rectangle(valid)
    if rect is not None and (rect[2] < _MIN_CROP_EDGE or rect[3] < _MIN_CROP_EDGE):
        rect = None
    if rect is None:
        return corrected, {
            "valid_roi": [0, 0, 0, 0],
            "crop_fraction": 0.0,
            "roi_degenerate": True,
            "residual_void_px": int((~valid).sum()),
        }
    x, y, rw, rh = rect
    return corrected[y : y + rh, x : x + rw], {
        "valid_roi": [int(x), int(y), int(rw), int(rh)],
        "crop_fraction": 1.0 - (rw * rh) / float(w * h),
        "roi_degenerate": False,
        "residual_void_px": 0,
    }
