"""Lens-distortion calibration from checkerboards, and image correction.

**Why the corner loop does not call pcv.transform.checkerboard_calib.** That
function print()s per-frame detection failures to stdout and then calls
cv2.calibrateCamera unconditionally: a directory where zero frames detect
crashes with a raw cv2 error, and one where a single frame detects "succeeds"
with a garbage calibration and no warning. The loop here is the same algorithm
(findChessboardCorners + cornerSubPix with the same criteria); what changes is
the accounting — every frame is reported as used or skipped by name, and fewer
than MIN_CALIBRATION_FRAMES detections is a typed refusal, not a crash.

**Why the correction crops.** pcv.transform.calibrate_camera computes
getOptimalNewCameraMatrix with alpha=1 and discards the valid-pixel ROI it
returns. The remap then fabricates black void pixels wherever the corrected
frame has no source data — measured on the real fisheye tutorial photo those
voids are large ellipses that any value/darkness threshold would select as
objects. The same arithmetic is used here (alpha=1 + cv2.undistort), and then
the frame is cropped to that ROI so every remaining pixel is real.

Why correction matters at all, measured on that same photo: uncorrected
distortion inflated the centre-frame plant's area 2.13x and was ANISOTROPIC —
the same pot's rim, face height and base scaled 1.728x/1.313x/1.591x — so no
px_per_mm, however calibrated, can compensate. Correct first, then calibrate
scale on the corrected image.
"""

import os
from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV's own guidance is ~10 views; the tutorial camera ships 9. Below three
# the optimiser will still return numbers, and they are numbers about nothing.
MIN_CALIBRATION_FRAMES = 3

_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


class LensCalibrationError(Exception):
    """Raised when a usable calibration cannot be built from the directory."""


@dataclass(frozen=True)
class LensCalibration:
    mtx: np.ndarray
    dist: np.ndarray
    rms: float
    frames_used: list[str]
    frames_skipped: list[str]


def calibrate_lens(
    checkerboard_dir: str, row_corners: int, col_corners: int
) -> LensCalibration:
    """Build a camera calibration from a directory of checkerboard photos.

    row_corners / col_corners count INNER corners (a standard 10x7-square board
    has 9x6). Frames where no board is detected — wrong counts, blur, or a file
    that is not an image — are skipped and named, because a calibration that
    silently dropped half its frames is worth less than it claims.
    """
    if row_corners < 2 or col_corners < 2:
        raise LensCalibrationError(
            f"row_corners={row_corners} col_corners={col_corners}: both must be "
            "at least 2, counting INNER corners of the checkerboard."
        )
    try:
        entries = sorted(os.listdir(checkerboard_dir))
    except OSError as exc:
        raise LensCalibrationError(
            f"Cannot list checkerboard directory {checkerboard_dir!r}: {exc}"
        ) from exc

    objp = np.zeros((row_corners * col_corners, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col_corners, 0:row_corners].T.reshape(-1, 2)

    used: list[str] = []
    skipped: list[str] = []
    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for name in entries:
        path = os.path.join(checkerboard_dir, name)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped.append(name)
            continue
        if shape is not None and img.shape != shape:
            skipped.append(name)
            continue
        found, corners = cv2.findChessboardCorners(img, (col_corners, row_corners))
        if not found:
            skipped.append(name)
            continue
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
        shape = img.shape
        used.append(name)
        objpoints.append(objp)
        imgpoints.append(corners)

    if len(used) < MIN_CALIBRATION_FRAMES:
        raise LensCalibrationError(
            f"Only {len(used)} of {len(entries)} file(s) in "
            f"{checkerboard_dir!r} contain a detectable "
            f"{col_corners}x{row_corners}-inner-corner checkerboard"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
            + f". A calibration needs at least {MIN_CALIBRATION_FRAMES} views "
            "— and the corner counts must be the INNER corners of the board, "
            "one less per side than its squares."
        )

    assert shape is not None
    rms, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, shape[::-1], None, None
    )
    return LensCalibration(
        mtx=mtx,
        dist=dist,
        rms=float(rms),
        frames_used=used,
        frames_skipped=skipped,
    )


def undistort_image(img: np.ndarray, calib: LensCalibration) -> tuple[np.ndarray, dict]:
    """Correct an image with a calibration; crop to the all-valid-pixel region.

    Returns (corrected, info) where info carries the valid ROI as (x, y, w, h)
    in the UNCROPPED corrected frame, the fraction of pixels the crop removed,
    and roi_degenerate=True when OpenCV could not find a valid rectangle at all
    (severe distortion) — in that case the frame is returned uncropped and
    still contains fabricated void pixels.
    """
    h, w = img.shape[:2]
    newmtx, roi = cv2.getOptimalNewCameraMatrix(
        calib.mtx, calib.dist, (w, h), 1, (w, h)
    )
    corrected = cv2.undistort(img, calib.mtx, calib.dist, None, newmtx)
    x, y, rw, rh = (int(v) for v in roi)
    degenerate = rw <= 0 or rh <= 0
    info = {
        "valid_roi": [x, y, rw, rh],
        "crop_fraction": 0.0,
        "roi_degenerate": degenerate,
    }
    if not degenerate:
        info["crop_fraction"] = 1.0 - (rw * rh) / float(w * h)
        corrected = corrected[y : y + rh, x : x + rw]
    return corrected, info
