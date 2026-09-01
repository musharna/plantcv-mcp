"""Lens-distortion calibration from checkerboards, and image correction.

**Why the corner loop does not call pcv.transform.checkerboard_calib.** That
function print()s per-frame detection failures to stdout and then calls
cv2.calibrateCamera unconditionally: a directory where zero frames detect
crashes with a raw cv2 error, and one where a single frame detects "succeeds"
with a garbage calibration and no warning. The loop here is the same algorithm
(findChessboardCorners + cornerSubPix with the same criteria); what changes is
the accounting and the refusals — every frame is reported as used or skipped
by name, fewer than MIN_CALIBRATION_FRAMES detections refuses, and so do a set
fewer than MIN_DISTINCT_ORIENTATIONS board orientations (measured with that
refusal disabled, on a synthetic camera whose intrinsics are known: ten copies
of one view "calibrated" to rms 0.19 with fx=252 against a true 400; eight
frontal translations to fx=228 at rms 0.19; two orientations repeated to
fx=883 — LOW-rms wrong answers no error gate could catch), and a non-finite or
absurd fit. Mirrored frames are a documented limit, not a guard: a set that is
ALL mirrored calibrates the reflected camera consistently, but a set with SOME
frames mirrored returns a camera that is neither (measured with cx=285: the
mixed set fitted cx=319 at rms 0.35) and nothing here can tell — do not mix.

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

# Zhang's method determines the intrinsics from the board's ORIENTATIONS, not
# its positions: every view of a board facing the camera — translated, nearer,
# farther, rotated in its own plane — is the same orientation, and a set of
# them fits a wrong camera perfectly (measured on the synthetic camera, gate
# disabled: eight frontal translations fx 228 against 400 at rms 0.19; depth
# changes 333; in-plane rotations 314 at rms 0.074; two orientations repeated
# four times each fx 883). Fewer than three is refused as the interpretable
# floor; it is NOT sufficient (see MAX_FOCAL_CONDITIONING). Orientations are
# read from the calibration's own rotation vectors; two within this angle of
# each other count as one, transitively, so the count does not depend on the
# order the frames happen to sort in (panel audit of 1.9.0: greedy packing
# counted 0/4/8/12/16° as three orientations or two by filename).
MIN_DISTINCT_ORIENTATIONS = 3
ORIENTATION_DISTINCT_DEG = 5.0

# Three orientations is a proxy; what the fit needs is DATA THAT DETERMINE
# THE FOCAL LENGTH. Measured (panel audit of 1.9.0, codex): a board tilted
# -7/0/+7° about one axis at fourteen spread positions passed the count,
# 48% coverage and an rms of 0.03% of the diagonal — and calibrated to fx 334
# against 400, the applied correction 391 px wrong at the corners. The
# statistic that separated every such set from every set that recovered the
# camera is the calibration's own uncertainty on fx (cv2.calibrateCameraExtended)
# per pixel of reprojection rms — the conditioning of the fit, invariant to
# resolution: weak sets measured 28–52, sound synthetic sets 4–22 (a
# fourteen-view fixture 3.9, five views 10.5, three views 14), the real
# PlantCV tutorial set 3.5. Above this the focal length is a guess.
MAX_FOCAL_CONDITIONING = 20.0

# A frame whose own reprojection rms stands far above the others' is a bad
# detection, not a bad camera, and the optimiser bends the camera to fit it:
# on the real PlantCV tutorial set one frame at 37 px against a median of 4
# moved fx by 13% (5169 → 4520) and the set's rms from 13.1 to 3.7; three
# such frames in a synthetic one-sided set drove fx to 612 with the field
# 2814 px wrong. A frame is an outlier above BOTH this multiple of the median
# and this fraction of the diagonal (the fixture's own honest spread runs
# 0.08–1.14 px, a 14× range, all under 0.15%); it is dropped by name and the
# camera refitted.
OUTLIER_VIEW_RATIO = 3.0
OUTLIER_VIEW_FRACTION = 0.0025

# A fit this bad is not a camera model. Reprojection rms is in pixels, so the
# threshold is a fraction of the frame diagonal: the same geometric fit scaled
# 8x would otherwise cross a pixel threshold on resolution alone. The real
# tutorial calibration fitted at 13 px of a 3461-px diagonal (0.38%) with its
# one outlier frame in, 3.7 px without it, and visibly straightens either
# way; the duplicate-pose failure that motivated the gate measured 5.95e10 in
# one reproduction. 3% separates "imperfect" from "meaningless" with an order
# of magnitude to spare.
MAX_CALIBRATION_RMS_FRACTION = 0.03

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
    # Fraction of the frame covered by the union of the detected corner
    # grids. The model is FITTED where boards were and extrapolated
    # everywhere else (measured: 24% coverage left the corners 31 px wrong
    # with k3 fixed). Defaulted only so hand-built calibrations in unit tests
    # stay terse; calibrate_lens_from_frames always measures it.
    coverage: float = 1.0
    # Frames dropped as outliers, (name, per-view rms px), and the median
    # per-view rms of the frames that were kept.
    frames_outliers: tuple[tuple[str, float], ...] = ()
    median_view_rms: float = 0.0
    # fx uncertainty per pixel of rms (see MAX_FOCAL_CONDITIONING).
    focal_conditioning: float = 0.0


def rms_fraction(rms: float, shape: tuple[int, int]) -> float:
    """Reprojection rms as a fraction of the frame diagonal — the unit in
    which a fit means the same thing at every resolution."""
    return rms / math.hypot(shape[0], shape[1])


def _distinct_orientations(rvecs, deg: float = ORIENTATION_DISTINCT_DEG) -> int:
    """Count board orientations, two within `deg` of each other counting as
    one — transitively (single linkage), so the count is a property of the
    set and not of the order the frames arrived in."""
    normals = [cv2.Rodrigues(np.asarray(r))[0][:, 2] for r in rvecs]
    parent = list(range(len(normals)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(normals):
        for j in range(i + 1, len(normals)):
            angle = math.degrees(math.acos(min(1.0, abs(float(np.dot(a, normals[j]))))))
            if angle <= deg:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(normals))})


def _fit(objpoints, imgpoints, size):
    """One calibrateCamera run with k3 fixed, plus the two things the plain
    call throws away: the per-view reprojection rms and the intrinsics'
    standard deviations."""
    rms, mtx, dist, rvecs, _, sd_intrinsics, _, per_view = cv2.calibrateCameraExtended(
        objpoints, imgpoints, size, None, None, flags=cv2.CALIB_FIX_K3
    )
    return (
        float(rms),
        mtx,
        dist,
        rvecs,
        np.asarray(sd_intrinsics).ravel(),
        np.asarray(per_view).ravel(),
    )


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
            # IMREAD_UNCHANGED, exactly as imaging.decode_image reads the scene:
            # IMREAD_GRAYSCALE honours the EXIF orientation tag and the scene
            # path does not, so a camera's own portrait frames calibrated at
            # 640x480 while its scenes decoded at 480x640 and were refused as
            # a resolution mismatch. Both paths now read the stored raster.
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            skipped.append(name)
            continue
        if img.ndim == 3:
            img = cv2.cvtColor(
                img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            )
        if img.dtype != np.uint8:
            top = float(img.max()) or 1.0
            img = (img.astype(np.float32) * (255.0 / top)).astype(np.uint8)
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

    assert shape is not None
    frame_shape = (int(shape[0]), int(shape[1]))
    diagonal = math.hypot(*frame_shape)

    def meaningless(rms: float, n: int) -> None:
        if (
            not math.isfinite(rms)
            or rms_fraction(rms, frame_shape) > MAX_CALIBRATION_RMS_FRACTION
        ):
            raise LensCalibrationError(
                f"The calibration fit is meaningless (rms reprojection error "
                f"{rms:.3g} px, {rms_fraction(rms, frame_shape):.1%} of the "
                f"frame diagonal, over {n} views). This happens when the views "
                "do not constrain the model — motion blur, or wrong corner "
                "counts. Re-shoot varied, sharp views."
            )

    # k3 is fixed at zero. Left free, the sixth-order term is fitted from
    # the boards' footprint alone and folds the correction over outside it:
    # measured, k3=-0.16 put the frame corners 659 px wrong from boards
    # covering 24% of the frame, and a wider 48% set drove the free fit to
    # fx 655 with k3=-5.4 (31,000 px). Fixed, the same sets recover k1 and k2
    # to the second decimal; the real fisheye tutorial set loses 0.1 px rms.
    # The price is a documented limit: a lens that truly needs k3 is
    # corrected to fourth order only, the error growing toward the corners
    # (measured, k3=0.1: rms 0.34 px with no advisory, 77 px at the corners),
    # and nothing in the residuals can tell.
    rms, mtx, dist, rvecs, sd, per_view = _fit(objpoints, imgpoints, shape[::-1])
    meaningless(rms, len(used))

    # A bad frame is a bad detection, not a bad camera: drop it and refit.
    bar = max(
        OUTLIER_VIEW_RATIO * float(np.median(per_view)),
        OUTLIER_VIEW_FRACTION * diagonal,
    )
    outliers = tuple((name, float(px)) for name, px in zip(used, per_view) if px > bar)
    if outliers:
        dropped = {name for name, _ in outliers}
        if len(used) - len(dropped) < MIN_CALIBRATION_FRAMES:
            raise LensCalibrationError(
                f"{len(dropped)} of the {len(used)} detected views fit far "
                "worse than the rest ("
                + ", ".join(f"{n} at {px:.1f} px" for n, px in outliers)
                + f", against a median of {float(np.median(per_view)):.1f} px) "
                f"and fewer than {MIN_CALIBRATION_FRAMES} would remain without "
                "them. Those frames are blurred, mis-detected or shot at a "
                "different zoom; re-shoot or remove them."
            )
        keep = [i for i, name in enumerate(used) if name not in dropped]
        used = [used[i] for i in keep]
        objpoints = [objpoints[i] for i in keep]
        imgpoints = [imgpoints[i] for i in keep]
        rms, mtx, dist, rvecs, sd, per_view = _fit(objpoints, imgpoints, shape[::-1])
        meaningless(rms, len(used))

    # The optimiser fits ONE camera to however many views; what it needs is
    # views of the board at different angles. Copies, translations, zooms and
    # in-plane rotations of one orientation all leave the focal length free to
    # trade against distortion — plausible numbers, wrong camera, low rms.
    orientations = _distinct_orientations(rvecs)
    if orientations < MIN_DISTINCT_ORIENTATIONS:
        raise LensCalibrationError(
            f"The {len(used)} detected views show the checkerboard at "
            f"{orientations} distinct orientation(s) (poses within "
            f"{ORIENTATION_DISTINCT_DEG:.0f}° of each other count as one). A "
            f"camera model needs at least {MIN_DISTINCT_ORIENTATIONS}: moving "
            "the board around the frame, nearer or farther, or copies of one "
            "photo do not add any — measured, eight such views 'calibrated' "
            "to a focal length 43% short at a low rms. Photograph the board "
            "TILTED differently in each view."
        )

    # Three orientations is the floor, not a guarantee: the fit's own
    # uncertainty on fx says whether the views actually determined it.
    # Either focal length undetermined is the same failure (they track each
    # other on any square-pixel camera), so the larger uncertainty is judged.
    conditioning = max(float(sd[0]), float(sd[1])) / rms if rms > 0 else 0.0
    if conditioning > MAX_FOCAL_CONDITIONING:
        raise LensCalibrationError(
            f"The {len(used)} detected views leave the focal length "
            f"undetermined: the fit's own uncertainty on it is {float(sd[0]):.0f} px "
            f"per pixel of corner error ({conditioning:.0f}; usable "
            f"calibrations sit under {MAX_FOCAL_CONDITIONING:.0f}). The board "
            "was tilted too little, or only symmetrically about one axis — "
            "measured, such a set 'calibrated' to a focal length 16% short at "
            "a low rms with the correction 391 px wrong at the corners. Tilt "
            "the board MORE, ten degrees and beyond, in different directions."
        )

    hull = np.zeros(frame_shape, np.uint8)
    for pts in imgpoints:
        cv2.fillConvexPoly(
            hull, cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32)), 1
        )
    return LensCalibration(
        mtx=mtx,
        dist=dist,
        rms=rms,
        frames_used=used,
        frames_skipped=skipped,
        shape=frame_shape,
        coverage=float(hull.mean()),
        frames_outliers=outliers,
        median_view_rms=float(np.median(per_view)),
        focal_conditioning=conditioning,
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
