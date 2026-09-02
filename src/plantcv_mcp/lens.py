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
frontal translations to fx=228 at rms 0.19 — LOW-rms wrong answers no error
gate could catch; "two orientations repeated to fx=883" was also measured, and
turned out to be a wrong START, not wrong data — see _best_fit), and a
non-finite or absurd fit. Mirrored frames are a documented limit, not a guard: a set that is
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

import hashlib
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

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
# changes 333; in-plane rotations 314 at rms 0.074). Fewer than three is
# refused as the textbook floor; it is NOT sufficient (see
# MAX_FOCAL_UNCERTAINTY). "Two orientations repeated four times each fx 883"
# was measured here too, and was a wrong start rather than wrong data — from
# a better start the same set recovers the camera (_best_fit); the floor of
# three stays because it is the floor the method is derived for, not because
# of that number. Orientations are read from the calibration's own rotation
# vectors; two within this angle of each other count as one, judged in a
# canonical order so the count does not depend on the order the frames
# happen to sort in (panel of 1.9.0), and NOT transitively, so a smooth
# sweep is not one orientation (panel of 1.10.1).
MIN_DISTINCT_ORIENTATIONS = 3
ORIENTATION_DISTINCT_DEG = 5.0

# Three orientations is the textbook floor for a five-intrinsic, four-
# distortion camera; it is NOT a sufficiency test, and the panel of 1.10.1
# showed that no statistic of the fit's self-consistency is one either (the
# "conditioning" gate of 1.10.0, fx uncertainty per pixel of rms, refused a
# ±5° one-axis set whose correction was 8 px right and passed a ±7° two-axis
# set whose correction was 54 px wrong). What can be said honestly is the
# fit's own uncertainty on the focal length RELATIVE to the focal length —
# refused when the views leave it undetermined, said when it is loose.
# Measured (multi-start fits, tmp/stat_1101.py): fixture 0.5%, a sound
# three-view set 0.6%, five views 2.0%, one-sided 10/20/30° 1.3%, ±7° one
# axis 2.4% (field 16 px), ±7° two axes 3.6% (field 54 px — said), two
# mis-detected views of four
# 4.9% (field 1462 px), three of six 10.9%, ±3° one axis 12.3% (field 538).
MAX_FOCAL_UNCERTAINTY = 0.04
FOCAL_UNCERTAINTY_ADVISORY = 0.025

# A view whose own reprojection rms stands far above the OTHERS' median is a
# bad detection, not a bad camera, and the optimiser bends the camera to fit
# it: on the real PlantCV tutorial set one frame at 37 px against a median of
# 4 moved fx by 13% (5169 → 4520). The median is of the other views so that
# two bad views of four cannot hide each other behind a median they inflate
# (panel of 1.10.1, codex: [0.83, 0.90, 2.19, 2.76] px dropped nothing under
# a whole-set median). A view is an outlier above BOTH this multiple and
# this fraction of the diagonal (the fixture's own honest spread runs
# 0.08–1.14 px, a 14× range, all under 0.15%); it is dropped by name, the
# camera refitted, and the test repeated until nothing stands out.
OUTLIER_VIEW_RATIO = 3.0
OUTLIER_VIEW_FRACTION = 0.0025
MIN_VIEWS_FOR_RESIDUAL_DROP = 4

# A view can be wrong and still fit: a board that bent, a rolling-shutter
# frame, an image sheared by the camera moving — its corners are consistent
# with SOME camera, so the optimiser compromises and the view's residual
# never stands out (measured: one view sheared by 5% among eight moved fx
# from 400 to 459 with the correction 111 px wrong, per-view rms 1.7× the
# median). What does stand out is its influence: refit without it and the
# focal length moves. Influence alone is not guilt — the steepest view of a
# weakly tilted set moves the answer too, because it carries the most
# information — so the shift is judged against the uncertainty of the fit
# made WITHOUT the view: a wrong view disagrees with a tight remainder by
# many sigma (sheared 5% of eight: 6.7σ; a 4-px ripple of six: 4.8σ; the
# PlantCV tutorial set's bad frame: 33σ), an informative view leaves a
# loose remainder and disagrees by few (every honest view measured ≤ 2.2σ;
# a 2% shear, correction 38 px, sits at 2.7σ and is the soft edge). Judged
# only where there are enough views for "the fit without it" to mean
# something, and only for a shift that matters.
INFLUENCE_SHIFT = 0.03
INFLUENCE_SIGMA = 4.0
MIN_VIEWS_FOR_INFLUENCE = 5


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
    # Views dropped from the fit, (name, per-view rms px, reason); the median
    # per-view rms of the views that were KEPT; frames whose bytes were
    # identical to an earlier frame, (name, name of the one kept).
    frames_outliers: tuple[tuple[str, float, str], ...] = ()
    median_view_rms: float = 0.0
    frames_duplicates: tuple[tuple[str, str], ...] = ()
    # The fit's own uncertainty on the focal length, as a fraction of it
    # (see MAX_FOCAL_UNCERTAINTY).
    focal_uncertainty: float = 0.0


def rms_fraction(rms: float, shape: tuple[int, int]) -> float:
    """Reprojection rms as a fraction of the frame diagonal — the unit in
    which a fit means the same thing at every resolution."""
    return rms / math.hypot(shape[0], shape[1])


def _distinct_orientations(rvecs, deg: float = ORIENTATION_DISTINCT_DEG) -> int:
    """Count board orientations more than `deg` apart.

    A greedy packing — keep a normal if it is more than `deg` from every
    normal already kept — in a CANONICAL order (by tilt from the optical
    axis, then azimuth), so the count is a property of the set and not of
    the order the frames sorted in (panel of 1.9.0). Not single linkage:
    that made a board swept smoothly through 40° in 4° steps ONE
    orientation and refused a set that calibrated to 1 px (panel of 1.10.1,
    every judge); what the fit needs is spread, and a chain has it.
    """
    normals = []
    for r in rvecs:
        n = cv2.Rodrigues(np.asarray(r))[0][:, 2]
        normals.append(n if n[2] >= 0 else -n)

    def canonical(n: np.ndarray) -> tuple[float, float]:
        tilt = math.degrees(math.acos(min(1.0, abs(float(n[2])))))
        return (round(tilt, 6), round(math.atan2(float(n[1]), float(n[0])), 6))

    kept: list[np.ndarray] = []
    for n in sorted(normals, key=canonical):
        if all(
            math.degrees(math.acos(min(1.0, abs(float(np.dot(n, q)))))) > deg
            for q in kept
        ):
            kept.append(n)
    return len(kept)


class _Fit(NamedTuple):
    rms: float
    mtx: np.ndarray
    dist: np.ndarray
    rvecs: tuple
    sd: np.ndarray  # standard deviations of the intrinsics, fx first
    per_view: np.ndarray  # per-view reprojection rms, in frame order


def _fit(objpoints, imgpoints, size, guess=None) -> _Fit:
    """One calibrateCamera run with k3 fixed, keeping the two things the
    plain call throws away: the per-view reprojection rms and the
    intrinsics' standard deviations. `guess` (a camera matrix) starts the
    optimiser there instead of at OpenCV's own initial estimate."""
    flags = cv2.CALIB_FIX_K3
    mtx0 = None
    if guess is not None:
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        mtx0 = np.array(guess, dtype=np.float64)
    rms, mtx, dist, rvecs, _, sd, _, per_view = cv2.calibrateCameraExtended(
        objpoints, imgpoints, size, mtx0, None, flags=flags
    )
    return _Fit(
        float(rms),
        mtx,
        dist,
        tuple(rvecs),
        np.asarray(sd).ravel(),
        np.asarray(per_view).ravel(),
    )


def _best_fit(objpoints, imgpoints, size) -> _Fit:
    """The lowest-rms fit over several starting focal lengths.

    OpenCV's initial estimate ignores distortion, and on a strongly distorted
    lens seen through near-frontal boards it can be far off (this module's
    fixture: 1356 for a true 400) — from there Levenberg–Marquardt settles in
    a wrong basin with a residual a few times the right one and a camera
    nothing downstream can tell is wrong (panel of 1.10.1: 41 of 91
    twelve-view subsets of the honest fixture calibrated to fx 480–8270 at
    rms 0.7–4.1 px; the one-sided 10/20/30° set that 1.10.0 documented as an
    open limit fitted fx 670 cold and 393 from a start at the frame width).
    Starting at half, one and two frame widths as well and keeping the
    lowest rms recovered every one of them. A true-basin fit is start-
    invariant (the PlantCV tutorial set: identical from every start).
    """
    w, h = size
    fits = [_fit(objpoints, imgpoints, size)]
    for f in (0.5 * w, 1.0 * w, 2.0 * w):
        guess = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
        try:
            fits.append(_fit(objpoints, imgpoints, size, guess))
        except cv2.error:
            continue
    return min(fits, key=lambda f: f.rms if math.isfinite(f.rms) else math.inf)


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
    duplicates: list[tuple[str, str]] = []
    seen: dict[str, str] = {}
    for name, data in frames:
        if data:
            # The same bytes under two names are one observation. Copies
            # do not add geometry, but they do shrink every uncertainty the
            # fit reports by the square root of their number (panel of
            # 1.10.1, codex: eight copies of a weak set passed a gate the
            # set itself failed). Kept once, named.
            key = hashlib.sha256(data).hexdigest()
            if key in seen:
                duplicates.append((name, seen[key]))
                continue
            seen[key] = name
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
        copies = (
            f" {len(duplicates)} of the files are byte-for-byte copies of "
            "another and count once ("
            + ", ".join(f"{n} = {first}" for n, first in duplicates)
            + ")."
            if duplicates
            else ""
        )
        raise LensCalibrationError(
            f"Only {len(used)} of {len(frames)} file(s) contain a detectable "
            f"{col_corners}x{row_corners}-inner-corner checkerboard"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "")
            + copies
            + f". A calibration needs at least {MIN_CALIBRATION_FRAMES} views "
            "— and the corner counts must be the INNER corners of the board, "
            "one less per side than its squares."
        )

    assert shape is not None
    frame_shape = (int(shape[0]), int(shape[1]))
    diagonal = math.hypot(*frame_shape)
    size = shape[::-1]

    def meaningless(fit: _Fit, n: int) -> None:
        if (
            not math.isfinite(fit.rms)
            or rms_fraction(fit.rms, frame_shape) > MAX_CALIBRATION_RMS_FRACTION
        ):
            raise LensCalibrationError(
                f"The calibration fit is meaningless (rms reprojection error "
                f"{fit.rms:.3g} px, {rms_fraction(fit.rms, frame_shape):.1%} of "
                f"the frame diagonal, over {n} views). This happens when the "
                "views do not constrain the model — motion blur, or wrong "
                "corner counts. Re-shoot varied, sharp views."
            )

    def too_few_orientations(count: int, n: int, dropped: str = "") -> None:
        raise LensCalibrationError(
            f"The {n} detected views show the checkerboard at {count} distinct "
            "orientation(s) (poses more than "
            f"{ORIENTATION_DISTINCT_DEG:g}° apart). A camera model needs at "
            f"least {MIN_DISTINCT_ORIENTATIONS}: moving the board around the "
            "frame, nearer or farther, or turning it in its own plane all "
            "count as ONE orientation — measured, such sets calibrate to a "
            "confidently wrong camera at a low rms. "
            + (dropped + " " if dropped else "")
            + "Photograph the board TILTED differently in each view — ten "
            "degrees and beyond, in different directions."
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
    fit = _best_fit(objpoints, imgpoints, size)

    # The optimiser fits ONE camera to however many views; what it needs is
    # views of the board at different angles. Judged first, on every view,
    # so a set that never had the geometry is refused for that reason and
    # not for whatever the drop loop below would have made of it.
    count = _distinct_orientations(fit.rvecs)
    if count < MIN_DISTINCT_ORIENTATIONS:
        too_few_orientations(count, len(used))

    # Drop views that fit far worse than the others, or that move the answer
    # far more than the others, one at a time, refitting after each: a bad
    # view bends the camera, and the bent camera hides the next one.
    outliers: list[tuple[str, float, str]] = []
    while len(used) >= MIN_VIEWS_FOR_RESIDUAL_DROP:
        n = len(used)
        pv = fit.per_view
        shifts: list[tuple[float, float]] | None = None
        if n >= MIN_VIEWS_FOR_INFLUENCE:
            fx = float(fit.mtx[0, 0])
            shifts = []
            for k in range(n):
                without = _fit(
                    objpoints[:k] + objpoints[k + 1 :],
                    imgpoints[:k] + imgpoints[k + 1 :],
                    size,
                    guess=fit.mtx,
                )
                fx_without = float(without.mtx[0, 0])
                shift = abs(fx_without - fx) / fx
                loose = max(float(without.sd[0]), float(without.sd[1])) / fx_without
                shifts.append((shift, shift / loose if loose > 0 else math.inf))
        worst: tuple[float, int, str] | None = None
        for k in range(n):
            others_rms = float(np.median(np.delete(pv, k)))
            ratio = pv[k] / others_rms if others_rms > 0 else math.inf
            score = 0.0
            reasons: list[str] = []
            if ratio > OUTLIER_VIEW_RATIO and pv[k] > OUTLIER_VIEW_FRACTION * diagonal:
                score = ratio / OUTLIER_VIEW_RATIO
                reasons.append(
                    f"fits at {pv[k]:.1f} px against the other views' median of "
                    f"{others_rms:.1f} px"
                )
            if shifts is not None:
                shift, sigma = shifts[k]
                if shift > INFLUENCE_SHIFT and sigma > INFLUENCE_SIGMA:
                    score = max(score, sigma / INFLUENCE_SIGMA)
                    reasons.append(
                        f"moves the focal length by {shift:.0%} on its own, "
                        f"{sigma:.0f} times the uncertainty of the fit without it"
                        + (
                            ""
                            if reasons
                            else " — a view consistent with some camera but not "
                            "this one: a bent board, a rolling-shutter frame"
                        )
                    )
            if reasons and (worst is None or score > worst[0]):
                worst = (score, k, ", and ".join(reasons))
        if worst is None:
            break
        _, k, reason = worst
        outliers.append((used[k], float(pv[k]), reason))
        del used[k], objpoints[k], imgpoints[k]
        fit = _best_fit(objpoints, imgpoints, size)

    # Only now is the fit judged as a whole: one grossly mis-detected view
    # is dropped and named above, not refused here as "re-shoot everything".
    meaningless(fit, len(used))

    if outliers:
        count = _distinct_orientations(fit.rvecs)
        if count < MIN_DISTINCT_ORIENTATIONS:
            names = ", ".join(n for n, _, _ in outliers)
            too_few_orientations(
                count,
                len(used),
                f"The view(s) dropped as unreliable ({names}) supplied the "
                "missing orientation(s); re-shoot them.",
            )

    # Three orientations is the floor, not a guarantee: the fit's own
    # uncertainty on the focal length says how well the views determined
    # it. Either focal length undetermined is the same failure (they track
    # each other on any square-pixel camera), so the larger is judged.
    fx = float(fit.mtx[0, 0])
    sd_f = max(float(fit.sd[0]), float(fit.sd[1]))
    uncertainty = sd_f / fx if fx > 0 else math.inf
    if not math.isfinite(uncertainty) or uncertainty > MAX_FOCAL_UNCERTAINTY:
        raise LensCalibrationError(
            f"The {len(used)} detected views leave the focal length "
            f"undetermined: the fit's own uncertainty on it is "
            f"{'not finite' if not math.isfinite(uncertainty) else format(uncertainty, '.1%')}"
            f" of its value (usable calibrations sit under "
            f"{MAX_FOCAL_UNCERTAINTY:.0%}). The board was tilted too little, or "
            "the views disagree with each other. Tilt the board MORE, ten "
            "degrees and beyond, in different directions."
        )

    hull = np.zeros(frame_shape, np.uint8)
    for pts in imgpoints:
        cv2.fillConvexPoly(
            hull, cv2.convexHull(pts.reshape(-1, 1, 2).astype(np.int32)), 1
        )
    return LensCalibration(
        mtx=fit.mtx,
        dist=fit.dist,
        rms=fit.rms,
        frames_used=used,
        frames_skipped=skipped,
        shape=frame_shape,
        coverage=float(hull.mean()),
        frames_outliers=tuple(outliers),
        median_view_rms=float(np.median(fit.per_view)),
        frames_duplicates=tuple(duplicates),
        focal_uncertainty=float(uncertainty),
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
