"""Deriving px_per_mm from a size marker of known physical length.

**Why this does not use pcv.report_size_marker_area.** That function takes an ROI,
but measured against a synthetic disc of known 80 px diameter it returns
major_axis=79.1 with a whole-frame ROI and 348.0 with a TIGHT ROI drawn around the
marker — a silent 4.35x scale error under the most intuitive usage, because its
ROI-filtering step can select a background component that merely intersects the
ROI. A whole-frame ROI is not a workaround either: a real image contains a plant as
well as a marker, and the plant would be measured instead.

So the region is CROPPED before thresholding. Nothing outside the box can be
selected, because nothing outside the box is passed in. That removes the mechanism
rather than compensating for it.
"""

from dataclasses import dataclass

import numpy as np

from .diagnostics import Advisory
from .segmentation import threshold_mask


class MarkerNotFoundError(Exception):
    """Raised when no object can be found inside the marker region."""


@dataclass(frozen=True)
class ScaleEstimate:
    px_per_mm: float
    marker_length_px: int
    marker_length_mm: float
    marker_area_px: int
    crop_fraction: float
    warnings: list[Advisory]


def calibrate_scale(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    marker_length_mm: float,
    channel: str = "v",
    method: str = "otsu",
    object_type: str = "dark",
) -> ScaleEstimate:
    """Measure a marker inside a crop and return pixels per millimetre.

    (x, y, w, h) bounds a region containing ONLY the marker. marker_length_mm is
    the marker's longest real dimension — the diameter of a circular marker, the
    side of a square one.

    The returned estimate carries the measured pixel length so the caller can check
    it against what they expect, because a wrong scale silently rescales every
    subsequent trait.
    """
    if marker_length_mm <= 0:
        raise ValueError(f"marker_length_mm must be > 0, got {marker_length_mm}")
    if w <= 0 or h <= 0:
        raise ValueError(f"crop must have positive size, got w={w} h={h}")

    frame_h, frame_w = img.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + int(w), y0 + int(h)
    if x0 < 0 or y0 < 0 or x1 > frame_w or y1 > frame_h:
        # No clamping: a crop quietly shrunk to fit can cut the marker at the
        # frame edge and yield a plausible but wrong scale, and the edge-contact
        # warning below would then point at the wrong culprit (polarity).
        raise ValueError(
            f"crop ({x}, {y}, {w}, {h}) does not lie inside the "
            f"{frame_w}x{frame_h} image. Move or shrink the region so the whole "
            "marker box is in frame."
        )

    crop = img[y0:y1, x0:x1]
    mask = threshold_mask(crop, channel, method, object_type=object_type)

    import cv2

    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise MarkerNotFoundError(
            f"No object found inside the marker region ({x}, {y}, {w}, {h}) using "
            f"channel={channel!r} method={method!r} object_type={object_type!r}. "
            "Try the opposite object_type, or check the region actually contains "
            "the marker."
        )

    # Largest component in the crop is the marker; background cannot compete,
    # because the crop excludes it.
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = int(np.argmax(areas)) + 1
    bw = int(stats[idx, cv2.CC_STAT_WIDTH])
    bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
    # The marker's length comes from the MINIMUM-AREA rotated rectangle, not
    # the axis-aligned bbox: a square marker photographed at 45 degrees has a
    # bbox sqrt(2)x its side, which would overstate px_per_mm by ~41% — and its
    # bbox is square, so no roundness check on bbox sides can catch it. The +1
    # converts the extent of pixel CENTRES to the pixel footprint (an
    # axis-aligned run of n pixels spans n-1 centre-to-centre).
    ys, xs = np.nonzero(labels == idx)
    rect_points = np.column_stack([xs, ys]).astype(np.float32)
    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(rect_points)
    side_long = float(max(rect_w, rect_h)) + 1.0
    side_short = float(min(rect_w, rect_h)) + 1.0
    marker_length_px = round(side_long)
    marker_area_px = int(stats[idx, cv2.CC_STAT_AREA])
    crop_fraction = float(marker_area_px) / float(binary.size)

    warnings: list[Advisory] = []

    # A marker sitting inside its own crop should not reach the crop boundary.
    # When the polarity is wrong the selected component is the BACKGROUND, which
    # spans the crop and touches every edge. Measured on a centred 80 px disc in a
    # 100x100 crop, the correct polarity gives crop_fraction 0.50 and the INVERTED
    # one also gives 0.50 — coverage cannot tell them apart, so edge contact is the
    # discriminator that can.
    bx = int(stats[idx, cv2.CC_STAT_LEFT])
    by = int(stats[idx, cv2.CC_STAT_TOP])
    touches = bx <= 0 or by <= 0 or (bx + bw) >= (x1 - x0) or (by + bh) >= (y1 - y0)
    if touches:
        warnings.append(
            Advisory(
                code="marker_touches_crop_edge",
                message=(
                    f"The detected object ({bw}x{bh} px) reaches the edge of the "
                    "crop. A marker fully inside its own region should not. This "
                    "usually means the polarity is wrong and the BACKGROUND was "
                    "selected, which would rescale every later trait. Try the "
                    "opposite object_type, or draw the region with a margin around "
                    "the marker. If instead the object simply continues beyond the "
                    "crop — a pot contiguous with its soil and plant is ONE object "
                    "— there is no isolatable marker in this scene, and any crop "
                    "yields a different scale: leave the traits in pixels rather "
                    "than trusting this one."
                ),
            )
        )

    if crop_fraction > 0.9:
        warnings.append(
            Advisory(
                code="marker_fills_crop",
                message=(
                    f"The detected object covers {crop_fraction:.0%} of the crop, so "
                    "the crop may be selecting background rather than the marker. "
                    "Check marker_length_px below against the size you expect, and "
                    "try the opposite object_type."
                ),
            )
        )
    if side_short > 0 and (side_long / side_short) > 1.5:
        warnings.append(
            Advisory(
                code="marker_not_round",
                message=(
                    f"The detected object measures {side_long:.0f}x{side_short:.0f} "
                    "px in its own orientation, which is far from square. If your "
                    "marker is circular or square this is probably not it, and the "
                    "resulting scale would be wrong."
                ),
            )
        )

    if marker_length_px <= 0:
        raise MarkerNotFoundError("Detected marker has zero length in pixels.")

    return ScaleEstimate(
        px_per_mm=side_long / float(marker_length_mm),
        marker_length_px=marker_length_px,
        marker_length_mm=float(marker_length_mm),
        marker_area_px=marker_area_px,
        crop_fraction=crop_fraction,
        warnings=warnings,
    )
