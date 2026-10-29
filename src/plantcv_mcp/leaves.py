"""Leaf instances by distance-transform watershed (`pcv.watershed_segmentation`).

`measure_morphology()` already reports a `leaf_count`, from the skeleton: it
counts leaf-like skeleton SEGMENTS, which suits a side-view plant with a stem and
does not yield a region per leaf. This module is the top-view counterpart: it
splits one plant mask into labelled regions, one per distance-transform peak,
and returns the count, each region's area and centroid, and the labelled overlay.

KNOWN WEAKNESS, measured rather than assumed (docs/EVAL.md, "Leaf instances"):
the watershed knows nothing about leaves. It splits the mask wherever two
distance-transform peaks are at least `min_distance` pixels apart, so

* overlapping rosette leaves with no concavity between them stay merged
  (under-segmentation), and a long or lobed leaf with two width maxima is cut
  in two (over-segmentation);
* `min_distance` is in PIXELS and decides the count. On real Arabidopsis trays
  (held-out tray, annotation's own mask) the mean count error moved from +8.5
  leaves at 3 px to -3.4 at 15 px on the same 20 plants, and no rule tied to plant size did better than a fixed value,
  because a rosette's smallest leaves do not grow with the plant.

So the count is an estimate to be read WITH the overlay, and the result carries
the count at half and at twice `min_distance` so its sensitivity is visible.
"""

import colorsys
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import (
    Advisory,
    analyze_mask,
    assert_not_degenerate,
    implausible_coverage_warning,
    plant_fills_frame_advisory,
)
from .imaging import render_overlay
from .measurement import isolated_pcv_outputs

LABEL = "leaves"
# PlantCV's own default for watershed_segmentation(distance=). Kept rather than
# tuned: the value that minimised error on one set of trays (6-8 px) is a
# property of that camera height, not of plants.
DEFAULT_MIN_DISTANCE = 10
SENSITIVITY_FRACTION = 0.30
# The watershed runs on the mask's bounding box set inside a ring of zeros
# WIDER THAN THE PEAK DISTANCE. skimage's peak_local_max, which PlantCV calls
# with its defaults, discards every peak within min_distance of the array's
# border (exclude_border=True): measured here, a lone 70x24 ellipse in a tight
# crop returned ZERO instances at distance 30, and called on a full frame the
# same rule silently drops any leaf near the photo's edge. The ring makes the
# border irrelevant; it is background, so it changes no label.
RING_EXTRA = 2


class LeafCountRefusedError(Exception):
    """Raised when the mask cannot yield a meaningful leaf count."""


@dataclass
class LeafInstancesResult:
    leaf_count: int
    instances: list[dict[str, Any]]
    min_distance: int
    # Count at half and at twice min_distance, keyed by the distance used.
    sensitivity: dict[str, int]
    units: dict[str, str]
    warnings: list[Advisory]
    overlay: np.ndarray
    labels: np.ndarray


def _instance_color(index: int) -> tuple[int, int, int]:
    """A fixed BGR colour per instance id. PlantCV's watershed debug palette is
    random, so the overlay is drawn here: the same mask gives the same picture."""
    r, g, b = colorsys.hsv_to_rgb((index * 0.6180339887) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _enclosed_holes(mask255: np.ndarray) -> int:
    """Background components that do not touch the (padded) crop's border."""
    padded = cv2.copyMakeBorder(mask255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    count, _ = cv2.connectedComponents((padded == 0).astype(np.uint8), connectivity=4)
    # Label 0 is the mask itself; one background component is the outside.
    return max(count - 2, 0)


def _watershed(img_c: np.ndarray, mask_c: np.ndarray, distance: int) -> np.ndarray:
    return pcv.watershed_segmentation(
        rgb_img=img_c, mask=mask_c, distance=distance, label=LABEL
    )


def describe_instances(
    img: np.ndarray,
    mask255: np.ndarray,
    labels_c: np.ndarray,
    y0: int,
    x0: int,
    px_per_mm: float | None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """The instance table and the numbered overlay for a label image.

    `labels_c` is a CROP whose top-left corner sits at (x0, y0) of the full
    frame (either may be negative: the watershed's crop carries a zero ring).
    Shared by count_leaves() and segment_leaves_sam(), so an instance `id`
    means the same thing on both pictures.
    """
    area_scale = 1.0 if px_per_mm is None else 1.0 / (px_per_mm * px_per_mm)
    instances: list[dict[str, Any]] = []
    overlay = render_overlay(img, mask255)
    n = int(labels_c.max())
    for i in range(1, n + 1):
        # Per-instance work stays on the crop (a full-frame comparison per leaf
        # is what made morphology slow on 16 MP photos); coordinates are moved
        # back to the full frame here.
        region = labels_c == i
        area_px = int(region.sum())
        ry, rx = np.nonzero(region)
        ry, rx = ry + y0, rx + x0
        cx, cy = float(rx.mean()), float(ry.mean())
        instances.append(
            {
                "id": i,
                "area": area_px * area_scale,
                "area_px": area_px,
                "centroid": [round(cx, 1), round(cy, 1)],
                "bbox": [
                    int(rx.min()),
                    int(ry.min()),
                    int(rx.max() - rx.min() + 1),
                    int(ry.max() - ry.min() + 1),
                ],
            }
        )
        contours, _ = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(overlay, contours, -1, _instance_color(i), 2, offset=(x0, y0))
    # Numbers last, so no later contour paints over an earlier label.
    for inst in instances:
        cx, cy = inst["centroid"]
        org = (int(cx) - 6, int(cy) + 5)
        cv2.putText(
            overlay, str(inst["id"]), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3
        )
        cv2.putText(
            overlay,
            str(inst["id"]),
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return instances, overlay


def count_leaves(
    img: np.ndarray,
    mask: np.ndarray,
    min_distance: int = DEFAULT_MIN_DISTANCE,
    px_per_mm: float | None = None,
) -> LeafInstancesResult:
    if isinstance(min_distance, bool) or not isinstance(min_distance, int):
        raise TypeError(f"min_distance must be an integer, got {min_distance!r}")
    if min_distance < 1:
        raise ValueError(f"min_distance must be >= 1 pixel, got {min_distance}")
    if px_per_mm is not None and (px_per_mm <= 0 or not math.isfinite(px_per_mm)):
        raise ValueError(f"px_per_mm must be a positive finite number, got {px_per_mm}")

    diag = analyze_mask(mask)
    # An empty mask is a named refusal, never a count of zero: "0 leaves" reads
    # as a measurement of a plant, and an empty mask is a failed segmentation.
    assert_not_degenerate(diag)
    coverage = implausible_coverage_warning(diag)
    filled = plant_fills_frame_advisory(img, mask, diag) if coverage else None
    if coverage and filled is None:
        raise LeafCountRefusedError(
            f"implausible_coverage: {coverage.message} A watershed of the "
            "background counts background blobs as leaves; fix the segmentation "
            "first."
        )
    warnings: list[Advisory] = [filled] if filled else []
    # measure_morphology() refuses a multi-object mask; this cannot. A rosette
    # whose petioles fall below the threshold IS several comparably sized
    # objects (measured: most plants in the Aberystwyth ground-truth masks), so
    # a refusal would reject the ordinary case. The mask cannot say whether the
    # pieces are one plant's leaves or several plants; the overlay can.
    if diag.major_object_count >= 2:
        warnings.append(
            Advisory(
                code="multi_object_mask",
                message=(
                    f"The mask holds {diag.major_object_count} comparably sized "
                    "objects and the count covers ALL of them. That is expected "
                    "for one rosette whose leaves segment apart; if they are "
                    "separate plants, refine(keep_largest) or crop to one plant "
                    "and count that session."
                ),
            )
        )

    mask255 = np.where(mask > 0, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(mask255)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    extent = max(y1 - y0, x1 - x0)
    if min_distance > extent:
        # Checked against the MASK, before anything is sized by it: the ring
        # below is 2 x min_distance wide on every side, so an unbounded value
        # is an unbounded allocation from one tool argument (200000 asked
        # OpenCV for 1.9 TB). It is also meaningless — no two points of the
        # mask are that far apart, so the answer could only be "one".
        raise ValueError(
            f"min_distance={min_distance} exceeds the mask's own extent, "
            f"{extent} px. Peaks cannot be further apart than the mask is wide; "
            "use a value near the half-width of the smallest leaf to be counted."
        )
    # The comparison pass is capped at the extent for the same reason, and the
    # result is keyed by the distance actually run.
    half, double = max(min_distance // 2, 1), min(2 * min_distance, extent)
    ring = double + RING_EXTRA  # the widest distance any pass below uses
    img_c = cv2.copyMakeBorder(
        img[y0:y1, x0:x1], ring, ring, ring, ring, cv2.BORDER_CONSTANT, value=0
    )
    mask_c = cv2.copyMakeBorder(
        mask255[y0:y1, x0:x1], ring, ring, ring, ring, cv2.BORDER_CONSTANT, value=0
    )
    # From here on, crop coordinates map to the full frame by this offset.
    y0, x0 = y0 - ring, x0 - ring

    holes = _enclosed_holes(mask_c)
    if holes:
        warnings.append(
            Advisory(
                code="mask_has_holes",
                message=(
                    f"The mask has {holes} enclosed hole(s). Every hole bends the "
                    "distance transform around itself and adds peaks, so a leaf "
                    "with pinholes is counted several times (measured on 13 real "
                    "rosettes at min_distance=10: mean error +9.6 leaves with the "
                    "holes, -5.4 after fill_holes — the holes were hiding an "
                    "undercount, not cancelling it). refine() with fill_holes and "
                    "count the refined session, unless the holes are real gaps "
                    "between leaves."
                ),
            )
        )

    with isolated_pcv_outputs():
        labels_c = _watershed(img_c, mask_c, min_distance)
        n_half = int(_watershed(img_c, mask_c, half).max())
        n_double = int(_watershed(img_c, mask_c, double).max())

    n = int(labels_c.max())
    if n == 0:
        # Reached once, before the zero ring existed (peaks discarded at the
        # crop border). Kept: a zero must never leave as a leaf count.
        raise LeafCountRefusedError(
            "The watershed found no distance-transform peak in a non-empty mask. "
            "Check the overlay from segment()."
        )

    instances, overlay = describe_instances(img, mask255, labels_c, y0, x0, px_per_mm)

    if max(abs(n_half - n), abs(n_double - n)) / n > SENSITIVITY_FRACTION:
        warnings.append(
            Advisory(
                code="min_distance_sensitive",
                message=(
                    f"min_distance={min_distance} yields {n} instances, but "
                    f"{half} yields {n_half} and {double} yields {n_double}. The "
                    "count depends on this parameter more than on the plant; "
                    "choose min_distance from the overlay (about the half-width "
                    "of the smallest leaf you want counted, in pixels) and keep "
                    "it fixed across images taken at the same scale."
                ),
            )
        )

    labels = np.zeros(mask255.shape, np.int32)
    inner = labels_c[ring:-ring, ring:-ring]
    labels[
        y0 + ring : y0 + ring + inner.shape[0], x0 + ring : x0 + ring + inner.shape[1]
    ] = inner
    return LeafInstancesResult(
        leaf_count=n,
        instances=instances,
        min_distance=min_distance,
        sensitivity={str(half): n_half, str(double): n_double},
        units={
            "area": "pixels" if px_per_mm is None else "mm2",
            "area_px": "pixels",
            "centroid": "pixels",
            "bbox": "pixels",
        },
        warnings=warnings,
        overlay=overlay,
        labels=labels,
    )


# --- seeded split: one marker per object, placed by the caller -----------------
#
# The distance-transform watershed above finds its own markers, one per
# distance peak, so leaves that overlap with no notch between them share a
# peak and stay merged (agent benchmark 2026-10-17: the model gave up on
# count_leaves and counted by eye). A seeded watershed takes the markers from
# the caller instead, one [x, y] per object, and floods the same landscape
# PlantCV's watershed uses (the negated distance transform) with skimage's
# compact watershed, so a mask is split between the seeds, preferring the
# narrowest neck. The count is then the number of seeds by construction; what
# the split adds is a region per seed, its area, and an overlay that shows
# whether each region is one object.
#
# Why compact, and why not the image's colour gradient (both measured on a
# 5-leaf rendered rosette, 2026-10-18): a plain watershed floods from each
# marker's OWN landscape value, so a seed placed on a slope or on a leaf vein
# starts after every other basin has filled and keeps 1-5 px (regions
# [1, 1, 7227, 2895, 5782] from the gradient, [3, 5, 55, 105, 15738] from the
# distance). Compactness charges each flood for its distance from its seed,
# which removes the trap: at 1.0 the five regions were 2,389-3,844 px. The
# gradient trapped seeds at every compactness up to 10, because a vein is a
# ridge as high as a leaf's edge.

MAX_SEEDS = 500
SEED_COMPACTNESS = 1.0
# A region under this share of the median region is named: two seeds on one
# object split it into a full region and a sliver.
SMALL_REGION_FRACTION = 0.15


class SeedSpecError(ValueError):
    """Seeds that are malformed, off the frame or off the mask."""


def validate_seeds(seeds: Any, mask: np.ndarray) -> list[tuple[int, int]]:
    """[[x, y], ...] in full-frame pixels, each on the mask, none repeated."""
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, (list, tuple)):
        raise SeedSpecError(f"seeds must be a list of [x, y] pairs, got {seeds!r}.")
    if not seeds:
        raise SeedSpecError("seeds is empty; give one [x, y] per object.")
    if len(seeds) > MAX_SEEDS:
        raise SeedSpecError(f"at most {MAX_SEEDS} seeds, got {len(seeds)}.")
    h, w = mask.shape[:2]
    out: list[tuple[int, int]] = []
    for i, s in enumerate(seeds):
        if (
            not isinstance(s, (list, tuple))
            or len(s) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in s)
        ):
            raise SeedSpecError(f"seed {i}: must be [x, y] integers, got {s!r}.")
        x, y = s
        if not (0 <= x < w and 0 <= y < h):
            raise SeedSpecError(f"seed {i}: [{x}, {y}] is outside the {w}x{h} frame.")
        if not mask[y, x]:
            raise SeedSpecError(
                f"seed {i}: [{x}, {y}] is not on the mask; place every seed on "
                "the object it stands for (the segment() overlay shows the mask)."
            )
        if (x, y) in out:
            raise SeedSpecError(
                f"seed {i}: [{x}, {y}] repeats seed {out.index((x, y))}."
            )
        out.append((x, y))
    return out


def split_by_seeds(
    img: np.ndarray,
    mask: np.ndarray,
    seeds: Any,
    px_per_mm: float | None = None,
) -> LeafInstancesResult:
    """Split a mask into one region per seed by a compact marker-controlled
    watershed on the negated distance transform. Mask pixels that no seed's
    flood reaches (a component with no seed on it) are left out and reported."""
    from skimage.segmentation import watershed

    if px_per_mm is not None and (px_per_mm <= 0 or not math.isfinite(px_per_mm)):
        raise ValueError(f"px_per_mm must be a positive finite number, got {px_per_mm}")
    diag = analyze_mask(mask)
    assert_not_degenerate(diag)
    coverage = implausible_coverage_warning(diag)
    filled = plant_fills_frame_advisory(img, mask, diag) if coverage else None
    if coverage and filled is None:
        raise LeafCountRefusedError(
            f"implausible_coverage: {coverage.message} Splitting the background "
            "splits nothing of the plant; fix the segmentation first."
        )
    points = validate_seeds(seeds, mask > 0)

    mask255 = np.where(mask > 0, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(mask255)
    pad = 2
    y0 = max(int(ys.min()) - pad, 0)
    x0 = max(int(xs.min()) - pad, 0)
    y1 = min(int(ys.max()) + 1 + pad, mask.shape[0])
    x1 = min(int(xs.max()) + 1 + pad, mask.shape[1])
    inside = mask255[y0:y1, x0:x1] > 0
    landscape = -cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    markers = np.zeros(inside.shape, np.int32)
    for i, (x, y) in enumerate(points, start=1):
        markers[y - y0, x - x0] = i
    labels_c = watershed(
        landscape, markers, mask=inside, compactness=SEED_COMPACTNESS
    ).astype(np.int32)

    instances, overlay = describe_instances(img, mask255, labels_c, y0, x0, px_per_mm)
    warnings: list[Advisory] = [
        Advisory(
            code="seeded_count",
            message=(
                f"The count is the number of seeds given ({len(points)}), not a "
                "measurement: the split only assigns pixels to them. Look at the "
                "overlay: each numbered region should be one whole object."
            ),
        )
    ]
    if filled:
        warnings.append(filled)
    unassigned = int((inside & (labels_c == 0)).sum())
    if unassigned:
        warnings.append(
            Advisory(
                code="unseeded_mask",
                message=(
                    f"{unassigned} mask px ({unassigned / inside.sum():.1%}) lie in "
                    "components with no seed and belong to no region."
                ),
            )
        )
    areas = sorted(int(inst["area_px"]) for inst in instances)
    median = areas[len(areas) // 2]
    small = [
        inst["id"]
        for inst in instances
        if inst["area_px"] < SMALL_REGION_FRACTION * median
    ]
    if small:
        warnings.append(
            Advisory(
                code="small_seeded_region",
                message=(
                    f"Region(s) {small} are under {SMALL_REGION_FRACTION:.0%} of the "
                    f"median region ({median} px): two seeds on one object, or a "
                    "seed on a sliver. Check them on the overlay."
                ),
            )
        )
    labels = np.zeros(mask255.shape, np.int32)
    labels[y0:y1, x0:x1] = labels_c
    return LeafInstancesResult(
        leaf_count=len(points),
        instances=instances,
        min_distance=0,
        sensitivity={},
        units={
            "area": "pixels" if px_per_mm is None else "mm2",
            "area_px": "pixels",
            "centroid": "pixels",
            "bbox": "pixels",
        },
        warnings=warnings,
        overlay=overlay,
        labels=labels,
    )
