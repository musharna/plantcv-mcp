"""Pure mask diagnostics. No I/O, no PlantCV — testable in isolation.

This is the safety core. PlantCV's own in_bounds / object_in_frame flags
report True on an all-zero mask, so they cannot be used to detect a failed
segmentation. Everything here computes our own signal instead.
"""

from dataclasses import dataclass

import cv2
import numpy as np


class DegenerateMaskError(Exception):
    """Raised when a mask is too empty to yield meaningful traits."""


@dataclass(frozen=True)
class MaskDiagnostics:
    component_count: int
    areas: list[int]
    largest_area: int
    mask_fraction: float
    major_object_count: int


def component_areas(mask: np.ndarray) -> list[int]:
    """Connected-component areas, descending, background excluded."""
    binary = (mask > 0).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [int(a) for a in stats[1:, cv2.CC_STAT_AREA]]
    return sorted(areas, reverse=True)


def analyze_mask(mask: np.ndarray, major_threshold: float = 0.25) -> MaskDiagnostics:
    """Summarise a binary mask.

    major_threshold: a component counts as "major" if its area is at least
    this fraction of the largest component's area. 0.25 is a calibrated
    starting value, not a constant — see plan Task 3.
    """
    areas = component_areas(mask)
    largest = areas[0] if areas else 0
    major = sum(1 for a in areas if largest and a >= major_threshold * largest)
    return MaskDiagnostics(
        component_count=len(areas),
        areas=areas,
        largest_area=largest,
        mask_fraction=float((mask > 0).sum()) / float(mask.size),
        major_object_count=major,
    )


def assert_not_degenerate(diag: MaskDiagnostics, min_fraction: float = 0.001) -> None:
    """Raise DegenerateMaskError if traits would be meaningless.

    Degenerate if ANY of: zero components; zero largest area; mask fraction
    below min_fraction. Callers must invoke this BEFORE returning traits —
    PlantCV will happily return 17 zero-valued traits otherwise.
    """
    if diag.component_count == 0 or diag.largest_area == 0:
        raise DegenerateMaskError(
            "Segmentation produced no objects. The mask is empty. "
            "Re-run segment() with a different channel or method."
        )
    if diag.mask_fraction < min_fraction:
        raise DegenerateMaskError(
            f"Mask covers {diag.mask_fraction:.4%} of the frame, below the "
            f"{min_fraction:.2%} minimum. This is almost certainly a failed "
            "segmentation, not a very small plant. Re-run segment() with a "
            "different channel or method."
        )


@dataclass(frozen=True)
class Advisory:
    code: str
    message: str


def implausible_coverage_warning(
    diag: MaskDiagnostics, max_fraction: float = 0.5
) -> "Advisory | None":
    """Warn when the mask covers implausibly much of the frame.

    This is the OTHER half of mask validity. assert_not_degenerate only rejects
    masks that are too SMALL, which leaves the dominant failure of any threshold
    operation — selecting the background instead of the foreground — outside the
    set of things this system could express as a failure at all. Measured on the
    fixture with otsu: the plant masks land at 0.031-0.046 of the frame and the
    inverted ones at 0.959-0.967, so 0.5 sits in a very wide empty gap rather
    than being a hopeful guess. Like the other thresholds here it is a calibrated
    starting value, not a constant.

    This WARNS rather than raises: a legitimate macro shot of a single leaf can
    fill most of the frame, and refusing to measure it would be its own silent
    wrongness. Channels whose two polarities both land near 0.5 (l and v on the
    fixture) are genuinely ambiguous and will not trip this — the polarity report
    from suggest_segmentation is the remedy there.
    """
    if diag.mask_fraction <= max_fraction:
        return None
    return Advisory(
        code="implausible_coverage",
        message=(
            f"The mask covers {diag.mask_fraction:.1%} of the frame. If you "
            "expected a plant against a background, this mask is probably "
            "INVERTED — it is the background, and every trait would describe "
            "that instead of the plant. Re-run segment() with the opposite "
            "object_type ('light' instead of 'dark', or vice versa). Ignore "
            "this if the subject genuinely fills the frame, such as a macro "
            "shot of a single leaf."
        ),
    )


def empty_mask_warning(diag: MaskDiagnostics) -> "Advisory | None":
    """Say plainly that the segmentation found nothing.

    segment() previously returned component_count=0 with no warning at all, so
    the failure only surfaced if measure() happened to be called afterwards. The
    tool whose entire job is to show whether a segmentation can be trusted has to
    state this itself.
    """
    if diag.component_count > 0:
        return None
    return Advisory(
        code="empty_mask",
        message=(
            "Segmentation found no objects at all — the mask is empty. No traits "
            "can be measured from it. Try the opposite object_type, a different "
            "channel or method, or a smaller fill_size."
        ),
    )


def multi_specimen_warning(diag: MaskDiagnostics) -> "Advisory | None":
    """Warn when the mask holds two or more comparably-sized objects.

    Calibrated on a real failure: a 4-view render segmented to areas
    8628/7981/7106/6748 (all >= 78% of the largest -> 4 major objects) with a
    tail at 570 and below (<= 6.6% -> excluded). A whole-image ROI merges them
    into one "plant" and every size trait becomes meaningless.
    """
    if diag.major_object_count < 2:
        return None
    return Advisory(
        code="multi_specimen",
        message=(
            f"{diag.major_object_count} comparably-sized objects detected "
            f"(areas: {diag.areas[: diag.major_object_count]}). A whole-image "
            "ROI will merge them into one object and every size trait will "
            "describe the group, not a plant. Consider roi.auto_grid "
            "(phase 2) or pass an explicit single-plant roi to measure()."
        ),
    )


def frame_clipping_warning(mask: np.ndarray) -> "Advisory | None":
    """Warn when mask pixels touch the frame edge.

    Computed here rather than trusting PlantCV's in_bounds, which reports True
    on an all-zero mask and so cannot discriminate this case.
    """
    binary = mask > 0
    if not (
        binary[0, :].any()
        or binary[-1, :].any()
        or binary[:, 0].any()
        or binary[:, -1].any()
    ):
        return None
    return Advisory(
        code="frame_clipping",
        message=(
            "Plant material touches the frame edge, so it is cut off by the "
            "image boundary. Size traits (area, width, height, perimeter) are "
            "a LOWER BOUND on the true plant, not a measurement of it."
        ),
    )
