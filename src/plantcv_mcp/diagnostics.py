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
class Warning:
    code: str
    message: str


def multi_specimen_warning(diag: MaskDiagnostics) -> "Warning | None":
    """Warn when the mask holds two or more comparably-sized objects.

    Calibrated on a real failure: a 4-view render segmented to areas
    8628/7981/7106/6748 (all >= 78% of the largest -> 4 major objects) with a
    tail at 570 and below (<= 6.6% -> excluded). A whole-image ROI merges them
    into one "plant" and every size trait becomes meaningless.
    """
    if diag.major_object_count < 2:
        return None
    return Warning(
        code="multi_specimen",
        message=(
            f"{diag.major_object_count} comparably-sized objects detected "
            f"(areas: {diag.areas[: diag.major_object_count]}). A whole-image "
            "ROI will merge them into one object and every size trait will "
            "describe the group, not a plant. Consider roi.auto_grid "
            "(phase 2) or pass an explicit single-plant roi to measure()."
        ),
    )


def frame_clipping_warning(mask: np.ndarray) -> "Warning | None":
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
    return Warning(
        code="frame_clipping",
        message=(
            "Plant material touches the frame edge, so it is cut off by the "
            "image boundary. Size traits (area, width, height, perimeter) are "
            "a LOWER BOUND on the true plant, not a measurement of it."
        ),
    )
