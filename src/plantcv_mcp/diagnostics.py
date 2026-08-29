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


# Remedy sentences are written per modality. The RGB ones name segment(),
# channels and methods; a thermal frame has a temperature band and a cube has
# an index threshold, and advice naming the wrong knob misdirects (found on a
# real FLIR tray and a real leaf cube: every refusal said "channel or method").
@dataclass(frozen=True)
class Remedies:
    coverage: str  # implausible_coverage: how to get the other polarity
    empty: str  # empty_mask: what to change when nothing was selected
    degenerate: str  # DegenerateMaskError: which segmenter to go back to


RGB_REMEDIES = Remedies(
    coverage=(
        "Re-run segment() with the opposite object_type ('light' instead of "
        "'dark', or vice versa)."
    ),
    empty=(
        "Try the opposite object_type, a different channel or method, or a "
        "smaller fill_size."
    ),
    degenerate="Re-run segment() with a different channel or method.",
)
HSI_REMEDIES = Remedies(
    coverage=(
        "Re-run segment_hyperspectral() with the opposite object_type, or a "
        "threshold inside the index_range."
    ),
    empty=(
        "Re-run segment_hyperspectral() with a threshold inside the reported "
        "index_range (or the opposite object_type), or a smaller fill_size."
    ),
    degenerate=(
        "Re-run segment_hyperspectral() with a threshold inside the index_range, "
        "or a different index."
    ),
)
THERMAL_REMEDIES = Remedies(
    coverage=(
        "Narrow the temperature band (min_c / max_c) to the plant's own "
        "temperatures and re-run segment_thermal()."
    ),
    empty=(
        "Re-run segment_thermal() with a band (min_c / max_c) inside the "
        "frame_range, or a smaller fill_size."
    ),
    degenerate=(
        "Re-run segment_thermal() with a band (min_c / max_c) that covers the "
        "plant's temperatures."
    ),
)
RGB_COVERAGE_REMEDY = RGB_REMEDIES.coverage


def assert_not_degenerate(
    diag: MaskDiagnostics,
    min_fraction: float = 0.001,
    remedy: str = RGB_REMEDIES.degenerate,
) -> None:
    """Raise DegenerateMaskError if traits would be meaningless.

    Degenerate if ANY of: zero components; zero largest area; mask fraction
    below min_fraction. Callers must invoke this BEFORE returning traits —
    PlantCV will happily return 17 zero-valued traits otherwise.
    """
    if diag.component_count == 0 or diag.largest_area == 0:
        raise DegenerateMaskError(
            f"Segmentation produced no objects. The mask is empty. {remedy}"
        )
    if diag.mask_fraction < min_fraction:
        raise DegenerateMaskError(
            f"Mask covers {diag.mask_fraction:.4%} of the frame, below the "
            f"{min_fraction:.2%} minimum. This is almost certainly a failed "
            f"segmentation, not a very small plant. {remedy}"
        )


@dataclass(frozen=True)
class Advisory:
    code: str
    message: str


def implausible_coverage_warning(
    diag: MaskDiagnostics,
    max_fraction: float = 0.5,
    remedy: str = RGB_COVERAGE_REMEDY,
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
            f"that instead of the plant. {remedy} Ignore this if the subject "
            "genuinely fills the frame, such as a macro shot of a single leaf."
        ),
    )


def empty_mask_warning(
    diag: MaskDiagnostics, remedy: str = RGB_REMEDIES.empty
) -> "Advisory | None":
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
            f"can be measured from it. {remedy}"
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
            "describe the group, not a plant. Call measure_regions() "
            "instead: it measures each plant separately and returns an "
            "overlay with the regions outlined and numbered."
        ),
    )


# Warnings meaning the traits would describe something other than the plant.
# Anything listed here BLOCKS an unattended measurement: with nobody looking at an
# overlay, these are the cases where a number must not be returned at all.
BLOCKING_CODES: frozenset[str] = frozenset(
    {"empty_mask", "fill_erased_mask", "implausible_coverage"}
)


def mask_warnings(
    mask: np.ndarray,
    diag: MaskDiagnostics,
    remedies: Remedies = RGB_REMEDIES,
) -> list["Advisory"]:
    """Advisories derivable from the mask alone, with no segmentation history.

    Shared by segment-time reporting AND measure-time re-reporting: the trait
    table is the artifact people keep, so it must carry the same caveats the
    overlay did — computed by the same code, or the two would drift.
    """
    warnings: list[Advisory] = []

    coverage = implausible_coverage_warning(diag, remedy=remedies.coverage)
    if coverage:
        warnings.append(coverage)

    multi = multi_specimen_warning(diag)
    if multi:
        warnings.append(multi)

    # frame_clipping asserts that size traits are a LOWER BOUND, which presumes the
    # mask IS the plant. On an implausibly large (probably inverted) mask that claim
    # actively misleads, so it is withheld rather than stacked on top.
    if not coverage:
        clipping = frame_clipping_warning(mask)
        if clipping:
            warnings.append(clipping)

    return warnings


def segmentation_warnings(
    mask: np.ndarray,
    diag: MaskDiagnostics,
    pre_fill_diag: MaskDiagnostics,
    fill_size: int,
    remedies: Remedies = RGB_REMEDIES,
) -> list["Advisory"]:
    """Every advisory a segmented mask earns, in one place.

    Shared by the interactive segment() path and the unattended batch path so the
    two cannot drift apart — a batch applying weaker guards than the interactive
    tool would be the worst of both worlds.
    """
    warnings: list[Advisory] = []

    if diag.component_count == 0 and pre_fill_diag.component_count > 0:
        warnings.append(
            Advisory(
                code="fill_erased_mask",
                message=(
                    f"Thresholding found {pre_fill_diag.component_count} object(s), "
                    f"the largest {pre_fill_diag.largest_area} px, and then "
                    f"fill_size={fill_size} removed every one of them. This is a "
                    "fill_size problem, not a channel or method problem — the "
                    "specimen is smaller than the speckle filter. Re-run with "
                    f"fill_size below {pre_fill_diag.largest_area}."
                ),
            )
        )
    else:
        empty = empty_mask_warning(diag, remedy=remedies.empty)
        if empty:
            warnings.append(empty)

    warnings.extend(mask_warnings(mask, diag, remedies=remedies))
    return warnings


def implausible_longest_path_warning(traits: dict) -> "Advisory | None":
    """Flag PlantCV's longest_path when it cannot be a path through the object.

    Observed live: 7 px against a 343 px tall region (a fragmented mask on the
    regions path) — an artefact that reads exactly like a measurement. A path
    through an object shorter than a tenth of its bounding box's long side is
    not a plausible skeleton path.
    """
    lp = (traits.get("longest_path") or {}).get("value")
    w = (traits.get("width") or {}).get("value")
    h = (traits.get("height") or {}).get("value")
    if (
        not isinstance(lp, int | float)
        or not isinstance(w, int | float)
        or not isinstance(h, int | float)
    ):
        return None
    extent = max(float(w), float(h))
    if extent <= 0 or lp >= 0.1 * extent:
        return None
    return Advisory(
        code="implausible_longest_path",
        message=(
            f"PlantCV reports longest_path={float(lp):.1f} against a "
            f"{w}x{h} bounding box. A path through the object cannot be that "
            "short; this happens on fragmented masks and is an artefact, not a "
            "measurement — ignore this trait for this object."
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


def threshold_outside_range_warning(
    selects_everything: bool,
    selects_nothing: bool,
    what: str,
    lo: float,
    hi: float,
    unit: str,
) -> "Advisory | None":
    """Say when the selection rule lies outside the data's own range.

    Found on a real leaf cube (NDVI 0.19-0.89): the default threshold 0.2 sat
    below the minimum and selected every pixel, and 0.95 sat above the
    maximum and selected none; neither result said the threshold was the
    problem. `what` names the rule ("threshold 0.95 with object_type='light'"
    or "band 10.0-40.0 C").
    """
    if not (selects_everything or selects_nothing):
        return None
    outcome = "every pixel" if selects_everything else "nothing"
    return Advisory(
        code="threshold_outside_range",
        message=(
            f"The {what} selects {outcome}: the data span {lo:.4g} to {hi:.4g}"
            f"{unit}, so no pixel is on the other side of the rule. The mask "
            "says nothing about the plant. Choose a value inside that range."
        ),
    )
