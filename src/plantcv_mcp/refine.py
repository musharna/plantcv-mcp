"""Mask refinement: validated morphological ops applied to a session's mask.

Two rules make this more than a pass-through to PlantCV.

1. **Validate everything before applying anything.** PlantCV silently no-ops on
   `fill(size=-1)`, `erode(i=0)` and an even `median_blur` kernel; a refinement
   that quietly did nothing would be recorded in the lineage as if it had run.
   Every op is checked — name, required params, ranges, no unknown params —
   before the first one touches the mask, and the error names the op's index.
2. **A refinement that deletes the plant is refused, not minted.** An erosion
   that empties the mask must not become a session that then measures zeros
   with believable units. The refusal carries the before/after numbers.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import (
    Advisory,
    DegenerateMaskError,
    MaskDiagnostics,
    analyze_mask,
    assert_not_degenerate,
    frame_clipping_warning,
    implausible_coverage_warning,
    multi_specimen_warning,
)


class RefineSpecError(Exception):
    """Raised for an op list that is malformed, before any op is applied."""


class RefinementErasedMaskError(Exception):
    """Raised when the refined mask is degenerate; no session is minted."""


# Published verbatim by list_methods(): name -> {doc, params: {name -> constraint}}.
# `example` is a value that must run on any non-empty mask; the test suite applies
# every documented op with its example so a published op can never be a dead one.
REFINE_OPS: dict[str, dict[str, Any]] = {
    "fill_holes": {
        "doc": "Fill enclosed background pixels inside the plant.",
        "params": {},
    },
    "fill": {
        "doc": "Remove connected components smaller than `size` pixels.",
        "params": {"size": {"type": "int", "min": 1, "example": 50}},
    },
    "erode": {
        "doc": "Shrink the mask by a `ksize` x `ksize` kernel, `iterations` times.",
        "params": {
            "ksize": {"type": "int", "min": 2, "example": 3},
            "iterations": {"type": "int", "min": 1, "default": 1, "example": 1},
        },
    },
    "dilate": {
        "doc": "Grow the mask by a `ksize` x `ksize` kernel, `iterations` times.",
        "params": {
            "ksize": {"type": "int", "min": 2, "example": 3},
            "iterations": {"type": "int", "min": 1, "default": 1, "example": 1},
        },
    },
    "opening": {
        "doc": "Erode then dilate: removes specks and thin bridges.",
        "params": {"ksize": {"type": "int", "min": 2, "example": 3}},
    },
    "closing": {
        "doc": "Dilate then erode: closes small gaps and holes.",
        "params": {"ksize": {"type": "int", "min": 2, "example": 3}},
    },
    "median_blur": {
        "doc": "Median filter; `ksize` must be odd.",
        "params": {"ksize": {"type": "int", "min": 3, "odd": True, "example": 3}},
    },
    "keep_largest": {
        "doc": "Keep only the `n` largest connected components.",
        "params": {"n": {"type": "int", "min": 1, "example": 1}},
    },
}

# Above this relative change in mask_fraction the refinement reshaped the mask
# enough that the overlay must be looked at, whatever the diagnostics say.
LARGE_CHANGE_FRACTION = 0.25

# A component that vanishes under one op counts as a dropped OBJECT (not a
# speck) when it was at least this fraction of the largest component present
# before that op. Matches the "major object" threshold order of magnitude in
# diagnostics: a leaf split off by opening() is ~10-40% of the plant; the
# specks keep_largest exists to remove are <1%.
DROPPED_OBJECT_FRACTION = 0.10
DROPPED_OBJECTS_LISTED = 3  # the advisory names the largest N; the rest are counted


@dataclass(frozen=True)
class DroppedObject:
    """A major component that one op removed entirely."""

    op_index: int
    op_name: str
    area: int
    largest_area: int  # the largest component present just before the op
    split_by_op_index: int | None  # the last earlier op that raised component_count
    split_by_op_name: str | None


def validate_ops(ops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Check every op and return the normalised list (defaults filled in).

    Raises RefineSpecError naming the first offending op by index. Nothing is
    applied here, so a bad third op costs nothing.
    """
    if not ops:
        raise RefineSpecError("ops is empty; give at least one operation.")
    normalised: list[dict[str, Any]] = []
    for i, raw in enumerate(ops):
        if not isinstance(raw, Mapping) or "op" not in raw:
            raise RefineSpecError(
                f"op {i}: each entry needs an 'op' name; got {raw!r}."
            )
        name = raw["op"]
        if name not in REFINE_OPS:
            raise RefineSpecError(
                f"op {i}: unknown op {name!r}. Valid: {sorted(REFINE_OPS)}."
            )
        spec = REFINE_OPS[name]["params"]
        extra = sorted(set(raw) - {"op"} - set(spec))
        if extra:
            raise RefineSpecError(
                f"op {i} ({name}): unknown parameter(s) {extra}; "
                f"{name} takes {sorted(spec) or 'no parameters'}."
            )
        out: dict[str, Any] = {"op": name}
        for pname, constraint in spec.items():
            if pname in raw:
                value = raw[pname]
            elif "default" in constraint:
                value = constraint["default"]
            else:
                raise RefineSpecError(f"op {i} ({name}): missing parameter {pname!r}.")
            # bool is an int subclass; True as a kernel size is a bug, not a 1.
            if isinstance(value, bool) or not isinstance(value, int):
                raise RefineSpecError(
                    f"op {i} ({name}): {pname} must be an integer, got {value!r}."
                )
            if value < constraint["min"]:
                raise RefineSpecError(
                    f"op {i} ({name}): {pname} must be >= {constraint['min']}, got {value}."
                )
            if constraint.get("odd") and value % 2 == 0:
                raise RefineSpecError(
                    f"op {i} ({name}): {pname} must be odd, got {value}."
                )
            out[pname] = value
        normalised.append(out)
    return normalised


def _keep_largest(mask: np.ndarray, n: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.argsort(areas)[::-1][:n] + 1  # label 0 is background
    return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)


def _apply_one(mask: np.ndarray, op: Mapping[str, Any]) -> np.ndarray:
    name = op["op"]
    if name == "fill_holes":
        return pcv.fill_holes(bin_img=mask)
    if name == "fill":
        return pcv.fill(bin_img=mask, size=op["size"])
    if name == "erode":
        return pcv.erode(gray_img=mask, ksize=op["ksize"], i=op["iterations"])
    if name == "dilate":
        return pcv.dilate(gray_img=mask, ksize=op["ksize"], i=op["iterations"])
    if name == "opening":
        kernel = np.ones((op["ksize"], op["ksize"]), np.uint8)
        return pcv.opening(gray_img=mask, kernel=kernel)
    if name == "closing":
        kernel = np.ones((op["ksize"], op["ksize"]), np.uint8)
        return pcv.closing(gray_img=mask, kernel=kernel)
    if name == "median_blur":
        return pcv.median_blur(gray_img=mask, ksize=op["ksize"])
    if name == "keep_largest":
        return _keep_largest(mask, op["n"])
    raise RefineSpecError(f"unhandled op {name!r}")  # unreachable after validate_ops


def _dropped_by(
    before: np.ndarray,
    after: np.ndarray,
    op_index: int,
    op_name: str,
    split_by: tuple[int, str] | None,
) -> list[DroppedObject]:
    """Components of `before` with no surviving pixel in `after`, if major."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (before > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.max())
    survivors = set(np.unique(labels[after > 0])) - {0}
    return [
        DroppedObject(
            op_index,
            op_name,
            int(a),
            largest,
            split_by[0] if split_by else None,
            split_by[1] if split_by else None,
        )
        for label, a in enumerate(areas, start=1)
        if label not in survivors and a >= DROPPED_OBJECT_FRACTION * largest
    ]


def apply_refinements_traced(
    mask: np.ndarray, ops: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, list[DroppedObject]]:
    """Apply a validated op list in order; return the new uint8 {0,255} mask
    and every major object an op discarded along the way.

    Raises RefineSpecError (nothing applied) or RefinementErasedMaskError (the
    result is degenerate and must not become a session).
    """
    validated = validate_ops(ops)
    out = np.where(mask > 0, 255, 0).astype(np.uint8)
    dropped: list[DroppedObject] = []
    split_by: tuple[int, str] | None = None
    n_components = analyze_mask(out).component_count
    for i, op in enumerate(validated):
        nxt = np.where(_apply_one(out, op) > 0, 255, 0).astype(np.uint8)
        dropped.extend(_dropped_by(out, nxt, i, op["op"], split_by))
        n_next = analyze_mask(nxt).component_count
        if n_next > n_components:
            split_by = (i, op["op"])
        n_components = n_next
        out = nxt
    before, after = analyze_mask(mask), analyze_mask(out)
    try:
        assert_not_degenerate(after)
    except DegenerateMaskError as exc:
        raise RefinementErasedMaskError(
            f"The refinement left no measurable plant, so no session was created. "
            f"before: mask_fraction={before.mask_fraction:.4f}, "
            f"component_count={before.component_count}, "
            f"largest_area={before.largest_area}; "
            f"after: mask_fraction={after.mask_fraction:.4f}, "
            f"component_count={after.component_count}, "
            f"largest_area={after.largest_area}. Use a smaller kernel, fewer "
            f"iterations, or a smaller fill size. ({exc})"
        ) from exc
    return out, dropped


def apply_refinements(mask: np.ndarray, ops: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """apply_refinements_traced without the trace."""
    return apply_refinements_traced(mask, ops)[0]


def dropped_object_warning(dropped: Sequence[DroppedObject]) -> Advisory | None:
    """One advisory naming every major object the op list threw away."""
    if not dropped:
        return None
    ranked = sorted(dropped, key=lambda d: d.area, reverse=True)
    parts = []
    for d in ranked[:DROPPED_OBJECTS_LISTED]:
        # One slot records the LAST op that raised the component count, so
        # with two splitting ops the first leaf is attributed to the second
        # split. Say what is actually known rather than which op cut it.
        origin = (
            f" (the last op that raised the component count before it was op "
            f"{d.split_by_op_index} ({d.split_by_op_name}))"
            if d.split_by_op_index is not None
            else ""
        )
        parts.append(
            f"op {d.op_index} ({d.op_name}) discarded a {d.area}-px object "
            f"({d.area / d.largest_area:.0%} of the largest, {d.largest_area} px){origin}"
        )
    rest = len(ranked) - DROPPED_OBJECTS_LISTED
    if rest > 0:
        parts.append(f"and {rest} more discarded objects")
    return Advisory(
        code="refine_dropped_object",
        message=(
            "; ".join(parts) + ". At that size it is more likely a leaf or a "
            "second specimen than a speck (objects under 10% of the largest are "
            "not reported): look at the overlay, and if it belongs to the plant, "
            "drop the op that split it (a smaller opening/erode kernel) or raise "
            "n on keep_largest."
        ),
    )


def refinement_warnings(
    mask: np.ndarray,
    before: MaskDiagnostics,
    after: MaskDiagnostics,
    dropped: Sequence[DroppedObject] = (),
) -> list[Advisory]:
    """Advisories for a refined mask: the shared mask guards, a change alarm,
    and any major object an op discarded."""
    warnings: list[Advisory] = []
    dropped_warning = dropped_object_warning(dropped)
    if dropped_warning:
        warnings.append(dropped_warning)
    coverage = implausible_coverage_warning(after)
    if coverage:
        warnings.append(coverage)
    multi = multi_specimen_warning(after)
    if multi:
        warnings.append(multi)
    if not coverage:
        clipping = frame_clipping_warning(mask)
        if clipping:
            warnings.append(clipping)
    if before.mask_fraction > 0:
        change = abs(after.mask_fraction - before.mask_fraction) / before.mask_fraction
        if change > LARGE_CHANGE_FRACTION:
            warnings.append(
                Advisory(
                    code="refine_large_change",
                    message=(
                        f"The refinement changed the mask by {change:.0%} "
                        f"(mask_fraction {before.mask_fraction:.4f} -> "
                        f"{after.mask_fraction:.4f}). That is a different plant "
                        "outline, not a cleanup: look at the overlay before "
                        "measuring."
                    ),
                )
            )
    return warnings
