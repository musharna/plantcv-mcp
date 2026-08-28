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


def apply_refinements(mask: np.ndarray, ops: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Apply a validated op list in order; return the new uint8 {0,255} mask.

    Raises RefineSpecError (nothing applied) or RefinementErasedMaskError (the
    result is degenerate and must not become a session).
    """
    validated = validate_ops(ops)
    out = np.where(mask > 0, 255, 0).astype(np.uint8)
    for op in validated:
        out = np.where(_apply_one(out, op) > 0, 255, 0).astype(np.uint8)
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
    return out


def refinement_warnings(
    mask: np.ndarray, before: MaskDiagnostics, after: MaskDiagnostics
) -> list[Advisory]:
    """Advisories for a refined mask: the shared mask guards plus a change alarm."""
    warnings: list[Advisory] = []
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
