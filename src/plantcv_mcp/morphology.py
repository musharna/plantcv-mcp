"""Skeleton-based morphology traits: leaves, stem, branch points, cycles.

PlantCV's morphology pipeline reports almost everything through the process-global
`pcv.outputs` (15 of 19 functions), so the whole run sits inside
`isolated_pcv_outputs()` — the same lock `measure()` and `measure_regions()` take.

What this module adds to a pass-through, all measured on PlantCV 4.11.3 with a
synthetic plant of known geometry (docs/superpowers/specs/2026-08-27-morphology-design.md):

* A perfectly vertical stem makes `analyze_stem` return stem_angle = -14373
  degrees (its slope blows up). A number that is not an angle is not returned as
  one: the value becomes null with a named warning.
* `prune_size` decides how many segments a skeleton has. If the count changes by
  more than 30% at twice the prune size, the traits are parameter-sensitive and
  the caller is told so.
* A mask with several plants produces one merged skeleton; it is refused by name,
  pointing at measure_regions() and refine(keep_largest).
* The numbered-segment overlay is returned with the table, because a per-segment
  number is unreadable without the picture saying which segment is which.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import (
    Advisory,
    analyze_mask,
    assert_not_degenerate,
    multi_specimen_warning,
)
from .imaging import render_overlay
from .measurement import isolated_pcv_outputs

LABEL = "morphology"
SENSITIVITY_FRACTION = 0.30
# Doubling prune_size and keeping this share of the segments means pruning is
# not the lever (real sorghum photo: 126 segments at 100 AND at 200).
PRUNE_STALL_FRACTION = 0.80

# Traits that scale with ONE spatial dimension (px_per_mm applies); everything
# angular is in degrees and is never scaled; curvature is a ratio.
LINEAR_PLANT_TRAITS = (
    "stem_height",
    "stem_length",
    "mean_segment_width",
    "segment_width_std",
    "segment_width_max",
)
LINEAR_SEGMENT_TRAITS = ("path_length", "euclidean_length")
ANGULAR_SEGMENT_TRAITS = ("angle", "tangent_angle", "insertion_angle")


class MorphologyRefusedError(Exception):
    """Raised when the mask cannot yield a meaningful single-plant skeleton."""


@dataclass
class MorphologyResult:
    plant: dict[str, Any]
    segments: list[dict[str, Any]]
    units: dict[str, str]
    warnings: list[Advisory]
    overlay: np.ndarray
    prune_size: int
    tangent_size: int


def _f(value: Any) -> float | None:
    """PlantCV emits None, numpy floats, and the STRING 'NA'; all become null."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _group(observations: dict, key: str) -> Any:
    """Read one observation's value from every group PlantCV filed under our label.

    Functions disagree on whether the label becomes the group name outright or
    gets a `_1` suffix (segment_width does the latter), so both are consulted.
    """
    for name, obs in observations.items():
        if (name == LABEL or name.startswith(LABEL + "_")) and key in obs:
            return obs[key].get("value")
    return None


def _segment_count(pruned_source: np.ndarray, mask: np.ndarray, size: int) -> int:
    pruned, _, _ = pcv.morphology.prune(skel_img=pruned_source, size=size, mask=mask)
    _, segments = pcv.morphology.segment_skeleton(skel_img=pruned, mask=mask)
    return len(segments)


def measure_morphology(
    img: np.ndarray,
    mask: np.ndarray,
    prune_size: int = 15,
    tangent_size: int = 25,
    px_per_mm: float | None = None,
) -> MorphologyResult:
    """tangent_size default of 25 px is measured, not guessed: insertion-angle bias
    on a known plant fell from 24.5 deg (10) to 5.9 deg (25) to 4.2 deg (30), and
    then a window longer than a leaf collapsed that leaf's angle to 0.0 (40 on a
    70 px leaf). 25 keeps margin before that cliff on small plants."""
    if prune_size < 0:
        raise ValueError(f"prune_size must be >= 0, got {prune_size}")
    if tangent_size < 1:
        raise ValueError(f"tangent_size must be >= 1, got {tangent_size}")
    if px_per_mm is not None and (px_per_mm <= 0 or not math.isfinite(px_per_mm)):
        raise ValueError(f"px_per_mm must be a positive finite number, got {px_per_mm}")

    diag = analyze_mask(mask)
    assert_not_degenerate(diag)
    if multi_specimen_warning(diag):
        raise MorphologyRefusedError(
            f"The mask holds {diag.major_object_count} comparably sized objects; a "
            "skeleton of several plants merges their leaves and stems into one "
            "meaningless graph. Use measure_regions() for a tray, or "
            "refine(keep_largest=1) to isolate one plant, then measure that session."
        )

    warnings: list[Advisory] = []
    mask255 = np.where(mask > 0, 255, 0).astype(np.uint8)

    n_here = n_double = 0
    try:
        with isolated_pcv_outputs():
            skel = pcv.morphology.skeletonize(mask=mask255)
            pruned, _, _ = pcv.morphology.prune(
                skel_img=skel, size=prune_size, mask=mask255
            )
            _, segments = pcv.morphology.segment_skeleton(skel_img=pruned, mask=mask255)
            if not segments:
                raise MorphologyRefusedError(
                    "Skeletonisation produced no segments: the mask is too small or too "
                    "compact to carry a skeleton. Check the overlay from segment()."
                )
            leaf_objects, stem_objects = pcv.morphology.segment_sort(
                skel_img=pruned, objects=segments, mask=mask255, first_stem=True
            )
            # Sensitivity first, because it is cheap and because the per-segment
            # functions below can abort on a fragmented skeleton — and the refusal
            # then needs these counts to say what to do.
            n_here = len(segments)
            n_double = _segment_count(skel, mask255, max(2 * prune_size, 1))

            # A skeleton with no leaf-like segment (a bare stem, a ring) is still
            # reported — cycles, tips, branch points and stem traits are exactly
            # what tells the user WHY there are no leaves — with an empty table.
            if leaf_objects:
                # segment_id reuses a process-global cached palette
                # (params.saved_color_scale, color_palette(saved=True)) and
                # indexes past it when this plant has more segments than the
                # one that filled the cache: IndexError from inside PlantCV.
                # Reset it here, inside the lock, and hand it back afterwards.
                saved_palette = pcv.params.saved_color_scale
                pcv.params.saved_color_scale = None
                try:
                    # Two returns: the plain colored segments (what the other
                    # segment_* functions consume) and the labeled copy with the
                    # id DIGITS drawn on. The overlay must use the labeled one —
                    # the table's `id` column is unreadable without the digits.
                    segmented_img, id_img = pcv.morphology.segment_id(
                        skel_img=pruned, objects=leaf_objects, mask=mask255
                    )
                finally:
                    pcv.params.saved_color_scale = saved_palette
                pcv.morphology.segment_path_length(
                    segmented_img=segmented_img, objects=leaf_objects, label=LABEL
                )
                pcv.morphology.segment_euclidean_length(
                    segmented_img=segmented_img, objects=leaf_objects, label=LABEL
                )
                pcv.morphology.segment_curvature(
                    segmented_img=segmented_img, objects=leaf_objects, label=LABEL
                )
                pcv.morphology.segment_angle(
                    segmented_img=segmented_img, objects=leaf_objects, label=LABEL
                )
                pcv.morphology.segment_tangent_angle(
                    segmented_img=segmented_img,
                    objects=leaf_objects,
                    size=tangent_size,
                    label=LABEL,
                )
            else:
                segmented_img = np.zeros_like(img)
                segmented_img[pruned > 0] = (255, 255, 255)
                id_img = segmented_img
                warnings.append(
                    Advisory(
                        code="no_leaf_segments",
                        message=(
                            "Every skeleton segment was classed as stem, so the segment "
                            "table is empty. A closed outline (see num_cycles) or a bare "
                            "stem does this; refine() with fill_holes, or check the "
                            "overlay."
                        ),
                    )
                )
            if stem_objects:
                if leaf_objects:
                    pcv.morphology.segment_insertion_angle(
                        skel_img=pruned,
                        segmented_img=segmented_img,
                        leaf_objects=leaf_objects,
                        stem_objects=stem_objects,
                        size=tangent_size,
                        label=LABEL,
                    )
                pcv.morphology.analyze_stem(
                    rgb_img=img, stem_objects=stem_objects, label=LABEL
                )
            else:
                warnings.append(
                    Advisory(
                        code="no_stem_segment",
                        message=(
                            "No segment was classed as stem, so insertion angles and stem "
                            "traits are unavailable for this mask."
                        ),
                    )
                )
            pcv.morphology.find_tips(skel_img=pruned, mask=mask255, label=LABEL)
            pcv.morphology.find_branch_pts(skel_img=pruned, mask=mask255, label=LABEL)
            pcv.morphology.check_cycles(skel_img=pruned, label=LABEL)
            if leaf_objects:
                pcv.morphology.segment_width(
                    segmented_img=segmented_img,
                    skel_img=pruned,
                    labeled_mask=np.where(mask > 0, 1, 0).astype(np.uint8),
                    n_labels=1,
                    label=LABEL,
                )
            observations = {k: dict(v) for k, v in pcv.outputs.observations.items()}
    except RuntimeError as exc:
        # PlantCV's fatal_error() on a skeleton it cannot analyse ("Too many
        # tips found per segment, try pruning again"). Refuse with the
        # sensitivity numbers, which are exactly what the user needs to act.
        counts = (
            f"prune_size={prune_size} leaves {n_here} segments and prune_size="
            f"{max(2 * prune_size, 1)} leaves {n_double}"
        )
        if n_here and n_double >= PRUNE_STALL_FRACTION * n_here:
            # Doubling the prune barely moves the count: the spurs are wider
            # than any prune reaches (a jagged mask edge on a real photo), and
            # "raise prune_size" sends the user up a ladder that never ends.
            remedy = (
                "; raising prune_size does not help here. refine() the mask "
                "(median_blur, opening) so the skeleton has fewer spurs, then "
                "re-measure."
            )
        else:
            remedy = (
                "; raise prune_size, or refine() the mask (opening, median_blur) "
                "so the skeleton has fewer spurs, then re-measure."
            )
        raise MorphologyRefusedError(
            f"PlantCV could not analyse this skeleton: {exc}. {counts}{remedy}"
        ) from exc

    if n_here and abs(n_double - n_here) / n_here > SENSITIVITY_FRACTION:
        warnings.append(
            Advisory(
                code="prune_size_sensitive",
                message=(
                    f"prune_size={prune_size} yields {n_here} skeleton segments but "
                    f"prune_size={max(2 * prune_size, 1)} yields {n_double}. The "
                    "segment table depends on this parameter more than on the plant; "
                    "refine() the mask (opening, median_blur) or choose prune_size "
                    "deliberately from the overlay."
                ),
            )
        )

    def per_segment(key: str) -> list[float | None]:
        values = _group(observations, key) or []
        return [_f(v) for v in values]

    path = per_segment("segment_path_length")
    euclid = per_segment("segment_eu_length")
    curvature = per_segment("segment_curvature")
    angle = per_segment("segment_angle")
    tangent = per_segment("segment_tangent_angle")
    insertion = per_segment("segment_insertion_angle")
    n = len(leaf_objects)

    def at(values: list[float | None], i: int) -> float | None:
        return values[i] if i < len(values) else None

    segments_out: list[dict[str, Any]] = [
        {
            "id": i,
            "path_length": at(path, i),
            "euclidean_length": at(euclid, i),
            "curvature": at(curvature, i),
            "angle": at(angle, i),
            "tangent_angle": at(tangent, i),
            "insertion_angle": at(insertion, i),
        }
        for i in range(n)
    ]

    # PlantCV takes `size` pixels from EACH end of a segment for the tangent fit
    # (segment_tangent_angle.py: "if size*2 > len(obj) then pruning will remove
    # the segment completely"), and then reports that leaf's insertion angle as
    # exactly 0.0 — a window artefact that reads like a measurement. Say so
    # instead (checked in pixels, before any scaling).
    short = [
        s["id"]
        for s in segments_out
        if s["path_length"] is not None and 2 * tangent_size > s["path_length"]
    ]
    if short:
        warnings.append(
            Advisory(
                code="tangent_window_exceeds_segment",
                message=(
                    f"2 x tangent_size={tangent_size} px exceeds the length of "
                    f"segment(s) {short}; PlantCV fits the tangent on `size` pixels "
                    "from each end, so their tangent and insertion angles are "
                    "window artefacts (typically 0.0), not measurements. Use a "
                    "smaller tangent_size for this plant."
                ),
            )
        )

    num_cycles = _group(observations, "num_cycles")
    stem_angle = _f(_group(observations, "stem_angle"))
    if stem_objects and (stem_angle is None or not -180 <= stem_angle <= 180):
        warnings.append(
            Advisory(
                code="stem_angle_undefined",
                message=(
                    f"PlantCV reported stem_angle={stem_angle}, which is not an angle: "
                    "its slope-based estimate is undefined for a (near-)vertical stem. "
                    "Returned as null rather than as a number."
                ),
            )
        )
        stem_angle = None

    widths = _group(observations, "mean_segment_width") or []
    width_std = _group(observations, "segment_width_std") or []
    width_max = _group(observations, "segment_width_max") or []
    plant: dict[str, Any] = {
        "leaf_count": n,
        "stem_count": len(stem_objects),
        "tip_count": len(_group(observations, "tips") or []),
        "branch_point_count": len(_group(observations, "branch_pts") or []),
        "num_cycles": int(num_cycles) if num_cycles is not None else 0,
        "stem_height": _f(_group(observations, "stem_height")),
        "stem_length": _f(_group(observations, "stem_length")),
        "stem_angle": stem_angle,
        "mean_segment_width": _f(widths[0]) if widths else None,
        "segment_width_std": _f(width_std[0]) if width_std else None,
        "segment_width_max": _f(width_max[0]) if width_max else None,
    }

    if plant["num_cycles"] > 0:
        warnings.append(
            Advisory(
                code="skeleton_has_cycles",
                message=(
                    f"The skeleton contains {plant['num_cycles']} closed loop(s): the "
                    "mask has holes (a gap between overlapping leaves, or noise). "
                    "Segment lengths around a loop are not leaf lengths. "
                    "refine() with fill_holes and re-measure."
                ),
            )
        )

    linear_unit = "pixels"
    if px_per_mm is not None:
        linear_unit = "mm"
        for key in LINEAR_PLANT_TRAITS:
            if plant[key] is not None:
                plant[key] = plant[key] / px_per_mm
        for seg in segments_out:
            for key in LINEAR_SEGMENT_TRAITS:
                if seg[key] is not None:
                    seg[key] = seg[key] / px_per_mm

    units = {key: linear_unit for key in LINEAR_PLANT_TRAITS + LINEAR_SEGMENT_TRAITS}
    units.update({key: "degrees" for key in ANGULAR_SEGMENT_TRAITS + ("stem_angle",)})
    units["curvature"] = "ratio"

    # The overlay: the tinted mask with PlantCV's numbered segments — digits
    # included — drawn on top.
    overlay = render_overlay(img, mask255)
    drawn = id_img.any(axis=2)
    overlay[drawn] = id_img[drawn]

    return MorphologyResult(
        plant=plant,
        segments=segments_out,
        units=units,
        warnings=warnings,
        overlay=overlay,
        prune_size=prune_size,
        tangent_size=tangent_size,
    )
