"""Trait extraction, gated on mask validity."""

import math
from typing import Any, TypedDict

import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import analyze_mask, assert_not_degenerate


class TraitValue(TypedDict):
    """A single measured trait and the unit it is expressed in."""

    value: Any
    unit: Any


ANALYSES: tuple[str, ...] = ("size", "color")

# Traits whose value scales with ONE spatial dimension.
LINEAR_TRAITS: frozenset[str] = frozenset(
    {
        "perimeter",
        "total_edge_length",
        "width",
        "height",
        "longest_path",
        "ellipse_major_axis",
        "ellipse_minor_axis",
    }
)

# Traits whose value scales with TWO spatial dimensions.
AREA_TRAITS: frozenset[str] = frozenset({"area", "convex_hull_area"})

# The histograms are 180 + 256 + 256 = 692 numbers. Useful for a plot, ruinous
# for a model's context window, so they are opt-in.
HISTOGRAM_TRAITS: frozenset[str] = frozenset(
    {"hue_frequencies", "saturation_frequencies", "value_frequencies"}
)


class UnknownAnalysisError(Exception):
    """Raised for an analysis outside ANALYSES."""


def convert_units(
    traits: dict[str, TraitValue], px_per_mm: float
) -> dict[str, TraitValue]:
    """Convert pixel traits to millimetres.

    **PlantCV labels both `area` and `width` as "pixels".** Scaling everything
    carrying that label by px_per_mm would leave every area wrong by exactly a
    factor of px_per_mm, silently and plausibly. Which traits are linear and
    which are areal is therefore an explicit table above, never inferred from
    the unit string.

    Positions (`center_of_mass`, `ellipse_center`) are left in pixels: without a
    defined origin, a millimetre coordinate is meaningless.
    """
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm must be > 0, got {px_per_mm}")

    out: dict[str, TraitValue] = {}
    for name, trait in traits.items():
        value = trait.get("value")
        if name in LINEAR_TRAITS and isinstance(value, int | float):
            out[name] = {"value": value / px_per_mm, "unit": "mm"}
        elif name in AREA_TRAITS and isinstance(value, int | float):
            out[name] = {"value": value / (px_per_mm**2), "unit": "mm2"}
        else:
            out[name] = dict(trait)
    return out


def _read_group() -> dict[str, dict]:
    """Read the observation group PlantCV just wrote, by explicit key."""
    expected_key = f"{pcv.params.sample_label}_1"
    if expected_key not in pcv.outputs.observations:
        raise KeyError(
            f"Expected observation group '{expected_key}' not found. "
            f"Available keys: {list(pcv.outputs.observations.keys())}. "
            f"This may indicate a change in PlantCV's labeling behavior or "
            f"an incomplete analysis."
        )
    return pcv.outputs.observations[expected_key]


def measure_traits(
    img: np.ndarray,
    mask: np.ndarray,
    analyses: tuple[str, ...] = ("size",),
    px_per_mm: float | None = None,
    include_histograms: bool = False,
) -> dict[str, TraitValue]:
    """Return PlantCV traits for a mask.

    Raises DegenerateMaskError BEFORE calling PlantCV when the mask is empty —
    PlantCV would otherwise return a full 17-trait set of zeros with
    in_bounds=True, which is indistinguishable from a real zero-area plant.

    analyses: any of ("size", "color"). "color" adds hue/saturation/value
        statistics; its three frequency histograms are 692 numbers in total and
        are omitted unless include_histograms is set.
    px_per_mm: when given, spatial traits are converted to mm and mm2. See
        convert_units for why the mapping is explicit rather than unit-derived.
    """
    unknown = [a for a in analyses if a not in ANALYSES]
    if unknown:
        raise UnknownAnalysisError(
            f"Unknown analyses {unknown}. Valid: {list(ANALYSES)}."
        )
    if not analyses:
        raise UnknownAnalysisError(
            f"No analyses requested. Choose at least one of {list(ANALYSES)}."
        )
    if px_per_mm is not None and (
        px_per_mm <= 0 or not math.isfinite(float(px_per_mm))
    ):
        raise ValueError(f"px_per_mm must be a positive finite number, got {px_per_mm}")

    assert_not_degenerate(analyze_mask(mask))

    # pcv.outputs is PROCESS-GLOBAL. Clearing it outright destroyed the
    # observations of any host application that also uses PlantCV directly, so
    # snapshot and restore instead. We still start from an empty table so our own
    # keyed lookup cannot pick up a foreign group.
    saved = dict(pcv.outputs.observations)
    pcv.outputs.clear()
    try:
        roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
        labeled, n = pcv.create_labels(mask=mask, rois=roi, roi_type="partial")

        if "size" in analyses:
            pcv.analyze.size(img=img, labeled_mask=labeled, n_labels=n)
        if "color" in analyses:
            pcv.analyze.color(
                rgb_img=img, labeled_mask=labeled, n_labels=n, colorspaces="hsv"
            )

        traits = {
            name: {"value": obs.get("value"), "unit": obs.get("label")}
            for name, obs in _read_group().items()
        }
    finally:
        # Restore unconditionally — an exception mid-analysis must not leave the
        # host's observations destroyed.
        pcv.outputs.clear()
        pcv.outputs.observations.update(saved)

    if not include_histograms:
        traits = {k: v for k, v in traits.items() if k not in HISTOGRAM_TRAITS}
    if px_per_mm is not None:
        traits = convert_units(traits, float(px_per_mm))
    return traits
