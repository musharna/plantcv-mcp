"""The analysis entry points, in one place, with picklable arguments and results.

Every function here is what a tool calls AFTER it has resolved its session and
loaded its image, and BEFORE it renders anything: pure compute over numpy arrays
and plain values. That boundary is what lets `workers.py` run the same function
in a subprocess without the server changing shape — the tool layer calls
`workers.dispatch("measure", ...)` and gets the same return value either way.

Nothing in this module may depend on the session store or the MCP server.
"""

from typing import Any

import numpy as np

from .batch import measure_batch
from .hyperspectral import (
    CubeLoad,
    HsiSegmentation,
    SpectralResult,
    measure_spectral,
    segment_hyperspectral,
)
from .measurement import measure_traits
from .morphology import MorphologyResult, measure_morphology
from .refine import DroppedObject, apply_refinements_traced
from .regions import build_regions, grid_misalignment_warning, measure_regions
from .thermal import (
    ThermalLoad,
    ThermalResult,
    ThermalSegmentation,
    measure_thermal,
    segment_thermal,
)


def measure(
    img: np.ndarray,
    mask: np.ndarray,
    analyses: tuple[str, ...],
    px_per_mm: float | None,
    include_histograms: bool,
) -> dict:
    return measure_traits(
        img,
        mask,
        analyses=analyses,
        px_per_mm=px_per_mm,
        include_histograms=include_histograms,
    )


def regions(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    mode: str,
    nrows: int,
    ncols: int,
    coord: tuple[int, int] | None,
    height: int | None,
    width: int | None,
    spacing: tuple[int, int] | None,
    radius: int | None,
    analyses: tuple[str, ...],
    px_per_mm: float | None,
    include_histograms: bool,
) -> dict[str, Any]:
    """build_regions + measure_regions, returning only picklable geometry.

    RegionSet carries PlantCV's ROI objects, which the server never needs once
    the bboxes are known; they stay on this side of the boundary.
    """
    region_set = build_regions(
        img,
        mask,
        mode=mode,
        nrows=nrows,
        ncols=ncols,
        coord=coord,
        height=height,
        width=width,
        spacing=spacing,
        radius=radius,
    )
    measurements = measure_regions(
        img,
        mask,
        region_set,
        analyses=analyses,
        px_per_mm=px_per_mm,
        include_histograms=include_histograms,
    )
    set_warnings = list(region_set.warnings)
    misaligned = grid_misalignment_warning(region_set.mode, measurements)
    if misaligned:
        set_warnings.append(misaligned)
    return {
        "measurements": measurements,
        "bboxes": list(region_set.bboxes),
        "mode": region_set.mode,
        "nrows": region_set.nrows,
        "ncols": region_set.ncols,
        "warnings": [(w.code, w.message) for w in set_warnings],
    }


def morphology(
    img: np.ndarray,
    mask: np.ndarray,
    prune_size: int,
    tangent_size: int,
    px_per_mm: float | None,
) -> MorphologyResult:
    return measure_morphology(
        img,
        mask,
        prune_size=prune_size,
        tangent_size=tangent_size,
        px_per_mm=px_per_mm,
    )


def batch(image_paths: list[str], **recipe: Any) -> dict:
    return measure_batch(image_paths, **recipe)


def refine(mask: np.ndarray, ops: list[dict]) -> tuple[np.ndarray, list[DroppedObject]]:
    return apply_refinements_traced(mask, ops)


def hsi_segment(cube_load: CubeLoad, **kwargs: Any) -> HsiSegmentation:
    return segment_hyperspectral(cube_load.raw_path, cube_load=cube_load, **kwargs)


def hsi_measure(cube_load: CubeLoad, mask: np.ndarray, **kwargs: Any) -> SpectralResult:
    return measure_spectral(cube_load.raw_path, mask, cube_load=cube_load, **kwargs)


def thermal_segment(load: ThermalLoad, path: str, **kwargs: Any) -> ThermalSegmentation:
    return segment_thermal(path, load=load, **kwargs)


def thermal_measure(
    load: ThermalLoad, path: str, mask: np.ndarray, **kwargs: Any
) -> ThermalResult:
    return measure_thermal(path, mask, load=load, **kwargs)


REGISTRY: dict[str, Any] = {
    "hsi_segment": hsi_segment,
    "hsi_measure": hsi_measure,
    "thermal_segment": thermal_segment,
    "thermal_measure": thermal_measure,
    "measure": measure,
    "regions": regions,
    "morphology": morphology,
    "batch": batch,
    "refine": refine,
}
