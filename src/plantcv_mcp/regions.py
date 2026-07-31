"""Per-region measurement — the multi-plant case.

Phase 1 measured a whole image as ONE region of interest, so a tray of
seedlings could only ever raise `multi_specimen`: the tool could say "there are
several plants here and I am about to merge them" but not "here is each one".
This module measures each region separately.

**The trap this module exists to avoid.** Measured on PlantCV 4.11.3 with a
2x2 grid where one cell was deliberately left empty:

    create_labels returned n = 4          (all four ROIs, empties included)
    observation groups: default_1 .. default_4
    default_3 (the empty cell) area = 0.0

So an empty cell does NOT go missing — it reports a complete, plausible trait
set full of zeros, which is the same failure `assert_not_degenerate` exists to
stop in the single-region path. `np.unique(labeled)` is the discriminator: a
region whose label never appears in the labelled mask has no plant in it, and
this module returns a REASON for that region instead of a row of zeros.

That measurement also settled the index mapping: group `default_{i+1}`
corresponds to region `i`, with no shifting when a cell is empty. Had empties
been dropped, every trait after the first gap would have been attributed to the
wrong plant — silently, and in the direction of looking perfectly reasonable.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from plantcv import plantcv as pcv
from typing_extensions import TypedDict

from .diagnostics import (
    Advisory,
    analyze_mask,
    implausible_coverage_warning,
)
from .measurement import (
    ANALYSES,
    HISTOGRAM_TRAITS,
    TraitValue,
    UnknownAnalysisError,
    convert_units,
)

# Modes map one-to-one onto real PlantCV constructors. No mode is synthesised
# here, so a caller reading PlantCV's docs sees the same behaviour.
REGION_MODES: tuple[str, ...] = ("auto_grid", "rect_grid")

# A grid larger than this is far likelier to be a typo than a real tray, and
# every cell costs a full PlantCV analysis pass.
MAX_REGIONS: int = 400


class RegionMeasurement(TypedDict):
    """One region: either traits, or the reason there are none."""

    index: int
    row: int
    col: int
    bbox: list[int]
    measured: bool
    reason: Any
    region_coverage: float
    traits: Any
    warnings: list[dict[str, str]]


@dataclass
class RegionSet:
    """ROIs plus the geometry needed to draw and label them."""

    rois: Any
    bboxes: list[tuple[int, int, int, int]]
    nrows: int
    ncols: int
    mode: str
    warnings: list[Advisory] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.bboxes)


class RegionSpecError(ValueError):
    """Raised for a region specification PlantCV could not be given."""


def _bboxes_from(rois: Any) -> list[tuple[int, int, int, int]]:
    """Axis-aligned bounding box per ROI contour, in ROI order.

    Used for drawing and for per-region coverage. Order is preserved because the
    label-to-region mapping depends on it.

    PlantCV's `Objects.contours` is a list of LISTS of point arrays — one list
    per ROI, each holding one or more contours — not a flat list of arrays.
    Passing an entry straight to cv2.boundingRect raises "array is not a numpy
    array", which is how this was found. Entries are stacked so an ROI made of
    several contours still yields the box that encloses all of it.
    """
    boxes: list[tuple[int, int, int, int]] = []
    for entry in rois.contours:
        parts = entry if isinstance(entry, list | tuple) else [entry]
        points = np.vstack([np.asarray(p).reshape(-1, 1, 2) for p in parts])
        x, y, w, h = cv2.boundingRect(points)
        boxes.append((int(x), int(y), int(w), int(h)))
    return boxes


def build_regions(
    img: np.ndarray,
    mask: np.ndarray,
    mode: str = "auto_grid",
    nrows: int = 1,
    ncols: int = 1,
    coord: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    spacing: tuple[int, int] | None = None,
    radius: int | None = None,
) -> RegionSet:
    """Construct the regions to measure.

    auto_grid: PlantCV infers the grid geometry from the MASK itself, so the
        caller supplies only how many rows and columns of plants there are. This
        is the right default for a tray photographed from above, where the plants
        define the layout better than any hand-entered coordinate.
    rect_grid: explicit rectangles — top-left `coord`, cell `height`/`width`, and
        `spacing` between cell origins. Use when the layout is known, when cells
        must line up with physical pots, or when the mask is too sparse for
        auto_grid to infer anything.
    """
    if mode not in REGION_MODES:
        raise RegionSpecError(f"Unknown mode {mode!r}. Valid: {list(REGION_MODES)}.")
    if nrows < 1 or ncols < 1:
        raise RegionSpecError(f"nrows and ncols must be >= 1, got {nrows}x{ncols}.")
    if nrows * ncols > MAX_REGIONS:
        raise RegionSpecError(
            f"{nrows}x{ncols} = {nrows * ncols} regions exceeds the {MAX_REGIONS} "
            "cap. Each region costs a full analysis pass; split the image instead."
        )

    warnings: list[Advisory] = []

    if mode == "auto_grid":
        if not (mask > 0).any():
            raise RegionSpecError(
                "auto_grid infers the grid from the mask, and this mask is empty, "
                "so there is no layout to infer. Re-run segment() first, or use "
                "mode='rect_grid' to state the geometry explicitly."
            )
        rois = pcv.roi.auto_grid(
            mask=mask, nrows=nrows, ncols=ncols, radius=radius, img=img
        )
    else:
        missing = [
            name
            for name, value in (
                ("coord", coord),
                ("height", height),
                ("width", width),
            )
            if value is None
        ]
        if missing:
            raise RegionSpecError(
                f"mode='rect_grid' needs {missing} as well as nrows/ncols. "
                "Nothing is guessed here: a wrong cell origin silently measures "
                "the neighbouring plant."
            )
        if (nrows > 1 or ncols > 1) and spacing is None:
            raise RegionSpecError(
                "mode='rect_grid' with more than one cell needs `spacing` "
                "(x, y) between cell origins."
            )
        rois = pcv.roi.multi_rect(
            img=img,
            coord=coord,
            h=height,
            w=width,
            spacing=spacing,
            nrows=nrows,
            ncols=ncols,
        )

    bboxes = _bboxes_from(rois)
    if len(bboxes) != nrows * ncols:
        # Not an error: auto_grid can return fewer if it cannot resolve the
        # layout. Say so rather than silently measuring a different grid.
        warnings.append(
            Advisory(
                code="region_count_mismatch",
                message=(
                    f"Asked for {nrows}x{ncols} = {nrows * ncols} regions but "
                    f"{len(bboxes)} were constructed. Row/column numbering below "
                    "follows the regions that exist, so it may not line up with "
                    "the physical tray. Use mode='rect_grid' to pin the geometry."
                ),
            )
        )

    return RegionSet(
        rois=rois,
        bboxes=bboxes,
        nrows=nrows,
        ncols=ncols,
        mode=mode,
        warnings=warnings,
    )


def _read_group(index: int) -> dict[str, dict]:
    """Read observation group `{label}_{index}` by explicit key.

    Keyed rather than positional for the same reason the single-region path is:
    a group that is not there must raise, not silently resolve to a neighbour's.
    """
    key = f"{pcv.params.sample_label}_{index}"
    if key not in pcv.outputs.observations:
        raise KeyError(
            f"Expected observation group {key!r} not found. Available: "
            f"{list(pcv.outputs.observations.keys())}. PlantCV's labelling "
            "behaviour may have changed."
        )
    return pcv.outputs.observations[key]


def measure_regions(
    img: np.ndarray,
    mask: np.ndarray,
    regions: RegionSet,
    analyses: tuple[str, ...] = ("size",),
    px_per_mm: float | None = None,
    include_histograms: bool = False,
) -> list[RegionMeasurement]:
    """Measure each region, refusing the empty ones by name.

    An empty region returns `measured=False` and a reason. It does NOT return
    zeros: PlantCV reports area 0.0 with a full trait set for a cell containing
    nothing, and a zero that looks like a measurement is worse than no answer.
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

    saved = dict(pcv.outputs.observations)
    pcv.outputs.clear()
    try:
        labeled, n = pcv.create_labels(mask=mask, rois=regions.rois, roi_type="partial")
        # Labels actually carrying pixels. A region whose label is absent here is
        # empty -- this is the only reliable discriminator, since the analysis
        # emits a full zero-valued group for it either way.
        present = {int(v) for v in np.unique(labeled)} - {0}

        if "size" in analyses:
            pcv.analyze.size(img=img, labeled_mask=labeled, n_labels=n)
        if "color" in analyses:
            pcv.analyze.color(
                rgb_img=img, labeled_mask=labeled, n_labels=n, colorspaces="hsv"
            )

        out: list[RegionMeasurement] = []
        for i, bbox in enumerate(regions.bboxes):
            label = i + 1
            x, y, w, h = bbox
            row, col = divmod(i, regions.ncols) if regions.ncols else (0, i)

            cell = labeled[y : y + h, x : x + w] if w and h else labeled[:0, :0]
            coverage = (
                float((cell == label).sum()) / float(cell.size) if cell.size else 0.0
            )

            if label not in present:
                out.append(
                    RegionMeasurement(
                        index=i,
                        row=row,
                        col=col,
                        bbox=list(bbox),
                        measured=False,
                        reason=(
                            "No plant material in this region. PlantCV reports a "
                            "complete trait set of zeros for an empty region, which "
                            "is indistinguishable from a real zero-area plant, so "
                            "no traits are returned for it."
                        ),
                        region_coverage=0.0,
                        traits=None,
                        warnings=[],
                    )
                )
                continue

            traits: dict[str, TraitValue] = {
                name: TraitValue(value=obs.get("value"), unit=obs.get("label"))
                for name, obs in _read_group(label).items()
            }
            if not include_histograms:
                traits = {k: v for k, v in traits.items() if k not in HISTOGRAM_TRAITS}
            if px_per_mm is not None:
                traits = convert_units(traits, float(px_per_mm))

            region_warnings: list[Advisory] = []
            cover = implausible_coverage_warning(
                analyze_mask((cell == label).astype(np.uint8) * 255)
            )
            if cover:
                region_warnings.append(cover)

            out.append(
                RegionMeasurement(
                    index=i,
                    row=row,
                    col=col,
                    bbox=list(bbox),
                    measured=True,
                    reason=None,
                    region_coverage=coverage,
                    traits=traits,
                    warnings=[
                        {"code": w.code, "message": w.message} for w in region_warnings
                    ],
                )
            )
        return out
    finally:
        pcv.outputs.clear()
        pcv.outputs.observations.update(saved)
