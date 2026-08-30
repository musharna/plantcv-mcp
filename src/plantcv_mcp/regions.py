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
    DegenerateMaskError,
    analyze_mask,
    assert_not_degenerate,
    implausible_longest_path_warning,
    multi_specimen_warning,
)
from .hyperspectral import compute_index
from .measurement import (
    ANALYSES,
    HISTOGRAM_TRAITS,
    TraitValue,
    UnknownAnalysisError,
    convert_units,
    isolated_pcv_outputs,
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
    if radius is not None and radius <= 0:
        raise RegionSpecError(f"radius must be positive, got {radius}.")

    warnings: list[Advisory] = []

    if mode == "auto_grid":
        if not (mask > 0).any():
            raise RegionSpecError(
                "auto_grid infers the grid from the mask, and this mask is empty, "
                "so there is no layout to infer. Re-run segment() first, or use "
                "mode='rect_grid' to state the geometry explicitly."
            )
        try:
            rois = pcv.roi.auto_grid(
                mask=mask, nrows=nrows, ncols=ncols, radius=radius, img=img
            )
        except (ValueError, cv2.error) as exc:
            # PlantCV fits one mixture component per row and per column:
            # fewer objects than components is sklearn's "Found array with 1
            # sample(s)" ValueError; objects that do not spread into the rows
            # asked give it NaN centres, and OpenCV refuses to draw them.
            # Both leaked raw — the batch quoted sklearn as its reason.
            n_objects = int(cv2.connectedComponents((mask > 0).astype(np.uint8))[0]) - 1
            raise RegionSpecError(
                f"auto_grid could not infer a {nrows}x{ncols} layout from the "
                f"{n_objects} object(s) in this mask. It fits one cluster per "
                "row and per column, so it needs at least that many objects, "
                "spread over the rows and columns asked. Give the geometry with "
                "mode='rect_grid', or measure() a single plant. "
                f"(PlantCV: {type(exc).__name__}: {str(exc).splitlines()[0][:120]})"
            ) from exc
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
        # `height`/`width`/`coord` are checked for None above, so the narrowing
        # here is for the type checker; the values are real by this point.
        assert height is not None and width is not None and coord is not None
        if height <= 0 or width <= 0:
            raise RegionSpecError(
                f"height and width must be positive, got {height}x{width}. A "
                "zero-height cell measured a two-pixel sliver and called it a "
                "plant; a negative one is a cell drawn backwards."
            )
        if spacing is None:
            # Required even for one cell: pinned PlantCV rejects a grid without
            # it, and its message blames the wrong argument.
            raise RegionSpecError(
                "mode='rect_grid' needs `spacing` [x, y] between cell origins, "
                "even for a single cell (use [0, 0] there)."
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

    if mode == "rect_grid":
        # Hand-entered geometry can put a cell partly or wholly off the frame.
        # Such a rectangle does not measure nothing — it reached native
        # OpenCV/PlantCV code inside measure_regions and killed the process
        # with SIGSEGV (PlantCV 4.11.3: coord=(-10,-10), 50x50 on a 100x100
        # image). Refuse it here, before anything native runs.
        img_h, img_w = int(mask.shape[0]), int(mask.shape[1])
        for i, (x, y, w, h) in enumerate(bboxes):
            if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
                row, col = divmod(i, ncols)
                raise RegionSpecError(
                    f"Region row {row} col {col} at ({x}, {y}) size {w}x{h} lies "
                    f"outside the {img_w}x{img_h} image. Every cell must fit "
                    "entirely inside the frame; check coord, height, width and "
                    "spacing."
                )
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


OWNED_MATERIAL_FRACTION = 0.2
"""Below this share of the cell's mask material, the cell's own object is a
fragment of a neighbour's (see partition_regions)."""

EXCEEDS_CELL_RATIO = 1.25
"""Object bbox / cell bbox above which the object is not this cell's plant.

Measured on real trays: a clean auto_grid draws tight cells around centroids,
so leaf tips overhang (ratios up to 1.02, half a plant's pixels outside the
cell) while the traits are still that one plant's, because partial labelling
measures the whole object. A misaligned grid that merges two plants gives
1.68-2.13. Pixel counts outside the cell cannot tell those apart; the bbox
ratio can.
"""


def object_exceeds_region_warning(
    labeled: np.ndarray, label: int, bbox: tuple[int, int, int, int]
) -> Advisory | None:
    """Warn when the object a cell reports is much larger than the cell.

    create_labels(roi_type="partial") labels any object that OVERLAPS a cell
    whole, so the traits describe the whole object, cell boundary or not. On a
    real X-Rite tray photo a misaligned auto_grid reported objects 785 px wide
    inside 369 px cells -- two plants per row -- with no signal at all.
    """
    x, y, w, h = bbox
    obj = labeled == label
    ys, xs = np.nonzero(obj)
    if xs.size == 0 or not (w and h):
        return None
    ox, oy = int(xs.min()), int(ys.min())
    ow, oh = int(xs.max() - ox + 1), int(ys.max() - oy + 1)
    ratio = max(ow / w, oh / h)
    if ratio < EXCEEDS_CELL_RATIO:
        return None
    total = int(obj.sum())
    outside = total - int(obj[y : y + h, x : x + w].sum())
    return Advisory(
        code="object_exceeds_region",
        message=(
            f"The object this region reports is {ratio:.1f}x the size of its cell "
            f"(object {ow}x{oh} px at ({ox}, {oy}) vs cell {w}x{h}; {outside} of "
            f"{total} px lie outside). Its traits describe the WHOLE object, which "
            "is almost certainly a neighbouring plant merged with this one. Look "
            "at the overlay, then give rect_grid geometry that puts one plant per "
            "cell."
        ),
    )


def grid_misalignment_warning(mode: str, measurements: list) -> Advisory | None:
    """Set-level signal: an inferred grid that does not sit on the tray.

    auto_grid infers cells from the mask's centroids. When it lands between the
    pots, the result looks healthy -- most cells measured -- while some cells
    are empty and others report objects spilling out of them. Both together,
    under auto_grid only: rect_grid geometry is the user's own and is not
    second-guessed.
    """
    if mode != "auto_grid":
        return None
    empty = sum(1 for m in measurements if not m["measured"])
    spill = sum(
        1
        for m in measurements
        if any(
            w["code"] in ("object_exceeds_region", "object_claimed_by_neighbour")
            for w in m["warnings"]
        )
    )
    if not (empty and spill):
        return None
    return Advisory(
        code="grid_misaligned",
        message=(
            f"auto_grid inferred a grid in which {empty} cell(s) hold nothing and "
            f"{spill} cell(s) report objects extending outside them. That is the "
            "signature of a grid that does not line up with the tray: neighbouring "
            "plants are being merged into one cell while another cell is empty. "
            "Check the numbered overlay, then re-run with mode='rect_grid' and "
            "explicit coord/width/height/spacing so each cell holds one plant."
        ),
    )


@dataclass
class RegionSlot:
    """One cell of the grid after labelling: measurable, or refused with a reason.

    The partition is the same for every modality -- which cell holds which
    object, which cells are empty, which had their material claimed by a
    neighbour, which hold too little to measure. Only the statistic computed
    on a measurable slot differs (traits, temperature, index values).
    """

    index: int
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    label: int
    coverage: float
    measured: bool
    reason: str | None
    warnings: list[Advisory]


def _refused_row(slot: RegionSlot) -> dict[str, Any]:
    return {
        "index": slot.index,
        "row": slot.row,
        "col": slot.col,
        "bbox": list(slot.bbox),
        "measured": False,
        "reason": slot.reason,
        "region_coverage": slot.coverage,
        "warnings": [{"code": w.code, "message": w.message} for w in slot.warnings],
    }


def partition_regions(
    mask: np.ndarray, regions: RegionSet
) -> tuple[np.ndarray, int, list[RegionSlot]]:
    """Label the mask by region and decide, per cell, whether it can be measured.

    Must be called under the `isolated_pcv_outputs` lock: PlantCV's
    create_labels is native code sharing state with the analyses that follow.
    """
    labeled, n = pcv.create_labels(mask=mask, rois=regions.rois, roi_type="partial")
    # Labels actually carrying pixels. A region whose label is absent here is
    # empty -- this is the only reliable discriminator, since the analysis
    # emits a full zero-valued group for it either way.
    present = {int(v) for v in np.unique(labeled)} - {0}

    slots: list[RegionSlot] = []
    for i, bbox in enumerate(regions.bboxes):
        label = i + 1
        x, y, w, h = bbox
        row, col = divmod(i, regions.ncols) if regions.ncols else (0, i)

        cell = labeled[y : y + h, x : x + w] if w and h else labeled[:0, :0]
        coverage = float((cell == label).sum()) / float(cell.size) if cell.size else 0.0

        if label not in present:
            # "Empty" per the labels is not "empty" per the mask: with
            # roi_type="partial" an object straddling two cells is handed
            # WHOLE to one of them, and the other reads as nothing. Say so,
            # naming the region that took it, instead of calling it empty.
            claimed = {int(v) for v in np.unique(cell)} - {0}
            in_cell = int((mask[y : y + h, x : x + w] > 0).sum()) if w and h else 0
            if claimed and in_cell:
                takers = ", ".join(f"region {c - 1}" for c in sorted(claimed))
                reason = (
                    f"Not empty: {in_cell} px of plant material lie in this cell "
                    f"but PlantCV assigned the whole object to {takers} because "
                    "it overlaps both. The neighbour's numbers include this "
                    "material. Look at the overlay; if these are two plants, "
                    "give rect_grid geometry that puts one plant per cell."
                )
                cell_warnings = [
                    Advisory(code="object_claimed_by_neighbour", message=reason)
                ]
            else:
                reason = (
                    "No plant material in this region. PlantCV reports a "
                    "complete result of zeros for an empty region, which is "
                    "indistinguishable from a real zero-area plant, so nothing "
                    "is returned for it."
                )
                cell_warnings = []
            # Coverage is the cell's own material, so a claimed cell's row
            # agrees with its reason ("N px lie in this cell").
            claimed_cov = in_cell / float(cell.size) if cell.size else 0.0
            slots.append(
                RegionSlot(
                    i, row, col, bbox, label, claimed_cov, False, reason, cell_warnings
                )
            )
            continue

        # A cell whose own object is a sliver of the material inside it is
        # reporting a fragment of a neighbour's object: an inverted tray under
        # a 1x2 rect_grid gave one cell the whole 400x200 background (caught
        # as exceeding) and the other a 544-px OUTLINE of it, 195x195 —
        # inside the exceeds ratio, above the floor, measured as a plant.
        # Calibrated on real trays: a clean tray owns >= 0.999 of every cell;
        # the misaligned X-Rite tray's intruded-upon cells own 0.35-0.39 and
        # their own object IS their plant (kept); the fragment owned 0.049.
        in_cell = int((mask[y : y + h, x : x + w] > 0).sum()) if w and h else 0
        owned = int((cell == label).sum())
        if in_cell and owned < OWNED_MATERIAL_FRACTION * in_cell:
            takers = ", ".join(
                f"region {c - 1}"
                for c in sorted({int(v) for v in np.unique(cell)} - {0, label})
            )
            slots.append(
                RegionSlot(
                    i,
                    row,
                    col,
                    bbox,
                    label,
                    coverage,
                    False,
                    "object_claimed_by_neighbour — traits withheld: this cell's "
                    f"own object is {owned} of the {in_cell} px of plant material "
                    f"in it; the rest belongs to the object {takers or 'a neighbour'} "
                    "reports, and what is left here is a fragment PlantCV cut off "
                    "at the cell edge, not a plant. Look at the numbered overlay; "
                    "if the mask is the background, re-run segment() with the "
                    "opposite object_type.",
                    [],
                )
            )
            continue

        # Each cell is held to the same degeneracy floor measure() applies
        # to a whole frame: a 2x2 speck of threshold noise otherwise comes
        # back as a full result indistinguishable from a real seedling.
        cell_diag = analyze_mask((cell == label).astype(np.uint8) * 255)
        try:
            assert_not_degenerate(cell_diag)
        except DegenerateMaskError as exc:
            slots.append(
                RegionSlot(
                    i,
                    row,
                    col,
                    bbox,
                    label,
                    coverage,
                    False,
                    "Too little plant material in this region to measure (of "
                    f"the cell, not the frame): {exc}",
                    [],
                )
            )
            continue

        # No coverage check here: cell_diag.mask_fraction is the plant's share
        # of the CELL, and a plant filling a tight cell is the happy path (a
        # disc in a tight square is ~79%). The whole-frame inversion check
        # already ran at segment()/measure(). What a cell CAN hide is several
        # plants: with roi_type="partial" every object in the cell gets the
        # cell's label, so four seedlings in one cell fit its bbox and look
        # like one plant to the exceeds check.
        region_warnings: list[Advisory] = []
        # Judged on the WHOLE labelled object, not the cell crop: a single
        # plant whose leaf leaves and re-enters the cell is two pieces inside
        # the crop (a 20,533-px arabidopsis read as two objects that way).
        whole_diag = analyze_mask((labeled == label).astype(np.uint8) * 255)
        multi = multi_specimen_warning(whole_diag, scope="cell")
        if multi:
            region_warnings.append(multi)
        spill = object_exceeds_region_warning(labeled, label, bbox)
        if spill:
            region_warnings.append(spill)
        slots.append(
            RegionSlot(i, row, col, bbox, label, coverage, True, None, region_warnings)
        )
    return labeled, n, slots


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

    # The SAME lock as measure_traits: both paths write and read the one global.
    with isolated_pcv_outputs():
        labeled, n, slots = partition_regions(mask, regions)

        if "size" in analyses:
            pcv.analyze.size(img=img, labeled_mask=labeled, n_labels=n)
        if "color" in analyses:
            pcv.analyze.color(
                rgb_img=img, labeled_mask=labeled, n_labels=n, colorspaces="hsv"
            )

        out: list[RegionMeasurement] = []
        for slot in slots:
            if not slot.measured:
                out.append(RegionMeasurement(traits=None, **_refused_row(slot)))
                continue

            traits: dict[str, TraitValue] = {
                name: TraitValue(value=obs.get("value"), unit=obs.get("label"))
                for name, obs in _read_group(slot.label).items()
            }
            # Checked on the pixel values, before any unit conversion.
            lp_warning = implausible_longest_path_warning(traits)
            if not include_histograms:
                traits = {k: v for k, v in traits.items() if k not in HISTOGRAM_TRAITS}
            if px_per_mm is not None:
                traits = convert_units(traits, float(px_per_mm))

            region_warnings = list(slot.warnings)
            if lp_warning:
                region_warnings.append(lp_warning)

            out.append(
                RegionMeasurement(
                    index=slot.index,
                    row=slot.row,
                    col=slot.col,
                    bbox=list(slot.bbox),
                    measured=True,
                    reason=None,
                    region_coverage=slot.coverage,
                    traits=traits,
                    warnings=[
                        {"code": w.code, "message": w.message} for w in region_warnings
                    ],
                )
            )
        return out


def _single_group(name: str) -> dict[str, dict]:
    """The observation group for a single-label analysis run under `name`.

    PlantCV writes `name` for n_labels=1 in some versions and `name_1` in
    others; either is accepted, anything else raises rather than resolving to
    a neighbour's group.
    """
    for key in (name, f"{name}_1"):
        if key in pcv.outputs.observations:
            return pcv.outputs.observations[key]
    raise KeyError(
        f"Expected observation group {name!r} not found. Available: "
        f"{list(pcv.outputs.observations.keys())}."
    )


def _measured_row(slot: RegionSlot, labeled: np.ndarray) -> dict[str, Any]:
    return {
        "index": slot.index,
        "row": slot.row,
        "col": slot.col,
        "bbox": list(slot.bbox),
        "measured": True,
        "reason": None,
        "region_coverage": slot.coverage,
        "pixel_count": int((labeled == slot.label).sum()),
        "warnings": [{"code": w.code, "message": w.message} for w in slot.warnings],
    }


def measure_regions_thermal(
    celsius: np.ndarray, mask: np.ndarray, regions: RegionSet
) -> list[dict[str, Any]]:
    """Per-region temperature statistics, through PlantCV's analyze.thermal.

    Same partition, same refusals as the RGB path; the statistic is the one
    measure_thermal() reports for the whole mask.
    """
    if celsius.shape != mask.shape:
        raise ValueError(f"mask {mask.shape} does not match the frame {celsius.shape}")
    with isolated_pcv_outputs():
        labeled, _n, slots = partition_regions(mask, regions)
        out: list[dict[str, Any]] = []
        for slot in slots:
            if not slot.measured:
                out.append(_refused_row(slot))
                continue
            # One label at a time: unlike analyze.size, analyze.thermal raises
            # on an empty label instead of reporting zeros, so a multi-label
            # call dies on the first empty cell.
            one = np.where(labeled == slot.label, 1, 0).astype(np.uint8)
            name = f"region{slot.index}"
            pcv.analyze.thermal(
                thermal_img=celsius, labeled_mask=one, n_labels=1, bins=100, label=name
            )
            obs = _single_group(name)
            out.append(
                {
                    **_measured_row(slot, labeled),
                    "temperature": {
                        "max": float(obs["max_temp"]["value"]),
                        "min": float(obs["min_temp"]["value"]),
                        "mean": float(obs["mean_temp"]["value"]),
                        "median": float(obs["median_temp"]["value"]),
                        "unit": "celsius",
                    },
                }
            )
        return out


def measure_regions_spectral(
    cube: Any,
    mask: np.ndarray,
    regions: RegionSet,
    indices: tuple[str, ...] = ("ndvi",),
) -> list[dict[str, Any]]:
    """Per-region index statistics on a PREPARED (float, calibrated) cube,
    through PlantCV's analyze.spectral_index -- the same numbers
    measure_spectral() reports for the whole mask."""
    if cube.array_data.shape[:2] != mask.shape:
        raise ValueError(
            f"mask {mask.shape} does not match the cube's frame "
            f"{cube.array_data.shape[:2]}"
        )
    computed = {name: compute_index(cube, name) for name in indices}
    with isolated_pcv_outputs():
        labeled, _n, slots = partition_regions(mask, regions)
        out: list[dict[str, Any]] = []
        for slot in slots:
            if not slot.measured:
                out.append(_refused_row(slot))
                continue
            sel = labeled == slot.label
            one = np.where(sel, 1, 0).astype(np.uint8)
            group = f"region{slot.index}"
            stats: dict[str, dict[str, float | int | None]] = {}
            for name, idx in computed.items():
                pcv.analyze.spectral_index(
                    index_img=idx,
                    labeled_mask=one,
                    n_labels=1,
                    bins=100,
                    min_bin="auto",
                    max_bin="auto",
                    label=group,
                )
                obs = _single_group(group)
                raw = idx.array_data.astype(np.float64)[sel]
                values = raw[np.isfinite(raw)]
                stats[name] = {
                    "mean": float(obs[f"mean_index_{name}"]["value"]),
                    "median": float(obs[f"med_index_{name}"]["value"]),
                    "std": float(obs[f"std_index_{name}"]["value"]),
                    "min": float(values.min()) if values.size else None,
                    "max": float(values.max()) if values.size else None,
                    "finite_pixel_count": int(values.size),
                }
            out.append({**_measured_row(slot, labeled), "indices": stats})
        return out
