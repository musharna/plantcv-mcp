"""Unattended measurement across many images.

**The design tension, stated rather than hidden.** This server's central claim is
that you cannot obtain a number without first being handed the picture it came
from. A batch of two hundred images cannot honour that literally: nobody is going
to look at two hundred overlays, and returning two hundred base64 images would be
useless anyway.

So the overlay is replaced by the only honest substitute — automated validation
plus explicit refusal. Every image runs the SAME guards as the interactive path
(`segmentation_warnings`). Any image tripping a BLOCKING guard gets **no traits at
all**, only the reason and an instruction to inspect it with `segment()`. Advisory
warnings that do not invalidate the measurement, like `multi_specimen`, are
attached to the traits rather than suppressing them.

The result is that a batch never hands back a number the system could not validate.
That is weaker than a human looking at a mask, and it is stated here so nobody
mistakes it for the same thing.
"""

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from plantcv import plantcv as pcv

from . import plantcv_version
from .color import correct_color
from .diagnostics import BLOCKING_CODES, Advisory, analyze_mask, segmentation_warnings
from .imaging import load_image
from .measurement import ANALYSES, UnknownAnalysisError, measure_traits
from .regions import (
    MAX_REGIONS,
    REGION_MODES,
    build_regions,
    grid_misalignment_warning,
    measure_regions,
)
from .segmentation import (
    CHANNELS,
    METHODS,
    OBJECT_TYPES,
    UnknownChannelError,
    UnknownMethodError,
    UnknownObjectTypeError,
    threshold_mask,
)

# A cap, so a model passing a directory glob of ten thousand files fails fast and
# loudly instead of pinning a CPU for an hour.
MAX_BATCH = 200

# Wall-clock budget for one call. Measured on real 3000-px photographs: 9-11 s
# per image, serial, so the 200-image cap alone allowed a ~30-minute call with
# no output at all -- longer than most MCP clients wait. Images not started
# before the budget runs out are returned as not run, so the caller gets the
# partial result and the list to resubmit.
DEFAULT_MAX_SECONDS = 300.0


class BatchTooLargeError(Exception):
    """Raised when more images are submitted than MAX_BATCH."""


@dataclass
class BatchEntry:
    image_path: str
    measured: bool
    mask_fraction: float | None = None
    component_count: int | None = None
    warnings: list[Advisory] = field(default_factory=list)
    traits: dict | None = None
    refused_because: str | None = None
    seconds: float | None = None
    not_run: bool = False
    regions: list[dict] | None = None  # per-plant rows when a grid was given


def _validate_recipe(
    channel: str,
    method: str,
    object_type: str,
    analyses: tuple[str, ...],
    max_seconds: float | None,
    grid: dict[str, Any] | None,
) -> None:
    """A recipe error is ONE error, raised before any image is loaded.

    Found on a real batch: channel='zz' ran every image and returned N
    identical UnknownChannelError rows with measured=0.
    """
    if channel not in CHANNELS:
        raise UnknownChannelError(
            f"Unknown channel {channel!r}. Valid channels: {sorted(CHANNELS)}."
        )
    if method not in METHODS:
        raise UnknownMethodError(f"Unknown method {method!r}. Valid: {list(METHODS)}.")
    if object_type not in OBJECT_TYPES:
        raise UnknownObjectTypeError(
            f"Unknown object_type {object_type!r}. Valid: {list(OBJECT_TYPES)}."
        )
    unknown = [a for a in analyses if a not in ANALYSES]
    if unknown or not analyses:
        raise UnknownAnalysisError(
            f"Unknown analyses {unknown}. Valid: {list(ANALYSES)}."
            if unknown
            else f"No analyses requested. Choose at least one of {list(ANALYSES)}."
        )
    if max_seconds is not None and max_seconds < 0:
        raise ValueError(f"max_seconds must be >= 0 or null, got {max_seconds}.")
    if grid is not None:
        if grid["nrows"] is None or grid["ncols"] is None:
            missing = "ncols" if grid["ncols"] is None else "nrows"
            raise ValueError(
                f"A grid needs both nrows and ncols; {missing} was not given. "
                "Nothing is guessed for an unattended batch: a 4x1 partition of "
                "a 4x6 tray would present row strips as plants."
            )
        if grid["nrows"] < 1 or grid["ncols"] < 1:
            raise ValueError(
                f"nrows and ncols must be >= 1, got {grid['nrows']}x{grid['ncols']}."
            )
        if grid["nrows"] * grid["ncols"] > MAX_REGIONS:
            raise ValueError(
                f"{grid['nrows']}x{grid['ncols']} = {grid['nrows'] * grid['ncols']} "
                f"regions exceeds the {MAX_REGIONS} cap per image."
            )
        if grid["mode"] not in REGION_MODES:
            raise ValueError(
                f"Unknown mode {grid['mode']!r}. Valid: {list(REGION_MODES)}."
            )
        if grid["radius"] is not None and grid["radius"] <= 0:
            raise ValueError(f"radius must be positive, got {grid['radius']}.")
        if grid["mode"] == "rect_grid" and None in (
            grid["coord"],
            grid["height"],
            grid["width"],
            grid["spacing"],
        ):
            raise ValueError(
                "mode='rect_grid' needs coord, height, width and spacing; nothing "
                "is guessed for an unattended batch."
            )


def measure_batch(
    image_paths: list[str],
    channel: str,
    method: str,
    object_type: str = "dark",
    fill_size: int = 200,
    ksize: int = 11,
    offset: int = 2,
    analyses: tuple[str, ...] = ("size",),
    px_per_mm: float | None = None,
    include_histograms: bool = False,
    color_correct: bool = False,
    max_seconds: float | None = DEFAULT_MAX_SECONDS,
    nrows: int | None = None,
    ncols: int | None = None,
    mode: str = "auto_grid",
    coord: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    spacing: tuple[int, int] | None = None,
    radius: int | None = None,
) -> dict:
    """Segment and measure many images with one fixed recipe.

    With nrows/ncols the mask of each image is measured per region exactly as
    measure_regions() does (same partition, same refusals), and the row
    carries `regions` instead of `traits`: a batch of trays otherwise returns
    the group's area with only an advisory, which real batches are mostly
    made of. Blocking guards still apply to the whole mask first.

    max_seconds bounds the call: images not started before it runs out are
    returned as not run, listed for resubmission, so the caller gets a partial
    result instead of a hung call.

    The recipe is EVERYTHING segment() takes — channel, method, object_type,
    fill_size, ksize, offset, color_correct — so a recipe settled interactively
    reproduces here exactly. A batch that silently ran default ksize/offset for a
    'mean' threshold, or skipped the colour correction, measured a different mask
    than the one the user had looked at: 1900 px vs 4536 px on the same file.

    Returns a per-image entry and a summary. Images whose guards fire are reported
    with `measured: false` and a reason, never with traits. An image that cannot
    be colour-corrected when asked is refused the same way — never measured raw.
    """
    if not image_paths:
        raise ValueError("image_paths is empty; nothing to measure.")
    if len(image_paths) > MAX_BATCH:
        raise BatchTooLargeError(
            f"{len(image_paths)} images submitted, limit is {MAX_BATCH}. Split the "
            "batch, or narrow the selection."
        )
    grid: dict[str, Any] | None = None
    if nrows is not None or ncols is not None:
        grid = {
            "mode": mode,
            "nrows": nrows,
            "ncols": ncols,
            "coord": coord,
            "height": height,
            "width": width,
            "spacing": spacing,
            "radius": radius,
        }
    _validate_recipe(channel, method, object_type, analyses, max_seconds, grid)

    # The same file twice is measured once; the summary says what was dropped.
    unique: list[str] = []
    duplicates: list[str] = []
    for path in image_paths:
        (duplicates if path in unique else unique).append(path)

    started = time.perf_counter()
    entries: list[BatchEntry] = []
    for path in unique:
        elapsed = time.perf_counter() - started
        if max_seconds is not None and entries and elapsed > max_seconds:
            entries.append(
                BatchEntry(
                    image_path=path,
                    measured=False,
                    not_run=True,
                    refused_because=(
                        f"not run: the {max_seconds:g} s time budget was used up "
                        f"after {len(entries)} image(s) ({elapsed:.0f} s). "
                        "Resubmit the not_run_paths, or raise max_seconds."
                    ),
                )
            )
            continue
        t0 = time.perf_counter()
        try:
            img = load_image(path)
            if color_correct:
                # Raises ColorCardNotFoundError without a card; caught below and
                # reported per-image, so the run continues and the image is
                # refused rather than measured uncorrected.
                img = correct_color(img)
            pre_fill = threshold_mask(
                img,
                channel,
                method,
                object_type=object_type,
                ksize=ksize,
                offset=offset,
            )
            mask = pcv.fill(bin_img=pre_fill, size=fill_size)
            diag = analyze_mask(mask)
            warnings = segmentation_warnings(
                mask, diag, analyze_mask(pre_fill), fill_size
            )
        except Exception as exc:  # noqa: BLE001 — see below
            # Deliberately blind: one unreadable or unsegmentable file must not
            # abort a 200-image run. The failure is reported per-image with its
            # exception type, so nothing is swallowed silently.
            entries.append(
                BatchEntry(
                    image_path=path,
                    measured=False,
                    refused_because=f"{type(exc).__name__}: {exc}",
                    seconds=time.perf_counter() - t0,
                )
            )
            continue

        blocking = [w for w in warnings if w.code in BLOCKING_CODES]
        if grid is not None:
            # "Noisy" is judged on the whole mask: a 96-well plate with ten
            # germinated wells and 86 late ones is 90 minor components, which
            # reads as texture until the grid says each is a well. With a
            # grid the per-cell degeneracy floor guards each cell, so the
            # advisory travels with the rows instead of withholding them.
            blocking = [w for w in blocking if w.code != "noisy_segmentation"]
        if blocking:
            entries.append(
                BatchEntry(
                    image_path=path,
                    measured=False,
                    mask_fraction=diag.mask_fraction,
                    component_count=diag.component_count,
                    warnings=warnings,
                    refused_because=(
                        f"{', '.join(w.code for w in blocking)} — traits withheld "
                        "because the mask probably does not describe the plant. "
                        "Inspect this image with segment() and look at the overlay."
                    ),
                    seconds=time.perf_counter() - t0,
                )
            )
            continue

        try:
            if grid is None:
                traits = measure_traits(
                    img,
                    mask,
                    analyses=analyses,
                    px_per_mm=px_per_mm,
                    include_histograms=include_histograms,
                )
                regions = None
            else:
                region_set = build_regions(img, mask, **grid)
                rows = measure_regions(
                    img,
                    mask,
                    region_set,
                    analyses=analyses,
                    px_per_mm=px_per_mm,
                    include_histograms=include_histograms,
                )
                traits = None
                regions = []
                for r in rows:
                    r = dict(r)
                    # Interactive measure_regions keeps the number beside the
                    # numbered overlay; here nobody looks, so a row the guard
                    # has already called a merge is withheld, not returned.
                    spill = [
                        w for w in r["warnings"] if w["code"] == "object_exceeds_region"
                    ]
                    if r["measured"] and spill:
                        r["measured"] = False
                        r["traits"] = None
                        r["reason"] = (
                            "object_exceeds_region — traits withheld: "
                            + spill[0]["message"]
                        )
                    regions.append(r)
                # multi_specimen says "this number describes a group"; with a
                # grid there is no group number, so the advisory would mislead.
                warnings = [w for w in warnings if w.code != "multi_specimen"]
                warnings.extend(region_set.warnings)
                misaligned = grid_misalignment_warning(region_set.mode, rows)
                if misaligned:
                    warnings.append(misaligned)
        except Exception as exc:  # noqa: BLE001 — same rationale as above:
            # a single image failing to measure must not lose the other 199.
            entries.append(
                BatchEntry(
                    image_path=path,
                    measured=False,
                    mask_fraction=diag.mask_fraction,
                    component_count=diag.component_count,
                    warnings=warnings,
                    refused_because=f"{type(exc).__name__}: {exc}",
                    seconds=time.perf_counter() - t0,
                )
            )
            continue

        entries.append(
            BatchEntry(
                image_path=path,
                measured=True,
                mask_fraction=diag.mask_fraction,
                component_count=diag.component_count,
                warnings=warnings,
                traits=traits,
                regions=regions,
                seconds=time.perf_counter() - t0,
            )
        )

    measured = [e for e in entries if e.measured]
    not_run = [e for e in entries if e.not_run]
    refused = [e for e in entries if not e.measured and not e.not_run]
    advisory_counts = Counter(w.code for e in measured for w in e.warnings)
    return {
        "recipe": {
            "channel": channel,
            "method": method,
            "object_type": object_type,
            "fill_size": fill_size,
            "ksize": ksize,
            "offset": offset,
            "color_correct": color_correct,
            "analyses": list(analyses),
            "px_per_mm": px_per_mm,
            "max_seconds": max_seconds,
            "regions": (
                {k: v for k, v in grid.items() if v is not None} if grid else None
            ),
        },
        "elapsed_s": time.perf_counter() - started,
        # Which PlantCV produced these numbers, travelling WITH them — the same
        # provenance measure() carries, for the same reason.
        "engine": {"name": "PlantCV", "version": plantcv_version()},
        "summary": {
            "submitted": len(image_paths),
            "unique": len(unique),
            "duplicates_dropped": duplicates,
            "measured": len(measured),
            "with_advisories": sum(1 for e in measured if e.warnings),
            "advisory_counts": dict(advisory_counts),
            "needs_review": len(refused),
            "review_paths": [e.image_path for e in refused],
            "not_run": len(not_run),
            "not_run_paths": [e.image_path for e in not_run],
        },
        "results": [
            {
                "image_path": e.image_path,
                "measured": e.measured,
                "mask_fraction": e.mask_fraction,
                "component_count": e.component_count,
                "warnings": [
                    {"code": w.code, "message": w.message} for w in e.warnings
                ],
                "traits": e.traits,
                "refused_because": e.refused_because,
                "seconds": e.seconds,
                **(
                    {
                        "regions": e.regions,
                        "regions_measured": sum(1 for r in e.regions if r["measured"]),
                    }
                    if e.regions is not None
                    else {}
                ),
            }
            for e in entries
        ],
    }
