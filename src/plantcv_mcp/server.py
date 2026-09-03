"""MCP server. Fourteen tools over a typed session store (rgb / hsi / thermal).

segment() mints a session and returns the overlay but NO traits; measure()
requires that session. The split is deliberate: it forces the visual evidence
into the model's context before a number can be obtained.

Tool functions are synchronous, and mcp 2.x runs synchronous tools on WORKER
THREADS (mcp/server/mcpserver/utilities/func_metadata.py: anyio.to_thread
.run_sync), so two tool calls can execute at the same time. An earlier version of
this docstring argued the opposite from mcp 1.28.1, which ran them inline; the
dependency floor moved to mcp>=2 and the argument silently stopped being true.
Nothing here may rely on serialisation. PlantCV's process-global `pcv.outputs`
is guarded by measurement.PCV_OUTPUTS_LOCK (via isolated_pcv_outputs), and the
session store carries its own lock.
"""

import functools
import hashlib
import json
import math
import os
import threading
from typing import Any, NotRequired

import numpy as np

# mcp 2.x renamed FastMCP to MCPServer and removed mcp.server.fastmcp. Same
# class, same decorator, same kwargs — a rename, not a rewrite. Image moved with
# it. ToolAnnotations stayed in mcp.types, though its fields are snake_case now.
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from plantcv import plantcv as pcv

# See measurement.py: typing.TypedDict breaks schema generation on 3.11.
from typing_extensions import TypedDict

from . import __version__, plantcv_version
from .batch import DEFAULT_MAX_SECONDS
from .color import (
    color_card_excluded_advisory,
    correct_color,
    detect_card_region,
    exclude_card,
)
from .diagnostics import (
    analyze_mask,
    implausible_longest_path_warning,
    mask_warnings,
    segmentation_warnings,
)
from .hyperspectral import load_cube
from .imaging import (
    downscale,
    encode_png,
    load_image,
    load_image_with_digest,
    open_regular_file,
    render_overlay,
    render_region_overlay,
    write_image,
)
from .lens import (
    FOCAL_UNCERTAINTY_ADVISORY,
    LensCalibration,
    calibrate_lens_from_frames,
    rms_fraction,
    undistort_image,
)
from .measurement import ANALYSES, TraitValue, validate_analyses
from .paths import check_readable, configured_roots, set_roots
from .refine import (
    REFINE_OPS,
    RefinementErasedMaskError,
    refinement_warnings,
    validate_ops,
)
from .regions import REGION_MODES, RegionSpecError
from .scale import calibrate_scale
from .segmentation import CHANNELS, METHODS, OBJECT_TYPES, threshold_mask
from .session import SessionStore
from .suggest import colorspace_sheet, polarity_report, threshold_sheet
from .thermal import grey_frame, load_thermal
from .workers import dispatch, set_isolation

_store = SessionStore()

# Shown to clients at connection time. The product is a discipline, not just a set
# of functions, and the discipline has to be stated somewhere the model will read
# it before it starts calling things.
INSTRUCTIONS = """\
PlantCV as a measurement instrument for plant images.

Work in this order: suggest_segmentation -> segment -> LOOK AT THE OVERLAY -> measure.

segment() deliberately returns no traits. It returns an overlay image with the
measured pixels tinted red and outlined in cyan, plus diagnostics. Look at that image before you trust
any number that follows: a segmentation that selected the background instead of
the plant still produces seventeen traits with correct units and believable
magnitudes. The overlay is the only thing that distinguishes the two.

Read the `warnings` array on every segment() response. `implausible_coverage`
means the mask is probably inverted — re-run with the opposite `object_type`.
`multi_specimen` means several plants were merged into one measurement.
`fill_erased_mask` means `fill_size` deleted the specimen. `empty_mask` means the
segmentation found nothing. `noisy_segmentation` means the mask is dozens of specks
around whatever plants there are — background texture, not a plant.

Do not guess `channel` or `object_type`. suggest_segmentation reports what both
polarities actually yield on the image in front of you. Its `noisy_segmentation`
warning means the recommended polarity is specks, not a plant: pick another
channel from the colourspace sheet. `ambiguous` is a polarity comparison, not a
quality verdict.

If the overlay is nearly right — a hole in the leaf, specks on the background, a
second object that is not the plant — use refine() instead of hunting for a
different threshold: [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}] is
the usual cleanup. refine() returns a NEW session and its overlay; look at that
overlay too, and measure the refined session. `refine_dropped_object` means an
op threw away a leaf-sized component (a leaf that opening/erode split off and
keep_largest discarded) — decide from the overlay whether it belonged to the
plant. Trait tables carry `lineage`, the ops that produced their mask.

Traits are in PIXELS unless you pass px_per_mm to measure(), and pixel sizes are
not comparable between images taken at different distances or zoom levels.
If the camera has visible lens distortion — straight edges bow, a fisheye rig —
run correct_lens_distortion() FIRST and work on the corrected image: measured on
a real fisheye photo, distortion inflated plant area 2.13x and the error is
anisotropic, so no px_per_mm can compensate for it.

Hyperspectral ENVI cubes and thermal frames have their own segmenters
(segment_hyperspectral, segment_thermal) and their own measurers
(measure_spectral, measure_thermal); measure_regions() works on all three kinds
(traits, temperatures or index statistics per plant). Sessions are typed, and a
session handed to the wrong measurer is refused naming the right one. segment_thermal()
without a band refuses WITH the frame's temperature range and percentiles; use
them. `threshold_outside_range` means the threshold or band lies past the data's
own range and the mask is meaningless. The same rule holds everywhere: look at
the returned overlay before trusting any number.\
"""


class MeasureResult(TypedDict):
    """Return type of measure(). Annotated so MCP can publish an output_schema."""

    session_id: str
    analyses: list[str]
    px_per_mm: float | None
    traits: dict[str, TraitValue]
    # The refine() ops that produced this session's mask, in order; [] when the
    # mask came straight from segment(). A trait table that cannot say how its
    # mask was made cannot be compared with one made differently.
    lineage: list[dict]
    # Which PlantCV produced these numbers, travelling WITH them. The version
    # was already reachable via list_methods, but a stored or forwarded trait
    # table could not say what measured it, and two tables could not be told
    # apart. Trait definitions shift between PlantCV releases, so a measurement
    # that cannot name its version is not reproducible.
    engine: dict[str, str]
    # Mask-level advisories (frame_clipping, multi_specimen, ...), re-derived
    # from the session mask at measure time. They were reported at segment()
    # time and then dropped, so the trait table — the artifact people keep —
    # could not say its own area was a lower bound.
    warnings: list[dict]


class WarningItem(TypedDict):
    """One advisory attached to a result."""

    code: str
    message: str


class ScaleResult(TypedDict):
    """Return type of calibrate_scale_from_marker()."""

    px_per_mm: float
    marker_length_px: int
    marker_length_mm: float
    marker_area_px: int
    crop_fraction: float
    warnings: list[WarningItem]


class BatchRecipe(TypedDict):
    """The one segmentation recipe a batch applied to every image."""

    channel: str
    method: str
    object_type: str
    fill_size: int
    # The 'mean'/'gaussian' kernel parameters and the colour-correction flag are
    # part of the recipe: without them the record cannot say what produced the
    # numbers, and a batch could not reproduce a settled segment() call.
    ksize: int
    offset: int
    color_correct: bool
    analyses: list[str]
    px_per_mm: float | None


class BatchSummary(TypedDict):
    """Counts, plus the paths that still need a human with an overlay."""

    submitted: int
    measured: int
    needs_review: int
    review_paths: list[str]


class BatchImageResult(TypedDict):
    """One image's outcome. traits is null whenever measured is false."""

    image_path: str
    measured: bool
    mask_fraction: float | None
    component_count: int | None
    warnings: list[WarningItem]
    traits: dict[str, TraitValue] | None
    refused_because: str | None


class BatchResult(TypedDict):
    """Return type of measure_images()."""

    recipe: BatchRecipe
    summary: BatchSummary
    results: list[BatchImageResult]
    engine: dict[str, str]


class SpectralMeasureResult(TypedDict):
    """Return type of measure_spectral()."""

    session_id: str
    kind: str
    indices: dict[str, dict[str, Any]]
    band_count: int
    wavelength_range: list[float]
    calibration: str
    pixel_count: int
    warnings: list[WarningItem]
    spectrum: NotRequired[dict[str, Any]]
    histograms: NotRequired[dict[str, Any]]
    engine: dict[str, str]


class ThermalMeasureResult(TypedDict):
    """Return type of measure_thermal()."""

    session_id: str
    kind: str
    temperature: dict[str, Any]
    pixel_count: int
    frame_range: list[float]
    histogram: NotRequired[dict[str, Any]]
    engine: dict[str, str]


class MethodsInfo(TypedDict):
    """Return type of list_methods()."""

    # This server's own version. The engine version alone cannot say which
    # plantcv-mcp answered — a stale cached server is indistinguishable from
    # current over the tool surface without it.
    server_version: str
    plantcv_version: str
    channels: list[str]
    methods: list[str]
    object_types: list[str]
    analyses: list[str]
    region_modes: list[str]
    refine_ops: dict[str, dict]
    # The read-root policy in force: realpaths, or null when unrestricted.
    read_roots: list[str] | None
    guidance: str


class ImageChangedSinceSegmentationError(Exception):
    """Raised when the file on disk is no longer the image that was segmented.

    Checked two ways: the frame shape, and a SHA-256 of the file's bytes. Shape
    alone let a same-dimension content swap through, silently measuring a stale
    mask against new pixels.
    """


def list_methods_impl() -> MethodsInfo:
    return {
        "server_version": __version__,
        "plantcv_version": plantcv_version(),
        "channels": sorted(CHANNELS),
        "methods": list(METHODS),
        "object_types": list(OBJECT_TYPES),
        "analyses": list(ANALYSES),
        "region_modes": list(REGION_MODES),
        "refine_ops": REFINE_OPS,
        "read_roots": configured_roots(),
        "guidance": (
            "Pick a channel AND the object_type that goes with it. object_type "
            "says which side of the threshold is the plant: 'dark' selects "
            "pixels below it, 'light' above it. Choosing the wrong one returns "
            "the BACKGROUND as the plant, with traits that still look "
            "plausible. Measured on a green-plant render: 'a' of LAB needs "
            "object_type='dark' (the usual starting point for green tissue on a "
            "light background), while 's' and 'b' need object_type='light' — "
            "with 'dark' they select 96% of the frame. 'l' and 'v' were "
            "ambiguous on that image. These are starting points, not rules: "
            "call suggest_segmentation(), which reports what BOTH polarities "
            "actually yield on your image."
        ),
    }


def _segment_impl(
    image_path: str,
    channel: str,
    method: str,
    object_type: str = "dark",
    fill_size: int = 200,
    ksize: int = 11,
    offset: int = 2,
    color_correct: bool = False,
    exclude_color_card: bool = False,
) -> dict:
    # The digest is of the SAME bytes the mask is about to be drawn on. Hashing
    # the path afterwards left a window in which a same-shape replacement was
    # recorded as this mask's identity.
    img, digest = load_image_with_digest(check_readable(image_path))
    card = None
    if color_correct:
        # Raises if no card is found. Silently measuring an uncorrected image after
        # being asked to correct it would be the same confident wrongness as an
        # inverted mask.
        img, card = correct_color(img)
    elif exclude_color_card:
        # The instrument is kept out of the mask without touching the colours;
        # asking for it with no card in the frame raises the same way.
        card = detect_card_region(img)

    # Threshold and fill run separately so a mask erased by fill_size can be
    # named as such instead of looking like a bad channel/method choice.
    pre_fill = threshold_mask(
        img, channel, method, object_type=object_type, ksize=ksize, offset=offset
    )
    card_note = None
    removed = 0
    if card is not None:
        # The card the correction just located is the instrument, not a
        # specimen. Left in, its chips merged into the largest "plant" on a
        # real photo, suppressing multi_specimen and dominating group traits.
        pre_fill, removed = exclude_card(pre_fill, card)
        card_note = color_card_excluded_advisory(removed, card)
    mask = pcv.fill(bin_img=pre_fill, size=fill_size)

    diag = analyze_mask(mask)
    # Shared with the batch path so the two cannot apply different guards.
    warnings = segmentation_warnings(mask, diag, analyze_mask(pre_fill), fill_size)
    if card_note is not None:
        warnings.append(card_note)

    session = _store.create(
        image_path,
        mask,
        channel,
        method,
        digest=digest,
        color_correct=color_correct,
        card_region=card,
        card_excluded_px=removed,
    )
    overlay, scale = downscale(render_overlay(img, mask))
    png = encode_png(overlay)
    return {
        "session_id": session.session_id,
        "channel": channel,
        "method": method,
        "object_type": object_type,
        "fill_size": fill_size,
        "color_correct": color_correct,
        "mask_fraction": diag.mask_fraction,
        "component_count": diag.component_count,
        "major_object_count": diag.major_object_count,
        "largest_area": diag.largest_area,
        "overlay_scale": scale,
        "overlay_png_bytes": len(png),
        "_png": png,
        "warnings": [{"code": w.code, "message": w.message} for w in warnings],
    }


def _load_session_image(session) -> np.ndarray:
    """Re-read the session's image, refusing it if the file changed underneath.

    Shared by every measuring path. Extracted rather than duplicated when
    per-region measurement was added: a second entry point with its own copy of
    these guards is a copy that can drift, and the failure mode of a drifted
    stale-image check is measuring a mask against pixels it was never drawn on —
    silently, with plausible numbers.
    """
    # Re-read (sessions do not hold RGB), hashing the bytes that are decoded
    # rather than the path afterwards — the same one-read rule as segment().
    img, digest = load_image_with_digest(session.image_path)
    current_shape = (int(img.shape[0]), int(img.shape[1]))
    if current_shape != session.shape:
        raise ImageChangedSinceSegmentationError(
            f"Image at {session.image_path!r} is now {current_shape} but was "
            f"{session.shape} when segment() produced this session's mask. "
            "The file changed on disk since segmentation, so the mask no "
            "longer corresponds to its current content. Re-run segment() on "
            "the current file before measuring it."
        )
    if digest != session.digest:
        raise ImageChangedSinceSegmentationError(
            f"Image at {session.image_path!r} still has shape {current_shape}, "
            "but its CONTENT changed since segment() produced this session's "
            "mask (SHA-256 mismatch). The mask describes pixels that are no "
            "longer there. Re-run segment() on the current file."
        )
    if session.color_correct:
        # Applied AFTER the integrity guards, so a missing colour card cannot mask
        # a stale-image error. The mask was drawn on corrected pixels, so the
        # traits have to be measured on corrected pixels too. The card region is
        # not needed here: it was already excluded from the stored mask.
        img, _ = correct_color(img)
    return img


def _measure_impl(
    session_id: str,
    analyses: list[str] | None = None,
    px_per_mm: float | None = None,
    include_histograms: bool = False,
) -> MeasureResult:
    requested = tuple(analyses) if analyses is not None else ("size",)
    session = _session_of(session_id, "rgb")
    img = _load_session_image(session)
    traits = dispatch(
        "measure", img, session.mask, requested, px_per_mm, include_histograms
    )
    warnings = mask_warnings(session.mask, analyze_mask(session.mask))
    lp_warning = implausible_longest_path_warning(traits)
    if lp_warning:
        warnings.append(lp_warning)
    if session.card_region is not None:
        # The mask cannot say a card was cut out of it; the session can.
        card_note = color_card_excluded_advisory(
            session.card_excluded_px, session.card_region
        )
        if card_note is not None:
            warnings.append(card_note)
    return {
        "session_id": session_id,
        "analyses": list(requested),
        "px_per_mm": px_per_mm,
        "lineage": [dict(op) for op in session.lineage],
        "traits": traits,
        "engine": {"name": "PlantCV", "version": plantcv_version()},
        "warnings": [{"code": w.code, "message": w.message} for w in warnings],
    }


def _refine_impl(session_id: str, ops: list[dict]) -> dict:
    session = _session_of(session_id, "rgb")
    validated = validate_ops(ops)  # all-or-nothing, before anything runs
    # refuses a degenerate result; reports every major object an op threw away
    mask, dropped = dispatch("refine", session.mask, validated)
    regrown = 0
    if session.card_region is not None:
        # The stored mask had the card cut out; a dilation, closing or
        # fill_holes grows the plant back into it, and measure() then samples
        # card pixels as plant. The exclusion is an invariant of the session,
        # not of the first mask.
        mask, regrown = exclude_card(mask, session.card_region)
        if regrown and not (mask > 0).any():
            raise RefinementErasedMaskError(
                "Every pixel the refinement produced lay inside the colour card; "
                "nothing measurable is left. The card is the instrument, not a "
                "specimen — drop the op that grew the mask into it."
            )
    before = analyze_mask(session.mask)
    after = analyze_mask(mask)
    warnings = refinement_warnings(mask, before, after, dropped)
    if regrown:
        note = color_card_excluded_advisory(
            regrown,
            session.card_region,
            note=" An op grew the mask into the card region; those pixels were "
            "removed again.",
        )
        if note is not None:
            warnings.append(note)

    # Re-read through the same integrity guards measure() uses: refining a
    # session whose file changed underneath would draw the overlay on pixels
    # the mask was never made from.
    img = _load_session_image(session)
    child = _store.create(
        session.image_path,
        mask,
        session.channel,
        session.method,
        digest=session.digest,
        color_correct=session.color_correct,
        card_region=session.card_region,
        card_excluded_px=session.card_excluded_px + regrown,
        lineage=[*session.lineage, *validated],
        parent_id=session.session_id,
    )
    overlay, scale = downscale(render_overlay(img, mask))
    png = encode_png(overlay)

    def _summary(d) -> dict:
        return {
            "mask_fraction": d.mask_fraction,
            "component_count": d.component_count,
            "largest_area": d.largest_area,
        }

    return {
        "session_id": child.session_id,
        "parent_session_id": session.session_id,
        "ops": validated,
        "lineage": [dict(op) for op in child.lineage],
        "before": _summary(before),
        "after": _summary(after),
        "overlay_scale": scale,
        "overlay_png_bytes": len(png),
        "warnings": [{"code": w.code, "message": w.message} for w in warnings],
        "engine": {"name": "PlantCV", "version": plantcv_version()},
        "_png": png,
    }


def _measure_morphology_impl(
    session_id: str,
    prune_size: int = 15,
    tangent_size: int = 25,
    px_per_mm: float | None = None,
) -> dict:
    session = _session_of(session_id, "rgb")
    img = _load_session_image(session)
    res = dispatch("morphology", img, session.mask, prune_size, tangent_size, px_per_mm)
    small, scale = downscale(res.overlay)
    png = encode_png(small)
    return {
        "session_id": session_id,
        "lineage": [dict(op) for op in session.lineage],
        "prune_size": res.prune_size,
        "tangent_size": res.tangent_size,
        "px_per_mm": px_per_mm,
        "plant": res.plant,
        "segments": res.segments,
        "units": res.units,
        "warnings": [{"code": w.code, "message": w.message} for w in res.warnings],
        "overlay_scale": scale,
        "engine": {"name": "PlantCV", "version": plantcv_version()},
        "_png": png,
    }


TOOL_FOR_KIND = {
    "rgb": "measure(), measure_regions(), measure_morphology() or refine()",
    "hsi": "measure_spectral() or measure_regions()",
    "thermal": "measure_thermal() or measure_regions()",
}


class WrongSessionKindError(Exception):
    """A session of one modality was handed to a tool for another."""


def _session_of(session_id: str, kind: str):
    """The session, if it is of `kind`; otherwise refuse naming the right tool."""
    session = _store.get(session_id)
    if session.kind != kind:
        raise WrongSessionKindError(
            f"Session {session_id!r} is a {session.kind} session (from "
            f"{'segment_hyperspectral' if session.kind == 'hsi' else 'segment_thermal' if session.kind == 'thermal' else 'segment'}()), "
            f"not {kind}. Use {TOOL_FOR_KIND[session.kind]} for it."
        )
    return session


def _load_session_cube(session):
    load = load_cube(session.image_path)
    if load.digest != session.digest:
        raise ImageChangedSinceSegmentationError(
            f"The ENVI cube at {session.image_path!r} (or its .hdr) changed since "
            "segment_hyperspectral() produced this session's mask (SHA-256 "
            "mismatch). Re-run segment_hyperspectral() on the current files."
        )
    return load


def _load_session_thermal(session):
    load = load_thermal(session.image_path)
    if load.digest != session.digest:
        raise ImageChangedSinceSegmentationError(
            f"The thermal file at {session.image_path!r} changed since "
            "segment_thermal() produced this session's mask (SHA-256 mismatch). "
            "Re-run segment_thermal() on the current file."
        )
    if load.celsius.shape != session.shape:
        raise ImageChangedSinceSegmentationError(
            f"The thermal frame at {session.image_path!r} is now "
            f"{load.celsius.shape} but was {session.shape} at segmentation."
        )
    return load


def _summary(diag) -> dict:
    return {
        "mask_fraction": diag.mask_fraction,
        "component_count": diag.component_count,
        "major_object_count": diag.major_object_count,
        "largest_area": diag.largest_area,
    }


def _segment_hsi_impl(
    envi_path: str,
    index: str = "ndvi",
    threshold: float = 0.2,
    object_type: str = "light",
    white_reference: str | None = None,
    dark_reference: str | None = None,
    fill_size: int = 200,
) -> dict:
    # Containment is enforced by read_image_bytes() at the read itself (the
    # derived ENVI sibling included); these loads happen HERE, in the parent,
    # so the worker never touches the filesystem.
    load = load_cube(envi_path)
    white_load = load_cube(white_reference) if white_reference is not None else None
    dark_load = load_cube(dark_reference) if dark_reference is not None else None
    seg = dispatch(
        "hsi_segment",
        load,
        index=index,
        threshold=threshold,
        object_type=object_type,
        white_reference=white_reference,
        dark_reference=dark_reference,
        white_load=white_load,
        dark_load=dark_load,
        fill_size=fill_size,
    )
    session = _store.create(
        load.raw_path,
        seg.mask,
        channel=f"index:{index}",
        method=f"threshold:{threshold}",
        digest=load.digest,
        kind="hsi",
        extra={
            "index": index,
            "threshold": float(threshold),
            "object_type": object_type,
            "calibration": seg.calibration,
            "calibration_args": seg.calibration_args,
        },
    )
    small, scale = downscale(seg.overlay)
    png = encode_png(small)
    return {
        "session_id": session.session_id,
        "kind": "hsi",
        "envi_path": load.raw_path,
        "index": index,
        "threshold": float(threshold),
        "object_type": object_type,
        "calibration": seg.calibration,
        "index_range": list(seg.index_range),
        "band_count": seg.band_count,
        "wavelength_range": list(seg.wavelength_range),
        **_summary(seg.diagnostics),
        "overlay_scale": scale,
        "overlay_png_bytes": len(png),
        "warnings": [{"code": w.code, "message": w.message} for w in seg.warnings],
        "engine": {"name": "PlantCV", "version": plantcv_version()},
        "_png": png,
    }


def _measure_spectral_impl(
    session_id: str,
    indices: list[str] | None = None,
    include_spectrum: bool = False,
    include_histograms: bool = False,
) -> SpectralMeasureResult:
    session = _session_of(session_id, "hsi")
    load = _load_session_cube(session)
    cal = session.extra.get("calibration_args") or {}
    # Refs are re-read in the parent (contained reads); measure_spectral checks
    # them against the digests pinned at segmentation before calibrating.
    wref, dref = cal.get("white_reference"), cal.get("dark_reference")
    res = dispatch(
        "hsi_measure",
        load,
        session.mask,
        indices=tuple(indices) if indices else (session.extra.get("index", "ndvi"),),
        calibration=cal,
        white_load=load_cube(wref) if wref is not None else None,
        dark_load=load_cube(dref) if dref is not None else None,
        include_spectrum=include_spectrum,
        include_histograms=include_histograms,
    )
    return {
        "session_id": session_id,
        "kind": "hsi",
        **res.as_dict(),
        "engine": {"name": "PlantCV", "version": plantcv_version()},
    }


def _segment_thermal_impl(
    path: str,
    min_c: float | None = None,
    max_c: float | None = None,
    fill_size: int = 200,
) -> dict:
    check_readable(path)
    load = load_thermal(path)
    seg = dispatch(
        "thermal_segment", load, path, min_c=min_c, max_c=max_c, fill_size=fill_size
    )
    session = _store.create(
        path,
        seg.mask,
        channel="celsius",
        method=f"band:{min_c}-{max_c}",
        digest=load.digest,
        kind="thermal",
        extra={"min_c": min_c, "max_c": max_c, "source": seg.source},
    )
    small, scale = downscale(seg.overlay)
    png = encode_png(small)
    return {
        "session_id": session.session_id,
        "kind": "thermal",
        "path": path,
        "source": seg.source,
        "min_c": min_c,
        "max_c": max_c,
        "frame_range": list(seg.frame_range),
        **_summary(seg.diagnostics),
        "overlay_scale": scale,
        "overlay_png_bytes": len(png),
        "warnings": [{"code": w.code, "message": w.message} for w in seg.warnings],
        "engine": {"name": "PlantCV", "version": plantcv_version()},
        "_png": png,
    }


def _measure_thermal_impl(
    session_id: str, include_histograms: bool = False
) -> ThermalMeasureResult:
    session = _session_of(session_id, "thermal")
    load = _load_session_thermal(session)
    res = dispatch(
        "thermal_measure",
        load,
        session.image_path,
        session.mask,
        include_histograms=include_histograms,
    )
    return {
        "session_id": session_id,
        "kind": "thermal",
        **res.as_dict(),
        "engine": {"name": "PlantCV", "version": plantcv_version()},
    }


def _as_xy(pair: list[int] | None, name: str) -> tuple[int, int] | None:
    """Coerce an [x, y] list from the wire into a checked 2-tuple.

    Length is validated rather than sliced or padded. A three-element `spacing`
    is a caller who thinks it means something else, and silently taking the
    first two would lay the grid down in the wrong place and measure the
    neighbouring plant — with numbers that look entirely reasonable.
    """
    if pair is None:
        return None
    if len(pair) != 2:
        raise RegionSpecError(
            f"{name} must be exactly [x, y]; got {len(pair)} values: {pair!r}."
        )
    return (int(pair[0]), int(pair[1]))


def _measure_regions_impl(
    session_id: str,
    mode: str = "auto_grid",
    nrows: int = 1,
    ncols: int = 1,
    coord: list[int] | None = None,
    height: int | None = None,
    width: int | None = None,
    spacing: list[int] | None = None,
    radius: int | None = None,
    analyses: list[str] | None = None,
    px_per_mm: float | None = None,
    include_histograms: bool = False,
    indices: list[str] | None = None,
) -> dict:
    session = _store.get(session_id)
    geometry = {
        "mode": mode,
        "nrows": nrows,
        "ncols": ncols,
        "coord": _as_xy(coord, "coord"),
        "height": height,
        "width": width,
        "spacing": _as_xy(spacing, "spacing"),
        "radius": radius,
    }
    # Arguments that belong to one modality are refused on another rather than
    # ignored: a px_per_mm silently dropped on a thermal tray would leave the
    # caller believing the numbers were converted.
    if session.kind != "rgb":
        for name, value in (
            ("analyses", analyses),
            ("px_per_mm", px_per_mm),
            ("include_histograms", include_histograms or None),
        ):
            if value is not None:
                raise WrongSessionKindError(
                    f"{name} applies to rgb sessions; session {session_id!r} is "
                    f"{session.kind}. Its regions report "
                    f"{'temperature' if session.kind == 'thermal' else 'index'} "
                    "statistics."
                )
    if session.kind != "hsi" and indices is not None:
        raise WrongSessionKindError(
            f"indices applies to hsi sessions; session {session_id!r} is "
            f"{session.kind}."
        )

    extra: dict = {}
    if session.kind == "rgb":
        requested = tuple(analyses) if analyses is not None else ("size",)
        validate_analyses(requested)  # before any geometry is built
        img = _load_session_image(session)
        regions = dispatch(
            "regions",
            img,
            session.mask,
            **geometry,
            analyses=requested,
            px_per_mm=px_per_mm,
            include_histograms=include_histograms,
        )
        extra = {
            "analyses": list(requested),
            "px_per_mm": px_per_mm,
            "lineage": [dict(op) for op in session.lineage],
        }
    elif session.kind == "thermal":
        load = _load_session_thermal(session)
        img = grey_frame(load.celsius)
        regions = dispatch("thermal_regions", load, session.mask, geometry=geometry)
    elif session.kind == "hsi":
        load = _load_session_cube(session)
        img = np.ascontiguousarray(load.cube.pseudo_rgb)
        cal = session.extra.get("calibration_args") or {}
        wref, dref = cal.get("white_reference"), cal.get("dark_reference")
        wanted = tuple(indices) if indices else (session.extra.get("index", "ndvi"),)
        regions = dispatch(
            "hsi_regions",
            load,
            session.mask,
            indices=wanted,
            calibration=cal,
            white_load=load_cube(wref) if wref is not None else None,
            dark_load=load_cube(dref) if dref is not None else None,
            geometry=geometry,
        )
        extra = {"indices": list(wanted), "calibration": regions["calibration"]}
    else:  # pragma: no cover - the store only mints the three kinds
        raise WrongSessionKindError(f"Unknown session kind {session.kind!r}.")
    measurements = regions["measurements"]

    overlay = render_region_overlay(
        img,
        session.mask,
        regions["bboxes"],
        [bool(m["measured"]) for m in measurements],
    )
    small, scale = downscale(overlay)
    png = encode_png(small)

    measured = [m for m in measurements if m["measured"]]
    return {
        "session_id": session_id,
        "kind": session.kind,
        "mode": regions["mode"],
        "nrows": regions["nrows"],
        "ncols": regions["ncols"],
        "regions_total": len(measurements),
        "regions_measured": len(measured),
        "regions_empty": len(measurements) - len(measured),
        **extra,
        "regions": measurements,
        "warnings": [{"code": c, "message": m} for c, m in regions["warnings"]],
        "overlay_scale": scale,
        "engine": {"name": "PlantCV", "version": plantcv_version()},
        "_png": png,
    }


# Calibrations are cached against the DIGEST of the checkerboard directory's
# contents, not its path: a directory whose frames changed is a different
# camera as far as the maths is concerned, and a stale calibration silently
# mis-corrects every image it touches.
_lens_cache: dict[tuple[str, int, int], LensCalibration] = {}
_lens_cache_lock = threading.Lock()


def _load_checkerboard_frames(directory: str) -> tuple[list[tuple[str, bytes]], str]:
    """Read every member file ONCE: the same bytes feed the cache digest and
    the calibration, so a directory mutated mid-call cannot cache one state's
    calibration under another state's digest. Each member passes
    check_readable — a symlink inside the directory pointing outside the
    configured roots is refused here exactly as segment() would refuse it —
    and an unreadable regular file becomes a named skipped frame instead of
    an exception from deep inside the digest. The digest serialization is
    length-prefixed: bare name||bytes concatenation had deterministic
    collisions (a file holding A||"1.png"||B hashed like two files A and B).
    """
    digest = hashlib.sha256()
    frames: list[tuple[str, bytes]] = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            continue
        if not os.path.isfile(path):
            # A FIFO, socket or device sitting at a member's name is not a
            # frame; it is named as a skipped one rather than vanishing from
            # the accounting (panel of 1.10.1: only a member SWAPPED for a
            # FIFO after this check was named, one present from the start
            # was silently omitted).
            frames.append((name, b""))
            encoded = name.encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            digest.update((0).to_bytes(8, "little"))
            continue
        real = check_readable(path)
        try:
            # The bytes come from a descriptor that open_regular_file proved
            # to be a regular file inside the roots — judged on the OPENED
            # file, not on the name: a member swapped for an outside symlink
            # (panel of 1.8.2), or its whole directory swapped for one, or the
            # member replaced by a FIFO that would block the open (panel of
            # 1.9.0) is a named skip or a refusal, never outside bytes.
            fd = open_regular_file(real, path)
            with os.fdopen(fd, "rb") as fh:
                data = fh.read()
        except OSError:
            data = b""
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
        frames.append((name, data))
    return frames, digest.hexdigest()


def _checkerboard_digest(directory: str) -> str:
    return _load_checkerboard_frames(directory)[1]


def _lens_calibration(
    checkerboard_dir: str, row_corners: int, col_corners: int
) -> LensCalibration:
    frames, digest = _load_checkerboard_frames(checkerboard_dir)
    key = (digest, int(row_corners), int(col_corners))
    with _lens_cache_lock:
        cached = _lens_cache.get(key)
    if cached is not None:
        return cached
    calib = calibrate_lens_from_frames(
        frames, row_corners=row_corners, col_corners=col_corners
    )
    with _lens_cache_lock:
        _lens_cache[key] = calib
    return calib


# Above this the correction is only as good as the fit and deserves saying so.
# A fraction of the frame diagonal, so one fit means one thing at every
# resolution: the real fisheye tutorial set calibrated at rms 13 px of a
# 3461-px diagonal (0.38%) while its one outlier frame was in, 3.7 px (0.11%)
# once the outlier gate dropped it — visibly helpful, demonstrably imperfect.
HIGH_REPROJECTION_FRACTION = 0.0015

# Below this the boards left most of the frame to extrapolation: measured on
# the synthetic camera, 24% coverage put the frame corners 31 px wrong (k3
# fixed) while 48% held them within 10 px; the real tutorial set covers 72%.
MIN_CALIBRATION_COVERAGE = 0.4


def _lens_advisories(calib: LensCalibration, info: dict, out: str) -> list[dict]:
    """The correction tool's warnings, assembled purely so they are testable."""
    warnings: list[dict] = []
    fit = rms_fraction(calib.rms, calib.shape)
    if info["roi_degenerate"] or info["residual_void_px"] > 0:
        crop_note = (
            f"No usable fully-valid crop exists for this calibration; "
            f"{info['residual_void_px']} fabricated void pixels remain in the "
            "output."
        )
    elif info["crop_fraction"] > 0:
        crop_note = (
            f"The corrected frame was cropped to the largest fully-valid "
            f"rectangle — {info['crop_fraction']:.0%} of the frame (remap "
            "voids and the contaminated border around them) was removed."
        )
    else:
        crop_note = "No void crop was needed."
    warnings.append(
        {
            "code": "lens_corrected",
            "message": (
                f"Lens distortion corrected from {len(calib.frames_used)} "
                f"checkerboard frame(s), rms reprojection error "
                f"{calib.rms:.2f} px. {crop_note} Segment and measure "
                f"{out!r} from here on — and take any px_per_mm from "
                "calibrate_scale_from_marker on the CORRECTED image. A scale "
                "calibrated on the original mixes two geometries: distortion "
                "is anisotropic, and no single number can compensate for it."
            ),
        }
    )
    if calib.frames_outliers:
        named = "; ".join(f"{n} {why}" for n, _, why in calib.frames_outliers)
        warnings.append(
            {
                "code": "outlier_frames_dropped",
                "message": (
                    f"{len(calib.frames_outliers)} checkerboard frame(s) were "
                    f"left out of the calibration, each judged against the "
                    f"views around it at the time: {named}. The "
                    f"{len(calib.frames_used)} views kept fit at a median of "
                    f"{calib.median_view_rms:.1f} px. A frame like that is "
                    "usually blurred, mis-detected, shot at a different zoom, "
                    "or of a board that was not flat; the camera was refitted "
                    "without it (measured: one such frame moved the focal "
                    "length 13% on the PlantCV tutorial set). Check those "
                    "frames, and re-shoot or remove them."
                ),
            }
        )
    if calib.frames_skipped:
        # A skip used to be named only inside thin_calibration, so a set with
        # five or more usable views reported nothing at all: a real run left
        # bad_checkerboard.png out of the calibration silently while every
        # outlier drop was announced. A skip is the more likely user error of
        # the two -- the wrong corner counts, a blurred frame, a file that is
        # not a photo of the board -- so it is named whatever the view count.
        warnings.append(
            {
                "code": "frames_skipped",
                "message": (
                    f"{len(calib.frames_skipped)} file(s) in the checkerboard "
                    "directory were left out before any camera was fitted: "
                    f"{', '.join(calib.frames_skipped)}. No board of the "
                    "given corner counts was detected there, the file did not "
                    "decode, or its size differs from the other frames. The "
                    f"calibration used the remaining {len(calib.frames_used)}. "
                    "If you expected those frames to count, check the corner "
                    "counts (INNER corners, one less per side than squares) "
                    "and that every photo is of the same board at the same "
                    "resolution."
                ),
            }
        )
    if calib.frames_duplicates:
        named = ", ".join(
            f"{n} (same bytes as {first})" for n, first in calib.frames_duplicates
        )
        warnings.append(
            {
                "code": "duplicate_frames_ignored",
                "message": (
                    f"{len(calib.frames_duplicates)} checkerboard file(s) are "
                    f"byte-for-byte copies of another and were counted once: "
                    f"{named}. Copies add no geometry; they only make every "
                    "uncertainty the fit reports look smaller than it is."
                ),
            }
        )
    # Judged on the PER-VIEW uncertainty, `sd/f · √N`, not on the reported
    # ratio: the ratio shrinks as ~1/√N whatever the geometry, so a weak set
    # used to buy silence by adding frames of the same weakness (panel of
    # 1.12.0 — see FOCAL_UNCERTAINTY_ADVISORY for the measurements). The
    # reported number stays the plain ratio, which is what the fit actually
    # says about itself; only the decision to speak changed.
    per_view_uncertainty = calib.focal_uncertainty * math.sqrt(
        max(1, len(calib.frames_used))
    )
    if per_view_uncertainty > FOCAL_UNCERTAINTY_ADVISORY:
        warnings.append(
            {
                "code": "focal_length_uncertain",
                "message": (
                    f"The {len(calib.frames_used)} views determine the focal "
                    f"length only to about ±{calib.focal_uncertainty:.1%} (the "
                    "fit's own uncertainty on it), so the correction is "
                    "correspondingly loose toward the frame corners (measured: "
                    "a set at 3.6% left the corners 54 px off). Judged per "
                    "view, because that ratio falls as frames are added "
                    "whether or not the geometry improves: these views are "
                    f"worth ±{per_view_uncertainty:.1%} each. The board was "
                    "tilted too little, or only about one axis; more strongly "
                    "and more variously tilted views tighten it — more frames "
                    "of the SAME view will not."
                ),
            }
        )
    if len(calib.frames_used) < 5:
        warnings.append(
            {
                "code": "thin_calibration",
                "message": (
                    f"Only {len(calib.frames_used)} checkerboard frame(s) "
                    "went into this calibration"
                    + (
                        f" ({len(calib.frames_skipped)} skipped: "
                        f"{', '.join(calib.frames_skipped)})"
                        if calib.frames_skipped
                        else ""
                    )
                    + ". Ten or more views, tilted and spread across the "
                    "frame, make a materially better model — the correction "
                    "here may be partial."
                ),
            }
        )
    if fit > HIGH_REPROJECTION_FRACTION:
        warnings.append(
            {
                "code": "high_reprojection_error",
                "message": (
                    f"The calibration fits its own checkerboards only to "
                    f"{calib.rms:.1f} px rms — {fit:.2%} of the frame diagonal "
                    f"(good calibrations sit under "
                    f"{HIGH_REPROJECTION_FRACTION:.2%}). That number is the "
                    "fit at the detected corners, not a bound on the residual "
                    "distortion elsewhere: the correction is only as good as "
                    "the fit. Sharper, more varied board views tighten it."
                ),
            }
        )
    if calib.coverage < MIN_CALIBRATION_COVERAGE:
        warnings.append(
            {
                "code": "low_calibration_coverage",
                "message": (
                    f"The detected checkerboards covered only "
                    f"{calib.coverage:.0%} of the frame (under "
                    f"{MIN_CALIBRATION_COVERAGE:.0%}). The model is fitted "
                    "where boards were and extrapolated everywhere else, so "
                    "the correction is least trustworthy toward the frame "
                    "edges and corners. Add views with the board near each "
                    "edge and corner."
                ),
            }
        )
    if info["roi_degenerate"] or info["residual_void_px"] > 0:
        warnings.append(
            {
                "code": "distortion_voids_remain",
                "message": (
                    "The corrected image still contains fabricated pixels at "
                    "its edges — black voids and the void-blended border "
                    "around them (residual_void_px counts both). A "
                    "value/darkness threshold will select them as objects; "
                    "segment on a channel where black is neutral (such as "
                    "'a'), and check the overlay."
                ),
            }
        )
    return warnings


def _correct_lens_impl(
    image_path: str,
    checkerboard_dir: str,
    row_corners: int,
    col_corners: int,
    output_path: str | None = None,
) -> dict:
    real_image = check_readable(image_path)
    img, _ = load_image_with_digest(real_image)
    real_dir = check_readable(checkerboard_dir)

    # The checkerboard directory is calibration INPUT and nothing else: every
    # file in it is read as a frame, and the cache is keyed on the digest of
    # its contents. A correction written there becomes a candidate frame on
    # the next call -- measured on real photos, the tool's own
    # <image>_undistorted.png came back in frames_skipped, caught only by the
    # majority-size rule because the correction happened to crop; a correction
    # that cropped nothing would be fitted as a distorted view of the camera
    # that produced it. Even skipped it changes the digest, so every image of
    # a batch recalibrates from scratch (3.7 s per call on nine 5-MP frames)
    # instead of reusing the cache this tool advertises. Refused rather than
    # filtered by name: filtering leaves the directory a sink, and a renamed
    # correction walks straight back in.
    intended = (
        os.path.splitext(real_image)[0] + "_undistorted.png"
        if output_path is None
        else output_path
    )
    if os.path.dirname(os.path.realpath(intended)) == os.path.realpath(real_dir):
        raise ValueError(
            f"The corrected image would be written into the checkerboard "
            f"directory {checkerboard_dir!r}, which is read as calibration "
            "input: every file there is a candidate frame and the cached "
            "calibration is keyed on the directory's contents, so the "
            "correction would be offered back to the next calibration and "
            "each image would refit from scratch. Nothing was written. Pass "
            "output_path pointing outside that directory, or move the image "
            "you are correcting out of it."
        )

    calib = _lens_calibration(real_dir, row_corners, col_corners)
    corrected, info = undistort_image(img, calib)

    if output_path is None:
        # The derived name is this tool's own to overwrite (re-running with a
        # better calibration is the normal workflow); write_image refuses to
        # follow a symlink squatting on it.
        out = os.path.splitext(real_image)[0] + "_undistorted.png"
        write_image(out, corrected)
    else:
        # A user-supplied path pointing at an existing file is not ours to
        # replace; O_EXCL makes the refusal atomic rather than a check-then-
        # write race. The NAME the caller gave is judged before it is
        # resolved: a dangling symlink there used to be followed and its
        # target created (panel of 1.9.0).
        if os.path.islink(output_path):
            raise OSError(
                f"output_path {output_path!r} is a symlink; images are written "
                "only to real files, so nothing was written. Remove the link "
                "or pass a different output_path."
            )
        if os.path.isdir(output_path):
            raise OSError(
                f"output_path {output_path!r} is a directory; pass the path of "
                "the file to create (it must not exist yet)."
            )
        out = check_readable(output_path)
        try:
            write_image(out, corrected, exclusive=True)
        except FileExistsError as exc:
            raise FileExistsError(
                f"output_path {output_path!r} already exists. Pass a new path, "
                "or omit output_path to write (and overwrite) the derived "
                "<image>_undistorted.png next to the input."
            ) from exc

    warnings = _lens_advisories(calib, info, out)
    preview, scale = downscale(corrected)
    png = encode_png(preview)
    return {
        "corrected_image_path": out,
        "frames_used": len(calib.frames_used),
        "frames_skipped": calib.frames_skipped,
        "frames_outliers": [
            {"name": n, "rms_px": round(px, 2), "reason": why}
            for n, px, why in calib.frames_outliers
        ],
        "frames_duplicates": [
            {"name": n, "same_as": first} for n, first in calib.frames_duplicates
        ],
        "rms_reprojection_error": calib.rms,
        "focal_uncertainty": calib.focal_uncertainty,
        "valid_roi": info["valid_roi"],
        "crop_fraction": info["crop_fraction"],
        "residual_void_px": info["residual_void_px"],
        "board_coverage": calib.coverage,
        "corrected_size": [int(corrected.shape[0]), int(corrected.shape[1])],
        "overlay_scale": scale,
        "warnings": warnings,
        "_png": png,
    }


def _loud(fn):
    """Convert every exception into a ToolError carrying its text.

    mcp 2.1 masks any exception other than ToolError to a bare `Error executing
    tool <name>` — the right contract for a network service that must not leak
    internals, and exactly the wrong one here: this is a local stdio instrument
    (paths.py states the trust boundary) whose refusal messages ARE the product.
    Verified live on 2.1.1: a WrongSessionKindError's "use measure_thermal()"
    guidance reached the client as nothing at all. Converting at the tool
    boundary — below the SDK, above the impls — keeps the impls raising their
    own types (tests and library callers see those) while every message crosses
    the wire. The class name is kept so a genuine bug still reads as one.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def build_server() -> MCPServer:
    # version= is what a client sees at initialize; MCPServer defaults it to "".
    mcp = MCPServer("plantcv-mcp", instructions=INSTRUCTIONS, version=__version__)

    # Every tool here only reads from disk and computes. None mutates anything,
    # none reaches the network. Saying so lets a client decide what is safe to
    # run without asking, instead of treating all four as opaque.
    #
    # snake_case since mcp 2.x. The camelCase spellings still work here as
    # constructor kwargs — pydantic keeps them as aliases — but the ATTRIBUTES
    # are snake_case now, so camelCase would keep this line green while every
    # read of these annotations broke.
    READ_ONLY = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @mcp.tool(title="List segmentation methods", annotations=READ_ONLY)
    @_loud
    def list_methods() -> MethodsInfo:
        """List available segmentation channels, methods, object types, analyses,
        and the pinned PlantCV version."""
        return list_methods_impl()

    @mcp.tool(title="Suggest a segmentation", annotations=READ_ONLY)
    @_loud
    def suggest_segmentation(
        image_path: str, channel: str = "a", method: str = "otsu"
    ) -> list:
        """Return colourspace and threshold contact sheets, plus what each
        object_type would yield on this image, so the channel/method/polarity
        choice is informed rather than blind. Call this before segment().
        polarity.warnings carries `noisy_segmentation` when the recommended
        polarity is many components with no dominant one — background texture,
        not a plant — which the `ambiguous` flag alone cannot tell you."""
        img = load_image(check_readable(image_path))
        cs, cs_scale = downscale(colorspace_sheet(img))
        th, th_scale = downscale(threshold_sheet(img, channel))
        return [
            json.dumps(
                {
                    "colorspace_sheet_scale": cs_scale,
                    "threshold_sheet_scale": th_scale,
                    "channel": channel,
                    "method": method,
                    "polarity": polarity_report(img, channel, method),
                }
            ),
            Image(data=encode_png(cs), format="png"),
            Image(data=encode_png(th), format="png"),
        ]

    @mcp.tool(title="Segment an image (returns the overlay)", annotations=READ_ONLY)
    @_loud
    def segment(
        image_path: str,
        channel: str,
        method: str,
        object_type: str = "dark",
        fill_size: int = 200,
        ksize: int = 11,
        offset: int = 2,
        color_correct: bool = False,
        exclude_color_card: bool = False,
    ) -> list:
        """Segment an image. Returns the overlay image and mask diagnostics —
        NOT traits. Use the returned session_id with measure() to get traits.

        object_type picks which side of the threshold is the plant ('dark' or
        'light'); the wrong one returns the background with plausible-looking
        traits, so call suggest_segmentation() if unsure. fill_size drops
        components smaller than itself and will erase a genuinely small
        specimen. ksize and offset apply to the 'mean' and 'gaussian' methods.
        color_correct requires a ColorChecker card in the frame and RAISES if it
        cannot find one; it makes colour traits comparable across lighting. The
        card's own region is then excluded from the mask (`color_card_excluded`
        advisory) — the card is the instrument, not a specimen. A card that is
        incomplete or partly covered is refused: a missing chip distorts the
        correction for every pixel. exclude_color_card=true keeps the card out
        of the mask WITHOUT correcting colours (and raises if there is none);
        with neither flag the server never looks for a card, so keep cards out
        of the frame or out of the measured regions.
        """
        result = _segment_impl(
            image_path,
            channel,
            method,
            object_type=object_type,
            fill_size=fill_size,
            ksize=ksize,
            offset=offset,
            color_correct=color_correct,
            exclude_color_card=exclude_color_card,
        )
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool(title="Refine a mask (returns the new overlay)", annotations=READ_ONLY)
    @_loud
    def refine(session_id: str, ops: list[dict]) -> list:
        """Apply morphological cleanup to a session's mask and get a NEW session.

        ops is an ordered list like [{"op": "fill_holes"}, {"op": "keep_largest",
        "n": 1}]; list_methods() documents every op and its parameters. The
        original session is untouched and still measurable, so a refinement that
        looks wrong is simply discarded. Returns the refined overlay — look at it
        — plus before/after mask diagnostics and warnings; NOT traits. Every op
        is validated before any runs, and a refinement that leaves no measurable
        plant is refused rather than minted. A `refine_dropped_object` warning
        names any component ≥10% of the largest that an op removed entirely,
        and the earlier op that split it off. measure() results carry
        `lineage`, the ops that produced their mask.
        """
        result = _refine_impl(session_id, ops)
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool(title="Measure traits from a segmentation", annotations=READ_ONLY)
    @_loud
    def measure(
        session_id: str,
        analyses: list[str] | None = None,
        px_per_mm: float | None = None,
        include_histograms: bool = False,
    ) -> MeasureResult:
        """Return plant traits for a segmentation produced by segment().
        Raises if the mask is degenerate rather than returning zeros.

        analyses defaults to ["size"]; add "color" for hue/saturation/value
        statistics. px_per_mm converts spatial traits to mm and mm2 — without it
        every size is in PIXELS and is not comparable between images shot at
        different distances or zoom. include_histograms adds three frequency
        arrays totalling 692 numbers, off by default.
        """
        return _measure_impl(
            session_id,
            analyses=analyses,
            px_per_mm=px_per_mm,
            include_histograms=include_histograms,
        )

    @mcp.tool(
        title="Measure each plant in a tray (returns the labelled overlay)",
        annotations=READ_ONLY,
    )
    @_loud
    def measure_regions(
        session_id: str,
        nrows: int,
        ncols: int,
        mode: str = "auto_grid",
        coord: list[int] | None = None,
        height: int | None = None,
        width: int | None = None,
        spacing: list[int] | None = None,
        radius: int | None = None,
        analyses: list[str] | None = None,
        px_per_mm: float | None = None,
        include_histograms: bool = False,
        indices: list[str] | None = None,
    ) -> list:
        """Measure EACH plant in a multi-plant image separately.

        Works on every session kind: an RGB session returns traits per
        region (analyses / px_per_mm apply); a thermal session returns
        temperature statistics per region; a hyperspectral session returns
        index statistics per region (`indices`, default the session's index).
        Arguments belonging to another modality are refused, not ignored.

        measure() treats the whole frame as one region, so on a tray it merges
        every plant into a single object and every size trait describes the
        group. This returns one row per region, plus an overlay with each region
        outlined and numbered so you can tell which row is which plant.

        mode='auto_grid' (default) infers the grid geometry from the MASK — give
        only nrows and ncols. mode='rect_grid' takes explicit geometry: coord
        [x, y] of the first cell, height and width of a cell, and spacing [x, y]
        between cell origins; use it when the mask is too sparse to infer a
        layout or the cells must line up with physical pots. auto_grid needs at
        least as many objects as rows and as columns, spread over both; a mask
        that cannot support the layout is refused by name (a single plant is
        measure(); a known layout is rect_grid).

        A region with no plant in it returns measured=false and a reason, NOT
        zeros: PlantCV reports a full trait set of zeros for an empty region,
        which is indistinguishable from a genuinely zero-area plant.
        """
        result = _measure_regions_impl(
            session_id,
            mode=mode,
            nrows=nrows,
            ncols=ncols,
            coord=coord,
            height=height,
            width=width,
            spacing=spacing,
            radius=radius,
            analyses=analyses,
            px_per_mm=px_per_mm,
            include_histograms=include_histograms,
            indices=indices,
        )
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool(
        title="Measure morphology: leaves, stem, branch points (returns the overlay)",
        annotations=READ_ONLY,
    )
    @_loud
    def measure_morphology(
        session_id: str,
        prune_size: int = 15,
        tangent_size: int = 25,
        px_per_mm: float | None = None,
    ) -> list:
        """Skeleton-based traits for ONE plant: per-segment path and euclidean
        length, curvature, angle, tangent angle and insertion angle, plus stem
        height/length/angle, tip and branch-point counts, cycles and widths.
        Returns the numbered-segment overlay with the table — segment `id` is
        the number drawn on the picture.

        prune_size removes skeleton spurs shorter than itself; the response
        warns (prune_size_sensitive) when the segment count changes by >30% at
        twice this value, because then the table describes the parameter, not
        the plant. tangent_size is the pixel window for tangent/insertion
        angles (default 25, measured: smaller windows bias insertion angles
        upward, a window longer than a leaf collapses its angle to 0 — flagged
        as tangent_window_exceeds_segment). A vertical stem has no defined angle:
        stem_angle is null (stem_angle_undefined) and so is every
        insertion_angle (insertion_angle_undefined) rather than a nonsense
        number. Multi-plant and inverted (implausible_coverage) masks are
        refused; use measure_regions() or refine(keep_largest) first. px_per_mm
        scales lengths only; angles stay in degrees.
        """
        result = _measure_morphology_impl(
            session_id,
            prune_size=prune_size,
            tangent_size=tangent_size,
            px_per_mm=px_per_mm,
        )
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool(title="Calibrate pixels per millimetre", annotations=READ_ONLY)
    @_loud
    def calibrate_scale_from_marker(
        image_path: str,
        x: int,
        y: int,
        w: int,
        h: int,
        marker_length_mm: float,
        channel: str = "v",
        method: str = "otsu",
        object_type: str = "dark",
    ) -> ScaleResult:
        """Measure a size marker of known real length and return px_per_mm, which
        you then pass to measure() to get traits in mm instead of pixels.

        (x, y, w, h) bounds a region containing ONLY the marker, with a little
        margin. marker_length_mm is its longest real dimension — the diameter of a
        circular marker. Check the returned marker_length_px against what you
        expect: a wrong scale silently rescales every trait you measure afterwards.
        """
        img = load_image(check_readable(image_path))
        est = calibrate_scale(
            img,
            x,
            y,
            w,
            h,
            marker_length_mm,
            channel=channel,
            method=method,
            object_type=object_type,
        )
        return {
            "px_per_mm": est.px_per_mm,
            "marker_length_px": est.marker_length_px,
            "marker_length_mm": est.marker_length_mm,
            "marker_area_px": est.marker_area_px,
            "crop_fraction": est.crop_fraction,
            "warnings": [{"code": a.code, "message": a.message} for a in est.warnings],
        }

    # The one tool that writes: a corrected copy of the input image, under a
    # derived name it owns (an explicit output_path refuses to overwrite).
    WRITES_A_FILE = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @mcp.tool(
        title="Correct lens distortion (writes a corrected image)",
        annotations=WRITES_A_FILE,
    )
    @_loud
    def correct_lens_distortion(
        image_path: str,
        checkerboard_dir: str,
        row_corners: int,
        col_corners: int,
        output_path: str | None = None,
    ) -> list:
        """Correct lens/fisheye distortion using checkerboard calibration photos,
        write the corrected image next to the input as <image>_undistorted.png
        (or to output_path, refused if it exists), and return its path plus a
        preview. Segment, calibrate scale, and measure the CORRECTED file.
        The correction is never written INTO checkerboard_dir: everything there
        is read as a calibration frame, so pass output_path elsewhere when the
        image you are correcting sits with the calibration photos.

        Use this FIRST when straight edges bow in the image: measured on a real
        fisheye photo, distortion inflated plant area 2.13x and varied by
        direction, so no px_per_mm can compensate — a marker and the plant it
        scales distort differently. checkerboard_dir holds several photos of a
        checkerboard taken with the SAME camera; row_corners/col_corners count
        its INNER corners (a 10x7-square board has 9x6). Frames where no board
        is detected are skipped and named; fewer than 3 detections is refused.
        The corrected frame is cropped to the region where every pixel is real
        (remap voids removed); calibration is cached on the directory's content,
        so correcting a batch re-uses it.
        """
        result = _correct_lens_impl(
            image_path,
            checkerboard_dir,
            row_corners=row_corners,
            col_corners=col_corners,
            output_path=output_path,
        )
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool(title="Measure many images with one recipe", annotations=READ_ONLY)
    @_loud
    def measure_images(
        image_paths: list[str],
        channel: str,
        method: str,
        object_type: str = "dark",
        fill_size: int = 200,
        ksize: int = 11,
        offset: int = 2,
        color_correct: bool = False,
        exclude_color_card: bool = False,
        analyses: list[str] | None = None,
        px_per_mm: float | None = None,
        max_seconds: float | None = DEFAULT_MAX_SECONDS,
        nrows: int | None = None,
        ncols: int | None = None,
        mode: str = "auto_grid",
        coord: list[int] | None = None,
        height: int | None = None,
        width: int | None = None,
        spacing: list[int] | None = None,
        radius: int | None = None,
    ) -> BatchResult:
        """Run one fixed segmentation recipe across many images.

        For trays give nrows/ncols (and rect_grid geometry if needed): each
        image is then measured per plant exactly as measure_regions() does and
        the row carries `regions` instead of `traits`. Without a grid a tray
        returns the GROUP's traits with a `multi_specimen` advisory.

        Real photographs take ~10 s each; max_seconds (default 300) bounds the
        call and images not started in time come back as not_run with their
        paths listed for resubmission. Each row reports its own `seconds`.

        There is no overlay here — nobody reviews two hundred of them — so instead
        every image runs the SAME guards as segment(), and any image that trips a
        blocking guard is returned with NO traits and a reason to inspect it
        individually. A batch never returns a number the server could not validate.
        With a grid, noisy_segmentation stops blocking only when the grid explains
        the components (at most four per cell) and implausible_coverage only with
        two or more cells; an image none of whose cells measured is refused as
        no_region_measured with the per-cell reasons.

        Settle the recipe on one representative image with suggest_segmentation()
        and segment() first, looking at the overlay, then apply it here with the
        SAME arguments — including ksize/offset for the 'mean' and 'gaussian'
        methods and color_correct if you used it; the returned `recipe` records
        exactly what ran. An image that cannot be colour-corrected when asked is
        refused, not measured raw, and the detected card's region is excluded
        from every corrected mask (`color_card_excluded`) — the card is the
        instrument, not a specimen; exclude_color_card=true does the same
        without correcting colours. Under a grid, a cell that is nearly all
        material in a mask covering most of the frame is refused as
        probable_background, and a cell holding several comparable specks in
        a mask that is texture overall refuses the image as
        noisy_segmentation. Limit is 200 images per call.
        """
        # Every path is checked BEFORE any is loaded: a batch with one stray
        # path is refused whole, not partially run.
        image_paths = [check_readable(p) for p in image_paths]
        return dispatch(
            "batch",
            image_paths,
            channel=channel,
            method=method,
            object_type=object_type,
            fill_size=fill_size,
            ksize=ksize,
            offset=offset,
            color_correct=color_correct,
            exclude_color_card=exclude_color_card,
            analyses=tuple(analyses) if analyses is not None else ("size",),
            px_per_mm=px_per_mm,
            max_seconds=max_seconds,
            nrows=nrows,
            ncols=ncols,
            mode=mode,
            coord=_as_xy(coord, "coord"),
            height=height,
            width=width,
            spacing=_as_xy(spacing, "spacing"),
            radius=radius,
        )

    @mcp.tool(
        title="Segment a hyperspectral cube by a spectral index (returns the overlay)",
        annotations=READ_ONLY,
    )
    @_loud
    def segment_hyperspectral(
        envi_path: str,
        index: str = "ndvi",
        threshold: float = 0.2,
        object_type: str = "light",
        white_reference: str | None = None,
        dark_reference: str | None = None,
        fill_size: int = 200,
    ) -> list:
        """Compute one spectral index over an ENVI cube (.raw/.hdr pair; give
        either path) and threshold it into a mask, returning the overlay on the
        cube's pseudo-RGB plus diagnostics — NOT numbers. Use the session_id
        with measure_spectral().

        Cubes are usually integer counts: an index computed on counts wraps
        around silently, so the cube is either CALIBRATED to reflectance from
        white_reference + dark_reference (ENVI pairs; refused if white - dark is
        not positive everywhere) or cast to float with the `uncalibrated_cube`
        warning, meaning indices are relative, not reflectance. index is any
        PlantCV spectral index (list_methods reports them); one the cube's
        wavelength range cannot support is refused by name. object_type picks
        the side of the threshold that is the plant.
        """
        result = _segment_hsi_impl(
            envi_path,
            index=index,
            threshold=threshold,
            object_type=object_type,
            white_reference=white_reference,
            dark_reference=dark_reference,
            fill_size=fill_size,
        )
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool(title="Measure spectral indices and reflectance", annotations=READ_ONLY)
    @_loud
    def measure_spectral(
        session_id: str,
        indices: list[str] | None = None,
        include_spectrum: bool = False,
        include_histograms: bool = False,
    ) -> SpectralMeasureResult:
        """Per requested index: mean, median, std, min, max over the session's
        mask, computed on the calibrated (or float-cast) cube exactly as
        segment_hyperspectral() prepared it. indices defaults to the index the
        session was segmented with. include_spectrum adds the per-band mean/
        max/min/std reflectance — hundreds of numbers per list, off by default;
        band_count is always reported. Refuses RGB and thermal sessions.
        """
        return _measure_spectral_impl(
            session_id,
            indices=indices,
            include_spectrum=include_spectrum,
            include_histograms=include_histograms,
        )

    @mcp.tool(
        title="Segment a thermal frame by temperature (returns the overlay)",
        annotations=READ_ONLY,
    )
    @_loud
    def segment_thermal(
        path: str,
        min_c: float | None = None,
        max_c: float | None = None,
        fill_size: int = 200,
    ) -> list:
        """Read a FLIR radiometric .jpg (via flyr), a .csv, or a .npz of degrees
        Celsius and select the pixels between min_c and max_c (give at least
        one) as the plant. Returns the overlay on a grey rendering of the frame,
        the frame's temperature range and diagnostics — NOT numbers; use the
        session_id with measure_thermal(). A thermal frame is a different
        sensor from the RGB camera, so a mask is never borrowed from an RGB
        session. A band that selects nothing is refused.
        """
        result = _segment_thermal_impl(
            path, min_c=min_c, max_c=max_c, fill_size=fill_size
        )
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool(title="Measure temperatures under a thermal mask", annotations=READ_ONLY)
    @_loud
    def measure_thermal(
        session_id: str, include_histograms: bool = False
    ) -> ThermalMeasureResult:
        """max, min, mean and median degrees Celsius over the session's mask,
        via PlantCV's analyze.thermal, plus the pixel count and the frame's
        range. include_histograms adds the 100-bin temperature histogram.
        Refuses RGB and hyperspectral sessions.
        """
        return _measure_thermal_impl(session_id, include_histograms=include_histograms)

    return mcp


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="plantcv-mcp",
        description="PlantCV as an MCP measurement instrument (stdio).",
    )
    parser.add_argument(
        "--isolate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "run every PlantCV analysis in a worker subprocess so a native crash "
            "becomes a tool error instead of killing the server. ON by default "
            "(measured +7.7%% wall time); --no-isolate or PLANTCV_MCP_ISOLATE=0 "
            "runs analyses in-process."
        ),
    )
    parser.add_argument(
        "--root",
        action="append",
        metavar="DIR",
        help=(
            "only read images — and write correct_lens_distortion's output — "
            "under DIR (repeatable; also PLANTCV_MCP_ROOTS, os.pathsep-separated). "
            "Unset: read anything the host user can read."
        ),
    )
    args = parser.parse_args()
    if args.isolate is not None:
        set_isolation(args.isolate)
    if args.root:
        set_roots(args.root)
    build_server().run()
