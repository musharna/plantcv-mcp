"""MCP server. Six tools over a session store.

segment() mints a session and returns the overlay but NO traits; measure()
requires that session. The split is deliberate: it forces the visual evidence
into the model's context before a number can be obtained.

Tool functions are deliberately SYNCHRONOUS. mcp 1.28.1 invokes sync tools inline
on the event loop (fastmcp/utilities/func_metadata.py: `return fn(...)`, with no
anyio.to_thread), which serialises them. That costs latency — a slow segmentation
blocks the whole server — but it is what makes PlantCV's process-global
`pcv.outputs` safe to use here. Making any tool `async`, or offloading to a
thread, would allow two analyses to interleave on that global and must not be
done without a lock around the measurement section.
"""

import json

from mcp.server.fastmcp import FastMCP, Image
from mcp.types import ToolAnnotations
from plantcv import plantcv as pcv

# See measurement.py: typing.TypedDict breaks schema generation on 3.11.
from typing_extensions import TypedDict

from . import plantcv_version
from .batch import measure_batch
from .color import correct_color
from .diagnostics import analyze_mask, segmentation_warnings
from .imaging import downscale, encode_png, file_digest, load_image, render_overlay
from .measurement import ANALYSES, TraitValue, measure_traits
from .scale import calibrate_scale
from .segmentation import CHANNELS, METHODS, OBJECT_TYPES, threshold_mask
from .session import SessionStore
from .suggest import colorspace_sheet, polarity_report, threshold_sheet

_store = SessionStore()

# Shown to clients at connection time. The product is a discipline, not just a set
# of functions, and the discipline has to be stated somewhere the model will read
# it before it starts calling things.
INSTRUCTIONS = """\
PlantCV as a measurement instrument for plant images.

Work in this order: suggest_segmentation -> segment -> LOOK AT THE OVERLAY -> measure.

segment() deliberately returns no traits. It returns an overlay image with the
measured pixels tinted red, plus diagnostics. Look at that image before you trust
any number that follows: a segmentation that selected the background instead of
the plant still produces seventeen traits with correct units and believable
magnitudes. The overlay is the only thing that distinguishes the two.

Read the `warnings` array on every segment() response. `implausible_coverage`
means the mask is probably inverted — re-run with the opposite `object_type`.
`multi_specimen` means several plants were merged into one measurement.
`fill_erased_mask` means `fill_size` deleted the specimen. `empty_mask` means the
segmentation found nothing.

Do not guess `channel` or `object_type`. suggest_segmentation reports what both
polarities actually yield on the image in front of you.

Traits are in PIXELS unless you pass px_per_mm to measure(), and pixel sizes are
not comparable between images taken at different distances or zoom levels.\
"""


class MeasureResult(TypedDict):
    """Return type of measure(). Annotated so MCP can publish an outputSchema."""

    session_id: str
    analyses: list[str]
    px_per_mm: float | None
    traits: dict[str, TraitValue]


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


class MethodsInfo(TypedDict):
    """Return type of list_methods()."""

    plantcv_version: str
    channels: list[str]
    methods: list[str]
    object_types: list[str]
    analyses: list[str]
    guidance: str


class ImageChangedSinceSegmentationError(Exception):
    """Raised when the file on disk is no longer the image that was segmented.

    Checked two ways: the frame shape, and a SHA-256 of the file's bytes. Shape
    alone let a same-dimension content swap through, silently measuring a stale
    mask against new pixels.
    """


def list_methods_impl() -> MethodsInfo:
    return {
        "plantcv_version": plantcv_version(),
        "channels": sorted(CHANNELS),
        "methods": list(METHODS),
        "object_types": list(OBJECT_TYPES),
        "analyses": list(ANALYSES),
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
) -> dict:
    img = load_image(image_path)
    if color_correct:
        # Raises if no card is found. Silently measuring an uncorrected image after
        # being asked to correct it would be the same confident wrongness as an
        # inverted mask.
        img = correct_color(img)

    # Threshold and fill run separately so a mask erased by fill_size can be
    # named as such instead of looking like a bad channel/method choice.
    pre_fill = threshold_mask(
        img, channel, method, object_type=object_type, ksize=ksize, offset=offset
    )
    mask = pcv.fill(bin_img=pre_fill, size=fill_size)

    diag = analyze_mask(mask)
    # Shared with the batch path so the two cannot apply different guards.
    warnings = segmentation_warnings(mask, diag, analyze_mask(pre_fill), fill_size)

    session = _store.create(
        image_path,
        mask,
        channel,
        method,
        digest=file_digest(image_path),
        color_correct=color_correct,
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


def _measure_impl(
    session_id: str,
    analyses: list[str] | None = None,
    px_per_mm: float | None = None,
    include_histograms: bool = False,
) -> MeasureResult:
    requested = tuple(analyses) if analyses else ("size",)
    session = _store.get(session_id)
    img = load_image(session.image_path)  # re-read; sessions do not hold RGB
    current_shape = (int(img.shape[0]), int(img.shape[1]))
    if current_shape != session.shape:
        raise ImageChangedSinceSegmentationError(
            f"Image at {session.image_path!r} is now {current_shape} but was "
            f"{session.shape} when segment() produced this session's mask. "
            "The file changed on disk since segmentation, so the mask no "
            "longer corresponds to its current content. Re-run segment() on "
            "the current file before measuring it."
        )
    if session.digest and file_digest(session.image_path) != session.digest:
        raise ImageChangedSinceSegmentationError(
            f"Image at {session.image_path!r} still has shape {current_shape}, "
            "but its CONTENT changed since segment() produced this session's "
            "mask (SHA-256 mismatch). The mask describes pixels that are no "
            "longer there. Re-run segment() on the current file."
        )
    if session.color_correct:
        # Applied AFTER the integrity guards, so a missing colour card cannot mask
        # a stale-image error. The mask was drawn on corrected pixels, so the
        # traits have to be measured on corrected pixels too.
        img = correct_color(img)
    return {
        "session_id": session_id,
        "analyses": list(requested),
        "px_per_mm": px_per_mm,
        "traits": measure_traits(
            img,
            session.mask,
            analyses=requested,
            px_per_mm=px_per_mm,
            include_histograms=include_histograms,
        ),
    }


def build_server() -> FastMCP:
    mcp = FastMCP("plantcv-mcp", instructions=INSTRUCTIONS)

    # Every tool here only reads from disk and computes. None mutates anything,
    # none reaches the network. Saying so lets a client decide what is safe to
    # run without asking, instead of treating all four as opaque.
    READ_ONLY = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @mcp.tool(title="List segmentation methods", annotations=READ_ONLY)
    def list_methods() -> MethodsInfo:
        """List available segmentation channels, methods, object types, analyses,
        and the pinned PlantCV version."""
        return list_methods_impl()

    @mcp.tool(title="Suggest a segmentation", annotations=READ_ONLY)
    def suggest_segmentation(
        image_path: str, channel: str = "a", method: str = "otsu"
    ) -> list:
        """Return colourspace and threshold contact sheets, plus what each
        object_type would yield on this image, so the channel/method/polarity
        choice is informed rather than blind. Call this before segment()."""
        img = load_image(image_path)
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
    def segment(
        image_path: str,
        channel: str,
        method: str,
        object_type: str = "dark",
        fill_size: int = 200,
        ksize: int = 11,
        offset: int = 2,
        color_correct: bool = False,
    ) -> list:
        """Segment an image. Returns the overlay image and mask diagnostics —
        NOT traits. Use the returned session_id with measure() to get traits.

        object_type picks which side of the threshold is the plant ('dark' or
        'light'); the wrong one returns the background with plausible-looking
        traits, so call suggest_segmentation() if unsure. fill_size drops
        components smaller than itself and will erase a genuinely small
        specimen. ksize and offset apply to the 'mean' and 'gaussian' methods.
        color_correct requires a ColorChecker card in the frame and RAISES if it
        cannot find one; it makes colour traits comparable across lighting.
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
        )
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool(title="Measure traits from a segmentation", annotations=READ_ONLY)
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

    @mcp.tool(title="Calibrate pixels per millimetre", annotations=READ_ONLY)
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
        img = load_image(image_path)
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

    @mcp.tool(title="Measure many images with one recipe", annotations=READ_ONLY)
    def measure_images(
        image_paths: list[str],
        channel: str,
        method: str,
        object_type: str = "dark",
        fill_size: int = 200,
        analyses: list[str] | None = None,
        px_per_mm: float | None = None,
    ) -> BatchResult:
        """Run one fixed segmentation recipe across many images.

        There is no overlay here — nobody reviews two hundred of them — so instead
        every image runs the SAME guards as segment(), and any image that trips a
        blocking guard is returned with NO traits and a reason to inspect it
        individually. A batch never returns a number the server could not validate.

        Settle the recipe on one representative image with suggest_segmentation()
        and segment() first, looking at the overlay, then apply it here. Limit is
        200 images per call.
        """
        return measure_batch(
            image_paths,
            channel,
            method,
            object_type=object_type,
            fill_size=fill_size,
            analyses=tuple(analyses) if analyses else ("size",),
            px_per_mm=px_per_mm,
        )

    return mcp


def main() -> None:
    build_server().run()
