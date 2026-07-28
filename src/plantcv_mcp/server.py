"""MCP server. Four tools over a session store.

segment() mints a session and returns the overlay but NO traits; measure()
requires that session. The split is deliberate: it forces the visual evidence
into the model's context before a number can be obtained.
"""

import json

from mcp.server.fastmcp import FastMCP, Image

from . import plantcv_version
from .diagnostics import (
    analyze_mask,
    frame_clipping_warning,
    multi_specimen_warning,
)
from .imaging import downscale, encode_png, load_image, render_overlay
from .measurement import measure_traits
from .segmentation import CHANNELS, METHODS, segment_mask
from .session import SessionStore
from .suggest import colorspace_sheet, threshold_sheet

_store = SessionStore()


def list_methods_impl() -> dict:
    return {
        "plantcv_version": plantcv_version(),
        "channels": sorted(CHANNELS),
        "methods": list(METHODS),
        "guidance": (
            "The 'a' channel of LAB separates green tissue from most backgrounds; "
            "'s' of HSV can work on non-green tissue. Call suggest_segmentation() "
            "to compare on your actual image rather than guessing."
        ),
    }


def _segment_impl(image_path: str, channel: str, method: str) -> dict:
    img = load_image(image_path)
    mask = segment_mask(img, channel=channel, method=method)
    diag = analyze_mask(mask)
    warnings = [
        w for w in (multi_specimen_warning(diag), frame_clipping_warning(mask)) if w
    ]
    session = _store.create(image_path, mask, channel, method)
    overlay, scale = downscale(render_overlay(img, mask))
    png = encode_png(overlay)
    return {
        "session_id": session.session_id,
        "mask_fraction": diag.mask_fraction,
        "component_count": diag.component_count,
        "major_object_count": diag.major_object_count,
        "largest_area": diag.largest_area,
        "overlay_scale": scale,
        "overlay_png_bytes": len(png),
        "_png": png,
        "warnings": [{"code": w.code, "message": w.message} for w in warnings],
    }


def _measure_impl(session_id: str) -> dict:
    session = _store.get(session_id)
    img = load_image(session.image_path)  # re-read; sessions do not hold RGB
    return {"session_id": session_id, "traits": measure_traits(img, session.mask)}


def build_server() -> FastMCP:
    mcp = FastMCP("plantcv-mcp")

    @mcp.tool()
    def list_methods() -> dict:
        """List available segmentation channels, methods, and the pinned PlantCV version."""
        return list_methods_impl()

    @mcp.tool()
    def suggest_segmentation(image_path: str, channel: str = "a") -> list:
        """Return colourspace and threshold contact sheets so the channel/method
        choice is informed rather than blind. Call this before segment()."""
        img = load_image(image_path)
        cs, _ = downscale(colorspace_sheet(img))
        th, _ = downscale(threshold_sheet(img, channel))
        return [
            Image(data=encode_png(cs), format="png"),
            Image(data=encode_png(th), format="png"),
        ]

    @mcp.tool()
    def segment(image_path: str, channel: str, method: str) -> list:
        """Segment an image. Returns the overlay image and mask diagnostics —
        NOT traits. Use the returned session_id with measure() to get traits."""
        result = _segment_impl(image_path, channel, method)
        png = result.pop("_png")
        return [json.dumps(result), Image(data=png, format="png")]

    @mcp.tool()
    def measure(session_id: str) -> dict:
        """Return plant traits for a segmentation produced by segment().
        Raises if the mask is degenerate rather than returning zeros."""
        return _measure_impl(session_id)

    return mcp


def main() -> None:
    build_server().run()
