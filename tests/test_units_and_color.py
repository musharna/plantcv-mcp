"""Real-world units, colour analysis, and MCP protocol metadata.

The unit conversion has one trap worth stating: PlantCV labels BOTH `area` and
`width` as "pixels". Scaling everything with that label linearly would leave every
area wrong by exactly a factor of px_per_mm — plausible, correctly-united, and
silently wrong, which is the failure mode this whole server exists to catch.
"""

import asyncio
import json

import pytest

from plantcv_mcp.imaging import load_image
from plantcv_mcp.measurement import (
    AREA_TRAITS,
    HISTOGRAM_TRAITS,
    LINEAR_TRAITS,
    UnknownAnalysisError,
    convert_units,
    measure_traits,
)
from plantcv_mcp.segmentation import segment_mask
from plantcv_mcp.server import _measure_impl, _segment_impl, build_server

FIXTURE = "tests/fixtures/multi_specimen.png"


def _traits(**kwargs):
    img = load_image(FIXTURE)
    return measure_traits(img, segment_mask(img, "a", "otsu"), **kwargs)


# --------------------------------------------------------------------------
# unit conversion
# --------------------------------------------------------------------------


def test_area_scales_quadratically_and_length_linearly():
    """The trap: both are labelled 'pixels' upstream, but they must not scale the
    same way. At 10 px/mm a length divides by 10 and an area by 100."""
    px = _traits()
    mm = _traits(px_per_mm=10.0)

    assert mm["width"]["unit"] == "mm"
    assert mm["width"]["value"] == pytest.approx(px["width"]["value"] / 10.0)

    assert mm["area"]["unit"] == "mm2"
    assert mm["area"]["value"] == pytest.approx(px["area"]["value"] / 100.0)

    # If area had been treated as linear it would be 10x larger than this. Assert
    # the wrong answer is NOT what we produced, so a regression cannot hide.
    assert mm["area"]["value"] != pytest.approx(px["area"]["value"] / 10.0)


def test_every_linear_and_area_trait_is_actually_converted():
    px, mm = _traits(), _traits(px_per_mm=2.0)
    for name in LINEAR_TRAITS & px.keys():
        assert mm[name]["unit"] == "mm", f"{name} was left in pixels"
        assert mm[name]["value"] == pytest.approx(px[name]["value"] / 2.0)
    for name in AREA_TRAITS & px.keys():
        assert mm[name]["unit"] == "mm2", f"{name} was left in pixels"
        assert mm[name]["value"] == pytest.approx(px[name]["value"] / 4.0)


def test_positions_and_unitless_traits_are_left_alone():
    """A millimetre coordinate has no meaning without an origin, and solidity is
    a ratio. Converting either would be a fabrication."""
    px, mm = _traits(), _traits(px_per_mm=3.0)
    assert mm["center_of_mass"]["value"] == px["center_of_mass"]["value"]
    assert mm["solidity"]["value"] == pytest.approx(px["solidity"]["value"])
    assert mm["ellipse_angle"]["unit"] == "degrees"


def test_nonpositive_scale_is_refused():
    for bad in (0, -5):
        with pytest.raises(ValueError):
            convert_units({"area": {"value": 1.0, "unit": "pixels"}}, bad)
    # Positive control in the same test: a valid scale still converts.
    ok = convert_units({"area": {"value": 100.0, "unit": "pixels"}}, 10.0)
    assert ok["area"]["value"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# colour analysis
# --------------------------------------------------------------------------


def test_color_analysis_adds_hue_stats_and_size_alone_does_not():
    size_only = _traits(analyses=("size",))
    with_color = _traits(analyses=("size", "color"))

    assert "hue_circular_mean" not in size_only, "size must not smuggle in colour"
    assert "hue_circular_mean" in with_color
    assert with_color["hue_circular_mean"]["unit"] == "degrees"
    assert "saturation_mean" in with_color
    # size traits survive alongside colour
    assert with_color["area"]["value"] == pytest.approx(size_only["area"]["value"])


def test_histograms_are_withheld_unless_asked_for():
    """692 numbers is a context-window problem, not a feature."""
    without = _traits(analyses=("color",))
    with_hist = _traits(analyses=("color",), include_histograms=True)

    assert not (HISTOGRAM_TRAITS & without.keys()), "histograms leaked by default"
    assert HISTOGRAM_TRAITS <= with_hist.keys(), "histograms missing when requested"
    assert len(with_hist["hue_frequencies"]["value"]) == 180


def test_unknown_or_empty_analysis_is_refused():
    with pytest.raises(UnknownAnalysisError):
        _traits(analyses=("size", "nonsense"))
    with pytest.raises(UnknownAnalysisError):
        _traits(analyses=())
    # Positive control: the valid names still work.
    assert _traits(analyses=("size", "color"))


# --------------------------------------------------------------------------
# server wiring
# --------------------------------------------------------------------------


def test_measure_tool_passes_scale_and_analyses_through():
    """Correct in the library but dropped at the server is the exact shape of the
    original object_type bug, so assert the passthrough at the server."""
    seg = _segment_impl(FIXTURE, "a", "otsu")
    plain = _measure_impl(seg["session_id"])
    scaled = _measure_impl(seg["session_id"], analyses=["size", "color"], px_per_mm=5.0)

    assert plain["px_per_mm"] is None
    assert plain["analyses"] == ["size"]
    assert scaled["px_per_mm"] == 5.0
    assert scaled["analyses"] == ["size", "color"]
    assert scaled["traits"]["width"]["unit"] == "mm"
    assert scaled["traits"]["width"]["value"] == pytest.approx(
        plain["traits"]["width"]["value"] / 5.0
    )
    assert "hue_circular_mean" in scaled["traits"]


# --------------------------------------------------------------------------
# MCP protocol metadata
# --------------------------------------------------------------------------


def test_server_publishes_instructions_that_state_the_discipline():
    server = build_server()
    assert server.instructions, "server instructions must not be empty"
    text = server.instructions.lower()
    assert "overlay" in text, "instructions must tell the model to look at the overlay"
    assert "implausible_coverage" in text, "instructions must name the warnings"


def test_every_tool_publishes_annotations_and_a_title():
    tools = asyncio.run(build_server().list_tools())
    assert len(tools) == 9
    for t in tools:
        assert t.title, f"{t.name} has no title"
        assert t.annotations is not None, f"{t.name} has no annotations"
        # snake_case since mcp 2.x. camelCase survives as a pydantic ALIAS for
        # constructing annotations, but that does not extend to reading them.
        assert t.annotations.read_only_hint is True, f"{t.name} not marked read-only"
        assert t.annotations.open_world_hint is False, (
            f"{t.name} not marked closed-world"
        )


def test_typed_tools_publish_an_output_schema():
    """A bare `-> dict` yields output_schema=None; a TypedDict return generates a
    real schema, which is what lets a client consume traits without parsing text."""
    by_name = {t.name: t for t in asyncio.run(build_server().list_tools())}

    for name in ("measure", "list_methods"):
        schema = by_name[name].output_schema
        assert schema is not None, f"{name} publishes no output_schema"
        assert schema.get("type") == "object"

    assert "traits" in by_name["measure"].output_schema["properties"]
    assert "px_per_mm" in by_name["measure"].output_schema["properties"]


def test_every_structured_tool_publishes_an_output_schema():
    """Generalises the check above so a NEW tool cannot quietly regress it.

    Written after exactly that happened: calibrate_scale_from_marker and
    measure_images were first added with a bare `-> dict`, so they returned a JSON
    string in a text block and no schema, while the older tools returned structured
    content. Only the two tools that return IMAGE blocks are exempt.
    """
    returns_images = {
        "segment",
        "suggest_segmentation",
        "measure_regions",
        "refine",
        "measure_morphology",
    }
    # measure_regions joined this set when per-region measurement shipped: it
    # returns the labelled overlay alongside the rows, because per-region
    # numbers are unreadable without a picture saying which region is which.
    # refine returns the refined overlay for the same reason: a cleanup nobody
    # looked at is how a hole-fill swallows the second plant.
    for tool in asyncio.run(build_server().list_tools()):
        if tool.name in returns_images:
            assert tool.output_schema is None, (
                f"{tool.name} returns image content; it should have no output_schema"
            )
            continue
        assert tool.output_schema is not None, (
            f"{tool.name} returns structured data but publishes no output_schema — "
            "annotate its return type with a TypedDict"
        )


def test_measure_over_the_real_mcp_layer_returns_structured_content():
    server = build_server()
    seg = asyncio.run(
        server.call_tool(
            "segment", {"image_path": FIXTURE, "channel": "a", "method": "otsu"}
        )
    )
    # mcp 2.x returns a CallToolResult; 1.x returned a (content, structured)
    # tuple, which is what the old isinstance sniff handled. We are 2.x-only, so
    # read the fields. A shape sniff is worse than useless here: under 2.x it
    # falls through and hands back the RESULT OBJECT, so the mismatch surfaces
    # somewhere downstream instead of at the call.
    blocks = seg.content
    sid = json.loads(
        next(b for b in blocks if getattr(b, "type", None) == "text").text
    )["session_id"]

    res = server.call_tool("measure", {"session_id": sid, "px_per_mm": 4.0})
    payload = asyncio.run(res).structured_content
    assert isinstance(payload, dict), (
        f"expected structured content, got {type(payload)}"
    )
    assert payload["px_per_mm"] == 4.0
    assert payload["traits"]["area"]["unit"] == "mm2"
