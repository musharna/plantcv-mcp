"""measure() does not hand back the traits of a mask that is probably the background.

It used to return all seventeen, units and all, with `implausible_coverage`
beside them, on the same mask `count_leaves`, `measure_morphology` and the batch
refused (MCP-server audit, 2026-11-10). A warning beside the numbers is lost the
moment someone keeps only the numbers. measure() now follows the leaf counters'
rule: a mask over half the frame is refused unless it is plant green, and a
subject that really fills the frame is measured only when the caller says so
with `allow_implausible_coverage`.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from plantcv_mcp.server import (
    MeasureRefusedError,
    _measure_impl,
    _refine_impl,
    _segment_impl,
    build_server,
)

FIXTURES = Path(__file__).parent / "fixtures"
TRAY = str(FIXTURES / "multi_specimen.png")
POTS = FIXTURES / "aberystwyth"


def _codes(result: dict) -> list[str]:
    return [w["code"] for w in result["warnings"]]


def _pixels(path: str) -> int:
    height, width = cv2.imread(path).shape[:2]
    return height * width


def test_an_inverted_mask_is_refused_and_the_plant_side_is_measured():
    inverted = _segment_impl(TRAY, "s", "otsu", object_type="dark")
    assert "implausible_coverage" in _codes(inverted)  # the premise
    with pytest.raises(MeasureRefusedError, match="implausible_coverage") as exc:
        _measure_impl(inverted["session_id"])
    # The refusal names the way through.
    assert "allow_implausible_coverage=true" in str(exc.value)

    # Same image, other polarity: measured, and nothing about coverage.
    plants = _segment_impl(TRAY, "s", "otsu", object_type="light")
    out = _measure_impl(plants["session_id"])
    assert 0 < out["traits"]["area"]["value"] < 0.5 * _pixels(TRAY)
    assert "implausible_coverage" not in _codes(out)


def test_the_override_returns_the_traits_with_the_warning_attached():
    inverted = _segment_impl(TRAY, "s", "otsu", object_type="dark")
    out = _measure_impl(inverted["session_id"], allow_implausible_coverage=True)
    assert out["traits"]["area"]["value"] > 0.5 * _pixels(TRAY)
    assert "implausible_coverage" in _codes(out)

    # On a mask under the line the flag changes nothing.
    plants = _segment_impl(TRAY, "s", "otsu", object_type="light")
    sid = plants["session_id"]
    assert _measure_impl(sid, allow_implausible_coverage=True) == _measure_impl(sid)


def test_a_plant_that_fills_its_pot_crop_is_measured_and_says_why():
    """A grown rosette cropped to its pot cell covers 86% of the crop and is not
    inverted. The counters measure it as plant green; so does measure(), and it
    says so. A real soil mask over the line, from another pot, is refused."""
    grown = str(POTS / "tray032_2016-01-17_potcell_r1c5.png")
    seg = _segment_impl(grown, "a", "otsu", object_type="dark", fill_size=200)
    plant = _refine_impl(
        seg["session_id"], [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}]
    )
    out = _measure_impl(plant["session_id"])
    assert out["traits"]["area"]["value"] > 0.5 * _pixels(grown)
    assert {"implausible_coverage", "plant_fills_frame"} <= set(_codes(out))

    pot = str(POTS / "tray032_2015-12-14_potcell_r3c3.png")
    soil = _segment_impl(pot, "a", "otsu", object_type="light", fill_size=200)
    assert "implausible_coverage" in _codes(soil)
    with pytest.raises(MeasureRefusedError, match="implausible_coverage"):
        _measure_impl(soil["session_id"])


@pytest.mark.anyio
async def test_the_refusal_and_the_override_cross_the_real_mcp_layer():
    """Through the server's own tool table: the argument is in the schema, the
    refusal reaches the client with its class and its text, and the override
    returns a structured result."""
    from mcp.server.mcpserver.exceptions import ToolError

    mcp = build_server()
    seg = json.loads(
        (
            await mcp.call_tool(
                "segment",
                {
                    "image_path": TRAY,
                    "channel": "s",
                    "method": "otsu",
                    "object_type": "dark",
                },
            )
        )
        .content[0]
        .text
    )
    with pytest.raises(ToolError, match="MeasureRefusedError: implausible_coverage"):
        await mcp.call_tool("measure", {"session_id": seg["session_id"]})

    allowed = await mcp.call_tool(
        "measure",
        {"session_id": seg["session_id"], "allow_implausible_coverage": True},
    )
    out = allowed.structured_content
    assert out["traits"]["area"]["value"] > 0.5 * _pixels(TRAY)
    assert "implausible_coverage" in [w["code"] for w in out["warnings"]]
