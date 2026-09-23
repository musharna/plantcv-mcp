"""Findings of the 2026-09-22 MCP bug audit, reproduced through the tool boundary.

Every test here drives a real in-process `mcp.client.Client` against
`build_server()`: the argument model, the tool function, the output-schema
validation and the wire serialisation all run. Calling the impl functions
directly is what let the batch output schema drift from what the batch
returned (finding H1) with the whole suite green.
"""

import json

import cv2
import jsonschema
import numpy as np
import pytest
from mcp.client import Client

from plantcv_mcp.server import build_server

pytestmark = pytest.mark.anyio


def _png(path, img) -> str:
    cv2.imwrite(str(path), img)
    return str(path)


def _tray(tmp_path) -> str:
    """Four green discs on a light background, one per cell of a 2x2 grid."""
    img = np.full((400, 400, 3), 240, np.uint8)
    for r in range(2):
        for c in range(2):
            cv2.circle(img, (100 + 200 * c, 100 + 200 * r), 40, (40, 160, 40), -1)
    return _png(tmp_path / "tray.png", img)


def _plant(tmp_path, name="plant.png") -> str:
    img = np.full((300, 300, 3), 240, np.uint8)
    cv2.circle(img, (150, 150), 40, (40, 160, 40), -1)
    return _png(tmp_path / name, img)


async def _tools(client) -> dict:
    return {t.name: t for t in (await client.list_tools()).tools}


# --- H1: the structured channel of measure_images carries what the batch returned


async def test_measure_images_structured_content_is_the_whole_batch_result(tmp_path):
    """The outputSchema was a hand-written TypedDict narrower than batch.py's
    dict, and pydantic drops undeclared keys. A 2x2 grid batch came back on
    the structured channel with `traits: null` and no `regions` — every
    per-plant number gone — while the text channel had them all. Covers every
    optional branch at once: a grid, a duplicate path, and a not_run image
    (max_seconds=0 runs exactly one)."""
    tray = _tray(tmp_path)
    plant = _plant(tmp_path)
    args = {
        "image_paths": [tray, tray, plant],
        "channel": "a",
        "method": "otsu",
        "nrows": 2,
        "ncols": 2,
        "max_seconds": 0,
    }
    async with Client(build_server()) as client:
        result = await client.call_tool("measure_images", args)
        schema = (await _tools(client))["measure_images"].output_schema
    assert not result.is_error, result.content
    text = json.loads(result.content[0].text)
    structured = result.structured_content

    # Positive control: the grid really was measured, per plant, in the text.
    row = text["results"][0]
    assert row["measured"] is True
    assert row["regions_measured"] == 4
    assert [r["traits"]["area"]["value"] > 0 for r in row["regions"]] == [True] * 4
    assert text["summary"]["not_run_paths"] == [plant]
    assert text["summary"]["duplicates_dropped"] == [tray]

    # The finding: the two channels must be the same document.
    assert structured == text
    jsonschema.validate(structured, schema)


async def test_every_structured_tool_publishes_a_closed_output_schema():
    """The class behind H1: an output TypedDict that ignores undeclared keys
    turns producer/schema drift into silent data loss. Every output model now
    forbids extra keys, so a key the producer adds without declaring it fails
    the call loudly instead of vanishing. Checked on the published schema of
    every tool that has one, so a new tool cannot opt out by omission."""
    async with Client(build_server()) as client:
        tools = await _tools(client)
    structured = {n: t.output_schema for n, t in tools.items() if t.output_schema}
    # Positive control: the tools known to return structured output are here.
    assert {"measure_images", "measure", "list_methods"} <= set(structured)
    open_ = sorted(
        n for n, s in structured.items() if s.get("additionalProperties") is not False
    )
    assert open_ == [], f"output schemas that silently drop keys: {open_}"
