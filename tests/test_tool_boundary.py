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


def _thermal_csv(tmp_path, name="t.csv", frame=None) -> str:
    if frame is None:
        frame = np.full((60, 80), 20.0)
        frame[20:40, 30:50] = 30.0
    path = tmp_path / name
    np.savetxt(path, frame, delimiter=",")
    return str(path)


def _hsi_cube(tmp_path) -> str:
    from test_hyperspectral import _known_cube, _write_cube

    return _write_cube(tmp_path, "cube", _known_cube())


def _text(result) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


# --- M2: a recipe error is one tool error, not N identical per-image refusals


@pytest.mark.parametrize(
    ("bad", "good", "named"),
    [
        ({"px_per_mm": 0}, {"px_per_mm": 2.0}, "px_per_mm"),
        ({"px_per_mm": -1}, {"px_per_mm": 2.0}, "px_per_mm"),
        ({"method": "mean", "ksize": 1}, {"method": "mean", "ksize": 31}, "ksize"),
    ],
)
async def test_measure_images_refuses_a_bad_recipe_value_as_one_error(
    tmp_path, bad, good, named
):
    """px_per_mm and ksize were checked per image, inside the per-image
    try/except, so every image was 'refused' with the same ValueError and the
    call reported isError=false with measured=0 — a failure dressed as a
    result (audit 2026-09-22, M2)."""
    paths = [_plant(tmp_path, "a.png"), _plant(tmp_path, "b.png")]
    base = {"image_paths": paths, "channel": "a", "method": "otsu"}
    async with Client(build_server()) as client:
        ok = await client.call_tool("measure_images", {**base, **good})
        refused = await client.call_tool("measure_images", {**base, **bad})
    # Positive control: the same batch with a valid value measures both images.
    assert not ok.is_error, _text(ok)
    assert ok.structured_content["summary"]["measured"] == 2
    assert refused.is_error, refused.structured_content
    assert named in _text(refused)


# --- bounds on the segmentation recipe (L10 fill_size; the threshold kernel)


async def test_negative_fill_size_is_refused_by_every_segmenter(tmp_path):
    """PlantCV treats fill(size<0) as 'remove nothing', so fill_size=-5 ran,
    recorded fill_size=-5 in the recipe, and meant 0 (audit 2026-09-22, L10).
    Four tools take fill_size; all four refuse, and 0 is still accepted."""
    plant = _plant(tmp_path)
    calls = {
        "segment": {"image_path": plant, "channel": "a", "method": "otsu"},
        "measure_images": {"image_paths": [plant], "channel": "a", "method": "otsu"},
        "segment_thermal": {"path": _thermal_csv(tmp_path), "min_c": 25},
        "segment_hyperspectral": {"envi_path": _hsi_cube(tmp_path)},
    }
    async with Client(build_server()) as client:
        for tool, args in calls.items():
            ok = await client.call_tool(tool, {**args, "fill_size": 0})
            assert not ok.is_error, (tool, _text(ok))
            bad = await client.call_tool(tool, {**args, "fill_size": -5})
            assert bad.is_error, (tool, _text(bad))
            assert "fill_size" in _text(bad), (tool, _text(bad))


async def test_adaptive_threshold_kernel_has_a_ceiling(tmp_path):
    """ksize had a floor (PlantCV's own, 3) and no ceiling: a 'gaussian' block
    of 10^6 took 40 s on a 300 px image and 2^31 overflowed inside OpenCV.
    The same bound applies to segment() and measure_images()."""
    plant = _plant(tmp_path)
    seg = {"image_path": plant, "channel": "a", "method": "mean"}
    batch = {"image_paths": [plant], "channel": "a", "method": "mean"}
    async with Client(build_server()) as client:
        assert not (await client.call_tool("segment", {**seg, "ksize": 31})).is_error
        for tool, args in (("segment", seg), ("measure_images", batch)):
            r = await client.call_tool(tool, {**args, "ksize": 10**6})
            assert r.is_error, (tool, _text(r))
            assert "ksize must be between 3 and 1001" in _text(r), _text(r)
        # The global methods ignore ksize; a default carried along is fine.
        r = await client.call_tool("segment", {**seg, "method": "otsu", "ksize": 10**6})
        assert not r.is_error, _text(r)
