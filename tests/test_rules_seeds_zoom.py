"""The 2026-10-18 benchmark fixes: several-channel segmentation (`rules` +
`combine`), the seeded split in count_leaves, zoom_image, and the `traits`
selection on measure / measure_regions.

Each is driven through the real MCP layer (build_server + call_tool), where
the argument model, the wrapper and the wire serialisation all run, and each
known-answer test carries the control that shows its premise: the one-channel
mask that misses the yellow leaf, the full trait table the selection is cut
from, the unseeded watershed that merges the two lobes.
"""

import json

import cv2
import numpy as np
import pytest
from mcp.client import Client

from plantcv_mcp.server import build_server

pytestmark = pytest.mark.anyio

SOIL_BGR = (40, 70, 110)  # brown
GREEN_BGR = (40, 160, 40)
YELLOW_BGR = (40, 200, 220)


def _png(path, img) -> str:
    cv2.imwrite(str(path), img)
    return str(path)


def _yellowed_plant(tmp_path):
    """A green disc and a touching yellow disc on brown soil: one plant whose
    yellow leaf LAB a* alone does not select. Returns (path, true area)."""
    img = np.full((300, 300, 3), SOIL_BGR, np.uint8)
    green = np.zeros((300, 300), np.uint8)
    yellow = np.zeros((300, 300), np.uint8)
    cv2.circle(green, (120, 150), 45, 255, -1)
    cv2.circle(yellow, (190, 150), 35, 255, -1)
    img[green > 0] = GREEN_BGR
    img[yellow > 0] = YELLOW_BGR
    return _png(tmp_path / "yellowed.png", img), int(((green | yellow) > 0).sum())


async def _call(server, tool, args):
    """One call through a real in-process client. Sessions live in the
    server module's store, so they outlive the client that minted them."""
    async with Client(server) as client:
        return await client.call_tool(tool, args)


def _payload(result) -> dict:
    assert not result.is_error, result.content[0].text
    return json.loads(result.content[0].text)


async def test_rules_or_joins_the_masks_one_channel_cannot(tmp_path):
    path, truth = _yellowed_plant(tmp_path)
    mcp = build_server()
    # Control: one channel finds the green leaf only (the premise of the fix).
    one = _payload(
        await _call(
            mcp,
            "segment",
            {
                "image_path": path,
                "channel": "a",
                "method": "otsu",
                "object_type": "dark",
            },
        )
    )
    assert one["largest_area"] < 0.75 * truth
    both = _payload(
        await _call(
            mcp,
            "segment",
            {
                "image_path": path,
                "rules": [
                    {"channel": "a", "method": "otsu", "object_type": "dark"},
                    {"channel": "b", "method": "otsu", "object_type": "light"},
                ],
                "combine": "or",
            },
        )
    )
    assert both["largest_area"] == truth
    assert both["component_count"] == 1
    assert len(both["rule_coverage"]) == 2
    assert "channel" not in both and both["combine"] == "or"
    area = _payload(
        await _call(
            mcp, "measure", {"session_id": both["session_id"], "traits": ["area"]}
        )
    )
    assert area["traits"] == {"area": {"value": float(truth), "unit": "pixels"}}


async def test_rules_and_intersects_explicit_inclusive_ranges(tmp_path):
    """A hue range AND a saturation floor select the yellow disc alone. The hue
    range is the yellow's own OpenCV hue at both ends (min = max = 27), so an
    exclusive bound would select nothing: the range is inclusive."""
    path, _ = _yellowed_plant(tmp_path)
    yellow = np.zeros((300, 300), np.uint8)
    cv2.circle(yellow, (190, 150), 35, 255, -1)
    hsv = cv2.cvtColor(np.uint8([[YELLOW_BGR]]), cv2.COLOR_BGR2HSV)[0, 0]
    assert int(hsv[0]) == 27  # the premise: OpenCV hue of the yellow
    rules = [{"channel": "h", "min": 27, "max": 27}, {"channel": "s", "min": 150}]
    mcp = build_server()
    res = _payload(
        await _call(
            mcp,
            "segment",
            {"image_path": path, "rules": rules, "combine": "and", "fill_size": 0},
        )
    )
    assert res["largest_area"] == int((yellow > 0).sum())
    assert res["component_count"] == 1
    # The same two rules OR-ed also take the soil (saturation ~163): combine
    # is applied, not ignored.
    union = _payload(
        await _call(
            mcp,
            "segment",
            {"image_path": path, "rules": rules, "combine": "or", "fill_size": 0},
        )
    )
    assert union["mask_fraction"] > 0.5 > res["mask_fraction"]


@pytest.mark.parametrize(
    ("args", "needle"),
    [
        (
            {"rules": [{"channel": "a", "min": 10}, {"channel": "b", "min": 140}]},
            "combine is required",
        ),
        (
            {"rules": [{"channel": "a", "min": 10, "method": "otsu"}], "combine": "or"},
            "not both",
        ),
        ({"rules": [{"channel": "a", "min": 0, "max": 255}]}, "whole"),
        ({"rules": [{"channel": "h", "min": 0, "max": 200}]}, "degrees / 2"),
        ({"rules": [{"channel": "a", "min": True}]}, "integer"),
        ({"rules": [{"channel": "a", "method": "otsu"}]}, "object_type"),
        (
            {"rules": [{"channel": "a", "min": 10}], "channel": "a"},
            "cannot be combined",
        ),
        ({"rules": [{"channel": "a", "min": 10}], "combine": "xor"}, "combine must be"),
        ({"channel": "a", "method": "otsu", "combine": "or"}, "rules only"),
        ({}, "Give channel and method"),
    ],
)
async def test_malformed_rules_are_refused_by_name(tmp_path, args, needle):
    path, _ = _yellowed_plant(tmp_path)
    mcp = build_server()
    result = await _call(mcp, "segment", {"image_path": path, **args})
    assert result.is_error and needle in result.content[0].text, result.content[0].text
    # Positive control in the same server: a well-formed rule still segments.
    ok = _payload(
        await _call(
            mcp,
            "segment",
            {
                "image_path": path,
                "rules": [{"channel": "a", "method": "otsu", "object_type": "dark"}],
            },
        )
    )
    assert ok["largest_area"] > 0


def _dumbbell(tmp_path):
    """Two discs joined by a wide neck: one blob to the distance watershed at a
    large min_distance, two objects to the eye."""
    img = np.full((240, 400, 3), 235, np.uint8)
    cv2.circle(img, (130, 120), 60, GREEN_BGR, -1)
    cv2.circle(img, (270, 120), 60, GREEN_BGR, -1)
    cv2.rectangle(img, (130, 85), (270, 155), GREEN_BGR, -1)
    return _png(tmp_path / "dumbbell.png", img)


async def test_seeds_split_touching_objects_one_region_per_seed(tmp_path):
    path = _dumbbell(tmp_path)
    mcp = build_server()
    seg = _payload(
        await _call(
            mcp, "segment", {"image_path": path, "channel": "a", "method": "otsu"}
        )
    )
    sid = seg["session_id"]
    # Control: the automatic split with min_distance past the 140 px between
    # the two distance peaks sees one object.
    auto = _payload(
        await _call(mcp, "count_leaves", {"session_id": sid, "min_distance": 150})
    )
    assert auto["leaf_count"] == 1
    # One seed near the left rim, one at the right disc's centre. A plain
    # (non-compact) watershed floods from each seed's own landscape value, so
    # the rim seed starts after the centre seed has taken everything: measured
    # 11 px against 24,524.
    res = _payload(
        await _call(
            mcp, "count_leaves", {"session_id": sid, "seeds": [[80, 120], [270, 120]]}
        )
    )
    assert res["method"] == "seeded_distance_watershed"
    assert res["leaf_count"] == 2
    areas = [i["area_px"] for i in res["instances"]]
    assert sum(areas) == seg["largest_area"]
    assert min(areas) > 0.35 * seg["largest_area"]
    # The split falls in the middle third, between the two discs.
    for inst, side in zip(res["instances"], ("left", "right"), strict=True):
        x, _, w, _ = inst["bbox"]
        assert (x + w <= 240) if side == "left" else (x >= 160)
    assert [w["code"] for w in res["warnings"]] == ["seeded_count"]


async def test_bad_seeds_are_refused_and_good_ones_still_split(tmp_path):
    path = _dumbbell(tmp_path)
    mcp = build_server()
    sid = _payload(
        await _call(
            mcp, "segment", {"image_path": path, "channel": "a", "method": "otsu"}
        )
    )["session_id"]
    for seeds, needle in (
        ([[5, 5]], "not on the mask"),
        ([[900, 5]], "outside"),
        ([[130, 120], [130, 120]], "repeats"),
        ([[130.5, 120]], "integer"),
        ([], "empty"),
    ):
        bad = await _call(mcp, "count_leaves", {"session_id": sid, "seeds": seeds})
        assert bad.is_error and needle in bad.content[0].text, bad.content[0].text
    both = await _call(
        mcp,
        "count_leaves",
        {"session_id": sid, "seeds": [[130, 120]], "min_distance": 5},
    )
    assert both.is_error and "markers are the seeds" in both.content[0].text
    good = _payload(
        await _call(
            mcp, "count_leaves", {"session_id": sid, "seeds": [[130, 120], [270, 120]]}
        )
    )
    assert good["leaf_count"] == 2


async def test_zoom_image_returns_the_box_enlarged(tmp_path):
    img = np.zeros((200, 300, 3), np.uint8)
    img[40:60, 100:140] = (255, 255, 255)  # a white bar inside the box
    path = _png(tmp_path / "z.png", img)
    mcp = build_server()
    result = await _call(
        mcp, "zoom_image", {"image_path": path, "x": 90, "y": 30, "w": 64, "h": 40}
    )
    meta = _payload(result)
    assert meta["box"] == [90, 30, 64, 40] and meta["clipped"] is False
    # 1024 // 64 = 16 would fill the edge; the cap is 8.
    assert meta["scale"] == 8
    pic = cv2.imdecode(
        np.frombuffer(__import__("base64").b64decode(result.content[1].data), np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert pic.shape[:2] == (40 * meta["scale"], 64 * meta["scale"])


async def test_zoom_image_clips_and_refuses_by_name(tmp_path):
    path = _png(tmp_path / "z.png", np.zeros((100, 120, 3), np.uint8))
    mcp = build_server()
    clipped = _payload(
        await _call(
            mcp,
            "zoom_image",
            {"image_path": path, "x": 100, "y": 80, "w": 50, "h": 50, "scale": 2},
        )
    )
    assert clipped["box"] == [100, 80, 20, 20] and clipped["clipped"] is True
    assert clipped["scale"] == 2
    outside = await _call(
        mcp, "zoom_image", {"image_path": path, "x": 120, "y": 0, "w": 5, "h": 5}
    )
    assert outside.is_error and "outside" in outside.content[0].text
    big = await _call(
        mcp,
        "zoom_image",
        {"image_path": path, "x": 0, "y": 0, "w": 5, "h": 5, "scale": 9},
    )
    assert big.is_error and "scale" in big.content[0].text


async def test_traits_selection_is_a_subset_of_the_full_table(tmp_path):
    img = np.full((400, 400, 3), 235, np.uint8)
    for r in range(2):
        for c in range(2):
            cv2.circle(img, (100 + 200 * c, 100 + 200 * r), 40, GREEN_BGR, -1)
    path = _png(tmp_path / "tray.png", img)
    mcp = build_server()
    sid = _payload(
        await _call(
            mcp, "segment", {"image_path": path, "channel": "a", "method": "otsu"}
        )
    )["session_id"]
    full = _payload(
        await _call(mcp, "measure_regions", {"session_id": sid, "nrows": 2, "ncols": 2})
    )
    cut = _payload(
        await _call(
            mcp,
            "measure_regions",
            {"session_id": sid, "nrows": 2, "ncols": 2, "traits": ["area", "width"]},
        )
    )
    assert len(full["regions"][0]["traits"]) > 10  # the control: a full table
    for f, c in zip(full["regions"], cut["regions"], strict=True):
        assert c["traits"] == {k: f["traits"][k] for k in ("area", "width")}
    unknown = await _call(
        mcp, "measure", {"session_id": sid, "traits": ["hue_circular_mean"]}
    )
    assert unknown.is_error and "available" in unknown.content[0].text
    ok = _payload(
        await _call(
            mcp,
            "measure",
            {
                "session_id": sid,
                "analyses": ["size", "color"],
                "traits": ["hue_circular_mean"],
            },
        )
    )
    assert list(ok["traits"]) == ["hue_circular_mean"]
