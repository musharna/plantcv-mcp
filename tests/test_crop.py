"""crop_image: one plant of a tray as its own file (field benchmark F2).

No tool took a region, so in the 2026-10-24 field benchmark every server run
counted one pot's leaves by eye: count_leaves and segment_leaves_sam work on
the whole frame's mask, and the frame held 20 plants. The crop is a file every
other tool already accepts. Driven through the real MCP layer, each refusal
beside the call that succeeds.
"""

import json
import os

import cv2
import numpy as np
import pytest
from mcp.client import Client

from plantcv_mcp.paths import set_roots
from plantcv_mcp.server import build_server

pytestmark = pytest.mark.anyio

SOIL_BGR = (40, 70, 110)
GREEN_BGR = (40, 160, 40)


async def _call(server, tool, args):
    async with Client(server) as client:
        return await client.call_tool(tool, args)


def _payload(result) -> dict:
    assert not result.is_error, result.content[0].text
    return json.loads(result.content[0].text)


def _two_plants(tmp_path):
    """Two green discs on soil, the right one larger. Returns the path and
    each disc's own pixel count."""
    img = np.full((240, 400, 3), SOIL_BGR, np.uint8)
    left = np.zeros(img.shape[:2], np.uint8)
    right = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(left, (100, 120), 40, 255, -1)
    cv2.circle(right, (290, 120), 60, 255, -1)
    img[(left | right) > 0] = GREEN_BGR
    path = tmp_path / "tray.png"
    cv2.imwrite(str(path), img)
    return str(path), int((left > 0).sum()), int((right > 0).sum())


async def test_crop_image_gives_the_other_tools_one_plant(tmp_path):
    path, left_px, right_px = _two_plants(tmp_path)
    mcp = build_server()
    seg_args = {"channel": "a", "method": "otsu", "object_type": "dark"}

    # Control: on the whole frame the mask is both plants, and the largest
    # object is the OTHER one.
    whole = _payload(await _call(mcp, "segment", {"image_path": path, **seg_args}))
    assert whole["component_count"] == 2
    assert whole["largest_area"] == right_px

    crop = _payload(
        await _call(
            mcp,
            "crop_image",
            {"image_path": path, "x": 40, "y": 50, "w": 120, "h": 140},
        )
    )
    assert crop["box"] == [40, 50, 120, 140] and crop["clipped"] is False
    out = crop["cropped_image_path"]
    assert out == os.path.join(str(tmp_path), "tray_crop_x40_y50_w120_h140.png")
    # The image's own pixels, lossless.
    np.testing.assert_array_equal(cv2.imread(out), cv2.imread(path)[50:190, 40:160])

    one = _payload(await _call(mcp, "segment", {"image_path": out, **seg_args}))
    assert one["component_count"] == 1
    assert one["largest_area"] == left_px


async def test_crop_image_owns_its_derived_name_but_not_an_output_path(tmp_path):
    path, _, _ = _two_plants(tmp_path)
    mcp = build_server()
    box = {"image_path": path, "x": 0, "y": 0, "w": 50, "h": 50}

    first = _payload(await _call(mcp, "crop_image", box))
    again = _payload(await _call(mcp, "crop_image", box))  # same box: overwritten
    assert first["cropped_image_path"] == again["cropped_image_path"]

    target = tmp_path / "pot.png"
    made = _payload(await _call(mcp, "crop_image", {**box, "output_path": str(target)}))
    assert made["cropped_image_path"] == str(target) and target.is_file()

    taken = await _call(mcp, "crop_image", {**box, "output_path": str(target)})
    assert taken.is_error and "already exists" in taken.content[0].text

    jpg = await _call(
        mcp, "crop_image", {**box, "output_path": str(tmp_path / "p.jpg")}
    )
    assert jpg.is_error and ".png" in jpg.content[0].text
    assert not (tmp_path / "p.jpg").exists()

    victim = tmp_path / "victim.png"
    victim.write_bytes(b"keep me")
    link = tmp_path / "link.png"
    link.symlink_to(victim)
    linked = await _call(mcp, "crop_image", {**box, "output_path": str(link)})
    assert linked.is_error and "is a symlink" in linked.content[0].text
    assert victim.read_bytes() == b"keep me"


async def test_crop_image_clips_and_refuses_a_box_outside(tmp_path):
    path, _, _ = _two_plants(tmp_path)  # 400 x 240
    mcp = build_server()
    clipped = _payload(
        await _call(
            mcp,
            "crop_image",
            {"image_path": path, "x": 350, "y": 200, "w": 100, "h": 100},
        )
    )
    assert clipped["box"] == [350, 200, 50, 40] and clipped["clipped"] is True
    assert cv2.imread(clipped["cropped_image_path"]).shape[:2] == (40, 50)

    outside = await _call(
        mcp, "crop_image", {"image_path": path, "x": 400, "y": 0, "w": 5, "h": 5}
    )
    assert outside.is_error and "outside" in outside.content[0].text


async def test_crop_image_writes_only_inside_the_read_roots(tmp_path):
    inside = tmp_path / "in"
    outside = tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    path, _, _ = _two_plants(inside)
    box = {"image_path": path, "x": 0, "y": 0, "w": 50, "h": 50}
    mcp = build_server()
    set_roots([str(inside)])
    try:
        refused = await _call(
            mcp, "crop_image", {**box, "output_path": str(outside / "c.png")}
        )
        assert (
            refused.is_error
            and "outside the configured read" in refused.content[0].text
        )
        assert list(outside.iterdir()) == []
        ok = _payload(
            await _call(
                mcp, "crop_image", {**box, "output_path": str(inside / "c.png")}
            )
        )
        assert os.path.isfile(ok["cropped_image_path"])
    finally:
        set_roots(None)
