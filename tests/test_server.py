import json

import numpy as np
import pytest

from plantcv_mcp.server import (
    _measure_impl,
    _segment_impl,
    build_server,
    list_methods_impl,
)


def _write_green_png(tmp_path):
    import cv2

    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[50:150, 50:150] = (60, 180, 60)
    p = tmp_path / "green.png"
    cv2.imwrite(str(p), img)
    return str(p)


def _write_multi_specimen_png(tmp_path):
    """Two disjoint, comparably-sized (40000px each) green blobs on a light
    background -> component_count=2, major_object_count=2."""
    import cv2

    img = np.full((400, 900, 3), 200, dtype=np.uint8)
    img[50:250, 50:250] = (60, 180, 60)
    img[50:250, 650:850] = (60, 180, 60)
    p = tmp_path / "multi.png"
    cv2.imwrite(str(p), img)
    return str(p)


def _write_clipped_png(tmp_path):
    """A green blob touching the top and left frame edges."""
    import cv2

    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[0:100, 0:100] = (60, 180, 60)
    p = tmp_path / "clipped.png"
    cv2.imwrite(str(p), img)
    return str(p)


def _write_large_green_png(tmp_path):
    """A 2000x2000 image, well over the default max_edge=1024, with an
    interior (non-edge-touching) green blob."""
    import cv2

    img = np.full((2000, 2000, 3), 128, dtype=np.uint8)
    img[500:1500, 500:1500] = (60, 180, 60)
    p = tmp_path / "large.png"
    cv2.imwrite(str(p), img)
    return str(p)


def _write_huge_green_png(tmp_path):
    """A 3000x3000 image with an interior green blob. Big enough that
    colorspace_sheet's contact-sheet grid (which is larger than the input
    itself) also clears max_edge=1024, exercising a second downscale()
    call with a different scale than the overlay's."""
    import cv2

    img = np.full((3000, 3000, 3), 128, dtype=np.uint8)
    img[500:2500, 500:2500] = (60, 180, 60)
    p = tmp_path / "huge.png"
    cv2.imwrite(str(p), img)
    return str(p)


def test_list_methods_reports_pinned_plantcv_version():
    info = list_methods_impl()
    assert info["plantcv_version"] == "4.11.3"
    assert "a" in info["channels"]
    assert "otsu" in info["methods"]


def test_segment_returns_no_traits_and_measure_needs_its_session(tmp_path):
    """The load-bearing API constraint: traits are unreachable without first
    receiving a segmentation overlay."""
    path = _write_green_png(tmp_path)
    seg = _segment_impl(path, channel="a", method="otsu")
    assert "traits" not in seg
    assert seg["overlay_png_bytes"] > 0
    assert "session_id" in seg
    # Deterministic geometry: a 100x100 green square on a 200x200 frame.
    assert seg["largest_area"] == 10000
    assert seg["mask_fraction"] == pytest.approx(0.25)

    traits = _measure_impl(seg["session_id"])
    assert traits["traits"]["area"]["value"] > 0


def test_measure_rejects_an_unknown_session_id():
    with pytest.raises(Exception) as exc:
        _measure_impl("not-a-real-session")
    assert "not-a-real-session" in str(exc.value)


def test_segment_warnings_fire_for_real_failures_and_not_for_a_clean_plant(tmp_path):
    """Diagnostics warnings (Tasks 3-4) first reach a user through segment()'s
    response here. If the warning list-comprehension in _segment_impl is ever
    dropped, this must go red rather than silently stop surfacing them."""
    multi_path = _write_multi_specimen_png(tmp_path)
    seg_multi = _segment_impl(multi_path, channel="a", method="otsu")
    codes_multi = {w["code"] for w in seg_multi["warnings"]}
    assert "multi_specimen" in codes_multi

    clipped_path = _write_clipped_png(tmp_path)
    seg_clipped = _segment_impl(clipped_path, channel="a", method="otsu")
    codes_clipped = {w["code"] for w in seg_clipped["warnings"]}
    assert "frame_clipping" in codes_clipped

    # POSITIVE CONTROL, same test: a clean, centered, single plant must fire
    # NEITHER warning -- otherwise an always-warn bug would pass the two
    # assertions above undetected.
    clean_path = _write_green_png(tmp_path)
    seg_clean = _segment_impl(clean_path, channel="a", method="otsu")
    assert seg_clean["warnings"] == []
    assert seg_clean["largest_area"] == 10000
    assert seg_clean["mask_fraction"] == pytest.approx(0.25)


def test_segment_reports_real_overlay_scale(tmp_path):
    """overlay_scale must reflect the actual downscale() call, not a stub 1.0."""
    small_path = _write_green_png(tmp_path)  # 200x200, under max_edge=1024
    seg_small = _segment_impl(small_path, channel="a", method="otsu")
    assert seg_small["overlay_scale"] == 1.0

    large_path = _write_large_green_png(tmp_path)  # 2000x2000
    seg_large = _segment_impl(large_path, channel="a", method="otsu")
    assert seg_large["overlay_scale"] == pytest.approx(1024 / 2000)


def test_measure_raises_when_the_image_changed_shape_since_segmentation(tmp_path):
    """Session.shape is stored but, before this fix, nothing read it. The final
    review proved the gap: replace the file at the segmented path with a
    differently-sized image between segment() and measure(), and measure()
    silently succeeded -- traits computed from a stale mask, attributed to a
    file whose current content was never segmented and whose overlay nobody
    saw. This is the project's own thesis failure class at the one seam the
    design left open."""
    import cv2

    from plantcv_mcp.server import ImageChangedSinceSegmentationError

    path = _write_green_png(tmp_path)  # 200x200
    seg = _segment_impl(path, channel="a", method="otsu")

    # Replace the file, same path, with a differently-sized image.
    bigger = np.full((400, 400, 3), 128, dtype=np.uint8)
    bigger[100:300, 100:300] = (60, 180, 60)
    cv2.imwrite(path, bigger)

    with pytest.raises(ImageChangedSinceSegmentationError) as exc:
        _measure_impl(seg["session_id"])
    msg = str(exc.value)
    assert "(200, 200)" in msg  # shape at segment() time
    assert "(400, 400)" in msg  # shape now

    # POSITIVE CONTROL, same test: a session whose file was never touched
    # after segment() must still measure fine -- otherwise an always-raises
    # bug would masquerade as a working guard.
    control_img = np.full((200, 200, 3), 128, dtype=np.uint8)
    control_img[50:150, 50:150] = (60, 180, 60)
    control_path = str(tmp_path / "unchanged.png")
    cv2.imwrite(control_path, control_img)
    seg_control = _segment_impl(control_path, channel="a", method="otsu")
    traits = _measure_impl(seg_control["session_id"])
    assert traits["traits"]["area"]["value"] > 0


@pytest.mark.anyio
async def test_suggest_segmentation_reports_downscale_factors(tmp_path):
    """Design spec S5: any downsampling must be reported, never silent.
    server.py previously did `cs, _ = downscale(...)` / `th, _ = downscale(...)`
    for suggest_segmentation, discarding both scale factors -- the one tool
    whose entire purpose is making the channel/method choice informed. Values
    independently reproduced before writing this test: for a 3000x3000 input,
    colorspace_sheet's contact-sheet grid is (4125, 9000, 3) -> scale
    1024/9000 = 0.11377..., and threshold_sheet is (3000, 3000, 3) ->
    scale 1024/3000 = 0.34133...
    """
    path = _write_huge_green_png(tmp_path)
    mcp = build_server()
    result = await mcp.call_tool(
        "suggest_segmentation", {"image_path": path, "channel": "a"}
    )
    payload = json.loads(result[0].text)
    assert payload["colorspace_sheet_scale"] == pytest.approx(1024 / 9000)
    assert payload["threshold_sheet_scale"] == pytest.approx(1024 / 3000)


@pytest.mark.anyio
async def test_server_registers_exactly_the_expected_tools():
    """No test previously referenced build_server/list_tools/call_tool -- every
    server test drove the private _impl helpers instead. Dropping a
    @mcp.tool() decorator would still pass all of those. Go through the real
    FastMCP registration."""
    mcp = build_server()
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "list_methods",
        "suggest_segmentation",
        "segment",
        "measure",
        "calibrate_scale_from_marker",
        "measure_images",
    }


@pytest.mark.anyio
async def test_segment_tool_carries_no_traits_through_the_real_mcp_layer(tmp_path):
    """The load-bearing assertion `assert "traits" not in seg` elsewhere in this
    file checks the HELPER's dict, not the actual tool payload a client
    receives. Drive it through build_server() + call_tool("segment", ...) so
    a `traits` key added inside the segment() WRAPPER (rather than
    _segment_impl) would be caught."""
    path = _write_green_png(tmp_path)
    mcp = build_server()
    result = await mcp.call_tool(
        "segment", {"image_path": path, "channel": "a", "method": "otsu"}
    )
    text_block, image_block = result
    payload = json.loads(text_block.text)
    assert "traits" not in payload
    assert payload["session_id"]
    assert payload["largest_area"] == 10000
    assert image_block.type == "image"
