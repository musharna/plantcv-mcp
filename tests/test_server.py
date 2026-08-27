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
    # .content since mcp 2.x: call_tool returns a CallToolResult, not the bare
    # block sequence that could be indexed directly.
    payload = json.loads(result.content[0].text)
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
        "measure_regions",
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
    text_block, image_block = result.content
    payload = json.loads(text_block.text)
    assert "traits" not in payload
    assert payload["session_id"]
    assert payload["largest_area"] == 10000
    assert image_block.type == "image"


# --- the digest must describe the bytes the mask was drawn on ---


def test_file_swapped_during_segmentation_is_refused_at_measure(tmp_path, monkeypatch):
    """TOCTOU in the stale-image guard.

    segment() used to decode the file, build the mask, and only THEN hash the
    path. A same-shape replacement landing in that window bound the OLD mask to
    the NEW file's hash, and measure() then passed the integrity check while
    measuring pixels the mask was never drawn on. The hash has to be taken from
    the bytes that were decoded, not from the path afterwards.
    """
    import cv2

    from plantcv_mcp import server as srv
    from plantcv_mcp.server import (
        ImageChangedSinceSegmentationError,
        _load_session_image,
    )

    path = _write_green_png(tmp_path)

    # A different image of IDENTICAL shape: the blob moved.
    other = np.full((200, 200, 3), 128, dtype=np.uint8)
    other[100:190, 100:190] = (60, 180, 60)

    real_threshold = srv.threshold_mask

    def swap_file_then_threshold(*args, **kwargs):
        cv2.imwrite(path, other)  # the file changes while segment() is mid-flight
        return real_threshold(*args, **kwargs)

    monkeypatch.setattr(srv, "threshold_mask", swap_file_then_threshold)
    seg = _segment_impl(path, "a", "otsu")
    monkeypatch.setattr(srv, "threshold_mask", real_threshold)

    session = srv._store.get(seg["session_id"])
    with pytest.raises(ImageChangedSinceSegmentationError, match="CONTENT"):
        _load_session_image(session)

    # Positive control: an undisturbed session on the (now settled) file loads.
    settled = _segment_impl(path, "a", "otsu")
    assert _load_session_image(srv._store.get(settled["session_id"])).shape == (
        200,
        200,
        3,
    )


# --- batch recipe parity: what you settled interactively is what the batch runs ---


def _write_png(tmp_path, name, img):
    import cv2

    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return str(p)


@pytest.mark.anyio
async def test_measure_images_reproduces_an_interactive_mean_threshold_recipe(tmp_path):
    """measure_images tells the user to settle a recipe with segment() and then
    apply it. For the 'mean'/'gaussian' methods the recipe INCLUDES ksize and
    offset; a batch that cannot take them silently runs a different threshold
    and the two paths disagree on the same file."""
    path = _write_green_png(tmp_path)
    recipe = {"channel": "a", "method": "mean", "ksize": 31, "offset": 5}

    seg = _segment_impl(path, **recipe)
    interactive = _measure_impl(seg["session_id"])["traits"]["area"]["value"]

    mcp = build_server()
    result = await mcp.call_tool("measure_images", {"image_paths": [path], **recipe})
    out = result.structured_content
    entry = out["results"][0]
    assert entry["measured"] is True, entry
    assert entry["traits"]["area"]["value"] == interactive
    assert out["recipe"]["ksize"] == 31
    assert out["recipe"]["offset"] == 5
    assert out["recipe"]["color_correct"] is False


@pytest.mark.anyio
async def test_measure_images_honours_color_correct_and_refuses_cardless_images(
    tmp_path,
):
    from test_scale_color_batch import _color_card  # pytest puts tests/ on sys.path

    card = _color_card()
    card[380:500, 400:560] = (40, 150, 40)  # a plant below the chips
    with_card = _write_png(tmp_path, "card.png", card)
    no_card = _write_green_png(tmp_path)

    seg = _segment_impl(with_card, "a", "otsu", color_correct=True)
    interactive = _measure_impl(seg["session_id"], analyses=["size", "color"])["traits"]

    mcp = build_server()
    result = await mcp.call_tool(
        "measure_images",
        {
            "image_paths": [with_card, no_card],
            "channel": "a",
            "method": "otsu",
            "color_correct": True,
            "analyses": ["size", "color"],
        },
    )
    out = result.structured_content
    by_path = {r["image_path"]: r for r in out["results"]}

    corrected = by_path[with_card]
    assert corrected["measured"] is True, corrected
    assert corrected["traits"]["hue_circular_mean"] == interactive["hue_circular_mean"]
    assert corrected["traits"]["area"] == interactive["area"]
    assert out["recipe"]["color_correct"] is True

    # Asked to correct, unable to: refused with the reason, never measured raw.
    refused = by_path[no_card]
    assert refused["measured"] is False
    assert refused["traits"] is None
    assert "ColorCardNotFoundError" in refused["refused_because"]


# --- provenance: every number names the engine; the server names its version ---


@pytest.mark.anyio
async def test_every_number_bearing_result_names_the_plantcv_engine(tmp_path):
    from plantcv_mcp import plantcv_version

    path = _write_green_png(tmp_path)
    mcp = build_server()
    seg = json.loads(
        (
            await mcp.call_tool(
                "segment", {"image_path": path, "channel": "a", "method": "otsu"}
            )
        )
        .content[0]
        .text
    )
    expected = {"name": "PlantCV", "version": plantcv_version()}

    measured = await mcp.call_tool("measure", {"session_id": seg["session_id"]})
    assert measured.structured_content["engine"] == expected

    regions = await mcp.call_tool(
        "measure_regions",
        {
            "session_id": seg["session_id"],
            "mode": "rect_grid",  # auto_grid needs >= 2 plants to fit its mixture
            "nrows": 1,
            "ncols": 1,
            "coord": [40, 40],
            "height": 120,
            "width": 120,
            "spacing": [0, 0],
        },
    )
    assert json.loads(regions.content[0].text)["engine"] == expected

    batch = await mcp.call_tool(
        "measure_images", {"image_paths": [path], "channel": "a", "method": "otsu"}
    )
    assert batch.structured_content["engine"] == expected


@pytest.mark.anyio
async def test_initialize_advertises_the_package_version():
    """MCPServer defaults `version` to "", and that empty string is what a client
    sees in the initialize handshake. Checked through a real client session,
    not by reading the attribute back."""
    from mcp.client import Client

    from plantcv_mcp import __version__

    async with Client(build_server()) as client:
        info = client.server_info
    assert info is not None, "server did not identify itself at initialize"
    assert info.name == "plantcv-mcp"
    assert info.version == __version__


# --- the remaining tools, driven through the real MCP layer ---


@pytest.mark.anyio
async def test_calibrate_scale_from_marker_over_the_real_mcp_layer(tmp_path):
    import cv2

    img = np.full((300, 300, 3), 240, np.uint8)
    cv2.circle(img, (150, 150), 40, (30, 30, 30), -1)  # an 80 px disc
    path = _write_png(tmp_path, "marker.png", img)

    mcp = build_server()
    result = await mcp.call_tool(
        "calibrate_scale_from_marker",
        {
            "image_path": path,
            "x": 100,
            "y": 100,
            "w": 100,
            "h": 100,
            "marker_length_mm": 20.0,
        },
    )
    out = result.structured_content
    assert out["px_per_mm"] == pytest.approx(4.0, rel=0.03)
    assert out["marker_length_px"] == pytest.approx(80, abs=2)
    assert out["warnings"] == []


@pytest.mark.anyio
async def test_off_image_region_geometry_is_a_tool_error_not_a_crash(tmp_path):
    """The wire schema accepts any integers. This exact geometry used to reach
    native code and SIGSEGV the whole stdio server."""
    from mcp.server.mcpserver.exceptions import ToolError

    path = _write_green_png(tmp_path)
    mcp = build_server()
    seg = json.loads(
        (
            await mcp.call_tool(
                "segment", {"image_path": path, "channel": "a", "method": "otsu"}
            )
        )
        .content[0]
        .text
    )
    with pytest.raises(ToolError, match="outside"):
        await mcp.call_tool(
            "measure_regions",
            {
                "session_id": seg["session_id"],
                "mode": "rect_grid",
                "nrows": 1,
                "ncols": 1,
                "coord": [-10, -10],
                "height": 50,
                "width": 50,
                "spacing": [0, 0],
            },
        )
    # Positive control: a well-formed rect_grid on the same session measures.
    ok = await mcp.call_tool(
        "measure_regions",
        {
            "session_id": seg["session_id"],
            "mode": "rect_grid",
            "nrows": 1,
            "ncols": 1,
            "coord": [40, 40],
            "height": 120,
            "width": 120,
            "spacing": [0, 0],
        },
    )
    assert json.loads(ok.content[0].text)["regions_measured"] == 1
