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
    small_path = _write_green_png(tmp_path)  # 200x200, under min_edge=256
    seg_small = _segment_impl(small_path, channel="a", method="otsu")
    assert seg_small["overlay_scale"] == 2.0  # upscaled so it can be looked at

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
        "refine",
        "measure",
        "measure_morphology",
        "measure_regions",
        "calibrate_scale_from_marker",
        "measure_images",
        "segment_hyperspectral",
        "measure_spectral",
        "segment_thermal",
        "measure_thermal",
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


# --- refine: a session-to-session operation that returns the picture ---


@pytest.mark.anyio
async def test_refine_mints_a_new_session_with_overlay_and_lineage(tmp_path):
    from plantcv_mcp.server import _refine_impl

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
    result = await mcp.call_tool(
        "refine",
        {"session_id": seg["session_id"], "ops": [{"op": "erode", "ksize": 3}]},
    )
    text_block, image_block = result.content
    payload = json.loads(text_block.text)
    assert image_block.type == "image"
    assert payload["parent_session_id"] == seg["session_id"]
    assert payload["session_id"] != seg["session_id"]
    assert payload["lineage"] == [{"op": "erode", "ksize": 3, "iterations": 1}]
    assert payload["after"]["mask_fraction"] < payload["before"]["mask_fraction"]
    assert payload["engine"]["name"] == "PlantCV"
    assert "traits" not in payload

    # The original session is untouched and still measurable; the refined one
    # measures a smaller plant and SAYS how its mask was made.
    original = await mcp.call_tool("measure", {"session_id": seg["session_id"]})
    refined = await mcp.call_tool("measure", {"session_id": payload["session_id"]})
    assert original.structured_content["lineage"] == []
    assert refined.structured_content["lineage"] == payload["lineage"]
    assert (
        refined.structured_content["traits"]["area"]["value"]
        < original.structured_content["traits"]["area"]["value"]
    )

    # Chained refinements accumulate lineage.
    again = _refine_impl(payload["session_id"], [{"op": "dilate", "ksize": 3}])
    assert again["lineage"] == payload["lineage"] + [
        {"op": "dilate", "ksize": 3, "iterations": 1}
    ]


@pytest.mark.anyio
async def test_refine_that_erases_the_plant_is_a_tool_error_and_mints_nothing(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    from plantcv_mcp.server import _store

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
    n_before = len(_store)
    with pytest.raises(ToolError, match="before"):
        await mcp.call_tool(
            "refine",
            {
                "session_id": seg["session_id"],
                "ops": [{"op": "erode", "ksize": 5, "iterations": 60}],
            },
        )
    assert len(_store) == n_before
    with pytest.raises(ToolError, match="op 0"):
        await mcp.call_tool(
            "refine",
            {"session_id": seg["session_id"], "ops": [{"op": "fill", "size": -1}]},
        )


def test_list_methods_publishes_the_refine_ops():
    info = list_methods_impl()
    assert set(info["refine_ops"]) == {
        "fill_holes",
        "fill",
        "erode",
        "dilate",
        "opening",
        "closing",
        "median_blur",
        "keep_largest",
    }
    assert info["refine_ops"]["erode"]["params"]["ksize"]["min"] == 2


# --- morphology over the real MCP layer ---


@pytest.mark.anyio
async def test_measure_morphology_over_the_real_mcp_layer(tmp_path):
    """Table + numbered overlay, engine and lineage; a vertical stem's angle is
    null with the warning, not a nonsense number."""
    import cv2

    from plantcv_mcp import plantcv_version

    img = np.full((400, 400, 3), 200, np.uint8)
    cv2.line(img, (200, 380), (200, 120), (40, 150, 40), 9)
    cv2.line(img, (200, 300), (260, 220), (40, 150, 40), 7)
    cv2.line(img, (200, 220), (140, 150), (40, 150, 40), 7)
    path = _write_png(tmp_path, "plant.png", img)

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
    result = await mcp.call_tool(
        "measure_morphology", {"session_id": seg["session_id"], "px_per_mm": 10.0}
    )
    text_block, image_block = result.content
    payload = json.loads(text_block.text)
    assert image_block.type == "image"
    assert payload["engine"] == {"name": "PlantCV", "version": plantcv_version()}
    assert payload["lineage"] == []
    assert payload["units"]["path_length"] == "mm"
    assert payload["units"]["insertion_angle"] == "degrees"
    assert payload["plant"]["leaf_count"] >= 2
    assert payload["plant"]["stem_angle"] is None
    assert "stem_angle_undefined" in [w["code"] for w in payload["warnings"]]
    assert all(s["id"] == i for i, s in enumerate(payload["segments"]))


def test_measure_recomputes_and_carries_mask_level_warnings(tmp_path):
    """frame_clipping was reported at segment() time and then dropped: the
    trait table — the artifact people actually keep — could not say its own
    area was a lower bound. measure() re-derives mask-level advisories."""
    import cv2

    from plantcv_mcp.server import _measure_impl, _segment_impl

    img = np.full((200, 200, 3), 128, np.uint8)
    img[0:100, 50:150] = (60, 180, 60)  # plant cut by the top frame edge
    p = str(tmp_path / "clipped.png")
    cv2.imwrite(p, img)
    seg = _segment_impl(p, "a", "otsu")
    assert "frame_clipping" in [w["code"] for w in seg["warnings"]]
    res = _measure_impl(seg["session_id"])
    assert "frame_clipping" in [w["code"] for w in res["warnings"]]

    # Positive control: a fully in-frame plant measures with no warnings.
    img2 = np.full((200, 200, 3), 128, np.uint8)
    img2[50:150, 50:150] = (60, 180, 60)
    p2 = str(tmp_path / "clean.png")
    cv2.imwrite(p2, img2)
    res2 = _measure_impl(_segment_impl(p2, "a", "otsu")["session_id"])
    assert res2["warnings"] == []


def test_list_methods_names_the_server_itself():
    """The engine version alone cannot say WHICH plantcv-mcp answered — found
    live: a stale 1.0.0 server was indistinguishable from current over the
    tool surface."""
    from plantcv_mcp import __version__
    from plantcv_mcp.server import list_methods_impl

    assert list_methods_impl()["server_version"] == __version__


@pytest.mark.anyio
async def test_grayscale_image_is_refused_by_name_everywhere(tmp_path):
    """A 1-channel PNG used to die three different ways (raw cvtColor traceback
    from segment() and the batch, PlantCV's bare 'not RGB' from suggest). The
    guard lives at the one load boundary, so every RGB tool now says the same
    thing and names segment_thermal() as the likely intent."""
    import cv2
    from mcp.server.mcpserver.exceptions import ToolError

    gray = tmp_path / "gray.png"
    cv2.imwrite(str(gray), np.full((40, 60), 90, dtype=np.uint8))
    mcp = build_server()
    for name, args in [
        ("segment", {"image_path": str(gray), "channel": "a", "method": "otsu"}),
        ("suggest_segmentation", {"image_path": str(gray)}),
    ]:
        with pytest.raises(ToolError, match="1 channel.*segment_thermal"):
            await mcp.call_tool(name, args)
    result = await mcp.call_tool(
        "measure_images",
        {"image_paths": [str(gray)], "channel": "a", "method": "otsu"},
    )
    entry = json.loads(result.content[0].text)["results"][0]
    assert entry["measured"] is False
    assert "1 channel" in entry["refused_because"]
    assert "cvtColor" not in entry["refused_because"]


@pytest.mark.anyio
async def test_marker_calibration_round_trip_yields_millimetres_from_a_real_frame():
    """The one path a user actually walks with calibrate_scale_from_marker, on a
    CHECKED-IN frame that holds both a plant and a marker (the earlier positive
    control used a temp disc alone): calibrate -> segment -> measure(px_per_mm)
    must land on the plant's known physical size, and the marker must not be
    counted as plant. Geometry is in tests/fixtures/make_plant_with_marker.py:
    a 200x100 px green rectangle and a 100 px black disc that is 20 mm, so
    5 px/mm and the plant is 40 x 20 mm, 800 mm2."""
    import importlib.util
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    spec = importlib.util.spec_from_file_location(
        "make_plant_with_marker", fixtures / "make_plant_with_marker.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    path = str(fixtures / "plant_with_marker.png")
    x, y, w, h = gen.MARKER_CROP
    mcp = build_server()

    cal = (
        await mcp.call_tool(
            "calibrate_scale_from_marker",
            {
                "image_path": path,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "marker_length_mm": gen.MARKER_LENGTH_MM,
            },
        )
    ).structured_content
    assert cal["marker_length_px"] == 100
    assert cal["px_per_mm"] == pytest.approx(5.0, rel=1e-3)
    assert cal["warnings"] == []

    # Negative control inside the same test: the wrong polarity on the same crop
    # selects the background and is flagged, so a passing calibration above is
    # evidence and not a harness that accepts anything.
    bad = (
        await mcp.call_tool(
            "calibrate_scale_from_marker",
            {
                "image_path": path,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "marker_length_mm": gen.MARKER_LENGTH_MM,
                "object_type": "light",
            },
        )
    ).structured_content
    assert "marker_touches_crop_edge" in [a["code"] for a in bad["warnings"]]

    seg = json.loads(
        (
            await mcp.call_tool(
                "segment",
                {"image_path": path, "channel": "a", "method": "otsu"},
            )
        )
        .content[0]
        .text
    )
    # Exactly the rectangle: the black marker in the same frame is not plant.
    assert seg["component_count"] == 1
    assert seg["largest_area"] == 200 * 100
    assert seg["warnings"] == []

    mm = (
        await mcp.call_tool(
            "measure", {"session_id": seg["session_id"], "px_per_mm": cal["px_per_mm"]}
        )
    ).structured_content
    t = mm["traits"]
    assert t["area"]["unit"] == "mm2"
    assert t["area"]["value"] == pytest.approx(800.0, rel=1e-3)
    assert t["width"]["value"] == pytest.approx(40.0, rel=1e-3)
    assert t["height"]["value"] == pytest.approx(20.0, rel=1e-3)


@pytest.mark.anyio
async def test_measure_images_takes_grid_geometry_and_a_time_budget():
    mcp = build_server()
    result = await mcp.call_tool(
        "measure_images",
        {
            "image_paths": ["tests/fixtures/multi_specimen.png"],
            "channel": "a",
            "method": "otsu",
            "nrows": 2,
            "ncols": 2,
            "max_seconds": 120,
        },
    )
    payload = json.loads(result.content[0].text)
    assert payload["results"][0]["regions_measured"] == 4
    assert payload["elapsed_s"] > 0
    assert payload["summary"]["not_run"] == 0


@pytest.mark.anyio
async def test_an_explicit_empty_analyses_list_is_refused_not_defaulted(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    path = _write_multi_specimen_png(tmp_path)
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
    with pytest.raises(ToolError, match="No analyses"):
        await mcp.call_tool(
            "measure", {"session_id": seg["session_id"], "analyses": []}
        )
    with pytest.raises(ToolError, match="No analyses"):
        await mcp.call_tool(
            "measure_regions",
            {"session_id": seg["session_id"], "nrows": 2, "ncols": 2, "analyses": []},
        )
    with pytest.raises(ToolError, match="No analyses"):
        await mcp.call_tool(
            "measure_images",
            {"image_paths": [path], "channel": "a", "method": "otsu", "analyses": []},
        )
    with pytest.raises(ToolError, match="indices"):
        await mcp.call_tool(
            "measure_regions",
            {
                "session_id": seg["session_id"],
                "nrows": 2,
                "ncols": 2,
                "indices": ["ndvi"],
            },
        )
