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


# --- M7 + #117: refine kernels bounded by the mask they act on


async def test_refine_kernel_reach_is_bounded_by_the_mask(tmp_path):
    """REFINE_OPS had a floor on every integer and no ceiling. A 3x3 dilation
    200000 times made OpenCV ask for 160 GB (audit 2026-09-22, M7); an erode
    ksize of 2e9 raised a raw MemoryError (#117). Both are now refused by
    validation, naming the op, before anything runs; the same ops at a sane
    size still refine."""
    plant = _plant(tmp_path)
    async with Client(build_server()) as client:
        seg = await client.call_tool(
            "segment", {"image_path": plant, "channel": "a", "method": "otsu"}
        )
        sid = json.loads(seg.content[0].text)["session_id"]

        async def refine(op):
            return await client.call_tool("refine", {"session_id": sid, "ops": [op]})

        ok = await refine({"op": "dilate", "ksize": 3, "iterations": 2})
        assert not ok.is_error, _text(ok)
        for op in (
            {"op": "dilate", "ksize": 3, "iterations": 200000},
            {"op": "erode", "ksize": 2_000_000_000, "iterations": 1},
            {"op": "opening", "ksize": 2_000_000_000},
            {"op": "closing", "ksize": 100_000},
            {"op": "median_blur", "ksize": 100_001},
        ):
            r = await refine(op)
            assert r.is_error, (op, _text(r))
            assert "wider than the mask's longest edge (300 px)" in _text(r), (
                op,
                _text(r),
            )


def test_refine_library_entry_point_applies_the_same_bound():
    """#117's repro was the library call, not the tool: apply_refinements()
    must refuse with the RefineSpecError every other malformed op gets, not
    leak MemoryError."""
    from plantcv_mcp.refine import RefineSpecError, apply_refinements

    mask = np.zeros((100, 100), np.uint8)
    mask[30:70, 30:70] = 255
    assert (apply_refinements(mask, [{"op": "erode", "ksize": 3}]) > 0).any()
    with pytest.raises(RefineSpecError, match="longest edge"):
        apply_refinements(mask, [{"op": "erode", "ksize": 2_000_000_000}])


# --- M6: checkerboard corner counts


async def test_checkerboard_corner_counts_have_a_ceiling(tmp_path):
    """row_corners/col_corners had a floor of 2 and no ceiling; 10^6 x 10^6
    sized a 10.9 TiB object-point grid (audit 2026-09-22, M6)."""
    from test_lens import POSES, _distort, _view, _write_frames

    boards = tmp_path / "boards"
    _write_frames(boards)
    scene = tmp_path / "scene.png"
    cv2.imwrite(
        str(scene), cv2.cvtColor(_distort(_view(*POSES[0])), cv2.COLOR_GRAY2BGR)
    )
    base = {"image_path": str(scene), "checkerboard_dir": str(boards)}
    async with Client(build_server()) as client:
        ok = await client.call_tool(
            "correct_lens_distortion",
            {**base, "row_corners": 6, "col_corners": 9},
        )
        assert not ok.is_error, _text(ok)
        for rc, cc in ((10**6, 10**6), (6, 201)):
            r = await client.call_tool(
                "correct_lens_distortion",
                {
                    **base,
                    "row_corners": rc,
                    "col_corners": cc,
                    "output_path": str(tmp_path / f"o_{rc}_{cc}.png"),
                },
            )
            assert r.is_error, _text(r)
            assert "must be between 2 and 200" in _text(r), _text(r)


# --- L8 + #117: a marker scale that is not a usable px_per_mm


def _marker(tmp_path) -> str:
    img = np.full((300, 300, 3), 240, np.uint8)
    cv2.circle(img, (150, 150), 40, (30, 30, 30), -1)
    return _png(tmp_path / "marker.png", img)


async def test_calibrate_scale_never_returns_an_unusable_scale(tmp_path):
    """marker_length_mm=1e-320 passed the `> 0` guard and returned
    px_per_mm=inf, which serialised as null and failed the client's schema
    check (audit 2026-09-22, L8); 1e308 returned 8e-307, which measure()
    itself refuses. The tool now refuses both with measure()'s own rule."""
    base = {"image_path": _marker(tmp_path), "x": 100, "y": 100, "w": 100, "h": 100}
    async with Client(build_server()) as client:
        ok = await client.call_tool(
            "calibrate_scale_from_marker", {**base, "marker_length_mm": 20.0}
        )
        assert not ok.is_error, _text(ok)
        assert ok.structured_content["px_per_mm"] == pytest.approx(4.0, rel=0.03)
        for mm in (1e-320, 1e308):
            r = await client.call_tool(
                "calibrate_scale_from_marker", {**base, "marker_length_mm": mm}
            )
            assert r.is_error, (mm, _text(r))
            assert "px_per_mm must be a positive finite number" in _text(r), _text(r)


@pytest.mark.parametrize("mm", [float("nan"), float("inf"), 1e-320])
def test_calibrate_scale_library_refuses_non_finite_marker_lengths(tmp_path, mm):
    """#117 item 1: NaN fails every comparison, so `<= 0` let it through and
    px_per_mm came back NaN; inf came back 0.0."""
    from plantcv_mcp.imaging import load_image
    from plantcv_mcp.scale import calibrate_scale

    img = load_image(_marker(tmp_path))
    assert calibrate_scale(img, 100, 100, 100, 100, 20.0).px_per_mm > 0
    with pytest.raises(ValueError, match="positive finite"):
        calibrate_scale(img, 100, 100, 100, 100, mm)


# --- M3: a call that fails after the mask is built leaves the store untouched


async def test_a_failed_render_mints_no_session_and_evicts_none(tmp_path, monkeypatch):
    """segment(), refine(), segment_hyperspectral() and segment_thermal() each
    created the store session BEFORE rendering the overlay. The store is an LRU
    of 8, so a call that then failed in the render (a 2x3000 image did, in
    downscale) still inserted a session nobody was told about and evicted the
    oldest live one — measure() on it then answered UnknownSessionError (audit
    2026-09-22, M3). A render failure is simulated at encode_png, the last step
    of every one of these calls."""
    from plantcv_mcp import server

    plant = _plant(tmp_path)
    async with Client(build_server()) as client:
        seg = await client.call_tool(
            "segment", {"image_path": plant, "channel": "a", "method": "otsu"}
        )
        sid = json.loads(seg.content[0].text)["session_id"]
        calls = {
            "segment": {"image_path": plant, "channel": "a", "method": "otsu"},
            "refine": {"session_id": sid, "ops": [{"op": "fill_holes"}]},
            "segment_thermal": {"path": _thermal_csv(tmp_path), "min_c": 25},
            "segment_hyperspectral": {"envi_path": _hsi_cube(tmp_path)},
        }
        real_encode = server.encode_png

        def broken_encode(img):
            raise RuntimeError("render failed")

        for tool, args in calls.items():
            before = set(server._store._sessions)
            monkeypatch.setattr(server, "encode_png", broken_encode)
            r = await client.call_tool(tool, args)
            assert r.is_error and "render failed" in _text(r), (tool, _text(r))
            assert set(server._store._sessions) == before, tool
            # Positive control: the same call, rendering, mints exactly one.
            monkeypatch.setattr(server, "encode_png", real_encode)
            r = await client.call_tool(tool, args)
            assert not r.is_error, (tool, _text(r))
            minted = set(server._store._sessions) - before
            assert minted == {json.loads(r.content[0].text)["session_id"]}, tool


# --- M4: a frame's range is the range of its finite values


def _strict_json(text: str):
    """json.loads that refuses the non-standard Infinity/NaN tokens."""

    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")

    return json.loads(text, parse_constant=refuse)


async def test_thermal_frame_range_and_overlay_ignore_non_finite_pixels(tmp_path):
    """frame_range was np.nanmin/np.nanmax, which skip NaN but not ±Inf: one
    inf pixel made segment_thermal print `Infinity` (not JSON) and
    measure_thermal fail the client's output-schema check (inf serialises as
    null); grey_frame scaled by that inf and rendered the plant black (audit
    2026-09-22, M4)."""
    from plantcv_mcp.thermal import grey_frame

    frame = np.full((60, 80), 20.0)
    frame[20:40, 30:50] = 30.0
    clean = _thermal_csv(tmp_path, "clean.csv", frame)
    dirty = frame.copy()
    dirty[0, 0], dirty[0, 1] = np.inf, -np.inf
    bad = _thermal_csv(tmp_path, "inf.csv", dirty)
    async with Client(build_server()) as client:
        for path in (clean, bad):  # clean is the positive control
            seg = await client.call_tool("segment_thermal", {"path": path, "min_c": 25})
            assert not seg.is_error, _text(seg)
            out = _strict_json(seg.content[0].text)
            assert out["frame_range"] == [20.0, 30.0], (path, out["frame_range"])
            m = await client.call_tool(
                "measure_thermal", {"session_id": out["session_id"]}
            )
            assert not m.is_error, _text(m)
            assert m.structured_content["frame_range"] == [20.0, 30.0]
            assert m.structured_content["temperature"]["mean"] == 30.0

    grey = grey_frame(dirty)
    assert grey[30, 40].tolist() == [255, 255, 255]  # the plant, at the top
    assert grey[50, 5].tolist() == [0, 0, 0]


async def test_hyperspectral_index_range_ignores_non_finite_pixels(tmp_path):
    """Same class in the cube path: one pixel with a zero 670 nm band made a
    ratio index infinite, and index_range printed `Infinity`."""
    from test_hyperspectral import _known_cube, _write_cube

    cube = _known_cube()
    clean = _write_cube(tmp_path, "clean", cube)
    cube[0, 0, 1] = 0.0  # 670 nm; pssr_chla = r800 / r680
    bad = _write_cube(tmp_path, "zero670", cube)
    async with Client(build_server()) as client:
        ranges = []
        for path in (clean, bad):
            r = await client.call_tool(
                "segment_hyperspectral",
                {"envi_path": path, "index": "pssr_chla", "threshold": 2.0},
            )
            assert not r.is_error, _text(r)
            ranges.append(_strict_json(r.content[0].text)["index_range"])
    assert ranges[0] == pytest.approx([2 / 3, 4.0], rel=1e-5)
    assert ranges[1] == ranges[0]


# --- M5: the decode gate checks bit depth as well as channel count


async def test_sixteen_bit_colour_images_are_refused_at_decode(tmp_path):
    """The gate at the one place pixels enter checked for 3 channels but not
    for 8 bits, so a 16-bit colour PNG passed it and died in cvtColor inside
    PlantCV with an OpenCV assertion (audit 2026-09-22, M5). Every RGB tool
    reads through the gate; three are driven here, with the same picture at
    8 bits as the positive control."""
    eight = cv2.imread(_plant(tmp_path, "p8.png"))
    p16 = _png(tmp_path / "p16.png", eight.astype(np.uint16) * 257)
    p8 = str(tmp_path / "p8.png")
    calls = {
        "segment": lambda p: {"image_path": p, "channel": "a", "method": "otsu"},
        "suggest_segmentation": lambda p: {"image_path": p},
        "calibrate_scale_from_marker": lambda p: {
            "image_path": p,
            "x": 100,
            "y": 100,
            "w": 100,
            "h": 100,
            "marker_length_mm": 10.0,
        },
    }
    async with Client(build_server()) as client:
        for tool, args in calls.items():
            ok = await client.call_tool(tool, args(p8))
            assert not ok.is_error, (tool, _text(ok))
            r = await client.call_tool(tool, args(p16))
            assert r.is_error, (tool, _text(r))
            assert "16-bit" in _text(r) and "OpenCV" not in _text(r), (tool, _text(r))
        batch = await client.call_tool(
            "measure_images",
            {"image_paths": [p8, p16], "channel": "a", "method": "otsu"},
        )
    rows = {r["image_path"]: r for r in batch.structured_content["results"]}
    assert rows[p8]["measured"] is True
    assert "16-bit" in rows[p16]["refused_because"], rows[p16]


# --- L9: thin images get a preview, not a zero-size resize


async def test_thin_and_tiny_images_render_their_previews(tmp_path):
    """downscale() computed int(edge * scale), which is 0 for a 2 px edge of a
    3000 px image, and OpenCV refused the resize: segment() failed after
    thresholding (audit 2026-09-22, L9). suggest_segmentation() on a 1 px
    image failed the same way inside PlantCV's own half-size sheet."""
    thin = np.full((2, 3000, 3), 240, np.uint8)
    thin[:, 1000:1500] = (40, 160, 40)
    thin_p = _png(tmp_path / "thin.png", thin)
    dot_p = _png(tmp_path / "dot.png", np.full((1, 1, 3), 40, np.uint8))
    plant = _plant(tmp_path)
    async with Client(build_server()) as client:
        for p in (plant, thin_p):  # plant: positive control
            r = await client.call_tool(
                "segment",
                {"image_path": p, "channel": "a", "method": "otsu", "fill_size": 1},
            )
            assert not r.is_error, (p, _text(r))
            assert any(c.type == "image" for c in r.content)
        for p in (plant, thin_p, dot_p):
            r = await client.call_tool("suggest_segmentation", {"image_path": p})
            assert not r.is_error, (p, _text(r))

    from plantcv_mcp.imaging import downscale

    small, scale = downscale(thin)
    assert small.shape[0] >= 1 and small.shape[1] == 1024, (small.shape, scale)


# --- L10: an empty indices list is a request for nothing, not for the default


async def test_empty_indices_is_refused_not_replaced_by_the_default(tmp_path):
    """`tuple(indices) if indices else (default,)` read [] as 'not given' and
    silently measured NDVI (audit 2026-09-22, L10). measure_spectral() and
    measure_regions() both refuse it; omitting indices still measures the
    session's index."""
    async with Client(build_server()) as client:
        seg = await client.call_tool(
            "segment_hyperspectral", {"envi_path": _hsi_cube(tmp_path)}
        )
        sid = json.loads(seg.content[0].text)["session_id"]
        grid = {
            "nrows": 1,
            "ncols": 1,
            "mode": "rect_grid",
            "coord": [10, 5],
            "height": 50,
            "width": 60,
            "spacing": [0, 0],
        }
        for tool, extra in (("measure_spectral", {}), ("measure_regions", grid)):
            ok = await client.call_tool(tool, {"session_id": sid, **extra})
            assert not ok.is_error, (tool, _text(ok))
            r = await client.call_tool(
                tool, {"session_id": sid, "indices": [], **extra}
            )
            assert r.is_error, (tool, _text(r))
            assert "No indices requested" in _text(r), (tool, _text(r))


async def test_module_docstring_does_not_miscount_the_tools():
    """server.py opened with 'Fifteen tools' after the sixteenth was added."""
    import re

    from plantcv_mcp import server

    words = "one two three four five six seven eight nine ten eleven twelve "
    words += "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
    number = {w: i for i, w in enumerate(words.split(), start=1)}
    async with Client(build_server()) as client:
        n_tools = len(await _tools(client))
    stated = re.findall(r"\b(\w+) tools\b", server.__doc__.lower())
    assert n_tools >= 16
    assert all(number.get(w, n_tools) == n_tools for w in stated), stated
