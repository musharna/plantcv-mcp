"""Read-root allow-list.

Unset, the server reads anything the host user can read (the documented trust
boundary). Configured, every path argument must resolve — symlinks and `..`
followed FIRST — to a location inside one of the roots. The tests that refuse
also assert the legitimate path succeeds, so a broken harness cannot read as
"blocked".
"""

import json
import os

import numpy as np
import pytest

from plantcv_mcp import paths
from plantcv_mcp.paths import PathOutsideRootsError, check_readable
from plantcv_mcp.server import build_server, list_methods_impl


@pytest.fixture
def roots(tmp_path):
    allowed = tmp_path / "allowed"
    other = tmp_path / "other"
    allowed.mkdir()
    other.mkdir()
    paths.set_roots([str(allowed)])
    try:
        yield allowed, other
    finally:
        paths.set_roots(None)


def _png(path):
    import cv2

    img = np.full((200, 200, 3), 128, np.uint8)
    img[50:150, 50:150] = (60, 180, 60)
    cv2.imwrite(str(path), img)
    return str(path)


def test_unconfigured_roots_allow_anything(tmp_path):
    paths.set_roots(None)
    os.environ.pop("PLANTCV_MCP_ROOTS", None)
    p = _png(tmp_path / "x.png")
    assert check_readable(p) == os.path.realpath(p)
    assert paths.configured_roots() is None


def test_inside_a_root_is_allowed_and_outside_is_refused_by_name(roots):
    allowed, other = roots
    inside = _png(allowed / "in.png")
    outside = _png(other / "out.png")
    assert check_readable(inside) == os.path.realpath(inside)
    with pytest.raises(PathOutsideRootsError, match=str(allowed)):
        check_readable(outside)


def test_dotdot_and_symlink_escapes_are_refused(roots):
    allowed, other = roots
    outside = _png(other / "out.png")
    sneaky = str(allowed / ".." / "other" / "out.png")
    with pytest.raises(PathOutsideRootsError):
        check_readable(sneaky)
    link = allowed / "link.png"
    link.symlink_to(outside)
    with pytest.raises(PathOutsideRootsError):
        check_readable(str(link))
    # Positive control: a real file inside the root, via its own `..`, is fine.
    inside = _png(allowed / "in.png")
    assert check_readable(str(allowed / "sub" / ".." / "in.png")) == os.path.realpath(
        inside
    )


def test_a_second_root_is_honoured(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    paths.set_roots([str(a), str(b)])
    try:
        assert check_readable(_png(b / "x.png"))
    finally:
        paths.set_roots(None)


def test_roots_come_from_the_environment(tmp_path, monkeypatch):
    paths.set_roots(None)
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("PLANTCV_MCP_ROOTS", os.pathsep.join([str(a), str(b)]))
    assert paths.configured_roots() == [os.path.realpath(a), os.path.realpath(b)]
    with pytest.raises(PathOutsideRootsError):
        check_readable(_png(tmp_path / "x.png"))


@pytest.mark.anyio
async def test_every_path_taking_tool_is_gated_over_the_mcp_layer(roots):
    from mcp.server.mcpserver.exceptions import ToolError

    allowed, other = roots
    inside = _png(allowed / "in.png")
    outside = _png(other / "out.png")
    mcp = build_server()

    for tool, args in [
        ("segment", {"channel": "a", "method": "otsu"}),
        ("suggest_segmentation", {"channel": "a"}),
        (
            "calibrate_scale_from_marker",
            {"x": 50, "y": 50, "w": 100, "h": 100, "marker_length_mm": 20.0},
        ),
    ]:
        with pytest.raises(ToolError, match="outside"):
            await mcp.call_tool(tool, {"image_path": outside, **args})

    # A batch with ONE stray path is refused whole: nothing is measured.
    with pytest.raises(ToolError, match="outside"):
        await mcp.call_tool(
            "measure_images",
            {"image_paths": [inside, outside], "channel": "a", "method": "otsu"},
        )

    # Positive controls: the same tools on the inside path work.
    seg = json.loads(
        (
            await mcp.call_tool(
                "segment", {"image_path": inside, "channel": "a", "method": "otsu"}
            )
        )
        .content[0]
        .text
    )
    assert seg["largest_area"] == 10000
    batch = await mcp.call_tool(
        "measure_images", {"image_paths": [inside], "channel": "a", "method": "otsu"}
    )
    assert batch.structured_content["summary"]["measured"] == 1


def test_list_methods_reports_the_policy(roots):
    allowed, _ = roots
    assert list_methods_impl()["read_roots"] == [os.path.realpath(allowed)]
    paths.set_roots(None)
    os.environ.pop("PLANTCV_MCP_ROOTS", None)
    assert list_methods_impl()["read_roots"] is None


def test_main_accepts_root_flags(monkeypatch, tmp_path):
    from plantcv_mcp import server, workers

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setattr(server.MCPServer, "run", lambda self: None)
    monkeypatch.setattr("sys.argv", ["plantcv-mcp", "--root", str(a), "--root", str(b)])
    paths.set_roots(None)
    try:
        server.main()
        assert paths.configured_roots() == [os.path.realpath(a), os.path.realpath(b)]
    finally:
        paths.set_roots(None)
        workers.set_isolation(None)


def test_the_read_boundary_itself_is_contained(roots):
    """Containment lives at read_image_bytes, not only at the tool layer.

    Derived paths (ENVI siblings) and any future reader go through this one
    function, so a path no tool ever validated still cannot leave the roots.
    """
    from plantcv_mcp.imaging import read_image_bytes

    allowed, other = roots
    outside = other / "secret.bin"
    outside.write_bytes(b"outside-bytes")
    link = allowed / "sneaky.bin"
    link.symlink_to(outside)
    with pytest.raises(PathOutsideRootsError):
        read_image_bytes(str(link))
    # Positive control: an in-root file reads, through an in-root symlink too.
    target = allowed / "data.bin"
    target.write_bytes(b"inside-bytes")
    via = allowed / "via.bin"
    via.symlink_to(target)
    assert read_image_bytes(str(via)) == b"inside-bytes"
