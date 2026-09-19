"""segment_leaves_sam() WITHOUT the model: everything a base install must get
right, plus the post-filter and the checkpoint handling, which need no torch.

The tests that load Segment Anything are in test_sam_leaves_model.py and run in
their own CI job. These run everywhere, including that job.
"""

import hashlib
import importlib.util
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from plantcv_mcp import sam_leaves
from plantcv_mcp.paths import PathOutsideRootsError, set_roots
from plantcv_mcp.sam_leaves import (
    CheckpointDownloadError,
    CheckpointMissingError,
    CheckpointVerificationError,
    SamDeviceError,
    SamNotInstalledError,
    open_verified,
    resolve_checkpoint,
    select_instances,
)
from plantcv_mcp.server import (
    _count_leaves_impl,
    _segment_impl,
    _segment_leaves_sam_impl,
    build_server,
)

SAM_JOB = os.environ.get("PLANTCV_MCP_SAM_TESTS") == "1"


def _plant_png(tmp_path) -> str:
    img = np.full((240, 320, 3), 200, np.uint8)
    for centre in ((110, 120), (170, 120), (140, 80)):
        cv2.ellipse(img, centre, (34, 22), 0, 0, 360, (40, 150, 40), -1)
    path = str(tmp_path / "plant.png")
    cv2.imwrite(path, img)
    return path


@pytest.mark.skipif(SAM_JOB, reason="the SAM CI job installs the extra on purpose")
def test_the_ordinary_environment_has_no_torch():
    """The extra must not ride along on `uv sync --all-extras`: every ordinary
    CI job would download torch on every run. If this fails, a workflow or a
    dependency group started installing the `sam` extra."""
    for module in ("torch", "torchvision", "segment_anything"):
        assert importlib.util.find_spec(module) is None, (
            f"{module} is installed in the base test environment"
        )


def test_without_the_extra_the_tool_refuses_with_the_install_command(
    tmp_path, monkeypatch
):
    """A named error carrying the exact command, over the real MCP layer, and
    NOT a watershed count in its place. In the base environment the modules are
    really absent; in the SAM job their absence is forced the way Python itself
    reports one (a None entry in sys.modules raises ImportError)."""
    import asyncio

    from mcp.server.mcpserver.exceptions import ToolError

    for module in ("torch", "torchvision", "segment_anything"):
        monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.setenv(sam_leaves.CACHE_ENV, str(tmp_path / "cache"))
    seg = _segment_impl(_plant_png(tmp_path), "a", "otsu")
    server = build_server()
    with pytest.raises(ToolError) as refused:
        asyncio.run(
            server.call_tool(
                "segment_leaves_sam",
                {"session_id": seg["session_id"], "download_checkpoint": True},
            )
        )
    text = str(refused.value)
    assert "segment_leaves_sam: SamNotInstalledError:" in text
    assert 'pip install "plantcv-mcp[sam]"' in text
    assert "NOT run in its place" in text
    # Refused before anything was fetched, although download was permitted.
    assert not (tmp_path / "cache").exists()
    # Positive control: the session is good and the base tool counts it.
    assert _count_leaves_impl(seg["session_id"])["leaf_count"] >= 1


def test_the_tool_is_registered_and_declares_its_network_and_write_access():
    import asyncio

    tools = {t.name: t for t in asyncio.run(build_server().list_tools())}
    sam = tools["segment_leaves_sam"].annotations
    assert sam.read_only_hint is False and sam.open_world_hint is True
    # Control: the watershed tool keeps the read-only, closed-world contract.
    base = tools["count_leaves"].annotations
    assert base.read_only_hint is True and base.open_world_hint is False


def _disc(shape, centre, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2 <= radius**2


def test_the_post_filter_keeps_leaves_and_drops_everything_else():
    shape = (200, 300)
    left, right = _disc(shape, (100, 100), 40), _disc(shape, (100, 200), 40)
    third = _disc(shape, (40, 150), 30)
    plant = left | right | third
    speckle = _disc(shape, (100, 100), 2)  # 13 px of ~12900: under 0.2%
    background = _disc(shape, (170, 150), 25) & ~plant
    half_off = _disc(shape, (100, 40), 40)  # about half of it is off the plant
    repeat = _disc(shape, (100, 100), 38)  # the left leaf again, slightly smaller
    candidates = [
        (left | right, 0.99, 0.99),  # spans two leaves: over 40% of the plant
        (plant, 0.99, 0.99),  # the whole rosette
        (left, 0.95, 0.95),
        (right, 0.95, 0.95),
        (third, 0.79, 0.99),  # predicted IoU under the threshold
        (speckle, 0.99, 0.99),
        (background, 0.99, 0.99),
        (half_off, 0.99, 0.99),
        (repeat, 0.99, 0.99),
    ]
    labels = select_instances(plant, candidates)
    assert int(labels.max()) == 2
    # Smallest first: `repeat` is taken, and `left` is then >50% covered.
    assert set(np.unique(labels[repeat])) == {1}
    assert set(np.unique(labels[right])) == {2}
    assert not labels[third & ~left & ~right].any()
    assert not labels[~plant].any()

    # Each rule, alone, is what removed its candidate: lift the rule's reason
    # and that candidate comes back.
    assert int(select_instances(plant, [*candidates, (third, 0.81, 0.99)]).max()) == 3
    assert int(select_instances(plant, [(third, 0.99, 0.84)]).max()) == 0
    assert int(select_instances(plant, [(third, 0.99, 0.85)]).max()) == 1
    # The size ceiling alone: with nothing smaller kept, the overlap rule cannot
    # be what removes a rosette-sized mask.
    assert int(select_instances(plant, [(plant, 0.99, 0.99)]).max()) == 0
    assert int(select_instances(plant, [(left | right, 0.99, 0.99)]).max()) == 0
    assert int(select_instances(plant, [(left, 0.99, 0.99)]).max()) == 1


def test_overlapping_leaves_both_survive_up_to_half_covered():
    shape = (200, 300)
    a, b = _disc(shape, (100, 110), 40), _disc(shape, (100, 150), 41)
    plant = a | b | _disc(shape, (100, 230), 45)  # keeps each leaf under 40%
    labels = select_instances(plant, [(a, 0.9, 0.9), (b, 0.9, 0.9)])
    assert int(labels.max()) == 2  # b is ~39% covered by a
    # A kept mask labels only what no earlier one took: no pixel is relabelled.
    assert set(np.unique(labels[a])) == {1}
    assert set(np.unique(labels[b & ~a])) == {2}
    nearer = _disc(shape, (100, 125), 41)  # ~77% covered by a
    assert int(select_instances(plant, [(a, 0.9, 0.9), (nearer, 0.9, 0.9)]).max()) == 1


def _pin(monkeypatch, payload: bytes) -> None:
    monkeypatch.setattr(
        sam_leaves, "CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(sam_leaves, "CHECKPOINT_BYTES", len(payload))


def test_a_corrupted_checkpoint_is_refused_and_the_good_one_opens(
    tmp_path, monkeypatch
):
    payload = os.urandom(4096)
    _pin(monkeypatch, payload)
    good, bad = tmp_path / "good.pth", tmp_path / "bad.pth"
    good.write_bytes(payload)
    bad.write_bytes(payload[:100] + bytes([payload[100] ^ 1]) + payload[101:])
    with pytest.raises(CheckpointVerificationError, match="was not loaded"):
        open_verified(str(bad))
    with open_verified(str(good)) as handle:
        assert handle.tell() == 0 and handle.read() == payload


def test_a_checkpoint_outside_the_read_roots_is_refused(tmp_path, monkeypatch):
    payload = os.urandom(512)
    _pin(monkeypatch, payload)
    inside, outside = tmp_path / "root", tmp_path / "elsewhere"
    inside.mkdir()
    outside.mkdir()
    (inside / "c.pth").write_bytes(payload)
    (outside / "c.pth").write_bytes(payload)
    set_roots([str(inside)])
    try:
        with pytest.raises(PathOutsideRootsError):
            open_verified(str(outside / "c.pth"))
        open_verified(str(inside / "c.pth")).close()
    finally:
        set_roots(None)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_nothing_is_downloaded_without_permission(tmp_path, monkeypatch):
    calls = []
    payload = os.urandom(2048)
    _pin(monkeypatch, payload)
    monkeypatch.setenv(sam_leaves.CACHE_ENV, str(tmp_path / "cache"))

    def fake_urlopen(url, timeout):
        calls.append(url)
        return _Response(payload)

    monkeypatch.setattr(sam_leaves.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(CheckpointMissingError, match="download_checkpoint=true"):
        resolve_checkpoint(None, download=False)
    assert calls == [] and not (tmp_path / "cache").exists()
    # With permission, the same call fetches, verifies and caches it.
    path = resolve_checkpoint(None, download=True)
    assert calls == [sam_leaves.CHECKPOINT_URL]
    assert Path(path) == tmp_path / "cache" / sam_leaves.CHECKPOINT_NAME
    assert Path(path).read_bytes() == payload
    # And the cached file is used afterwards without another request.
    assert resolve_checkpoint(None, download=False) == path
    assert len(calls) == 1


def test_a_download_with_the_wrong_hash_is_discarded_not_cached(tmp_path, monkeypatch):
    payload = os.urandom(2048)
    _pin(monkeypatch, payload)
    cache = tmp_path / "cache"
    monkeypatch.setenv(sam_leaves.CACHE_ENV, str(cache))
    served = {"body": payload[:-1] + bytes([payload[-1] ^ 1])}
    monkeypatch.setattr(
        sam_leaves.urllib.request,
        "urlopen",
        lambda url, timeout: _Response(served["body"]),
    )
    with pytest.raises(CheckpointVerificationError, match="discarded, not cached"):
        resolve_checkpoint(None, download=True)
    assert list(cache.iterdir()) == []  # neither the file nor a .part
    served["body"] = payload + b"x"  # longer than the pinned size
    with pytest.raises(CheckpointVerificationError, match="more than the expected"):
        resolve_checkpoint(None, download=True)
    assert list(cache.iterdir()) == []

    def unreachable(url, timeout):
        raise OSError("network is unreachable")

    monkeypatch.setattr(sam_leaves.urllib.request, "urlopen", unreachable)
    with pytest.raises(CheckpointDownloadError, match="network is unreachable"):
        resolve_checkpoint(None, download=True)
    assert list(cache.iterdir()) == []
    # Positive control: the right bytes through the same seam are cached.
    served["body"] = payload
    monkeypatch.setattr(
        sam_leaves.urllib.request,
        "urlopen",
        lambda url, timeout: _Response(served["body"]),
    )
    resolve_checkpoint(None, download=True)
    assert [p.name for p in cache.iterdir()] == [sam_leaves.CHECKPOINT_NAME]


def test_the_cache_is_held_to_the_read_roots_before_any_download(tmp_path, monkeypatch):
    payload = os.urandom(1024)
    _pin(monkeypatch, payload)
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        return _Response(payload)

    monkeypatch.setattr(sam_leaves.urllib.request, "urlopen", fake_urlopen)
    root = tmp_path / "root"
    root.mkdir()
    set_roots([str(root)])
    try:
        monkeypatch.setenv(sam_leaves.CACHE_ENV, str(tmp_path / "outside"))
        with pytest.raises(PathOutsideRootsError):
            resolve_checkpoint(None, download=True)
        assert calls == [] and not (tmp_path / "outside").exists()
        monkeypatch.setenv(sam_leaves.CACHE_ENV, str(root / "cache"))
        path = resolve_checkpoint(None, download=True)
        assert calls == [sam_leaves.CHECKPOINT_URL]
        open_verified(path).close()
    finally:
        set_roots(None)


def test_download_url_is_the_pinned_https_constant():
    """What the `nosec B310` on urlopen rests on: the URL is a constant, https,
    on the upstream host, and the pinned digest is a full SHA-256."""
    assert sam_leaves.CHECKPOINT_URL == (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    )
    assert len(sam_leaves.CHECKPOINT_SHA256) == 64
    int(sam_leaves.CHECKPOINT_SHA256, 16)


def test_a_missing_checkpoint_path_is_named(tmp_path):
    with pytest.raises(CheckpointMissingError, match="is not a file"):
        resolve_checkpoint(str(tmp_path / "absent.pth"), download=True)
    present = tmp_path / "present.pth"
    present.write_bytes(b"x")
    assert resolve_checkpoint(str(present), download=False) == str(present)


def test_the_device_is_cpu_unless_a_real_gpu_is_asked_for():
    def torch_with(gpus):
        return SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: gpus > 0, device_count=lambda: gpus
            )
        )

    resolve = sam_leaves._resolve_device
    assert resolve(torch_with(0), "cpu") == "cpu"
    assert resolve(torch_with(1), "cpu") == "cpu"  # present is not asked for
    with pytest.raises(SamDeviceError, match="no CUDA device"):
        resolve(torch_with(0), "cuda")
    assert resolve(torch_with(1), "cuda") == "cuda"
    assert resolve(torch_with(2), "cuda:1") == "cuda:1"
    with pytest.raises(SamDeviceError, match="sees 2"):
        resolve(torch_with(2), "cuda:2")
    for bad in ("gpu", "auto", "cuda:", "cuda:x", "mps", ""):
        with pytest.raises(SamDeviceError, match="must be 'cpu'"):
            resolve(torch_with(1), bad)


def test_a_thermal_session_is_refused_before_the_extra_is_even_checked(monkeypatch):
    from plantcv_mcp.server import WrongSessionKindError, _segment_thermal_impl

    fixture = Path(__file__).parent / "fixtures" / "plantcv" / "thermal_img.npz"
    thermal = _segment_thermal_impl(str(fixture), min_c=30.0, max_c=34.0)
    with pytest.raises(WrongSessionKindError, match=r"measure_thermal\(\)"):
        _segment_leaves_sam_impl(thermal["session_id"])
    # Control: an RGB session passes that gate and reaches the next refusal.
    for module in ("torch", "torchvision", "segment_anything"):
        monkeypatch.setitem(sys.modules, module, None)
    rgb = Path(__file__).parent / "fixtures" / "aberystwyth"
    seg = _segment_impl(str(rgb / "tray032_2015-12-21_plant0.png"), "a", "otsu")
    with pytest.raises(SamNotInstalledError):
        _segment_leaves_sam_impl(seg["session_id"])


# --- The pipeline around the model, with ONLY the model call replaced. -------
# `_candidates` is the single seam that needs torch. These stubs return masks
# cut from the crop they are handed, so the crop, the filter, the offset back to
# the full frame, the advisories and the envelope are the real code.


def _strips(n_strips: int, keep: int | None = None):
    """A stand-in for `_candidates`: the plant crop cut into vertical strips of
    equal pixel count, each offered as one high-scoring mask."""

    def fake(model, rgb, plant):
        assert rgb.shape[:2] == plant.shape and rgb.shape[2] == 3
        cols = np.nonzero(plant)[1]
        edges = np.quantile(cols, np.linspace(0, 1, n_strips + 1))
        xx = np.broadcast_to(np.arange(plant.shape[1]), plant.shape)
        out = []
        for k in range(n_strips):
            upper = xx <= edges[k + 1] if k == n_strips - 1 else xx < edges[k + 1]
            out.append((plant & (xx >= edges[k]) & upper, 0.95, 0.95))
        return out[:keep]

    return fake


@pytest.fixture
def no_model(monkeypatch, tmp_path):
    """The extra 'present', a checkpoint file that is never opened, the model
    never loaded, in-process so the stubs are seen. Returns the dummy path and
    the list of load_model calls."""
    from plantcv_mcp import server
    from plantcv_mcp.workers import set_isolation

    loads: list = []
    monkeypatch.setattr(server, "require_sam", lambda: (None, None))
    monkeypatch.setattr(
        sam_leaves, "load_model", lambda ckpt, device: loads.append(device) or object()
    )
    dummy = tmp_path / "unused.pth"
    dummy.write_bytes(b"never opened")
    set_isolation(False)
    yield str(dummy), loads
    set_isolation(None)


def test_the_envelope_offsets_units_and_overlay_without_the_model(
    tmp_path, monkeypatch, no_model
):
    from plantcv_mcp.leaves import _instance_color
    from plantcv_mcp.server import _store

    checkpoint, loads = no_model
    monkeypatch.setattr(sam_leaves, "_candidates", _strips(3))
    seg = _segment_impl(_plant_png(tmp_path), "a", "otsu")
    out = _segment_leaves_sam_impl(
        seg["session_id"], checkpoint_path=checkpoint, px_per_mm=2.0
    )
    assert loads == ["cpu"]  # the default device, and nothing else
    assert out["leaf_count"] == 3 and out["candidate_masks"] == 3
    assert out["method"] == "segment_anything_point_grid"
    assert out["device"] == "cpu" and out["model"]["license"] == "Apache-2.0"
    assert out["mask_coverage"] == 1.0
    assert out["units"]["area"] == "mm2"
    assert "low_instance_coverage" not in [w["code"] for w in out["warnings"]]
    mask = _store.get(seg["session_id"]).mask
    ys, xs = np.nonzero(mask)
    # Full-frame coordinates: the plant starts at x=76, far from the crop's 0.
    assert min(i["bbox"][0] for i in out["instances"]) == xs.min() > 50
    assert min(i["bbox"][1] for i in out["instances"]) == ys.min() > 30
    assert sum(i["area_px"] for i in out["instances"]) == int((mask > 0).sum())
    for inst in out["instances"]:
        assert inst["area"] == inst["area_px"] / 4.0  # the SQUARE of px_per_mm
        assert mask[int(inst["centroid"][1]), int(inst["centroid"][0])] > 0
    overlay = cv2.imdecode(np.frombuffer(out["_png"], np.uint8), cv2.IMREAD_COLOR)
    assert (overlay == np.array(_instance_color(1), np.uint8)).all(axis=2).any()
    # Without px_per_mm the same call reports pixels.
    px = _segment_leaves_sam_impl(seg["session_id"], checkpoint_path=checkpoint)
    assert px["units"]["area"] == "pixels"
    assert px["instances"][0]["area"] == px["instances"][0]["area_px"]


def test_low_coverage_is_flagged_and_zero_instances_is_refused(
    tmp_path, monkeypatch, no_model
):
    from plantcv_mcp.leaves import LeafCountRefusedError

    checkpoint, _ = no_model
    seg = _segment_impl(_plant_png(tmp_path), "a", "otsu")
    monkeypatch.setattr(sam_leaves, "_candidates", _strips(3, keep=1))
    out = _segment_leaves_sam_impl(seg["session_id"], checkpoint_path=checkpoint)
    assert out["leaf_count"] == 1 and 0.30 < out["mask_coverage"] < 0.37
    assert "low_instance_coverage" in [w["code"] for w in out["warnings"]]
    # Two of three strips cover two thirds: counted, and not flagged.
    monkeypatch.setattr(sam_leaves, "_candidates", _strips(3, keep=2))
    out = _segment_leaves_sam_impl(seg["session_id"], checkpoint_path=checkpoint)
    assert out["leaf_count"] == 2
    assert "low_instance_coverage" not in [w["code"] for w in out["warnings"]]
    # No surviving mask is a refusal, never a count of zero.
    monkeypatch.setattr(sam_leaves, "_candidates", lambda model, rgb, plant: [])
    with pytest.raises(LeafCountRefusedError, match="none passed the leaf filter"):
        _segment_leaves_sam_impl(seg["session_id"], checkpoint_path=checkpoint)


def test_empty_and_inverted_masks_are_refused_before_the_model_loads(
    monkeypatch, no_model
):
    from plantcv_mcp.diagnostics import DegenerateMaskError
    from plantcv_mcp.leaves import LeafCountRefusedError

    checkpoint, loads = no_model
    monkeypatch.setattr(sam_leaves, "_candidates", _strips(3))
    img = np.full((200, 300, 3), 128, np.uint8)
    good = np.zeros((200, 300), np.uint8)
    good[60:140, 80:220] = 255
    with pytest.raises(DegenerateMaskError):
        sam_leaves.segment_leaves_sam(img, np.zeros_like(good), checkpoint)
    with pytest.raises(LeafCountRefusedError, match="implausible_coverage"):
        sam_leaves.segment_leaves_sam(img, 255 - good, checkpoint)
    with pytest.raises(ValueError, match="px_per_mm must be a positive"):
        sam_leaves.segment_leaves_sam(img, good, checkpoint, px_per_mm=0.0)
    assert loads == []
    res = sam_leaves.segment_leaves_sam(img, good, checkpoint)
    assert res.leaf_count == 3 and loads == ["cpu"]
    assert [w.code for w in res.warnings] == []


def test_several_objects_and_a_clipped_mask_carry_their_advisories(
    tmp_path, monkeypatch, no_model
):
    checkpoint, _ = no_model
    monkeypatch.setattr(sam_leaves, "_candidates", _strips(4))
    img = np.full((200, 300, 3), 128, np.uint8)
    img[0:90, 40:120] = (60, 180, 60)  # cut by the top edge
    img[60:150, 180:260] = (60, 180, 60)  # a second, comparable object
    two = str(tmp_path / "two.png")
    cv2.imwrite(two, img)
    sid = _segment_impl(two, "a", "otsu")["session_id"]
    codes = [
        w["code"]
        for w in _segment_leaves_sam_impl(sid, checkpoint_path=checkpoint)["warnings"]
    ]
    assert "multi_object_mask" in codes and "frame_clipping" in codes
    assert "multi_specimen" not in codes  # replaced by multi_object_mask
    one = np.full((200, 300, 3), 128, np.uint8)
    one[60:150, 100:220] = (60, 180, 60)
    single = str(tmp_path / "one.png")
    cv2.imwrite(single, one)
    sid = _segment_impl(single, "a", "otsu")["session_id"]
    assert _segment_leaves_sam_impl(sid, checkpoint_path=checkpoint)["warnings"] == []


def test_no_workflow_installs_the_sam_extra_except_the_sam_job():
    """`--all-extras` would pull torch into every ordinary CI job. Each use must
    carry `--no-extra sam`; only ci.yml's `sam` job may name `--extra sam`."""
    workflows = Path(__file__).parent.parent / ".github" / "workflows"
    all_extras, naming_sam = [], []
    for path in sorted(workflows.glob("*.yml")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if "--all-extras" in line:
                all_extras.append((path.name, number, line))
            if "--extra sam" in line:
                naming_sam.append(path.name)
    assert len(all_extras) >= 5, "the scan found no --all-extras: it is broken"
    bare = [(f, n) for f, n, line in all_extras if "--no-extra sam" not in line]
    assert bare == [], f"--all-extras without --no-extra sam at {bare}"
    assert naming_sam and set(naming_sam) == {"ci.yml"}
