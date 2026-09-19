"""segment_leaves_sam() WITH the model. Needs the `sam` extra and the ViT-B
checkpoint, so it runs in its own CI job (ci.yml, `sam`), on CPU.

Outside that job the module is skipped when the extra is absent. Inside it
(PLANTCV_MCP_SAM_TESTS=1) a missing extra is an ERROR: a skip there would turn
a broken install into a green job.

The checkpoint comes from the tool's own download path, into
PLANTCV_MCP_SAM_CACHE (the job caches that directory), so the real URL, the
real size and the real pinned SHA-256 are exercised, not a stand-in.
"""

import asyncio
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported: CPU only
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from plantcv_mcp import sam_leaves
from plantcv_mcp.sam_leaves import (
    CheckpointVerificationError,
    SamDeviceError,
    resolve_checkpoint,
)
from plantcv_mcp.server import (
    ImageChangedSinceSegmentationError,
    _refine_impl,
    _segment_impl,
    _segment_leaves_sam_impl,
    build_server,
)
from plantcv_mcp.workers import set_isolation

SAM_JOB = os.environ.get("PLANTCV_MCP_SAM_TESTS") == "1"
if SAM_JOB:
    import segment_anything  # noqa: F401
    import torch
    import torchvision  # noqa: F401
else:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("segment_anything")

FIXTURES = Path(__file__).parent / "fixtures" / "aberystwyth"
YOUNG = FIXTURES / "tray032_2015-12-21_plant0.png"  # 11 annotated leaves
GROWN = FIXTURES / "tray032_2016-01-05_plant0.png"  # 25 annotated leaves


@pytest.fixture(scope="module")
def checkpoint() -> str:
    if (
        not SAM_JOB
        and not (sam_leaves.cache_dir() / sam_leaves.CHECKPOINT_NAME).is_file()
    ):
        pytest.skip("no cached SAM checkpoint; the CI `sam` job downloads it")
    return resolve_checkpoint(None, download=SAM_JOB)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _session(path: Path) -> str:
    sid = _segment_impl(str(path), "a", "otsu")["session_id"]
    return _refine_impl(sid, [{"op": "fill_holes"}, {"op": "keep_largest", "n": 1}])[
        "session_id"
    ]


def test_the_real_checkpoint_is_verified_corrupted_refused_good_loaded(
    checkpoint, tmp_path, monkeypatch
):
    """One flipped byte in the real 375 MB file is refused and never reaches
    torch.load; the untouched file loads, with weights_only=True."""
    bad = tmp_path / "flipped.pth"
    shutil.copyfile(checkpoint, bad)
    with open(bad, "r+b") as handle:
        handle.seek(200_000_000)
        byte = handle.read(1)
        handle.seek(200_000_000)
        handle.write(bytes([byte[0] ^ 1]))
    sam_leaves._loaded.clear()
    with pytest.raises(CheckpointVerificationError, match="was not loaded"):
        sam_leaves.load_model(str(bad), "cpu")
    assert sam_leaves._loaded == {}
    seen = []
    real_load = torch.load

    def spy(*args, **kwargs):
        seen.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    model = sam_leaves.load_model(checkpoint, "cpu")
    assert seen == [True]  # the restricted unpickler, and exactly one load
    assert next(model.parameters()).device.type == "cpu"
    assert sam_leaves.load_model(checkpoint, "cpu") is model  # cached


def test_counts_on_the_two_real_rosettes_over_the_mcp_layer(checkpoint):
    """Through build_server() and the isolation worker (on by default), so the
    result type crosses the pipe. The pinned counts are what this model and
    this filter give on the two fixture plants (11 and 25 annotated leaves);
    the watershed gives 7 and 13 on the same sessions (test_leaves.py)."""
    server = build_server()
    counts = {}
    for name, path in (("young", YOUNG), ("grown", GROWN)):
        result = asyncio.run(
            server.call_tool(
                "segment_leaves_sam",
                {"session_id": _session(path), "checkpoint_path": checkpoint},
            )
        )
        text_block, image_block = result.content
        payload = json.loads(text_block.text)
        counts[name] = payload["leaf_count"]
        assert payload["method"] == "segment_anything_point_grid"
        assert payload["device"] == "cpu"
        assert payload["model"]["sha256"] == sam_leaves.CHECKPOINT_SHA256
        assert [op["op"] for op in payload["lineage"]][-2:] == [
            "fill_holes",
            "keep_largest",
        ]
        assert len(payload["instances"]) == payload["leaf_count"]
        assert [i["id"] for i in payload["instances"]] == list(
            range(1, payload["leaf_count"] + 1)
        )
        assert 0.5 <= payload["mask_coverage"] <= 1.0
        assert image_block.type == "image"
    assert counts == {"young": YOUNG_COUNT, "grown": GROWN_COUNT}


# Measured 2026-09-19 (torch 2.14.0+cpu, segment-anything 1.0) through this same
# path; the annotations say 11 and 25. They pin the pipeline, not its accuracy.
YOUNG_COUNT = 9
GROWN_COUNT = 28


def test_instances_lie_on_the_mask_and_the_overlay_shows_them(checkpoint):
    set_isolation(False)
    try:
        img = cv2.imread(str(YOUNG))
        sid = _session(YOUNG)
        from plantcv_mcp.server import _store

        mask = _store.get(sid).mask
        res = sam_leaves.segment_leaves_sam(img, mask, checkpoint)
    finally:
        set_isolation(None)
    assert res.labels.shape == mask.shape
    assert not res.labels[mask == 0].any()
    assert res.leaf_count == int(res.labels.max()) == YOUNG_COUNT
    for inst in res.instances:
        region = res.labels == inst["id"]
        assert inst["area_px"] == int(region.sum()) > 0
        x, y, w, h = inst["bbox"]
        ys, xs = np.nonzero(region)
        assert (x, y, w, h) == (
            xs.min(),
            ys.min(),
            xs.max() - xs.min() + 1,
            ys.max() - ys.min() + 1,
        )
    from plantcv_mcp.leaves import _instance_color

    assert (res.overlay == np.array(_instance_color(1), np.uint8)).all(axis=2).any()


def test_an_unavailable_gpu_is_an_error_and_cpu_still_runs(checkpoint):
    """CPU is the default and the tests force it (CUDA_VISIBLE_DEVICES is empty
    here), so torch reports no GPU: asking for one is refused, not downgraded."""
    sid = _session(YOUNG)
    assert not torch.cuda.is_available()
    with pytest.raises(SamDeviceError, match="no CUDA device"):
        _segment_leaves_sam_impl(sid, checkpoint_path=checkpoint, device="cuda")
    assert (
        _segment_leaves_sam_impl(sid, checkpoint_path=checkpoint)["leaf_count"]
        == YOUNG_COUNT
    )


def test_a_changed_image_is_refused_before_the_model_runs(checkpoint, tmp_path):
    copy = tmp_path / "plant.png"
    shutil.copyfile(YOUNG, copy)
    sid = _session(copy)
    assert _segment_leaves_sam_impl(sid, checkpoint_path=checkpoint)["leaf_count"] > 0
    img = cv2.imread(str(copy))
    img[0, 0] = (1, 2, 3)
    cv2.imwrite(str(copy), img)
    with pytest.raises(ImageChangedSinceSegmentationError, match="CONTENT changed"):
        _segment_leaves_sam_impl(sid, checkpoint_path=checkpoint)
