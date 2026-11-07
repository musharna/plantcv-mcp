"""sam_leaves.py at its edges, without torch (mutation burn-down, #97).

The nightly mutation job installs every extra but `sam`, so the model path
(load_model, _candidates, _prompt_points) had no test that runs there: 102
mutants were "no tests", and 136 more survived in the code around it
(2026-11-06). These tests fake only what needs torch or the network: the
`require_sam()` pair, `torch.load`, the mask generator, and `urlopen`.
Everything between those seams is the real code.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from plantcv_mcp import sam_leaves
from plantcv_mcp.leaves import LeafCountRefusedError
from plantcv_mcp.paths import PathOutsideRootsError, set_roots
from plantcv_mcp.sam_leaves import (
    CHECKPOINT_NAME,
    CHECKPOINT_URL,
    CheckpointDownloadError,
    CheckpointMissingError,
    CheckpointVerificationError,
    SamDeviceError,
    SamNotInstalledError,
    _candidates,
    _download,
    _prompt_points,
    _resolve_device,
    _sha256_of,
    cache_dir,
    load_model,
    open_verified,
    require_sam,
    resolve_checkpoint,
    segment_leaves_sam,
    select_instances,
)

GREEN = (40, 180, 40)


def _frame(h, w, *boxes, value=255):
    """A grey frame with green plant boxes `(y0, y1, x0, x1)`, and its mask."""
    img = np.full((h, w, 3), 90, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    for y0, y1, x0, x1 in boxes:
        img[y0:y1, x0:x1] = GREEN
        mask[y0:y1, x0:x1] = value
    return img, mask


def _quarters(keep=4, record=None):
    """A stand-in for `_candidates`: the plant crop cut into four vertical
    quarters of equal pixel count, `keep` of them offered as leaves."""

    def fake(model, rgb, plant):
        if record is not None:
            record.append((model, rgb.copy(), plant.copy()))
        cols = np.nonzero(plant.any(axis=0))[0]
        edges = np.linspace(cols.min(), cols.max() + 1, 5).astype(int)
        xx = np.broadcast_to(np.arange(plant.shape[1]), plant.shape)
        out = [
            (plant & (xx >= a) & (xx < b), 0.95, 0.95)
            for a, b in itertools.pairwise(edges)
        ]
        return out[:keep]

    return fake


@pytest.fixture
def stub_model(monkeypatch):
    """load_model returns a sentinel and records what it was asked for."""
    calls: list = []
    model = object()

    def fake_load(checkpoint, device):
        calls.append((checkpoint, device))
        return model

    monkeypatch.setattr(sam_leaves, "load_model", fake_load)
    return model, calls


# --- the crop handed to the model ---


@pytest.mark.parametrize(
    ("shape", "box", "crop"),
    [
        # extent 80 -> margin ceil(8.0) = 8
        ((200, 300), (50, 100, 100, 180), (42, 108, 92, 188)),
        # extent 81 -> margin ceil(8.1) = 9, clipped to 0 at the top and left
        ((200, 300), (3, 84, 2, 63), (0, 93, 0, 72)),
        # a tall frame: the bottom of the crop is clipped by the HEIGHT
        ((300, 100), (220, 300, 20, 60), (212, 300, 12, 68)),
    ],
)
def test_the_crop_is_the_mask_box_plus_a_tenth_of_its_longer_side(
    shape, box, crop, monkeypatch, stub_model
):
    model, calls = stub_model
    seen: list = []
    monkeypatch.setattr(sam_leaves, "_candidates", _quarters(record=seen))
    img, mask = _frame(*shape, box)
    out = segment_leaves_sam(img, mask, "ckpt.pth")
    assert calls == [("ckpt.pth", "cpu")]
    [(got_model, rgb, plant)] = seen
    assert got_model is model  # the loaded model, not a placeholder
    y0, y1, x0, x1 = crop
    assert plant.shape == (y1 - y0, x1 - x0)
    assert (plant == (mask[y0:y1, x0:x1] > 0)).all()
    assert (rgb == img[y0:y1, x0:x1, ::-1]).all()  # BGR -> RGB
    # The labels come back in full-frame coordinates, on the mask and only there.
    assert out.labels.shape == mask.shape and out.labels.dtype == np.int32
    assert ((out.labels > 0) == (mask > 0)).all()
    assert sorted(np.unique(out.labels).tolist()) == [0, 1, 2, 3, 4]


def test_a_0_1_mask_and_a_0_1_observed_read_like_0_255(monkeypatch, stub_model):
    monkeypatch.setattr(sam_leaves, "_candidates", _quarters())
    img, mask = _frame(200, 300, (50, 100, 100, 180))
    ref = segment_leaves_sam(img, mask, "c")
    ones = segment_leaves_sam(img, (mask > 0).astype(np.uint8), "c")
    seen = segment_leaves_sam(img, mask, "c", observed=(mask > 0).astype(np.uint8))
    for out in (ones, seen):
        assert out.leaf_count == ref.leaf_count == 4
        assert (out.labels == ref.labels).all()


def test_the_result_envelope_in_full(monkeypatch, stub_model):
    monkeypatch.setattr(sam_leaves, "_candidates", _quarters(keep=3))
    img, mask = _frame(200, 300, (50, 100, 100, 180))
    out = segment_leaves_sam(img, mask, "c", px_per_mm=0.5)
    assert out.leaf_count == 3 and out.candidates == 3 and out.device == "cpu"
    assert out.mask_coverage == 0.75
    assert out.units == {
        "area": "mm2",
        "area_px": "pixels",
        "centroid": "pixels",
        "bbox": "pixels",
    }
    # A scale below 1 px/mm is still a scale.
    assert out.instances[0]["area"] == out.instances[0]["area_px"] / 0.25
    with pytest.raises(ValueError) as exc:
        segment_leaves_sam(img, mask, "c", px_per_mm=0.0)
    assert str(exc.value) == "px_per_mm must be a positive finite number, got 0.0"


def test_mask_coverage_is_rounded_to_four_places(monkeypatch, stub_model):
    """One leaf of 1,333 px on a 4,000-px plant: 0.33325, kept to 0.3332."""
    img, mask = _frame(200, 300, (50, 100, 100, 180))  # 4,000 px

    def one_leaf(model, rgb, plant):
        leaf = np.zeros_like(plant)
        leaf.flat[np.flatnonzero(plant)[:1333]] = True
        return [(leaf, 0.95, 0.95)]

    monkeypatch.setattr(sam_leaves, "_candidates", one_leaf)
    out = segment_leaves_sam(img, mask, "c")
    assert out.mask_coverage == round(1333 / 4000, 4)
    assert out.mask_coverage != round(1333 / 4000, 5)


def test_low_coverage_is_flagged_below_half_and_not_at_it(monkeypatch, stub_model):
    img, mask = _frame(200, 300, (50, 100, 100, 180))
    monkeypatch.setattr(sam_leaves, "_candidates", _quarters(keep=2))
    half = segment_leaves_sam(img, mask, "c")
    assert half.mask_coverage == 0.5
    assert "low_instance_coverage" not in [w.code for w in half.warnings]
    monkeypatch.setattr(sam_leaves, "_candidates", _quarters(keep=1))
    quarter = segment_leaves_sam(img, mask, "c")
    assert [
        w.message for w in quarter.warnings if w.code == "low_instance_coverage"
    ] == [
        (
            "The 1 instances cover 25% of the plant mask. The rest got no "
            "leaf-sized mask from the model — leaves there are NOT in the count. "
            "Look at the overlay for unoutlined green."
        )
    ]


def test_no_leaf_from_the_model_is_refused_in_full(monkeypatch, stub_model):
    img, mask = _frame(200, 300, (50, 100, 100, 180))
    monkeypatch.setattr(sam_leaves, "_candidates", lambda model, rgb, plant: [])
    with pytest.raises(LeafCountRefusedError) as exc:
        segment_leaves_sam(img, mask, "c")
    assert str(exc.value) == (
        "Segment Anything returned 0 candidate mask(s) and none passed the leaf "
        "filter (score, size between 0.2% and 40% of the plant, on the mask). A "
        "zero is not reported as a leaf count; check the overlay from segment()."
    )


def test_an_inverted_mask_is_refused_before_the_model_loads(monkeypatch, stub_model):
    _, calls = stub_model
    img = np.full((100, 100, 3), 90, np.uint8)  # grey: no plant fills this frame
    mask = np.full((100, 100), 255, np.uint8)
    mask[40:60, 40:60] = 0
    with pytest.raises(LeafCountRefusedError) as exc:
        segment_leaves_sam(img, mask, "c")
    msg = str(exc.value)
    assert msg.startswith("implausible_coverage: ")
    assert msg.endswith(
        " Prompts are placed on the mask, so an inverted mask prompts the "
        "background; fix the segmentation first."
    )
    assert calls == []


def test_several_objects_carry_the_multi_object_advisory_in_full(
    monkeypatch, stub_model
):
    monkeypatch.setattr(sam_leaves, "_candidates", _quarters())
    img, mask = _frame(200, 300, (50, 100, 40, 120), (50, 100, 180, 260))
    out = segment_leaves_sam(img, mask, "c")
    assert [w.message for w in out.warnings if w.code == "multi_object_mask"] == [
        (
            "The mask holds 2 comparably sized objects and the count covers ALL "
            "of them. That is expected for one rosette whose leaves segment "
            "apart; if they are separate plants, refine(keep_largest) or crop to "
            "one plant and count that session. The area limits below are "
            "fractions of the WHOLE mask, so several plants also shift them."
        )
    ]


# --- the post-filter at its exact thresholds ---


def _box(shape, y0, y1, x0, x1):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_select_instances_keeps_each_threshold_exactly_and_drops_just_past_it():
    plant = np.ones((100, 100), bool)  # 10,000 px: leaves 20..4,000 px
    shape = plant.shape
    kept = select_instances(
        plant,
        [
            (_box(shape, 0, 2, 0, 10), 0.95, 0.95),  # 20 px, the smallest kept
            (_box(shape, 10, 50, 0, 100), 0.95, 0.95),  # 4,000 px, the largest
            (_box(shape, 60, 70, 0, 10), 0.80, 0.95),  # predicted IoU exactly 0.80
            (np.zeros(shape, bool), 0.95, 0.95),  # an empty candidate is ignored
        ],
    )
    assert kept.dtype == np.int32
    assert kept.max() == 3
    dropped = select_instances(
        plant,
        [
            (_box(shape, 0, 1, 0, 19), 0.95, 0.95),  # 19 px
            (_box(shape, 10, 51, 0, 100), 0.95, 0.95),  # 4,100 px
            (_box(shape, 60, 70, 0, 10), 0.79, 0.95),
        ],
    )
    assert dropped.max() == 0


def test_a_candidate_exactly_80_percent_on_observed_pixels_is_kept():
    plant = np.ones((100, 100), bool)
    observed = np.ones((100, 100), bool)
    observed[0:2, 0:10] = False  # 20 of the candidate's 100 px were not seen
    cand = _box(plant.shape, 0, 10, 0, 10)
    assert select_instances(plant, [(cand, 0.95, 0.95)], observed).max() == 1
    observed[2, 0] = False  # 21 unseen: 79%
    assert select_instances(plant, [(cand, 0.95, 0.95)], observed).max() == 0


def test_overlap_is_judged_against_every_mask_kept_so_far():
    plant = np.ones((100, 100), bool)
    shape = plant.shape
    a = _box(shape, 0, 10, 0, 10)  # 100 px
    b = _box(shape, 50, 60, 50, 65)  # 150 px, far from a
    # 200 px, exactly half on a: kept (only MORE than half is dropped).
    half = _box(shape, 0, 10, 0, 20)
    assert select_instances(plant, [(a, 0.9, 0.9), (half, 0.9, 0.9)]).max() == 2
    # Taken smallest first: a, b, then the third. 100 of its 190 px are on a
    # and none on b, so it is dropped although b, not a, was kept last.
    heavy = a | _box(shape, 80, 89, 80, 90)
    assert int(heavy.sum()) == 190 and not (heavy & b).any()
    labels = select_instances(plant, [(a, 0.9, 0.9), (b, 0.9, 0.9), (heavy, 0.9, 0.9)])
    assert labels.max() == 2
    # Control: 100 of 280 px on a is under half, and the same third is kept.
    light = a | _box(shape, 80, 98, 80, 90)
    assert int(light.sum()) == 280
    labels = select_instances(plant, [(a, 0.9, 0.9), (b, 0.9, 0.9), (light, 0.9, 0.9)])
    assert labels.max() == 3


# --- the prompt grid ---


def test_prompt_points_are_the_grid_cell_centres_that_fall_on_the_plant():
    centres = (np.arange(32) + 0.5) / 32
    full = _prompt_points(np.ones((64, 32), bool))
    assert full.shape == (1024, 2)
    assert full[0].tolist() == [centres[0], centres[0]]
    assert full[1].tolist() == [centres[1], centres[0]]  # x varies fastest
    assert full[32].tolist() == [centres[0], centres[1]]
    # Only the top two rows of a 64-row plant: the first grid row (y = 1/64
    # of the height lands on row 1), every x.
    top = np.zeros((64, 32), bool)
    top[:2] = True
    pts = _prompt_points(top)
    assert pts[:, 1].tolist() == [centres[0]] * 32
    assert pts[:, 0].tolist() == centres.tolist()
    # Only the first column: the first grid column, every y.
    left = np.zeros((64, 32), bool)
    left[:, 0] = True
    pts = _prompt_points(left)
    assert pts[:, 0].tolist() == [centres[0]] * 32
    assert pts[:, 1].tolist() == centres.tolist()
    # The last grid row and column land exactly on the plant's last row
    # (31.5/32 x 64 = 63) and last column (31.5/32 x 32 = 31.5 -> 31).
    bottom = np.zeros((64, 32), bool)
    bottom[63] = True
    assert _prompt_points(bottom)[:, 1].tolist() == [centres[31]] * 32
    right = np.zeros((64, 32), bool)
    right[:, 31] = True
    assert _prompt_points(right)[:, 0].tolist() == [centres[31]] * 32
    assert len(_prompt_points(np.zeros((64, 32), bool))) == 0


# --- the model calls, through a fake torch and segment_anything ---


class _Gen:
    made: ClassVar[list] = []

    def __init__(self, model, **kwargs):
        _Gen.made.append((model, kwargs))

    def generate(self, rgb):
        assert _Gen.inside, "generate() ran outside torch.inference_mode()"
        _Gen.rgb = rgb
        return [
            {
                "segmentation": np.ones(rgb.shape[:2], bool),
                "predicted_iou": np.float32(0.875),
                "stability_score": np.float64(0.9),
            }
        ]


@contextlib.contextmanager
def _inference_mode():
    _Gen.inside = True
    try:
        yield
    finally:
        _Gen.inside = False


def _fake_sam(monkeypatch, cuda=False, count=0):
    torch = SimpleNamespace(
        inference_mode=_inference_mode,
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count),
        loads=[],
    )

    def load(handle, **kwargs):
        torch.loads.append((handle.tell(), handle.read(), kwargs))
        return {"weights": 1}

    torch.load = load
    models: list = []

    class Model:
        def __init__(self):
            self.calls = []
            models.append(self)

        def load_state_dict(self, state):
            self.calls.append(("load_state_dict", state))

        def to(self, device):
            self.calls.append(("to", device))
            return self

        def eval(self):
            self.calls.append(("eval",))
            return self

    sam = SimpleNamespace(
        SamAutomaticMaskGenerator=_Gen, sam_model_registry={"vit_b": Model}
    )
    monkeypatch.setattr(sam_leaves, "require_sam", lambda: (torch, sam))
    _Gen.made, _Gen.inside = [], False
    return torch, models


def test_candidates_asks_the_generator_for_the_plant_points_only(monkeypatch):
    _fake_sam(monkeypatch)
    model = object()
    plant = np.zeros((64, 32), bool)
    plant[:, 0] = True
    rgb = np.zeros((64, 32, 3), np.uint8)
    out = _candidates(model, rgb, plant)
    [(got_model, kwargs)] = _Gen.made
    assert got_model is model and _Gen.rgb is rgb
    grids = kwargs.pop("point_grids")
    assert len(grids) == 1 and (grids[0] == _prompt_points(plant)).all()
    assert kwargs == {
        "points_per_side": None,
        "pred_iou_thresh": 0.80,
        "stability_score_thresh": 0.85,
        "box_nms_thresh": 0.95,
        "crop_n_layers": 0,
        "min_mask_region_area": 0,
    }
    [(mask, iou, stability)] = out
    assert mask.shape == (64, 32)
    assert (iou, stability) == (0.875, 0.9)
    assert type(iou) is float and type(stability) is float
    # No point on the plant: no generator, no call, no candidates.
    _Gen.made = []
    assert _candidates(model, rgb, np.zeros((64, 32), bool)) == []
    assert _Gen.made == []


def test_load_model_loads_the_verified_bytes_once_per_file_and_device(
    tmp_path, monkeypatch
):
    payload = b"a checkpoint, as bytes"
    ckpt = tmp_path / "c.pth"
    ckpt.write_bytes(payload)
    monkeypatch.setattr(
        sam_leaves, "CHECKPOINT_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(sam_leaves, "_loaded", {})
    torch, models = _fake_sam(monkeypatch, cuda=True, count=1)

    model = load_model(str(ckpt), "cpu")
    assert torch.loads == [(0, payload, {"map_location": "cpu", "weights_only": True})]
    assert models == [model]
    assert model.calls == [
        ("load_state_dict", {"weights": 1}),
        ("to", "cpu"),
        ("eval",),
    ]
    # The same file under another name, same device: the cached model.
    link = tmp_path / "link.pth"
    link.symlink_to(ckpt)
    assert load_model(str(link), "cpu") is model
    assert len(torch.loads) == 1
    # Another device: loaded again, and only one model is kept in memory.
    gpu = load_model(str(ckpt), "cuda")
    assert gpu is not model and gpu.calls[1] == ("to", "cuda")
    assert len(sam_leaves._loaded) == 1
    assert load_model(str(ckpt), "cuda") is gpu


def test_load_model_refuses_a_checkpoint_with_the_wrong_hash(tmp_path, monkeypatch):
    ckpt = tmp_path / "c.pth"
    ckpt.write_bytes(b"not it")
    monkeypatch.setattr(sam_leaves, "_loaded", {})
    torch, models = _fake_sam(monkeypatch)
    with pytest.raises(CheckpointVerificationError):
        load_model(str(ckpt), "cpu")
    assert torch.loads == [] and models == []


# --- devices ---


def test_device_names_are_checked_in_full():
    no_gpu = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    many = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 200)
    )
    assert _resolve_device(many, "cuda:123") == "cuda:123"  # 8 characters
    with pytest.raises(SamDeviceError) as exc:
        _resolve_device(many, "cuda:1234")  # 9: not a device name
    assert str(exc.value) == (
        "device must be 'cpu', 'cuda' or 'cuda:N', got 'cuda:1234'. The default "
        "is 'cpu'; a GPU is used only when asked for."
    )
    with pytest.raises(SamDeviceError) as exc:
        _resolve_device(no_gpu, "cuda")
    assert str(exc.value) == (
        "device='cuda' was asked for and torch reports no CUDA device. Nothing "
        "ran; use device='cpu'."
    )
    with pytest.raises(SamDeviceError) as exc:
        _resolve_device(many, "cuda:200")
    assert str(exc.value) == "device='cuda:200': torch sees 200 CUDA device(s)."


# --- the extra, the cache, the checkpoint ---


def _block_import(monkeypatch, exc):
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name in ("segment_anything", "torch", "torchvision"):
            raise exc
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


@pytest.mark.parametrize(
    ("exc", "named"),
    [
        (ImportError("x", name="torch"), "torch"),
        (ImportError("x"), "one of its modules"),
    ],
)
def test_the_missing_extra_is_named_in_full(monkeypatch, exc, named):
    _block_import(monkeypatch, exc)
    with pytest.raises(SamNotInstalledError) as info:
        require_sam()
    assert str(info.value) == (
        f"segment_leaves_sam() needs the optional `sam` extra, and {named} is not "
        'importable. Install it with: pip install "plantcv-mcp[sam]" — that adds '
        "torch, torchvision and segment-anything. count_leaves() needs none of "
        "them and was NOT run in its place."
    )


def test_the_cache_follows_the_override_then_xdg_then_home(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANTCV_MCP_SAM_CACHE", str(tmp_path / "over"))
    assert cache_dir() == tmp_path / "over"
    monkeypatch.delenv("PLANTCV_MCP_SAM_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_dir() == tmp_path / "xdg" / "plantcv-mcp"
    monkeypatch.setenv("XDG_CACHE_HOME", "")  # set but empty: the home default
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert cache_dir() == tmp_path / "home" / ".cache" / "plantcv-mcp"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert cache_dir() == tmp_path / "home" / ".cache" / "plantcv-mcp"


def test_a_missing_cached_checkpoint_is_refused_in_full(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANTCV_MCP_SAM_CACHE", str(tmp_path))
    with pytest.raises(CheckpointMissingError) as exc:
        resolve_checkpoint(None, download=False)
    cached = str(tmp_path / CHECKPOINT_NAME)
    assert str(exc.value) == (
        f"No SAM checkpoint at {cached!r}. Either call again with "
        f"download_checkpoint=true to let the server fetch {CHECKPOINT_URL} "
        "(375 MB, SHA-256 verified before use, cached in that directory; "
        "PLANTCV_MCP_SAM_CACHE moves it), or download it yourself and pass "
        "checkpoint_path. Nothing was downloaded."
    )


def test_a_wrong_hash_is_refused_in_full(tmp_path, monkeypatch):
    ckpt = tmp_path / "c.pth"
    ckpt.write_bytes(b"something else")
    with pytest.raises(CheckpointVerificationError) as exc:
        open_verified(str(ckpt))
    actual = hashlib.sha256(b"something else").hexdigest()
    assert str(exc.value) == (
        f"{str(ckpt)!r} has SHA-256 {actual}, not the pinned "
        f"{sam_leaves.CHECKPOINT_SHA256} of {CHECKPOINT_NAME}. It was not loaded. "
        "A truncated download or a different checkpoint both read this way; "
        "delete the file (or pass the right one) and try again."
    )


def test_the_opened_file_is_checked_against_the_roots_by_its_name(
    tmp_path, monkeypatch
):
    """If the name passed the check and the file opened is elsewhere (the path
    changed in between), the refusal names the path the caller gave."""
    inside, outside = tmp_path / "root", tmp_path / "elsewhere"
    inside.mkdir()
    outside.mkdir()
    target = outside / "c.pth"
    target.write_bytes(b"x")
    monkeypatch.setattr(sam_leaves, "check_readable", lambda path: path)  # the race
    set_roots([str(inside)])
    try:
        with pytest.raises(PathOutsideRootsError) as exc:
            open_verified(str(target))
    finally:
        set_roots(None)
    assert str(exc.value).startswith(f"{str(target)!r} was opened as ")


class _Reads(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.sizes = []

    def read(self, size=-1):
        self.sizes.append(size)
        return super().read(size)


def test_hashing_reads_in_one_mib_chunks():
    handle = _Reads(b"z" * (3 << 20))
    assert _sha256_of(handle) == hashlib.sha256(b"z" * (3 << 20)).hexdigest()
    assert handle.sizes == [1 << 20] * 4
    assert handle.tell() == 0


# --- the download ---


class _Response(_Reads):
    def __init__(self, chunks, on_read=None):
        super().__init__(b"".join(chunks))
        self.chunks = list(chunks)
        self.on_read = on_read

    def read(self, size=-1):
        self.sizes.append(size)
        if self.on_read:
            self.on_read()
        return self.chunks.pop(0) if self.chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(monkeypatch, response):
    seen = {}

    def urlopen(url, timeout):
        seen["url"], seen["timeout"] = url, timeout
        return response

    monkeypatch.setattr(sam_leaves.urllib.request, "urlopen", urlopen)
    return seen


def test_a_download_streams_into_a_part_file_beside_the_target(tmp_path, monkeypatch):
    payload = [b"a" * 1000, b"b" * 500]
    data = b"".join(payload)
    monkeypatch.setattr(
        sam_leaves, "CHECKPOINT_SHA256", hashlib.sha256(data).hexdigest()
    )
    monkeypatch.setattr(sam_leaves, "CHECKPOINT_BYTES", len(data))
    target = tmp_path / "a" / "b" / "c" / CHECKPOINT_NAME  # no parent exists yet
    parts: list = []

    def look():
        if target.parent.is_dir():
            parts.append(sorted(p.name for p in target.parent.iterdir()))

    response = _Response(payload, on_read=look)
    seen = _serve(monkeypatch, response)
    _download(target)
    assert seen == {"url": CHECKPOINT_URL, "timeout": 60}
    assert response.sizes == [1 << 20] * 3
    assert parts and all(len(p) == 1 and p[0].endswith(".part") for p in parts)
    assert target.read_bytes() == data
    assert sorted(p.name for p in target.parent.iterdir()) == [CHECKPOINT_NAME]


def test_a_download_that_runs_past_the_expected_size_is_abandoned(
    tmp_path, monkeypatch
):
    """Each chunk alone is under the limit; their sum is not."""
    monkeypatch.setattr(sam_leaves, "CHECKPOINT_BYTES", 10)
    _serve(monkeypatch, _Response([b"x" * 6, b"x" * 6]))
    target = tmp_path / CHECKPOINT_NAME
    with pytest.raises(CheckpointVerificationError) as exc:
        _download(target)
    assert str(exc.value) == (
        f"{CHECKPOINT_URL} sent more than the expected 10 bytes; download abandoned."
    )
    assert list(tmp_path.iterdir()) == []


def test_an_unreachable_download_is_refused_in_full(tmp_path, monkeypatch):
    def urlopen(url, timeout):
        raise OSError("no route to host")

    monkeypatch.setattr(sam_leaves.urllib.request, "urlopen", urlopen)
    with pytest.raises(CheckpointDownloadError) as exc:
        _download(tmp_path / CHECKPOINT_NAME)
    assert str(exc.value) == (
        f"Could not download {CHECKPOINT_URL}: no route to host. Fetch it "
        "yourself and pass checkpoint_path; it is verified the same way."
    )
    assert list(tmp_path.iterdir()) == []


def test_the_module_never_imports_torch_itself():
    """Every fake above stands in for torch; the real one must stay optional."""
    src = Path(sam_leaves.__file__).read_text()
    assert "\nimport torch" not in src and "\nfrom torch" not in src
