"""Leaf instances from Segment Anything (ViT-B), behind the optional `sam` extra.

`count_leaves()` is a distance-transform watershed and merges overlapping
leaves. This module asks a learned, promptable segmenter instead: one point
prompt per grid cell that falls inside the plant mask, then a fixed post-filter
that keeps leaf-sized masks and drops the ones that repeat or span the rosette.

What was measured, and on what (docs/EVAL.md, "Leaf instances: Segment
Anything"): every constant below was chosen on Aberystwyth Tray 031 and the
held-out Tray 032 is what is quoted. SAM was not trained on leaves and is not
fine-tuned here; the count is still an estimate to be read with the overlay.

Nothing in this module imports torch at import time. The base install imports
it, registers the tool, and refuses a call with `SamNotInstalledError` and the
install command. There is no fallback to the watershed: a caller who asked for
this method gets this method or a named error.
"""

import hashlib
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .diagnostics import (
    Advisory,
    analyze_mask,
    assert_not_degenerate,
    implausible_coverage_warning,
)
from .leaves import LeafCountRefusedError, describe_instances
from .paths import check_open_fd, check_readable

INSTALL_COMMAND = 'pip install "plantcv-mcp[sam]"'
MODEL_TYPE = "vit_b"
# The official ViT-B checkpoint (Apache-2.0, as the code). Upstream publishes no
# SHA-256; the `01ec64` in the file name is the first six hex digits of its MD5,
# which the download measured here matched. The SHA-256 below was computed from
# that download (2026-09-19) and is what every load is checked against.
CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/" + CHECKPOINT_NAME
CHECKPOINT_SHA256 = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
CHECKPOINT_BYTES = 375_042_383
CACHE_ENV = "PLANTCV_MCP_SAM_CACHE"
DOWNLOAD_TIMEOUT_S = 60

# Prompt grid: GRID x GRID points over the crop, kept where they fall on the mask.
GRID = 32
# The crop handed to SAM is the mask's bounding box plus this fraction of its
# longer side on every edge. SAM rescales its input to 1024 px, so the crop, not
# the photo, sets the resolution a leaf is seen at. Chosen on Tray 031 from
# 0.10 / 0.25 / 0.50 by grown-rosette MAE (2.00 / 2.21 / 2.21; young 1.35 /
# 1.25 / 1.20): a small difference on 14 plants, and not a sensitive choice.
# The crop is clipped to the image, so on a tightly cropped photo a larger
# margin changes nothing. A smaller margin puts more of the grid on the plant
# and so sends SAM more prompts.
CROP_MARGIN_FRACTION = 0.10
# Mask generator thresholds (SAM's own quality scores), then the post-filter:
PRED_IOU_MIN = 0.80
STABILITY_MIN = 0.85
INSIDE_MIN = 0.80  # a candidate must lie at least this much on the plant mask
MIN_AREA_FRACTION = 0.002  # of the plant mask: smaller is speckle
MAX_AREA_FRACTION = 0.40  # larger is several leaves or the whole rosette
OVERLAP_MAX = 0.50  # with masks already kept, taken smallest first
LOW_COVERAGE = 0.50


class SamNotInstalledError(Exception):
    """The optional `sam` extra (torch, torchvision, segment-anything) is absent."""


class CheckpointMissingError(Exception):
    """No verified checkpoint is available and downloading was not permitted."""


class CheckpointVerificationError(Exception):
    """A checkpoint's SHA-256 is not the pinned one. It is never loaded."""


class CheckpointDownloadError(Exception):
    """The checkpoint could not be fetched."""


class SamDeviceError(Exception):
    """The requested torch device is not valid or not available."""


@dataclass
class SamLeavesResult:
    leaf_count: int
    instances: list[dict[str, Any]]
    mask_coverage: float
    candidates: int
    device: str
    units: dict[str, str]
    warnings: list[Advisory]
    overlay: np.ndarray
    labels: np.ndarray


def require_sam() -> tuple[Any, Any]:
    """(torch, segment_anything), or SamNotInstalledError naming the command."""
    try:
        import segment_anything
        import torch
        import torchvision  # noqa: F401 — segment_anything needs it at call time
    except ImportError as exc:
        raise SamNotInstalledError(
            f"segment_leaves_sam() needs the optional `sam` extra, and "
            f"{exc.name or 'one of its modules'} is not importable. Install it "
            f"with: {INSTALL_COMMAND} — that adds torch, torchvision and "
            "segment-anything. count_leaves() needs none of them and was NOT "
            "run in its place."
        ) from exc
    return torch, segment_anything


def cache_dir() -> Path:
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "plantcv-mcp"


def _sha256_of(handle) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    while chunk := handle.read(1 << 20):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def open_verified(path: str):
    """Open `path`, check the OPEN file against the pinned SHA-256, and return
    the handle rewound to 0. The bytes that were hashed are the bytes that get
    loaded: hashing a name and then re-opening it would check one file and
    load another."""
    check_readable(path)
    handle = open(path, "rb")  # noqa: SIM115 — returned open, closed by caller
    try:
        check_open_fd(handle.fileno(), path)
        actual = _sha256_of(handle)
        if actual != CHECKPOINT_SHA256:
            raise CheckpointVerificationError(
                f"{path!r} has SHA-256 {actual}, not the pinned "
                f"{CHECKPOINT_SHA256} of {CHECKPOINT_NAME}. It was not loaded. "
                "A truncated download or a different checkpoint both read this "
                "way; delete the file (or pass the right one) and try again."
            )
    except BaseException:
        handle.close()
        raise
    return handle


def _download(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    received = 0
    fd, part = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out:
            try:
                # The URL is the https constant above, never caller input.
                with urllib.request.urlopen(  # nosec B310 - test_sam_leaves.py::test_download_url_is_the_pinned_https_constant: fixed https URL, no caller-supplied scheme
                    CHECKPOINT_URL, timeout=DOWNLOAD_TIMEOUT_S
                ) as response:
                    while chunk := response.read(1 << 20):
                        received += len(chunk)
                        if received > CHECKPOINT_BYTES:
                            raise CheckpointVerificationError(
                                f"{CHECKPOINT_URL} sent more than the expected "
                                f"{CHECKPOINT_BYTES} bytes; download abandoned."
                            )
                        digest.update(chunk)
                        out.write(chunk)
            except OSError as exc:
                raise CheckpointDownloadError(
                    f"Could not download {CHECKPOINT_URL}: {exc}. Fetch it "
                    "yourself and pass checkpoint_path; it is verified the same way."
                ) from exc
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise CheckpointVerificationError(
                f"The download from {CHECKPOINT_URL} has SHA-256 "
                f"{digest.hexdigest()} ({received} bytes), not the pinned "
                f"{CHECKPOINT_SHA256}. It was discarded, not cached."
            )
        os.replace(part, target)
    finally:
        if os.path.exists(part):
            os.unlink(part)


def resolve_checkpoint(checkpoint_path: str | None, download: bool) -> str:
    """The path of a checkpoint to load: the caller's, or the cached one,
    downloading it first only when `download` is true. Verification proper
    happens at load (open_verified), on the file that is opened."""
    if checkpoint_path is not None:
        if not os.path.isfile(checkpoint_path):
            raise CheckpointMissingError(
                f"checkpoint_path {checkpoint_path!r} is not a file."
            )
        return checkpoint_path
    cached = cache_dir() / CHECKPOINT_NAME
    # With read roots configured the cache is held to them like any other path,
    # and BEFORE a download: fetching 375 MB into a directory the server will
    # then refuse to read is a refusal that arrives late.
    check_readable(str(cached))
    if cached.is_file():
        return str(cached)
    if not download:
        raise CheckpointMissingError(
            f"No SAM checkpoint at {str(cached)!r}. Either call again with "
            f"download_checkpoint=true to let the server fetch {CHECKPOINT_URL} "
            f"({CHECKPOINT_BYTES // 1_000_000} MB, SHA-256 verified before use, "
            f"cached in that directory; {CACHE_ENV} moves it), or download it "
            "yourself and pass checkpoint_path. Nothing was downloaded."
        )
    _download(cached)
    return str(cached)


_loaded: dict[tuple[str, str], Any] = {}


def _resolve_device(torch: Any, device: str) -> str:
    if device == "cpu":
        return device
    if device == "cuda" or (
        device.startswith("cuda:") and device[5:].isdigit() and len(device) < 9
    ):
        if not torch.cuda.is_available():
            raise SamDeviceError(
                f"device={device!r} was asked for and torch reports no CUDA "
                "device. Nothing ran; use device='cpu'."
            )
        if device != "cuda" and int(device[5:]) >= torch.cuda.device_count():
            raise SamDeviceError(
                f"device={device!r}: torch sees {torch.cuda.device_count()} "
                "CUDA device(s)."
            )
        return device
    raise SamDeviceError(
        f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}. The default "
        "is 'cpu'; a GPU is used only when asked for."
    )


def load_model(checkpoint: str, device: str) -> Any:
    torch, segment_anything = require_sam()
    device = _resolve_device(torch, device)
    key = (os.path.realpath(checkpoint), device)
    if key not in _loaded:
        with open_verified(checkpoint) as handle:
            # The checkpoint is a plain state_dict of tensors, so the
            # restricted unpickler is enough: no code in the file can run.
            state = torch.load(handle, map_location="cpu", weights_only=True)
        model = segment_anything.sam_model_registry[MODEL_TYPE]()
        model.load_state_dict(state)
        model.to(device).eval()
        _loaded.clear()  # one model in memory at a time
        _loaded[key] = model
    return _loaded[key]


def select_instances(
    plant: np.ndarray, candidates: list[tuple[np.ndarray, float, float]]
) -> np.ndarray:
    """Label image from SAM candidates `(mask, predicted_iou, stability)`.

    Kept: scores at or above the thresholds, at least INSIDE_MIN on the plant,
    area within [MIN_AREA_FRACTION, MAX_AREA_FRACTION] of the plant. Survivors
    are taken smallest first, and one is dropped when more than OVERLAP_MAX of
    it is already covered — so a mask spanning two leaves loses to the two
    leaves. A kept mask labels only the pixels no earlier one took.
    """
    area = int(plant.sum())
    kept: list[tuple[int, np.ndarray]] = []
    for mask, predicted_iou, stability in candidates:
        if predicted_iou < PRED_IOU_MIN or stability < STABILITY_MIN:
            continue
        total = int(mask.sum())
        on_plant = mask & plant
        size = int(on_plant.sum())
        if total == 0 or size / total < INSIDE_MIN:
            continue
        if size < MIN_AREA_FRACTION * area or size > MAX_AREA_FRACTION * area:
            continue
        kept.append((size, on_plant))
    kept.sort(key=lambda item: item[0])
    labels = np.zeros(plant.shape, np.int32)
    taken = np.zeros(plant.shape, bool)
    count = 0
    for size, on_plant in kept:
        if int((on_plant & taken).sum()) / size > OVERLAP_MAX:
            continue
        count += 1
        labels[on_plant & ~taken] = count
        taken |= on_plant
    return labels


def _prompt_points(plant: np.ndarray) -> np.ndarray:
    height, width = plant.shape
    centres = (np.arange(GRID) + 0.5) / GRID
    points = np.stack(np.meshgrid(centres, centres), -1).reshape(-1, 2)  # x, y
    rows = np.minimum((points[:, 1] * height).astype(int), height - 1)
    cols = np.minimum((points[:, 0] * width).astype(int), width - 1)
    return points[plant[rows, cols]]


def _candidates(model: Any, rgb: np.ndarray, plant: np.ndarray) -> list:
    torch, segment_anything = require_sam()
    points = _prompt_points(plant)
    if len(points) == 0:
        return []
    generator = segment_anything.SamAutomaticMaskGenerator(
        model,
        points_per_side=None,
        point_grids=[points],
        pred_iou_thresh=PRED_IOU_MIN,
        stability_score_thresh=STABILITY_MIN,
        box_nms_thresh=0.95,  # near-duplicates only; select_instances does the rest
        crop_n_layers=0,
        min_mask_region_area=0,
    )
    with torch.inference_mode():
        raw = generator.generate(rgb)
    return [
        (m["segmentation"], float(m["predicted_iou"]), float(m["stability_score"]))
        for m in raw
    ]


def segment_leaves_sam(
    img: np.ndarray,
    mask: np.ndarray,
    checkpoint: str,
    device: str = "cpu",
    px_per_mm: float | None = None,
) -> SamLeavesResult:
    if px_per_mm is not None and (px_per_mm <= 0 or not math.isfinite(px_per_mm)):
        raise ValueError(f"px_per_mm must be a positive finite number, got {px_per_mm}")
    diag = analyze_mask(mask)
    assert_not_degenerate(diag)
    coverage = implausible_coverage_warning(diag)
    if coverage:
        raise LeafCountRefusedError(
            f"implausible_coverage: {coverage.message} Prompts are placed on the "
            "mask, so an inverted mask prompts the background; fix the "
            "segmentation first."
        )
    warnings: list[Advisory] = []
    if diag.major_object_count >= 2:
        warnings.append(
            Advisory(
                code="multi_object_mask",
                message=(
                    f"The mask holds {diag.major_object_count} comparably sized "
                    "objects and the count covers ALL of them. That is expected "
                    "for one rosette whose leaves segment apart; if they are "
                    "separate plants, refine(keep_largest) or crop to one plant "
                    "and count that session. The area limits below are fractions "
                    "of the WHOLE mask, so several plants also shift them."
                ),
            )
        )
    model = load_model(checkpoint, device)

    mask255 = np.where(mask > 0, 255, 0).astype(np.uint8)
    ys, xs = np.nonzero(mask255)
    extent = max(int(ys.max() - ys.min()), int(xs.max() - xs.min())) + 1
    margin = math.ceil(CROP_MARGIN_FRACTION * extent)
    y0, y1 = (
        max(int(ys.min()) - margin, 0),
        min(int(ys.max()) + 1 + margin, mask.shape[0]),
    )
    x0, x1 = (
        max(int(xs.min()) - margin, 0),
        min(int(xs.max()) + 1 + margin, mask.shape[1]),
    )
    plant = mask255[y0:y1, x0:x1] > 0
    rgb = cv2.cvtColor(np.ascontiguousarray(img[y0:y1, x0:x1]), cv2.COLOR_BGR2RGB)

    candidates = _candidates(model, rgb, plant)
    labels_c = select_instances(plant, candidates)
    n = int(labels_c.max())
    if n == 0:
        raise LeafCountRefusedError(
            f"Segment Anything returned {len(candidates)} candidate mask(s) and "
            "none passed the leaf filter (score, size between "
            f"{MIN_AREA_FRACTION:.1%} and {MAX_AREA_FRACTION:.0%} of the plant, "
            "on the mask). A zero is not reported as a leaf count; check the "
            "overlay from segment()."
        )
    covered = float((labels_c > 0).sum()) / float(plant.sum())
    if covered < LOW_COVERAGE:
        warnings.append(
            Advisory(
                code="low_instance_coverage",
                message=(
                    f"The {n} instances cover {covered:.0%} of the plant mask. "
                    "The rest got no leaf-sized mask from the model — leaves "
                    "there are NOT in the count. Look at the overlay for "
                    "unoutlined green."
                ),
            )
        )
    instances, overlay = describe_instances(img, mask255, labels_c, y0, x0, px_per_mm)
    labels = np.zeros(mask255.shape, np.int32)
    labels[y0:y1, x0:x1] = labels_c
    return SamLeavesResult(
        leaf_count=n,
        instances=instances,
        mask_coverage=round(covered, 4),
        candidates=len(candidates),
        device=device,
        units={
            "area": "pixels" if px_per_mm is None else "mm2",
            "area_px": "pixels",
            "centroid": "pixels",
            "bbox": "pixels",
        },
        warnings=warnings,
        overlay=overlay,
        labels=labels,
    )
