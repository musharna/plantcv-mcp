# plantcv-mcp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An MCP server that exposes PlantCV as a measurement instrument — it returns plant trait numbers *and* the segmentation overlay they were computed from, and refuses to return numbers when the segmentation is degenerate.

**Architecture:** Four MCP tools over a session store. `segment` mints a `session_id` and returns an overlay image plus mask diagnostics but **no traits**; `measure` requires that `session_id` to return traits. This split structurally forces the visual evidence into the model's context before a number can be obtained. Pure functions (diagnostics, segmentation, measurement) are separated from I/O (imaging) and state (session) so each is testable alone.

**Tech Stack:** Python 3.11+, `plantcv==4.11.3`, `mcp>=1.28.1,<2`, `opencv-python`, `numpy`, `pytest`. Managed with `uv`.

## Global Constraints

- **Python** `>=3.11`. **PlantCV pinned to `4.11.3`** — trait values may shift across releases.
- **`mcp>=1.28.1,<2`** — 1.28.1 is a security floor (GHSA-vj7q-gjh5-988w, GHSA-jpw9-pfvf-9f58, GHSA-hvrp-rf83-w775), matching `data-aggregator-mcp`.
- **`plantcv.__version__` DOES NOT EXIST.** Always use `importlib.metadata.version("plantcv")`.
- **Never treat `in_bounds` / `object_in_frame` as validity signals.** They are bounds checks. On an all-zero mask PlantCV reports `in_bounds=True`. Compute our own degeneracy gate.
- **Never return traits on a degenerate mask.** Raise instead.
- **Test corpus is `~/bio3d-arena/data/assets/renders` only.** `~/orchid-data` images carry third-party copyright watermarks (e.g. "© Gerrit Verhellen") and must NOT appear in tests, fixtures, examples, or README assets.
- **Fail loud.** No silent fallbacks, no silent downsampling. Any downscale is reported in the response.
- Package name `plantcv-mcp` — verified available on PyPI (404) and GitHub (shape-matched null) 2026-07-28.

**Resolved from spec §8 (open questions):**
- **Session eviction:** LRU, **max 8 sessions**. A session stores the mask (`uint8` HxW), image path, params and image shape — **not** the RGB image, which is re-read from disk on demand.
- **Overlay transport:** overlays are downscaled so the longest edge is **≤ 1024 px**, encoded PNG, and the applied scale factor is reported.

---

## File Structure

- `src/plantcv_mcp/__init__.py` — version export.
- `src/plantcv_mcp/diagnostics.py` — pure functions: component analysis, degeneracy gate, the two warnings. No I/O. **The safety core.**
- `src/plantcv_mcp/imaging.py` — image load, downscale-with-report, overlay rendering, PNG encode.
- `src/plantcv_mcp/session.py` — `SessionStore`, LRU eviction.
- `src/plantcv_mcp/segmentation.py` — channel/method dispatch onto PlantCV.
- `src/plantcv_mcp/measurement.py` — analyze dispatch, trait extraction, degeneracy refusal.
- `src/plantcv_mcp/suggest.py` — contact sheets.
- `src/plantcv_mcp/server.py` — MCP tool registration.
- `tests/` — one test module per source module, plus `test_integration.py`.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `src/plantcv_mcp/__init__.py`, `tests/test_version.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `plantcv_mcp.__version__` (str), `plantcv_mcp.plantcv_version() -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_version.py
from plantcv_mcp import __version__, plantcv_version


def test_package_version_is_a_string():
    assert isinstance(__version__, str)
    assert __version__


def test_plantcv_version_is_pinned_4_11_3():
    # plantcv.__version__ does NOT exist; we must read package metadata.
    assert plantcv_version() == "4.11.3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_version.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plantcv_mcp'`

- [ ] **Step 3: Write pyproject.toml**

```toml
[project]
name = "plantcv-mcp"
version = "0.1.0"
description = "MCP server exposing PlantCV as a measurement instrument — traits plus the segmentation overlay they came from"
readme = "README.md"
license = "MIT"
requires-python = ">=3.11"
dependencies = [
    # 1.28.1 is a SECURITY floor, not a feature floor.
    "mcp>=1.28.1,<2",
    "plantcv==4.11.3",
    "opencv-python>=4.9",
    "numpy>=1.26",
]

[project.scripts]
plantcv-mcp = "plantcv_mcp.server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/plantcv_mcp"]

[dependency-groups]
dev = ["pytest>=8.0"]
```

- [ ] **Step 4: Write the module**

```python
# src/plantcv_mcp/__init__.py
"""plantcv-mcp — PlantCV as an MCP measurement instrument."""

from importlib.metadata import version

__version__ = "0.1.0"


def plantcv_version() -> str:
    """Return the installed PlantCV version.

    plantcv.__version__ does not exist — metadata is the only source.
    """
    return version("plantcv")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv sync && uv run pytest tests/test_version.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/plantcv_mcp/__init__.py tests/test_version.py
git commit -m "feat: project scaffolding with pinned plantcv 4.11.3"
```

---

### Task 2: Component analysis and the degeneracy gate

**Files:**
- Create: `src/plantcv_mcp/diagnostics.py`, `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `component_areas(mask: np.ndarray) -> list[int]` — areas descending, background excluded.
  - `MaskDiagnostics` dataclass with fields `component_count: int`, `areas: list[int]`, `largest_area: int`, `mask_fraction: float`, `major_object_count: int`.
  - `analyze_mask(mask: np.ndarray, major_threshold: float = 0.25) -> MaskDiagnostics`
  - `DegenerateMaskError(Exception)`
  - `assert_not_degenerate(diag: MaskDiagnostics, min_fraction: float = 0.001) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_diagnostics.py
import numpy as np
import pytest

from plantcv_mcp.diagnostics import (
    DegenerateMaskError,
    analyze_mask,
    assert_not_degenerate,
    component_areas,
)


def _mask_with_squares(shape, squares):
    """squares = [(row, col, size), ...] -> uint8 mask with disjoint filled squares."""
    m = np.zeros(shape, dtype=np.uint8)
    for r, c, s in squares:
        m[r : r + s, c : c + s] = 255
    return m


def test_component_areas_descending_excludes_background():
    mask = _mask_with_squares((100, 100), [(0, 0, 10), (50, 50, 5)])
    assert component_areas(mask) == [100, 25]


def test_analyze_mask_reports_fraction_and_counts():
    mask = _mask_with_squares((100, 100), [(0, 0, 10)])
    diag = analyze_mask(mask)
    assert diag.component_count == 1
    assert diag.largest_area == 100
    assert diag.mask_fraction == pytest.approx(100 / 10000)


def test_degenerate_empty_mask_raises_and_valid_mask_does_not():
    # NEGATIVE CASE plus its POSITIVE CONTROL, in the same test, so an
    # always-raises bug cannot masquerade as working detection.
    empty = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(DegenerateMaskError):
        assert_not_degenerate(analyze_mask(empty))

    valid = _mask_with_squares((100, 100), [(0, 0, 30)])
    assert_not_degenerate(analyze_mask(valid))  # must NOT raise


def test_degenerate_below_min_fraction_raises_and_just_above_does_not():
    # 0.1% of 100x100 = 10 px. A 3x3 square (9 px) is below; 4x4 (16 px) is above.
    below = _mask_with_squares((100, 100), [(0, 0, 3)])
    with pytest.raises(DegenerateMaskError):
        assert_not_degenerate(analyze_mask(below))

    above = _mask_with_squares((100, 100), [(0, 0, 4)])
    assert_not_degenerate(analyze_mask(above))  # positive control
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plantcv_mcp.diagnostics'`

- [ ] **Step 3: Write the implementation**

```python
# src/plantcv_mcp/diagnostics.py
"""Pure mask diagnostics. No I/O, no PlantCV — testable in isolation.

This is the safety core. PlantCV's own in_bounds / object_in_frame flags
report True on an all-zero mask, so they cannot be used to detect a failed
segmentation. Everything here computes our own signal instead.
"""

from dataclasses import dataclass

import cv2
import numpy as np


class DegenerateMaskError(Exception):
    """Raised when a mask is too empty to yield meaningful traits."""


@dataclass(frozen=True)
class MaskDiagnostics:
    component_count: int
    areas: list[int]
    largest_area: int
    mask_fraction: float
    major_object_count: int


def component_areas(mask: np.ndarray) -> list[int]:
    """Connected-component areas, descending, background excluded."""
    binary = (mask > 0).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = [int(a) for a in stats[1:, cv2.CC_STAT_AREA]]
    return sorted(areas, reverse=True)


def analyze_mask(mask: np.ndarray, major_threshold: float = 0.25) -> MaskDiagnostics:
    """Summarise a binary mask.

    major_threshold: a component counts as "major" if its area is at least
    this fraction of the largest component's area. 0.25 is a calibrated
    starting value, not a constant — see plan Task 3.
    """
    areas = component_areas(mask)
    largest = areas[0] if areas else 0
    major = sum(1 for a in areas if largest and a >= major_threshold * largest)
    return MaskDiagnostics(
        component_count=len(areas),
        areas=areas,
        largest_area=largest,
        mask_fraction=float((mask > 0).sum()) / float(mask.size),
        major_object_count=major,
    )


def assert_not_degenerate(
    diag: MaskDiagnostics, min_fraction: float = 0.001
) -> None:
    """Raise DegenerateMaskError if traits would be meaningless.

    Degenerate if ANY of: zero components; zero largest area; mask fraction
    below min_fraction. Callers must invoke this BEFORE returning traits —
    PlantCV will happily return 17 zero-valued traits otherwise.
    """
    if diag.component_count == 0 or diag.largest_area == 0:
        raise DegenerateMaskError(
            "Segmentation produced no objects. The mask is empty. "
            "Re-run segment() with a different channel or method."
        )
    if diag.mask_fraction < min_fraction:
        raise DegenerateMaskError(
            f"Mask covers {diag.mask_fraction:.4%} of the frame, below the "
            f"{min_fraction:.2%} minimum. This is almost certainly a failed "
            "segmentation, not a very small plant. Re-run segment() with a "
            "different channel or method."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: mask diagnostics and degeneracy gate"
```

---

### Task 3: Multi-specimen warning

**Files:**
- Modify: `src/plantcv_mcp/diagnostics.py`
- Modify: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `MaskDiagnostics`, `analyze_mask` from Task 2.
- Produces: `Warning` dataclass (`code: str`, `message: str`); `multi_specimen_warning(diag: MaskDiagnostics) -> Warning | None`.

- [ ] **Step 1: Write the failing test**

Add this helper beside `_mask_with_squares` in `tests/test_diagnostics.py`:

```python
def _mask_from_areas(shape, areas):
    """Build a mask with disjoint square blobs of the given pixel areas."""
    m = np.zeros(shape, dtype=np.uint8)
    col = 0
    for a in areas:
        side = int(np.sqrt(a))
        m[0:side, col : col + side] = 255
        col += side + 5  # gap keeps components disjoint
    return m
```

Then the test:

```python
from plantcv_mcp.diagnostics import multi_specimen_warning


def test_multi_specimen_fires_on_measured_failure_and_not_on_single_plant():
    """Calibrated on the real mode-1 failure. The positive control lives in the
    SAME test so an always-fires bug cannot masquerade as detection."""
    # Real measured areas from bio3d-arena/.../736_multi4.png
    four_plants = _mask_from_areas((400, 400), [8628, 7981, 7106, 6748, 570, 454])
    warn = multi_specimen_warning(analyze_mask(four_plants))
    assert warn is not None
    assert warn.code == "multi_specimen"
    assert "auto_grid" in warn.message

    # POSITIVE CONTROL: one plant + disconnected leaf tips -> must NOT fire
    one_plant = _mask_from_areas((400, 400), [8628, 570, 454, 274])
    assert multi_specimen_warning(analyze_mask(one_plant)) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnostics.py -k multi_specimen -v`
Expected: FAIL with `ImportError: cannot import name 'multi_specimen_warning'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/plantcv_mcp/diagnostics.py
@dataclass(frozen=True)
class Warning:
    code: str
    message: str


def multi_specimen_warning(diag: MaskDiagnostics) -> "Warning | None":
    """Warn when the mask holds two or more comparably-sized objects.

    Calibrated on a real failure: a 4-view render segmented to areas
    8628/7981/7106/6748 (all >= 78% of the largest -> 4 major objects) with a
    tail at 570 and below (<= 6.6% -> excluded). A whole-image ROI merges them
    into one "plant" and every size trait becomes meaningless.
    """
    if diag.major_object_count < 2:
        return None
    return Warning(
        code="multi_specimen",
        message=(
            f"{diag.major_object_count} comparably-sized objects detected "
            f"(areas: {diag.areas[: diag.major_object_count]}). A whole-image "
            "ROI will merge them into one object and every size trait will "
            "describe the group, not a plant. Consider roi.auto_grid "
            "(phase 2) or pass an explicit single-plant roi to measure()."
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: multi-specimen warning calibrated on measured failure"
```

---

### Task 4: Frame-clipping warning

**Files:**
- Modify: `src/plantcv_mcp/diagnostics.py`
- Modify: `tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `Warning` from Task 3.
- Produces: `frame_clipping_warning(mask: np.ndarray) -> Warning | None`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_diagnostics.py
from plantcv_mcp.diagnostics import frame_clipping_warning


def test_frame_clipping_fires_when_touching_edge_and_not_when_interior():
    clipped = np.zeros((100, 100), dtype=np.uint8)
    clipped[0:40, 0:40] = 255  # touches top and left edges
    warn = frame_clipping_warning(clipped)
    assert warn is not None
    assert warn.code == "frame_clipping"
    assert "lower bound" in warn.message.lower()

    # POSITIVE CONTROL in the same test: interior object must NOT fire
    interior = np.zeros((100, 100), dtype=np.uint8)
    interior[20:60, 20:60] = 255
    assert frame_clipping_warning(interior) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_diagnostics.py -k frame_clipping -v`
Expected: FAIL with `ImportError: cannot import name 'frame_clipping_warning'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/plantcv_mcp/diagnostics.py
def frame_clipping_warning(mask: np.ndarray) -> "Warning | None":
    """Warn when mask pixels touch the frame edge.

    Computed here rather than trusting PlantCV's in_bounds, which reports True
    on an all-zero mask and so cannot discriminate this case.
    """
    binary = mask > 0
    if not (
        binary[0, :].any()
        or binary[-1, :].any()
        or binary[:, 0].any()
        or binary[:, -1].any()
    ):
        return None
    return Warning(
        code="frame_clipping",
        message=(
            "Plant material touches the frame edge, so it is cut off by the "
            "image boundary. Size traits (area, width, height, perimeter) are "
            "a LOWER BOUND on the true plant, not a measurement of it."
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_diagnostics.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: frame-clipping warning computed independently of in_bounds"
```

---

### Task 5: Imaging — load, reported downscale, overlay

**Files:**
- Create: `src/plantcv_mcp/imaging.py`, `tests/test_imaging.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_image(path: str) -> np.ndarray` (BGR; raises `RuntimeError` naming the path)
  - `downscale(img: np.ndarray, max_edge: int = 1024) -> tuple[np.ndarray, float]`
  - `render_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray`
  - `encode_png(img: np.ndarray) -> bytes`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_imaging.py
import numpy as np
import pytest

from plantcv_mcp.imaging import downscale, encode_png, load_image, render_overlay


def test_load_image_raises_on_missing_file_with_path_in_message():
    with pytest.raises(RuntimeError) as exc:
        load_image("/nonexistent/definitely_not_here.png")
    assert "definitely_not_here.png" in str(exc.value)


def test_downscale_shrinks_large_and_reports_scale():
    big = np.zeros((2048, 1024, 3), dtype=np.uint8)
    out, scale = downscale(big, max_edge=1024)
    assert max(out.shape[:2]) == 1024
    assert scale == pytest.approx(0.5)


def test_downscale_leaves_small_untouched_and_reports_one():
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    out, scale = downscale(small, max_edge=1024)
    assert out.shape == small.shape
    assert scale == 1.0


def test_render_overlay_tints_only_masked_pixels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, :] = 255
    out = render_overlay(img, mask)
    assert out[0, 0].sum() > 0   # masked -> tinted
    assert out[9, 0].sum() == 0  # unmasked -> untouched


def test_encode_png_returns_png_magic_bytes():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert encode_png(img)[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_imaging.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plantcv_mcp.imaging'`

- [ ] **Step 3: Write the implementation**

```python
# src/plantcv_mcp/imaging.py
"""Image I/O and rendering. The only module that touches the filesystem."""

import cv2
import numpy as np
from plantcv import plantcv as pcv

OVERLAY_BGR = np.array([0, 0, 255], dtype=np.float64)  # red in BGR
OVERLAY_ALPHA = 0.55


def load_image(path: str) -> np.ndarray:
    """Load an image as BGR.

    pcv.readimage already raises RuntimeError("Failed to open <path>") for both
    missing and non-image files, so we let it propagate with the path intact
    rather than wrapping it into something vaguer.
    """
    img, _, _ = pcv.readimage(path)
    return img


def downscale(img: np.ndarray, max_edge: int = 1024) -> tuple[np.ndarray, float]:
    """Shrink so the longest edge is <= max_edge. Returns (image, scale).

    Scale is always returned so downsampling is never silent.
    """
    longest = max(img.shape[:2])
    if longest <= max_edge:
        return img, 1.0
    scale = max_edge / longest
    resized = cv2.resize(
        img,
        (int(img.shape[1] * scale), int(img.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def render_overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint masked pixels so a viewer can see what was measured."""
    out = img.copy()
    sel = mask > 0
    out[sel] = (
        (1 - OVERLAY_ALPHA) * out[sel] + OVERLAY_ALPHA * OVERLAY_BGR
    ).astype(np.uint8)
    return out


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to PNG-encode image")
    return buf.tobytes()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_imaging.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/imaging.py tests/test_imaging.py
git commit -m "feat: image load, reported downscale, overlay render"
```

---

### Task 6: Session store with LRU eviction

**Files:**
- Create: `src/plantcv_mcp/session.py`, `tests/test_session.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Session` dataclass: `session_id`, `image_path`, `mask`, `channel`, `method`, `shape`
  - `SessionStore(max_sessions: int = 8)` with `create(image_path, mask, channel, method) -> Session`, `get(session_id) -> Session`, `__len__`
  - `UnknownSessionError(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_session.py
import numpy as np
import pytest

from plantcv_mcp.session import SessionStore, UnknownSessionError


def _store_with(n, max_sessions=8):
    store = SessionStore(max_sessions=max_sessions)
    ids = [
        store.create("/img.png", np.zeros((4, 4), np.uint8), "a", "otsu").session_id
        for _ in range(n)
    ]
    return store, ids


def test_create_then_get_roundtrips():
    store, ids = _store_with(1)
    assert store.get(ids[0]).channel == "a"


def test_unknown_session_id_names_what_was_passed():
    store, _ = _store_with(1)
    with pytest.raises(UnknownSessionError) as exc:
        store.get("bogus-id")
    assert "bogus-id" in str(exc.value)


def test_lru_evicts_oldest_beyond_max():
    store, ids = _store_with(9, max_sessions=8)
    assert len(store) == 8
    with pytest.raises(UnknownSessionError):
        store.get(ids[0])      # oldest evicted
    assert store.get(ids[-1])  # newest survives (positive control)


def test_get_refreshes_recency_so_used_sessions_survive():
    store, ids = _store_with(8, max_sessions=8)
    store.get(ids[0])  # touch the oldest
    store.create("/img.png", np.zeros((4, 4), np.uint8), "a", "otsu")
    assert store.get(ids[0])  # survived because touched
    with pytest.raises(UnknownSessionError):
        store.get(ids[1])     # the new oldest went instead
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plantcv_mcp.session'`

- [ ] **Step 3: Write the implementation**

```python
# src/plantcv_mcp/session.py
"""In-memory session store.

Sessions hold the mask (uint8 HxW) but NOT the RGB image — that is re-read from
disk on demand, keeping memory bounded when several sessions are live.
"""

import uuid
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


class UnknownSessionError(Exception):
    """Raised when a session_id is not in the store."""


@dataclass
class Session:
    session_id: str
    image_path: str
    mask: np.ndarray
    channel: str
    method: str
    shape: tuple[int, int]


class SessionStore:
    def __init__(self, max_sessions: int = 8) -> None:
        self._max = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()

    def create(
        self, image_path: str, mask: np.ndarray, channel: str, method: str
    ) -> Session:
        session = Session(
            session_id=str(uuid.uuid4()),
            image_path=image_path,
            mask=mask,
            channel=channel,
            method=method,
            shape=(int(mask.shape[0]), int(mask.shape[1])),
        )
        self._sessions[session.session_id] = session
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)  # evict least-recently-used
        return session

    def get(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            raise UnknownSessionError(
                f"Unknown session_id {session_id!r}. Sessions are in-memory and "
                f"capped at {self._max}; the oldest are evicted. Re-run segment()."
            )
        self._sessions.move_to_end(session_id)  # refresh recency
        return self._sessions[session_id]

    def __len__(self) -> int:
        return len(self._sessions)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_session.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/session.py tests/test_session.py
git commit -m "feat: session store with LRU eviction"
```

---

### Task 7: Segmentation dispatch

**Files:**
- Create: `src/plantcv_mcp/segmentation.py`, `tests/test_segmentation.py`

**Interfaces:**
- Consumes: nothing (takes an image array).
- Produces:
  - `CHANNELS: dict[str, str]` — keys `l,a,b,h,s,v` mapping to PlantCV colourspace functions.
  - `METHODS: tuple[str, ...]` — `("otsu", "triangle", "mean", "gaussian")`
  - `UnknownChannelError(Exception)`, `UnknownMethodError(Exception)`
  - `segment_mask(img, channel: str, method: str, object_type: str = "dark", fill_size: int = 200) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segmentation.py
import numpy as np
import pytest

from plantcv_mcp.segmentation import (
    CHANNELS,
    METHODS,
    UnknownChannelError,
    UnknownMethodError,
    segment_mask,
)


def _green_blob():
    """A synthetic BGR image with a green square on a grey field."""
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[50:150, 50:150] = (60, 180, 60)  # BGR green
    return img


def test_unknown_channel_names_the_valid_options():
    with pytest.raises(UnknownChannelError) as exc:
        segment_mask(_green_blob(), channel="zzz", method="otsu")
    assert "zzz" in str(exc.value)
    for key in CHANNELS:
        assert key in str(exc.value)


def test_unknown_method_names_the_valid_options():
    with pytest.raises(UnknownMethodError) as exc:
        segment_mask(_green_blob(), channel="a", method="nope")
    assert "nope" in str(exc.value)
    for m in METHODS:
        assert m in str(exc.value)


def test_segment_mask_finds_the_green_blob():
    mask = segment_mask(_green_blob(), channel="a", method="otsu")
    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    # the blob is 100x100 = 10000 px of 40000; expect the mask near that
    assert 5000 < (mask > 0).sum() < 20000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plantcv_mcp.segmentation'`

- [ ] **Step 3: Write the implementation**

```python
# src/plantcv_mcp/segmentation.py
"""Channel and threshold dispatch onto PlantCV.

The server never picks a channel or method itself — callers pass both
explicitly. suggest_segmentation() exists to make that choice informed.
"""

import numpy as np
from plantcv import plantcv as pcv

CHANNELS: dict[str, str] = {
    "l": "lab", "a": "lab", "b": "lab",
    "h": "hsv", "s": "hsv", "v": "hsv",
}

METHODS: tuple[str, ...] = ("otsu", "triangle", "mean", "gaussian")


class UnknownChannelError(Exception):
    """Raised for a channel outside CHANNELS."""


class UnknownMethodError(Exception):
    """Raised for a method outside METHODS."""


def _to_gray(img: np.ndarray, channel: str) -> np.ndarray:
    space = CHANNELS[channel]
    if space == "lab":
        return pcv.rgb2gray_lab(rgb_img=img, channel=channel)
    return pcv.rgb2gray_hsv(rgb_img=img, channel=channel)


def segment_mask(
    img: np.ndarray,
    channel: str,
    method: str,
    object_type: str = "dark",
    fill_size: int = 200,
) -> np.ndarray:
    """Produce a binary mask. Raises on unknown channel or method — never guesses."""
    if channel not in CHANNELS:
        raise UnknownChannelError(
            f"Unknown channel {channel!r}. Valid channels: {sorted(CHANNELS)}. "
            "Call suggest_segmentation() to see which separates plant from background."
        )
    if method not in METHODS:
        raise UnknownMethodError(
            f"Unknown method {method!r}. Valid methods: {list(METHODS)}. "
            "Call suggest_segmentation() to compare them on this image."
        )
    gray = _to_gray(img, channel)
    if method == "otsu":
        mask = pcv.threshold.otsu(gray_img=gray, object_type=object_type)
    elif method == "triangle":
        mask = pcv.threshold.triangle(gray_img=gray, object_type=object_type, xstep=1)
    elif method == "mean":
        mask = pcv.threshold.mean(gray_img=gray, ksize=11, offset=2, object_type=object_type)
    else:
        mask = pcv.threshold.gaussian(gray_img=gray, ksize=11, offset=2, object_type=object_type)
    return pcv.fill(bin_img=mask, size=fill_size)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmentation.py -v`
Expected: 3 passed

> If `test_segment_mask_finds_the_green_blob` fails on pixel count, flip
> `object_type` to `"light"` for the synthetic fixture and record which
> polarity the `a` channel needs — do NOT loosen the assertion bounds, since
> the bounds are what makes the test capable of failing.

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/segmentation.py tests/test_segmentation.py
git commit -m "feat: explicit channel/method segmentation dispatch"
```

---

### Task 8: Measurement with degeneracy refusal

**Files:**
- Create: `src/plantcv_mcp/measurement.py`, `tests/test_measurement.py`

**Interfaces:**
- Consumes: `analyze_mask`, `assert_not_degenerate`, `DegenerateMaskError` (Task 2).
- Produces: `measure_traits(img: np.ndarray, mask: np.ndarray) -> dict[str, dict]` — mapping trait name to `{"value": ..., "unit": ...}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_measurement.py
import numpy as np
import pytest

from plantcv_mcp.diagnostics import DegenerateMaskError
from plantcv_mcp.measurement import measure_traits


def _img_and_mask(fill=True):
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    if fill:
        mask[50:150, 50:150] = 255
    return img, mask


def test_empty_mask_refuses_and_valid_mask_returns_traits():
    """The single most important test in the suite. PlantCV returns 17
    zero-valued traits with in_bounds=True on an empty mask; we must refuse.
    The positive control is in the SAME test."""
    img, empty = _img_and_mask(fill=False)
    with pytest.raises(DegenerateMaskError):
        measure_traits(img, empty)

    img, valid = _img_and_mask(fill=True)
    traits = measure_traits(img, valid)  # must NOT raise
    assert traits["area"]["value"] > 0


def test_traits_carry_units():
    img, mask = _img_and_mask()
    traits = measure_traits(img, mask)
    assert traits["area"]["unit"] == "pixels"
    assert "solidity" in traits


def test_area_matches_the_known_mask_size():
    img, mask = _img_and_mask()  # 100x100 filled square
    traits = measure_traits(img, mask)
    assert traits["area"]["value"] == pytest.approx(10000, rel=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_measurement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plantcv_mcp.measurement'`

- [ ] **Step 3: Write the implementation**

```python
# src/plantcv_mcp/measurement.py
"""Trait extraction, gated on mask validity."""

import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import analyze_mask, assert_not_degenerate


def measure_traits(img: np.ndarray, mask: np.ndarray) -> dict[str, dict]:
    """Return PlantCV size traits for a mask.

    Raises DegenerateMaskError BEFORE calling PlantCV when the mask is empty —
    PlantCV would otherwise return a full 17-trait set of zeros with
    in_bounds=True, which is indistinguishable from a real zero-area plant.
    """
    assert_not_degenerate(analyze_mask(mask))

    pcv.outputs.clear()  # observations accumulate globally; start clean
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
    labeled, n = pcv.create_labels(mask=mask, rois=roi, roi_type="partial")
    pcv.analyze.size(img=img, labeled_mask=labeled, n_labels=n)

    group = next(iter(pcv.outputs.observations.values()))
    return {
        name: {"value": obs.get("value"), "unit": obs.get("label")}
        for name, obs in group.items()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_measurement.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/measurement.py tests/test_measurement.py
git commit -m "feat: trait measurement gated on mask validity"
```

---

### Task 9: Segmentation suggestion contact sheets

**Files:**
- Create: `src/plantcv_mcp/suggest.py`, `tests/test_suggest.py`

**Interfaces:**
- Consumes: `downscale` (Task 5).
- Produces: `colorspace_sheet(img) -> np.ndarray`, `threshold_sheet(img, channel: str) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_suggest.py
import numpy as np

from plantcv_mcp.suggest import colorspace_sheet, threshold_sheet


def _img():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[50:150, 50:150] = (60, 180, 60)
    return img


def test_colorspace_sheet_is_larger_than_the_input():
    sheet = colorspace_sheet(_img())
    assert sheet.ndim == 3
    assert sheet.shape[0] * sheet.shape[1] > 200 * 200


def test_threshold_sheet_returns_an_image():
    sheet = threshold_sheet(_img(), channel="a")
    assert sheet.ndim in (2, 3)
    assert sheet.size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_suggest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plantcv_mcp.suggest'`

- [ ] **Step 3: Write the implementation**

```python
# src/plantcv_mcp/suggest.py
"""Contact sheets that make the channel/method choice informed rather than blind."""

import numpy as np
from plantcv import plantcv as pcv

from .segmentation import _to_gray


def colorspace_sheet(img: np.ndarray) -> np.ndarray:
    """Grid of L,A,B,H,S,V,C,M,Y,K plus the original — which channel separates plant from background."""
    return pcv.visualize.colorspaces(rgb_img=img, original_img=True)


def threshold_sheet(img: np.ndarray, channel: str) -> np.ndarray:
    """Grid of auto-threshold methods on one channel — which method works here."""
    gray = _to_gray(img, channel)
    result = pcv.visualize.auto_threshold_methods(gray_img=gray, grid_img=True)
    return result[0] if isinstance(result, list) else result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_suggest.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/suggest.py tests/test_suggest.py
git commit -m "feat: colorspace and threshold contact sheets"
```

---

### Task 10: MCP server — the four tools

**Files:**
- Create: `src/plantcv_mcp/server.py`, `tests/test_server.py`

**Interfaces:**
- Consumes: everything from Tasks 2–9.
- Produces: `build_server() -> FastMCP`, `main() -> None`, and the four tool callables `suggest_segmentation`, `segment`, `measure`, `list_methods`.

> **Verify the import path first.** Run
> `uv run python -c "from mcp.server.fastmcp import FastMCP, Image; print('ok')"`.
> If it fails, run `uv run python -c "import mcp, pkgutil; print([m.name for m in pkgutil.iter_modules(mcp.__path__)])"`
> and adapt — do NOT guess a second time.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py
import numpy as np
import pytest

from plantcv_mcp.server import _measure_impl, _segment_impl, _store, list_methods_impl


def _write_green_png(tmp_path):
    import cv2
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[50:150, 50:150] = (60, 180, 60)
    p = tmp_path / "green.png"
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

    traits = _measure_impl(seg["session_id"])
    assert traits["traits"]["area"]["value"] > 0


def test_measure_rejects_an_unknown_session_id():
    with pytest.raises(Exception) as exc:
        _measure_impl("not-a-real-session")
    assert "not-a-real-session" in str(exc.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'plantcv_mcp.server'`

- [ ] **Step 3: Write the implementation**

```python
# src/plantcv_mcp/server.py
"""MCP server. Four tools over a session store.

segment() mints a session and returns the overlay but NO traits; measure()
requires that session. The split is deliberate: it forces the visual evidence
into the model's context before a number can be obtained.
"""

from mcp.server.fastmcp import FastMCP, Image

from . import plantcv_version
from .diagnostics import (
    analyze_mask,
    frame_clipping_warning,
    multi_specimen_warning,
)
from .imaging import downscale, encode_png, load_image, render_overlay
from .measurement import measure_traits
from .segmentation import CHANNELS, METHODS, segment_mask
from .session import SessionStore
from .suggest import colorspace_sheet, threshold_sheet

_store = SessionStore()


def list_methods_impl() -> dict:
    return {
        "plantcv_version": plantcv_version(),
        "channels": sorted(CHANNELS),
        "methods": list(METHODS),
        "guidance": (
            "The 'a' channel of LAB separates green tissue from most backgrounds; "
            "'s' of HSV can work on non-green tissue. Call suggest_segmentation() "
            "to compare on your actual image rather than guessing."
        ),
    }


def _segment_impl(image_path: str, channel: str, method: str) -> dict:
    img = load_image(image_path)
    mask = segment_mask(img, channel=channel, method=method)
    diag = analyze_mask(mask)
    warnings = [
        w for w in (multi_specimen_warning(diag), frame_clipping_warning(mask)) if w
    ]
    session = _store.create(image_path, mask, channel, method)
    overlay, scale = downscale(render_overlay(img, mask))
    png = encode_png(overlay)
    return {
        "session_id": session.session_id,
        "mask_fraction": diag.mask_fraction,
        "component_count": diag.component_count,
        "major_object_count": diag.major_object_count,
        "largest_area": diag.largest_area,
        "overlay_scale": scale,
        "overlay_png_bytes": len(png),
        "_png": png,
        "warnings": [{"code": w.code, "message": w.message} for w in warnings],
    }


def _measure_impl(session_id: str) -> dict:
    session = _store.get(session_id)
    img = load_image(session.image_path)  # re-read; sessions do not hold RGB
    return {"session_id": session_id, "traits": measure_traits(img, session.mask)}


def build_server() -> FastMCP:
    mcp = FastMCP("plantcv-mcp")

    @mcp.tool()
    def list_methods() -> dict:
        """List available segmentation channels, methods, and the pinned PlantCV version."""
        return list_methods_impl()

    @mcp.tool()
    def suggest_segmentation(image_path: str, channel: str = "a") -> list:
        """Return colourspace and threshold contact sheets so the channel/method
        choice is informed rather than blind. Call this before segment()."""
        img = load_image(image_path)
        cs, _ = downscale(colorspace_sheet(img))
        th, _ = downscale(threshold_sheet(img, channel))
        return [
            Image(data=encode_png(cs), format="png"),
            Image(data=encode_png(th), format="png"),
        ]

    @mcp.tool()
    def segment(image_path: str, channel: str, method: str) -> list:
        """Segment an image. Returns the overlay image and mask diagnostics —
        NOT traits. Use the returned session_id with measure() to get traits."""
        result = _segment_impl(image_path, channel, method)
        png = result.pop("_png")
        return [str(result), Image(data=png, format="png")]

    @mcp.tool()
    def measure(session_id: str) -> dict:
        """Return plant traits for a segmentation produced by segment().
        Raises if the mask is degenerate rather than returning zeros."""
        return _measure_impl(session_id)

    return mcp


def main() -> None:
    build_server().run()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/plantcv_mcp/server.py tests/test_server.py
git commit -m "feat: MCP server with segment/measure split"
```

---

### Task 11: Real-execution test against a real image file

**Files:**
- Create: `tests/fixtures/README.md`, `tests/test_integration.py`
- Copy: one render from `~/bio3d-arena/data/assets/renders/` into `tests/fixtures/`

**Interfaces:**
- Consumes: `_segment_impl`, `_measure_impl` (Task 10).
- Produces: nothing (test-only).

> **Copyright:** copy ONLY from `~/bio3d-arena/data/assets/renders/`. Never use
> `~/orchid-data` — those carry third-party watermarks.

- [ ] **Step 1: Copy the fixture and document its provenance**

```bash
mkdir -p tests/fixtures
cp ~/bio3d-arena/data/assets/renders/736_multi4.png tests/fixtures/multi_specimen.png
cat > tests/fixtures/README.md <<'EOF'
# Test fixtures

`multi_specimen.png` — copied from bio3d-arena (our own render, no third-party
rights). It is a FOUR-VIEW panel: segmenting it with a whole-image ROI merges
four plants into one object. Measured with PlantCV 4.11.3: 9 connected
components, top-four areas 8628 / 7981 / 7106 / 6748. It exists to prove the
multi-specimen warning fires on a real, known-bad input.

Do NOT add images from ~/orchid-data — third-party copyright watermarks.
EOF
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_integration.py
"""Real execution: drives the full tool path against a real file on disk."""

from pathlib import Path

import pytest

from plantcv_mcp.server import _measure_impl, _segment_impl

FIXTURE = Path(__file__).parent / "fixtures" / "multi_specimen.png"


def test_real_render_fires_multi_specimen_warning_end_to_end():
    result = _segment_impl(str(FIXTURE), channel="a", method="otsu")
    codes = {w["code"] for w in result["warnings"]}
    assert "multi_specimen" in codes, (
        "The multi-specimen guard did not fire on a known four-view render. "
        f"diagnostics: {result['component_count']} components, "
        f"{result['major_object_count']} major objects."
    )
    assert result["major_object_count"] >= 2
    traits = _measure_impl(result["session_id"])["traits"]
    assert traits["area"]["value"] > 0


def test_real_render_segmentation_is_not_degenerate():
    """Positive control for the integration path: a real image must produce a
    usable mask, so a always-degenerate bug cannot hide behind the warning test."""
    result = _segment_impl(str(FIXTURE), channel="a", method="otsu")
    assert result["mask_fraction"] > 0.001
    assert result["component_count"] > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_integration.py -v`
Expected: FAIL (fixture missing) until Step 1 is done; then PASS.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_integration.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures tests/test_integration.py
git commit -m "test: real-execution integration against a known four-view render"
```

---

### Task 12: Mutation checks and determinism

**Files:**
- Create: `tests/test_determinism.py`, `docs/MUTATION-CHECKS.md`

**Interfaces:**
- Consumes: `_segment_impl` (Task 10), `measure_traits` (Task 8).
- Produces: nothing (test/doc only).

- [ ] **Step 1: Write the determinism test**

```python
# tests/test_determinism.py
from pathlib import Path

from plantcv_mcp.server import _measure_impl, _segment_impl

FIXTURE = Path(__file__).parent / "fixtures" / "multi_specimen.png"


def test_same_image_and_params_give_identical_traits():
    a = _measure_impl(_segment_impl(str(FIXTURE), "a", "otsu")["session_id"])["traits"]
    b = _measure_impl(_segment_impl(str(FIXTURE), "a", "otsu")["session_id"])["traits"]
    assert a == b, "Identical inputs produced different traits — not deterministic."
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_determinism.py -v`
Expected: 1 passed

- [ ] **Step 3: Run the mutation checks by hand and record the results**

A guard whose test passes with the guard disabled is not a test. For each guard,
break it, confirm the named test goes RED, then restore.

```bash
# 1. Degeneracy gate — comment out the body of assert_not_degenerate()
#    EXPECT RED: tests/test_measurement.py::test_empty_mask_refuses_and_valid_mask_returns_traits
# 2. Multi-specimen — change `if diag.major_object_count < 2:` to `< 99:`
#    EXPECT RED: tests/test_diagnostics.py::test_multi_specimen_fires_on_measured_failure_and_not_on_single_plant
#               tests/test_integration.py::test_real_render_fires_multi_specimen_warning_end_to_end
# 3. Frame clipping — make frame_clipping_warning() return None unconditionally
#    EXPECT RED: tests/test_diagnostics.py::test_frame_clipping_fires_when_touching_edge_and_not_when_interior
uv run pytest -v
```

- [ ] **Step 4: Write `docs/MUTATION-CHECKS.md`**

```markdown
# Mutation checks

Each guard was disabled and its test confirmed to fail. A guard whose test
passes with the guard removed is not a test.

| guard | mutation applied | tests that went red | date |
|---|---|---|---|
| degeneracy gate | body of `assert_not_degenerate` removed | `test_empty_mask_refuses_and_valid_mask_returns_traits` | (fill in) |
| multi-specimen | threshold `< 2` changed to `< 99` | `test_multi_specimen_fires_...`, `test_real_render_fires_...` | (fill in) |
| frame clipping | `frame_clipping_warning` returns None | `test_frame_clipping_fires_...` | (fill in) |

Re-run these whenever a guard's logic changes.
```

- [ ] **Step 5: Commit**

```bash
git add tests/test_determinism.py docs/MUTATION-CHECKS.md
git commit -m "test: determinism plus recorded mutation checks for all three guards"
```

---

### Task 13: README and packaging

**Files:**
- Create: `README.md`, `LICENSE`

**Interfaces:**
- Consumes: `list_methods_impl` (Task 10).
- Produces: nothing.

- [ ] **Step 1: Write the README**

```markdown
# plantcv-mcp

PlantCV as an MCP **measurement instrument**: it returns plant trait numbers
**and the segmentation overlay they were computed from**, and refuses to return
numbers when the segmentation is degenerate.

## Why the two-step API

`segment()` returns an overlay and diagnostics but **no traits**. `measure()`
requires the `session_id` that `segment()` mints. You cannot get a number
without first being handed the picture it came from.

This is not a style preference. Measured on real images with PlantCV 4.11.3:

| failure | what you get without the overlay |
|---|---|
| four-view render, whole-image ROI | 17 plausible traits describing four merged plants |
| plant clipped by the frame | size traits that are silently lower bounds |
| empty mask | 17 traits of zeros, with PlantCV reporting `in_bounds=True` |

All three produce correctly-united, entirely believable numbers.

## Install

```bash
uv add plantcv-mcp
```

## Tools

- `suggest_segmentation(image_path, channel)` — colourspace and threshold contact sheets
- `segment(image_path, channel, method)` — overlay + diagnostics + warnings, no traits
- `measure(session_id)` — traits, or a raised error on a degenerate mask
- `list_methods()` — channels, methods, pinned PlantCV version

## Limitations

Phase 1 is single-ROI. Multi-plant grids, morphology traits (leaf angles, stem,
skeleton) and iterative mask refinement are phase 2.
```

- [ ] **Step 2: Add the MIT LICENSE file**

Use the standard MIT text with `Copyright (c) 2026 Jaret Arnold`.

- [ ] **Step 3: Verify the package builds**

Run: `uv build`
Expected: wheel and sdist produced in `dist/` without error.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add README.md LICENSE
git commit -m "docs: README and MIT license"
```

---

## Plan self-review

**Spec coverage** — every spec section maps to a task: §3 tool surface → Tasks 7–10;
§4 warnings → Tasks 3–4; §5 error handling → Tasks 2, 5, 6, 8; §6 testing → Tasks 11–12;
§8 open questions → all three resolved (name verified 2026-07-28; eviction in Task 6;
overlay downscale in Task 5).

**Deliberately deferred** — §7 non-goals (morphology, refine, auto_grid, batch,
hyperspectral) are out of phase 1 by design.

**Known gaps to watch during execution:**
1. The `mcp.server.fastmcp` import path is unverified — Task 10 opens with a check rather than an assumption.
2. `object_type` polarity in Task 7 may need flipping for the synthetic fixture; the plan says record the answer, not loosen the assertion.
3. The 25% multi-specimen threshold is calibrated on ONE image. Re-calibrate against more renders during Task 11; treat it as tunable.
