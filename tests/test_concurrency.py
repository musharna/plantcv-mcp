"""Concurrent measurements must not corrupt each other through `pcv.outputs`.

`pcv.outputs` is PROCESS-GLOBAL. The server's original safety argument was that
mcp 1.x ran synchronous tools inline on the event loop, so two measurements could
never interleave. mcp 2.0 offloads synchronous tools to worker threads
(`mcp/server/mcpserver/utilities/func_metadata.py`: `anyio.to_thread.run_sync`),
which made that argument false without changing a line of this package.

The natural race window is narrow — two plain concurrent calls rarely collide —
so these tests WIDEN it deliberately: `pcv.analyze.size` is wrapped to sleep
after it has written its observations, which is exactly the moment another
thread's `pcv.outputs.clear()` would erase them. Without a lock this fails on
every round; with one it cannot fail.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from plantcv import plantcv as pcv

from plantcv_mcp.measurement import measure_traits
from plantcv_mcp.regions import build_regions, measure_regions

ROUNDS = 10
WINDOW_S = 0.05


def _square_scene(side: int, size: int = 200):
    """A `side`×`side` plant on a `size`×`size` frame; area is side**2 exactly."""
    img = np.full((size, size, 3), 200, np.uint8)
    mask = np.zeros((size, size), np.uint8)
    lo = (size - side) // 2
    img[lo : lo + side, lo : lo + side] = (40, 150, 40)
    mask[lo : lo + side, lo : lo + side] = 255
    return img, mask


@pytest.fixture
def widened_race_window(monkeypatch):
    """Sleep AFTER analyze.size has populated pcv.outputs, before it is read."""
    real = pcv.analyze.size
    seen_threads: set[int] = set()

    def slow_size(**kwargs):
        seen_threads.add(threading.get_ident())
        out = real(**kwargs)
        time.sleep(WINDOW_S)
        return out

    monkeypatch.setattr(pcv.analyze, "size", slow_size)
    return seen_threads


def test_measure_and_measure_regions_do_not_cross_contaminate(widened_race_window):
    """The cross-path pair: whole-image and per-region analyses share the global."""
    img_a, mask_a = _square_scene(40)  # area 1600
    img_b, mask_b = _square_scene(100)  # area 10000
    regions_b = build_regions(
        img_b,
        mask_b,
        mode="rect_grid",
        nrows=1,
        ncols=1,
        coord=(40, 40),
        height=120,
        width=120,
        spacing=(120, 120),
    )

    # Sequential ground truth — the positive control that the scenes measure
    # what they claim BEFORE any concurrency is involved.
    assert measure_traits(img_a, mask_a)["area"]["value"] == 1600.0
    assert measure_regions(img_b, mask_b, regions_b)[0]["traits"]["area"]["value"] == (
        10000.0
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(ROUNDS):
            fa = pool.submit(measure_traits, img_a, mask_a)
            fb = pool.submit(measure_regions, img_b, mask_b, regions_b)
            a, b = fa.result(), fb.result()
            assert a["area"]["value"] == 1600.0, a["area"]
            assert b[0]["measured"] is True, b[0]
            assert b[0]["traits"]["area"]["value"] == 10000.0, b[0]["traits"]["area"]

    # The harness must actually have interleaved on two WORKER threads, or the
    # pass above is vacuous. The ground-truth calls ran on this thread.
    workers = widened_race_window - {threading.get_ident()}
    assert len(workers) == 2, widened_race_window


def test_two_whole_image_measurements_do_not_cross_contaminate(widened_race_window):
    img_a, mask_a = _square_scene(40)
    img_b, mask_b = _square_scene(100)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(ROUNDS):
            fa = pool.submit(measure_traits, img_a, mask_a)
            fb = pool.submit(measure_traits, img_b, mask_b)
            assert fa.result()["area"]["value"] == 1600.0
            assert fb.result()["area"]["value"] == 10000.0
    assert len(widened_race_window) == 2


def test_host_pcv_outputs_state_is_fully_restored():
    """`pcv.outputs.clear()` wipes measurements, images, observations AND
    metadata. Restoring only observations leaves a host application that uses
    PlantCV directly with three of its four tables silently emptied."""
    img, mask = _square_scene(40)
    pcv.outputs.clear()
    pcv.outputs.observations["host"] = {"trait": {"value": 1, "label": "px"}}
    pcv.outputs.measurements["host"] = {"legacy": 1}
    pcv.outputs.images.append("host.png")
    pcv.outputs.metadata["host"] = {"value": ["x"]}
    try:
        measure_traits(img, mask)
        assert pcv.outputs.observations == {
            "host": {"trait": {"value": 1, "label": "px"}}
        }
        assert pcv.outputs.measurements == {"host": {"legacy": 1}}
        assert pcv.outputs.images == ["host.png"]
        assert pcv.outputs.metadata == {"host": {"value": ["x"]}}
    finally:
        pcv.outputs.clear()
