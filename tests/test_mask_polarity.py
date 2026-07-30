"""Tests for the two-sided mask-validity model and the exposed segmentation tunables.

The bug these cover: the validity model encoded "invalid = too small", so an
inverted mask — the dominant failure of any threshold — could not be expressed as
a failure at all. Measured before the fix: channel 's' on the fixture gave
mask_fraction 0.961 and measure() returned area=1007829 (the whole frame) with no
error. Every test here pairs the failing case with a passing positive control in
the same function, so an always-fires guard cannot masquerade as detection.
"""

import json

import numpy as np
import pytest

from plantcv_mcp.diagnostics import (
    analyze_mask,
    frame_clipping_warning,
    implausible_coverage_warning,
)
from plantcv_mcp.imaging import load_image
from plantcv_mcp.segmentation import segment_mask, threshold_mask

FIXTURE = "tests/fixtures/multi_specimen.png"


def _mostly_full(shape=(200, 200), fraction=0.96):
    """A mask covering `fraction` of the frame — the shape an inverted mask has."""
    mask = np.zeros(shape, np.uint8)
    rows = int(shape[0] * fraction)
    mask[:rows, :] = 255
    return mask


def _small_object(shape=(200, 200)):
    """A compact object well inside the frame — a plausible plant."""
    mask = np.zeros(shape, np.uint8)
    mask[80:120, 80:120] = 255
    return mask


def test_implausible_coverage_fires_on_inverted_and_not_on_a_plant():
    inverted = analyze_mask(_mostly_full())
    plant = analyze_mask(_small_object())

    assert implausible_coverage_warning(inverted) is not None, (
        "a mask covering 96% of the frame must be flagged"
    )
    # Positive control in the same test: a real plant must NOT trip it, or an
    # always-fires bug would look like working detection.
    assert implausible_coverage_warning(plant) is None, (
        "a 4% plant mask must not be flagged as implausible coverage"
    )


def test_implausible_coverage_names_the_remedy():
    w = implausible_coverage_warning(analyze_mask(_mostly_full()))
    assert w.code == "implausible_coverage"
    assert "object_type" in w.message, (
        "the warning must name the parameter that fixes it"
    )


def test_frame_clipping_is_not_claimed_when_coverage_is_implausible():
    """frame_clipping says size traits are a LOWER BOUND, which implies a real
    plant is cut off. On an inverted mask that is actively misleading, so the
    server must withhold it — while still reporting it on a genuine clip."""
    from plantcv_mcp.server import _segment_impl

    # The raw check does fire on a mask that fills the frame: it truly does
    # touch every edge. Suppression therefore has to happen at the server.
    assert frame_clipping_warning(_mostly_full()) is not None

    inverted = _segment_impl(FIXTURE, "s", "otsu", object_type="dark")
    codes = [w["code"] for w in inverted["warnings"]]
    assert "implausible_coverage" in codes, f"expected inversion warning, got {codes}"
    assert "frame_clipping" not in codes, (
        f"frame_clipping must be withheld on an inverted mask, got {codes}"
    )

    # Positive control in the same test: a genuinely clipped plant, whose mask is
    # small enough to be plausible, MUST still report frame_clipping — otherwise
    # this suppression would be silently swallowing a real advisory.
    import tempfile

    import cv2

    img = np.full((200, 200, 3), 240, np.uint8)
    img[120:200, 60:140] = (40, 160, 40)  # 16% of frame, running off the bottom edge
    d = tempfile.mkdtemp()
    p = f"{d}/clipped.png"
    cv2.imwrite(p, img)

    clipped = _segment_impl(p, "a", "otsu")
    clipped_codes = [w["code"] for w in clipped["warnings"]]
    assert clipped["mask_fraction"] < 0.5, "control mask must be plausibly sized"
    assert "frame_clipping" in clipped_codes, (
        f"a real edge-touching plant must still be flagged, got {clipped_codes}"
    )


def test_object_type_is_reachable_and_changes_the_mask():
    img = load_image(FIXTURE)
    dark = analyze_mask(segment_mask(img, "s", "otsu", object_type="dark"))
    light = analyze_mask(segment_mask(img, "s", "otsu", object_type="light"))
    assert dark.mask_fraction > 0.5, (
        "channel 's' with object_type=dark is the background"
    )
    assert light.mask_fraction < 0.2, "channel 's' with object_type=light is the plant"


def test_server_honours_object_type_end_to_end():
    """The P0 was that _segment_impl hardcoded object_type='dark'. Testing
    segment_mask alone would not catch a regression there, because the parameter
    could be correct in the library and dropped at the server."""
    from plantcv_mcp.server import _segment_impl

    dark = _segment_impl(FIXTURE, "s", "otsu", object_type="dark")
    light = _segment_impl(FIXTURE, "s", "otsu", object_type="light")

    assert dark["mask_fraction"] > 0.5, "dark on 's' is the background"
    assert light["mask_fraction"] < 0.2, "light on 's' is the plant"
    assert light["object_type"] == "light", "the response must echo what was used"


def test_server_honours_fill_size_end_to_end():
    """Same argument for fill_size: correct in the library, droppable at the server."""
    import tempfile

    import cv2

    from plantcv_mcp.server import _segment_impl

    img = np.full((100, 100, 3), 240, np.uint8)
    img[44:56, 44:56] = (40, 160, 40)  # 144 px
    d = tempfile.mkdtemp()
    p = f"{d}/small.png"
    cv2.imwrite(p, img)

    erased = _segment_impl(p, "a", "otsu")  # default fill_size=200
    kept = _segment_impl(p, "a", "otsu", fill_size=10)

    assert erased["component_count"] == 0, "default fill_size erases a 144 px object"
    assert kept["component_count"] >= 1, "a smaller fill_size must preserve it"


def test_threshold_mask_is_separable_from_fill():
    """The server needs the pre-fill mask to tell 'nothing was found' apart from
    'fill deleted what was found'."""
    img = np.full((100, 100, 3), 240, np.uint8)
    img[44:56, 44:56] = (40, 160, 40)  # 144 px, under the default fill_size of 200

    pre = threshold_mask(img, "a", "otsu", object_type="dark")
    post = segment_mask(img, "a", "otsu", object_type="dark")  # default fill_size

    assert analyze_mask(pre).component_count >= 1, "threshold alone must find the blob"
    assert analyze_mask(post).component_count == 0, "default fill_size erases it"


def test_segment_reports_fill_erasure_by_name_not_as_a_bad_channel_choice():
    import tempfile

    import cv2

    from plantcv_mcp.server import _segment_impl

    img = np.full((100, 100, 3), 240, np.uint8)
    img[44:56, 44:56] = (40, 160, 40)
    d = tempfile.mkdtemp()
    p = f"{d}/small.png"
    cv2.imwrite(p, img)

    r = _segment_impl(p, "a", "otsu")
    codes = [w["code"] for w in r["warnings"]]
    assert "fill_erased_mask" in codes, f"expected fill_erased_mask, got {codes}"
    msg = next(w["message"] for w in r["warnings"] if w["code"] == "fill_erased_mask")
    assert "fill_size" in msg, "the warning must name fill_size as the cause"

    # Positive control: the real fixture must NOT report fill erasure.
    r2 = _segment_impl(FIXTURE, "a", "otsu")
    assert "fill_erased_mask" not in [w["code"] for w in r2["warnings"]]


def test_segment_warns_on_a_wholly_empty_mask():
    import tempfile

    import cv2

    from plantcv_mcp.server import _segment_impl

    d = tempfile.mkdtemp()
    blank = f"{d}/blank.png"
    cv2.imwrite(blank, np.full((200, 200, 3), 255, np.uint8))

    r = _segment_impl(blank, "a", "otsu")
    assert r["component_count"] == 0
    assert "empty_mask" in [w["code"] for w in r["warnings"]], (
        "segment() must say the segmentation failed, not just report zero silently"
    )

    # Positive control: a real image must not be called empty.
    r2 = _segment_impl(FIXTURE, "a", "otsu")
    assert "empty_mask" not in [w["code"] for w in r2["warnings"]]


def test_measure_preserves_foreign_plantcv_observations():
    """measure_traits used to call pcv.outputs.clear(), destroying the state of any
    host process that also uses PlantCV directly."""
    from plantcv import plantcv as pcv

    from plantcv_mcp.measurement import measure_traits

    img = load_image(FIXTURE)
    mask = segment_mask(img, "a", "otsu")

    pcv.outputs.clear()
    pcv.outputs.add_observation(
        sample="host_sample",
        variable="host_var",
        trait="a host measurement",
        method="host",
        scale="none",
        datatype=int,
        value=1234,
        label="none",
    )

    traits = measure_traits(img, mask)

    # Positive control in the same test: our own measurement still works.
    assert traits["area"]["value"] > 0, "our traits must still be produced"
    assert "host_sample" in pcv.outputs.observations, (
        "foreign observations must survive measure_traits"
    )
    assert pcv.outputs.observations["host_sample"]["host_var"]["value"] == 1234, (
        "foreign observation value must be unchanged"
    )


def test_measure_detects_a_same_shape_content_swap():
    """The stale-image guard was shape-only, so swapping in a different image of
    identical dimensions silently measured the old mask against new content."""
    import tempfile

    import cv2

    from plantcv_mcp.server import (
        ImageChangedSinceSegmentationError,
        _measure_impl,
        _segment_impl,
    )

    d = tempfile.mkdtemp()
    p = f"{d}/swap.png"
    original = cv2.imread(FIXTURE)
    cv2.imwrite(p, original)

    r = _segment_impl(p, "a", "otsu")
    sid = r["session_id"]

    # Same dimensions, different content.
    swapped = original.copy()
    swapped[:, :] = (10, 200, 10)
    assert swapped.shape == original.shape
    cv2.imwrite(p, swapped)

    with pytest.raises(ImageChangedSinceSegmentationError):
        _measure_impl(sid)


def test_measure_still_works_when_the_file_is_untouched():
    """Positive control for the content guard: an unmodified file must measure."""
    from plantcv_mcp.server import _measure_impl, _segment_impl

    r = _segment_impl(FIXTURE, "a", "otsu")
    out = _measure_impl(r["session_id"])
    assert out["traits"]["area"]["value"] > 0


def test_suggest_reports_both_polarities_so_the_choice_is_informed():
    import asyncio

    from plantcv_mcp.server import build_server

    server = build_server()
    res = asyncio.run(
        server.call_tool(
            "suggest_segmentation", {"image_path": FIXTURE, "channel": "s"}
        )
    )
    # .content since mcp 2.x: call_tool returns a CallToolResult. The old
    # isinstance sniff would fall through here and hand back the result object.
    blocks = res.content
    payload = json.loads(
        next(b for b in blocks if getattr(b, "type", None) == "text").text
    )

    assert "polarity" in payload, "suggest must report what each object_type would give"
    assert payload["polarity"]["dark"]["mask_fraction"] > 0.5
    assert payload["polarity"]["light"]["mask_fraction"] < 0.2
    assert payload["polarity"]["recommended"] == "light"
