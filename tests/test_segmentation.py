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


def _small_dark_object_on_soil():
    """BGR image whose `a` channel is a 1.5%-of-frame dark disc (a ~114) on
    textured soil (a ~130). The F1 field set's young rosettes look like this:
    Otsu splits the soil's own spread and selects ~40% of the frame."""
    rng = np.random.default_rng(0)
    a = rng.normal(130, 4, (200, 200))
    yy, xx = np.mgrid[:200, :200]
    disc = (yy - 100) ** 2 + (xx - 100) ** 2 < 14**2
    a[disc] = rng.normal(114, 3, int(disc.sum()))
    lab = np.dstack(
        [np.full((200, 200), 128), np.clip(a, 0, 255), np.full((200, 200), 128)]
    ).astype(np.uint8)
    import cv2

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), disc


def test_triangle_is_the_triangle_threshold_not_otsu():
    """PlantCV 4.11.3's pcv.threshold.triangle computes a triangle value and
    then thresholds with THRESH_OTSU, so cv2 ignores the value: `triangle`
    was Otsu. On a small object the two differ by an order of magnitude."""
    import cv2

    from plantcv_mcp.segmentation import threshold_mask, to_gray

    img, disc = _small_dark_object_on_soil()
    gray = to_gray(img, "a")
    _, want = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    tri = threshold_mask(img, "a", "triangle", object_type="dark")
    otsu = threshold_mask(img, "a", "otsu", object_type="dark")
    assert np.array_equal(tri > 0, want > 0)
    # positive control in the same test: Otsu really does take the soil here
    assert (otsu > 0).mean() > 0.3
    assert (tri > 0).mean() < 2 * disc.mean()
    assert (tri > 0)[disc].mean() > 0.95
    # and "light" is the complement side of the same threshold
    light = threshold_mask(img, "a", "triangle", object_type="light")
    assert not ((light > 0) & (tri > 0)).any()
