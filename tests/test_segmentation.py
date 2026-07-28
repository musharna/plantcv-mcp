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
