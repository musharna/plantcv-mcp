import numpy as np
import pytest

from plantcv_mcp.segmentation import UnknownChannelError
from plantcv_mcp.suggest import colorspace_sheet, threshold_sheet


def _img():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    img[50:150, 50:150] = (60, 180, 60)
    return img


def test_colorspace_sheet_returns_exact_shape():
    sheet = colorspace_sheet(_img())
    assert sheet.ndim == 3
    assert sheet.shape == (275, 600, 3)


def test_threshold_sheet_returns_exact_shape():
    sheet = threshold_sheet(_img(), channel="a")
    assert sheet.ndim == 3
    assert sheet.shape == (200, 200, 3)


def test_threshold_sheet_unknown_channel_raises_and_valid_channel_works():
    img = _img()
    # Negative control: invalid channel must raise
    with pytest.raises(UnknownChannelError) as exc:
        threshold_sheet(img, channel="zzz")
    assert "zzz" in str(exc.value)

    # Positive control: valid channel must still work
    sheet = threshold_sheet(img, channel="a")
    assert sheet.shape == (200, 200, 3)
