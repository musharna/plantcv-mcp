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
