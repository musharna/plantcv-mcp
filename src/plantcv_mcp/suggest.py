"""Contact sheets that make the channel/method choice informed rather than blind."""

import numpy as np
from plantcv import plantcv as pcv

from .segmentation import to_gray


def colorspace_sheet(img: np.ndarray) -> np.ndarray:
    """Grid of L,A,B,H,S,V,C,M,Y,K plus the original — which channel separates plant from background."""
    return pcv.visualize.colorspaces(rgb_img=img, original_img=True)


def threshold_sheet(img: np.ndarray, channel: str) -> np.ndarray:
    """Grid of auto-threshold methods on one channel — which method works here."""
    gray = to_gray(img, channel)
    result = pcv.visualize.auto_threshold_methods(gray_img=gray, grid_img=True)
    return result[0] if isinstance(result, list) else result
