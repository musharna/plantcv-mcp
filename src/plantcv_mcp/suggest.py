"""Contact sheets that make the channel/method choice informed rather than blind."""

import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import analyze_mask
from .segmentation import OBJECT_TYPES, segment_mask, to_gray


def colorspace_sheet(img: np.ndarray) -> np.ndarray:
    """Grid of L,A,B,H,S,V,C,M,Y,K plus the original — which channel separates plant from background."""
    return pcv.visualize.colorspaces(rgb_img=img, original_img=True)


def threshold_sheet(img: np.ndarray, channel: str) -> np.ndarray:
    """Grid of auto-threshold methods on one channel — which method works here."""
    gray = to_gray(img, channel)
    result = pcv.visualize.auto_threshold_methods(gray_img=gray, grid_img=True)
    return result[0] if isinstance(result, list) else result


def polarity_report(
    img: np.ndarray,
    channel: str,
    method: str = "otsu",
    fill_size: int = 200,
    ambiguity_margin: float = 0.1,
) -> dict:
    """What each object_type would actually yield on THIS image.

    object_type is the single easiest way to get a confidently wrong answer: pick
    the wrong polarity and the mask is the background, while every trait remains
    plausible and correctly united. The server refuses to guess it, so this
    measures both and hands back the numbers.

    'recommended' assumes the subject occupies LESS of the frame than its
    background, which is true of plant photography but not of a macro shot that
    fills the frame. When the two polarities are within `ambiguity_margin` the
    assumption cannot discriminate, so `ambiguous` is set and the caller is told
    to look at the overlay rather than trust the recommendation.
    """
    per_polarity = {}
    for object_type in OBJECT_TYPES:
        mask = segment_mask(
            img, channel, method, object_type=object_type, fill_size=fill_size
        )
        diag = analyze_mask(mask)
        per_polarity[object_type] = {
            "mask_fraction": diag.mask_fraction,
            "component_count": diag.component_count,
        }

    fractions = {k: v["mask_fraction"] for k, v in per_polarity.items()}
    recommended = min(fractions, key=lambda k: fractions[k])
    ambiguous = abs(fractions["dark"] - fractions["light"]) < ambiguity_margin

    report = dict(per_polarity)
    report["recommended"] = recommended
    report["ambiguous"] = ambiguous
    report["basis"] = (
        "The polarity covering less of the frame is assumed to be the plant. "
        "Check the overlay before trusting this on a subject that fills the frame."
        if not ambiguous
        else (
            "Both polarities cover a similar share of the frame, so this cannot "
            "be decided by coverage. Inspect the overlay from segment() under "
            "each object_type."
        )
    )
    return report
