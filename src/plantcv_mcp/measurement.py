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
