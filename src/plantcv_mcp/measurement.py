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

    # pcv.outputs is PROCESS-GLOBAL. Clearing it outright destroyed the
    # observations of any host application that also uses PlantCV directly, so
    # snapshot and restore instead. We still start from an empty table so our own
    # keyed lookup cannot pick up a foreign group.
    saved = dict(pcv.outputs.observations)
    pcv.outputs.clear()
    try:
        roi = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])
        labeled, n = pcv.create_labels(mask=mask, rois=roi, roi_type="partial")
        pcv.analyze.size(img=img, labeled_mask=labeled, n_labels=n)

        # Select the observation group by explicit key, not by insertion order.
        # PlantCV keys observations by sample_label (default: 'default') + '_1'.
        expected_key = f"{pcv.params.sample_label}_1"
        if expected_key not in pcv.outputs.observations:
            raise KeyError(
                f"Expected observation group '{expected_key}' not found. "
                f"Available keys: {list(pcv.outputs.observations.keys())}. "
                f"This may indicate a change in PlantCV's labeling behavior or "
                f"an incomplete analysis."
            )
        group = pcv.outputs.observations[expected_key]
        return {
            name: {"value": obs.get("value"), "unit": obs.get("label")}
            for name, obs in group.items()
        }
    finally:
        # Restore unconditionally — an exception mid-analysis must not leave the
        # host's observations destroyed.
        pcv.outputs.clear()
        pcv.outputs.observations.update(saved)
