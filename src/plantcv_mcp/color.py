"""Colour-card correction.

Colour traits are only comparable across images if the images are colour-corrected
first: the same leaf under two lighting setups yields two different hues. PlantCV
can detect a Macbeth-style ColorChecker in the frame and correct to a standard
reference, which is what this wraps.

The important behaviour here is the failure mode. If correction is requested and no
card can be found, this RAISES. Silently returning the uncorrected image would hand
back colour traits that look corrected and are not — the same class of confident
wrongness as an inverted mask.
"""

import numpy as np
from plantcv.plantcv import transform as tr


class ColorCardNotFoundError(Exception):
    """Raised when colour correction is requested but no colour card is detected."""


def correct_color(img: np.ndarray) -> np.ndarray:
    """Colour-correct an image against a ColorChecker in the frame.

    Corrects to PlantCV's standard reference matrix, so the result is comparable
    across images and sessions rather than merely internally consistent.

    Raises ColorCardNotFoundError when no card is detected — never falls back to
    the uncorrected image.
    """
    try:
        corrected = tr.auto_correct_color(rgb_img=img)
    except Exception as exc:  # PlantCV raises assorted types when detection fails
        raise ColorCardNotFoundError(
            "Colour correction was requested but no colour card could be detected "
            f"in this image ({type(exc).__name__}: {exc}). Correction needs a "
            "Macbeth-style ColorChecker visible in the frame. Re-run with "
            "color_correct=false to measure the image as shot — but note that "
            "colour traits will then not be comparable with other images."
        ) from exc

    if corrected is None:
        raise ColorCardNotFoundError(
            "Colour correction returned nothing, which means no colour card was "
            "detected. Re-run with color_correct=false to measure as shot."
        )
    return corrected
