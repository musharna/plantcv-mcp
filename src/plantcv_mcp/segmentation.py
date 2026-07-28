"""Channel and threshold dispatch onto PlantCV.

The server never picks a channel or method itself — callers pass both
explicitly. suggest_segmentation() exists to make that choice informed.
"""

import numpy as np
from plantcv import plantcv as pcv

CHANNELS: dict[str, str] = {
    "l": "lab",
    "a": "lab",
    "b": "lab",
    "h": "hsv",
    "s": "hsv",
    "v": "hsv",
}

METHODS: tuple[str, ...] = ("otsu", "triangle", "mean", "gaussian")


class UnknownChannelError(Exception):
    """Raised for a channel outside CHANNELS."""


class UnknownMethodError(Exception):
    """Raised for a method outside METHODS."""


def to_gray(img: np.ndarray, channel: str) -> np.ndarray:
    """Convert image to greyscale using the specified channel.

    Args:
        img: RGB or BGR image array.
        channel: One of the keys in CHANNELS.

    Returns:
        Greyscale image as uint8 array.

    Raises:
        UnknownChannelError: if channel is not in CHANNELS.
    """
    if channel not in CHANNELS:
        raise UnknownChannelError(
            f"Unknown channel {channel!r}. Valid channels: {sorted(CHANNELS)}. "
            "Call suggest_segmentation() to see which separates plant from background."
        )
    space = CHANNELS[channel]
    if space == "lab":
        return pcv.rgb2gray_lab(rgb_img=img, channel=channel)
    return pcv.rgb2gray_hsv(rgb_img=img, channel=channel)


OBJECT_TYPES: tuple[str, ...] = ("dark", "light")


class UnknownObjectTypeError(Exception):
    """Raised for an object_type outside OBJECT_TYPES."""


def threshold_mask(
    img: np.ndarray,
    channel: str,
    method: str,
    object_type: str = "dark",
    ksize: int = 11,
    offset: int = 2,
) -> np.ndarray:
    """Threshold only — no morphological fill.

    Kept separate from the fill step so callers can tell "the threshold found
    nothing" apart from "the threshold found something and fill deleted it".
    Collapsing the two made a fill_size problem look like a bad channel choice.

    Args:
        img: RGB or BGR image array.
        channel: One of the keys in CHANNELS (e.g., "a", "s").
        method: One of the strings in METHODS (e.g., "otsu", "triangle").
        object_type: "dark" or "light" — which side of the threshold is the
            object. Getting this wrong yields the background as the mask, so it
            is a first-class choice rather than a hidden default.
        ksize: neighbourhood size for the adaptive methods (mean, gaussian).
        offset: constant subtracted from the local mean (mean, gaussian).

    Raises:
        UnknownChannelError, UnknownMethodError, UnknownObjectTypeError.
    """
    if channel not in CHANNELS:
        raise UnknownChannelError(
            f"Unknown channel {channel!r}. Valid channels: {sorted(CHANNELS)}. "
            "Call suggest_segmentation() to see which separates plant from background."
        )
    if method not in METHODS:
        raise UnknownMethodError(
            f"Unknown method {method!r}. Valid methods: {list(METHODS)}. "
            "Call suggest_segmentation() to compare them on this image."
        )
    if object_type not in OBJECT_TYPES:
        raise UnknownObjectTypeError(
            f"Unknown object_type {object_type!r}. Valid: {list(OBJECT_TYPES)}. "
            "'dark' selects pixels below the threshold, 'light' above it. "
            "Call suggest_segmentation() to see which one yields the plant."
        )
    gray = to_gray(img, channel)
    if method == "otsu":
        return pcv.threshold.otsu(gray_img=gray, object_type=object_type)
    if method == "triangle":
        return pcv.threshold.triangle(gray_img=gray, object_type=object_type, xstep=1)
    if method == "mean":
        return pcv.threshold.mean(
            gray_img=gray, ksize=ksize, offset=offset, object_type=object_type
        )
    return pcv.threshold.gaussian(
        gray_img=gray, ksize=ksize, offset=offset, object_type=object_type
    )


def segment_mask(
    img: np.ndarray,
    channel: str,
    method: str,
    object_type: str = "dark",
    fill_size: int = 200,
    ksize: int = 11,
    offset: int = 2,
) -> np.ndarray:
    """Threshold then fill. Raises on unknown channel, method or object_type.

    fill_size removes speckle, but it removes ANY component smaller than itself —
    including a genuinely small specimen. Callers that need to tell those two
    outcomes apart should use threshold_mask() and pcv.fill() separately, as the
    server does.
    """
    mask = threshold_mask(
        img, channel, method, object_type=object_type, ksize=ksize, offset=offset
    )
    return pcv.fill(bin_img=mask, size=fill_size)
