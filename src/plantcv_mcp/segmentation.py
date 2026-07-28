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


def segment_mask(
    img: np.ndarray,
    channel: str,
    method: str,
    object_type: str = "dark",
    fill_size: int = 200,
) -> np.ndarray:
    """Produce a binary mask. Raises on unknown channel or method — never guesses.

    Args:
        img: RGB or BGR image array.
        channel: One of the keys in CHANNELS (e.g., "a", "s").
        method: One of the strings in METHODS (e.g., "otsu", "triangle").
        object_type: "dark" or "light" (default "dark").
        fill_size: size threshold for morphological fill.

    Returns:
        Binary uint8 mask with the same shape as img[:2].

    Raises:
        UnknownChannelError: if channel not in CHANNELS.
        UnknownMethodError: if method not in METHODS.
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
    gray = to_gray(img, channel)
    if method == "otsu":
        mask = pcv.threshold.otsu(gray_img=gray, object_type=object_type)
    elif method == "triangle":
        mask = pcv.threshold.triangle(gray_img=gray, object_type=object_type, xstep=1)
    elif method == "mean":
        mask = pcv.threshold.mean(
            gray_img=gray, ksize=11, offset=2, object_type=object_type
        )
    else:
        mask = pcv.threshold.gaussian(
            gray_img=gray, ksize=11, offset=2, object_type=object_type
        )
    return pcv.fill(bin_img=mask, size=fill_size)
