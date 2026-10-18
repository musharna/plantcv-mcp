"""Channel and threshold dispatch onto PlantCV.

The server never picks a channel or method itself — callers pass both
explicitly. suggest_segmentation() exists to make that choice informed.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from plantcv import plantcv as pcv

from .limits import require_threshold_params

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
    require_threshold_params(method, ksize, offset)
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


# --- several channels at once -------------------------------------------------
#
# One channel cannot always separate a whole plant from its background: on a
# plant with yellowed leaves, LAB a* finds the green leaves and b* the yellow
# ones, and neither alone finds both (agent benchmark 2026-10-17: the server
# arm added two partial masks by hand and missed q05, q06 and q08, while code
# that combined a hue range with a saturation floor got all of them). A rule is
# one channel test; `combine` joins the rules' masks pixel by pixel.

COMBINES: tuple[str, ...] = ("and", "or")
# The value a rule's min/max compares against is the channel's 8-bit grey, as
# PlantCV's rgb2gray_lab/rgb2gray_hsv return it: l, a, b, s, v are 0-255, and
# h is OpenCV's hue, 0-179 (degrees / 2; pure green is 60).
CHANNEL_RANGE: dict[str, tuple[int, int]] = {
    "l": (0, 255),
    "a": (0, 255),
    "b": (0, 255),
    "h": (0, 179),
    "s": (0, 255),
    "v": (0, 255),
}
MAX_RULES = 6
_RANGE_KEYS = {"channel", "min", "max"}
_AUTO_KEYS = {"channel", "method", "object_type", "ksize", "offset"}


class SegmentRuleError(ValueError):
    """A rules list that is malformed; raised before any pixel is read."""


def _int(value: Any, what: str) -> int:
    # bool is an int subclass; True as a threshold is a bug, not a 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise SegmentRuleError(f"{what} must be an integer, got {value!r}.")
    return value


def validate_rules(
    rules: Sequence[Mapping[str, Any]], combine: str | None
) -> list[dict]:
    """Check every rule and return the normalised list (defaults filled in).

    A rule is EITHER a range, {"channel", "min" and/or "max"}, selecting the
    pixels whose channel value lies in [min, max] inclusive, OR an automatic
    threshold, {"channel", "method", "object_type"} (plus ksize/offset for the
    adaptive methods), exactly as segment() thresholds one channel. Errors
    name the rule by index; nothing is applied here.
    """
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)) or not rules:
        raise SegmentRuleError("rules must be a non-empty list of rule objects.")
    if len(rules) > MAX_RULES:
        raise SegmentRuleError(f"at most {MAX_RULES} rules, got {len(rules)}.")
    if combine is not None and combine not in COMBINES:
        raise SegmentRuleError(
            f"combine must be one of {list(COMBINES)}, got {combine!r}."
        )
    if len(rules) > 1 and combine is None:
        raise SegmentRuleError(
            f"combine is required with {len(rules)} rules: 'or' keeps a pixel any "
            "rule selects (green leaves OR yellow leaves), 'and' keeps a pixel "
            "every rule selects (a hue range AND a saturation floor)."
        )
    out: list[dict] = []
    for i, raw in enumerate(rules):
        if not isinstance(raw, Mapping):
            raise SegmentRuleError(f"rule {i}: must be an object, got {raw!r}.")
        channel = raw.get("channel")
        if not isinstance(channel, str) or channel not in CHANNELS:
            raise SegmentRuleError(
                f"rule {i}: channel must be one of {sorted(CHANNELS)}, got {channel!r}."
            )
        is_range = "min" in raw or "max" in raw
        extra = sorted(set(raw) - (_RANGE_KEYS if is_range else _AUTO_KEYS))
        if extra:
            raise SegmentRuleError(
                f"rule {i}: unexpected key(s) {extra}. A rule is either a range "
                "(channel, min, max) or an automatic threshold (channel, method, "
                "object_type, and ksize/offset for mean/gaussian), not both."
            )
        if is_range:
            lo_ok, hi_ok = CHANNEL_RANGE[channel]
            lo = _int(raw.get("min", lo_ok), f"rule {i}: min")
            hi = _int(raw.get("max", hi_ok), f"rule {i}: max")
            if not lo_ok <= lo <= hi <= hi_ok:
                raise SegmentRuleError(
                    f"rule {i}: need {lo_ok} <= min <= max <= {hi_ok} for channel "
                    f"{channel!r}, got min={lo}, max={hi}."
                    + (" h is OpenCV hue, degrees / 2." if channel == "h" else "")
                )
            if (lo, hi) == (lo_ok, hi_ok):
                raise SegmentRuleError(
                    f"rule {i}: min={lo}, max={hi} spans channel {channel!r}'s whole "
                    "range and selects every pixel; narrow it."
                )
            out.append({"channel": channel, "min": lo, "max": hi})
            continue
        method = raw.get("method")
        if method not in METHODS:
            raise SegmentRuleError(
                f"rule {i}: method must be one of {list(METHODS)}, or give min/max; "
                f"got {method!r}."
            )
        object_type = raw.get("object_type")
        if object_type not in OBJECT_TYPES:
            raise SegmentRuleError(
                f"rule {i}: object_type must be one of {list(OBJECT_TYPES)}, got "
                f"{object_type!r}. It is required: the wrong side of the threshold "
                "is the background."
            )
        rule: dict[str, Any] = {
            "channel": channel,
            "method": method,
            "object_type": object_type,
        }
        if method in ("mean", "gaussian"):
            ksize = _int(raw.get("ksize", 11), f"rule {i}: ksize")
            offset = _int(raw.get("offset", 2), f"rule {i}: offset")
            require_threshold_params(method, ksize, offset)
            rule |= {"ksize": ksize, "offset": offset}
        elif "ksize" in raw or "offset" in raw:
            raise SegmentRuleError(
                f"rule {i}: ksize/offset apply to the 'mean' and 'gaussian' "
                f"methods only, not {method!r}."
            )
        out.append(rule)
    return out


def rule_mask(img: np.ndarray, rule: Mapping[str, Any]) -> np.ndarray:
    """One validated rule's mask, uint8 {0, 255}."""
    if "min" in rule:
        gray = to_gray(img, rule["channel"])
        inside = (gray >= rule["min"]) & (gray <= rule["max"])
        return np.where(inside, 255, 0).astype(np.uint8)
    return threshold_mask(
        img,
        rule["channel"],
        rule["method"],
        object_type=rule["object_type"],
        ksize=rule.get("ksize", 11),
        offset=rule.get("offset", 2),
    )


def rules_mask(
    img: np.ndarray, rules: Sequence[Mapping[str, Any]], combine: str | None
) -> tuple[np.ndarray, list[float]]:
    """The combined mask of validated rules, uint8 {0, 255}, and each rule's own
    coverage (fraction of the frame), so a rule that selected nothing or
    everything is visible in the response, not hidden inside the combination."""
    masks = [rule_mask(img, r) > 0 for r in rules]
    out = masks[0].copy()
    for m in masks[1:]:
        out = (out & m) if combine == "and" else (out | m)
    return np.where(out, 255, 0).astype(np.uint8), [float(m.mean()) for m in masks]


def describe_rules(rules: Sequence[Mapping[str, Any]], combine: str | None) -> str:
    """A one-line name for a rules segmentation, stored as the session's
    channel so a trait table can say how its mask was made."""
    parts = [
        f"{r['min']}<={r['channel']}<={r['max']}"
        if "min" in r
        else f"{r['channel']}:{r['method']}:{r['object_type']}"
        for r in rules
    ]
    return (
        f" {combine.upper()} ".join(parts) if len(parts) > 1 and combine else parts[0]
    )
