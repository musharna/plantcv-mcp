"""Thermal frames: FLIR radiometric JPEGs, CSV and .npz temperature arrays.

A thermal frame is degrees Celsius per pixel from a different sensor than the
RGB camera, so a thermal mask is never borrowed from an RGB session: the plant is
segmented on temperature itself, between `min_c` and `max_c`, and measured with
PlantCV's `analyze.thermal` under the same lock every other analysis takes.

Readers, by extension: `.jpg/.jpeg` via `flyr` (what PlantCV's `readimage(mode=
"thermal")` uses), `.csv` via numpy, `.npz` (first array). The bytes are read
once and hashed; the decoders work on exactly those bytes.
"""

import io
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import (
    THERMAL_REMEDIES,
    Advisory,
    DegenerateMaskError,
    MaskDiagnostics,
    analyze_mask,
    assert_not_degenerate,
    finite_range,
    segmentation_warnings,
    threshold_outside_range_warning,
)
from .imaging import digest_bytes, read_image_bytes, render_overlay
from .limits import require_fill_size
from .measurement import isolated_pcv_outputs

LABEL = "thermal"


@dataclass
class ThermalLoad:
    celsius: np.ndarray
    digest: str
    source: str  # "flir", "csv", "npz"


def _undecodable(path: str, expected: str, exc: BaseException) -> ValueError:
    """The one error a decoder failure becomes, whatever the library raised.

    The bytes are user content, so every way they can be malformed (empty,
    truncated, not a zip, a pickled object array, text that is not numbers) is
    an invalid-input case naming the file and what it should have held — the
    same contract as every other refusal in this module. The library's own
    type and message ride along as the reason; nothing is hidden, only named.
    """
    reason = str(exc).splitlines()[0][:120] if str(exc) else ""
    return ValueError(
        f"Could not decode {path!r} as {expected}: {type(exc).__name__}: {reason}"
    )


def _decode_flir(data: bytes) -> np.ndarray:
    import flyr

    with tempfile.TemporaryDirectory(prefix="plantcv-mcp-flir-") as d:
        copy = os.path.join(d, "frame.jpg")
        with open(copy, "wb") as fh:
            fh.write(data)
        return np.asarray(flyr.unpack(copy).celsius, dtype=np.float64)


def _decode_npz(data: bytes, path: str) -> np.ndarray:
    # Only the library call is inside the try: this module's own refusals
    # below (no arrays, several arrays) must not be re-wrapped as "undecodable".
    try:
        z = np.load(io.BytesIO(data))
    except Exception as exc:  # the decoder's failure, whatever type it picks
        raise _undecodable(path, "a .npz holding one 2-D Celsius array", exc) from exc
    with z:
        names = list(z.files)
        if not names:
            raise ValueError(f"{path!r} holds no arrays")
        if len(names) > 1:
            # Picking one silently would be a guess between candidate
            # frames; the wrong one still measures beautifully.
            raise ValueError(
                f"{path!r} holds {len(names)} arrays ({sorted(names)}); "
                "a thermal .npz must hold exactly one 2-D Celsius array. "
                "Re-save just the temperature frame."
            )
        try:
            return np.asarray(z[names[0]], dtype=np.float64)
        except Exception as exc:  # a member that is not a loadable numeric .npy
            raise _undecodable(
                path, f"a .npz whose array {names[0]!r} is a numeric frame", exc
            ) from exc


def load_thermal(path: str) -> ThermalLoad:
    data = read_image_bytes(path)
    digest = digest_bytes(data)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        try:
            celsius = _decode_flir(data)
        except Exception as exc:
            raise _undecodable(path, "a FLIR radiometric JPEG", exc) from exc
        source = "flir"
    elif ext == ".csv":
        try:
            celsius = np.loadtxt(io.BytesIO(data), delimiter=",", dtype=np.float64)
        except Exception as exc:
            raise _undecodable(path, "a .csv of Celsius values", exc) from exc
        source = "csv"
    elif ext == ".npz":
        celsius = _decode_npz(data, path)
        source = "npz"
    else:
        raise ValueError(
            f"Unsupported thermal file {path!r}: expected a FLIR radiometric .jpg, "
            "a .csv of temperatures, or a .npz array."
        )
    if celsius.ndim != 2:
        raise ValueError(
            f"A thermal frame must be 2-D degrees Celsius; got {celsius.shape}"
        )
    return ThermalLoad(celsius=celsius, digest=digest, source=source)


def grey_frame(celsius: np.ndarray) -> np.ndarray:
    """Min-max scaled BGR rendering of a temperature frame, for overlays."""
    lo, hi = finite_range(celsius, "temperature")
    scaled = np.zeros_like(celsius) if hi <= lo else (celsius - lo) / (hi - lo)
    # Scaled over the finite range; a non-finite pixel is clipped to an end
    # (NaN to 0) instead of stretching the scale until the plant is black.
    grey = (np.clip(np.nan_to_num(scaled), 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


@dataclass
class ThermalSegmentation:
    mask: np.ndarray
    overlay: np.ndarray
    diagnostics: MaskDiagnostics
    warnings: list[Advisory]
    frame_range: tuple[float, float]
    min_c: float | None
    max_c: float | None
    source: str


def segment_thermal(
    path: str,
    min_c: float | None = None,
    max_c: float | None = None,
    fill_size: int = 200,
    load: ThermalLoad | None = None,
) -> ThermalSegmentation:
    require_fill_size(fill_size)
    if min_c is not None and max_c is not None and min_c >= max_c:
        raise ValueError(f"min_c must be below max_c, got {min_c} >= {max_c}")
    load = load or load_thermal(path)
    c = load.celsius
    finite = c[np.isfinite(c)]
    if finite.size == 0:
        raise ValueError(f"No finite temperatures in {path!r}; nothing to segment.")
    lo, hi = finite_range(c, "temperature")
    if min_c is None and max_c is None:
        # Refusing without the range made the first call a blind guess; the
        # percentiles show where the plant (usually the cool tail) sits.
        pct = np.percentile(finite, [5, 25, 50, 75, 95])
        pct_txt = ", ".join(
            f"p{q}={v:.1f}" for q, v in zip((5, 25, 50, 75, 95), pct, strict=True)
        )
        raise ValueError(
            "Give min_c and/or max_c: the band of temperatures that is the plant. "
            f"This frame spans {lo:.1f} to {hi:.1f} C ({pct_txt}). Re-run "
            "segment_thermal() with a band inside that range, e.g. max_c near "
            "the cool tail for a transpiring plant on a warmer background."
        )
    band_lo = lo if min_c is None else min_c
    band_hi = hi if max_c is None else max_c
    if band_lo > hi or band_hi < lo:
        raise ValueError(
            f"The band {band_lo:.1f}-{band_hi:.1f} C lies entirely outside this "
            f"frame's {lo:.1f}-{hi:.1f} C, so it can select nothing. Re-run "
            "segment_thermal() with a band inside the frame range."
        )
    sel = np.isfinite(c)
    if min_c is not None:
        sel &= c >= min_c
    if max_c is not None:
        sel &= c <= max_c
    pre_fill = np.where(sel, 255, 0).astype(np.uint8)
    with isolated_pcv_outputs():
        mask = pcv.fill(bin_img=pre_fill, size=fill_size)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    diag = analyze_mask(mask)
    warnings = segmentation_warnings(
        mask, diag, analyze_mask(pre_fill), fill_size, remedies=THERMAL_REMEDIES
    )
    # The warnings are built BEFORE the degeneracy refusal so that a plant
    # fill_size erased is refused with the fill_size sentence, not blamed on
    # the band (a 150-px cool plant under the default 200 was told to widen
    # the band, which then selects background).
    erased = next((w for w in warnings if w.code == "fill_erased_mask"), None)
    try:
        assert_not_degenerate(diag, remedy=THERMAL_REMEDIES.degenerate)
    except DegenerateMaskError as exc:
        if erased is not None:
            raise DegenerateMaskError(erased.message) from exc
        raise
    rng = threshold_outside_range_warning(
        selects_everything=band_lo <= lo and band_hi >= hi,
        selects_nothing=False,  # refused above, before selection
        what=f"band {band_lo:.1f}-{band_hi:.1f} C",
        lo=lo,
        hi=hi,
        unit=" C",
    )
    if rng:
        warnings.insert(0, rng)
    n_bad = int(c.size - np.isfinite(c).sum())
    if n_bad:
        warnings.append(
            Advisory(
                code="nan_pixels",
                message=(
                    f"{n_bad} of {c.size} pixels are not finite (NaN/Inf) and "
                    "could never be selected by the temperature band. If they "
                    "cluster on the plant, the mask has a hole the diagnostics "
                    "cannot see; check the sensor export."
                ),
            )
        )
    return ThermalSegmentation(
        mask=mask,
        overlay=render_overlay(grey_frame(c), mask),
        diagnostics=diag,
        warnings=warnings,
        frame_range=finite_range(c, "temperature"),
        min_c=min_c,
        max_c=max_c,
        source=load.source,
    )


@dataclass
class ThermalResult:
    temperature: dict[str, float]
    pixel_count: int
    frame_range: tuple[float, float]
    histogram: dict[str, list[float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "temperature": {**self.temperature, "unit": "celsius"},
            "pixel_count": self.pixel_count,
            "frame_range": list(self.frame_range),
        }
        if self.histogram is not None:
            out["histogram"] = self.histogram
        return out


def _value(observations: dict, key: str) -> Any:
    for name, obs in observations.items():
        if (name == LABEL or name.startswith(LABEL + "_")) and key in obs:
            return obs[key]
    return None


def measure_thermal(
    path: str,
    mask: np.ndarray,
    include_histograms: bool = False,
    load: ThermalLoad | None = None,
) -> ThermalResult:
    diag = analyze_mask(mask)
    assert_not_degenerate(diag, remedy=THERMAL_REMEDIES.degenerate)
    load = load or load_thermal(path)
    c = load.celsius
    if c.shape != mask.shape:
        raise ValueError(f"mask {mask.shape} does not match the frame {c.shape}")
    labeled = np.where(mask > 0, 1, 0).astype(np.uint8)
    with isolated_pcv_outputs():
        pcv.analyze.thermal(
            thermal_img=c, labeled_mask=labeled, n_labels=1, bins=100, label=LABEL
        )
        obs = {k: dict(v) for k, v in pcv.outputs.observations.items()}
    temperature = {
        "max": float(_value(obs, "max_temp")["value"]),
        "min": float(_value(obs, "min_temp")["value"]),
        "mean": float(_value(obs, "mean_temp")["value"]),
        "median": float(_value(obs, "median_temp")["value"]),
    }
    histogram = None
    if include_histograms:
        freq = _value(obs, "thermal_frequencies")
        histogram = {
            "bins": [float(b) for b in freq["label"]],
            "counts": [float(f) for f in freq["value"]],
        }
    return ThermalResult(
        temperature=temperature,
        pixel_count=int((mask > 0).sum()),
        frame_range=finite_range(c, "temperature"),
        histogram=histogram,
    )
