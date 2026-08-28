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
    Advisory,
    MaskDiagnostics,
    analyze_mask,
    assert_not_degenerate,
    segmentation_warnings,
)
from .imaging import digest_bytes, read_image_bytes, render_overlay
from .measurement import isolated_pcv_outputs

LABEL = "thermal"


@dataclass
class ThermalLoad:
    celsius: np.ndarray
    digest: str
    source: str  # "flir", "csv", "npz"


def load_thermal(path: str) -> ThermalLoad:
    data = read_image_bytes(path)
    digest = digest_bytes(data)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        import flyr

        with tempfile.TemporaryDirectory(prefix="plantcv-mcp-flir-") as d:
            copy = os.path.join(d, "frame.jpg")
            with open(copy, "wb") as fh:
                fh.write(data)
            celsius = np.asarray(flyr.unpack(copy).celsius, dtype=np.float64)
        source = "flir"
    elif ext == ".csv":
        celsius = np.loadtxt(io.BytesIO(data), delimiter=",", dtype=np.float64)
        source = "csv"
    elif ext == ".npz":
        with np.load(io.BytesIO(data)) as z:
            if not z.files:
                raise ValueError(f"{path!r} holds no arrays")
            celsius = np.asarray(z[z.files[0]], dtype=np.float64)
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


def _grey(celsius: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(celsius)), float(np.nanmax(celsius))
    scaled = np.zeros_like(celsius) if hi <= lo else (celsius - lo) / (hi - lo)
    grey = (np.nan_to_num(scaled) * 255).astype(np.uint8)
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
    if min_c is None and max_c is None:
        raise ValueError(
            "Give min_c and/or max_c: the band of temperatures that is the plant."
        )
    if min_c is not None and max_c is not None and min_c >= max_c:
        raise ValueError(f"min_c must be below max_c, got {min_c} >= {max_c}")
    load = load or load_thermal(path)
    c = load.celsius
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
    assert_not_degenerate(diag)
    warnings = segmentation_warnings(mask, diag, analyze_mask(pre_fill), fill_size)
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
        overlay=render_overlay(_grey(c), mask),
        diagnostics=diag,
        warnings=warnings,
        frame_range=(float(np.nanmin(c)), float(np.nanmax(c))),
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
    assert_not_degenerate(diag)
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
        frame_range=(float(np.nanmin(c)), float(np.nanmax(c))),
        histogram=histogram,
    )
