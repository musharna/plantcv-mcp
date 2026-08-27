"""Hyperspectral cubes: reading, calibration, spectral indices, reflectance.

Everything measured on PlantCV 4.11.3 (2026-08-27) that shapes this module:

* `read_data` wants the RAW path and derives `<name>.hdr` by stripping the
  extension; handed the `.hdr` itself it reads the header as pixels. This loader
  accepts `.raw`, `.hdr`, or the bare name and resolves both files itself.
* Cubes are commonly integer counts (uint16). `spectral_index.*` computes on the
  array as-is, so `(r800 - r670)` on uint16 WRAPS AROUND: NDVI on a synthetic
  count cube read 65.3 on the background, on a [-1, 1] index. No index here is
  ever computed on integer data: with white/dark references the cube is
  calibrated to reflectance (float64); without, it is cast to float64 and the
  session carries `uncalibrated_cube` — the numbers are relative, not reflectance.
* `hyperspectral.calibrate` with white == dark clips to 1.0 rather than yielding
  NaN, so a degenerate reference pair would silently produce a flat cube. It is
  refused here before PlantCV sees it.
* `_package_index` keeps the true index in `array_data` and a 0-255 rescale in
  `pseudo_rgb`; thresholds and statistics use `array_data`, the overlay uses the
  cube's pseudo-RGB.

The read-once-hash-decode rule (imaging.py) holds: both files' bytes are read
once and hashed; PlantCV decodes a private copy of exactly those bytes.
"""

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from plantcv import plantcv as pcv

from .diagnostics import (
    Advisory,
    MaskDiagnostics,
    analyze_mask,
    assert_not_degenerate,
    segmentation_warnings,
)
from .imaging import read_image_bytes, render_overlay
from .measurement import isolated_pcv_outputs

LABEL = "hsi"
INDEX_DISTANCE = 20  # PlantCV's leniency (nm) in finding the bands an index needs


class CalibrationDegenerateError(Exception):
    """white - dark is not positive somewhere: reflectance would be undefined."""


class IndexUnavailableError(Exception):
    """The index is unknown, or the cube's wavelengths cannot support it."""


def available_indices() -> list[str]:
    return sorted(
        n
        for n in dir(pcv.spectral_index)
        if not n.startswith("_") and callable(getattr(pcv.spectral_index, n))
    )


@dataclass
class CubeLoad:
    cube: Any  # plantcv Spectral_data
    digest: str
    raw_path: str
    hdr_path: str


def _resolve_pair(path: str) -> tuple[str, str]:
    """(raw, hdr) for a .raw, .hdr, or bare ENVI name; both must exist."""
    stem, ext = os.path.splitext(path)
    candidates_raw = [path] if ext.lower() not in {".hdr"} else []
    candidates_raw += [stem + ".raw", stem, path + ".raw"]
    raw = next(
        (c for c in candidates_raw if os.path.isfile(c) and not c.endswith(".hdr")),
        None,
    )
    hdr = next(
        (
            c
            for c in (stem + ".hdr", path + ".hdr", path)
            if c.endswith(".hdr") and os.path.isfile(c)
        ),
        None,
    )
    if raw is None or hdr is None:
        raise FileNotFoundError(
            f"An ENVI cube needs a raw file and its .hdr header side by side; for "
            f"{path!r} found raw={raw!r}, hdr={hdr!r}. PlantCV derives the header "
            "name by stripping the raw file's extension."
        )
    return raw, hdr


def load_cube(path: str) -> CubeLoad:
    raw_path, hdr_path = _resolve_pair(path)
    raw_bytes = read_image_bytes(raw_path)
    hdr_bytes = read_image_bytes(hdr_path)
    h = hashlib.sha256()
    h.update(hdr_bytes)
    h.update(b"\0")
    h.update(raw_bytes)
    digest = h.hexdigest()
    # Decode a private copy of exactly the bytes that were hashed.
    with tempfile.TemporaryDirectory(prefix="plantcv-mcp-envi-") as d:
        base = os.path.join(d, "cube")
        with open(base + ".raw", "wb") as fh:
            fh.write(raw_bytes)
        with open(base + ".hdr", "wb") as fh:
            fh.write(hdr_bytes)
        cube = pcv.readimage(base + ".raw", mode="envi")
    cube.filename = raw_path
    return CubeLoad(cube=cube, digest=digest, raw_path=raw_path, hdr_path=hdr_path)


def _as_float(cube: Any) -> Any:
    from plantcv.plantcv import Spectral_data

    data = cube.array_data.astype(np.float64)
    return Spectral_data(
        array_data=data,
        max_wavelength=cube.max_wavelength,
        min_wavelength=cube.min_wavelength,
        max_value=float(data.max()),
        min_value=float(data.min()),
        d_type=np.float64,
        wavelength_dict=cube.wavelength_dict,
        samples=cube.samples,
        lines=cube.lines,
        interleave=cube.interleave,
        wavelength_units=cube.wavelength_units,
        array_type=cube.array_type,
        pseudo_rgb=cube.pseudo_rgb,
        filename=cube.filename,
        default_bands=cube.default_bands,
        metadata=getattr(cube, "metadata", None),
    )


def prepare_cube(
    cube: Any, white_reference: str | None, dark_reference: str | None
) -> tuple[Any, str, list[Advisory]]:
    """Return (float cube, calibration label, advisories). Never integer data."""
    if (white_reference is None) != (dark_reference is None):
        raise ValueError(
            "white_reference and dark_reference must be given together; a "
            "single reference cannot calibrate."
        )
    if white_reference is not None and dark_reference is not None:
        white = load_cube(white_reference).cube
        dark = load_cube(dark_reference).cube
        span = np.mean(white.array_data, axis=0, keepdims=True).astype(
            np.float64
        ) - np.mean(dark.array_data, axis=0, keepdims=True).astype(np.float64)
        if not np.all(span > 0):
            raise CalibrationDegenerateError(
                "white_reference - dark_reference is not positive at every band and "
                f"column (min {float(span.min())}), so reflectance is undefined there. "
                "PlantCV would clip this to 1.0 silently; check that the white and "
                "dark references are the right files and not the same one."
            )
        calibrated = pcv.hyperspectral.calibrate(
            raw_data=cube, white_reference=white, dark_reference=dark
        )
        if not np.issubdtype(calibrated.array_data.dtype, np.floating):
            calibrated = _as_float(calibrated)
        calibrated.pseudo_rgb = cube.pseudo_rgb
        return calibrated, "white/dark", []
    warnings = []
    if not np.issubdtype(cube.array_data.dtype, np.floating):
        warnings.append(
            Advisory(
                code="uncalibrated_cube",
                message=(
                    f"The cube holds integer counts ({cube.array_data.dtype}, max "
                    f"{float(cube.array_data.max()):g}), not reflectance. Indices are "
                    "computed on a float cast of the counts, so they are RELATIVE "
                    "(comparable within this cube, not across sensors or sessions). "
                    "Pass white_reference and dark_reference to calibrate."
                ),
            )
        )
        return _as_float(cube), "none", warnings
    return cube, "none", warnings


def compute_index(cube: Any, index: str) -> Any:
    """A spectral index as a Spectral_data (true values in array_data)."""
    names = available_indices()
    if index not in names:
        raise IndexUnavailableError(f"Unknown index {index!r}. Valid: {names}.")
    if np.issubdtype(cube.array_data.dtype, np.integer):
        raise IndexUnavailableError(
            "Refusing to compute an index on integer counts (unsigned wraparound); "
            "this is a bug in the caller, prepare_cube() must run first."
        )
    fn = getattr(pcv.spectral_index, index)
    with isolated_pcv_outputs():  # PlantCV emits warnings/debug through globals
        result = fn(hsi=cube, distance=INDEX_DISTANCE)
    if result is None:
        raise IndexUnavailableError(
            f"Index {index!r} cannot be computed from wavelengths "
            f"{float(cube.min_wavelength):g}-{float(cube.max_wavelength):g} "
            f"{cube.wavelength_units} (within {INDEX_DISTANCE} nm of the bands it "
            "needs). Choose an index the cube's range supports."
        )
    return result


@dataclass
class HsiSegmentation:
    mask: np.ndarray
    overlay: np.ndarray
    diagnostics: MaskDiagnostics
    warnings: list[Advisory]
    calibration: str
    calibration_args: dict[str, str | None]
    index: str
    threshold: float
    object_type: str
    index_range: tuple[float, float]
    band_count: int
    wavelength_range: tuple[float, float]


def segment_hyperspectral(
    path: str,
    index: str = "ndvi",
    threshold: float = 0.2,
    object_type: str = "light",
    white_reference: str | None = None,
    dark_reference: str | None = None,
    fill_size: int = 200,
    cube_load: CubeLoad | None = None,
) -> HsiSegmentation:
    if object_type not in ("light", "dark"):
        raise ValueError(f"object_type must be 'light' or 'dark', got {object_type!r}")
    load = cube_load or load_cube(path)
    prepared, calibration, warnings = prepare_cube(
        load.cube, white_reference, dark_reference
    )
    idx = compute_index(prepared, index)
    values = idx.array_data.astype(np.float64)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    with isolated_pcv_outputs():
        pre_fill = pcv.threshold.binary(
            gray_img=values, threshold=float(threshold), object_type=object_type
        )
        mask = pcv.fill(bin_img=pre_fill, size=fill_size)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    diag = analyze_mask(mask)
    warnings = warnings + segmentation_warnings(
        mask, diag, analyze_mask(pre_fill), fill_size
    )
    pseudo = load.cube.pseudo_rgb
    overlay = render_overlay(np.ascontiguousarray(pseudo), mask)
    wl = sorted(float(w) for w in load.cube.wavelength_dict)
    return HsiSegmentation(
        mask=mask,
        overlay=overlay,
        diagnostics=diag,
        warnings=warnings,
        calibration=calibration,
        calibration_args={
            "white_reference": white_reference,
            "dark_reference": dark_reference,
        },
        index=index,
        threshold=float(threshold),
        object_type=object_type,
        index_range=(lo, hi),
        band_count=len(wl),
        wavelength_range=(wl[0], wl[-1]),
    )


@dataclass
class SpectralResult:
    indices: dict[str, dict[str, float | None]]
    band_count: int
    wavelength_range: tuple[float, float]
    calibration: str
    warnings: list[Advisory]
    pixel_count: int
    spectrum: dict[str, Any] | None = None
    histograms: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "indices": self.indices,
            "band_count": self.band_count,
            "wavelength_range": list(self.wavelength_range),
            "calibration": self.calibration,
            "pixel_count": self.pixel_count,
            "warnings": [{"code": w.code, "message": w.message} for w in self.warnings],
        }
        if self.spectrum is not None:
            out["spectrum"] = self.spectrum
        if self.histograms:
            out["histograms"] = self.histograms
        return out


def _value(observations: dict, key: str) -> Any:
    for name, obs in observations.items():
        if (name == LABEL or name.startswith(LABEL + "_")) and key in obs:
            return obs[key].get("value")
    return None


def measure_spectral(
    path: str,
    mask: np.ndarray,
    indices: list[str] | tuple[str, ...] = ("ndvi",),
    calibration: dict[str, str | None] | None = None,
    include_spectrum: bool = False,
    include_histograms: bool = False,
    cube_load: CubeLoad | None = None,
) -> SpectralResult:
    if not indices:
        raise IndexUnavailableError(
            f"No indices requested. Valid: {available_indices()}."
        )
    diag = analyze_mask(mask)
    assert_not_degenerate(diag)
    load = cube_load or load_cube(path)
    calibration = calibration or {}
    prepared, label, warnings = prepare_cube(
        load.cube, calibration.get("white_reference"), calibration.get("dark_reference")
    )
    if prepared.array_data.shape[:2] != mask.shape:
        raise ValueError(
            f"mask {mask.shape} does not match the cube's frame "
            f"{prepared.array_data.shape[:2]}"
        )
    labeled = np.where(mask > 0, 1, 0).astype(np.uint8)
    selected = mask > 0
    out: dict[str, dict[str, float | None]] = {}
    histograms: dict[str, Any] = {}
    for name in indices:
        idx = compute_index(prepared, name)
        values = idx.array_data.astype(np.float64)[selected]
        values = values[np.isfinite(values)]
        with isolated_pcv_outputs():
            pcv.analyze.spectral_index(
                index_img=idx,
                labeled_mask=labeled,
                n_labels=1,
                bins=100,
                min_bin="auto",
                max_bin="auto",
                label=LABEL,
            )
            obs = {k: dict(v) for k, v in pcv.outputs.observations.items()}
        out[name] = {
            "mean": float(_value(obs, f"mean_index_{name}")),
            "median": float(_value(obs, f"med_index_{name}")),
            "std": float(_value(obs, f"std_index_{name}")),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }
        if include_histograms:
            freq = _value(obs, f"index_frequencies_index_{name}")
            for gname, group in obs.items():
                if (
                    gname.startswith(LABEL)
                    and f"index_frequencies_index_{name}" in group
                ):
                    histograms[name] = {
                        "bins": [
                            float(b)
                            for b in group[f"index_frequencies_index_{name}"]["label"]
                        ],
                        "frequencies": [float(f) for f in freq],
                    }
    spectrum = None
    if include_spectrum:
        with isolated_pcv_outputs():
            pcv.analyze.spectral_reflectance(
                hsi=prepared, labeled_mask=labeled, n_labels=1, label=LABEL
            )
            obs = {k: dict(v) for k, v in pcv.outputs.observations.items()}
        wavelengths = [float(w) for w in _label_of(obs, "wavelength_means")]
        spectrum = {
            "wavelengths": wavelengths,
            "wavelength_means": [float(v) for v in _value(obs, "wavelength_means")],
            "max_reflectance": [float(v) for v in _value(obs, "max_reflectance")],
            "min_reflectance": [float(v) for v in _value(obs, "min_reflectance")],
            "spectral_std": [float(v) for v in _value(obs, "spectral_std")],
            "global_mean_reflectance": float(_value(obs, "global_mean_reflectance")),
            "global_median_reflectance": float(
                _value(obs, "global_median_reflectance")
            ),
            "global_spectral_std": float(_value(obs, "global_spectral_std")),
        }
    wl = sorted(float(w) for w in load.cube.wavelength_dict)
    return SpectralResult(
        indices=out,
        band_count=len(wl),
        wavelength_range=(wl[0], wl[-1]),
        calibration=label,
        warnings=warnings,
        pixel_count=int(selected.sum()),
        spectrum=spectrum,
        histograms=histograms,
    )


def _label_of(observations: dict, key: str) -> Any:
    for name, obs in observations.items():
        if (name == LABEL or name.startswith(LABEL + "_")) and key in obs:
            return obs[key].get("label")
    return []
