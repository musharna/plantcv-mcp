# Hyperspectral + thermal (1.0.0)

**Parent:** `2026-08-27-backlog-integration-plan-of-attack.md`, sub-project E.
**Owner decision:** validate on real public data. **Data (MPL-2.0, PlantCV
`tests/testdata`, vendored under `tests/fixtures/plantcv/` with a NOTICE):**
`corn-kernel-hyperspectral.{hdr,raw}` (31×43 px, 580 bands 366–1048 nm, uint16
counts 0–2887), `darkReference{,.hdr}`, `FLIR_test.jpg` (FLIR radiometric JPEG,
480×640, 19–24 °C via `flyr`), `FLIR2600.csv` (480×640 °C), `thermal_img.npz` +
`thermal_img_mask.png` (480×640, 30–38 °C, a hand-made plant mask).

## Measured PlantCV behaviours that shape the design

| Observation (4.11.3, 2026-08-27)                                                                                                                                                                                                                                         | Design response                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `readimage(mode="envi")` wants the **RAW** path and derives `<raw>.hdr` (`_find_hdr`); given the `.hdr` it reads the header as data and fails on reshape.                                                                                                                | `segment_hyperspectral(envi_path)` accepts either; resolve to the raw path, require the `.hdr` sibling, fail loud naming both.                                                                                                                                                   |
| The cube is **uint16 counts**. `spectral_index.ndvi` computes `(r800 − r670)/(r800 + r670)` on `array_data` as-is → **unsigned wraparound**: NDVI came out 0.0008–205.9 on a [-1, 1] index. `hyperspectral.calibrate(raw, white, dark)` casts to float64 and normalises. | **Never compute an index on integer counts.** With `white_reference`/`dark_reference` paths → `calibrate()`; without → cast to float64 and attach `uncalibrated_cube` (values are counts, not reflectance; indices are relative only). Cast + calibrate happen before any index. |
| `_package_index` keeps the true index in `array_data` (`min_value`/`max_value` = observed) and a 0–255 rescale in `pseudo_rgb`.                                                                                                                                          | Threshold and analyse `array_data`; use `pseudo_rgb` only for the overlay.                                                                                                                                                                                                       |
| `analyze.spectral_index` bins with `min_bin=0, max_bin=1` by default; `spectral_reflectance` returns per-band lists of 580 values ×4.                                                                                                                                    | Report `mean/median/std` per index; histogram off by default. Full spectra (`wavelength_means` etc., 580 × 4 numbers) are **opt-in** (`include_spectrum`), band count always reported — the large-result rule.                                                                   |
| `readimage(mode="thermal")` → float64 °C array via `flyr`; `mode="csv"` → float64 array; `analyze.thermal` reports max/min/mean/median °C + histogram.                                                                                                                   | `segment_thermal(path, min_c, max_c)` thresholds the °C array (`.jpg` → flyr, `.csv` → csv, `.npz` → first array); `measure_thermal` → `analyze.thermal`, histogram opt-in.                                                                                                      |
| A thermal frame is a different sensor/frame from an RGB session.                                                                                                                                                                                                         | A thermal mask is never borrowed from an RGB session; thermal sessions are typed.                                                                                                                                                                                                |

## Sessions become typed

`Session.kind ∈ {"rgb","hsi","thermal"}` (default `"rgb"`); `Session.source` holds
the modality payload the analysis needs (`Spectral_data` cube or °C array is NOT
kept — like RGB, it is re-read through the digest guard). `measure()`,
`measure_regions()`, `measure_morphology()`, `refine()` refuse non-RGB sessions by
name, pointing at the right tool; `measure_spectral`/`measure_thermal` refuse RGB
sessions likewise.

## Tools (four; 13 total)

- `segment_hyperspectral(envi_path, index="ndvi", threshold, object_type="light",
white_reference=None, dark_reference=None, fill_size=200)` → session (kind hsi)
  - overlay on the cube's **pseudo-RGB** + diagnostics (`segmentation_warnings`) +
    `calibration: "white/dark" | "none"` + the index's observed range.
- `measure_spectral(session_id, indices=["ndvi"], include_spectrum=False,
include_histograms=False)` → per index `{mean, median, std, min, max}`, band
  count, wavelength range, optional full spectrum, `engine`, `calibration`.
- `segment_thermal(path, min_c=None, max_c=None, fill_size=200)` → session (kind
  thermal) + overlay on a grey rendering of the °C frame + diagnostics + frame
  temperature range.
- `measure_thermal(session_id, include_histograms=False)` → `{max, min, mean,
median}` °C, pixel count, `engine`.

All index names are validated against `pcv.spectral_index` (31 functions) and an
index PlantCV declines for the cube's wavelength range (`ndvi` needs 670 & 800 nm)
is refused naming the range, not returned as `None`.

## Evals

- **Real data:** corn cube → `segment_hyperspectral(index="ndvi", threshold≈…)`
  segments the kernel (mask fraction in a sane band, `uncalibrated_cube` present);
  with `darkReference` as both white and dark → refused (white == dark divides by
  zero → `calibration_degenerate`), so the calibration path is exercised for its
  refusal; FLIR JPEG → `segment_thermal` on 19–24 °C; `thermal_img.npz` with the
  vendored mask → `measure_thermal` mean within 0.01 °C of PlantCV's own 33.509.
- **Synthetic known-value:** a cube written with `hyperspectral.write_data` whose
  bands at 670/800 nm are chosen so NDVI is exactly 0.6 inside a disc and −0.2
  outside (float32, "calibrated") → `measure_spectral` mean NDVI within 0.01 of
  0.6, positive control that the uint16 version of the same cube WITHOUT the cast
  would not; a °C array with a known warm disc → mean within 0.01 °C.
- MCP layer for all four; tool-list assertions (13); typed-session refusals both
  ways.
