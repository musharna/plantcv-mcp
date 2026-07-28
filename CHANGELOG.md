# Changelog

All notable changes to `plantcv-mcp` are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-28

Phase 1. PlantCV exposed as an MCP **measurement instrument**: the model proposes a
segmentation, the server returns the numbers _and the overlay it measured_.

### Added

- Four MCP tools:
  - `list_methods()` — available segmentation channels and methods, plus the resolved
    PlantCV version and guidance on which channel suits which image.
  - `suggest_segmentation(image_path, channel="a")` — a colourspace contact sheet and an
    auto-threshold contact sheet, so the channel/method choice is informed rather than
    blind. The server never picks silently.
  - `segment(image_path, channel, method)` — mints a `session_id` and returns the overlay
    image alongside mask diagnostics (mask fraction, component count, major-object count,
    largest area) and any fired warnings. Returns **no traits**.
  - `measure(session_id)` — plant traits with units for a segmentation `segment` produced.
- **Degeneracy gate.** `measure` refuses to return traits on an empty or degenerate mask
  (zero connected components, zero largest-component area, or mask fraction below 0.1% of
  the frame), raising `DegenerateMaskError` with the mask fraction and a re-segmentation
  suggestion instead.
- **Multi-specimen warning** when two or more connected components each reach at least 25%
  of the largest component's area — a whole-image ROI would merge them into one "plant"
  and every size trait would describe the group.
- **Frame-clipping warning** when mask pixels touch any of the four frame edges, flagging
  that size traits are a lower bound rather than a measurement.
- **Session store** with LRU eviction (default 8 sessions), holding a copy of the mask and
  the image path rather than the decoded RGB image.
- `ImageChangedSinceSegmentationError` — `measure` re-reads the image from disk and refuses
  to proceed if its shape no longer matches the mask's, rather than silently applying a
  stale mask or letting PlantCV raise an opaque `IndexError`.
- Overlay and contact-sheet downscaling is always **reported** in the response
  (`overlay_scale`, `colorspace_sheet_scale`, `threshold_sheet_scale`), never silent.
- 47 tests, including a real-execution integration test over a tracked fixture render and
  determinism checks across repeat runs.
- `docs/MUTATION-CHECKS.md` — the mutants each guard test was confirmed to go red against.

### Notes

- **`segment` and `measure` are deliberately separate tools.** A single `analyze(image)`
  call would hand the model 17 plausible numbers having never shown it the mask. Because
  `measure` requires a `session_id` only `segment` mints, and `segment` returns the overlay,
  the protocol structurally puts the visual evidence in the model's context before a number
  can be obtained.
- PlantCV's own `in_bounds` and `object_in_frame` are treated as _additional information_,
  never as validity signals: on an all-zero mask PlantCV reports both `True` while returning
  a full 17-trait set of zeros. They are bounds checks, not success checks, so this server
  computes its own degeneracy gate instead.
- The 25% multi-specimen threshold and the 0.1% degeneracy floor are calibrated starting
  values, not constants, and are expected to be re-tuned against a wider image set.

### Not included (phase 2)

`measure` uses the whole image as its region of interest; explicit and multi-plant ROIs
(`roi.rectangle`, `roi.circle`, `roi.auto_grid`) are not yet exposed — which is why the
multi-specimen warning can only advise, not correct. Also deferred: mask refinement
(`refine`), morphology traits (leaf insertion/tangent angles, stem, skeleton segments),
batch processing, and hyperspectral/thermal imaging.
