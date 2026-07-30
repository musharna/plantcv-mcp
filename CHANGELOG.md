# Changelog

All notable changes to `plantcv-mcp` are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2026-07-30

A docs-and-metadata release. No behaviour changes; it exists because two claims
the package made about itself were wrong, and one of them could only be corrected
by publishing.

### Fixed

- **The README told users the package was unpublished.** The Install section read
  "Not published to PyPI. Install from the repository", so anyone following it
  built from a git checkout while `pip install plantcv-mcp` had worked since
  0.2.0. Corrected on `master` before this release — but PyPI renders the
  description captured at release time, so the project page kept serving the
  false claim regardless. **That is what this release is for.**

- **`plantcv_mcp.__version__` reported `0.1.0` from the published 0.2.0.** It was
  a literal sitting beside a `pyproject.toml` that said 0.2.0, with nothing
  enforcing agreement. It now reads from installed metadata — the source the
  packaging already enforces — so the two cannot drift again.

  `tests/test_version.py` asserted only that the version was a non-empty string,
  which is exactly what a wrong version is, so it passed throughout. It now
  compares the reported version against the one `pyproject.toml` declares, and
  was confirmed to fail on the old code before being kept.

### Changed

- Status is now derived rather than asserted: PyPI version and `pyversions`
  badges replace the hand-written publication claim and the static python badge.
  A hand-typed status line is only true until someone forgets, which is how the
  Install section went stale in the first place.

## [0.2.0] — 2026-07-28

First release published to PyPI. 0.1.0 was tagged in this changelog but never uploaded,
so everything below shipped together.

### Added

- **`calibrate_scale_from_marker`** — measures a marker of known real size and returns
  `px_per_mm`, closing the half of real-world units that was previously left to the caller.
  It does **not** wrap `pcv.report_size_marker_area`: that function takes an ROI, and against
  a synthetic disc of known 80 px diameter it returns `major_axis=79.1` with a whole-frame ROI
  but **348.0 with a tight ROI around the marker** — a silent 4.35× scale error under the most
  intuitive usage, because its ROI filter can select a background component that merely
  intersects the region. Here the region is **cropped before thresholding**, so nothing outside
  the box can be selected. That removes the mechanism instead of compensating for it.
  A `marker_touches_crop_edge` warning catches the wrong-polarity case: measured on a centred
  80 px disc in a 100×100 crop, the correct and inverted polarities both give `crop_fraction`
  0.50, so coverage cannot discriminate and edge contact is the guard that can.
- **Colour-card correction.** `segment(..., color_correct=true)` detects a ColorChecker and
  corrects to a standard reference; `measure()` re-applies it so traits are measured on the
  pixels the mask was drawn on. If no card is found it **raises** rather than silently
  measuring the uncorrected image. Previously deferred for want of a test fixture — a
  synthetic Macbeth chart driving the real detector, plus a known colour distortion, gives
  ground truth: mean absolute error to the undistorted original falls from 8.77 to 3.43.
- **`measure_images`** — one recipe across up to 200 images. Batch cannot honour "no number
  without the picture" literally, so the overlay is replaced by automated validation plus
  explicit refusal: every image runs the **same** guards as `segment()` via shared code, and
  any image tripping a blocking guard returns **no traits**, only a reason. That is weaker
  than a human reading a mask and is documented as such rather than implied.
- **Real-world units.** `measure(session_id, px_per_mm=…)` converts spatial traits to `mm`
  and `mm2`. Without it every size is in pixels, and pixel sizes are not comparable between
  images shot at different distance or zoom — the largest practical limitation the tool had.
  Lengths divide by `px_per_mm` and areas by its square, from an explicit table rather than
  from PlantCV's unit strings: PlantCV labels **both** `area` and `width` as `"pixels"`, so
  a unit-derived rule would leave every area wrong by exactly a factor of `px_per_mm`.
  Positions stay in pixels, having no meaning in mm without an origin.
- **Colour analysis.** `measure(session_id, analyses=["size", "color"])` adds hue,
  saturation and value statistics. Their three frequency histograms total 692 numbers and
  are withheld unless `include_histograms=true` — that is a context-window cost, not a
  feature. This takes the server from 1 to 2 of PlantCV's 11 `analyze` functions.
- **Server instructions.** The server now publishes MCP `instructions` telling the client to
  look at the overlay before trusting a number, and what each warning code means. The product
  is a discipline as much as a set of functions, and nothing was conveying that.
- **Tool metadata.** Every tool now publishes a human title and `ToolAnnotations`
  (`readOnlyHint`, `destructiveHint=false`, `idempotentHint`, `openWorldHint=false`) so a
  client can tell they only read and compute. `measure` and `list_methods` publish an
  `outputSchema` derived from typed returns, so callers get structured content instead of
  parsing JSON out of a text block — as do `calibrate_scale_from_marker` and
  `measure_images`. `segment` and `suggest_segmentation` return image blocks and so have
  no structured schema, by nature; a test asserts exactly that split so a new tool cannot
  quietly ship without one.

### Previously deferred, now resolved

Both of the items this section used to hold have shipped. Automatic scale is implemented
without wrapping the ROI-based API that produced the 4.35× error, and colour correction is
implemented now that a synthetic ColorChecker gives its happy path a ground-truth test. The
measured reasons are preserved in the entries above so neither is re-litigated from scratch.

### Documentation

- **The README now shows an overlay.** A tool whose entire argument is "you get the picture
  the numbers came from" had no pictures in it. The top of the page is now the same image
  segmented correctly (3.1% mask, `area=32427`) beside the same image segmented with the
  wrong polarity (96.1% mask — the background — `area=1007829`), which makes the case in
  one glance instead of three paragraphs. Assets render from the bundled fixture via
  committed code; each is a standalone image laid out with a table, not a stitched grid.
- **The seventeen traits are listed.** Previously the README named exactly one (`area`) and
  only as part of a failure example, so a reader could not tell whether the tool measured
  what they needed.
- Added a `segment` parameter table, a verbatim example response, a colourspace contact
  sheet, CI/Python/licence badges, and links to the changelog and the mutation-check log.
  Reordered so the reader sees what the tool produces before installation instructions, and
  promoted "Getting the polarity right" to a top-level section since it is the main
  correctness trap.

### Fixed

- **Mask validity is now two-sided.** `assert_not_degenerate` only ever rejected masks
  that were too _small_, which left inverting a threshold — the dominant failure of any
  threshold operation — outside the set of outcomes this system could express as a
  failure. A new `implausible_coverage` warning fires above 50% frame coverage.
  Measured on the fixture: plant masks land at 0.031–0.046 and inverted ones at
  0.959–0.967, so the boundary sits in a wide empty gap. It warns rather than raises,
  because a macro shot of one leaf can legitimately fill the frame.
- **`object_type` is exposed on `segment()`.** It was hardcoded to `"dark"` and
  unreachable, so channels `s` and `b` returned the **background** as the plant:
  `mask_fraction` 0.961, and `measure()` reported `area=1007829, width=1024,
height=1024` — the whole frame — with no error. `list_methods()` made this worse by
  recommending `'s'`. Its guidance is now correct and names the polarity each channel
  needs.
- **`suggest_segmentation` reports both polarities.** It now measures what `dark` and
  `light` each yield on your image and recommends one, flagging the case where the two
  are too close to call rather than guessing.
- **`fill_size`, `ksize` and `offset` are exposed.** `fill_size=200` was hardcoded and
  silently erased any specimen smaller than itself — a measured 144 px object became an
  empty mask. Thresholding and filling are now separate steps, so this reports
  `fill_erased_mask` naming `fill_size` and the size to drop below, instead of
  presenting as a bad channel choice.
- **`segment()` warns on an empty mask.** It previously returned `component_count=0`
  with no warning at all, so the failure surfaced only if `measure()` happened to be
  called afterwards.
- **`frame_clipping` is withheld when coverage is implausible.** It asserts that size
  traits are a lower bound, which presumes the mask is the plant; on an inverted mask
  that misleads. A genuinely clipped, plausibly-sized plant still reports it.
- **`measure()` no longer destroys the host's PlantCV state.** `pcv.outputs.clear()`
  wiped a process-global table shared with any application also using PlantCV directly.
  Observations are now snapshotted and restored in a `finally`.
- **The stale-image guard compares content, not just shape.** Swapping the file for a
  different image of identical dimensions previously passed, measuring a stale mask
  against new pixels. A SHA-256 of the file is now recorded and re-checked.

### Notes

- Tool functions remain deliberately **synchronous**. mcp 1.28.1 runs sync tools inline
  on the event loop, which serialises them and is what makes PlantCV's process-global
  `pcv.outputs` safe here. Making a tool `async` would allow two analyses to interleave
  on that global and needs a lock first.

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
- Continuous integration: lint, format check, and the full suite on Python 3.11, 3.12 and
  3.13; a packaging job that asserts the sdist ships `NOTICE`/`LICENSE`/`CHANGELOG.md` and
  leaks none of `.superpowers`, `docs/superpowers` or `.claude`, and that the built wheel
  imports and still registers exactly the expected tool surface; and an absolute-path check that runs its
  own negative control so a silently-disabled check fails the build.
- `NOTICE` — attribution for PlantCV (MPL-2.0), an explicit statement that this project is
  unofficial and unaffiliated with the Donald Danforth Plant Science Center, and the
  provenance of the test fixture.
- PyPI metadata: authors, keywords, classifiers, and project URLs.
- README sections covering MCP client configuration (Claude Code and Claude Desktop) and the
  server's security/trust boundary.

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
- **The server reads image files from anywhere the host user can read**, with no directory
  allow-list or sandbox, and returns them to the model as base64 images. This is the same
  trust boundary as any local filesystem MCP server, and it is documented rather than
  silently assumed. Restricting reads to a configured root is a candidate for a later release.
- `plantcv` is hard-pinned to `==4.11.3` on purpose: trait values can shift between PlantCV
  releases and determinism is a tested guarantee. `opencv-python` is deliberately left without
  an upper bound, because PlantCV already caps it and duplicating that ceiling would drift.

### Not included (phase 2)

`measure` uses the whole image as its region of interest; explicit and multi-plant ROIs
(`roi.rectangle`, `roi.circle`, `roi.auto_grid`) are not yet exposed — which is why the
multi-specimen warning can only advise, not correct. Also deferred: mask refinement
(`refine`), morphology traits (leaf insertion/tangent angles, stem, skeleton segments),
batch processing, and hyperspectral/thermal imaging.
