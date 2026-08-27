# Changelog

All notable changes to `plantcv-mcp` are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] — 2026-08-27

Sub-project B of the roadmap. Design: `docs/superpowers/specs/2026-08-27-morphology-design.md`.

### Added

- **`measure_morphology(session_id, prune_size=15, tangent_size=25, px_per_mm)` —
  skeleton-based traits for one plant**, returning PlantCV's per-segment table
  (path/euclidean length, curvature, angle, tangent and insertion angle), per-plant
  stem height/length/angle, tip and branch-point counts, cycles and segment widths,
  together with the **numbered-segment overlay**: a segment `id` is the number on
  the picture. Runs entirely under `PCV_OUTPUTS_LOCK` — 15 of PlantCV's 19
  morphology functions report only through `pcv.outputs`.
  Every guard below was measured on PlantCV 4.11.3 against a synthetic plant of
  known geometry (stem + three leaves at 30/45/60°, lengths 90/80/70 px):
  - **A vertical stem yields `stem_angle = -14373°`** (PlantCV's slope-based
    estimate blows up). Returned as `null` with `stem_angle_undefined` instead of
    as a number.
  - **`tangent_size` default 25 is measured, not guessed.** Insertion-angle bias
    fell from 24.5° (10 px) to 5.9° (25) to 4.2° (30); PlantCV fits `size` pixels
    from each end, so `2 × size` longer than a leaf collapses its angle to `0.0`
    (`segment_tangent_angle.py:102`). `tangent_window_exceeds_segment` flags it.
  - **`prune_size_sensitive`** when the segment count changes by >30% at twice
    the prune size — then the table describes the parameter, not the plant.
  - **PlantCV's own abort on a fragmented skeleton** ("Too many tips found per
    segment, try pruning again") arrives as a refusal carrying the segment counts
    at `prune_size` and `2×prune_size`, not as a stack trace.
  - PlantCV's `segment_id` reuses a process-global cached colour palette
    (`params.saved_color_scale`) and indexes past it when a later plant has more
    segments; the cache is reset inside the lock.
  - Leaf-less skeletons (a bare stem, a ring) are reported with an empty table and
    `no_leaf_segments`, so cycles/tips/stem can still say why. Multi-plant masks
    are refused naming `measure_regions()` and `refine(keep_largest)`.
  - Known-geometry eval: leaves recovered in order, insertion angles within 12°,
    lengths within 15%; `stem_height` is PlantCV's base→topmost-junction height.
  - `'NA'` strings PlantCV emits for undefined traits become `null`.

## [0.6.0] — 2026-08-27

First step of the post-0.5.0 roadmap
(`docs/superpowers/specs/2026-08-27-backlog-integration-plan-of-attack.md`).

### Added

- **`refine(session_id, ops)` — mask refinement as a session→session operation.**
  Applies an ordered list of morphological ops (`fill_holes`, `fill`, `erode`,
  `dilate`, `opening`, `closing`, `median_blur`, and our own `keep_largest`) to a
  session's mask and mints a NEW session, returning the refined overlay plus
  before/after diagnostics and warnings — never traits. The original session is
  untouched, so a refinement that looks wrong is discarded rather than undone.
  - **Validation is all-or-nothing and stricter than PlantCV's.** PlantCV silently
    no-ops on `fill(size=-1)`, `erode(i=0)` and an even `median_blur` kernel
    (measured); a no-op recorded in the lineage as if it had run would be a lie,
    so every op is checked — name, required params, ranges, unknown params —
    before the first one touches the mask, and the error names the op index.
  - **A refinement that leaves no measurable plant is refused, not minted**, with
    before/after `mask_fraction`, `component_count` and `largest_area`.
  - `refine_large_change` advisory when the mask changed by more than 25%.
  - Known-value eval: a disc with a hole and salt noise measures wrong first
    (positive control), then `[fill_holes, keep_largest(1)]` recovers the area
    within 1%.
- **`lineage` on every trait table.** `measure()` and `measure_regions()` results
  carry the ops that produced their mask (`[]` for an unrefined session), so two
  tables made differently can be told apart. `Session` gains `lineage` and
  `parent_id`.
- `list_methods()` publishes `refine_ops` with each op's parameter constraints;
  the test suite applies every published op with its example so a documented op
  can never be a dead one.

## [0.5.0] — 2026-08-27

Findings of an adversarial audit run against 0.4.1 (Codex consult, every item
reproduced before it was fixed). Three were correctness defects in the shipped
runtime, not backlog.

### Fixed

- **Concurrent measurements corrupted each other through `pcv.outputs`.** The
  server's safety argument — synchronous tools run inline on the event loop, so
  two analyses can never interleave — was true for mcp 1.28.1 and became false
  when the dependency floor moved to `mcp>=2`: mcp 2.x runs synchronous tools on
  worker threads (`func_metadata.py`: `anyio.to_thread.run_sync`) and dispatches
  requests concurrently. Reproduced with two sessions and a widened window: one
  thread's `pcv.outputs.clear()` erased the other's observations (`Expected
observation group 'default_1' not found`), and one call returned the OTHER
  session's area as its own. One process-wide lock (`measurement.PCV_OUTPUTS_LOCK`,
  via `isolated_pcv_outputs`) now guards the snapshot → clear → analyze → read →
  restore section in both `measure()` and `measure_regions()`; the regression test
  drives a cross-path pair on two threads and fails without it.
- **Malformed `rect_grid` geometry crashed the whole server.** `coord=[-10, -10]`
  with a 50×50 cell on a 100×100 image passed `build_regions` and reached native
  OpenCV/PlantCV inside `measure_regions`, which died with SIGSEGV (exit 139) —
  taking every session with it, since this is a stdio server. `build_regions` now
  refuses non-positive `height`/`width`, non-positive `radius`, a missing
  `spacing` (required even for one cell: pinned PlantCV rejects it with a message
  that blames the wrong argument), and any cell that lies partly outside the
  image. An off-image cell previously reported "No plant material" — misleading,
  since the cell was off the frame, not empty.
- **Time-of-check/time-of-use hole in the stale-image guard, in both
  directions.** `segment()` decoded the file, built the mask, and only then hashed
  the path; `measure()` decoded and then hashed separately. A same-shape
  replacement landing in either window bound the OLD mask to the NEW file's hash,
  and the integrity check then passed. Both paths now read the bytes once, hash
  those bytes, and decode those bytes (`imaging.load_image_with_digest`); the
  regression swaps the file mid-segmentation and expects `measure()` to refuse.
- **`SessionStore` was not thread-safe.** `get()` is check-then-act and
  `create()` is insert-then-evict; under mcp 2.x worker threads an interleaving
  raised a bare `KeyError` for a session present a moment earlier. The store now
  carries its own lock (deliberately not the PlantCV analysis lock).
- **`pcv.outputs` restore was incomplete.** `clear()` resets `measurements`,
  `images`, `observations` and `metadata`; only `observations` was restored, so a
  host application using PlantCV directly had three tables silently emptied.

### Changed

- **`measure_images` now takes the whole recipe: `ksize`, `offset` and
  `color_correct`.** The tool tells users to settle a recipe with `segment()` and
  apply it, but could not accept the kernel parameters of the `mean`/`gaussian`
  methods or the colour-correction flag, so it silently ran a different threshold:
  1900 px vs 4536 px on the same file. The returned `recipe` records all three.
  An image that cannot be colour-corrected when asked is refused with the reason
  (`ColorCardNotFoundError`), never measured raw.
- **Every number-bearing result names its engine.** `engine: {name, version}`
  was added to `measure()` in 0.4.0 and is now on `measure_regions` and
  `measure_images` too; a stored trait table from any tool can say what produced
  it.
- **The server identifies its version at initialize.** `MCPServer` defaults
  `version` to `""`, and that empty string is what clients saw in `serverInfo`.
  Now `version=__version__`, checked through a real in-memory client handshake.
- The `plantcv==4.11.3` pin is described as PlantCV-version stability rather
  than determinism: `uvx plantcv-mcp` does not consume this repository's
  `uv.lock`, so the rest of the dependency set may vary within bounds.

### CI

- The sdist "exactly what we intend" step asserts every file in the allow-list
  (`CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` were not checked).
- The citation workflow also holds `CITATION.cff` `date-released` to the
  CHANGELOG date for the pyproject version; its error message had instructed
  updating both fields while checking one.
- The installed-wheel smoke test now runs the `plantcv-mcp` console script as a
  real stdio subprocess and completes an initialize handshake, instead of only
  importing the package and listing tools in-process.
- `tests/test_concurrency.py` (forced-interleave races), MCP-layer tests for
  `calibrate_scale_from_marker` and invalid geometry (`ToolError`, not a crash),
  and batch-parity tests through `call_tool`.

## [0.4.1] — 2026-08-02

### Added

- **Community-health and repo-hygiene files, matching the standard set by
  `data-aggregator-mcp` and `plant-genomics-mcp`.** An earlier parity audit
  compared this repo only against `ldraw-mcp`, which is itself thin on these, so
  the whole tier went unnoticed: `CONTRIBUTING.md`, `SECURITY.md`, issue forms
  (bug report + feature request + a config pointing security reports at private
  advisories), a pull-request template, `.editorconfig`, `.mcp.json`, `glama.json`,
  a CodeQL workflow, and a Dependabot config.

  **Dependabot uses the `uv` ecosystem, not `pip`.** This is a uv-locked project;
  the pip ecosystem would update `pyproject.toml` and leave `uv.lock` stale, which
  CI installs with `--frozen` and would fail on. Dependabot's native uv support
  reads both together.

  `CONTRIBUTING.md` and `SECURITY.md` were added to the sdist allow-list.
  hatchling's allow-list drops anything unlisted **silently** — verified with
  `tar tzf` on a real build rather than assumed, the same way a `NOTICE` was
  previously found missing.

  `SECURITY.md` documents this server's actual trust boundary and defers to the README's
  "Security and trust boundary" section as authoritative rather than restating it:
  this server reads image files anywhere the running user can read, deliberately and
  without a sandbox. It also states what is **in** scope — a path the caller did not
  ask for, non-image content disclosure, or an escape from the running user's own
  permissions — so "it read the file I asked it to read" is not filed as a
  vulnerability.

- **README gained a Glama badge, and its licence badge is now derived rather than
  hardcoded.** It read `badge/license-MIT-green`, a literal that would keep claiming
  MIT if the licence ever changed; it now reads `pypi/l/plantcv-mcp`. Verified to
  resolve to MIT before switching.

- **`server.json` is validated against the registry's own published schema.**
  `breedsim-mcp` v0.4.0 was tagged, uploaded to PyPI and GitHub-released before
  the MCP registry refused it with a 422: its description had grown past a
  100-character cap that nothing local measured. The publish workflow is the only
  thing that checks registry constraints, and it runs on tag push — after the
  version is already burned. This server's description is **91 characters**: it
  passes today with nine to spare, and nothing here was measuring it.

  Rather than copy the one constant, the check validates the whole document
  against the dated `$schema` `server.json` already declares, which is the
  registry's own statement of what it accepts. That covers the four other length
  caps and the required-field list as well. The schema is vendored at
  `tests/server.schema.json` rather than fetched, keeping the suite offline and
  deterministic, and a test asserts the vendored copy's `$id` still matches the
  declared `$schema` so the pin cannot drift silently.

  Verified by mutation rather than assumed: a 282-character description was
  written into the real `server.json` and the suite watched to fail on it, with a
  non-length failure (a missing required field) and an in-test positive control
  so a validator that raised on everything could not read as a working guard.

## [0.4.0] — 2026-08-01

### Added

- **`measure_regions()` — one row per plant.** `measure()` treats the whole frame
  as a single region, so a tray of seedlings was merged into one object and every
  size trait described the group; `multi_specimen` could warn about it but nothing
  could measure it. Regions come from `mode="auto_grid"` (PlantCV infers the
  layout from the mask; give only rows and columns) or `mode="rect_grid"`
  (explicit geometry). The response carries an overlay with every region outlined
  and numbered, because per-region numbers are unreadable without a picture
  saying which region is which.

  **An empty cell is refused by name, not reported as zero.** Measured on PlantCV
  4.11.3: for a 2×2 grid with one empty cell, `create_labels` returns n=4 and the
  analysis emits a complete trait set with `area = 0.0` for the empty one. That is
  the same failure `assert_not_degenerate` exists to stop in the single-region
  path. `np.unique(labeled)` is the discriminator, and such a region returns
  `measured: false` with a reason.

  That same measurement settled the index mapping: group `default_{i+1}`
  corresponds to region `i`, with **no shifting when a cell is empty**. Had
  empties been dropped, every trait after a gap would have been attributed to the
  neighbouring plant. The tests use plants of deliberately different sizes so a
  one-off shift is detectable at all — a fixture of identical plants could not
  fail that test.

- **Traits are now checked against shapes whose geometry is known in advance.**
  An 80×80 square must measure 6400 px, and it does. Every prior test compared
  this server against its own previous output, which cannot catch a change that
  is consistently wrong. This one can: an external oracle, arithmetic rather than
  a stored baseline.

- **`measure()` reports the engine that produced the numbers.** The result now
  carries `engine` with the PlantCV version. It was previously reachable only
  through `list_methods`, so a saved trait table could not be traced back to the
  version that measured it — and PlantCV's own trait definitions do change
  between versions.

- **A fresh-process test that drives the tool layer and nothing else.** Three
  sequential `call_tool` invocations in a subprocess that has never touched
  PlantCV directly, asserting the 6400 px oracle at the end.

  This exists because of a bug in a SIBLING server, not one found here: in
  breedsim-mcp, rpy2 published its conversion rules into a `ContextVar` at import
  time, the import happened inside the first request, and the rules were
  discarded when that request returned — while 27 tests passed throughout,
  because a test that imports the dependency into pytest's root context masks the
  whole class of failure. Nothing here is known to be broken; what was missing was
  the ability to notice. PlantCV has import-time global state of its own
  (`pcv.params.sample_label`, which this package reads), so a case covers a host
  process that sets it before importing us. The file carries its own positive
  control proving the driver can actually fail.

### Changed

- The `multi_specimen` warning pointed at "roi.auto_grid (phase 2)". Phase 2 is
  this release; it now names `measure_regions()`. A warning pointing at unbuilt
  work is worse than no pointer.
- `_load_session_image()` extracted so `measure()` and `measure_regions()` share
  one copy of the stale-image guards. A second entry point with its own copy is a
  copy that can drift, and a drifted staleness check measures a mask against
  pixels it was never drawn on.

## [0.3.2] — 2026-07-31

### Added

- **Zenodo archival.** This release exists to be archived: the Zenodo↔GitHub
  integration mints a DOI from the tag's tarball, and the previous tag predated
  `.zenodo.json` and `CITATION.cff` entirely — those files were added after it was
  cut. Zenodo archives the tag, not the default branch, so a release was the only
  way to get the metadata into an archived snapshot.

### Changed

- **`.zenodo.json` now uses Zenodo's lowercase licence identifier**
  (`mit` rather than `MIT`). That is the canonical spelling —
  `zenodo.org/api/vocabularies/licenses/<id>` returns 200 for the lowercase form
  and 404 for the SPDX-cased one. See the correction below: it fixed nothing.

### Correction — added after this release was published

This release was originally described here as **fixing** a defect in which the
SPDX casing "silently dropped the licence from the published record". **That was
wrong, and the entry is corrected rather than quietly deleted.**

Zenodo normalises the licence identifier on ingest. The sibling `ldraw-mcp`
archived with `"MIT"` still in place and its record reads `license: mit-license`;
this project's record reads `license: mit`. The licence was never dropped.

The apparent evidence was two of my own measurement errors, both the same
mistake — probing a proxy instead of the artifact:

1. Querying the licence **vocabulary endpoint** and treating a 404 there as what
   the ingest accepts. It is not; the ingest normalises casing.
2. Reading the **RDM-era field names** (`rights`, `subjects`,
   `creators[].person_or_org`) against an API endpoint that returns the **legacy**
   shape (`metadata.license`, `metadata.keywords`, `creators[].orcid`). Every
   field reported as absent was present throughout.

What remains true is the reason this release exists: the previous tag predated
`.zenodo.json` and `CITATION.cff`, and Zenodo archives the **tag**, not the
default branch. DOI: [10.5281/zenodo.21713869](https://doi.org/10.5281/zenodo.21713869).

### Notes

No functional change. Tools, guards and dependency pins are identical to 0.3.1.

## [0.3.1] — 2026-07-31

### Added

- **Published to the official MCP registry** (`io.github.musharna/plantcv-mcp`)
  via `server.json` and an OIDC workflow, so the server is discoverable from MCP
  clients and directories rather than only from PyPI.

  This needed a release rather than a docs commit. The registry proves PyPI
  ownership by finding an `mcp-name` marker in the package README **as published
  to PyPI**, and PyPI captures `long_description` at release time — so a marker
  sitting on `master` verifies nothing. That is the same mechanism that kept this
  package's "Not published to PyPI" line live on its project page after the fix
  had merged, and it is why 0.2.1 existed too.

- **`tests/test_registry_metadata.py`.** `server.json` states the version in
  three places and nothing else makes them agree with `pyproject.toml`; a stale
  one is rejected by the registry during a release, after the version is spent.
  The README marker is checked against the name `server.json` declares, since
  that exact string is what the registry greps for.

### Notes

No functional change. Tools, guards and dependency pins are identical to 0.3.0.

Running the real `mcp-publisher validate` against `server.json` is what caught a
100-character cap on `description` in the sibling `breedsim-mcp` — a constraint
no schema read surfaced, and one that would otherwise have failed the publish
after the version was already on PyPI. The workflow validates before
authenticating for that reason.

## [0.3.0] — 2026-07-30

### Changed

- **Migrated to `mcp` 2.x.** `FastMCP` was renamed, not removed: it is now
  `mcp.server.mcpserver.MCPServer`, and `Image` moved with it. The dependency
  **moves** to `mcp>=2,<3` rather than widening to `<3` — this package imports
  `mcp.server.mcpserver`, absent in 1.x, so a range spanning both majors could
  resolve to a version that cannot import the server. mcp 2.x requires Python

  > =3.10, below this package's >=3.11 floor, so the support matrix is unchanged.

- **`call_tool` returns a `CallToolResult`**, not a bare block sequence or a
  `(content, structured)` tuple. Tests read `.content` and `.structured_content`;
  the old `res[0] if isinstance(res, tuple) else res` shims are removed rather
  than extended, because under 2.x that sniff falls through and hands back the
  result object, moving the failure away from its cause.

- **`mcp.types` fields are snake_case** (`outputSchema` → `output_schema`,
  `readOnlyHint` → `read_only_hint`). Constructing `ToolAnnotations` still
  accepts camelCase via pydantic aliases, but reading the attributes does not —
  so construction would have stayed green while every read broke. Both spellings
  are now aligned, including in the comments that named the old field.

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
