# plantcv-mcp — design

> Status: design approved 2026-07-28 (sections 1–3). Not yet planned, not yet built.
> Working name `plantcv-mcp` — **name availability NOT yet verified** (see Open Questions).

## 1. Motivation and seam

An MCP server exposing PlantCV as a **measurement instrument** for vision-capable models:
the model proposes a segmentation, the server returns numbers **and the segmentation overlay**,
so the model can confirm the measurement was taken on the right pixels.

**Grounding (7-registry sweep, shape-matched positive controls, 2026-07-27/28):**
`plantcv` is null on npm; `phenotyping mcp`, `leaf segmentation`, `field trial`, `genebank`
all verified-null. The BioContextAI registry lists 66 curated biomedical MCP servers and the
only plant entry is our own `musharna-plant-genomics-mcp`.

**Tier thesis.** Domains accumulate *retrieval* MCP servers; almost nobody ships the *compute*
tier. Biomedical crossed that line (`scvi-tools-mcp`, `galaxy-mcp`, `royerlab-napari-mcp`,
`vrtejus-pymol-mcp`). Plant science has not. This server is the compute tier for plant imaging.

**Gate 1 (shell-out) — passed on 3 of 4 conditions.** PlantCV is a local library, so an MCP
surface must justify itself. It does via: (a) **state across calls** — an iterative
segment → look → adjust → measure loop; (b) **typed schemas** — 17 united traits the model
should not parse from stdout; (c) **routing** — ~60 functions across 6 namespaces where
"which threshold for which image" is real routing work a `--help` cannot do.
Not remote/authed. (a) is the strongest: a one-shot script cannot model the iteration.

## 2. Empirical basis (verified 2026-07-28, PlantCV 4.11.3, real images)

Everything below was measured, not assumed.

**Three distinct silent-wrongness modes were observed. All produced plausible,
correctly-united numbers.**

| # | case | mask | what was wrong | mechanically detectable via |
|---|---|---|---|---|
| 1 | `bio3d-arena/.../736_multi4.png` | good | a 4-view render; whole-image ROI merged 4 plants into ONE object, so `width=686 height=843 area=32427` span four specimens | 9 connected components; top-4 areas **8628 / 7981 / 7106 / 6748** (near-equal) |
| 2 | real orchid photo | excellent (pot + dark bg correctly excluded) | plant **clipped by frame**; `width=1712 height=1141` are exactly the image dims, so size traits are lower bounds | `in_bounds=False` |
| 3 | all-zero mask | empty | `analyze.size` returns a **full 17-trait set of zeros** with **`in_bounds=True` and `object_in_frame=True`** | nothing in PlantCV — its QC flags certify the failure as fine |

**Mode 3 is the critical one.** PlantCV's `in_bounds` / `object_in_frame` are *bounds* checks,
not *success* checks. On a total segmentation failure they report success. They cannot
discriminate the failure they appear to cover.

**Verified API facts:**
- `plantcv.__version__` does **not** exist; use `importlib.metadata.version("plantcv")`.
- `pcv.readimage` **raises** `RuntimeError: Failed to open <path>` for missing AND non-image
  files (it does not silently return `None` as bare `cv2.imread` would).
- `pcv.analyze` (11 fns) · `pcv.morphology` (19) · `pcv.roi` (13) · `pcv.threshold` (11) ·
  `pcv.visualize` (15) · `pcv.filters` (2). One `analyze.size` yields **17 traits, each with a unit**.
- Native comparison helpers exist: `visualize.colorspaces`, `visualize.auto_threshold_methods`,
  `visualize.obj_sizes`, `visualize.obj_size_ecdf` (returns an Altair chart).

## 3. Tool surface (phase 1)

Four tools. Verb-shaped, following `data-aggregator-mcp` convention.

### `suggest_segmentation(image_path)`
Returns two contact sheets — `visualize.colorspaces` (L,A,B,H,S,V,C,M,Y,K) and
`visualize.auto_threshold_methods` — plus a diagnostic summary. **The server never picks
silently.** It makes an informed choice cheap instead of making a blind one automatic.

### `segment(image_path, channel, method, params) -> session_id + overlay + diagnostics`
Requires an **explicit** channel and method. Returns the segmentation **overlay image**, mask
fraction, object count, solidity, and any fired warnings. **Returns no traits.**

### `measure(session_id, analyses, roi) -> traits`
Typed traits with units, plus `in_bounds` / `object_in_frame` surfaced as *additional*
information (never as validity signals). Requires a `session_id`, which only `segment` mints.

### `list_methods()`
Available channels/methods with applicability guidance, **and the pinned PlantCV version**.

### The load-bearing decision
**Splitting `segment` from `measure` is the feature.** A single `analyze(image)` call would
hand the model 17 plausible numbers having never shown it the mask — reproducing modes 1–3
exactly. Because `measure` requires a `session_id` only `segment` can produce, and `segment`
returns the overlay, the protocol **structurally forces the visual evidence into the model's
context before a number can be obtained**. The model may still ignore it; it cannot avoid
being handed it. This is the differentiator expressed as an API constraint rather than a docstring.

## 4. Automatic warnings

Two deterministic guards, each derived from an observed failure — not an imagined one.

1. **Multi-specimen** — when the top-N object areas fall within tolerance of each other and
   N ≥ 2: *"likely multiple specimens; a whole-image ROI will merge them; consider `roi.auto_grid`."*
   Would have caught mode 1 with no human looking.
2. **Frame clipping** — `in_bounds=False` promoted to an explicit warning that size traits are
   lower bounds. Covers mode 2.

## 5. Error handling — fail loud

**Hard rule: refuse to return traits on an empty or degenerate mask.** Zeros-with-green-flags
(mode 3) is the most dangerous output this server could emit. Raise instead, returning mask
fraction and object count plus a re-segmentation suggestion.

- Never treat `in_bounds` / `object_in_frame` as validity signals; compute our own gate.
- Let `readimage`'s `RuntimeError` propagate with the path intact.
- Unknown `session_id` → explicit error naming what was passed.
- Any downsampling of large images is **reported in the response**, never silent.

## 6. Testing strategy

Three controls, all required.

1. **Real execution.** Corpus = `bio3d-arena` renders (ours; no copyright encumbrance).
   ⚠️ `~/orchid-data` photos carry third-party watermarks (e.g. "© Gerrit Verhellen") and must
   **not** ship in tests, examples, or README assets. At least one test drives the full tool
   path against a real file on disk, not a synthetic array.
2. **Never trust a test you have not seen fail.** Each guard is run against its known-broken
   input and confirmed to fire *for the stated reason*: empty-mask refusal (all-zero mask);
   multi-specimen (`736_multi4.png`, measured at 9 components / top-4 near-equal); clipping
   (a render cropped to cross the frame edge). Then each guard is **mutated off** and its test
   confirmed to go red. A guard whose test passes with the guard disabled is not a test.
3. **A negative result needs a positive control in the same test.** Every guard test also
   asserts the guard does **not** fire on a clean single-plant image, in the same test function,
   so an always-fires bug cannot masquerade as working detection.

**Determinism (lens 2):** identical image + params yields an identical trait dict across repeat
runs. PlantCV version pinned and surfaced by `list_methods()`. Golden trait values pinned with
tolerance, not exact float equality, since values may shift across PlantCV releases.

## 7. Non-goals (phase 1)

No morphology traits, no iterative mask refinement, no multi-plant ROI grids, no batch
processing, no hyperspectral/thermal, no training or ML. Phase 2 adds `refine(session_id, ops)`,
morphology (leaf insertion/tangent angles, stem, skeleton segments) and `roi.auto_grid` —
all additive over the same session model.

## 8. Open questions

1. **Package name unverified.** `plantcv` is null on npm, but the GitHub check for it was in a
   batch that manufactured false nulls. Run a shape-matched check before committing to a name.
2. **Session lifetime / eviction** — masks are full-size arrays; an eviction policy and a memory
   ceiling are needed. Not yet specified.
3. **Overlay transport size** — MCP image blocks are base64; large overlays inflate context.
   Downscale policy for returned overlays is not yet specified.
