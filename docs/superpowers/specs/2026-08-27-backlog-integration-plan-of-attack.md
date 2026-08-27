# Plan of attack: integrating the post-0.5.0 backlog

**Date:** 2026-08-27 · **Baseline:** v0.5.0 (`c64c6b5`), 7 tools, 140 tests · **Status:** ordering APPROVED by the owner 2026-08-27; sub-project specs follow one at a time

This is the ordering-and-scoping document for the five remaining backlog items. Each
numbered sub-project below gets its own spec → implementation plan → release cycle;
nothing here is implementation. Every claim about PlantCV was checked against the
installed 4.11.3 by introspection on the date above, not recalled.

## The five items, and what grounding changed about them

| #   | Item                                                     | What introspection showed                                                                                                                                                                                                                                                                                         | Consequence                                                                                                                                                                                                                   |
| --- | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A   | Mask refinement (`refine`)                               | PlantCV has the primitives: `fill`, `fill_holes`, `erode`, `dilate`, `opening`, `closing`, `median_blur`. None writes `pcv.outputs`.                                                                                                                                                                              | Cheap, self-contained, and a hard prerequisite for B.                                                                                                                                                                         |
| B   | Morphology traits                                        | 19 `pcv.morphology` functions; the pipeline is `skeletonize → prune → segment_skeleton → segment_sort → segment_id → {path_length, euclidean_length, curvature, angle, tangent_angle, insertion_angle, width} + find_tips/find_branch_pts/check_cycles/analyze_stem`. **15 of 19 report only via `pcv.outputs`.** | Must run under the existing `PCV_OUTPUTS_LOCK`. Skeleton traits are meaningless on noisy masks → A first.                                                                                                                     |
| C   | "Drop `pcv.outputs` reliance / worker-process isolation" | B makes dropping `pcv.outputs` impossible without re-implementing PlantCV's morphology reporting.                                                                                                                                                                                                                 | **Pivot:** the deliverable is _process isolation_ — each analysis in a worker subprocess. This also gives the one thing validation cannot: a native crash (SIGSEGV) becomes a tool error instead of killing the stdio server. |
| D   | Read-root allow-list                                     | Trust boundary is documented (README §"Security and trust boundary"); `main()` has no CLI parsing today.                                                                                                                                                                                                          | Small; add `PLANTCV_MCP_ROOTS` + `--root`; default stays unrestricted.                                                                                                                                                        |
| E   | Hyperspectral + thermal                                  | ENVI reader is self-contained (`hyperspectral/read_data.py` imports only cv2/numpy); `flyr` and `nd2` are already hard deps of plantcv; `analyze.spectral_index`, `analyze.spectral_reflectance`, `analyze.thermal`, and 31 `spectral_index.*` functions exist. No new packages.                                  | Feasible, but a **new session type** (cube / float array), i.e. the largest change. Last.                                                                                                                                     |

## Ordering and releases

```
0.6.0  A  refine                     ~1 session   unblocks B; teaches the new-tool pattern
0.7.0  B  morphology traits          ~2–3         the feature users will ask for
0.8.0  C  worker-process isolation   ~1–2         crash containment + global-state isolation; opt-in first
0.9.0  D  read-root allow-list       ~0.5         needed only before any non-local deployment
1.0.0  E  hyperspectral + thermal    ~3–4         new modalities; 1.0 = tool surface complete
```

Why this order and not another:

- **A before B** — morphology is the only consumer that _needs_ refinement, and B's eval fixture (a synthetic plant with known leaf angles) is the same fixture that proves A does not distort geometry.
- **B before C** — C's benefit is measured against real workloads; B is the workload that exercises `pcv.outputs` hardest (15 writers per call) and the one most likely to hit native code with odd inputs.
- **C before E** — E's cubes are the largest payloads; C's worker design must be fixed (pickling cost, spawn vs fork) before the payloads grow, or C gets redesigned twice.
- **D is order-independent** — it sits at 0.9 only because it is the thing to do _right before_ anyone exposes the server beyond a local stdio client, which E's data (often on shared storage) makes more likely.

Each release keeps the invariants 0.5.0 established: every number-bearing result carries `engine`, `warnings`, and — for anything that segments — the overlay it came from; version moves across all 7 surfaces; CI's exact-tool-list assertions (`tests/test_server.py`, `tests/test_tool_layer_sequence.py`, `ci.yml` wheel smoke) are updated in the same change.

## Design gates applied (mcp-server-seams)

- **Shell-out gate:** PlantCV has no CLI for any of these; the MCP surface earns its keep through session state (masks, lineage), typed schemas, and the overlay-before-number rule. Pass.
- **Retrieval vs compute:** all five are compute. Open field.
- **Non-compute lenses each sub-project must carry:** provenance (`engine` + a `lineage`/`recipe` record of every op that produced the mask), determinism (pinned PlantCV; known-geometry evals), QC (advisory/blocking warnings, refusal over zeros), units (`px_per_mm` conversion table extended for new linear/angular traits), large-result handling (E: a 200-band mean spectrum is opt-in, summarised by default), eval (synthetic ground truth per sub-project, as `tests/test_eval_known_geometry.py` does today).
- **Positive controls:** every refusal test asserts the legitimate path succeeds in the same test (repo rule 3 in `CONTRIBUTING.md`).

## Sub-project A — `refine` (0.6.0)

**Tool:** `refine(session_id, ops: list[{op, ...params}]) -> new session_id + overlay + diagnostics + delta + warnings`

- **Ops** (thin, named after PlantCV): `fill_holes`, `fill(size)`, `erode(ksize, iterations)`, `dilate(ksize, iterations)`, `opening(ksize)`, `closing(ksize)`, `median_blur(ksize)`, plus one of our own, `keep_largest(n)` (via `diagnostics.component_areas`) — the op users actually want and PlantCV lacks.
- **Session model:** refinement mints a **new** session (the original stays measurable); `Session` gains `lineage: list[dict]` (ops applied, in order) and `parent_id`. `measure()`/`measure_regions()` results echo `lineage` so a trait table can say how its mask was made.
- **Guards:** same `segmentation_warnings` as `segment()`; an op sequence that empties the mask or trips `fill_erased_mask` is **refused** (no session minted) with the before/after numbers; `delta` reports mask_fraction and component_count before → after so a large erosion is visible even when not blocking.
- **Picture rule:** returns the overlay of the refined mask — refinement without seeing the result is exactly how a hole-fill swallows a second plant.
- **Eval:** synthetic disc with salt-and-pepper noise and one interior hole → `fill_holes` + `keep_largest(1)` recovers area within 1% of the clean disc; positive control that the noisy mask measures _wrong_ first.
- **Files:** new `refine.py` (ops table + validation), `session.py` (lineage/parent), `server.py` (tool + INSTRUCTIONS), tests `test_refine.py` + MCP-layer case in `test_server.py`.

## Sub-project B — morphology traits (0.7.0)

**Tool:** `measure_morphology(session_id, prune_size: int = 0, tangent_size: int = 5, px_per_mm=None) -> per-plant scalars + per-segment table + labelled-segment overlay`

- **Pipeline (single plant, v1):** `skeletonize(mask) → prune(skel, prune_size, mask) → segment_skeleton → segment_sort(first_stem=True) → segment_id`; then per segment `segment_path_length`, `segment_euclidean_length`, `segment_curvature`, `segment_angle`, `segment_tangent_angle(size=tangent_size)`, `segment_insertion_angle(size)`, `segment_width`; per plant `find_tips`, `find_branch_pts`, `check_cycles`, `analyze_stem` (stem height/angle).
- **Multi-plant:** deferred to B.2 — run the pipeline per `measure_regions` cell. v1 refuses `multi_specimen` masks by name, pointing at `measure_regions` + `refine(keep_largest)`.
- **Guards:** empty skeleton → refuse; `check_cycles` cycles > 0 → advisory (`skeleton_has_cycles`, means the mask still has holes → point back to `refine`); zero leaf segments after `segment_sort` → refuse (`no_leaf_segments`); `prune_size` sensitivity advisory when segment count changes by >30% between `prune_size` and `2×prune_size` (computed, cheap). Angular traits are unit-labelled `degrees` and excluded from `px_per_mm` scaling; path/euclidean lengths and width are added to `LINEAR_TRAITS`.
- **Global state:** every call runs inside `isolated_pcv_outputs()`; the per-segment reads key on the `label` PlantCV assigns — the same keyed-read discipline as `regions._read_group`.
- **Picture rule:** overlay = `segment_id`'s labelled image over the RGB; the per-segment table indexes into it.
- **Eval:** synthetic plant — a vertical stem with N leaves drawn at known insertion angles and known lengths — asserting recovered angles within 5° and lengths within 5% (the same approach as `tests/test_eval_known_geometry.py`), plus a fresh-process tool-layer run.
- **Files:** new `morphology.py`, `measurement.py` (trait tables), `server.py`, `test_morphology.py`, `test_eval_known_geometry.py` (extend).

## Sub-project C — worker-process isolation (0.8.0)

**Deliverable:** every PlantCV analysis (`measure_traits`, `regions.measure_regions`, morphology, and E later) runs in a worker subprocess; the server process never executes native PlantCV/OpenCV analysis code.

- **Why, restated:** (1) a native crash — the SIGSEGV class 0.5.0 closed by _validation_ — becomes a `ToolError` instead of a dead server; validation stays, isolation is the backstop. (2) `pcv.outputs` and `pcv.params` are then per-worker; the lock remains (it is cheap and correct) but no longer load-bearing. (3) memory from large images is reclaimed when a worker is recycled.
- **Shape:** `multiprocessing` with the **spawn** context (the server has anyio worker threads; forking a threaded process is unsafe), a pool of 1–2 warm workers, `maxtasksperchild` for recycling, arrays passed as pickled numpy (measured, not assumed — see gate below). Worker death → `ToolError("analysis worker crashed (signal N); the server is still running; re-run segment()")`.
- **Gate before committing:** measure round-trip overhead on a 3000×3000 image (the existing `_write_huge_green_png` fixture). If overhead > 25% of analysis time, ship as **opt-in** (`--isolate`, default off) and revisit; if < 25%, default on.
- **Regression:** a test-only hook that makes the worker call `os.abort()` — the server must return a `ToolError` and answer the _next_ call. Positive control: the same call without the hook measures normally.
- **Files:** new `workers.py`, `server.py` (routing + CLI flag), `tests/test_isolation.py`; `test_concurrency.py` gains the cross-process case.

## Sub-project D — read-root allow-list (0.9.0)

- `PLANTCV_MCP_ROOTS` (`os.pathsep`-separated) and/or repeated `--root PATH` on the console script; `main()` grows `argparse`. When any root is set, every path argument (`segment`, `suggest_segmentation`, `calibrate_scale_from_marker`, `measure_images`, and E's readers) must `realpath` to a location under a root; otherwise refused naming the configured roots. Unset → unchanged behaviour, still documented as the trust boundary.
- **Tests:** `..` traversal, a symlink pointing outside a root, an absolute path outside; positive control: a path inside measures. README §"Security and trust boundary" and `SECURITY.md` updated.
- **Files:** new `paths.py`, `server.py`, `tests/test_paths.py`.

## Sub-project E — hyperspectral + thermal (1.0.0)

- **Sessions become typed:** `Session.kind ∈ {"rgb", "hsi", "thermal"}`; a mask is always uint8 HxW, but the source array differs (`Spectral_data` cube; float °C array). `measure()` on a non-RGB session refuses with the right tool name.
- **Hyperspectral tools:** `segment_hyperspectral(envi_path, index="ndvi", method, object_type, ...)` — computes one `spectral_index.*` image, thresholds it with the existing `threshold_mask` machinery, returns the **pseudo-RGB overlay** (`Spectral_data.pseudo_rgb`) and diagnostics; `measure_spectral(session_id, indices=[...], include_spectrum=False)` — `analyze.spectral_index` per requested index (mean/median/std) and, opt-in, `analyze.spectral_reflectance` (the full mean spectrum is the large-result case: hundreds of bands, off by default, band count reported always).
- **Thermal tools:** `segment_thermal(path, min_c, max_c)` for FLIR radiometric JPEG (`flyr`) or CSV; `measure_thermal(session_id)` via `analyze.thermal` (mean/median/max/min °C + optional histogram). A thermal mask cannot be borrowed from an RGB session — different sensor, different frame — so thermal segments on temperature itself.
- **Evals:** ENVI cubes are synthesisable with `pcv.hyperspectral.write_data` (a cube whose NDVI is known per pixel); thermal CSV arrays likewise. FLIR JPEG needs a real file — PlantCV's own test data ships one; vendor it under a documented licence or mark the FLIR path "tested against PlantCV's sample only".
- **Decision (owner, 2026-08-27):** validate against real public data — source an ENVI plant cube and FLIR radiometric plant JPEGs online (PlantCV's own test-data samples are the fallback), vendored or fetched under a documented licence; synthetic cubes/CSVs remain the known-value evals. Sourcing that data is the first task of E's plan.
- **Files:** new `hyperspectral.py`, `thermal.py`, `session.py` (kind), `imaging.py` (cube/CSV readers with the same read-once-hash-decode discipline), `server.py`, `tests/test_hyperspectral.py`, `tests/test_thermal.py`.

## What is deliberately not in scope

- Auto-selecting `prune_size` or thresholds (a guessed parameter is the failure mode this server exists to prevent — report sensitivity, do not guess).
- A generic "run any PlantCV function" tool (destroys the guards; the registry has such wrappers already).
- Streaming/HTTP transport (D would have to come first, and nobody has asked).

## Next step

Sequence approved. Sub-project A gets a spec (`docs/superpowers/specs/2026-08-xx-refine-design.md`) and an implementation plan via the writing-plans skill, executed with TDD as 0.5.0 was.
