# `refine` — mask refinement as a session→session operation (0.6.0)

**Parent:** `2026-08-27-backlog-integration-plan-of-attack.md`, sub-project A. **Status:** approved as part of the roadmap; owner delegated design detail.

## What it is

`refine(session_id, ops)` applies a validated list of morphological operations to a
session's mask and mints a **new** session for the result. The original session is
untouched and still measurable, so a refinement that turns out wrong is discarded,
not undone. It returns the overlay of the refined mask — the picture-before-number
rule — plus before/after diagnostics and warnings.

## Ops

| op             | params                                    | PlantCV call                              | notes                                   |
| -------------- | ----------------------------------------- | ----------------------------------------- | --------------------------------------- |
| `fill_holes`   | —                                         | `pcv.fill_holes(bin_img)`                 | flood-fills enclosed background         |
| `fill`         | `size ≥ 0`                                | `pcv.fill(bin_img, size)`                 | drops components smaller than `size` px |
| `erode`        | `ksize ≥ 2`, `iterations ≥ 1` (default 1) | `pcv.erode(gray_img, ksize, i)`           | PlantCV rejects `ksize < 2`             |
| `dilate`       | same                                      | `pcv.dilate`                              |                                         |
| `opening`      | `ksize ≥ 2`                               | `pcv.opening(gray_img, kernel=ones(k,k))` | erode→dilate: removes specks            |
| `closing`      | `ksize ≥ 2`                               | `pcv.closing(...)`                        | dilate→erode: closes gaps               |
| `median_blur`  | odd `ksize ≥ 3`                           | `pcv.median_blur(gray_img, ksize)`        | cv2 requires odd                        |
| `keep_largest` | `n ≥ 1`                                   | ours, via connected components            | the op users want and PlantCV lacks     |

Validation is **all-or-nothing**: every op is checked (name, required params, ranges,
no unknown params) before any is applied; failure raises `RefineSpecError` naming the
offending op index. Ops apply in the order given. The list must be non-empty.

## Session model

`Session` gains `lineage: list[dict]` (ops applied to reach this mask, in order,
cumulative across chained refinements) and `parent_id: str | None`. `refine` copies
`image_path`, `digest`, `shape`, `channel`, `method`, `color_correct` from the parent.
`measure()` and `measure_regions()` results carry `lineage` so a trait table states how
its mask was made; an unrefined session has `lineage: []`.

## Guards

- Result degenerate (`assert_not_degenerate` fails) → **refused**, no session minted:
  `RefinementErasedMaskError` carrying before/after `mask_fraction` and
  `component_count`. An erosion that deletes the plant must not become a session
  that then measures zeros.
- Otherwise advisory warnings from the shared diagnostics: `implausible_coverage`,
  `multi_specimen`, `frame_clipping`; plus `refine_large_change` when
  `|after.mask_fraction − before.mask_fraction| / before.mask_fraction > 0.25` — the
  refinement changed more than a quarter of the mask, look at the overlay.

## Result

```
session_id, parent_session_id, ops (as applied), lineage,
before: {mask_fraction, component_count, largest_area},
after:  {mask_fraction, component_count, largest_area},
overlay_scale, warnings, engine, + the overlay image block
```

`list_methods()` gains `refine_ops`: `{name: {param: constraint}}`.

## Tests

- Ops table: each op on a synthetic mask asserts the intended effect (`fill_holes`
  closes a 1-px hole; `keep_largest(1)` drops a speck; `erode` shrinks; etc.).
- Known-value eval: a disc of known radius with salt-and-pepper noise and an interior
  hole; `[fill_holes, keep_largest(1)]` recovers `area` within 1% of the clean disc,
  after a positive control that the noisy mask measures wrong.
- Validation: unknown op, missing param, out-of-range param, unknown param, empty list
  → `RefineSpecError` with the op index; positive control in the same test.
- Refusal: an erosion that empties the mask → `RefinementErasedMaskError`, no new
  session, parent still measurable.
- Lineage: chained refinements accumulate; `measure()` echoes it.
- MCP layer: `call_tool("refine", ...)` returns JSON + image; tool-list assertions
  updated in `test_server.py`, `ci.yml`.
