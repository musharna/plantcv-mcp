# `measure_morphology` — skeleton-based traits (0.7.0)

**Parent:** `2026-08-27-backlog-integration-plan-of-attack.md`, sub-project B.
**Grounding:** every PlantCV behaviour below was measured on 4.11.3 with a synthetic
plant (vertical 9-px stem, three 7-px leaves at 30°/45°/60° from vertical, lengths
90/80/70 px) on 2026-08-27.

## Tool

`measure_morphology(session_id, prune_size=15, tangent_size=15, px_per_mm=None)`
→ per-plant scalars + per-segment table + the **numbered-segment overlay** (image block).

## Pipeline (PlantCV calls, in order, inside `isolated_pcv_outputs()` with `label="morphology"`)

```
skeletonize(mask)
prune(skel, size=prune_size, mask)                → pruned, _, _
segment_skeleton(pruned, mask)                    → _, segments
segment_sort(pruned, segments, mask, first_stem)  → leaf_objects, stem_objects
segment_id(pruned, leaf_objects, mask)            → segmented_img (the overlay), _
segment_path_length / segment_euclidean_length / segment_curvature /
segment_angle / segment_tangent_angle(size) / segment_insertion_angle(size)
find_tips / find_branch_pts / check_cycles / analyze_stem(rgb, stem_objects)
segment_width(segmented_img, pruned, labeled_mask, n_labels=1)
```

15 of these write `pcv.outputs`; per-segment traits arrive as lists indexed by the
segment ids drawn on the overlay, which is what makes the table readable.

## Measured PlantCV behaviours the design must absorb

| Observation                                                                                                                          | Design response                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `segment_sort` returned 4 "leaves" for 3 drawn: the stem tip above the last junction is classed as a leaf (angle 90°, insertion 0°). | Report what PlantCV reports; do not filter. `leaf_count` is PlantCV's count, documented as "leaf-like segments".                                                                                                              |
| Leaf path lengths came out 80/71/68 for 90/80/70 drawn (junction trimming + skeleton starting inside the stem). Ordering preserved.  | Eval asserts ordering and length within **15%**; documents the systematic under-estimate.                                                                                                                                     |
| Insertion angles 38.6/50.3/64.9 for 30/45/60 drawn. Ordering preserved.                                                              | Eval asserts ordering and angle within **12°**. Advisory text tells users the angles are relative, not absolute, without a calibration image.                                                                                 |
| `analyze_stem` returned `stem_angle = -14373°` for a perfectly vertical stem (slope → 2.3e7, PlantCV prints "cannot be plotted").    | **Guard:** `stem_angle` outside [-180, 180] → value `null` + warning `stem_angle_undefined` ("stem is (near-)vertical; PlantCV's slope-based angle is undefined"). A number that is not an angle must not be returned as one. |
| `tips` includes stem base and top (5 for 3 leaves).                                                                                  | Report `tip_count` as PlantCV's; the overlay shows them.                                                                                                                                                                      |
| `segment_euclidean_length`/`path_length` also write a `backend` group.                                                               | Read only the `morphology` label group; the lock+restore discards the rest.                                                                                                                                                   |
| `prune(size)` changes segment count; a bad `prune_size` fragments or merges leaves.                                                  | **Sensitivity advisory** `prune_size_sensitive`: rerun `segment_skeleton` at `2×prune_size`; if the segment count changes by >30%, warn.                                                                                      |
| Empty/near-empty skeleton on a degenerate mask.                                                                                      | Reuse `assert_not_degenerate`; zero segments after sort → **refuse** (`NoSkeletonSegmentsError`) naming `refine()`; `num_cycles > 0` → advisory `skeleton_has_cycles` pointing at `refine(fill_holes)`.                       |
| Multi-plant masks: skeletons merge across plants.                                                                                    | v1 refuses `multi_specimen` masks by name (points at `measure_regions` + `refine(keep_largest)`).                                                                                                                             |

## Calibration: `tangent_size` (measured 2026-08-27, three leaves at 60/45/30°)

| tangent_size | max bias | note |
|---|---|---|
| 10 | 24.5° | |
| 15 | 14.1° | |
| 20 | 8.6° | |
| 25 | **5.9°** | **default** — margin before the cliff |
| 30 | 4.2° | |
| 40 | 30.0° | the 30°/70-px leaf collapses to 0.0: window longer than the leaf |

Guard `tangent_window_exceeds_segment` fires when `tangent_size` exceeds any leaf
segment's path length. `stem_height` is PlantCV's: base → topmost junction (202 px
for a 260 px drawn stem whose tip is sorted as a leaf).

## Result

```
session_id, lineage, prune_size, tangent_size, px_per_mm,
plant: { leaf_count, stem_count, tip_count, branch_point_count, num_cycles,
         stem_height, stem_length, stem_angle (nullable),
         mean_segment_width, segment_width_std, segment_width_max },
segments: [ { id, path_length, euclidean_length, curvature, angle,
              tangent_angle, insertion_angle } ... ]   # id = number on the overlay
units: { path_length: "pixels"|"mm", ... angles: "degrees", curvature: "ratio" },
warnings, overlay_scale, engine, + image block (segment_id overlay)
```

Lengths/heights/widths are linear traits (`px_per_mm` converts them); angles are
degrees and never scaled; curvature is a ratio.

## Tests

- Pipeline on the synthetic plant: 3 drawn leaves recovered among the segments with
  ordering preserved, lengths within 15%, insertion angles within 12°; `num_cycles == 0`;
  `stem_height` within 10% of 260 px; positive control that a _different_ plant
  (two leaves) yields a different segment count.
- Vertical stem → `stem_angle` is `null` with `stem_angle_undefined`; a tilted stem
  (drawn at 20° from vertical) → a finite angle, no warning (positive control).
- Ring-shaped mask → `skeleton_has_cycles` advisory; refined with `fill_holes` → gone.
- Empty region / `multi_specimen` mask → refused by name.
- `px_per_mm` scales only linear traits; angles unchanged.
- MCP layer: JSON + image block; tool-list assertions updated (9 tools).
