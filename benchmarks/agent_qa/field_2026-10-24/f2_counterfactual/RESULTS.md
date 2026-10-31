# F2 diagnosis, 2026-10-27: can the server's leaf counters win F2?

No model calls, no spend. The question: the pilot's mcp arm scored 2/10 exact on
F2 and never called `count_leaves` or `segment_leaves_sam`. Was that the agent's
choice, or can the server not do better?

## Why the agents counted by eye

All 10 F2 mcp transcripts make 1-3 calls: `Read` the tray, `zoom_image`, then a
count by eye. Both leaf counters were offered (init event, 17 plantcv tools).
They cannot reach the asked plant: every one of the 18 tools on 1.17.2
(`origin/master` 6a68a6b, read from the AST) takes a whole image or a session, and
none takes a box or region. `zoom_image` returns a picture, not a file. The
counters' docstrings say "crop to one plant" and "isolate one plant first", and
the mcp arm has no Bash to crop with. So on a 4x5 tray there is no path from
"row 3, column 3" to a count. The pilot server also lacked the `[sam]` extra, so
`segment_leaves_sam` would have refused anyway.

## If the agent could crop (`f2_counterfactual.py`, `rows.json`)

plantcv-mcp[sam] 1.17.2, CPU. Positive control first: `scripts/eval_leaf_count_sam.py`
re-run on Tray 032 reproduces docs/EVAL.md exactly (young −0.60 / 1.10 / 6 of 20,
grown +0.23 / 1.77 / 2 of 13). F2's truth equals EVAL's for every matched plant
(asserted). Two reruns gave identical counts.

- **A, counter ceiling:** EVAL's path with the plant's box from the annotation.
- **B, realistic:** the pot cell the question names, polarity from `polarity_report`.

| | exact | \|DiC\| ≤ 1 | DiC | \|DiC\| |
|---|---|---|---|---|
| alt, pilot (by eye) | 1/10 | 4/10 | −3.80 | 3.80 |
| mcp, pilot (by eye) | 2/10 | 3/10 | −3.50 | 3.50 |
| A `count_leaves` | 0/10 | 0/10 | −7.80 | 7.80 |
| A `segment_leaves_sam` | 1/10 | 5/10 | +1.90 | 2.30 |
| B `count_leaves` | 0/10 | 0/10 | 9 answered | 16.89 |
| B `segment_leaves_sam` | 1/10 | 2/10 | 8 answered | 7.75 |
| date median (shortcut) | 6/10 | 8/10 | −0.90 | 1.30 |

- **Even with a perfect crop the server does not win F2's primary.** SAM is exact
  on 1 of 10, the agents on 1 and 2. On the secondary it is ahead (5 vs 3-4 of 10),
  which n = 10 cannot separate from noise. SAM over-counts grown plants (+3 to +5 on
  f2_05, f2_06, f2_08-f2_10); the watershed under-counts every plant.
- **A pot cell is not a plant box.** f2_01: the pot rim segments as a closed ring
  and `fill_holes` floods it (99.6% mask, refused). On grown plants (f2_08-f2_10)
  the plant covers over half the pot, and `polarity_report`'s rule ("the polarity
  covering less of the frame is the plant") picks the soil.
- **F2 does not discriminate.** The date's median count beats every arm and tool,
  as the PREREG warned ("An age guess does well on F2").

## Verdict

F2 stays a pilot result with no winner and no re-run. Two server seams are real and
worth fixing for users; neither would flip F2:
1. no tool restricts segmentation or a leaf count to one region of a tray;
2. `polarity_report` assumes the plant is the smaller side, which fails on a crop
   the plant fills.

## Since then (added 2026-10-31)

Both seams were fixed in 1.18.0 (#163): `crop_image` cuts a session to a box, and the
polarity rule now prefers the greener side instead of the smaller one. The SAM
over-count above was soil that `fill_holes` encloses in a grown rosette, counted as
leaves; 1.18.0 (#168) stopped counting it. Path A re-run on 1.18.0 for the five grown
plants only (the young five were not re-run):

| path A, `segment_leaves_sam`, f2_05 f2_06 f2_08 f2_09 f2_10 | counts         | exact | \|DiC\| ≤ 1 | DiC  |
| ------------------------------------------------------------ | -------------- | ----- | ----------- | ---- |
| 1.17.2 (above)                                               | 27 21 26 28 33 | 0/5   | 0/5         | +3.8 |
| 1.18.0                                                       | 21 16 22 21 29 | 2/5   | 3/5         | −1.4 |

Truth is 23 16 23 25 29. The remaining error is an undercount of the small leaves
packed at the rosette centre; docs/EVAL.md (2026-10-29 and 2026-10-31) has the trace.
No agent arm was re-run, so the verdict above stands as a pilot result.
