# F1 confirmatory on tray 031: NOT CONFIRMED

Pre-registered in [`../PREREG_2026-10-24_field-families.md`](../PREREG_2026-10-24_field-families.md)
(amendment 2026-10-31, pushed as `a9285c8` before the draw; key `ccead11`, sha256
`1934e4a3…`, pushed before the run). Run 2026-10-31, 20:29–21:25 EDT, launcher case
`plantcv-f1confirm`. 60 runs, none invalid, no timeouts, no API failures, no leaks,
no `text_mentions`. Cost $3.99 (alt $1.75, mcp $2.24).

## The rule's result

| 30 questions, 1 rep, `claude-sonnet-5-5` | right | px right | mm² right | median s |
| ---------------------------------------- | ----- | -------- | --------- | -------- |
| alt (Bash + Read, PlantCV 4.11.3)        | 16    | 19       | 18        | 94       |
| mcp (plantcv-mcp 1.18.1, `287a049`)      | 23    | 25       | 26        | 112      |

Paired by question: both right 12, mcp only 11, alt only 4, neither 3. The two-sided
exact sign test on 11 against 4 gives p = 0.118. The rule needed p < 0.05, so **the
pilot's F1 gap is not confirmed**. The server is ahead on every readout, but this run
cannot tell that lead from chance.

The prediction (mcp at least 27, alt about 18) was wrong for mcp.

## How each arm is wrong

Read from the replies against the key. Not part of the rule.

- **alt, 14 wrong: too few pixels.** 11 of 14 have the pixel count low, by 4% to 34%;
  its colour thresholds leave leaf pixels out. The other 3 miss on mm² alone, 6% to
  7% high; the cause was not traced (each run corrects the lens in its own frame, so
  its scale cannot be read against the key's 7.43).
- **mcp, 7 wrong: too many pixels on young plants.**
  - 5 are plants 22 to 30 days after sowing (f1c_02, 03, 04, 07, 09). Four of them
    have the pixel count 21% to 51% high.
  - Traced on 2026-11-01 (exploratory): Otsu on the `a` channel puts its threshold
    inside the soil's own range when the plant is a few percent of the crop, and the
    soil specks touching the leaves survive `fill_size` and `keep_largest`. f1c_09's
    crop reproduces the reply exactly (12,520 px against 8,280): the plant is 5.3% of
    it, Otsu cuts at 128, the soil runs 124 to 137 and the plant tops out at 122. On
    40 annotated pot crops (these 30 and the pilot's 10), Otsu overcounts all 14
    plants under about 5% of the crop and none of the larger ones. The server's
    `method="triangle"` could not help: PlantCV 4.11.3's triangle thresholds with
    Otsu (plantcv-mcp #180).
  - f1c_21 (42 days) is the one mcp undercount: 18% low.
- **The scale is not a server fault (corrected 2026-11-01).** An earlier version of
  this file set the runs' scales from `calibrate_scale_from_marker` (6.93 to 7.28
  px/mm) against the key's 7.43 and charged them 6% to 12% of mm². The 7.43 belongs
  to the key's corrected frame. The server's `correct_lens_distortion` returns a
  frame about 3.5% smaller (interior pots 428 px across against 444 in the raw
  image), where the pots give about 7.13 px/mm; the runs were within 3% of that.
  f1c_09's mm² was inside its window. Of the two mm² misses with the pixel count
  right, a scale 1.4% high (7.23) explains about half of f1c_23's 6% shortfall
  (window ±5.5%); f1c_03's 26% excess is mostly its corrected-image mask.

Split by age, after the fact (exploratory, not pre-registered): on the 21 plants
older than 30 days mcp is right on 19 and alt on 11; on the 9 younger ones, 4 and 5.

## What it says

- On fresh draws from a tray the pilot did not use, the current server answers more
  F1 questions than Bash with PlantCV (23 against 16), but not by enough for this
  pre-registered test at n = 30. The pilot's 10 of 10 did not carry over: young
  plants cost the server 5 of its 7.
- No further confirmatory run is possible on this dataset: trays 031 and 032 are the
  only annotated ones, and both are now used.
- The young-plant overcount is reproducible on these questions and worth fixing for
  users. A fix tested on these 30 would be exploratory.

## Files

- `questions.json`: the key (built by `field_2026-10-24/build.py`, flags in the PREREG).
- `results.jsonl`: one row per run. `versions.json`: the arms' environments.
- Transcripts: `~/plantcv-mcp-artifacts/agent_qa/f1confirm-2026-10-31/`.
- `diag_2026-11-01/`: the scripts and data behind the traced overcount.
