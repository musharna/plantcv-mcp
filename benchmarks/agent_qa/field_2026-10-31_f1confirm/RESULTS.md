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
  7% high, from the arm's own rim scale.
- **mcp, 7 wrong: too many pixels on young plants, and the rim scale.**
  - 5 are plants 22 to 30 days after sowing (f1c_02, 03, 04, 07, 09). Four of them
    have the pixel count 21% to 51% high. The transcripts show `segment` with Otsu
    on the `a` channel of a crop around the pot, sometimes with a large `fill_size`,
    and not always `keep_largest`; which of those adds the pixels was not traced.
  - The scale: the runs that passed a scale from `calibrate_scale_from_marker` on
    the pot rim used 7.02 to 7.23 px/mm (f1c_07, 09, 23) where the key's is 7.43,
    and f1c_03's answer implies 6.85. By itself that raises mm² by 6% to 12% (18% at
    6.85). Two miss on mm² with the pixel count inside its window: f1c_03 (26% high)
    and f1c_23 (6% low, so the scale is not the whole story there). The pilot saw the
    same tool on rim crops.
  - f1c_21 (42 days) is the one mcp undercount: 18% low.

Split by age, after the fact (exploratory, not pre-registered): on the 21 plants
older than 30 days mcp is right on 19 and alt on 11; on the 9 younger ones, 4 and 5.

## What it says

- On fresh draws from a tray the pilot did not use, the current server answers more
  F1 questions than Bash with PlantCV (23 against 16), but not by enough for this
  pre-registered test at n = 30. The pilot's 10 of 10 did not carry over: young
  plants and the rim scale cost the server 7.
- No further confirmatory run is possible on this dataset: trays 031 and 032 are the
  only annotated ones, and both are now used.
- Both server faults are reproducible on these questions and worth fixing for users:
  extra pixels on small rosettes in a pot crop, and a low rim scale from
  `calibrate_scale_from_marker`. A fix tested on these 30 would be exploratory.

## Files

- `questions.json`: the key (built by `field_2026-10-24/build.py`, flags in the PREREG).
- `results.jsonl`: one row per run. `versions.json`: the arms' environments.
- Transcripts: `~/plantcv-mcp-artifacts/agent_qa/f1confirm-2026-10-31/`.
