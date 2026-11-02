# Result: `threshold_inside_mode` in `suggest_segmentation`, 2026-11-01

Pre-registered in `PREREG_suggest_inside_mode.md` (`23424b5`); run on `23424b5`
with `suggest_check.py`, 4 shards. Every crop's Otsu density matched
`heldout.json` before it was scored.

**S1: PASS.** On all 739 held-out cells `suggest_segmentation` reports
`threshold_inside_mode` exactly where `segment` did: 0 mismatches.

| `heldout.json` label | cells | fires |
| -------------------- | ----: | ----: |
| OVER                 |   351 |   351 |
| RIGHT                |   354 |     6 |
| UNDER                |    34 |     0 |

Before this change, 295 of the 351 OVER cells got a recommendation of `dark`,
not ambiguous, with no warning at all. The other 56 carried another warning or
another polarity.

Control: the same script run on 12 cells with the pre-change `suggest.py`
reported 12 mismatches, so the check can fail.

The agent's own f1c_09 crop (from the F1 confirmatory run, not committed) now
returns `dark`, not ambiguous, with `threshold_inside_mode`: Otsu 128, density
0.55.
