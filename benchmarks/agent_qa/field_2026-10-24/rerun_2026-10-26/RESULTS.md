# F1 mcp re-run on 1.17.2, 2026-10-26

Pre-registered in `PREREG_2026-10-24_field-families.md`, "Amendment, 2026-10-26"
(a7905c8, 981d187; committed before the run). A pilot: 1 rep, n = 10, tray 032.
Descriptive, not confirmatory.

| question | alt (pilot) | mcp 1.17.1 (pilot) | mcp 1.17.2 |
|---|---|---|---|
| f1_01 | right | right, 79 s | right, 80 s |
| f1_02 | right | wrong, 173 s | right, 99 s |
| f1_03 | wrong | timeout, 900 s | right, 141 s |
| f1_04 | right | wrong, 91 s | right, 130 s |
| f1_05 | right | right, 346 s | right, 102 s |
| f1_06 | wrong | right, 352 s | right, 89 s |
| f1_07 | right | right, 742 s | right, 108 s |
| f1_08 | wrong | timeout, 900 s | right, 112 s |
| f1_09 | right | timeout, 900 s | right, 87 s |
| f1_10 | wrong | timeout, 900 s | right, 120 s |
| **right** | **6/10** | **4/10** | **10/10** |

- **Primary.** mcp 1.17.2 is right on 10 of 10, against the pilot alt arm's 6.
  Paired by question: both 6, mcp only 4, alt only 0, neither 0. An exact sign
  test on 4:0 gives p = 0.125. With n = 10 and one rep that is not a win by any
  rule this set has; it is reported as a pilot.
- **Mechanism: engaged.** 900 s timeouts fell from 4 to 0. The four pilot
  timeouts (f1_03, f1_08, f1_09, f1_10) all answered right, in 87-141 s. Every
  re-run took 80-141 s; the pilot's right answers took 79-742 s.
- **px and mm² both right** on all 10. All 10 rows valid, no leak flags, no
  key access, no API failures.
- **The two flipped wrong answers are not the fix.** f1_02 and f1_04 were wrong
  at 1.17.1 and right at 1.17.2. The amendment expected them to stay wrong, and
  the transcripts show why they did not: the agent chose differently
  (f1_02 a single `a`-channel threshold instead of an `a` and `b` rule;
  f1_04 calibrated on another pot rim and re-segmented twice more). The server's
  segmentation and calibration are unchanged between the releases, so this is
  run-to-run variation that one rep cannot separate from an effect.
- **So, of the gain from 4 to 10:** 4 are the fix (timeouts to answers) and
  2 are agent variation. Against alt, the 4 mcp-only questions are f1_03,
  f1_08 and f1_10 (pilot timeouts) and f1_06 (right at 1.17.1 too).
- **Cost:** $1.30 by the runs' own records (10 runs). The spend ledger
  moved $137.45 → $138.64.
- **Not re-run:** the alt arm (its 6/10 is one rep from 10-24) and F2.

## Since then (added 2026-10-31)

The confirmatory run on 30 fresh tray 031 questions did **not** confirm this gap:
mcp 1.18.1 right on 23, alt on 16, paired 11:4, p = 0.118 against a pre-registered
p < 0.05. Young rosettes cost the server 5 of its 7 misses (Otsu overcounts a plant
that is a few percent of the crop; the rim scale, first blamed too, was not at
fault). See
[`../../field_2026-10-31_f1confirm/RESULTS.md`](../../field_2026-10-31_f1confirm/RESULTS.md).
