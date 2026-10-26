# Pre-registration amendment: field families, 2026-10-24

Written before any run of this set. Anything run that is not listed here is
exploratory and is labeled so.

**Why.** The rendered set sits at ceiling (Haiku and Sonnet tied, 10-20), with
crisp edges a single color cut answers. These two families use raw tray photos
scored against hand annotations, the way the field scores its tools.

**Set.** `field_2026-10-24/questions.json`: 20 questions from tray 032 of the
Aberystwyth Leaf Evaluation Dataset (Bell and Dee 2016, CC BY 4.0). Built and
documented in [`field_2026-10-24/README.md`](field_2026-10-24/README.md).

**Caveat, carried to every result from this set.**

- **Tray 032 is not unseen.** It is the server's own held-out reporting tray
  (docs/EVAL.md): its figures were seen during development, though nothing
  was chosen from them.
- **Tray 031 is excluded**, because `count_leaves` was designed on it.
- Trays 033 and 034 have no hand annotations.

The set is reported separately from the rendered set.

**Families, n, scoring.** All windows are fixed in the key before any arm runs.

- **F1, plant area** (n = 10): px and mm², from one tray image plus the
  shipped checkerboards. A question is right only if both numbers fall in
  their windows:
  - px: the hand mask eroded and dilated by 1 px;
  - mm²: the same band after lens correction, widened by the lens-model spread
    and the pot-scale standard error.

  px-right and mm²-right are also reported separately.
- **F2, leaf count** (n = 10): primary is exact. Secondary is |DiC| ≤ 1, and
  DiC, |DiC|, MSE and % agreement are reported as in the CVPPP LCC.

**Pilot.**

- Arm and model: alt only, `claude-sonnet-5-5`, 1 rep, 20 runs.
- The alt arm has Bash and Read only: web off, enforced by the set and
  checked in every run's init event.
- The prompt, alt prompt note, validity rule and leak scan are unchanged
  from 10-20. Added: the dataset's URLs and file names are leak markers, and
  web tools present in an alt run make it invalid.

```
uv run python benchmarks/agent_qa/run.py \
  --questions benchmarks/agent_qa/field_2026-10-24/questions.json \
  --arms alt --reps 1 --model claude-sonnet-5-5 --workers 1
```

**Headroom rule.**

- A family has headroom if alt is wrong on ≥ 3 of its 10 questions (invalid
  runs counted as not right).
- Only a headroom family gets an mcp arm, run within the remaining pilot
  budget.
- That run uses the server spec of the latest pre-registration (1.17.1 at
  `30e12b3`, `--exclude-newer 2026-10-17`) unless amended before it starts.
- A tie on a full run is an acceptable result.

**Known limits, stated now.**

- **The lens shortcut barely bites.** Uncorrected mm² fails only 1 of the 10
  F1 questions (3 of 171 eligible plants). The ±1 px band (median ±13%) is
  wider than the edge distortion (median −3.7%) for most plants, so F1
  separates segmentation more than lens correction.
- **An age guess does well on F2.** The date's median count is exact on 6 of
  the 10 F2 questions.
- **Grown, overlapping rosettes are mostly excluded.** Plants touching a
  neighbour cannot be separated in the annotation (39 of 210).

## Amendment, 2026-10-26: F1 mcp re-run on 1.17.2

Written before the re-run starts. The pilot rows above it are committed with it.

**Why.** The pilot's mcp arm on F1 (1.17.1 at `30e12b3`) was 4 right, 2 wrong
and 4 timeouts at 900 s (f1_03, f1_08, f1_09, f1_10). The alt arm was 6 right.
All four timeouts were spent inside `measure_regions`, in PlantCV's
`create_labels(roi_type="partial")`. Releases 1.17.1 → 1.17.2 replace it with
a pixel-identical labeller (PR #158 for `measure_regions`, #159 for
`measure_traits`). Nothing else about the server changed what it measures.

**Run.** mcp arm only, F1 only (n = 10), 1 rep, same model, prompt, timeout,
validity rule and leak scan. Server: 1.17.2 at its release merge commit, with
the pilot's `--exclude-newer`, so only the server's own code differs.
Results go to their own file; the pilot rows are not touched.

```
uv run python benchmarks/agent_qa/run.py \
  --questions benchmarks/agent_qa/field_2026-10-24/questions.json \
  --results benchmarks/agent_qa/field_2026-10-24/rerun_2026-10-26/results.jsonl \
  --arms mcp --reps 1 --model claude-sonnet-5-5 --workers 1 \
  --only f1_01,f1_02,f1_03,f1_04,f1_05,f1_06,f1_07,f1_08,f1_09,f1_10 \
  --server-spec git+https://github.com/musharna/plantcv-mcp@6a68a6b7069a8b83503474b11d720d6a75dc29a3 \
  --exclude-newer 2026-10-17
```

**Readouts, fixed now.**

- Primary: mcp right of 10 at 1.17.2, against the pilot's alt 6 of 10,
  paired by question (both, mcp only, alt only, neither).
- Mechanism: the number of 900 s timeouts. The fix is shown to have engaged
  only if it falls from 4; any timeout left is read from its transcript, not
  assumed to be load.
- Also reported: mcp at 1.17.1 vs 1.17.2 per question, px-right and mm²-right
  separately, and cost.

**What it can and cannot say.**

- It is a pilot: 1 rep, n = 10, on tray 032, which the server was developed
  beside (see the caveat above). It is reported as descriptive.
- No confirmatory run can follow on this dataset: only trays 031 and 032 are
  hand-annotated, and 031 is excluded.
- The pilot's two wrong mcp answers came from the colour threshold and from
  `calibrate_scale_from_marker` on rim crops, which this release does not touch.
  If they stay wrong, that is the expected result, not a new finding.
