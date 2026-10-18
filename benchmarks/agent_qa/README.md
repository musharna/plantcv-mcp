# Agent benchmark: measuring plant images with and without the server

We asked Claude Opus 5.5 46 measurement questions about rendered plant images,
each with an exactly known answer, three times each: with no tools, with Bash
and its own Python (PlantCV and OpenCV installed), and with plantcv-mcp plus
Read. Averaged over questions, Claude's own code was right on 95.7% of answers
and the server on 86.2%, and the two are not separated on accuracy (exact McNemar p = 0.18 over
questions: 7 questions were right in every run with Bash only, 2 with the
server only). What separates them is cost and time, both in Bash's favor: the
median answer cost $0.0385 [0.0345, 0.0432] with Bash against $0.0713
[0.0601, 0.0902] with the server, and took 12.9 s [11.9, 14.7] against 17.7 s [15.8,
18.9]. The main limitation is that the images are rendered with crisp edges,
so a single color cut with no model already answers 28 of the 46 questions.

![Right, wrong, declined or invalid per setup](agent_qa.png)

## Results

Each setup gave 138 answers (46 questions × 3 runs); each no-model control
answered once per question. Shares are the mean over questions of the share
of valid runs right, with 95% bootstrap intervals over questions (4,000
resamples).

| Setup | Right | Wrong | Declined | Invalid | Questions right in all runs | Share right, all 46 | Share right, the 18 a color cut misses | Cost / answer, median | Seconds / answer, median | Tool calls, median |
|---|---|---|---|---|---|---|---|---|---|---|
| No model: median answer of its kind | 2 | 44 | 0 | 0 | 2 of 46 (1 run) | 4.3% [0, 10.9] | 0% | - | - | - |
| No model: one color cut | 28 | 10 | 8 | 0 | 28 of 46 (1 run) | 60.9% [45.7, 73.9] | 0% | - | - | - |
| Claude alone | 0 | 0 | 135 | 3 | 0 of 46 | 0% | 0% | $0.0102 [0.0095, 0.0108] | 4.8 [4.5, 5.1] | 0 |
| Claude + Bash/PlantCV/OpenCV + web | 131 | 6 | 0 | 1 | 41 of 46 | 95.7% [90.6, 99.3] | 88.9% [75.9, 98.2] | $0.0385 [0.0345, 0.0432] | 12.9 [11.9, 14.7] | 3 [3, 3] |
| Claude + plantcv-mcp + Read | 119 | 19 | 0 | 0 | 37 of 46 | 86.2% [76.8, 94.9] | 72.2% [53.7, 88.9] | $0.0713 [0.0601, 0.0902] | 17.7 [15.8, 18.9] | 3 [3, 4] |

[`summary.tsv`](summary.tsv), [`stats.tsv`](stats.tsv) and
[`stats_paired.tsv`](stats_paired.tsv), written by `plot.R` from
[`results.jsonl`](results.jsonl) and [`baselines.jsonl`](baselines.jsonl), hold
every number here. Per answer, Claude with Bash was right on 131 of 137 valid
answers and the server on 119 of 138. Invalid runs are left out of every
share, interval and test, which slightly favors the Bash setup: its one
invalid run (q25, rep 2) timed out, and the paired test counts q25 as right in
its two valid runs, so it has Bash right in every run on 42 questions, against
41 in the table. The paired test is exact McNemar on the discordant questions:
7 against 2 (p = 0.18) on all 46, and 5 against 2 (p = 0.45) on the 18 a color
cut misses. The server's mean cost and time ($0.091, 23.3 s) sit above its
medians because of a few long runs.

![Share right with 95% interval](agent_qa_accuracy_ci.png)

**Where each setup missed.**

- Server, areas with yellowed leaves (6 wrong, q05, q06, q08): `segment`
  thresholds one channel and `refine` cannot join two masks, so the model
  added partial masks by hand. Claude with Bash got all 12 right.
- Server, leaf counts (7 wrong, q31 × 3, q33, q36 × 3): `count_leaves`'
  watershed cannot split leaves that overlap without gaps, so the model
  counted by eye. It said 13 for the 14-leaf rosette (q36) in all 3 runs;
  Claude with Bash said 14.
- Server, q26 tray seedling: 1,571 px for 1,599 in all 3 runs, cause not
  investigated. q21 mm² area: 591.9 for 607.9 once, just outside the 2.5%
  tolerance. q46 yellow share: 52.2 for 17.2 in 2 of 3 runs.
- Claude with Bash, q31 leaf count: 11 for 12 in all 3 runs (the server also
  missed it every time). Yellow shares: one run each of q42, q44 and q46; the
  server got all runs of q42 and q44.

![Cost and time per answer, every run](agent_qa_cost_time.png)

![Right, wrong or declined by kind of question](agent_qa_by_category.png)

## Controls

- **Oracle.** Before any run, every key answer written as a reply must score
  right and `UNKNOWN` must score declined. In CI,
  `tests/test_agent_qa_score.py` also checks that a shifted answer scores
  wrong for every question.
- **Baselines.** Two answers that need no model were scored the same way:
  the median answer of each kind of question, taken from the key, and one
  color cut (a pixel is plant if its green beats red and blue by 10, and the
  largest blob is the plant; `baselines.py`). The color cut answers every
  height/width and mm² question and 6 of the 8 tray questions. Only 18
  questions need more: 4 areas with yellowed leaves, 2 tray counts, 8 leaf
  counts and 4 yellow shares.
- **Validity.** One rule applies to every setup (`run.py` `assess()`). A run
  is invalid if it did not exit 0 (a timeout included); if its start-up event
  lacks the setup's tools (for the server setup, Read plus the connected
  server's tools; for Claude alone, none); if a tool setup made no tool call,
  since the image cannot be measured unopened; or if any tool input or output
  contains a leak marker. Invalid runs are recorded and shown, never retried or
  overwritten. There were 4: three Claude alone runs exited 1 after trying to
  call a tool they do not have ("tool call could not be parsed"), and one
  Claude with Bash run hit the 900 s timeout.
- **Leak scan.** The leak markers are the key's file names, the benchmark
  directory (local, on github.com or on raw.githubusercontent.com), the
  development checkout's directory name and the transcript directory. All 414
  transcripts were re-scanned after the sweep (`run.py --audit`), tool inputs
  and outputs both, with 0 hits. No tool input named the key or the server's
  code (`touched_key`: 0).
- **Tests.** `tests/test_agent_qa_score.py` rejects six wrong scorers (among
  them a 5% area tolerance, hue compared without wrapping and the first ANSWER
  line read instead of the last), and `tests/test_agent_qa_harness.py` rejects
  four wrong validity rules (among them tool outputs not scanned for leaks and
  the exit code ignored). Each was seen to fail under its planted mutation.

## Limitations

- **The images are rendered, not photographed.** They have textured soil,
  algae, pebbles, moss, straw, shadows, pot rims, overlapping leaves of
  different shades and a printed marker card. But the plant's edge is crisp,
  and a color rule (LAB a* or b*) always separates the plant from what
  touches it, which keeps every answer exact. Real photos have soft edges,
  glare and leaves that blend into the soil. The server's real-photo figures
  are in [`docs/EVAL.md`](../../docs/EVAL.md).
- **Most questions cannot show a win.** On the 28 questions a color cut
  answers, a setup can only show that it did not lose. On the other 18 the
  setups split the same way as overall, and the difference is not separated
  (p = 0.45).
- **Claude with Bash used OpenCV, not PlantCV.** Its Bash calls imported
  `cv2` in 123 of 138 runs and PlantCV in none, although the prompt named
  PlantCV. So the result compares the server with Claude's own OpenCV and
  numpy code. We counted this from the raw transcripts, which are not in the
  repository (`results.jsonl` stores only tool names).
- **Both model setups can see the image.** Some answers, such as a leaf count
  or a count of plants, can come from looking rather than from code, in
  either setup.
- **The server ran without the `[sam]` extra** (torch). Its
  `segment_leaves_sam` tool fails with `SamNotInstalledError`; the model
  called it twice, both in one run.
- **One model, one date, one server version, 46 questions.** A difference of a
  few questions is within the noise.

## Method

**Setups.** Claude Opus 5.5 in Claude Code, 3 runs per question per setup:

- **Claude alone**: no tools. It cannot open the image, so the right reply is
  "unknown". This is the control for guessing.
- **Claude + Bash/PlantCV/OpenCV + web**: what a user has without this
  server: Bash, Read (which shows Claude the image), WebSearch and WebFetch.
  The prompt says that the python3 on PATH has PlantCV 4.11.3 and OpenCV 4.11
  installed; plantcv-mcp is not installed. Claude writes its own code.
- **Claude + plantcv-mcp + Read**: this server plus Read, so it can look at
  the image just as the other setup can.

**Isolation.** Each run is a separate `claude -p` with `--setting-sources ""`,
an empty working directory holding only the image (as `image.png`, so the
file name says nothing about the answer), and `--strict-mcp-config`, so no
other MCP servers load. Every setup gets the same short system prompt and
question prompt, plus the one sentence naming PlantCV and OpenCV for the Bash
setup. Claude Code connects MCP servers in the background by default, so
`run.py` sets `MCP_CONNECTION_NONBLOCKING=false`. Transcripts and the Bash
setup's Python live outside the repository, so a Bash call that searches the
disk does not find them beside the key.

| | Claude + Bash/PlantCV/OpenCV | Claude + plantcv-mcp |
|---|---|---|
| Python | 3.12 venv outside the repository (`--make-alt-venv`) | `uvx --python 3.12 --with opencv-python==4.11.0.86 plantcv-mcp==1.16.1` |
| PlantCV | 4.11.3 | 4.11.3 |
| opencv-python | 4.11.0.86 (pinned) | 4.11.0.86 (pinned) |
| NumPy / scikit-image | 2.4.6 / 0.26.0 | 2.4.6 / 0.26.0 |

`run.py` checks these versions in both environments before a sweep. It stops
if a pin or the server version differs, or if NumPy or scikit-image differ
between the two. The model is `claude-opus-5-5` and Claude Code is 2.1.302 in
every row.

**Questions.** [`questions.json`](questions.json) holds the 46 questions, their
answers and each image's sha256, which is checked before every use.
`build_questions.py` renders the images into `images/` (about 21 MB,
gitignored) from seeded generators (seed 20261016), and a rebuild is
byte-identical. Shapes, sizes, colors, textures and positions are random
draws, not picks: mottled soil with algae, pebbles and moss, overlapping
leaves from bright to dull olive, yellowed leaves and straw in half the area
questions, trays of square pots, side views with a terracotta pot, and an mm²
marker card turned 10-35°.

| Kind | Count | What is asked | Answer key |
|---|---|---|---|
| `area` | 8 | plant area in pixels | plant pixels drawn (4-7 overlapping leaves; in 4, 1-2 of them yellowed) |
| `extent` | 8 | side-view plant height or width in pixels, lowest to highest (leftmost to rightmost) point | bounding box of the plant pixels drawn, both ends counted |
| `area_mm` | 6 | plant area in mm², given a tilted black square of known size | pixels drawn ÷ (square's side in px ÷ its side in mm)² |
| `tray` | 8 | how many pots hold a seedling (4); area of the seedling in row r, column c (4) | seedlings drawn; pixels drawn for that seedling |
| `leaf_count` | 8 | leaves on a rosette (5 to 14) | overlapping elliptical leaves drawn, each at least 25% visible |
| `color` | 8 | mean HSV hue in degrees (4); percentage of the plant that is yellow (4) | circular mean of the plant pixels' HSV hue (as `colorsys` computes it); visible yellow-leaf pixels ÷ plant pixels |

`build_questions.py` never imports plantcv-mcp or PlantCV, so a bug in the
server cannot also be in the key. Every answer is the generator's own mask,
counted with numpy, and texture is painted after the mask is fixed. For every
plant, the builder asserts that a LAB rule separates the plant from
everything within 4 px of it.

**Scoring.** Each reply ends with `ANSWER: <value>` or `ANSWER: UNKNOWN`, and
`score.py` decides the verdict from that line alone: right within the
tolerance, wrong outside it, declined for `UNKNOWN`, and no answer if the line
is absent. Tolerances:

- pixel areas, 1%. Segmenting these images is exact. `cv2.contourArea` traces
  through pixel centers and comes out 2-8% smaller, a different quantity that
  scores wrong;
- height and width, 1 pixel, so both max − min and max − min + 1 count;
- mm² areas, 2.5%. A 1-pixel misreading of the marker's side costs 1% or less
  on the 100-120 px markers and 2.8% (once squared) on the smallest, 72 px;
  2.5% accepts the usual misreading and refuses a wrong scale. cm² is
  converted;
- counts, exact; hue, 3 degrees around the circle (OpenCV stores hue in
  2-degree steps); yellow share, 1 percentage point, with a bare value of at
  most 1 read as a fraction (every true share is 10-60%).

A wrong answer equal to a named other quantity, such as `contour_area` or
`opencv_half_degrees`, gets a `slip` label, named only when its tolerance
window cannot overlap the right answer's.

## History

Each pilot asked 1-2 questions per setup once, with the server setup lacking
Read. A pilot question every tool setup got right was too easy, so the set was
hardened. Claude alone declined every pilot question.

1. **Flat colors on flat soil.** Both tool setups were right on both
   questions, so we added textured soil, a pot and moss. The harness now waits
   for the server to connect, after one server run started without its tools.
2. **Textured soil, pot.** Both were right again (Bash 15,796 against
   15,795), so we added algae, dull leaves, 5-14-leaf rosettes, a tilted
   marker and drooping side-view leaves.
3. **Hardened round 2.** Both missed the 14-leaf count (15 and 13), so we
   added yellowed leaves and straw to 4 area questions.
4. **Final set.** Both missed q05. A scorer fix followed: a slip is named only
   when its window cannot overlap the right answer's. No verdict changed.
5. **Sweep 1, stopped at 188 rows** (rep 1 of all 46 questions, rep 2 of the
   first 17). The server setup lacked Read, and the
   Bash setup had not been told PlantCV was installed (it used it in 0 of 63
   runs). Bash got 59 of 63 right, the server 55 of 62, and Claude alone
   declined 61 and exited 1 twice. We stopped it when a review found the
   server setup handicapped; the rows are in
   [`results_sweep1_stopped.jsonl`](results_sweep1_stopped.jsonl).
6. **Sweep 2 (this one).** The server setup got Read, the Bash prompt names
   PlantCV and OpenCV, one validity rule replaced a server-only retry, OpenCV
   is pinned and versions are asserted, and the leak scan was added. The Bash
   setup's Python and the transcripts moved outside the repository, because in
   sweep 1 a traceback printed the venv's path inside the benchmark directory
   (the one leak hit). The Bash and server rows were run again. Claude alone's
   63 rows were kept, since its prompt and setup did not change; under the new
   rule 2 of them are invalid.

Transcripts from the pilots and both sweeps are kept by the maintainer, not
published.

## Reproduce

```bash
uv sync --python 3.12
uv run python benchmarks/agent_qa/build_questions.py      # renders images/ (gitignored); questions.json is byte-identical
uv run python benchmarks/agent_qa/baselines.py            # the two no-model controls -> baselines.jsonl
uv run python benchmarks/agent_qa/run.py --make-alt-venv  # the Bash setup's Python, at $AGENTQA_ALT_VENV
uv run python benchmarks/agent_qa/run.py --reps 3 --workers 1   # 414 claude -p runs; resumable
uv run python benchmarks/agent_qa/run.py --audit          # re-assess validity and re-scan every transcript for leaks
uv run python benchmarks/agent_qa/run.py --rescore        # re-score stored replies after a key or scorer fix
Rscript benchmarks/agent_qa/plot.R                        # summary.tsv, stats*.tsv + figures
```

Transcripts go to `$AGENTQA_ARTIFACTS/runs/` (default
`~/plantcv-mcp-artifacts/agent_qa/`) and the Bash setup's Python to
`$AGENTQA_ALT_VENV` (default `~/.local/share/agentqa-plantcv-alt-venv`), both
outside the repository.
