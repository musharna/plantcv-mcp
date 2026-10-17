# Agent benchmark: measuring plant images with and without the server

Does Claude measure plant images better with plantcv-mcp than without it?
46 questions about synthetic plant images, each asked 3 times to Claude Opus
5.5 in Claude Code under three setups:

- **Claude alone**: no tools. It cannot open the image, so the right reply is
  "unknown"; it is the control for guessing.
- **Claude + Bash/PlantCV/OpenCV + web**: what a user has without this
  server. Bash, Read (which shows Claude the image), WebSearch and WebFetch,
  with a Python on PATH that has PlantCV 4.11.3 and OpenCV installed and not
  plantcv-mcp. Claude writes its own code.
- **Claude + plantcv-mcp**: this server (1.16.1 from PyPI, via `uvx`) only.

Run on TODO with Claude Code TODO.

![Right, wrong or declined per setup](agent_qa.png)

| Setup | Right | Wrong | Declined | Questions right in all 3 runs | Cost / answer, median (mean) | Seconds / answer, median (mean) | Slowest |
|---|---|---|---|---|---|---|---|
| Claude alone | TODO | TODO | TODO | TODO of 46 | TODO | TODO | TODO |
| Claude + Bash/PlantCV/OpenCV + web | TODO | TODO | TODO | TODO of 46 | TODO | TODO | TODO |
| Claude + plantcv-mcp | TODO | TODO | TODO | TODO of 46 | TODO | TODO | TODO |

Out of 138 answers per setup. The three runs of a question are not
independent, so the questions column is the fairer count. The numbers come
from [`summary.tsv`](summary.tsv), which `plot.R` writes from
[`results.jsonl`](results.jsonl).

## What it shows

TODO after the full sweep.

![Cost and time per answer, every run](agent_qa_cost_time.png)

![Right, wrong or declined by kind of question](agent_qa_by_category.png)

## What it does not show

- **The images are rendered, not photographed.** They have textured soil,
  pebbles, moss, shadows, pot rims, overlapping leaves of different shades and
  a printed marker card. But the plant's edge is crisp, and a colour rule
  (LAB a* or b*) always separates the plant from what touches it, which keeps
  every answer exact. Real photos have soft
  edges, glare and leaves that blend into the soil. The server's real-photo
  figures are in [`docs/EVAL.md`](../../docs/EVAL.md).
- **The alternative setup can see the image.** Read shows it to Claude, so
  some answers (a leaf count, a count of plants) can come from looking, not
  from code. The server returns overlays, so the same holds there.
- **One model, one date, one server version.**

## Questions

[`questions.json`](questions.json) holds the 46 questions, their answers and
each image's sha256. `build_questions.py` renders the images into `images/`
(about 21 MB, gitignored) from seeded generators (seed 20261016), and a
rebuild is byte-identical. `run.py` refuses an image whose hash does not
match. Shapes, sizes, colours, textures and positions are random draws, not
picks. Each run sees only its image, copied into an empty directory as
`image.png`, so the file name says nothing about the answer.

The pictures are meant to make the shortcut fail. A flat green shape on flat
soil can be measured by counting one colour. The first draft of this set was
exactly that, and in its pilot Claude with Bash got both questions right in
two tool calls. A second draft added soil texture and a pot. Claude with Bash
still found a plant area in 2 calls: green ahead of red and blue by 10, then
the largest blob, gave 15,796 px against 15,795. Now:

- the soil is mottled and grainy, with green-brown algae, pebbles, olive moss
  clumps (which are not the plant) and the plant's shadow;
- top views have a dark pot rim, trays are black with square pots (some
  empty, some with moss or algae), and side views have a wall and a
  terracotta pot, with leaves that can hang down in front of the pot;
- leaves run from bright to dull olive, close to the algae, and each has its
  own shade, a lighter midrib and a darker edge; leaves overlap;
- in half the area questions 1-2 old leaves have yellowed and straw lies on
  the soil, so "greener than red" misses plant and "green or yellow" picks up
  straw;
- rosettes have 5-14 leaves of nearly one shade, overlapping well beyond the
  centre, so neither shade, connected parts nor convexity defects give the
  count;
- the mm² marker is a black square on a white card turned 10-35°, so its
  bounding box is not its side.

The builder asserts the colour rule above for every plant: a* below the
lowest a* within 4 px of the plant, or b* above the highest b* there.

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
counted with numpy. Texture is painted after the mask is fixed, and the
builder asserts that every plant pixel's hue stays between 40° and 160°.

## Scoring

Each reply ends with `ANSWER: <value>` or `ANSWER: UNKNOWN`. `score.py`
decides the verdict from that line alone:

- **right**: the value is within the tolerance below.
- **wrong**: it is not.
- **declined**: `UNKNOWN`.
- **no answer**: there is no `ANSWER:` line.

Tolerances, by kind:

- **Pixel areas: 1%.** Segmenting these images is exact, so the only
  legitimate spread is rounding. Measuring with `cv2.contourArea`, which
  traces through pixel centres, comes out smaller by about half the perimeter
  (2-8% here); that is a different quantity, and it is scored wrong.
- **Height and width: 1 pixel**, so that both max − min and max − min + 1
  count.
- **Areas in mm²: 2.5%.** The square's side in pixels can be read 1 pixel
  off either way: 1.4% on the smallest square in the set (72 px), which is
  2.8% squared, and 1% or less on the 100-120 px squares. 2.5% accepts the
  usual ±1 px reading and refuses a wrong scale. An answer in cm² is
  converted.
- **Counts: exact.** One leaf off on a 4-leaf rosette is a 25% error.
- **Hue: 3 degrees**, measured around the circle. OpenCV stores 8-bit hue in
  2-degree steps.
- **Yellow share: 1 percentage point.** A bare value of at most 1 is read as
  a fraction; every true share is between 10% and 60%, so this cannot be
  confused with a real percentage.

A wrong answer that equals a specific other quantity gets a `slip`, checked
against that quantity's number, never guessed from the wording:
`contour_area`, `pixels` or `linear_scale` (mm² answers), `other_axis`,
`opencv_half_degrees`, `green_share`. A slip is named only when its window
and the right answer's window do not overlap. On a large plant the contour
area is under 2% below the pixel count, so an answer near it may just as well
come from a loose threshold. In the pilot, Claude with Bash answered 20,039
for 20,354 from a saturation threshold, which an unguarded check had labelled
`contour_area`. A run whose tool calls mention the answer key or the server's
code is flagged `touched_key`.

Tests: `tests/test_agent_qa_score.py`. Six deliberately wrong scorers were
each failed by it (`mutate-run`): a 5% area tolerance, hue compared without
wrapping, `UNKNOWN` scored wrong, the first ANSWER line read instead of the
last, cm² not converted, and a contour slip named without the overlap check.

## Reproduce

```bash
uv sync --python 3.12
uv run python benchmarks/agent_qa/build_questions.py     # renders images/ (gitignored); questions.json is byte-identical
uv run python benchmarks/agent_qa/run.py --make-alt-venv  # the alt setup's Python: PlantCV + OpenCV, no plantcv-mcp
uv run python benchmarks/agent_qa/run.py --reps 3 --workers 1   # 414 claude -p runs; resumable
uv run python benchmarks/agent_qa/run.py --rescore        # re-score stored replies after a key or scorer fix
Rscript benchmarks/agent_qa/plot.R                        # summary.tsv + figures
```

Each run is an isolated `claude -p`:
- `--setting-sources ""` and an empty working directory holding only the image.
- `--strict-mcp-config`, so no other MCP servers.
- The same short system prompt in every setup.

The server is started as `uvx --python 3.12 plantcv-mcp==1.16.1`: on a
machine whose default Python is 3.14, uvx otherwise tries to build PlantCV's
pinned SciPy from source. Claude Code connects MCP servers in the background
by default, and in the pilot one run began before the server had started and
answered without it. `run.py` therefore sets `MCP_CONNECTION_NONBLOCKING=false`,
and refuses to record any server run whose start-up event does not show the
server connected; such a run is retried on the next invocation.

Raw transcripts go to `runs/`, which is gitignored.
