# plantcv-mcp

plantcv-mcp is an MCP server that lets AI agents run [PlantCV](https://plantcv.org) plant
phenotyping. It returns trait numbers and the segmentation overlay they were measured from, and
refuses to return numbers when the segmentation is degenerate.

[![CI](https://img.shields.io/github/actions/workflow/status/musharna/plantcv-mcp/ci.yml?branch=master&label=CI)](https://github.com/musharna/plantcv-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/plantcv-mcp)](https://pypi.org/project/plantcv-mcp/)
[![Python](https://img.shields.io/pypi/pyversions/plantcv-mcp)](https://pypi.org/project/plantcv-mcp/)
[![License](https://img.shields.io/pypi/l/plantcv-mcp)](https://github.com/musharna/plantcv-mcp/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21713516.svg)](https://doi.org/10.5281/zenodo.21713516)
[![Glama](https://glama.ai/mcp/servers/musharna/plantcv-mcp/badges/score.svg)](https://glama.ai/mcp/servers/musharna/plantcv-mcp)

<!-- mcp-name: io.github.musharna/plantcv-mcp -->

<p align="center">
  <img src="https://raw.githubusercontent.com/musharna/plantcv-mcp/master/docs/assets/overlay-correct.png" width="45%" alt="Correct segmentation: four views of a rendered plant, stems and leaves tinted red and outlined in cyan, the gray background untouched">
  <img src="https://raw.githubusercontent.com/musharna/plantcv-mcp/master/docs/assets/overlay-inverted.png" width="45%" alt="Inverted segmentation: the whole gray background tinted red, the four plant views left out">
</p>

**Left**, `channel="a", object_type="dark"`: the mask covers 3.1% of the frame, 9 components,
`area=32427`. **Right**, `channel="s", object_type="dark"`: the mask covers 96.1%; it is the
background, `area=1007829`.

## Why you get the overlay

Both images come from the same file and the same threshold method; only the channel differs.
Without the picture, both runs return 17 traits with correct units and believable
magnitudes. The one on the right is measuring the wall behind the plants. Red marks the
measured pixels; a cyan line traces the mask's edge.

So `segment()` returns the overlay and diagnostics but **no traits**, and `measure()` needs the
`session_id` that `segment()` mints: no single image is measured before you are shown it. The
exception is `measure_images()`, which applies one recipe to up to 200 images without overlays.
Settle the recipe on one image's overlay first (the server does not check); every image still
passes the `segment()` guards, and one that trips a blocking guard gets no traits.

Measured with PlantCV 4.11.3 on a synthetic render (first row:
`tests/fixtures/multi_specimen.png`) and on photographs, each of these returns believable
numbers with correct units:

| Failure                           | What you get without the overlay                            |
| --------------------------------- | ----------------------------------------------------------- |
| four-view render, whole-image ROI | 17 plausible traits describing four merged plants           |
| plant clipped by the frame        | size traits that are silently lower bounds                  |
| empty mask                        | 17 traits of zeros, with PlantCV reporting `in_bounds=True` |

## Does it help?

Not on these test images: the server did not make Claude measurably more accurate than writing
its own code (mostly OpenCV), and its own code was cheaper.

<p align="center">
  <img src="https://raw.githubusercontent.com/musharna/plantcv-mcp/master/benchmarks/agent_qa/rerun_2026-10-18/agent_qa.png" width="780" alt="Stacked bar chart of right and wrong answers in the 2026-10-18 re-run, 46 questions times 3 runs per setup. Claude + plantcv-mcp + Read: 132 right, 6 wrong. Claude + Bash/PlantCV/OpenCV + web: 134 right, 4 wrong. No-model controls, one answer per question: one color cut 28 right, 10 wrong, 8 declined; median answer of its kind 2 right, 44 wrong.">
</p>

In a pre-registered re-run on 2026-10-18, against an unreleased commit (051aff0) that adds
several-channel segmentation rules and `zoom_image`, Claude Opus 5.5 was right on 97.1% of 46
measurement questions about rendered plant images with its own code and 95.7% with the server
(three runs each; 44 against 42 questions right in every run, exact McNemar p = 0.63, a tie).
Its own code cost $0.038 against $0.066 per answer (medians; the intervals do not overlap). On
release 1.16.1 the server was right on 86.2% against 95.7% (p = 0.18). The images have crisp
edges, so a single color cut with no model answers 28 of the 46. Method, controls and
limitations:
[benchmarks/agent_qa](https://github.com/musharna/plantcv-mcp/blob/master/benchmarks/agent_qa/README.md).

## Install

With [uv](https://docs.astral.sh/uv/) nothing needs installing: `uvx --python 3.13 plantcv-mcp` fetches and
runs the current release. Otherwise:

```bash
pip install plantcv-mcp
```

Requires Python 3.11 to 3.13: PlantCV 4.11.3 needs SciPy below 1.16, which has no Python 3.14
wheels, so `--python 3.13` keeps `uvx` off a 3.14 default. PlantCV and its scientific stack make the first install (or first `uvx`
run) slow. From a checkout: `uv add /path/to/plantcv-mcp`. The one optional tool,
`segment_leaves_sam()`, needs `pip install "plantcv-mcp[sam]"` (torch, torchvision,
segment-anything); without it, that tool refuses with that command and every other tool works.

## Configure your MCP client

```bash
claude mcp add plantcv -- uvx --python 3.13 plantcv-mcp
```

Claude Desktop and other stdio hosts:

```json
{ "mcpServers": { "plantcv": { "command": "uvx", "args": ["--python", "3.13", "plantcv-mcp"] } } }
```

With a pip install the command is `plantcv-mcp` ([checkout config](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#client-configuration)).
Verify with `list_methods()`.

Flags: `--root DIR` (repeatable, or `PLANTCV_MCP_ROOTS`) confines every read, and every file
the server keeps, to your imagery: `plantcv-mcp --root /data/phenotyping`. `--no-isolate` (or
`PLANTCV_MCP_ISOLATE=0`) runs analyses in-process instead of in the crash-containing worker.

## Tools

| Tool                                          | Returns                                                          |
| --------------------------------------------- | ---------------------------------------------------------------- |
| `suggest_segmentation`                        | contact sheets, and what each `object_type` would yield          |
| `segment`                                     | overlay + diagnostics + warnings, **no traits**                  |
| `refine`                                      | a new session with a cleaned-up mask, plus its overlay           |
| `measure`                                     | traits, or a raised error on a degenerate mask                   |
| `measure_images`                              | one recipe across many images; traits only where valid           |
| `measure_regions`                             | one row per plant in a tray, plus the numbered overlay           |
| `measure_morphology`                          | leaf/stem skeleton traits + the numbered-segment overlay         |
| `count_leaves`                                | an estimated leaf count by watershed + the numbered overlay      |
| `segment_leaves_sam`                          | the same count from Segment Anything (optional `sam` extra)      |
| `calibrate_scale_from_marker`                 | `px_per_mm` from a marker of known real size                     |
| `correct_lens_distortion`                     | a fisheye or wide-angle image undistorted from checkerboards     |
| `segment_hyperspectral`, `measure_spectral`   | an HSI session from an index threshold; index statistics         |
| `segment_thermal`, `measure_thermal`          | a thermal session from a °C band; °C statistics under the mask   |
| `zoom_image`                                  | part of an image, enlarged to look closely (not a measurement)   |
| `list_methods`                                | channels, methods, object types, pinned PlantCV version          |

Typical loop: `suggest_segmentation` → `segment` → **look at the overlay** → `segment` again
with another channel, method or polarity if it is wrong (or `refine` if it is nearly right) →
`measure`. With a ColorChecker in the frame, `color_correct=true` corrects colors before
segmenting and excludes the card from the mask. Parameters, an example response and the
measured facts behind each tool:
[docs/GUIDE.md](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md).

## What it refuses

Each guard was added after a specific failure and names the next action. Blocking guards
withhold numbers; advisories travel with them.

- **Inverted mask** (`implausible_coverage`): the right-hand image above, 17 believable
  traits describing the wall.
- **Nothing selected, or `fill_size` deleted the specimen** (`empty_mask`, `fill_erased_mask`):
  PlantCV returns 17 zeros with `in_bounds=True`.
- **Background texture** (`noisy_segmentation`): specks of chamber wall measured as one plant.
- **Several plants in one mask** (`multi_specimen`): use `measure_regions()`.
- **Wrong scale, wrong session kind, a file changed after segmentation**: refused, naming the
  right tool.
- **A lens calibration the boards do not determine, or no color card when one was asked for**:
  refused.

Why each guard exists, and every warning code:
[docs/GUIDE.md#warnings-and-refusals](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#warnings-and-refusals).

## Security

With no `--root`, the server reads any image file the host user can read and returns it to the
model. It writes only the lens-corrected image, the Segment Anything checkpoint when asked to
download it, and short-lived temporary copies of hyperspectral and thermal inputs. Run it as a
non-root user with `--root` set. Analyses run in a worker subprocess, so a native crash is a
tool error, not a dead server. Details:
[docs/GUIDE.md#security-and-trust-boundary](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#security-and-trust-boundary).

## Attribution and license

MIT licensed. It depends on [PlantCV](https://github.com/danforthcenter/plantcv), which is
licensed under the **Mozilla Public License 2.0**. No PlantCV source is vendored or
redistributed here; it is an ordinary runtime dependency, so the MIT license applies to this
project's own files. Unofficial: not affiliated with, endorsed by, or sponsored by the Donald
Danforth Plant Science Center or the PlantCV maintainers. See
[NOTICE](https://github.com/musharna/plantcv-mcp/blob/master/NOTICE).

Cite it with [CITATION.cff](https://github.com/musharna/plantcv-mcp/blob/master/CITATION.cff)
(archived at [doi:10.5281/zenodo.21713516](https://doi.org/10.5281/zenodo.21713516)). Changes:
[CHANGELOG.md](https://github.com/musharna/plantcv-mcp/blob/master/CHANGELOG.md).
