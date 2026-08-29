# plantcv-mcp

**Plant phenotyping over MCP — traits, plus the segmentation overlay they were measured from.**

[![ci](https://github.com/musharna/plantcv-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/musharna/plantcv-mcp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/plantcv-mcp)](https://pypi.org/project/plantcv-mcp/)
[![python](https://img.shields.io/pypi/pyversions/plantcv-mcp)](https://pypi.org/project/plantcv-mcp/)
[![license](https://img.shields.io/pypi/l/plantcv-mcp)](LICENSE)
[![Glama](https://glama.ai/mcp/servers/musharna/plantcv-mcp/badges/score.svg)](https://glama.ai/mcp/servers/musharna/plantcv-mcp)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21713516.svg)](https://doi.org/10.5281/zenodo.21713516)

<!-- mcp-name: io.github.musharna/plantcv-mcp -->

[PlantCV](https://plantcv.org) as an MCP **measurement instrument**: it returns plant trait
numbers **and the picture they were computed from**, and refuses to return numbers when the
segmentation is degenerate.

> Unofficial. Not affiliated with, endorsed by, or sponsored by the Donald
> Danforth Plant Science Center or the PlantCV maintainers. See [NOTICE](https://github.com/musharna/plantcv-mcp/blob/master/NOTICE).

## Why you are handed the overlay

Red marks the pixels that were measured, and a cyan line traces the mask's own boundary — the
tint alone was invisible on a photo of red beans, so the outline is drawn on the mask's edge
pixels and never touches anything unmasked. Both images below come from the same file and the
same threshold method — the only difference is one parameter.

| ✅ `channel="a", object_type="dark"`                                                                                   | ❌ `channel="s", object_type="dark"`                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| ![correct segmentation](https://raw.githubusercontent.com/musharna/plantcv-mcp/master/docs/assets/overlay-correct.png) | ![inverted segmentation](https://raw.githubusercontent.com/musharna/plantcv-mcp/master/docs/assets/overlay-inverted.png) |
| Mask covers **3.1%** of the frame, 9 components. `area=32427`                                                          | Mask covers **96.1%** — it is the **background**. `area=1007829`                                                         |

The failure on the right is what this server exists to prevent. Without the picture, both
runs return seventeen traits with correct units and entirely believable magnitudes. The one
on the right is measuring the wall behind the plants.

`segment()` returns the overlay and diagnostics but **no traits**. `measure()` requires the
`session_id` that `segment()` mints. You cannot get a number without first being handed the
image it came from.

That is not a style preference. Measured on real images with PlantCV 4.11.3:

| failure                           | what you get without the overlay                            |
| --------------------------------- | ----------------------------------------------------------- |
| four-view render, whole-image ROI | 17 plausible traits describing four merged plants           |
| plant clipped by the frame        | size traits that are silently lower bounds                  |
| empty mask                        | 17 traits of zeros, with PlantCV reporting `in_bounds=True` |

All three produce correctly-united, entirely believable numbers.

## Install

```bash
pip install plantcv-mcp
```

Requires Python 3.11+. Installing pulls PlantCV and its scientific stack, so the first
install is not fast. From a checkout: `uv add /path/to/plantcv-mcp`.

## Configure your MCP client

```bash
claude mcp add plantcv -- plantcv-mcp
```

Claude Desktop and other stdio hosts:

```json
{ "mcpServers": { "plantcv": { "command": "plantcv-mcp" } } }
```

If `plantcv-mcp` is not on the host's `PATH`, use `"command": "uv", "args": ["run",
"--directory", "/path/to/plantcv-mcp", "plantcv-mcp"]`. Verify with `list_methods()`.
To confine reads to your imagery: `plantcv-mcp --root /data/phenotyping`.

## Tools

| tool                                                                    | returns                                                                                                      |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `suggest_segmentation(image_path, channel, method)`                     | contact sheets, and what each `object_type` would yield                                                      |
| `segment(image_path, channel, method, ...)`                             | overlay + diagnostics + warnings — **no traits**                                                             |
| `refine(session_id, ops)`                                               | a NEW session with a cleaned-up mask, plus its overlay                                                       |
| `measure(session_id, analyses, px_per_mm, ...)`                         | traits, or a raised error on a degenerate mask                                                               |
| `calibrate_scale_from_marker(image_path, x, y, w, h, marker_length_mm)` | `px_per_mm` from a marker of known real size                                                                 |
| `measure_regions(session_id, nrows, ncols, ...)`                        | one row per plant in a tray (RGB traits, thermal temperatures or HSI index stats), plus the numbered overlay |
| `measure_morphology(session_id, prune_size, tangent_size, ...)`         | leaf/stem skeleton traits + the numbered-segment overlay                                                     |
| `measure_images(image_paths, channel, method, ...)`                     | one recipe across many images (per plant with a grid); traits only where valid; time-budgeted                |
| `segment_hyperspectral(envi_path, index, threshold, ...)`               | an HSI session from a spectral-index threshold + pseudo-RGB overlay                                          |
| `measure_spectral(session_id, indices, ...)`                            | index statistics (and, opt-in, per-band reflectance)                                                         |
| `segment_thermal(path, min_c, max_c, ...)`                              | a thermal session from a °C band + grey-frame overlay                                                        |
| `measure_thermal(session_id, ...)`                                      | max/min/mean/median °C under the mask                                                                        |
| `list_methods()`                                                        | channels, methods, object types, pinned PlantCV version                                                      |

Typical loop: `suggest_segmentation` → `segment` → **look at the overlay** → `segment` again
with a different channel, method or polarity if it is wrong (or `refine` if it is nearly
right) → `measure`.

The `segment()` response for the image above — verbatim, apart from a shortened
`session_id` and an elided message — and the overlay arrives beside it as an image:

```json
{
  "session_id": "9d2384c8-…",
  "channel": "a",
  "method": "otsu",
  "object_type": "dark",
  "fill_size": 200,
  "mask_fraction": 0.031,
  "component_count": 9,
  "major_object_count": 4,
  "largest_area": 8628,
  "overlay_scale": 1.0,
  "overlay_png_bytes": 748233,
  "warnings": [
    {
      "code": "multi_specimen",
      "message": "4 comparably-sized objects detected (areas: [8628, 7981, 7106, 6748]). …"
    }
  ]
}
```

## What it refuses, and why

Every guard was calibrated against a real failure and names the next action. Blocking
guards withhold numbers; advisories travel with them.

- **Inverted mask** (`implausible_coverage`) — the right-hand image above: 96% of the frame
  selected, seventeen believable traits, all describing the wall.
- **Nothing selected, or `fill_size` deleted the specimen** (`empty_mask`,
  `fill_erased_mask`) — PlantCV returns seventeen zeros with `in_bounds=True`.
- **Background texture** (`noisy_segmentation`) — a sorghum photo measured as one
  650,000-px plant made of 118 chamber-wall specks.
- **Several plants in one mask** (`multi_specimen`) — the number describes the group; use
  `measure_regions()`, which measures each plant and numbers the overlay.
- **Wrong scale, wrong kind, changed file** — a marker measured 4.35× wrong by PlantCV's own
  ROI method; a thermal session handed to an RGB measurer; an image edited after
  segmentation. Each is refused naming the right tool.

All 26 warning codes, every tool's parameters, and the measured facts behind each guard:
**[docs/GUIDE.md](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md)** — [segmenting](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#segmenting) · [traits and units](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#what-it-measures) ·
[polarity](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#getting-the-polarity-right) · [refining](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#refining-a-mask) ·
[trays](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#measuring-a-tray) · [morphology](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#morphology-leaves-stem-branch-points) ·
[batches](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#measuring-many-images) · [hyperspectral and thermal](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#hyperspectral-and-thermal) ·
[warning reference](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#warnings-and-refusals).

## Security

This server reads image files the host user can read and returns them to the model as
images; with no `--root` there is no allow-list. Run it as a user whose read access you are
comfortable exposing, set `--root`, and do not run it as root. PlantCV/OpenCV analyses run in
a worker subprocess, so a native crash is a tool error, not a dead server. Details:
[security](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#security-and-trust-boundary) · [read roots](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#restricting-what-the-server-may-read) ·
[crash containment](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#crash-containment-the-analysis-worker) · [limitations](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md#limitations).

## Attribution and licensing

This project is MIT licensed. It depends on
[PlantCV](https://github.com/danforthcenter/plantcv), which is licensed under the
**Mozilla Public License 2.0**. No PlantCV source is vendored or redistributed
here — it is an ordinary runtime dependency — so the MIT license applies to this
project's own files. See [NOTICE](https://github.com/musharna/plantcv-mcp/blob/master/NOTICE) for the full statement.

## More

- [docs/GUIDE.md](https://github.com/musharna/plantcv-mcp/blob/master/docs/GUIDE.md) — the full guide
- [CHANGELOG.md](https://github.com/musharna/plantcv-mcp/blob/master/CHANGELOG.md) — what changed, and why
- [docs/MUTATION-CHECKS.md](https://github.com/musharna/plantcv-mcp/blob/master/docs/MUTATION-CHECKS.md) — every guard disabled on purpose, and
  the test that went red for it

Images on this page are rendered from `tests/fixtures/multi_specimen.png`, an original render
by the author, and regenerate from committed code.
