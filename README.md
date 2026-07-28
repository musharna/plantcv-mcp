# plantcv-mcp

PlantCV as an MCP **measurement instrument**: it returns plant trait numbers
**and the segmentation overlay they were computed from**, and refuses to return
numbers when the segmentation is degenerate.

## Why the two-step API

`segment()` returns an overlay and diagnostics but **no traits**. `measure()`
requires the `session_id` that `segment()` mints. You cannot get a number
without first being handed the picture it came from.

This is not a style preference. Measured on real images with PlantCV 4.11.3:

| failure                           | what you get without the overlay                            |
| --------------------------------- | ----------------------------------------------------------- |
| four-view render, whole-image ROI | 17 plausible traits describing four merged plants           |
| plant clipped by the frame        | size traits that are silently lower bounds                  |
| empty mask                        | 17 traits of zeros, with PlantCV reporting `in_bounds=True` |

All three produce correctly-united, entirely believable numbers.

## Install

```bash
uv add plantcv-mcp
```

## Tools

- `suggest_segmentation(image_path, channel)` — colourspace and threshold contact sheets
- `segment(image_path, channel, method)` — overlay + diagnostics + warnings, no traits
- `measure(session_id)` — traits, or a raised error on a degenerate mask
- `list_methods()` — channels, methods, pinned PlantCV version

## Limitations

Phase 1 is single-ROI. Multi-plant grids, morphology traits (leaf angles, stem,
skeleton) and iterative mask refinement are phase 2.
