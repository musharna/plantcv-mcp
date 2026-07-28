# plantcv-mcp

PlantCV as an MCP **measurement instrument**: it returns plant trait numbers
**and the segmentation overlay they were computed from**, and refuses to return
numbers when the segmentation is degenerate.

> Unofficial. Not affiliated with, endorsed by, or sponsored by the Donald
> Danforth Plant Science Center or the PlantCV maintainers. See [NOTICE](NOTICE).

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

Not published to PyPI. Install from the repository:

```bash
uv add git+https://github.com/musharna/plantcv-mcp
```

Or from a local checkout:

```bash
uv add /path/to/plantcv-mcp
```

Requires Python 3.11+. Installing pulls PlantCV and its scientific stack
(scikit-image, dask, scipy), so the first install is not fast.

## Configure your MCP client

The server speaks stdio. The console script installed by the package is
`plantcv-mcp`.

**Claude Code**

```bash
claude mcp add plantcv -- plantcv-mcp
```

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "plantcv": {
      "command": "plantcv-mcp"
    }
  }
}
```

If the executable is not on your `PATH` (common when it lives in a project
virtualenv), give the absolute path to it, or invoke it through uv:

```json
{
  "mcpServers": {
    "plantcv": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/plantcv-mcp", "plantcv-mcp"]
    }
  }
}
```

Verify it is wired up by calling `list_methods()`, which reports the channels,
the methods, and the pinned PlantCV version.

## Tools

- `suggest_segmentation(image_path, channel="a")` — colourspace and threshold contact sheets
- `segment(image_path, channel, method)` — overlay + diagnostics + warnings, no traits
- `measure(session_id)` — traits, or a raised error on a degenerate mask
- `list_methods()` — channels, methods, pinned PlantCV version

Typical loop: `suggest_segmentation` → `segment` → look at the overlay →
`segment` again with a different channel or method if it is wrong → `measure`.

## Security and trust boundary

**This server reads image files anywhere on the host filesystem, and returns
them to the model as images.**

`suggest_segmentation` and `segment` take an `image_path` and pass it straight to
PlantCV's reader. There is no directory allow-list and no sandbox. Any path the
model asks for — that the operating-system user running the server can read — will
be decoded and returned as a base64 image in the model's context.

Practical consequences:

- Treat it like any other local filesystem MCP server. Run it as a user whose
  read access you are comfortable exposing to the model driving it.
- A prompt-injected or adversarial model can use it to view arbitrary **image**
  files on the machine. Non-image files fail to decode and raise, but the error
  message discloses whether the path exists.
- Do not run it as root, and do not expose it to untrusted input on a machine
  holding sensitive imagery.

Restricting reads to a configured root directory is a candidate for a future
release; it is deliberately **not** implemented today, and this section exists so
that is a decision you make rather than a surprise you discover.

## Attribution and licensing

This project is MIT licensed. It depends on
[PlantCV](https://github.com/danforthcenter/plantcv), which is licensed under the
**Mozilla Public License 2.0**. No PlantCV source is vendored or redistributed
here — it is an ordinary runtime dependency — so the MIT license applies to this
project's own files. See [NOTICE](NOTICE) for the full statement.

## Limitations

Phase 1 is single-ROI: `measure()` uses the whole image as its region of
interest, which is why the multi-specimen warning can only advise rather than
correct. Multi-plant grids, morphology traits (leaf angles, stem, skeleton) and
iterative mask refinement are phase 2.

Sessions are in-memory and capped (8 by default, LRU-evicted). They do not
survive a server restart.
