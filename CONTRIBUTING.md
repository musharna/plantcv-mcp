# Contributing

Thanks for helping improve `plantcv-mcp`. It is an MCP server exposing PlantCV as a measurement instrument: it returns trait numbers together with the segmentation overlay they were computed from, and refuses to return numbers when the segmentation is degenerate.

## Dev setup

Requires Python >=3.11 and [`uv`](https://docs.astral.sh/uv/).

No system packages are required, but note that PlantCV depends on **non-headless**
`opencv-python`. Swapping in `opencv-python-headless` will not give you a headless
install; it will give you a broken one.

PlantCV is **hard-pinned** at `4.11.3` because trait values can shift between
releases and determinism is a tested, advertised guarantee. Re-pin explicitly when
validating a new PlantCV release — do not loosen the pin.

```bash
uv sync
```

## Running the tests

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The suite exercises real images, including deliberately inverted masks: an inverted threshold reports the background as the plant with no error at all, so two-sided mask validity is the thing under test.

## Testing rules

These are the rules this repo is actually held to. They share one idea: **a test is
worth only what it can fail on.**

### 1. Never trust a test you have not seen fail

Before a test is trusted, run it against the broken state — for a bug fix, the
pre-fix code you still have — and confirm it fails *for the stated reason*, not
merely that it fails. A test asserting "some exception was raised" passes against
broken code too.

### 2. A negative result needs a positive control

A test asserting that something is refused must also assert, **in the same test**,
that the legitimate path still succeeds. Otherwise a harness that raises on
everything reads as "the guard works". `tests/test_registry_metadata.py` is the
worked example: it rejects an over-long `server.json` description *and* validates
the real document in the same function.

### 3. Prefer an external oracle to self-comparison

A test that compares the server against itself will keep passing when the server is
consistently wrong. Where a ground truth exists — a known geometry, a simulated
tree, an analytic expectation — assert against that instead. See `docs/EVAL.md`.

### 4. Mutation checks

`docs/MUTATION-CHECKS.md` records deliberate mutations introduced to confirm the
suite catches them. **A surviving mutant is the coverage report.** If you change
behaviour in a guarded area, add the mutation you used to prove the guard works.

## Pull requests

- Update `CHANGELOG.md` for any user-facing change (Keep a Changelog format).
- Update the README if tool signatures or configuration change.
- Bump nothing else: version lives in `pyproject.toml`, `server.json` (x3) and
  `CITATION.cff`, and a release PR moves them together. `tests/test_registry_metadata.py`
  enforces that they agree, and validates `server.json` against the MCP registry's
  published schema so a constraint violation fails here rather than after a tag is cut.
- Fail loud. Do not add silent fallbacks or swallow errors; surface enough context
  (inputs, what was attempted) that the failure is debuggable.
