# Read-root allow-list (0.9.0)

**Parent:** `2026-08-27-backlog-integration-plan-of-attack.md`, sub-project D.

## What changes

Today every tool that takes a path (`suggest_segmentation`, `segment`,
`calibrate_scale_from_marker`, `measure_images`, and E's readers) reads anything the
host user can read; README §"Security and trust boundary" documents this. The
allow-list makes that boundary configurable without changing the default.

- `PLANTCV_MCP_ROOTS` — `os.pathsep`-separated directories — and/or repeated
  `--root PATH` on the console script (CLI wins when both are given).
- When at least one root is configured, every path argument must resolve
  (`os.path.realpath`, so symlinks and `..` are followed _before_ the check) to a
  location inside one of the roots; otherwise the call is refused with
  `PathOutsideRootsError` naming the configured roots. `measure_images` checks every
  path **before** loading any: a batch with one stray path is refused whole, not
  partially run.
- Unset → unchanged behaviour. `list_methods()` reports `read_roots` (the configured
  list, or `null`) so a client can see the policy.

## Where

`paths.py`: `configured_roots() -> list[str] | None`, `set_roots(list | None)`,
`check_readable(path) -> str` (returns the realpath). The check is called at the
top of each tool, before `load_image*`; `batch.py` receives already-checked paths.

## Tests

- Inside a root → measures (positive control in every refusal test).
- `..` traversal out of a root, a symlink inside the root pointing outside, an
  absolute path outside → refused, message names the roots.
- Two roots: a path under the second is allowed.
- No roots configured → an arbitrary temp path is allowed (today's behaviour).
- `measure_images` with one stray path → refused whole, nothing measured.
- `--root` on `main()` sets the policy; `PLANTCV_MCP_ROOTS` alone sets it too.
- README + SECURITY.md updated.
