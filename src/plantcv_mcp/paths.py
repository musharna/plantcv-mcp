"""Read-root allow-list for every path a tool accepts.

Unset, the server reads anything the host user can read — the documented trust
boundary of a local stdio server. Configured (PLANTCV_MCP_ROOTS, os.pathsep-
separated, or `plantcv-mcp --root DIR` repeated), every path argument must
resolve to a location inside one of the roots.

`os.path.realpath` runs BEFORE the containment check, so `..` and symlinks are
followed first: a symlink inside the root that points outside is refused, and a
path that walks out and back in is allowed. The refusal names the roots, so a
client can see the policy rather than guess at it.
"""

import os

_ENV = "PLANTCV_MCP_ROOTS"
_roots: list[str] | None = None


class PathOutsideRootsError(Exception):
    """Raised when a path resolves outside every configured read root."""


def set_roots(roots: list[str] | None) -> None:
    """Override the environment (None restores it). The CLI and tests use this."""
    global _roots
    _roots = None if roots is None else [os.path.realpath(r) for r in roots]


def configured_roots() -> list[str] | None:
    """The active roots as realpaths, or None when reads are unrestricted."""
    if _roots is not None:
        return list(_roots)
    raw = os.environ.get(_ENV)
    if raw is None or not raw.strip():
        return None
    return [os.path.realpath(r) for r in raw.split(os.pathsep) if r.strip()]


def check_readable(path: str) -> str:
    """Return the realpath of `path`, or raise PathOutsideRootsError."""
    real = os.path.realpath(path)
    roots = configured_roots()
    if roots is None:
        return real
    for root in roots:
        try:
            if os.path.commonpath([root, real]) == root:
                return real
        except ValueError:  # different drives on Windows
            continue
    raise PathOutsideRootsError(
        f"{path!r} resolves to {real!r}, which is outside the configured read "
        f"roots {roots}. This server only reads under those directories "
        "(PLANTCV_MCP_ROOTS / --root)."
    )
