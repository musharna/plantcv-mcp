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


def _inside(real: str, roots: list[str]) -> bool:
    for root in roots:
        try:
            if os.path.commonpath([root, real]) == root:
                return True
        except ValueError:  # different drives on Windows
            continue
    return False


def check_readable(path: str) -> str:
    """Return the realpath of `path`, or raise PathOutsideRootsError.

    This judges a NAME, and a name can be re-pointed between the check and
    the open (any ancestor directory swapped for a symlink — O_NOFOLLOW only
    guards the last component). Readers therefore also call check_open_fd on
    the descriptor they actually opened; this check is the early, named
    refusal, that one is the binding.
    """
    real = os.path.realpath(path)
    roots = configured_roots()
    if roots is None or _inside(real, roots):
        return real
    raise PathOutsideRootsError(
        f"{path!r} resolves to {real!r}, which is outside the configured read "
        f"roots {roots}. This server only reads under those directories "
        "(PLANTCV_MCP_ROOTS / --root)."
    )


def fd_path(fd: int) -> str:
    """The current path of an OPEN descriptor, from the kernel — not from
    any name the caller holds. Linux answers through /proc; macOS through
    F_GETPATH. Anywhere else there is no way to ask, and a containment check
    that cannot be made is refused rather than skipped."""
    try:
        return os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        pass
    try:
        import fcntl

        getpath = fcntl.F_GETPATH  # macOS only
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "This platform cannot report where an open file lives, so read "
            "roots cannot be enforced on it (Linux and macOS can). Run without "
            "PLANTCV_MCP_ROOTS / --root, or on a supported platform."
        ) from exc
    raw = fcntl.fcntl(fd, getpath, bytes(1024))
    return os.fsdecode(raw.split(b"\0", 1)[0])


def check_open_fd(fd: int, path: str) -> str:
    """Check the file BEHIND `fd` against the roots; return its path.

    Panel audit of 1.9.0: a member directory renamed and replaced by an
    outside symlink between check_readable and os.open was followed — the
    resolved pathname was re-resolved by the kernel, ancestors included.
    The descriptor is the thing that was opened; ask the kernel where it is.
    """
    roots = configured_roots()
    if roots is None:
        return path
    real = fd_path(fd)
    if _inside(real, roots):
        return real
    raise PathOutsideRootsError(
        f"{path!r} was opened as {real!r}, which is outside the configured "
        f"read roots {roots} — the path changed between the check and the "
        "open. Nothing was read."
    )
