"""Every tool that can write a file is named where the docs list what the server writes.

The README's Security section and the guide's trust-boundary section both said the
server writes only the lens-corrected image (plus the SAM checkpoint and temporary
copies). `crop_image`, added in 1.18.0, writes `<image>_crop_x.._y.._w.._h...png`
beside its input and replaces it on a repeat crop, and neither list named it
(audit of 2026-11-10, P-M3). A list kept by hand drifts the moment a writing tool is
added, so it is checked here against the server's own declaration: a tool whose
annotations say it is not read-only is a tool that writes.
"""

import asyncio
from pathlib import Path

from plantcv_mcp.server import build_server

ROOT = Path(__file__).resolve().parents[1]


def _section(path: Path, heading: str) -> str:
    """The text under `## <heading>` up to the next level-2 heading."""
    blocks = path.read_text().split("\n## ")
    found = [b for b in blocks if b.split("\n", 1)[0] == heading]
    assert len(found) == 1, f"{path.name} has {len(found)} '## {heading}' sections"
    return found[0]


def _undocumented(writers: list[str], section: str) -> list[str]:
    """Writers the section does not name (as `name` or `name()`)."""
    return [name for name in writers if f"`{name}" not in section]


def test_every_writing_tool_is_named_in_both_write_lists():
    tools = asyncio.run(build_server().list_tools())
    writers = sorted(t.name for t in tools if t.annotations.read_only_hint is False)
    # Premise: the declaration this test reads does mark the writers. If the
    # annotation stopped saying so, an empty list would pass everything below.
    assert "crop_image" in writers and "correct_lens_distortion" in writers, writers

    sections = {
        "README.md": _section(ROOT / "README.md", "Security"),
        "docs/GUIDE.md": _section(
            ROOT / "docs" / "GUIDE.md", "Security and trust boundary"
        ),
    }
    missing = {doc: _undocumented(writers, text) for doc, text in sections.items()}
    assert missing == {doc: [] for doc in sections}, (
        f"tools that write a file but are not named in the doc's write list: {missing}"
    )

    # Control: the check can fail. The same section with one writer's name
    # taken out is reported as missing exactly that writer.
    readme = sections["README.md"]
    stripped = readme.replace("`crop_image", "`a_tool")
    assert stripped != readme, "premise: the README section names crop_image"
    assert _undocumented(writers, stripped) == ["crop_image"]
