"""The server must survive more than one tool call, in a process that is not pytest.

This file exists because of a bug in a SIBLING server, not one found here. In
breedsim-mcp, rpy2 published its conversion rules into a `contextvars.ContextVar`
at import time; the import happened lazily inside the first request, so the rules
were discarded when that request returned and every subsequent tool call failed.
Twenty-seven tests passed throughout, because a test that calls the library
directly imports the dependency into pytest's root context first and masks the
whole thing.

The audit that prompted this file found plantcv-mcp had no fresh-process test at
all, so that class of failure — state established at import time that only
misbehaves inside a request — was undetectable here by construction. Nothing is
known to be broken; what was missing was the ability to notice.

PlantCV is not free of import-time global state either: `pcv.params` is a
module-level singleton, and this package READS `pcv.params.sample_label` when it
builds trait keys. A process that never sets it and a process that does are
different worlds, and only a fresh one tells you which you are in.
"""

import os
import subprocess
import sys

# Drive every compute tool in order, through call_tool and nothing else. A direct
# library call here would re-mask exactly the bug this exists to expose.
DRIVER = """
import asyncio, json, sys, tempfile, os
import numpy as np
from PIL import Image
from plantcv_mcp.server import build_server

srv = build_server()

def call(name, args):
    r = asyncio.run(srv.call_tool(name, args))
    txt = next((c.text for c in (r.content or []) if getattr(c, "text", None)), None)
    return json.loads(txt) if txt else None

d = tempfile.mkdtemp()
p = os.path.join(d, "leaf.png")
a = np.full((160, 160, 3), 200, dtype=np.uint8)
a[40:120, 40:120] = (40, 150, 40)          # an 80x80 square: area is 6400 px
Image.fromarray(a).save(p)

call("list_methods", {})
s = call("segment", {"image_path": p, "channel": "a", "method": "otsu"})
m = call("measure", {"session_id": s["session_id"]})

assert s["largest_area"] == 6400, s["largest_area"]
assert m["traits"]["area"]["value"] == 6400.0, m["traits"]["area"]
assert m["engine"]["version"], "no engine version on the result"
print("SEQUENCE_OK")
"""


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,  # a non-zero exit is the failure mode under test, not an error
        env={**os.environ},
    )


def test_three_tool_calls_in_one_fresh_process():
    """Three sequential calls must all succeed in a process that has never
    touched PlantCV outside the tool layer."""
    proc = _run(DRIVER)
    assert "SEQUENCE_OK" in proc.stdout, (
        "the tool layer could not complete three sequential calls.\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr[-3000:]}"
    )


def test_the_driver_can_actually_fail():
    """Positive control for the harness above.

    Asserting on stdout is only meaningful if a broken server actually withholds
    SEQUENCE_OK. A subprocess that dies for an unrelated reason — a missing
    dependency, a bad path, an import error — would otherwise read identically to
    a passing one, and this file would be decoration.
    """
    proc = _run(DRIVER.replace('call("list_methods", {})', 'raise SystemExit("boom")'))
    assert "SEQUENCE_OK" not in proc.stdout
    assert proc.returncode != 0


def test_a_preset_sample_label_does_not_change_the_measurement():
    """`pcv.params.sample_label` is a module-level singleton this package reads.

    A host process that set it before importing us must not shift trait keys or
    values. Checked in a FRESH process because the point is the state a real
    client's process might arrive in, which pytest's cannot represent.
    """
    preset = (
        "import plantcv.plantcv as pcv\npcv.params.sample_label = 'host_set'\n" + DRIVER
    )
    proc = _run(preset)
    assert "SEQUENCE_OK" in proc.stdout, (
        "a sample_label set before import changed the result.\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr[-3000:]}"
    )
