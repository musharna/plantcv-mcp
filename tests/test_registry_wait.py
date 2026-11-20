"""The registry workflow's wait for PyPI, run as the workflow runs it.

The step is shell inside a workflow file: not imported, not linted as code and
executed only when a tag is pushed, so a flaw in it shows up during a release.
One did. For about a minute after an upload PyPI's answers for the new version
disagree, and the step asked twice: once to see the version was live and once
to read its description. The first answer was a 200, the second a 404, the
read got an empty body, and tmodloader-mcp's 0.8.0 registry publish failed
before it began. This repo carried the same step.

These tests take the step's `run:` text out of the workflow and run it with a
stand-in `curl` that answers from a script of status codes, so they exercise
what the workflow executes, not a copy of it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
WORKFLOW = REPO / ".github" / "workflows" / "publish-registry.yml"
STEP_NAME = "Wait for the PyPI release, then verify it carries the mcp-name marker"
MARKER = "mcp-name: io.github.musharna/plantcv-mcp"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None,
    reason="the workflow step is bash; no bash on this host",
)

STAND_IN_CURL = """\
#!/usr/bin/env bash
# Answers request N with line N of $STUB/codes (the last line from then on).
n=$(cat "$STUB/count")
echo $((n + 1)) > "$STUB/count"
code=$(sed -n "$((n + 1))p" "$STUB/codes")
[ -n "$code" ] || code=$(tail -n 1 "$STUB/codes")
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in -o) out=$2; shift ;; esac
  shift
done
if [ "$code" = 200 ]; then
  if [ -n "$out" ]; then cp "$STUB/body" "$out"; else cat "$STUB/body"; fi
  exit 0
fi
echo "curl: (22) The requested URL returned error: $code" >&2
exit 22
"""


def step_script() -> str:
    """The `run:` text of the wait step, as written in the workflow."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.strip() == f"- name: {STEP_NAME}"
    )
    assert lines[start + 1].strip() == "run: |", lines[start + 1]
    body = []
    for line in lines[start + 2 :]:
        if line.strip().startswith("- name:"):
            break
        body.append(line)
    script = textwrap.dedent("\n".join(body))
    assert "pypi" in script.lower(), (
        "the wait step no longer mentions PyPI; this test reads the wrong step"
    )
    return script


def run_step(
    tmp_path: Path, codes: list[int], description: str
) -> tuple[subprocess.CompletedProcess[str], int]:
    """Run the step against a PyPI that answers `codes` in order; also return the request count."""
    stub = tmp_path / "stub"
    stub.mkdir(parents=True)
    (stub / "codes").write_text("\n".join(str(c) for c in codes) + "\n")
    (stub / "count").write_text("0\n")
    (stub / "body").write_text(json.dumps({"info": {"description": description}}))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(STAND_IN_CURL)
    curl.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        STUB=str(stub),
        PYPI_WAIT_PAUSE="0",
        PYPI_WAIT_TRIES="8",
    )
    # GitHub runs a `run:` block with `bash -e`, from the checkout's root.
    done = subprocess.run(
        ["bash", "-e", "-c", step_script()],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return done, int((stub / "count").read_text())


def test_an_answer_that_flips_back_to_404_does_not_fail_the_step(tmp_path):
    """200, then 404, then steady 200s: what PyPI served during tmodloader-mcp's 0.8.0 release."""
    done, requests = run_step(
        tmp_path, [200, 404, 200, 200, 200], f"readme\n{MARKER}\n"
    )

    assert done.returncode == 0, done.stdout + done.stderr
    assert "marker present: True" in done.stdout
    # The 404 restarts the count, so "live" is three answers in a row after it,
    # and the marker is read from the last of them, not from a further request.
    assert requests == 5, (requests, done.stdout)


def test_a_steady_pypi_passes_and_a_missing_marker_or_version_fails(tmp_path):
    steady, requests = run_step(tmp_path / "steady", [200], f"readme\n{MARKER}\n")
    assert steady.returncode == 0, steady.stdout + steady.stderr
    assert requests == 3, requests

    no_marker, _ = run_step(tmp_path / "no_marker", [200], "a readme with no marker")
    assert no_marker.returncode == 1, no_marker.stdout + no_marker.stderr
    assert MARKER in no_marker.stdout and "does not carry" in no_marker.stdout

    never, requests = run_step(tmp_path / "never", [404], f"readme\n{MARKER}\n")
    assert never.returncode == 1, never.stdout + never.stderr
    assert "did not serve" in never.stdout
    assert requests == 8, requests
