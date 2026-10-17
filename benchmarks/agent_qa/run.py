"""Ask every question to Claude Code under each arm, and score the replies.

Arms (same model, same prompt, isolated: no user settings, hooks or other MCP
servers, an empty working directory per run holding only the question's image):

- ``bare``: no tools at all. It cannot open the image, so the right reply is
  ``ANSWER: UNKNOWN``; it is kept as the control for guessing.
- ``alt``:  what a user has without this server: Bash, Read (which shows Claude
  the image), WebSearch and WebFetch, with a Python on PATH that has PlantCV
  and OpenCV installed (``.alt-venv``, made by ``--make-alt-venv``) and NOT
  plantcv-mcp. Claude writes its own PlantCV/OpenCV code.
- ``mcp``:  plantcv-mcp (pinned PyPI release via uvx) only.

Each finished run is appended to ``results.jsonl`` with its score; a rerun
skips (arm, question, rep) triples already there, so an interrupted sweep
resumes. Raw stream-json transcripts go to ``runs/<arm>/<qid>-<rep>.jsonl``.

    uv run python benchmarks/agent_qa/run.py --make-alt-venv   # once
    uv run python benchmarks/agent_qa/run.py --reps 3 --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from score import score

HERE = Path(__file__).resolve().parent
MODEL = "claude-opus-5-5"
SERVER = "plantcv-mcp==1.16.1"
# uvx resolves without this repo's lock. Its default interpreter here is 3.14,
# for which the pinned PlantCV's scipy has no wheel, so the server is started
# on 3.12, a version the package declares and CI tests.
SERVER_PYTHON = "3.12"
PLANTCV = "plantcv==4.11.3"  # the version plantcv-mcp 1.16.1 pins, so both arms run the same PlantCV
ALT_VENV = HERE / ".alt-venv"
RUN_TIMEOUT_S = 900
# Every arm gets this in place of Claude Code's own system prompt. That prompt
# describes tools even when --tools "" removes them all, and in the sibling
# plant-genomics-mcp pilot the no-tools arm wrote fake <invoke name="Bash">
# calls as text instead of answering. Tools still reach the model through the
# API in the other arms.
SYSTEM = "You are a helpful assistant. Answer the user's question."
PROMPT = (
    "{question}\n\nThe image is the file {path}\n\nGive the answer as {answer_format}. "
    "Keep your reply short, and end it with one line of the form\nANSWER: <answer>\n"
    "If you cannot determine the answer, end with\nANSWER: UNKNOWN"
)
# A tool call whose input mentions one of these could have read the answer key
# or the server's code instead of measuring the image. Checked per run and
# reported, not assumed away.
KEY_MARKERS = (
    "questions.json",
    "agent_qa",
    "plantcv_mcp",
    "plantcv-mcp",
    "build_questions",
)


class HarnessError(RuntimeError):
    """A run that did not test what its arm says (e.g. the server was not
    connected when the model started). Never scored; the run is retried on
    the next invocation."""


def clean_env(arm: str) -> dict[str, str]:
    """The caller's environment minus this repo's virtualenv (``uv run`` puts
    it first on PATH, and it has plantcv-mcp installed); the alt arm gets
    ``.alt-venv/bin`` in its place."""
    env = {
        k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")
    }
    repo_venv = str(HERE.parent.parent / ".venv")
    path = [
        p for p in env.get("PATH", "").split(os.pathsep) if not p.startswith(repo_venv)
    ]
    if arm == "alt":
        path.insert(0, str(ALT_VENV / "bin"))
        env["VIRTUAL_ENV"] = str(ALT_VENV)
    if arm == "mcp":
        # Claude Code connects MCP servers in the background by default, and in
        # the pilot the first turn started while this server (uvx + PlantCV
        # import, ~10-30 s) was still "pending": the model saw no tools and
        # declined. Blocking on the connection makes every mcp run start with
        # the server's tools; run_one also refuses a run whose init event does
        # not show the server connected.
        env["MCP_CONNECTION_NONBLOCKING"] = "false"
        env["MCP_TIMEOUT"] = "180000"
    env["PATH"] = os.pathsep.join(path)
    return env


def arm_flags(arm: str, tmp: Path) -> list[str]:
    empty = tmp.parent / f"{tmp.name}-no-mcp.json"
    empty.write_text('{"mcpServers": {}}')
    if arm == "bare":
        return ["--tools", "", "--strict-mcp-config", "--mcp-config", str(empty)]
    if arm == "alt":
        tools = "Bash,Read,WebSearch,WebFetch"
        return [
            "--tools", tools, "--allowedTools", tools,
            "--strict-mcp-config", "--mcp-config", str(empty),
        ]  # fmt: skip
    if arm == "mcp":
        cfg = tmp.parent / f"{tmp.name}-mcp.json"
        cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "plantcv": {
                            "command": "uvx",
                            "args": ["--python", SERVER_PYTHON, SERVER],
                        }
                    }
                }
            )
        )
        return [
            "--tools", "", "--allowedTools", "mcp__plantcv",
            "--strict-mcp-config", "--mcp-config", str(cfg),
        ]  # fmt: skip
    raise ValueError(arm)


def tool_uses(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        b
        for e in events
        if e.get("type") == "assistant"
        for b in e["message"].get("content", [])
        if b.get("type") == "tool_use"
    ]


def run_one(arm: str, q: dict[str, Any], rep: int) -> dict[str, Any]:
    out = HERE / "runs" / arm / f"{q['id']}-{rep}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agentqa-") as d:
        # The working directory holds the image and nothing else; the MCP
        # config files sit beside it, not in it.
        tmp = Path(d) / "work"
        tmp.mkdir()
        image = tmp / ("image" + Path(q["image"]).suffix)
        shutil.copyfile(HERE / "images" / q["image"], image)
        if hashlib.sha256(image.read_bytes()).hexdigest() != q["sha256"]:
            raise HarnessError(
                f"{q['id']}: images/{q['image']} does not match questions.json; "
                "run build_questions.py"
            )
        prompt = PROMPT.format(
            question=q["question"], path=image, answer_format=q["answer_format"]
        )
        cmd = [
            "claude", "-p", prompt, "--model", MODEL, "--setting-sources", "",
            "--output-format", "stream-json", "--verbose", "--system-prompt", SYSTEM,
            *arm_flags(arm, tmp),
        ]  # fmt: skip
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                env=clean_env(arm),
                check=False,
                capture_output=True,
                text=True,
                timeout=RUN_TIMEOUT_S,
            )
            stdout, rc = proc.stdout, proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (
                (exc.stdout or b"").decode()
                if isinstance(exc.stdout, bytes)
                else exc.stdout or ""
            )
            rc = "timeout"
        seconds = round(time.monotonic() - t0, 1)
    out.write_text(stdout)
    events = [json.loads(ln) for ln in stdout.splitlines() if ln.startswith("{")]
    init = next(
        (e for e in events if e.get("type") == "system" and e.get("subtype") == "init"),
        {},
    )
    result = next((e for e in events if e.get("type") == "result"), {})
    uses = tool_uses(events)
    reply = result.get("result") or ""
    if arm == "mcp":
        status = [s.get("status") for s in init.get("mcp_servers") or []]
        if status != ["connected"]:
            raise HarnessError(
                f"mcp {q['id']} rep{rep}: server status at init {status}, not connected"
            )
    return {
        "arm": arm,
        "qid": q["id"],
        "category": q["category"],
        "rep": rep,
        "exit": rc,
        "seconds": seconds,
        "claude_code": init.get("claude_code_version"),
        "model": init.get("model"),
        "mcp_servers": init.get("mcp_servers"),
        "tools_available": init.get("tools") or [],
        "tool_calls": [u["name"] for u in uses],
        "touched_key": any(
            m in json.dumps(u.get("input", {})) for u in uses for m in KEY_MARKERS
        ),
        "cost_usd": result.get("total_cost_usd"),
        "reply": reply,
        **score(q, reply),
    }


def rescore(results: Path, by_id: dict[str, dict[str, Any]]) -> None:
    """Score each stored reply again, so a truth or scorer fix reaches runs
    made before it; prints every verdict that changed."""
    rows = [json.loads(ln) for ln in results.read_text().splitlines()]
    for r in rows:
        new = score(by_id[r["qid"]], r["reply"])
        if new["verdict"] != r["verdict"] or new["slip"] != r["slip"]:
            print(f"{r['arm']:4} {r['qid']} rep{r['rep']} {r['verdict']} -> {new['verdict']}"
                  f" ({new['value']!r}, slip {new['slip']})")  # fmt: skip
        r.update(new)
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))


def make_alt_venv() -> None:
    """PlantCV and OpenCV in a venv of their own, so the alt arm cannot import
    plantcv-mcp (this repo's .venv has it installed, editable)."""
    py = ALT_VENV / "bin" / "python"
    subprocess.run(
        ["uv", "venv", str(ALT_VENV), "--python", SERVER_PYTHON, "--seed"], check=True
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(py), PLANTCV, "opencv-python>=4.9"],
        check=True,
    )
    check_alt_venv()


def check_alt_venv() -> None:
    """The alt arm's Python must have PlantCV and OpenCV and must not have
    plantcv-mcp; both are checked by running it, before any question is asked."""
    py = ALT_VENV / "bin" / "python"
    has = subprocess.run(
        [str(py), "-c", "import plantcv, cv2"],
        env=clean_env("alt"),
        capture_output=True,
        check=False,
    )
    lacks = subprocess.run(
        [str(py), "-c", "import plantcv_mcp"],
        env=clean_env("alt"),
        capture_output=True,
        check=False,
    )
    if has.returncode != 0 or lacks.returncode == 0:
        sys.exit(
            f"{py}: plantcv+cv2 import exit {has.returncode} (want 0), plantcv_mcp import "
            f"exit {lacks.returncode} (want non-zero). Run with --make-alt-venv."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bare,alt,mcp")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument("--results", default=str(HERE / "results.jsonl"))
    ap.add_argument(
        "--make-alt-venv", action="store_true", help="build .alt-venv, run nothing"
    )
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="re-score every stored reply against the current questions.json, run nothing",
    )
    a = ap.parse_args()
    questions = json.loads((HERE / "questions.json").read_text())["questions"]
    if a.make_alt_venv:
        make_alt_venv()
        return
    if a.rescore:
        rescore(Path(a.results), {q["id"]: q for q in questions})
        return
    if "alt" in a.arms.split(","):
        check_alt_venv()
    if a.only:
        keep = set(a.only.split(","))
        questions = [q for q in questions if q["id"] in keep]
    results = Path(a.results)
    done = set()
    if results.exists():
        for ln in results.read_text().splitlines():
            r = json.loads(ln)
            done.add((r["arm"], r["qid"], r["rep"]))
    todo = [
        (arm, q, rep)
        for rep in range(1, a.reps + 1)
        for q in questions
        for arm in a.arms.split(",")
        if (arm, q["id"], rep) not in done
    ]
    print(f"{len(todo)} runs to do ({len(done)} already in {results.name})", flush=True)
    lock = threading.Lock()
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_one, *t) for t in todo]
        for n, f in enumerate(as_completed(futures), 1):
            try:
                r = f.result()
            except HarnessError as exc:
                failures.append(str(exc))
                print(
                    f"[{n}/{len(todo)}] HARNESS ERROR, not recorded: {exc}", flush=True
                )
                continue
            with lock, results.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(
                f"[{n}/{len(todo)}] {r['arm']:4} {r['qid']} rep{r['rep']} {r['verdict']:9}"
                f" exit={r['exit']} {r['seconds']}s calls={len(r['tool_calls'])}"
                f" ${r['cost_usd']} {r['value']!r}",
                flush=True,
            )
    if failures:
        sys.exit(
            f"{len(failures)} runs hit a harness error and were not recorded; rerun to retry them"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
