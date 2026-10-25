"""Ask every question to Claude Code under each arm, and score the replies.

Arms (same model, same prompt, isolated: no user settings, hooks or other MCP
servers, an empty working directory per run holding only the question's image):

- ``bare``: no tools at all. It cannot open the image, so the right reply is
  ``ANSWER: UNKNOWN``; it is kept as the control for guessing.
- ``alt``:  what a user has without this server: Bash, Read (which shows Claude
  the image), WebSearch and WebFetch, with a Python on PATH that has PlantCV
  and OpenCV installed (``.alt-venv``, made by ``--make-alt-venv``) and NOT
  plantcv-mcp. Claude writes its own PlantCV/OpenCV code.
- ``mcp``:  plantcv-mcp (pinned PyPI release via uvx) plus Read, so it can
  see the image as the alt arm can.

Every attempt is appended to ``results.jsonl`` with its score and a validity
verdict decided by one rule for every arm (``assess``); invalid attempts are
kept and reported, never retried or overwritten. A rerun skips (arm, question,
rep) triples already there, so an interrupted sweep resumes. Raw stream-json
transcripts go OUTSIDE the worktree (``ARTIFACTS/runs/<arm>/<qid>-<rep>.jsonl``),
so a Bash call that searches the disk does not find them beside the key.

    uv run python benchmarks/agent_qa/run.py --make-alt-venv   # once
    uv run python benchmarks/agent_qa/run.py --reps 3 --workers 2
"""

from __future__ import annotations

import argparse
import datetime as dt
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
# Both arms get the same PlantCV and OpenCV; check_versions() asserts it before
# a sweep. OpenCV is pinned in the server's uvx environment with --with.
OPENCV = "opencv-python==4.11.0.86"
VERSIONS = {"plantcv": "4.11.3", "opencv-python": "4.11.0.86"}
ARTIFACTS = Path(
    os.environ.get(
        "AGENTQA_ARTIFACTS", Path.home() / "plantcv-mcp-artifacts" / "agent_qa"
    )
)
RUNS = ARTIFACTS / "runs"
# uvx resolves without this repo's lock. Its default interpreter here is 3.14,
# for which the pinned PlantCV's scipy has no wheel, so the server is started
# on 3.12, a version the package declares and CI tests.
SERVER_PYTHON = "3.12"
SERVER_UVX = ["--python", SERVER_PYTHON, "--with", OPENCV, SERVER]
# --server-spec runs an unreleased commit instead of the PyPI pin: the git URL
# of a commit pushed to the PR branch, by full SHA so it cannot move. Set once
# by main() before any run; recorded in every row and in versions.json.
SPEC_PREFIX = "git+https://github.com/musharna/plantcv-mcp@"
SERVER_SPEC: str | None = None
# --exclude-newer pins the server environment's dependencies to what was on
# PyPI at that time, so a git install resolves the same NumPy/scikit-image as
# the alt arm's venv (check_versions asserts they match).
EXCLUDE_NEWER: str | None = None
# The account's usage windows, read from each transcript's rate_limit_event.
# A run that sees the five-hour window at or above PAUSE_AT, or a rejected
# request, pauses the whole sweep until the window resets: it sets PAUSE_UNTIL,
# and every worker waits for it before starting a run (a run already under way
# on another worker finishes). Each pause is logged in pauses.jsonl. A seven-day window at or above STOP_AT
# stops the sweep: runs not yet started are left for a resume.
PAUSE_AT = 0.85
STOP_AT = 0.80
PAUSES: Path | None = None
STOP = threading.Event()
PAUSE_UNTIL = 0.0
PAUSE_LOCK = threading.Lock()


def server_uvx() -> list[str]:
    """uvx arguments that start the server under test."""
    pins = ["--exclude-newer", EXCLUDE_NEWER] if EXCLUDE_NEWER else []
    if SERVER_SPEC is None:
        return [*pins, *SERVER_UVX]
    return [
        *pins, "--python", SERVER_PYTHON, "--with", OPENCV,
        "--from", SERVER_SPEC, "plantcv-mcp",
    ]  # fmt: skip


def expected_server_version() -> str:
    """The plantcv-mcp version the server must report: the PyPI pin, or the
    version in pyproject.toml at the spec's commit (read from git, so a spec
    whose commit is not in this clone fails here, not mid-sweep)."""
    if SERVER_SPEC is None:
        return SERVER.split("==")[1]
    sha = SERVER_SPEC.removeprefix(SPEC_PREFIX)
    toml = subprocess.run(
        ["git", "-C", str(HERE), "show", f"{sha}:pyproject.toml"],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return next(
        ln.split('"')[1] for ln in toml.splitlines() if ln.startswith("version = ")
    )


def set_server_spec(spec: str | None, exclude_newer: str | None) -> None:
    global SERVER_SPEC, EXCLUDE_NEWER
    if spec is not None:
        sha = spec.removeprefix(SPEC_PREFIX)
        if (
            not spec.startswith(SPEC_PREFIX)
            or len(sha) != 40
            or not all(c in "0123456789abcdef" for c in sha)
        ):
            sys.exit(
                f"--server-spec must be {SPEC_PREFIX}<full 40-hex sha>, got {spec!r}"
            )
    SERVER_SPEC, EXCLUDE_NEWER = spec, exclude_newer


DEFAULT_MODEL = MODEL


def set_model(model: str) -> None:
    """--model runs every arm on another model. Its transcripts go to their own
    runs-<model>/ so they never overwrite the default model's."""
    global MODEL, RUNS
    MODEL = model
    RUNS = ARTIFACTS / ("runs" if model == DEFAULT_MODEL else f"runs-{model}")


PLANTCV = "plantcv==4.11.3"  # the version plantcv-mcp 1.16.1 pins, so both arms run the same PlantCV
# Outside the worktree: a traceback from the alt arm prints this path, and it
# must not point the model at the benchmark directory (sweep 1, alt q40).
ALT_VENV = Path(
    os.environ.get(
        "AGENTQA_ALT_VENV", Path.home() / ".local/share/agentqa-plantcv-alt-venv"
    )
)
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
# What each arm is told about its own setup, appended to the prompt. The server
# describes its tools itself; the alt arm's Python is invisible unless named
# (in the stopped first sweep it used PlantCV in 0 of 63 runs).
ARM_NOTE = {
    "bare": "",
    "alt": "\n\nThe python3 on your PATH has PlantCV 4.11.3 and OpenCV 4.11 installed.",
    "mcp": "",
}
# The tools each arm must have at start-up for a run to count (init event).
REQUIRED_TOOLS = {
    "bare": set(),
    "alt": {"Bash", "Read", "WebSearch", "WebFetch"},
    "mcp": {"Read"},
}
WEB_TOOLS = ("WebSearch", "WebFetch")
# A question set can turn the alt arm's web tools off ("alt_web": false in its
# questions file): the field set's truth is public online. Set once by main()
# from the questions file before any run, like SERVER_SPEC.
ALT_WEB = True


def set_alt_web(on: bool) -> None:
    global ALT_WEB
    ALT_WEB = bool(on)
    REQUIRED_TOOLS["alt"] = {"Bash", "Read", *(WEB_TOOLS if ALT_WEB else ())}


def alt_tools() -> str:
    return ",".join(["Bash", "Read", *(WEB_TOOLS if ALT_WEB else ())])


# Anything that could carry the answer key into a run: the key and builder,
# the benchmark directory (local, and on GitHub for WebFetch/Bash curl), the
# worktree, and the artifacts directory. Scanned in every tool input AND
# output; a hit makes the run invalid.
LEAK_MARKERS = (
    "questions.json",
    "build_questions",
    "agent_qa",
    "plantcv-mcp-artifacts",
    "bench-1016",
    "/benchmarks/",
    "github.com/musharna/plantcv-mcp",
    "raw.githubusercontent.com/musharna",
)
# The field set (field_2026-10-24/) is real Aberystwyth trays whose hand
# annotations and leaf counts are public (Zenodo; CVPPP LCC A4). Matched
# without case, in every run of every set.
SOURCE_LEAK_MARKERS = (
    "zenodo",
    "zenodo.168158",  # (a bare 168158 could be a pixel count)
    "records/168158",
    "aberystwyth",
    "plant-phenotyping.org",
    "cvppp",
    "psi_tray",
    "_top_gt",
    "_top-gt",
    "leaf-evaluation",
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
    """The harness itself is broken (an image that does not match the key):
    stop before asking anything. Not used for a run that went wrong; those are
    recorded as invalid."""


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
        tools = alt_tools()
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
                            "args": server_uvx(),
                        }
                    }
                }
            )
        )
        return [
            "--tools", "Read", "--allowedTools", "Read,mcp__plantcv",
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


def tool_results(events: list[dict[str, Any]]) -> list[str]:
    """Text of every tool result (images left out)."""
    out = []
    for e in events:
        if e.get("type") != "user":
            continue
        for b in e["message"].get("content", []) or []:
            if not isinstance(b, dict) or b.get("type") != "tool_result":
                continue
            c = b.get("content")
            if isinstance(c, str):
                out.append(c)
            elif isinstance(c, list):
                out += [x.get("text", "") for x in c if x.get("type") == "text"]
    return out


def leaks(events: list[dict[str, Any]]) -> list[str]:
    """Leak markers found in any tool input or tool output."""
    text = "\n".join(
        [json.dumps(u.get("input", {})) for u in tool_uses(events)]
        + tool_results(events)
    )
    low = text.lower()
    return [m for m in LEAK_MARKERS if m in text] + [
        m for m in SOURCE_LEAK_MARKERS if m in low
    ]


def assess(arm: str, stdout: str, rc: int | str) -> dict[str, Any]:
    """Everything about one attempt except its score, from its transcript, by
    one rule for every arm. Invalid: exit not 0 (timeouts included); no init
    event; the arm's tools (and, for mcp, its connected server) missing at
    start-up; for alt and mcp, no tool call at all (the image cannot be
    measured without opening it); any leak marker in a tool input or output."""
    events = [json.loads(ln) for ln in stdout.splitlines() if ln.startswith("{")]
    init = next(
        (e for e in events if e.get("type") == "system" and e.get("subtype") == "init"),
        {},
    )
    result = next((e for e in events if e.get("type") == "result"), {})
    uses = tool_uses(events)
    tools = set(init.get("tools") or [])
    found = leaks(events)
    why = []
    if rc != 0:
        why.append(f"exit {rc}")
    if not init:
        why.append("no init event")
    else:
        if arm == "bare" and tools:
            why.append(f"tools present: {sorted(tools)}")
        if not REQUIRED_TOOLS[arm] <= tools:
            why.append(f"missing tools {sorted(REQUIRED_TOOLS[arm] - tools)}")
        if arm == "alt" and not ALT_WEB and tools & set(WEB_TOOLS):
            why.append(f"web tools present: {sorted(tools & set(WEB_TOOLS))}")
        if arm == "mcp":
            status = [s.get("status") for s in init.get("mcp_servers") or []]
            if status != ["connected"] or not any(
                t.startswith("mcp__plantcv__") for t in tools
            ):
                why.append(f"server status {status}")
    if arm != "bare" and not uses:
        why.append("no tool call")
    if found:
        why.append(f"leak {found}")
    return {
        "exit": rc,
        "claude_code": init.get("claude_code_version"),
        "model": init.get("model"),
        "mcp_servers": init.get("mcp_servers"),
        "tools_available": init.get("tools") or [],
        "tool_calls": [u["name"] for u in uses],
        # Argument NAMES per call (values left out: they hold paths), so the
        # use of a new tool or parameter can be counted per run.
        "tool_arg_names": [sorted((u.get("input") or {}).keys()) for u in uses],
        "touched_key": any(
            m in json.dumps(u.get("input", {})) for u in uses for m in KEY_MARKERS
        ),
        "leaks": found,
        "valid": not why,
        "invalid_reason": "; ".join(why) or None,
        "cost_usd": result.get("total_cost_usd"),
        "reply": result.get("result") or "",
    }


def rate_limit(events: list[dict[str, Any]]) -> dict[str, Any]:
    """The last rate_limit_event's status and five-hour / seven-day windows."""
    info = next(
        (e["rate_limit_info"] for e in reversed(events)
         if e.get("type") == "rate_limit_event" and e.get("rate_limit_info")),
        {},
    )  # fmt: skip
    win = info.get("unifiedWindows") or {}
    return {
        "status": info.get("status"),
        "five_hour": (win.get("five_hour") or {}).get("utilization"),
        "seven_day": (win.get("seven_day") or {}).get("utilization"),
        "resets_at": (win.get("five_hour") or {}).get("resetsAt")
        or info.get("resetsAt"),
    }


def api_failure(events: list[dict[str, Any]]) -> bool:
    """The API ended the run before the model answered: a request was rejected
    for the rate limit and no assistant message came back. Read from the
    transcript, never from the reply or its score."""
    rejected = rate_limit(events)["status"] == "rejected"
    answered = any(e.get("type") == "assistant" for e in events)
    return rejected and not answered


def pause_if_limited(rl: dict[str, Any], where: str) -> None:
    """Sleep until the five-hour window resets when it is at PAUSE_AT or a
    request was rejected, logging the pause; set STOP when the seven-day
    window is at STOP_AT."""
    if (rl["seven_day"] or 0) >= STOP_AT and not STOP.is_set():
        STOP.set()
        print(
            f"STOP after {where}: seven_day {rl['seven_day']} >= {STOP_AT}", flush=True
        )
    hot = rl["status"] == "rejected" or (rl["five_hour"] or 0) >= PAUSE_AT
    if not hot or not rl["resets_at"]:
        return
    global PAUSE_UNTIL
    with PAUSE_LOCK:
        PAUSE_UNTIL = max(PAUSE_UNTIL, rl["resets_at"] + 120)
    wait = max(0.0, PAUSE_UNTIL - time.time())
    start = dt.datetime.now().astimezone()
    print(f"pausing {wait / 60:.0f} min after {where}: {rl}", flush=True)
    if PAUSES is not None:
        with PAUSES.open("a") as fh:
            fh.write(json.dumps({"after": where, "at": start.isoformat(timespec="seconds"),
                                 "sleep_s": round(wait), **rl}) + "\n")  # fmt: skip
    wait_for_pause()


def spend_stop() -> str | None:
    """Why an API-billed sweep must stop before its next run, or None.

    Set ``BENCH_SPEND_GATE`` to the shared spend meter's script and each run
    first asks it (``check``, track from ``BENCH_TRACK``); any non-zero exit
    (spend cap, billing violation, checkpoint) stops the sweep cleanly.
    """
    gate = os.environ.get("BENCH_SPEND_GATE")
    if not gate:
        return None
    p = subprocess.run(
        [sys.executable, gate, "check"], capture_output=True, text=True, check=False
    )
    return (
        None
        if p.returncode == 0
        else (p.stderr.strip() or f"spend gate exit {p.returncode}")
    )


def wait_for_pause() -> None:
    """Sleep until PAUSE_UNTIL, if a pause is set; every worker calls this
    before it starts a run, so one worker's pause holds the whole sweep."""
    wait = PAUSE_UNTIL - time.time()
    if wait > 0:
        time.sleep(wait)


def stopped_before(arm: str, q: dict[str, Any], rep: int) -> bool:
    """True if the sweep is stopped, asking the spend gate if it is not yet."""
    if not STOP.is_set():
        why = spend_stop()
        if why and not STOP.is_set():
            STOP.set()
            print(f"STOP before {arm} {q['id']} rep{rep}: {why}", flush=True)
    return STOP.is_set()


# Where the questions' images live: HERE/images for the rendered set, or the
# questions file's "images_dir" (the field set keeps its dataset images
# outside the repo). Set once by main() from the questions file.
IMAGES = HERE / "images"


def prepare_workdir(q: dict[str, Any], tmp: Path) -> Path:
    """Copy the question's image (renamed image.<ext>) and any extra
    directories it names into the empty working directory, checking every
    file against the key's sha256 first. Returns the image's path."""
    image = tmp / ("image" + Path(q["image"]).suffix)
    shutil.copyfile(IMAGES / q["image"], image)
    if hashlib.sha256(image.read_bytes()).hexdigest() != q["sha256"]:
        raise HarnessError(
            f"{q['id']}: {IMAGES / q['image']} does not match the questions file; "
            "rebuild the set"
        )
    for name, rel in (q.get("extra_dirs") or {}).items():
        shutil.copytree(IMAGES / rel, tmp / name)
    want = q.get("extra_sha256") or {}
    have = {
        str(p.relative_to(tmp)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in (q.get("extra_dirs") or {})
        for p in sorted((tmp / name).rglob("*"))
        if p.is_file()
    }
    if have != want:
        raise HarnessError(f"{q['id']}: extra files differ from the questions file")
    return image


def render_prompt(arm: str, q: dict[str, Any], image: Path) -> str:
    return (
        PROMPT.format(
            question=q["question"], path=image, answer_format=q["answer_format"]
        )
        + ARM_NOTE[arm]
    )


def run_one(arm: str, q: dict[str, Any], rep: int) -> dict[str, Any] | None:
    """One attempt, scored; None if the sweep was stopped before it began.

    The stop checks run before a pending pause, so a stopped sweep returns at
    once rather than sleeping out the pause, and again after it, so the spend
    gate is read when the billed call is about to fire."""
    if stopped_before(arm, q, rep):
        return None
    wait_for_pause()
    if stopped_before(arm, q, rep):
        return None
    out = RUNS / arm / f"{q['id']}-{rep}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agentqa-") as d:
        # The working directory holds the image and nothing else; the MCP
        # config files sit beside it, not in it.
        tmp = Path(d) / "work"
        tmp.mkdir()
        image = prepare_workdir(q, tmp)
        prompt = render_prompt(arm, q, image)
        cmd = [
            "claude", "-p", prompt, "--model", MODEL, "--setting-sources", "",
            "--output-format", "stream-json", "--verbose", "--system-prompt", SYSTEM,
            *arm_flags(arm, tmp),
        ]  # fmt: skip
        # Each run gets its own TMPDIR beside (not in) the working directory.
        run_tmp = Path(d) / "tmp"
        run_tmp.mkdir()
        env = {**clean_env(arm), "TMPDIR": str(run_tmp)}
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                env=env,
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
    row = assess(arm, stdout, rc)
    events = [json.loads(ln) for ln in stdout.splitlines() if ln.startswith("{")]
    rl = rate_limit(events)
    pause_if_limited(rl, f"{arm} {q['id']} rep{rep}")
    return {
        "arm": arm,
        "qid": q["id"],
        "category": q["category"],
        "rep": rep,
        "seconds": seconds,
        "prompt_note": ARM_NOTE[arm],
        "server_spec": (SERVER_SPEC or SERVER) if arm == "mcp" else None,
        "exclude_newer": EXCLUDE_NEWER if arm == "mcp" else None,
        # Asked for; the model served is "model", from the init event.
        "model_requested": MODEL,
        "rate_limit": rl,
        "api_failure": api_failure(events),
        **row,
        **score(q, row["reply"]),
    }


def audit(results: Path) -> None:
    """Re-assess every stored row from its transcript (validity and leak scan)
    without touching its reply or score; prints every row that changed."""
    rows = [json.loads(ln) for ln in results.read_text().splitlines()]
    for r in rows:
        f = RUNS / r["arm"] / f"{r['qid']}-{r['rep']}.jsonl"
        new = assess(r["arm"], f.read_text(), r["exit"])
        if new["reply"] != r["reply"]:
            sys.exit(f"{f}: transcript reply differs from the stored row")
        if new["valid"] != r.get("valid") or new["leaks"] != r.get("leaks"):
            print(f"{r['arm']:4} {r['qid']} rep{r['rep']} valid {r.get('valid')} -> "
                  f"{new['valid']} ({new['invalid_reason']})")  # fmt: skip
        r.update({k: new[k] for k in ("valid", "invalid_reason", "leaks")})
        r.setdefault("prompt_note", "")
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n = len(rows)
    bad = sum(not r["valid"] for r in rows)
    hits = sum(bool(r["leaks"]) for r in rows)
    print(f"{n} rows audited: {bad} invalid, {hits} with a leak-marker hit")


def oracle(questions: list[dict[str, Any]]) -> None:
    """Positive control before any run: every key answer, written as a reply,
    scores right, and UNKNOWN scores declined. Stops the sweep otherwise."""
    from score import truth_answer

    bad = [
        q["id"]
        for q in questions
        if score(q, f"ANSWER: {truth_answer(q)}")["verdict"] != "right"
        or score(q, "ANSWER: UNKNOWN")["verdict"] != "declined"
    ]
    if bad:
        sys.exit(f"oracle check failed for {bad}")


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


def drop_api_failures(results: Path) -> None:
    """Move every row the API ended to results.api_failures.jsonl, unless its
    triple was already dropped once (then it stays, and counts as invalid)."""
    side = results.with_name("results.api_failures.jsonl")
    dropped = set()
    if side.exists():
        for ln in side.read_text().splitlines():
            r = json.loads(ln)
            dropped.add((r["arm"], r["qid"], r["rep"]))
    rows = [json.loads(ln) for ln in results.read_text().splitlines()]
    keep, move = [], []
    for r in rows:
        t = (r["arm"], r["qid"], r["rep"])
        (move if r.get("api_failure") and t not in dropped else keep).append(r)
    with side.open("a") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in move)
    results.write_text("".join(json.dumps(r) + "\n" for r in keep))
    print(
        f"moved {len(move)} API-failure rows to {side.name}; resume to re-run them once"
    )


def record_models_served(results: Path) -> None:
    """Add to versions.json the models the init events reported, so the file
    says what was served and not only what was asked for."""
    vf = results.parent / "versions.json"
    rows = [json.loads(ln) for ln in results.read_text().splitlines()]
    v = json.loads(vf.read_text())
    v["models_served"] = sorted({str(r.get("model")) for r in rows})
    vf.write_text(json.dumps(v, indent=1) + "\n")


def make_alt_venv() -> None:
    """PlantCV and OpenCV in a venv of their own, so the alt arm cannot import
    plantcv-mcp (this repo's .venv has it installed, editable)."""
    py = ALT_VENV / "bin" / "python"
    subprocess.run(
        ["uv", "venv", str(ALT_VENV), "--python", SERVER_PYTHON, "--seed"], check=True
    )
    subprocess.run(
        ["uv", "pip", "install", "--python", str(py), PLANTCV, OPENCV],
        check=True,
    )
    check_alt_venv()


VERSION_PROBE = (
    "import importlib.metadata as m, json; print(json.dumps({k: m.version(k) "
    "for k in ('plantcv', 'opencv-python', 'numpy', 'scikit-image')}))"
)


def check_versions() -> dict[str, dict[str, str]]:
    """The alt arm's Python and the server's uvx environment have the pinned
    PlantCV and OpenCV, and the same NumPy and scikit-image; exits otherwise."""
    alt = subprocess.run(
        [str(ALT_VENV / "bin" / "python"), "-c", VERSION_PROBE],
        env=clean_env("alt"), capture_output=True, text=True, check=False,
    )  # fmt: skip
    srv = subprocess.run(
        ["uvx", *server_uvx()[:-1], *(["--from", SERVER] if SERVER_SPEC is None else []),
         "python", "-c", VERSION_PROBE.replace("'plantcv',", "'plantcv', 'plantcv-mcp',")],
        env=clean_env("mcp"), capture_output=True, text=True, check=False, timeout=600,
    )  # fmt: skip
    if alt.returncode or srv.returncode:
        sys.exit(f"version probe failed: {alt.stderr[-800:]} {srv.stderr[-800:]}")
    got = {"alt": json.loads(alt.stdout), "mcp": json.loads(srv.stdout)}
    want = {**VERSIONS}
    wrong = [
        f"{arm} {k}={v.get(k)} (want {w})"
        for arm, v in got.items()
        for k, w in want.items()
        if v.get(k) != w
    ]
    wrong += [
        f"{k}: alt {got['alt'][k]} vs mcp {got['mcp'][k]}"
        for k in ("numpy", "scikit-image")
        if got["alt"][k] != got["mcp"][k]
    ]
    if got["mcp"].get("plantcv-mcp") != expected_server_version():
        wrong.append(f"server plantcv-mcp={got['mcp'].get('plantcv-mcp')}")
    if wrong:
        sys.exit("library versions differ from the pins: " + "; ".join(wrong))
    return got


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


def load_set(path: Path) -> list[dict[str, Any]]:
    """Read a questions file and set what it governs: where its images are,
    whether the alt arm has the web, and (for a named set) the transcripts'
    directory, so two sets' runs never share one."""
    global IMAGES, RUNS
    doc = json.loads(path.read_text())
    images = doc.get("images_dir")
    IMAGES = Path(images).expanduser() if images else HERE / "images"
    set_alt_web(doc.get("alt_web", True))
    if doc.get("set"):
        RUNS = RUNS.parent / f"{RUNS.name}-{doc['set']}"
    return doc["questions"]


def dry_run(questions: list[dict[str, Any]], arms: list[str]) -> None:
    """Everything a sweep would send, with no model call: each question's
    work directory is built and checked (image and extra files against their
    sha256), its prompt rendered, and each arm's claude flags printed."""
    for arm in arms:
        with tempfile.TemporaryDirectory(prefix="agentqa-dry-") as d:
            flags = arm_flags(arm, Path(d) / "work")
        print(
            f"== arm {arm}: required tools {sorted(REQUIRED_TOOLS[arm])}; flags {flags}"
        )
    for q in questions:
        with tempfile.TemporaryDirectory(prefix="agentqa-dry-") as d:
            tmp = Path(d) / "work"
            tmp.mkdir()
            image = prepare_workdir(q, tmp)
            files = sorted(
                str(p.relative_to(tmp)) for p in tmp.rglob("*") if p.is_file()
            )
            for arm in arms:
                print(f"-- {q['id']} [{arm}] work dir: {len(files)} files")
                print(render_prompt(arm, q, image))
    print(f"dry run: {len(questions)} questions x {len(arms)} arms, no model called")


def main() -> None:
    global PAUSES, STOP_AT
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bare,alt,mcp")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="", help="comma-separated question ids")
    ap.add_argument(
        "--questions",
        default=str(HERE / "questions.json"),
        help="the question set (default the rendered set, questions.json)",
    )
    ap.add_argument(
        "--results",
        default=None,
        help="results.jsonl (default: beside the questions file)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build every work dir, render every prompt, print each arm's tools; "
        "call no model",
    )
    ap.add_argument(
        "--make-alt-venv",
        action="store_true",
        help="build the alt arm's venv at $AGENTQA_ALT_VENV, run nothing",
    )
    ap.add_argument(
        "--audit",
        action="store_true",
        help="re-assess validity and leak markers of every stored row, run nothing",
    )
    ap.add_argument(
        "--rescore",
        action="store_true",
        help="re-score every stored reply against the current questions.json, run nothing",
    )
    ap.add_argument(
        "--server-spec",
        help=f"run the server from {SPEC_PREFIX}<full sha> instead of {SERVER}",
    )
    ap.add_argument(
        "--exclude-newer",
        help="uvx --exclude-newer for the server environment's dependencies",
    )
    ap.add_argument("--model", default=MODEL, help=f"claude --model (default {MODEL})")
    ap.add_argument(
        "--drop-api-failures",
        action="store_true",
        help="move rows the API ended (api_failure) to results.api_failures.jsonl "
        "so a resume re-runs each once, run nothing",
    )
    ap.add_argument(
        "--stop-at",
        type=float,
        default=STOP_AT,
        help=f"stop the sweep when seven_day usage reaches this (default {STOP_AT})",
    )
    a = ap.parse_args()
    qfile = Path(a.questions).resolve()
    if a.results is None:
        a.results = str(qfile.parent / "results.jsonl")
    PAUSES = Path(a.results).parent / "pauses.jsonl"
    STOP_AT = a.stop_at
    set_model(a.model)
    set_server_spec(a.server_spec, a.exclude_newer)
    questions = load_set(qfile)
    if a.dry_run:
        oracle(questions)
        keep = set(a.only.split(",")) if a.only else None
        dry_run(
            [q for q in questions if keep is None or q["id"] in keep],
            a.arms.split(","),
        )
        return
    if a.make_alt_venv:
        make_alt_venv()
        return
    if a.drop_api_failures:
        drop_api_failures(Path(a.results))
        return
    if a.rescore:
        rescore(Path(a.results), {q["id"]: q for q in questions})
        return
    if a.audit:
        audit(Path(a.results))
        return
    oracle(questions)
    if "alt" in a.arms.split(","):
        check_alt_venv()
    if {"alt", "mcp"} & set(a.arms.split(",")):
        versions = check_versions()
        print("versions:", json.dumps(versions), flush=True)
        # What this sweep ran, beside its results. Written at every start, so
        # a resumed sweep that resolved anything differently shows it in git.
        Path(a.results).parent.mkdir(parents=True, exist_ok=True)
        (Path(a.results).parent / "versions.json").write_text(
            json.dumps(
                {
                    "server_spec": SERVER_SPEC or SERVER,
                    "server_uvx": server_uvx(),
                    "exclude_newer": EXCLUDE_NEWER,
                    "model": MODEL,
                    "versions": versions,
                },
                indent=1,
            )
            + "\n"
        )
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
            if r is None:  # not started: the sweep was stopped
                continue
            with lock, results.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(
                f"[{n}/{len(todo)}] {r['arm']:4} {r['qid']} rep{r['rep']} {r['verdict']:9}"
                f" exit={r['exit']} {r['seconds']}s calls={len(r['tool_calls'])}"
                f" ${r['cost_usd']} {r['value']!r}"
                + ("" if r["valid"] else f" INVALID ({r['invalid_reason']})"),
                flush=True,
            )
    if results.exists() and (results.parent / "versions.json").exists():
        record_models_served(results)
    if failures:
        sys.exit(f"{len(failures)} runs stopped by a harness error: {failures[:3]}")
    if STOP.is_set():
        sys.exit(
            "stopped by the usage or spend gate (see the STOP line above); resume after it clears"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
