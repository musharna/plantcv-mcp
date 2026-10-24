"""The agent benchmark's validity rule: one rule for every setup, decided from
the transcript. Each test pairs a run that must count with one that must not,
so a rule that passed (or failed) everything fails here."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "benchmarks" / "agent_qa")
)

from run import assess

SERVER_TOOLS = ["mcp__plantcv__segment", "mcp__plantcv__measure"]


def _transcript(tools, servers=None, calls=(), results=(), reply="ANSWER: 7") -> str:
    events = [
        {
            "type": "system",
            "subtype": "init",
            "tools": tools,
            "mcp_servers": servers or [],
        }
    ]
    for name, inp in calls:
        events.append(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "tool_use", "name": name, "input": inp}]
                },
            }
        )
    for text in results:
        events.append(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": [{"type": "text", "text": text}],
                        }
                    ]
                },
            }
        )
    events.append({"type": "result", "result": reply, "total_cost_usd": 0.01})
    return "\n".join(json.dumps(e) for e in events)


MCP_OK = {
    "tools": ["Read", *SERVER_TOOLS],
    "servers": [{"name": "plantcv", "status": "connected"}],
    "calls": [
        ("mcp__plantcv__segment", {"image_path": "/tmp/agentqa-x/work/image.png"})
    ],
}


def test_a_server_run_counts_only_with_the_server_connected_and_read_present() -> None:
    assert assess("mcp", _transcript(**MCP_OK), 0)["valid"]
    pending = dict(MCP_OK, servers=[{"name": "plantcv", "status": "pending"}])
    assert not assess("mcp", _transcript(**pending), 0)["valid"]
    no_read = dict(MCP_OK, tools=SERVER_TOOLS)
    got = assess("mcp", _transcript(**no_read), 0)
    assert not got["valid"] and "Read" in got["invalid_reason"]


def test_the_same_exit_and_tool_call_rules_hold_for_every_setup() -> None:
    alt_ok = {
        "tools": ["Bash", "Read", "WebSearch", "WebFetch"],
        "calls": [("Read", {})],
    }
    assert assess("alt", _transcript(**alt_ok), 0)["valid"]
    for arm, ok in (("alt", alt_ok), ("mcp", MCP_OK)):
        assert not assess(arm, _transcript(**ok), 1)["valid"]  # exit 1
        assert not assess(arm, _transcript(**ok), "timeout")["valid"]
        assert not assess(arm, _transcript(**dict(ok, calls=())), 0)[
            "valid"
        ]  # never looked
    assert assess("bare", _transcript(tools=[]), 0)["valid"]
    assert not assess("bare", _transcript(tools=[]), 1)["valid"]
    assert not assess("bare", _transcript(tools=["Bash"]), 0)["valid"]


def test_a_leak_marker_in_a_tool_input_or_output_makes_the_run_invalid() -> None:
    url = "https://raw.githubusercontent.com/musharna/plantcv-mcp/x/benchmarks/agent_qa/questions.json"
    by_input = dict(MCP_OK, calls=[*MCP_OK["calls"], ("WebFetch", {"url": url})])
    got = assess("mcp", _transcript(**by_input), 0)
    assert not got["valid"] and "questions.json" in got["leaks"]
    by_output = dict(
        MCP_OK, results=["found ~/x/bench-1016/benchmarks/agent_qa/questions.json"]
    )
    assert not assess("mcp", _transcript(**by_output), 0)["valid"]
    clean = dict(MCP_OK, results=["area 15795 px, image /tmp/agentqa-x/work/image.png"])
    assert assess("mcp", _transcript(**clean), 0)["leaks"] == []


def test_server_spec_is_a_pinned_commit_and_reaches_the_uvx_command() -> None:
    """--server-spec runs an unreleased commit: only a full SHA on this repo's
    git URL is accepted (a branch name or short SHA can move under a sweep),
    and the accepted spec, with --exclude-newer, is what uvx starts."""
    import pytest
    import run

    sha = "0123456789abcdef0123456789abcdef01234567"
    try:
        for bad in (
            f"{run.SPEC_PREFIX}master",
            f"{run.SPEC_PREFIX}{sha[:12]}",
            f"git+https://github.com/someone/plantcv-mcp@{sha}",
        ):
            with pytest.raises(SystemExit, match="full 40-hex sha"):
                run.set_server_spec(bad, None)
        run.set_server_spec(f"{run.SPEC_PREFIX}{sha}", "2026-10-17")
        argv = run.server_uvx()
        assert argv[:2] == ["--exclude-newer", "2026-10-17"]
        assert argv[-3:] == ["--from", f"{run.SPEC_PREFIX}{sha}", "plantcv-mcp"]
        assert run.SERVER not in argv
        run.set_server_spec(None, None)
        assert run.server_uvx() == run.SERVER_UVX  # the 10-17 command, unchanged
    finally:
        run.set_server_spec(None, None)


def test_tool_argument_names_are_recorded_without_their_values() -> None:
    """The mechanism readout counts runs that used a new parameter, so the
    argument names must be in the row; the values (image paths) must not."""
    call = ("mcp__plantcv__segment", {"image_path": "/x/image.png", "rules": []})
    row = assess("mcp", _transcript(**dict(MCP_OK, calls=[call])), 0)
    assert row["valid"]
    assert row["tool_arg_names"] == [["image_path", "rules"]]
    assert "/x/image.png" not in json.dumps(row["tool_arg_names"])


def test_model_flag_keeps_the_default_and_routes_other_models_to_their_own_runs() -> (
    None
):
    """--model runs every arm on another model; its transcripts must not
    overwrite the default model's runs/."""
    import run

    assert run.DEFAULT_MODEL == "claude-opus-5-5"
    try:
        run.set_model("claude-haiku-5-5")
        assert run.MODEL == "claude-haiku-5-5"
        assert run.RUNS == run.ARTIFACTS / "runs-claude-haiku-5-5"
        run.set_model(run.DEFAULT_MODEL)
        assert run.RUNS == run.ARTIFACTS / "runs"
    finally:
        run.set_model(run.DEFAULT_MODEL)


def _rl_event(status: str, five: float, seven: float) -> dict:
    return {
        "type": "rate_limit_event",
        "rate_limit_info": {
            "status": status,
            "resetsAt": 2000,
            "unifiedWindows": {
                "five_hour": {"utilization": five, "resetsAt": 1000},
                "seven_day": {"utilization": seven, "resetsAt": 2000},
            },
        },
    }


def test_only_a_rejection_before_any_answer_is_an_api_failure() -> None:
    """A rate-limit rejection that ended the run before the model answered is
    a harness failure (re-run once); one after an answer, or no rejection, is
    a run like any other."""
    import run

    answer = {"type": "assistant", "message": {"content": []}}
    rejected = _rl_event("rejected", 1.0, 0.5)
    assert run.api_failure([rejected])
    assert not run.api_failure([answer, rejected])
    assert not run.api_failure([_rl_event("allowed", 0.2, 0.5)])
    got = run.rate_limit([_rl_event("allowed", 0.2, 0.5), rejected])
    assert got == {"status": "rejected", "five_hour": 1.0, "seven_day": 0.5,
                   "resets_at": 1000}  # fmt: skip


def test_usage_windows_pause_and_stop_the_sweep(tmp_path, monkeypatch) -> None:
    """five_hour at 0.85 sleeps until its reset and logs the pause; below, no
    pause. seven_day at 0.80 stops the sweep: a run not yet begun returns None. A pause holds every
    worker: a run another worker starts waits for the reset too."""
    import run

    slept: list[float] = []
    monkeypatch.setattr(run.time, "sleep", slept.append)
    monkeypatch.setattr(run.time, "time", lambda: 400.0)
    monkeypatch.setattr(run, "PAUSES", tmp_path / "pauses.jsonl")
    monkeypatch.setattr(run, "PAUSE_UNTIL", 0.0)
    try:
        run.pause_if_limited(run.rate_limit([_rl_event("allowed", 0.84, 0.79)]), "a")
        assert slept == [] and not run.STOP.is_set()
        run.wait_for_pause()
        assert slept == []  # no pause set: a run starts at once
        run.pause_if_limited(run.rate_limit([_rl_event("allowed", 0.85, 0.79)]), "b")
        assert slept == [600 + 120] and not run.STOP.is_set()
        assert json.loads((tmp_path / "pauses.jsonl").read_text())["after"] == "b"
        run.pause_if_limited(run.rate_limit([_rl_event("allowed", 0.1, 0.80)]), "c")
        assert run.STOP.is_set() and len(slept) == 1
        assert run.run_one("bare", {"id": "q01"}, 1) is None
        assert slept == [720]  # a stopped sweep does not sleep out the pause
    finally:
        run.STOP.clear()


def test_a_run_waits_for_the_pause_then_asks_the_spend_gate_again(
    tmp_path, monkeypatch
) -> None:
    """A pause holds every worker, and the spend gate is asked again once it
    clears: here the gate passes before the pause and refuses after it, so the
    run sleeps out the pause and then stops without starting."""
    import run

    calls = tmp_path / "calls"
    gate = tmp_path / "gate.py"
    gate.write_text(
        "import pathlib, sys\n"
        f"p = pathlib.Path({str(calls)!r})\n"
        "n = int(p.read_text()) + 1 if p.exists() else 1\n"
        "p.write_text(str(n))\n"
        "print('cap reached', file=sys.stderr)\n"
        "sys.exit(0 if n == 1 else 3)\n"
    )
    slept: list[float] = []
    monkeypatch.setattr(run.time, "sleep", slept.append)
    monkeypatch.setattr(run.time, "time", lambda: 400.0)
    monkeypatch.setattr(run, "PAUSE_UNTIL", 1120.0)
    monkeypatch.setenv("BENCH_SPEND_GATE", str(gate))
    try:
        assert run.run_one("bare", {"id": "q01"}, 1) is None
        assert slept == [720]  # the run waited for the pause first
        assert calls.read_text() == "2"  # and asked the gate again after it
        assert run.STOP.is_set()
    finally:
        run.STOP.clear()


def test_spend_gate_stops_the_sweep_before_a_run(tmp_path, monkeypatch) -> None:
    """API-billed sweeps (2026-10-23): run_one asks BENCH_SPEND_GATE first; a
    non-zero exit stops the sweep before the run starts. Under the cap, or
    with the gate unset, nothing stops."""
    import run

    gate = tmp_path / "gate.py"
    gate.write_text(
        "import os, sys\nprint('cap reached', file=sys.stderr)\nsys.exit(int(os.environ['GATE_RC']))\n"
    )
    monkeypatch.setattr(run, "PAUSE_UNTIL", 0.0)
    monkeypatch.setenv("BENCH_SPEND_GATE", str(gate))
    try:
        monkeypatch.setenv("GATE_RC", "0")
        assert run.spend_stop() is None  # under the cap
        monkeypatch.setenv("GATE_RC", "3")
        assert run.spend_stop() == "cap reached"
        assert run.run_one("bare", {"id": "q01"}, 1) is None
        assert run.STOP.is_set()
        monkeypatch.delenv("BENCH_SPEND_GATE")
        assert run.spend_stop() is None  # unset: subscription sweeps unchanged
    finally:
        run.STOP.clear()
