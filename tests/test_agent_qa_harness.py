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
