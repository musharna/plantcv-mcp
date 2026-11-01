"""The F1 confirmatory set (benchmarks/agent_qa/field_2026-10-31_f1confirm/):
30 per-plant area questions from tray 031 of the Aberystwyth Leaf Evaluation
Dataset (Bell and Dee 2016, CC BY 4.0), drawn after the PREREG amendment of
2026-10-31 was pushed.

The key is frozen by its sha256 here. Tests that read the dataset need the
files `fetch.py --tray 031` puts in
~/plantcv-mcp-artifacts/field_2026-10-31_f1confirm/aberystwyth; without them
they skip, saying so.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

AQ = Path(__file__).resolve().parent.parent / "benchmarks" / "agent_qa"
FIELD = AQ / "field_2026-10-24"
SET = AQ / "field_2026-10-31_f1confirm"
sys.path.insert(0, str(AQ))
sys.path.insert(0, str(FIELD))

import run
from score import score, truth_answer

# Committed with the key, before any arm ran (PREREG amendment 2026-10-31).
KEY_SHA256 = "1934e4a30cd46e3c4f889f21144b3f0fd85190eab60f02542de9e90fb14b9bcd"
KEY = json.loads((SET / "questions.json").read_text())
QS = KEY["questions"]
PILOT = json.loads((FIELD / "questions.json").read_text())
DATA = Path(KEY["images_dir"]).expanduser()
needs_data = pytest.mark.skipif(
    not (DATA / "MANIFEST.json").exists(),
    reason=f"dataset not fetched to {DATA} (run fetch.py --tray 031)",
)


def _v(q: dict, reply: str) -> str:
    return score(q, reply)["verdict"]


# ------------------------------------------------------------- the frozen key
def test_the_key_is_the_one_committed_before_any_run() -> None:
    got = hashlib.sha256((SET / "questions.json").read_bytes()).hexdigest()
    assert got == KEY_SHA256


def test_thirty_f1_questions_from_thirty_tray_031_images() -> None:
    assert [q["id"] for q in QS] == [f"f1c_{n:02d}" for n in range(1, 31)]
    assert {q["family"] for q in QS} == {"F1"}
    assert len({q["image"] for q in QS}) == 30
    for q in QS:
        assert q["image"].startswith("PSI_Tray031/tv/PSI_Tray031_"), q["id"]
        assert q["category"] == "tray_plant_area"
    assert KEY["seed"] == {"F1": 20261031}
    assert KEY["source"]["tray"] == "031"
    assert KEY["alt_web"] is False
    # nothing shared with the pilot: other ids, other images
    assert not {q["id"] for q in QS} & {q["id"] for q in PILOT["questions"]}
    assert not {q["image"] for q in QS} & {q["image"] for q in PILOT["questions"]}


def test_each_window_holds_its_truth_and_every_lens_model() -> None:
    for q in QS:
        t = q["truth"]
        assert t["px_lo"] < t["area_px"] < t["px_hi"], q["id"]
        assert t["mm2_lo"] < t["area_mm2"] < t["mm2_hi"], q["id"]
        for lens, v in t["mm2_by_lens"].items():
            assert t["mm2_lo"] <= v <= t["mm2_hi"], (q["id"], lens)


def test_oracle_every_key_answer_is_right_and_a_shifted_one_wrong() -> None:
    assert [_v(q, f"ANSWER: {truth_answer(q)}") for q in QS] == ["right"] * 30
    wrong = []
    for q in QS:
        t = q["truth"]
        wrong.append(_v(q, f"ANSWER: {t['px_hi'] + 1} px, {t['area_mm2']} mm²"))
        wrong.append(_v(q, f"ANSWER: {t['area_px']} px, {t['mm2_hi'] * 1.01} mm²"))
    assert wrong == ["wrong"] * 60


# ------------------------------------------------------ the builder's guards
def test_tray_031_never_serves_leaf_counts_and_another_tray_needs_its_own_set() -> None:
    import build

    with pytest.raises(SystemExit, match="never serves F2"):
        build.configure(["--tray", "031", "--set", "x", "--families", "F1,F2"])
    with pytest.raises(SystemExit, match="its own --set"):
        build.configure(["--tray", "031"])
    with pytest.raises(SystemExit, match="exactly one family"):
        build.configure(["--tray", "032", "--set", "x", "--id-prefix", "f"])
    # positive control: the pilot's flags leave every constant as it was
    before = (build.TRAY, build.SET, build.FAMILIES, build.N, build.OUT)
    build.configure([])
    assert (build.TRAY, build.SET, build.FAMILIES, build.N, build.OUT) == before


def test_build_refuses_a_flag_it_would_ignore_and_an_unknown_family() -> None:
    import build

    # the pilot set's name with another flag: that flag would be dropped
    with pytest.raises(SystemExit, match="its own --set"):
        build.configure(["--families", "F2", "--n", "5"])
    with pytest.raises(SystemExit, match="unknown --families"):
        build.configure(["--tray", "031", "--set", "x", "--families", "F1,F3"])


class _Stop(Exception):
    pass


def test_fetch_reads_the_tray_in_any_order_and_refuses_a_bad_flag(
    monkeypatch, tmp_path, capsys
) -> None:
    """main() is driven for real up to the first archive member it would read;
    the archives themselves are stubbed, so no network is touched."""
    import fetch

    assert fetch.GT_COUNT == {"032": 13, "031": 43}
    asked: list[str] = []

    def wanted(images, software, harvest, tray):
        asked.append(tray)
        raise _Stop

    monkeypatch.setattr(fetch, "open_archive", lambda name: None)
    monkeypatch.setattr(fetch, "wanted", wanted)
    for argv in ([str(tmp_path), "--tray", "031"], ["--tray", "031", str(tmp_path)]):
        with pytest.raises(_Stop):
            fetch.main(argv)
    # positive control: no flag is the pilot's tray
    with pytest.raises(_Stop):
        fetch.main([str(tmp_path)])
    assert asked == ["031", "031", "032"]
    for bad in (["--tray"], ["--tray", "033"], [str(tmp_path), str(tmp_path)]):
        with pytest.raises(SystemExit):
            fetch.main(bad)
    assert "invalid choice: '033'" in capsys.readouterr().err
    assert asked == ["031", "031", "032"]  # no bad flag reached the archive


# --------------------------------------------- tests that read the dataset
@needs_data
def test_positive_control_a_reference_without_the_server_passes_every_question() -> (
    None
):
    import reference

    verdicts = {q["id"]: _v(q, reference.answer(q, DATA)) for q in QS}
    assert verdicts == {q["id"]: "right" for q in QS}


@needs_data
def test_dry_run_renders_every_prompt_and_calls_no_model() -> None:
    p = subprocess.run(
        [sys.executable, str(AQ / "run.py"), "--questions", str(SET / "questions.json"),
         "--arms", "alt", "--dry-run"],
        capture_output=True, text=True, timeout=600, check=False,
        env={**run.os.environ, "PATH": "/usr/bin:/bin"},  # no claude on PATH
    )  # fmt: skip
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.count("ANSWER: <answer>") == 30
    assert "dry run: 30 questions x 1 arms, no model called" in p.stdout
