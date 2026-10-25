"""The field set (benchmarks/agent_qa/field_2026-10-24/): F1 per-plant area in
px and mm², F2 leaf count, from tray 032's hand annotations (Aberystwyth Leaf
Evaluation Dataset, Bell and Dee 2016, CC BY 4.0).

Every test pairs what must pass with what must fail. The tests that read the
dataset itself (the positive control, the work-directory check) need the files
fetch.py puts in ~/plantcv-mcp-artifacts/field_2026-10-24/aberystwyth; without
them they skip, saying so.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

AQ = Path(__file__).resolve().parent.parent / "benchmarks" / "agent_qa"
FIELD = AQ / "field_2026-10-24"
sys.path.insert(0, str(AQ))
sys.path.insert(0, str(FIELD))

import run
from score import area_pair, score, truth_answer

KEY = json.loads((FIELD / "questions.json").read_text())
QS = KEY["questions"]
DATA = Path(KEY["images_dir"]).expanduser()
HAVE_DATA = (DATA / "MANIFEST.json").exists()
needs_data = pytest.mark.skipif(
    not HAVE_DATA, reason=f"dataset not fetched to {DATA} (run fetch.py)"
)


def _v(q: dict, reply: str) -> str:
    return score(q, reply)["verdict"]


@pytest.fixture
def field_set():
    """run.py with the field set loaded; the module state is put back after."""
    saved = (run.IMAGES, run.RUNS, run.ALT_WEB)
    run.load_set(FIELD / "questions.json")
    yield
    run.IMAGES, run.RUNS = saved[0], saved[1]
    run.set_alt_web(saved[2])


# ------------------------------------------------------------- the frozen key
def test_the_key_is_ten_per_family_from_tray_032_only() -> None:
    fam = {f: [q for q in QS if q["family"] == f] for f in ("F1", "F2")}
    assert {f: len(v) for f, v in fam.items()} == {"F1": 10, "F2": 10}
    assert len({q["id"] for q in QS}) == 20
    for f, v in fam.items():
        # 10 distinct images per family, spread over the dates
        assert len({q["image"] for q in v}) == 10, f
    for q in QS:
        assert q["image"].startswith("PSI_Tray032/tv/PSI_Tray032_"), q["id"]
        assert "Tray031" not in json.dumps(q)
        assert len(q["sha256"]) == 64
    assert KEY["alt_web"] is False
    assert KEY["source"]["doi"] == "10.5281/zenodo.168158"


def test_each_f1_window_holds_its_truth_and_every_lens_model() -> None:
    for q in (q for q in QS if q["family"] == "F1"):
        t = q["truth"]
        assert t["px_lo"] < t["area_px"] < t["px_hi"], q["id"]
        assert t["mm2_lo"] < t["area_mm2"] < t["mm2_hi"], q["id"]
        for lens, v in t["mm2_by_lens"].items():
            assert t["mm2_lo"] <= v <= t["mm2_hi"], (q["id"], lens)


# ------------------------------------------------------------------ scoring
F1 = {
    "category": "tray_plant_area",
    "truth": {"area_px": 1000, "px_lo": 900, "px_hi": 1100,
              "area_mm2": 20.0, "mm2_lo": 18.0, "mm2_hi": 22.0},
}  # fmt: skip


def test_f1_is_right_only_when_both_numbers_are_inside_their_windows() -> None:
    assert _v(F1, "ANSWER: 1000 px, 20.0 mm²") == "right"
    assert _v(F1, "ANSWER: 20.0 mm², 1000 px") == "right"  # labels beat order
    assert _v(F1, "ANSWER: 0.2 cm², 1000 pixels") == "right"  # cm² converted
    assert _v(F1, "ANSWER: 1000 px, 20 mm^2") == "right"  # the 2 of ^2 is no number
    got = score(F1, "ANSWER: 1200 px, 20.0 mm²")
    assert got["verdict"] == "wrong" and not got["px_right"] and got["mm2_right"]
    got = score(F1, "ANSWER: 1000 px, 25 mm²")
    assert got["verdict"] == "wrong" and got["px_right"] and not got["mm2_right"]
    assert _v(F1, "ANSWER: 1000 px") == "wrong"  # one number is not two
    assert _v(F1, "ANSWER: UNKNOWN") == "declined"


def test_area_pair_reads_units_and_order() -> None:
    assert area_pair("12,345 px, 678.9 mm²") == (12345.0, 678.9)
    assert area_pair("12345, 678.9") == (12345.0, 678.9)
    assert area_pair("12345 px; 6.789 cm²") == (12345.0, 678.9)
    assert area_pair("678.9 mm2, 12345 px") == (12345.0, 678.9)


def test_f2_is_exact_and_keeps_the_difference_in_count() -> None:
    q = {"category": "tray_leaf_count", "truth": {"count": 12}}
    assert score(q, "ANSWER: 12") == {
        "verdict": "right", "value": "12", "number": 12.0, "slip": None, "dic": 0.0,
    }  # fmt: skip
    got = score(q, "ANSWER: 13 leaves")
    assert got["verdict"] == "wrong" and got["dic"] == 1.0
    assert score(q, "ANSWER: 10")["dic"] == -2.0


def test_oracle_every_key_answer_is_right_and_a_shifted_one_wrong() -> None:
    assert [_v(q, f"ANSWER: {truth_answer(q)}") for q in QS] == ["right"] * 20
    wrong = []
    for q in QS:
        t = q["truth"]
        if q["family"] == "F2":
            wrong.append(_v(q, f"ANSWER: {t['count'] + 1}"))
        else:
            wrong.append(_v(q, f"ANSWER: {t['px_hi'] + 1} px, {t['area_mm2']} mm²"))
            wrong.append(_v(q, f"ANSWER: {t['area_px']} px, {t['mm2_hi'] * 1.01} mm²"))
    assert wrong == ["wrong"] * 30


# ------------------------------------------------------------ shortcut controls
def test_shortcut_uncorrected_mm2_fails_only_where_the_distortion_outgrows_the_band() -> (
    None
):
    """The uncorrected conversion (raw px over the raw interior pots' px/mm,
    squared) under-reads edge plants by a median 3.7% of area; it lands
    outside the key's window only where the plant is large enough that its
    +-1 px band is narrower than that. Measured on all 171 eligible plants:
    3 fail, all at the tray edge, none interior. Of the 10 sampled, 1 fails
    (f1_10, 223,448 px at row 1 column 5)."""
    c = KEY["controls"]["shortcut_uncorrected_all_eligible"][
        "uncorrected_px_per_mm_interior"
    ]
    assert (c["fails"], c["of"], c["fails_edge"], c["fails_interior"]) == (3, 171, 3, 0)
    assert (
        c["rel_error_edge_median"]
        < -0.03
        < 0.0
        < c["rel_error_interior_range"][1] + 0.01
    )
    fails = []
    for q in (q for q in QS if q["family"] == "F1"):
        t = q["truth"]
        shortcut = f"ANSWER: {t['area_px']} px, {t['uncorrected_mm2']['interior']} mm²"
        corrected = f"ANSWER: {t['area_px']} px, {t['area_mm2']} mm²"
        assert _v(q, corrected) == "right", q["id"]  # same px, corrected: right
        if _v(q, shortcut) == "wrong":
            fails.append(q["id"])
    assert fails == ["f1_10"]
    assert QS[9]["id"] == "f1_10" and QS[9]["where"]["edge"]


def test_shortcut_date_median_leaf_count_is_reported() -> None:
    """Not a shortcut an arm can take (it needs the key), but the floor a
    good guess from the plant's age reaches: the median count of that date's
    plants is exact on 60 of 171 eligible plants and within one on 121."""
    c = KEY["controls"]["shortcut_date_median_leaf_count"]
    assert (c["exact"], c["within_one"], c["of"]) == (60, 121, 171)
    f2 = [q for q in QS if q["family"] == "F2"]
    exact = sum(q["truth"]["date_median"] == q["truth"]["count"] for q in f2)
    assert exact == 6


def test_retention_accounts_for_every_annotated_plant() -> None:
    c = KEY["controls"]
    assert c["annotated_plants"] == 210  # dataset PDF, Table 3
    rows = c["retention"]
    assert sum(r["plants"] for r in rows) == 210
    assert sum(r["kept"] for r in rows) == c["eligible_plants"] == 171
    assert all(
        r["kept"] + r["touching"] + r["not_found"] + r["stray_group"] == r["plants"]
        for r in rows
    )
    assert sum(r["touching"] for r in rows) == 39


# ------------------------------------------------------------- harness wiring
def test_the_field_set_turns_the_alt_arms_web_off(field_set, tmp_path) -> None:
    flags = run.arm_flags("alt", tmp_path / "work")
    tools = flags[flags.index("--tools") + 1].split(",")
    allowed = flags[flags.index("--allowedTools") + 1].split(",")
    assert tools == allowed == ["Bash", "Read"]
    calls = [("Bash", {"command": "python3 x.py"})]
    ok = _transcript(["Bash", "Read"], calls)
    webbed = _transcript(["Bash", "Read", "WebSearch", "WebFetch"], calls)
    assert run.assess("alt", ok, 0)["valid"]
    got = run.assess("alt", webbed, 0)
    assert not got["valid"] and "web tools present" in got["invalid_reason"]
    # the rendered set keeps its web, and its alt prompt note is unchanged
    run.set_alt_web(True)
    flags = run.arm_flags("alt", tmp_path / "work2")
    assert flags[flags.index("--tools") + 1] == "Bash,Read,WebSearch,WebFetch"
    assert run.ARM_NOTE["alt"] == (
        "\n\nThe python3 on your PATH has PlantCV 4.11.3 and OpenCV 4.11 installed."
    )


def test_dataset_urls_are_leak_markers_but_a_pixel_count_is_not(field_set) -> None:
    clean = [("Bash", {"command": "python3 measure.py"})]
    assert run.assess(
        "alt", _transcript(["Bash", "Read"], clean, ["area 168158 px"]), 0
    )["valid"]
    for leak in (
        "curl https://zenodo.org/records/168158",
        "curl https://ZENODO.org/api/records/168158/files",
        "wget https://www.plant-phenotyping.org/datasets",
        "find / -name '*_top_gt.png'",
        "ls PSI_Tray032",
    ):
        got = run.assess(
            "alt", _transcript(["Bash", "Read"], [("Bash", {"command": leak})]), 0
        )
        assert not got["valid"] and got["invalid_reason"].startswith("leak"), leak


def _transcript(tools, calls=(), results=(), reply="ANSWER: 7") -> str:
    events = [{"type": "system", "subtype": "init", "tools": tools, "mcp_servers": []}]
    for name, inp in calls:
        events.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": name, "input": inp}]}})  # fmt: skip
    for text in results:
        events.append({"type": "user", "message": {"content": [
            {"type": "tool_result", "content": [{"type": "text", "text": text}]}]}})  # fmt: skip
    events.append({"type": "result", "result": reply, "total_cost_usd": 0.01})
    return "\n".join(json.dumps(e) for e in events)


# --------------------------------------------- tests that read the dataset
@needs_data
def test_positive_control_a_reference_without_the_server_passes_every_question() -> (
    None
):
    """reference.py: the authors' shipped calibration, a resampled mask and
    its own pot scale (not the builder's route), and no plantcv-mcp."""
    import reference

    replies = {q["id"]: reference.answer(q, DATA) for q in QS}
    verdicts = {i: _v(q, replies[i]) for q in QS for i in [q["id"]]}
    assert verdicts == {q["id"]: "right" for q in QS}, replies
    for f in ("reference.py", "field_lib.py", "build.py", "fetch.py"):
        assert "plantcv_mcp" not in (FIELD / f).read_text(), f


@needs_data
def test_every_work_dir_is_built_from_files_matching_the_key(
    field_set, tmp_path
) -> None:
    """Real files, real hashes: the image and (F1) the 23 checkerboard photos
    copied into an empty work dir and checked; a tampered board is refused."""
    q = next(q for q in QS if q["family"] == "F1")
    work = tmp_path / "work"
    work.mkdir()
    image = run.prepare_workdir(q, work)
    assert image.name == "image.png"
    boards = {f"checkerboards/{p.name}" for p in (work / "checkerboards").iterdir()}
    assert boards == set(q["extra_sha256"]) and len(boards) == 23
    bad = dict(q, extra_sha256={**q["extra_sha256"], "checkerboards/0.png": "0" * 64})
    work2 = tmp_path / "work2"
    work2.mkdir()
    with pytest.raises(run.HarnessError, match="extra files differ"):
        run.prepare_workdir(bad, work2)


@needs_data
def test_dry_run_renders_every_prompt_and_calls_no_model() -> None:
    p = subprocess.run(
        [sys.executable, str(AQ / "run.py"), "--questions", str(FIELD / "questions.json"),
         "--arms", "alt", "--dry-run"],
        capture_output=True, text=True, timeout=600, check=False,
        env={**run.os.environ, "PATH": "/usr/bin:/bin"},  # no claude on PATH
    )  # fmt: skip
    assert p.returncode == 0, p.stderr[-2000:]
    assert "flags ['--tools', 'Bash,Read', '--allowedTools', 'Bash,Read'" in p.stdout
    assert p.stdout.count("ANSWER: <answer>") == 20
    assert "dry run: 20 questions x 1 arms, no model called" in p.stdout
