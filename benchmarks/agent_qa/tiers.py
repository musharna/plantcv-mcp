"""Analyses for the model-tier re-runs (PREREG_2026-10-20_model-tiers.md)
that plot.R does not do itself.

    python tiers.py substitute TIER_RESULTS OPUS_RESULTS OUTDIR
        OUTDIR/results.jsonl = the tier's mcp rows + the Opus run's alt rows,
        plus baselines.jsonl, so `Rscript plot.R OUTDIR/results.jsonl OUTDIR`
        computes the same paired test and CI for "tier + server vs Opus + Bash".
        The two sets of rows were not run together (non-concurrent).

    python tiers.py strict RESULTS
        Sensitivity: the primary count with every invalid run counted as not
        right (plot.R leaves invalid runs out). Exact two-sided McNemar.
"""

from __future__ import annotations

import json
import shutil
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def substitute(tier: Path, opus: Path, outdir: Path) -> None:
    t = [r for r in rows(tier) if r["arm"] == "mcp"]
    o = [r for r in rows(opus) if r["arm"] == "alt"]
    models = {r["model"] for r in t}, {r["model"] for r in o}
    if models[1] != {"claude-opus-5-5"} or "claude-opus-5-5" in models[0]:
        sys.exit(f"want tier mcp rows vs Opus alt rows, got {models}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in t + o))
    shutil.copyfile(HERE / "baselines.jsonl", outdir / "baselines.jsonl")
    print(f"{len(t)} mcp rows ({models[0]}) + {len(o)} alt rows ({models[1]})")


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def strict(path: Path) -> None:
    rs = rows(path)
    right: dict[tuple[str, str], bool] = {}
    for r in rs:
        k = (r["arm"], r["qid"])
        right[k] = right.get(k, True) and r["valid"] and r["verdict"] == "right"
    qs = sorted({r["qid"] for r in rs})
    m = [right[("mcp", q)] for q in qs]
    a = [right[("alt", q)] for q in qs]
    only_m = sum(x and not y for x, y in zip(m, a, strict=True))
    only_a = sum(y and not x for x, y in zip(m, a, strict=True))
    print(
        "subset\tquestions\tmcp_all_right\talt_all_right\tonly_mcp\tonly_alt\texact_mcnemar_p"
    )
    print(f"invalid_as_not_right\t{len(qs)}\t{sum(m)}\t{sum(a)}\t{only_m}\t{only_a}\t"
          f"{mcnemar(only_m, only_a):.3g}")  # fmt: skip


if __name__ == "__main__":
    if sys.argv[1:2] == ["substitute"]:
        substitute(*map(Path, sys.argv[2:5]))
    elif sys.argv[1:2] == ["strict"]:
        strict(Path(sys.argv[2]))
    else:
        sys.exit(__doc__)
