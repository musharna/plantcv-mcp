# Pre-registration: model tiers, 2026-10-20

Committed and pushed before the Haiku sweep. Anything run after a tier's sweep
that is not listed here is exploratory and is labeled so.

**Why.** On Opus 5.5 the server tied Claude's own code (10-18: questions right
in every run 42 vs 44, p = 0.63). Opus writes PlantCV/OpenCV code itself, so
accuracy may be at a ceiling. The question here: does the server lift a
weaker model's accuracy?

**Tiers.** Claude Haiku 5.5 (`claude-haiku-5-5`) now; Claude Sonnet 5.5 later,
with this same design and analysis. Both are confirmatory.

**Server under test.** plantcv-mcp 1.17.1, the current PyPI release, run as
`git+https://github.com/musharna/plantcv-mcp@30e12b37eda71cb8600ba5f130d4853688f6a508`
(tag v1.17.1) with `--exclude-newer 2026-10-17`, as the 10-18 re-run pinned
it. The git spec is needed because the PyPI upload (2026-10-20) is newer than
the date pin, which keeps NumPy 2.4.6 and scikit-image 0.26.0 equal to the alt
arm's (checked by `run.py` before the sweep). 1.17.1 is the 10-18 commit
051aff0 plus the fix that names tools with their client prefix (374653a).

**Arms**, all three re-run on the tier's model, interleaved by question (for
each rep, each question runs bare, alt, mcp). No Opus row is reused in the
primary comparison.

- `bare`: no tools (the floor; the right reply is UNKNOWN).
- `alt`: Bash/PlantCV/OpenCV + Read + web.
- `mcp`: this server + Read.

Prompt, isolation, tools, questions (46, images checked by sha256), scorer
(`score.py`) and validity rule (`run.py assess()`) are unchanged from 10-18.
3 reps: 414 runs per tier. `--workers 2`: each run has its own working
directory, TMPDIR and server process, so rows stay valid; seconds per answer
come from runs sharing the machine with one other run (10-18 used 1 worker),
so time against Opus is not comparable.

**Primary endpoint per tier.** Questions right in every valid rep (with at
least one valid rep), mcp vs alt, paired over questions: exact two-sided
McNemar and the 95% percentile bootstrap CI of the paired difference (4,000
resamples, plot.R's seed). This is `stats_paired.tsv`, subset `all`, as on
10-18.

**Win rule** (per server, across tiers). Holm over the 2 tiers' McNemar p.

- **Win**: a significant lift (Holm-adjusted p < 0.05, mcp ahead) at ≥ 1
  tier, AND no significant loss (Holm-adjusted p < 0.05, alt ahead) at the
  other.
- **Loss**: a significant loss at either tier and no win.
- **Tie**: otherwise.

The verdict waits for Sonnet; after Haiku only the raw p is reported.
**Power:** a lift needs at least 6 discordant questions, all for mcp
(6-0, p = 0.031), before Holm; at Holm's first step p must be < 0.025
(7-0, p = 0.016).

**Secondary.**

- **Substitute claim** (non-concurrent): tier + server vs Opus + Bash, the
  tier's mcp rows against the 10-18 Opus alt rows
  (`rerun_2026-10-18/results.jsonl`), same statistic, by `tiers.py
  substitute` + `plot.R`. It holds if the tier's mcp count of questions right
  in every rep is ≥ Opus alt's − 2 (the 10-18 non-inferiority margin). The
  runs were not concurrent; it is labeled so wherever it is reported.
- **Sensitivity:** the primary count with invalid runs counted as not right
  (`tiers.py strict`), since a weaker model may skip the tools.
- **Mechanism:** server calls per valid mcp run, mcp runs with 0 server calls
  ("mechanism did not fire"), bare-name calls, and the per-tool counts
  (`diagnose.py --mechanism`).
- Share right with CIs (all 46, and the 18 a color cut misses), the bare
  floor, cost and seconds per answer.

**Controls, all active.**

- Oracle before the sweep (run by `run.py`; passed 2026-10-20 on all 46).
  The key is rendered from fixed images, so no live state can move it.
- No-model baselines (`baselines.jsonl`) reused unchanged.
- Validity rule and leak scan in every run; `run.py --audit` after the sweep.
- **Usage gate:** a run whose transcript shows five_hour ≥ 0.85 or a rejected
  request pauses the sweep until the window resets (logged to
  `pauses.jsonl`). seven_day ≥ 0.80 stops the sweep; it resumes after the
  reset.
- **Rate-limit re-run rule:** a row the API ended before any model answer
  (a rejected request and no assistant message, `api_failure`) is a harness
  failure: `run.py --drop-api-failures` moves it aside and the resume re-runs
  it once. A second such failure stays, as invalid.
- The model served is the init event's `model` in every row (with
  `model_requested`), and `versions.json` gets `models_served` after the
  sweep. A row served by another model is reported.

**Pilot before this file** (not part of any result): 9 Haiku runs, q01, q31,
q39 rep 1, all arms, all valid, $0.0002–0.0042 per run; the mcp runs called
the server 2–4 times. Used for feasibility and cost only. Kept in
`~/plantcv-mcp-artifacts/agent_qa/pilot-haiku-2026-10-20/`.

**Sweep** (from `benchmarks/agent_qa`; first `df -h /`, stop under 5 GB).

```bash
AGENTQA_ARTIFACTS=~/plantcv-mcp-artifacts/agent_qa_haiku_1020 \
flock ~/.claude/jobs/9cc8114d/tmp/sweep.lock env HEAVY_RUN_MEM=6G heavy-run \
  uv run --python 3.13 --no-project --with jsonschema python run.py \
  --model claude-haiku-5-5 --arms bare,alt,mcp --reps 3 --workers 2 \
  --results haiku_2026-10-20/results.jsonl \
  --server-spec git+https://github.com/musharna/plantcv-mcp@30e12b37eda71cb8600ba5f130d4853688f6a508 \
  --exclude-newer 2026-10-17
```

**Analysis** (repo root; `A=~/plantcv-mcp-artifacts/agent_qa_haiku_1020`,
`Q=benchmarks/agent_qa`).

```bash
AGENTQA_ARTIFACTS=$A uv run python $Q/run.py --model claude-haiku-5-5 --audit --results $Q/haiku_2026-10-20/results.jsonl
Rscript $Q/plot.R $Q/haiku_2026-10-20/results.jsonl $Q/haiku_2026-10-20
uv run python $Q/tiers.py strict $Q/haiku_2026-10-20/results.jsonl
uv run python $Q/diagnose.py --mechanism $Q/haiku_2026-10-20/results.jsonl
uv run python $Q/tiers.py substitute $Q/haiku_2026-10-20/results.jsonl $Q/rerun_2026-10-18/results.jsonl $Q/haiku_2026-10-20/substitute
Rscript $Q/plot.R $Q/haiku_2026-10-20/substitute/results.jsonl $Q/haiku_2026-10-20/substitute
```

Every result is reported, whatever it shows.

## Amendment, 2026-10-20 ~19:00 EDT (before any Sonnet run)

Received after this file was pushed and after the Haiku sweep started at
18:50 EDT (no Haiku row had been written). Nothing above changes except:

- **Sonnet runs this session**, right after the Haiku sweep, with the same
  arms, reps, key, controls and analysis (`--model claude-sonnet-5-5`,
  results in `sonnet_2026-10-20/`, transcripts under
  `~/plantcv-mcp-artifacts/agent_qa_sonnet_1020/`). The win/tie/loss verdict
  (Holm over both tiers) is then reported from these two sweeps.
- **Usage stop for the Sonnet sweep: seven_day ≥ 0.90** (`run.py --stop-at
  0.90`; Haiku keeps 0.80). It is read from every run's transcript; if reached,
  the sweep stops, its rows are kept and reported, and the unrun triples are
  listed as not run.
- **Pause fix, ~23:20 EDT, before any Sonnet run** (CI review finding): with
  `--workers 2` the five-hour pause slept only the worker that saw it. It now
  holds every worker before its next run. The Haiku sweep never reached it
  (five_hour at most 0.20, no rejection, no pause).
- **Sonnet deferred** by the coordinator at 22:50 EDT (seven_day 0.84): the
  other servers' Haiku sweeps run first, and Sonnet runs on whatever budget is
  left, still stopping at 0.90.

## Amendment, 2026-10-23 ~19:20 EDT: API billing (before any further run)

- **Billing.** The user's decision: every remaining run in this design (both
  tiers) bills an Anthropic API key, not the Claude subscription. Same model
  IDs, prompts, arms, reps, harness and controls. Claude Code is 2.1.309 now
  (recorded per row).
- **How.** A benchmark-only `claude` wrapper, first on `PATH` for the sweep,
  loads the key and then `exec`s the real CLI (so the harness's timeout still
  kills Claude itself). It records each run's cost (Claude Code's
  `total_cost_usd`; for a killed run, estimated from its token counts) and
  refuses to start a run once the total reaches $171 of the user's $175 cap
  (exit 75, no transcript). A run whose `init` event does not show
  `apiKeySource: ANTHROPIC_API_KEY` stops every later launch.
- **Gate.** API calls report no subscription windows, so the seven-day stop
  and the five-hour pause no longer fire. `run.py` instead asks the spend gate
  (`BENCH_SPEND_GATE`) before every run and, on a refusal, stops cleanly as
  at the seven-day stop: finished rows stay, nothing new starts. Test added,
  seen to fail with the check disabled (`mutate-run`).
- **Set aside by exit code, not outcome.** A run the wrapper refuses (exit
  75) is a harness failure, not a result: it is set aside and re-run once.
- **Concurrency.** Up to 4 servers' sweeps run at once (2 before). The
  workers per sweep are unchanged.
