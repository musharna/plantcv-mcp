# Worker-process isolation (0.8.0)

**Parent:** `2026-08-27-backlog-integration-plan-of-attack.md`, sub-project C.

## Why

Two facts from 0.5.0–0.7.0: (1) a native crash inside PlantCV/OpenCV takes the whole
stdio server down — 0.5.0 closed one such SIGSEGV by _validation_, which only covers
inputs someone thought of; (2) PlantCV keeps process-global state (`pcv.outputs`,
`pcv.params.saved_color_scale`, `pcv.params.sample_label`) that 0.5.0 and 0.7.0
guard with one lock and explicit resets — again only the globals someone found.
Running every PlantCV analysis in a **worker process** turns both from "found so
far" into structural: a crash becomes a `ToolError` and the server answers the next
call; the worker's globals are nobody else's.

"Drop `pcv.outputs` reliance" (the original backlog wording) is infeasible: 15 of 19
morphology functions report only through it.

## Shape

- `workers.py`: `run_isolated(fn_name, *args, **kwargs)` dispatches to a registry of
  analysis callables (`measure_traits`, `regions.measure_regions`, `measure_morphology`,
  `batch` per-image analysis) executed in a `multiprocessing` **spawn** context
  (the server has anyio worker threads; forking a threaded process is unsafe).
- One warm worker, lazily started, recycled after `WORKER_MAX_TASKS` calls to bound
  memory; arguments and results pickled (numpy arrays). Worker death (signal, exit)
  → `WorkerCrashedError("analysis worker died with signal N ...; the server is still
running; re-run segment()")`, the worker is restarted on the next call.
- Exceptions raised _inside_ the worker are re-raised in the server with their
  original type where possible (pickled), so `MorphologyRefusedError` etc. keep their
  meaning at the tool layer.
- A lock around the single worker serialises analyses (as today's `PCV_OUTPUTS_LOCK`
  does); the in-process lock stays for the non-isolated mode.
- **Mode:** `PLANTCV_MCP_ISOLATE=1` or `plantcv-mcp --isolate`. Default decided by the
  overhead gate below.

## Overhead gate (measured, then decided)

Round-trip overhead on the 3000×3000 fixture (`_write_huge_green_png`) for
`measure()`; if the isolated path costs > 25% more wall time than in-process, ship
**opt-in** (default off) and record the number; else default on.

## Tests

- `os.abort()` injected in the worker via a test-only registry entry → the tool call
  raises `WorkerCrashedError`/`ToolError`, and the **next** call on the same server
  measures normally (positive control in the same test).
- Results identical between isolated and in-process modes for `measure`,
  `measure_regions`, `measure_morphology` (same fixtures as their own tests).
- Worker exceptions keep their type (`MorphologyRefusedError` through the worker).
- Concurrency: `test_concurrency.py` cases pass in isolated mode.
- Fresh-process tool-layer run with `PLANTCV_MCP_ISOLATE=1`.
