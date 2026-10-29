# Self-check and run-directory layout

These facts are derived from the harness code named in each section. The module paths are given so a
maintainer can re-check them; the skill's test checks the names against that code.

## Request channel (`.qa_channel/`)

From the broker (`merlin_experiments.phase1.brokers.selfcheck`) and its in-sandbox client
(`agent_selfcheck.py`, `merlin_experiments.phase1.tools.selfcheck`):

| File | Direction | Content |
|---|---|---|
| `req_RID.json` | you to broker | `submission`, `capsules`, `workers`, `timeout`, any probe flags (`shape_coverage`, `offload_census`, `model_layers`, `candidate_diagnostics`, `compile_only`, `scale_probe`, `dynamic_census`), optional `sim`/`tiers` |
| `progress_RID.json` | broker to you | progress while the request runs |
| `resp_RID.json` | broker to you | the redacted self-check JSON, stamped with `selfcheck_request_id`, `submission_sha256` and `selfcheck_submission` |
| `done_RID` | broker to you | marks `resp_RID.json` as complete |
| `detached_RID` | you to broker | the client returned on purpose (`--wait-budget`), so the request keeps running |

`agent_selfcheck.py --attach RID` collects a request and `--list` shows the ones in flight. A
response's top level holds `n_passed`, `n_certified`, `all_pass`, `certified_complete`, `scope`,
`suite_size`, `per_capsule`, `note`, and `performance_brief` / `offload_census` when they were
produced.

## Per-case copies (`selfcheck_out/`)

From `merlin_experiments.phase1.feedback.selfcheck`:

- The artifact root is `selfcheck_out/requests/RID/` when the broker passes `--artifact-root`, which
  it does for every per-request child. Otherwise it is `selfcheck_out/`. The root is validated to be
  exactly that per-request directory.
- For each graded case the self-check copies `command_buffer.json`, `instruction_trace.json` and
  `lowered.llvm.mlir` from the run's `generated/` into `ROOT/CASE/`. It records their paths in the
  row as `your_artifacts`.
- When the original linked producer chain is available, `candidate_build_view` points to
  `candidate_build.json`, alongside the original input, commands, emitted LLVM and
  `kernel.linked.dis`. The last file contains only the retained candidate entry function's
  instructions from the original linked image. The harness ELF and its data stay private.
  Compile-only requests publish this view before their private build products are released.
  Analyze it with `--reference-analysis candidate --reference-format candidate_build
  --reference-path PATH`. A host-private index supplies the original build/source identity;
  editing the public manifest cannot register a producer. Historical views keep their original
  compiler identity when the live submission changes. Missing linked provenance remains UNKNOWN.
- A failing row also carries `tiers`, `numeric` (every statistic; `first_mismatch` keeps `output`,
  `index`, `your_value`, and the expected value is withheld), `trace_summary`, `trace_check`,
  `failure` and `sim_console_tail`.
- A passing row carries `pass`, the `barrier_*` fields and the measured cycles. Its full debug block
  is left out to keep responses short.
- A whole-model row shows only the closed codes `candidate_native_verification`, `failure.category`
  and `failure.plane`. No trace or console comes from the runner-owned host program.

## Full run directory (model layers)

`--model-layers` grades at the functional tier into `ROOT/model_layers/` as its runs root.
`run_capsule` lays results out at `RUNS_ROOT/runs/SUITE/CASE/` (`capsule_runner.suite_for`), and the
directories come from `aet.core.run_paths.RunPaths`:

```
CASE/
  capsule_result.json   verdict: status, tiers, trace_check, numeric, failure, work_volume, ...
  run_manifest.yaml     identity, required_oracle_tiers, tier_status, cycles per tier
  artifacts/            tier consoles, e.g. functional_console.log
  logs/  contracts/     (usually empty)
  generated/            every product of building and running the case
```

Inside `generated/`, grouped by who wrote the file:

- **Your compiler's outputs**:
  - `input.interface.mlir` is the input your tool received.
  - `lowered.target.mlir` is the stdout of `lower_interface_to_target`.
  - `command_buffer.json` comes from `emit_command_buffer` or `emit_analysis_bundle`.
  - `lowered.llvm.mlir` is the stdout of the target-artifact command (`merlin.targetgen.capsule_common.lower_interface`).
- **The object build** (`merlin.targetgen.contract.compile`). The harness builds these from your bytes:
  - `kernel.llvm.mlir` is a copy of the lowered IR, and `kernel.ll` is its `mlir-translate` output.
  - `kernel.o` is the object, compiled with `-fstack-usage`.
  - `kernel.su` is the compiler's stack report, and `kernel.measured.su` is its canonical view.
  - `kernel.stack_frame.json` is the preflight receipt (`entry_symbol`, `max_static_bytes`, `frame_bytes`, `headroom_bytes`, `status`, `diagnostic`).
  - `kernel.abi.json` is the object's ABI receipt.
  - `harness.c` and `package_kernel.elf` are the linked test program.
- **Executed evidence**:
  - `numeric_report.yaml` holds `status`, `mismatch_count`, `max_abs_error`, `first_mismatch` and `per_output`.
  - `oracle_console.log` is the simulator console.
  - `executed/ENGINE/DIGEST/` holds the exact program each engine ran (`merlin.targetgen.executed_program`).
- **Harness analyses of your output**:
  - `instruction_trace.json` is the decode of `lowered.llvm.mlir`.
  - `liveness_report.json` and `placement_census.json` are analyses of the same output.
  - `coverage_certificate.json` comes from the same analyses.

The operator grade keeps these trees on the host side. The linked candidate view exposes no
harness program or data. Reproduce additional object diagnostics when needed with `inspect-kernel`.

## Diffing

`case_artifacts.py diff A B` accepts a copied case directory or a full run directory; for a full run
directory it reads `generated/`. The rules for reading the result:

- Identical `command_buffer.json` with a differing `lowered.llvm.mlir` means only emission changed.
- An identical `instruction_trace.json` class histogram with different `barrier_cycles` points to
  operand values or loop trip counts. Compare the `loops` lists.
- A different command count or opcode delta means planning changed. Dump the stages of both trees
  (the `inspect-ir` skill, `dump_stages.py compare`) to find the first stage that differs.
