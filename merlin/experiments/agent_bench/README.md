# Agent benchmark — baseline vs Merlin-assisted (experiment ABI v0.1)

The substrate (`bench_contract/`, `oot_runner.py`, reference packages, AET recording, 18-cell
sweep) is frozen. This directory holds the **task packet** for the two-agent comparison. Nothing
here launches an agent; launch is a human-triggered step.

## Protocol

1. Stage a clean sandbox per agent (`setup_baseline_sandbox.sh`): only PUBLIC materials.
2. Launch the **baseline** agent (raw: contract + public docs only). Record telemetry.
3. Launch the **Merlin-assisted** agent (same task; may use Merlin tooling). Record telemetry.
4. Score both on PUBLIC (g0/g1/g2) and HIDDEN (h0/h1/h2 = renamed variants) via `grade.sh`.
5. Compare: highest rung, hidden pass rate, wall time, cost, tool calls, failure planes,
   artifact completeness.

## Public vs hidden

- **Public** (in the sandbox): `bench_contract/examples/{g0,g1,g2}.interface.mlir` + the whole
  `bench_contract/` bundle.
- **Hidden** (NOT in the sandbox; scored by the operator): `experiments/agent_bench/hidden/
  {h0_matmul,h1_relu,h2_acc_scale}.interface.mlir` — same single-tile structure as g0/g1/g2 but
  RENAMED tensors → different deterministic data. A package that hardcodes the public answer
  fails hidden; a correct data-independent kernel passes. (Verified: L0-consistent, data differs.)

## Allowed materials

| | baseline | merlin-assisted |
|---|---|---|
| `bench_contract/` (grammar, schemas, examples, integrity policy) | ✅ | ✅ |
| the grader (`oot_runner` via `grade.sh`) + LLVM/MLIR 23 install + RISCV gcc + spike/verilator | ✅ | ✅ |
| public Gemmini ISA headers (`software/libgemmini/gemmini.h`, `gemmini_params.h`) | ✅ | ✅ |
| MLIR `examples/standalone` OOT template | ✅ | ✅ |
| Merlin `targetgen` authoring helpers, xDSL scaffolding, package generator | ❌ | ✅ |
| reference packages (`merlin_native_v0`, `hand_smoke_oot`), the Merlin source tree, hidden tests | ❌ | ❌ |

The package itself must be self-contained and pass the **integrity scan** (no `import merlin`,
no `reference_outputs`, etc.). Both agents may iterate using `grade.sh` against the PUBLIC
examples; hidden scoring is operator-only.

## Telemetry sources

- `oot_runner` (per grade): run status, K-ladder, oracle kind/`derived_from_rtl`/`cycle_accurate`,
  cycles, origin-tagged artifacts, plane-routed failures → `runs/<run_id>/`.
- agent runtime (the launch harness): wall time, tool calls, token cost, transcript.

## Launch commands (operator runs these — they are the trigger)

```
# baseline (sandbox has ONLY public materials)
bash experiments/agent_bench/setup_baseline_sandbox.sh   # stages /scratch/agustin/agent_bench/baseline_ws
# then launch your agent runtime with TASK_baseline.md as the prompt, cwd = the sandbox.

# scoring (operator)
bash experiments/agent_bench/grade.sh <submission_dir> public   verilator
bash experiments/agent_bench/grade.sh <submission_dir> hidden   verilator
```

## Freeze checklist (all must hold before launch)

- [x] `bench_contract` v0.1 present + schemas valid; grammar frozen.
- [x] Grader runs a package from an arbitrary clean location (preflight passed: g0/g1).
- [x] `oot_runner` consumes the `mlir_oot_target_backend` artifact type.
- [x] Public examples g0/g1/g2 exist; hidden h0/h1/h2 exist and are separate/not in sandbox.
- [x] Telemetry records status, artifacts, oracle metadata (oot_runner) + wall/tool-calls (runtime).
- [x] Task prompt forbids C-compute-kernel-only and hidden-reference/harness imports (integrity).
- [ ] **bench_contract copied into the baseline sandbox** — done by `setup_baseline_sandbox.sh` at launch.
