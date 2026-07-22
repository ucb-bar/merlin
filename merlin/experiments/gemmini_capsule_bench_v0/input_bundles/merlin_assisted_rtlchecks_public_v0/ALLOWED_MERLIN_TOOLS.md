# Allowed Merlin tooling — `merlin_assisted` arm

This file is the authoritative, human-readable statement of what the **merlin_assisted** agent may and
may not use. The machine-readable source of truth is `input_bundle_manifest.yaml` (allowed/denied); if
the two ever disagree, the manifest wins. The *only* legitimate difference between this arm and
`raw_baseline` is the extra **authoring** tooling listed here — everything about grading is identical
(see `../../COORDINATION.md`).

## The one rule that governs everything

> Merlin tools may help you **author and debug** the package. The **final submitted package must be
> self-contained and integrity-clean**: it is graded only through its 4 CLI entrypoints, never imported,
> and it must pass the non-exempt integrity scan (no `import merlin` / `from merlin`, no
> `merlin.runtime.reference` / `merlin.runtime.simulator`, no `reference_outputs`, no copied/called
> kernels, no embedded outputs).

An authoring aid you used to *think* is fine; a runtime dependency on Merlin in the shipped package is a
hard fail.

## ALLOWED (authoring aids)

| Tool | Path | What it's for |
|---|---|---|
| Target/dialect/runtime **plan synthesis** | `merlin/python/merlin/targetgen/synthesize/` | Produce a *plan/spec* for a dialect, lowering, runtime adapter (scaffolding intent, not the answer). |
| **Scaffold generators** (xDSL + MLIR) | `merlin/python/merlin/targetgen/generate/` **except `runtime_adapter.py`** | Emit empty/structural dialect + pass + tablegen scaffolds you then fill in. |
| **xDSL dialect patterns** | `merlin/python/merlin/xdsl_dialects/` **except `lowering/`** | Reference IRDL op/type/verifier patterns for prototyping your input + target dialects. |
| **Interface grammar emit/parse** | `merlin/python/merlin/targetgen/contract/interface_emit.py` | Serialize/parse the `merlin_iface` v0.1 grammar (clean; imports only `re`/`typing`). |
| **Compiler-modification spine** (CCA) | `merlin/python/merlin/kernels/{cca,cca_compare,cca_contract,action_catalog,microkernel}.py` | Learn WHERE and HOW to modify the compiler: `cca` (the target-agnostic compute abstraction) + `cca_compare` (diff two behaviors) + `cca_contract.check_bijection("gemmini")` (the *what-to-build* checklist: which levers exist) + `action_catalog` (route a divergence to a compiler seam + the "which file" seam map) + `microkernel` (the tunable micro-kernel space). Answer-free analysis; none import the oracle or the grader. |
| **Gemmini codegen levers** | `merlin/python/merlin/llvmlower/gemmini_features.py`, `merlin/python/merlin/targetgen/gemmini_plugin.py` | The default-off Gemmini codegen features (`GemminiCodegenOpts`) + the backend plugin that registers the Gemmini routes/seams/features. The seams point at YOUR generated OOT package (OOT-relative), not our in-tree reference. |
| **Shared hardware spec (ISA/RTL)** | `experiments/.../contracts/hwbringup_gemmini_v0` (bound as `gemmini`) | RTL + ISA headers + README + one example — the *defined ISA* your backend must target. A shared constant across ALL arms (not Merlin assistance): a correct backend needs the target's ISA. |
| Public Gemmini facts | `tmp/.../gemmini-rocc-tests/include/gemmini.h`, `gemmini_params.h` | ISA encoding, DIM, dtypes. |
| Public capsule contract | `merlin/contract/` (schemas, grammar, command-buffer ABI, integrity policy) + public/dev capsule inputs | The interface you compile and the schemas you must satisfy. |
| Toolchain | `third_party/llvm-install/` | LLVM/MLIR 23 to build the OOT package. |

## FORBIDDEN (would break comparability or integrity)

| Forbidden | Why |
|---|---|
| `merlin.runtime.reference` / `merlin.runtime.simulator` / `reference_outputs` / `outputs_match` | The oracle. Using it = self-grading against the true answer instead of the redacted QA verdict. |
| `merlin/python/merlin/targetgen/generate/runtime_adapter.py` | Emits a `semantics.py` that does `from merlin.runtime import reference_outputs` — a **callable oracle route**. Denied even though `generate/` is allowed (deny-wins; the launcher stages `generate/` minus this file). |
| `merlin/python/merlin/xdsl_dialects/lowering/` | `pipeline.execute()` calls `reference_outputs()` and returns a correctness verdict — a **callable oracle route**. Denied even though `xdsl_dialects/` is allowed (launcher stages it minus `lowering/`). |
| Grader internals: `rocc_decode`, `trace_check.py`, `capsule_grade`, `capsule_golden`, `capsule_runner`, `oot_runner` | Reading the grader to conform to/reverse-engineer it instead of deriving from public facts. |
| Hidden capsules / hidden outputs / withheld `golden.yaml` / `expected_command_buffer*` | Answers. Masked + denied. |
| Prior backends: `agent_spec_v0/v1_mlir_oot`, `hand_smoke_oot`, `merlin_native_v0` | Copying a finished package. Denied + canary-guarded. |
| Copied/called C kernels, bareMetalC, high-level Gemmini C lib (`tiled_matmul_auto`) | The device path must be MLIR-lowered RoCC, computed genuinely. |
| Any embedded/hardcoded capsule outputs | Hidden capsules after freeze will fail a memorizer. |

## How "denied inside an allowed dir" is enforced

Under `--sandbox none` (the mode both arms run — bwrap crashes the `claude` binary here), the real
`merlin` package is importable from disk, so a deny-list alone cannot *prevent* access. The boundary is
enforced in layers:
1. **Workspace staging** copies the allowed tool dirs **minus** the denied sub-paths, so the workspace
   you're given carries no pointer to `runtime_adapter.py` or `lowering/`.
2. **Post-run transcript audit** (operator-side) flags any read of a denied/oracle/grader path *and*
   any oracle import/call in code you write or run (`from merlin.runtime`, `reference_outputs(`,
   `pipeline.execute(`, `outputs_match(`).
3. **Integrity scan** on the submitted package is the final gate.

Don't try to work around the sandbox or reach denied paths on the real filesystem — it is detected, and
it invalidates the run's comparability.
