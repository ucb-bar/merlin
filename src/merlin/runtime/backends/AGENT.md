# AGENT.md — merlin/python/merlin/runtime/backends

## Purpose

Merlin runtime **execution backends**: run the same Merlin command buffers the Python simulator executes, on real ISAs/simulators — spike (bare-metal multicore RVV) and pre-built Saturn VCS RTL sims.

## What belongs here

- `rvv_codegen.py` — command buffer → C driver around the hand-written RVV kernel.
- `spike.py` — compile (chipyard riscv gcc) + run (`spike --isa=rv64gcv_zfh_zvfh -pN`) + parse + normalize metrics.
- `vcs.py` — replay the same ELF on `MERLIN_SATURN_SIMV` (never builds RTL).

## What does not belong here

- Target dialects or lowering (that is `xdsl_dialects/`).
- The bare-metal harness C/asm (that is `merlin/runtime/baremetal/spike/`).
- RTL builds, vendored simulators, or anything that mutates the chipyard checkout.

## Interfaces

- Input: command-buffer dicts (command_buffer.schema.yaml), same as `merlin.runtime.simulate`.
- Output: `{outputs, metrics, raw_metrics, correct, console}` with metrics normalized onto `COMMON_METRIC_NAMES` (extras under `target_specific`).
- Env: `MERLIN_CHIPYARD` (default `/path/to/chipyard`), `MERLIN_RISCV_GCC`, `MERLIN_SPIKE`, `MERLIN_SATURN_SIMV`.

## Invariants

- A selected `runner.backend: chipyard_rocc` binds generic tooling directly from
  the selected contract and existing RTL facts before executable discovery. It
  imports no target provider. Missing or malformed selected data refuses; native
  binaries and receipt identities are checked when requested, without RTL builds.
- Target plugin execution requires a selected support provider on `MERLIN_TARGET_PATH`
  (unset, the checkout's vendored `examples/*/support` providers; see
  `target_registry.in_repo_support`). Reference metadata and generated-package discovery alone
  never authorize backend, dialect, or oracle imports. Direct plugin loads and
  already-loaded ownership checks enforce the same selection.
- Whole-model `MatrixRouting` requires separate support_target, unit and config.
  `plugin.matrix_lowering` owns geometry, selection policy, rewrite ABI/sidecar
  names and object construction. Shared build paths never choose a native matrix
  implementation. `matrix_routing.json` binds the preparation selection and
  signatures to the builder; changed or missing handoffs refuse reuse. It is a
  local consistency record, not frozen-source verification or certification.
- **Correctness gate**: every backend run is compared against `reference_outputs(cb)`; `correct` must be True for a run to count. Residency/parallelization must never change results.
- Generated epilogue C must match `tensor.py` semantics bit-exactly (rounding arithmetic shift, saturating i8).
- Tests must auto-skip (not fail) when the toolchain is absent (`spike.available()`).
- Keep the matmul kernel as hand-written `.S` — see `merlin/runtime/baremetal/spike/AGENT.md` for the GCC 13.2 intrinsics bug that forced this.

## Testing expectations

`merlin/python/tests/test_rvv_spike.py` (skips without toolchain); codegen-only tests run everywhere.

## Notes for future agents

`cycles` from spike is mcycle delta on hart 0 between the start barrier and the final barrier; counters (pack/hits/evictions/commits) are counted by the generated driver itself, byte counters are computed statically by codegen using the simulator's formulas.

- Whole-model `spike_model` build identity includes the actual linked device and
  matrix object bytes in link order, after compiling them and before emitting the
  harness marker. Paths are not identity. Final ELF hashes remain authoritative.
- The ordinary device build checks the complete routed kernel/precision roster
  and the caller's selected granularity before building or linking device work.
  Active device routes require an explicitly selected `LinkedElfAdmissionService`
  on the unchanged final link bytes, after legacy audits and before completing
  the recipe. Its optional `final_elf_audit` callback cannot replace that gate.
  The selected evaluator owns original source-symbol and instruction policy;
  shared orchestration grants no independent instruction/effect/runtime role.
  Zephyr preparation can emit the same routed device calls, but its builder
  constructs no device-object roster or final-link admission boundary. Active
  device signatures therefore refuse before LLVM/object compilation; host-only
  and inert routes remain available. A selected service alone cannot fill this
  missing device-link implementation.
  The legacy matrix-object routes bind support/unit/config but have no selected
  original linked-ELF policy correspondence. Nonempty matrix signatures refuse
  in both builders; host-only and inert routes remain available. Provider object
  presence, scalar diagnostic substitution or a successful link cannot supply
  the user's mandatory forbidden-instruction policy.
- Explicit whole-model operation profiling uses complete typed operation
  boundaries, including calls/stores with no result. Generic source printing
  retains the entry C interface; private marker callbacks retain their ABI.
  Profiling is optional, and marker costs/optimization perturbation must be
  measured against an uninstrumented control. A profile interval is not an
  automatic CPU/device attribution or accelerator-only timing measurement.
- The ordinary bare-metal model build also records owned model/runtime/harness
  compile commands and the link command in `compilation_recipe.json`, with actual
  executable and explicit input/output hashes. Ordered link inputs include provider
  objects. Completion follows final audits; a new refused/failed invocation cannot
  retain the previous success receipt. This observes the build, preserving flags,
  emitted bytes and existing marker identity. Header/library/provider compilation
  closure remains unknown rather than being inferred from a compiler name. The
  default math archive is an explicit exception: resolve `libm.a` with the selected
  GCC ISA/ABI flags, link its absolute regular-archive path, bind its bytes and
  driver to the marker and recipe, and refuse changed bytes before completion.
  Callers may additionally request a closed symbol roster from this archive:
  the actual link traces each defining member into the recipe, and the reader
  must independently select that roster. The default stays unchanged. This is
  no proof of source-call routing, other library/header closure or numerical
  host support.
- Device host ABI preparation is optional and source-bound. The provider's
  post-offload callback receives exact routed IR and an immutable sidecar; its
  selected host file enters the normal lowering/object identity path. ABI bridge
  code remains provider-owned; this hook makes no placement decision.
