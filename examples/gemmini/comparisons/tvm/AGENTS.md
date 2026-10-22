# Agent instructions: EL4 TVM comparison

Read the root AGENT.md/CLAUDE.md, this directory's README.md and UPSTREAM_AUDIT.md,
and the companion TVM `apps/gemmini/{README,OWNERSHIP,TUNING,FIRESIM,AGENTS}.md`.
This directory is a trusted comparison edge, not the generated compiler candidate.

## Goal and current boundary

Run the actual selected EL4 ResNet50, TinyLlama and SmolVLA sessions end to end
on queued FireSim, with unchanged quality/correctness gates and complete timing
and memory attribution. Historical host runs, INT8 cutouts and synthetic operator
packages are bounded evidence. They do not finish this goal.

The current source delivery provides real semantic JSON intake for tuning and
compilation, session route/reference planners and `build_package_session.py`.
The package bridge has small two-stage Spike positive/negative controls in both
near-code and explicit absolute layouts (9 GiB weights, 13 GiB arena). Actual
model qualification, full-size memory use and FireSim loading remain unfinished. Read
`EL4_ASSETS.md` for recorded capture/bundle paths and required independent
references. A matching FireSim driver, actual loader/high-DRAM evidence and full
SmolVLA numerical closure remain explicit execution prerequisites.

## Source isolation and ownership

- Start from pinned EL4 `0cc289de52a10bed256f32c90cfbb54a21b3c81f` plus the delivered
  comparison commits. Record actual current hashes; do not silently follow a
  moving branch. TVM starts from Apache v0.19.0 and is a separate repository.
- Preserve the original comparison and older TVM worktrees, including their
  uncommitted changes. Use isolated branches; never reset someone else's work.
- Keep target-specific comparison recipes here and device ISA/schedule code in
  TVM contrib or the explicitly selected target backend. Do not add target-name
  literals, ISA constants or workload dispatch to Merlin core.
- Reuse Merlin's existing runtime, linker, source/reference owners, instruction
  admission, queue and receipt APIs. Do not build a parallel execution harness.
- TVM schedules/reference outputs must not become grants to the from-scratch
  generated compiler candidate. Preserve frozen Phase 1 and campaign identities.
- The stopped manual EL4 campaign must remain stopped. Do not edit its deadline,
  baseline or `manual-stop.json` to reuse a convenient runner.

## Practical sequence

1. Verify source/library/toolchain identities and read final handoff validation.
   Create outputs with `merlin.common.artifacts.new_product/new_measurement`.
2. Reconcile actual selected EL4 inputs and numerical policy against
   `examples/gemmini/numerics/models.yaml` and the frozen run contract.
   TinyLlama defaults in the comparison factory are diagnostic; select full
   prefill/decode explicitly. SmolVLA requires dataset observations, all stages,
   ten recurrent updates and every padded action output; report physical channels
   separately. Do not substitute the TVM factories' numerical recipes for EL4's
   frozen capture scales and integerized IR.
3. Save semantic stage JSON with `--save-programs`, hash it and run the complete
   host reference/candidate session checks. Import-only does not establish
   candidate correctness. Keep calibration/development/held-out roles separate.
4. Use TVM `tune_models.py --program … --extract-only`; inspect ALL tasks and
   CPU exclusions. Tune and compile the same IR and capability fingerprint.
5. Qualify stages, then the entire linked recurrence/reset chain. Resolve
   SmolVLA errors rather than widening tolerances. All final outputs and carried
   state are required; cosine alone cannot replace elementwise checks.
   Export complete session UART with `read_package_session.py`, then bind its
   per-invocation NPZ files to independent references in the shared baseline
   checker. Preserve execution receipts; the exporter establishes transport and
   ABI correctness, not execution identity, numerical agreement or timing.
6. Bind the package ABI explicitly to the existing board-session runtime. Audit
   every symbol/instruction and all memory reservations after final link. Hardware
   loop FSM commands remain forbidden; raw-command OS/WS/software loops are legal.
7. Verify the exact full U250 bitstream/driver/params/DTS and loader. Run queued
   known-output positive/negative controls and required high-address memory probes.
8. Validate a checkpoint manifest with the existing CLI's `--dry-run`, then run
   actual complete workloads through the queue and seal the original logs.
9. Report E2E quality, compiler agreement, memory and measured cycles with the
   exact stage/repetition boundary. Never call summed operator minima E2E timing.

## Testing and known environment constraints

Comparison tests live in `merlin/tests/gemmini`. Use the selected Torch/ONNX/TVM
Python, `MERLIN_REPO_ROOT`, matching `TVM_LIBRARY_PATH`, and optional
`TVM_SESSION_MODEL2MLIR`/`MODEL2MLIR_ROOT`/`TVM_SESSION_MODERN_ONNX` as documented by
the tests. The base Merlin venv lacks Torch. Source-exact SiLU tests require the
pinned PyTorch git revision and AVX512 (one thread for SmolVLA, two for TinyLlama).
The selected Transformers 5.4.0 overlay passes the five cache/rotary/fixed-tree
integration checks; the SmolVLA environment's Transformers 4.57.1 is incompatible
with that model2MLIR cache protocol. Keep these environments explicit. Do not weaken those guards
to make an incompatible environment green. Record exclusions explicitly.

Run relevant tests and `python build_tools/scripts/check_structure.py`; report
inherited EL4 structure failures separately. New documentation here follows the
example-local format; durable docs under `docs/` require frontmatter/index checks.
Generated reports, binary packages, checkpoints and protected model data stay out
of Git. Never overwrite hardware evidence after code changes. No PR or force-push
is authorized by this handoff.
