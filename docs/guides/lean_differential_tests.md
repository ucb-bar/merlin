---
title: Qualify the Lean tensor pass gate
kind: guide
status: current
owner: verification
last_verified: 2026-10-28
related: [lean_pass_gate, merlin_lean_verification, compiler_verification]
code_refs:
  - build_tools/scripts/check_lean_guard.py
  - build_tools/scripts/verify_lean_pass_models.py
  - build_tools/toolchains/lean-mlir-requirements.txt
  - merlin/lean/LEDGER.md
  - merlin/tests/ir/test_lean_pass_gate.py
  - merlin/tests/ir/test_lean_dequant_contract.py
  - merlin/tests/ir/test_lean_pass_gate_fuzz.py
  - merlin/tests/rvv/test_lean_pass_gate_spike.py
---

# Qualify the Lean tensor pass gate

Build the checker using the pinned Lean and Mathlib versions in
[the gate guide](lean_pass_gate.md). Set `MERLIN_LEAN_SDK_BIN` to that SDK's `bin`,
`MERLIN_COMPILER_PYTHON` to the authoritative MLIR interpreter, and
`MERLIN_LEAN_REQUIRED=1` so missing prerequisites fail rather than skip.
Keep temporary directories under the project's `out/build/` root.

Run `python build_tools/scripts/check_lean_guard.py` with the selected SDK on
`PATH`. It rebuilds the checker and auditor, checks frozen-file and elaborated
statement identities, and audits every imported project theorem for unapproved
axioms and executable substitutions. `--lock-only` checks file identities only;
it does not replace the full audit. Specification changes require independent
review; regenerating a hash is not approval.

Run the real gate and quantization regression files listed in `code_refs`.
The pass-gate fuzzer executes the actual preprocessing passes and compares
mutations with an independent NumPy oracle. Increase coverage with
`MERLIN_FUZZ_SEED` and `MERLIN_FUZZ_CASES`. These tests exercise checker acceptance
and refusal; they do not directly qualify every operation of Lean's evaluator
against native MLIR.

`test_lean_pass_gate_spike.py` additionally requires a captured model, its golden
output, and the simulator/toolchain selected by Merlin's normal backend APIs.
It checks a successful gated build and injected compiler faults. Report missing
simulator prerequisites separately from passed checks.

To retain exported model boundaries from ordinary compilation, run
`PYTHONPATH=src python build_tools/scripts/verify_lean_pass_models.py` with
`--sdk-bin`, `--compiler-python`, and one `--capture /absolute/path` per model.
Each capture contains `model.mlir`. The tool compiles both xDSL and textual
routes with and without validation, requires identical returned LLVM bytes,
and retains the exact boundary texts, authoritative JSON exports, bound
receipts and verdicts in a fresh verification artifact directory. This is
pass-boundary evidence; downstream LLVM correctness remains trusted.

The broader `feat/pass-verification` delivery also retains direct Lean/native
semantics differential tooling. It is separate from this integration checkout's
gate regression suite. Historical model-fidelity observations and unresolved
cases remain in [the assumption ledger](../../merlin/lean/LEDGER.md); gate
acceptance does not discharge those assumptions.
