---
title: Gate compiler passes on Merlin's Lean proofs
kind: guide
status: current
owner: verification
last_verified: 2026-10-27
related: [verification, compiler_verification]
code_refs:
  - src/merlin/verify/pass_validation.py
  - src/merlin/verify/gate_checks.py
  - src/merlin/verify/speculative_gate.py
  - src/merlin/verify/lean/check.py
  - src/merlin/llvmlower/passes_xdsl.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/runtime/backends/zephyr_model.py
  - src/merlin/compile/command.py
  - merlin/lean/lakefile.toml
  - merlin/tests/ir/test_lean_pass_gate.py
  - merlin/tests/ir/test_lean_pass_gate_fuzz.py
  - merlin/tests/rvv/test_lean_pass_gate_spike.py
---

# Gate compiler passes on Merlin's Lean proofs

Merlin can check selected preprocessing passes against the `MerlinVerify` proofs while it compiles.
Each pass's exact IR before and after goes to a worker process; compilation carries on as if the
check passes, and a check that does not accept stops the compile, prints the check, the reason and
the paths of the input and output, and renames any `.ll`, `.so` or `.o` already produced to
`*.rejected`.

| Validated pass | Where it runs | Hook |
| --- | --- | --- |
| `prune_dead_pure_tensor_ops` | `lower_model` preprocessing (xDSL route) | `passes_xdsl._preprocess_module` |
| `lower_quant_ext` | `lower_model` preprocessing, and every `prepare_for_lowering` | `_preprocess_module`, `zephyr_model._prepare_model_mlir` |
| `preprocess_text_textual` | `lower_model(textual=True)`: the whole-model RVV route, including Spike | `passes_xdsl.preprocess_text_textual` |

The whole-model RVV route lowers textually, so in a Spike or `merlin-compile` RVV build the checked
boundary is `preprocess_text_textual` (dequantization, slice repair and the C interface together);
the xDSL prune and quantization run only on the xDSL route and in preparation.

## What is proved

For every validated boundary the worker requires all of:

1. `MerlinVerify.Check.TypedModule.check_sound` on the boundary, inputs of the declared types, both
   buffers exported by MLIR's own parser and round-tripped identically through Lean;
2. `MerlinVerify.Check.supportCheck_sound` on every function of the input (it never gets stuck),
   which makes the refinement strict rather than conditional;
3. `compose_chain` replaying both, with `refinement_verdict: verified`.

It then attempts two strengthenings and reports them, never requiring them: the universal boundary
theorem (`moduleCheck_sound`, for all inputs) and source definedness (no undefined behaviour or
poison). The receipt's `claim` says exactly what was established, for example "strict refinement
for all inputs; definedness not established". Axioms: `propext`, `Classical.choice`, `Quot.sound`.
Assumptions: `merlin/lean/LEDGER.md` A1–A8 and T1–T6 (notably T2, the exporter and Python bridge,
and T3, fidelity of the op semantics to MLIR). Only the 64-bit index width is admitted: at index32
the model and the native pipeline disagree on a retained example (LEDGER S6).

`quant_ext` dequantization is proved against `MerlinVerify.Spec.Dequant`, a narrow contract written
from PyTorch v2.10.0 (`_decomposed.py`): per-tensor converts then subtracts; per-channel promotes to
i32, subtracts, converts once, then multiplies — the order lowering implements since 03a22c5fd ("match PyTorch per-channel dequantization order").
Signed i8 input, i32 zero points and f32 scale only. The contract still needs human specification
review and capture-route binding; draft policy, native and capture premises remain;
an unexplained SmolVLA complete-output comparison failure is recorded upstream. With `int8_compute`
the W8A8 passes do most of the quantization before `lower_quant_ext` and are not gated here.

## Set it up once

```
uv venv --python 3.12 <venv>
grep '^torch-mlir' build_tools/toolchains/lean-mlir-requirements.txt > torch-mlir.txt
uv pip install --python <venv>/bin/python --no-deps --require-hashes -r torch-mlir.txt
uv pip install --python <venv>/bin/python --no-deps numpy==2.4.6   # numpy carries no hash in the file

cd merlin/lean
PATH=<lean-4.35.0>/bin:$PATH lake exe cache get        # prebuilt Mathlib, about 7 GB
PATH=<lean-4.35.0>/bin:$PATH taskset -c 0-7 nice -n 19 lake build MerlinVerify merlin-lean merlin-lean-audit
```

A checker that is out of date with `merlin/lean` (`lake build --no-build`) is refused.

## Turn it on

```
merlin-compile ... --validate-passes all --lean-sdk-bin <lean-4.35.0>/bin
```

or set `MERLIN_VALIDATE_PASSES` (`all` or a comma list) and `MERLIN_LEAN_SDK_BIN`, with
`MERLIN_COMPILER_PYTHON=<venv>/bin/python`, or pass `pass_validation=PassValidation(gate, tools)` to
`lower_model`. It is off by default; a missing or stale tool stops the compile before it starts.
`prepare_for_lowering` and `lower_model` each run under their own guard; a step called inside an
active validation joins it, and a different explicit validation is refused.

## Outcomes

| Verdict | Meaning | Compile |
| --- | --- | --- |
| `accepted` | the three required checks passed | continues |
| `not_established` | the boundary or the composition did not verify | stops |
| `unsupported` | MLIR's parser refused the IR, or source support is not established | stops |
| `unavailable` | a tool could not run or the check ran out of time | stops |

A pass that changes nothing submits nothing. Raw int8 captures that read a value before its
definition are rejected by MLIR's parser and so reported `unsupported`.

## Known checker gaps

Found by fuzzing and pinned by tests in `test_lean_pass_gate_fuzz.py` (they flip when fixed). Both
are correct rewrites the checker does not prove, so the gate would stop a compile that contains them:

* deleting a dead `quant_ext` dequantization (no totality fact for dequantization);
* deleting a dead constant-fed generic placed before live code (the same cone after live code is proved).

Neither occurs in the captured models checked so far, but random programs hit them often (32 of 40
in one campaign), so they block turning the prune gate on by default.

## Tests

* `merlin/tests/ir/test_lean_pass_gate.py` — real lowering of captured models with each pass proved,
  injected bugs in each pass stopping the compile, nesting, tool and freshness refusals.
* `merlin/tests/rvv/test_lean_pass_gate_spike.py` — whole-model builds on Spike: a gated build matches
  its golden output; an injected dequantization bug reaches the output without the gate and is
  stopped, before any ELF exists, with it.
* `merlin/tests/ir/test_lean_pass_gate_fuzz.py` — seeded random programs through the real passes.
  Completeness: each real output is accepted, or it is identical to an independently rendered expected
  output that is refused too (a checker gap). Soundness, at all three boundaries: a mutated output a
  numpy oracle can tell apart is never accepted. `MERLIN_FUZZ_SEED` and `MERLIN_FUZZ_CASES` scale it.

Set `MERLIN_LEAN_REQUIRED=1` to turn a missing tool into a failure instead of a skip.

## Add a pass

A candidate's boundary must lie in the checker's typed module domain (the theorems in
`merlin/lean/MerlinVerify/Check/`). Add it to `pass_validation.CHECKS`, report its exact before/after
IR with `pass_validation.observe` (or `submit` for text) where it runs, and add a real-capture
acceptance, an injected-bug test and fuzz coverage.
