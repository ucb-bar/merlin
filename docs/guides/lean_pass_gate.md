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
  - src/merlin/llvmlower/leading_cse.py
  - merlin/lean/MerlinVerify/Check/ExactCse.lean
  - merlin/lean/MerlinVerify/Rules/Dce.lean
  - src/merlin/runtime/backends/zephyr_model.py
  - src/merlin/compile/command.py
  - merlin/lean/lakefile.toml
  - merlin/tests/ir/test_lean_pass_gate.py
  - merlin/tests/ir/test_lean_pass_gate_fuzz.py
  - merlin/tests/rvv/test_lean_pass_gate_spike.py
  - merlin/tests/ir/test_lean_dequant_contract.py
  - merlin/lean/MerlinVerify/Spec/Dequant.lean
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
| `cse` | the tensor-level `cse` every lowering pipeline starts with, on every route | `leading_cse.run` |

The whole-model RVV route lowers textually, so in a Spike or `merlin-compile` RVV build the checked
boundaries are `preprocess_text_textual` (dequantization, slice repair and the C interface together)
and `cse`; the xDSL prune and quantization run only on the xDSL route and in preparation.

Every pipeline opens with `canonicalize,cse`. With `cse` validated, `leading_cse` runs those two passes
before the pipeline (same pass registry and fusion guard as the build's runner), checks the exact `cse`
pair, and hands the result on; it then runs them once more and checks any change, so the pipeline's own
head provably changes nothing and the build lowers exactly what was checked. `canonicalize` is not
validated (folds are outside the checker). This is exact only when nothing rewrites the module before
the pass manager, so with features or source-carrier, observation or masked-contraction stages selected
`cse` validation is refused; validate the other passes then.

On the textual (RVV, Spike) route the gated build's `model.ll` is byte-identical to the ungated one.
On the xDSL route 15 commutative `fadd`s of `small_llama_int8` have their operands swapped: the
pipeline now parses the module once more after `cse`, and the later elementwise fusion builds bodies
in use order, which a print and parse does not keep (the likely cause; not separately confirmed).

## What is proved

For every validated boundary the worker requires all of:

1. `MerlinVerify.Check.TypedModule.check_sound` on the boundary, inputs of the declared types, both
   buffers exported by MLIR's own parser and round-tripped identically through Lean;
2. `MerlinVerify.Check.supportCheck_sound` on every function of the input (it never gets stuck),
   which makes the refinement strict rather than conditional;
3. `compose_chain` replaying both, with `refinement_verdict: verified`.

Dead-code deletion is proved by the C1 DCE rule; dead ops are erased before alignment is
attempted (`synthC1Live`), so a dead op is never matched with an identical live one. Common
subexpressions are proved by `Check/ExactCse.lean`, in one forward walk as upstream `cse` makes them:
a later effect-free op with the same head as an earlier one, operands holding the same values, and
regions reading the same outside values computes the earlier op's value in the same run, so its uses
are renamed to it. The facts come from the earlier op's actual execution, plus an alias fact for
every merge, so merges cascade (an empty, then the fill of it, then a generic of that) and regions
may read outside constants. Like upstream `cse`, which erases an op nothing reads when it reaches
it, nothing is merged into an op nothing reads; the comparison afterwards may erase such unused ops
(or, for pruning, whole dead cones). Both programs are normalized this way before comparison. It covers what
upstream `cse` merges in the captured models: `tensor.empty`, `tensor.collapse_shape`/`expand_shape`,
`linalg.fill` and generics.

It then attempts two strengthenings and reports them, never requiring them: the universal boundary
theorem (`moduleCheck_sound`, for all inputs) and source definedness (no undefined behaviour or
poison). The receipt's `claim` says exactly what was established, for example "strict refinement
for all inputs; definedness not established". Axioms: `propext`, `Classical.choice`, `Quot.sound`.
Assumptions: `merlin/lean/LEDGER.md` A1–A9 and T1–T6 (A9: the two modeled `quant_ext` dequantizations are
effect-free and may be erased when dead) (notably T2, the exporter and Python bridge,
and T3, fidelity of the op semantics to MLIR). Only the 64-bit index width is admitted: at index32
the model and the native pipeline disagree on a retained example (LEDGER S6).

`quant_ext` dequantization is proved against `MerlinVerify.Spec.Dequant`, a narrow contract written
from PyTorch v2.10.0 (`_decomposed.py`): per-tensor converts then subtracts (line 289); per-channel
promotes to i32, subtracts, converts once, then multiplies (lines 734–738) — the order lowering
implements since 03a22c5fd ("match PyTorch per-channel dequantization order"). Signed i8 input and
f32 scale; per-tensor with an i32 or i64 zero point, per-channel with i32. `quant_min`/`quant_max`
follow PyTorch: ignored per-tensor, checked against the i8 range per-channel, never clamped. Every
per-tensor dequantization in the tracked capsules (627 ops, i64 zero points, ranges [-128, 127] and
[-127, 127]) is proved (`test_lean_dequant_contract.py`); a whole capsule still is not, because its
`quant_ext.quantize_per_tensor` ops have no contract. f64 scales and per-channel i64 zero points are
excluded (`LEDGER.md`). Deleting a dead dequantization relies on ledger assumption A9 (both modeled
dequantizations are effect-free). The contract still needs human specification review and
capture-route binding; draft policy, native and capture premises remain;
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

## Cost

Each check takes 14–20 s on the captured models (1,000 ops); checks run one at a time, at nice 19,
beside the compile, so a gated `small_llama_int8` lowering takes about 48 s (xDSL route, three checks)
or 33 s (textual route, two), against 1.4 s and 0.6 s ungated.

## Outcomes

| Verdict | Meaning | Compile |
| --- | --- | --- |
| `accepted` | the three required checks passed | continues |
| `not_established` | the boundary or the composition did not verify | stops |
| `unsupported` | MLIR's parser refused the IR, or source support is not established | stops |
| `unavailable` | a tool could not run or the check ran out of time | stops |

A pass that changes nothing submits nothing. Raw int8 captures that read a value before its
definition are rejected by MLIR's parser and so reported `unsupported`.

## Checker gaps

None known. Fuzzing found two correct prune rewrites the checker did not prove (deleting a dead
dequantization; a dead cone before live code whose `tensor.empty` matched a live one). Both are now
proved and kept as regression tests in `test_lean_pass_gate_fuzz.py`, beside a test that deleting a
used value is still refused. A refusal the fuzzer cannot attribute to the checker fails the test.

## Tests

* `merlin/tests/ir/test_lean_pass_gate.py` — real lowering of captured models with each pass proved,
  injected bugs in each pass stopping the compile, nesting, tool and freshness refusals.
* `merlin/tests/rvv/test_lean_pass_gate_spike.py` — whole-model builds on Spike: a gated build matches
  its golden output; an injected dequantization bug reaches the output without the gate and is
  stopped, before any ELF exists, with it.
* `merlin/tests/ir/test_lean_pass_gate_fuzz.py` — seeded random programs through the real passes.
  Completeness: each real output is accepted, or it is identical to an independently rendered expected
  output that is refused too (a checker gap). Soundness, at every boundary: a mutated output a
  numpy oracle can tell apart is never accepted. `MERLIN_FUZZ_SEED` and `MERLIN_FUZZ_CASES` scale it.

Set `MERLIN_LEAN_REQUIRED=1` to turn a missing tool into a failure instead of a skip.

## Add a pass

A candidate's boundary must lie in the checker's typed module domain (the theorems in
`merlin/lean/MerlinVerify/Check/`). Add it to `pass_validation.CHECKS`, report its exact before/after
IR with `pass_validation.observe` (or `submit` for text) where it runs, and add a real-capture
acceptance, an injected-bug test and fuzz coverage.
