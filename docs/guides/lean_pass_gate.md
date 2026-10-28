---
title: Gate dead-code pruning and cse on Merlin's Lean proofs
kind: guide
status: current
owner: verification
last_verified: 2026-10-28
related: [verification, compiler_verification]
code_refs:
  - src/merlin/verify/pass_validation.py
  - src/merlin/verify/gate_checks.py
  - src/merlin/verify/speculative_gate.py
  - src/merlin/verify/lean/check.py
  - src/merlin/llvmlower/passes_xdsl.py
  - src/merlin/llvmlower/lower.py
  - src/merlin/llvmlower/leading_cse.py
  - src/merlin/llvmlower/pipeline.py
  - src/merlin/compile/command.py
  - merlin/lean/MerlinVerify/Check/ExactCse.lean
  - merlin/lean/MerlinVerify/Rules/Dce.lean
  - merlin/lean/LEDGER.md
  - merlin/lean/lakefile.toml
  - build_tools/scripts/check_lean_guard.py
  - build_tools/scripts/verify_lean_pass_models.py
  - merlin/tests/ir/test_lean_pass_gate.py
  - merlin/tests/ir/lean_gate_chunks.py
  - merlin/tests/ir/test_lean_pass_gate_fuzz.py
  - merlin/tests/rvv/test_lean_pass_gate_spike.py
---

# Gate dead-code pruning and cse on Merlin's Lean proofs

Merlin can check two of its compile steps against the `MerlinVerify` proofs while it compiles: the
dead-code pruning in preprocessing and the tensor-level `cse` every lowering pipeline starts with.
Each step's exact IR before and after goes to a worker process; compilation carries on as if the
check passes, and a check that does not accept stops the compile, prints the check, the reason and
the paths of the input and output, and renames any `.ll`, `.so` or `.o` already produced to
`*.rejected`.

| Validated pass | Where it runs | Hook |
| --- | --- | --- |
| `prune_dead_pure_tensor_ops` | `lower_model` preprocessing (xDSL route) | `passes_xdsl._preprocess_module` |
| `cse` | every tensor-level `cse` the build's runner runs (the pipeline's leading one), on every route and with any features | `leading_cse.bind_runner`, `leading_cse.submit` |

The `cse` check is recorded inside the build's own runner, after whatever the selected features and
source stages do before the pass manager, so it checks the `cse` the build actually ran and the
build's output is byte-identical to an ungated build. Quantization lowering (`lower_quant_ext`, the
textual whole-model preprocessing) is not gated here.

## What is proved

For every validated boundary the worker requires all of:

1. `MerlinVerify.Check.TypedModule.check_sound` on the boundary, inputs of the declared types, both
   buffers exported by MLIR's own parser and round-tripped identically through Lean;
2. `MerlinVerify.Check.supportCheck_sound` on every function of the input (it never gets stuck),
   which makes the refinement strict rather than conditional;
3. `compose_chain` replaying both, with `refinement_verdict: verified`.

It attempts two strengthenings and reports them without requiring them: the universal boundary
theorem (`moduleCheck_sound`, for all inputs) and source definedness. The receipt's `claim` says
exactly what was established. Axioms: `propext`, `Classical.choice`, `Quot.sound`.

* Pruning is proved by the C1 DCE rule. Dead source ops are erased before alignment
  (`synthC1Live`, and `synthC1Unused` for what upstream `cse` erases), so a dead op is never matched
  with an identical live one.
* `cse` is proved by `Check/ExactCse.lean`, in one forward walk as upstream `cse` makes it: a later
  effect-free op with the same head as an earlier one, operands holding the same values and regions
  reading the same outside values computes the earlier op's value in the same run, so its uses are
  renamed to it. Facts come from the earlier op's actual execution plus an alias per merge, so
  merges cascade and regions may read outside constants; nothing is merged into an op nothing reads,
  matching `cse`'s erasure of unused ops. Both programs are normalized this way before comparison.

## Human review before relying on it

Lean checks every proof step; review establishes that what is proved is what is intended.

1. **The op semantics the claims are stated over** (`merlin/lean/LEDGER.md` A1–A8, T3): the
   `MerlinVerify` meaning of `arith`, `math`, `tensor` and `linalg` ops, which both checks rely on.
2. **The purity scope** (`Rules/Dce.lean` `pureOp`/`erasable`, ledger A9): which ops are treated as
   only computing their results, and so may be erased when dead and reused by `cse`. A9 extends this
   to the three modeled `quant_ext` ops, so a dead quantization op can be pruned.
3. **The quantization source semantics, for pruning only** (`Spec/Dequant.lean`,
   `Spec/Quantize.lean`): a prune boundary whose input still contains `quant_ext` ops needs their
   meaning to show the input never gets stuck. A module without them does not depend on these files.
4. **The trusted components** (ledger T1–T6): the Lean kernel and compiler, the `torch_mlir` exporter
   and the Python bridge, and everything below the tensor stage.
5. **The locked statements** (`merlin/lean/SPEC_LOCK`): the hashes pin the semantics files and the
   theorem statements the gate cites; review confirms they say what the claims need.

Not covered: `canonicalize`, the trailing LLVM-dialect `canonicalize,cse`, and lowering below the
tensor stage stay trusted. Only the 64-bit index width is admitted (LEDGER S6).

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
`build_tools/scripts/check_lean_guard.py` audits every theorem's axioms and the locked files.

## Turn it on

```
merlin-compile ... --validate-passes all --lean-sdk-bin <lean-4.35.0>/bin
```

or set `MERLIN_VALIDATE_PASSES` (`all` or a comma list) and `MERLIN_LEAN_SDK_BIN`, with
`MERLIN_COMPILER_PYTHON=<venv>/bin/python`, or pass `pass_validation=PassValidation(gate, tools)` to
`lower_model`. It is off by default; a missing or stale tool stops the compile before it starts. A
lowering called inside an active validation joins it, and a different explicit validation is refused.

## Outcomes

| Verdict | Meaning | Compile |
| --- | --- | --- |
| `accepted` | the three required checks passed | continues |
| `not_established` | the boundary or the composition did not verify | stops |
| `unsupported` | MLIR's parser refused the IR, or source support is not established | stops |
| `unavailable` | a tool could not run or the check ran out of time | stops |

A pass that changes nothing submits nothing.

## Tests

* `merlin/tests/ir/test_lean_pass_gate.py` — the gate's behaviour (injected prune and `cse` bugs stop
  the compile and leave no artifact, nesting, tool and freshness refusals, features around the
  runner's `cse`), small exact `cse` cases (cascades, captured constants, unused-op erasure, wrong
  merges), and every representative chunk of the retained capsules (`lean_gate_chunks.py`) through
  real lowering, live and with dead code, with byte-identical gated output.
* `merlin/tests/ir/test_lean_pass_gate_fuzz.py` — seeded random programs through the real passes;
  completeness against an independent reference and soundness against a numpy oracle.
* `merlin/tests/rvv/test_lean_pass_gate_spike.py` — whole-model builds on Spike: a gated build matches
  its golden output, and a wrong `cse` is stopped before any ELF exists.

Set `MERLIN_LEAN_REQUIRED=1` to turn a missing tool into a failure instead of a skip. Whole models
are qualified with `build_tools/scripts/verify_lean_pass_models.py --capture <mlir> ...`, which
compiles each with and without the gate and requires byte-identical LLVM output.
