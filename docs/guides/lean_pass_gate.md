---
title: Gate a compiler pass on Merlin's Lean proofs
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
  - src/merlin/compile/command.py
  - merlin/lean/lakefile.toml
  - merlin/tests/ir/test_lean_prune_gate.py
---

# Gate a compiler pass on Merlin's Lean proofs

`lower_model` can check `prune_dead_pure_tensor_ops`, the dead-tensor pruning every lowering runs,
against the `MerlinVerify` proofs while it compiles. The pass's exact IR before and after goes to a
worker process; lowering carries on as if the check passes, and a check that does not accept stops
the compile, printing the check, the reason, and the paths of the input and output.

## What is proved

The worker accepts only when all three hold:

1. `MerlinVerify.Check.TypedModule.check_sound` verifies the boundary on inputs of the declared
   types (both buffers exported by MLIR's own parser, round-tripped identically through Lean);
2. `MerlinVerify.Check.supportCheck_sound` verifies every function of the input (it never gets
   stuck), which makes the refinement strict rather than conditional;
3. `compose_chain` replays both and reports `refinement_verdict: verified`.

Claim: every behaviour of the pruned program is a behaviour of the original on inputs of the
declared types. Definedness (no undefined behaviour or poison) is not claimed:
`DefinedExtended.check_sound` is outside its proved fragment on the captured models. The theorems use
only `propext`, `Classical.choice` and `Quot.sound`. Their assumptions are `merlin/lean/LEDGER.md`
A1–A8 and T1–T6 (notably T2, the exporter and Python bridge, and T3, fidelity of the op semantics).

## Set it up once

```
# compiler Python with the pinned MLIR bindings (no PyTorch needed)
uv venv --python 3.12 <venv>
grep '^torch-mlir' build_tools/toolchains/lean-mlir-requirements.txt > torch-mlir.txt
uv pip install --python <venv>/bin/python --no-deps --require-hashes -r torch-mlir.txt
uv pip install --python <venv>/bin/python --no-deps numpy==2.4.6   # numpy carries no hash in the file

# Mathlib (prebuilt, about 7 GB) and the checker, from merlin/lean
cd merlin/lean
PATH=<lean-4.35.0>/bin:$PATH lake exe cache get
PATH=<lean-4.35.0>/bin:$PATH taskset -c 0-7 nice -n 19 lake build MerlinVerify merlin-lean merlin-lean-audit
```

The build writes `out/build/lean/build/bin/merlin-lean`, which the gate uses. A checker that is out of
date with `merlin/lean` (as `lake build --no-build` reports) is refused.

## Turn it on

```
merlin-compile ... --validate-passes all --lean-sdk-bin <lean-4.35.0>/bin
```

or set `MERLIN_VALIDATE_PASSES=all` and `MERLIN_LEAN_SDK_BIN=<lean-4.35.0>/bin` for any entry point
that lowers, with `MERLIN_COMPILER_PYTHON=<venv>/bin/python`, or pass
`lower_model(..., pass_validation=PassValidation(gate, CheckerTools(sdk_bin, compiler_python)))`.
A missing tool stops the compile before it starts. It is off by default.

## Outcomes

| Verdict | Meaning | Compile |
| --- | --- | --- |
| `accepted` | the three checks above passed | continues |
| `not_established` | the boundary or the composition did not verify: a pass bug | stops |
| `unsupported` | MLIR's parser refused the IR, or source support is not established | stops |
| `unavailable` | a tool could not run or the check ran out of time | stops |

A pass that changes nothing submits nothing. Measured on captured models with real lowering:

| Capture | Pruned | Gate | Lowering with the gate |
| --- | --- | --- | --- |
| small_llama_int8_consistent | 30 | accepted | 10.2 s |
| small_consistent | 30 | accepted | 9.9 s |
| tiny_llama_fp32_consistent | 0 | nothing to check | 1.0 s |

Raw int8 captures with `quant_ext` operations can read a value before its definition; MLIR's parser
rejects such IR, so the gate reports `unsupported` rather than checking it.

## Add a pass

A pass is a candidate when its boundary is in the checker's typed module domain (see the theorems in
`merlin/lean/MerlinVerify/Check/`). Add it to `pass_validation.CHECKS`, report its exact before/after
IR through `pass_validation.selected`/`submit` at the pass (as `passes_xdsl._preprocess_module` does),
and add a real-capture acceptance and an injected-bug test to `merlin/tests/ir/test_lean_prune_gate.py`.
