---
title: Verify the program and its execution
kind: guide
status: current
owner: verification
last_verified: 2026-11-09
related: [07-building, 10-compiler-generation, ../reference/repo_structure]
code_refs:
  - src/merlin/compiler/quantization/contract.py
  - src/merlin/compiler/quantization/admission.py
  - src/merlin/transforms/validation.py
  - src/merlin/verification/cli.py
  - src/merlin/verification/scalar_ir.py
  - src/merlin/verification/partition.py
  - src/merlin/verification/lean_checker.py
  - src/merlin/model_check/metrics.py
  - src/merlin/model_check/full_quantized.py
  - src/merlin/model_check/full_chain_validation.py
  - src/merlin/model_check/binding.py
  - src/merlin/evaluation/capsule/numerics.py
  - src/merlin/runtime/dispatch_placement.py
  - src/merlin/runtime/elf_audit.py
  - src/merlin/runtime/service.py
  - build_tools/scripts/check_test_budget.py
  - .github/workflows/ci.yml
---

# 9. Verify the program and its execution

[Previous](08-running.md) | [Guide index](../README.md) | [Next](10-compiler-generation.md)

A build can succeed while computing the wrong values. A numerically correct
result can also come from an unintended scalar fallback. Verification therefore
answers several questions, using the original program, its numerical contract,
the compiled artifacts and observations of the actual run.

Keep three programs distinct: the original model, the realized quantized program,
and the compiled executable. If no quantization is requested, the middle step
can be absent. A single score against the original model cannot explain which
stage introduced a difference.

## Check the right comparison

| Question | What to compare or inspect |
| --- | --- |
| Is the numerical change acceptable? | Original model versus the quantized reference on selected evaluation inputs, using approved task-quality/error criteria. |
| Did capture/import preserve the prepared program? | The realized framework program and imported representation, including signatures, constants, shapes and quantization behavior. |
| Did compilation preserve that meaning? | Quantized/imported reference versus complete executable outputs under the selected numerical policy. |
| Did the intended route execute? | Selected regions/resources versus emitted instructions, calls and actual execution evidence. |
| Is composed execution valid? | Buffers, effects, aliases, ABI, transfers, completion and resource lifetimes across boundaries. |

Calibration inputs help determine quantization parameters. Held-out evaluation
inputs assess quality or generalization; they have a different role. A compiler
candidate must not choose a looser tolerance because its output differs.

Quantization means more than storage dtype. Preserve scales and zero points,
accumulator type, rounding, saturation, bias domain and the order of scaling,
activation and clamping. Runtime-computed parameters remain part of the program.
`compiler/quantization/contract.py` and `admission.py` describe these contracts
and their checks; they do not by themselves establish model quality or importer
fidelity. Use the reference appropriate to the realized numerical form.

## Follow the verification owners

| Owner | Responsibility |
| --- | --- |
| `verification/` | Independent program/transformation checks, including scalar SMT, pure outlining, partition reconstruction, packing/pointer obligations and Lean checking. |
| `transforms/validation.py` | Checks the actual structure and witnesses of supported before/after rewrites; a structural witness is not a formal receipt. |
| `evaluation/capsule/` | Runs selected capsule evaluation paths and judges their declared numerical/execution obligations. |
| `model_check/` | Builds and checks complete model paths, references, all-output comparisons and declared simulator/hardware bindings. |
| `runtime/` | Executes selected images and checks applicable deployment, ELF, dispatch and observation boundaries. |

For example, `verification/partition.py` reconstructs the original typed SSA
program from outlined calls. That checks partitioning fidelity, not subsequent
device lowering. `model_check/full_chain_validation.py` requires the selected
quantized-reference gates and every declared step, rather than trusting a
standalone pass string.

The applicable verifier must bind its claim to the exact source and artifacts.
A saved receipt for an older program cannot establish a changed program. A
missing solver, unsupported operation or missing run observation stays explicit.
The candidate compiler cannot replace the independent checker that accepts it.

## Try a small scalar equivalence check

Save this as `out/before.mlir`:

```mlir
builtin.module {
  func.func @forward(%x: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %result = arith.addi %x, %zero : i32
    func.return %result : i32
  }
}
```

Save the simplified version as `out/after.mlir`:

```mlir
builtin.module {
  func.func @forward(%x: i32) -> i32 {
    func.return %x : i32
  }
}
```

With the `verify` extra installed, run:

```sh
merlin verify scalar-receipt --before out/before.mlir --after out/after.mlir --output out/scalar.json
```

This uses the supported pure scalar-integer SMT checker: it asks whether any
input distinguishes the two functions under its modeled arithmetic. SMT is a
solver-based check, not a sample-input test. Its result is scoped to those
functions and bytes; it establishes no accelerator, pointer or hardware behavior.
Use fresh output paths. `merlin verify --help` lists the other actual entrypoints;
there is no universal command that proves every backend or model.

The CLI distinguishes `verified`, `refuted` and `abstained`, with exit codes
0, 1 and 2. Abstention means the checker could not establish the requested claim,
not that the compiler is correct or necessarily wrong. A timeout or unavailable
tool must not be recorded as a pass.

## Check route, effects and ABI as well as numbers

An ABI is the calling convention and data layout at a compiled boundary. Match
argument/result positions, scalar versus buffer arguments, element widths,
strides, alignment, ownership and aliases. Shared memory also needs the selected
visibility and completion rules. A transfer's issue is not its completion.

`runtime/dispatch_placement.py` checks structural placement requirements.
`runtime/elf_audit.py` checks a produced ELF against selected board properties.
`model_check/binding.py` checks declared simulator/RTL/board correspondence from
the selected sources. Each supplies a different part of the evidence.

If a requirement says Gemmini must execute, scalar fallback cannot satisfy it
merely by producing correct output. Conversely, an instruction appearing in a
binary does not prove it ran. Retain observations of the selected route and
whole required boundary, including every live output and required effect.
A command-stream or device-binary backend needs its matching decoder/reference
and runtime evidence; LLVM emission is not a prerequisite for verification.

## Keep regression tests small and verification separate

The permanent pytest suite is capped at 150 collected cases. It protects a
small set of distinct compiler, interface and workflow obligations. Prefer a
clear positive/negative example for an important contract over a test for every
helper, branch or tile combination.

Generated conformance cases, complete-output checks, solver obligations and
hardware qualification are separate workflows. Their coverage can grow without
turning every generated case into a permanent pytest test. They must still use
fixed references, recorded inputs and explicit acceptance criteria.

Hosted CI checks source/layout rules, packaging, retained regression cases and
public examples with available tools. Hardware-dependent checks need their
selected environment and belong in the corresponding qualification workflow.
CI green does not substitute for an unavailable board or a missing reference.

After [building](07-building.md), keep the build and run reports with the
verification results. The [repository map](../reference/repo_structure.md)
locates these owners; [compiler generation](10-compiler-generation.md) explains
how the same boundaries apply to an agent-authored backend.
