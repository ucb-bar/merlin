---
title: Optional lowering passes — list, select, copy
kind: guide
status: current
owner: compiler
last_verified: 2026-11-05
related: [model_lowering, whole_model_on_accelerator, extending_the_stack]
code_refs: [src/merlin/llvmlower/optional_passes.py,
            src/merlin/llvmlower/int_softmax_table.py,
            src/merlin/llvmlower/portable_math.py,
            src/merlin/llvmlower/normalization_reassociation.py,
            src/merlin/llvmlower/passes_xdsl.py,
            src/merlin/compile/command.py,
            src/merlin/perf/whole_model_builder.py]
---

# Optional lowering passes

Merlin's optional transforms are **global and opt-in**. Each one matches IR structure, never a model,
shape or target name. You can turn one on by name, or copy it into your own out-of-tree package and
change it there. `merlin.llvmlower.optional_passes` is the one list of them. For each pass it records
the stage it acts in, whether it is exact, its default, and what it changes in the program.

## List them

```bash
merlin-compile --list-passes          # a table
merlin-compile --list-passes --json   # the same rows as data, with the switch each one maps to
```

`exact` means bit-identical output on every input the original program defines. `numerics-changing`
means a different numerical model, graded against its own reference.

## Select them

| where | how |
|---|---|
| `merlin-compile` | `--pass NAME` (repeatable), `--no-pass NAME` for one that is on by default |
| whole-model builder (a launch config's `build_options`) | `lowering_passes: [NAME, -NAME]` |
| any process | `MERLIN_PASSES=NAME,-NAME` |

All three set `MERLIN_PASSES`. The lowering reads it wherever it decides, including in the child
processes that lower, so the same spelling reaches every path. With no selection, every pass keeps
its default and the emitted code is unchanged. A selection is recorded with the result:
`lowering_passes` in a `merlin-compile` result, and `notes.lowering_passes` in a builder record. Host
object caches key on it, because the open-model lowering identity digests every `MERLIN_*` switch.

A pass marked `capture` acts while the model is captured, so it cannot be selected at compile time.
Name it in the capture variant instead (for example `quant_integer_nonlinear: true` in a descriptor's
`claim_objective_variants`).

## What a pass maps to

Each entry names the switch that already existed, and selecting it flips that switch:

* a **lowering feature** (`merlin.llvmlower.impr_features`), added to or removed from the feature set
  `lower_to_llvm_ir` builds;
* a **`MERLIN_*` variable** (`MERLIN_FUSION_GUARD`, `MERLIN_SINK_DEALLOCS`, `MERLIN_STATIC_ARENA`,
  `MERLIN_LAYER_NORM_CHUNKED_SUMS`, `MERLIN_PORTABLE_TRANSCENDENTALS`). An
  explicit selection wins over the variable, which still works on its own for A/B builds;
* a pass of the **integer datapath** (`merlin.llvmlower.quant_passes`), added to or removed from the
  set `apply_quant` runs.

## portable-transcendentals

On by default. The lowered model's `erff`/`expf`/`tanhf`/`logf` calls go to Merlin's own
functions (`merlin_erff` and the rest, in `merlin/runtime/abi/portable_math.h`). These are linked
with the MLIR runtime into the host build that writes a golden and into every device image, so
the two evaluate the same function bit for bit. The functions are correctly rounded (logf on all
but five inputs).

`--no-pass portable-transcendentals` restores each target's C library. In that mode a host golden
and a RISC-V image can disagree in the last ulp, and requantization can turn that ulp into a
per-element gate failure. Each build records the policy in `stats["transcendental_math"]` and
each W8A8 golden in its provenance. `merlin-compile` warns when the two disagree. Design,
accuracy and evidence: [the design note](../design/portable_transcendentals.md).

## int-softmax-table

A capture made with integer nonlinears spells softmax per element: a fixed exponent grid index, an
integer exp, a 64-bit floor division, an i64 row sum, then the next contraction's per-row int8
quantization of the probabilities. `int-softmax-table` (`merlin.llvmlower.int_softmax_table`)
restructures that in the captured IR before the lowering pipeline runs, so the capture itself is
unchanged:

* the numerator becomes a read of a table, evaluated at compile time with the IR's own integer
  semantics;
* the clamp's upper bound, which never binds, is dropped, and its lower bound becomes a
  compare-and-select;
* the row sum is accumulated in i32 when it cannot overflow;
* the probabilities are quantized once per row, on their candidate values;
* the attention scale moves before the reshape that hid it from fusion.

It matches the structure only. Anything that differs is left as it was, and the build prints the
reason (`int_softmax_table_report.json` beside the lowered IR). `merlin/tests/ir/test_int_softmax_table.py`
runs the captured and the rewritten modules on the same inputs and requires identical bits. It covers
the table at every grid index, the softmax alone, and two whole int8 attentions, including flat rows
whose quantization scale clamps to its eps. The fixtures live in `merlin/tests/data/int_softmax_table/`.

## layer-norm-chunked-sums

A LayerNorm's row sum is one long chain of f32 additions. `layer-norm-chunked-sums`
(`merlin.llvmlower.normalization_reassociation`) splits the row into contiguous 32-element chunks, sums
each chunk into a fresh `+0.0` partial, and sums the partials. It runs in the lowering's xDSL
preprocessing, after the quantization rewrite and before the module is handed to upstream MLIR.

It is **numerics-changing**. Selecting it is the explicit permission for two things the source does not
grant: reassociating those additions, and assuming the intermediates stay finite. A partial sum that
overflows in one order may stay finite in another, and NaN payloads and exception order are not
preserved. Grade a build that selects it against its own reference.

It matches structure only. A `linalg.reduce` is rewritten only when its provenance names the operation
kind `layer_norm`, it adds f32 values over the innermost axis, that axis has a static extent of at
least two chunks and divisible by the chunk, and its initial value is a `+0.0` splat. Every other
reduction is left byte-identical. The rewritten reduction and its new partial-sum generic carry
`merlin.normalization_reassociation`, so a second application finds nothing. When the lowering
records a source transform map, the reduction owns the operations the rewrite adds.
`merlin/tests/ir/test_normalization_reassociation.py` checks the scope, both permissions, the default-off
seam, and that the rewritten sum equals the source sum on inputs whose partial sums are exact.

## Adding one

1. Write the transform so that it matches structure only, and so that a build which does not select
   it is byte-identical to one made before it existed.
2. Register it as one of the switches above.
3. Add an `OptionalPass` entry. Give it an honest `exactness`: if you claim `exact`, add a test that
   runs the original and the rewritten program on the same inputs and compares the bits.
4. `merlin/tests/ir/test_optional_passes.py` checks that every entry names a real switch.
