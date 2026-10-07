---
title: Optional lowering passes — list, select, copy
kind: guide
status: current
owner: compiler
last_verified: 2026-10-06
related: [model_lowering, whole_model_on_accelerator, extending_the_stack]
code_refs: [src/merlin/llvmlower/optional_passes.py,
            src/merlin/llvmlower/int_softmax_table.py,
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
* a **`MERLIN_*` variable** (`MERLIN_FUSION_GUARD`, `MERLIN_SINK_DEALLOCS`, `MERLIN_STATIC_ARENA`). An
  explicit selection wins over the variable, which still works on its own for A/B builds;
* a pass of the **integer datapath** (`merlin.llvmlower.quant_passes`), added to or removed from the
  set `apply_quant` runs.

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

## Adding one

1. Write the transform so that it matches structure only, and so that a build which does not select
   it is byte-identical to one made before it existed.
2. Register it as one of the switches above.
3. Add an `OptionalPass` entry. Give it an honest `exactness`: if you claim `exact`, add a test that
   runs the original and the rewritten program on the same inputs and compares the bits.
4. `merlin/tests/ir/test_optional_passes.py` checks that every entry names a real switch.
