---
title: Generate tests from declared requirements
kind: guide
status: current
owner: targetgen
last_verified: 2026-11-09
related: [cca, scheduling, getting_started, targetgen, shared_pass_compiler_generation]
code_refs: [src/merlin/experiments/phase0/component_coverage_inputs.py, src/merlin/experiments/phase0/component_coverage_plan.py, src/merlin/experiments/phase0/component_coverage.py, src/merlin/experiments/phase0/generation.py, src/merlin/experiments/phase0/writer.py, src/merlin/targetgen/corpus_spec.py, src/merlin/targetgen/capsule_golden.py, src/merlin/targetgen/software_spec.py]
---

# Generate tests from declared requirements

**Tutorial 2 of 5** | [Start here](../../examples/tutorials/README.md) | [Previous: Contracts and facts](target_contracts.md) | [Next: Scheduling](scheduling.md)

Start with a small shape domain, choose which combinations to cover, and build
one generated program. This example uses the same coverage algorithm and capsule
builder as Phase 0. It runs without Torch or hardware.

Use the [shared tutorial launcher](../../examples/tutorials/README.md) for setup.
Then run this ordinary Python example in that environment:

```python
from merlin.experiments.phase0.component_coverage_inputs import covering_points
from merlin.targetgen.capsule_golden import golden
from merlin.targetgen.corpus_spec import CorpusBinding, build

points, coverage = covering_points(
    {"M": [2, 3], "K": [2, 4], "N": [2, 3]},
    [], max_members=8, max_cells=100,
)
binding = CorpusBinding(
    target="illustrative", tile_dim=None,
    operand_dtype="int8", accum_dtype="i32", integer=True,
    tiers=[], compare="exact_int", readout_inputs={},
)
entry = {
    "name": "illustrative_matmul", "op": "matmul", "kind": "layer",
    "source_role": "illustrative_declaration",
    "source_reference": "Tutorial shape domain",
    "label": "public", "output_dtype": "i32", "stimulus_range": [-2, 2],
    **points[0],
}
capsule, interface_mlir = build(entry, binding)
print(coverage)
print(interface_mlir)
print(golden(capsule))
```

The output contains five shape points covering 18 single-axis and pair
combinations, an i8-input/i32-output matmul program in `merlin_iface`, and its
expected integer outputs. `capsule` is the generated declaration: operation,
operand shapes, numerical policy and expected instruction classes.

Open [the notebook](../../examples/tutorials/02_test_generation.ipynb) to change the
member budget, request the full three-axis interaction and inspect each point.
The point menu follows the generated cases; the display shows covered/missing cells
and the selected program. Ordinary Python calls also work without widgets. The
notebook keeps its results in memory.

## What coverage means

`covering_points` covers every declared single-axis and pair projection, plus
requested larger interactions. Adding `["M", "K", "N"]` as an interaction requires
all eight shapes in this example. Reducing `max_members` reports `incomplete` and
the number of missing cells. Coverage is over the finite domain you supplied;
a shape sweep alone does not test every numerical corner case.

## Follow the real generation path

The small example calls the inner algorithms directly. For a target corpus,
[`generate_target`](../../src/merlin/experiments/phase0/generation.py) connects them
with input review, references and corpus writing:

| Step | Source under `src/merlin/` | Responsibility |
| --- | --- | --- |
| Load requirements | `experiments/phase0/component_coverage_plan.py` | Bind the reviewed plan to exact software, numerical and hardware inputs; validate operation/effect owners, roles and budgets. |
| Expand requirements | `experiments/phase0/component_coverage_inputs.py` | Use `covering_points` and facts-derived resource boundaries to produce entries. Keep unavailable obligations visible. |
| Build programs | `targetgen/corpus_spec.py` | Turn each entry and target binding into a declaration and interface MLIR. |
| Write capsules | `experiments/phase0/writer.py` | Check numerical bounds, select references, write programs, declarations, instruction-coverage files, full-output goldens and the capsule README. |
| Check generated members | `experiments/phase0/component_coverage.py` | Reopen written programs, goldens and effect witnesses against the original plan. |
| Compile and evaluate | `evaluation/capsule/` | Use the selected compiler and engines, compare complete outputs and inspect required instruction use. |

The software specification defines supported operation forms and arithmetic:
operand/accumulator/readout types, rounding, overflow, layouts, tails, composition
and quantization constraints. The hardware specification identifies source and
extraction inputs; RTL facts supply observed resource and instruction data. The
capability contract describes backend implementations. These are separate inputs.
`ComponentCoveragePlan.load` checks their exact hashes and the numerical-semantics
digest so generation cannot quietly substitute different arithmetic or capacity.

The example's `CorpusBinding` is hand-written because it describes an illustrative
program. A real run derives its binding from those selected inputs. `tile_dim=None`
selects no hardware geometry, and the empty instruction/tier lists request no
hardware evidence.

## Choose development and held-out roles before tuning

Component plans name three **cohorts**, meaning groups with different uses:

| Cohort | Generated visibility | Use |
| --- | --- | --- |
| `development` | `label: dev`, performance category | Public development and tuning inputs. |
| `functional_guard` | `label: public`, declared kind | Functional checks with their declared mandatory status. |
| `withheld_transfer` | `label: hidden`, hidden category | Operator-owned transfer evaluation. |

The notebook creates no split. Real plans select these roles before tuning and
preserve them through writing, seals and evaluation. Calibration observations and
held-out quality evaluation serve different purposes. Keep held-out selectors
and answers out of author views. The private coverage receipt retains member
identities; `component_coverage.public_summary` exposes commitments and counts.

## Generate a target corpus

Use the public CLI below after selecting real inputs. The uppercase shell variables
name operator-supplied files for the same target. The reviewed coverage plan must
bind those exact inputs. The component recipe selects the shared performance
template rather than authoring its own `capsules` or `sweeps`. Keep destinations
under `out/`; `OUTPUT_ROOT` must be fresh.

```bash
python -m merlin.experiments.phase0 \
  --target "$TARGET" --descriptor "$DESCRIPTOR" \
  --recipe "$RECIPE" --performance-template "$PERFORMANCE_TEMPLATE" \
  --software-spec "$SOFTWARE_SPEC" --hardware-spec "$HARDWARE_SPEC" \
  --capability-contract "$CAPABILITY_CONTRACT" --rtl-facts "$RTL_FACTS" \
  --component-only --component-coverage "$COVERAGE_PLAN" \
  --evidence-mode diagnostic \
  --evidence-root "$EVIDENCE_ROOT" --output-root "$OUTPUT_ROOT"
```

The component entry contract requires `diagnostic` mode, which retains unresolved
evidence. Real entries may also require their selected reference framework or
model tooling. Inspect the generated manifest and coverage report before
separately selecting compilation and execution.

## Read the result correctly

This tutorial generates declarations, interface IR and expected values; it does
not write an admitted corpus or run an executable. The integer host reference
computes the declared operation independently of candidate command buffers while
sharing Merlin's Tensor arithmetic and deterministic inputs. Floating, model and
packed formats need their matching references. Numerical execution, required
instruction use and held-out quality still need separate checks. Phase 1 transcript
conformance checks authoring/tool evidence, not numerical correctness or whether
a compiler pass actually ran.
