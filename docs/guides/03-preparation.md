---
title: 3. Capture and prepare numerical semantics
kind: guide
status: current
owner: compiler
last_verified: 2026-11-21
related: [02-targets, 04-planning]
code_refs: [src/merlin/capture, src/merlin/compiler/frontend/export.py, src/merlin/compiler/frontend/staged.py, src/merlin/compiler/frontend/bindings.py, src/merlin/compiler/quantization, src/merlin/compiler/lower/options/quantization_scope.py, src/merlin/compiler/ir/provenance.py, src/merlin/target/spec/quantization.py]
---

# 3. Capture and prepare numerical semantics

[Previous](02-targets.md) | [Guide index](../README.md) | [Next](04-planning.md)


Preparation turns a supported source program into the numerical program that
the compiler must preserve. Keep three programs distinct: the original model,
the realized quantized program, and the compiled executable.

## Start from the original program

Capture preserves the graph, call signature, constants, shape constraints and
source information. The shared capture code lives in `src/merlin/capture/`;
the public export/import adapters live in `compiler/frontend/`. Experiment corpus
construction lives in `src/merlin/experiments/corpus/`.

The public saved-export path imports and binds existing captured values; it does
not select a new quantization recipe. `StagedProgram` selects existing
compiler-only inputs with a source-bound precision policy and requires explicit
provider admission. It runs no capture, calibration or quantization.

Imported standalone TTIR stages retain their complete scalar/tensor interface
and actual standard operations, including guards and observable effects.
Their replayed compiler projection supplies a non-transforming source policy;
that policy is separate from backend support and numerical qualification.
Preparation neither freezes runtime state nor replaces the program with a
static tensor subset. Its nested source evidence remains part of the published
compiler-only artifact roster.

Derived TTIR models also replay their parent graph, exact kernel implementations,
shape/effect checks and compiler constant declarations. Staging retains only the
parent receipt fields used by those checks; original receipt hashes are observed
identities rather than authenticated lineage. Parameter archives are hashed and
copied in bounded chunks. Public preparation substitutes admitted immutable
parameters/constants using exact storage and checks every original operation
under that substitution. Captured BUFFER arguments remain explicit runtime state;
their captured snapshots grant no freeze permission. The prepared runtime
signature preserves outputs and allocation limits and remaps original input-axis
and input-word coordinates after immutable argument removal.

An authored MLIR kernel can enter directly with its numerical contract. It need
not pass through TorchAO. A Python kernel or FX graph needs a supported capture
and import route; it need not be an `nn.Linear` module.

## Where quantization is defined

The target software spec describes the numerical forms the target exposes.
Application preparation selects a supported form under the user's precision and
quality requirements. The backend then has to implement the realized arithmetic.
These are separate decisions; the software spec does not silently choose a
different model-quality target.

`target/spec/quantization.py` validates authored declarations.
`compiler/quantization/` owns numerical contracts, preparation policy, source
scope and the adapters used by compilation. TorchAO-specific preparation stays
in its adapter code rather than in generic numerical records.

The contract needs more than a dtype:

* Storage, expressed and accumulator types, signedness and value ranges.
* Scales and zero points, including computations that produce them at runtime.
* Per-tensor, per-axis or per-block granularity.
* Rounding, saturation and overflow behavior.
* Bias domain and the order of requantization, clamp and activation.
* Logical tensor layout separately from target packing.

For example, `requantize(acc + bias)` cannot generally be replaced by
`requantize(acc) + quantized_bias`. The compiler must retain the intended order.

## TorchAO routes

Module/tensor preparation and exported-graph preparation have different
interfaces. Supported module workflows apply the selected quantization recipe
to eligible modules. Graph workflows annotate supported patterns, prepare the
graph, calibrate when required, convert, and import the result.

Use the recipe and tool versions selected by the existing capture workflow.
The presence of an adapter does not mean every packed format, functional kernel
or QAT configuration is implemented. Check the selected recipe's applicability
and resulting graph before planning device regions.

Calibration inputs should reflect the activations that actually reach each
operation in the full model. Keep calibration provenance separate from held-out
task-quality evaluation. Dynamic quantization parameters remain computations.

## Preserve operation identity

A shared module can be called at multiple sites. Equal tensor shapes do not
identify the same operation. Preserve source lineage through export and import,
then bind implementation proposals to the actual imported operations.

Before preparation, backend capabilities can constrain feasible precision
choices. Final placement happens after import against the realized arithmetic.
If preparation invalidates an intended route, record the refusal and make a
bounded new attempt from the original program. Do not quantize an already
quantized previous attempt or change numerical meaning during lowering.

## Open the implementation

| Question | Source |
| --- | --- |
| Which contractions retain captured quantization scope during integer lowering? | [`lower/options/quantization_scope.py`](../../src/merlin/compiler/lower/options/quantization_scope.py) |
| How are numerical requirements represented? | [`quantization/contract.py`](../../src/merlin/compiler/quantization/contract.py) |
| Where is preparation policy selected? | [`quantization/preparation.py`](../../src/merlin/compiler/quantization/preparation.py) |
| How are supported TorchAO recipes projected? | [`quantization/torchao/recipes.py`](../../src/merlin/compiler/quantization/torchao/recipes.py) |
| What reaches import from a saved export? | [`frontend/export.py`](../../src/merlin/compiler/frontend/export.py) |
| How are captured parameters bound without changing their storage bits? | [`frontend/bindings.py`](../../src/merlin/compiler/frontend/bindings.py) |
