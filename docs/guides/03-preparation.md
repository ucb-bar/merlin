---
title: 3. Capture and prepare numerical semantics
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [02-targets, 04-planning]
code_refs: [src/merlin/capture, src/merlin/compiler/frontend/torch, src/merlin/compiler/quantization, src/merlin/compiler/ir/provenance.py, src/merlin/target/spec/quantization.py]
---

# 3. Capture and prepare numerical semantics

[Previous](02-targets.md) | [Guide index](../README.md) | [Next](04-planning.md)


Preparation turns a supported source program into the numerical program that
the compiler must preserve. Keep three programs distinct: the original model,
the realized quantized program, and the compiled executable.

## Start from the original program

Capture preserves the graph, call signature, constants, shape constraints and
source information. The shared capture code lives in `src/merlin/capture/`;
the public export/import adapters live in `compiler/frontend/torch/`.
Corpus construction belongs to `experiments/corpus/`, not the compiler frontend.

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
| Which source sites belong to a captured quantization scope? | [`quantization/scope.py`](../../src/merlin/compiler/quantization/scope.py) |
| How are numerical requirements represented? | [`quantization/contract.py`](../../src/merlin/compiler/quantization/contract.py) |
| Where is preparation policy selected? | [`quantization/preparation.py`](../../src/merlin/compiler/quantization/preparation.py) |
| How are supported TorchAO recipes projected? | [`quantization/torchao/recipes.py`](../../src/merlin/compiler/quantization/torchao/recipes.py) |
| What reaches import from a saved export? | [`frontend/torch/export.py`](../../src/merlin/compiler/frontend/torch/export.py) |
