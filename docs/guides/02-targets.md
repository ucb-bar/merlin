---
title: 2. Select a target system and backend
kind: guide
status: current
owner: targetgen
last_verified: 2026-11-09
related: [01-inputs, 03-preparation]
code_refs: [src/merlin/target/spec/load.py, src/merlin/target/system/topology.py, src/merlin/compiler/backend/service.py, src/merlin/compiler/backend/load.py, src/merlin/development/targetgen/target_registry.py, src/merlin/development/targetgen/rtl/facts.py]
---

# 2. Select a target system and backend

[Previous](01-inputs.md) | [Guide index](../README.md) | [Next](03-preparation.md)


The hardware configuration and the compiler implementation answer different
questions. Keep both visible when choosing how a program will run.

| Term | Meaning | Example |
| --- | --- | --- |
| Target system | The selected SoC and its topology | CPU, accelerator and shared memory |
| Execution resource | A concrete place to execute | `cpu0`, `array0` |
| Backend | A compiler implementation | An installed scalar provider or an OOT compiler |
| Implementation | One way to compute a region | Vector matmul or accelerator matmul |
| Placement | The chosen implementation and resource | Matmul assigned to `array0` |

Scalar CPU code and RVV code can share one CPU resource. Two compiler providers
do not establish that their work can execute concurrently. The selected topology
must describe shared resources, memory access and synchronization.

## Read the target inputs

Target declarations used by compilation examples live under
`examples/targets/<target>/target/`. Compiler-generation workflows keep their
inputs beside the workflow at `examples/compiler-generation/<target>/target/`.
Some examples use `software-spec.yaml`; others select `spec.yaml` and
`hardware.yaml`. Read the example's README for its actual input names.

The software specification declares operation meaning: operand and accumulator
types, quantization, instruction behavior and constraints. Hardware selection
identifies the sources, configuration and evidence used for that system.
Extracted facts record what the selected readers observed, including unknowns.

```python
from merlin.api import inspect_spec

record = inspect_spec("examples/compiler-generation/gemmini/target/software-spec.yaml", expanded=True)
```

Compact specifications are loaded through `target/spec/load.py` and expanded into
the view consumed by existing code. Inspection does not execute authored Python
or establish that a backend implements every declared operation.

Try [notebook 1](../../examples/tutorials/01_contracts.ipynb): change a software
dtype and an RTL bit slice, then inspect the two results side by side. An
eight-bit wire alone does not reveal whether it carries an opcode or a number.
That meaning needs a checked correspondence to the selected target.

## Select an OOT package

An OOT repository can supply dialects, proposals, transformations, lowering,
encoders and runtime requirements. A dialect definition by itself is not a
complete backend.

There are two current selection boundaries:

* The public compile service resolves `merlin.backends` entry points.
  `inspect_backends()` reports installed names and declared origins without
  importing the providers. Install a local OOT project in the same environment
  when that project exports the required entry point.
* Target-definition discovery resolves contracts and support resources. Set
  `MERLIN_TARGET_PATH` to the explicitly selected package root or shelf of package
  roots. Its order matters. A support package's `provider.yaml` is separate from
  a candidate compiler's `manifest.yaml`.

```python
from merlin.api import inspect_backends

print(inspect_backends())
```

Selecting support metadata does not install a compiler service. Likewise, an
installed compiler is not an independent simulator or correctness reference.
Generated experiment candidates are selected by their run's compiler records;
they should not silently become the default installed backend.

## Keep facts and implementation claims separate

Hardware facts, backend capabilities, importer coverage and observed execution
are different evidence. A backend can propose an implementation only under the
selected numerical, memory and resource contract. It cannot loosen application
accuracy requirements to make its implementation acceptable.

RTL extraction uses the configured ModeLIR/CIRCT tools through
`targetgen/rtl/circt_introspect.py` and `introspect.py`. Start with an explicit
hardware source selection. Retain the reader/tool identities and unknown fields;
do not replace a missing fact with a guessed capacity or datatype.

## Open the implementation

| Responsibility | Source |
| --- | --- |
| Authored spec loading and expansion | [`target/spec/load.py`](../../src/merlin/target/spec/load.py) |
| Resource and memory topology | [`system/topology.py`](../../src/merlin/system/topology.py) |
| Installed compiler/runtime selection | [`backend/service.py`](../../src/merlin/compiler/backend/service.py) |
| Provider roles and selected resources | [`backend/load.py`](../../src/merlin/compiler/backend/load.py) |
| Target-definition search order | [`target_registry.py`](../../src/merlin/targetgen/target_registry.py) |
| Facts loading and validation | [`rtl/facts.py`](../../src/merlin/targetgen/rtl/facts.py) |
