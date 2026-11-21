---
title: 1. Choose an input and compile a small program
kind: guide
status: current
owner: compiler
last_verified: 2026-11-21
related: [02-targets, ../reference/repo_structure]
code_refs: [src/merlin/api.py, src/merlin/compiler/records.py, src/merlin/compiler/compilation.py, src/merlin/compiler/frontend/program.py, src/merlin/compiler/frontend/export.py, src/merlin/compiler/frontend/staged.py, src/merlin/compiler/frontend/intake/staged_projection.py, examples/compilation/quickstart]
---

# 1. Choose an input and compile a small program

[Guide index](../README.md) | [Next](02-targets.md)


Merlin compiles through an explicitly selected backend. Start with the small
layout example, then use the same Python interface for a kernel or model.

## Try the smallest example

Use Python 3.12 or newer and uv. From the repository root:

```sh
uv venv --python 3.12
uv pip install -e '.[compiler]'
uv run --no-sync merlin compile examples/compilation/quickstart/flatten.mlir \
  --backend merlin.layout --output out/build/quickstart/compiled
uv run --no-sync merlin inspect out/build/quickstart/compiled
```

Enter multiline shell commands on one line in PowerShell. Choose a fresh output
directory on each run. This example reshapes six integers and emits portable C;
it needs no Torch installation, accelerator or native MLIR build.

The same operation is available from Python:

```python
from pathlib import Path
from merlin.api import compile, inspect

result = compile(
    Path("examples/compilation/quickstart/flatten.mlir"),
    backend="merlin.layout",
    output="out/build/quickstart/python",
)
print(inspect(result))
```

A Python string means source text. Use `Path` for a filename. `inspect` checks
the recorded artifacts; execution is a separate step in [chapter 8](08-running.md).
The layout backend supports a small set of layout operations. For arithmetic,
select the scalar compiler or an OOT backend with the required implementation.

## Choose the input form

| Input | Public form | What Merlin needs |
| --- | --- | --- |
| Authored MLIR | `Path`, source text or `Source` | An input representation accepted by the selected backend |
| Saved PyTorch export | A `.pt2` `Path` or `SavedExport` | The configured capture/import tools; the archive carries its example inputs |
| PyTorch module, function or FX graph | Python object | `example_inputs=(...)` and a separate capture workspace |
| PyTorch `ExportedProgram` | Python object | Its recorded example inputs; omit `example_inputs` |
| Existing compiler stage | `StagedProgram` | The stage's recorded inputs and explicit backend acceptance of its preparation policy |

Framework imports are lazy. Working with MLIR does not require importing Torch.
A captured program retains its signature, constants, shape constraints and input
bindings. An FX graph still needs a supported export/import route; accepting the
Python object does not imply support for every operation inside it.

Capture and compilation use separate directories. For Python objects and `.pt2`
paths, `capture_output=` selects the capture workspace; its default is the
compilation output name with `.capture` appended. A `SavedExport` selects that
workspace through its `capture` field. A `StagedProgram` uses existing compiler
inputs and performs no capture or quantization. The original program stays
available when a different numerical preparation must be tried.

## Receipt-bound Triton input

`merlin.compiler.frontend.inputs.stage_compile_inputs(capture, output)` also
accepts the existing triton2MLIR standalone tensor/effect receipts and bound-model
receipts. It checks their original typed source and argument correspondence and
excludes evaluator inputs, golden outputs and execution reports. Provenance
describes derivation; it grants no numerical or hardware implementation verdict.

Select `result_mode="execution-owned"` explicitly for execution-produced result
descriptors. Input constraints and finite allocation limits remain checked. A
separate `precision_policy=` cannot override a Triton packet's source-bound policy.

Fresh standalone tensor/effect stages also work with `load_compile_inputs` and
`StagedProgram`. Their `source/intake.json` compiler projection replays the same
typed source, memory/effect and ABI builder before accepting the complete nested
member roster. Resealing a changed program, signature or policy does not establish
that correspondence. No capture/evaluator directory is opened during replay.

The generated `merlin.imported-source-policy.v1` records retention of imported
operations and an empty numerical-transform selection. A backend must explicitly
admit that policy through `validate_preparation`; the document authorizes no
numerical rewrite or hardware verdict. Runtime scalars, dynamic shapes,
execution-owned results and void effects keep their existing contracts. Public
preparation retains a standalone program byte-for-byte and publishes every nested
compiler member with its artifact identity. Restage older standalone captures to
obtain the replay projection. Bound-model staging still uses its existing adapter;
its generic public-stage integration remains separate work.

The repository-layout merge retains receipt schemas and standard MLIR payloads,
while changing Python owners:

| Previous owner | Current owner |
| --- | --- |
| `merlin.frontends.compile_inputs` | `merlin.compiler.frontend.inputs` |
| `merlin.frontends.linalg_mlir` | `merlin.compiler.frontend.linalg_mlir` |
| Receipt and shape helpers under `merlin.frontends` | `merlin.compiler.frontend.intake` |
| `merlin.common.mlir_query` | `merlin.compiler.ir.mlir_query` |

Already-externalized model2MLIR compiler constants retain their source/byte checks.
Their adapter emits typed argument bindings and a source-derived input order,
alongside `compiler-constants.json`; evaluator archives do not reconstruct those
bound constants. Other materialized captures retain the refactor's initializer
handling. The newer `StagedProgram` service separately requires an accepted
source-bound precision policy and its supported staging profile. Receipt staging
alone does not qualify a Triton packet for every backend or preparation service.

## Select options explicitly

`CompileOptions` carries `system`, `platform`, `harts`, and a `parameters` mapping.
Parameter names have a namespace, such as `frontend.max-initialization-bytes`.
The public frontend consumes `frontend.max-initialization-bytes` for captured or
staged inputs. The selected provider validates the remaining parameters before
compiling. An option accepted by one provider is not automatically meaningful to
every other provider.

The public CLI and Python API share the same service. Use `merlin --help` and
`merlin compile --help` for the options in this checkout.
[The package declarations](../../pyproject.toml) name installed commands and
backend entry points. Older saved-model workflows also retain their
specialized entry points; they are not aliases for every public compile mode.

## Open the implementation

| Question | Source to open |
| --- | --- |
| Where does a compile request enter? | [`api.py`](../../src/merlin/api.py) |
| What are the framework-independent input and option records? | [`compiler/records.py`](../../src/merlin/compiler/records.py) |
| How is a compile request validated and invoked? | [`compiler/compilation.py`](../../src/merlin/compiler/compilation.py) |
| How is a Python program frozen? | [`frontend/program.py`](../../src/merlin/compiler/frontend/program.py) |
| How is a saved export imported? | [`frontend/export.py`](../../src/merlin/compiler/frontend/export.py) |
| How is an existing compiler stage selected? | [`frontend/staged.py`](../../src/merlin/compiler/frontend/staged.py) |
