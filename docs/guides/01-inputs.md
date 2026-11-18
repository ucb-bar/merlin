---
title: 1. Choose an input and compile a small program
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [02-targets, ../reference/repo_structure]
code_refs: [src/merlin/api.py, src/merlin/compiler/records.py, src/merlin/compiler/compilation.py, src/merlin/compiler/frontend/program.py, src/merlin/compiler/frontend/export.py, src/merlin/compiler/frontend/staged.py, examples/compilation/quickstart]
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
