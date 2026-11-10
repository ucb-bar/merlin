---
title: 1. Choose an input and compile a small program
kind: guide
status: current
owner: compiler
last_verified: 2026-11-09
related: [02-targets, ../reference/repo_structure]
code_refs: [src/merlin/api.py, src/merlin/compiler/request.py, src/merlin/compiler/frontend/torch/program.py, src/merlin/compiler/frontend/torch/export.py, examples/compilation/quickstart]
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
| Existing compiler stage | `StagedProgram` | The stage's recorded inputs and explicit backend acceptance of its preparation policy |

Framework imports are lazy. Working with MLIR does not require importing Torch.
A captured program retains its signature, constants, shape constraints and input
bindings. An FX graph still needs a supported export/import route; accepting the
Python object does not imply support for every operation inside it.

Capture and compilation use separate directories. The default capture location is
the compilation output name with `.capture` appended. The original program stays
available when a different numerical preparation must be tried.

## Select options explicitly

`CompileOptions` carries `system`, `platform`, `harts`, and a `parameters` mapping.
Parameter names have a namespace, such as `frontend.max-initialization-bytes`.
Each provider validates its own supported options before compiling. An option
accepted by one provider is not automatically meaningful to every other provider.

The public CLI and Python API share the same service. Use `merlin --help`,
`merlin compile --help`, and [the command declarations](../../pyproject.toml) to find
the options in this checkout. Older saved-model workflows also retain their
specialized entry points; they are not aliases for every public compile mode.

## Open the implementation

| Question | Source to open |
| --- | --- |
| Where does a compile request enter? | [`api.py`](../../src/merlin/api.py) |
| What are the framework-independent input and option records? | [`compiler/request.py`](../../src/merlin/compiler/request.py) |
| How is a Python program frozen? | [`frontend/torch/program.py`](../../src/merlin/compiler/frontend/torch/program.py) |
| How is a saved export imported? | [`frontend/torch/export.py`](../../src/merlin/compiler/frontend/torch/export.py) |
