# Compile and run six values

This is the smallest complete Merlin compilation example. `flatten.mlir` changes
a dense `2x3xi32` tensor into `6xi32`, preserving storage bytes. The built-in
`merlin.layout` provider emits portable scalar C for this limited layout subset;
it does not compile arbitrary arithmetic or models.

From the repository root, with Python 3.12 or newer:

```sh
uv venv --python 3.12
uv pip install -e '.[compiler]'
uv run --no-sync merlin compile examples/quickstart/flatten.mlir \
  --backend merlin.layout --output out/build/quickstart/compiled
uv run --no-sync merlin inspect out/build/quickstart/compiled
```

No Torch, Triton, native MLIR tools, submodules or OOT checkout is required.
Use a fresh output directory. Compilation produces `compilation.json`,
`layout.json`, and a C source/header pair whose names bind the source hash.
Inspection checks the recorded artifact hashes; it does not execute the code.

The same service is callable from Python:

```python
from pathlib import Path
from merlin.api import compile

compilation = compile(Path("examples/quickstart/flatten.mlir"),
                      backend="merlin.layout", output="out/build/quickstart/python")
print(compilation.manifest)
```

For an actual run, use **Linux x86-64** and a host C compiler named `cc` on PATH
(or pass `--cc /path/to/clang`). Windows users can run this optional step in WSL2.
The script validates the manifest's C interface, builds its emitted source with
the checked-in `main.c`, and executes the resulting ELF through `merlin.api.run`:

```sh
uv run --no-sync python examples/quickstart/run.py out/build/quickstart/compiled \
  --output out/build/quickstart/native
```

Expected stdout is exactly:

```text
1 2 3 4 5 6
```

The native directory contains `flatten` and `run.json`. The driver checks all six
values independently. The runtime selects an explicit Linux deployment and one
initial affinity CPU; it does not promise exclusive CPU ownership or one thread.
This establishes this layout example's output on the executing host, without
claiming device execution or broader numerical qualification.

Open [api.py](../../src/merlin/api.py) for the shared service,
[backend/layout.py](../../src/merlin/compiler/backend/layout.py) for this provider,
and [lower/c/layout.py](../../src/merlin/compiler/lower/c/layout.py) for its emitted C algorithm.
For model capture and target workflows, continue with the [example catalog](../README.md).
