# Learn Merlin

Start with **1. Contracts and facts**, then follow the notebooks in order. Each
notebook runs on its own; the order builds the explanation, not shared Python state.

| Step | Notebook | What you will do |
| --- | --- | --- |
| 1 | [Contracts and facts](01_contracts.ipynb) | Read a software spec and inspect facts from a small RTL example. |
| 2 | [Test generation](02_test_generation.ipynb) | Choose coverage requirements and generate test inputs and IR. |
| 3 | [Scheduling](03_scheduling.ipynb) | Change loop and storage choices and see which schedules are legal. |
| 4 | [Transformations](04_transformations.ipynb) | Compose passes, inspect changed IR and check the result. |
| 5 | [CCA](05_cca.ipynb) | Compare compiler output with a reference and find possible compiler edits. |

## Open the notebooks

From the repository root, with Python 3.12 or newer and uv:

```sh
uv run --frozen --extra compiler --with 'jupyterlab>=4,<5' \
  --with 'ipywidgets>=8.1,<9' jupyter lab examples/tutorials
```

In PowerShell, enter the command on one line. Open `01_contracts.ipynb` and choose
**Run All Cells**. Then try the controls or edit the small Python examples.
Use the Python kernel supplied by this command so it can import Merlin.

For VS Code, select the Python environment containing your editable Merlin
installation, with `ipykernel` and `ipywidgets` installed there. These notebook
packages are optional; they do not change Merlin's core dependencies or lockfile.
The [uv Jupyter guide](https://docs.astral.sh/uv/guides/integration/jupyter/)
explains kernel selection.

The notebooks also expose ordinary functions you can call directly. GitHub shows
a static preview: live controls need a running Python kernel.

## Read more

The guides follow the same order:
[contracts and facts](../../docs/guides/target_contracts.md),
[test generation](../../docs/guides/test_generation.md),
[scheduling](../../docs/guides/scheduling.md),
[transformations](../../docs/guides/transformations.md), and
[CCA](../../docs/guides/cca.md).

All default examples run without a board, simulator, Torch model or native MLIR
build. They call Merlin's actual parsers, generators, passes and checkers using
small example inputs. Checking these examples is different from checking your
accelerator: the guides explain what RTL, tools and execution results you need
for that next step.

## Run all five without a browser

```sh
uv run --frozen --extra compiler --with 'nbclient>=0.11,<0.12' \
  --with 'ipykernel>=7,<8' --with 'ipywidgets>=8.1,<9' \
  python build_tools/scripts/check_tutorials.py --output out/build/python/tutorials
```

This runs each notebook once in that Python interpreter. It saves executed copies
and `report.json` in the output directory and leaves the source notebooks alone.
Cell errors and widget callback errors make the command fail. CI runs these same
five examples; it does not sweep every control combination.
