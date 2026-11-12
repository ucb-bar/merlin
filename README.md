<p align="center">
  <img src="docs/assets/merlin_transparent.png" width="360" alt="merlin logo">
</p>

<h1 align="center">merlin</h1>

Merlin is a compiler-generation framework. It provides shared transformations,
explicit out-of-tree backend interfaces, and independent evaluation for developing
target compilers. APIs are under active development.

## First compile

Use Python **3.12 or newer**. A plain clone is enough; the first example needs no
submodules, Torch, target checkout, or native MLIR toolchain.

```sh
uv venv --python 3.12
uv pip install -e '.[compiler]'
uv run --no-sync merlin --help
uv run --no-sync merlin compile examples/compilation/quickstart/flatten.mlir \
  --backend merlin.layout --output out/build/quickstart/compiled
uv run --no-sync merlin inspect out/build/quickstart/compiled
```

The output directory must be fresh. This example reshapes six `i32` values from
`2x3` to `6`, preserving their bytes. It emits C, a C header, `layout.json`, and
`compilation.json`. The built-in `merlin.layout` provider supports this small layout
subset; it is not a general model compiler. On Linux x86-64, an optional C compiler
step exercises the runtime and prints `1 2 3 4 5 6`; see
[the complete example](examples/compilation/quickstart/README.md).

The Python API uses the same compilation service:

```python
from pathlib import Path
from merlin.api import compile

result = compile(Path("examples/compilation/quickstart/flatten.mlir"), backend="merlin.layout",
                 output="out/build/quickstart/python")
print(result.manifest)
```

`merlin compile` accepts an explicit MLIR source, saved `.pt2` export, or admitted
compiler-stage directory. The Python API also accepts supported PyTorch modules,
FX graphs and functions with example inputs. Those routes require their selected
capture/import dependencies and a provider that admits the resulting program.
See [setup and prerequisites](docs/guides/01-inputs.md).

## Choose a workflow

| Task | Start here |
| --- | --- |
| Compile and run a small input | [First example](examples/compilation/quickstart/README.md) |
| Explore specs, RTL facts, test generation, scheduling, transformations and CCA | [Interactive tutorials](examples/tutorials/README.md) |
| Compile a model or use a target | [Examples](examples/README.md), [setup](docs/guides/01-inputs.md) |
| Add a compiler transformation | [Contributor entry points](docs/guides/11-optimization.md), [shared pass library](docs/guides/12-experiments.md) |
| Add or select an OOT backend | [Target integration](docs/guides/02-targets.md), [backend resolution](docs/guides/02-targets.md) |
| Run agentic experiments | [Experiment catalog](experiments/README.md) |

Backend, system, platform, and runner are separate selections. A **backend** is a
compiler provider; a **system** selects hardware; a **platform** selects an image
environment such as bare metal or Zephyr; a **runner** handles existing artifacts.
Providers validate the selections and options they actually support. Use
`merlin inspect --backends` for installed provider metadata and
`merlin compile --help` / `merlin run --help` for the shared service arguments.
The retained `merlin-compile` command is the older workload/operator adapter;
its options are documented by `merlin-compile --help`.

## Find the code

There is one Python namespace, `merlin`, beneath `src/merlin/`. The `src/` layout
keeps imports tied to the installed package instead of accidentally importing a
checkout directory. Repository inputs and generated results stay outside it.

| Path | Responsibility |
| --- | --- |
| `src/merlin/compiler/` | Backend protocols, planning, quantization, lowering and build orchestration |
| `src/merlin/transforms/`, `src/merlin/sched/` | Shared transformations and scheduling primitives |
| `src/merlin/runtime/`, `src/merlin/verification/` | Artifact execution and independent verification, with separate contracts |
| `src/merlin/development/` | Test generation and backend authoring engines |
| `src/merlin/experiments/`, `src/merlin/analysis/` | Experiment selection, tracking and analysis |
| `src/merlin/resources/` | Installed schemas, prompts, public contracts and benchmark definitions |
| `examples/` | Small walkthroughs and target-owned public inputs |
| `experiments/` | Nine authored catalog, definition, template and instruction files |
| `tools/` | Standalone console and independent timing observer |
| `tests/` | Compact permanent regression suite, capped at 150 collected cases |
| `docs/`, `build_tools/` | Guides and references; build, release and repository checks |
| `out/` | Ignored generated builds, reviewed corpora, input bundles, captures, caches and runs |

The [repository map](docs/reference/repo_structure.md) gives the complete source
flow, including active legacy owners. Generic scalar and control generation live
in Merlin; RVV and accelerator implementations belong in their OOT repositories.
The [contributing guide](CONTRIBUTING.md) explains ownership, module size and review.

Hardware workflows require explicitly selected toolchains, support and execution
inputs. FireSim execution is external: Merlin prepares files and collects results;
it does not provision or launch FireSim. Listing experiments and inspecting
manifests do not establish numerical correctness or observed device execution.
Historical results apply to their recorded revisions and prerequisites.

## License

Merlin is Apache-2.0 except for components with their own notices. See [LICENSE](LICENSE)
and [NOTICE](NOTICE). The optional Atlas LSTMNetViT loader retains its upstream
commercial-use restriction. Logos and artwork have separate terms in LICENSE.
