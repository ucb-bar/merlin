# Merlin Compiler Infrastructure

Merlin is an MLIR/IREE-based compiler stack for compiling models to CPU and custom RISC-V targets (including SpacemiT and Saturn OPU flows).

<p align="center">
  <img src="docs/assets/merlin_transparent.png" width="400">
</p>

## Quick Start

### 1) Environment

```bash
conda env create -f env_linux.yml
conda activate merlin-dev
uv sync
pre-commit install
```

Optional convenience alias for your shell session:

```bash
alias merlin='uv run tools/merlin.py'
```

### 2) Build host compiler tools

```bash
conda activate merlin-dev
uv run tools/merlin.py build --target host --config release
```

This creates host tools under:

- `build/host-vanilla-release/install/bin/`
- or `build/host-merlin-release/install/bin/` when using `--with-plugin`

### 3) Compile one model with `compile.py`

```bash
conda activate merlin-dev
uv run tools/merlin.py compile models/dronet/dronet.mlir --target spacemit_x60
```

Expected output artifact:

- `build/compiled_models/dronet/spacemit_x60_RVV_dronet/dronet.vmfb`

### 4) Build one runtime/sample binary

```bash
conda activate merlin-dev
uv run tools/merlin.py build --target spacemit --config release --with-plugin --cmake-target merlin_baseline_dual_model_async_run
find build/spacemit-merlin-release -name baseline-dual-model-async-run
```

Typical binary location:

- `build/spacemit-merlin-release/samples/SpacemiTX60/baseline_dual_model_async/baseline-dual-model-async-run`

## Where Build Outputs Go

- Host tool binaries: `build/host-*/install/bin/`
- Cross-target build trees: `build/spacemit-*`, `build/firesim-*`
- Compiled model artifacts (`.mlir`, `.vmfb`, optional dumps): `build/compiled_models/<model>/...`
- Generated docs site (local): `site/`

## Repository Map (Where To Put Things)

```text
merlin/
├── tools/              # Main developer CLIs (build.py, compile.py, setup.py, ci.py)
├── models/             # Model sources and target YAML configs (add new model flows here)
├── compiler/
│   ├── src/merlin/     # Merlin MLIR dialects/passes/codegen
│   └── plugins/        # Merlin IREE plugin registration/target glue
├── samples/            # Runtime/sample applications and board-specific executables
│   ├── common/         # Samples for that are not dependant on target
│       └── AppName0    # Your application or example name
│   ...
│   └── TargetName/     # Samples built for a specifc target usecase
│       └── AppName1    # Your application that is dependant on the compilation for that target
├── benchmarks/         # Benchmark scripts and profiling workflows
│   ├── BenchName/      # Name of an interesting third party benchmark
│   ...
│   └── TargetName/     # Benchmarks you create for your specific target
├── build_tools/        # Toolchains, target support scripts, patch-related helpers
├── docs/               # MkDocs source for architecture + generated reference
├── third_party/        # Submodules (IREE fork, turbine, llama.cpp, shark_ai, ...)
└── build/              # Local build outputs and generated artifacts
```

Detailed folder guide with auto-generated tracked tree:

- [docs/repository_guide.md](docs/repository_guide.md)

## If You Want To Extend Merlin

- New compiler pass/dialect: add under `compiler/src/merlin/...`
- New plugin/target wiring: add under `compiler/plugins/...`
- New model path/config: add model files in `models/<model>/` and flags in `models/<target>.yaml`
- New sample runtime app: add under `samples/<platform>/...`
- New benchmark flow: add under `benchmarks/<target>/...`

## Documentation

Primary docs:

- [docs/index.md](docs/index.md)
- [docs/getting_started.md](docs/getting_started.md)
- [docs/iree_setup.md](docs/iree_setup.md)
- [docs/architecture/plugin_and_patch_model.md](docs/architecture/plugin_and_patch_model.md)
- [docs/architecture/cmake_presets.md](docs/architecture/cmake_presets.md)

Build docs locally:

```bash
conda activate merlin-dev
MLIR_TBLGEN=build/host-vanilla-release/llvm-project/bin/mlir-tblgen \
  uv run --with-requirements docs/requirements.txt mkdocs serve
```

Unified CLI help:

```bash
conda activate merlin-dev
uv run tools/merlin.py --help
```
Published docs URL (after GitHub Pages is enabled): `https://ucb-bar.github.io/merlin/`

Then open `http://127.0.0.1:8000`.

## Formatting and Checks

```bash
conda activate merlin-dev
pre-commit run --all-files
uv run tools/merlin.py ci lint
```

## Contributing

- [CONTRIBUTING.md](CONTRIBUTING.md)
