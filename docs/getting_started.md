# Getting Started

This page is the fastest path for first-time Merlin usage:

1. Create/activate environment.
2. Build host compiler tools.
3. Compile one model.
4. Build one sample/runtime binary.
5. Locate generated outputs.

## 1) Environment

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

## 2) Build host compiler tools

```bash
conda activate merlin-dev
uv run tools/merlin.py build --target host --config release
```

Expected tool locations:

- `build/host-vanilla-release/install/bin/`
- `build/host-merlin-release/install/bin/` (when built with `--with-plugin`)

## 3) Compile one model

```bash
conda activate merlin-dev
uv run tools/merlin.py compile models/dronet/dronet.mlir --target spacemit_x60
```

Expected output:

- `build/compiled_models/dronet/spacemit_x60_RVV_dronet/dronet.vmfb`

## 4) Build one sample binary

```bash
conda activate merlin-dev
uv run tools/merlin.py build --target spacemit --config release --with-plugin --cmake-target merlin_baseline_dual_model_async_run
find build/spacemit-merlin-release -name baseline-dual-model-async-run
```

Typical binary path:

- `build/spacemit-merlin-release/samples/SpacemiTX60/baseline_dual_model_async/baseline-dual-model-async-run`

## 5) Where outputs go

- Host tools: `build/host-*/install/bin/`
- Cross builds: `build/spacemit-*`, `build/firesim-*`
- Compiled models/artifacts: `build/compiled_models/<model>/...`
- Local rendered docs: `site/`

## Next

- Repository layout and placement conventions: `repository_guide.md`
- Full command and API reference: `reference/`
- Setup details and prerequisites: `iree_setup.md`
