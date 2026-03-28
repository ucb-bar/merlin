# Getting Started

This page is the fastest path for first-time Merlin usage:

1. Select the Merlin branch and create/activate the environment.
2. Sync core submodules for that branch.
3. Build host compiler tools.
4. Compile one model.
5. Build one sample/runtime binary.
6. Locate generated outputs.

If this is your first time in the repo, you can ignore most of the tree.
Start with `tools/`, `models/`, `docs/`, and the generated `build/` outputs.
Use [user_paths.md](user_paths.md) if you want the repo explained by workflow
instead of by folder name.

## 1) Environment

```bash
git checkout dev/main
conda env create -f env_linux.yml
conda activate merlin-dev
uv sync
pre-commit install
```

Optional convenience alias for your shell session:

```bash
alias merlin='uv run tools/merlin.py'
```

## 2) Sync core submodules

```bash
conda activate merlin-dev
uv run tools/merlin.py setup submodules --submodules-profile core --submodule-sync
```

Submodule SHAs follow the current Merlin commit. If you switch branches later,
rerun the same setup command before building.

## 3) Build host compiler tools

```bash
conda activate merlin-dev
uv run tools/merlin.py build --profile vanilla
```

Expected tool locations:

- `build/host-vanilla-release/install/bin/`
- `build/host-merlin-release/install/bin/` (when built with `--with-plugin`)

## 4) Compile one model

```bash
conda activate merlin-dev
uv run tools/merlin.py compile models/dronet/dronet.mlir --target spacemit_x60
```

Expected output:

- `build/compiled_models/dronet/spacemit_x60_RVV_dronet/dronet.vmfb`

## 5) Build one sample binary

```bash
conda activate merlin-dev
uv run tools/merlin.py build --profile spacemit --cmake-target merlin_baseline_dual_model_async_run
find build/spacemit-merlin-release -name baseline-dual-model-async-run
```

Typical binary path:

- `build/spacemit-merlin-release/runtime/plugins/merlin-samples/SpacemiTX60/baseline_dual_model_async/baseline-dual-model-async-run`

## 6) Where outputs go

- Host tools: `build/host-*/install/bin/`
- Cross builds: `build/spacemit-*`, `build/firesim-*`
- Compiled models/artifacts: `build/compiled_models/<model>/...`
- Local rendered docs: `site/`

## Next

- Practical extension guides (dialects/HAL/samples/targets): `how_to/index.md`
- User-first repo navigation: `user_paths.md`
- Repository layout and placement conventions: `repository_guide.md`
- Active implementation notes and debugging logs: `dev_blog/index.md`
- Full command and API reference: `reference/`
- Setup details and prerequisites: `iree_setup.md`
