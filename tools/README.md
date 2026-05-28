# tools/

Python developer entrypoints behind `./merlin <subcommand>`. If you are using
Merlin you almost certainly want the `./merlin` wrapper at the repo root, not
to invoke these scripts directly.

## Layout

- `merlin.py` — unified CLI dispatcher.
- `utils.py` — shared helpers (`run`, `eprint`, `REPO_ROOT`, target config
  loading, toolchain-binary resolution).
- `archive/` — frozen scripts from past investigations / migrations.

**Single-file subcommands** (small, no co-resident concerns):

- `benchmark.py`, `ci.py`, `patches.py`, `setup.py`, `sim.py`, `spike.py`.

**Subcommand packages** (each has `cli.py` as the entry + topic helpers):

- `build/` — configure + build host/cross targets.
- `compile/` — MLIR/ONNX → VMFB pipeline + kernel embedding.
- `chipyard/` — Chipyard hardware-backend orchestration (9 modules).
- `coverage/` — accelerator coverage check for VMFBs.
- `mcp/` — MCP servers (`./merlin mcp <name>` + 6 per-domain registries).
- `perf/` — per-dispatch performance decomposition + plotters.
- `quantize/` — INT8 quantization helpers.
- `ray/` — Ray control plane (jobs, resources, artifacts).
- `run/` — execute compiled models on a target board (6 modes).
- `targetgen/` — TargetGen planner framework + 13 cli subactions + MCP.
- `verify/` — output cross-hash verification.

**Supporting infrastructure** (not subcommands):

- `kernels/` — kernel-embedding pipeline (manifests, precompile, QNN
  emitter + 14 recognizers). Imported by `compile/qnn.py`.
- `spike-hetero/` — C++/Make build artifact for the Saturn-OPU + Gemmini
  spike extension.

## Pattern

See `docs/architecture/tools_architecture.md` for the canonical write-up:
*one subcommand = one package, `cli.py` is the entry, helpers as siblings.*

For the CLI reference see [`docs/reference/cli.md`](../docs/reference/cli.md).
