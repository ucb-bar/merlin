# tools/

Python developer entrypoints behind `./merlin <subcommand>`. If you are using
Merlin you almost certainly want the `./merlin` wrapper at the repo root, not
to invoke these scripts directly.

- `merlin.py` — the unified CLI dispatcher.
- `build.py`, `compile.py`, `setup.py`, `ci.py`, `patches.py`, `benchmark.py`,
  `chipyard.py`, `ray_cmd.py`, `targetgen_cmd.py` — per-subcommand backends.
- `utils.py` — shared helpers (`run`, `eprint`, path resolution, target
  config loading).
- `targetgen/`, `raycp/` — package-style modules for larger subsystems.
- `analyze_quant_ir.py`, `install_prebuilt.py`, `strip_mlir_weights.py` —
  focused one-off helpers; not exposed as `./merlin` subcommands.

For the canonical CLI reference see [`docs/reference/cli.md`](../docs/reference/cli.md).
