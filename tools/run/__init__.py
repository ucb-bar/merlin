"""Implementation package for the `run` subcommand (`./merlin run <mode>`).

The registered shim is `tools/run/cli.py`; this package holds one module
per mode. Each mode is a self-contained driver with its own argparse.

Modes (extension points):

- `schedule.py` — execute a multi-model `combined_schedule.json` on an
  aarch64 board by spawning iree-run-module per instance with taskset
  pinning. Extend for new scheduler output formats or pin strategies.
- `multi_device.py` — drive merlin_multi_device_runner (one IREE device
  per `--cluster`). Extend for new device-group topologies.
- `het_e2e.py` — end-to-end heterogeneous schedule runner:
  compile-per-target → push → run → fold. Extend for new transports or
  scheduler integrations.
- `het_matrix.py` — sweep `het_e2e` across (model, granularity, target)
  cells. Extend for new sweep dimensions.
- `full_loop.py` — profile → schedule → run → fold → repeat until
  convergence. The big driver. Extend for new convergence criteria.
- `roundtrip.py` — single-shot compile → board run → timing capture →
  manifest. Extend for new round-trip artifact formats.

External paths come from env vars: `MERLIN_XPU_RT_ROOT`, `MERLIN_BOARD_HOST`,
`MERLIN_BOARD_SSH_KEY`. The `run/cli.py:_MODE_TO_SCRIPT` dict is the
mode-discovery surface — a new file here must be registered there to be
reachable via `./merlin run`.
"""
