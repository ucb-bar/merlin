"""`./merlin run` — drive a compiled model on a target board.

Dispatches to one of the executors under `tools/run/`. Each mode is a
self-contained script with its own argparse; this shim passes remaining
arguments straight through, so:

    ./merlin run <mode> --foo --bar

is equivalent to:

    uv run python tools/run/<mode>.py --foo --bar

Modes:
    schedule       — execute a multi-model schedule on the QRB5165 (or any
                     aarch64-linux board).
    multi-device   — drive the merlin_multi_device_runner on the QRB5165
                     from the host (one process per device).
    het-e2e        — end-to-end heterogeneous schedule runner for QRB5165.
    het-matrix     — sweep `het-e2e` across (model, granularity) cells.
    full-loop      — end-to-end XPU-RT loop: profile -> schedule -> run ->
                     fold -> repeat until convergence.
    roundtrip      — round-trip a compiled model through a remote board.
"""

from __future__ import annotations

import argparse
import sys

import utils

_MODE_TO_SCRIPT = {
    "schedule": "tools/run/schedule.py",
    "multi-device": "tools/run/multi_device.py",
    "het-e2e": "tools/run/het_e2e.py",
    "het-matrix": "tools/run/het_matrix.py",
    "full-loop": "tools/run/full_loop.py",
    "roundtrip": "tools/run/roundtrip.py",
}


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "mode",
        choices=sorted(_MODE_TO_SCRIPT.keys()),
        help="Which board-execution flow to run. See module docstrings under tools/run/ for per-mode flags.",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded verbatim to the underlying script.",
    )


def main(args: argparse.Namespace) -> int:
    script = utils.REPO_ROOT / _MODE_TO_SCRIPT[args.mode]
    if not script.exists():
        utils.eprint(f"run mode '{args.mode}' not found at {script}")
        return 2
    return utils.run([sys.executable, str(script), *args.passthrough])
