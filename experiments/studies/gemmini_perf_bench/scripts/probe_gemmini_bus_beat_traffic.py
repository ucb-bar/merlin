#!/usr/bin/env python3
"""Write the CIRCT-derived physical-bus measurement feasibility report."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from merlin.common.paths import artifacts_dir


def _probe_module():
    from merlin.runtime.backends.base import get_backend

    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.bus_beat_probe")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Derive beat-monitor candidates from CIRCT and report whether the selected "
                     "prebuilt RTL simulator can measure them without semantic guesses."))
    parser.add_argument(
        "--output-json",
        default=str(artifacts_dir() / "perf-bench" / "gemmini" / "bus_beat_probe.json"),
        help="generated artifact path (defaults below the configured artifacts directory)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_json).resolve()
    artifact = _probe_module().probe_bus_beat_traffic()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # UNKNOWN is an honest capability result, not a tool failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
