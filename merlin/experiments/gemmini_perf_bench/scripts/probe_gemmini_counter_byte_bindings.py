#!/usr/bin/env python3
"""Write Gemmini's fail-closed CIRCT counter-binding probe under generated outputs."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from merlin.common.paths import artifacts_dir


def _probe_module():
    from merlin.runtime.backends.base import get_backend

    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.counter_byte_bindings")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract fail-closed physical-counter candidates from Gemmini CIRCT HW.")
    parser.add_argument(
        "--output-json",
        default=str(artifacts_dir() / "perf-bench" / "gemmini" / "counter_byte_bindings.json"),
        help="generated artifact path (defaults below the configured artifacts directory)",
    )
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--run-differential", action="store_true",
                        help="run the complete 12-point RTL promotion campaign")
    stages.add_argument("--run-read-witness", action="store_true",
                        help="run one automatically derived read point; records but cannot promote")
    parser.add_argument("--rtl-facts", help="exact generated RTL facts JSON override")
    parser.add_argument("--simulator", choices=("verilator", "gsim"), default="verilator")
    parser.add_argument("--timeout", type=int, default=600, help="per-point RTL timeout in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_json).resolve()
    probe = _probe_module()
    if args.run_differential or args.run_read_witness:
        artifact = probe.run_differential_probe(
            facts_path=args.rtl_facts, simulator=args.simulator, timeout=args.timeout,
            workdir=output.parent / (output.stem + "_runs"),
            full_campaign=args.run_differential,
        )
    else:
        artifact = probe.probe_counter_byte_bindings()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # UNKNOWN is the expected honest current result, not a tool execution failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
