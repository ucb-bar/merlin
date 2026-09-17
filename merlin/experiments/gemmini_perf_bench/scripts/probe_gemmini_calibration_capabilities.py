#!/usr/bin/env python3
"""Emit Gemmini's target-owned, content-addressed calibration capabilities.

Only demonstrated native-emitter operations are reported.  DMA payload coordinates come from the
target's generated header and are advertised only after independent read/write/copy emitter probes;
observed cache state and physical traffic remain UNKNOWN until an RTL measurement supplies them.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path


def _probe_module():
    """Load the configured Gemmini plugin, then its target-owned probe module.

    The experiment is allowed to name the target it measures; generic performance code is not.
    Resolving through the runtime registry keeps this CLI on the exact OOT/reference package selected
    for the active target rather than importing a second copy by filesystem path.
    """
    from merlin.runtime.backends.base import get_backend

    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.calibration_capabilities")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Gemmini native emitter/compiler calibration capabilities from RTL facts.")
    parser.add_argument("--stage", choices=("emission", "compile"), default="emission",
                        help="emission is toolchain-light; compile additionally lowers every probe")
    parser.add_argument("--rtl-facts", help="optional exact CIRCT facts JSON override")
    parser.add_argument("--dma-transfer-size", action="append", type=int, default=[], metavar="BYTES",
                        help=("exact DMA payload size to probe; repeatable. If absent, the target probe "
                              "derives the required ladder from its generated header capability"))
    parser.add_argument("--output-json", required=True, help="exact capability artifact output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output_json).resolve()
    if args.rtl_facts and output == Path(args.rtl_facts).resolve():
        _parser().error("output path must be distinct from the explicit RTL facts input")
    probe = _probe_module()
    artifact = probe.probe_calibration_capabilities(
        stage=args.stage, facts_path=args.rtl_facts,
        dma_transfer_sizes=tuple(args.dma_transfer_size))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if artifact.get("status") in {"complete", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
