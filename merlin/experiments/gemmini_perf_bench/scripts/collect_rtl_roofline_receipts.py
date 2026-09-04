#!/usr/bin/env python3
"""Write fail-closed roofline receipts below the configured generated-output root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.paths import out_dir
from merlin.perf.receipt_bridge import build


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fail-closed raw roofline receipts from exact RTL runner evidence.")
    parser.add_argument("--rtl-facts", required=True)
    parser.add_argument("--campaign-manifest", required=True)
    parser.add_argument("--calibration-results", required=True)
    parser.add_argument("--workload-results", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-calibration", required=True)
    parser.add_argument("--output-observations", required=True)
    return parser


def _under_output_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(out_dir().resolve())
    except ValueError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_paths = {Path(getattr(args, name)).resolve() for name in (
        "rtl_facts", "campaign_manifest", "calibration_results", "workload_results")}
    output_paths = {Path(getattr(args, name)).resolve() for name in (
        "output_report", "output_calibration", "output_observations")}
    if input_paths & output_paths or len(output_paths) != 3:
        _parser().error("all explicit input and output paths must be distinct")
    if not all(_under_output_root(path) for path in output_paths):
        _parser().error(
            f"generated products must be below the configured output root {out_dir().resolve()}")
    report, calibration, observations, status = build(
        Path(args.rtl_facts), Path(args.campaign_manifest), Path(args.calibration_results),
        Path(args.workload_results))
    for path, value in ((Path(args.output_report), report),
                        (Path(args.output_calibration), calibration),
                        (Path(args.output_observations), observations)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
