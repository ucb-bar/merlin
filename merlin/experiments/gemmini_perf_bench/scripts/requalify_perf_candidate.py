#!/usr/bin/env python3
"""Requalify an immutable performance candidate refused only by a corrected transcript audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import perf_agent_stage as PAS
from merlin.targetgen.target_experiment import load_target_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-record", type=Path, required=True)
    parser.add_argument("--output-record", type=Path, required=True,
                        help="record path beneath a fresh output directory")
    parser.add_argument(
        "--descriptor", type=Path,
        help="target descriptor; defaults to the exact descriptor named by the source record")
    args = parser.parse_args(argv)
    try:
        source = json.loads(args.source_record.read_text(encoding="utf-8"))
        descriptor = args.descriptor or Path(source["target"]["descriptor"])
        target = load_target_experiment(descriptor)
        record = PAS.requalify_audit_only_candidate(
            args.source_record, args.output_record, target)
    except (OSError, KeyError, ValueError, PAS.StageGateError) as exc:
        print(f"NO-GO: {exc}")
        return 2
    print(f"SEALED: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
