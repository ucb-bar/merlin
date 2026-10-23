#!/usr/bin/env python3
"""Run the canonical Core ATen cases through model2MLIR and record each frontend result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.paths import artifacts_dir
from merlin.targetgen.core_aten_capture import capture_case_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None, help="generated core_aten_test_cases.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="per-case capture result tree")
    parser.add_argument("--overload", action="append", default=[], help="capture only this exact overload")
    parser.add_argument("--m2m-dir", type=Path, default=None, help="model2MLIR checkout")
    parser.add_argument("--m2m-python", type=Path, default=None, help="model2MLIR PyTorch interpreter")
    args = parser.parse_args(argv)

    artifact_root = artifacts_dir() / "verification" / "core-aten"
    corpus_path = args.corpus or artifact_root / "core_aten_test_cases.json"
    destination = args.output_dir or artifact_root / "canonical-case-captures"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    requested = set(args.overload) if args.overload else None
    report = capture_case_corpus(
        corpus,
        destination,
        overloads=requested,
        m2m_dir=args.m2m_dir,
        python=args.m2m_python,
    )
    report_path = destination / "capture_report.json"
    if requested and report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        merged = {**previous.get("results", {}), **report["results"]}
        counts: dict[str, int] = {}
        for record in merged.values():
            status = record["status"]
            counts[status] = counts.get(status, 0) + 1
        report = {
            "schema_version": 1,
            "case_count": len(merged),
            "status_counts": dict(sorted(counts.items())),
            "results": dict(sorted(merged.items())),
        }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["status_counts"], sort_keys=True))
    print(f"report: {report_path}")
    return 0 if report["status_counts"].get("capture_failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
