#!/usr/bin/env python3
"""Generate the compiler-independent Core ATen denominator and executable case corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from merlin.common.paths import artifacts_dir
from merlin.targetgen._aten_opset_worker import core_opset
from merlin.targetgen.core_aten_cases import corpus_document
from merlin.targetgen.core_aten_cover import denominator_document, json_bytes


def _documents() -> dict[str, bytes]:
    denominator = denominator_document(core_opset())
    corpus = corpus_document(denominator["overloads"], pytorch_version=denominator["pytorch_version"])
    corpus["denominator_sha256"] = denominator["overload_sha256"]
    return {
        "core_aten_ops.json": json_bytes(denominator),
        "core_aten_test_cases.json": json_bytes(corpus),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="artifact directory (defaults beneath the configured generated-artifact root)",
    )
    args = parser.parse_args(argv)
    destination = args.output_dir or artifacts_dir() / "verification" / "core-aten"

    first = _documents()
    second = _documents()
    if first != second:
        print("FAIL: Core ATen suite artifacts changed across identical reruns", file=sys.stderr)
        return 2
    destination.mkdir(parents=True, exist_ok=True)
    for name, payload in sorted(first.items()):
        (destination / name).write_bytes(payload)
    corpus = json.loads(first["core_aten_test_cases.json"])
    print(
        f"{corpus['case_count']}/{corpus['overload_count']} Core ATen overloads have "
        "deterministic executable source cases"
    )
    print(f"artifacts: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
