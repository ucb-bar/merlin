#!/usr/bin/env python3
"""Evaluate all Core ATen cases through one external compiler/target adapter invocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.paths import artifacts_dir
from merlin.targetgen.core_aten_eval import (
    JsonCommandSuiteAdapter,
    evaluate_core_aten_suite,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-name", required=True)
    parser.add_argument("--execution-kind", default="hardware")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "adapter_command",
        nargs=argparse.REMAINDER,
        help="command containing required {request} and {response} placeholders (precede with --)",
    )
    args = parser.parse_args(argv)
    command = list(args.adapter_command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("an adapter command is required after --")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8")) if args.corpus else None
    adapter = JsonCommandSuiteAdapter(
        argv=tuple(command),
        name=args.adapter_name,
        timeout_seconds=args.timeout,
    )
    report = evaluate_core_aten_suite(
        adapter,
        required_execution_kind=args.execution_kind,
        corpus=corpus,
    )
    safe_name = "".join(character if character.isalnum() else "-" for character in args.adapter_name)
    destination = args.output or (artifacts_dir() / "verification" / "core-aten" / f"semantic-{safe_name}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{report['passed_count']}/{report['case_count']} Core ATen cases passed on "
        f"{report['required_execution_kind']} via {report['adapter']}"
    )
    print(f"report: {destination}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
