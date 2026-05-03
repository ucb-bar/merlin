#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Smoke-tests cuda_new against existing cuda_tile VMFB artifacts.

This is intentionally a developer harness, not a hermetic test. It consumes
artifacts produced by the Python/Torch-to-cuda_tile export flow and verifies
that cuda_new can run the same VMFBs with the same textual result as cuda_tile.
"""

from __future__ import annotations

import argparse
import difflib
import pathlib
import shlex
import subprocess
import sys


DEFAULT_CASES = (
    "linear",
    "linear_bias_relu",
    "softmax",
    "alexnet_nonuniform",
)


def _read_reference_command(case_dir: pathlib.Path) -> list[str]:
    cmd_file = case_dir / "run.stdout.cmd.txt"
    if not cmd_file.exists():
        raise FileNotFoundError(f"missing command file: {cmd_file}")
    first_line = cmd_file.read_text().splitlines()[0]
    if not first_line:
        raise ValueError(f"empty command file: {cmd_file}")
    return shlex.split(first_line)


def _rewrite_for_cuda_new(cmd: list[str], iree_run_module: pathlib.Path) -> list[str]:
    rewritten = [str(iree_run_module)]
    for arg in cmd[1:]:
        if arg == "--device=cuda_tile":
            rewritten.append("--device=cuda_new")
        else:
            rewritten.append(arg)
    return rewritten


def _normalize_stdout(text: str) -> list[str]:
    return [line.rstrip() for line in text.strip().splitlines()]


def _run_case(
    case: str,
    artifacts_dir: pathlib.Path,
    iree_run_module: pathlib.Path,
    verbose: bool,
) -> bool:
    case_dir = artifacts_dir / case
    reference_stdout_path = case_dir / "run.stdout.txt"
    if not reference_stdout_path.exists():
        print(f"[{case}] FAIL: missing reference stdout: {reference_stdout_path}")
        return False

    try:
        reference_cmd = _read_reference_command(case_dir)
        cmd = _rewrite_for_cuda_new(reference_cmd, iree_run_module)
    except Exception as exc:  # noqa: BLE001 - show actionable harness errors.
        print(f"[{case}] FAIL: {exc}")
        return False

    if verbose:
        print(f"[{case}] RUN: {' '.join(shlex.quote(arg) for arg in cmd)}")

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print(f"[{case}] FAIL: exit code {result.returncode}")
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        return False

    actual = _normalize_stdout(result.stdout)
    expected = _normalize_stdout(reference_stdout_path.read_text())
    if actual != expected:
        print(f"[{case}] FAIL: stdout differs from cuda_tile reference")
        diff = difflib.unified_diff(
            expected,
            actual,
            fromfile=f"{case}/cuda_tile",
            tofile=f"{case}/cuda_new",
            lineterm="",
        )
        for line in list(diff)[:80]:
            print(line)
        return False

    print(f"[{case}] PASS")
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/codex_torch_cuda_tile_exports"),
        help="Directory containing per-case cuda_tile export artifacts.",
    )
    parser.add_argument(
        "--iree-run-module",
        type=pathlib.Path,
        default=pathlib.Path(
            "/scratch/ashvin/merlin/build/host-merlin-release/tools/iree-run-module"
        ),
        help="iree-run-module binary to use.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Case to run. May be repeated. Defaults to core smoke cases.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print rewritten cuda_new commands before running.",
    )
    args = parser.parse_args(argv)

    cases = tuple(args.cases) if args.cases else DEFAULT_CASES
    if not args.iree_run_module.exists():
        print(f"missing iree-run-module: {args.iree_run_module}", file=sys.stderr)
        return 2
    if not args.artifacts_dir.exists():
        print(f"missing artifacts dir: {args.artifacts_dir}", file=sys.stderr)
        return 2

    failures = 0
    for case in cases:
        if not _run_case(case, args.artifacts_dir, args.iree_run_module, args.verbose):
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
