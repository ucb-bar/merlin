#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Focused Step 8 audit/stress gate for cuda_new CUDA graph support."""

from __future__ import annotations

import argparse
import difflib
import os
import pathlib
import re
import shlex
import subprocess
import sys


REPO_ROOT = pathlib.Path("/scratch/ashvin/merlin")
CUDA_NEW_DIR = REPO_ROOT / "runtime/src/iree/hal/drivers/cuda_new"
DEFAULT_BUILD_DIR = REPO_ROOT / "build/host-merlin-release"
DEFAULT_ARTIFACTS_DIR = pathlib.Path("/tmp/codex_torch_cuda_tile_exports")
DEFAULT_CASES = (
    "linear",
    "linear_bias_relu",
    "softmax",
    "alexnet_nonuniform",
)


class CheckResult:
    def __init__(self, name: str, ok: bool, detail: str = "") -> None:
        self.name = name
        self.ok = ok
        self.detail = detail


def _read(path: pathlib.Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _contains(path: pathlib.Path, pattern: str) -> bool:
    return re.search(pattern, _read(path), flags=re.MULTILINE | re.DOTALL) is not None


def _run(
    name: str,
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: pathlib.Path = REPO_ROOT,
) -> CheckResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    result = subprocess.run(
        cmd,
        check=False,
        cwd=cwd,
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode == 0:
        return CheckResult(name, True)
    output = result.stdout.strip()
    if len(output) > 5000:
        output = output[-5000:]
    return CheckResult(name, False, output)


def _source_checks() -> list[CheckResult]:
    dynamic_symbols = CUDA_NEW_DIR / "dynamic_symbol_table.h"
    graph_c = CUDA_NEW_DIR / "graph_command_buffer.c"
    graph_h = CUDA_NEW_DIR / "graph_command_buffer.h"
    logical_device = CUDA_NEW_DIR / "logical_device.c"
    cmake = CUDA_NEW_DIR / "CMakeLists.txt"

    required_symbols = (
        "cuStreamBeginCapture",
        "cuStreamEndCapture",
        "cuGraphInstantiate",
        "cuGraphLaunch",
        "cuGraphExecDestroy",
        "cuGraphDestroy",
    )
    required_graph_calls = (
        "cuCtxSetCurrent",
        "cuStreamBeginCapture",
        "iree_hal_deferred_command_buffer_apply",
        "cuStreamEndCapture",
        "cuGraphInstantiate",
        "cuGraphDestroy",
    )

    results = [
        CheckResult(
            "graph_command_buffer.c exists",
            graph_c.exists(),
            str(graph_c),
        ),
        CheckResult(
            "graph_command_buffer.h exists",
            graph_h.exists(),
            str(graph_h),
        ),
        CheckResult(
            "CMake compiles graph command buffer",
            '"graph_command_buffer.c"' in _read(cmake)
            and '"graph_command_buffer.h"' in _read(cmake),
            "graph_command_buffer sources must be in cuda_new CMake SRCS/HDRS.",
        ),
        CheckResult(
            "required CUDA graph symbols are declared",
            all(symbol in _read(dynamic_symbols) for symbol in required_symbols),
            ", ".join(required_symbols),
        ),
        CheckResult(
            "graph capture helper performs capture, replay, instantiate, destroy",
            all(call in _read(graph_c) for call in required_graph_calls),
            ", ".join(required_graph_calls),
        ),
        CheckResult(
            "failed capture path ends capture and destroys discard graph",
            "cuStreamEndCapture(capture_stream, &discard)" in _read(graph_c)
            and "cuGraphDestroy(discard)" in _read(graph_c),
            "A failed replay must leave the stream out of capture mode.",
        ),
        CheckResult(
            "logical_device has explicit graph mode switch",
            "IREE_CUDA_NEW_USE_GRAPHS" in _read(logical_device),
            "Graph mode should be explicitly selectable for testing.",
        ),
        CheckResult(
            "queue_execute launches and destroys graph executable",
            "cuGraphLaunch" in _read(logical_device)
            and "cuGraphExecDestroy" in _read(logical_device),
            "Graph exec lifetime must be explicit.",
        ),
        CheckResult(
            "queue_execute has completion mechanism (sync or async)",
            "cuStreamSynchronize" in _read(logical_device)
            or "cuLaunchHostFunc" in _read(logical_device),
            "Must have either sync fallback or async completion callback.",
        ),
        CheckResult(
            "signal events are recorded before completion",
            _contains(logical_device, r"cuEventRecord"),
            "Signal events must be recorded on stream for ordering.",
        ),
    ]
    return results


def _read_reference_command(case_dir: pathlib.Path) -> list[str]:
    cmd_file = case_dir / "run.stdout.cmd.txt"
    lines = cmd_file.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty command file: {cmd_file}")
    return shlex.split(lines[0])


def _rewrite_for_cuda_new(cmd: list[str], iree_run_module: pathlib.Path) -> list[str]:
    rewritten = [str(iree_run_module)]
    for arg in cmd[1:]:
        rewritten.append("--device=cuda_new" if arg == "--device=cuda_tile" else arg)
    return rewritten


def _normalize(text: str) -> list[str]:
    return [line.rstrip() for line in text.strip().splitlines()]


def _run_case(
    *,
    case: str,
    artifacts_dir: pathlib.Path,
    iree_run_module: pathlib.Path,
    use_graphs: bool,
    iteration: int,
    verbose: bool,
) -> CheckResult:
    mode = "graph" if use_graphs else "fallback"
    name = f"{mode} runtime {case} iteration {iteration}"
    case_dir = artifacts_dir / case
    reference_stdout = case_dir / "run.stdout.txt"
    try:
        cmd = _rewrite_for_cuda_new(
            _read_reference_command(case_dir), iree_run_module
        )
        expected = _normalize(reference_stdout.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - harness should report details.
        return CheckResult(name, False, str(exc))

    if verbose:
        print(f"[RUN] {name}: {' '.join(shlex.quote(arg) for arg in cmd)}")

    env = {"IREE_CUDA_NEW_USE_GRAPHS": "1"} if use_graphs else {}
    result = subprocess.run(
        cmd,
        check=False,
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stdout + result.stderr
        if len(detail) > 5000:
            detail = detail[-5000:]
        return CheckResult(name, False, detail.strip())

    actual = _normalize(result.stdout)
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                expected,
                actual,
                fromfile=f"{case}/cuda_tile_reference",
                tofile=f"{case}/cuda_new_{mode}",
                lineterm="",
            )
        )
        return CheckResult(name, False, diff[:5000])
    return CheckResult(name, True)


def _dynamic_checks(args: argparse.Namespace) -> list[CheckResult]:
    iree_run_module = args.build_dir / "tools/iree-run-module"
    results = [
        _run(
            "Step 7 dynamic gate still passes",
            [
                "python3",
                str(CUDA_NEW_DIR / "audit_step6_sync.py"),
                "--dynamic",
                "--build-dir",
                str(args.build_dir),
            ],
        ),
    ]

    for iteration in range(1, args.iterations + 1):
        for case in args.cases:
            results.append(
                _run_case(
                    case=case,
                    artifacts_dir=args.artifacts_dir,
                    iree_run_module=iree_run_module,
                    use_graphs=False,
                    iteration=iteration,
                    verbose=args.verbose,
                )
            )
            results.append(
                _run_case(
                    case=case,
                    artifacts_dir=args.artifacts_dir,
                    iree_run_module=iree_run_module,
                    use_graphs=True,
                    iteration=iteration,
                    verbose=args.verbose,
                )
            )
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=DEFAULT_BUILD_DIR,
        help="IREE build directory.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=pathlib.Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing existing cuda_tile export artifacts.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=DEFAULT_CASES,
        help="Case to run. May be repeated. Defaults to all Step 8 stress cases.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of fallback/graph repetitions per case.",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Only run source checks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print runtime commands.",
    )
    args = parser.parse_args(argv)
    args.cases = tuple(args.cases) if args.cases else DEFAULT_CASES

    results = _source_checks()
    if not args.source_only:
        results.extend(_dynamic_checks(args))

    failures = 0
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}")
        if not result.ok:
            failures += 1
            if result.detail:
                print(f"       {result.detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
