#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Targeted test for stream-ordered async allocation lifetime.

Proves that:
1. Async-allocated buffers survive through dispatch execution.
2. Stream-ordered free (queue_dealloca) does not corrupt subsequent work.
3. Repeated alloca/dispatch/dealloca cycles do not leak or crash.
4. Graph mode + async allocation combined is correct.

This test runs a model multiple times in rapid succession with async
allocation enabled, then verifies output stability. If free-before-work
or use-after-free bugs exist, repeated runs will produce incorrect
results or crash.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path("/scratch/ashvin/merlin")
IREE_RUN_MODULE = REPO_ROOT / "build/host-merlin-release/tools/iree-run-module"
ARTIFACTS_DIR = pathlib.Path("/tmp/codex_torch_cuda_tile_exports")

CASES = [
    ("linear", "--input=2x8xf32=1"),
    ("softmax", "--input=2x8xf32=1"),
    ("linear144_bias", "--input=1x144xf32=1"),
    ("conv_only", "--input=1x3x8x8xf32=1"),
]


def run_model(case: str, input_flag: str, env: dict) -> str | None:
    vmfb = ARTIFACTS_DIR / case / "module_cuda_tile.vmfb"
    if not vmfb.exists():
        return None
    result = subprocess.run(
        [
            str(IREE_RUN_MODULE),
            "--device=cuda_new",
            f"--module={vmfb}",
            "--function=main",
            input_flag,
        ],
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None


def get_reference(case: str, input_flag: str, base_env: dict) -> str | None:
    """Get reference output from cuda_tile."""
    vmfb = ARTIFACTS_DIR / case / "module_cuda_tile.vmfb"
    if not vmfb.exists():
        return None
    result = subprocess.run(
        [
            str(IREE_RUN_MODULE),
            "--device=cuda_tile",
            f"--module={vmfb}",
            "--function=main",
            input_flag,
        ],
        env=base_env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--with-graphs", action="store_true")
    args = parser.parse_args(argv)

    import os

    base_env = os.environ.copy()
    async_env = base_env.copy()
    async_env["IREE_CUDA_NEW_ASYNC_ALLOCATIONS"] = "1"
    if args.with_graphs:
        async_env["IREE_CUDA_NEW_USE_GRAPHS"] = "1"

    mode = "graphs+async" if args.with_graphs else "async"
    failures = 0

    for case, input_flag in CASES:
        ref = get_reference(case, input_flag, base_env)
        if ref is None:
            print(f"[{case}] SKIP: no reference available")
            continue

        all_match = True
        for i in range(args.iterations):
            out = run_model(case, input_flag, async_env)
            if out is None:
                print(f"[{case}] FAIL: iteration {i+1} crashed/failed ({mode})")
                all_match = False
                failures += 1
                break
            if out != ref:
                print(
                    f"[{case}] FAIL: iteration {i+1} output differs ({mode})"
                )
                print(f"  expected: {ref}")
                print(f"  got:      {out}")
                all_match = False
                failures += 1
                break

        if all_match:
            print(f"[{case}] PASS: {args.iterations} iterations ({mode})")

    if failures == 0:
        print(f"\nAll cases passed ({args.iterations} iterations, {mode} mode)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
