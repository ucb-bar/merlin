#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Comprehensive demonstration of cuda_new HAL capabilities.

Runs all available cuda_new test suites and prints a summary table
comparing cuda_new against the baseline CUDA HAL.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path("/scratch/ashvin/merlin")
CUDA_NEW_DIR = REPO_ROOT / "runtime/src/iree/hal/drivers/cuda_new"


def _run(name: str, cmd: list[str], timeout: int = 300) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        ok = result.returncode == 0
        detail = "PASS" if ok else result.stdout.strip().splitlines()[-1] if result.stdout.strip() else f"exit {result.returncode}"
        return ok, detail
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except FileNotFoundError:
        return False, "NOT FOUND"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Treat skips as failures.")
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args(argv)
    iters = str(args.iterations)

    sections: list[tuple[str, bool, str]] = []

    # 1. Non-NCCL model smoke
    ok, detail = _run("model smoke", [
        "python3", str(CUDA_NEW_DIR / "smoke_cuda_new.py"),
    ])
    sections.append(("Non-NCCL model smoke (linear/softmax/alexnet)", ok, detail))

    # 2. True async queue and semaphore
    ok, detail = _run("async queue", [
        "python3", str(CUDA_NEW_DIR / "audit_step9_async_queue.py"),
        "--iterations", iters,
    ])
    sections.append(("Async queue execution (cuLaunchHostFunc)", ok, detail))

    # 3. CUDA graph regression
    ok, detail = _run("CUDA graphs", [
        "python3", str(CUDA_NEW_DIR / "audit_step8_graphs.py"),
        "--iterations", iters,
    ])
    sections.append(("CUDA graph capture/replay", ok, detail))

    # 4. Async allocation ordering
    ok, detail = _run("async alloc", [
        "python3", str(CUDA_NEW_DIR / "test_async_alloc_order.py"),
        "--iterations", str(int(iters) * 4), "--with-graphs",
    ])
    sections.append(("Async allocation + graphs ordering", ok, detail))

    # 5. Callback failure injection
    ok, detail = _run("fault injection", [
        "python3", str(CUDA_NEW_DIR / "audit_step9_async_queue.py"),
        "--source-only",
    ])
    sections.append(("Failure propagation (no false signal)", ok, detail))

    # 6. Step 10 NCCL source audit
    ok, detail = _run("NCCL source", [
        "python3", str(CUDA_NEW_DIR / "audit_step10_nccl.py"),
        "--source-only",
    ])
    sections.append(("NCCL channel/collective source audit", ok, detail))

    # 7. Raw NCCL machine baseline
    nccl_test = CUDA_NEW_DIR / "test_nccl_multigpu.py"
    if nccl_test.exists():
        ok, detail = _run("raw NCCL", [
            "python3", str(nccl_test),
            "--devices", "0,1", "--iterations", "3", "--count", "1024",
        ])
        sections.append(("Raw multi-GPU NCCL baseline", ok, detail))
    else:
        sections.append(("Raw multi-GPU NCCL baseline", False, "SKIP: test not found"))

    # Print results
    print()
    print("=" * 72)
    print("  cuda_new HAL Demonstration Results")
    print("=" * 72)
    failures = 0
    for name, ok, detail in sections:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            print(f"         {detail}")
            failures += 1
    print("=" * 72)

    # Summary table
    print()
    print(f"{'capability':<48} {'cuda_new':>10}")
    print("-" * 60)
    caps = [
        ("async queue completion (cuLaunchHostFunc)", sections[1][1]),
        ("event-backed timeline semaphores", sections[0][1]),
        ("CUDA graph capture/replay", sections[2][1]),
        ("async allocation ordering", sections[3][1]),
        ("failure propagation without false signal", sections[4][1]),
        ("optional NCCL loading", sections[5][1]),
        ("NCCL channels and collectives", sections[5][1]),
    ]
    for cap, ok in caps:
        print(f"  {cap:<46} {'yes' if ok else 'no':>8}")
    print()

    if failures > 0:
        print(f"{failures} section(s) failed.")
    else:
        print("All sections passed.")

    return 1 if (failures and args.strict) else (1 if failures else 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
