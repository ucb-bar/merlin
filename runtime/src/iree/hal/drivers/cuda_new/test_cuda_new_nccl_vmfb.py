#!/usr/bin/env python3
# Copyright 2026 UCB-BAR
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""End-to-end collective VMFB test for cuda_new via MPI.

Compiles allreduce_f32.mlir, launches N MPI ranks, each on a distinct GPU,
and validates rank-dependent all-reduce results.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path("/scratch/ashvin/merlin")
BUILD_DIR = REPO_ROOT / "build/host-merlin-release"
IREE_COMPILE = BUILD_DIR / "tools/iree-compile"
IREE_RUN_MODULE = BUILD_DIR / "tools/iree-run-module"
MLIR_SRC = (
    REPO_ROOT
    / "runtime/src/iree/hal/drivers/cuda_new/test/allreduce_f32.mlir"
)


def _count_cuda_devices() -> int:
    try:
        result = subprocess.run(
            [str(IREE_RUN_MODULE), "--list_devices=cuda_new"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return len([l for l in result.stdout.splitlines() if l.strip()])
    except Exception:
        return 0


def _compile(mlir_path: pathlib.Path, vmfb_path: pathlib.Path) -> bool:
    result = subprocess.run(
        [
            str(IREE_COMPILE),
            str(mlir_path),
            "--iree-hal-target-backends=cuda_tile",
            "--iree-cuda-tile-enable-codegen=false",
            "-o",
            str(vmfb_path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        print(f"[FAIL] compile: {result.stdout.strip()}")
        return False
    print("[PASS] compile")
    return True


def _run_mpi(
    vmfb_path: pathlib.Path, ranks: int, count: int, iterations: int,
    timeout: int,
) -> bool:
    mpirun = shutil.which("mpirun")
    if not mpirun:
        print("[SKIP] mpirun not found")
        return True

    all_ok = True
    for it in range(1, iterations + 1):
        # Build rank-dependent inputs: rank r element i = (r+1)*10000 + i
        # Expected output element i = sum over ranks = sum_{r=0}^{ranks-1} ((r+1)*10000 + i)
        #   = 10000 * sum(1..ranks) + ranks*i = 10000*ranks*(ranks+1)/2 + ranks*i
        base_sum = 10000 * ranks * (ranks + 1) // 2

        script = (
            'r=${OMPI_COMM_WORLD_RANK}; '
            'INPUT=$(python3 -c "'
            "import sys; r=int(sys.argv[1]); n=int(sys.argv[2]); "
            "vals=','.join(str((r+1)*10000+i) for i in range(n)); "
            "print(f'{n}xf32={vals}')"
            f'" $r {count}); '
            f'{IREE_RUN_MODULE} '
            f'--device=cuda_new://$r '
            f'--cuda_new_default_index_from_mpi=false '
            f'--module={vmfb_path} '
            f'--function=all_reduce_sum --input=$INPUT'
        )

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
        )
        result = subprocess.run(
            [mpirun, "--allow-run-as-root", "-n", str(ranks),
             "-x", "LD_LIBRARY_PATH", "bash", "-c", script],
            check=False,
            text=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )

        if result.returncode != 0:
            print(f"[FAIL] run iteration {it}: exit {result.returncode}")
            print(result.stdout[-2000:] if result.stdout else "")
            all_ok = False
            continue

        # Parse and validate outputs
        outputs = [
            l for l in result.stdout.splitlines()
            if l.strip().startswith(f"{count}xf32=")
        ]
        if len(outputs) != ranks:
            print(f"[FAIL] iteration {it}: expected {ranks} outputs, got {len(outputs)}")
            all_ok = False
            continue

        expected_vals = [base_sum + ranks * i for i in range(count)]
        expected_str = " ".join(str(v) for v in expected_vals)

        ok = True
        for rank_idx, out_line in enumerate(outputs):
            actual = out_line.split("=", 1)[1].strip()
            if actual != expected_str:
                print(f"[FAIL] iteration {it} rank {rank_idx}: expected [{expected_str}] got [{actual}]")
                ok = False
                all_ok = False

        if ok:
            print(f"[PASS] iteration {it}/{iterations} ({ranks} ranks, {count} elements)")

    return all_ok


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args(argv)

    num_devices = _count_cuda_devices()
    if num_devices < args.ranks:
        print(f"[SKIP] need {args.ranks} CUDA devices, found {num_devices}")
        return 0

    with tempfile.TemporaryDirectory() as tmpdir:
        vmfb = pathlib.Path(tmpdir) / "allreduce_f32.vmfb"
        if not _compile(MLIR_SRC, vmfb):
            return 1
        if not _run_mpi(vmfb, args.ranks, args.count, args.iterations, args.timeout):
            return 1

    print(f"\nAll passed ({args.ranks} ranks, {args.count} elements, {args.iterations} iterations)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
