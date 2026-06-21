#!/usr/bin/env python3
"""Approach (d): the DEPRECATED-merlin hand-written C++ Gemmini dialect via IREE.

For each feasible matmul kernel in the corpus, emit a tensor-domain linalg.matmul fixture, compile it
through the REAL IREE Gemmini plugin (`iree-compile --iree-plugin=gemmini --iree-gemmini-enable
--iree-gemmini-lower-back-to-iree=false`), embed the resulting .vmfb into the bare-metal
`bench_gemmini_spike_matmul` runner, run on spike (--isa=...zicntr enables rdcycle), and parse the
runner's `METRIC cycles` + PASS/FAIL.

The runner fills A,B with all-ones and verifies each output == K (data-independent, so the systolic-array
cycle count is directly comparable to the other arms). Cycles INCLUDE IREE HAL dispatch overhead (the
cost of the IREE runtime wrapper the deprecated merlin ships) — a real architectural difference, noted
in the report. Spike-only: verilator would need the full IREE runtime on RTL (out of scope).

NOTE the convention transpose: the corpus encodes shapes as M x K x N (matmul ins(MxK, KxN)); the IREE
fixture filename is matmul_<M>x<N>x<K> (the CMake regex binds group1=M, group2=N, group3=K). So the
fixture name swaps N and K relative to the corpus id.

Writes <run>/iree_arm_results.json: {kernel_id: {cycles, util_pct, correct, shape_mkn, fixture, macs,
error?}}. Merge into a perf run with merge_iree_arm.py, then regenerate the report.

Usage: run_iree_arm.py [--run-id perf_full_0001] [--max-macs 3000000] [--kernels all|id,id]
All local tools (iree-compile + spike) — no Opus budget.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

import _pbcommon as PB

MERLIN = Path("/scratch2/agustin/merlin")          # DEPRECATED IREE-based merlin (NOT oscar-merlin)
BUILD = MERLIN / "build" / "firesim-merlin-release"
FIXTURES = MERLIN / "tests" / "integration" / "gemmini_spike" / "fixtures"
RUNNER_TARGET = "bench_gemmini_spike_matmul"
ELF = (BUILD / "runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel"
       / "bench_gemmini_spike_matmul")
SPIKE = Path("/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/spike")
SPIKE_ISA = "rv64gcv_zicntr_zihpm"                  # zicntr enables rdcycle for the cycle metric

BUILD_ENV = {
    **os.environ,
    "RISCV_TOOLCHAIN_ROOT": str(MERLIN / "build_tools/riscv-tools-iree/toolchain/clang/linux/RISCV"),
    "RISCV_NEWLIB_SYSROOT": "/scratch2/agustin/chipyard/.conda-env/riscv-tools/riscv64-unknown-elf",
}

FIXTURE_TMPL = """\
func.func @matmul_{shape}(%A: tensor<{M}x{K}xi8>, %B: tensor<{K}x{N}xi8>) -> tensor<{M}x{N}xi32> {{
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<{M}x{N}xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%init : tensor<{M}x{N}xi32>) -> tensor<{M}x{N}xi32>
  %res = linalg.matmul ins(%A, %B : tensor<{M}x{K}xi8>, tensor<{K}x{N}xi8>)
                       outs(%fill : tensor<{M}x{N}xi32>) -> tensor<{M}x{N}xi32>
  return %res : tensor<{M}x{N}xi32>
}}
"""


def ensure_fixture(M: int, N: int, K: int) -> tuple[str, Path]:
    """Fixture name is matmul_MxNxK (CMake binds M,N,K in that order); A=MxK, B=KxN."""
    shape = f"{M}x{N}x{K}"
    path = FIXTURES / f"matmul_{shape}_tensor.mlir"
    if not path.is_file():
        path.write_text(FIXTURE_TMPL.format(shape=shape, M=M, N=N, K=K))
    return shape, path


def build_shape(shape: str, log: Path) -> None:
    """Reconfigure the runner for SHAPE then build it. The env carries the clang cross-toolchain;
    we touch build.ninja so ninja's env-stripped auto-regen doesn't clobber the configure."""
    with log.open("w") as f:
        subprocess.run(
            ["cmake", "-S", str(MERLIN / "third_party/iree_bar"), "-B", str(BUILD),
             f"-DGEMMINI_SPIKE_MATMUL_SHAPE={shape}"],
            cwd=MERLIN, env=BUILD_ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
        (BUILD / "build.ninja").touch()
        subprocess.run(["ninja", "-C", str(BUILD), RUNNER_TARGET],
                       cwd=MERLIN, env=BUILD_ENV, stdout=f, stderr=subprocess.STDOUT, check=True)


_CYC_RE = re.compile(r"METRIC cycles (\d+)")


def run_spike(log: Path, timeout: int) -> tuple[int | None, bool]:
    out = subprocess.run([str(SPIKE), "--extension=gemmini", f"--isa={SPIKE_ISA}", str(ELF)],
                         capture_output=True, text=True, timeout=timeout).stdout
    log.write_text(out)
    m = _CYC_RE.search(out)
    cyc = int(m.group(1)) if m else None
    passed = "[gemmini-spike] PASS" in out
    return cyc, passed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="perf_full_0001")
    ap.add_argument("--max-macs", type=int, default=3_000_000,
                    help="skip matmuls above this (spike-infeasible giants, e.g. tiny_llama lm_head)")
    ap.add_argument("--kernels", default="all")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args(argv)

    doc = yaml.safe_load((PB.KERNELS / "kernel_corpus.yaml").read_text())
    corpus = ((doc.get("golden_kernels") or []) + (doc.get("model_kernels") or [])
              + (doc.get("attention_kernels") or []))
    corpus = [k for k in corpus if k.get("op") == "matmul"]
    if a.kernels != "all":
        want = set(a.kernels.split(","))
        corpus = [k for k in corpus if k["id"] in want]

    out_dir = PB.RUNS / a.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "_iree_logs"
    logs.mkdir(exist_ok=True)
    res_path = out_dir / "iree_arm_results.json"
    results = json.loads(res_path.read_text()) if res_path.is_file() else {}

    for k in corpus:
        kid = k["id"]
        # golden/model entries carry M/K/N keys; attention entries carry only a "MxKxN" shape string.
        if k.get("M") is not None:
            M, K, N = int(k["M"]), int(k["K"]), int(k["N"])
        else:
            M, K, N = (int(x) for x in str(k["shape"]).split("x"))
        rec = {"shape_mkn": f"{M}x{K}x{N}", "macs": k["macs"]}
        if k["macs"] > a.max_macs:
            rec["error"] = f"skipped: macs {k['macs']:,} > {a.max_macs:,} (spike-infeasible)"
            results[kid] = rec
            print(f"[{kid:34s}] SKIP (macs {k['macs']:,})", flush=True)
            res_path.write_text(json.dumps(results, indent=2))
            continue
        t0 = time.time()
        try:
            shape, fx = ensure_fixture(M, N, K)
            rec["fixture"] = fx.name
            build_shape(shape, logs / f"{kid}.build.log")
            cyc, passed = run_spike(logs / f"{kid}.spike.log", a.timeout)
            rec["cycles"] = cyc
            rec["correct"] = passed
            rec["util_pct"] = PB.utilization_pct(k["macs"], cyc)
            rec["wall_s"] = round(time.time() - t0, 1)
            print(f"[{kid:34s}] cyc={cyc} pass={passed} util={rec['util_pct']}% "
                  f"({rec['wall_s']}s)", flush=True)
        except subprocess.CalledProcessError as e:
            rec["error"] = f"build failed (see {kid}.build.log)"
            print(f"[{kid:34s}] BUILD-FAIL ({round(time.time()-t0,0)}s)", flush=True)
        except subprocess.TimeoutExpired:
            rec["error"] = f"spike timeout > {a.timeout}s"
            print(f"[{kid:34s}] TIMEOUT", flush=True)
        except Exception as e:  # noqa: BLE001 — record honestly, keep going
            rec["error"] = f"{type(e).__name__}: {str(e)[-160:]}"
            print(f"[{kid:34s}] ERR {rec['error']}", flush=True)
        results[kid] = rec
        res_path.write_text(json.dumps(results, indent=2))

    ok = sum(1 for r in results.values() if r.get("cycles") is not None and r.get("correct"))
    print(f"\nwrote {res_path}  ({ok}/{len(results)} kernels with valid cycles+PASS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
