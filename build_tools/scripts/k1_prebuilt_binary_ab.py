#!/usr/bin/env python3
"""Interleaved, full-output A/B for prebuilt Merlin K1 binaries.

This is the short feedback loop between an LLVM/codegen candidate and a whole-model
rebuild.  It never recompiles or changes the model: every arm must use the same work
directory (and therefore the same generated ABI and weights), and every candidate
output must be byte-identical to the first control launch before timing is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from merlin.mining import k1
from merlin.mining.registry import load_rvv_package
from merlin.runtime.backends import zephyr_model as zm


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _save(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _arm(value: str) -> tuple[str, Path]:
    try:
        label, binary = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("arm must be LABEL=/path/to/elf") from exc
    path = Path(binary).resolve()
    if not label or not path.is_file():
        raise argparse.ArgumentTypeError(f"invalid arm {value!r}")
    return label, path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="shared completed build directory containing ABI and weights")
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--arm", action="append", type=_arm, required=True,
                    help="repeat LABEL=/path/to/elf; first arm is the control")
    ap.add_argument("--order", required=True,
                    help="comma-separated interleaving using the arm labels")
    ap.add_argument("--cores", default="1,8")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--expected-sha256")
    ap.add_argument("--staged-weights",
                    help="existing board path for the work directory's mmap weights")
    ap.add_argument("--staged-weights-sha256",
                    help="required digest when --staged-weights is used")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms = dict(args.arm)
    control = args.arm[0][0]
    order = [item.strip() for item in args.order.split(",") if item.strip()]
    unknown = set(order) - set(arms)
    if unknown:
        ap.error(f"order names unknown arms: {sorted(unknown)}")
    if control not in order or any(name not in order for name in arms):
        ap.error("order must contain every arm, including the first/control arm")
    if len({path.parent for path in arms.values()}) != 1 or next(iter(arms.values())).parent != args.work_dir.resolve():
        ap.error("every ELF must live directly in --work-dir; cross-build A/Bs are refused")

    model_dir = args.model_dir.resolve()
    work_dir = args.work_dir.resolve()
    pkg0 = load_rvv_package(args.baseline.resolve())
    refs = {}
    if (model_dir / "golden.npy").is_file():
        refs["fp32"] = np.load(model_dir / "golden.npy")
    independent = model_dir / "golden_w8a8.independent.npy"
    ordinary = model_dir / "golden_w8a8.npy"
    if pkg0.is_int8 and (independent.is_file() or ordinary.is_file()):
        refs["w8a8"] = np.load(independent if independent.is_file() else ordinary)

    run_work = work_dir
    staged_weights = None
    if args.staged_weights:
        if not args.staged_weights_sha256:
            ap.error("--staged-weights requires --staged-weights-sha256")
        marker = work_dir / "USE_MMAP_WEIGHTS"
        if not marker.is_file():
            ap.error("--staged-weights requires USE_MMAP_WEIGHTS in --work-dir")
        local_digest = _sha_bytes(Path(marker.read_text().strip()).read_bytes())
        if local_digest != args.staged_weights_sha256:
            ap.error(f"local mmap weights digest {local_digest} does not match the staged digest")
        remote = k1._ssh(  # The board adapter owns credentials, port, and bounded execution.
            "sha256sum -- " + shlex.quote(args.staged_weights), timeout=180)
        remote_digest = remote.stdout.split()[0] if remote.returncode == 0 else ""
        if remote_digest != local_digest:
            raise RuntimeError(
                f"staged board weights digest {remote_digest!r} != local {local_digest}")
        # run_binary_on_k1 only consults bwork for its mmap marker and full-output sink. A marker-free
        # run directory lets the explicitly verified, immutable board file be reused across launches.
        run_work = args.out.parent / (args.out.stem + ".runio")
        run_work.mkdir(parents=True, exist_ok=True)
        staged_weights = {"path": args.staged_weights, "sha256": local_digest}

    report = {
        "schema": "merlin.k1.prebuilt_binary_ab.v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_dir": str(model_dir),
        "shared_work_dir": str(work_dir),
        "staged_weights": staged_weights,
        "control": control,
        "protocol": {"order": order, "cores": [int(x) for x in args.cores.split(",")],
                     "warmup": args.warmup, "timed_iterations": args.iters,
                     "full_output_required": True, "bit_exact_across_arms": True},
        "artifacts": {name: {"elf": str(path), "sha256": _sha_bytes(path.read_bytes())}
                      for name, path in arms.items()},
        "launches": [],
    }
    canonical: bytes | None = None
    expected = args.expected_sha256
    for cores in report["protocol"]["cores"]:
        for position, name in enumerate(order):
            binary = arms[name]
            pkg = replace(pkg0, run_id=f"k1_ab_{name}_c{cores}_p{position}")
            row = {"cores": cores, "position": position, "arm": name}
            try:
                run_env = {"MERLIN_WARMUP": str(args.warmup),
                           "MERLIN_ITERS": str(args.iters),
                           "MERLIN_AB_PAD": "X" * (64 << position)}
                if staged_weights:
                    run_env["MERLIN_WEIGHTS"] = staged_weights["path"]
                result = k1.run_binary_on_k1(
                    model_dir, run_work, pkg, binary, env=run_env,
                    timeout=args.timeout, capture_full_output=True,
                    parallel_harts=cores)
                output = np.ascontiguousarray(result["outputs"], dtype="<f4").tobytes()
                digest = _sha_bytes(output)
                if expected and digest != expected:
                    raise RuntimeError(f"output SHA {digest} != expected {expected}")
                if canonical is None:
                    if name != control:
                        raise RuntimeError("the first launch must be the control arm")
                    canonical = output
                    expected = digest
                if output != canonical:
                    raise RuntimeError("candidate output is not byte-identical to control")
                samples = [int(x) for x in result.get("iter_wall_ns", ())]
                if len(samples) != args.iters:
                    raise RuntimeError(f"expected {args.iters} samples, got {len(samples)}")
                row.update(status="pass", samples_ns=samples,
                           launch_mean_ns=statistics.fmean(samples),
                           launch_median_ns=statistics.median(samples),
                           output_sha256=digest,
                           output_elements=len(output) // 4,
                           independent_reference_gate=zm._gate(
                               np.frombuffer(output, dtype="<f4"), refs, min_coverage=1.0)
                               if refs else None,
                           board_conditions=result.get("board_conditions"),
                           affinity_mask=result.get("affinity_mask"))
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                report["launches"].append(row)
                _save(args.out, report)
                raise
            report["launches"].append(row)
            _save(args.out, report)
            print(f"c{cores} p{position} {name}: {row['launch_mean_ns']/1e6:.6f} ms", flush=True)

    summary = {}
    for cores in report["protocol"]["cores"]:
        per_arm = {}
        for name in arms:
            means = [row["launch_mean_ns"] for row in report["launches"]
                     if row["cores"] == cores and row["arm"] == name]
            per_arm[name] = {"launch_means_ns": means,
                             "median_launch_mean_ns": statistics.median(means)}
        control_ns = per_arm[control]["median_launch_mean_ns"]
        for name, values in per_arm.items():
            values["speedup_vs_control"] = control_ns / values["median_launch_mean_ns"]
        summary[str(cores)] = per_arm
    report["expected_output_sha256"] = expected
    report["summary"] = summary
    report["completed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _save(args.out, report)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
