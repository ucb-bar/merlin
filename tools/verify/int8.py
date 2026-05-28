#!/usr/bin/env python3
"""Verify INT8 backend output correctness via cross-hash comparison.

The merlin runtime (`merlin_hetero_runner`) prints one
`[hetero] job=N hart=H cycles=C hash=0xXXXX rc=R` line per inference
on the FireSim UART. The `hash` is a 64-bit FNV/CRC-style fold over
the int8 output tensor of the model — different backends running the
SAME int8 quantized graph SHOULD produce identical hashes, because
int8 arithmetic is deterministic (no fp rounding drift).

This script does two complementary jobs:

  1. Computes a CPU "golden" reference hash by running the
     ORIGINAL .q.int8.onnx model on x86 via onnxruntime with a
     fixed-seed deterministic input, and folding the output with the
     same hash function the on-board runtime uses.

  2. Parses one or more FireSim uartlog files (or hashes given on
     the command line) and compares every backend's hash against the
     golden reference + against each other. Prints a pass/fail
     table per (model, backend, hart, job).

Examples:

    # Golden hash + compare against a logged FireSim hash:
    ./merlin verify-output models/dronet/dronet.q.int8.onnx \\
            --shape 1,3,200,200 \\
            --observed 0x498d8553b619f5da:gemmini \\
            --observed 0xd4d44793e1099c94:opu

    # Pull hashes directly from a FireSim uartlog:
    ./merlin verify-output models/dronet/dronet.q.int8.onnx \\
            --shape 1,3,200,200 \\
            --uartlog tmp/firesim_dronet_GEM_OPU_PASSED_2026-05-17.log
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


# Hash function used by `merlin_record_job_result` in the embedded
# Zephyr runner (see zephyr-chipyard-sw/samples/merlin_hetero_runner/
# src/hetero_worker.c). The runtime hash is a 64-bit FNV-1a fold over
# the model's output bytes. Mirror it here so the CPU reference and
# the on-board hash are directly comparable.
def _fnv1a_64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _parse_shape(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def _golden_hash(onnx_path: Path, shapes: list[list[int]], zero_input: bool = True, seed: int = 0xCAFE) -> int:
    """Run model on x86 via onnxruntime and hash the output.

    The merlin runtime (hetero_worker.c::worker_init) pre-allocates a
    ZERO-FILLED input buffer once and reuses it for every job. So the
    on-board "first_job_hash" the runner reports is the FNV-1a hash
    of the model's output when fed all-zero input.

    Pass zero_input=False to fall back to a fixed-seed random input
    (useful for sanity-checking different inputs produce different
    outputs).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit("onnxruntime not installed in this env. install with:\n" "    uv pip install onnxruntime")

    sess_opts = ort.SessionOptions()
    # Disable optimizations that could change int8 saturation/round
    # semantics — we want bit-for-bit what the backend sees.
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts)
    input_specs = sess.get_inputs()
    if len(input_specs) != len(shapes):
        raise SystemExit(
            f"model has {len(input_specs)} inputs ({[i.name for i in input_specs]}) "
            f"but you passed {len(shapes)} --shape arguments"
        )
    feeds = {}
    for spec, shape in zip(input_specs, shapes):
        if zero_input:
            # Matches `hetero_worker.c::worker_init` which calls
            # `iree_hal_buffer_allocator_allocate(... ZERO_FILL)` for
            # every per-job input.
            feeds[spec.name] = np.zeros(tuple(shape), dtype=np.float32)
        else:
            rng = np.random.default_rng(seed=seed)
            feeds[spec.name] = rng.standard_normal(tuple(shape)).astype(np.float32)
    outputs = sess.run(None, feeds)
    # Match the runner's multi-output fold:
    #   hash = FNV1A_OFFSET (0xcbf29ce484222325)
    #   for each output: hash ^= fnv1a64(output_bytes) + 0x9e3779b97f4a7c15
    #                            + (hash << 6) + (hash >> 2)
    # For single-output models this still produces a value different
    # from raw fnv1a64(output_bytes), so we must mirror exactly.
    hash_seed = 0xCBF29CE484222325
    h = hash_seed
    for out in outputs:
        h_i = _fnv1a_64(out.tobytes())
        h = (h ^ (h_i + 0x9E3779B97F4A7C15 + (h << 6) + (h >> 2))) & 0xFFFFFFFFFFFFFFFF
    return h


_UARTLOG_HASH_RE = re.compile(r"\[hetero\] job=(\d+) hart=(\d+) cycles=\d+ hash=0x([0-9a-fA-F]+) rc=(-?\d+)")


def _parse_uartlog(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        m = _UARTLOG_HASH_RE.search(line)
        if m:
            rows.append(
                {
                    "job": int(m.group(1)),
                    "hart": int(m.group(2)),
                    "hash": int(m.group(3), 16),
                    "rc": int(m.group(4)),
                }
            )
    return rows


def _parse_observed(spec: str) -> tuple[int, str]:
    if ":" not in spec:
        raise SystemExit(f"--observed needs HASH:LABEL form, got: {spec}")
    h, label = spec.split(":", 1)
    return int(h, 16), label


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", type=Path, help="Quantized .q.int8.onnx model")
    p.add_argument(
        "--shape",
        action="append",
        required=True,
        help="Input shape comma-separated (repeat per input)",
    )
    p.add_argument(
        "--observed",
        action="append",
        default=[],
        help="Backend hash to verify: <hex_hash>:<label> (e.g. 0x498...:gemmini)",
    )
    p.add_argument(
        "--uartlog",
        action="append",
        type=Path,
        default=[],
        help="FireSim uartlog file to extract hashes from",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0xCAFE,
        help="RNG seed for x86 reference input when --random-input (default 0xCAFE)",
    )
    p.add_argument(
        "--random-input",
        action="store_true",
        help=(
            "Use random input instead of all-zero (the runner uses zeros "
            "via ZERO_FILL buffer alloc). Use this only for sanity checks."
        ),
    )
    p.add_argument(
        "--skip-golden",
        action="store_true",
        help="Skip the onnxruntime baseline (just cross-check observed hashes)",
    )
    args = p.parse_args()

    shapes = [_parse_shape(s) for s in args.shape]

    rows = []
    for spec in args.observed:
        h, label = _parse_observed(spec)
        rows.append({"source": "--observed", "label": label, "hash": h})
    for uart in args.uartlog:
        if not uart.exists():
            raise SystemExit(f"uartlog not found: {uart}")
        for r in _parse_uartlog(uart):
            rows.append(
                {
                    "source": uart.name,
                    "label": f"job={r['job']} hart={r['hart']}",
                    "hash": r["hash"],
                    "rc": r["rc"],
                }
            )

    golden = None
    if not args.skip_golden:
        print(f"==> computing CPU x86 golden hash from {args.model}")
        golden = _golden_hash(
            args.model,
            shapes,
            zero_input=not args.random_input,
            seed=args.seed,
        )
        print(f"    golden hash = 0x{golden:016x} (zero_input={not args.random_input})")

    if not rows:
        print("no observed hashes to compare. Pass --observed HASH:LABEL " "or --uartlog FILE.")
        return 0 if args.skip_golden else None

    print()
    print(f"{'source':<46} {'label':<32} {'hash':<20} {'verdict'}")
    print("-" * 110)
    any_mismatch = False
    for r in rows:
        h = r["hash"]
        verdict = "OK"
        if golden is not None and h != golden:
            verdict = "DIFF-FROM-GOLDEN"
            any_mismatch = True
        elif r.get("rc", 0) != 0:
            verdict = f"rc={r['rc']}"
            any_mismatch = True
        print(f"{r['source']:<46} {r['label']:<32} 0x{h:016x}   {verdict}")

    # Also cross-compare each pair of observed hashes
    distinct_hashes = sorted({r["hash"] for r in rows})
    if len(distinct_hashes) > 1:
        print()
        print(f"WARNING: {len(distinct_hashes)} DISTINCT hashes across observed runs:")
        for h in distinct_hashes:
            labels = [r["label"] for r in rows if r["hash"] == h]
            print(f"  0x{h:016x}  →  {', '.join(labels)}")
        any_mismatch = True

    return 1 if any_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
