#!/usr/bin/env python3
"""Run merlin-dispatch-bench across per-dispatch QNN VMFBs on QRB5165.

Mirrors XPU-RT/scripts/profile_per_target_on_board.py but wraps the
merlin-built dispatch_bench (which speaks the QNN HAL URI scheme:
`qnn://gpu`, `qnn://hta`) instead of stock iree-benchmark-module
(which has no QNN driver registered on the legacy on-board runtime).

For every dispatch in <output-dir>/breakdowns/manifest.json, ssh to
the board, invoke `merlin-dispatch-bench` with the right module/
function/device/zero-input shape, parse the mean wall-time line,
and emit `profiled_manifest.json` keyed by dispatch name with
`mean_time_us` populated where successful and `infeasible: true`
where the runtime rejected the variant (placeholder, finalize
failure, dtype mismatch, etc).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

_HERE = pathlib.Path(__file__).resolve()
_MERLIN = _HERE.parent.parent

_RE_MEAN = re.compile(r"mean[:\s=]+([0-9]+\.?[0-9]*)\s*us", re.IGNORECASE)
_RE_MEDIAN = re.compile(r"median[:\s=]+([0-9]+\.?[0-9]*)\s*us", re.IGNORECASE)
_RE_SETUP = re.compile(r"setup[:\s=]+([0-9]+\.?[0-9]*)\s*us", re.IGNORECASE)


def _shape_str_for(input_type: str) -> str | None:
    # `tensor<3x320x320xf32>` -> `3x320x320xf32`
    m = re.match(r"^tensor<([^>]+)>$", input_type.strip())
    if not m:
        return None
    return m.group(1)


def _ssh_run(host: str, key: str, cmd: str, timeout: float = 60.0) -> tuple[int, str]:
    full = ["ssh", "-o", "BatchMode=yes", "-i", key, host, cmd]
    try:
        p = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return 124, "ssh timeout"


def _parse_metrics(text: str) -> dict:
    out: dict[str, float] = {}
    for label, rgx in (("mean_time_us", _RE_MEAN), ("median_time_us", _RE_MEDIAN), ("setup_time_us", _RE_SETUP)):
        m = rgx.search(text)
        if m:
            out[label] = float(m.group(1))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="Per-target build dir, expects breakdowns/manifest.json + per-dispatch VMFBs.",
    )
    p.add_argument(
        "--remote-dir",
        required=True,
        help="Board path with the same per-dispatch VMFBs (typically /root/iree_run/<job>).",
    )
    p.add_argument("--device", required=True, help="QNN URI: qnn://gpu | qnn://hta | qnn://htp.")
    p.add_argument(
        "--ssh-host",
        default=os.environ.get("MERLIN_BOARD_HOST", "qdev"),
        help="SSH host alias or user@addr. Override via $MERLIN_BOARD_HOST.",
    )
    p.add_argument(
        "--ssh-key",
        default=os.environ.get("MERLIN_BOARD_SSH_KEY", ""),
        help="SSH identity file. Override via $MERLIN_BOARD_SSH_KEY (empty = use ssh-config).",
    )
    p.add_argument(
        "--bench-bin",
        default=os.environ.get("MERLIN_BOARD_BENCH_BIN", "/tmp/merlin_e2e/merlin-dispatch-bench"),
        help="Remote path to merlin-dispatch-bench. Override via $MERLIN_BOARD_BENCH_BIN.",
    )
    p.add_argument(
        "--qnn-lib-dir",
        default=os.environ.get("MERLIN_QNN_LIB_DIR", "/root/qairt/lib/target"),
        help="Remote QNN runtime libs directory. Override via $MERLIN_QNN_LIB_DIR.",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument(
        "--profile-key",
        required=True,
        help="Tag forwarded into profiled_manifest.json (used by ingest). "
        "Pick a label that identifies (board, target, hw), e.g. 'qrb5165_qnn_GPU'.",
    )
    args = p.parse_args()

    manifest_path = args.output_dir / "breakdowns" / "manifest.json"
    if not manifest_path.exists():
        print(f"missing {manifest_path}; run tools/breakdown_vmfb.py first", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    dispatches = manifest["dispatches"]
    out_manifest = json.loads(manifest_path.read_text())  # deep-ish copy via reload
    out_manifest["profile_key"] = args.profile_key

    n_total = len(dispatches)
    n_measured = 0
    n_infeasible = 0
    rdir = pathlib.Path(args.remote_dir)
    # Accept either layout: <remote-dir>/breakdowns/dispatch_*.vmfb (the
    # XPU-RT default) or <remote-dir>/dispatch_*.vmfb (flat). Probe once.
    flat_layout = None  # determined on first dispatch
    for dname, entry in sorted(dispatches.items(), key=lambda kv: kv[1].get("id", 0)):
        if flat_layout is None:
            flat_layout = _ssh_run(args.ssh_host, args.ssh_key, f"test -f {rdir}/{dname}.vmfb", timeout=10)[0] == 0
        vmfb = rdir / f"{dname}.vmfb" if flat_layout else rdir / "breakdowns" / f"{dname}.vmfb"
        func = entry.get("module_name", "")
        if not func:
            out_manifest["dispatches"][dname]["infeasible"] = True
            n_infeasible += 1
            continue

        # Build CLI: zero-input only for first input (matches dispatch_bench
        # contract: arity 0 or 1 i32 OR arity 1 zero tensor).
        cmd_parts = [
            f"LD_LIBRARY_PATH={args.qnn_lib_dir}",
            args.bench_bin,
            f"--module={vmfb}",
            f"--function='{func}'",
            f"--device={args.device}",
            f"--warmup={args.warmup}",
            f"--iterations={args.iterations}",
        ]
        # Add a zero-input for each input declared in shapes (best-effort).
        inputs = entry.get("inputs", [])
        # dispatch_bench supports a SINGLE --zero-input. Pick the first
        # shaped input. Multi-input dispatches will fall through to ABI
        # mismatch and get marked infeasible — that's fine for the cost
        # matrix.
        if inputs:
            shape = _shape_str_for(inputs[0])
            if shape:
                cmd_parts.append(f"--zero-input={shape}")

        rc, out = _ssh_run(args.ssh_host, args.ssh_key, " ".join(cmd_parts), timeout=60)
        metrics = _parse_metrics(out)
        if rc == 0 and "mean_time_us" in metrics:
            entry_out = out_manifest["dispatches"][dname]
            entry_out.update(metrics)
            entry_out["infeasible"] = False
            n_measured += 1
            print(f"[ok ] {dname}: mean={metrics['mean_time_us']:.1f}us")
        else:
            entry_out = out_manifest["dispatches"][dname]
            entry_out["infeasible"] = True
            # Pick the most informative line: prefer one with a known
            # error keyword and (if possible) a path/file reference. The
            # bytecode-trace-only line is least informative — fall back
            # to it only if nothing better matches.
            lines = [l for l in out.splitlines() if l.strip()]
            best = None
            for kw in (
                "graphFinalize",
                "graphAddNode",
                "QnnContext_create",
                "tensorCreateGraphTensor",
                "INVALID_ARGUMENT",
                "FAILED_PRECONDITION",
                "NOT_FOUND",
                "QnnHta",
                "QnnGpu",
                "validation",
            ):
                hits = [l for l in lines if kw in l]
                if hits:
                    best = hits[0]
                    break
            if best is None and lines:
                best = lines[-1]
            entry_out["reason"] = (best or f"rc={rc}")[:200]
            n_infeasible += 1
            print(f"[xx ] {dname}: {entry_out['reason'][:80]}")

    out_path = args.output_dir / "breakdowns" / "profiled_manifest.json"
    out_path.write_text(json.dumps(out_manifest, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    print(f"  measured: {n_measured} / {n_total}")
    print(f"  infeasible: {n_infeasible} / {n_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
