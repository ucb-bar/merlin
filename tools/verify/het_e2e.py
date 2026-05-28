#!/usr/bin/env python3
"""End-to-end numerical verification harness for heterogeneous schedules.

Phase G of the heterogeneous-scheduling pipeline. Two modes:

* `gen-ref`: compile a CPU-only VMFB of the model, run it with a fixed-seed
  input, and save the output as a reference binary. Use this to capture
  the gold output before invoking a heterogeneous run.
* `compare`: given a reference output file and a candidate output file
  (typically produced by `merlin-dispatch-scheduler --data-flow-mode
  --output-data-to=<file>`), assert element-wise numerical equivalence
  within configurable tolerance.

Together these let us verify a heterogeneous schedule's output is
numerically equivalent to the CPU baseline.

Usage:
  # Generate reference once:
  ./merlin … tools/verify_het_e2e.py gen-ref \\
      --source benchmarks/QRB5165/mlir/yolov8.mlir \\
      --function module.main --input-shape 1x3x640x640xi8 --seed 42 \\
      --out-dir /tmp/yolov8_ref

  # Compare a heterogeneous run's output:
  ./merlin … tools/verify_het_e2e.py compare \\
      --reference /tmp/yolov8_ref/output.bin \\
      --candidate /tmp/yolov8_ref/het_output.bin \\
      --shape 1x84x8400xf32 --rtol 5e-3 --atol 1e-2
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import subprocess
import sys
from collections.abc import Iterable

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_IREE_COMPILE = REPO_ROOT / "build/host-merlin-release-qrb/tools/iree-compile"
DEFAULT_IREE_RUN = REPO_ROOT / "build/host-merlin-release-qrb/tools/iree-run-module"

_DTYPE_TO_NP: dict[str, np.dtype] = {
    "i8": np.int8,
    "u8": np.uint8,
    "i16": np.int16,
    "u16": np.uint16,
    "i32": np.int32,
    "u32": np.uint32,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}


def _parse_shape(shape: str) -> tuple[list[int], np.dtype]:
    """Parse `1x3x640x640xf32` into ([1,3,640,640], np.float32)."""
    if "x" not in shape:
        raise ValueError(f"shape must end with x<dtype>: {shape}")
    *dims_str, dtype_str = shape.split("x")
    dtype = _DTYPE_TO_NP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"unknown dtype {dtype_str!r}; known: {list(_DTYPE_TO_NP)}")
    return [int(d) for d in dims_str], np.dtype(dtype)


def _gen_random(shape: list[int], dtype: np.dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)
    return rng.standard_normal(shape).astype(dtype)


def _save_npy(path: pathlib.Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def _load_bin(path: pathlib.Path, shape: list[int], dtype: np.dtype) -> np.ndarray:
    raw = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(f"file {path} has {raw.size} elements but shape {shape} needs {expected}")
    return raw.reshape(shape)


def _compile_cpu(source: pathlib.Path, vmfb: pathlib.Path, iree_compile: pathlib.Path, log: pathlib.Path) -> bool:
    cmd = [
        str(iree_compile),
        '--iree-hal-target-device=#hal.device.target<"local", '
        '[#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", '
        '{target_triple = "x86_64-linux-gnu"}>]>',
        "-o",
        str(vmfb),
        str(source),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode == 0


def _run_module(
    iree_run: pathlib.Path, vmfb: pathlib.Path, function: str, inputs: Iterable[str], device: str = "local-task"
) -> bytes:
    cmd = [
        str(iree_run),
        f"--module={vmfb}",
        f"--device={device}",
        f"--function={function}",
        "--output=-",
    ]
    for inp in inputs:
        cmd.append(f"--input={inp}")
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"iree-run-module failed: rc={proc.returncode}\n" f"stderr: {proc.stderr.decode(errors='replace')[:2000]}"
        )
    return proc.stdout


def cmd_gen_ref(args: argparse.Namespace) -> int:
    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shape, dtype = _parse_shape(args.input_shape)
    arr = _gen_random(shape, dtype, args.seed)
    in_path = out_dir / "input.bin"
    arr.tofile(in_path)
    print(f"input: {in_path}  shape={shape}  dtype={dtype}  bytes={arr.nbytes}")

    vmfb = out_dir / "ref_cpu.vmfb"
    log = out_dir / "compile.log"
    ok = _compile_cpu(args.source, vmfb, args.iree_compile, log)
    if not ok:
        print(f"compile FAILED — see {log}", file=sys.stderr)
        return 1
    print(f"compiled: {vmfb}")

    # Run with iree-run-module, materializing the input from in_path.
    input_arg = f"{args.input_shape}=@{in_path}"
    raw = _run_module(args.iree_run, vmfb, args.function, [input_arg])
    out_path = out_dir / "output.bin"
    out_path.write_bytes(raw)
    print(f"reference output: {out_path}  bytes={len(raw)}")

    # Stash a small metadata file for the compare step.
    (out_dir / "meta.txt").write_text(
        f"source={args.source}\nfunction={args.function}\n" f"input_shape={args.input_shape}\nseed={args.seed}\n"
    )
    return 0


@dataclasses.dataclass
class CompareReport:
    n_total: int
    n_within: int
    max_abs_diff: float
    max_rel_diff: float
    first_bad_idx: int | None
    first_bad_pair: tuple[float, float] | None


def _compare(ref: np.ndarray, cand: np.ndarray, rtol: float, atol: float) -> CompareReport:
    diff = np.abs(ref.astype(np.float64) - cand.astype(np.float64))
    rel = diff / (np.maximum(np.abs(ref.astype(np.float64)), 1e-12))
    within = (diff <= atol) | (rel <= rtol)
    n_within = int(within.sum())
    n_total = int(within.size)
    bad = np.where(~within)[0] if within.ndim == 1 else np.argwhere(~within)
    first_bad_idx: int | None = None
    first_bad_pair: tuple[float, float] | None = None
    if (bad.size if hasattr(bad, "size") else len(bad)) > 0:
        if within.ndim == 1:
            i = int(bad[0])
        else:
            i = tuple(int(x) for x in bad[0])
        first_bad_idx = i if isinstance(i, int) else int(np.ravel_multi_index(i, ref.shape))
        first_bad_pair = (float(ref.flat[first_bad_idx]), float(cand.flat[first_bad_idx]))
    return CompareReport(
        n_total=n_total,
        n_within=n_within,
        max_abs_diff=float(diff.max()),
        max_rel_diff=float(rel.max()),
        first_bad_idx=first_bad_idx,
        first_bad_pair=first_bad_pair,
    )


def cmd_compare(args: argparse.Namespace) -> int:
    shape, dtype = _parse_shape(args.shape)
    ref = _load_bin(args.reference, shape, dtype).flatten()
    cand = _load_bin(args.candidate, shape, dtype).flatten()
    rep = _compare(ref, cand, args.rtol, args.atol)
    pass_pct = 100.0 * rep.n_within / rep.n_total
    print(f"shape={shape} dtype={dtype}")
    print(f"within tolerance: {rep.n_within}/{rep.n_total} ({pass_pct:.3f}%)")
    print(f"max abs diff: {rep.max_abs_diff:.6g}")
    print(f"max rel diff: {rep.max_rel_diff:.6g}")
    if rep.first_bad_idx is not None:
        print(
            f"first divergence: index={rep.first_bad_idx} " f"ref={rep.first_bad_pair[0]} cand={rep.first_bad_pair[1]}"
        )
    if rep.n_within == rep.n_total:
        print("PASS")
        return 0
    print(f"FAIL: rtol={args.rtol} atol={args.atol}")
    return 1


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-ref", help="Compile + run CPU reference, save output")
    g.add_argument("--source", type=pathlib.Path, required=True, help="Source MLIR file")
    g.add_argument("--function", default="module.main", help="Entry function (default module.main)")
    g.add_argument("--input-shape", required=True, help="Shape with dtype suffix, e.g. 1x3x640x640xi8")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--out-dir", type=pathlib.Path, required=True)
    g.add_argument("--iree-compile", type=pathlib.Path, default=DEFAULT_IREE_COMPILE)
    g.add_argument("--iree-run", type=pathlib.Path, default=DEFAULT_IREE_RUN)
    g.set_defaults(func=cmd_gen_ref)

    c = sub.add_parser("compare", help="Element-wise compare two binary tensors")
    c.add_argument("--reference", type=pathlib.Path, required=True)
    c.add_argument("--candidate", type=pathlib.Path, required=True)
    c.add_argument("--shape", required=True, help="Shape with dtype suffix, e.g. 1x84x8400xf32")
    c.add_argument("--rtol", type=float, default=1e-3)
    c.add_argument("--atol", type=float, default=1e-3)
    c.set_defaults(func=cmd_compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
