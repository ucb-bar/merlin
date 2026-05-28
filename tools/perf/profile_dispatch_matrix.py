#!/usr/bin/env python3
"""On-board per-(dispatch, target) execution-time profiler.

Phase C of the heterogeneous-scheduling pipeline (model-agnostic). Consumes
Phase B's matrix.json, compiles each feasible per-dispatch MLIR to a
standalone VMFB, SCPs it to an aarch64-linux board (typically QRB5165), and
runs `merlin-dispatch-bench` to measure setup_ms + mean_ms / median_ms /
p99_ms.

Outputs `profiled_manifest.json` extending matrix.json with timing fields
per (dispatch, target) cell: `setup_us`, `mean_us`, `median_us`, `p99_us`.

Usage:
  python tools/perf/profile_dispatch_matrix.py \\
      --matrix <matrix_dir>/matrix.json \\
      --ssh-host <user>@<host> --ssh-identity <key_path> \\
      --board-bench <remote_bench_path> \\
      --out <matrix_dir>/profiled_manifest.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_IREE_COMPILE = REPO_ROOT / "build/host-merlin-release-qrb/tools/iree-compile"
DEFAULT_SSH_IDENTITY = REPO_ROOT.parent / "DIMA_SLICE"

# device URI per target on the board
_DEVICE_URI = {
    "cpu": "local-task",
    "qnn_gpu": "qnn://gpu",
    "qnn_hta": "qnn://hta",
}


_BENCH_RE = re.compile(
    r"setup_ms=(?P<setup>\S+) iters=\d+ warmup=\d+ "
    r"mean_ms=(?P<mean>\S+) median_ms=(?P<median>\S+) "
    r"p99_ms=(?P<p99>\S+) min_ms=(?P<min>\S+) max_ms=(?P<max>\S+)"
)


def _compile(
    iree_compile: pathlib.Path, mlir: pathlib.Path, target: str, vmfb: pathlib.Path, log: pathlib.Path
) -> bool:
    """Compile a per-dispatch MLIR to a standalone VMFB for the given target."""
    args = [str(iree_compile)]
    if target == "cpu":
        args.extend(
            [
                '--iree-hal-target-device=#hal.device.target<"local", '
                '[#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", '
                '{target_triple = "aarch64-linux-gnu"}>]>',
            ]
        )
    elif target == "qnn_gpu":
        args.extend(
            [
                "--iree-plugin=hal_target_qnn",
                '--iree-hal-target-device=#hal.device.target<"qnn", '
                '[#hal.executable.target<"qnn", "qnn-context-binary", '
                '{qnn_backend = "gpu", opaque_binary = true}>]>',
                # Phase 0: failures must be loud. No --iree-hal-qnn-allow-placeholder.
            ]
        )
    elif target == "qnn_hta":
        args.extend(
            [
                "--iree-plugin=hal_target_qnn",
                '--iree-hal-target-device=#hal.device.target<"qnn", '
                '[#hal.executable.target<"qnn", "qnn-context-binary", '
                '{qnn_backend = "hta", opaque_binary = true}>]>',
                # Phase 0: failures must be loud. No --iree-hal-qnn-allow-placeholder.
            ]
        )
    else:
        raise ValueError(f"unknown target {target}")
    args.extend(["-o", str(vmfb), str(mlir)])
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        f.write("# " + " ".join(args) + "\n")
        f.flush()
        return subprocess.run(args, stdout=f, stderr=subprocess.STDOUT).returncode == 0


def _ssh_base(ssh_host: str, ssh_identity: pathlib.Path | None) -> list[str]:
    cmd = ["ssh"]
    if ssh_identity:
        cmd.extend(["-i", str(ssh_identity)])
    cmd.append(ssh_host)
    return cmd


def _scp_base(ssh_identity: pathlib.Path | None) -> list[str]:
    cmd = ["scp", "-q"]
    if ssh_identity:
        cmd.extend(["-i", str(ssh_identity)])
    return cmd


def _bench_on_board(
    ssh_host: str,
    ssh_identity: pathlib.Path | None,
    board_bench: str,
    remote_vmfb: str,
    function: str,
    device_uri: str,
    input_flags: list[str],
    iterations: int,
    warmup: int,
    timeout_s: int,
    qnn_lib_dir: str = "/root/qairt/lib/target",
) -> dict | None:
    """Run merlin-dispatch-bench on the board; parse the output line."""
    # Build the remote command line as a single shell string so we can prefix
    # LD_LIBRARY_PATH (needed for QNN backends to dlopen libQnn*.so).
    # Per-dispatch VMFBs are emitted with default module name "module",
    # so the bench's --function expects "module.<func>" qualification.
    qualified_function = function if "." in function else f"module.{function}"
    # Single-quote the function so the remote shell does not interpret
    # `$` (per-dispatch names like `main_graph$async_dispatch_N` would
    # otherwise be split into `main_graph` plus an empty var expansion).
    # Single-quote --module= and --function= so the remote shell does
    # not expand `$` in canonical names like `main_graph$async_dispatch_N`.
    remote_cmd = (
        f"LD_LIBRARY_PATH={qnn_lib_dir} {board_bench} "
        f"--module={shlex.quote(remote_vmfb)} "
        f"--device={shlex.quote(device_uri)} "
        f"--function={shlex.quote(qualified_function)} "
        f"--iterations={iterations} --warmup={warmup}"
    )
    if input_flags:
        remote_cmd += " " + " ".join(shlex.quote(flag) for flag in input_flags)
    cmd = [*_ssh_base(ssh_host, ssh_identity), remote_cmd]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="replace")[:1000]
        out = proc.stdout.decode(errors="replace")[:500]
        return {"error": f"rc={proc.returncode}", "stderr": err, "stdout": out}
    out = proc.stdout.decode(errors="replace")
    m = _BENCH_RE.search(out)
    if not m:
        return {"error": "parse-failed", "stdout": out[-500:]}
    return {
        "setup_us": float(m.group("setup")) * 1000.0,
        "mean_us": float(m.group("mean")) * 1000.0,
        "median_us": float(m.group("median")) * 1000.0,
        "p99_us": float(m.group("p99")) * 1000.0,
        "min_us": float(m.group("min")) * 1000.0,
        "max_us": float(m.group("max")) * 1000.0,
        "stdout": out[-500:],
    }


def _scp(ssh_host: str, ssh_identity: pathlib.Path | None, local: pathlib.Path, remote: str) -> None:
    # Use default SFTP protocol (no `-O`); legacy SCP runs the remote
    # path through a shell which would expand `$` in canonical names
    # like `main_graph$async_dispatch_N`, silently losing the suffix.
    subprocess.run([*_scp_base(ssh_identity), str(local), f"{ssh_host}:{remote}"], check=True)


def _remote_mkdir(ssh_host: str, ssh_identity: pathlib.Path | None, remote_dir: str) -> None:
    subprocess.run(
        [*_ssh_base(ssh_host, ssh_identity), f"mkdir -p {shlex.quote(remote_dir)}"],
        check=True,
    )


def _sanitize_dispatch_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-$") else "_" for c in name) or "unnamed_dispatch"


def _input_flags_for_cell(
    canonical: str,
    cell: dict,
    input_mode: str,
    capture_dir: pathlib.Path | None,
) -> tuple[list[str], list[pathlib.Path]]:
    sizes = list(cell.get("binding_byte_sizes", []))
    if input_mode == "zero":
        return ([f"--zero-input={size}xi8" for size in sizes], [])
    if capture_dir is None:
        raise ValueError("--capture-dir is required with --input-mode=captured")
    dispatch_dir = capture_dir / _sanitize_dispatch_name(canonical)
    flags: list[str] = []
    local_files: list[pathlib.Path] = []
    for i, size in enumerate(sizes):
        local_file = dispatch_dir / f"input_{i}.bin"
        if not local_file.exists():
            raise FileNotFoundError(f"missing captured input for {canonical}: {local_file}")
        flags.append(f"--input={size}xi8=@INPUT_{i}")
        local_files.append(local_file)
    return flags, local_files


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix", type=pathlib.Path, required=True)
    p.add_argument("--ssh-host", default="root@10.44.120.201")
    p.add_argument("--ssh-identity", type=pathlib.Path, default=DEFAULT_SSH_IDENTITY)
    p.add_argument("--board-bench", default="/root/merlin-dispatch-bench")
    p.add_argument("--board-vmfb-dir", default="/root/dispatch_profile")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timeout-per-cell-s", type=int, default=120)
    p.add_argument("--iree-compile", type=pathlib.Path, default=DEFAULT_IREE_COMPILE)
    p.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("/tmp/dispatch_profile_work"))
    p.add_argument("--targets", default="cpu,qnn_gpu,qnn_hta", help="Subset of targets to profile")
    p.add_argument(
        "--input-mode",
        choices=("captured", "zero"),
        default="captured",
        help="Replay standalone dispatches with captured files or " "with zero-filled synthetic buffers",
    )
    p.add_argument(
        "--capture-dir",
        type=pathlib.Path,
        default=None,
        help="Per-dispatch captured inputs root; layout " "<capture-dir>/<canonical>/input_<n>.bin",
    )
    p.add_argument("--skip-compile", action="store_true", help="Reuse existing per-dispatch VMFBs from workdir")
    args = p.parse_args(argv)

    matrix = json.loads(args.matrix.read_text())
    targets = [t.strip() for t in args.targets.split(",")]
    args.workdir.mkdir(parents=True, exist_ok=True)
    # Ensure remote dir exists
    _remote_mkdir(args.ssh_host, args.ssh_identity, args.board_vmfb_dir)

    profiled = {
        "matrix_path": str(args.matrix),
        "targets": targets,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "dispatches": {},
    }
    if "dispatch_graph" in matrix:
        profiled["dispatch_graph"] = matrix["dispatch_graph"]

    n_total = len(matrix["dispatches"]) * len(targets)
    n_done = 0
    for canonical, row in matrix["dispatches"].items():
        out_row = {}
        for target in targets:
            cell = row.get(target, {"feasible": False})
            n_done += 1
            tag = f"[{n_done:>3}/{n_total}] {canonical}/{target}"
            if not cell.get("feasible"):
                print(f"{tag} SKIP (infeasible)")
                out_row[target] = {**cell, "profile": None}
                continue
            mlir = pathlib.Path(cell["mlir"])
            func = cell["func"]
            local_vmfb = args.workdir / target / f"{canonical}.vmfb"
            local_log = args.workdir / target / f"{canonical}.compile.log"
            if not args.skip_compile or not local_vmfb.exists():
                ok = _compile(args.iree_compile, mlir, target, local_vmfb, local_log)
                if not ok:
                    print(f"{tag} compile-FAIL (see {local_log})")
                    out_row[target] = {**cell, "profile": {"error": "compile-failed", "log": str(local_log)}}
                    continue
            remote_vmfb = f"{args.board_vmfb_dir}/{target}__{canonical}.vmfb"
            try:
                _scp(args.ssh_host, args.ssh_identity, local_vmfb, remote_vmfb)
            except subprocess.CalledProcessError as exc:
                print(f"{tag} scp-FAIL: {exc}")
                out_row[target] = {**cell, "profile": {"error": "scp-failed"}}
                continue
            try:
                input_flags, local_inputs = _input_flags_for_cell(canonical, cell, args.input_mode, args.capture_dir)
            except (ValueError, FileNotFoundError) as exc:
                print(f"{tag} input-FAIL: {exc}")
                out_row[target] = {
                    **cell,
                    "profile": {
                        "error": "input-prep-failed",
                        "detail": str(exc),
                    },
                }
                continue
            remote_inputs: list[str] = []
            if local_inputs:
                remote_input_dir = f"{args.board_vmfb_dir}/" f"{_sanitize_dispatch_name(canonical)}.inputs"
                try:
                    _remote_mkdir(args.ssh_host, args.ssh_identity, remote_input_dir)
                    for i, local_input in enumerate(local_inputs):
                        remote_input = f"{remote_input_dir}/input_{i}.bin"
                        _scp(args.ssh_host, args.ssh_identity, local_input, remote_input)
                        remote_inputs.append(remote_input)
                except subprocess.CalledProcessError as exc:
                    print(f"{tag} input-scp-FAIL: {exc}")
                    out_row[target] = {**cell, "profile": {"error": "input-scp-failed"}}
                    continue
            resolved_input_flags: list[str] = []
            for flag in input_flags:
                resolved = flag
                for i, remote_input in enumerate(remote_inputs):
                    resolved = resolved.replace(f"@INPUT_{i}", f"@{remote_input}")
                resolved_input_flags.append(resolved)
            t0 = time.time()
            bench = _bench_on_board(
                args.ssh_host,
                args.ssh_identity,
                args.board_bench,
                remote_vmfb,
                func,
                _DEVICE_URI[target],
                resolved_input_flags,
                args.iterations,
                args.warmup,
                args.timeout_per_cell_s,
            )
            elapsed = time.time() - t0
            if bench is not None and "error" not in bench:
                print(
                    f"{tag} mean_us={bench['mean_us']:8.1f} " f"setup_us={bench['setup_us']:8.1f}  " f"({elapsed:.1f}s)"
                )
            else:
                err = bench.get("error", "unknown") if bench else "no-result"
                print(f"{tag} bench-FAIL: {err}")
            out_row[target] = {
                **cell,
                "profile": bench,
                "vmfb_remote": remote_vmfb,
                "input_mode": args.input_mode,
                "captured_inputs_remote": remote_inputs,
            }
        profiled["dispatches"][canonical] = out_row

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profiled, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
