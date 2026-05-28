"""On-board QNN kernel benchmark runner.

Walks a compile bundle (produced by `qnn_e2e_compile_all.py`), pushes each
per-kernel `.qnn-ctx` to the QRB5165, runs it through `qnn-net-run`
with profiling, and dumps the timing report next to the kernel artifact.

Bundle layout this consumes:
    <bundle>/targets/qnn_gpu/kernels_cache/board_gpu_<hash>/<kernel>.qnn-ctx

Output appended under:
    <bundle>/targets/qnn_gpu/benchmarks/<kernel>/
        input.raw
        input_list.txt
        output.raw
        bench.log

`benchmarks_summary.md` rolls everything up into one table.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re
import struct
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from kernels import manifest as _kmanifest  # noqa: E402

_LOG = logging.getLogger(__name__)


def _ssh(host: str, cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", host, cmd], capture_output=True, text=True, check=False)


def _scp_to(host: str, src: pathlib.Path, dst: str) -> None:
    res = subprocess.run(
        ["scp", "-q", str(src), f"{host}:{dst}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"scp {src} -> {host}:{dst} failed: {res.stderr}")


def _scp_from(host: str, src: str, dst: pathlib.Path) -> None:
    subprocess.run(
        ["scp", "-q", f"{host}:{src}", str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )


def _bench_one(
    kernel: _kmanifest.KernelEntry,
    ctxbin: pathlib.Path,
    bench_dir: pathlib.Path,
    ssh_host: str,
    board_qairt: str,
    backend_lib: str,
    iters: int,
) -> dict:
    bench_dir.mkdir(parents=True, exist_ok=True)
    # Build a deterministic input matching the manifest signature's `in`
    # operands. For the static-fp32 1×16 kernels we just write zero-filled
    # raw fp32 buffers (timing doesn't depend on values).
    ins = [op for op in kernel.signature.operands if op.role == "in"]
    in_lines: list[str] = []
    for i, op in enumerate(ins):
        # Parse element count from `tensor<DIMSxTYPE>`.
        m = re.match(r"tensor<([0-9x?]+)x([a-z0-9]+)>", op.tensor)
        if m is None:
            raise ValueError(f"unsupported tensor type: {op.tensor}")
        dims = m.group(1).split("x")
        if "?" in dims:
            raise ValueError(f"dynamic shapes not supported in bench: {op.tensor}")
        n = 1
        for d in dims:
            n *= int(d)
        # Always write fp32 (works for f32 kernels — int8 kernels would
        # need different packing).
        raw = bench_dir / f"in{i}.raw"
        raw.write_bytes(struct.pack(f"<{n}f", *([0.0] * n)))
        # Tensor name comes from the .qnn.cpp's APP_WRITE tensor; we use
        # generic fallback names (input, in0, a, etc.) — try the canonical
        # one for our kernels.
        # For now hardcode a name list per common kernel; the entry symbol
        # itself doesn't carry the tensor name so we must guess.
        tensor_name = "input" if len(ins) == 1 else ("a" if i == 0 else "b")
        _scp_to(ssh_host, raw, f"/tmp/qnn_probe/bench_in{i}.raw")
        in_lines.append(f"{tensor_name}:=/tmp/qnn_probe/bench_in{i}.raw")
    list_path = bench_dir / "input_list.txt"
    list_path.write_text(" ".join(in_lines) + "\n")
    _scp_to(ssh_host, list_path, "/tmp/qnn_probe/bench_input_list.txt")
    _scp_to(ssh_host, ctxbin, "/tmp/qnn_probe/bench_kernel.qnn-ctx")

    # Run with profiling. iters is encoded by repeating the input in the
    # input list; qnn-net-run runs once per line. We replicate by writing N
    # lines in the list file.
    if iters > 1:
        list_text = list_path.read_text().strip()
        list_path.write_text("\n".join([list_text] * iters) + "\n")
        _scp_to(ssh_host, list_path, "/tmp/qnn_probe/bench_input_list.txt")

    cmd = (
        "cd /tmp/qnn_probe && rm -rf bench_out && mkdir bench_out && "
        "export LD_LIBRARY_PATH=/tmp/qnn_probe/lib && "
        f"./bin/qnn-net-run "
        f"--retrieve_context bench_kernel.qnn-ctx "
        f"--backend lib/{backend_lib} "
        f"--input_list bench_input_list.txt "
        f"--output_dir bench_out "
        f"--profiling_level basic 2>&1"
    )
    res = _ssh(ssh_host, cmd)
    log_path = bench_dir / "bench.log"
    log_path.write_text((res.stdout or "") + "\n--- stderr ---\n" + (res.stderr or ""))

    # Pull the profile log for human inspection.
    profile_log = bench_dir / "qnn-profiling-data_0.log"
    _scp_from(ssh_host, "/tmp/qnn_probe/bench_out/qnn-profiling-data_0.log", profile_log)

    # Decode it via the host-side qnn-profile-viewer (the .log is a binary
    # format).
    ips: float | None = None
    if profile_log.exists():
        qairt = pathlib.Path(
            os.environ.get(
                "QAIRT_SDK_ROOT",
                "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326",
            )
        )
        viewer = qairt / "bin" / "x86_64-linux-clang" / "qnn-profile-viewer"
        if viewer.exists():
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{qairt}/lib/x86_64-linux-clang:" f"{qairt}/lib/clang14-runtime:" + env.get(
                "LD_LIBRARY_PATH", ""
            )
            vres = subprocess.run(
                [str(viewer), "--input_log", str(profile_log)],
                capture_output=True,
                text=True,
                env=env,
            )
            (bench_dir / "qnn-profile-viewer.txt").write_text(vres.stdout)
            m = re.search(r"NetRun IPS[^:]*:\s*([\d.]+)\s*inf/sec", vres.stdout)
            if m:
                ips = float(m.group(1))

    return {
        "kernel": kernel.name,
        "ok": res.returncode == 0,
        "iters": iters,
        "ips": ips,
        "us_per_inf": (1e6 / ips) if ips else None,
        "log": str(log_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=pathlib.Path, help="Bundle dir produced by qnn_e2e_compile_all.py")
    p.add_argument("--kernel-manifest", required=True, type=pathlib.Path)
    p.add_argument("--ssh-host", default="qdev")
    p.add_argument("--board-qairt-root", default="/tmp/qnn_probe")
    p.add_argument("--backend-lib", default="libQnnGpu.so", help="QNN backend lib basename for qnn-net-run.")
    p.add_argument("--iters", type=int, default=20, help="How many iterations to time (replicates input list).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    bundle = args.bundle.resolve()
    qnn_dir = bundle / "targets" / "qnn_gpu"
    if not qnn_dir.exists():
        _LOG.error("no qnn_gpu target dir at %s — run qnn_e2e_compile_all.py first", qnn_dir)
        return 1

    full = _kmanifest.load(args.kernel_manifest)
    by_name = {k.name: k for k in full.kernels}

    bench_root = qnn_dir / "benchmarks"
    bench_root.mkdir(parents=True, exist_ok=True)

    cache_dir = qnn_dir / "kernels_cache"
    results: list[dict] = []
    for ctxbin in sorted(cache_dir.glob("board_gpu_*/*.qnn-ctx")):
        kname = ctxbin.stem
        kernel = by_name.get(kname)
        if kernel is None:
            _LOG.warning("ctxbin %s has no matching manifest entry; skipping", ctxbin)
            continue
        # Skip kernels with non-fp32 signatures for this round (the input
        # generator only writes fp32 buffers).
        if any("f32" not in op.tensor for op in kernel.signature.operands):
            _LOG.info("skipping %s (non-fp32)", kname)
            continue
        _LOG.info("benching %s …", kname)
        try:
            r = _bench_one(
                kernel,
                ctxbin,
                bench_root / kname,
                args.ssh_host,
                args.board_qairt_root,
                args.backend_lib,
                args.iters,
            )
        except Exception as e:
            _LOG.warning("bench %s failed: %s", kname, e)
            r = {"kernel": kname, "ok": False, "error": str(e)}
        results.append(r)

    # Roll up into a markdown summary.
    summary = bench_root / "benchmarks_summary.md"
    lines: list[str] = ["# QNN GPU on-board benchmarks\n"]
    lines.append(f"- backend: `{args.backend_lib}` on `{args.ssh_host}`")
    lines.append(f"- iters per kernel: {args.iters}\n")
    lines.append("| Kernel | OK | inf/sec | µs/inf | Log |")
    lines.append("|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: -(x.get("ips") or 0)):
        ok = "✅" if r.get("ok") else "❌"
        ips = f"{r['ips']:.1f}" if r.get("ips") else "?"
        us = f"{r['us_per_inf']:.0f}" if r.get("us_per_inf") else "?"
        lines.append(f"| `{r['kernel']}` | {ok} | " f"{ips} | {us} | " f"`benchmarks/{r['kernel']}/bench.log` |")
    summary.write_text("\n".join(lines) + "\n")
    print(f"\n  bench summary: {summary}")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
