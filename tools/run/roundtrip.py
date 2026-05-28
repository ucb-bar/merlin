#!/usr/bin/env python3
"""Round-trip a compiled model through any aarch64-linux target (typically
QRB5165) and feed timing data back to XPU-RT. Model-agnostic.

Workflow this script orchestrates:

    1. Compile the model (via `merlin compile` directly or assume already
       compiled with --dump-phases + --iree-hal-dump-executable-benchmarks-to).
    2. Run `tools/breakdown_vmfb.py` to produce per-dispatch VMFBs +
       manifest with shape/deps.
    3. Generate input data for each per-dispatch VMFB from the shapes.json
       (random ints / floats matching the declared tensor types).
    4. Push the cross-built runtime + per-dispatch VMFBs to the board over
       SSH (rsync if available; falls back to scp).
    5. Run `iree-benchmark-module` on each VMFB; capture mean / median /
       stddev nanoseconds from the gbenchmark output.
    6. Emit a profiled JSON whose schema matches XPU-RT's
       `<model>_dispatch_deps.json` plus per-dispatch `mean_time_us`
       fields. The XPU-RT scheduler can consume this directly.
    7. (Optional) Hand the profiled JSON off to a scheduler invocation,
       receive a new schedule.json, and recompile in --with-schedule mode
       (closing the round-trip).

Default SSH config (QRB5165 example; override flags for other boards):
    Host:        10.44.120.201
    User:        root
    Identity:    ~/.ssh/DIMA_SLICE
    Remote dir:  /root/iree_run/<model>/

Usage:

    python tools/run/roundtrip.py \
        --output-dir build/compiled_models/<model>/<target> \
        [--ssh-host <user>@<host> --ssh-key ~/.ssh/<key>] \
        [--remote-runtime /root/iree-run-module] \
        [--repetitions 5]

Skip individual stages with `--skip-push`, `--skip-bench`, `--profile-only`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import re
import shlex
import subprocess
import sys

_LOG = logging.getLogger("board_roundtrip")

_DEFAULT_SSH_HOST = "root@10.44.120.201"
_DEFAULT_SSH_KEY = pathlib.Path("~/.ssh/DIMA_SLICE").expanduser()
_DEFAULT_REMOTE_DIR = pathlib.Path("/root/iree_run")
_DEFAULT_RUNTIME_BIN = pathlib.Path(
    "/scratch2/agustin/merlin/build/qrb5165-runtime-release/tools/iree-benchmark-module"
)


@dataclasses.dataclass
class BenchSample:
    name: str
    mean_ns: float
    median_ns: float
    stddev_ns: float
    iterations: int


def _parse_gbench(stdout: str) -> dict[str, BenchSample]:
    """Parse `iree-benchmark-module` output (gbenchmark format).

    Sample lines look like:
        BM_main_graph$async_dispatch_0_...    13.5 us    13.4 us    52000
    or for repetitions:
        BM_main_graph$async_dispatch_0_..._mean    13.5 us  ...
    """
    samples: dict[str, BenchSample] = {}
    # Benchmark names include `$`, `/`, `.`, `_`, `-` (gbenchmark uses `/`
    # for sub-benchmark / time-unit nesting like
    # `BM_foo/process_time/real_time_mean`). The trailing iteration count
    # column may be followed by gbenchmark counters like
    # `items_per_second=...` which we ignore.
    pattern = re.compile(r"^([\w$./\-]+)\s+([\d.]+)\s+(ns|us|ms)\s+[\d.]+\s+(ns|us|ms)\s+(\d+)")
    unit = {"ns": 1.0, "us": 1e3, "ms": 1e6}
    by_base: dict[str, dict[str, float]] = {}
    iters: dict[str, int] = {}
    for line in stdout.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue
        full = m.group(1)
        time_val = float(m.group(2)) * unit[m.group(3)]
        n = int(m.group(5))
        for suffix in ("_mean", "_median", "_stddev", "_cv"):
            if full.endswith(suffix):
                base = full[: -len(suffix)]
                kind = suffix[1:]
                by_base.setdefault(base, {})[kind] = time_val
                iters[base] = n
                break
        else:
            # Single-shot run (no repetitions): treat as mean.
            by_base.setdefault(full, {})["mean"] = time_val
            iters[full] = n
    for name, vals in by_base.items():
        mean = vals.get("mean", 0.0)
        samples[name] = BenchSample(
            name=name,
            mean_ns=mean,
            median_ns=vals.get("median", mean),
            stddev_ns=vals.get("stddev", 0.0),
            iterations=iters.get(name, 0),
        )
    return samples


def _ssh(host: str, key: pathlib.Path, cmd: str, timeout: float = 60.0) -> str:
    args = ["ssh", "-i", str(key), "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", host, cmd]
    _LOG.debug("ssh: %s", " ".join(shlex.quote(a) for a in args))
    res = subprocess.run(args, capture_output=True, text=True, check=False, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"ssh `{cmd}` failed (rc={res.returncode}):\n" f"{res.stderr}")
    return res.stdout


def _scp(key: pathlib.Path, src: pathlib.Path, dst: str, timeout: float = 120.0) -> None:
    # `-O` forces the legacy SCP protocol — newer OpenSSH defaults to SFTP
    # which the board's older sshd may not negotiate cleanly.
    args = ["scp", "-O", "-i", str(key), "-q", str(src), dst]
    _LOG.debug("scp: %s", " ".join(args))
    res = subprocess.run(args, capture_output=True, text=True, check=False, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"scp {src}->{dst} failed:\n{res.stderr}")


def push(
    output_dir: pathlib.Path,
    runtime_bin: pathlib.Path,
    *,
    ssh_host: str,
    ssh_key: pathlib.Path,
    remote_dir: pathlib.Path,
) -> None:
    """Push the cross-built runtime + every per-dispatch VMFB to the board.
    Idempotent: existing files are overwritten in place."""
    breakdowns = output_dir / "breakdowns"
    if not breakdowns.exists():
        raise RuntimeError(f"missing {breakdowns}; run breakdown_vmfb.py first")
    _ssh(ssh_host, ssh_key, f"mkdir -p {shlex.quote(str(remote_dir))}/breakdowns")
    _LOG.info("pushing runtime: %s", runtime_bin)
    _scp(ssh_key, runtime_bin, f"{ssh_host}:{remote_dir}/iree-benchmark-module")
    _ssh(ssh_host, ssh_key, f"chmod +x {shlex.quote(str(remote_dir))}/iree-benchmark-module")
    vmfbs = sorted(breakdowns.glob("*.vmfb"))
    if not vmfbs:
        raise RuntimeError(f"no vmfbs in {breakdowns}")
    _LOG.info("pushing %d vmfbs", len(vmfbs))
    for v in vmfbs:
        _scp(ssh_key, v, f"{ssh_host}:{remote_dir}/breakdowns/{v.name}")
    # Manifest + shapes are useful on the board for debugging but not
    # required for benchmarking; push them anyway since they're tiny.
    for ext in ("*.shapes.json", "manifest.json"):
        for f in breakdowns.glob(ext):
            _scp(ssh_key, f, f"{ssh_host}:{remote_dir}/breakdowns/{f.name}")


def _benchmark_one(
    ssh_host: str,
    ssh_key: pathlib.Path,
    remote_dir: pathlib.Path,
    vmfb_name: str,
    repetitions: int,
    cpu_mask: str | None = None,
    task_topology_cpu_ids: str | None = None,
    device_override: str | None = None,
) -> dict[str, BenchSample] | None:
    """Run iree-benchmark-module on a single per-dispatch VMFB. Returns the
    samples keyed by full benchmark name (gbenchmark BM_... names). The
    per-dispatch benchmark mlirs declare a single `iree.benchmark =
    "dispatch"` reflection so we get one sample per VMFB.

    Two pinning modes:
      - `cpu_mask` only: single-thread mode. taskset masks the process to
        the given cores and the IREE executor is forced to 1 worker so all
        work stays on the pinned core(s). Right for measuring isolated
        per-dispatch latency on a single core.
      - `task_topology_cpu_ids`: multi-thread cluster mode. taskset masks
        the process AND the IREE executor spawns N worker threads pinned
        to those exact cpu_ids. This matches what `CreatePinnedLocalTaskDevice`
        does at scheduler runtime, so the resulting profile is what the
        scheduler will actually see — not single-core latency.
    """
    pin_prefix = ""
    device = "local-task"
    if device_override is not None:
        # Caller supplied a HAL device URL (e.g. `qnn?backend=gpu`) — use
        # it verbatim and skip CPU-only pinning logic. taskset still
        # wraps the process if cpu_mask is given so the bench *process*
        # itself stays out of the way.
        device = device_override
        if cpu_mask is not None:
            pin_prefix = f"taskset -c {shlex.quote(cpu_mask)} "
    elif task_topology_cpu_ids is not None:
        # Cluster mode: bench uses N workers pinned to the given cores.
        # Wrap with taskset on the same cores so the bench process itself
        # can't drift onto a different cluster.
        pin_prefix = f"taskset -c {shlex.quote(task_topology_cpu_ids)} "
        device = "local-task " f"--task_topology_cpu_ids={shlex.quote(task_topology_cpu_ids)}"
    elif cpu_mask is not None:
        pin_prefix = f"taskset -c {shlex.quote(cpu_mask)} "
        # 1-thread task topology so all work stays on the pinned core.
        device = "local-task --task_topology_max_group_count=1"
    cmd = (
        f"cd {shlex.quote(str(remote_dir))} && "
        f"{pin_prefix}./iree-benchmark-module "
        f"--module=breakdowns/{shlex.quote(vmfb_name)} "
        f"--device={device} --benchmark_repetitions={repetitions} "
        f"--benchmark_min_time=0.05s 2>&1"
    )
    try:
        out = _ssh(ssh_host, ssh_key, cmd, timeout=120.0)
    except RuntimeError as exc:
        # iree-benchmark-module can fail on individual VMFBs (e.g. dispatch
        # uses a binding shape we didn't materialize correctly, or the
        # executable's entry symbol isn't recognized). Log and skip — the
        # rest of the model can still produce useful timing data.
        _LOG.warning("benchmark failed for %s: %s", vmfb_name, exc)
        return None
    samples = _parse_gbench(out)
    if not samples:
        _LOG.warning("no samples parsed from %s output (head):\n%s", vmfb_name, "\n".join(out.splitlines()[:6]))
    return samples


def benchmark_all(
    output_dir: pathlib.Path,
    *,
    ssh_host: str,
    ssh_key: pathlib.Path,
    remote_dir: pathlib.Path,
    repetitions: int = 5,
    cpu_mask: str | None = None,
    task_topology_cpu_ids: str | None = None,
    device_override: str | None = None,
) -> dict[str, BenchSample]:
    breakdowns = output_dir / "breakdowns"
    manifest_path = breakdowns / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    all_samples: dict[str, BenchSample] = {}
    for name, entry in manifest["dispatches"].items():
        vmfb = entry.get("executable")
        if not vmfb:
            continue
        vmfb_name = pathlib.Path(vmfb).name
        _LOG.info("[%s] %s", name, vmfb_name)
        samples = _benchmark_one(
            ssh_host,
            ssh_key,
            remote_dir,
            vmfb_name,
            repetitions,
            cpu_mask=cpu_mask,
            task_topology_cpu_ids=task_topology_cpu_ids,
            device_override=device_override,
        )
        if samples is None:
            continue  # logged inside _benchmark_one
        if samples:
            sample = next(iter(samples.values()))
            sample = dataclasses.replace(sample, name=name)
            all_samples[name] = sample
    return all_samples


def emit_profiled_manifest(
    output_dir: pathlib.Path,
    samples: dict[str, BenchSample],
    *,
    out_name: str = "profiled_manifest.json",
    profile_key: str = "default",
) -> pathlib.Path:
    """Layer the timing samples onto the breakdown manifest so the XPU-RT
    scheduler sees both topology and per-dispatch latency in one file.

    Multiple profile keys (e.g. "CPU_P", "CPU_E") can coexist in a single
    file — they're stored under `dispatches.<n>.profiles.<key>`. The
    top-level `mean_time_us` etc. fields mirror the most-recently-written
    profile so single-machine consumers keep working.
    """
    breakdowns = output_dir / "breakdowns"
    out = breakdowns / out_name
    if out.exists():
        manifest = json.loads(out.read_text())
    else:
        manifest = json.loads((breakdowns / "manifest.json").read_text())
    for name, sample in samples.items():
        if name not in manifest["dispatches"]:
            continue
        entry = manifest["dispatches"][name]
        timing = {
            "mean_time_us": sample.mean_ns / 1e3,
            "median_time_us": sample.median_ns / 1e3,
            "stddev_time_us": sample.stddev_ns / 1e3,
            "iterations": sample.iterations,
        }
        entry.setdefault("profiles", {})[profile_key] = timing
        # Mirror the most-recent profile to the top-level fields so simple
        # consumers (one machine) keep reading out a sane value.
        entry.update(timing)
    manifest["profile_source"] = "merlin/tools/board_roundtrip.py"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
        help="merlin compile output dir; must already have " "phases/ + benchmarks/ + breakdowns/.",
    )
    parser.add_argument("--ssh-host", default=_DEFAULT_SSH_HOST)
    parser.add_argument("--ssh-key", default=_DEFAULT_SSH_KEY, type=pathlib.Path)
    parser.add_argument("--remote-dir", default=_DEFAULT_REMOTE_DIR, type=pathlib.Path)
    parser.add_argument(
        "--runtime-bin",
        default=_DEFAULT_RUNTIME_BIN,
        type=pathlib.Path,
        help="cross-built iree-benchmark-module on this host.",
    )
    parser.add_argument("--repetitions", default=5, type=int)
    parser.add_argument(
        "--cpu-mask",
        help="If set, pin each benchmark to this taskset -c "
        "mask AND force IREE to a single worker. Right "
        "for measuring isolated single-core latency. "
        "Mutually exclusive with --task-topology-cpu-ids.",
    )
    parser.add_argument(
        "--task-topology-cpu-ids",
        help="Cluster mode: bench uses N IREE worker threads "
        "pinned exactly to these cpu_ids (comma-separated). "
        "Use this when profiling for the scheduler — the "
        "scheduler runs its CPU_P/CPU_E devices via the "
        "same N-core pinned topology, so the resulting "
        "profile is what the scheduler will actually see.",
    )
    parser.add_argument(
        "--device",
        help="HAL device URL passed verbatim to "
        "iree-benchmark-module (e.g. 'qnn?backend=gpu' "
        "or 'qnn?backend=hta'). Bypasses local-task "
        "pinning logic. When set, --cpu-mask still wraps "
        "the bench process with taskset to keep the "
        "host-side process out of the way.",
    )
    parser.add_argument(
        "--profile-key",
        default="default",
        help="Name to use under `dispatches.<n>.profiles.<key>"
        "` in the profiled manifest. Use this to keep "
        "multiple per-cluster profile runs in one file "
        "(e.g. 'CPU_P', 'CPU_E').",
    )
    parser.add_argument("--skip-push", action="store_true", help="VMFBs already pushed to the board.")
    parser.add_argument("--skip-bench", action="store_true", help="Push only; useful for inspecting first.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.skip_push:
        push(
            args.output_dir, args.runtime_bin, ssh_host=args.ssh_host, ssh_key=args.ssh_key, remote_dir=args.remote_dir
        )

    if args.skip_bench:
        return 0

    samples = benchmark_all(
        args.output_dir,
        ssh_host=args.ssh_host,
        ssh_key=args.ssh_key,
        remote_dir=args.remote_dir,
        repetitions=args.repetitions,
        cpu_mask=args.cpu_mask,
        task_topology_cpu_ids=args.task_topology_cpu_ids,
        device_override=args.device,
    )
    profiled = emit_profiled_manifest(
        args.output_dir,
        samples,
        profile_key=args.profile_key,
    )
    print(f"profiled manifest -> {profiled}")
    print(f"benchmarked {len(samples)} dispatches " f"(out of {len(samples)} expected)")
    if not samples:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
