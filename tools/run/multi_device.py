#!/usr/bin/env python3
"""Drive merlin_multi_device_runner on any aarch64-linux board from the host.

The runner (samples/common/xpu-rt/multi_device_runner.cc) instantiates
one pinned local-task IREE device per ``--cluster=<name>:<cpu_ids>`` flag
and loads a single full-model VMFB against a multi-device group. This
script handles the host side: push artifacts, run on board over SSH,
pull the output dump, and (optionally) md5-compare to a single-device
baseline run of the same VMFB.

Model-agnostic, board-agnostic. Cluster→CPU pinning is supplied by the
caller; the script doesn't bake in any board topology.

Per-dispatch routing on the board follows the ``stream.affinity`` stamps
emitted by --iree-merlin-schedule-spec at compile time.

Usage (cluster topology is board-specific; the example below matches a
typical QRB5165 big.LITTLE layout — CPU_P = 4-7, CPU_E = 0,1 — but pass
whatever your board uses):

    tools/run/multi_device.py \\
        --vmfb <model.scheduled.vmfb> \\
        --runner <build_dir>/merlin_multi_device_runner \\
        --function main \\
        --input '<shape>=@input.bin' \\
        --cluster device_a:<cpu_ids> \\
        --cluster device_b:<cpu_ids> \\
        --baseline-vmfb <model.unscheduled.vmfb> \\
        --baseline-iree-run-module <build_dir>/iree-run-module
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_HOST = "qdev"
DEFAULT_REMOTE_DIR = "/data/local/tmp/merlin_multi_device"


@dataclasses.dataclass
class Args:
    vmfb: Path
    runner: Path
    function: str
    inputs: list[str]
    clusters: list[str]
    extra_files: list[Path]
    baseline_vmfb: Path | None
    baseline_iree_run_module: Path | None
    baseline_cluster: str
    host: str
    remote_dir: str
    output_dump_prefix: str
    keep_remote: bool
    trace: bool


def parse() -> Args:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--vmfb", required=True, type=Path)
    p.add_argument("--runner", required=True, type=Path, help="Path to cross-built merlin_multi_device_runner binary")
    p.add_argument("--function", default="main")
    p.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=[],
        help="Repeatable; iree-run-module style input spec. "
        "Use absolute @<path> for binary inputs; the file "
        "will be pushed to the board.",
    )
    p.add_argument(
        "--cluster",
        action="append",
        dest="clusters",
        default=[],
        help="Repeatable; <name>:<cpu_ids_csv>. Order = HAL " "device-group index. Required >=1.",
    )
    p.add_argument(
        "--extra-file",
        action="append",
        dest="extra_files",
        default=[],
        type=Path,
        help="Extra files to push to remote_dir (e.g. .npy " "input arrays).",
    )
    p.add_argument(
        "--baseline-vmfb",
        type=Path,
        default=None,
        help="Optional: single-device VMFB of the same model "
        "(unscheduled). When set, we run iree-run-module "
        "against it on the board and md5-compare outputs.",
    )
    p.add_argument(
        "--baseline-iree-run-module",
        type=Path,
        default=None,
        help="Path to cross-built iree-run-module on host; "
        "required with --baseline-vmfb if not already on "
        "the board at <remote_dir>/iree-run-module.",
    )
    p.add_argument("--baseline-cluster", default="4,5,6,7", help="CPU ids for taskset around the baseline run.")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    p.add_argument(
        "--output-dump-prefix",
        default="out_multi",
        help="Multi-device runner dumps each output as " "<prefix>.<i>.bin (pulled back to host).",
    )
    p.add_argument("--keep-remote", action="store_true", help="Don't rm -rf the remote dir at the end.")
    p.add_argument(
        "--trace",
        action="store_true",
        help="Set IREE_HAL_TRACE=1 on board so per-dispatch " "device routing is observable on stderr.",
    )
    args = p.parse_args()
    if not args.clusters:
        p.error("--cluster required at least once")
    return Args(**vars(args))


def ssh(host: str, cmd: str, *, capture: bool = False) -> subprocess.CompletedProcess:
    full = ["ssh", host, "bash", "-lc", cmd]
    print(f"[ssh] {host}: {cmd}", file=sys.stderr)
    return subprocess.run(full, check=True, text=True, capture_output=capture)


def scp(src: Path, host: str, dst: str) -> None:
    print(f"[scp] {src} -> {host}:{dst}", file=sys.stderr)
    subprocess.run(["scp", "-q", str(src), f"{host}:{dst}"], check=True)


def scp_pull(host: str, src: str, dst: Path) -> None:
    print(f"[scp] {host}:{src} -> {dst}", file=sys.stderr)
    subprocess.run(["scp", "-q", f"{host}:{src}", str(dst)], check=True)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_input_specs(specs: list[str], remote_dir: str) -> list[str]:
    """Rewrite ``shape=@host_path`` to ``shape=@remote_dir/basename``."""
    out: list[str] = []
    for s in specs:
        if "=@" in s:
            shape, _, path = s.partition("=@")
            out.append(f"{shape}=@{remote_dir}/{Path(path).name}")
        else:
            out.append(s)
    return out


def main() -> int:
    args = parse()
    host = args.host
    remote = args.remote_dir

    # 1. Set up remote dir.
    ssh(host, f"mkdir -p {shlex.quote(remote)} && rm -f {shlex.quote(remote)}/*.bin")

    # 2. Push runner + vmfb + extra files + binary inputs.
    scp(args.runner, host, f"{remote}/{args.runner.name}")
    ssh(host, f"chmod +x {shlex.quote(remote)}/{args.runner.name}")
    scp(args.vmfb, host, f"{remote}/{args.vmfb.name}")
    for f in args.extra_files:
        scp(f, host, f"{remote}/{f.name}")
    for s in args.inputs:
        if "=@" in s:
            _, _, p = s.partition("=@")
            scp(Path(p), host, f"{remote}/{Path(p).name}")

    # 3. Build the runner command.
    runner_args = [
        f"./{args.runner.name}",
        f"--module={args.vmfb.name}",
        f"--function={args.function}",
        f"--output_dump={args.output_dump_prefix}",
    ]
    for c in args.clusters:
        runner_args.append(f"--cluster={c}")
    for s in remote_input_specs(args.inputs, "."):
        runner_args.append(f"--input={s}")
    if args.trace:
        runner_args.append("--trace_execution")
    cmd = " ".join(shlex.quote(a) for a in runner_args)
    env = "IREE_HAL_TRACE=1 " if args.trace else ""
    ssh(host, f"cd {shlex.quote(remote)} && {env}{cmd}")

    # 4. Pull output dumps + md5.
    multi_md5: dict[int, str] = {}
    listing = (
        ssh(host, f"cd {shlex.quote(remote)} && ls {args.output_dump_prefix}.*.bin", capture=True)
        .stdout.strip()
        .splitlines()
    )
    for line in listing:
        idx = int(line.rsplit(".", 2)[-2])
        local = Path(f"./{Path(line).name}")
        scp_pull(host, f"{remote}/{Path(line).name}", local)
        multi_md5[idx] = md5_of(local)
        print(f"[multi_device] output[{idx}] md5={multi_md5[idx]} ({local})")

    # 5. Optional: baseline single-device run for comparison.
    baseline_md5: dict[int, str] = {}
    if args.baseline_vmfb is not None:
        if args.baseline_iree_run_module is not None:
            scp(args.baseline_iree_run_module, host, f"{remote}/iree-run-module")
            ssh(host, f"chmod +x {shlex.quote(remote)}/iree-run-module")
        scp(args.baseline_vmfb, host, f"{remote}/{args.baseline_vmfb.name}")
        # iree-run-module dumps outputs to stdout in the same shaped buffer
        # format; we redirect to file and parse the binary form via the
        # `--output=@...` flag.
        bargs = [
            "taskset",
            "-c",
            args.baseline_cluster,
            "./iree-run-module",
            f"--module={args.baseline_vmfb.name}",
            f"--function={args.function}",
            "--device=local-task",
        ]
        for s in remote_input_specs(args.inputs, "."):
            bargs.append(f"--input={s}")
        # We dump each output as a binary blob via the `--output` flag.
        # Note: iree-run-module's --output expects one spec per output;
        # for a single-output model that means one --output=@file.
        bargs.append("--output=@out_baseline.0.bin")
        bcmd = " ".join(shlex.quote(a) for a in bargs)
        ssh(host, f"cd {shlex.quote(remote)} && {bcmd}")
        scp_pull(host, f"{remote}/out_baseline.0.bin", Path("./out_baseline.0.bin"))
        baseline_md5[0] = md5_of(Path("./out_baseline.0.bin"))
        print(f"[baseline]      output[0] md5={baseline_md5[0]}")

    # 6. Compare.
    rc = 0
    if baseline_md5:
        for k, v in baseline_md5.items():
            mv = multi_md5.get(k)
            if mv == v:
                print(f"[OK] output[{k}] md5 matches: {v}")
            else:
                print(f"[FAIL] output[{k}] md5 mismatch: " f"baseline={v} multi={mv}")
                rc = 1

    # 7. Cleanup.
    if not args.keep_remote:
        ssh(host, f"rm -rf {shlex.quote(remote)}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
