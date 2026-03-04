#!/usr/bin/env python3
"""
Remote OFAT sweep runner over SSH (host-side).

Key features:
- OFAT sweeps: vary ONE factor at a time vs baseline
- optional min_all / max_all corner runs
- optional small pairwise 2D grids for interactions
- supports up to 8-core masks easily
- parses CSV_HEADER/CSV_ROW from the benchmark binary output
- stores metadata + parsed metrics into one results CSV
- writes full logs per run

Example:
  python3 run_remote_sweep.py \
    --ssh_host 10.44.86.251 --ssh_user root --remote_dir /home/baseline \
    --bin_name benchmark-baseline-dual-model-async-run \
    --dronet_vmfb dronet.q.int8.vmfb --mlp_vmfb mlp.q.int8.vmfb \
    --out_csv results/sweep.csv --logs_dir results/logs \
    --repeats 3 \
    --mlp_hz 10,20,30 \
    --duration_s 10 --warmup_s 2 \
    --report_hz 0 \
    --dronet_sensor_hz 30,60,120 \
    --dronet_inflight 4,8,16 \
    --mlp_inflight 1,2 \
    --auto_core_masks_1to8 \
    --include_minmax 1 \
    --pairwise_grids core_mask:dronet_inflight,mlp_hz:mlp_inflight
"""

import argparse
import csv
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import hashlib


# ---------------------------
# Helpers
# ---------------------------

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sanitize(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def parse_csv_list(s: str) -> List[str]:
    if s is None:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in parse_csv_list(s)]


def parse_int_list(s: str) -> List[int]:
    vals: List[int] = []
    for x in parse_csv_list(s):
        if x.startswith(("0x", "0X")):
            vals.append(int(x, 16))
        else:
            vals.append(int(x))
    return vals


def popcount_u64(x: int) -> int:
    x &= (1 << 64) - 1
    return bin(x).count("1")


def first(xs, name: str):
    if not xs:
        raise SystemExit(f"Empty list for {name}")
    return xs[0]


def minv(xs, name: str):
    if not xs:
        raise SystemExit(f"Empty list for {name}")
    return min(xs)


def maxv(xs, name: str):
    if not xs:
        raise SystemExit(f"Empty list for {name}")
    return max(xs)


def extract_csv(output_text: str) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    header = None
    row = None
    for line in output_text.splitlines():
        line = line.strip()
        if line.startswith("CSV_HEADER,"):
            header = line[len("CSV_HEADER,"):].split(",")
        elif line.startswith("CSV_ROW,"):
            row = line[len("CSV_ROW,"):].split(",")
    return header, row


# ---------------------------
# Run config
# ---------------------------

@dataclass(frozen=True)
class RunConfig:
    mlp_hz: float
    duration_s: float
    warmup_s: float
    report_hz: float
    dronet_sensor_hz: float
    mlp_sensor_hz: float
    dronet_inflight: int
    mlp_inflight: int
    core_mask: int

    dronet_fn: str
    mlp_fn: str
    driver: str

    plan_kind: str       # baseline/sweep/min_all/max_all/grid2d
    sweep_param: str     # which param varied for OFAT; "" otherwise
    sweep_value: str     # value used for the sweep param; "" otherwise

    grid_params: str     # e.g. "core_mask:dronet_inflight"; "" otherwise


def build_remote_cmd(bin_name: str, dronet_vmfb: str, mlp_vmfb: str, rc: RunConfig) -> str:
    # Always request CSV summary.
    parts = [
        f"./{bin_name}",
        dronet_vmfb,
        mlp_vmfb,
        f"--mlp_hz={rc.mlp_hz}",
        f"--duration_s={rc.duration_s}",
        f"--warmup_s={rc.warmup_s}",
        f"--dronet_fn='{rc.dronet_fn}'",
        f"--mlp_fn='{rc.mlp_fn}'",
        f"--driver='{rc.driver}'",
        f"--report_hz={rc.report_hz}",
        f"--dronet_sensor_hz={rc.dronet_sensor_hz}",
        f"--mlp_sensor_hz={rc.mlp_sensor_hz}",
        f"--dronet_inflight={rc.dronet_inflight}",
        f"--mlp_inflight={rc.mlp_inflight}",
        "--csv=1",
    ]
    if rc.core_mask != 0:
        parts.append(f"--core_mask=0x{rc.core_mask:x}")
    return " ".join(parts)


def ssh_run(
    host: str,
    user: str,
    remote_dir: str,
    remote_cmd: str,
    timeout_s: Optional[float],
    use_sshpass: bool,
    sshpass_password: Optional[str],
    ssh_port: int,
) -> subprocess.CompletedProcess:
    full_remote = f"cd {remote_dir} && {remote_cmd}"
    ssh_base = [
        "ssh",
        "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{host}",
        full_remote,
    ]

    if use_sshpass:
        if not sshpass_password:
            raise RuntimeError("sshpass enabled but no password provided")
        cmd = ["sshpass", "-p", sshpass_password] + ssh_base
    else:
        cmd = ssh_base

    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )


def open_csv_writer(out_csv: Path, fieldnames: List[str]) -> Tuple[csv.DictWriter, object]:
    ensure_parent_dir(out_csv)
    exists = out_csv.exists()
    f = out_csv.open("a", newline="")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
    return w, f


# ---------------------------
# Plan building
# ---------------------------

def make_plan_ofat(
    include_minmax: bool,
    mlp_hz_list: List[float],
    duration_list: List[float],
    warmup_list: List[float],
    report_list: List[float],
    dronet_sensor_list: List[float],
    mlp_sensor_list: List[float],
    dronet_inflight_list: List[int],
    mlp_inflight_list: List[int],
    core_mask_list: List[int],
    dronet_fn: str,
    mlp_fn: str,
    driver: str,
) -> List[RunConfig]:
    # Baseline uses first element of each list.
    base = RunConfig(
        mlp_hz=first(mlp_hz_list, "mlp_hz"),
        duration_s=first(duration_list, "duration_s"),
        warmup_s=first(warmup_list, "warmup_s"),
        report_hz=first(report_list, "report_hz"),
        dronet_sensor_hz=first(dronet_sensor_list, "dronet_sensor_hz"),
        mlp_sensor_hz=(first(mlp_sensor_list, "mlp_sensor_hz")
                       if mlp_sensor_list else first(mlp_hz_list, "mlp_hz")),
        dronet_inflight=first(dronet_inflight_list, "dronet_inflight"),
        mlp_inflight=first(mlp_inflight_list, "mlp_inflight"),
        core_mask=first(core_mask_list, "core_mask"),
        dronet_fn=dronet_fn,
        mlp_fn=mlp_fn,
        driver=driver,
        plan_kind="baseline",
        sweep_param="",
        sweep_value="",
        grid_params="",
    )

    plan: List[RunConfig] = [base]

    def add_sweep(param: str, values: List, setter):
        for v in values:
            rc = setter(base, v)
            if rc == base:
                continue
            plan.append(rc)

    # OFAT sweeps
    add_sweep("mlp_hz", mlp_hz_list,
              lambda b, v: RunConfig(**{
                  **asdict(b),
                  "mlp_hz": float(v),
                  # If no explicit mlp_sensor list, keep it coupled to mlp_hz.
                  "mlp_sensor_hz": (float(v) if not mlp_sensor_list else b.mlp_sensor_hz),
                  "plan_kind": "sweep",
                  "sweep_param": "mlp_hz",
                  "sweep_value": str(v),
              }))

    add_sweep("dronet_sensor_hz", dronet_sensor_list,
              lambda b, v: RunConfig(**{
                  **asdict(b),
                  "dronet_sensor_hz": float(v),
                  "plan_kind": "sweep",
                  "sweep_param": "dronet_sensor_hz",
                  "sweep_value": str(v),
              }))

    add_sweep("dronet_inflight", dronet_inflight_list,
              lambda b, v: RunConfig(**{
                  **asdict(b),
                  "dronet_inflight": int(v),
                  "plan_kind": "sweep",
                  "sweep_param": "dronet_inflight",
                  "sweep_value": str(v),
              }))

    add_sweep("mlp_inflight", mlp_inflight_list,
              lambda b, v: RunConfig(**{
                  **asdict(b),
                  "mlp_inflight": int(v),
                  "plan_kind": "sweep",
                  "sweep_param": "mlp_inflight",
                  "sweep_value": str(v),
              }))

    add_sweep("core_mask", core_mask_list,
              lambda b, v: RunConfig(**{
                  **asdict(b),
                  "core_mask": int(v),
                  "plan_kind": "sweep",
                  "sweep_param": "core_mask",
                  "sweep_value": f"0x{int(v):x}",
              }))

    if mlp_sensor_list:
        add_sweep("mlp_sensor_hz", mlp_sensor_list,
                  lambda b, v: RunConfig(**{
                      **asdict(b),
                      "mlp_sensor_hz": float(v),
                      "plan_kind": "sweep",
                      "sweep_param": "mlp_sensor_hz",
                      "sweep_value": str(v),
                  }))

    # Min/max corners
    if include_minmax:
        min_all = RunConfig(
            mlp_hz=minv(mlp_hz_list, "mlp_hz"),
            duration_s=first(duration_list, "duration_s"),   # keep duration stable by default
            warmup_s=first(warmup_list, "warmup_s"),
            report_hz=first(report_list, "report_hz"),
            dronet_sensor_hz=minv(dronet_sensor_list, "dronet_sensor_hz"),
            mlp_sensor_hz=(minv(mlp_sensor_list, "mlp_sensor_hz")
                           if mlp_sensor_list else minv(mlp_hz_list, "mlp_hz")),
            dronet_inflight=minv(dronet_inflight_list, "dronet_inflight"),
            mlp_inflight=minv(mlp_inflight_list, "mlp_inflight"),
            core_mask=minv(core_mask_list, "core_mask"),
            dronet_fn=dronet_fn,
            mlp_fn=mlp_fn,
            driver=driver,
            plan_kind="min_all",
            sweep_param="",
            sweep_value="",
            grid_params="",
        )

        max_all = RunConfig(
            mlp_hz=maxv(mlp_hz_list, "mlp_hz"),
            duration_s=first(duration_list, "duration_s"),
            warmup_s=first(warmup_list, "warmup_s"),
            report_hz=first(report_list, "report_hz"),
            dronet_sensor_hz=maxv(dronet_sensor_list, "dronet_sensor_hz"),
            mlp_sensor_hz=(maxv(mlp_sensor_list, "mlp_sensor_hz")
                           if mlp_sensor_list else maxv(mlp_hz_list, "mlp_hz")),
            dronet_inflight=maxv(dronet_inflight_list, "dronet_inflight"),
            mlp_inflight=maxv(mlp_inflight_list, "mlp_inflight"),
            core_mask=maxv(core_mask_list, "core_mask"),
            dronet_fn=dronet_fn,
            mlp_fn=mlp_fn,
            driver=driver,
            plan_kind="max_all",
            sweep_param="",
            sweep_value="",
            grid_params="",
        )

        plan.append(min_all)
        plan.append(max_all)

    return plan


def add_pairwise_grids(
    plan: List[RunConfig],
    base: RunConfig,
    pairwise_specs: List[str],
    values_by_param: Dict[str, List],
) -> List[RunConfig]:
    """
    Adds small 2D grids keeping all other params at baseline.
    pairwise_specs: ["core_mask:dronet_inflight", "mlp_hz:mlp_inflight", ...]
    values_by_param keys must include both params.
    """
    out = list(plan)
    for spec in pairwise_specs:
        spec = spec.strip()
        if not spec:
            continue
        if ":" not in spec:
            raise SystemExit(f"Bad --pairwise_grids entry '{spec}'. Expected 'a:b'.")
        a, b = spec.split(":", 1)
        a = a.strip()
        b = b.strip()
        if a not in values_by_param or b not in values_by_param:
            raise SystemExit(f"--pairwise_grids '{spec}' refers to unknown params. "
                             f"Known: {sorted(values_by_param.keys())}")

        a_vals = values_by_param[a]
        b_vals = values_by_param[b]
        for av in a_vals:
            for bv in b_vals:
                d = asdict(base)
                # Set fields by name:
                d[a] = av
                d[b] = bv

                # Keep mlp_sensor_hz coupled if user didn't explicitly sweep it:
                if a == "mlp_hz" and "mlp_sensor_hz" not in values_by_param:
                    d["mlp_sensor_hz"] = float(av)

                rc = RunConfig(**{
                    **d,
                    "plan_kind": "grid2d",
                    "sweep_param": "",
                    "sweep_value": "",
                    "grid_params": f"{a}:{b}",
                })
                out.append(rc)
    return out


# ---------------------------
# Main
# ---------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Remote OFAT sweep runner over SSH (host-side).")

    ap.add_argument("--ssh_host", required=True)
    ap.add_argument("--ssh_user", default="root")
    ap.add_argument("--ssh_port", type=int, default=22)
    ap.add_argument("--remote_dir", default="/home/baseline")
    ap.add_argument("--bin_name", required=True)
    ap.add_argument("--dronet_vmfb", default="dronet.q.int8.vmfb")
    ap.add_argument("--mlp_vmfb", default="mlp.q.int8.vmfb")

    ap.add_argument("--use_sshpass", action="store_true")
    ap.add_argument("--sshpass_password", default="")

    ap.add_argument("--out_csv", default="results/remote_sweep.csv")
    ap.add_argument("--logs_dir", default="results/remote_logs")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout_s", type=float, default=None)
    ap.add_argument("--include_minmax", type=int, default=1)

    ap.add_argument("--pairwise_grids", default="",
                    help="Optional 2D interaction grids, e.g. 'core_mask:dronet_inflight,mlp_hz:mlp_inflight'")

    ap.add_argument("--dronet_fn", default="dronet.main_graph$async")
    ap.add_argument("--mlp_fn", default="mlp.main_graph$async")
    ap.add_argument("--driver", default="local-task")

    ap.add_argument("--mlp_hz", default="20")
    ap.add_argument("--duration_s", default="10")
    ap.add_argument("--warmup_s", default="2")
    ap.add_argument("--report_hz", default="0")
    ap.add_argument("--dronet_sensor_hz", default="60")
    ap.add_argument("--mlp_sensor_hz", default="",
                    help="If empty, defaults to mlp_hz per run")
    ap.add_argument("--dronet_inflight", default="8")
    ap.add_argument("--mlp_inflight", default="2")
    ap.add_argument("--core_mask", default="0")

    ap.add_argument("--auto_core_masks_1to8", action="store_true",
                    help="If set, ignore --core_mask and use 1..8 core masks: 0x1,0x3,...,0xff")

    args = ap.parse_args()

    out_csv = Path(args.out_csv)
    logs_dir = Path(args.logs_dir)
    ensure_parent_dir(out_csv)
    ensure_parent_dir(logs_dir / "x")

    mlp_hz_list = parse_float_list(args.mlp_hz)
    duration_list = parse_float_list(args.duration_s)
    warmup_list = parse_float_list(args.warmup_s)
    report_list = parse_float_list(args.report_hz)
    dronet_sensor_list = parse_float_list(args.dronet_sensor_hz)
    mlp_sensor_list = parse_float_list(args.mlp_sensor_hz) if args.mlp_sensor_hz else []
    dronet_inflight_list = parse_int_list(args.dronet_inflight)
    mlp_inflight_list = parse_int_list(args.mlp_inflight)

    if args.auto_core_masks_1to8:
        core_mask_list = [(1 << n) - 1 for n in range(1, 9)]  # 1..8 cores
    else:
        core_mask_list = parse_int_list(args.core_mask)
        if not core_mask_list:
            core_mask_list = [0]

    # Plan: baseline + OFAT sweeps + (optional) min/max corners.
    plan = make_plan_ofat(
        include_minmax=(args.include_minmax != 0),
        mlp_hz_list=mlp_hz_list,
        duration_list=duration_list,
        warmup_list=warmup_list,
        report_list=report_list,
        dronet_sensor_list=dronet_sensor_list,
        mlp_sensor_list=mlp_sensor_list,
        dronet_inflight_list=dronet_inflight_list,
        mlp_inflight_list=mlp_inflight_list,
        core_mask_list=core_mask_list,
        dronet_fn=args.dronet_fn,
        mlp_fn=args.mlp_fn,
        driver=args.driver,
    )

    # Optional: add pairwise grids (small interaction studies).
    pair_specs = parse_csv_list(args.pairwise_grids)
    if pair_specs:
        base = plan[0]
        values_by_param: Dict[str, List] = {
            "mlp_hz": mlp_hz_list,
            "dronet_sensor_hz": dronet_sensor_list,
            "mlp_sensor_hz": (mlp_sensor_list if mlp_sensor_list else [base.mlp_sensor_hz]),
            "dronet_inflight": dronet_inflight_list,
            "mlp_inflight": mlp_inflight_list,
            "core_mask": core_mask_list,
        }
        plan = add_pairwise_grids(plan, base, pair_specs, values_by_param)

    print(f"[remote] Plan configs={len(plan)} repeats={args.repeats}", flush=True)

    def do_run(rc: RunConfig, run_id: int, rep_idx: int) -> Dict[str, str]:
        remote_cmd = build_remote_cmd(args.bin_name, args.dronet_vmfb, args.mlp_vmfb, rc)

        start_iso = datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        cp = ssh_run(
            host=args.ssh_host,
            user=args.ssh_user,
            remote_dir=args.remote_dir,
            remote_cmd=remote_cmd,
            timeout_s=args.timeout_s,
            use_sshpass=args.use_sshpass,
            sshpass_password=args.sshpass_password,
            ssh_port=args.ssh_port,
        )
        t1 = time.time()
        end_iso = datetime.now().isoformat(timespec="seconds")

        out = cp.stdout or ""
        header, row = extract_csv(out)

        # Make an informative log filename.
        cores = popcount_u64(rc.core_mask)
        tag = f"run{run_id:05d}_rep{rep_idx:02d}_{rc.plan_kind}"
        if rc.sweep_param:
            tag += f"_sweep_{rc.sweep_param}={sanitize(rc.sweep_value)}"
        if rc.grid_params:
            tag += f"_grid_{sanitize(rc.grid_params)}"
        tag += f"_mlp{rc.mlp_hz}_ds{rc.dronet_sensor_hz}_di{rc.dronet_inflight}_mi{rc.mlp_inflight}_cores{cores}_mask{rc.core_mask:x}"

        log_file = logs_dir / (sanitize(tag) + ".log")
        log_file.write_text(out)

        res: Dict[str, str] = {
            "run_id": str(run_id),
            "rep_idx": str(rep_idx),
            "start_time": start_iso,
            "end_time": end_iso,
            "elapsed_s": f"{(t1 - t0):.3f}",
            "exit_code": str(cp.returncode),

            "plan_kind": rc.plan_kind,
            "sweep_param": rc.sweep_param,
            "sweep_value": rc.sweep_value,
            "grid_params": rc.grid_params,

            "ssh_host": args.ssh_host,
            "remote_dir": args.remote_dir,
            "bin_name": args.bin_name,
            "dronet_vmfb": args.dronet_vmfb,
            "mlp_vmfb": args.mlp_vmfb,

            "dronet_fn": rc.dronet_fn,
            "mlp_fn": rc.mlp_fn,
            "driver": rc.driver,

            "mlp_hz": str(rc.mlp_hz),
            "duration_s": str(rc.duration_s),
            "warmup_s": str(rc.warmup_s),
            "report_hz": str(rc.report_hz),
            "dronet_sensor_hz": str(rc.dronet_sensor_hz),
            "mlp_sensor_hz": str(rc.mlp_sensor_hz),
            "dronet_inflight": str(rc.dronet_inflight),
            "mlp_inflight": str(rc.mlp_inflight),
            "core_mask": f"0x{rc.core_mask:x}",
            "core_count": str(cores),

            "log_file": str(log_file),
        }

        if header is None or row is None or len(header) != len(row):
            res["csv_parse_ok"] = "0"
            res["csv_error"] = "missing CSV_HEADER/CSV_ROW or mismatched lengths"
            return res

        res["csv_parse_ok"] = "1"
        res["csv_error"] = ""
        for k, v in zip(header, row):
            res[k] = v
        return res

    # Probe to learn CSV schema.
    probe = do_run(plan[0], run_id=0, rep_idx=0)

    base_fields = [
        "run_id", "rep_idx",
        "start_time", "end_time", "elapsed_s", "exit_code",
        "csv_parse_ok", "csv_error",
        "plan_kind", "sweep_param", "sweep_value", "grid_params",
        "ssh_host", "remote_dir", "bin_name", "dronet_vmfb", "mlp_vmfb",
        "dronet_fn", "mlp_fn", "driver",
        "mlp_hz", "duration_s", "warmup_s", "report_hz",
        "dronet_sensor_hz", "mlp_sensor_hz",
        "dronet_inflight", "mlp_inflight",
        "core_mask", "core_count",
        "log_file",
    ]

    csv_fields: List[str] = []
    if probe.get("csv_parse_ok") == "1":
        log_text = Path(probe["log_file"]).read_text()
        header, _ = extract_csv(log_text)
        if header:
            csv_fields = header
    else:
        print("[remote] WARNING: probe did not yield CSV_HEADER/CSV_ROW; only metadata will be stored.", flush=True)

    fieldnames = base_fields + csv_fields
    writer, f = open_csv_writer(out_csv, fieldnames)

    writer.writerow({k: probe.get(k, "") for k in fieldnames})
    f.flush()

    run_id = 1
    for rc in plan:
        for rep in range(args.repeats):
            # Skip re-running baseline rep0 since probe already did it.
            if rc == plan[0] and rep == 0:
                continue

            label = rc.plan_kind
            if rc.sweep_param:
                label += f"/{rc.sweep_param}={rc.sweep_value}"
            if rc.grid_params:
                label += f"/grid={rc.grid_params}"
            print(f"[remote] Run {run_id} rep={rep} {label} "
                  f"mlp_hz={rc.mlp_hz} ds={rc.dronet_sensor_hz} di={rc.dronet_inflight} "
                  f"mi={rc.mlp_inflight} mask={rc.core_mask:#x}",
                  flush=True)

            try:
                res = do_run(rc, run_id, rep)
            except subprocess.TimeoutExpired:
                res = {
                    "run_id": str(run_id),
                    "rep_idx": str(rep),
                    "start_time": datetime.now().isoformat(timespec="seconds"),
                    "end_time": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_s": "",
                    "exit_code": "TIMEOUT",
                    "csv_parse_ok": "0",
                    "csv_error": "timeout",
                    "plan_kind": rc.plan_kind,
                    "sweep_param": rc.sweep_param,
                    "sweep_value": rc.sweep_value,
                    "grid_params": rc.grid_params,
                    "ssh_host": args.ssh_host,
                    "remote_dir": args.remote_dir,
                    "bin_name": args.bin_name,
                    "dronet_vmfb": args.dronet_vmfb,
                    "mlp_vmfb": args.mlp_vmfb,
                    "dronet_fn": rc.dronet_fn,
                    "mlp_fn": rc.mlp_fn,
                    "driver": rc.driver,
                    "mlp_hz": str(rc.mlp_hz),
                    "duration_s": str(rc.duration_s),
                    "warmup_s": str(rc.warmup_s),
                    "report_hz": str(rc.report_hz),
                    "dronet_sensor_hz": str(rc.dronet_sensor_hz),
                    "mlp_sensor_hz": str(rc.mlp_sensor_hz),
                    "dronet_inflight": str(rc.dronet_inflight),
                    "mlp_inflight": str(rc.mlp_inflight),
                    "core_mask": f"0x{rc.core_mask:x}",
                    "core_count": str(popcount_u64(rc.core_mask)),
                    "log_file": "",
                }

            writer.writerow({k: res.get(k, "") for k in fieldnames})
            f.flush()
            run_id += 1

    f.close()
    print(f"[remote] Done. Wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())