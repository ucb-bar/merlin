#!/usr/bin/env python3
"""Plane A capsule cycles on FireSim: the SAME capsule ELFs the Verilator oracle certified, on the FPGA.

Each cell is an L3 run directory written by ``capsule_h2h.py`` (``<arm>__<capsule>__<simulator>[__tag]``).
Its ``generated/package_kernel.elf`` is exactly the binary the RTL oracle passed, and its
``artifacts/console.log`` holds that run's ``METRIC cycles`` and ``OUT Y0``. This script queues those
bytes with the existing submit tool (``run_firesim.sh`` from ``--tools``). That tool builds the per-job
chipyard view and calls ``firesim-queue runworkload-full``, which owns kill -> infrasetup -> runworkload
-> kill under the FPGA lock. The script then records the FPGA's ``METRIC cycles`` and checks its
``OUT Y0`` byte for byte against the capsule's certified Verilator output.

Submissions are sequential and happen only while the queue is empty and no FireSim simulation is
running: a busy FPGA is waited on, never preempted. Identical ELFs (same sha256) are measured once and
attributed to every cell that uses them. Nothing here invokes the ``firesim`` CLI.

Usage (merlin venv):
    python merlin/experiments/voyager_h2h/scripts/firesim_h2h.py --target <target> --tools <dir> \
        --cell voyager_v1__C0_mlp_linear1__verilator__nofence --cell merlin_la1__C0_mlp_linear1__verilator__order
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from merlin.common import provenance
from merlin.common.artifacts import new_product
from merlin.common.paths import runs_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_cycles(text: str) -> int | None:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "METRIC" and parts[1] == "cycles" and parts[2].isdigit():
            return int(parts[2])
    return None


def out_values(text: str, name: str = "Y0") -> str | None:
    """The ``OUT <name> <rows> <cols> <values...>`` line, whitespace-normalized."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] == "OUT" and parts[1] == name:
            return " ".join(parts)
    return None


def _field(log: str, key: str) -> str | None:
    """First ``key=value`` token in the submit log (``job_id=656``, ``workload=...``)."""
    for line in log.splitlines():
        for token in line.split():
            if token.startswith(key + "="):
                return token.split("=", 1)[1]
    return None


def fpga_idle(queue: Path) -> tuple[bool, str]:
    status = subprocess.run([str(queue), "status"], capture_output=True, text=True, check=False)
    text = status.stdout + status.stderr
    procs = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True, check=False).stdout
    simulating = any("FireSim-xilinx" in line for line in procs.splitlines())
    return ("daemon: ALIVE" in text and "no jobs match" in text and not simulating), text.strip()


def _find_uart(queue_root: Path, workload: str, job_id: str) -> Path | None:
    results = queue_root / "jobs" / job_id / "deploy_overlay" / "results-workload"
    for directory in sorted(results.glob(f"*-{workload}-q{job_id}")):
        uart = directory / f"{workload}0" / "uartlog"
        if uart.is_file():
            return uart
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", required=True)
    parser.add_argument(
        "--tools", required=True, type=Path, help="directory holding the existing run_firesim.sh submit tool"
    )
    parser.add_argument(
        "--queue", type=Path, default=None, help="firesim-queue client (default: the one run_firesim.sh uses)"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="capsule_h2h run directories (default: <runs>/<target>/voyager-h2h/runs/<target>-contract)",
    )
    parser.add_argument("--cell", action="append", required=True, help="run directory name")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="seconds the daemon may hold the FPGA per job (MERLIN_FIRESIM_TIMEOUT)",
    )
    parser.add_argument(
        "--max-wait", type=int, default=6 * 3600, help="seconds to wait for an idle FPGA before giving up"
    )
    parser.add_argument("--project", default="voyager-h2h")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    tools = args.tools.resolve()
    submit_tool = tools / "run_firesim.sh"
    if not submit_tool.is_file():
        raise SystemExit(f"no run_firesim.sh under {tools}")
    queue = (
        args.queue or Path(os.environ.get("MERLIN_FIRESIM_QUEUE", "/scratch/firesim_queue/bin/firesim-queue"))
    ).resolve()
    queue_root = queue.parent.parent
    runs_root = args.runs_root or runs_dir() / args.target / "voyager-h2h" / "runs" / f"{args.target}-contract"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    work = runs_dir() / args.target / "voyager-h2h" / "firesim" / stamp

    # Cells, and the certified output each capsule must reproduce (every passing arm of a capsule
    # produced the same OUT under the RTL oracle; disagreement between them is itself an error).
    cells, expected = [], {}
    for name in args.cell:
        run = runs_root / name
        parts = name.split("__")
        if len(parts) < 3:
            raise SystemExit(f"{name}: not an <arm>__<capsule>__<simulator>[__tag] run directory")
        elf = run / "generated" / "package_kernel.elf"
        if not elf.is_file():
            raise SystemExit(f"{name}: no ELF at {elf}")
        console = run / "artifacts" / "console.log"
        text = console.read_text(errors="replace") if console.is_file() else ""
        cell = {
            "cell": name,
            "arm": parts[0],
            "capsule": parts[1],
            "elf": str(elf),
            "elf_sha256": _sha256(elf),
            "verilator_cycles": metric_cycles(text),
            "verilator_out": out_values(text),
        }
        if cell["verilator_out"]:
            seen = expected.setdefault(cell["capsule"], cell["verilator_out"])
            if seen != cell["verilator_out"]:
                raise SystemExit(f"{name}: certified outputs disagree across arms for {cell['capsule']}")
        cells.append(cell)

    measured: dict[str, dict] = {}  # elf sha256 -> FPGA result
    for cell in cells:
        sha = cell["elf_sha256"]
        if sha in measured:
            continue
        bundle = work / cell["cell"]
        bundle.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cell["elf"], bundle / "package_kernel.elf")
        prefix = "vh2h-" + "-".join(p.replace("_", "-").lower() for p in (cell["arm"], cell["capsule"].split("_")[0]))
        if args.dry_run:
            print(f"[dry-run] {cell['cell']}: {submit_tool} {bundle} {prefix} {args.project}")
            measured[sha] = {"status": "dry-run"}
            continue
        waited = 0
        while True:
            idle, status = fpga_idle(queue)
            if idle:
                break
            if waited >= args.max_wait:
                raise SystemExit(f"FPGA not idle after {waited} s; last status:\n{status}")
            print(f"FPGA busy, waiting ({waited} s): {status.splitlines()[-1] if status else ''}", flush=True)
            time.sleep(60)
            waited += 60
        started = time.time()
        env = dict(os.environ, MERLIN_FIRESIM_TIMEOUT=str(args.timeout))
        proc = subprocess.run(
            [str(submit_tool), str(bundle), prefix, f"{args.project}-{prefix}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
            timeout=args.timeout + 3600,
        )
        log = proc.stdout
        (bundle / "submit.log").write_text(log)
        job_id, workload, view = _field(log, "job_id"), _field(log, "workload"), _field(log, "view")
        result = {
            "status": "fail",
            "job_id": job_id,
            "workload": workload,
            "view": view,
            "submit_rc": proc.returncode,
            "wall_s": round(time.time() - started, 1),
            "submit_log": str(bundle / "submit.log"),
        }
        if view and Path(view, "queue_identity.json").is_file():
            identity = json.loads(Path(view, "queue_identity.json").read_text())
            result["identity"] = {
                k: identity.get(k)
                for k in (
                    "deploy_quintuplet",
                    "bitstream",
                    "bitstream_sha256",
                    "driver_sha256",
                    "bootbinary_sha256",
                    "workload_template",
                )
            }
        uart = _find_uart(queue_root, workload, job_id) if (job_id and workload) else None
        if proc.returncode == 0 and "terminal state=DONE" in log and uart is not None:
            text = uart.read_text(errors="replace")
            (bundle / "uartlog").write_text(text)
            result.update(
                uart=str(bundle / "uartlog"),
                uart_sha256=_sha256(bundle / "uartlog"),
                cycles=metric_cycles(text),
                out=out_values(text),
            )
            want = expected.get(cell["capsule"])
            result["out_matches_certified"] = want is not None and result["out"] == want
            if result["cycles"] is not None and result["out_matches_certified"]:
                result["status"] = "pass"
        measured[sha] = result
        print(
            f"{cell['cell']}: {result['status']} job={job_id} cycles={result.get('cycles')} "
            f"out_ok={result.get('out_matches_certified')} ({result['wall_s']} s)",
            flush=True,
        )

    rows = [{**cell, "firesim": measured.get(cell["elf_sha256"])} for cell in cells]
    if args.dry_run:
        print(json.dumps({"cells": len(rows), "unique_elfs": len(measured), "work_dir": str(work)}))
        return 0
    pins = {}
    try:
        pins["gemmini_rtl"] = provenance.verify("gemmini_rtl")
    except Exception as exc:  # noqa: BLE001 -- recorded, not fatal: drift is the finding
        print(f"pin gemmini_rtl: {exc}")
    tool_files = {
        p.name: p for p in (submit_tool, tools / "prepare_queue_view.sh", tools / "config_hwdb.yaml") if p.is_file()
    }
    record = provenance.record(pins=pins, artifacts=tool_files)
    product = new_product("compare", version=1, target=args.target, notes="voyager_h2h capsule cycles on FireSim")
    doc = {
        "target": args.target,
        "engine": "firesim",
        "cells": rows,
        "work_dir": str(work),
        "timeout_s": args.timeout,
        "provenance": record,
    }
    (product.path / "results.json").write_text(json.dumps(doc, indent=1, default=str))
    lines = [
        "# Capsule cycles on FireSim (same ELFs as the Verilator L3 runs)",
        "",
        "| cell | FireSim cycles | Verilator cycles | OUT matches certified | job |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        fs = row["firesim"] or {}
        lines.append(
            f"| {row['cell']} | {fs.get('cycles', '-')} | {row['verilator_cycles'] or '-'} | "
            f"{fs.get('out_matches_certified', '-')} | {fs.get('job_id', '-')} |"
        )
    (product.path / "table.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "product": str(product.path),
                "passes": sum(1 for r in rows if (r["firesim"] or {}).get("status") == "pass"),
                "cells": len(rows),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
