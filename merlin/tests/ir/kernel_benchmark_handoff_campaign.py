"""Run full-size handoff cases on at most eight leased single-CPU lanes.

The caller authorizes the lane set. Every case uses an isolated owned pytest
process, one Lean worker/thread, and its own retained directory. This runner
never turns compilation or numerical equality into formal proof credit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Event
from time import perf_counter
from xml.etree import ElementTree as ET

from merlin.common.owned_process import CleanupUnknown
from merlin.common.owned_process import run as run_tree
from merlin.common.paths import repo_root

CASES = tuple(
    [f"G{i:02d}" for i in range(1, 7)] + [f"C{i:02d}" for i in range(1, 4)] + [f"T{i:02d}" for i in range(1, 8)]
)
TEST = "merlin/tests/ir/test_kernel_benchmark_formal_gates.py"
METHOD = "test_kernel_benchmark_handoff_formal_gate_and_native"
MAX_JOBS = 8


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def write(path, value):
    """Each aggregate has one main-thread writer and atomic replacement."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_cpus(value):
    if ".." in value or "-" in value:
        separator = ".." if ".." in value else "-"
        pieces = value.split(separator)
        if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
            raise argparse.ArgumentTypeError("CPU range must be start..end")
        lower, upper = map(int, pieces)
        if upper < lower or upper - lower + 1 > (os.cpu_count() or 1):
            raise argparse.ArgumentTypeError("bounded increasing CPU range required")
        result = tuple(range(lower, upper + 1))
    else:
        pieces = value.split(",")
        if not all(piece.isdigit() for piece in pieces):
            raise argparse.ArgumentTypeError("CPU IDs must be comma-separated integers")
        result = tuple(map(int, pieces))
    if not result or len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("distinct CPU lanes required")
    return result


def snapshot(paths):
    try:
        return [ref(path) for path in paths], None
    except OSError as error:
        return None, str(error)


def run_case(case, cpu, *, output, root, environment, timeout, source_files, initial, owned_run=run_tree):
    """No shared aggregate writes: this owns only the case directory/log."""
    directory = output / case
    directory.mkdir()
    row = {"case": case, "cpu": cpu, "case_directory": str(directory), "source_cut": initial}
    before, error = snapshot(source_files)
    row["pre_source_cut_matches"] = before == initial
    if before != initial:
        row.update(status="FAILED_SOURCE_CUT_DRIFT_NO_CAMPAIGN_CREDIT", reason=error or "source changed before case")
    xml = directory / "pytest.xml"
    command = [
        "nice",
        "-n",
        "19",
        "taskset",
        "-c",
        str(cpu),
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-o",
        "addopts=",
        "-o",
        "tmp_path_retention_policy=all",
        "-o",
        "timeout=0",
        "--basetemp=" + str(directory / "pytest-tmp"),
        "--junitxml=" + str(xml),
        TEST + "::" + METHOD + "[" + case + "]",
    ]
    write(
        directory / "COMMAND.json",
        {
            "command": command,
            "cwd": str(root),
            "cpu": cpu,
            "timeout": timeout,
            "nice": 19,
            "threads": 1,
            "lean_workers": 1,
        },
    )
    started = perf_counter()
    if before == initial:
        print("START", case, "CPU", cpu, flush=True)
        try:
            process = owned_run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=timeout)
            row["returncode"] = process.returncode
            cleanup = getattr(process, "owned_process_cleanup", {})
            row["owned_process_cleanup"] = cleanup
            (directory / "pytest.log").write_text(process.stdout + process.stderr)
            if not cleanup.get("complete") or not cleanup.get("guardian_reaped"):
                row.update(status="FAILED_CLEANUP_UNKNOWN_NO_CAMPAIGN_CREDIT", unsafe_to_release_lane=True)
        except subprocess.TimeoutExpired as error:
            cleanup = getattr(error, "owned_process_cleanup", {})
            row.update(
                status="FAILED_RESOURCE_TIMEOUT_NO_PROOF_CREDIT", reason=str(error), owned_process_cleanup=cleanup
            )
            if not cleanup.get("complete") or not cleanup.get("guardian_reaped"):
                row["unsafe_to_release_lane"] = True
            chunks = [getattr(error, "output", "") or "", getattr(error, "stderr", "") or ""]
            (directory / "pytest.log").write_text(
                "".join(value.decode(errors="replace") if isinstance(value, bytes) else value for value in chunks)
            )
        except CleanupUnknown as error:
            row.update(
                status="FAILED_CLEANUP_UNKNOWN_NO_CAMPAIGN_CREDIT", reason=str(error), unsafe_to_release_lane=True
            )
            (directory / "pytest.log").write_text(str(error))
        except OSError as error:
            row.update(status="FAILED_EXECUTION_UNAVAILABLE_NO_PROOF_CREDIT", reason=str(error))
            (directory / "pytest.log").write_text(str(error))
    row["elapsed_seconds"] = perf_counter() - started
    after, error = snapshot(source_files)
    row["post_source_cut_matches"] = after == initial
    if after != initial:
        row.update(
            status="FAILED_SOURCE_CUT_DRIFT_NO_CAMPAIGN_CREDIT",
            post_source_cut=after,
            reason=error or "source changed during case",
        )
    passed = False
    try:
        reports = list((directory / "pytest-tmp").rglob("CASE-REPORT.json"))
        if len(reports) == 1:
            report = json.loads(reports[0].read_bytes())
            row.update(report=ref(reports[0]), case_status=report["status"], phase=report["phase"])
            if "gate_coverage" in report:
                coverage = report["gate_coverage"]
                row.update(
                    selector_status=coverage["selectors"],
                    verified_boundaries=coverage["record"]["verified_boundaries"],
                    unverified_boundaries=coverage["record"]["unverified_boundaries"],
                )
        if xml.is_file():
            suite = ET.parse(xml).getroot().find("testsuite")
            row["pytest"] = dict(suite.attrib)
            passed = suite.attrib.get("tests") == "1" and all(
                suite.attrib.get(k) == "0" for k in ("failures", "errors", "skipped")
            )
    except (OSError, ValueError, KeyError, AttributeError, ET.ParseError) as error:
        row.update(status="FAILED_INCOMPLETE_CASE_EVIDENCE", reason=str(error))
    if "status" not in row:
        row["status"] = (
            "PASS" if row.get("returncode") == 0 and passed and row.get("case_status") == "PASS" else "FAILED"
        )
    return row


def dispatch(cases, cpus, jobs, execute, on_result):
    """Lease one distinct CPU per task; deliver results only on caller thread."""
    if type(jobs) is not int or not 1 <= jobs <= MAX_JOBS or jobs > len(cpus) or len(set(cpus)) != len(cpus):
        raise ValueError("jobs1..8 must fit distinct leased CPU lanes")
    lanes = Queue()
    cleanup_unknown = Event()
    for cpu in cpus:
        lanes.put(cpu)

    def leased(case):
        if cleanup_unknown.is_set():
            return {"case": case, "cpu": None, "status": "FAILED_NOT_LAUNCHED_AFTER_CLEANUP_UNKNOWN"}
        cpu = lanes.get()
        release = True
        try:
            row = execute(case, cpu)
            if row.get("unsafe_to_release_lane"):
                release = False
                cleanup_unknown.set()
            return row
        except Exception as error:
            return {"case": case, "cpu": cpu, "status": "FAILED_CASE_ORCHESTRATION", "reason": str(error)}
        finally:
            if release:
                lanes.put(cpu)

    records = []
    with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="kernel-lane") as pool:
        futures = {pool.submit(leased, case): case for case in cases}
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            on_result(row, tuple(records))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cpus", type=parse_cpus)
    parser.add_argument("--cpu", type=int, help="single-lane compatibility spelling")
    parser.add_argument("--case-timeout", type=float, default=1200)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    if args.cpu is not None and args.cpus is not None:
        raise ValueError("choose --cpu or --cpus")
    cpus = args.cpus if args.cpus is not None else (28 if args.cpu is None else args.cpu,)
    if not 1 <= args.jobs <= MAX_JOBS or args.jobs > len(cpus):
        raise ValueError("jobs1..8 must fit selected CPU lanes")
    if not set(cpus) <= os.sched_getaffinity(0) or os.getpriority(os.PRIO_PROCESS, 0) < 19:
        raise ValueError("requested CPU lanes must be available inside a nice19 parent affinity")
    if len(args.cases) != len(set(args.cases)) or args.case_timeout <= 0:
        raise ValueError("distinct cases and positive per-case timeout required")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = repo_root()
    source_files = [
        root / TEST,
        root / "merlin/tests/ir/kernel_benchmark_handoff_inputs.py",
        root / "merlin/tests/ir/kernel_benchmark_handoff_oracle.py",
        root / "merlin/tests/data/kernel_benchmark_handoff_contracts.json",
        Path(__file__).resolve(),
    ]
    initial = [ref(path) for path in source_files]
    environment = dict(
        os.environ,
        MERLIN_LEAN_REQUIRED="1",
        MERLIN_RUN_KERNEL_HANDOFF="1",
        MERLIN_KERNEL_HANDOFF_REQUIRED="1",
        LEAN_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    write(
        output / "CAMPAIGN-SOURCE-CUT.json",
        {
            "files": initial,
            "cpus": list(cpus),
            "jobs": args.jobs,
            "threads_per_case": 1,
            "lean_workers_per_case": 1,
            "cases": args.cases,
            "case_timeout": args.case_timeout,
            "kernel_sizes": "original complete handoff dimensions",
            "proof_claim": "only actual bounded pass coverage; native/reference comparison is finite evidence",
        },
    )

    def execute(case, cpu):
        return run_case(
            case,
            cpu,
            output=output,
            root=root,
            environment=environment,
            timeout=args.case_timeout,
            source_files=source_files,
            initial=initial,
        )

    def completed(row, records):
        (output / row["case"]).mkdir(exist_ok=True)
        write(output / row["case"] / "PROCESS-RESULT.json", row)
        write(
            output / "CAMPAIGN-RESULT.json",
            {
                "schema": "merlin.kernel-handoff-formal-campaign.v1",
                "status": "RUNNING" if len(records) < len(args.cases) else "COMPLETE",
                "jobs": args.jobs,
                "cpus": list(cpus),
                "cases_requested": list(args.cases),
                "cases_attempted": len(records),
                "passed": sum(r["status"] == "PASS" for r in records),
                "failed": sum(r["status"] != "PASS" for r in records),
                "cases": sorted(records, key=lambda r: args.cases.index(r["case"])),
                "all_ten_transformations_automatically_enabled": False,
                "whole_kernel_or_LLVM_proof": False,
            },
        )
        print("END", row["case"], row["status"], "CPU", row["cpu"], round(row.get("elapsed_seconds", 0), 2), flush=True)

    records = dispatch(args.cases, cpus, args.jobs, execute, completed)
    return 0 if all(row["status"] == "PASS" for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
