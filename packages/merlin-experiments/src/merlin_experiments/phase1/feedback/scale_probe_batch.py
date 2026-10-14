"""Original-issued public scale workers with one restored owner per sealed batch.

Only process-local read leases persist. Every case and actual command retains
its original checks; accepted results wait for the worker's final owner exit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from merlin.common import cpu_pool
from merlin_experiments.phase2.contracts import StageGateError

from . import compiler_profile_child as CHILD
from . import compiler_profile_runtime as R
from . import scale_probe_child as S

SCHEMA = "merlin.original_public_scale_batch.v1"


def _jobs(document, spec):
    """Validate the exact per-case producer roster before using any worker slot."""
    fields = {"schema", "package_dir", "target", "jobs", "deadline", "output"}
    if (set(document) != fields or document["schema"] != SCHEMA
            or CHILD._plain(document["output"]) != spec.parent
            or type(document["deadline"]) not in (int, float) or not math.isfinite(document["deadline"])
            or type(document["jobs"]) is not list or not 1 <= len(document["jobs"]) <= 8192):
        raise StageGateError("public scale worker requires its closed original batch")
    jobs, indices, paths, outputs = [], set(), set(), set()
    for item in document["jobs"]:
        if (type(item) is not dict or set(item) != {"index", "spec", "case"}
                or type(item["index"]) is not int or item["index"] < 0):
            raise StageGateError("public scale batch has an invalid original case index")
        path = CHILD._plain(item["spec"]["path"])
        if CHILD._pin(path) != item["spec"] or path.stat().st_size > 1 << 20:
            raise StageGateError("public scale batch case specification changed")
        job = json.loads(path.read_bytes())
        output = CHILD._plain(job["output"])
        if (item["index"] in indices or path in paths or output in outputs or output != path.parent
                or output.parent != spec.parent.parent
                or job["package_dir"] != document["package_dir"] or job["target"] != document["target"]
                or job["deadline"] != document["deadline"] or job["case"] != item["case"]):
            raise StageGateError("public scale batch substituted or repeated an original case owner")
        indices.add(item["index"])
        paths.add(path)
        outputs.add(output)
        jobs.append((item, path, job))
    return jobs


def _verify_payload(root, payload, budget):
    if payload.get("lowering_status") in {"emitted", "resource_refused"}:
        analysis = S._verify_products(root, payload)
        expected = budget.evaluate(analysis) if budget is not None else {"mode": "not_selected"}
        if (CHILD._digest(payload.get("resource_budget")) != CHILD._digest(expected)
                or payload["lowering_status"] != (
                    "resource_refused" if expected.get("status") == "refused" else "emitted")):
            raise StageGateError("public scale batch changed its original resource policy")
    elif set(payload) != {"lowering_status", "reason"} or payload["lowering_status"] != "unavailable":
        raise StageGateError("public scale batch supplied an unrecognized case observation")


def emit_cases(*, owner, candidate, target_experiment, capsules, cases, contract, facts_path,
               work_root, deadline, ir_budget, sandbox_inputs=None, timing=None):
    from merlin.targetgen.capsule_runner import _kill_tree

    from . import compile_only as C

    entered, entered_cpu = time.monotonic(), time.process_time()
    count = min(S.batch_workers(), len(cases)) if cases else 0
    rows, case_jobs = [C._case_row(case) for case in cases], []
    groups, active, completed = [], [], []
    with contextlib.ExitStack() as lifetime:
        with C._snapshot_operation(owner):
            lifetime.enter_context(R._original_execution_lifetime(owner,
                target_experiment=target_experiment, candidate=candidate,
                work_root=work_root / "child-owner", sandbox_inputs=sandbox_inputs))
            common = {"contract": S._input_pin(contract), "facts": CHILD._pin(facts_path)}
            for index, case in enumerate(cases):
                case.verify()
                root = work_root / f"case-{index:03d}"
                root.mkdir(mode=0o700)
                source = work_root / "sources" / f"case-{index:03d}.mlir"
                source.write_text(case.source_ir)
                spec = root / ".scale_child_spec.json"
                selected = [cap for cap in capsules
                    if str(cap.get("name") or Path(cap["__dir__"]).name) == case.member]
                S._write(spec, {"schema": S.SCHEMA, "package_dir": str(candidate),
                    "target": target_experiment.target, "case": case.record(), "capsules": CHILD._data(selected),
                    "source": CHILD._pin(source), **common, "output": str(root), "deadline": deadline,
                    "ir_budget": ir_budget.record() if ir_budget is not None else None})
                case_jobs.append({"index": index, "spec": CHILD._pin(spec), "case": case.record()})
            for worker in range(count):
                root = work_root / f"worker-{worker:03d}"
                root.mkdir(mode=0o700)
                spec, result = root / ".scale_batch_spec.json", root / ".scale_batch_result.json"
                S._write(spec, {"schema": SCHEMA, "package_dir": str(candidate),
                    "target": target_experiment.target, "jobs": case_jobs[worker::count],
                    "deadline": deadline, "output": str(root)})
                groups.append((spec, result))
            environment = dict(os.environ)
            transports = lifetime.enter_context(CHILD.prepare_model_children(
                spec_paths=tuple(spec for spec, _ in groups), environment=environment,
                package_dir=candidate)) if groups else ()
        prepared_at, prepared_cpu = time.monotonic(), time.process_time()
        try:
            for (spec, result), transport in zip(groups, transports, strict=True):
                if time.monotonic() >= deadline:
                    break
                log = lifetime.enter_context((spec.parent / ".scale_batch.log").open("w"))
                command = [sys.executable, "-m", __name__, "--spec", str(spec),
                           "--result", str(result), *transport.argv]
                process = subprocess.Popen(cpu_pool.worker_pool_command(command, environment), env=environment,
                    pass_fds=transport.pass_fds, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((spec, result, transport, process, time.monotonic()))
            while active:
                for item in tuple(active):
                    process = item[3]
                    if process.poll() is not None:
                        process.wait()
                        completed.append((*item, time.monotonic(), "returned"))
                        active.remove(item)
                if time.monotonic() >= deadline:
                    for item in active:
                        if item[3].poll() is None:
                            _kill_tree(item[3].pid)
                        item[3].wait(timeout=10)
                        completed.append((*item, time.monotonic(), "original_compile_deadline"))
                    active.clear()
                if active:
                    time.sleep(0.02)
        finally:
            for item in active:
                if item[3].poll() is None:
                    _kill_tree(item[3].pid)
                item[3].wait(timeout=10)
        joining_at, joining_cpu = time.monotonic(), time.process_time()
        with C._snapshot_operation(owner):
            for spec, output, transport, process, started, ended, outcome in completed:
                if CHILD._pin(spec) != transport.document["spec"]:
                    raise StageGateError("public scale original batch specification changed")
                jobs = _jobs(json.loads(spec.read_bytes()), spec)
                result = None
                if (process.returncode == 0 and output.is_file() and not output.is_symlink()
                        and output.stat().st_size <= 8 << 20):
                    result = transport.verify_result(json.loads(output.read_bytes()))
                    if (set(result) != {"schema", "observations", "original_compiler_child"}
                            or result["schema"] != SCHEMA or type(result["observations"]) is not list
                            or len(result["observations"]) != len(jobs)):
                        raise StageGateError("public scale batch completion changed its original case roster")
                for position, (item, case_spec, job) in enumerate(jobs):
                    if not 0 <= item["index"] < len(cases):
                        raise StageGateError("public scale batch has another original case index")
                    index, row = item["index"], rows[item["index"]]
                    if item["case"] != cases[index].record():
                        raise StageGateError("public scale batch lost its original producer case")
                    for name in ("source", "contract", "facts"):
                        if CHILD._digest(S._input_pin(job[name]["path"])) != CHILD._digest(job[name]):
                            raise StageGateError("public scale batch original case input changed before publication")
                    if result is None:
                        row.update(lowering_status="unavailable", reason=(
                            "original aggregate compile budget exhausted" if outcome == "original_compile_deadline"
                            else "original public scale batch refused"))
                    else:
                        actual = result["observations"][position]
                        if (set(actual) != {"index", "case", "observation", "started_monotonic", "finished_monotonic"}
                                or actual["index"] != index or actual["case"] != cases[index].record()):
                            raise StageGateError("public scale batch completed another original case")
                        stamps = (actual["started_monotonic"], actual["finished_monotonic"])
                        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in stamps)
                                or not started <= stamps[0] <= stamps[1] <= ended):
                            raise StageGateError("public scale batch supplied an invalid case observation interval")
                        _verify_payload(case_spec.parent, actual["observation"], ir_budget)
                        row.update(actual["observation"])
                        row["case_observation"] = {"started_monotonic": actual["started_monotonic"],
                            "finished_monotonic": actual["finished_monotonic"], "pending_until_worker_exit": False}
                    cases[index].verify()
                    row["preparation_process"] = {"pid": process.pid, "started_monotonic": started,
                        "finished_monotonic": ended, "mode": "original_sealed_persistent_batch", "outcome": outcome}
                    row["wall_seconds"] = ended - started
            for row in rows:
                if row["lowering_status"] == "not_attempted":
                    row["reason"] = "original aggregate compile budget exhausted"
            lifetime.close()
    ended, ended_cpu = time.monotonic(), time.process_time()
    if timing is not None:
        timing.update(parent_setup_s=prepared_at - entered, dispatch_wait_s=joining_at - prepared_at,
            parent_final_join_s=ended - joining_at, total_dispatch_s=ended - entered,
            workers=count, children_launched=len(completed), mode="original_sealed_persistent_batch",
            parent_process_cpu_s={"setup": prepared_cpu - entered_cpu,
                "dispatch_wait": joining_cpu - prepared_cpu, "final_join": ended_cpu - joining_cpu,
                "total": ended_cpu - entered_cpu}, scope="complete parent stages; no parent lease during child waits")
    return rows


def main(argv=None):
    from merlin.targetgen.capsule_runner import _die_with_parent

    from .public_scalability_gate import selected_budget

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--original-profile-fd", type=int, required=True)
    args = parser.parse_args(argv)
    _die_with_parent()
    raw, observations, events = args.spec.read_bytes(), [], []
    with CHILD.restore_model_child(args.original_profile_fd, spec_path=args.spec, spec_bytes=raw) as receipt:
        document = json.loads(raw)
        if args.result != args.spec.parent / ".scale_batch_result.json":
            raise StageGateError("public scale batch completion has another original output owner")
        jobs = _jobs(document, args.spec)
        executor = R.active_original_execution()
        budget = selected_budget(executor.owner)

        def progress():
            path = args.spec.parent / ".scale_batch_progress.json"
            temporary = path.with_suffix(".pending")
            S._write(temporary, {"schema": SCHEMA, "accepted": False,
                "source_inventory_sha256": executor.source_sha256,
                "scope": "observations pending final worker custody", "events": events})
            os.replace(temporary, path)

        for item, spec, job in jobs:
            if CHILD._pin(spec) != item["spec"]:
                raise StageGateError("public scale case changed before worker consumption")
            started = time.monotonic()
            events.append({"index": item["index"], "case": item["case"], "event": "begin",
                "at_monotonic": started, "at_utc_ns": time.time_ns(), "pid": os.getpid()})
            progress()
            observation = S.observe_case(job, spec_path=spec, executor=executor, original_budget=budget)
            if CHILD._pin(spec) != item["spec"]:
                raise StageGateError("public scale case changed during worker consumption")
            finished = time.monotonic()
            observations.append({"index": item["index"], "case": item["case"], "observation": observation,
                "started_monotonic": started, "finished_monotonic": finished})
            events.append({"index": item["index"], "case": item["case"], "event": "end",
                "at_monotonic": finished, "at_utc_ns": time.time_ns(), "pid": os.getpid()})
            progress()
        # Reopen every earlier case as well as the original batch before closing
        # its service/import/emission lifetime and making completion visible.
        _jobs(document, args.spec)
        for _, _, job in jobs:
            for name in ("source", "contract", "facts"):
                if CHILD._digest(S._input_pin(job[name]["path"])) != CHILD._digest(job[name]):
                    raise StageGateError("public scale earlier case input changed during its batch")
    S._write(args.result, {"schema": SCHEMA, "observations": observations, "original_compiler_child": receipt})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
