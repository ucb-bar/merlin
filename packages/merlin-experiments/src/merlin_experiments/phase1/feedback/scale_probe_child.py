"""Independent public emission cases under original sealed compiler authority.

The parent derives the fixed public ladder and issues only source-bound jobs.
Each fresh child restores its own namespace. No parent compute slot survives
into a child wait, and only verified children enter the final ordered report.
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

from merlin.common import cpu_pool, invocation_record
from merlin.common.digest import sha256_file
from merlin.targetgen import package_runtime as P
from merlin.targetgen.rtl.facts import observed_facts
from merlin_experiments.phase2.contracts import StageGateError, exact_tree_record

from . import compiler_profile_child as CHILD
from . import compiler_profile_runtime as R

SCHEMA = "merlin.original_public_scale_child.v1"


def workers():
    if not cpu_pool.shared_compute_enabled():
        return 1
    value = os.environ.get("MERLIN_GRADE_WORKERS", "1")
    if not value.isdecimal() or int(value) < 1:
        raise ValueError("public scale preparation requires a positive original worker count")
    return min(int(value), len(cpu_pool.grading_cpus()))


def _write(path, document):
    path.write_text(json.dumps(document, sort_keys=True, allow_nan=False), encoding="utf-8")


def _input_pin(value):
    path = CHILD._plain(value)
    if path.is_dir():
        return {"path": str(path), "kind": "directory", "tree": exact_tree_record(path)}
    return CHILD._pin(path)


def _verify_products(root, row):
    """Reopen command receipts and lexical observations before parent publication."""
    from merlin.llvmlower.device_ir_scalability import analyze_device_ir

    fields = {"lowering_status", "compilation_status", "code_growth", "resource_budget",
              "emission_invocation_sha256", "translation_invocation_sha256", "static_obligations"}
    unknown = {key: "UNKNOWN" for key in (
        "semantic_coverage", "index_bounds", "resource_legality", "complete_output_coverage")}
    if set(row) != fields or row["compilation_status"] != "not_attempted" or row["static_obligations"] != unknown:
        raise StageGateError("public scale child cannot add execution/static qualification claims")

    paths = sorted(root.rglob("invocation.json"))
    if not paths:
        raise StageGateError("public scale child supplied no actual invocation evidence")
    records = [(path, invocation_record.verify(path)) for path in paths]
    for field, stage in (("emission_invocation_sha256", "public_scale_source_lowering"),
                         ("translation_invocation_sha256", "public_scale_llvm_translation")):
        matches = [path for path, record in records if record.get("stage") == stage]
        if len(matches) != 1 or sha256_file(matches[0]) != row.get(field):
            raise StageGateError("public scale child lost its exact emission/translation producer")
    analysis = analyze_device_ir(root / "generated/lowered.ll")
    saved = row.get("code_growth") or {}
    if any(saved.get(key) != value for key, value in analysis.items() if key != "elapsed_s"):
        raise StageGateError("public scale child lexical observations changed")
    mlir = root / "generated/lowered.llvm.mlir"
    if mlir.stat().st_size != saved.get("emitted_mlir_bytes") or sha256_file(mlir) != saved.get("emitted_mlir_sha256"):
        raise StageGateError("public scale child emitted source changed")
    return analysis


def emit_cases(*, owner, candidate, target_experiment, capsules, cases, contract, facts_path,
               work_root, deadline, ir_budget, sandbox_inputs=None, timing=None):
    from . import compile_only as C

    entered, count = time.monotonic(), workers()
    rows = [C._case_row(case) for case in cases]
    prepared, active = [], []
    with contextlib.ExitStack() as lifetime:
        # Setup closes before Popen. The original namespace remains parent-private;
        # every child issues a distinct restored namespace in its fresh interpreter.
        with C._snapshot_operation(owner):
            lifetime.enter_context(R._original_execution_lifetime(owner,
                target_experiment=target_experiment, candidate=candidate,
                work_root=work_root / "child-owner", sandbox_inputs=sandbox_inputs))
            for index, case in enumerate(cases):
                case.verify()
                root = work_root / f"case-{index:03d}"
                root.mkdir(mode=0o700)
                source = work_root / "sources" / f"case-{index:03d}.mlir"
                source.write_text(case.source_ir)
                spec, result = root / ".scale_child_spec.json", root / ".scale_child_result.json"
                remaining = min(600, math.floor(deadline - time.monotonic()))
                if remaining < 1:
                    rows[index]["reason"] = "original aggregate compile budget exhausted"
                    continue
                selected = [cap for cap in capsules
                            if str(cap.get("name") or Path(cap["__dir__"]).name) == case.member]
                document = {"schema": SCHEMA, "package_dir": str(candidate), "target": target_experiment.target,
                    "case": case.record(), "capsules": CHILD._data(selected), "source": CHILD._pin(source),
                    "contract": _input_pin(contract), "facts": CHILD._pin(facts_path),
                    "output": str(root), "deadline": deadline,
                    "ir_budget": ir_budget.record() if ir_budget is not None else None}
                _write(spec, document)
                environment = dict(os.environ)
                transport = lifetime.enter_context(CHILD.prepare_model_child(
                    spec_path=spec, environment=environment, package_dir=candidate))
                if transport is None:
                    raise StageGateError("public scale children require the original sealed owner")
                prepared.append((index, spec, result, environment, transport))
        setup_finished = time.monotonic()
        pending = iter(prepared)

        def launch(item):
            index, spec, result, environment, transport = item
            command = [sys.executable, "-m", __name__, "--spec", str(spec), "--result", str(result), *transport.argv]
            log = lifetime.enter_context((spec.parent / ".scale_child.log").open("w"))
            process = subprocess.Popen(cpu_pool.worker_pool_command(command, environment), env=environment,
                pass_fds=transport.pass_fds, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active.append((item, process, time.monotonic()))

        def refill():
            while len(active) < count and time.monotonic() < deadline:
                item = next(pending, None)
                if item is None:
                    break
                launch(item)

        completed = []
        try:
            refill()
            while active:
                for item, process, started in tuple(active):
                    if process.poll() is None:
                        continue
                    process.wait()
                    active.remove((item, process, started))
                    completed.append((item, process, started, time.monotonic()))
                if time.monotonic() >= deadline and active:
                    from merlin.targetgen.capsule_runner import _kill_tree
                    for item, process, started in active:
                        if process.poll() is None:
                            _kill_tree(process.pid)
                        process.wait(timeout=10)
                        completed.append((item, process, started, time.monotonic()))
                    active.clear()
                refill()
                if active:
                    time.sleep(0.02)
            for index, *_ in pending:
                rows[index]["reason"] = "original aggregate compile budget exhausted"
        finally:
            from merlin.targetgen.capsule_runner import _kill_tree
            for _, process, _ in active:
                if process.poll() is None:
                    _kill_tree(process.pid)
                process.wait(timeout=10)
        # No child or RPC is running here. A mandatory final full custody read
        # encloses result joins, then closes before the aggregate is published.
        join_started = time.monotonic()
        with C._snapshot_operation(owner):
            for item, process, started, finished in completed:
                index, spec, output, _, transport = item
                if CHILD._pin(spec) != transport.document["spec"]:
                    raise StageGateError("public scale child original specification changed")
                original_inputs = json.loads(spec.read_bytes())
                for name in ("source", "contract", "facts"):
                    if CHILD._digest(_input_pin(original_inputs[name]["path"])) != CHILD._digest(original_inputs[name]):
                        raise StageGateError("public scale child original input changed before parent publication")
                row = rows[index]
                if (process.returncode != 0 or not output.is_file() or output.is_symlink()
                        or output.stat().st_size > 1024 * 1024):
                    row.update(lowering_status="unavailable", reason="original public scale child refused")
                    emitted = output.parent / "generated/lowered.llvm.mlir"
                    if emitted.is_file() and not emitted.is_symlink():
                        row["partial_emission"] = {"bytes": emitted.stat().st_size, "sha256": sha256_file(emitted),
                                                   "scope": "unverified retained MLIR bytes"}
                else:
                    result = transport.verify_result(json.loads(output.read_bytes()))
                    if (set(result) != {"schema", "case", "observation", "original_compiler_child"}
                            or result.get("schema") != SCHEMA
                            or CHILD._digest(result.get("case")) != CHILD._digest(cases[index].record())):
                        raise StageGateError("public scale child completed another original case")
                    payload = result["observation"]
                    if payload.get("lowering_status") in {"emitted", "resource_refused"}:
                        analysis = _verify_products(output.parent, payload)
                        expected = ir_budget.evaluate(analysis) if ir_budget is not None else {"mode": "not_selected"}
                        if (CHILD._digest(payload.get("resource_budget")) != CHILD._digest(expected)
                                or payload["lowering_status"] != (
                                    "resource_refused" if expected.get("status") == "refused" else "emitted")):
                            raise StageGateError("public scale child changed its selected resource policy")
                    elif set(payload) != {"lowering_status", "reason"} or payload["lowering_status"] != "unavailable":
                        raise StageGateError("public scale child supplied an unrecognized completion")
                    row.update(payload)
                cases[index].verify()
                row["wall_seconds"] = finished - started
                row["preparation_process"] = {"pid": process.pid, "started_monotonic": started,
                    "finished_monotonic": finished, "mode": "fresh_original_sealed_child"}
            # Only custody accessors and closed-descriptor/namespace cleanup
            # remain. Include the namespace's last check in this operation,
            # then perform its independent full exit before report publication.
            lifetime.close()
    finished = time.monotonic()
    if timing is not None:
        timing.update(parent_setup_s=setup_finished - entered,
                      dispatch_wait_s=join_started - setup_finished,
                      parent_final_join_s=finished - join_started,
                      total_dispatch_s=finished - entered, workers=count,
                      children_launched=len(completed),
                      scope="actual parent stage wall times, including all custody checks")
    return rows


def main(argv=None):
    from merlin.llvmlower.device_ir_scalability import DeviceIRBudget
    from merlin.targetgen import capsule_runner as CR
    from merlin_experiments.phase0 import public_scale_sources

    from . import compile_only as C

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--original-profile-fd", type=int, required=True)
    args = parser.parse_args(argv)
    CR._die_with_parent()
    raw = args.spec.read_bytes()
    with CHILD.restore_model_child(args.original_profile_fd, spec_path=args.spec, spec_bytes=raw) as receipt:
        job = json.loads(raw)
        executor = R.active_original_execution()
        owner, target = executor.owner, executor.target
        if job.get("schema") != SCHEMA or args.result != args.spec.parent / ".scale_child_result.json":
            raise StageGateError("public scale child has another protocol/output owner")
        for name in ("source", "contract", "facts"):
            if CHILD._digest(_input_pin(job[name]["path"])) != CHILD._digest(job[name]):
                raise StageGateError("public scale child original input changed")
        facts = json.loads(Path(job["facts"]["path"]).read_bytes())
        edges = {row.get("cols") for row in (facts.get("facts") or {}).get("arrays", [])
                 if type(row.get("cols")) is int and row["cols"] > 0}
        cases, _ = public_scale_sources.derive(job["capsules"], selected=job["case"]["member"],
            index_bits=owner.profile.get("host_index_bits"), tile_edge=next(iter(edges)) if len(edges) == 1 else None)
        matched = [case for case in cases if case.record() == job["case"]]
        if len(matched) != 1 or Path(job["source"]["path"]).read_text() != matched[0].source_ir:
            raise StageGateError("public scale child is not the original producer-derived variant")
        from .public_scalability_gate import selected_budget

        original_budget = selected_budget(owner)
        if CHILD._digest(job["ir_budget"]) != CHILD._digest(
                original_budget.record() if original_budget is not None else None):
            raise StageGateError("public scale child changed its original resource budget")
        budget = DeviceIRBudget(**job["ir_budget"]) if job["ir_budget"] is not None else None
        root, candidate = Path(job["output"]), Path(job["package_dir"])
        if root != args.spec.parent:
            raise StageGateError("public scale child products lost their original owner")
        remaining = min(600, math.floor(job["deadline"] - time.monotonic()))
        observation = {"lowering_status": "unavailable", "reason": "original aggregate compile budget exhausted"}
        if remaining >= 1:
            with (P.scoped_package_executor(C._Executor(target, candidate, root,
                    common_service=owner, inputs=executor.inputs)),
                  observed_facts(target.target, facts, job["facts"]["path"])):
                observation = C._emit_scale_case(candidate, Path(job["source"]["path"]), root,
                    contract=Path(job["contract"]["path"]), target=target.target,
                    timeout_s=remaining, ir_budget=budget)
        for name in ("source", "contract", "facts"):
            if CHILD._digest(_input_pin(job[name]["path"])) != CHILD._digest(job[name]):
                raise StageGateError("public scale child original input changed before completion")
        matched[0].verify()
    # Completion becomes visible only after the restored owner's fresh final checks.
    _write(args.result, {"schema": SCHEMA, "case": job["case"], "observation": observation,
                         "original_compiler_child": receipt})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
