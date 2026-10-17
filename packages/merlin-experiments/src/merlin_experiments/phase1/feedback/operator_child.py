"""Fresh operator graders under the existing sealed original-host capability.

Scheduling remains in the parent. Children use the ordinary capsule ladder and
stable evidence paths; the parent publishes the result only after reopening the
original child completion. No callback or Python namespace crosses processes.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from pathlib import Path

import yaml

from merlin.common import cpu_pool
from merlin.common.bundle_snapshot import shared_entry_verification, verified_snapshot_scope
from merlin.targetgen import package_runtime as P
from merlin.targetgen import tier_policy
from merlin.targetgen.contract.readback_policy import ReadbackPolicy
from merlin_experiments.phase2.contracts import StageGateError

from . import compiler_profile_child as CHILD
from . import compiler_profile_runtime as R

ENV = "MERLIN_OPERATOR_GRADE_PROCESSES"
_ADAPTERS = weakref.WeakKeyDictionary()
#: original executor -> the snapshot entry its grade verified once before fanning out.
_GRADE_ENTRIES = weakref.WeakKeyDictionary()
_LOCK = threading.RLock()


def enabled():
    choice = os.environ.get(ENV, "0")
    if choice not in ("", "0", "1"):
        raise ValueError(f"{ENV} requires 0 or 1")
    return choice == "1"


def register_adapter(adapter, *, engine, target, selection, readback_policy):
    """Record only an original simulator factory's live, non-executable selection."""
    with _LOCK:
        _ADAPTERS[adapter] = (engine, target, selection, readback_policy)


def adapter_selection(adapters, *, target):
    from merlin.targetgen import capsule_runner as runner

    chosen = runner._resolve_oracle_adapters(target) if adapters is None else adapters
    document = {}
    with _LOCK:
        for tier, adapter in chosen.items():
            original = _ADAPTERS.get(adapter)
            if original is None or original[1] != target or tier not in {"L2", "L3", "L4"}:
                raise StageGateError("operator processes require original selected simulator adapters")
            engine, _, selection, policy = original
            document[tier] = {
                "engine": engine,
                "selection": CHILD._data(selection),
                "readback_policy": policy.record() if policy is not None else None,
            }
    return copy.deepcopy(document)


def validate(*, target, adapters, config, perf_extractor):
    if config is not None or perf_extractor is not None:
        raise StageGateError("operator processes cannot transport arbitrary configuration or performance callbacks")
    if tier_policy.budget_seconds() is not None:
        raise StageGateError("operator processes require an unlimited shared certify budget")
    if type(P.active_package_executor()) is not R._Executor:
        raise StageGateError("operator processes require the original live profile executor")
    selection = adapter_selection(adapters, target=target)
    _open_grade_entry(P.active_package_executor())
    return selection


def _open_grade_entry(executor):
    """Verify the original bundle snapshot once, before this grade's capsules fan out.

    Every capsule's parent preparation used to repeat the complete entry read
    (payload digest, roster scan and guard setup) on its own thread, all of it
    GIL-bound in one controller process. Capsules now start from this entry
    while its kernel guard observes no change; each keeps its own fresh final
    verification before its child is launched and again before its result is
    accepted. Without a live guard every capsule keeps its full entry read.
    """
    owner = executor.owner
    with _LOCK:
        previous = _GRADE_ENTRIES.pop(executor, None)
    if previous is not None:
        previous.close()
    if type(owner) is not CHILD.CommonTransformService:
        return
    entry = shared_entry_verification(owner.workspace, owner.bundle, repo=owner.repo)
    if entry is None:
        return
    with _LOCK:
        _GRADE_ENTRIES[executor] = entry
    weakref.finalize(executor, entry.close)


def _publish(path, row):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _communicate_or_defer(process, *, timeout_s):
    """Wait for an original child, or return the scheduler's observed deferral reason.

    A direct/self-check grade has no deferral probe and keeps the ordinary single
    wait. A background grade polls only while its original source guard exists;
    the scheduler, not the candidate or child, decides when work is superseded.
    """
    from merlin.targetgen.capsule_scheduling import active_defer_probe

    probe = active_defer_probe()
    if probe is None:
        process.communicate(timeout=timeout_s)
        return None
    deadline = time.monotonic() + timeout_s
    while True:
        if reason := probe():
            return reason
        try:
            process.communicate(timeout=min(5.0, max(0.001, deadline - time.monotonic())))
            return None
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline:
                raise


@contextlib.contextmanager
def _prepared_child(document):
    """Batch only synchronous original-owner issuance; close custody before child launch.

    Holds no CPU lease itself. Snapshot verification leases one shared CPU per
    stage behind its process admission gate, so a grade's capsules waiting to
    verify never reserve pool CPUs that author requests are queued for; a held
    slot is never preempted, so a lease held across this GIL-bound preparation
    used to keep the pool idle and locked for minutes per grade.
    """
    executor = P.active_package_executor()
    if type(executor) is not R._Executor or type(executor.owner) is not CHILD.CommonTransformService:
        raise StageGateError("operator child requires its original live profile owner")
    owner = executor.owner
    with _LOCK:
        entry = _GRADE_ENTRIES.get(executor)
    with contextlib.ExitStack() as lifetime:
        with verified_snapshot_scope(owner.workspace, owner.bundle, repo=owner.repo, entry=entry):
            root = CHILD.model_child_temporary_root()  # Retain original import/owner verification.
            if root is None:
                raise StageGateError("operator child lost its original profile owner")
            private = Path(lifetime.enter_context(tempfile.TemporaryDirectory(prefix="operator-grade-", dir=root)))
            spec, output = private / "spec.json", private / "result.json"
            spec.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
            environment = cpu_pool.child_environment()
            if environment.get("MERLIN_PASS_LOG"):
                environment["MERLIN_PASS_LOG_CAPSULE"] = str(document["capsule"]["name"])
                environment["MERLIN_PASS_LOG_REQUIREMENTS"] = json.dumps(
                    list(document["capsule"].get("pass_requirements") or [])
                )
            transport = lifetime.enter_context(
                CHILD.prepare_model_child(spec_path=spec, environment=environment, package_dir=document["package_dir"])
            )
            if transport is None:
                raise StageGateError("operator child requires its original sealed capability")
        # Final independent custody closes before exposing the transport to Popen. Its FD and
        # temporary directory remain owned here until final child-result verification completes.
        yield spec, output, environment, transport


@contextlib.contextmanager
def _restored_child(*args, **kwargs):
    """Lease synchronous custody entry/exit, never the grader body or its waits."""
    original = CHILD.restore_model_child(*args, **kwargs)
    with cpu_pool.shared_compute_pool():
        receipt = original.__enter__()
    try:
        yield receipt
    except BaseException:
        with cpu_pool.shared_compute_pool():
            suppressed = original.__exit__(*sys.exc_info())
        if not suppressed:
            raise
    else:
        with cpu_pool.shared_compute_pool():
            original.__exit__(None, None, None)


def completed_tier_costs(result):
    """``{tier: adapter wall seconds}`` for each tier this child executed to a compared verdict.

    The parent prices its ladder from these (``tier_policy.record_cost``), exactly as the in-process
    ladder prices a tier only once its adapter has returned. A tier whose adapter raised -- unavailable,
    refused before dispatch, timed out -- carries only a ``partial`` wall clock, and pricing it would
    teach the order that the tier costs what its refusal cost: one ELF build in place of a simulation.
    """
    return {
        tier: record["timing"]["adapter_wall_s"]
        for tier, record in (result.get("tiers") or {}).items()
        if isinstance(record, dict)
        and isinstance(record.get("timing"), dict)
        and not record["timing"].get("partial")
        and type(record["timing"].get("adapter_wall_s")) in (int, float)
    }


def grade(capsule, package_dir, *, runs_root, contract, target, suite, dtype, adapters, timeout, workers, no_oracle):
    from merlin.targetgen import capsule_runner as runner

    selection = adapter_selection(adapters, target=target)
    if capsule.get("kind") == "model":
        raise StageGateError("operator child cannot replace the whole-model route")
    costs = {tier: tier_policy.observed_cost(target, tier) for tier in selection}
    document = {
        "grade_kind": "operator",
        "capsule": CHILD._data(capsule),
        "package_dir": str(package_dir),
        "target": target,
        "runs_root": str(runs_root),
        "contract": str(contract) if contract else None,
        "suite": suite,
        "dtype": dtype,
        "adapters": selection,
        "timeout": timeout,
        "workers": workers,
        "no_oracle": no_oracle,
        "observed_costs_s": costs,
    }
    with _prepared_child(document) as (spec, output, environment, transport):
        command = [
            sys.executable,
            "-c",
            runner._CHILD_PREAMBLE,
            "--operator-grade",
            str(spec),
            "--operator-grade-out",
            str(output),
            *transport.argv,
        ]
        process = subprocess.Popen(
            cpu_pool.worker_pool_command(command, environment),
            env=environment,
            pass_fds=transport.pass_fds,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            deferred = _communicate_or_defer(process, timeout_s=max(1, timeout) * (len(selection) + 6))
            if deferred:
                runner._kill_tree(process.pid)
                process.wait(timeout=10)
                row = {
                    "capsule": capsule["name"],
                    "kind": capsule.get("kind"),
                    "label": capsule.get("label"),
                    "status": "gated",
                    "background_deferred": deferred,
                    "failure": {"plane": "gate", "category": "BACKGROUND_DEFERRED", "detail": deferred},
                }
                cfg = runner._config_for_target(target, suite, dtype)
                result_path = Path(runs_root) / "runs" / cfg.suite / capsule["name"] / "capsule_result.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                _publish(result_path, row)
                return row
        except BaseException:
            runner._kill_tree(process.pid)
            process.wait(timeout=10)
            raise
        if process.returncode != 0 or not output.is_file():
            raise StageGateError(f"operator grade child exited {process.returncode} without a verified completion")
        # Fresh and complete, and unleased here for the same reason as preparation: its snapshot
        # read leases its own stage behind the admission gate.
        row = transport.verify_result(json.loads(output.read_bytes()))
        if row.get("capsule") != capsule["name"] or adapter_selection(adapters, target=target) != selection:
            raise StageGateError("operator child result or selected oracle changed")
        for tier, value in (row.pop("operator_child_costs_s", {}) or {}).items():
            tier_policy.record_cost(target, tier, value)
        tier_policy.note_spend(target, row.pop("operator_child_spend_s", None))
        row["operator_grade_process"] = {"pid": process.pid, "workers": workers, "mode": "fresh_sealed_child"}
        cfg = runner._config_for_target(target, suite, dtype)
        result_path = Path(runs_root) / "runs" / cfg.suite / capsule["name"] / "capsule_result.json"
        manifest = yaml.safe_load((result_path.parent / ".operator_child_manifest.yaml").read_bytes())
        manifest["metadata"]["original_compiler_child"] = row["original_compiler_child"]
        _publish(result_path.parent / "run_manifest.yaml", manifest)
        _publish(result_path, row)
        return row


def main(argv):
    from merlin.targetgen import capsule_runner as runner
    from merlin.targetgen.owned_host_leaf_execution import selected_host_leaf

    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-grade", type=Path, required=True)
    parser.add_argument("--operator-grade-out", type=Path, required=True)
    parser.add_argument("--original-profile-fd", type=int, required=True)
    args = parser.parse_args(argv)
    runner._die_with_parent()
    raw = args.operator_grade.read_bytes()
    job = json.loads(raw)
    with _restored_child(args.original_profile_fd, spec_path=args.operator_grade, spec_bytes=raw) as receipt:
        capsule, target = job["capsule"], job["target"]
        if job.get("grade_kind") != "operator" or capsule.get("kind") == "model" or selected_host_leaf(capsule):
            raise StageGateError("operator child requires the original accelerator capsule route")
        adapters = {
            tier: runner.simulator_adapter(
                record["engine"],
                target,
                selection=record["selection"],
                readback_policy=(
                    ReadbackPolicy.from_record(record["readback_policy"])
                    if record["readback_policy"] is not None
                    else None
                ),
            )
            for tier, record in job["adapters"].items()
        }
        for tier, value in job["observed_costs_s"].items():
            tier_policy.record_cost(target, tier, value)
        with cpu_pool.shared_compute_pool():
            pkg = P.load_package(job["package_dir"], contract=job["contract"])
            P.integrity_scan(pkg)
        result = runner.run_capsule(
            capsule,
            job["package_dir"],
            runs_root=job["runs_root"],
            run_id=capsule["name"],
            contract=job["contract"],
            oracle_adapters=adapters,
            pkg=pkg,
            timeout=job["timeout"],
            target=target,
            suite=job["suite"],
            dtype=job["dtype"],
            no_oracle=job["no_oracle"],
            workers=job["workers"],
            _defer_result_publication=True,
        )
        result["operator_child_costs_s"] = completed_tier_costs(result)
        result["operator_child_spend_s"] = tier_policy.spent(target)
    result["original_compiler_child"] = receipt
    args.operator_grade_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0
