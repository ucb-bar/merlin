"""Fresh operator graders under the existing sealed original-host capability.

Scheduling remains in the parent. Children use the ordinary capsule ladder and
stable evidence paths; the parent publishes the result only after reopening the
original child completion. No callback or Python namespace crosses processes.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import threading
import weakref
from pathlib import Path

import yaml

from merlin.common import cpu_pool
from merlin.targetgen import package_runtime as P
from merlin.targetgen import tier_policy
from merlin.targetgen.contract.readback_policy import ReadbackPolicy
from merlin_experiments.phase2.contracts import StageGateError

from . import compiler_profile_child as CHILD
from . import compiler_profile_runtime as R

ENV = "MERLIN_OPERATOR_GRADE_PROCESSES"
_ADAPTERS = weakref.WeakKeyDictionary()
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
            document[tier] = {"engine": engine, "selection": CHILD._data(selection),
                              "readback_policy": policy.record() if policy is not None else None}
    return copy.deepcopy(document)


def validate(*, target, adapters, config, perf_extractor):
    if config is not None or perf_extractor is not None:
        raise StageGateError("operator processes cannot transport arbitrary configuration or performance callbacks")
    if tier_policy.budget_seconds() is not None:
        raise StageGateError("operator processes require an unlimited shared certify budget")
    if type(P.active_package_executor()) is not R._Executor:
        raise StageGateError("operator processes require the original live profile executor")
    return adapter_selection(adapters, target=target)


def _publish(path, row):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def grade(capsule, package_dir, *, runs_root, contract, target, suite, dtype, adapters, timeout, workers, no_oracle):
    from merlin.targetgen import capsule_runner as runner

    selection = adapter_selection(adapters, target=target)
    if capsule.get("kind") == "model":
        raise StageGateError("operator child cannot replace the whole-model route")
    root = CHILD.model_child_temporary_root()
    if root is None:
        raise StageGateError("operator child lost its original profile owner")
    costs = {tier: tier_policy.observed_cost(target, tier) for tier in selection}
    with tempfile.TemporaryDirectory(prefix="operator-grade-", dir=root) as temporary:
        private = Path(temporary)
        spec, output = private / "spec.json", private / "result.json"
        document = {
            "grade_kind": "operator", "capsule": CHILD._data(capsule), "package_dir": str(package_dir),
            "target": target, "runs_root": str(runs_root), "contract": str(contract) if contract else None,
            "suite": suite, "dtype": dtype, "adapters": selection, "timeout": timeout,
            "workers": workers, "no_oracle": no_oracle, "observed_costs_s": costs,
        }
        spec.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
        environment = dict(os.environ)
        if environment.get("MERLIN_PASS_LOG"):
            environment["MERLIN_PASS_LOG_CAPSULE"] = str(capsule["name"])
            environment["MERLIN_PASS_LOG_REQUIREMENTS"] = json.dumps(list(capsule.get("pass_requirements") or []))
        with CHILD.prepare_model_child(spec_path=spec, environment=environment, package_dir=package_dir) as transport:
            if transport is None:
                raise StageGateError("operator child requires its original sealed capability")
            command = [sys.executable, "-c", runner._CHILD_PREAMBLE,
                       "--operator-grade", str(spec), "--operator-grade-out", str(output), *transport.argv]
            process = subprocess.Popen(cpu_pool.worker_pool_command(command, environment), env=environment,
                pass_fds=transport.pass_fds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True)
            try:
                process.communicate(timeout=max(1, timeout) * (len(selection) + 6))
            except BaseException:
                runner._kill_tree(process.pid)
                process.wait(timeout=10)
                raise
            if process.returncode != 0 or not output.is_file():
                raise StageGateError(f"operator grade child exited {process.returncode} without a verified completion")
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
    with CHILD.restore_model_child(args.original_profile_fd, spec_path=args.operator_grade, spec_bytes=raw) as receipt:
        capsule, target = job["capsule"], job["target"]
        if job.get("grade_kind") != "operator" or capsule.get("kind") == "model" or selected_host_leaf(capsule):
            raise StageGateError("operator child requires the original accelerator capsule route")
        adapters = {
            tier: runner.simulator_adapter(record["engine"], target, selection=record["selection"],
                        readback_policy=(ReadbackPolicy.from_record(record["readback_policy"])
                                         if record["readback_policy"] is not None else None))
            for tier, record in job["adapters"].items()
        }
        for tier, value in job["observed_costs_s"].items():
            tier_policy.record_cost(target, tier, value)
        with cpu_pool.shared_compute_pool():
            pkg = P.load_package(job["package_dir"], contract=job["contract"])
            P.integrity_scan(pkg)
        result = runner.run_capsule(capsule, job["package_dir"], runs_root=job["runs_root"], run_id=capsule["name"],
            contract=job["contract"], oracle_adapters=adapters, pkg=pkg, timeout=job["timeout"], target=target,
            suite=job["suite"], dtype=job["dtype"], no_oracle=job["no_oracle"], workers=job["workers"],
            _defer_result_publication=True)
        result["operator_child_costs_s"] = {
            tier: record["timing"]["adapter_wall_s"] for tier, record in result.get("tiers", {}).items()
            if isinstance(record, dict) and isinstance(record.get("timing"), dict)
            and type(record["timing"].get("adapter_wall_s")) in (int, float)
        }
        result["operator_child_spend_s"] = tier_policy.spent(target)
    result["original_compiler_child"] = receipt
    args.operator_grade_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0
