"""Unqualified compiler/complete-output Spike loop over already captured inputs.

This deliberately runs no publication, pass factory, tier audit or source-effect
qualification. The normal native numerical runner still binds the exact inputs
and compares every output. Final qualification remains a separate operation.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path

from merlin.common.cpu_pool import shared_compute_pool
from merlin.targetgen import native_model_execution as NATIVE
from merlin.targetgen import package_runtime as PACKAGE

SCHEMA = "merlin.phase2.lean_native_evaluation.v1"
INTAKE = "merlin.native_pointer_source_intake.v2"


def _pin(path):
    return NATIVE._digest(Path(path))


def _tree(path):
    root = Path(path).resolve(strict=True)
    rows = []
    for leaf in sorted(root.rglob("*")):
        if ".git" in leaf.relative_to(root).parts or "__pycache__" in leaf.relative_to(root).parts:
            continue
        if leaf.is_symlink():
            raise ValueError("lean source/cache identity requires ordinary files")
        if leaf.is_file():
            rows.append(dict(member=str(leaf.relative_to(root)), **_pin(leaf)))
    return rows


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _source_bytes(rows):
    return [{key: row[key] for key in ("member", "sha256", "size_bytes")} for row in rows]


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def member_identity(member):
    """Exact saved source, all operands/references and selected target bytes."""
    required = ("id", "capsule_dir", "interface_mlir", "runtime_bundle", "target_descriptor", "rtl_facts")
    if not isinstance(member, Mapping) or any(not member.get(key) for key in required):
        raise ValueError("lean evaluation needs an explicit saved member mapping")
    import yaml

    descriptor = yaml.safe_load(Path(member["target_descriptor"]).read_bytes())
    return dict(
        member={key: str(value) if isinstance(value, Path) else value for key, value in member.items()},
        capsule=_tree(member["capsule_dir"]),
        interface=_pin(member["interface_mlir"]),
        runtime_bundle=_tree(member["runtime_bundle"]),
        target_descriptor=_pin(member["target_descriptor"]),
        target_contract=_pin(descriptor["hardware_spec"]["target_contract"]),
        rtl_facts=_pin(member["rtl_facts"]),
        source_pins=[_pin(path) for path in member.get("source_pins", ())],
    )


def _remaining(deadline):
    value = deadline - time.monotonic()
    if value < 1:
        raise TimeoutError("lean evaluation remaining budget expired")
    return value


def _complete_numeric(native, command_buffer):
    numeric = native.get("numeric") or {}
    outputs = (command_buffer.get("kernel_abi") or {}).get("outputs") or []
    rows = numeric.get("per_output") or {}
    return bool(
        not native.get("failure")
        and numeric.get("status") == "pass"
        and numeric.get("policy") == "exact_int"
        and outputs
        and set(rows) == set(outputs)
        and all(
            row.get("status") == "pass"
            and row.get("mismatch_count") == 0
            and row.get("n_elements") == math.prod(command_buffer["tensors"][name]["shape"])
            for name, row in rows.items()
        )
    )


def _native_compile_join(native, command_buffer, lowered_pin):
    candidate = native.get("candidate") or {}
    return (
        candidate.get("command_buffer_sha256")
        == hashlib.sha256(json.dumps(command_buffer, sort_keys=True).encode()).hexdigest()
        and candidate.get("lowered_mlir_sha256") == lowered_pin["sha256"]
    )


def evaluate_member(
    *, member, candidate_source, work_root, timeout_s, count_runtime=None, count_observer=None, progress_callback=None
):
    """Compile one retained package and run every saved output through Spike."""
    deadline = time.monotonic() + float(timeout_s)
    root, package = Path(work_root), Path(candidate_source).resolve(strict=True)
    root.mkdir(parents=True, exist_ok=False)
    record = dict(
        schema=SCHEMA,
        member=member["id"],
        work_root=str(root),
        candidate_source=str(package),
        qualification="UNQUALIFIED",
        source_effects="PENDING",
        status="incomplete",
        failure_category=None,
        public_failure_summary=None,
        complete_outputs_checked=False,
        count_observation=None,
    )
    started = time.monotonic()

    def checkpoint(phase):
        record["phase"] = phase
        record["wall_seconds"] = time.monotonic() - started
        _write_json(root / "evaluation.json", record)
        if progress_callback is not None:
            progress_callback(record)

    try:
        record["input_identity"] = member_identity(member)
        record["package_identity"] = _tree(package)
        checkpoint("compiling")
        # One worker process uses one target selection. The campaign's members
        # share this descriptor; no source/correspondence owner is constructed.
        descriptor = Path(member["target_descriptor"])
        import yaml

        declaration = yaml.safe_load(descriptor.read_bytes())
        selected_contract = declaration["hardware_spec"]["target_contract"]
        selected_target = member.get("target") or declaration.get("target")
        if not selected_target:
            contract = yaml.safe_load(Path(selected_contract).read_bytes())
            selected_target = contract.get("target") or contract["name"]
        os.environ["MERLIN_TARGET_EXPERIMENT"] = str(descriptor)
        os.environ["MERLIN_TARGET_CONTRACT"] = str(selected_contract)
        os.environ["MERLIN_RTL_FACTS"] = str(member["rtl_facts"])
        compiled = root / "compile"
        compiled.mkdir()
        command_path, llvm_path = compiled / "command_buffer.json", compiled / "lowered.llvm.mlir"
        with shared_compute_pool(timeout_s=_remaining(deadline)):
            emitted = PACKAGE.run_entrypoint(
                PACKAGE.load_package(package),
                "emit_analysis_bundle",
                Path(member["interface_mlir"]),
                command_path,
                timeout=int(min(120, _remaining(deadline))),
            )
        llvm_path.write_text(emitted.stdout)
        (compiled / "compiler.stderr").write_text(emitted.stderr)
        record["compile_returncode"] = emitted.returncode
        record["lowered_mlir"] = _pin(llvm_path)
        if command_path.is_file():
            record["command_buffer"] = _pin(command_path)
        if emitted.returncode:
            record.update(
                status="compile_failed",
                detail=emitted.stderr[-4000:],
                failure_category="compiler_error",
                public_failure_summary="Compiler emission failed; run a local compiler self-check for this member.",
            )
        else:
            cb = json.loads(command_path.read_bytes())
            checkpoint("checking_complete_outputs")
            with shared_compute_pool(timeout_s=_remaining(deadline)):
                native = NATIVE.execute_candidate_model(
                    command_buffer=cb,
                    lowered_mlir_text=emitted.stdout,
                    capsule_dir=Path(member["capsule_dir"]),
                    capture_bundle=Path(member["runtime_bundle"]),
                    target=selected_target,
                    out_dir=root / "native",
                    simulator="spike",
                    timeout=int(_remaining(deadline)),
                    oracle_tier_ceiling="L2",
                    complete_call_timing=False,
                    rtl_facts=Path(member["rtl_facts"]),
                    board_config=member.get("board_config"),
                    source_intake_protocol=INTAKE,
                )
            record["native_result"] = _pin(root / "native/result.json")
            record["numeric"] = native.get("numeric")
            record["elf"] = native.get("elf")
            complete = _complete_numeric(native, cb) and _native_compile_join(native, cb, record["lowered_mlir"])
            record.update(status="numeric_pass" if complete else "numeric_failed", complete_outputs_checked=complete)
            if not complete:
                record["detail"] = native.get("failure") or native.get("numeric")
                record["failure_category"] = "native_execution_error" if native.get("failure") else "numeric_mismatch"
                record["public_failure_summary"] = (
                    "The complete-output Spike run did not complete."
                    if native.get("failure")
                    else "The complete-output Spike comparison failed or omitted required outputs."
                )
            if complete and count_runtime is not None:
                if _tree(package) != record["package_identity"] or member_identity(member) != record["input_identity"]:
                    raise ValueError("compiler, saved source, input or reference changed during numerical evaluation")
                checkpoint("observing_counts")
                if count_observer is None:
                    from .lean_count_observation import observe_evaluation

                    count_observer = observe_evaluation
                try:
                    record["count_observation"] = count_observer(
                        record,
                        runtime=count_runtime,
                        target_experiment=descriptor,
                        package=package,
                        directory=root / "counts",
                        floor_cycles=member.get("floor_cycles"),
                        target_fraction=member.get("target_fraction", 0.3),
                        remaining_s=_remaining(deadline),
                    )
                except Exception as error:
                    record["count_observation"] = dict(
                        status="UNKNOWN", detail=str(error), qualification="UNQUALIFIED", source_effects="PENDING"
                    )
        if _tree(package) != record["package_identity"]:
            raise ValueError("retained compiler changed during lean evaluation")
        if member_identity(member) != record["input_identity"]:
            raise ValueError("saved source, input or reference changed during lean evaluation")
    except Exception as error:
        record.update(
            status="budget_exhausted" if isinstance(error, TimeoutError) else "error",
            detail=f"{type(error).__name__}: {error}",
            complete_outputs_checked=False,
            failure_category="deadline" if isinstance(error, TimeoutError) else "evaluation_error",
            public_failure_summary=(
                "The evaluation exhausted its remaining time budget."
                if isinstance(error, TimeoutError)
                else "The numerical evaluation could not complete; inspect the local compiler self-check."
            ),
        )
    record["wall_seconds"] = time.monotonic() - started
    checkpoint("finished")
    return record


def _reopen_cached(record):
    if record.get("status") != "numeric_pass" or record.get("complete_outputs_checked") is not True:
        return False
    for role in ("native_result", "command_buffer", "lowered_mlir"):
        if record.get(role) != _pin(record[role]["path"]):
            return False
    native = json.loads(Path(record["native_result"]["path"]).read_bytes())
    cb = json.loads(Path(record["command_buffer"]["path"]).read_bytes())
    if not _complete_numeric(native, cb) or not _native_compile_join(native, cb, record["lowered_mlir"]):
        return False
    elf = native.get("elf")
    return elf is not None and elf == _pin(elf["path"])


def evaluate_candidate(
    *,
    candidate_source,
    baseline_source,
    selected_members,
    work_root,
    timeout_s,
    baseline_cache_root,
    count_runtime=None,
    count_identity=None,
    count_observer=None,
):
    """Paired lean callback; unchanged controls are cached by all exact inputs."""
    deadline = time.monotonic() + float(timeout_s)
    root, cache = Path(work_root), Path(baseline_cache_root)
    root.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    if count_runtime is not None and not count_identity:
        raise ValueError("counted baseline cache requires explicit frozen model/runtime identity")
    rows = []
    progress = dict(
        schema=SCHEMA,
        status="running",
        qualification="UNQUALIFIED",
        source_effects="PENDING",
        members=[
            dict(id=member["id"], baseline=dict(phase="pending"), candidate=dict(phase="pending"))
            for member in selected_members
        ],
    )

    def checkpoint(index, arm, record):
        counts = record.get("count_observation") or {}
        progress["members"][index][arm] = dict(
            phase=record.get("phase", "pending"),
            status=record.get("status", "incomplete"),
            complete_outputs_checked=record.get("complete_outputs_checked") is True,
            count_status=counts.get("status", "PENDING" if count_runtime is not None else "not_requested"),
        )
        _write_json(root / "progress.json", progress)

    _write_json(root / "progress.json", progress)
    for member in selected_members:
        index = len(rows)
        identity = dict(source=member_identity(member), baseline=_tree(baseline_source), counts=count_identity)
        checkpoint(index, "baseline", dict(phase="waiting_for_control"))
        key = _hash(identity)
        lock_path = cache / (key + ".lock")
        with lock_path.open("a+b") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    _remaining(deadline)
                    time.sleep(0.05)
            cached = cache / (key + ".json")
            try:
                baseline = json.loads(cached.read_bytes()) if cached.is_file() else None
            except (OSError, ValueError):
                baseline = None
            hit = False
            try:
                hit = (
                    baseline is not None
                    and baseline.get("cache_identity") == identity
                    and baseline.get("input_identity") == identity["source"]
                    and baseline.get("package_identity") == identity["baseline"]
                    and _reopen_cached(baseline)
                )
            except (OSError, KeyError, ValueError):
                pass
            if not hit:
                destination = cache / key
                if destination.exists():
                    destination = cache / (key + "-retry-" + str(time.monotonic_ns()))
                baseline = evaluate_member(
                    member=member,
                    candidate_source=baseline_source,
                    work_root=destination,
                    timeout_s=_remaining(deadline),
                    count_runtime=count_runtime,
                    count_observer=count_observer,
                    progress_callback=lambda record: checkpoint(index, "baseline", record),
                )
                baseline["cache_identity"] = identity
                (destination / "evaluation.json").write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
                temporary = cached.with_suffix(".new")
                temporary.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
                temporary.replace(cached)
            baseline = dict(baseline, baseline_cache_hit=hit)
        checkpoint(index, "baseline", baseline)
        candidate_tree = _tree(candidate_source)
        if (
            baseline.get("package_identity") == identity["baseline"]
            and _source_bytes(candidate_tree) == _source_bytes(identity["baseline"])
            and baseline.get("input_identity") == member_identity(member)
            and _reopen_cached(baseline)
        ):
            candidate = copy.deepcopy(baseline)
            candidate.update(
                candidate_source=str(Path(candidate_source).resolve(strict=True)),
                package_identity=candidate_tree,
                observation_origin="identical_source_control_reuse",
                reused_control_evaluation=str(Path(baseline["work_root"]) / "evaluation.json"),
            )
        else:
            candidate = evaluate_member(
                member=member,
                candidate_source=candidate_source,
                work_root=root / ("member-" + str(index)),
                timeout_s=_remaining(deadline),
                count_runtime=count_runtime,
                count_observer=count_observer,
                progress_callback=lambda record: checkpoint(index, "candidate", record),
            )
        checkpoint(index, "candidate", candidate)
        rows.append(dict(id=member["id"], baseline=baseline, candidate=candidate))
    result = dict(
        schema=SCHEMA,
        qualification="UNQUALIFIED",
        source_effects="PENDING",
        members=rows,
        all_complete_outputs_pass=bool(rows)
        and all(row[arm]["complete_outputs_checked"] for row in rows for arm in ("baseline", "candidate")),
    )
    progress["status"] = "finished"
    _write_json(root / "progress.json", progress)
    (root / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
