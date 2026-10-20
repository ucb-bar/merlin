"""Explicit original-host launch of an unscored public retained-queue proof.

The plan selects source-pinned original admission, not saved capability verdicts.
This entry never issues formal Phase2 admission or a performance decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

from merlin.common.cpu_pool import current_thread_owns_cpu_lease, shared_compute_pool
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, restore_original_factory

from .contracts import StageGateError, write_json
from .retained_revisions import source_inventory

SCHEMA = "merlin.public_diagnostic_launch.v1"
NATIVE_CLOCK_SCHEMA = "phase2.public_native_diagnostic_infrastructure_packet.v2"
FIELDS = {
    "schema",
    "sources",
    "operator_exclusion",
    "planning_inventory",
    "compiler",
    "compiler_inventory",
    "edit_contract",
    "contract_root",
    "target_experiment",
    "source_root",
    "profile",
    "library_selection",
    "factory",
    "clock",
    "cpu_ids",
    "board_config",
}


def read_pin(row, *, max_bytes=2 << 20):
    if type(row) is not dict or set(row) != {"path", "sha256"}:
        raise StageGateError("diagnostic launch requires an exact ordinary leaf pin")
    path = Path(row["path"])
    if not path.is_absolute() or path.resolve() != path or path.is_symlink() or not path.is_file():
        raise StageGateError("diagnostic launch input is not an ordinary canonical leaf")
    if path.stat().st_size > max_bytes:
        raise StageGateError("diagnostic launch metadata exceeds its byte ceiling")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise StageGateError("diagnostic launch input differs from its selected bytes")
    return payload


def require_current_native_contract(target, expected=None):
    """Check an explicit operator pin against the current native contract selection.

    This read-only setup check neither selects a contract nor issues execution or
    qualification authority. Call it after setting the operator environment and
    before constructing native owners or compiling. With no expected pin, the
    existing selection behavior is unchanged. Later producers must still perform
    their original contract, recipe and source checks.
    """
    if expected is None:
        return None
    if (
        type(target) is not str
        or not target
        or type(expected) is not dict
        or set(expected) != {"path", "sha256"}
        or type(expected["path"]) is not str
        or type(expected["sha256"]) is not str
        or len(expected["sha256"]) != 64
        or any(char not in "0123456789abcdef" for char in expected["sha256"])
    ):
        raise StageGateError("native contract preflight requires an exact target and ordinary leaf pin")
    pin = dict(expected)
    before = read_pin(pin)
    from merlin.targetgen.target_registry import resolve

    selected = resolve(target).capability_contract_path
    if str(selected) != pin["path"]:
        raise StageGateError("current native contract path differs from the operator's expected contract")
    if read_pin(pin) != before:
        raise StageGateError("current native contract changed during setup preflight")
    return pin


def read_plan(path, sha256):
    plan = json.loads(read_pin({"path": str(Path(path).absolute()), "sha256": sha256}))
    if type(plan) is not dict or set(plan) != FIELDS or plan["schema"] != SCHEMA:
        raise StageGateError("unknown public diagnostic launch plan")
    if (
        type(plan["sources"]) is not list
        or not 1 <= len(plan["sources"]) <= 32
        or type(plan["cpu_ids"]) is not list
        or not 1 <= len(plan["cpu_ids"]) <= 256
        or any(type(cpu) is not int or cpu < 0 for cpu in plan["cpu_ids"])
        or len(set(plan["cpu_ids"])) != len(plan["cpu_ids"])
        or type(plan["board_config"]) is not str
        or not 1 <= len(plan["board_config"]) <= 128
    ):
        raise StageGateError("diagnostic launch requires bounded public sources and an explicit unique CPU assignment")
    for field in (
        "operator_exclusion",
        "planning_inventory",
        "edit_contract",
        "target_experiment",
        "library_selection",
        "clock",
    ):
        read_pin(plan[field], max_bytes=8 << 20 if field == "planning_inventory" else 2 << 20)
    if type(plan["profile"]) is not dict or set(plan["profile"]) != {
        "archive",
        "workspace",
        "repo",
        "profile_sha256",
        "snapshot_sha256",
    }:
        raise StageGateError("diagnostic launch requires the original selected profile identity")
    read_pin(plan["profile"]["archive"])
    clock = json.loads(read_pin(plan["clock"]))
    cpu_limit = 1200 if type(clock) is dict and clock.get("schema") == NATIVE_CLOCK_SCHEMA else 600
    timestamps = ("start_unix", "deadline_unix", "target_admission_cutoff_unix")
    byte_limits = {
        "peak_rss_bytes": 1 << 30,
        "per_file_bytes": 64 << 20,
        "total_artifact_bytes": 768 << 20,
    }
    if (
        type(clock) is not dict
        or any(type(clock.get(key)) not in (int, float) or not math.isfinite(clock[key]) for key in timestamps)
        or type(clock.get("total_cpu_seconds")) not in (int, float)
        or not math.isfinite(clock["total_cpu_seconds"])
        or not 0 < clock["total_cpu_seconds"] <= cpu_limit
        or any(type(clock.get(key)) is not int or not 0 < clock[key] <= limit for key, limit in byte_limits.items())
        or clock.get("schema") not in {"phase2.public_diagnostic_infrastructure_packet.v1", NATIVE_CLOCK_SCHEMA}
        or clock.get("performance_campaign_restarted") is not False
        or not 0 < clock["deadline_unix"] - clock["start_unix"] <= 1800
        or type(clock.get("cleanup_reserve_seconds")) is not int
        or clock["cleanup_reserve_seconds"] < 480
        or not clock["start_unix"]
        <= time.time()
        < clock["target_admission_cutoff_unix"]
        <= clock["deadline_unix"] - clock["cleanup_reserve_seconds"]
    ):
        raise StageGateError("diagnostic infrastructure packet is expired or exceeds its fixed limits")
    return plan, clock


def _inventory(root):
    return json.loads(json.dumps(source_inventory(Path(root), max_bytes=32 << 20, max_files=8192)))


def project_contract(source, destination):
    """Copy exact ordinary contract leaves; examples/capsules are not selected."""
    source, destination = Path(source), Path(destination)
    destination.mkdir()
    records = []
    for path in sorted(source.iterdir()):
        if path.name in {"examples", "capsules"}:
            continue
        if path.is_symlink():
            raise StageGateError("selected contract contains a linked entry")
        target = destination / path.name
        if path.is_dir():
            before = _inventory(path)
            shutil.copytree(path, target)
            if before != _inventory(path) or before != _inventory(target):
                raise StageGateError("selected contract directory changed during projection")
        else:
            if not path.is_file():
                raise StageGateError("selected contract contains a special entry")
            before = path.read_bytes()
            shutil.copyfile(path, target)
            if before != path.read_bytes() or before != target.read_bytes():
                raise StageGateError("selected contract leaf changed during projection")
            before = hashlib.sha256(before).hexdigest()
        records.append({"path": str(path), "identity": before})
    for path in reversed((destination, *destination.rglob("*"))):
        path.chmod(path.stat().st_mode & ~0o222)
    _inventory(destination)
    return records


def _wait_for_job(pool, job, *, output, report, remaining):
    """Publish the actual current terminal state before any refusal escapes."""
    while True:
        status = pool.status(job["job_id"])
        write_json(output / "queue-status.json", status)
        report("queued_original_retained_pair", queue_state=status["state"])
        if status["terminal"]:
            break
        remaining()
        time.sleep(2)
    write_json(output / "queue-observations.json", pool.observations())
    return status


def run_plan(
    plan,
    clock,
    *,
    output,
    completed=None,
    native_exploration=None,
    external_boundary=None,
    candidate_source=None,
    compiler_pass_selection=None,
    compiler_pass_preparation=None,
    native_decision_policy=None,
    native_execution_protocol=None,
    diagnostic_pair_protocol=None,
):
    """Run under an optional original live envelope; never reconstruct one.

    The envelope must already enforce its declared kernel resources. It does
    not prove cumulative CPU-time or whole-call wall limits: the original
    enforcing operator remains required, including for setup. ``completed``
    receives the actual live pool/job, never restored JSON success.

    ``candidate_source`` is an explicit original-host source directory, not a
    V1 plan field or saved owner. Its bounded bytes must pass the same frozen
    edit authority and retained-source issuer as the initial control revision.

    Explicit live pass preparation runs on both independent copies before edit
    authority is frozen. It is not a V1 plan field, a saved owner, or scheduling
    qualification. Both seeds receive the same original host-selected policy.
    """
    from .evaluation_pool import AssignedResources
    from .external_resources import executing_resource_boundary

    scope = executing_resource_boundary(external_boundary) if external_boundary is not None else nullcontext()
    if native_execution_protocol is not None:
        from merlin.targetgen.checked_source_model_diagnostic import POINTER_PROTOCOL

        if type(native_execution_protocol) is not str or native_execution_protocol != POINTER_PROTOCOL:
            raise StageGateError("native execution requires its exact explicit pointer protocol")
    if native_decision_policy is not None:
        from .native_development_decision import POLICY

        if type(native_decision_policy) is not str or native_decision_policy != POLICY or native_exploration is None:
            raise StageGateError("native decision requires its explicit policy and original live evaluator")
    if diagnostic_pair_protocol is not None:
        from .diagnostic_pair_check import PROTOCOL

        if (
            type(diagnostic_pair_protocol) is not str
            or diagnostic_pair_protocol != PROTOCOL
            or compiler_pass_selection is None
            or compiler_pass_preparation is None
        ):
            raise StageGateError(
                "diagnostic pair checks require the explicit protocol and original prepared treatments"
            )
    with scope:
        if current_thread_owns_cpu_lease():
            raise StageGateError("diagnostic launch coordinator cannot hold a child CPU lease")
        if clock.get("schema") == NATIVE_CLOCK_SCHEMA:
            _verify_native_clock_owner(native_exploration, clock)
        resources = AssignedResources(
            tuple(plan["cpu_ids"]),
            clock["peak_rss_bytes"],
            clock["total_artifact_bytes"],
            1,
            author_cores=0,
            reserve_fraction=0,
            require_hard_limits=external_boundary is not None,
        )
        resources.verify(external_boundary=external_boundary)
        if external_boundary is not None:
            external_boundary.verify(writable_roots=(Path(output).absolute(),))
        if compiler_pass_selection is not None or compiler_pass_preparation is not None:
            with shared_compute_pool(timeout_s=min(120, max(0, clock["deadline_unix"] - time.time()))):
                _validate_preparation(compiler_pass_selection, compiler_pass_preparation)
        return _execute_plan(
            plan,
            clock,
            output=output,
            completed=completed,
            native_exploration=native_exploration,
            resources=resources,
            external_boundary=external_boundary,
            candidate_source=candidate_source,
            compiler_pass_selection=compiler_pass_selection,
            compiler_pass_preparation=compiler_pass_preparation,
            native_decision_policy=native_decision_policy,
            native_execution_protocol=native_execution_protocol,
            diagnostic_pair_protocol=diagnostic_pair_protocol,
        )


def _verify_native_clock_owner(native_exploration, clock):
    """The larger workflow envelope requires a freshly verified live evaluator.

    This changes setup/metadata CPU allowance only. Queue-job timeouts, program
    verification ceilings, memory/file limits and numerical checks stay fixed.
    The original process supervisor still enforces aggregate workflow resources.
    """
    from .public_native_exploration import PublicNativeExploration

    if type(native_exploration) is not PublicNativeExploration:
        raise StageGateError("native diagnostic clock requires its original live evaluator")
    with shared_compute_pool(timeout_s=min(120, max(0, clock["deadline_unix"] - time.time()))):
        native_exploration.verify()


def _validate_preparation(selection, preparation):
    from .compiler_pass_preparation import CompilerPassPreparation

    if selection is None and preparation is None:
        return None
    if type(selection) is not CompilerPassSelection or type(preparation) is not CompilerPassPreparation:
        raise StageGateError("diagnostic preparation requires the original live selection and typed host policies")
    choices = preparation.record()
    if (
        selection.factory is None
        or selection.mechanisms != (choices["mechanism"],)
        or choices["baseline_policy"] != choices["candidate_policy"]
    ):
        raise StageGateError("diagnostic preparation requires one original factory and equal seed policies")
    selection.revalidate()
    capability = selection.resolve(choices["mechanism"])
    if capability is None:
        raise StageGateError("diagnostic preparation lacks its exact original live capability")
    owner = capability.owner
    for name in ("prepare_compiler", "check_compiler_binding"):
        callback = getattr(owner, name, None)
        if type(callback) is not MethodType or callback.__self__ is not owner:
            raise StageGateError("diagnostic preparation requires original bound preparation/checker methods")
    return choices


def _prepare_diagnostic_sources(plan, *, output, selection, preparation, library, factory):
    """Use the existing original preparer; keep raw parent and compared bytes distinct."""
    from merlin.targetgen.compiler_pass_selection import copy_compiler_source

    from . import campaign
    from .compiler_pass_preparation import prepare_pair, verify_pair

    _validate_preparation(selection, preparation)
    if selection.binding.library != library.contract.sha256 or selection.factory.record() != factory.record():
        raise StageGateError("diagnostic preparation changed its original factory or selected library")
    original = Path(output) / "original-compiler"
    raw_sha = campaign.materialize_readonly_tree(Path(plan["compiler"]), original)
    record = prepare_pair(
        selection,
        preparation,
        original=original,
        functional_sha256=raw_sha,
        root=Path(output) / "compiler-preparation",
    )
    candidate, baseline = Path(output) / "candidate", Path(output) / "baseline"
    copy_compiler_source(Path(record["candidate_seed"]["path"]), candidate)
    comparison_sha = campaign.materialize_readonly_tree(Path(record["baseline"]["path"]), baseline)
    if (
        comparison_sha != record["baseline"]["sha256"]
        or _inventory(candidate) != _inventory(baseline)
        or _inventory(plan["compiler"]) != plan["compiler_inventory"]
    ):
        raise StageGateError("diagnostic preparation changed its raw parent or equal comparison seeds")
    verify_pair(record, selection, functional_sha256=raw_sha)
    return candidate, baseline, record


def _retain_candidate_source(revisions, initial, source, expected_inventory):
    """Retain actual selected bytes under existing authority, with fresh custody."""
    source = Path(source)
    if _inventory(source) != expected_inventory:
        raise StageGateError("selected diagnostic candidate source changed after preflight")
    revision = revisions.freeze(
        source,
        parent_revision=initial.revision_id,
        options={
            "hypothesis": "explicit host-selected candidate under the original frozen edit authority",
            "edit_family": "host-selected-source-candidate",
        },
    )
    if _inventory(source) != expected_inventory or _inventory(revision.source) != expected_inventory:
        raise StageGateError("selected diagnostic candidate source changed during retention")
    revision.verify(authority=revisions.authority)
    return revision, {
        "schema": "merlin.public_diagnostic_retained_candidate_selection.v1",
        "source": str(source),
        "selected_inventory": expected_inventory,
        "initial_revision": initial.public_record(),
        "submitted_revision": revision.public_record(),
        "distinct_source": revision.source_sha256 != initial.source_sha256,
        "scope": "source custody and existing edit authority only; execution and qualification remain independent",
    }


def _create_pool(*, policy, revisions, resources, external_boundary):
    from .evaluation_pool import ComponentEvaluationPool, JobResources

    return ComponentEvaluationPool(
        policy=policy,
        revisions=revisions,
        resources=resources,
        job_resources=JobResources(1, resources.ram_bytes, min(64 << 20, resources.disk_bytes), 1, files=16384),
        root=policy.feedback_evaluator.work_root / "queue",
        beam=1,
        max_pending=1,
        max_jobs=1,
        external_boundary=external_boundary,
    )


def _execute_plan(
    plan,
    clock,
    *,
    output,
    completed,
    native_exploration,
    resources,
    external_boundary,
    candidate_source=None,
    compiler_pass_selection=None,
    compiler_pass_preparation=None,
    native_decision_policy=None,
    native_execution_protocol=None,
    diagnostic_pair_protocol=None,
):
    from merlin.perf.phase2_edit_contract import validate_against_package
    from merlin.targetgen.checked_source_model_diagnostic import select_checked_source_diagnostic
    from merlin.targetgen.common_transforms import CommonTransformService
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.phase0 import operator_exclusion as E

    from . import campaign
    from .compiler_library_state import restore_selection
    from .edit_authority import FrozenEditAuthority
    from .public_diagnostic_corpus import freeze_public_diagnostic_corpus
    from .public_source_diagnostic import prepare_public_source_diagnostic
    from .retained_revisions import RetainedRevisionStore

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output or any(p.is_symlink() for p in output.parents):
        raise StageGateError("diagnostic launch needs a fresh canonical output directory")
    output.mkdir(parents=True)
    started = time.monotonic()
    stages = []
    state = {
        "schema": "merlin.public_diagnostic_launch_result.v1",
        "formal_phase2_launched": False,
        "promotion": "DIAGNOSTIC_ONLY",
        "timing_cycles": None,
        "timing": "UNKNOWN",
        "queue_submitted": 0,
        "stages": stages,
    }

    def report(stage, **values):
        state.update(stage=stage, elapsed_seconds=time.monotonic() - started, **values)
        write_json(output / "status.json", state)

    def remaining():
        value = clock["deadline_unix"] - time.time() - clock["cleanup_reserve_seconds"]
        if value <= 0:
            raise StageGateError("diagnostic packet verification reserve reached")
        return value

    report("original_public_source_profile_preflight")
    try:
        with shared_compute_pool(timeout_s=min(120, remaining())):
            original = _inventory(plan["compiler"])
            if original != plan["compiler_inventory"]:
                raise StageGateError("original compiler differs from the selected immutable parent")
            candidate_inventory = _inventory(candidate_source) if candidate_source is not None else None
            read_pin(plan["target_experiment"])
            target = load_target_experiment(
                Path(plan["target_experiment"]["path"]), source_root=Path(plan["source_root"])
            )
            exclusion = E.select(Path(plan["operator_exclusion"]["path"]), plan["operator_exclusion"]["sha256"])
            corpus = freeze_public_diagnostic_corpus(
                sources=tuple(Path(p) for p in plan["sources"]),
                destination=output / "smoke-corpus",
                target=target.target,
                operator_exclusion=exclusion,
                planning_inventory=Path(plan["planning_inventory"]["path"]),
                planning_inventory_sha256=plan["planning_inventory"]["sha256"],
            )
            write_json(output / "contract-origins.json", project_contract(plan["contract_root"], output / "contract"))
            selected = plan["profile"]
            read_pin(selected["archive"])
            service = CommonTransformService(
                workspace=Path(selected["workspace"]),
                bundle_manifest=Path(selected["archive"]["path"]),
                repo=Path(selected["repo"]),
            )
            service._verify()
            if (service.profile["sha256"], service.snapshot_digest) != (
                selected["profile_sha256"],
                selected["snapshot_sha256"],
            ):
                raise StageGateError("original compiler profile or snapshot changed")
            os.environ["MERLIN_TARGET_EXPERIMENT"] = str(target.path)
            library = restore_selection(json.loads(read_pin(plan["library_selection"]))["selection"])
            factory = restore_original_factory(plan["factory"])
            candidate, baseline = output / "candidate", output / "baseline"
            if compiler_pass_preparation is None:
                shutil.copytree(plan["compiler"], candidate)
                campaign.materialize_readonly_tree(candidate, baseline)
                if _inventory(plan["compiler"]) != original or _inventory(candidate) != original:
                    raise StageGateError("immutable compiler changed during preparation")
            else:
                candidate, baseline, prepared = _prepare_diagnostic_sources(
                    plan,
                    output=output,
                    selection=compiler_pass_selection,
                    preparation=compiler_pass_preparation,
                    library=library,
                    factory=factory,
                )
                state["compiler_preparation"] = prepared
                write_json(output / "compiler-preparation-selection.json", prepared)
            edit = json.loads(read_pin(plan["edit_contract"]))
            validate_against_package(edit, candidate)
            authority_root = output / "edit-authority"
            authority_root.mkdir()
            authority = FrozenEditAuthority(authority_root, guidance_contract=output / "contract")
            authority.freeze(candidate, edit, has_iterations=False)
        stages.append({"stage": "original_preflight", "wall_seconds": time.monotonic() - started})
        report("fresh_original_current_source_admission")
        tick = time.monotonic()
        registry = (
            factory.create(work_root=output / "source-admission")
            if compiler_pass_selection is None
            else compiler_pass_selection.registry
        )
        with shared_compute_pool(timeout_s=min(120, remaining())):
            # Only the diagnostic source capability is requested. No schedule or timing grant.
            if compiler_pass_selection is not None:
                _validate_preparation(compiler_pass_selection, compiler_pass_preparation)
            sources = [cap for cap in registry._capabilities if cap.mechanism == "source_execution_diagnostic"]
            if len(sources) != 1:
                raise StageGateError("diagnostic launch requires one exact original source execution owner")
            capability = sources[0]
            if compiler_pass_selection is not None and capability.binding != compiler_pass_selection.binding:
                raise StageGateError("diagnostic preparation and source execution bindings differ")
            passes = CompilerPassSelection(
                registry, capability.binding, mechanisms=("source_execution_diagnostic",), factory=factory
            )
            checked = select_checked_source_diagnostic(
                passes,
                target=target.target,
                rtl_facts=Path(factory.inputs["facts"]),
                board_config=plan["board_config"],
                native_execution_protocol=native_execution_protocol,
            )
            pair_check = None
            if diagnostic_pair_protocol is not None:
                from .diagnostic_pair_check import select_source_pair_check

                pair_check = select_source_pair_check(compiler_pass_selection, passes)
            policy = prepare_public_source_diagnostic(
                candidate=candidate,
                baseline=baseline,
                corpus=corpus,
                target_experiment=target,
                edit_authority=authority,
                service=service,
                compiler_library_selection=library,
                compiler_pass_selection=passes,
                checked_source_diagnostic=checked,
                contract_root=output / "contract",
                work_root=output / "execution",
                **({"diagnostic_pair_check": pair_check} if pair_check is not None else {}),
                **({"native_exploration": native_exploration} if native_exploration is not None else {}),
            )
            policy.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(policy.receipt_path, policy.binding())
            revisions = RetainedRevisionStore(
                output / "revisions",
                edit_authority=authority,
                max_bytes=32 << 20,
                max_files=8192,
                max_total_bytes=64 << 20,
            )
            revision = revisions.freeze(
                candidate,
                options={"hypothesis": "same original revision both arms", "edit_family": "unchanged-source-control"},
            )
            if candidate_source is not None:
                revision, candidate_selection = _retain_candidate_source(
                    revisions, revision, candidate_source, candidate_inventory
                )
                write_json(output / "candidate-source-selection.json", candidate_selection)
                state["candidate_source_selection"] = candidate_selection
        stages.append({"stage": "original_live_admission", "wall_seconds": time.monotonic() - tick})
        if current_thread_owns_cpu_lease():
            raise StageGateError("queue coordinator retained its metadata CPU grant")
        remaining()
        pool = _create_pool(
            policy=policy,
            revisions=revisions,
            resources=resources,
            external_boundary=external_boundary,
        )
        try:
            # Compute member selection before submit; no outer metadata leaf around either.
            members = tuple(sorted(policy.validate_search_inputs()))
            job = pool.submit(revision.revision_id, timeout_s=min(600, remaining()), selected_members=members)
            report(
                "queued_original_retained_pair",
                queue_submitted=1,
                job_id=job["job_id"],
                job_root=str(pool.root / job["job_id"]),
            )
            status = _wait_for_job(pool, job, output=output, report=report, remaining=remaining)
            if status["state"] != "completed":
                raise StageGateError("actual retained queue refused: " + str(status.get("refusal")))
            result = status["result"]
            if result["promotion"] != "DIAGNOSTIC_ONLY" or result["screening"]["status"] != "UNKNOWN":
                raise StageGateError("diagnostic retained feedback changed its permanent unscored scope")
            write_json(output / "diagnostic-feedback.json", result)
            if native_decision_policy is not None:
                from . import ordinary_fast_feedback as FAST
                from .native_development_decision import decide

                native_exploration.verify_for(policy)
                FAST.validate_feedback(result, policy=policy, revision=revision, kind=FAST.SELECTED_KIND)
                decision = decide(result, owner=native_exploration, required_members=members)
                write_json(output / "native-development-decision.json", decision)
                state["native_development_decision"] = decision
            if completed is not None:
                completed(pool, job, policy)
            report(
                "completed_unscored_same_revision_queue"
                if candidate_source is None
                else "completed_unscored_retained_candidate_queue",
                decision="REJECT performance promotion; timing UNKNOWN",
            )
            return state
        finally:
            pool.close()
    except BaseException as error:
        report("precise_diagnostic_refusal", failure={"type": type(error).__name__, "detail": str(error)})
        raise


def main(argv=None, *, native_exploration=None, external_boundary=None):
    from merlin.targetgen.checked_source_model_diagnostic import POINTER_PROTOCOL

    from . import compiler_pass_state as PASSES
    from .diagnostic_pair_check import PROTOCOL as PAIR_PROTOCOL
    from .native_development_decision import POLICY as DECISION_POLICY

    parser = argparse.ArgumentParser(description=__doc__)
    PASSES.add_selection_arguments(parser)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidate-source", type=Path, help="explicit retained candidate under frozen edit authority")
    parser.add_argument("--native-execution-protocol", choices=(POINTER_PROTOCOL,))
    parser.add_argument("--diagnostic-pair-protocol", choices=(PAIR_PROTOCOL,))
    parser.add_argument("--native-decision-policy", choices=(DECISION_POLICY,))
    parser.add_argument(
        "--check-plan",
        action="store_true",
        help="validate selected plan metadata and clock only; never execute admission or a queue",
    )
    args = parser.parse_args(argv)
    if args.check_plan and args.output is not None:
        parser.error("--output belongs to execution; --check-plan writes metadata to stdout only")
    if not args.check_plan and args.output is None:
        parser.error("execution requires --output")
    extra = any(
        value is not None
        for value in (
            args.compiler_pass_selection,
            args.compiler_pass_selection_sha256,
            args.compiler_pass_admission_root,
            args.compiler_pass_preparation,
            args.compiler_pass_preparation_sha256,
            args.candidate_source,
            args.native_execution_protocol,
            args.diagnostic_pair_protocol,
            args.native_decision_policy,
        )
    )
    if args.check_plan and extra:
        parser.error(
            "--check-plan checks only V1 plan metadata; execution selections require a separate bounded launch"
        )
    plan, clock = read_plan(args.plan, args.plan_sha256)
    if args.check_plan:
        now = time.time()
        print(
            json.dumps(
                {
                    "schema": "merlin.public_diagnostic_plan_check.v1",
                    "status": "NOT_EXECUTED",
                    "scope": "selected plan schema, exact declared metadata leaf pins and existing packet clock only",
                    "authority": "NONE",
                    "plan": {"path": str(args.plan.absolute()), "sha256": args.plan_sha256},
                    "selected": {
                        "public_sources": plan["sources"],
                        "compiler": plan["compiler"],
                        "target_experiment": plan["target_experiment"],
                        "source_root": plan["source_root"],
                        "profile_archive": plan["profile"]["archive"],
                        "profile_sha256": plan["profile"]["profile_sha256"],
                        "snapshot_sha256": plan["profile"]["snapshot_sha256"],
                        "library_selection": plan["library_selection"],
                        "cpu_ids": plan["cpu_ids"],
                        "board_config": plan["board_config"],
                    },
                    "clock": {
                        "deadline_unix": clock["deadline_unix"],
                        "admission_remaining_seconds": max(0, clock["target_admission_cutoff_unix"] - now),
                        "packet_remaining_seconds": max(0, clock["deadline_unix"] - now),
                    },
                    "not_checked": [
                        "public source/reference/private exclusion admission",
                        "compiler inventory/edit authority",
                        "genuine profile verification",
                        "original factory reconstruction/source closure/admission",
                        "actual resource admission",
                        "queue",
                        "numerical/effect correctness",
                        "timing",
                    ],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    kwargs = {}
    if native_exploration is not None:
        kwargs["native_exploration"] = native_exploration
    if external_boundary is not None:
        kwargs["external_boundary"] = external_boundary
    for name in ("candidate_source", "native_execution_protocol", "diagnostic_pair_protocol", "native_decision_policy"):
        if getattr(args, name) is not None:
            kwargs[name] = getattr(args, name)
    preparation = PASSES.preparation_from_arguments(args)
    if preparation is not None:
        # A JSON declaration pins an original factory; loading reruns it.
        # Never spend its admission cost when the selected clock lacks its live owner.
        if clock["schema"] == NATIVE_CLOCK_SCHEMA:
            _verify_native_clock_owner(native_exploration, clock)
        from .compiler_library_state import restore_selection

        library = restore_selection(json.loads(read_pin(plan["library_selection"]))["selection"])
        kwargs["compiler_pass_selection"] = PASSES.selection_from_arguments(
            args,
            library=library,
            stage_root=args.output,
            external_boundary=external_boundary,
        )
        kwargs["compiler_pass_preparation"] = preparation
    run_plan(plan, clock, output=args.output, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
