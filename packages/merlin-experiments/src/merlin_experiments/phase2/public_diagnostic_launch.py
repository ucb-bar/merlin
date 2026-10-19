"""Explicit original-host launch of an unscored public retained-queue proof.

The plan selects source-pinned original admission, not saved capability verdicts.
This entry never issues formal Phase2 admission or a performance decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

from merlin.common.cpu_pool import current_thread_owns_cpu_lease, shared_compute_pool
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, restore_original_factory

from .contracts import StageGateError, write_json
from .retained_revisions import source_inventory

SCHEMA = "merlin.public_diagnostic_launch.v1"
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
    if (
        clock.get("schema") != "phase2.public_diagnostic_infrastructure_packet.v1"
        or clock.get("performance_campaign_restarted") is not False
        or not 0 < clock["deadline_unix"] - clock["start_unix"] <= 1800
        or type(clock.get("cleanup_reserve_seconds")) is not int
        or clock["cleanup_reserve_seconds"] < 480
        or clock["target_admission_cutoff_unix"] > clock["deadline_unix"] - clock["cleanup_reserve_seconds"]
        or clock["total_cpu_seconds"] > 600
        or clock["peak_rss_bytes"] > 1 << 30
        or clock["per_file_bytes"] > 64 << 20
        or clock["total_artifact_bytes"] > 768 << 20
        or time.time() >= clock["target_admission_cutoff_unix"]
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


def run_plan(plan, clock, *, output, completed=None, native_exploration=None, external_boundary=None):
    """Run under an optional original live envelope; never reconstruct one.

    The envelope must already enforce its declared kernel resources. It does
    not prove cumulative CPU-time or whole-call wall limits: the original
    enforcing operator remains required, including for setup. ``completed``
    receives the actual live pool/job, never restored JSON success.
    """
    from .evaluation_pool import AssignedResources
    from .external_resources import executing_resource_boundary

    scope = executing_resource_boundary(external_boundary) if external_boundary is not None else nullcontext()
    with scope:
        if current_thread_owns_cpu_lease():
            raise StageGateError("diagnostic launch coordinator cannot hold a child CPU lease")
        resources = AssignedResources(
            tuple(plan["cpu_ids"]),
            1 << 30,
            768 << 20,
            1,
            author_cores=0,
            reserve_fraction=0,
            require_hard_limits=external_boundary is not None,
        )
        resources.verify(external_boundary=external_boundary)
        if external_boundary is not None:
            external_boundary.verify(writable_roots=(Path(output).absolute(),))
        return _execute_plan(
            plan,
            clock,
            output=output,
            completed=completed,
            native_exploration=native_exploration,
            resources=resources,
            external_boundary=external_boundary,
        )


def _create_pool(*, policy, revisions, resources, external_boundary):
    from .evaluation_pool import ComponentEvaluationPool, JobResources

    return ComponentEvaluationPool(
        policy=policy,
        revisions=revisions,
        resources=resources,
        job_resources=JobResources(1, 1 << 30, 64 << 20, 1, files=16384),
        root=policy.feedback_evaluator.work_root / "queue",
        beam=1,
        max_pending=1,
        max_jobs=1,
        external_boundary=external_boundary,
    )


def _execute_plan(plan, clock, *, output, completed, native_exploration, resources, external_boundary):
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
            candidate, baseline = output / "candidate", output / "baseline"
            shutil.copytree(plan["compiler"], candidate)
            campaign.materialize_readonly_tree(candidate, baseline)
            if _inventory(plan["compiler"]) != original or _inventory(candidate) != original:
                raise StageGateError("immutable compiler changed during preparation")
            edit = json.loads(read_pin(plan["edit_contract"]))
            validate_against_package(edit, candidate)
            authority_root = output / "edit-authority"
            authority_root.mkdir()
            authority = FrozenEditAuthority(authority_root, guidance_contract=output / "contract")
            authority.freeze(candidate, edit, has_iterations=False)
            library = restore_selection(json.loads(read_pin(plan["library_selection"]))["selection"])
            factory = restore_original_factory(plan["factory"])
        stages.append({"stage": "original_preflight", "wall_seconds": time.monotonic() - started})
        report("fresh_original_current_source_admission")
        tick = time.monotonic()
        registry = factory.create(work_root=output / "source-admission")
        with shared_compute_pool(timeout_s=min(120, remaining())):
            # Only the diagnostic source capability is requested. No schedule or timing grant.
            capability = registry._capabilities[0]
            passes = CompilerPassSelection(
                registry, capability.binding, mechanisms=("source_execution_diagnostic",), factory=factory
            )
            checked = select_checked_source_diagnostic(
                passes,
                target=target.target,
                rtl_facts=Path(factory.inputs["facts"]),
                board_config=plan["board_config"],
            )
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
            if completed is not None:
                completed(pool, job, policy)
            report("completed_unscored_same_revision_queue", decision="REJECT performance promotion; timing UNKNOWN")
            return state
        finally:
            pool.close()
    except BaseException as error:
        report("precise_diagnostic_refusal", failure={"type": type(error).__name__, "detail": str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path)
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
    run_plan(plan, clock, output=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
