"""AWS-only costs and calibration on the original normal Phase 1 corpus route.

Planning is the default. Explicit execution uses fixed original admission,
compile/Spike and certified RTL evaluators, never request-supplied commands.
Repeated fresh processes do not prove cold/warm hardware cache state.
"""

from __future__ import annotations

import argparse
import os
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from merlin.common.digest import is_sha256
from merlin.common.strict_json import loads
from merlin.perf.component_screen import ComponentScreenPolicy
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS

from . import ordinary_fast_feedback as FAST
from . import timing_study as COST
from .contracts import StageGateError, document_sha256, mapping_file, sha256_file, write_json

SCHEMA = "merlin.normal_aws_cpu_timing_study.v1"
_NORMAL = {
    "functional_runs_root",
    "functional_run_id",
    "functional_submission_sha256",
    "corpus_root",
    "corpus_manifest_sha256",
    "corpus_sha256",
    "baseline",
    "candidate",
    "target_descriptor",
    "contract_root",
    "edit_contract",
    "certificate",
    "rtl_facts",
}


def request_from_policy(
    policy, *, resources, alternatives, public_partition, edit_contract_path, replicates=2, max_stage_seconds=600
):
    """Export a reviewable data request from the actual original normal selection."""
    from merlin.benchharness import hash_tree

    from .ordinary_search_policy import OrdinarySearchPolicy

    if type(policy) is not OrdinarySearchPolicy:
        raise StageGateError("normal study request requires its original live Phase 1 policy")
    policy.validate_search_inputs()
    feedback, functional = policy.feedback_evaluator, policy._functional

    def pin(path):
        return {"path": str(path), "sha256": sha256_file(Path(path))}

    if mapping_file(Path(edit_contract_path)) != policy.edit_authority.contract:
        raise StageGateError("normal study edit contract differs from the original frozen authority")
    request = {
        "schema": SCHEMA,
        "normal_inputs": {
            "functional_runs_root": str(functional.run_dir.parent),
            "functional_run_id": functional.run_id,
            "functional_submission_sha256": feedback.baseline_sha256,
            "corpus_root": str(policy.search_corpus.root),
            "corpus_manifest_sha256": policy.search_corpus.manifest_sha256,
            "corpus_sha256": policy.search_corpus.capsules_sha256,
            "baseline": str(policy.search_baseline),
            "candidate": str(policy.candidate),
            "target_descriptor": pin(policy.target_experiment.path),
            "contract_root": str(policy._contract_root),
            "edit_contract": pin(edit_contract_path),
            "certificate": pin(feedback.certificate.path),
            "rtl_facts": pin(feedback.rtl_facts_path),
        },
        "resources": resources,
        "environment_sha256": document_sha256(dict(os.environ)),
        "replicates": replicates,
        "max_stage_seconds": max_stage_seconds,
        "public_partition": public_partition,
        "alternatives": [{"path": str(path), "sha256": str(hash_tree(path)["sha256"])} for path in alternatives],
    }
    OrdinaryCorpusCpuStudyAdapter(policy).verify(request)
    return request


def plan(request):
    """Validate a closed request without loading a runtime or executing any producer."""
    COST._fields(
        request,
        (
            "schema",
            "normal_inputs",
            "resources",
            "environment_sha256",
            "replicates",
            "max_stage_seconds",
            "public_partition",
            "alternatives",
        ),
        "normal CPU study",
    )
    if request["schema"] != SCHEMA or not is_sha256(request["environment_sha256"]):
        raise StageGateError("normal study needs an exact original environment identity")
    COST._fields(request["normal_inputs"], _NORMAL, "normal Phase 1 inputs")
    for key in ("target_descriptor", "edit_contract", "certificate", "rtl_facts"):
        row = request["normal_inputs"][key]
        COST._fields(row, ("path", "sha256"), key)
        if not is_sha256(row["sha256"]) or sha256_file(Path(row["path"])) != row["sha256"]:
            raise StageGateError("normal study selected input changed: " + key)
    resources = request["resources"]
    COST._fields(
        resources,
        ("provider", "instance_id", "cpu_ids", "memory_bytes", "disk_bytes", "work_root"),
        "allocated AWS resources",
    )
    if (
        resources["provider"] != "aws"
        or type(resources["instance_id"]) is not str
        or not resources["instance_id"].startswith("i-")
    ):
        raise StageGateError("normal timing study requires explicitly allocated AWS CPU")
    cpus = resources["cpu_ids"]
    if (
        type(cpus) is not list
        or not cpus
        or len(cpus) > 1024
        or len(set(cpus)) != len(cpus)
        or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
    ):
        raise StageGateError("normal study needs distinct assigned CPU identities")
    for key in ("memory_bytes", "disk_bytes"):
        COST._positive(resources[key], 1 << 60, key)
    root = Path(resources["work_root"])
    if not root.is_absolute() or root.resolve() != root or any(path.is_symlink() for path in (root, *root.parents)):
        raise StageGateError("normal study requires its canonical allocated output owner")
    COST._positive(request["replicates"], 100, "replicates")
    if request["replicates"] < 2:
        raise StageGateError("normal study requires repeated complete observations")
    COST._positive(request["max_stage_seconds"], int(ITERATION_MAX_SECONDS), "ordinary reduced deadline")
    alternatives = request["alternatives"]
    if type(alternatives) is not list or not 2 <= len(alternatives) <= 64:
        raise StageGateError("normal study requires multiple bounded public compiler alternatives")
    for row in alternatives:
        COST._fields(row, ("path", "sha256"), "public alternative")
        if not is_sha256(row["sha256"]):
            raise StageGateError("normal study compiler alternative lacks its source identity")
    return {
        "schema": SCHEMA,
        "request_sha256": document_sha256(request),
        "execution_performed": False,
        "route": "original Phase 1 functional admission, compile/Spike and certified development RTL",
        "stages": ["complete_compile_spike_features", "complete_certified_rtl", "calibration_and_held_validation"],
        "resources": resources,
        "screening_policy": ComponentScreenPolicy().to_dict(),
        "cache_state": "UNKNOWN",
        "whole_model_host_cost": "UNKNOWN",
        "promotion": "NONE",
        "break_even_rule": "N*((1-q)*S-E)>M",
    }


@dataclass(frozen=True)
class OrdinaryCorpusCpuStudyAdapter:
    policy: object

    def verify(self, request):
        from .ordinary_search_policy import OrdinarySearchPolicy

        planned = plan(request)
        if type(self.policy) is not OrdinarySearchPolicy:
            raise StageGateError("normal CPU study requires its original live Phase 1 policy")
        self.policy.validate_search_inputs()
        binding, selected = self.policy.binding(), request["normal_inputs"]
        if (
            binding["target_descriptor_sha256"] != selected["target_descriptor"]["sha256"]
            or binding["certificate"]["sha256"] != selected["certificate"]["sha256"]
            or self.policy.search_corpus.manifest_sha256 != selected["corpus_manifest_sha256"]
            or self.policy.search_corpus.capsules_sha256 != selected["corpus_sha256"]
            or str(self.policy.candidate) != selected["candidate"]
            or str(self.policy.search_baseline) != selected["baseline"]
            or sha256_file(self.policy.feedback_evaluator.rtl_facts_path) != selected["rtl_facts"]["sha256"]
            or document_sha256(dict(os.environ)) != request["environment_sha256"]
        ):
            raise StageGateError("normal CPU study changed original target/corpus/toolchain/runtime/environment")
        from .ordinary_calibration import public_partition

        public_partition(self.policy, request["public_partition"])
        return planned

    def profile(self, *, request, revision, baseline_revision, workspace, kind, call_index):
        from .supervised_feedback import bounded_feedback

        self.verify(request)
        COST._allocated(request)
        started, cpu = time.monotonic(), _cpu()
        result = bounded_feedback(
            _profile_fixed,
            timeout_s=request["max_stage_seconds"],
            output=Path(request["resources"]["work_root"]) / "profile_workers",
            kwargs={
                "adapter": self,
                "request": request,
                "revision": revision,
                "baseline_revision": baseline_revision,
                "workspace": workspace,
                "kind": kind,
                "call_index": call_index,
            },
        )
        result["evaluated_work_wall_seconds"] = result["wall_seconds"]
        result["wall_seconds"], result["cpu_seconds"] = time.monotonic() - started, _cpu() - cpu
        return result

    def _profile(self, *, request, revision, baseline_revision, workspace, kind, call_index):
        """Measure the complete fixed route; no subtraction manufactures stage costs."""
        self.verify(request)
        COST._allocated(request)
        if kind not in (FAST.KIND, "corpus-rtl-feedback"):
            raise StageGateError("normal study cannot select another command or evaluator")
        workspace = Path(workspace)
        root = Path(request["resources"]["work_root"])
        if workspace.resolve() != workspace or workspace.exists() or not workspace.is_relative_to(root):
            raise StageGateError("normal study profile requires fresh allocated outputs")
        previous = os.sched_getaffinity(0)
        os.sched_setaffinity(0, request["resources"]["cpu_ids"])
        before, wall, cpu = COST._disk_bytes(root), time.monotonic(), _cpu()
        peak, errors, stop = [0, before], [], threading.Event()

        def sample():
            while not stop.is_set():
                try:
                    peak[0] = max(peak[0], COST._tree_rss())
                    peak[1] = max(peak[1], COST._disk_bytes(root))
                except (OSError, StageGateError) as error:
                    errors.append(str(error))
                    break
                stop.wait(0.05)

        thread = threading.Thread(target=sample)
        thread.start()
        try:
            result = self.policy.evaluate_retained(
                revision,
                baseline_revision,
                kind=kind,
                work_root=workspace,
                call_index=call_index,
                timeout_s=request["max_stage_seconds"],
                workers=len(request["resources"]["cpu_ids"]),
            )
        finally:
            stop.set()
            thread.join(timeout=5)
            os.sched_setaffinity(0, previous)
        self.verify(request)
        if errors or thread.is_alive():
            raise StageGateError("normal study resource sample is incomplete")
        return {
            "schema": "merlin.normal_aws_cpu_profile.v1",
            "kind": kind,
            "wall_seconds": time.monotonic() - wall,
            "cpu_seconds": _cpu() - cpu,
            "peak_process_tree_rss_bytes": max(peak[0], COST._tree_rss()),
            "disk_bytes_before": before,
            "peak_output_disk_bytes": max(peak[1], COST._disk_bytes(root)),
            "disk_bytes_after": COST._disk_bytes(root),
            "cache_state": "UNKNOWN",
            "request_sha256": document_sha256(request),
            "result": result,
            "promotion": "NONE",
            "scope": "complete observed original route; external service cost UNKNOWN",
        }


def _cpu():
    return sum(
        row.ru_utime + row.ru_stime
        for row in (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))
    )


def _profile_fixed(*, adapter, **kwargs):
    if type(adapter) is not OrdinaryCorpusCpuStudyAdapter:
        raise StageGateError("normal study requires its fixed live adapter")
    return adapter._profile(**kwargs)


@contextmanager
def _complete_cost(root):
    """Observe complete setup/calibration calls, including owned descendants."""
    result = {}
    before, wall, cpu = COST._disk_bytes(root), time.monotonic(), _cpu()
    peak, errors, stop = [0, before], [], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                peak[0] = max(peak[0], COST._tree_rss())
                peak[1] = max(peak[1], COST._disk_bytes(root))
            except (OSError, StageGateError) as error:
                errors.append(str(error))
                break
            stop.wait(0.05)

    thread = threading.Thread(target=sample)
    thread.start()
    try:
        yield result
    finally:
        stop.set()
        thread.join(timeout=5)
    if errors or thread.is_alive():
        raise StageGateError("normal study complete resource observation is unresolved")
    result.update(
        wall_seconds=time.monotonic() - wall,
        cpu_seconds=_cpu() - cpu,
        peak_process_tree_rss_bytes=max(peak[0], COST._tree_rss()),
        disk_bytes_before=before,
        peak_output_disk_bytes=max(peak[1], COST._disk_bytes(root)),
        disk_bytes_after=COST._disk_bytes(root),
        cache_state="UNKNOWN",
    )


def _load(request):
    """Construct only the original normal admission factory from closed selected inputs."""
    from merlin.benchharness import hash_tree
    from merlin.targetgen.target_experiment import load_target_experiment

    from .corpus import load_frozen_performance_corpus
    from .edit_authority import FrozenEditAuthority
    from .ordinary_search_policy import prepare_ordinary_search_policy
    from .retained_revisions import RetainedRevisionStore

    inputs, root = request["normal_inputs"], Path(request["resources"]["work_root"])
    target = load_target_experiment(inputs["target_descriptor"]["path"])
    corpus = load_frozen_performance_corpus(
        inputs["corpus_root"], manifest_sha256=inputs["corpus_manifest_sha256"], capsules_sha256=inputs["corpus_sha256"]
    )
    control = root / "normal_study_control"
    if control.exists():
        raise StageGateError("normal study requires fresh host control outputs")
    control.mkdir(mode=0o700)
    authority_root = control / "authority"
    authority_root.mkdir()
    authority = FrozenEditAuthority(authority_root, guidance_contract=Path(inputs["contract_root"]))
    authority.freeze(
        Path(inputs["candidate"]), mapping_file(Path(inputs["edit_contract"]["path"])), has_iterations=False
    )
    policy = prepare_ordinary_search_policy(
        functional_runs_root=Path(inputs["functional_runs_root"]),
        functional_run_id=inputs["functional_run_id"],
        functional_submission_sha256=inputs["functional_submission_sha256"],
        corpus=corpus,
        baseline=Path(inputs["baseline"]),
        candidate=Path(inputs["candidate"]),
        target_experiment=target,
        receipt_path=control / "receipts.jsonl",
        work_root=control / "feedback",
        edit_authority=authority,
        contract_root=Path(inputs["contract_root"]),
        certificate_path=Path(inputs["certificate"]["path"]),
        certificate_sha256=inputs["certificate"]["sha256"],
        rtl_facts_path=Path(inputs["rtl_facts"]["path"]),
    )
    store = RetainedRevisionStore(
        control / "revisions",
        edit_authority=authority,
        max_bytes=min(64 << 20, request["resources"]["disk_bytes"]),
        max_files=10000,
        max_total_bytes=request["resources"]["disk_bytes"],
    )
    baseline = store.freeze(policy.search_baseline, options={"hypothesis": "original Phase 1 baseline"})
    alternatives = []
    for row in request["alternatives"]:
        source = Path(row["path"])
        if str(hash_tree(source)["sha256"]) != row["sha256"]:
            raise StageGateError("normal study public compiler alternative changed")
        alternatives.append(store.freeze(source))
    return OrdinaryCorpusCpuStudyAdapter(policy), tuple(alternatives), baseline


def execute(request):
    """Explicit allocated AWS execution; profile and qualify original public routes."""
    plan(request)
    COST._allocated(request)
    if document_sha256(dict(os.environ)) != request["environment_sha256"]:
        raise StageGateError("normal study must retain the explicitly selected Phase 1 environment")
    workers = str(len(request["resources"]["cpu_ids"]))
    if os.environ.get("MERLIN_PERF_SWEEP_WORKERS", "1") != workers:
        raise StageGateError("normal study member fanout must match its explicit allocation")
    root, profiles = Path(request["resources"]["work_root"]), []
    with _complete_cost(root) as setup:
        adapter, revisions, baseline = _load(request)
        adapter.verify(request)
    setup.update(engine_build_cost=None, scope="normal admission/source freeze; prior engine build cost UNKNOWN")
    for repeat in range(request["replicates"]):
        for index, revision in enumerate(revisions):
            for kind, label in ((FAST.KIND, "cheap"), ("corpus-rtl-feedback", "rtl")):
                profile = adapter.profile(
                    request=request,
                    revision=revision,
                    baseline_revision=baseline,
                    workspace=root / f"profile_{repeat}_{index}_{label}",
                    kind=kind,
                    call_index=index,
                )
                profiles.append(profile)
                write_json(root / f"profile_{repeat}_{index}_{label}.json", profile)
    from .ordinary_calibration import CalibrationUnavailable, domain, produce_calibration

    calibrated = None
    with _complete_cost(root) as calibration:
        try:
            calibrated = produce_calibration(
                adapter.policy,
                revisions,
                baseline,
                partition=request["public_partition"],
                work_root=root / "calibration",
                timeout_s=request["max_stage_seconds"],
                workers=int(workers),
                allocation=request["resources"],
            )
        except CalibrationUnavailable as error:
            qualification = {"status": "UNKNOWN", "exposable": False, "promotion": "NONE", "reason": str(error)}
    adapter.verify(request)
    if calibrated is not None:
        qualification = loads(calibrated.report_json)
    return {
        "schema": SCHEMA,
        "execution_performed": True,
        "request_sha256": document_sha256(request),
        "profiles": profiles,
        "setup": setup,
        "calibration_sha256": None if calibrated is None else calibrated.sha256,
        "calibration_and_held_validation": calibration,
        "domain_sha256": domain(adapter.policy) if calibrated is None else calibrated.domain_sha256,
        "public_partition": request["public_partition"],
        "source_revisions": [revision.public_record() for revision in revisions],
        "qualification": qualification,
        "cache_state": "UNKNOWN",
        "promotion": "NONE",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="execute only on the explicitly allocated AWS CPU")
    args = parser.parse_args(argv)
    try:
        if args.request.stat().st_size > 1 << 20 or args.output.exists() or args.output.is_symlink():
            raise StageGateError("normal study needs bounded metadata and a fresh plan/result destination")
        request = mapping_file(args.request)
        result = execute(request) if args.execute else plan(request)
        write_json(args.output, result)
    except (StageGateError, ValueError, OSError) as error:
        parser.error(str(error))
    print("normal AWS CPU study: execution_performed=" + str(result["execution_performed"]).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
