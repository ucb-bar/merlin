"""Neutral retained-queue pair controls; no accelerator or timing is qualified."""

import copy
import json
from contextlib import nullcontext

import pytest
import test_public_source_diagnostic as SOURCE
from merlin_experiments.phase2 import diagnostic_pair_check as P
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore

from merlin.common.digest import sha256_file
from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapabilities, ReviewedPassCapability
from merlin.targetgen import checked_source_model_diagnostic as CHECKED
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection


class OriginalPairControl:
    def __init__(self, evidence):
        self.evidence = evidence
        self.identity = sha256_file(evidence)
        self.calls = []
        self.status = "PASS"

    def revalidate(self):
        if sha256_file(self.evidence) != self.identity:
            raise ValueError("original neutral evidence changed")

    def check_source_pair(self, *, baseline_workspace, candidate_workspace, capsule, remaining_s):
        self.revalidate()
        assert remaining_s > 0 and capsule["kind"] == "model"
        assert all((path / "ordinary_observation.json").is_file() for path in (baseline_workspace, candidate_workspace))
        self.calls.append((baseline_workspace, candidate_workspace))
        return {"status": self.status, "scope": "neutral invocation wiring only"}


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    args, *_ = SOURCE.inputs.__wrapped__(tmp_path, monkeypatch)
    evidence = tmp_path / "pair-evidence"
    evidence.write_text("independent neutral fixture; no target effects\n")
    owner = OriginalPairControl(evidence)
    original = args["compiler_pass_selection"]
    registry = ReviewedPassCapabilities(
        (
            original.registry._capabilities[0],
            ReviewedPassCapability("pair_control", original.binding, owner),
        )
    )
    passes = CompilerPassSelection(registry, original.binding, mechanisms=("pair_control",), factory=original.factory)
    source_passes = CompilerPassSelection(
        registry,
        original.binding,
        mechanisms=("source_execution_diagnostic",),
        factory=original.factory,
    )
    args["compiler_pass_selection"] = source_passes
    args["checked_source_diagnostic"] = CHECKED.select_checked_source_diagnostic(
        source_passes,
        target=args["target_experiment"].target,
        rtl_facts=args["checked_source_diagnostic"].record()["rtl_facts"]["path"],
        board_config="fixture",
    )
    selected = P.select_source_pair_check(passes, source_passes)
    args["diagnostic_pair_check"] = selected
    return args, selected, owner, passes


def queue_context(prepared, tmp_path, monkeypatch, *, functional=True):
    args, selected, owner, _ = prepared
    fast = D.FAST
    monkeypatch.setattr(fast.CR, "simulator_adapter", lambda *a: "neutral-no-engine")
    monkeypatch.setattr(D.CAMPAIGN, "package_sandbox_policy", lambda *a: "neutral-no-box")
    monkeypatch.setattr(D.CAMPAIGN, "boxed_entrypoints", lambda *a: nullcontext())
    monkeypatch.setattr(fast, "compiled_observation", lambda *a, **kw: {"status": "UNKNOWN"})
    monkeypatch.setattr(fast, "observe_passed_census", lambda *a, **kw: {"status": "UNKNOWN"})
    monkeypatch.setattr(D.PublicSourceDiagnosticPolicy, "compiler_execution", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        fast.CR,
        "run_capsule",
        lambda *a, **kw: {
            "status": "incomplete",
            "checked_source_status": "complete_source_L0_L1_L2_diagnostic_pass" if functional else "refused",
            "tiers": {t: {"status": "pass"} for t in ("L0", "L1", "L2")},
        },
    )
    policy = D.prepare_public_source_diagnostic(**args)
    store = RetainedRevisionStore(
        tmp_path / "revisions",
        edit_authority=policy.edit_authority,
        max_bytes=1 << 20,
        max_files=100,
        max_total_bytes=4 << 20,
    )
    baseline = store.freeze(args["baseline"])
    (args["candidate"] / "compiler.py").write_text("def choose():\n return 1\n")
    candidate = store.freeze(args["candidate"], parent_revision=baseline.revision_id)
    return policy, baseline, candidate, selected, owner, store


def queue(prepared, tmp_path, monkeypatch, *, functional=True):
    args, *_ = prepared
    policy, baseline, candidate, selected, owner, _ = queue_context(
        prepared,
        tmp_path,
        monkeypatch,
        functional=functional,
    )
    result = policy.evaluate_retained(
        candidate,
        baseline,
        work_root=args["work_root"] / "job",
        call_index=0,
        timeout_s=30,
        selected_members=tuple(sorted(policy.validate_search_inputs())),
    )
    return policy, candidate, result, selected, owner


def test_live_pair_runs_after_both_arms_and_before_unscored_feedback(prepared, tmp_path, monkeypatch):
    policy, revision, result, _, owner = queue(prepared, tmp_path, monkeypatch)
    assert result["schema"] == D.PAIRED_FEEDBACK_SCHEMA
    assert result["promotion"] == "DIAGNOSTIC_ONLY" and result["certificate_sha256"] is None
    assert len(owner.calls) == len(result["cells"])
    for cell in result["cells"]:
        assert cell["diagnostic_pair_check"]["status"] == "PASS"
        for arm in ("baseline", "candidate"):
            assert cell[arm]["timing_cycles"] is None and cell[arm]["pending_tiers"] == ["L3"]
    assert D.FAST.validate_feedback(result, policy=policy, revision=revision, kind=D.FAST.SELECTED_KIND) == result


def test_refused_relation_never_publishes_feedback(prepared, tmp_path, monkeypatch):
    args, _, owner, _ = prepared
    owner.status = "REFUSED"
    with pytest.raises(P.StageGateError, match="relation refused"):
        queue(prepared, tmp_path, monkeypatch)
    assert not (args["work_root"] / "job/ordinary_fast_feedback.json").exists()
    assert (args["work_root"] / "job/pair-checks/m000.json").is_file()


def test_incomplete_arm_keeps_pair_unknown_without_invoking_target(prepared, tmp_path, monkeypatch):
    _, _, result, _, owner = queue(prepared, tmp_path, monkeypatch, functional=False)
    assert not owner.calls
    assert all(cell["diagnostic_pair_check"]["status"] == "UNKNOWN" for cell in result["cells"])


@pytest.mark.parametrize("fault", ["projection", "inplace_projection", "receipt", "arm", "saved_selection"])
def test_saved_or_modified_pair_cannot_replace_live_observation(prepared, tmp_path, monkeypatch, fault):
    policy, revision, result, selected, _ = queue(prepared, tmp_path, monkeypatch)
    if fault != "inplace_projection":
        result = copy.deepcopy(result)
    if fault in ("projection", "inplace_projection"):
        result["cells"][0]["diagnostic_pair_check"]["scope"] = "qualified timing"
    elif fault == "receipt":
        path = policy.feedback_evaluator.work_root / "job/pair-checks/m000.json"
        path.write_text(json.dumps({"status": "PASS"}))
    elif fault == "arm":
        result["cells"][0]["candidate"]["evidence_sha256"] = "0" * 64
    else:
        copied = copy.copy(selected)
        with pytest.raises(P.StageGateError, match="original issued"):
            copied.verify_for(policy._compiler_pass_selection)
        return
    with pytest.raises(P.StageGateError, match="pair feedback"):
        D.FAST.validate_feedback(result, policy=policy, revision=revision, kind=D.FAST.SELECTED_KIND)


@pytest.mark.parametrize("fault", ["registry", "callback", "evidence", "saved"])
def test_selection_reopens_original_registry_callback_and_sources(prepared, fault, monkeypatch):
    args, selected, owner, passes = prepared
    source = args["compiler_pass_selection"]
    if fault == "registry":
        with pytest.raises(P.StageGateError, match="same original"):
            P.select_source_pair_check(passes, args["checked_source_diagnostic"])
    elif fault == "callback":
        monkeypatch.setattr(owner, "check_source_pair", lambda **kw: {"status": "PASS"})
        with pytest.raises(P.StageGateError, match="callback"):
            selected.verify_for(source)
    elif fault == "evidence":
        owner.evidence.write_text("copied PASS")
        with pytest.raises(ValueError, match="original neutral"):
            selected.record()
    else:
        with pytest.raises(P.StageGateError, match="same original"):
            P.select_source_pair_check(selected.record(), source)


def test_forked_pair_requires_fresh_original_parent_check(prepared, tmp_path, monkeypatch):
    import multiprocessing
    import time

    policy, baseline, candidate, selected, owner, _ = queue_context(prepared, tmp_path, monkeypatch)
    members = tuple(sorted(policy.validate_search_inputs()))
    root = policy.feedback_evaluator.work_root / "forked"
    result_path = tmp_path / "child-result.json"

    def child():
        result = policy.evaluate_retained(
            candidate,
            baseline,
            work_root=root,
            call_index=0,
            timeout_s=30,
            selected_members=members,
        )
        result_path.write_text(json.dumps(result))

    process = multiprocessing.get_context("fork").Process(target=child)
    process.start()
    try:
        process.join(timeout=30)
        assert not process.is_alive() and process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()
    result = json.loads(result_path.read_bytes())
    assert owner.calls == []
    with pytest.raises(P.StageGateError, match="not observed"):
        D.FAST.validate_feedback(result, policy=policy, revision=candidate, kind=D.FAST.SELECTED_KIND)
    policy.consume_feedback_child_result(
        result,
        work_root=root,
        selected_members=members,
        deadline=time.monotonic() + 30,
    )
    assert len(owner.calls) == len(members)
    assert D.FAST.validate_feedback(result, policy=policy, revision=candidate, kind=D.FAST.SELECTED_KIND) == result


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object(), "x" * (1 << 20)])
def test_pair_report_must_be_bounded_closed_finite_data(value):
    with pytest.raises(P.StageGateError, match="bounded report"):
        P._closed({"unsupported": value})


def test_normal_pool_consumes_pair_through_original_parent_before_completion(prepared, tmp_path, monkeypatch):
    import os
    import time

    from merlin_experiments.phase2.evaluation_pool import AssignedResources, ComponentEvaluationPool, JobResources

    policy, _, candidate, _, owner, store = queue_context(prepared, tmp_path, monkeypatch)
    assigned = tuple(sorted(os.sched_getaffinity(0)))
    pool = ComponentEvaluationPool(
        policy=policy,
        revisions=store,
        resources=AssignedResources(assigned, 1 << 30, 64 << 20, 1, author_cores=0, reserve_fraction=0),
        job_resources=JobResources(1, 1 << 30, 32 << 20),
        root=policy.feedback_evaluator.work_root / "queue",
        beam=1,
        max_pending=1,
        max_jobs=1,
    )
    try:
        members = tuple(sorted(policy.validate_search_inputs()))
        job = pool.submit(candidate.revision_id, timeout_s=30, selected_members=members)
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            result = pool.status(job["job_id"])
            if result["terminal"]:
                break
            time.sleep(0.02)
        assert result["state"] == "completed", result
        assert len(owner.calls) == len(members)  # Only the parent's fresh callbacks are visible here.
        assert result["result"]["promotion"] == "DIAGNOSTIC_ONLY"
        assert all(row["diagnostic_pair_check"]["status"] == "PASS" for row in result["result"]["cells"])
        assert (pool.root / job["job_id"] / "cleanup.json").is_file()
    finally:
        pool.close()
