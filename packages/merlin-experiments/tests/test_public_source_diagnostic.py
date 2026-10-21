"""Original profile/corpus issuance and neutral queue controls; no target executes."""

import copy
import json
import os
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import test_checked_source_model_diagnostic as SOURCE
import test_compiler_profile_child as PROFILE
from merlin_experiments.phase2 import campaign as CAMPAIGN
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2 import search_inputs as SI
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore
from test_component_workflow import generated_corpus

from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.perf.phase2_edit_contract import seal
from merlin.targetgen import checked_source_model_diagnostic as S
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, OriginalPassFactory


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    # The original metadata owner now requires a real admitted CPU leaf.
    # Select an isolated slot namespace instead of relying on launcher state.
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", ",".join(map(str, sorted(os.sched_getaffinity(0)))))
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "cpu-slots"))
    profile_root = tmp_path / "profile"
    profile_root.mkdir()
    original = PROFILE.original.__wrapped__(profile_root, monkeypatch)
    library = PROFILE._additional_library(original)
    source_root = tmp_path / "source"
    source_root.mkdir()
    _, _, issuer, facts, _ = SOURCE.selected.__wrapped__(source_root, monkeypatch)
    binding = PassCapabilityBinding(S._pin(facts)[1], "software", "numeric", "runtime", library.contract.sha256)
    registry = ReviewedPassCapabilities((ReviewedPassCapability(S.MECHANISM, binding, issuer),))
    passes = CompilerPassSelection(
        registry,
        binding,
        mechanisms=("optional_not_qualified",),
        factory=OriginalPassFactory(SOURCE.original_factory, {}),
    )
    checked = S.select_checked_source_diagnostic(
        passes, target=original.target.target, rtl_facts=facts, board_config="fixture"
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("def choose():\n return 0\n")
    (candidate / "manifest.yaml").write_text(
        json.dumps(
            {
                "entrypoints": {"tool": "compiler.py"},
                "commands": {"compile": {"argv": ["python", "{tool}"]}},
            }
        )
    )
    baseline = tmp_path / "baseline"
    CAMPAIGN.materialize_readonly_tree(candidate, baseline)
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    authority = FrozenEditAuthority(authority_root)
    authority.freeze(
        candidate,
        seal(
            {
                "schema": "compiler_edit_contract_v1",
                "existing_symbols": [{"surface_id": "policy", "path": "compiler.py", "symbol": "choose"}],
                "helper_extensions": [],
            }
        ),
        has_iterations=False,
    )
    corpus = generated_corpus(tmp_path / "corpus", kind="model")
    contract = tmp_path / "contract"
    contract.mkdir()
    (contract / "readme").write_text("synthetic source-only metadata fixture")
    cap = dict(kind="model", label="dev", required_oracle_tiers=["L0", "L1", "L2", "L3"])
    monkeypatch.setattr(FAST.CR, "load_capsule", lambda *a, **kw: copy.deepcopy(cap))
    args = dict(
        candidate=candidate,
        baseline=baseline,
        corpus=corpus,
        target_experiment=original.target,
        edit_authority=authority,
        service=original.owner,
        compiler_library_selection=library,
        compiler_pass_selection=passes,
        checked_source_diagnostic=checked,
        contract_root=contract,
        work_root=tmp_path / "diagnostics",
    )
    return args, original, cap, issuer


def test_real_original_profile_issues_unscored_context_without_formal_gate(inputs, monkeypatch):
    from merlin_experiments.phase2 import functional_inputs as FI

    monkeypatch.setattr(FI, "require_formal_qualification", lambda *_a: pytest.fail("formal issuer is separate"))
    args, _, _, _ = inputs
    policy = D.prepare_public_source_diagnostic(**args)
    assert type(policy) is D.PublicSourceDiagnosticPolicy
    assert policy.binding()["formal_admission"] is False
    selected = SI.SearchInputs(policy)
    assert selected.diagnostic and selected.kinds == (FAST.SELECTED_KIND,)
    assert selected.identities()["certificate_sha256"] is None
    assert policy.feedback_evaluator._compiler_profile_service is args["service"]
    with pytest.raises(StageGateError, match="original host"):
        D.PublicSourceDiagnosticPolicy(_issuer=None)
    copied = copy.copy(policy)
    with pytest.raises(StageGateError, match="original host"):
        copied.validate_search_inputs()


@pytest.mark.parametrize("protocol,ceiling", [(None, 900), (D.FULL_SOURCE_FEEDBACK_BUDGET_PROTOCOL, 2400)])
def test_explicit_full_source_budget_preserves_default_and_dispatches_with_original_deadline(
    inputs, monkeypatch, tmp_path, protocol, ceiling
):
    args, *_ = inputs
    seen = []

    def observe(*a, **kw):
        seen.append(kw["timeout_s"])
        return {"neutral_original_dispatch": True}

    monkeypatch.setattr(FAST, "evaluate_retained", observe)
    policy = D.prepare_public_source_diagnostic(**args, feedback_budget_protocol=protocol)
    assert policy._feedback_timeout_seconds() == ceiling
    binding = policy.binding()
    if protocol is None:
        assert "feedback_budget" not in binding
    else:
        assert binding["feedback_budget"]["protocol"] == protocol
        assert binding["feedback_budget"]["maximum_seconds"] == ceiling
    store = RetainedRevisionStore(tmp_path / "budget-revisions", edit_authority=policy.edit_authority,
                                  max_bytes=1 << 20, max_files=100, max_total_bytes=4 << 20)
    baseline = store.freeze(args["baseline"])
    result = policy.evaluate_retained(baseline, baseline, work_root=args["work_root"] / "budget-job",
                                     call_index=0, timeout_s=ceiling)
    assert result == {"neutral_original_dispatch": True} and len(seen) == 1 and 0 < seen[0] <= ceiling
    for invalid in (ceiling + 1, True, 0, float("nan")):
        with pytest.raises(StageGateError, match="bounded selected"):
            policy.evaluate_retained(baseline, baseline, work_root=args["work_root"] / "invalid-job",
                                     call_index=1, timeout_s=invalid)
    if protocol is not None:
        policy._feedback_budget_protocol = None
        with pytest.raises(StageGateError, match="selection changed"):
            policy.validate_search_inputs()


@pytest.mark.parametrize("protocol", ["unknown", 2400, True, {}])
def test_unknown_full_source_budget_refuses_before_owner_issuance(inputs, protocol):
    args, *_ = inputs
    with pytest.raises(StageGateError, match="feedback budget protocol"):
        D.prepare_public_source_diagnostic(**args, feedback_budget_protocol=protocol)
    assert not args["work_root"].exists()


@pytest.mark.parametrize("fault", ["profile", "facts", "library", "source", "target", "authority"])
def test_original_owner_or_frozen_byte_changes_refuse(inputs, fault):
    args, original, *_ = inputs
    policy = D.prepare_public_source_diagnostic(**args)
    if fault == "profile":
        original.owner.bundle_manifest.write_text("foreign original archive")
    elif fault == "facts":
        policy.feedback_evaluator.rtl_facts_path.write_text("changed facts")
    elif fault == "library":
        path = args["compiler_library_selection"].root / "merlin/portable.py"
        path.write_text("changed import")
    elif fault == "source":
        path = args["baseline"] / "compiler.py"
        path.chmod(0o644)
        path.write_text("changed initial seed")
    elif fault == "target":
        policy.target_experiment = replace(policy.target_experiment, target="foreign")
    else:
        policy.edit_authority = object()
    with pytest.raises((ValueError, StageGateError)):
        policy.validate_search_inputs()


def test_hidden_or_missing_mandatory_roles_refuse_at_issuance(inputs):
    args, _, cap, _ = inputs
    cap["label"] = "hidden"
    with pytest.raises(StageGateError, match="complete public"):
        D.prepare_public_source_diagnostic(**args)
    cap["label"] = "dev"
    cap["required_oracle_tiers"].remove("L3")
    with pytest.raises(StageGateError, match="mandatory tier"):
        D.prepare_public_source_diagnostic(**args)


def test_selected_retained_route_uses_same_issued_policy_and_keeps_all_obligations(inputs, tmp_path, monkeypatch):
    args, _, _, _ = inputs
    calls = []
    monkeypatch.setattr(FAST.CR, "simulator_adapter", lambda *a: "synthetic-no-engine")
    monkeypatch.setattr(CAMPAIGN, "package_sandbox_policy", lambda *a: "synthetic-no-box")
    monkeypatch.setattr(CAMPAIGN, "boxed_entrypoints", lambda *a: nullcontext())
    monkeypatch.setattr(FAST, "compiled_observation", lambda *a, **kw: {"status": "UNKNOWN"})
    monkeypatch.setattr(FAST, "observe_passed_census", lambda *a, **kw: {"status": "UNKNOWN"})
    # Only execution is neutral. The original policy/profile/corpus/edit owners
    # are actual host objects; no synthetic facade can instantiate the policy.
    monkeypatch.setattr(D.PublicSourceDiagnosticPolicy, "compiler_execution", lambda *a, **kw: nullcontext())

    def run(cap, package, **kwargs):
        assert kwargs["checked_source_diagnostic"] is args["checked_source_diagnostic"]
        assert 0 < kwargs["model_budget_s"] <= kwargs["timeout"]
        kwargs["checked_source_diagnostic"].verify_for(cap, target=args["target_experiment"].target)
        calls.append(package)
        return {
            "status": "incomplete",
            "checked_source_status": "complete_source_L0_L1_L2_diagnostic_pass",
            "tiers": {t: {"status": "pass"} for t in ("L0", "L1", "L2")},
        }

    monkeypatch.setattr(FAST.CR, "run_capsule", run)
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
    selected = tuple(sorted(policy.validate_search_inputs()))
    result = policy.evaluate_retained(
        candidate,
        baseline,
        work_root=args["work_root"] / "job",
        call_index=0,
        timeout_s=30,
        selected_members=selected,
    )
    assert set(calls) == {candidate.source, baseline.source}
    assert result["schema"] == D.FEEDBACK_SCHEMA and result["certificate_sha256"] is None
    assert result["promotion"] == "DIAGNOSTIC_ONLY" and result["screening"]["status"] == "UNKNOWN"
    assert SI.SearchInputs(policy).validate_result(result, candidate, FAST.SELECTED_KIND) == result
    # The real fixed worker must carry its original deadline into the final
    # selection/binding validation too, after the retained evaluator returns.
    from merlin_experiments.phase2 import evaluation_pool as EP

    validation_deadlines = []
    validate = SI.SearchInputs.validate_result

    def validate_result(self, raw, revision, kind, *, deadline=None):
        validation_deadlines.append(deadline)
        return validate(self, raw, revision, kind, deadline=deadline)

    monkeypatch.setattr(SI.SearchInputs, "validate_result", validate_result)
    monkeypatch.setattr(EP.resource, "setrlimit", lambda *_a: None)
    monkeypatch.setattr(EP.WORKER_RESOURCES, "configure", lambda *_a, **_kw: None)
    worker_root = args["work_root"] / "fixed-worker"
    worker_root.mkdir()
    worker_result = EP._evaluate_fixed_routes(
        policy=policy,
        revision=candidate,
        baseline_revision=baseline,
        kind=FAST.SELECTED_KIND,
        selected_members=selected,
        timeout_s=900,
        root=worker_root,
        grant=SimpleNamespace(cores=1, engine_slots=1, ram_bytes=1 << 30, disk_bytes=1 << 20),
        cpus=(0,),
        index=0,
    )
    assert worker_result["promotion"] == "DIAGNOSTIC_ONLY"
    assert len(validation_deadlines) == 1
    assert 0 < validation_deadlines[0] - EP.time.monotonic() <= 900
    for cell in result["cells"]:
        for arm in ("baseline", "candidate"):
            assert cell[arm]["pending_tiers"] == ["L3"] and cell[arm]["timing_cycles"] is None
    with pytest.raises(StageGateError, match="selected compile/Spike"):
        policy.evaluate_retained(
            candidate, baseline, work_root=args["work_root"] / "full", call_index=1, timeout_s=30, kind=FAST.KIND
        )


def test_body_exception_reaches_original_scope_with_no_parent_lease_during_body(inputs, monkeypatch):
    from contextlib import contextmanager

    from merlin_experiments.phase1.feedback import compiler_profile_runtime as R

    from merlin.common import cpu_pool as C

    args, *_ = inputs
    errors, metadata_priorities = [], []
    monkeypatch.setenv(C.PRIORITY_ENV, str(C.GRADE))

    @contextmanager
    def execution(*a, **kw):
        assert C.current_thread_owns_cpu_lease()
        metadata_priorities.append(C.lease_priority())
        assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
        try:
            yield
        except RuntimeError as error:
            assert C.current_thread_owns_cpu_lease()
            metadata_priorities.append(C.lease_priority())
            assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
            errors.append(error)
            raise

    monkeypatch.setattr(R, "original_compiler_execution", execution)
    policy = D.prepare_public_source_diagnostic(**args)
    failure = RuntimeError("original observed body refusal")
    with C.cpu_priority(C.GRADE):
        with pytest.raises(RuntimeError, match="body refusal"):
            with policy.compiler_execution(args["candidate"], args["work_root"] / "scope"):
                assert not C.current_thread_owns_cpu_lease()
                assert C.lease_priority() == C.GRADE
                assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
                raise failure
        assert C.lease_priority() == C.GRADE
        assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
    assert metadata_priorities == [C.AUTHOR, C.AUTHOR]
    assert errors == [failure]


def test_screen_installation_and_expired_metadata_are_refused_before_execution(inputs):
    args, *_ = inputs
    policy = D.prepare_public_source_diagnostic(**args)
    with pytest.raises(StageGateError, match="aggregate deadline"):
        policy.validate_search_inputs(deadline=-1)
    policy._ordinary_screen = object()
    with pytest.raises(StageGateError, match="original execution owner"):
        policy.validate_search_inputs()


@pytest.mark.parametrize("deadline,accepted", [(900, True), (150, False), (None, False)])
def test_binding_preserves_operation_deadline_through_checked_pass_records(inputs, monkeypatch, deadline, accepted):
    args, *_ = inputs
    policy = D.prepare_public_source_diagnostic(**args)
    clock, waits = [10], []

    @contextmanager
    def lease(*, timeout_s):
        waits.append(timeout_s)
        yield
        # Actual original records were reopened; metadata finishes after the
        # queue ceiling but must retain the caller's original operation budget.
        clock[0] = 200

    monkeypatch.setattr(D.CHECKED, "monotonic", lambda: clock[0])
    monkeypatch.setattr(D.CHECKED, "shared_compute_pool", lease)
    if accepted:
        assert policy.binding(deadline=deadline)["promotion_authority"] == "NONE"
    else:
        with pytest.raises(StageGateError, match="aggregate deadline"):
            policy.binding(deadline=deadline)
    assert waits == [120]


@pytest.mark.parametrize(
    "owner", ["service", "compiler_library_selection", "compiler_pass_selection", "edit_authority"]
)
def test_saved_or_foreign_owner_records_cannot_issue_policy(inputs, owner):
    args, *_ = inputs
    args = dict(args, **{owner: {"status": "passed"}})
    with pytest.raises(StageGateError, match="original live"):
        D.prepare_public_source_diagnostic(**args)


@pytest.mark.parametrize("setup_seconds", [5, 11])
def test_original_profile_setup_time_clips_or_exhausts_child_budget(inputs, tmp_path, monkeypatch, setup_seconds):
    from contextlib import contextmanager
    from types import SimpleNamespace

    args, *_ = inputs
    clock = [90.0]
    timeouts = []

    @contextmanager
    def original_setup(*_a, **_kw):
        clock[0] += setup_seconds
        yield

    def run(*_a, timeout, **_kw):
        timeouts.append(timeout)
        raise RuntimeError("neutral child call reached")

    monkeypatch.setattr(D.PublicSourceDiagnosticPolicy, "compiler_execution", original_setup)
    monkeypatch.setattr(CAMPAIGN, "package_sandbox_policy", lambda *_a: "neutral-no-box")
    monkeypatch.setattr(CAMPAIGN, "boxed_entrypoints", lambda *_a: nullcontext())
    monkeypatch.setattr(FAST.CR, "simulator_adapter", lambda *_a: "neutral-no-engine")
    monkeypatch.setattr(FAST.CR, "run_capsule", run)
    policy = D.prepare_public_source_diagnostic(**args)
    revision = RetainedRevisionStore(
        tmp_path / "revisions",
        edit_authority=policy.edit_authority,
        max_bytes=1 << 20,
        max_files=100,
        max_total_bytes=4 << 20,
    ).freeze(args["baseline"])
    monkeypatch.setattr(FAST, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(FAST.CHECKED_SOURCE, "monotonic", lambda: clock[0])
    from merlin.targetgen.compiler_pass_selection import scoped_compiler_passes

    with scoped_compiler_passes(args["compiler_pass_selection"]):
        with pytest.raises((StageGateError, RuntimeError)) as error:
            FAST._member_arm(
                policy=policy,
                revision=revision,
                member=policy.search_corpus.capsules[0],
                workspace=args["work_root"] / "delayed-setup",
                deadline=100.0,
                workers=1,
            )
    if setup_seconds == 5:
        assert timeouts == [5] and "neutral child call" in str(error.value)
    else:
        assert timeouts == [] and "setup exhausted" in str(error.value)


def test_real_diagnostic_pool_is_constructed_after_releasing_metadata_grant(inputs, tmp_path):
    import os

    from merlin_experiments.phase2.evaluation_pool import (
        AssignedResources,
        ComponentEvaluationPool,
        JobResources,
    )

    from merlin.common.cpu_pool import current_thread_owns_cpu_lease, shared_compute_pool

    args, *_ = inputs
    policy = D.prepare_public_source_diagnostic(**args)
    store = RetainedRevisionStore(
        tmp_path / "revisions",
        edit_authority=policy.edit_authority,
        max_bytes=1 << 20,
        max_files=100,
        max_total_bytes=4 << 20,
    )
    assigned = tuple(sorted(os.sched_getaffinity(0)))
    # The coordinator keeps its original roster after the metadata grant ends;
    # its child workers will acquire their own actual leaf grants later.
    with shared_compute_pool(timeout_s=120):
        assert current_thread_owns_cpu_lease()
        unavailable = max(assigned) + 1
        with pytest.raises(StageGateError, match="explicit usable assigned"):
            AssignedResources(
                (*assigned, unavailable), 64 << 20, 64 << 20, 1, author_cores=0, reserve_fraction=0
            ).verify()
    assert not current_thread_owns_cpu_lease() and tuple(sorted(os.sched_getaffinity(0))) == assigned
    pool = ComponentEvaluationPool(
        policy=policy,
        revisions=store,
        resources=AssignedResources(assigned, 64 << 20, 64 << 20, 1, author_cores=0, reserve_fraction=0),
        job_resources=JobResources(1, 32 << 20, 16 << 20),
        root=args["work_root"] / "queue-control",
        beam=1,
        max_pending=1,
        max_jobs=1,
    )
    try:
        assert pool.inputs.diagnostic and pool.resources.cpu_ids == assigned
        assert pool.observations() == ()
    finally:
        pool.close()


def test_public_roster_preflight_and_binding_use_complete_batch(inputs, tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    import test_public_diagnostic_corpus as PUBLIC

    from merlin.common.paths import repo_root

    args, *_ = inputs
    public_root = tmp_path / "public"
    public_root.mkdir()
    public_args, *_ = PUBLIC.published.__wrapped__(public_root, monkeypatch)
    batches = []
    original = D.PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus.execution_capsules

    def batch(self, *, contract):
        result = original(self, contract=contract)
        batches.append(tuple(cap["__dir__"] for cap in result))
        return result

    monkeypatch.setattr(D.PUBLIC_CORPUS.FrozenPublicDiagnosticCorpus, "execution_capsules", batch)
    corpus = D.PUBLIC_CORPUS.freeze_public_diagnostic_corpus(**public_args)
    contract = tmp_path / "ordinary-contract"
    shutil.copytree(
        repo_root() / "merlin" / "contract",
        contract,
        ignore=lambda directory, names: [name for name in names if (Path(directory) / name).is_symlink()],
    )
    original_capsule = D._capsule
    monkeypatch.setattr(D, "_capsule", lambda *_a, **_kw: pytest.fail("binding must not repeat single-member views"))
    policy = D.prepare_public_source_diagnostic(**dict(args, corpus=corpus, contract_root=contract))
    binding = policy.binding()
    assert len(batches) >= 3
    assert all(batch == tuple(str(member.source_dir) for member in corpus.capsules) for batch in batches)
    assert len(binding["public_execution_views"]) == len(corpus.capsules)
    assert (
        binding["public_execution_views"][0]["complete_source_and_reference_tree_sha256"]
        == corpus.capsules[0].source_sha256
    )
    assert binding["formal_admission"] is False and binding["timing"] == "UNKNOWN"

    monkeypatch.setattr(D, "_capsule", original_capsule)
    # The genuine published view reaches both the retained producer and consumer.
    # Only target execution is neutral in this regression.
    monkeypatch.setattr(FAST.CR, "simulator_adapter", lambda *a: "neutral")
    monkeypatch.setattr(CAMPAIGN, "package_sandbox_policy", lambda *a: "neutral")
    monkeypatch.setattr(CAMPAIGN, "boxed_entrypoints", lambda *a: nullcontext())
    monkeypatch.setattr(FAST, "compiled_observation", lambda *a, **kw: {"status": "UNKNOWN"})
    monkeypatch.setattr(FAST, "observe_passed_census", lambda *a, **kw: {"status": "UNKNOWN"})
    monkeypatch.setattr(D.PublicSourceDiagnosticPolicy, "compiler_execution", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        FAST.CR,
        "run_capsule",
        lambda *a, **kw: {
            "status": "incomplete",
            "checked_source_status": "complete_source_L0_L1_L2_diagnostic_pass",
            "tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")},
        },
    )
    policy = D.prepare_public_source_diagnostic(**dict(args, corpus=corpus, contract_root=contract))
    store = RetainedRevisionStore(
        tmp_path / "view-revisions",
        edit_authority=policy.edit_authority,
        max_bytes=1 << 20,
        max_files=100,
        max_total_bytes=4 << 20,
    )
    revision = store.freeze(args["candidate"])
    baseline = store.freeze(args["baseline"])
    result = policy.evaluate_retained(
        revision,
        baseline,
        work_root=args["work_root"] / "published-view",
        call_index=0,
        timeout_s=120,
        selected_members=tuple(sorted(policy.validate_search_inputs())),
    )
    arm = result["cells"][0]["candidate"]
    assert arm["public_diagnostic_execution_view"] == binding["public_execution_views"][0]
    assert FAST.validate_feedback(result, policy=policy, revision=revision, kind=FAST.SELECTED_KIND) == result
    changed = copy.deepcopy(result)
    changed["cells"][0]["candidate"]["public_diagnostic_execution_view"]["schema"] = "forged"
    with pytest.raises(StageGateError, match="original public execution view"):
        FAST.validate_feedback(changed, policy=policy, revision=revision, kind=FAST.SELECTED_KIND)
    del changed["cells"][0]["candidate"]["public_diagnostic_execution_view"]
    with pytest.raises(StageGateError, match="private evidence digest"):
        FAST.validate_feedback(changed, policy=policy, revision=revision, kind=FAST.SELECTED_KIND)


@pytest.mark.parametrize("failure", [False, True])
def test_original_policy_completes_queue_publication_after_real_cleanup_protocol(
    inputs, tmp_path, monkeypatch, failure
):
    import test_evaluation_costs as COST_CONTROLS

    args, *_ = inputs
    policy = D.prepare_public_source_diagnostic(**args)
    # Only process/reaping boundaries are neutral. Exercise the original pool's
    # finalization code against the genuinely issued diagnostic policy object.
    fixture = COST_CONTROLS.parent.__wrapped__(tmp_path, monkeypatch)
    parent = next(fixture)
    try:
        parent.pool.policy = policy
        deadlines = []

        def validate_result(result, *_, deadline=None):
            deadlines.append(deadline)
            return copy.deepcopy(result)

        parent.pool.inputs.validate_result = validate_result
        job = parent.job()
        if failure:
            parent.pool._finish(job, error=StageGateError("independent current source refused"))
        else:
            parent.pool._finish(job)
        assert job.state == ("refused" if failure else "completed")
        assert deadlines == ([] if failure else [job.deadline])
        assert job.guardian_fd is None
        assert "release" in parent.events
        assert (job.root / "cleanup.json").is_file()
        if failure:
            assert (job.root / "refusal.json").is_file()
        else:
            assert (job.root / "feedback.json").is_file()
        assert policy.publish_completed_artifacts(job.root) == {"status": "UNAVAILABLE", "published": 0}
        assert policy.discard_cleaned_artifacts(job.root) == {"status": "UNAVAILABLE", "discarded": 0}
    finally:
        fixture.close()


def test_startup_binding_and_capsule_metadata_priority_keeps_original_deadline_and_environment(
    inputs, monkeypatch
):
    import time

    from merlin.common import cpu_pool as C

    args, *_ = inputs
    monkeypatch.setenv(C.PRIORITY_ENV, str(C.GRADE))
    original_pool = D.shared_compute_pool
    seen = []

    @contextmanager
    def original_metadata_leaf(*, timeout_s):
        seen.append((C.lease_priority(), os.environ.get(C.PRIORITY_ENV), timeout_s))
        assert C.lease_priority() == C.AUTHOR
        assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
        with original_pool(timeout_s=timeout_s):
            yield

    monkeypatch.setattr(D, "shared_compute_pool", original_metadata_leaf)
    with C.cpu_priority(C.GRADE):
        policy = D.prepare_public_source_diagnostic(**args)
        deadline = time.monotonic() + 30
        policy.binding(deadline=deadline)
        policy.load_capsule(args["corpus"].capsules[0], deadline=deadline)
        assert C.lease_priority() == C.GRADE
        assert os.environ[C.PRIORITY_ENV] == str(C.GRADE)
        assert not C.current_thread_owns_cpu_lease()
    assert len(seen) >= 6
    assert all(0 < wait <= 120 for _, _, wait in seen)
    assert any(wait < 30 for _, _, wait in seen)
