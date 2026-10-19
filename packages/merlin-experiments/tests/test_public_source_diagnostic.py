"""Original profile/corpus issuance and neutral queue controls; no target executes."""

import copy
import json
from contextlib import nullcontext
from dataclasses import replace

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

    from merlin.common.cpu_pool import current_thread_owns_cpu_lease

    args, *_ = inputs
    errors = []

    @contextmanager
    def execution(*a, **kw):
        assert current_thread_owns_cpu_lease()
        try:
            yield
        except RuntimeError as error:
            assert current_thread_owns_cpu_lease()
            errors.append(error)
            raise

    monkeypatch.setattr(R, "original_compiler_execution", execution)
    policy = D.prepare_public_source_diagnostic(**args)
    failure = RuntimeError("original observed body refusal")
    with pytest.raises(RuntimeError, match="body refusal"):
        with policy.compiler_execution(args["candidate"], args["work_root"] / "scope"):
            assert not current_thread_owns_cpu_lease()
            raise failure
    assert errors == [failure]


def test_screen_installation_and_expired_metadata_are_refused_before_execution(inputs):
    args, *_ = inputs
    policy = D.prepare_public_source_diagnostic(**args)
    with pytest.raises(StageGateError, match="aggregate deadline"):
        policy.validate_search_inputs(deadline=-1)
    policy._ordinary_screen = object()
    with pytest.raises(StageGateError, match="original execution owner"):
        policy.validate_search_inputs()


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
