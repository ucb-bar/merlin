"""Live explicit fast selection and original parent reopening; no engine starts."""

import copy
import time
from contextlib import nullcontext

import pytest
import test_public_source_diagnostic as PUBLIC
from merlin_experiments.phase2 import campaign as CAMPAIGN
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_numerical_counts as N
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2 import search_inputs as SI
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.retained_revisions import RetainedRevisionStore

from merlin.targetgen import checked_source_model_diagnostic as S


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    return PUBLIC.inputs.__wrapped__(tmp_path, monkeypatch)


def fast_selection(args):
    old = args["checked_source_diagnostic"].record()
    return S.select_checked_source_diagnostic(
        args["compiler_pass_selection"], target=old["target"],
        rtl_facts=old["rtl_facts"]["path"], board_config=old["board_config"],
        verification_protocol=S.NUMERICAL_COUNTS_PROTOCOL,
    )


def grade(selection):
    return dict(
        schema=S.FAST_RESULT_SCHEMA, checked_source_status=S.FAST_STATUS,
        checked_source_diagnostic=selection.record(), verification_protocol=S.NUMERICAL_COUNTS_PROTOCOL,
        source_effects={"status": "PENDING"}, status="incomplete", formal_admission=False,
        promotion_authority="NONE", incumbent_authority="NONE", roofline_convergence_authority="NONE",
        timing_cycles=None, tiers={t: {"status": "pass"} for t in ("L0", "L1", "L2")},
    )


def run_feedback(inputs, tmp_path, monkeypatch, *, fast=True, mutate_grade=None):
    args, *_ = inputs
    args = dict(args)
    if fast:
        args["checked_source_diagnostic"] = fast_selection(args)
    calls, reopened = [], []
    monkeypatch.setattr(FAST.CR, "simulator_adapter", lambda *a: "neutral-engine")
    monkeypatch.setattr(CAMPAIGN, "package_sandbox_policy", lambda *a: "neutral-box")
    monkeypatch.setattr(CAMPAIGN, "boxed_entrypoints", lambda *a: nullcontext())
    monkeypatch.setattr(FAST, "compiled_observation", lambda *a, **kw: {"status": "UNKNOWN"})
    monkeypatch.setattr(FAST, "observe_passed_census", lambda *a, **kw: {"status": "UNKNOWN"})
    monkeypatch.setattr(D.PublicSourceDiagnosticPolicy, "compiler_execution", lambda *a, **kw: nullcontext())

    def run(cap, package, **kwargs):
        selection = kwargs["checked_source_diagnostic"]
        selection.verify_for(cap, target=args["target_experiment"].target)
        calls.append(package)
        result = grade(selection) if fast else dict(
            status="incomplete", checked_source_status=S.FULL_STATUS,
            tiers={t: {"status": "pass"} for t in ("L0", "L1", "L2")},
        )
        if mutate_grade:
            mutate_grade(result)
        return result

    def reopen(directory, value, *, checked_source):
        assert checked_source == (args["checked_source_diagnostic"], args["compiler_pass_selection"])
        assert value["functional_status"] == "PASS"
        reopened.append(directory)

    monkeypatch.setattr(FAST.CR, "run_capsule", run)
    monkeypatch.setattr(N.NATIVE, "original_numerical_observation", reopen)
    policy = D.prepare_public_source_diagnostic(**args)
    store = RetainedRevisionStore(
        tmp_path / "revisions", edit_authority=policy.edit_authority,
        max_bytes=1 << 20, max_files=100, max_total_bytes=4 << 20,
    )
    baseline = store.freeze(args["baseline"])
    (args["candidate"] / "compiler.py").write_text("def choose():\n return 1\n")
    candidate = store.freeze(args["candidate"], parent_revision=baseline.revision_id)
    selected = tuple(sorted(policy.validate_search_inputs()))
    root = args["work_root"] / "job"
    result = policy.evaluate_retained(
        candidate, baseline, work_root=root, call_index=0, timeout_s=30, selected_members=selected,
    )
    return policy, candidate, selected, root, result, calls, reopened


def test_explicit_fast_queue_is_partial_and_parent_reopens_both_arms_without_counts(inputs, tmp_path, monkeypatch):
    policy, candidate, selected, root, result, calls, reopened = run_feedback(inputs, tmp_path, monkeypatch)
    assert result["schema"] == D.NUMERICAL_COUNTS_FEEDBACK_SCHEMA
    assert len(calls) == 2 * len(selected) and not reopened
    assert result["source_effects"]["status"] == "PENDING"
    assert result["source_effects"]["roofline_convergence_authority"] == "NONE"
    assert result["promotion"] == "DIAGNOSTIC_ONLY"
    assert SI.SearchInputs(policy).validate_result(result, candidate, FAST.SELECTED_KIND) == result
    policy.consume_feedback_child_result(
        result, work_root=root, selected_members=selected, deadline=time.monotonic() + 30,
    )
    assert len(reopened) == 2 * len(selected)
    assert {p.name.rsplit("_", 1)[1] for p in reopened} == {"baseline", "candidate"}


@pytest.mark.parametrize("fault", ["complete_effects", "authority", "timing", "schema", "arm_projection"])
def test_fast_feedback_cannot_upgrade_or_erase_pending_effect_obligations(inputs, tmp_path, monkeypatch, fault):
    policy, candidate, _, _, result, *_ = run_feedback(inputs, tmp_path, monkeypatch)
    result = copy.deepcopy(result)
    if fault == "complete_effects":
        result["source_effects"]["status"] = "PASS"
    elif fault == "authority":
        result["source_effects"]["promotion_authority"] = "PROMOTE"
    elif fault == "timing":
        result["source_effects"]["timing_cycles"] = 1
    elif fault == "schema":
        result["schema"] = D.FEEDBACK_SCHEMA
    else:
        result["cells"][0]["candidate"].pop("source_effects")
    with pytest.raises(StageGateError):
        SI.SearchInputs(policy).validate_result(result, candidate, FAST.SELECTED_KIND)


def test_new_fast_claim_under_old_live_selection_refuses(inputs, tmp_path, monkeypatch):
    def forge(value):
        value.update(schema=S.FAST_RESULT_SCHEMA, checked_source_status=S.FAST_STATUS,
                     verification_protocol=S.NUMERICAL_COUNTS_PROTOCOL)

    with pytest.raises(StageGateError, match="live verification selection"):
        run_feedback(inputs, tmp_path, monkeypatch, fast=False, mutate_grade=forge)


def test_old_live_selection_preserves_default_feedback_and_has_no_pending_projection(inputs, tmp_path, monkeypatch):
    policy, candidate, selected, root, result, _, reopened = run_feedback(inputs, tmp_path, monkeypatch, fast=False)
    assert result["schema"] == D.FEEDBACK_SCHEMA and "source_effects" not in result
    assert SI.SearchInputs(policy).validate_result(result, candidate, FAST.SELECTED_KIND) == result
    policy.consume_feedback_child_result(
        result, work_root=root, selected_members=selected, deadline=time.monotonic() + 30,
    )
    assert not reopened


@pytest.mark.parametrize("fault", ["foreign_root", "missing_arm", "source", "roster", "pending_projection"])
def test_parent_fast_reopening_refuses_changed_source_custody(inputs, tmp_path, monkeypatch, fault):
    policy, _, selected, root, result, _, reopened = run_feedback(inputs, tmp_path, monkeypatch)
    result = copy.deepcopy(result)
    if fault == "foreign_root":
        root = tmp_path / "foreign"
        root.mkdir()
    elif fault == "missing_arm":
        result["cells"][0]["baseline"] = None
    elif fault == "source":
        result["cells"][0]["source_sha256"] = "0" * 64
    elif fault == "roster":
        result["selected_members"] = []
    else:
        result["source_effects"]["status"] = "PASS"
    with pytest.raises(StageGateError):
        policy.consume_feedback_child_result(
            result, work_root=root, selected_members=selected, deadline=time.monotonic() + 30,
        )
    assert not reopened
