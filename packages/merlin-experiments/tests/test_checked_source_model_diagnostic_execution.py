"""Synthetic consumption controls; independent target correctness is not mocked as qualified."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import test_checked_source_model_diagnostic as SELECTION

from merlin.targetgen import checked_source_model_diagnostic_execution as E
from merlin.targetgen.compiler_pass_selection import scoped_compiler_passes


@pytest.fixture
def execution(tmp_path, monkeypatch):
    passes, selection, issuer, _facts, capsule = SELECTION.selected.__wrapped__(tmp_path, monkeypatch)
    capsule["__dir__"] = str(tmp_path)
    native = dict(original_completed_native_record="synthetic")
    native_path = tmp_path / "native.json"
    native_path.write_text(json.dumps(native))
    result = dict(candidate_native_execution=native, candidate_emission={"original": "emission"},
                  candidate_source_eligibility={"n_eligible": 1, "synthetic": "source-only certificate"})
    tiers = dict(status="unverified", required_tiers=["L0", "L1", "L2", "L3"], missing_tiers=["L3"],
                 tiers={t: dict(status="pending" if t == "L3" else "pass") for t in ("L0", "L1", "L2", "L3")})
    monkeypatch.setattr(E.NATIVE, "audit_candidate_tiers", lambda *_a, **_kw: copy.deepcopy(tiers))
    monkeypatch.setattr(E.NATIVE, "_pinned_native_file", lambda _pins, role: tmp_path / role)
    from merlin.runtime.backends import base as backends

    monkeypatch.setattr(backends, "harness_build_recipe", lambda _target: SimpleNamespace(
        require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="selected_entry")))
    active = []

    class Lease:
        def __enter__(self):
            active.append(True)

        def __exit__(self, *args):
            active.pop()

    monkeypatch.setattr(E, "shared_compute_pool", lambda **_kw: Lease())
    class SyntheticRealization:
        def revalidate(self):
            pass

        def source_rule(self):
            return {}

        def verify_source_execution(self, *_a):
            pytest.fail("the transport fixture has no independent target execution authority")

    issuer.realization = SyntheticRealization()
    monkeypatch.setattr(E.SOURCE, "scoped_realization_owner", lambda owner: nullcontext(owner))

    def qualify(_entry, _capsule, _directory):
        assert active == [True]
        assert len(issuer.calls) == 1
        return dict(status="qualified", authority="synthetic consumer wiring only")

    monkeypatch.setattr(E.SOURCE, "qualify", qualify)
    return SimpleNamespace(passes=passes, selection=selection, issuer=issuer, capsule=capsule,
                           native=native_path, result=result, tiers=tiers, active=active, root=tmp_path)


def finish(x, result=None, remaining=60):
    with scoped_compiler_passes(x.passes):
        return E.finish(x.selection, x.result if result is None else result, x.capsule,
                        target="fixture", native_result=x.native, work_root=x.root / "new-actual-trace",
                        remaining_s=remaining)


def test_full_checked_consumption_stays_diagnostic_and_retains_required_rtl(execution):
    x = execution
    result = finish(x)
    assert result["checked_source_status"] == "complete_source_L0_L1_L2_diagnostic_pass"
    assert result["status"] == "incomplete" and result["formal_admission"] is False
    assert result["promotion_authority"] == "NONE" and result["timing_cycles"] is None
    assert result["tiers"]["L3"]["status"] == "pending"
    assert result["original_tier_audit"]["required_tiers"] == ["L0", "L1", "L2", "L3"]
    assert x.active == [] and len(x.issuer.calls) == 1
    assert x.issuer.calls[0]["native_result"] == x.native


@pytest.mark.parametrize("role", ["L0", "L1", "L2"])
def test_original_compile_or_complete_output_failure_cannot_borrow_effect_pass(execution, role):
    x = execution
    x.tiers["tiers"][role]["status"] = "fail"
    result = finish(x)
    assert "checked_source_status" not in result and x.issuer.calls == []
    assert result["tiers"][role]["status"] == "fail"


def test_copied_success_cannot_survive_missing_actual_native_products(execution):
    x = execution
    copied = dict(complete_source_qualification={"status": "qualified"},
                  checked_source_status="complete_source_L0_L1_L2_diagnostic_pass",
                  tiers=x.tiers["tiers"])
    result = finish(x, copied)
    assert "checked_source_status" not in result and "complete_source_qualification" not in result
    assert "tiers" not in result and x.issuer.calls == []


def test_foreign_native_record_or_changed_original_roles_refuse(execution, monkeypatch):
    x = execution
    x.native.write_text('{"other":"program"}')
    with pytest.raises(ValueError, match="exact original native"):
        finish(x)
    x.native.write_text(json.dumps(x.result["candidate_native_execution"]))
    count = []

    def changed(*_a, **_kw):
        count.append(True)
        result = copy.deepcopy(x.tiers)
        if len(count) > 1:
            result["tiers"]["L2"]["status"] = "fail"
        return result

    monkeypatch.setattr(E.NATIVE, "audit_candidate_tiers", changed)
    with pytest.raises(ValueError, match="original tiers"):
        finish(x)


def test_actual_source_output_refusal_is_not_replaced_by_a_saved_verdict(execution, monkeypatch):
    x = execution

    def refused(*_a):
        raise ValueError("complete actual source outputs differ")

    monkeypatch.setattr(E.SOURCE, "qualify", refused)
    x.result["complete_source_qualification"] = {"status": "qualified"}
    with pytest.raises(ValueError, match="complete actual source outputs"):
        finish(x)


@pytest.mark.parametrize("eligible", [None, 0, -1, True, "1"])
def test_host_only_or_unowned_source_never_gets_accelerator_diagnostic_credit(execution, eligible):
    execution.result["candidate_source_eligibility"]["n_eligible"] = eligible
    result = finish(execution)
    assert "checked_source_status" not in result and execution.issuer.calls == []


def test_source_denominator_failure_stays_explicit_diagnostic_without_legacy_fallback(execution, monkeypatch):
    from merlin.targetgen import capsule_runner as runner

    x = execution
    x.capsule.update(name="synthetic-source", operation={"attributes": {}})

    def refuse(*_a, **_kw):
        raise ValueError("independent original source denominator is missing")

    monkeypatch.setattr(E.NATIVE, "independent_frozen_source_eligibility", refuse)
    with scoped_compiler_passes(x.passes):
        result = runner._grade_public_program_capsule_inline(
            x.capsule, target="fixture", timeout=60, package_dir=x.root, context={},
            diagnostic_selection=x.selection)
    assert result["status"] == "incomplete" and result["formal_admission"] is False
    assert "candidate_source_eligibility_failure" in result and "checked_source_status" not in result
    assert x.issuer.calls == []


@pytest.mark.parametrize("remaining", [0, -1, True, float("nan"), float("inf")])
def test_unknown_or_expired_deadline_never_admits_an_effect_job(execution, remaining):
    with pytest.raises(TimeoutError):
        finish(execution, remaining=remaining)
    assert execution.issuer.calls == []


def test_explicit_pointer_selection_uses_its_audit_on_both_sides_of_effect_check(execution, monkeypatch):
    import sys

    from merlin.targetgen import checked_source_model_diagnostic as S

    x = execution
    x.selection = S.select_checked_source_diagnostic(
        x.passes, target="fixture", rtl_facts=x.root / "facts.json", board_config="config",
        native_execution_protocol=S.POINTER_PROTOCOL,
    )
    calls = []

    def audit(emission, native, **kwargs):
        calls.append((emission, native, kwargs))
        return copy.deepcopy(x.tiers)

    monkeypatch.setitem(sys.modules, "merlin.targetgen.native_pointer_source_audit",
                        SimpleNamespace(audit_pointer_source_tiers=audit))
    monkeypatch.setattr(E.NATIVE, "audit_candidate_tiers", lambda *_a, **_kw: pytest.fail("wrong ABI audit"))
    result = finish(x)
    assert len(calls) == 2 and len(x.issuer.calls) == 1
    assert result["tiers"]["L3"]["status"] == "pending"
    assert result["promotion_authority"] == "NONE"


def test_native_result_field_cannot_select_pointer_audit(execution, monkeypatch):
    import sys

    from merlin.targetgen import checked_source_model_diagnostic as S

    x = execution
    x.result["candidate_native_execution"]["native_execution_protocol"] = S.POINTER_PROTOCOL
    x.native.write_text(json.dumps(x.result["candidate_native_execution"]))
    monkeypatch.setitem(sys.modules, "merlin.targetgen.native_pointer_source_audit", SimpleNamespace(
        audit_pointer_source_tiers=lambda *_a, **_kw: pytest.fail("result chose its own ABI audit")))
    result = finish(x)
    assert result["checked_source_status"] == "complete_source_L0_L1_L2_diagnostic_pass"


def test_explicit_live_owner_intake_is_passed_to_each_original_audit(execution, monkeypatch):
    from merlin.targetgen.native_pointer_source_binding import LARGE_INTAKE_PROTOCOL

    x, calls = execution, []
    monkeypatch.setattr(x.issuer, "record", lambda: dict(source_intake_protocol=LARGE_INTAKE_PROTOCOL), raising=False)

    def audit(*_args, **kwargs):
        calls.append(kwargs)
        return copy.deepcopy(x.tiers)

    monkeypatch.setattr(E.NATIVE, "audit_candidate_tiers", audit)
    finish(x)
    assert len(calls) == 2
    assert all(row["source_intake_protocol"] == LARGE_INTAKE_PROTOCOL for row in calls)


def fast_selection(x, *, native=None):
    from merlin.targetgen import checked_source_model_diagnostic as S

    x.selection = S.select_checked_source_diagnostic(x.passes, target="fixture", rtl_facts=x.root / "facts.json",
        board_config="config", native_execution_protocol=native, verification_protocol=S.NUMERICAL_COUNTS_PROTOCOL)
    return x


def test_fast_diagnostic_reopens_original_numerics_and_never_creates_effect_owner(execution, monkeypatch):
    from merlin.targetgen import checked_source_model_diagnostic as S

    x = fast_selection(execution)
    monkeypatch.setattr(E.SOURCE, "qualify", lambda *_a, **_k: pytest.fail("full source effects executed"))
    result = finish(x)
    assert result["schema"] == S.FAST_RESULT_SCHEMA and result["checked_source_status"] == S.FAST_STATUS
    assert S.numerical_grade_matches_selection(x.selection.record(), result)
    assert result["source_effects"]["status"] == "PENDING" and result["status"] == "incomplete"
    assert result["formal_admission"] is False and result["timing_cycles"] is None
    assert result["incumbent_authority"] == result["roofline_convergence_authority"] == "NONE"
    assert result["promotion_authority"] == "NONE" and result["tiers"]["L3"]["status"] == "pending"
    assert "complete_source_qualification" not in result and x.issuer.calls == []


@pytest.mark.parametrize("tier", ["L0", "L1", "L2"])
def test_fast_diagnostic_preserves_every_original_tier_guard(execution, tier):
    x = fast_selection(execution)
    x.tiers["tiers"][tier]["status"] = "fail"
    result = finish(x)
    assert "checked_source_status" not in result and x.issuer.calls == []
    assert result["source_effects"]["status"] == "PENDING"


def test_fast_diagnostic_preserves_source_eligibility_and_exact_native_record(execution):
    x = fast_selection(execution)
    x.result["candidate_source_eligibility"]["n_eligible"] = 0
    assert "checked_source_status" not in finish(x)
    x.result["candidate_source_eligibility"]["n_eligible"] = 1
    x.native.write_text("{}")
    with pytest.raises(ValueError, match="exact original native"):
        finish(x)


def test_fast_diagnostic_requires_unchanged_reopened_final_tiers(execution, monkeypatch):
    x, calls = fast_selection(execution), []

    def audit(*_args, **_kwargs):
        calls.append(True)
        result = copy.deepcopy(x.tiers)
        if len(calls) == 2:
            result["tiers"]["L2"]["status"] = "fail"
        return result

    monkeypatch.setattr(E.NATIVE, "audit_candidate_tiers", audit)
    with pytest.raises(ValueError, match="original tiers or native products changed"):
        finish(x)
    assert len(calls) == 2 and x.issuer.calls == []


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(checked_source_status="complete_source_L0_L1_L2_diagnostic_pass"),
    lambda row: row.update(complete_source_qualification={"status": "qualified"}),
    lambda row: row["source_effects"].update(status="PASS"),
    lambda row: row.update(promotion_authority="allowed"),
    lambda row: row.update(incumbent_authority="allowed"),
    lambda row: row.update(roofline_convergence_authority="allowed"),
    lambda row: row.update(formal_admission=True),
])
def test_fast_receipt_cannot_borrow_complete_effect_or_promotion_authority(execution, mutation):
    from merlin.targetgen import checked_source_model_diagnostic as S

    x = fast_selection(execution)
    result = finish(x)
    mutation(result)
    assert not S.numerical_grade_matches_selection(x.selection.record(), result)


def test_saved_fast_status_cannot_satisfy_old_full_selection(execution):
    from merlin.targetgen import checked_source_model_diagnostic as S

    old = execution.selection
    x = fast_selection(execution)
    result = finish(x)
    assert not S.numerical_grade_matches_selection(old.record(), result)
    result["checked_source_status"] = S.FULL_STATUS
    assert not S.numerical_grade_matches_selection(old.record(), result)
