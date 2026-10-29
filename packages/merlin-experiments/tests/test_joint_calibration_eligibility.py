"""Calibration cannot turn numerical success into structural qualification."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from merlin_experiments.phase1 import joint_calibration_eligibility as E


def capsule():
    return dict(name="public", kind="layer", label="public", required_oracle_tiers=["L0","L1","L2","L3"])


def valid_grade(*, pending=False, source="c"*64):
    row = dict(capsule="public", kind="layer", label="public", status="pass",
        numeric=dict(status="pass", mismatch_count=0, missing_outputs=[], golden_source="public semantics",
                     per_output={"y":{"status":"pass","n_elements":4}}),
        tiers={t:{"status":"pass", "mandatory":True} for t in ("L0","L1","L2","L3")},
        trace_check=dict(status="pass", violations=[], trace_requirement_status="pass", unavailable_requirements=[]),
        compiler_source=dict(status="stable", candidate_sha256=source))
    row["tiers"]["L2"]["engine"] = "spike"
    if pending:
        reason = "this capsule declares required oracle tier L3 (elaborated_rtl) but this phase supplies no adapter for it; reachable tiers here: ['L2']"
        row["status"] = "incomplete"
        row["tiers"]["L3"] = dict(status="unavailable", mandatory=True, reason=reason)
        row["failure"] = dict(plane="oracle_unavailable", category="NOT_RUN_IS_NOT_PASS", tier="L3",
                              tier_status="unavailable", tier_reason=reason)
    return row


@pytest.mark.parametrize("pending", [False, True])
def test_complete_or_exact_pending_required_l3_admitted(pending):
    row = valid_grade(pending=pending)
    assert E.eligible(row, capsule(), compiler_sha256="c"*64)
    E.require(row, capsule(), compiler_sha256="c"*64)
    assert not E.eligible(row, capsule(), compiler_sha256="d"*64)
    if pending:
        assert not E.eligible(row, dict(capsule(), required_oracle_tiers=["L0","L1","L2"]))


@pytest.mark.parametrize("damage", ["overall_fail", "placement_failure", "trace_failure", "trace_violation",
    "trace_missing", "requirements_missing", "placement_audit", "nested_placement", "malformed_placement",
    "other_tier_fail", "missing_required", "source_missing", "numeric_failure", "l0_missing"])
@pytest.mark.parametrize("pending", [False, True])
def test_numeric_pass_does_not_hide_independent_failure(damage, pending):
    row = valid_grade(pending=pending)
    if damage == "overall_fail": row["status"] = "fail"
    elif damage == "placement_failure": row["failure"] = dict(plane="placement", category="SOURCE_PLACEMENT")
    elif damage == "trace_failure": row["trace_check"]["status"] = "fail"
    elif damage == "trace_violation": row["trace_check"]["violations"] = ["required accelerator operation ran on host"]
    elif damage == "trace_missing": row.pop("trace_check")
    elif damage == "requirements_missing": row["trace_check"]["unavailable_requirements"] = ["placement"]
    elif damage == "placement_audit": row["placement"] = {"status":"violation"}
    elif damage == "nested_placement": row["model_execution_check"] = {"candidate_source_placement":{"status":"violation"}}
    elif damage == "malformed_placement": row["source_placement"] = True
    elif damage == "other_tier_fail": row["tiers"]["L1"]["status"] = "fail"
    elif damage == "missing_required": row["tiers"].pop("L2")
    elif damage == "source_missing": row.pop("compiler_source")
    elif damage == "numeric_failure": row["numeric"]["status"] = "fail"
    else: row["tiers"].pop("L0")
    assert not E.eligible(row, capsule(), compiler_sha256="c"*64)


@pytest.mark.parametrize("damage", ["wrong_reason", "not_mandatory", "wrong_tier", "missing_output", "failed_output", "missing_count"])
def test_pending_l3_exception_is_exact(damage):
    row = valid_grade(pending=True)
    if damage == "wrong_reason": row["tiers"]["L3"]["reason"] = "engine unavailable"
    elif damage == "not_mandatory": row["tiers"]["L3"]["mandatory"] = False
    elif damage == "wrong_tier": row["failure"]["tier"] = "L1"
    elif damage == "missing_output": row["numeric"]["missing_outputs"] = ["y"]
    elif damage == "failed_output": row["numeric"]["per_output"]["y"]["status"] = "fail"
    else: row["numeric"].pop("mismatch_count")
    assert not E.eligible(row, capsule())


def test_exact_whole_program_nonapplicability_preserved():
    row = valid_grade(pending=True)
    for tier, reason in {
        "L0":"command-buffer reference cannot interpret scalar regions of a whole-program ELF",
        "L1":"command-buffer simulate cannot execute scalar regions of a whole-program ELF"}.items():
        row["tiers"][tier] = dict(status="skipped", not_applicable=True, reason=reason, mandatory=True)
    assert E.eligible(row, capsule())
    row["tiers"]["L0"]["reason"] = "unsupported"
    assert not E.eligible(row, capsule())


def test_replay_rejects_bad_result_before_product_or_runtime_work(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_calibration as C
    from merlin_experiments.phase1.feedback import candidate_observations
    from merlin.benchharness import hash_tree
    compiler = tmp_path / "compiler"; compiler.mkdir(); (compiler / "source.py").write_text("pass")
    digest = hash_tree(compiler)["sha256"]
    grade = valid_grade(source=digest); grade["trace_check"]["status"] = "fail"
    monkeypatch.setattr(C.PERF, "_read", lambda pin: grade)
    monkeypatch.setattr(C, "_capsule", lambda pin: capsule())
    monkeypatch.setattr(candidate_observations, "observe_generated_artifacts",
                        lambda *a, **k: pytest.fail("bad calibration source must stop before product replay"))
    case = dict(generated_root=str(tmp_path), compiler=str(compiler), compiler_sha256=digest,
                result={}, capsule={})
    with pytest.raises(ValueError, match="structurally admitted"):
        C._observe(case, runtime=None, context=None, directory=tmp_path / "unused")


@pytest.mark.parametrize("pending,bad", [(False,True),(True,False)])
def test_freeze_checks_structural_admission_and_allows_only_valid_pending_l3(tmp_path, monkeypatch, pending, bad):
    from merlin_experiments.phase1 import joint_calibration as C
    from merlin_experiments.phase1.feedback import joint_timing_selection
    selection = {"runtime":{"path":"runtime"}, "facts":{}}
    grade = valid_grade(pending=pending)
    if bad: grade["placement"] = {"status":"violation"}
    model = {"prices":{"compute":{"latency_cycles":1}}}
    def read(pin):
        return {"selection":selection, "runtime":{"facts_alignment":{"path":"alignment"}},
                "alignment":{"mappings":[]}, "grade":grade}[pin["path"]]
    monkeypatch.setattr(C.PERF, "_read", read)
    monkeypatch.setattr(C.PERF, "_runtime", lambda *a, **kw: SimpleNamespace(model_json=json.dumps(model)))
    monkeypatch.setattr(C, "_capsule", lambda pin: capsule())
    monkeypatch.setattr(C.joint_control, "selected", lambda: {"policy":{"gsim":{"witness_capsules":["public"]}}})
    monkeypatch.setattr(joint_timing_selection, "choose", lambda *a: {"runtime":{"path":"runtime"}})
    monkeypatch.setattr(C.PERF, "selected_capsule", lambda selection, cap: cap)
    rows = [dict(id=f"case{i}", split="train" if i < 2 else "held", source_group_sha256=str(i),
                 capsule={}, result={"path":"grade"}, compiler_sha256="c"*64) for i in range(4)]
    kwargs = dict(selection_pin={"path":"selection"}, cases=rows,
        parameters=[dict(path=["prices","compute","latency_cycles"], values=[1,2], mechanism="test",reviewed_scope="test")],
        output=tmp_path/"plan.json", question="test bounded eligibility")
    if bad:
        with pytest.raises(ValueError, match="structurally admitted"): C.freeze_round(**kwargs)
        assert not kwargs["output"].exists()
    else:
        assert C.freeze_round(**kwargs)["sha256"]


def test_resumed_round_rechecks_old_results_before_cached_outcome(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_calibration as C, joint_calibration_admission
    grade = valid_grade(); grade["trace_check"]["violations"] = ["placement failure"]
    plan = dict(schema=C.SCHEMA, gates=dict(C.GATES), cases=[dict(result={"path":"grade"}, capsule={}, compiler_sha256="c"*64)])
    monkeypatch.setattr(C.PERF, "_read", lambda pin: plan if pin['path']=='plan' else grade)
    monkeypatch.setattr(C, "_capsule", lambda pin: capsule())
    monkeypatch.setattr(joint_calibration_admission, "assert_allowed", lambda plan: None)
    (tmp_path/'outcome.json').write_text('{"status":"previous outcome"}')
    with pytest.raises(ValueError, match="structurally admitted"):
        C.run_round({'path':'plan'}, context=None, backend=None, output=tmp_path)
