"""The prohibited instruction roles, judged on a capsule's decoded EXECUTED command stream.

The trace gate decodes every graded kernel's emitted program into a command stream with per-site
execution counts. A run that prohibits a role (the joint campaign's policy, the run's environment or the
capsule's sealed arm) fails a case whose stream executes a command with that role at L2, naming the role,
the command and its command index; an ordinary stream passes; a stream whose roles cannot be told is
UNKNOWN and never passes.

The target, its selectors and its role table are synthetic facts patched in; nothing here is a real
encoding.
"""

from __future__ import annotations

import json

import pytest
from merlin_experiments.phase1 import joint_control as JC

from merlin.perf import isa_prohibition as IP
from merlin.perf import task_instruction_evidence as TIE
from merlin.perf.whole_model_gate import ROLES_ENV
from merlin.targetgen import capsule_instruction_policy as CIP
from merlin.targetgen import capsule_runner as R
from merlin.targetgen.package_runtime import CertFailure

TARGET = "toy-accelerator"
#: The synthetic role table (made up; the rule derives the prohibited selectors from it).
FACTS = {
    "instruction_names": {"1": "MOVE", "4": "STEP", "5": "NEST", "9": "WAIT"},
    "roles_by_selector": {"1": ["operand_load"], "4": ["accumulate"], "5": ["loop_descriptor"], "9": ["sync"]},
}


def _row(index, funct, executions=1, cls=None):
    names = FACTS["instruction_names"]
    return {"index": index, "class": cls or names.get(str(funct), "UNKNOWN"), "funct": funct, "executions": executions}


FENCE = {"index": 0, "class": "FENCE", "funct": None, "decoded": {}}
ORDINARY = {"instructions": [FENCE, _row(1, 1, 8), _row(2, 4, 64), _row(3, 9), {**FENCE, "index": 4}]}
LOOPED = {"instructions": [FENCE, _row(1, 1), _row(2, 5, 3), _row(3, 4), _row(4, 5, 3), {**FENCE, "index": 5}]}


@pytest.fixture()
def synthetic(monkeypatch, tmp_path):
    """Synthetic facts, no environment roles, and a joint campaign whose policy prohibits the role."""
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: FACTS)
    monkeypatch.delenv(ROLES_ENV, raising=False)
    state = tmp_path / "joint_campaign.json"
    state.write_text(json.dumps({"policy": {"version": 1, "prohibited_instruction_roles": ["loop_descriptor"]}}))
    monkeypatch.setenv(JC.ENV, str(state))


def test_mutation_a_loop_command_in_the_executed_stream_fails_at_l2(synthetic):
    tiers = {}
    with pytest.raises(CertFailure) as raised:
        CIP.enforce_executed_stream({"name": "cap"}, LOOPED, target=TARGET, tiers=tiers)
    error = raised.value
    assert error.detail.startswith("uses prohibited instruction role loop_descriptor: NEST at command index 2 ")
    assert "(executed 3x); 1 more prohibited command site(s): NEST" in error.detail
    assert error.plane == "instruction_policy" and error.facts["tier"] == "L2"
    assert tiers["L2"].status == "fail" and tiers["L2"].reason in error.detail
    assert error.facts["instruction_policy"]["hit_count"] == 2
    assert error.facts["instruction_policy"]["prohibited"] == {"5": "NEST"}


def test_an_ordinary_stream_passes_and_keeps_its_verdict(synthetic):
    report = CIP.enforce_executed_stream({"name": "cap"}, ORDINARY, target=TARGET, tiers={})
    assert report["status"] == "measured" and report["clean"] is True and report["hits"] == []
    assert CIP.apply_executed_stream_rule("pass", None, {"instruction_policy": report}) == ("pass", None)


def test_a_site_proved_never_to_run_is_not_executed_but_an_unproved_one_is(synthetic):
    dead = {"instructions": [FENCE, _row(1, 4), _row(2, 5, 0)]}
    report = IP.stream_report(dead, target=TARGET, roles=["loop_descriptor"])
    assert report["clean"] is True and report["unexecuted_sites"][0]["index"] == 2
    unproved = {"instructions": [FENCE, _row(1, 5, None)]}
    report = IP.stream_report(unproved, target=TARGET, roles=["loop_descriptor"])
    assert report["clean"] is False and "(executions unproved)" in IP.stream_reason(report)


def test_mutation_an_underivable_role_mapping_is_unknown_never_a_pass(synthetic, monkeypatch):
    unroled = {**FACTS, "roles_by_selector": {k: [] for k in FACTS["roles_by_selector"]}}
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: unroled)
    report = CIP.enforce_executed_stream({"name": "cap"}, LOOPED, target=TARGET, tiers={})
    assert report["status"] == IP.UNKNOWN and report["clean"] is None
    status, failure = CIP.apply_executed_stream_rule("pass", None, {"instruction_policy": report})
    assert status == "incomplete" and failure["category"] == "PROHIBITION_NOT_MEASURED"
    assert failure["tier"] == "L2" and "UNKNOWN" in failure["detail"]

    def broken(_target):
        raise FileNotFoundError("no facts for this target")

    monkeypatch.setattr(TIE, "target_instruction_facts", broken)
    report = IP.stream_report(ORDINARY, target=TARGET, roles=["loop_descriptor"])
    assert report["status"] == IP.UNKNOWN and "no facts for this target" in report["detail"]


def test_a_command_whose_role_cannot_be_told_is_unknown_unless_a_hit_settles_it(synthetic):
    undecoded = {"index": 3, "class": "UNKNOWN", "funct": None, "raw": ".word 0", "decoded": {}}
    report = IP.stream_report(
        {"instructions": [*ORDINARY["instructions"], undecoded]}, target=TARGET, roles=["loop_descriptor"]
    )
    assert report["status"] == IP.UNKNOWN and "command index 3" in report["detail"]
    undeclared = IP.stream_report({"instructions": [_row(1, 77)]}, target=TARGET, roles=["loop_descriptor"])
    assert undeclared["status"] == IP.UNKNOWN
    settled = IP.stream_report({"instructions": [undecoded, _row(4, 5)]}, target=TARGET, roles=["loop_descriptor"])
    assert settled["clean"] is False


def test_a_capsule_held_to_no_role_is_not_judged(monkeypatch):
    monkeypatch.delenv(ROLES_ENV, raising=False)
    monkeypatch.delenv(JC.ENV, raising=False)
    assert CIP.enforce_executed_stream({"name": "cap"}, LOOPED, target=TARGET, tiers={}) is None
    assert CIP.apply_executed_stream_rule("pass", None, {"status": "pass", "violations": []}) == ("pass", None)


def test_the_roles_join_the_campaign_policy_the_environment_and_the_sealed_arm(synthetic, monkeypatch):
    assert CIP.executed_stream_roles({"name": "cap"}) == ("loop_descriptor",)
    monkeypatch.setenv(ROLES_ENV, "sync")
    sealed = {"performance": {"arms": {"candidate": {"instruction_policy": {"prohibited_instruction_roles": ["dma"]}}}}}
    assert CIP.executed_stream_roles(sealed) == ("dma", "loop_descriptor", "sync")


def test_the_whole_elf_rule_keeps_the_stream_reason_first():
    stream = {"plane": "instruction_policy", "category": "forbidden_pattern", "tier": "L2", "detail": "uses ..."}
    elf = {
        "clean": False,
        "roles": ["loop_descriptor"],
        "summary": {"NEST": 2},
        "scans": [{"prohibited": {"5": "NEST"}}],
    }
    status, failure = R.apply_prohibited_instruction_rule("fail", stream, elf)
    assert status == "fail" and failure["detail"] == "uses ..." and failure["tier"] == "L2"
    assert failure["whole_elf"] == {"summary": {"NEST": 2}, "prohibited": {"5": "NEST"}}


def test_the_finalizer_never_passes_an_unknown_stream(synthetic, monkeypatch, tmp_path):
    from merlin.targetgen.capsule_common import make_run_paths

    unroled = {**FACTS, "roles_by_selector": {k: [] for k in FACTS["roles_by_selector"]}}
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: unroled)
    report = IP.stream_report(ORDINARY, target=TARGET, roles=["loop_descriptor"])
    paths = make_run_paths(tmp_path / "runs", "cap", suite="t", target=TARGET, dtype="fp32", benchmark="cap")
    row = R._finalize_capsule_result(
        name="cap",
        capsule={"name": "cap", "kind": "isa", "label": "public"},
        status="pass",
        failure=None,
        tiers={"L2": R.TierResult("L2", "pass", True)},
        trace_check_res={"status": "pass", "violations": [], "instruction_policy": report},
        numeric={"status": "pass"},
        required={"L2"},
        no_oracle=False,
        eff_target=TARGET,
        paths=paths,
        run_id="cap",
        cfg=R._config_for_target(TARGET, "t", "fp32"),
        contract=None,
    )
    assert row["status"] == "incomplete" and row["failure"]["category"] == "PROHIBITION_NOT_MEASURED"


def test_the_trace_gate_holds_the_decoded_stream_to_the_rule():
    import inspect

    source = inspect.getsource(R.run_capsule)
    gate = source.index("trace_check_res = _grade_cpu_step(")
    oracle = source.index("--- oracle tiers ---")
    assert gate < source.index("enforce_executed_stream(capsule, trace,", gate) < oracle
