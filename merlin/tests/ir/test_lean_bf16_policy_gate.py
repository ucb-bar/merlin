"""Independent policy-gate qualification. Default tests are SDK-free plumbing.

Mocked reports are NEVER proof acceptance or human approval. Real tools require
separate F environment authorization, final frozen source and retained raw runs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.verify import bf16_policy_gate as gate
from merlin.verify import gate_worker, pass_validation
from merlin.verify.gate_errors import GateRejected, GateUnavailable, GateUnsupported


def op(name, *, results=(), attrs=None, props=None, children=None):
    return {
        "name": name,
        "results": list(results),
        "attrs": attrs or {},
        "props": props or {},
        "operands": [],
        "regions": [] if children is None else [{"blocks": [{"args": [], "ops": children}]}],
    }


def source_export():
    def contraction(identity):
        return op(
            "linalg.generic",
            results=[
                {
                    "id": identity,
                    "type": {"kind": "tensor", "shape": [2, 3], "element": {"kind": "float", "format": "bf16"}},
                }
            ],
            props={"iterator_types": {"kind": "array", "elements": [{"kind": "enum", "value": "reduction"}]}},
        )

    return {
        "schema": "merlin.lean.ir.v1",
        "module": {
            "ops": [
                op(
                    "func.func",
                    props={"sym_name": {"kind": "string", "value": "subject"}},
                    children=[
                        op("arith.constant"),
                        contraction(10),
                        op("arith.constant"),
                        contraction(20),
                        op("func.return"),
                    ],
                )
            ]
        },
    }


def raw_report():
    return {
        "checker": gate.policy.CHECKER,
        "theorem": gate.policy.THEOREM,
        "claim_type": "policy",
        "policy": "bf16-f32acc-contract",
        "index_bits": 64,
        "verdict": "abstracted",
        "policy_reviewed": False,
        "candidates": 2,
        "conditional_match": True,
        "qualification_passed": True,
        "sites": [{"mul": "contract", "add": "none"}, {"mul": "none", "add": "contract"}],
        "pass_boundary": {
            "checked": True,
            "semantic_match": True,
            "semantic_match_theorem": gate.EXACT,
            "site_qualification_passed": True,
            "site_qualification_theorem": gate.SITES,
            "expected_sites": 2,
            "covered_sites": 2,
            "site_obligations": [
                {"function": "subject", "source_operation": 1, "source_results": [10], "qualification_passed": True},
                {"function": "subject", "source_operation": 3, "source_results": [20], "qualification_passed": True},
            ],
        },
    }


def job(tmp_path):
    return {
        "directory": str(tmp_path),
        "check": "bf16_f32acc_policy",
        "label": gate.PASS,
        "before": str(tmp_path / "before.mlir"),
        "after": str(tmp_path / "after.mlir"),
        "params": {"sdk_bin": "/NEVER_OPEN_SDK", "compiler_python": "/NEVER_OPEN_COMPILER", "index_bits": 64},
    }


def forbidden(*args, **kwargs):
    raise AssertionError("SDK/native/auditor path forbidden in this SDK-free test")


def test_pending_policy_refuses_before_tool_or_boundary_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(gate, "_tools", forbidden)
    monkeypatch.setattr(gate, "_ref", forbidden)
    monkeypatch.setattr(gate.subprocess, "run", forbidden)
    with pytest.raises(GateRejected, match="human_review_pending"):
        gate.check_policy(job(tmp_path))


@pytest.mark.parametrize("field,value", [("approval", True), ("receipt", "saved.json"), ("policy_reviewed", True)])
def test_caller_cannot_supply_admission(tmp_path, field, value):
    request = job(tmp_path)
    request["params"][field] = value
    with pytest.raises(GateUnsupported):
        gate.check_policy(request)


@pytest.mark.parametrize("bits", [True, 32, "64", 0])
def test_exact_index64_only(tmp_path, bits):
    request = job(tmp_path)
    request["params"]["index_bits"] = bits
    with pytest.raises(GateUnsupported):
        gate.check_policy(request)


def test_report_consistency_is_separate_from_admission(tmp_path, monkeypatch):
    report = raw_report()
    # Whole-capture fields are weaker/separate. Their false value cannot erase
    # local proof acceptance, but this return is NOT a live gate verdict.
    report.update(conditional_match=False, qualification_passed=False)
    assert gate.validate_report(report, source_export()) == raw_report()["pass_boundary"]["site_obligations"]
    monkeypatch.setattr(gate, "repo_root", lambda: tmp_path)
    with pytest.raises(GateRejected):
        gate.check_policy(job(tmp_path))


@pytest.mark.parametrize("field", ["checked", "semantic_match", "site_qualification_passed"])
def test_weak_true_flags_cannot_replace_strong_false_flag(field):
    report = raw_report()
    report["pass_boundary"][field] = False
    with pytest.raises(GateRejected):
        gate.validate_report(report, source_export())


@pytest.mark.parametrize("field", ["checked", "semantic_match", "site_qualification_passed"])
def test_report_boolean_types_are_exact(field):
    report = raw_report()
    report["pass_boundary"][field] = 1
    with pytest.raises(GateUnavailable):
        gate.validate_report(report, source_export())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "reorder",
        "wrong_position",
        "wrong_result",
        "wrong_function",
        "bool_result",
        "unqualified",
    ],
)
def test_full_source_roster_is_recomputed(mutation):
    report = raw_report()
    sites = report["pass_boundary"]["site_obligations"]
    if mutation == "missing":
        sites.pop()
    elif mutation == "duplicate":
        sites[1] = copy.deepcopy(sites[0])
    elif mutation == "reorder":
        sites.reverse()
    elif mutation == "wrong_position":
        sites[1]["source_operation"] = 2
    elif mutation == "wrong_result":
        sites[1]["source_results"] = [21]
    elif mutation == "wrong_function":
        sites[1]["function"] = "other"
    elif mutation == "bool_result":
        sites[1]["source_results"] = [True]
    else:
        sites[1]["qualification_passed"] = False
    with pytest.raises(GateRejected):
        gate.validate_report(report, source_export())


@pytest.mark.parametrize("field,value", [("expected_sites", 1), ("covered_sites", 3), ("covered_sites", True)])
def test_denominator_cannot_shrink_or_change_type(field, value):
    report = raw_report()
    report["pass_boundary"][field] = value
    with pytest.raises(GateRejected):
        gate.validate_report(report, source_export())


@pytest.mark.parametrize("verdict", ["verified", "accepted", "human_review_pending", "unmeasured", "unknown"])
def test_driver_cannot_approve_itself(verdict):
    report = raw_report()
    report["verdict"] = verdict
    with pytest.raises(GateUnavailable):
        gate.validate_report(report, source_export())


@pytest.mark.parametrize("flag", ["fast", "reassoc", "nnan", "contract,nnan"])
def test_nonpermitted_flags_refuse(flag):
    report = raw_report()
    report["sites"][0]["add"] = flag
    with pytest.raises(GateRejected):
        gate.validate_report(report, source_export())


def test_nested_candidate_is_explicitly_unsupported():
    document = source_export()
    body = document["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    body[1] = op("scf.if", children=[body[1]])
    with pytest.raises(GateUnsupported):
        gate.validate_report(raw_report(), document)


def test_zero_candidate_is_not_positive_coverage():
    document = source_export()
    document["module"]["ops"].clear()
    with pytest.raises(GateUnsupported):
        gate.validate_report(raw_report(), document)


@pytest.mark.parametrize("outcome", ["pending", "unsupported", "error", "timeout"])
def test_worker_has_no_accepted_verdict_on_nonadmission(tmp_path, monkeypatch, outcome):
    def check(_):
        if outcome == "pending":
            raise GateRejected("human_review_pending")
        if outcome == "unsupported":
            raise GateUnsupported("outside domain")
        if outcome == "timeout":
            raise GateUnavailable("checker timeout")
        raise RuntimeError("checker crashed")

    (tmp_path / "request.json").write_text(json.dumps(job(tmp_path)))
    monkeypatch.setattr(gate_worker, "resolve", lambda _: check)
    assert gate_worker.main(["worker", str(tmp_path)]) == 0
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == {"pending": "not_established", "unsupported": "unsupported"}.get(
        outcome, "unavailable"
    )


def test_wip_policy_is_not_a_builtin_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(gate.subprocess, "run", forbidden)
    (tmp_path / "request.json").write_text(json.dumps(job(tmp_path)))
    assert gate_worker.main(["worker", str(tmp_path)]) == 0
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["verdict"] == "unavailable"
    assert "neither a built-in" in verdict["reason"]


def test_selected_auditor_command_never_prepends_elan(tmp_path, monkeypatch):
    tools = {
        key: {"path": "/NEW_REVIEWED_TOOLS/" + key, "sha256": "0" * 64, "bytes": 0}
        for key in ("lake", "lean", "compiler_python", "checker", "audit")
    }
    calls = []
    monkeypatch.setattr(gate, "_bound", lambda _: None)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[:3] == ["/NEW_REVIEWED_TOOLS/lake", "env", "/NEW_REVIEWED_TOOLS/audit"]
        assert kwargs["env"]["PATH"].split(":")[0] == "/NEW_REVIEWED_TOOLS"
        if "--statements" in command:
            value = {"statements": {root: "TEST TYPE" for root in gate.ROOTS}}
        elif "--axioms" in command:
            value = {"decl": command[command.index("--axioms") + 1], "axioms": ["propext"]}
        else:
            value = {"violations": [], "theorems_audited": 1}
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr="")

    monkeypatch.setattr(gate.subprocess, "run", run)
    result = gate.audit_candidate(tools, tmp_path)
    assert set(result["axioms"]) == set(gate.ROOTS)
    assert len(calls) == 2 + len(gate.ROOTS)


def test_policy_is_standalone_and_not_a_compiler_gate():
    assert gate.PASS not in pass_validation.CHECKS
    assert "bf16_f32acc_policy" not in gate_worker.BUILTIN_CHECKS


def test_actual_preparation_preserves_bytes_without_policy_observers(tmp_path, monkeypatch):
    from lean_bf16_f32acc_cases import bmm
    from merlin.runtime.backends import zephyr_model

    source = tmp_path / "source.mlir"
    source.write_text(bmm())
    plain, observed = tmp_path / "plain", tmp_path / "observed"
    plain.mkdir()
    observed.mkdir()
    expected = zephyr_model._prepare_model_mlir(source, plain).read_bytes()
    monkeypatch.setattr(gate, "require_admission", forbidden)
    monkeypatch.setattr(pass_validation, "observe", forbidden)
    actual = zephyr_model._prepare_model_mlir(source, observed).read_bytes()
    assert actual == expected
    assert b"arith.extf" in actual and b"arith.truncf" in actual


@pytest.mark.parametrize("source_init", ["0.000000e+00", "1.000000e+00"])
def test_pending_policy_cannot_block_normal_preparation(tmp_path, monkeypatch, source_init):
    from lean_bf16_f32acc_cases import bmm
    from merlin.runtime.backends import zephyr_model
    from merlin.verify.speculative_gate import SpeculativeGate

    source = tmp_path / "source.mlir"
    source.write_text(bmm(init=source_init))
    work = tmp_path / "preparation"
    work.mkdir()
    tools = object.__new__(pass_validation.CheckerTools)
    object.__setattr__(tools, "sdk_bin", Path("/NEVER_OPEN_SDK"))
    object.__setattr__(tools, "compiler_python", Path("/NEVER_OPEN_COMPILER"))
    runner = SpeculativeGate(tmp_path / "gate", job_timeout=20)
    validation = pass_validation.PassValidation(runner, tools)
    monkeypatch.setattr(pass_validation, "from_environment", lambda *a, **k: validation)
    monkeypatch.setattr(gate, "require_admission", forbidden)
    monkeypatch.setattr(gate, "check_policy", forbidden)
    try:
        result, _features = zephyr_model.prepare_for_lowering(source, work)
        assert result.is_file()
        assert runner.verdicts == []
    finally:
        runner.close()
    assert not runner._processes


def synthetic_admission(root: Path):
    """Build governance-fixture bytes under tmp_path ONLY, not real human approval."""
    # Stub source enumeration so this is strictly an admission-protocol test.
    sources = {f"merlin/lean/MerlinVerify/{suffix}.lean": "a" * 64 for suffix in gate.PIN_FILES}
    statements = {name: "TEST-ONLY-STATEMENT " + name for name in gate.ROOTS}
    lock = {name.removeprefix("merlin/lean/"): value for name, value in sources.items()}
    lock.update({"statement:" + name: gate._digest(value.encode()) for name, value in statements.items()})
    p = root / "merlin/lean/SPEC_LOCK"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(lock))
    review = {
        "schema": "merlin.bf16.policy.review.v1",
        "decision": "approved",
        "domain": gate.DOMAIN,
        "reviewer": "SYNTHETIC UNIT TEST",
        "independent_of_implementation": True,
        "record": "test-fixture-not-human-review",
        "sources_sha256": gate._digest(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()),
        "statements_sha256": gate._digest(json.dumps(statements, sort_keys=True, separators=(",", ":")).encode()),
    }
    review_path = root / gate.REVIEW
    review_path.parent.mkdir(parents=True)
    review_path.write_text(json.dumps(review))
    admission = {
        "schema": "merlin.bf16.policy.admission.v1",
        "domain": gate.DOMAIN,
        "review_sha256": gate._digest(review_path.read_bytes()),
        "sources": sources,
        "statements": statements,
        "lean": {},
        "exporter": {},
    }
    (root / gate.ADMISSION).write_text(json.dumps(admission))
    return sources, review, admission


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_review",
        "empty_reviewer",
        "empty_record",
        "pending",
        "not_independent",
        "wrong_domain",
        "stale_source",
        "stale_statement",
        "missing_policy_pin",
    ],
)
def test_matching_hashes_do_not_replace_attributable_review(tmp_path, monkeypatch, mutation):
    sources, review, admitted = synthetic_admission(tmp_path)
    monkeypatch.setattr(gate, "source_bindings", lambda _: sources)
    # Coherent synthetic fixture is only documentary parsing, never real admission.
    assert gate.require_admission(tmp_path) == admitted
    if mutation == "missing_review":
        (tmp_path / gate.REVIEW).unlink()
    elif mutation == "stale_source":
        sources[next(iter(sources))] = "b" * 64
    elif mutation == "stale_statement":
        admitted["statements"][next(iter(admitted["statements"]))] = "changed"
    elif mutation == "missing_policy_pin":
        p = tmp_path / "merlin/lean/SPEC_LOCK"
        lock = json.loads(p.read_text())
        lock.pop("MerlinVerify/Spec/Bf16F32AccContract.lean")
        p.write_text(json.dumps(lock))
    else:
        key, value = {
            "empty_reviewer": ("reviewer", ""),
            "empty_record": ("record", ""),
            "pending": ("decision", "pending"),
            "not_independent": ("independent_of_implementation", False),
            "wrong_domain": ("domain", "unapproved-nonzero-domain"),
        }[mutation]
        review[key] = value
        p = tmp_path / gate.REVIEW
        p.write_text(json.dumps(review))
        admitted["review_sha256"] = gate._digest(p.read_bytes())  # rehashing cannot bless the changed decision
    (tmp_path / gate.ADMISSION).write_text(json.dumps(admitted))
    with pytest.raises(GateRejected):
        gate.require_admission(tmp_path)
