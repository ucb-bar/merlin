"""SDK-free handler controls. Synthetic reports are not proof or approval evidence."""

import json
from pathlib import Path

import pytest

from merlin.verify import bf16_policy_gate as gate
from merlin.verify.gate_errors import GateRejected, GateUnavailable, GateUnsupported


def op(name, *, results=None, regions=None, attrs=None):
    return dict(
        name=name, results=results or [], operands=[], attrs=attrs or {}, props={}, regions=regions or [], successors=[]
    )


@pytest.fixture
def pair():
    contraction = op(
        "linalg.matmul",
        results=[
            {
                "id": 7,
                "type": {
                    "kind": "tensor",
                    "shape": [2, 2],
                    "encoding": None,
                    "element": {"kind": "float", "format": "bf16"},
                },
            }
        ],
    )
    source = {
        "schema": "merlin.lean.ir.v1",
        "module": {
            "ops": [
                op(
                    "func.func",
                    attrs={"sym_name": {"kind": "string", "value": "forward"}},
                    regions=[{"blocks": [{"args": [], "ops": [contraction]}]}],
                )
            ]
        },
    }
    local = {
        "checked": True,
        "semantic_match": True,
        "site_qualification_passed": True,
        "semantic_match_theorem": gate.EXACT,
        "site_qualification_theorem": gate.SITES,
        "expected_sites": 1,
        "covered_sites": 1,
        "site_obligations": gate.source_sites(source),
    }
    report = {
        "checker": gate.policy.CHECKER,
        "theorem": gate.policy.THEOREM,
        "claim_type": "policy",
        "policy": "bf16-f32acc-contract",
        "index_bits": 64,
        "verdict": "abstracted",
        "policy_reviewed": False,
        "pass_boundary": local,
        "candidates": 1,
        "sites": [{"mul": "contract", "add": "none"}],
        "conditional_match": True,
        "qualification_passed": False,
    }
    return report, source


def test_report_can_validate_without_granting_admission(pair):
    report, source = pair
    assert gate.validate_report(report, source) == gate.source_sites(source)
    assert report["verdict"] == "abstracted" and report["policy_reviewed"] is False


@pytest.mark.parametrize("field", ["checked", "semantic_match", "site_qualification_passed"])
def test_weak_conditional_green_does_not_replace_exact_or_sites(pair, field):
    report, source = pair
    report["pass_boundary"][field] = False
    with pytest.raises(GateRejected):
        gate.validate_report(report, source)


@pytest.mark.parametrize(
    "mutation", ["missing", "position", "results", "function", "bool-count", "truthy-site", "flags", "root"]
)
def test_report_roster_and_identity_refusals(pair, mutation):
    report, source = pair
    local = report["pass_boundary"]
    if mutation == "missing":
        local["site_obligations"] = []
    elif mutation == "position":
        local["site_obligations"][0]["source_operation"] = 3
    elif mutation == "results":
        local["site_obligations"][0]["source_results"] = [8]
    elif mutation == "function":
        local["site_obligations"][0]["function"] = "other"
    elif mutation == "bool-count":
        local["covered_sites"] = True
    elif mutation == "truthy-site":
        local["site_obligations"][0]["qualification_passed"] = 1
    elif mutation == "flags":
        report["sites"][0]["mul"] = "fast"
    else:
        report["checker"] = "different"
    with pytest.raises((GateRejected, GateUnavailable)):
        gate.validate_report(report, source)


def test_changed_nested_candidate_is_not_omitted(pair):
    report, source = pair
    ops = source["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
    ops[:] = [op("scf.if", regions=[{"blocks": [{"args": [], "ops": ops[:]}]}])]
    with pytest.raises(GateUnsupported, match="outside"):
        gate.validate_report(report, source)


def test_no_site_is_not_positive_coverage(pair):
    report, source = pair
    source["module"]["ops"] = []
    with pytest.raises(GateUnsupported, match="no half"):
        gate.validate_report(report, source)


@pytest.mark.parametrize("field,value", [("verdict", "verified"), ("policy_reviewed", True), ("index_bits", 32)])
def test_wrong_policy_status_width(pair, field, value):
    report, source = pair
    report[field] = value
    with pytest.raises((GateUnavailable, GateUnsupported)):
        gate.validate_report(report, source)


def test_missing_admission_never_opens_tools_or_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "installed_root", lambda: tmp_path)

    def forbidden(*a, **kw):
        raise AssertionError("tool/source invocation before approval")

    monkeypatch.setattr(gate, "_tools", forbidden)
    monkeypatch.setattr(gate, "audit_candidate", forbidden)
    monkeypatch.setattr(gate.policy, "check_stage", forbidden)
    monkeypatch.setattr(gate, "_ref", forbidden)
    with pytest.raises(GateRejected, match="human_review_pending"):
        gate.check_policy({"params": {"sdk_bin": "/forbidden", "compiler_python": "/forbidden", "index_bits": 64}})


def test_caller_approval_flag_and_saved_receipt_are_forbidden():
    for key in ("human_reviewed", "receipt", "admission"):
        with pytest.raises(GateUnsupported):
            gate.check_policy(
                {"params": {"sdk_bin": "/forbidden", "compiler_python": "/forbidden", "index_bits": 64, key: True}}
            )


def test_fake_installed_flag_is_not_a_review_record(tmp_path):
    path = tmp_path / gate.ADMISSION
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"human_reviewed": True, "status": "verified"}))
    with pytest.raises(GateRejected):
        gate.require_admission(tmp_path)


def test_unregistered_policy_selector_refuses_before_tool_construction(tmp_path, monkeypatch):
    from merlin.common import paths
    from merlin.verify import pass_validation

    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("MERLIN_VALIDATE_PASSES", gate.PASS)
    monkeypatch.setenv("MERLIN_LEAN_SDK_BIN", "/forbidden")
    monkeypatch.setattr(pass_validation, "CheckerTools", lambda *a: pytest.fail("opened SDK"))
    with pytest.raises(ValueError, match="nonempty subset"):
        pass_validation.from_environment(tmp_path)


@pytest.mark.parametrize(
    "detail", [None, {}, {"verdict": "abstracted"}, {"status": "human_review_pending"}, {"status": "admitted"}]
)
def test_worker_never_accepts_normal_pending_return(tmp_path, monkeypatch, detail):
    from merlin.verify import gate_worker

    (tmp_path / "request.json").write_text(json.dumps({"check": "compiler_macc_contract"}))
    monkeypatch.setattr(gate_worker, "resolve", lambda name: lambda request: detail)
    assert gate_worker.main(["worker", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "verdict.json").read_text())["verdict"] == "not_established"


@pytest.mark.parametrize("mutation", ["none", "empty", "unqualified", "boolean_index", "boolean_result", "duplicate"])
def test_worker_refuses_malformed_draft_sites(tmp_path, monkeypatch, mutation):
    from merlin.verify import compiler_macc, gate_worker

    site = {"function": "subject", "source_operation": 1, "source_results": [10], "qualification_passed": True}
    sites = [site]
    if mutation == "none":
        sites = [None]
    elif mutation == "empty":
        sites = [{}]
    elif mutation == "unqualified":
        site["qualification_passed"] = False
    elif mutation == "boolean_index":
        site["source_operation"] = True
    elif mutation == "boolean_result":
        site["source_results"] = [True]
    else:
        sites.append(dict(site))
    detail = {
        "status": "checked_draft_contract",
        "claim_type": "draft_policy_conformance",
        "theorem": compiler_macc.THEOREM,
        "policy_reviewed": False,
        "numerical_policy_verdict": "abstracted",
        "original_source_equivalence": False,
        "sites": sites,
    }
    (tmp_path / "request.json").write_text(json.dumps({"check": "compiler_macc_contract"}))
    monkeypatch.setattr(gate_worker, "resolve", lambda _: lambda request: detail)
    assert gate_worker.main(["worker", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "verdict.json").read_text())["verdict"] == "not_established"


def test_selected_auditor_is_threaded_without_default(monkeypatch, tmp_path):
    # Pure call-path test; the native implementation is replaced before entry.
    captured = {}

    def fake(*args, **kwargs):
        captured.update(kwargs)
        return {"verdict": "abstracted"}

    monkeypatch.setattr(gate.policy, "_check_stages", fake)

    def sentinel(name, module):
        return []

    gate.policy.check_stage(tmp_path, tmp_path / "receipt", m2m_python=Path("/unused"), theorem_auditor=sentinel)
    assert captured["theorem_auditor"] is sentinel


def test_normal_tool_freshness_is_required_before_audit(monkeypatch):
    from merlin.verify import pass_validation

    calls = []

    def unavailable(*args):
        calls.append(args)
        raise ValueError("checker is out of date")

    monkeypatch.setattr(pass_validation, "CheckerTools", unavailable)
    with pytest.raises(ValueError, match="out of date"):
        gate._tools({"sdk_bin": "/unused", "compiler_python": "/unused"}, {})
    assert len(calls) == 1


@pytest.mark.parametrize("override", ["alias", "worker_path"])
def test_selected_policy_cannot_be_rerouted_to_other_handler(tmp_path, override):
    from types import SimpleNamespace

    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    options = (
        {"aliases": {"compiler_macc_contract": "tensor_refinement"}}
        if override == "alias"
        else {"worker_path": [tmp_path]}
    )
    worker = SpeculativeGate(tmp_path / "gate", **options)
    validation = SimpleNamespace(passes={"bf16_f32acc_draft_contract"}, gate=worker)
    try:
        with pytest.raises(ValueError, match="canonical handler"):
            pass_validation.require_canonical_handler(validation, "compiler_macc_contract")
    finally:
        worker.close()


def test_tools_preserve_virtualenv_interpreter_path(tmp_path, monkeypatch):
    from merlin.verify import pass_validation
    from merlin.verify.lean import check

    sdk = tmp_path / "sdk"
    sdk.mkdir()
    checker = tmp_path / "merlin-lean"
    for path in (sdk / "lake", sdk / "lean", checker, tmp_path / "merlin-lean-audit"):
        path.write_bytes(b"tool fixture")
    base_python = tmp_path / "base-python"
    base_python.write_bytes(b"interpreter fixture")
    selected = tmp_path / "venv" / "bin" / "python"
    selected.parent.mkdir(parents=True)
    selected.symlink_to(base_python)
    calls = []
    monkeypatch.setattr(pass_validation, "CheckerTools", lambda sdk, python: calls.append(("tools", python)))
    monkeypatch.setattr(check, "lean_exe", lambda: checker)
    monkeypatch.setattr(check, "_lean_block", lambda: {"checker": "fixture"})

    def exporter(python):
        calls.append(("exporter", python))
        return {"revision": "fixture", "round_trip": "not_run"}

    monkeypatch.setattr(check, "_exporter_block", exporter)
    admitted = {"lean": {"checker": "fixture"}, "exporter": {"revision": "fixture", "round_trip": "identical"}}
    gate._tools({"sdk_bin": str(sdk), "compiler_python": str(selected)}, admitted)
    assert calls == [("tools", selected), ("exporter", selected)]
