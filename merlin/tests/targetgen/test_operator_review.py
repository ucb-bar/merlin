"""Authored scope extensions require acceptance without claiming completed review."""

import copy

import pytest

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen.operator_review import ACCEPT_ENV, PENDING, review_record
from merlin.targetgen.software_spec import admit_operation, load_software_spec

HOST_QUANTIZERS = ["host_quantize_exact", "host_dequantize_per_tensor"]


def specification(status=PENDING):
    return dict(
        status="reviewed",
        operations=[
            dict(
                id="linear",
                status=status,
                families=["elementwise_map"],
                placement="accelerator",
                signature=dict(numerical_family=["F2"], operand_dtypes=["int8"]),
            )
        ],
    )


def decision(spec):
    return admit_operation(
        spec,
        "elementwise_map",
        dict(family="elementwise_map", numerical_family="F2", operand_dtype="int8"),
        "accelerator",
    )


def test_compatibility_default_does_not_claim_operator_review(monkeypatch):
    monkeypatch.delenv(ACCEPT_ENV, raising=False)
    result = decision(specification())
    assert result["status"] == "admitted"
    assert result["review_status"] == PENDING and result["operator_accepted"] is True
    assert result["operator_acceptance"]["selection"] == "default_compatibility"


@pytest.mark.parametrize("selected,expected", [("0", "unknown"), ("1", "admitted")])
def test_explicit_acceptance_controls_pending_rows(monkeypatch, selected, expected):
    monkeypatch.setenv(ACCEPT_ENV, selected)
    result = decision(specification())
    assert result["status"] == expected
    assert result["review_status"] == PENDING
    assert result["operator_acceptance"]["selection"] == "environment"


@pytest.mark.parametrize("selected", ["0", "1"])
def test_acceptance_never_promotes_unreviewed_rows(monkeypatch, selected):
    monkeypatch.setenv(ACCEPT_ENV, selected)
    assert decision(specification("unreviewed"))["status"] == "unknown"
    assert decision(specification("reviewed"))["status"] == "admitted"


def test_reviewed_label_cannot_override_pending_evidence(monkeypatch):
    monkeypatch.setenv(ACCEPT_ENV, "0")
    spec = specification("reviewed")
    spec["operations"][0]["evidence"] = dict(review_status="authored, pending operator review")
    result = decision(spec)
    assert result["status"] == "unknown" and result["review_status"] == PENDING


def test_acceptance_cannot_relax_constraints(monkeypatch):
    monkeypatch.setenv(ACCEPT_ENV, "1")
    spec = specification()
    spec["operations"][0]["signature"]["operand_dtypes"] = ["i64"]
    assert decision(spec)["status"] == "unsupported"


def test_unclassified_source_cannot_borrow_component_family():
    spec = specification("reviewed")
    result = admit_operation(
        spec, "elementwise_map", dict(family="elementwise_map", operand_dtype="int8"), "accelerator"
    )
    assert result["status"] == "unsupported"
    assert "representation proof" in result["reason"]
    # An observed component family still leaves unrelated missing observations unknown.
    result = admit_operation(
        spec, "elementwise_map", dict(family="elementwise_map", numerical_family="F2"), "accelerator"
    )
    assert result["status"] == "unknown"


def test_invalid_acceptance_is_a_configuration_error(monkeypatch):
    monkeypatch.setenv(ACCEPT_ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        decision(specification())


def test_authored_native_family_rows_are_honestly_labeled():
    root = repo_root() / "examples/gemmini/target"
    software = load_software_spec(root / "software-spec.yaml")
    rows = {row["id"]: row for row in software["operations"]}
    for name in ["movement_bitwise", "component_linear", "component_bilinear"]:
        assert rows[name]["status"] == "reviewed"
        assert rows[name]["evidence"]["review_status"] == "reviewed"
        assert "review packet" in rows[name]["evidence"]["basis"]
    assert not set(HOST_QUANTIZERS) & set(rows)
    assert software["quantization_contract"]["status"] == "reviewed"
    assert software["numerical_semantics"]["native_unit_contract"]["status"] == "reviewed"
    for name in ["contraction", "elementwise_map", "elementwise_map_standalone", "pooling_on_readout"]:
        assert rows[name]["status"] == "reviewed"
        assert rows[name]["evidence"]["review_status"] == "reviewed"
        assert "packet2.md" in rows[name]["evidence"]["packet"]
    assert "host_quantize" not in rows
    host = load_yaml(root.parent / "platform" / "host-capabilities-core-aten.yaml")
    for row in host["operations"]:
        contract = row.get("numerical_contract", {})
        if "pending operator review" in contract.get("evidence", {}).get("review_status", ""):
            assert row["status"] == contract["status"] == PENDING
    host_rows = {row["id"]: row for row in host["operations"]}
    for name in ["host_exact_native_boundary"] + HOST_QUANTIZERS:
        row = host_rows[name]
        assert row["status"] == row["numerical_contract"]["status"] == "reviewed"
        assert row["numerical_contract"]["evidence"]["review_status"] == "reviewed"
        assert "packet2.md" in row["numerical_contract"]["evidence"]["packet"]


def test_pending_numerical_claim_does_not_review_a_host_placement(monkeypatch):
    monkeypatch.setenv(ACCEPT_ENV, "1")
    original = dict(status="reviewed", numerical_contract=dict(status=PENDING))
    before = copy.deepcopy(original)
    result = review_record(original)
    assert result["operator_accepted"] and result["review_status"] == PENDING
    assert original == before


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_host_screen_separates_acceptance_from_review(monkeypatch, enabled):
    from merlin.targetgen.host_capabilities import admit_host_operation

    monkeypatch.setenv(ACCEPT_ENV, enabled)
    declaration = specification()["operations"][0]
    declaration["placement"] = "host"
    document = dict(
        schema="merlin.host_capabilities.v1",
        status="reviewed",
        evidence={},
        compiler=dict(package_sha256="a" * 64, dtype_strategy="native"),
        operations=[declaration],
    )
    selected = {
        "native": dict(
            package_sha256="a" * 64, capability_spec_sha256="b" * 64, dtype_strategy="native", capability_spec=document
        )
    }
    signature = dict(family="elementwise_map", numerical_family="F2", operand_dtype="int8")
    result = admit_host_operation(selected, dict(mlir_operation="linalg.generic"), signature)
    assert result["status"] == ("admitted" if enabled == "1" else "unknown")
    assert result["review_status"] == PENDING and result["reviewed"] is False
    assert result["operator_accepted"] is (enabled == "1")


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_pending_numerical_contract_uses_the_same_acceptance_policy(monkeypatch, enabled):
    from merlin.targetgen.operation_numerics import _explicit_contract

    monkeypatch.setenv(ACCEPT_ENV, enabled)
    contract = dict(status=PENDING, semantics=dict(arithmetic="exact_integer"), evidence=dict(basis="authored"))
    result = _explicit_contract(contract)
    assert result["status"] == ("resolved" if enabled == "1" else "unknown")
    assert result["review_status"] == PENDING
    assert result["operator_accepted"] is (enabled == "1")


def test_numerical_contract_without_explicit_review_status_is_not_accepted(monkeypatch):
    from merlin.targetgen.operation_numerics import _explicit_contract

    monkeypatch.setenv(ACCEPT_ENV, "1")
    result = _explicit_contract(dict(semantics=dict(arithmetic="exact_integer"), evidence=dict(basis="authored")))
    assert result["status"] == "unknown" and "review_status" in result["unknowns"]


@pytest.mark.parametrize("missing", ["spec_review", "unchecked_row_review"])
def test_missing_required_review_cannot_be_accepted_implicitly(monkeypatch, missing):
    monkeypatch.setenv(ACCEPT_ENV, "1")
    spec = specification()
    if missing == "spec_review":
        spec.pop("status")
    else:
        row = spec["operations"][0]
        row.pop("status")
        row["admission_basis"] = "declared_unchecked"
    assert decision(spec)["status"] == "unknown"


def test_numerical_report_keeps_accepted_pending_host_claims(monkeypatch):
    from merlin.targetgen.operation_numerics import operation_numerical_contracts

    monkeypatch.setenv(ACCEPT_ENV, "1")
    claim = dict(status=PENDING, semantics=dict(arithmetic="exact_integer"), evidence=dict(basis="authored"))
    entry = dict(observed_signature={}, software_admissions=[], host_admission=dict(profiles=[
        dict(profile="native", status="admitted", reviewed=False, operator_accepted=True,
             decisions=[dict(status="admitted", declaration="host_integer")])]))
    selected = {"native": dict(capability_spec=dict(operations=[dict(id="host_integer", numerical_contract=claim)]))}
    result = operation_numerical_contracts(entry, {}, selected)
    assert result["host"]["status"] == "resolved"
    assert result["host"]["declarations"][0]["review_status"] == PENDING
