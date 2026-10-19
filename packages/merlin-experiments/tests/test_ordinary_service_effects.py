"""Neutral original-owner transport checks; fixtures qualify no target service."""

import json
from dataclasses import asdict

import pytest
from merlin_experiments.phase2 import _ordinary_service_effects as S
from merlin_experiments.phase2.contracts import StageGateError, document_sha256

from merlin.llvmlower.reviewed_pass_capabilities import PassCapabilityBinding, ReviewedPassCapability


class SyntheticServiceOwner:
    identity = "original-synthetic-emitted-service"

    def __init__(self, binding, receipt):
        self.binding = asdict(binding)
        self.receipt = receipt
        self.original = receipt.read_bytes()
        self.damage = None
        self.calls = 0

    def revalidate(self):
        if self.receipt.read_bytes() != self.original:
            raise ValueError("original service evidence changed")

    def selected_binding(self):
        return dict(self.binding)

    def evidence(self, report_path, _invocation, _review, *, source_sha256):
        self.calls += 1
        if self.damage == "unsupported":
            raise ValueError("unsupported counter semantics")
        report = json.loads(report_path.read_bytes())
        result = {
            "schema": "merlin.phase2.service_effect_evidence.v1",
            "source_sha256": source_sha256,
            "elf_sha256": report["elf_sha256"],
            "scope_sha256": report["scope_sha256"],
            "device_sha256": report["device_sha256"],
            "scope": "emitted_schedule",
            "resources": {"service": {"unit": "requests", "capacity_per_cycle_upper": 2,
                                        "provenance": "synthetic structural ceiling"}},
            "events": [{"id": "emitted", "demand": {"service": 10}, "compulsory": False,
                        "provenance": "synthetic same-ELF work"}],
            "completion_chains": [],
            "unresolved": {"complete_call": "UNKNOWN"},
            "attainment": None,
            "convergence_eligible": False,
        }
        if self.damage in {"elf_sha256", "source_sha256", "scope_sha256", "device_sha256"}:
            result[self.damage] = "f" * 64
        elif self.damage == "workload":
            result["scope"] = "workload"
        elif self.damage == "compulsory":
            result["events"][0]["compulsory"] = True
        elif self.damage == "attainment":
            result["attainment"] = 1
        elif self.damage == "fake_cycles":
            result["measured_cycles"] = 5
        elif self.damage == "rate":
            result["resources"]["service"]["capacity_per_cycle_upper"] = -1
        elif self.damage == "unpriced":
            result["resources"]["service"]["capacity_per_cycle_upper"] = None
        elif self.damage == "stale":
            self.receipt.write_text("changed")
        return result


@pytest.fixture
def original(tmp_path):
    binding = PassCapabilityBinding(*(str(i) * 64 for i in range(1, 6)))
    receipt = tmp_path / "original-owner-evidence"
    receipt.write_text("original")
    owner = SyntheticServiceOwner(binding, receipt)
    capability = ReviewedPassCapability(S.MECHANISM, binding, owner)
    selection = S.select(capability, binding)
    source = "a" * 64
    interface = "b" * 64
    compiled = {"status": "observed", "elf_sha256": "c" * 64, "candidate_diagnostics": {
        "artifacts": {"source": {"sha256": interface}},
        "source_emission_join": {"status": "verified", "source_sha256": interface},
    }}
    report = {
        "schema": S.O.REPORT_V2,
        "model_sha256": binding.runtime,
        "elf_sha256": compiled["elf_sha256"],
        "scope_sha256": "d" * 64,
        "device_sha256": "e" * 64,
        "input_sha256": document_sha256({"kind": "original_frozen_public_capsule", "sha256": source}),
        "instructions": 10,
        "feature_counts": {"cpu.integer": 6},
        "feature_observation": {"schema": S.O.COUNT_SCHEMA, "complete": True,
            "scope_instructions": 6, "counted_cpu_instructions": 6,
            "scope_kind": "counter_pc_pair", "window_index": 0,
            "primary_cpu_features": ["cpu.integer"], "unresolved": [], "qualification": "not_established"},
        "counter_windows": [{"instructions": 6, "begin_pc": 8, "end_pc": 16}],
        "scope_binding": {"occurrence": 0, "begin_pc": 8, "end_pc": 16},
        "execution_completed": True, "functional_status": "PASS", "raw_functional_exit_code": 0,
        "estimated_cycles": None, "timing_complete": False,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    arguments = {"member_source_sha256": source, "compiled": compiled, "report_path": path,
                 "invocation_path": tmp_path / "invocation.json", "counter_review_path": tmp_path / "review.json"}
    return locals()


def test_original_effects_compose_existing_partial_floor_without_cycles_or_attainment(original):
    result = original["selection"].project(**original["arguments"])
    assert result["floor"]["floor_cycles"] == 5
    assert result["floor"]["status"] == "partial"
    assert result["source_binding"] == {"member_sha256": original["source"], "interface_sha256": original["interface"]}
    assert result["attainment"] is result["measured_cycles"] is None
    assert result["objective_floor_eligible"] is result["convergence_eligible"] is False
    assert result["promotion_authority"] == "NONE"
    assert original["owner"].calls == 1


def test_missing_selection_is_unknown_and_saved_records_never_issue_owner(original):
    assert S.select(None, original["binding"]) is None
    with pytest.raises(StageGateError, match="original live"):
        S.select(original["capability"].record(), original["binding"])


@pytest.mark.parametrize("damage", ["elf_sha256", "source_sha256", "scope_sha256", "device_sha256",
                                    "workload", "compulsory", "attainment", "fake_cycles", "rate"])
def test_foreign_effects_and_compulsory_candidate_waste_refuse(original, damage):
    original["owner"].damage = damage
    with pytest.raises(StageGateError, match="foreign|compulsory|invalid original"):
        original["selection"].project(**original["arguments"])


def test_unpriced_effects_remain_unknown_and_refusal_retains_every_member(original):
    original["owner"].damage = "unpriced"
    result = original["selection"].project(**original["arguments"])
    assert result["floor"]["floor_cycles"] is None
    assert result["floor"]["status"] == "unknown"
    original["owner"].damage = "unsupported"
    result = original["selection"].project(**original["arguments"])
    assert result["status"] == "UNKNOWN" and result["evidence"] is None
    assert "unsupported counter semantics" in result["reason"]


@pytest.mark.parametrize("role", ["elf_sha256", "model_sha256", "input_sha256"])
def test_report_must_join_original_current_ELF_model_and_frozen_source(original, role):
    original["report"][role] = "f" * 64
    original["path"].write_text(json.dumps(original["report"]))
    with pytest.raises(StageGateError, match="current original count report"):
        original["selection"].project(**original["arguments"])


def test_missing_actual_source_join_never_synthesizes_effects(original):
    original["compiled"]["candidate_diagnostics"]["source_emission_join"]["status"] = "UNKNOWN"
    result = original["selection"].project(**original["arguments"])
    assert result["status"] == "UNKNOWN" and result["floor"] is None
    assert original["owner"].calls == 0


def test_foreign_binding_or_swapped_original_callback_refuses(original, monkeypatch):
    foreign = PassCapabilityBinding("foreign", "2" * 64, "3" * 64, "4" * 64, "5" * 64)
    with pytest.raises(StageGateError, match="foreign mechanism or binding"):
        S.select(original["capability"], foreign)
    monkeypatch.setattr(original["owner"], "evidence", lambda *_a, **_k: {})
    with pytest.raises(StageGateError, match="original bound owner"):
        original["selection"].project(**original["arguments"])


def test_stale_original_evidence_fails_after_callback_before_result_consumption(original):
    original["owner"].damage = "stale"
    with pytest.raises(ValueError, match="original service evidence changed"):
        original["selection"].project(**original["arguments"])


def test_records_are_defensive_copies_and_wrong_original_binding_cannot_be_asserted(original):
    record = original["selection"].record()
    record["capability"]["binding"]["hardware"] = "foreign"
    assert original["selection"].record()["capability"]["binding"] == asdict(original["binding"])
    original["owner"].binding["hardware"] = "foreign"
    with pytest.raises(StageGateError, match="owner or callback changed"):
        original["selection"].revalidate()
