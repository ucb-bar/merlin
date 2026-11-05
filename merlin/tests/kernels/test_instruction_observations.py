"""Rich observations survive canonical CCA/MLIR and never become semantic levers."""

from dataclasses import asdict, replace

import pytest

from merlin.kernels.cca import CCA
from merlin.kernels.cca_compare import compare_evidence
from merlin.kernels.cca_mlir import from_mlir, to_mlir
from merlin.kernels.instruction_comparison import OperandSummary
from merlin.kernels.instruction_observations import compact_comparison, compare_observations, from_projection


def observed(values, *, pins=(("selected_contract", "a" * 64),), **limits):
    summary = OperandSummary(**limits)
    for value in values:
        summary.observe("ISA_PACKET", dict(extent=value, enabled=False))
    projection = dict(
        summary.report(),
        instruction_classes=dict(ISA_PACKET=len(values)),
        decoder_inputs=[dict(path=path, sha256=sha) for path, sha in pins],
    )
    return from_projection(projection, evidence_id="original-decoded-input", unknowns=("dynamic_count_unknown",))


def test_typed_fields_and_unknowns_survive_real_cca_mlir_roundtrip():
    instructions = observed([False, 0, 0.0, 8, None])
    cca = CCA(op="accelerator_region", backend=["test_array"], instructions=instructions)
    restored = from_mlir(to_mlir(cca))
    assert restored == cca
    assert restored.instructions.fields[1].samples == instructions.fields[1].samples
    assert restored.instructions.fields[1].unresolved_sites == 1
    assert restored.instructions.unknowns == ("dynamic_count_unknown",)
    assert asdict(restored)["instructions"]["fields"]


def test_equal_semantic_summaries_expose_real_values_without_inventing_routable_axes():
    a = CCA(op="accelerator_region", backend=["test_array"], instructions=observed([4]))
    b = CCA(op="accelerator_region", backend=["test_array"], instructions=observed([16]))
    assert a.compute == b.compute and a.to_dict() != b.to_dict()
    assert compare_evidence(a, b).denominator == 0
    result = compare_observations(a.instructions, b.instructions)
    assert result["compared_fields"] == 2
    assert result["field_differences"][0]["field"] == "extent"
    assert result["field_differences"][0]["reference"]["samples"] == [dict(value=4, sites=1)]
    assert result["field_differences"][0]["candidate"]["samples"] == [dict(value=16, sites=1)]
    assert result["semantic_alignment"] == result["numerical_equivalence"] == result["performance"] == "unknown"


def test_partial_recovery_is_comparable_but_preserves_missingness_and_scope():
    a, b = observed([None, 4]), observed([4, 4])
    result = compare_observations(a, b)
    row = result["field_differences"][0]
    assert row["completeness"] == "partial_recovered_values"
    assert row["reference"]["unresolved_sites"] == 1
    assert result["count_basis"] == "listed_instruction_sites"
    unknown = compare_observations(observed([None]), observed([None]))
    assert unknown["unknown_fields"] == 1  # The independent enabled=False field remains observed.
    assert not unknown["field_differences"]


def test_population_budget_exhaustion_keeps_localized_sequence_changes():
    a = observed(range(40), max_population_entries=3)
    b = observed([*range(39), 99], max_population_entries=3)
    result = compare_observations(a, b)
    row = result["field_differences"][0]
    assert row["field"] == "extent" and row["completeness"] == "partial_recovered_values"
    assert row["reasons"] == ["recovered_sequence_changed_population_unknown"]
    assert result["performance"] == "unknown"


@pytest.mark.parametrize(
    "change",
    [
        dict(decoder_inputs=()),
        dict(count_basis="static_ir_sites"),
        dict(decoder_inputs=(("selected_contract", "b" * 64),)),
        dict(evidence_id=""),
    ],
)
def test_foreign_or_unbound_decoder_count_scope_and_evidence_refuse(change):
    a = observed([4])
    result = compare_observations(a, replace(a, **change))
    assert result["status"] == "unknown" and result["compared_fields"] == 0


def test_compact_response_keeps_scalar_values_and_marks_large_string_display():
    result = compare_observations(observed(["a" * 1000]), observed(["b" * 1000]))
    compact = compact_comparison(result)
    sample = compact["field_differences"][0]["reference"]["samples"][0]
    assert sample["display_truncated"] and len(sample["value"]) < 200
    assert len(result["field_differences"][0]["reference"]["samples"][0]["value"]) == 1000
    assert compact["compared_fields"] == result["compared_fields"]


def test_field_selectors_localize_requested_values_and_report_absent_fields():
    a, b = observed([4]), observed([16])
    result = compare_observations(a, b, requested_fields=["ISA_PACKET.extent"])
    assert result["compared_fields"] == result["total_fields"] == 1
    assert result["field_differences"][0]["field"] == "extent"
    missing = compare_observations(a, b, requested_fields=["ISA_PACKET.unknown"])
    assert missing["status"] == "unknown" and missing["unknown_fields"] == 1


def test_missing_value_changes_do_not_become_changed_known_value_sets():
    result = compare_observations(observed([None, 4]), observed([4, 4]))
    assert all("recovered_values_changed" not in row["reasons"] for row in result["field_differences"])
