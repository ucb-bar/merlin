"""Decoded field values reach CCA facets only through a target's declared field kinds."""

from __future__ import annotations

import json

import pytest

from merlin.compile.analysis.disassembly import RawInsn
from merlin.compile.analysis.schedule_structure import analyze_normalized_commands
from merlin.kernels.cca_mlir import from_mlir, to_mlir
from merlin.kernels.instruction_observations import field_roster_status
from merlin.kernels.machine_evidence import declared_field_kinds, machine_projection
from merlin.mining.reference_subset_cards import axis_knownness
from merlin.targetgen.rocc.normalized_operands import BoundOperandObserver


def _observer():
    facts = dict(
        interfaces=[
            dict(
                name="register_bundle_layouts",
                bundles={
                    "Packet": dict(
                        width=64, fields={"extent": dict(offset=20, width=5), "slot": dict(offset=0, width=8)}
                    )
                },
            )
        ]
    )
    protocol = dict(
        version=1,
        encoding=dict(semantic_class={4: "FETCH", 5: "SETUP"}),
        roles=dict(
            classes={
                "FETCH": dict(
                    fields=[
                        dict(name="height", operand="rs2", bundle="Packet", field="extent"),
                        dict(name="slot", operand="rs2", bundle="Packet", field="slot"),
                    ]
                ),
                "SETUP": dict(fields=[{"name": "pitch", "operand": "rs2", "as": "raw"}]),
            }
        ),
    )
    return BoundOperandObserver(
        target="array_fixture",
        protocol=protocol,
        facts=facts,
        raw_isa=dict(CUSTOM_OPCODE=43, FUNCT3=3, FUNCT_CLASS={4: "RAW4", 5: "RAW5"}),
        word_bits=64,
    )


def _word(funct, rs2=6):
    return f"{funct << 25 | rs2 << 20 | 3 << 12 | 43:08x}"


def _rows():
    # x6 = 17<<20 (height 17), x7 = 9<<20 (height 9); SETUP reads x6 raw.
    return [
        RawInsn(0, "lui", [], "01100337"),
        RawInsn(4, "lui", [], "009003b7"),
        RawInsn(8, ".insn", [], _word(5)),
        RawInsn(12, ".insn", [], _word(4)),
        RawInsn(16, ".insn", [], _word(4, rs2=7)),
        RawInsn(20, "fence", [], "0ff0000f"),
        RawInsn(24, "ret", [], "00008067"),
    ]


def _project(**structure):
    decoder = dict(
        semantics=_observer(),
        target="array_fixture",
        structure=dict(instruction_roles={"FETCH": ["operand_load"], "SETUP": ["config"]}, **structure),
    )
    model = analyze_normalized_commands([], source_kind="static_disassembly")
    return machine_projection(_rows(), model=model, evidence_id="kinds", decoder=decoder)


def test_undeclared_fields_stay_out_of_semantic_facets():
    cca, report = _project()
    assert cca.operand_fields is None
    assert report["field_profiles"]["FETCH.height"]
    assert report["operand_field_kinds"] == {}
    evidence = cca.provenance["axis_evidence"]["operand_fields.stride"]
    assert evidence["status"] == "unknown" and "declaration" in evidence["reason"]


def test_declared_kinds_bind_complete_static_value_sets_with_static_scope():
    cca, report = _project(
        operand_field_kinds={"FETCH.height": "extent", "SETUP.pitch": "stride", "FETCH.slot": "address"}
    )
    assert cca.operand_fields.extent == (("FETCH.height", (9, 17)),)
    assert cca.operand_fields.stride == (("SETUP.pitch", (17 << 20,)),)
    # Address kinds are accepted but never projected as semantics.
    assert cca.operand_fields.scale is None
    evidence = cca.provenance["axis_evidence"]
    assert evidence["operand_fields.extent"]["scope"] == "static_instruction_sites"
    assert report["operand_field_kinds"]["FETCH.slot"] == "address"
    restored = from_mlir(to_mlir(cca))
    assert restored.operand_fields == cca.operand_fields


def test_static_dispatch_counts_and_config_fraction_never_claim_executions():
    cca, _ = _project()
    assert cca.dispatch.static_command_sites == 3
    assert cca.dispatch.static_fence_sites == 1
    assert cca.dispatch.config_fraction == pytest.approx(1 / 3)
    evidence = cca.provenance["axis_evidence"]
    for axis in ("dispatch.static_command_sites", "dispatch.static_fence_sites", "dispatch.config_fraction"):
        assert evidence[axis]["scope"] == "static_instruction_sites"


def test_malformed_kind_declarations_refuse():
    with pytest.raises(ValueError):
        declared_field_kinds({"operand_field_kinds": {"FETCH.height": "tile_size_hint"}})
    with pytest.raises(ValueError):
        declared_field_kinds({"operand_field_kinds": {"height": "extent"}})


def test_roster_is_complete_only_when_the_selected_layout_roster_is():
    assert field_roster_status({"field_fingerprint_status": "complete"}) == "observed_fields_only"
    assert (
        field_roster_status(
            {"field_fingerprint_status": "complete", "operand_layout_roster": {"status": "declared_subset"}}
        )
        == "declared_subset"
    )
    assert (
        field_roster_status({"field_fingerprint_status": "complete", "operand_layout_roster": {"status": "complete"}})
        == "complete"
    )
    assert (
        field_roster_status(
            {
                "field_fingerprint_status": "unknown_field_budget_exhausted",
                "operand_layout_roster": {"status": "complete"},
            }
        )
        == "unknown_field_budget_exhausted"
    )
    _, report = _project(operand_layout_roster={"status": "declared_subset", "unprojected_fields": []})
    assert report["operand_layout_roster"]["status"] == "declared_subset"


def test_signature_knownness_reports_mostly_null_axes():
    sparse = [{"compute": {"op": None, "form": "x"}, "dispatch": None, "op": "region"}] * 3
    report = axis_knownness(sparse)
    assert report["occurrences"] == 3
    assert report["known_occurrences_by_axis"] == {"compute.form": 3, "compute.op": 0, "dispatch": 0, "op": 3}
    assert report["never_known_axes"] == 2
    assert report["known_axes_per_occurrence"] == {"min": 2, "median": 2, "max": 2}
    assert json.dumps(report)
