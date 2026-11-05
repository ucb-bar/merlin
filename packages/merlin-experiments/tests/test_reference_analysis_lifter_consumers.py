"""Reference analysis passes lifter trip counts and protocol decoder structure through."""

from __future__ import annotations

import hashlib
import json

from merlin_experiments.phase1.feedback import reference_analysis as analysis

from merlin.targetgen.rocc import selected_operand_protocol


def _selection(tmp_path, **extra):
    sidecar = dict(
        layouts=dict(fields=[dict(bundle="B", name="addr", status="derived"),
                             dict(bundle="B", name="rows", status="derived")]),
        protocol=dict(roles=dict(classes={"LOAD": dict(fields=[dict(name="addr", operand="rs1", bundle="B",
                                                                   field="addr")])})),
        **extra,
    )
    path = tmp_path / "sidecar.json"
    path.write_bytes(json.dumps(sidecar).encode())
    return dict(sidecar=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_trip_counts_pass_through_only_when_proved():
    assert analysis._trip_counts(None) == "unknown"
    nest = dict(value=dict(levels=[dict(header_pc=4, nest_depth=1, trip_count=dict(status="proved", value=16)),
                                   dict(header_pc=8, nest_depth=2, trip_count=dict(status="unknown"))]))
    counts = analysis._trip_counts(dict(loop_nest=nest))
    assert counts["status"] == "partial" and counts["proved"] == [dict(header_pc=4, nest_depth=1, value=16)]
    nest["value"]["levels"].pop()
    assert analysis._trip_counts(dict(loop_nest=nest))["status"] == "proved"


def test_layout_roster_derives_from_the_selected_sidecar(tmp_path, monkeypatch):
    monkeypatch.delattr(selected_operand_protocol, "selected_decoder_structure", raising=False)
    structure = analysis._protocol_decoder_structure(_selection(tmp_path))
    assert "operand_field_kinds" not in structure
    roster = structure["operand_layout_roster"]
    assert roster["status"] == "declared_subset"
    assert roster["unprojected_fields"] == [dict(bundle="B", field="rows")]


def test_declared_kinds_and_roster_are_copied(tmp_path, monkeypatch):
    monkeypatch.delattr(selected_operand_protocol, "selected_decoder_structure", raising=False)
    roster = dict(status="complete", derived_fields=2, unprojected_fields=[])
    structure = analysis._protocol_decoder_structure(
        _selection(tmp_path, operand_field_kinds={"LOAD.rows": "extent"}, operand_layout_roster=roster))
    assert structure == dict(operand_field_kinds={"LOAD.rows": "extent"}, operand_layout_roster=roster)


def test_protocol_owner_declarations_win(tmp_path, monkeypatch):
    seen = []

    def selected_decoder_structure(selection):
        seen.append(selection)
        return dict(operand_field_kinds={"LOAD.addr": "address"}, operand_layout_roster=None)

    monkeypatch.setattr(selected_operand_protocol, "selected_decoder_structure", selected_decoder_structure,
                        raising=False)
    selection = _selection(tmp_path)
    structure = analysis._protocol_decoder_structure(selection)
    assert seen == [selection]
    assert structure["operand_field_kinds"] == {"LOAD.addr": "address"}
    assert structure["operand_layout_roster"]["status"] == "declared_subset"


def test_no_selection_adds_nothing():
    assert analysis._protocol_decoder_structure(None) == {}
