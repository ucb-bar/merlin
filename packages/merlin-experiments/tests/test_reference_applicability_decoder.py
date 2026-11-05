"""Bank decoding and run resource admission are independent host selections."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_applicability as applicability
from merlin_experiments.phase1.feedback import reference_bank as bank

from merlin.targetgen.rocc.normalized_operands import BoundOperandObserver

TARGET = "array_fixture"


def _selection(root, *, opcode, offset, rows):
    root.mkdir()
    fields = {"nr": dict(offset=offset, width=4), "nc": dict(offset=offset + 4, width=4)}
    protocol = dict(version=1, encoding=dict(semantic_class={4: "FETCH"}), roles=dict(classes={
        "FETCH": dict(fields=[dict(name=name, operand="rs2", bundle="Packet", field=name)
                              for name in fields])}))
    facts = dict(interfaces=[dict(name="register_bundle_layouts", bundles={
        "Packet": dict(width=64, fields=fields)})],
        arrays=[dict(name="array", rows=4, cols=4)],
        memories=[dict(name="operands", bytes=4 * rows, depth=rows),
                  dict(name="accumulator", bytes=16 * 32, depth=32)],
        datapaths=[dict(name="operands", dtype="int8"), dict(name="accumulator", dtype="int32")])
    sidecar = dict(protocol=protocol, layouts=dict(fields=[dict(bundle="Packet", name=name, **field)
                                                         for name, field in fields.items()]),
                   load_state_selection=dict(transfer_bound={axis: dict(bundle="Packet", name=name)
                                                              for axis, name in (("rows", "nr"), ("cols", "nc"))}))
    sidecar_path = root / "operand-protocol.json"
    sidecar_path.write_text(json.dumps(sidecar))
    contract = root / "contract.yaml"
    contract.write_text(yaml.safe_dump(dict(rocc_operand_selection=dict(sidecar=dict(
        path=str(sidecar_path), sha256=hashlib.sha256(sidecar_path.read_bytes()).hexdigest())))))
    fact_path = root / "facts.json"
    fact_path.write_text(json.dumps(facts))
    observer = BoundOperandObserver(target=TARGET, protocol=protocol, facts=facts,
        raw_isa=dict(CUSTOM_OPCODE=opcode, FUNCT3=3, FUNCT_CLASS={4: "RAW4"}), word_bits=64)
    return dict(contract=contract, facts=fact_path, target=TARGET, descriptor=None), dict(
        target=TARGET, semantics=observer, structure=dict(instruction_roles={"FETCH": ["operand_load"]}))


def _region(name, *, computes):
    return dict(stable_id=name, address_range=[0, 12], cfg=dict(natural_loops=0),
                mechanism_projection=dict(instruction_classes={"FETCH": 1}),
                schedule_model=dict(role_counts=dict(operand_load=4, accumulate=computes)))


@pytest.mark.parametrize("run_opcode,run_offset", [(91, 0), (43, 8)])
def test_bank_decoder_is_shared_by_compare_and_applicability_while_run_capacity_stays_selected(
        tmp_path, monkeypatch, run_opcode, run_offset):
    bank_context, bank_decoder = _selection(tmp_path / "bank", opcode=43, offset=0, rows=256)
    run_context, run_decoder = _selection(tmp_path / "run", opcode=run_opcode, offset=run_offset, rows=64)
    workspace = tmp_path / "workspace"
    (workspace / "artifacts").mkdir(parents=True)
    # x6 = packed4x4; one neutral accelerator operation and return.
    word = 4 << 25 | 6 << 20 | 3 << 12 | 43
    (workspace / 'artifacts' / 'kernel.dis').write_text(
        f"00000000 <kernel>:\n 0: 04400313 addi t1,zero,68\n 4: {word:08x} .insn\n 8: 00008067 ret\n")
    selector = dict(path=(Path('artifacts') / 'kernel.dis').as_posix(), format="disassembly")
    candidate = _region("candidate", computes=2)
    decoder_calls = []

    def selected_decoder(contract, facts, target, *, descriptor=None, include_structure=False):
        decoder_calls.append(dict(contract=Path(contract), facts=Path(facts), target=target,
                                  descriptor=descriptor))
        return bank_decoder if Path(contract) == bank_context["contract"] else run_decoder

    monkeypatch.setattr(analysis, "_operand_decoder", selected_decoder)
    reference = _region("reference", computes=16)
    reference["derived_effects"], _, gap = applicability._candidate_effects(
        selector, candidate, workspace=workspace, **bank_context)
    assert gap is None
    # Prove the selections really disagree under actual instruction/operand decoding.
    wrong, _, _ = applicability._candidate_effects(selector, candidate, workspace=workspace, **run_context)
    expected_tiles = reference["derived_effects"]["facts"]["transfer_tiles"]
    assert expected_tiles["value"]["by_role"]["operand_load"] == [dict(rows=4, cols=4, sites=1)]
    assert wrong["facts"]["transfer_tiles"] != expected_tiles
    reference["bank_occurrence"] = dict(unit_id="fixture", region_id="reference", original_id="original")
    monkeypatch.setattr(bank, "open_bank", lambda *a, **k: (tmp_path / "bank", {}))
    monkeypatch.setattr(bank, "_bank_decoder", lambda *a: {k: bank_context[k] for k in ("facts", "contract")})
    monkeypatch.setattr(bank, "_lift", lambda *a, **k: (dict(regions=[reference]), None))
    analyzed_contexts = []

    def analyze(value, **kwargs):
        context = {k: kwargs[k] for k in bank_context}
        analyzed_contexts.append(context)
        effects, _, _ = applicability._candidate_effects(value, candidate, workspace=workspace, **context)
        return dict(regions=[dict(candidate, derived_effects=effects)]), None

    monkeypatch.setattr(analysis, "_analyze", analyze)
    compared = []
    monkeypatch.setattr(analysis, "_compare", lambda left, right, *a, **k: compared.append(right) or right)
    monkeypatch.setattr(analysis, "_response", lambda base, packet, report, *a, **k: report)
    request = dict(entry="fixture", region="reference", candidate=selector)
    options = dict(base={}, workspace=workspace, bundle_manifest=None, repo=None, pin="0" * 64, **run_context)
    bank.handle(dict(request, operation="compare"), **options)
    response = bank.handle(dict(request, operation="applicability", shape=dict(kh=3, kw=3, c=16, oc=4)), **options)
    from merlin_experiments.phase1.feedback.result_details import load_row

    report = load_row(response, workspace=workspace)["report"]
    assert analyzed_contexts == [bank_context, bank_context]
    assert decoder_calls[-1] == bank_context
    assert report["candidate"]["derived_effects"]["facts"]["transfer_tiles"] == expected_tiles
    assert compared[0]["regions"][0]["derived_effects"]["facts"]["transfer_tiles"] == expected_tiles
    assert report["decoder_structure"]["status"] == "same"
    result = report["applicability"]
    assert result["selected_geometry"]["operand_rows"] == 64
    difference = next(row for row in result["differences"] if row["effect"] == "operand_load_per_compute")
    route = next(row for row in difference["routes"]
                 if row["capability"] == "capacity_phased_convolution_rhs_residency")
    # Required148rows fitsbank256 but violates the selectedrun64; do not swap resource facts.
    assert route["admission"]["required_operand_rows"] == 148
    assert route["admission"]["operand_rows"] == 64
    assert route["admission"]["status"] == "violated"
