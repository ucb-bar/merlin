"""The real broker derives mechanism questions; supplied associations prove no join."""

from __future__ import annotations

import hashlib
import json

import pytest
from merlin_experiments.phase1.feedback import reference_analysis as analysis
from merlin_experiments.phase1.feedback import reference_experiments as reference
from merlin_experiments.phase1.feedback.reference_mechanisms import align_questions
from test_reference_analysis_tool import _request, _world

SOURCE = """module {func.func @entry(%a:tensor<2x3xf32>,%b:tensor<3x4xf32>,
      %d:tensor<3x4xf32>,%c:tensor<2x4xf32>,%e:tensor<2x4xf32>)
      -> (tensor<2x4xf32>,tensor<2x4xf32>) {
      %r = linalg.matmul ins(%a,%b:tensor<2x3xf32>,tensor<3x4xf32>)
        outs(%c:tensor<2x4xf32>) -> tensor<2x4xf32>
      %s = linalg.matmul ins(%a,%d:tensor<2x3xf32>,tensor<3x4xf32>)
        outs(%e:tensor<2x4xf32>) -> tensor<2x4xf32>
      func.return %r,%s : tensor<2x4xf32>,tensor<2x4xf32>
    }}"""

EMITTED = """module {
    llvm.func @selected(%a:!llvm.ptr) {
      llvm.call @leaf(%a) : (!llvm.ptr) -> ()
      llvm.call @leaf(%a) : (!llvm.ptr) -> ()
      llvm.return
    }
    llvm.func @leaf(%a:!llvm.ptr) {
      "llvm.inline_asm"() <{asm_string="fence",constraints="",has_side_effects}> : () -> ()
      llvm.return
    }
    llvm.func @unselected_alternative() {
      "llvm.inline_asm"() <{asm_string="fence",constraints="",has_side_effects}> : () -> ()
      llvm.return
    }
}"""


def test_broker_exposes_source_sharing_and_actual_selected_consumer_closure(tmp_path):
    world = _world(tmp_path)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "source.mlir").write_text(SOURCE)
    (directory / "emitted.mlir").write_text(EMITTED)
    with reference.invocation(world["path"], world["sha"]):
        source = analysis.handle(_request("candidate", "artifacts/source.mlir", format="source_mlir"),
                                 workspace=tmp_path)
        emitted = analysis.handle(_request("candidate", "artifacts/emitted.mlir", format="emitted_mlir",
                                          symbol="selected"), workspace=tmp_path)
    assert source["status"] == emitted["status"] == "analyzed", (source, emitted)
    mechanisms = source["report"]["mechanism_observations"]
    shared, = mechanisms["shared_inputs"]
    assert shared["consumers_total"] == 2 and shared["storage_reuse"] == "unknown"
    calls = emitted["report"]["consumer_structure"]
    assert calls["reachable_symbols"] == ["leaf", "selected"]
    assert calls["unselected_definitions"] == 1
    wrapper = next(row for row in calls["functions"] if row["symbol"] == "selected")
    assert wrapper["call_sites_total"] == 2
    assert wrapper["shared_pointer_arguments"][0]["uses_total"] == 2
    assert calls["optimization_application"] == calls["execution_multiplicity"] == "unknown"
    bundle = dict(views=dict(source=source["report"], emitted=emitted["report"]))
    alignment = align_questions(bundle, bundle)
    assert alignment["questions_total"] == 2
    assert alignment["producer_correspondence"] == alignment["performance"] == "unknown"
    assert all(row["optimization_application"] == "unknown" for row in alignment["questions"])


def test_consumer_closure_reports_external_bodies_and_budget_refusal():
    from xdsl.parser import Parser

    from merlin.kernels.consumer_structure import consumer_structure
    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    module = Parser(make_llvm_context(), """module {
      llvm.func @external(!llvm.ptr)
      llvm.func @entry(%a:!llvm.ptr) {llvm.call @external(%a) : (!llvm.ptr) -> ()
        llvm.return}
    }""").parse_module()
    observed = consumer_structure(module, symbol="entry")
    assert next(row for row in observed["functions"] if row["symbol"] == "external")["body"] == "external_unknown"
    refused = consumer_structure(module, symbol="entry", max_operations=1)
    assert refused["reason"] == "consumer_operation_budget_exhausted"


def test_existing_interface_views_expose_shared_rhs_without_claiming_retention():
    from test_reference_views import observe

    report = observe()
    shared, = report["source"]["mechanism_observations"]["shared_inputs"]
    assert shared["value"] == "w" and shared["consumers_total"] == 2
    assert shared["storage_reuse"] == "unknown"
    assert report["emitted"]["consumer_structure"]["roots"] == ["kernel_entry"]


@pytest.mark.parametrize("edge", [8, 16])
def test_source_packing_width_is_derived_from_the_selected_fact_bytes(tmp_path, edge):
    from merlin_experiments.phase1.feedback.reference_mechanisms import source_report

    from merlin.frontends.linalg_mlir import parse_mlir_text

    facts = dict(schema_version="2.0", inputs={}, facts=dict(
        arrays=[dict(name="unit", rows=edge, cols=edge)],
        datapaths=[dict(name="input", dtype="i8", evidence="store_a smem UInt<8>"),
                   dict(name="accumulator", dtype="i32", evidence="store_b smem SInt<32>")],
        memories=[dict(name="store_a", bytes=edge * 128, depth=32),
                  dict(name="store_b", bytes=edge * 256, depth=32)]))
    path = tmp_path / "selected-facts.json"
    path.write_text(json.dumps(facts))
    report = source_report(parse_mlir_text(SOURCE), dict(sha256=hashlib.sha256(SOURCE.encode()).hexdigest()),
                           facts=path, target="array_fixture")
    assert report["reduction_geometry"]["width"] == edge
    packing = report["reductions"][0]["packing"]
    assert packing["lanes"]["width"] == edge
    assert packing["alternatives"][0]["physical_extent"] == edge
    assert report["optimization_application"] == "unknown"
    facts["facts"]["arrays"] = []
    path.write_text(json.dumps(facts))
    missing = source_report(parse_mlir_text(SOURCE), dict(sha256="a" * 64), facts=path, target="array_fixture")
    assert missing["reduction_geometry"]["status"] == "unknown"
    assert missing["reductions"][0]["packing"]["lanes"] is None


def test_closed_interface_convolution_exposes_axis_packing_and_refuses_bad_geometry():
    from merlin_experiments.phase1.feedback.reference_views import source_view

    from merlin.kernels.mechanism_inference import ReductionLanes

    source = b'''module attributes {merlin_iface.version="0.1", merlin_iface.target="fixture"} {
      %a = merlin_iface.tensor {name="a", role="input"} : tensor<1x5x5x2xi8>
      %w = merlin_iface.tensor {name="w", role="weight"} : tensor<18x5xi8>
      %resident = merlin_iface.resident_pack %w {layout="packed_rhs"} : (tensor<18x5xi8>) -> !merlin_iface.resident
      %out = merlin_iface.conv2d %a,%resident {kernel=[3,3,2,5],stride=[1,1],padding=[0,0,0,0],
        dilation=[1,1],name="out",epilogue=[],output_dtype="i32",layout="nhwc"}
        : (tensor<1x5x5x2xi8>,!merlin_iface.resident) -> tensor<9x5xi32>
      merlin_iface.evict %resident : (!merlin_iface.resident) -> ()
    }'''.replace(b",\n        dilation", b",dilation").replace(b"}\n        :", b"} :")
    report = source_view(source, reduction_lanes=ReductionLanes(8, "selected-facts#edge"))
    packing = report["mechanism_observations"]["reductions"][0]["packing"]
    assert packing["reduction_extents"] == [3, 3, 2]
    assert [row["physical_extent"] for row in packing["alternatives"]] == [72, 24, 24]
    assert report["mechanism_observations"]["optimization_application"] == "unknown"
    coverage = report["mechanism_observations"]["shared_input_coverage"]
    assert coverage["status"] == "partial" and coverage["absence_of_sharing"] == "unknown"
    assert coverage["unsupported_commands"] == [dict(operation=1, opcode="CONV2D")]
    bad = source_view(source.replace(b"kernel=[3,3,2,5]", b"kernel=[3,3,4,5]"))
    assert bad["mechanism_observations"]["reductions"] == []
    assert bad["mechanism_observations"]["unknowns_total"] == 1


def test_large_reply_keeps_mechanism_questions_visible_and_full_evidence_in_detail(tmp_path):
    from merlin_experiments.phase1.feedback.result_details import load_row

    from merlin.verify.lean_contract import document_digest

    report = dict(kind="representation_comparison", mechanism_alignment=dict(
        status="independent_observations", questions=[dict(mechanism="shared_operand_lifetime",
        optimization_application="unknown", witness="actual-source-operation")] * 800),
        mechanism_observations=dict(status="observed", optimization_application="unknown"),
        consumer_structure=dict(roots=["actual_selected"], source_correspondence="unknown"))
    response = dict(status="analyzed", report=report, report_identity=document_digest(report),
                    semantic_equivalence="unknown", performance="unknown", qualification_issued=False)
    bounded = analysis._bounded_response(response, workspace=tmp_path)
    assert bounded["status"] == "analyzed" and bounded["report_in_detail"]
    assert bounded["report"]["mechanism_alignment"]["questions"][0]["mechanism"] == "shared_operand_lifetime"
    assert bounded["report"]["consumer_structure"]["roots"] == ["actual_selected"]
    assert bounded["report"]["mechanism_observations"]["optimization_application"] == "unknown"
    assert len(json.dumps(bounded).encode()) < 64 << 10
    assert load_row(bounded, workspace=tmp_path)["report"] == report
