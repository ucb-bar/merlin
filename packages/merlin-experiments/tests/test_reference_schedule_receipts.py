"""Recorded construction decisions never authenticate a compiler or execution."""

import copy
import hashlib
import json

import pytest
from merlin_experiments.phase1.feedback import reference_schedule_receipts as receipts
from merlin_experiments.phase1.feedback.reference_views import artifact_views, compare_views
from test_reference_views import inputs


def fixture():
    source, cb, emitted = inputs()
    identity = hashlib.sha256(source).hexdigest()
    text = '''"builtin.module"() ({
  "schedule.plan"() <{sym_name = "example"}> ({
    %0 = "schedule.bind"() <{target = "dense:SOURCE:CONTEXT"}> : () -> !schedule.handle
    "schedule.block_contraction"() {problem = array<i64: 48, 32, 24>,
      primitive_tile = array<i64: 8, 8, 8>, block_tiles = array<i64: 2, 2, 1>,
      element_bytes = array<i64: 1, 1, 4>, storage_bytes = array<i64: 384, 512>,
      storage_capacity = array<i64: 4096, 2048>, static_commands = 5 : i64,
      command_budget = 64 : i64, outer_order = ["m", "n"], row_bases = array<i64: 0, 256, 0>}
      : () -> ()
    "schedule.panel_pipeline"(%0) <{prefetch = false, kind = "retained_full_k"}>
      : (!schedule.handle) -> ()
  }) : () -> ()
}) : () -> ()'''.replace("SOURCE", identity).replace("CONTEXT", "a" * 64)
    record = dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest(), source_identity=identity,
                  pipeline_kind="retained_full_k", prefetch=False,
                  decisions_consumed=["block_tiles", "outer_order", "row_bases", "panel_pipeline"])
    cb["params"] = dict(construction=dict(schema="merlin.compact_dense_realization.v1",
                        kernel_sha256="b" * 64, schedule_ir=record))
    return source, cb, emitted, record


def inspect(source, cb, emitted):
    return receipts.inspect_schedule_receipts(source_mlir=source, command_buffer=json.dumps(cb).encode(),
                                              emitted_mlir=emitted)


def test_exact_source_bound_decisions_keep_capacity_and_emission_proofs_separate():
    source, cb, emitted, _ = fixture()
    report = inspect(source, cb, emitted)
    assert report["valid_records"] == 1
    row, = report["records"]
    assert row["choices"]["block"]["problem"] == [48, 32, 24]
    assert row["choices"]["block"]["primitive_tile"] == [8, 8, 8]
    assert row["choices"]["pipeline"] == dict(prefetch=False, kind="retained_full_k")
    assert row["capacity_condition"]["used_bytes"] == [384, 512]
    assert row["capacity_condition"]["selected_hardware_and_simultaneous_lifetimes"] == "unknown"
    assert row["source_shape_correspondence"] == row["emitted_correspondence"] == "unknown"
    assert report["schedule_consumption"] == "recorded_only"
    assert report["compiler_and_emission_binding"] == report["target_execution"] == "unknown"
    assert report["transformation_records"][0]["hash_links"] == dict(kernel_sha256="b" * 64)
    assert "executed_instruction_calls" not in json.dumps(report)


@pytest.mark.parametrize("mutation,reason", [
    ("source", "schedule_source_identity_mismatch"),
    ("digest", "schedule_text_digest_mismatch"),
    ("bind", "schedule_bind_source_mismatch"),
    ("pipeline", "receipt_pipeline_differs_from_decision_ir"),
    ("roster", "consumed_decision_roster_unknown"),
    ("capacity", "schedule_decision_parse_or_verification_refused"),
])
def test_changed_or_conflicting_receipts_are_unavailable(mutation, reason):
    source, cb, emitted, record = fixture()
    if mutation == "source":
        source += b"\n"
    elif mutation == "digest":
        record["text"] += "\n"
    elif mutation == "bind":
        record["text"] = record["text"].replace(record["source_identity"], "c" * 64)
        record["sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
    elif mutation == "pipeline":
        record["prefetch"] = True
    elif mutation == "capacity":
        record["text"] = record["text"].replace("array<i64: 4096, 2048>", "array<i64: 1, 1>")
        record["sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
    else:
        record["decisions_consumed"] += ["target_execution"]
    report = inspect(source, cb, emitted)
    assert report["valid_records"] == 0
    assert report["records"][0]["reason"] == reason
    assert report["schedule_consumption"] == report["compiler_and_emission_binding"] == "unknown"


def test_receipt_and_node_limits_report_partial_not_complete_absence(monkeypatch):
    source, cb, emitted, _ = fixture()
    cb["params"]["second"] = copy.deepcopy(cb["params"]["construction"])
    monkeypatch.setattr(receipts, "MAX_RECEIPTS", 1)
    report = inspect(source, cb, emitted)
    assert report["status"] == report["budget_status"] == "partial"
    assert "schedule_receipt_budget_exhausted" in report["missing"]
    monkeypatch.setattr(receipts, "MAX_NODES", 1)
    report = inspect(source, cb, emitted)
    assert report["status"] == "partial"
    assert "metadata_node_or_depth_budget_exhausted" in report["missing"]


def test_actual_cross_view_entry_exposes_receipts_without_count_or_semantic_promotion():
    source, cb, emitted, _ = fixture()
    report = artifact_views(source_mlir=source, command_buffer=json.dumps(cb).encode(), emitted_mlir=emitted)
    assert report["construction_receipts"]["valid_records"] == 1
    assert report["schedule_consumption"] == "recorded_only"
    assert report["mechanism_status"] == report["semantic_equivalence"] == report["numerical_status"] == "unknown"
    assert report["emitted"]["counted_work"]["command_counts"] == {}


def test_ordinary_source_structure_remains_visible_without_inventing_its_lowering():
    _, cb, emitted, _ = fixture()
    source = b'''builtin.module {
      func.func @ordinary(%arg: tensor<2x3xi8>) -> tensor<2x3xi8> {
        func.return %arg : tensor<2x3xi8>
      }
    }'''
    report = artifact_views(source_mlir=source, command_buffer=json.dumps(cb).encode(), emitted_mlir=emitted)
    assert report["source"]["status"] == "observed_source_structure"
    assert report["source"]["functions"][0]["inputs"] == [dict(shape=[2, 3], dtype="i8")]
    assert report["source_command_join"] == "unknown"
    assert report["construction_receipts"]["valid_records"] == 0  # It names another original source.
    compared = compare_views(report, report)
    assert compared["declared_source_join"] == "unknown_or_mismatch"
    assert all(row["status"] == "unknown" for row in compared["issue_comparison"].values())


def test_host_product_join_rechecks_all_reopened_domain_bytes_but_grants_no_execution():
    source, cb, emitted, _ = fixture()
    raw_cb = json.dumps(cb).encode()
    binding = dict(status="observed_original_build_consumption", compiler_sha256="a" * 64,
        source_sha256=hashlib.sha256(source).hexdigest(), command_buffer_sha256=hashlib.sha256(raw_cb).hexdigest(),
        emitted_sha256=hashlib.sha256(emitted).hexdigest(), object_sha256="b" * 64,
        linked_elf_sha256="c" * 64, function_view_sha256="d" * 64)
    report = receipts.inspect_schedule_receipts(source_mlir=source, command_buffer=raw_cb,
                                                emitted_mlir=emitted, producer_binding=binding)
    assert report["compiler_and_emission_binding"] == "observed_original_build_consumption"
    assert report["schedule_consumption"] == "recorded_in_observed_compiler_products"
    assert report["target_execution"] == report["semantic_equivalence"] == "unknown"
    assert report["records"][0]["emitted_correspondence"] == "unknown"
    changed = receipts.inspect_schedule_receipts(source_mlir=source, command_buffer=raw_cb,
        emitted_mlir=emitted + b"\n", producer_binding=binding)
    assert changed["compiler_and_emission_binding"] == "unknown"
    assert changed["schedule_consumption"] == "recorded_only"
    assert changed["producer_binding"]["reason"] == "original_build_product_pins_do_not_match"


def test_comparison_reports_recorded_choice_populations_without_pairing_groups_or_cca_axes():
    source, cb, emitted, record = fixture()
    left = inspect(source, cb, emitted)
    record["text"] = record["text"].replace('outer_order = ["m", "n"]', 'outer_order = ["n", "m"]')
    record["sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
    right = inspect(source, cb, emitted)
    compared = receipts.compare_schedule_receipts(left, right)
    assert compared["status"] == "compared_recorded_choice_populations"
    changed, = [row for row in compared["choices"] if row["differs"]]
    assert changed["axis"] == "schedule.block.outer_order"
    assert compared["canonical_cca_axes_added"] == 0
    assert compared["semantic_equivalence"] == compared["optimization_application"] == "unknown"
    right["budget_status"] = "partial"
    assert receipts.compare_schedule_receipts(left, right)["status"] == "independent_recorded_choices"
    assert all(row["differs"] is None for row in receipts.compare_schedule_receipts({}, right)["choices"])
