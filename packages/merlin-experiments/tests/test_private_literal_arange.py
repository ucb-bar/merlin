"""Neutral closed-source witnesses for literal signed-i64 ranges."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_literal_arange_admission as admission
from merlin_experiments.phase1.feedback.private_literal_arange import PENDING, prove_literal_arange_source

from merlin.common import mlir_query as mq
from merlin.frontends.capture_normalization import normalize_capture_mlir


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _source(start: int, step: int, extent: int) -> str:
    # Printed generic-form MLIR from a neutral frontend i64 arange capture,
    # with only its independent literal/extent dimensions parameterized.
    prov = 'prov.source_node_ids = ["g:prepared:root:n1"], prov.origin_node_ids = ["g:original:root:n1"]'
    return f""""builtin.module"() ({{
  "func.func"() <{{sym_name = "forward", function_type = () -> tensor<{extent}xi64>}}> ({{
  ^bb0:
    %0 = "tensor.empty"() {{{prov}, prov.aten = "aten.arange.start_step"}} : () -> tensor<{extent}xi64>
    %1 = "linalg.generic"(%0) <{{indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}}> ({{
    ^bb1(%2: i64):
      %3 = "linalg.index"() <{{dim = 0 : i64}}> {{{prov}}} : () -> index
      %4 = "arith.index_cast"(%3) {{{prov}}} : (index) -> i64
      %5 = "arith.constant"() <{{value = {step} : i64}}> {{{prov}}} : () -> i64
      %6 = "arith.muli"(%4, %5) <{{overflowFlags = #arith.overflow<none>}}> {{{prov}}} : (i64, i64) -> i64
      %7 = "arith.constant"() <{{value = {start} : i64}}> {{{prov}}} : () -> i64
      %8 = "arith.addi"(%7, %6) <{{overflowFlags = #arith.overflow<none>}}> {{{prov}}} : (i64, i64) -> i64
      "linalg.yield"(%8) {{{prov}}} : (i64) -> ()
    }}) {{{prov}, prov.aten = "aten.arange.start_step"}} : (tensor<{extent}xi64>) -> tensor<{extent}xi64>
    "func.return"(%1) : (tensor<{extent}xi64>) -> ()
  }}) : () -> ()
}}) : () -> ()
"""


def _write_capture(tmp_path, *, start=-7, end=8, step=3, extent=5, change_source=None, change_trace=None):
    capture = tmp_path / "capture"
    capture.mkdir()
    source = _source(start, step, extent)
    if change_source is not None:
        source = change_source(source)
    (capture / "model.mlir").write_text(source)
    module = mq.parse(source)
    operations = list(mq.walk(module))
    lower = [i for i, op in enumerate(operations) if mq.attr_str(op, "prov.aten") == "aten.arange.start_step"]
    assert len(lower) == 2
    source_ids = [i for i, op in enumerate(operations) if "prov.source_node_ids" in op.attributes]

    def node(graph):
        return {
            "id": f"g:{graph}:root:n1",
            "op": "call_function",
            "target": "aten.arange.start_step",
            "args": [start, end, step],
            "kwargs": {
                "dtype": {"kind": "dtype", "value": "torch.int64"},
                "device": {"kind": "device", "value": "cpu"},
            },
            "results": [
                {
                    "kind": "tensor",
                    "dtype": "int64",
                    "storage_dtype": "int64",
                    "device": "cpu",
                    "layout": "torch.strided",
                    "stride": [1],
                    "shape": [extent],
                }
            ],
            "origin_node_ids": ["g:original:root:n1"] if graph == "prepared" else [],
        }

    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "graphs": {graph: {"status": "complete", "nodes": [node(graph)]} for graph in ("original", "prepared")},
        "mlir": {
            "sha256": _sha(source.encode()),
            "bytes": len(source.encode()),
            "operations": [
                {
                    "ordinal": i,
                    "operation": mq.op_name(op),
                    "source_node_ids": ["g:prepared:root:n1"] if i in source_ids else [],
                }
                for i, op in enumerate(operations)
            ],
            "source_correspondence": [
                {"node_id": "g:prepared:root:n1", "status": "lowered", "mlir_ordinals": source_ids}
            ],
        },
    }
    if change_trace is not None:
        change_trace(trace)
    (capture / "frontend-trace.json").write_text(json.dumps(trace))
    (capture / "weights.safetensors").write_bytes(b"empty neutral weights")
    (capture / "weights.safetensors.manifest.json").write_text("{}")
    artifacts = {}
    for name in ("model.mlir", "frontend-trace.json", "weights.safetensors", "weights.safetensors.manifest.json"):
        data = (capture / name).read_bytes()
        artifacts[name] = {"sha256": _sha(data), "bytes": len(data)}
    (capture / "capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "artifacts": artifacts,
                "materialized_abi": {"complete": True},
            }
        )
    )
    return capture


@pytest.mark.parametrize(
    "start,end,step,extent",
    [
        (-7, 8, 3, 5),
        (8, -7, -3, 5),
        (4, 4, 1, 0),
        (-(1 << 63), (1 << 63) - 1, (1 << 63) - 1, 3),
    ],
)
def test_closed_source_including_negative_empty_and_wrapped_intermediate(tmp_path, start, end, step, extent):
    proof = prove_literal_arange_source(
        _write_capture(tmp_path, start=start, end=end, step=step, extent=extent), index_bits=64
    )
    assert proof["status"] == PENDING
    assert proof["count"] == 1
    assert proof["occurrences"][0]["extent"] == extent
    assert proof["index_bits_premise"] == 64
    assert len(proof["occurrences"][0]["lowering_ordinals"]) == 9
    assert proof["occurrences"][0]["original_to_prepared_equivalence"] == "not_proved"


def test_omitted_range_options_are_decided_by_the_observed_result(tmp_path):
    def omit_options(trace):
        for graph in trace["graphs"].values():
            graph["nodes"][0]["kwargs"] = {}

    proof = prove_literal_arange_source(_write_capture(tmp_path, change_trace=omit_options), index_bits=64)
    assert proof["count"] == 1

    def omit_and_change_result(trace):
        omit_options(trace)
        trace["graphs"]["prepared"]["nodes"][0]["results"][0]["device"] = "meta"

    changed = tmp_path / "changed"
    changed.mkdir()
    with pytest.raises(ValueError, match="literal arange source refusal"):
        prove_literal_arange_source(_write_capture(changed, change_trace=omit_and_change_result), index_bits=64)


def test_original_linspace_ancestry_is_recorded_without_equivalence_claim(tmp_path):
    def linspace(trace):
        original = trace["graphs"]["original"]["nodes"][0]
        original["target"] = "aten.linspace.default"
        original["args"] = [0.0, 1.0, 5]
        original["results"][0]["dtype"] = "float64"
        original["results"][0]["storage_dtype"] = "float64"

    proof = prove_literal_arange_source(_write_capture(tmp_path, change_trace=linspace), index_bits=64)
    assert proof["occurrences"][0]["original_target"] == "aten.linspace.default"
    assert proof["occurrences"][0]["original_to_prepared_equivalence"] == "not_proved"


@pytest.mark.parametrize(
    "values",
    [
        {"step": 0, "extent": 0},
        {"extent": 4},
        {"end": 9},
        {"start": 5, "end": 3, "step": 1, "extent": 0},
        {"start": 3, "end": 5, "step": -1, "extent": 0},
        {"start": (1 << 63) - 1, "end": (1 << 63) - 1, "step": 1, "extent": 1},
    ],
)
def test_invalid_literal_or_extent_refuses(tmp_path, values):
    with pytest.raises(ValueError, match="literal arange source refusal"):
        prove_literal_arange_source(_write_capture(tmp_path, **values), index_bits=64)


def test_selected_index_width_is_a_refusing_premise(tmp_path):
    with pytest.raises(ValueError, match="extent exceeds selected width"):
        prove_literal_arange_source(_write_capture(tmp_path), index_bits=3)
    (tmp_path / "limit").mkdir()
    with pytest.raises(ValueError, match="extent exceeds selected width"):
        prove_literal_arange_source(_write_capture(tmp_path / "limit", start=0, end=4, step=1, extent=4), index_bits=3)
    with pytest.raises(ValueError, match="index width"):
        prove_literal_arange_source(tmp_path, index_bits=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text.replace("affine_map<(d0) -> (d0)>", "affine_map<(d0) -> (0)>"),
        lambda text: text.replace('"arith.addi"(%7, %6)', '"arith.addi"(%6, %7)'),
        lambda text: text.replace("#arith.overflow<none>", "#arith.overflow<nsw>", 1),
        lambda text: text.replace('%8 = "arith.addi"', '%8 = "arith.addf"'),
    ],
)
def test_changed_body_or_map_refuses(tmp_path, mutation):
    capture = _write_capture(tmp_path, change_source=mutation)
    with pytest.raises(ValueError):
        prove_literal_arange_source(capture, index_bits=64)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text.replace(" : () -> index\n", " : () -> i64\n", 1).replace(
            " : (index) -> i64\n", " : (i64) -> i64\n", 1
        ),
        lambda text: text.replace("value = 3 : i64", "value = 3 : i32", 1),
        lambda text: text.replace('"func.return"(%1)', '"func.return"(%0)'),
    ],
)
def test_invalid_typed_source_or_unconsumed_result_refuses(tmp_path, mutation):
    capture = _write_capture(tmp_path, change_source=mutation)
    with pytest.raises(ValueError):
        prove_literal_arange_source(capture, index_bits=64)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda trace: trace["graphs"]["prepared"]["nodes"][0]["args"].__setitem__(0, {"node_id": "input"}),
        lambda trace: trace["graphs"]["prepared"]["nodes"][0].__setitem__("target", "aten.arange.default"),
        lambda trace: trace["graphs"]["prepared"]["nodes"][0]["origin_node_ids"].clear(),
        lambda trace: trace["mlir"]["source_correspondence"][0]["mlir_ordinals"].pop(),
        lambda trace: trace["mlir"].__setitem__("sha256", "0" * 64),
        lambda trace: trace["graphs"]["prepared"]["nodes"][0]["kwargs"].__setitem__("pin_memory", 0),
        lambda trace: trace["graphs"]["prepared"]["nodes"][0]["kwargs"].__setitem__(
            "device", {"kind": "device", "value": "meta"}
        ),
    ],
)
def test_changed_trace_literal_lineage_or_ordinal_refuses(tmp_path, mutation):
    capture = _write_capture(tmp_path, change_trace=mutation)
    with pytest.raises(ValueError, match="literal arange source refusal"):
        prove_literal_arange_source(capture, index_bits=64)


def test_reviewed_host_row_requires_exact_source_and_linked_build(tmp_path):
    capture = _write_capture(tmp_path)
    raw = (capture / "model.mlir").read_bytes()
    normalized, _ = normalize_capture_mlir(raw.decode())
    parsed = tuple(mq.walk(mq.parse(normalized)))
    ordinal = next(i for i, op in enumerate(parsed) if mq.op_name(op) == "linalg.generic")
    selected = {"schema": "merlin.selected-index-lowering.v1", "index_bits": 64}
    row = {
        "frontend_op": "aten.arange.start_step",
        "mlir_operation": "linalg.generic",
        "ordinals": [ordinal],
        "count": 1,
    }
    host = {
        "status": "admitted",
        "reviewed": True,
        "profiles": [
            {"status": "admitted", "reviewed": True, "profile": "selected", "capability_spec_sha256": "a" * 64}
        ],
    }
    witness = admission.begin(_sha(raw), _sha(normalized.encode()), len(parsed), selected)
    with pytest.raises(ValueError, match="reviewed host declaration"):
        admission.record(witness, capture, row, {**host, "reviewed": False}, parsed, {ordinal: row})
    unproved_form = {**row, "frontend_op": "aten.arange.default"}
    with pytest.raises(ValueError, match="unproved frontend form"):
        admission.record(witness, capture, unproved_form, host, parsed, {ordinal: unproved_form})
    admission.record(witness, capture, row, host, parsed, {ordinal: row})
    assert witness["source_proof"]["count"] == 1
    assert witness["admissions"][0]["ordinals"] == [ordinal]
    source = {
        admission.FIELD: witness,
        "source_sha256": _sha(raw),
        "normalized_source_sha256": _sha(normalized.encode()),
        "capture_receipt_sha256": witness["source_proof"]["capture_receipt_sha256"],
        "n_source_operations": len(parsed),
        "selected_index_observation": selected,
    }
    linked = {"candidate_tree_sha256": "b" * 64, "capture_tree_sha256": "c" * 64, "elf_sha256": "d" * 64}
    admission.link(source, selected, linked)
    entry = {
        "source_sha256": _sha(raw),
        "capture_tree_sha256": "c" * 64,
        "elf_sha256": "d" * 64,
        "index_lowering": selected,
    }
    assert admission.linked_complete(source, entry, "b" * 64)
    for mutate in (
        lambda record: record[admission.FIELD].pop("source_proof"),
        lambda record: record[admission.FIELD]["source_proof"].__setitem__("frontend_trace_sha256", "0" * 64),
        lambda record: record[admission.FIELD]["source_proof"].__setitem__("index_bits_premise", 32),
        lambda record: record[admission.FIELD]["admissions"][0].__setitem__("ordinals", [True]),
        lambda record: record[admission.FIELD]["linked_build"].__setitem__("elf_sha256", "0" * 64),
    ):
        changed = deepcopy(source)
        mutate(changed)
        assert not admission.linked_complete(changed, entry, "b" * 64)
    malformed_width = deepcopy(source)
    malformed_width["selected_index_observation"]["index_bits"] = True
    malformed_width[admission.FIELD]["selected_index_observation"]["index_bits"] = True
    malformed_width[admission.FIELD]["source_proof"]["index_bits_premise"] = True
    malformed_width[admission.FIELD]["source_proof_sha256"] = _sha(
        json.dumps(malformed_width[admission.FIELD]["source_proof"], sort_keys=True, separators=(",", ":")).encode()
    )
    malformed_entry = deepcopy(entry)
    malformed_entry["index_lowering"]["index_bits"] = True
    assert not admission.linked_complete(malformed_width, malformed_entry, "b" * 64)


def test_no_reviewed_host_arange_keeps_empty_mandatory_roster(tmp_path):
    capture = _write_capture(tmp_path)
    witness = admission.begin("a" * 64, "b" * 64, 3, None)
    row = {"frontend_op": "aten.add.Tensor", "ordinals": [0], "count": 1}
    parsed = tuple(mq.walk(mq.parse((capture / "model.mlir").read_text())))
    admission.record(witness, capture, row, {"status": "admitted", "reviewed": True}, parsed, {0: row})
    assert witness["source_proof"] is None and witness["admissions"] == []


def test_linked_two_range_roster_refuses_omitted_host_admission(tmp_path):
    capture = _write_capture(tmp_path)
    raw = (capture / "model.mlir").read_bytes()
    normalized, _ = normalize_capture_mlir(raw.decode())
    parsed = tuple(mq.walk(mq.parse(normalized)))
    generic = next(i for i, op in enumerate(parsed) if mq.op_name(op) == "linalg.generic")
    selected = {"schema": "merlin.selected-index-lowering.v1", "index_bits": 64}
    row = {
        "frontend_op": "aten.arange.start_step",
        "mlir_operation": "linalg.generic",
        "ordinals": [generic],
        "count": 1,
    }
    host = {
        "status": "admitted",
        "reviewed": True,
        "profiles": [
            {"status": "admitted", "reviewed": True, "profile": "selected", "capability_spec_sha256": "a" * 64}
        ],
    }
    witness = admission.begin(_sha(raw), _sha(normalized.encode()), len(parsed), selected)
    admission.record(witness, capture, row, host, parsed, {generic: row})
    source = {
        admission.FIELD: witness,
        "source_sha256": _sha(raw),
        "normalized_source_sha256": _sha(normalized.encode()),
        "capture_receipt_sha256": witness["source_proof"]["capture_receipt_sha256"],
        "n_source_operations": 2 * len(parsed),
        "selected_index_observation": selected,
    }
    witness["n_source_operations"] = source["n_source_operations"]
    second = deepcopy(witness["source_proof"]["occurrences"][0])
    second["lowering_ordinals"] = [value + len(parsed) for value in second["lowering_ordinals"]]
    second["source_ordinal"] = second["lowering_ordinals"][1]
    second["source_node_id"] = "g:prepared:root:n2"
    witness["source_proof"]["occurrences"].append(second)
    witness["source_proof"]["count"] = 2
    witness["source_proof_sha256"] = _sha(
        json.dumps(witness["source_proof"], sort_keys=True, separators=(",", ":")).encode()
    )
    linked = {"candidate_tree_sha256": "b" * 64, "capture_tree_sha256": "c" * 64, "elf_sha256": "d" * 64}
    admission.link(source, selected, linked)
    entry = {
        "source_sha256": _sha(raw),
        "capture_tree_sha256": "c" * 64,
        "elf_sha256": "d" * 64,
        "index_lowering": selected,
    }
    witness["admissions"].append(
        {
            "ordinals": [second["source_ordinal"]],
            "profile": "selected",
            "capability_spec_sha256": "a" * 64,
        }
    )
    assert admission.linked_complete(source, entry, "b" * 64)
    witness["source_proof"]["occurrences"][1]["extent"] = 1 << 63
    witness["source_proof_sha256"] = _sha(
        json.dumps(witness["source_proof"], sort_keys=True, separators=(",", ":")).encode()
    )
    assert not admission.linked_complete(source, entry, "b" * 64)
    witness["source_proof"]["occurrences"][1]["extent"] = 5
    witness["source_proof_sha256"] = _sha(
        json.dumps(witness["source_proof"], sort_keys=True, separators=(",", ":")).encode()
    )
    assert admission.linked_complete(source, entry, "b" * 64)
    witness["admissions"].pop()
    assert not admission.linked_complete(source, entry, "b" * 64)


def test_unrelated_global_trace_diagnostic_does_not_replace_scoped_range_proof(tmp_path):
    def diagnostic(trace):
        trace['status'] = 'diagnostic'
        trace['blockers'] = ['quantized -> prepared correspondence incomplete']
    capture = _write_capture(tmp_path, change_trace=diagnostic)
    proof = prove_literal_arange_source(capture, index_bits=64)
    assert proof['frontend_trace_status'] == 'diagnostic'
    assert proof['source_correspondence_scope'] == 'all captured prepared i64 literal ranges'
    assert proof['occurrences'][0]['original_to_prepared_equivalence'] == 'not_proved'


def test_diagnostic_trace_still_refuses_missing_range_join(tmp_path):
    def missing_range(trace):
        trace['status'] = 'diagnostic'
        trace['mlir']['source_correspondence'] = []
    capture = _write_capture(tmp_path, change_trace=missing_range)
    with pytest.raises(ValueError, match='range source join is ambiguous'):
        prove_literal_arange_source(capture, index_bits=64)
