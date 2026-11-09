"""Neutral BUILD-only accounting for reviewed host bucketize occurrences."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.bucketize_source import STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.runtime.modelinputs import strict_tree_sha256
from merlin.targetgen.host_capabilities import admit_host_operation
from merlin_experiments.phase1.feedback import private_bucketize_support as support
from merlin_experiments.phase1.feedback import private_host_source_dispatch as dispatch


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _source(
    boundaries: str = "0.0, 0.25, 0.25, 1.0",
    *,
    dynamic: bool = False,
    frontend_op="aten.bucketize.Tensor",
    input_extent=2,
) -> str:
    boundary_arg = ", %boundaries: tensor<4xf32>" if dynamic else ""
    dense = f'dense<"{boundaries}">' if boundaries.startswith("0x") else f"dense<[{boundaries}]>"
    constant = (
        ""
        if dynamic
        else f'%boundaries = "arith.constant"() <{{value = {dense} : tensor<4xf32>}}> : () -> tensor<4xf32>'
    )
    return f"""builtin.module {{
  func.func @forward(%input: tensor<{input_extent}xf32>{boundary_arg}) -> tensor<{input_extent}xi64> {{
    {constant}
    %zero = "arith.constant"() <{{value = 0 : i64}}> : () -> i64
    %one = "arith.constant"() <{{value = 1 : i64}}> : () -> i64
    %init = "tensor.splat"(%zero) : (i64) -> tensor<{input_extent}xi64>
    %out = "linalg.generic"(%input, %boundaries, %init)
      <{{indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                          affine_map<(d0, d1) -> (d1)>,
                          affine_map<(d0, d1) -> (d0)>],
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
        operandSegmentSizes = array<i32: 2, 1>}}> ({{
      ^bb0(%x: f32, %b: f32, %acc: i64):
        %less = "arith.cmpf"(%b, %x)
          <{{predicate = 11 : i64, fastmath = #arith.fastmath<none>}}> : (f32, f32) -> i1
        %increment = "arith.select"(%less, %one, %zero) : (i1, i64, i64) -> i64
        %count = "arith.addi"(%acc, %increment)
          <{{overflowFlags = #arith.overflow<none>}}> : (i64, i64) -> i64
        "linalg.yield"(%count) : (i64) -> ()
      }}) {{prov.source_node_ids = ["g:prepared:root:n1"], prov.aten = "{frontend_op}"}}
        : (tensor<{input_extent}xf32>, tensor<4xf32>, tensor<{input_extent}xi64>) -> tensor<{input_extent}xi64>
    func.return %out : tensor<{input_extent}xi64>
  }}
}}"""


def _capture(
    tmp_path,
    *,
    boundaries="0.0, 0.25, 0.25, 1.0",
    dynamic=False,
    index_bits=64,
    frontend_op="aten.bucketize.Tensor",
    input_extent=2,
):
    capture = tmp_path / "capture"
    capture.mkdir(parents=True)
    text = _source(boundaries, dynamic=dynamic, frontend_op=frontend_op, input_extent=input_extent)
    (capture / "model.mlir").write_text(text)
    parsed = tuple(mq.walk(mq.parse(text)))
    ordinal = next(index for index, op in enumerate(parsed) if mq.op_name(op) == "linalg.generic")
    original_id = "g:original:root:n1"
    quantized_id = "g:quantized:root:n1"
    prepared_id = "g:prepared:root:n1"
    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "mlir": {
            "sha256": _sha(text.encode()),
            "bytes": len(text.encode()),
            "operations": [
                {
                    "ordinal": index,
                    "operation": mq.op_name(op),
                    "source_node_ids": [prepared_id] if index == ordinal else [],
                    "operand_types": [str(value.type) for value in op.operands],
                    "result_types": [str(value.type) for value in op.results],
                }
                for index, op in enumerate(parsed)
            ],
            "source_correspondence": [{"node_id": prepared_id, "status": "lowered", "mlir_ordinals": [ordinal]}],
        },
        "graphs": {
            "original": {"nodes": [{"id": original_id, "target": frontend_op, "kwargs": {"right": False}}]},
            "quantized": {
                "nodes": [
                    {
                        "id": quantized_id,
                        "target": frontend_op,
                        "kwargs": {"right": False},
                        "origin_node_ids": [original_id, "g:staged:root:n1"],
                    }
                ]
            },
            "prepared": {
                "nodes": [
                    {
                        "id": prepared_id,
                        "target": frontend_op,
                        "kwargs": {"right": False},
                        "origin_node_ids": [original_id, "g:staged:root:n1", quantized_id],
                    }
                ]
            },
        },
        "transformations": [
            {
                "from_stage": "original",
                "to_stage": "quantized",
                "status": "complete",
                "relations": [{"kind": "introduced", "source_ids": [original_id], "destination_ids": [quantized_id]}],
            },
            {
                "from_stage": "quantized",
                "to_stage": "prepared",
                "status": "complete",
                "relations": [{"kind": "introduced", "source_ids": [quantized_id], "destination_ids": [prepared_id]}],
            },
        ],
    }
    (capture / "frontend-trace.json").write_text(json.dumps(trace))
    (capture / "weights.safetensors").write_bytes(b"neutral empty weights")
    (capture / "weights.safetensors.manifest.json").write_text("{}")
    artifacts = {
        name: {"sha256": _sha((capture / name).read_bytes()), "bytes": (capture / name).stat().st_size}
        for name in ("model.mlir", "frontend-trace.json", "weights.safetensors", "weights.safetensors.manifest.json")
    }
    (capture / "capture_receipt.json").write_text(
        json.dumps({"schema": "m2m.capture-receipt.v1", "artifacts": artifacts, "materialized_abi": {"complete": True}})
    )
    normalized = normalize_capture_mlir(text)[0]
    selected = (
        {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": "neutral-clang",
            "compiler_resolved": "/neutral/clang",
            "compiler_sha256": "c" * 64,
            "cross_flags": ["--target=neutral"],
            "data_layout": f"e-p:{index_bits}:{index_bits}",
            "index_bits": index_bits,
            "scope": "neutral selected compiler observation",
        }
        if index_bits
        else None
    )
    witness = support.begin(capture, parsed, _sha(text.encode()), _sha(normalized.encode()), selected)
    return witness, ordinal


def _reviewed(witness, ordinal):
    assert witness["source_proof"]["ordinals"][0][0] == ordinal
    pattern = dict(witness["source_proof"]["ordinals"][0][1])
    fact = witness["boundary_facts"][0]
    decision = {
        "status": "admitted",
        "reviewed": True,
        "source_body_proof": {
            "schema": STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA,
            "declaration": "neutral_closed_bucketize",
            "operation": "f32_literal_boundary_count_i64",
            "predicate": None,
            "patterns": [
                {
                    "operation": "f32_literal_boundary_count_i64",
                    "shape": pattern["input_shape"],
                    "ordered_types": ["f32", "f32", "i64", "i64"],
                    "boundary_count": pattern["boundary_count"],
                    "right": pattern["right"],
                    "comparison": pattern["comparison"],
                    "count_range": pattern["count_range"],
                    "literal_sha256": fact["literal_sha256"],
                }
            ],
            "profile": "neutral",
            "capability_spec_sha256": "a" * 64,
            "selected_index_observation": deepcopy(witness["selected_index_observation"]),
        },
        "profiles": [
            {
                "status": "admitted",
                "reviewed": True,
                "profile": "neutral",
                "capability_spec_sha256": "a" * 64,
            }
        ],
    }
    inner = {
        key: value
        for key, value in decision["source_body_proof"].items()
        if key not in {"profile", "capability_spec_sha256"}
    }
    decision["profiles"][0]["source_body_proof"] = deepcopy(inner)
    decision["profiles"][0]["decisions"] = [
        {
            "status": "admitted",
            "review_status": "reviewed",
            "declaration": inner["declaration"],
            "source_body_proof": deepcopy(inner),
        }
    ]
    return decision


def _row(ordinal):
    return {
        "mlir_operation": "linalg.generic",
        "frontend_op": "aten.bucketize.Tensor",
        "count": 1,
        "ordinals": [ordinal],
    }


@pytest.mark.parametrize("boundaries", ["0.0, 0.25, 0.25, 1.0", "-0.0, 0.0, 0.0, 1.0"])
def test_typed_literal_proves_nondecreasing_even_with_duplicates_and_signed_zero(tmp_path, boundaries):
    witness, ordinal = _capture(tmp_path, boundaries=boundaries)
    assert witness["expected_ordinals"] == [ordinal]
    assert witness["boundary_facts"][0]["status"] == "source_dense_non_decreasing_non_nan_proved"
    assert witness["original_to_prepared_equivalence"] == "not_proved"
    assert witness["source_proof"]["trace_bindings"][0][1]["unresolved_ancestry_ids"] == ["g:staged:root:n1"]


@pytest.mark.parametrize("boundaries", ["0.0, 0.75, 0.5, 1.0", "0x000000000000803E0000C07F0000803F"])
def test_unsorted_or_nan_literal_cannot_discharge_boundary_premise(tmp_path, boundaries):
    witness, ordinal = _capture(tmp_path, boundaries=boundaries)
    assert witness["boundary_facts"][0]["status"] == "not_proved"
    row = _row(ordinal)
    with pytest.raises(ValueError, match="ordering/non-NaN premise"):
        support.record(witness, row, _reviewed(witness, ordinal), {ordinal: row})


def test_dynamic_boundary_is_not_inferred_from_trace_or_example_inputs(tmp_path):
    witness, ordinal = _capture(tmp_path, dynamic=True)
    assert witness["boundary_facts"] == [
        {
            "ordinal": ordinal,
            "status": "not_proved",
            "required_precondition": "nondecreasing_non_nan_f32_boundaries",
            "literal_sha256": None,
        }
    ]


def test_boundary_count_must_fit_selected_index_width(tmp_path):
    with pytest.raises(ValueError, match="selected index width"):
        _capture(tmp_path, index_bits=3)


def test_parallel_input_extent_must_fit_selected_index_width(tmp_path):
    with pytest.raises(ValueError, match="selected index width"):
        _capture(tmp_path, index_bits=4, input_extent=8)


def test_empty_roster_accepts_diagnostic_selection_but_cannot_link_without_observation(tmp_path):
    witness, _ = _capture(tmp_path, frontend_op="aten.alias.default", index_bits=None)
    assert witness["expected_ordinals"] == []
    assert witness["selected_index_observation"] is None
    source = {
        support.FIELD: witness,
        "selected_index_observation": None,
        "capture_receipt_sha256": witness["capture_receipt_sha256"],
    }
    build = {"candidate_tree_sha256": "b" * 64, "capture_tree_sha256": "c" * 64, "elf_sha256": "d" * 64}
    with pytest.raises(ValueError, match="actual index lowering"):
        support.link(source, {}, build)


def test_boundary_bits_and_source_identity_change_with_literal(tmp_path):
    first, _ = _capture(tmp_path / "first", boundaries="0.0, 0.25, 0.25, 1.0")
    second, _ = _capture(tmp_path / "second", boundaries="0.0, 0.25, 0.5, 1.0")
    assert first["raw_source_sha256"] != second["raw_source_sha256"]
    assert first["boundary_facts"][0]["literal_sha256"] != second["boundary_facts"][0]["literal_sha256"]


def test_reviewed_host_and_exact_linked_identity_required(tmp_path, monkeypatch):
    witness, ordinal = _capture(tmp_path)
    row = _row(ordinal)
    source_rows = {ordinal: row}
    with pytest.raises(ValueError, match="reviewed host admission"):
        support.record(witness, row, {"status": "unsupported"}, source_rows)
    without_body = _reviewed(witness, ordinal)
    without_body.pop("source_body_proof")
    with pytest.raises(ValueError, match="closed source-body"):
        support.record(witness, row, without_body, source_rows)
    wrong_right = _reviewed(witness, ordinal)
    wrong_right["source_body_proof"]["patterns"][0]["right"] = True
    wrong_right["profiles"][0]["source_body_proof"]["patterns"][0]["right"] = True
    wrong_right["profiles"][0]["decisions"][0]["source_body_proof"]["patterns"][0]["right"] = True
    with pytest.raises(ValueError, match="differs from the traced source"):
        support.record(witness, row, wrong_right, source_rows)
    wrong_declaration = _reviewed(witness, ordinal)
    wrong_declaration["profiles"][0]["decisions"][0]["declaration"] = "unrelated"
    with pytest.raises(ValueError, match="exact selected reviewed declaration decision"):
        support.record(witness, row, wrong_declaration, source_rows)
    wrong_inner_proof = _reviewed(witness, ordinal)
    wrong_inner_proof["profiles"][0]["decisions"][0]["source_body_proof"]["patterns"][0]["right"] = True
    with pytest.raises(ValueError, match="exact selected reviewed declaration decision"):
        support.record(witness, row, wrong_inner_proof, source_rows)
    support.record(witness, row, _reviewed(witness, ordinal), source_rows)
    with pytest.raises(ValueError, match="admitted twice"):
        support.record(witness, row, _reviewed(witness, ordinal), source_rows)
    source = {
        support.FIELD: witness,
        "source_sha256": witness["raw_source_sha256"],
        "normalized_source_sha256": witness["normalized_source_sha256"],
        "capture_receipt_sha256": witness["capture_receipt_sha256"],
        "n_source_operations": witness["n_source_operations"],
        "selected_index_observation": witness["selected_index_observation"],
    }
    monkeypatch.setattr(support, "_record_matches_selected", lambda actual, selected: actual == selected)
    build = {
        "candidate_tree_sha256": "b" * 64,
        "capture_tree_sha256": strict_tree_sha256(tmp_path / "capture")["sha256"],
        "elf_sha256": "d" * 64,
    }
    with pytest.raises(ValueError, match="index lowering"):
        support.link(source, {"index_bits": 32}, build)
    support.link(source, witness["selected_index_observation"], build, capture_path=tmp_path / "capture")
    entry = {
        "source_sha256": source["source_sha256"],
        "capture_tree_sha256": build["capture_tree_sha256"],
        "elf_sha256": build["elf_sha256"],
        "index_lowering": witness["selected_index_observation"],
    }
    assert support.linked_source_complete(source, entry, build["candidate_tree_sha256"])
    for mutate in (
        lambda value: value[support.FIELD].update(admissions=[]),
        lambda value: value[support.FIELD]["source_proof"].update(trace_sha256="0" * 64),
        lambda value: value[support.FIELD]["boundary_facts"][0].update(status="not_proved"),
        lambda value: value[support.FIELD].update(original_to_prepared_equivalence="proved"),
        lambda value: value[support.FIELD]["linked_build"].update(elf_sha256="0" * 64),
        lambda value: value[support.FIELD].update(actual_index_observation={"index_bits": 32}),
        lambda value: value[support.FIELD].update(source_capture_path="/missing/capture"),
        lambda value: value[support.FIELD].update(source_capture_path=1),
    ):
        altered = deepcopy(source)
        mutate(altered)
        assert not support.linked_source_complete(altered, entry, build["candidate_tree_sha256"])
    trace = tmp_path / "capture" / "frontend-trace.json"
    original = trace.read_bytes()
    trace.write_bytes(original + b"\n")
    assert not support.linked_source_complete(source, entry, build["candidate_tree_sha256"])
    trace.write_bytes(original)
    assert support.linked_source_complete(source, entry, build["candidate_tree_sha256"])


def test_link_reproves_actual_capture_bytes_not_just_the_source_record(tmp_path, monkeypatch):
    witness, ordinal = _capture(tmp_path)
    row = _row(ordinal)
    support.record(witness, row, _reviewed(witness, ordinal), {ordinal: row})
    source = {
        support.FIELD: witness,
        "source_sha256": witness["raw_source_sha256"],
        "normalized_source_sha256": witness["normalized_source_sha256"],
        "capture_receipt_sha256": witness["capture_receipt_sha256"],
        "n_source_operations": witness["n_source_operations"],
        "selected_index_observation": witness["selected_index_observation"],
    }
    monkeypatch.setattr(support, "_record_matches_selected", lambda actual, selected: actual == selected)
    capture = tmp_path / "capture"
    build = {
        "candidate_tree_sha256": "b" * 64,
        "capture_tree_sha256": strict_tree_sha256(capture)["sha256"],
        "elf_sha256": "d" * 64,
    }
    trace = capture / "frontend-trace.json"
    trace.write_bytes(trace.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="linked capture source or trace changed"):
        support.link(source, witness["selected_index_observation"], build, capture_path=capture)


def test_actual_host_source_decision_joins_independent_trace_and_literal(tmp_path):
    witness, ordinal = _capture(tmp_path)
    row = _row(ordinal)
    parsed = tuple(mq.walk(mq.parse((tmp_path / "capture" / "model.mlir").read_text())))
    selected = {
        "neutral": {
            "package_sha256": "b" * 64,
            "capability_spec_sha256": "a" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "b" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral_closed_bucketize",
                        "ops": ["aten.bucketize.Tensor"],
                        "placement": "host",
                        "signature": {
                            "family": "reduction",
                            "ordered_operand_dtypes": ["f32", "f32", "i64"],
                            "ordered_result_dtypes": ["i64"],
                        },
                        "source_body": {
                            "schema": STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA,
                            "operation": "f32_literal_boundary_count_i64",
                        },
                        "numerical_contract": {"status": "unreviewed"},
                    }
                ],
                "evidence": {"scope": "neutral source/build-only placement"},
            },
        }
    }
    decision = admit_host_operation(
        selected,
        row,
        {
            "family": "reduction",
            "ordered_operand_dtypes": ["f32", "f32", "i64"],
            "ordered_result_dtypes": ["i64"],
            "rank": 1,
        },
        source_operations=(parsed[ordinal],),
        source_context={"selected_index_observation": witness["selected_index_observation"]},
    )
    assert decision["status"] == "admitted"
    support.record(witness, row, decision, {ordinal: row})
    assert witness["admissions"][0]["source_body_schema"] == STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA


def test_bucketize_schema_routes_away_from_unrelated_linalg_witness(monkeypatch):
    def wrong_witness(*_args, **_kwargs):
        raise AssertionError("bucketize source was sent through pointwise witness")

    monkeypatch.setattr(dispatch.linalg_support, "record", wrong_witness)
    dispatch.record(
        {},
        {},
        {},
        {},
        {"source_body_proof": {"schema": STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA}},
        (),
        {},
        None,
    )
