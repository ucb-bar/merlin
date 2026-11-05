"""Neutral source/trace checks for mask count and literal-bounded tensor indexing."""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_index_source as index_source
from xdsl.utils.exceptions import ParseError, VerifyException

from merlin.common import mlir_query as mq
from merlin.common.jsonio import canonical_json


def _range_ir(name: str, *, extent: int, start: int, step: int) -> str:
    ident = f"g:prepared:root:{name}"
    origin = f"g:original:root:{name}"
    prov = f'{{prov.source_node_ids = ["{ident}"], prov.origin_node_ids = ["{origin}"]}}'
    return f"""
    %{name}_empty = "tensor.empty"() {{prov.source_node_ids = ["{ident}"],
      prov.origin_node_ids = ["{origin}"], prov.aten = "aten.arange.start_step"}} : () -> tensor<{extent}xi64>
    %{name}_range = "linalg.generic"(%{name}_empty) <{{indexing_maps = [affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 0, 1>}}> ({{
      ^bb_{name}(%{name}_old: i64):
        %{name}_index = "linalg.index"() <{{dim = 0 : i64}}> {prov} : () -> index
        %{name}_cast = "arith.index_cast"(%{name}_index) {prov} : (index) -> i64
        %{name}_step = "arith.constant"() <{{value = {step} : i64}}> {prov} : () -> i64
        %{name}_scaled = "arith.muli"(%{name}_cast, %{name}_step)
          <{{overflowFlags = #arith.overflow<none>}}> {prov} : (i64, i64) -> i64
        %{name}_start = "arith.constant"() <{{value = {start} : i64}}> {prov} : () -> i64
        %{name}_value = "arith.addi"(%{name}_start, %{name}_scaled)
          <{{overflowFlags = #arith.overflow<none>}}> {prov} : (i64, i64) -> i64
        "linalg.yield"(%{name}_value) {prov} : (i64) -> ()
    }}) {{prov.source_node_ids = ["{ident}"], prov.origin_node_ids = ["{origin}"],
      prov.aten = "aten.arange.start_step"}} : (tensor<{extent}xi64>) -> tensor<{extent}xi64>
"""


def _source(*, offset: int = 0, second_step: int = 1, second_start: int = 0, projected: bool = False) -> str:
    a = _range_ir("a", extent=2, start=0, step=1)
    b = _range_ir("b", extent=3, start=second_start, step=second_step)
    source = f""""builtin.module"() ({{
  "func.func"() <{{sym_name = "forward",
    function_type = (tensor<2x3xi1>) -> tensor<2x3xi1>}}> ({{
  ^bb0(%data: tensor<2x3xi1>):
{a}
{b}
    %offset = "arith.constant"() <{{value = {offset} : i64}}>
      {{prov.source_node_ids = ["g:prepared:root:add"]}} : () -> i64
    %splat = "tensor.splat"(%offset)
      {{prov.source_node_ids = ["g:prepared:root:add"]}} : (i64) -> tensor<3xi64>
    %sum_init = "tensor.empty"()
      {{prov.source_node_ids = ["g:prepared:root:add"]}} : () -> tensor<3xi64>
    %sum = "linalg.generic"(%b_range, %splat, %sum_init) <{{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}}> ({{
      ^bb_add(%base: i64, %delta: i64, %old_sum: i64):
        %shifted = "arith.addi"(%base, %delta) <{{overflowFlags = #arith.overflow<none>}}>
          {{prov.source_node_ids = ["g:prepared:root:add"]}} : (i64, i64) -> i64
        "linalg.yield"(%shifted) {{prov.source_node_ids = ["g:prepared:root:add"]}} : (i64) -> ()
    }}) {{prov.source_node_ids = ["g:prepared:root:add"], prov.aten = "aten.add.Tensor"}}
      : (tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    %out_init = "tensor.empty"()
      {{prov.source_node_ids = ["g:prepared:root:index"]}} : () -> tensor<2x3xi1>
    %out = "linalg.generic"(%a_range, %sum, %out_init) <{{
      indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 2, 1>}}> ({{
      ^bb_index(%row: i64, %column: i64, %old: i1):
        %i = "arith.index_cast"(%row) {{prov.source_node_ids = ["g:prepared:root:index"]}}
          : (i64) -> index
        %j = "arith.index_cast"(%column) {{prov.source_node_ids = ["g:prepared:root:index"]}}
          : (i64) -> index
        %value = "tensor.extract"(%data, %i, %j)
          {{prov.source_node_ids = ["g:prepared:root:index"]}}
          : (tensor<2x3xi1>, index, index) -> i1
        "linalg.yield"(%value) {{prov.source_node_ids = ["g:prepared:root:index"]}} : (i1) -> ()
    }}) {{prov.source_node_ids = ["g:prepared:root:index"], prov.aten = "aten.index.Tensor"}}
      : (tensor<2xi64>, tensor<3xi64>, tensor<2x3xi1>) -> tensor<2x3xi1>
    "func.return"(%out) : (tensor<2x3xi1>) -> ()
  }}) : () -> ()
}}) : () -> ()
"""
    if projected:
        source = (
            source.replace(
                '    %out_init = "tensor.empty"()',
                """    %a_expanded = "tensor.expand_shape"(%a_range) <{
      reassociation = [[0 : i64, 1 : i64]], static_output_shape = array<i64: 2, 1>}>
      {prov.source_node_ids = ["g:prepared:root:reshape"], prov.aten = "aten.unsqueeze.default"}
      : (tensor<2xi64>) -> tensor<2x1xi64>
    %out_init = "tensor.empty"()""",
            )
            .replace(
                '"linalg.generic"(%a_range, %sum, %out_init)',
                '"linalg.generic"(%a_expanded, %sum, %out_init)',
            )
            .replace(
                "affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>",
                "affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d1)>",
            )
            .replace(
                "(tensor<2xi64>, tensor<3xi64>, tensor<2x3xi1>) -> tensor<2x3xi1>",
                "(tensor<2x1xi64>, tensor<3xi64>, tensor<2x3xi1>) -> tensor<2x3xi1>",
            )
        )
    return source


def _sha(data: bytes) -> str:
    return sha256(data).hexdigest()


def _capture(tmp_path, *, source: str | None = None, second_start: int = 0, second_step: int = 1, trace_change=None):
    capture = tmp_path / "capture"
    capture.mkdir()
    source = source or _source(second_start=second_start, second_step=second_step)
    module = mq.parse(source)
    module.verify()
    parsed = tuple(mq.walk(module))
    (capture / "model.mlir").write_text(source)
    grouped: dict[str, list[int]] = {}
    entries = []
    for ordinal, op in enumerate(parsed):
        ids = index_source._ids(op)
        assert len(ids) <= 1
        entries.append({"ordinal": ordinal, "operation": mq.op_name(op), "source_node_ids": list(ids)})
        for ident in ids:
            grouped.setdefault(ident, []).append(ordinal)

    def range_node(name: str, extent: int):
        start, step = (second_start, second_step) if name == "b" else (0, 1)
        return {
            "id": f"g:prepared:root:{name}",
            "op": "call_function",
            "target": "aten.arange.start_step",
            "args": [start, start + extent * step, step],
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
            "origin_node_ids": [f"g:original:root:{name}"],
        }

    prepared = [
        range_node("a", 2),
        range_node("b", 3),
        {"id": "g:prepared:root:add", "op": "call_function", "target": "aten.add.Tensor"},
        {"id": "g:prepared:root:reshape", "op": "call_function", "target": "aten.unsqueeze.default"},
        {"id": "g:prepared:root:index", "op": "call_function", "target": "aten.index.Tensor"},
    ]
    original = [
        {"id": f"g:original:root:{name}", "op": "call_function", "target": "aten.arange.start_step"}
        for name in ("a", "b")
    ]
    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "graphs": {
            "prepared": {"status": "complete", "nodes": prepared},
            "original": {"status": "complete", "nodes": original},
        },
        "mlir": {
            "sha256": _sha(source.encode()),
            "bytes": len(source.encode()),
            "operations": entries,
            "source_correspondence": [
                {"node_id": ident, "status": "lowered", "mlir_ordinals": ordinals}
                for ident, ordinals in grouped.items()
            ],
        },
    }
    if trace_change is not None:
        trace_change(trace)
    (capture / "frontend-trace.json").write_text(json.dumps(trace))
    (capture / "weights.safetensors").write_bytes(b"neutral empty weights")
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


def test_neutral_literal_indexed_extract_has_exact_trace_and_static_bounds(tmp_path):
    capture = _capture(tmp_path)
    proof = index_source.prove_index_source(capture, index_bits=64)
    assert proof["status"] == index_source.PENDING
    assert [record["kind"] for record in proof["records"]] == ["literal_indexed_extract"]
    assert len(proof["records"][0]["index_sources"]) == 2
    assert proof["original_to_prepared_equivalence"] == "not_proved"
    index_source.verify_index_source_record(capture, proof)


def test_reviewed_literal_index_role_requires_source_trace_and_linked_build(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import private_index_host_support as host_support

    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen.application_inventory import operation_structure
    from merlin.targetgen.host_capabilities import admit_host_operation

    capture = _capture(tmp_path)
    source = capture / "model.mlir"
    raw = source.read_text()
    normalized, receipt = normalize_capture_mlir(raw)
    parsed = tuple(mq.walk(mq.parse(normalized)))
    selected = {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "neutral selected compiler premise",
    }
    witness = host_support.begin(
        capture,
        parsed,
        _sha(raw.encode()),
        receipt["output_sha256"],
        _sha((capture / "capture_receipt.json").read_bytes()),
        selected,
    )
    proof = witness["source_proof"]
    ordinal = proof["records"][0]["compute_ordinals"][0]
    structure = operation_structure(parsed[ordinal])
    inputs, outputs = structure["ordered_operand_types"], structure["ordered_result_types"]
    row = {
        "mlir_operation": "linalg.generic",
        "frontend_op": "aten.index.Tensor",
        "ordinals": [ordinal],
        "count": 1,
        **structure,
    }
    signature = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": [item["dtype"] for item in inputs],
        "ordered_result_dtypes": [item["dtype"] for item in outputs],
        "rank": len(outputs[0]["shape"]),
    }
    declaration = {
        "id": "neutral_index_role",
        "ops": ["aten.index.Tensor"],
        "placement": "host",
        "signature": {key: value for key, value in signature.items() if key != "rank"} | {"ranks": [signature["rank"]]},
        "source_body": {"schema": "merlin.prepared_index_source_body.v1", "operation": "literal_indexed_extract"},
    }
    selected_host = {
        "neutral": {
            "package_sha256": "b" * 64,
            "capability_spec_sha256": "c" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "b" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [declaration],
                "evidence": {"scope": "neutral source-only test"},
            },
        }
    }
    context = {"selected_index_observation": selected, "verified_index_source": proof}
    assert (
        admit_host_operation(selected_host, row, signature, source_operations=(parsed[ordinal],))["status"] == "unknown"
    )
    decision = admit_host_operation(
        selected_host,
        row,
        signature,
        source_operations=(parsed[ordinal],),
        source_context=context,
    )
    assert decision["status"] == "admitted", decision
    wrong_role = deepcopy(selected_host)
    wrong_role["neutral"]["capability_spec"]["operations"][0]["source_body"]["operation"] = "boolean_extension"
    with pytest.raises(ValueError, match="prepared-index source_body"):
        admit_host_operation(
            wrong_role,
            row,
            signature,
            source_operations=(parsed[ordinal],),
            source_context=context,
        )
    malformed_role = deepcopy(wrong_role)
    malformed_role["neutral"]["capability_spec"]["operations"][0]["source_body"]["operation"] = []
    with pytest.raises(ValueError, match="closed role"):
        admit_host_operation(
            malformed_role,
            row,
            signature,
            source_operations=(parsed[ordinal],),
            source_context=context,
        )
    wrong_signature = deepcopy(signature)
    wrong_signature["ordered_result_dtypes"] = ["i64"]
    assert (
        admit_host_operation(
            selected_host,
            row,
            wrong_signature,
            source_operations=(parsed[ordinal],),
            source_context=context,
        )["status"]
        != "admitted"
    )
    wrong_ordinal = deepcopy(row)
    wrong_ordinal["ordinals"] = [ordinal + 1]
    assert (
        admit_host_operation(
            selected_host,
            wrong_ordinal,
            signature,
            source_operations=(parsed[ordinal],),
            source_context=context,
        )["status"]
        != "admitted"
    )
    host_support.record(witness, row, decision, parsed, {ordinal: row})
    host_support.verify_source(witness, capture)
    source_record = {
        "source_sha256": witness["raw_source_sha256"],
        "normalized_source_sha256": witness["normalized_source_sha256"],
        "capture_receipt_sha256": witness["capture_receipt_sha256"],
        "n_source_operations": len(parsed),
        "selected_index_observation": selected,
        host_support.FIELD: witness,
    }
    actual = {
        **selected,
        "effective_pipeline": ",".join(
            f"{name}{{index-bitwidth=64}}"
            for name in (
                "convert-index-to-llvm",
                "convert-arith-to-llvm",
                "finalize-memref-to-llvm",
                "convert-func-to-llvm",
                "convert-cf-to-llvm",
            )
        ),
    }
    linked = {
        "candidate_tree_sha256": "d" * 64,
        "capture_tree_sha256": strict_tree_sha256(capture)["sha256"],
        "elf_sha256": "f" * 64,
    }
    pending = deepcopy(source_record)
    with pytest.raises(ValueError, match="selected capture"):
        host_support.link(source_record, actual, linked)
    for name in ("model.mlir", "frontend-trace.json", "capture_receipt.json"):
        path = capture / name
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with pytest.raises(ValueError, match="selected capture"):
                host_support.link(deepcopy(pending), actual, linked, capture_path=capture)
        finally:
            path.write_bytes(original)

    original_verify = index_source.verify_index_source_record
    trace_path = capture / "frontend-trace.json"
    original_trace = trace_path.read_bytes()

    def changed_during_reproof(path, proof):
        original_verify(path, proof)
        trace_path.write_bytes(original_trace + b"\n")

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(index_source, "verify_index_source_record", changed_during_reproof)
            with pytest.raises(ValueError, match="selected capture"):
                host_support.link(deepcopy(pending), actual, linked, capture_path=capture)
    finally:
        trace_path.write_bytes(original_trace)
    host_support.link(source_record, actual, linked, capture_path=capture)
    entry = {
        "source_sha256": source_record["source_sha256"],
        "capture_tree_sha256": linked["capture_tree_sha256"],
        "elf_sha256": "f" * 64,
        "index_lowering": actual,
    }
    assert host_support.linked_complete(source_record, entry, "d" * 64)
    for name in ("model.mlir", "frontend-trace.json", "capture_receipt.json"):
        path = capture / name
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            assert not host_support.linked_complete(source_record, entry, "d" * 64)
        finally:
            path.write_bytes(original)
    assert host_support.linked_complete(source_record, entry, "d" * 64)
    changed = deepcopy(source_record)
    changed[host_support.FIELD]["admissions"] = []
    assert not host_support.linked_complete(changed, entry, "d" * 64)
    alternate_parent = tmp_path / "alternate"
    alternate_parent.mkdir()
    alternate = _capture(alternate_parent, second_start=2, second_step=-1)
    alternate_proof = index_source.prove_index_source(alternate, index_bits=64)
    substituted = deepcopy(source_record)
    substituted[host_support.FIELD]["source_proof"] = alternate_proof
    substituted[host_support.FIELD]["source_proof_sha256"] = _sha(canonical_json(alternate_proof))
    assert not host_support.linked_complete(substituted, entry, "d" * 64)
    pending_substitution = deepcopy(pending)
    pending_substitution[host_support.FIELD]["source_proof"] = alternate_proof
    pending_substitution[host_support.FIELD]["source_proof_sha256"] = _sha(canonical_json(alternate_proof))
    with pytest.raises(ValueError, match="selected capture"):
        host_support.link(pending_substitution, actual, linked, capture_path=capture)
    for edit in (
        lambda ctx: ctx["verified_index_source"]["records"][0]["output_shape"].__setitem__(0, 99),
        lambda ctx: ctx["verified_index_source"]["records"][0]["output_shape"].__setitem__(0, 2.0),
        lambda ctx: ctx["verified_index_source"]["records"][0].__setitem__("source_node_id", "other"),
        lambda ctx: ctx["verified_index_source"]["records"][0].__setitem__("kind", []),
        lambda ctx: ctx["verified_index_source"]["records"][0]["index_sources"][0].__setitem__("axis", False),
        lambda ctx: ctx["verified_index_source"]["records"][0].__setitem__("extra", 0),
        lambda ctx: ctx["selected_index_observation"].__setitem__("index_bits", 64.0),
        lambda ctx: ctx["selected_index_observation"].__setitem__("cross_flags", []),
    ):
        changed = deepcopy(context)
        edit(changed)
        assert (
            admit_host_operation(
                selected_host,
                row,
                signature,
                source_operations=(parsed[ordinal],),
                source_context=changed,
            )["status"]
            != "admitted"
        )
    forged = deepcopy(context)
    forged["verified_index_source"]["frontend_trace_sha256"] = "0" * 64
    # Core sees a caller premise; only the private reread establishes that a
    # valid-looking digest names these captured trace bytes.
    with pytest.raises(ValueError, match="serialized"):
        index_source.verify_index_source_record(capture, forged["verified_index_source"])
    bad_decision = deepcopy(decision)
    bad_decision["source_body_proof"]["roles"][0]["ordinal"] = float(ordinal)
    fresh = host_support.begin(
        capture,
        parsed,
        _sha(raw.encode()),
        receipt["output_sha256"],
        _sha((capture / "capture_receipt.json").read_bytes()),
        selected,
    )
    with pytest.raises(ValueError, match="source-body"):
        host_support.record(fresh, row, bad_decision, parsed, {ordinal: row})


def test_descending_literal_index_is_bounded_by_the_same_closed_source(tmp_path):
    proof = index_source.prove_index_source(_capture(tmp_path, second_start=2, second_step=-1), index_bits=64)
    assert proof["records"][0]["kind"] == "literal_indexed_extract"


def test_singleton_projected_index_and_malformed_reshape(tmp_path):
    proof = index_source.prove_index_source(_capture(tmp_path, source=_source(projected=True)), index_bits=64)
    assert proof["records"][0]["index_input_shapes"][0] == [2, 1]

    changed = _source(projected=True).replace("array<i64: 2, 1>", "array<i64: 2, 2>")
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises((ValueError, ParseError, VerifyException)):
        index_source.prove_index_source(_capture(other, source=changed), index_bits=64)


def test_offset_bounds_are_independent_of_a_matching_trace(tmp_path):
    capture = _capture(tmp_path, source=_source(offset=2))
    with pytest.raises(ValueError, match="out of bounds"):
        index_source.prove_index_source(capture, index_bits=64)


@pytest.mark.parametrize("offset", [0, 1])
def test_literal_rank_zero_offset_preserves_index_bounds(tmp_path, offset):
    source = (_source(offset=offset)
        .replace(": (i64) -> tensor<3xi64>", ": (i64) -> tensor<i64>")
        .replace("affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>",
                 "affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>")
        .replace("(tensor<3xi64>, tensor<3xi64>, tensor<3xi64>)",
                 "(tensor<3xi64>, tensor<i64>, tensor<3xi64>)"))
    capture = _capture(tmp_path, source=source)
    if offset:
        with pytest.raises(ValueError, match="out of bounds"):
            index_source.prove_index_source(capture, index_bits=64)
    else:
        proof = index_source.prove_index_source(capture, index_bits=64)
        assert proof["records"][0]["kind"] == "literal_indexed_extract"
        assert proof["original_to_prepared_equivalence"] == "not_proved"


def test_boolean_mask_count_is_a_separate_closed_reduction():
    source = """"builtin.module"() ({
  "func.func"() <{sym_name = "count", function_type = (tensor<7xi1>) -> i64}> ({
  ^bb0(%mask: tensor<7xi1>):
    %init = "tensor.empty"() : () -> tensor<7xi64>
    %wide = "linalg.generic"(%mask, %init) <{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb1(%bit: i1, %old: i64):
        %cell = "arith.extui"(%bit) : (i1) -> i64
        "linalg.yield"(%cell) : (i64) -> ()
    }) : (tensor<7xi1>, tensor<7xi64>) -> tensor<7xi64>
    %zero = "arith.constant"() <{value = 0 : i64}> : () -> i64
    %seed = "tensor.splat"(%zero) : (i64) -> tensor<i64>
    %sum = "linalg.reduce"(%wide, %seed) <{dimensions = array<i64: 0>}> ({
      ^bb2(%cell: i64, %acc: i64):
        %next = "arith.addi"(%cell, %acc) <{overflowFlags = #arith.overflow<none>}>
          : (i64, i64) -> i64
        "linalg.yield"(%next) : (i64) -> ()
    }) : (tensor<7xi64>, tensor<i64>) -> tensor<i64>
    %count = "tensor.extract"(%sum) : (tensor<i64>) -> i64
    "func.return"(%count) : (i64) -> ()
  }) : () -> ()
}) : () -> ()"""
    module = mq.parse(source)
    module.verify()
    parsed = tuple(mq.walk(module))
    ordinals = {id(op): i for i, op in enumerate(parsed)}
    reduce = next(op for op in parsed if mq.op_name(op) == "linalg.reduce")
    owned = {
        ordinals[id(reduce)],
        ordinals[id(reduce.operands[0].owner)],
        next(i for i, op in enumerate(parsed) if mq.op_name(op) == "tensor.extract"),
    }
    proof = index_source._mask_count(reduce, owned, ordinals, 64)
    assert proof["kind"] == "boolean_mask_count"
    assert proof["count_interval"] == [0, 7]
    with pytest.raises(ValueError):
        index_source._mask_count(reduce, owned - {proof["extension_ordinal"]}, ordinals, 64)


def _mask_capture(tmp_path):
    """A small independent prepared mask-count capture, not a validation shape."""
    capture = tmp_path / "capture"
    capture.mkdir()
    ident = "g:prepared:root:index"
    tag = f'{{prov.source_node_ids = ["{ident}"]}}'
    root = f'{{prov.source_node_ids = ["{ident}"], prov.aten = "aten.index.Tensor"}}'
    source = f""""builtin.module"() ({{
  "func.func"() <{{sym_name = "count", function_type = (tensor<7xi1>) -> i64}}> ({{
  ^bb0(%mask: tensor<7xi1>):
    %init = "tensor.empty"() : () -> tensor<7xi64>
    %wide = "linalg.generic"(%mask, %init) <{{
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}}> ({{
      ^bb1(%bit: i1, %old: i64):
        %cell = "arith.extui"(%bit) {tag} : (i1) -> i64
        "linalg.yield"(%cell) {tag} : (i64) -> ()
    }}) {root} : (tensor<7xi1>, tensor<7xi64>) -> tensor<7xi64>
    %zero = "arith.constant"() <{{value = 0 : i64}}> : () -> i64
    %seed = "tensor.splat"(%zero) : (i64) -> tensor<i64>
    %sum = "linalg.reduce"(%wide, %seed) <{{dimensions = array<i64: 0>}}> ({{
      ^bb2(%cell: i64, %acc: i64):
        %next = "arith.addi"(%cell, %acc) <{{overflowFlags = #arith.overflow<none>}}> {tag}
          : (i64, i64) -> i64
        "linalg.yield"(%next) {tag} : (i64) -> ()
    }}) {root} : (tensor<7xi64>, tensor<i64>) -> tensor<i64>
    %count = "tensor.extract"(%sum) {tag} : (tensor<i64>) -> i64
    "func.return"(%count) : (i64) -> ()
  }}) : () -> ()
}}) : () -> ()"""
    module = mq.parse(source)
    module.verify()
    parsed = tuple(mq.walk(module))
    (capture / "model.mlir").write_text(source)
    owned = [i for i, op in enumerate(parsed) if index_source._ids(op) == (ident,)]
    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "graphs": {
            "prepared": {"status": "complete", "nodes": [{"id": ident, "target": "aten.index.Tensor"}]},
            "original": {"status": "complete", "nodes": []},
        },
        "mlir": {
            "sha256": _sha(source.encode()),
            "bytes": len(source.encode()),
            "operations": [
                {"ordinal": i, "operation": mq.op_name(op), "source_node_ids": list(index_source._ids(op))}
                for i, op in enumerate(parsed)
            ],
            "source_correspondence": [{"node_id": ident, "status": "lowered", "mlir_ordinals": owned}],
        },
    }
    (capture / "frontend-trace.json").write_text(json.dumps(trace))
    (capture / "weights.safetensors").write_bytes(b"neutral empty weights")
    (capture / "weights.safetensors.manifest.json").write_text("{}")
    artifacts = {
        name: {"sha256": _sha((capture / name).read_bytes()), "bytes": (capture / name).stat().st_size}
        for name in ("model.mlir", "frontend-trace.json", "weights.safetensors", "weights.safetensors.manifest.json")
    }
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


def test_mask_extension_and_sum_have_distinct_closed_roles(tmp_path):
    from merlin.frontends.prepared_index_source_body import screen_source_body
    from merlin.targetgen.application_inventory import operation_structure

    capture = _mask_capture(tmp_path)
    proof = index_source.prove_index_source(capture, index_bits=64)
    assert [record["kind"] for record in proof["records"]] == ["boolean_mask_count"]
    selected = {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "neutral selected compiler premise",
    }
    parsed = tuple(mq.walk(mq.parse((capture / "model.mlir").read_text())))
    record = proof["records"][0]
    for role, ordinal in (
        ("boolean_extension", record["extension_ordinal"]),
        ("mask_sum", record["reduction_ordinal"]),
    ):
        op = parsed[ordinal]
        structure = operation_structure(op)
        operands, results = structure["ordered_operand_types"], structure["ordered_result_types"]
        row = {
            "frontend_op": "aten.index.Tensor",
            "mlir_operation": mq.op_name(op),
            "ordinals": [ordinal],
            "count": 1,
            **structure,
        }
        signature = {
            "ordered_operand_dtypes": [item["dtype"] for item in operands],
            "ordered_result_dtypes": [item["dtype"] for item in results],
            "rank": len(results[0]["shape"]),
        }
        declaration = {"id": role, "source_body": {"operation": role}}
        decision = screen_source_body(
            declaration,
            row,
            signature,
            (op,),
            {"selected_index_observation": selected, "verified_index_source": proof},
        )
        assert decision["status"] == "admitted", decision
        wrong = deepcopy(proof)
        wrong["records"][0]["compute_ordinals"] = [True, ordinal]
        assert (
            screen_source_body(
                declaration,
                row,
                signature,
                (op,),
                {"selected_index_observation": selected, "verified_index_source": wrong},
            )["status"]
            != "admitted"
        )


def test_mask_both_reviewed_roles_are_mandatory_for_linked_private_witness(tmp_path):
    from merlin_experiments.phase1.feedback import private_index_host_support as host_support

    from merlin.frontends.capture_normalization import normalize_capture_mlir
    from merlin.targetgen.application_inventory import operation_structure
    from merlin.targetgen.host_capabilities import admit_host_operation

    capture = _mask_capture(tmp_path)
    raw = (capture / "model.mlir").read_text()
    normalized, normalization = normalize_capture_mlir(raw)
    parsed = tuple(mq.walk(mq.parse(normalized)))
    selected = {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "a" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": "e-p:64:64",
        "index_bits": 64,
        "scope": "neutral selected compiler premise",
    }
    witness = host_support.begin(
        capture,
        parsed,
        _sha(raw.encode()),
        normalization["output_sha256"],
        _sha((capture / "capture_receipt.json").read_bytes()),
        selected,
    )
    proof = witness["source_proof"]
    owned = proof["records"][0]
    rows = []
    for role, ordinal in (("boolean_extension", owned["extension_ordinal"]), ("mask_sum", owned["reduction_ordinal"])):
        op = parsed[ordinal]
        structure = operation_structure(op)
        inputs, outputs = structure["ordered_operand_types"], structure["ordered_result_types"]
        row = {
            "frontend_op": "aten.index.Tensor",
            "mlir_operation": mq.op_name(op),
            "ordinals": [ordinal],
            "count": 1,
            **structure,
        }
        signature = {
            "family": "reduction" if role == "mask_sum" else "elementwise_map",
            "ordered_operand_dtypes": [item["dtype"] for item in inputs],
            "ordered_result_dtypes": [item["dtype"] for item in outputs],
            "rank": len(outputs[0]["shape"]),
        }
        declaration = {
            "id": f"neutral_{role}",
            "ops": ["aten.index.Tensor"],
            "placement": "host",
            "signature": {key: value for key, value in signature.items() if key != "rank"},
            "source_body": {"schema": "merlin.prepared_index_source_body.v1", "operation": role},
        }
        selected_host = {
            "neutral": {
                "package_sha256": "b" * 64,
                "capability_spec_sha256": "c" * 64,
                "dtype_strategy": "int8_w8a8",
                "capability_spec": {
                    "schema": "merlin.host_capabilities.v1",
                    "status": "reviewed",
                    "compiler": {"package_sha256": "b" * 64, "dtype_strategy": "int8_w8a8"},
                    "operations": [declaration],
                    "evidence": {"scope": "neutral source-only test"},
                },
            }
        }
        decision = admit_host_operation(
            selected_host,
            row,
            signature,
            source_operations=(op,),
            source_context={"selected_index_observation": selected, "verified_index_source": proof},
        )
        assert decision["status"] == "admitted", decision
        rows.append((row, decision))
    source_rows = {row["ordinals"][0]: row for row, _ in rows}
    host_support.record(witness, *rows[0], parsed, source_rows)
    with pytest.raises(ValueError, match="omit"):
        host_support.verify_source(witness, capture)
    host_support.record(witness, *rows[1], parsed, source_rows)
    host_support.verify_source(witness, capture)
    assert witness["expected_ordinals"] == sorted(source_rows)


@pytest.mark.parametrize(
    "changed",
    [
        lambda text: text.replace("value = 0 : i64", "value = 2 : i64", 1),
        lambda text: text.replace('%j = "arith.index_cast"(%column)', '%j = "arith.index_cast"(%row)'),
        lambda text: text.replace(
            '%value = "tensor.extract"(%data, %i, %j)', '%value = "tensor.extract"(%data, %j, %i)'
        ),
        lambda text: text.replace("affine_map<(d0, d1) -> (d1)>", "affine_map<(d0, d1) -> (d0)>", 1),
        lambda text: text.replace("#arith.overflow<none>", "#arith.overflow<nsw>", 1),
        lambda text: text.replace("value = 1 : i64", "value = -1 : i64", 1),
    ],
)
def test_indexed_extract_near_misses_refuse(tmp_path, changed):
    source = changed(_source())
    with pytest.raises((ValueError, ParseError, VerifyException)):
        capture = _capture(tmp_path, source=source)
        index_source.prove_index_source(capture, index_bits=64)


def test_different_source_node_and_serialized_record_changes_refuse(tmp_path):
    capture = _capture(
        tmp_path,
        trace_change=lambda trace: trace["mlir"]["source_correspondence"][-1].__setitem__(
            "node_id", "g:prepared:root:add"
        ),
    )
    with pytest.raises(ValueError, match="index source refusal"):
        index_source.prove_index_source(capture, index_bits=64)

    other = tmp_path / "other"
    other.mkdir()
    capture = _capture(other)
    record = index_source.prove_index_source(capture, index_bits=64)
    record["records"][0]["index_sources"][0]["axis"] = 1
    with pytest.raises(ValueError, match="serialized"):
        index_source.verify_index_source_record(capture, record)


def test_serialized_record_refuses_python_numeric_and_container_aliases(tmp_path):
    capture = _capture(tmp_path)
    original = index_source.prove_index_source(capture, index_bits=64)
    mutations = (
        lambda record: record["records"][0]["index_sources"][0].__setitem__("axis", False),
        lambda record: record["records"][0]["index_sources"][0].__setitem__("axis", 0.0),
        lambda record: record["records"][0]["index_sources"][0].__setitem__("axis", float("nan")),
        lambda record: record["records"][0].__setitem__("output_shape", tuple(record["records"][0]["output_shape"])),
        lambda record: record["records"][0].__setitem__("extra", 0),
    )
    for mutate in mutations:
        record = deepcopy(original)
        mutate(record)
        with pytest.raises(ValueError, match="serialized"):
            index_source.verify_index_source_record(capture, record)


def test_index_body_component_cannot_be_attributed_to_another_node(tmp_path):
    source = _source().replace(
        '"tensor.extract"(%data, %i, %j)\n          {prov.source_node_ids = ["g:prepared:root:index"]}',
        '"tensor.extract"(%data, %i, %j)\n          {prov.source_node_ids = ["g:prepared:root:add"]}',
    )
    assert source != _source()
    capture = _capture(tmp_path, source=source)
    with pytest.raises(ValueError, match="body component belongs to another prepared node"):
        index_source.prove_index_source(capture, index_bits=64)


def test_wrong_direction_or_missing_trace_literal_refuses(tmp_path):
    def bad_direction(trace):
        trace["graphs"]["prepared"]["nodes"][1]["args"] = [0, 3, -1]

    capture = _capture(tmp_path, trace_change=bad_direction)
    with pytest.raises(ValueError):
        index_source.prove_index_source(capture, index_bits=64)


def test_selected_index_width_and_literal_bounds_are_refusing_premises(tmp_path):
    capture = _capture(tmp_path)
    with pytest.raises(ValueError):
        index_source.prove_index_source(capture, index_bits=3)
