"""Compile-time tensor movement and wide scans preserve exact source bits."""

from dataclasses import replace

import numpy as np
import pytest

from merlin.semantic_compiler.constant_folding import fold_source_constants, verify_constant_folding
from merlin.semantic_compiler.linalg_constants import read_tensor_bytes
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType
from merlin.semantic_compiler.reference import evaluate_graph
from merlin.semantic_compiler.source_preparation import prepare_source, verify_source_preparation


def _request(raw, dtype="f32", policy="raw-bit-pattern"):
    width = {"i1": 1, "i8": 1, "i32": 4, "i64": 8, "bf16": 2, "f32": 4, "f64": 8, "fp8_e4m3": 1}[dtype]
    count = len(raw) // width
    type_ = TensorType((count,), dtype, policy)
    literal = SemanticNode("literal", "constant", (), type_, effect="constant")
    encoding, _ = read_tensor_bytes(type_, raw)
    return KernelRequest((literal,), ("literal",), ("external",), (), "synthetic",
                         constants=(ConstantBinding("literal", encoding, raw.hex()),))


def _evaluate(source):
    return evaluate_graph(source, {}, constants={c.node_id: read_tensor_bytes(
        source.node(c.node_id).type, bytes.fromhex(c.data_hex),
    )[1] for c in source.constants})


@pytest.mark.parametrize("dtype,policy,width", [
    ("i1", "boolean", 1), ("i8", "signed-i8", 1), ("i32", "exact-i32", 4),
    ("i64", "signed-i64", 8), ("bf16", "raw-bit-pattern", 2), ("f32", "raw-bit-pattern", 4),
    ("f64", "raw-bit-pattern", 8), ("fp8_e4m3", "raw-bit-pattern", 1),
])
def test_movement_preserves_arbitrary_bits_including_nan_payloads(dtype, policy, width):
    codes = [0, 1, (1 << (8 * width)) - 1, 1 << (8 * width - 1), 3, 7]
    if dtype == "i1":
        codes = [0, 1, 1, 0, 1, 0]
    source = _request(b"".join(code.to_bytes(width, "little") for code in codes), dtype, policy)
    matrix = SemanticNode("matrix", "reshape", ("literal",), TensorType((2, 3), dtype, policy))
    gathered = SemanticNode("transpose", "gather", ("matrix",), TensorType((3, 2), dtype, policy),
                            index_maps=(IndexMap(2, ((0, 1), (1, 0)), (0, 0)),
                                        IndexMap(2, ((1, 0), (0, 1)), (0, 0))))
    identity = SemanticNode("copy", "identity", ("transpose",), gathered.type)
    source = replace(source, nodes=(*source.nodes, matrix, gathered, identity), outputs=("copy", "transpose"),
                     output_storages=("external", "external"))
    folded, record = fold_source_constants(source, include_tensor_ops=True)
    assert len(record["folds"]) == 3 and _evaluate(folded) == _evaluate(source)
    assert verify_constant_folding(folded, record).digest() == source.digest()
    old, legacy = fold_source_constants(source)
    assert not legacy["folds"] and verify_constant_folding(old, legacy).digest() == source.digest()


@pytest.mark.parametrize("outer,extent,stride", [(1, 3, 1), (2, 3, 2), (3, 1, 4), (1, 50, 1)])
def test_ordered_scan_rounds_f64_accumulator_then_each_f32_output(outer, extent, stride):
    count = outer * extent * stride
    rng = np.random.default_rng(count)
    values = rng.normal(size=(outer, extent, stride)).astype("<f4")
    if extent == 3:
        values[0, :, 0] = [16777216.0, 1.0, -16777216.0]
    source = _request(values.tobytes())
    node = SemanticNode("prefix", "ordered_f32_scan_f64", ("literal",), source.nodes[0].type,
                        attrs=(("stride", stride), ("extent", extent),
                               ("numeric_semantics", "ordered-f32-prefix-f64-rne-v1")))
    source = replace(source, nodes=(*source.nodes, node), outputs=(node.id,))
    prepared, record = prepare_source(source)
    assert record["folds"] and verify_source_preparation(prepared, record).digest() == source.digest()
    expected = np.cumsum(values, axis=1, dtype=np.float64).astype("<f4").tobytes()
    assert bytes.fromhex(prepared.constants[-1].data_hex) == expected
    assert _evaluate(prepared) == _evaluate(source)
    limited, receipt = fold_source_constants(source, include_tensor_ops=True, element_budget=2 * count - 1)
    assert not receipt["folds"] and receipt["deferred"] and limited.digest() == source.digest()


@pytest.mark.parametrize("mutation", ["bounds", "output_map", "numerics", "shape"])
def test_invalid_constant_views_are_deferred(mutation):
    source = _request(np.arange(6, dtype="<f4").tobytes())
    node = SemanticNode("view", "gather", ("literal",), source.nodes[0].type,
                        index_maps=(IndexMap(1, ((1,),), (0,)), IndexMap(1, ((1,),), (0,))))
    if mutation == "bounds":
        node = replace(node, index_maps=(IndexMap(1, ((1,),), (1,)), node.index_maps[1]))
    elif mutation == "output_map":
        node = replace(node, index_maps=(node.index_maps[0], IndexMap(1, ((-1,),), (5,))))
    elif mutation == "numerics":
        node = replace(node, type=TensorType((6,), "f32", "unrelated"))
    else:
        node = replace(node, op="reshape", type=TensorType((7,), "f32", "raw-bit-pattern"), index_maps=())
    source = replace(source, nodes=(*source.nodes, node), outputs=(node.id,))
    result, receipt = fold_source_constants(source, include_tensor_ops=True)
    assert not receipt["folds"] and receipt["deferred"] and result.digest() == source.digest()


@pytest.mark.parametrize("bits", [0x7FC00001, 0xFFC00001])
def test_scan_with_unqualified_nan_remains_unfolded(bits):
    source = _request(bits.to_bytes(4, "little"))
    node = SemanticNode("scan", "ordered_f32_scan_f64", ("literal",), source.nodes[0].type,
                        attrs=(("stride", 1), ("extent", 1),
                               ("numeric_semantics", "ordered-f32-prefix-f64-rne-v1")))
    source = replace(source, nodes=(*source.nodes, node), outputs=(node.id,))
    result, record = fold_source_constants(source, include_tensor_ops=True)
    assert not record["folds"] and "NaN" in record["deferred"][0]["reason"]
    assert verify_constant_folding(result, record).digest() == source.digest()


def test_scan_signed_zero_subnormal_and_output_overflow():
    codes = [0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0x7F7FFFFF, 0xFF7FFFFF]
    source = _request(b"".join(word.to_bytes(4, "little") for word in codes))
    node = SemanticNode("prefix", "ordered_f32_scan_f64", ("literal",), source.nodes[0].type,
                        attrs=(("stride", 1), ("extent", len(codes)),
                               ("numeric_semantics", "ordered-f32-prefix-f64-rne-v1")))
    source = replace(source, nodes=(*source.nodes, node), outputs=(node.id,))
    folded, record = fold_source_constants(source, include_tensor_ops=True)
    assert _evaluate(folded) == _evaluate(source)
    values = _evaluate(folded)[0].elements
    assert values[0] == 0 and values[1] == 1 and values[4] == 0x7F800000 and values[5] == 0x7F7FFFFF
    assert verify_constant_folding(folded, record).digest() == source.digest()
