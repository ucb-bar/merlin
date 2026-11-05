"""Typed direct-source names, without a fabricated whole-program plan."""

import copy
import hashlib

import pytest
from merlin_experiments.phase0.connected_integer_bounds import derive

from merlin.common import mlir_query as mq
from merlin.llvmlower.direct_source_contraction import bind_direct_source_contraction

SOURCE = """module {
func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi32> {
%z = arith.constant 0 : i32
%s = tensor.splat %z : tensor<2x4xi32>
%r = linalg.generic {
indexing_maps = [affine_map<(m,n,k)->(m,k)>,affine_map<(m,n,k)->(k,n)>,affine_map<(m,n,k)->(m,n)>],
iterator_types = ["parallel","parallel","reduction"]
} ins(%a,%b : tensor<2x3xi8>,tensor<3x4xi8>) outs(%s : tensor<2x4xi32>) {
^bb0(%x:i8,%y:i8,%acc:i32):
%u = arith.extsi %x : i8 to i32
%v = arith.extsi %y : i8 to i32
%p = arith.muli %u,%v : i32
%sum = arith.addi %acc,%p : i32
linalg.yield %sum : i32
} -> tensor<2x4xi32>
func.return %r : tensor<2x4xi32>
}}
"""


def fixture(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    proof = derive(mq.parse(SOURCE), [(-3, 3), (-3, 3)])
    proof["interface_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    cb = dict(
        compiler_pointer_abi=dict(version=1, arguments=["a", "b", "out"]),
        tensors={
            "a": dict(shape=[2, 3], dtype="i8", role="input"),
            "b": dict(shape=[3, 4], dtype="i8", role="input"),
            "out": dict(shape=[2, 4], dtype="i32", role="output"),
        },
    )
    return source, cb, proof


def test_exact_single_source_dot_and_all_pointer_storage(tmp_path):
    source, cb, proof = fixture(tmp_path)
    before = source.read_bytes(), copy.deepcopy(cb), copy.deepcopy(proof)
    assert bind_direct_source_contraction(source, cb, proof) == dict(
        ordinal=2, lhs="a", rhs="b", output="out", m=2, k=3, n=4
    )
    assert before == (source.read_bytes(), cb, proof)
    assert "kernel_abi" not in cb and "params" not in cb


@pytest.mark.parametrize(
    "fault",
    ["source", "order", "shape", "tail", "initial", "unsigned", "extra_effect", "extra_tensor", "layout", "version"],
)
def test_foreign_source_recurrence_or_pointer_declaration_refuses(tmp_path, fault):
    source, cb, proof = fixture(tmp_path)
    if fault == "source":
        source.write_text(source.read_text() + "\n")
    elif fault == "order":
        cb["compiler_pointer_abi"]["arguments"].reverse()
    elif fault == "shape":
        cb["tensors"]["a"]["shape"] = [2, 4]
    elif fault == "tail":
        proof["operations"][2]["reduction_extent"] = 2
    elif fault == "initial":
        proof["operations"][2]["initial_interval"] = [1, 1]
    elif fault == "unsigned":
        source.write_text(source.read_text().replace("arith.extsi", "arith.extui"))
        proof["interface_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    elif fault == "extra_effect":
        source.write_text(
            source.read_text().replace("func.return %r", "%extra = arith.constant 1 : i32\nfunc.return %r")
        )
        proof["interface_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    elif fault == "extra_tensor":
        cb["tensors"]["foreign"] = dict(cb["tensors"]["a"])
    elif fault == "layout":
        cb["tensors"]["b"]["layout"] = "transposed"
    else:
        cb["compiler_pointer_abi"]["version"] = True
    with pytest.raises(ValueError):
        bind_direct_source_contraction(source, cb, proof)
