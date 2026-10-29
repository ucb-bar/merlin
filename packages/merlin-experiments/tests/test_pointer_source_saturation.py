"""Prospective exact final-saturation source grammar, independent of schedules."""

import copy
import hashlib

import pytest
from xdsl.utils.exceptions import VerifyException
from merlin_experiments.phase0.connected_integer_bounds import derive
from test_direct_source_contraction import SOURCE, fixture as raw_fixture

from merlin.common import mlir_query as mq
from merlin.llvmlower.pointer_source_saturation import bind_pointer_source_saturation


def native_text(*, comparisons=False):
    def body(kind):
        if comparisons:
            predicate = "slt" if kind == "maxsi" else "sgt"
            return f"%cmp = arith.cmpi {predicate}, %x, %bound : i32\n%v = arith.select %cmp, %bound, %x : i32"
        return f"%v = arith.{kind} %x, %bound : i32"

    text = SOURCE.replace("-> tensor<2x4xi32> {", "-> tensor<2x4xi8> {", 1)
    epilogue = ""
    previous = "%r"
    for suffix, kind, bound in [("lo", "maxsi", -128), ("hi", "minsi", 127)]:
        epilogue += f"""
%c{suffix} = arith.constant {bound} : i32
%s{suffix} = tensor.splat %c{suffix} : tensor<2x4xi32>
%e{suffix} = tensor.empty() : tensor<2x4xi32>
%r{suffix} = linalg.generic {{indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]}}
 ins({previous}, %s{suffix} : tensor<2x4xi32>,tensor<2x4xi32>) outs(%e{suffix} : tensor<2x4xi32>) {{
^bb0(%x:i32,%bound:i32,%old:i32):
{body(kind)}
linalg.yield %v : i32
}} -> tensor<2x4xi32>
"""
        previous = "%r" + suffix
    epilogue += """
%eo = tensor.empty() : tensor<2x4xi8>
%out = linalg.generic {indexing_maps = [affine_map<(m,n)->(m,n)>,affine_map<(m,n)->(m,n)>], iterator_types = ["parallel","parallel"]}
 ins(%rhi : tensor<2x4xi32>) outs(%eo : tensor<2x4xi8>) {
^bb0(%x:i32,%old:i8):
%v = arith.trunci %x : i32 to i8
linalg.yield %v : i8
} -> tensor<2x4xi8>
func.return %out : tensor<2x4xi8>
"""
    return text.replace("func.return %r : tensor<2x4xi32>", epilogue)


def fixture(tmp_path, *, comparisons=False, shape=(2, 3, 4)):
    m, k, n = shape
    text = native_text(comparisons=comparisons).replace("2x3", f"{m}x{k}").replace("3x4", f"{k}x{n}").replace("2x4", f"{m}x{n}")
    source = tmp_path / "native.mlir"
    source.write_text(text)
    proof = derive(mq.parse(text), [(-8, 8)] * 2)
    proof["interface_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    cb = dict(compiler_pointer_abi=dict(version=1, arguments=["a", "b", "out"]), tensors={
        "a": dict(shape=[m, k], dtype="i8", role="input"),
        "b": dict(shape=[k, n], dtype="i8", role="input"),
        "out": dict(shape=[m, n], dtype="i8", role="output")})
    return source, cb, proof


@pytest.mark.parametrize("comparisons", [False, True])
@pytest.mark.parametrize("shape", [(2, 3, 4), (17, 33, 47), (12544, 64, 256)])
def test_full_source_dot_final_saturation_and_complete_proof(tmp_path, comparisons, shape):
    source, cb, proof = fixture(tmp_path, comparisons=comparisons, shape=shape)
    original = source.read_bytes(), copy.deepcopy(cb), copy.deepcopy(proof)
    assert bind_pointer_source_saturation(source, cb, proof) == (
        dict(ordinal=2, lhs="a", rhs="b", output="out", m=shape[0], k=shape[1], n=shape[2], output_dtype="i8"),)
    assert original == (source.read_bytes(), cb, proof)


def test_raw_controls_preserve_exact_source_reduction_binding(tmp_path):
    source, cb, proof = raw_fixture(tmp_path)
    assert bind_pointer_source_saturation(source, cb, proof)[0]["output_dtype"] == "i32"


@pytest.mark.parametrize("fault", ["lower", "upper", "comparison", "unsigned", "trunc", "dtype", "proof", "alias", "source"])
def test_inexact_clamp_proof_or_output_contract_refuses(tmp_path, fault):
    source, cb, proof = fixture(tmp_path, comparisons=True)
    if fault in {"lower", "upper", "comparison", "unsigned", "trunc"}:
        old, new = {
            "lower": ("constant -128", "constant -127"),
            "upper": ("constant 127", "constant 126"),
            "comparison": ("cmpi slt", "cmpi sgt"),
            "unsigned": ("cmpi slt", "cmpi ult"),
            "trunc": ("arith.trunci", "arith.extsi"),
        }[fault]
        source.write_text(source.read_text().replace(old, new))
        proof["interface_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    elif fault == "dtype":
        cb["tensors"]["out"]["dtype"] = "i32"
    elif fault == "proof":
        proof["operations"][2]["reduction_extent"] -= 1
    elif fault == "alias":
        cb["compiler_pointer_abi"]["arguments"][-1] = "a"
    else:
        source.write_text(source.read_text() + "\n")
    with pytest.raises((ValueError, VerifyException)):
        bind_pointer_source_saturation(source, cb, proof)
