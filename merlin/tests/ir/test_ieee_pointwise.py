# ruff: noqa: E501 -- static MLIR fixtures retain one operation per line
"""IEEE classification covers raw words; finite arithmetic is absent from readout."""

import math
import random

import numpy as np
import pytest
from mlir_fixtures import emitter, float_value, forward, generic, pointwise, run_ops
from xdsl.dialects.builtin import bf16, f16, f32, f64
from xdsl.ir import Block

from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.ieee_pointwise import dependence_proof, overlay, special


@pytest.mark.parametrize("typ", [f16, bf16, f32, f64])
def test_classification_does_not_quiet_or_round_inputs(typ):
    block = Block(arg_types=[typ])
    result = special(block.args[0], emitter(block))
    width = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}[str(typ)]
    rng = random.Random(842)
    samples = range(1 << 16) if width == 16 else [rng.getrandbits(width) for _ in range(4096)]
    for raw in samples:
        values = {block.args[0]: raw}
        run_ops(block.ops, values)
        assert bool(values[result]) == (not math.isfinite(float_value(raw, typ)))


@pytest.mark.parametrize("operation", ["addf", "subf", "mulf"])
def test_floating_special_arithmetic_occurs_only_in_the_affected_branch(operation):
    typ = "tensor<2xf32>"
    source = pointwise([typ, typ], typ, f"%v=arith.{operation} %a,%b : f32")
    _, proofs = rewrite_floating_components(source, [(0, 1), (0, 1)], ieee_pointwise=True)
    assert proofs[0]["ieee_special_readout"] == "affected_elements_only"
    branches = [op for op in source.walk() if op.name == "scf.if"]
    assert len(branches) == 1
    chosen, retained = branches[0].regions
    assert not any(op.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"} for op in chosen.walk())
    assert [op.name for op in retained.walk()] == ["scf.yield"]
    source.verify()


@pytest.mark.parametrize("typ", [f16, bf16, f32, f64])
@pytest.mark.parametrize("operation", ["add", "sub", "mul"])
def test_explicit_special_words_match_native_cpu(typ, operation):
    torch = pytest.importorskip("torch")
    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.llvmlower.ieee_pointwise import special_arithmetic

    fraction, exponent, bias = FORMAT[str(typ)]
    width = 1 + fraction + exponent
    mask, sign = (1 << width) - 1, 1 << (fraction + exponent)
    infinity = ((1 << exponent) - 1) << fraction
    nan = infinity | 1 << (fraction - 1)
    words = [0, sign, bias << fraction, (bias << fraction) | sign, infinity, infinity | sign, nan, nan | sign]
    block = Block(arg_types=[typ, typ])
    result = special_arithmetic("arith." + operation + "f", block.args, emitter(block))
    dtype = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32, "f64": torch.float64}[str(typ)]
    storage = {16: torch.int16, 32: torch.int32, 64: torch.int64}[width]
    for a in words:
        for b in words:
            if max(a & ~sign, b & ~sign) < infinity:
                continue
            if a & ~sign > infinity and b & ~sign > infinity and a != b:
                continue  # The independent admission audit refuses competing payloads.
            values = dict(zip(block.args, [a, b]))
            run_ops(block.ops, values)
            pair = torch.tensor([v if v < sign else v - (1 << width) for v in [a, b]], dtype=storage).view(dtype)
            expected = int(getattr(torch, operation)(pair[0], pair[1]).view(storage)) & mask
            assert values[result] == expected


def test_special_readout_does_not_taint_an_unread_shape_operand():
    module = pointwise(["tensor<2xf32>"] * 3, "tensor<2xf32>", "%v=arith.addf %a,%b : f32")
    source = next(op for op in module.walk() if op.name == "linalg.generic")
    proof = dependence_proof(source)
    assert [d["operand"] for d in proof["input_dependencies"]] == [0, 1]
    readout = Block(arg_types=[f32] * 4)
    overlay(source, readout.args[:3], readout.args[3], emitter(readout))
    assert not readout.args[2].uses
    assert readout.args[0].uses and readout.args[1].uses


def test_overlay_dependence_follows_permuted_and_broadcast_input_maps():
    maps = [f"affine_map<(d0,d1)->({m})>" for m in ("d1,d0", "d1", "d0,d1", "d0,d1")]
    ins = [("%x", "tensor<3x2xf32>"), ("%y", "tensor<3xf32>"), ("%unread", "tensor<2x3xf32>")]
    params = "%x: tensor<3x2xf32>, %y: tensor<3xf32>, %unread: tensor<2x3xf32>"
    source = forward(params, "tensor<2x3xf32>", generic(ins, "tensor<2x3xf32>", "%v=arith.addf %a,%b : f32", maps=maps))
    op = next(op for op in source.walk() if op.name == "linalg.generic")
    proof = dependence_proof(op)
    assert proof["output_shape"] == [2, 3]
    assert [d["operand"] for d in proof["input_dependencies"]] == [0, 1]
    arrays = [np.array([[0., np.inf], [np.nan, 1.], [-0., 2.]]),
              np.array([0., 3., -np.inf]), np.full((2, 3), np.nan)]
    maps = [m.data for m in op.get_indexing_maps()]
    affected = np.zeros((2, 3), dtype=bool)
    for index in np.ndindex(affected.shape):
        affected[index] = any(not np.isfinite(arrays[d["operand"]][maps[d["operand"]].eval(index, ())])
                              for d in proof["input_dependencies"])
    assert np.array_equal(affected, ~np.isfinite(arrays[0].T) | ~np.isfinite(arrays[1][None, :]))
    assert not affected[0, 0]  # Unread NaN storage must never taint this finite output.


def test_overlay_refuses_reduction_dependence_without_a_closed_proof():
    maps = ["affine_map<(d0)->(d0)>", "affine_map<(d0)->()>"]
    total = generic([("%x", "tensor<3xf32>")], "tensor<f32>", "%v=arith.addf %a,%acc : f32", maps=maps, iterators=["reduction"], out="%seed", acc="%acc")
    module = forward("%x: tensor<3xf32>,%seed: tensor<f32>", "tensor<f32>", total)
    op = next(op for op in module.walk() if op.name == "linalg.generic")
    with pytest.raises(ValueError, match="closed parallel output index map"):
        dependence_proof(op)
