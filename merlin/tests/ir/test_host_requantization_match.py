"""Source epilogue proofs work independently of device shape selection."""

import pytest
from xdsl.dialects.builtin import IntegerAttr, i64

from merlin.common.paths import merlin_dir
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.captured_requant import inspect_chain
from merlin.llvmlower.integer_gemm import IntegerGemm, match_integer_gemm
from merlin.llvmlower.requantization_match import match


def _source(name):
    # Preserved source-only fixtures from the original host compiler tests.
    return (merlin_dir() / "tests/data" / name).read_text()


def test_immutable_bias_proof_needs_no_device_shape_or_resources():
    module = parse_mlir_text(_source("fused_requant.mlir"))
    epilogue = next(op for op in module.walk() if op.name == "linalg.generic")
    proof = match(epilogue)
    assert proof.dimensions == IntegerGemm(1, 17, 19, 20)
    assert proof.integer_bias == [j % 5 - 2 for j in range(19)]
    assert proof.proof["scale"] == 0.03125
    assert not hasattr(proof, "shape")
    for wrong, reason in (("math.round", "nearest-even"),):
        changed = parse_mlir_text(_source("fused_requant.mlir").replace("math.roundeven", wrong))
        with pytest.raises(ValueError, match=reason):
            match(next(op for op in changed.walk() if op.name == "linalg.generic"))


def test_captured_chain_keeps_original_bias_scales_and_refuses_changed_zero_point():
    module = parse_mlir_text(_source("captured_requant.mlir"))
    module.verify()
    contraction = next(op for op in module.walk() if match_integer_gemm(op) is not None)
    chain = inspect_chain(contraction)
    assert len(chain["scales"]) == 2 and chain["bias"].index == 2
    assert chain["dimensions"] == IntegerGemm(1, 3136, 64, 64)
    assert chain["pool"] is None
    constant = chain["quantize"].operands[2].owner.operands[0].owner
    constant.properties["value"] = IntegerAttr(1, i64)
    with pytest.raises(ValueError, match="zero point"):
        inspect_chain(contraction)


def test_nonzero_initial_accumulator_and_fanout_refuse():
    module = parse_mlir_text(_source("captured_requant.mlir"))
    contraction = next(op for op in module.walk() if match_integer_gemm(op) is not None)
    chain = inspect_chain(contraction)
    consumer = chain["operations"][0]
    consumer.parent.insert_op_after(consumer.clone(), consumer)
    with pytest.raises(ValueError, match="fanout"):
        inspect_chain(contraction)
    zero = contraction.operands[-1].owner.operands[0].owner
    zero.properties["value"] = IntegerAttr(1, 32)
    assert match_integer_gemm(contraction) is None
