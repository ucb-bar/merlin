"""Whole-model compilability over REAL model2MLIR linalg (the increment-3 deliverable).

The staged pipeline consumes a real m2m whole-model linalg module and reports, structurally, what
the compiler makes of it: the matmul backbone (real shapes, int8 weight-only dequant idiom resolved
back to the weight function argument), the ops it already lowers, and the fundamental gaps — rmsnorm
/ softmax / rope, whose bodies need rsqrt/exp/div the Python engine has no operator for. The
synthetic tests run everywhere; the tiny_llama test skips unless the m2m checkout is resolvable
(``MERLIN_M2M_DIR`` / ``MERLIN_MODEL2MLIR``).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def _tiny_llama_int8():
    for var in ("MERLIN_M2M_DIR", "MERLIN_MODEL2MLIR"):
        base = os.environ.get(var)
        if base:
            p = Path(base) / "workloads" / "tiny_llama" / "tiny_llama_int8.mlir"
            if p.is_file():
                return p
    return None


def test_fully_modeled_block_is_compilable():
    """A matmul+vector block (add/mul/relu only) reports fully modeled and pipeline-ready."""
    from merlin.xdsl_dialects.lowering.compilability import compilability_report
    from merlin.xdsl_dialects.lowering.input_workload import build_vector_block

    r = compilability_report(build_vector_block(combine="add", relu=True))
    assert r.modeled is True
    assert r.pipeline_ready is True
    assert r.blockers == []
    assert len(r.matmuls) == 2
    assert r.op_classes.get("elementwise", 0) >= 1  # the residual add + relu are vector ops


def test_divf_op_is_a_fundamental_blocker():
    """A generic whose body divides is a fundamental gap — no engine operator — and is reported."""
    from xdsl.ir import Block, Region
    from xdsl.dialects import arith
    from xdsl.dialects import tensor as td
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FunctionType, ModuleOp, TensorType, f32
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import ops as lo
    from xdsl.ir.affine import AffineMap

    from merlin.xdsl_dialects.lowering.compilability import classify_op, compilability_report

    t = TensorType(f32, [4, 4])
    blk = Block(arg_types=[t, t])
    a, b = blk.args
    e = td.EmptyOp((), t)
    idmap = AffineMapAttr(AffineMap.identity(2))
    body = Block(arg_types=[f32, f32, f32])
    d = arith.DivfOp(body.args[0], body.args[1])
    body.add_ops([d, lo.YieldOp(d.result)])
    gen = lo.GenericOp(
        inputs=(a, b), outputs=(e.tensor,),
        body=Region([body]),
        indexing_maps=ArrayAttr([idmap, idmap, idmap]),
        iterator_types=ArrayAttr([lo.IteratorTypeAttr.parallel(),
                                  lo.IteratorTypeAttr.parallel()]),
        result_types=((t,),))
    blk.add_ops([e, gen, ReturnOp(gen.results[0])])
    fn = FuncOp("f", FunctionType.from_lists([t, t], [t]), Region([blk]))
    r = compilability_report(ModuleOp([fn]))
    assert classify_op(gen) == "unmodeled"
    assert r.modeled is False
    assert any("arith.divf" in b["body_math"] for b in r.blockers)


@pytest.mark.skipif(_tiny_llama_int8() is None,
                    reason="model2MLIR checkout not resolvable (set MERLIN_M2M_DIR)")
def test_real_tiny_llama_int8_backbone_and_gaps():
    """The compiler consumes a real int8 tiny_llama: it inventories the matmul backbone (real
    shapes, weight traced through the dequant idiom to an i8 arg) and honestly reports the rmsnorm/
    softmax/rope math it cannot yet model."""
    from merlin.xdsl_dialects.lowering.compilability import report_from_file

    r = report_from_file(_tiny_llama_int8())

    # 15 int8 linears; every one resolves its weight through dequantize_per_channel to an i8 arg
    # with a per-channel scale — the int8 weight-only idiom handled structurally.
    assert len(r.matmuls) == 15
    for s in r.matmuls:
        assert s.m and s.k and s.n                       # real 2D shapes
        assert s.weight_arg is not None                  # traced to a function argument
        assert s.scale_arg is not None                   # per-channel scale argument
        assert s.quant == "quant_ext.dequantize_per_channel"
        assert s.weight_dtype == "i8"

    # A full transformer is NOT engine-runnable here — honest, not a silent pass.
    assert r.modeled is False
    assert r.pipeline_ready is False

    # The blockers are the transformer nonlinearities, and they name the exact missing math.
    needed = {m for b in r.blockers for m in b["body_math"]}
    assert {"math.rsqrt", "math.exp", "arith.divf"} <= needed
    assert "normalization" in r.unmodeled_families
