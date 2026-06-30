"""runtime dialect: metadata, build+verify, invalid cases, round-trip, analyses."""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common, runtime as r
from merlin.xdsl_dialects.lowering import analyses

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def test_metadata():
    assert r.DIALECT_NAME == "runtime"
    d = r.get_dialect()
    names = {op.name for op in d.operations}
    assert names == {"runtime." + o for o in r.OPS}


def test_build_verify_roundtrip():
    m = r.build_example()
    m.verify()
    m2 = _common.roundtrip(m, r.get_dialect())
    m2.verify()
    assert _common.text(m) == _common.text(m2)


def _dev():
    from xdsl.dialects.builtin import StringAttr

    return r.DeviceGetOp(result_types=[r.DeviceType()], properties={
        "device": StringAttr("toy_npu0"),
        "backend": r.BackendAttr(r.Backend.SIMULATOR)})


def test_buffer_alloc_rejects_nonpositive_bytes():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import IntegerAttr

    op = r.BufferAllocOp(operands=[_dev().dev], result_types=[r.BufferType()],
                         properties={"bytes": IntegerAttr(0, 64)})
    with pytest.raises(VerifyException, match="positive"):
        op.verify()


def test_metrics_read_rejects_unknown_metric():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    op = r.MetricsReadOp(operands=[_dev().dev], result_types=[r.MetricsType()],
                         properties={"metrics": ArrayAttr([StringAttr("vibes")])})
    with pytest.raises(VerifyException, match="unknown"):
        op.verify()


def test_metrics_read_allows_target_specific_prefix():
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    op = r.MetricsReadOp(operands=[_dev().dev], result_types=[r.MetricsType()],
                         properties={"metrics": ArrayAttr(
                             [StringAttr("target_specific.rocc_stalls")])})
    op.verify()


def test_append_arg_values_must_be_names():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import DictionaryAttr, IntegerAttr, StringAttr

    dev = _dev()
    cb = r.CommandBufferCreateOp(operands=[dev.dev],
                                 result_types=[r.CommandBufferType()],
                                 properties={"target": StringAttr("toy_npu")})
    op = r.CommandBufferAppendOp(operands=[cb.cb], properties={
        "opcode": StringAttr("RES_PACK"),
        "args": DictionaryAttr({"src": IntegerAttr(3, 64)})})
    with pytest.raises(VerifyException, match="must name a tensor"):
        op.verify()


def test_unsupported_queue_kind_flagged_by_analysis():
    from xdsl.ir import Block, Region
    from xdsl.dialects.builtin import (DictionaryAttr, FunctionType, ModuleOp,
                                       StringAttr)
    from xdsl.dialects.func import FuncOp, ReturnOp

    blk = Block()
    dev = _dev()
    cb = r.CommandBufferCreateOp(operands=[dev.dev],
                                 result_types=[r.CommandBufferType()],
                                 properties={"target": StringAttr("toy_npu")})
    ap = r.CommandBufferAppendOp(operands=[cb.cb], properties={
        "opcode": StringAttr("RES_PACK"),
        "args": DictionaryAttr({"src": StringAttr("W"), "dst": StringAttr("W_res")}),
        "queue": r.QueueKindAttr(r.QueueKind.HOST)})  # simulator has no host queue
    blk.add_ops([dev, cb, ap, ReturnOp()])
    fn = FuncOp("bad", FunctionType.from_lists([], []), Region([blk]))
    problems = analyses.check_command_buffer_consistency(ModuleOp([fn]))
    assert problems and "unsupported" in problems[0]
