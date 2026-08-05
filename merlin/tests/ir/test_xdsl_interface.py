"""interface dialect: metadata, build+verify, commit verifier, round-trip, analyses."""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common, interface as i
from merlin.xdsl_dialects.lowering import analyses

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def test_metadata():
    assert i.DIALECT_NAME == "interface"
    d = i.get_dialect()
    names = {op.name for op in d.operations}
    assert names == {"interface." + o for o in i.OPS}


def test_build_verify_roundtrip():
    m = i.build_example(reuse=3)
    m.verify()
    m2 = _common.roundtrip(m, i.get_dialect())
    m2.verify()
    assert _common.text(m) == _common.text(m2)


def _acc():
    from xdsl.ir import Block
    from xdsl.dialects.builtin import TensorType, i32

    return Block(arg_types=[i.AccumulatorType(TensorType(i32, [4, 4]))]).args[0]


def _commit(props, result_elem="i8"):
    from xdsl.dialects.builtin import TensorType, i8, i32

    out_t = TensorType(i8 if result_elem == "i8" else i32, [4, 4])
    return i.CommitOp(operands=[_acc()], result_types=[out_t], properties=props)


def test_commit_rejects_unknown_stage():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    with pytest.raises(VerifyException, match="epilogue stage"):
        _commit({"epilogue": ArrayAttr([StringAttr("frobnicate")])}).verify()


def test_commit_rejects_bias_stage_without_bias_name():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    with pytest.raises(VerifyException, match="no `bias` tensor name"):
        _commit({"epilogue": ArrayAttr([StringAttr("bias_add")])}).verify()


def test_commit_rejects_requant_without_shift():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    with pytest.raises(VerifyException, match="requant"):
        _commit({"epilogue": ArrayAttr([StringAttr("requant")])}).verify()


def test_commit_rejects_output_dtype_mismatch():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    with pytest.raises(VerifyException, match="does not match result element type"):
        _commit({"epilogue": ArrayAttr([]), "output_dtype": StringAttr("i8")},
                result_elem="i32").verify()


def test_resident_pack_layout_must_match_result_type():
    from xdsl.ir import Block
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import StringAttr, TensorType, i8

    Wt = TensorType(i8, [8, 8])
    w = Block(arg_types=[Wt]).args[0]
    op = i.ResidentPackOp(
        operands=[w],
        result_types=[i.ResidentTensorType(Wt, StringAttr("canonical"))],
        properties={"layout": i.LayoutAttr(i.Layout.PACKED_RHS)})
    with pytest.raises(VerifyException, match="does not match result type"):
        op.verify()


def test_matmul_rejects_incompatible_shapes():
    from xdsl.ir import Block
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import TensorType, i8, i32

    blk = Block(arg_types=[TensorType(i8, [4, 5]), TensorType(i8, [6, 4])])
    a, b = blk.args
    op = i.MatmulOp(operands=[a, b],
                    result_types=[i.AccumulatorType(TensorType(i32, [4, 4]))])
    with pytest.raises(VerifyException, match="inner dims disagree"):
        op.verify()


def test_fifo_push_type_mismatch():
    from xdsl.ir import Block
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import TensorType, i8, i32

    elem = TensorType(i8, [2, 2])
    blk = Block(arg_types=[i.FifoType(elem), TensorType(i32, [2, 2])])
    fifo, wrong = blk.args
    with pytest.raises(VerifyException, match="does not match fifo element"):
        i.FifoPushOp(operands=[fifo, wrong]).verify()


def test_use_after_evict_is_flagged():
    from xdsl.ir import Block, Region
    from xdsl.dialects.builtin import (FunctionType, ModuleOp, StringAttr, TensorType,
                                       i8, i32)
    from xdsl.dialects.func import FuncOp, ReturnOp

    At, Wt = TensorType(i8, [4, 8]), TensorType(i8, [8, 4])
    blk = Block(arg_types=[At, Wt])
    a, w = blk.args
    pack = i.ResidentPackOp(
        operands=[w],
        result_types=[i.ResidentTensorType(Wt, StringAttr("packed_rhs"))],
        properties={"layout": i.LayoutAttr(i.Layout.PACKED_RHS)})
    ev = i.ResidentEvictOp(operands=[pack.res])
    mm = i.MatmulOp(operands=[a, pack.res],   # use AFTER evict
                    result_types=[i.AccumulatorType(TensorType(i32, [4, 4]))])
    blk.add_ops([pack, ev, mm, ReturnOp()])
    fn = FuncOp("bad", FunctionType.from_lists([At, Wt], []), Region([blk]))
    problems = analyses.check_no_use_after_evict(ModuleOp([fn]))
    assert problems and "after evict" in problems[0]


def test_clean_module_passes_evict_analysis():
    assert analyses.check_no_use_after_evict(i.build_example(2)) == []


def _parse_type_str(ty: str) -> str:
    """Parse a type string through xDSL's builtin parser and print it back."""
    import io

    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin
    from xdsl.parser import Parser
    from xdsl.printer import Printer

    ctx = Context()
    ctx.load_dialect(Builtin)
    parsed = Parser(ctx, ty).parse_type()
    s = io.StringIO()
    Printer(stream=s).print_attribute(parsed)
    return s.getvalue()


@pytest.mark.parametrize("elem", ["f8E4M3FN", "f8E5M2"])
def test_fp8_element_types_parse_and_roundtrip(elem):
    """fp8 capsule interfaces spell tensors with MLIR's 8-bit float names; the pinned
    xDSL builtin parser lacks them, so merlin's kit registers them (import side effect
    of importing the interface dialect). Parsing must succeed and print back exactly."""
    assert _parse_type_str(f"tensor<16x32x{elem}>") == f"tensor<16x32x{elem}>"


@pytest.mark.parametrize("elem", ["i8", "i32", "f32", "bf16", "f16"])
def test_non_fp8_element_types_still_parse(elem):
    """The fp8 parser hook must not regress the existing integer/float element types."""
    assert _parse_type_str(f"tensor<8x8x{elem}>") == f"tensor<8x8x{elem}>"
