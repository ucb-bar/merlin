"""schedule dialect: metadata, build+verify, invalid cases, round-trip."""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common, contract as c, schedule as s

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def test_metadata():
    assert s.DIALECT_NAME == "schedule"
    d = s.get_dialect()
    names = {op.name for op in d.operations}
    assert names == {"schedule." + o for o in s.OPS}


def test_build_verify_roundtrip():
    m = s.build_example()
    m.verify()
    m2 = _common.roundtrip(m, s.get_dialect(), c.get_dialect())
    m2.verify()
    assert _common.text(m) == _common.text(m2)


def _value():
    from xdsl.ir import Block
    from xdsl.dialects.builtin import TensorType, i8

    return Block(arg_types=[TensorType(i8, [4, 4])]).args[0]


def test_select_interface_rejects_unknown_interface():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import StringAttr

    op = s.SelectInterfaceOp(operands=[_value()],
                             properties={"interface": StringAttr("warp_drive")})
    with pytest.raises(VerifyException, match="not a known interface"):
        op.verify()


def test_vector_strategy_rejects_non_tail_policy():
    from xdsl.utils.exceptions import VerifyException

    op = s.VectorStrategyOp(operands=[_value()], properties={
        "strategy": s.VectorStrategyAttr(s.VectorStrategy.SCALABLE_VL),
        "tail": s.VectorStrategyAttr(s.VectorStrategy.FIXED_WIDTH)})
    with pytest.raises(VerifyException, match="tail must be a tail policy"):
        op.verify()


def test_group_dispatch_rejects_empty():
    from xdsl.utils.exceptions import VerifyException

    op = s.GroupDispatchOp(operands=[[]], properties={
        "granularity": s.DispatchGranularityAttr(s.DispatchGranularity.OP)})
    with pytest.raises(VerifyException, match="at least one item"):
        op.verify()


def test_bad_enum_spelling_rejected_by_parser():
    from xdsl.parser import Parser
    from xdsl.utils.exceptions import ParseError

    ctx = _common.make_context(s.get_dialect())
    ir = '"schedule.bind"() <{target = "@x"}> : () -> !schedule.handle\n'
    Parser(ctx, ir).parse_op()  # sanity: valid op parses
    bad = ('"schedule.vector_strategy"(%0) <{strategy = '
           '#schedule<vector_strategy warp_speed>}> : (!schedule.handle) -> ()')
    with pytest.raises(Exception):
        Parser(ctx, '%0 = ' + ir + bad).parse_module()
