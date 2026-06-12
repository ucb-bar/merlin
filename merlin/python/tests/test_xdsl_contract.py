"""contract dialect: metadata, build+verify, invalid cases, round-trip."""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common, contract as c

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def test_metadata():
    assert c.DIALECT_NAME == "contract"
    d = c.get_dialect()
    assert d.name == "contract"
    names = {op.name for op in d.operations}
    assert names == {"contract." + o for o in c.OPS}


def test_build_verify_roundtrip():
    m = c.build_example()
    m.verify()
    m2 = _common.roundtrip(m, c.get_dialect())
    m2.verify()
    assert _common.text(m) == _common.text(m2)


def _block_with_tensor():
    from xdsl.ir import Block
    from xdsl.dialects.builtin import TensorType, i8

    blk = Block(arg_types=[TensorType(i8, [4, 4])])
    return blk, blk.args[0]


def test_assume_rejects_unknown_kind():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import StringAttr

    _, v = _block_with_tensor()
    op = c.AssumeOp(operands=[v], properties={"kind": StringAttr("vibes")})
    with pytest.raises(VerifyException, match="not registered"):
        op.verify()


def test_fact_rejects_negative_reuse():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import IntegerAttr

    _, v = _block_with_tensor()
    op = c.FactOp(operands=[v], properties={"reuse_count": IntegerAttr(-1, 64)})
    with pytest.raises(VerifyException, match="non-negative"):
        op.verify()


def test_require_rejects_unknown_predicate():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    op = c.RequireOp(properties={
        "feature": StringAttr("resident_packed_tensor"),
        "requires": ArrayAttr([StringAttr("totally_made_up")])})
    with pytest.raises(VerifyException, match="not registered"):
        op.verify()


def test_check_rejects_mismatched_proof():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import StringAttr

    _, v = _block_with_tensor()
    prove = c.ProveOp(operands=[v],
                      result_types=[c.ProofType(StringAttr("capacity_fit"))],
                      properties={"requirement": StringAttr("capacity_fit")})
    check = c.CheckOp(operands=[v, [prove.proof]],
                      properties={"requirement": StringAttr("rhs_immutable")})
    with pytest.raises(VerifyException, match="does not discharge"):
        check.verify()


def test_capability_rejects_bad_feature_identifier():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    op = c.CapabilityOp(
        result_types=[c.CapabilityType(StringAttr("t"))],
        properties={"sym_name": StringAttr("t"),
                    "features": ArrayAttr([StringAttr("not a name!")])})
    with pytest.raises(VerifyException, match="not a valid identifier"):
        op.verify()
