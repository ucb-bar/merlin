"""dse dialect: metadata, build+verify, invalid cases, round-trip."""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common, dse

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")


def test_metadata():
    assert dse.DIALECT_NAME == "dse"
    d = dse.get_dialect()
    names = {op.name for op in d.operations}
    assert names == {"dse." + o for o in dse.OPS}


def test_build_verify_roundtrip():
    m = dse.build_example()
    m.verify()
    m2 = _common.roundtrip(m, dse.get_dialect())
    m2.verify()
    assert _common.text(m) == _common.text(m2)


def test_candidate_rejects_non_interface_ops():
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    op = dse.CandidateOp(
        result_types=[dse.InterfaceCandidateType(StringAttr("x"))],
        properties={"candidate_name": StringAttr("x"),
                    "interface_ops": ArrayAttr([StringAttr("toynpu.res_pack")])})
    with pytest.raises(VerifyException, match="interface"):
        op.verify()


def test_result_rejects_non_integer_metric():
    from xdsl.ir import Block
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import DictionaryAttr, StringAttr

    cand = Block(arg_types=[dse.InterfaceCandidateType(StringAttr("x"))]).args[0]
    op = dse.ResultOp(operands=[cand], properties={
        "variant": dse.VariantAttr(_common.Visibility.BASELINE),
        "workload": StringAttr("w"),
        "backend": StringAttr("simulator"),
        "metrics": DictionaryAttr({"cycles": StringAttr("fast")})})
    with pytest.raises(VerifyException, match="integer"):
        op.verify()
