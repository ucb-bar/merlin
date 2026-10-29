"""Scalar assertion support retains frozen host admission, never new compute credit."""
from types import SimpleNamespace

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen.owned_host_model_execution import _untagged_assert_comparisons


def operations(*, scalar="index", tag="shape_guard", extra_use=False):
    source = f'''module {{
      func.func @forward(%a: {scalar}, %b: {scalar}) {{
        %c = arith.cmpi sge, %a, %b : {scalar}
        "cf.assert"(%c) <{{msg = "bound"}}> {{prov.op = "{tag}"}} : (i1) -> ()
        {('"fixture.consume"(%c) : (i1) -> ()' if extra_use else '')}
        func.return
      }}
    }}'''
    module = mq.parse(source)
    return list(next(op for op in module.body.block.ops if op.name == "func.func").body.block.ops)


def test_bound_scalar_assert_support_is_one_demand_with_complete_host_inventory():
    ops = operations()
    frozen = {"host_operations": {"arith.cmpi": 1, "cf.assert": 1}}
    assert _untagged_assert_comparisons(ops, frozen, []) == {id(ops[0])}
    with pytest.raises(ValueError, match="frozen host admission"):
        _untagged_assert_comparisons(ops, {"host_operations": {}}, [])
    device = SimpleNamespace(placement="device", members=[ops[0]])
    assert not _untagged_assert_comparisons(ops, frozen, [device])


@pytest.mark.parametrize("kwargs", [{"scalar": "i64"}, {"tag": "other"}, {"extra_use": True}])
def test_other_untagged_work_is_not_removed_from_demand_inventory(kwargs):
    ops = operations(**kwargs)
    assert not _untagged_assert_comparisons(ops, {"host_operations": {"arith.cmpi": 1, "cf.assert": 1}}, [])
