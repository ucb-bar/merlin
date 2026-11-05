"""Raw group pointer bindings keep their original singleton index domains."""

import pytest
from test_compact_dense_group import fixture, schedule

from merlin.compile.scheduling.compact_dense_group_check import _range
from merlin.sched.ir import Loop, Var


def test_actual_group_retains_singleton_scopes_for_original_pointer_bindings():
    model = fixture(same_shape=True)
    result = schedule(model)
    assert result is not None

    def loops(body):
        for statement in body:
            if isinstance(statement, Loop):
                yield statement
                yield from loops(statement.body)

    assert any(loop.extent == 1 for loop in loops(result.kernel.body))
    assert result.realization["requested_lhs_bytes"] == model[0][0].m * model[0][0].k
    result.revalidate()


@pytest.mark.parametrize("scopes", [(), ((1, "i", 1), (2, "i", 1))])
def test_missing_or_ambiguous_pointer_domain_still_refuses(scopes):
    with pytest.raises(ValueError, match="unbound or ambiguous"):
        _range(Var("i"), scopes)
