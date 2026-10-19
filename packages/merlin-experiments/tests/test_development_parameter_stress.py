"""Real parameter ownership and unchanged connected integer arithmetic."""

import ast

import numpy as np
import pytest
from merlin_experiments.phase0 import development_parameter_stress as P
from merlin_experiments.phase0 import development_reference as R
from merlin_experiments.phase0 import development_sources as D


def test_parameter_payload_scales_separately_from_repeated_work_and_shared_storage():
    distinct = P.declaration("public_distinct", width=512, depth=4)
    shared = P.declaration("public_shared", width=512, depth=4, shared=True)
    a, b = D.describe(distinct), D.describe(shared)
    assert a["parameter_bytes"] == 1024**2 and b["parameter_bytes"] == 256 * 1024
    assert a["scalar_products"] == b["scalar_products"] == 1024**2
    assert len(a["parameters"]) == 4 and len(b["parameters"]) == 1
    for d, n in [(distinct, 4), (shared, 1)]:
        tree = ast.parse(D.render(d))
        calls = [x for x in ast.walk(tree) if isinstance(x, ast.Call)]
        assert sum(ast.unparse(x.func) == "self.register_parameter" for x in calls) == n
        assert sum(ast.unparse(x.func) == "torch._int_mm" for x in calls) == 4
        assert all(
            "requires_grad=False" in ast.unparse(x) for x in calls if ast.unparse(x.func) == "torch.nn.Parameter"
        )
    big = P.declaration("public_three_mib", width=1024, depth=3)
    assert D.describe(big)["parameter_bytes"] == 3 * 1024**2
    refused = {**big, "arithmetic": dict(contract=D.CONTRACT, readout_shift=4)}
    with pytest.raises(ValueError, match="existing exact"):
        D.describe(refused)


def test_independent_complete_reference_preserves_signed_intermediates_and_both_liveouts():
    d = P.declaration("toy", width=5, depth=3)
    values = R.inputs(d)
    state = values[0].astype(np.int64)
    for i in range(3):
        weight = P.parameter(d, i)
        # A scalar checked reduction is independent of BLAS and the generated Torch graph.
        wide = np.asarray(
            [[sum(int(state[0, k]) * int(weight[k, j]) for k in range(5)) for j in range(5)]], dtype=np.int64
        )
        state = np.asarray([[max(-128, min(127, (int(x) + 4) // 8)) for x in wide[0]]], dtype=np.int8)
    got = R.evaluate(d, values)
    np.testing.assert_array_equal(got[0], wide)
    np.testing.assert_array_equal(got[1], state)
    assert [x.dtype for x in got] == [np.dtype("int32"), np.dtype("int8")]
    with pytest.raises(ValueError, match="unchanged independent"):
        R.evaluate(d, [values[0] + 1])
    with pytest.raises(ValueError, match="unchanged independent"):
        R.evaluate(d, [values[0].astype(np.float32)])
    with pytest.raises(ValueError, match="ordinal"):
        P.parameter(d, True)
