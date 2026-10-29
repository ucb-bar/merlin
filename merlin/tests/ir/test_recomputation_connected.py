"""Public synthetic host fan-out before two exact integer contractions.

The fixed source and ordered Python reference provide no capture, target
capability, admission or timing qualification. Inputs are invocation-owned.
"""

from pathlib import Path

import numpy as np

from merlin.common.paths import repo_root
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.transforms.certificates import ir_text
from merlin.transforms.dense_ir import static_zero_integer_matmul_geometry
from merlin.transforms.pipeline_state import PipelineContext
from merlin.transforms.recomputation import (
    RecomputationBudgets,
    RecomputationPermissions,
    plan_source_recomputation,
)

INPUTS = (
    ("x", (2, 5), "i8"),
    ("bias0", (2, 5), "i8"),
    ("bias1", (2, 5), "i8"),
    ("w0", (5, 5), "i8"),
    ("w1", (5, 3), "i8"),
    ("state", (2, 5), "i32"),
)
OUTPUTS = (("next_state", (2, 5), "i32"), ("c0", (2, 5), "i32"), ("c1", (2, 3), "i32"), *INPUTS)


def signed(value, width):
    value = int(value) % (1 << width)
    return value - (1 << width) if value >= 1 << (width - 1) else value


def initial_state():
    return np.array([[2147483647, -2147483648, 255, -129, 128], [-1, 0, 127, -128, 65535]], dtype=np.int32)


def inputs(epoch, state):
    assert type(epoch) is int and 0 <= epoch < 3
    assert state.shape == (2, 5) and state.dtype == np.int32
    x = np.array([[-128, 127, -1, 0, 1], [126, -127, 2, -3, 99]], dtype=np.int8)
    x = np.array([[signed(int(v) + 7 * epoch, 8) for v in row] for row in x], dtype=np.int8)
    b0 = np.array([[127, 1, -128, -1, 3], [-128, 127, 6, -8, 11]], dtype=np.int8)
    b1 = np.array([[-1, 2, 127, -128, 1], [127, -128, -3, 5, 9]], dtype=np.int8)
    w0 = np.array([[signed(-128 + 37 * r + 23 * c, 8) for c in range(5)] for r in range(5)], dtype=np.int8)
    w1 = np.array([[signed(127 - 41 * r + 19 * c, 8) for c in range(3)] for r in range(5)], dtype=np.int8)
    return x, b0, b1, w0, w1, state.copy()


def reference(x, bias0, bias1, w0, w1, state):
    a, b = np.empty((2, 5), dtype=np.int8), np.empty((2, 5), dtype=np.int8)
    for m in range(2):
        for k in range(5):
            narrow = signed(int(state[m, k]), 8)
            p = signed(int(x[m, k]) + narrow, 8)
            a[m, k] = signed(p + int(bias0[m, k]), 8)
            b[m, k] = signed(p * int(bias1[m, k]), 8)

    def contraction(left, right):
        result = np.empty((2, right.shape[1]), dtype=np.int32)
        for m in range(2):
            for n in range(right.shape[1]):
                acc = 0
                for k in range(5):
                    product = signed(int(left[m, k]) * int(right[k, n]), 32)
                    acc = signed(acc + product, 32)
                    assert -(1 << 19) <= acc < 1 << 19
                result[m, n] = acc
        return result

    c0, c1 = contraction(a, w0), contraction(b, w1)
    state_after = np.array(
        [[signed(int(c0[m, k]) + int(state[m, k]), 32) for k in range(5)] for m in range(2)], dtype=np.int32
    )
    return state_after, c0, c1, *(value.copy() for value in (x, bias0, bias1, w0, w1, state))


def source_path() -> Path:
    return repo_root() / "merlin/tests/fixtures/recomputation_connected.mlir"


def checked_plan():
    source = source_path().read_text()
    module = parse_mlir_text(source)
    module.verify()
    permissions = RecomputationPermissions(
        "original-public-total-integer-source-v1",
        repeat_evaluation_no_faults=True,
        repeat_evaluation_unobserved=True,
        suppressed_materialization_no_faults=True,
        suppressed_materialization_unobserved=True,
        inputs_stable_during_region=True,
    )
    budgets = RecomputationBudgets(8, 4096, 2, 32, 4096, 4096, 65536)
    plan = plan_source_recomputation(
        module,
        PipelineContext(numerical_policy="exact-modular-source", execution="pure-host-region"),
        permissions,
        budgets,
    )
    assert plan is not None
    return plan


def test_mixed_width_host_fanout_and_reductions_are_admitted_without_rewriting_them():
    from merlin.compile.scheduling.multioutput_fusion import read_source_multioutput
    from merlin.llvmlower.layout_conformance import scalar_bodies
    from merlin.targetgen.capsule_source import model_op_demands_checked

    plan = checked_plan()
    members = tuple(op for op in plan.source.walk() if op.name == "linalg.generic")
    source = read_source_multioutput(plan.source, members[:3])
    assert source.producer.graph.input_bits == (8, 32) and source.producer.graph.output_bits == 8
    for module in (plan.source, plan.candidate):
        demands = model_op_demands_checked(ir_text(module), "i8", source_kind="typed_static_integer_v1")
        assert demands
    reductions = [
        tuple(
            op
            for op in module.walk()
            if op.name == "linalg.generic" and any(it.data.value == "reduction" for it in op.iterator_types)
        )
        for module in (plan.source, plan.candidate)
    ]
    assert [static_zero_integer_matmul_geometry(op) for op in reductions[0]] == [(2, 5, 5), (2, 5, 3)]
    for before, after in zip(*reductions, strict=True):
        assert scalar_bodies(before) == scalar_bodies(after) and before.properties == after.properties
        assert tuple(v.type for v in before.operands) == tuple(v.type for v in after.operands)
    proof = plan.record()["conformance"]
    assert (proof["before_scalar_evaluations"], proof["after_scalar_evaluations"]) == (50, 70)
    assert (proof["before_producer_evaluations"], proof["after_producer_evaluations"]) == (10, 20)
    assert proof["logical_producer_bytes_removed"] == 10
    assert len(plan.source.body.block.first_op.function_type.outputs) == 9
    plan.revalidate()


def test_reference_preserves_all_input_bytes_and_returns_actual_state_path():
    state = initial_state()
    for epoch in range(3):
        arrays = inputs(epoch, state)
        before = [value.tobytes() for value in arrays]
        output = reference(*arrays)
        assert len(output) == 9 and sum(value.size for value in output) == 106
        assert before == [value.tobytes() for value in arrays]
        assert [value.tobytes() for value in output[3:]] == before
        assert output[0].dtype == output[1].dtype == output[2].dtype == np.int32
        for row in range(2):
            for col in range(5):
                assert int(output[0][row, col]) == signed(int(output[1][row, col]) + int(state[row, col]), 32)
        state = output[0].copy()
