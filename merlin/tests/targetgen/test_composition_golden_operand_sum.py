"""The composition reference widens per-operand loads before their accumulator sum."""

import numpy as np

from merlin.runtime.exact_readout import f32_readout
from merlin.targetgen.capsule_golden import materialize_capsule_leaves
from merlin.targetgen.composition_golden import evaluate


def test_per_operand_sum_does_not_wrap_at_the_operand_width():
    m, n = 4, 8
    case = dict(producer="operand_sum", operand_multipliers=[1.0, 1.0], multipliers=[0.25], exit="requant",
                accumulator_stages=[], readout_stages=["convert", "scale", "requant"])
    scheme = dict(weight_scale_granularity="tensor", readout_strategy="f32_product_rne",
                  operand_sum_rounding="per_operand")
    capsule = dict(
        inputs=[dict(name="Left", shape=[m, n], dtype="i8", role="input"),
                dict(name="Right", shape=[m, n], dtype="i8", role="input")],
        stimulus_range=[100, 127],
        operation=dict(attributes=dict(case=case, quantization_scheme=scheme, M=m, K=m, N=n, outputs=["readout"])),
    )
    leaves = materialize_capsule_leaves(capsule)
    left, right = (np.asarray(leaves[name].to_list(), np.int64) for name in ("Left", "Right"))
    assert np.all(left + right > 127)  # A byte-width sum would wrap negative.
    expected = f32_readout(left + right, np.float32(0.25), "i8")
    np.testing.assert_array_equal(np.asarray(evaluate(capsule)["readout"]), expected)
