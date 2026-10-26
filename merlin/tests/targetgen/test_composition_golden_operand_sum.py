"""The composition reference widens per-operand loads before their accumulator sum."""

import numpy as np

from merlin.runtime.exact_readout import f32_readout
from merlin.targetgen.capsule_golden import materialize_capsule_leaves
from merlin.targetgen.composition_golden import evaluate


def test_per_operand_sum_does_not_wrap_at_the_operand_width():
    m, n = 4, 8
    case = dict(
        producer="operand_sum",
        operand_multipliers=[1.0, 1.0],
        multipliers=[0.25],
        exit="requant",
        accumulator_stages=[],
        readout_stages=["convert", "scale", "requant"],
    )
    scheme = dict(
        weight_scale_granularity="tensor", readout_strategy="f32_product_rne", operand_sum_rounding="per_operand"
    )
    capsule = dict(
        inputs=[
            dict(name="Left", shape=[m, n], dtype="i8", role="input"),
            dict(name="Right", shape=[m, n], dtype="i8", role="input"),
        ],
        stimulus_range=[100, 127],
        operation=dict(attributes=dict(case=case, quantization_scheme=scheme, M=m, K=m, N=n, outputs=["readout"])),
    )
    leaves = materialize_capsule_leaves(capsule)
    left, right = (np.asarray(leaves[name].to_list(), np.int64) for name in ("Left", "Right"))
    assert np.all(left + right > 127)  # A byte-width sum would wrap negative.
    expected = f32_readout(left + right, np.float32(0.25), "i8")
    np.testing.assert_array_equal(np.asarray(evaluate(capsule)["readout"]), expected)


def _single_rounding_capsule(left, right, scales):
    case = dict(
        producer="operand_sum",
        operand_multipliers=scales,
        multipliers=[1.0],
        exit="accumulator",
        accumulator_stages=[],
        readout_stages=[],
    )
    scheme = dict(
        scale_value="arbitrary_f32",
        weight_scale_granularity="tensor",
        readout_strategy="f32_product_rne",
        operand_sum_rounding="single_rounding",
    )
    shape = list(np.shape(left))
    capsule = dict(
        inputs=[dict(name=name, shape=shape, dtype="i8", role="input") for name in ("Left", "Right")],
        operation=dict(
            attributes=dict(case=case, quantization_scheme=scheme, M=shape[0], K=1, N=shape[1], outputs=["Y"])
        ),
    )
    return capsule, dict(Left=np.asarray(left, np.int64), Right=np.asarray(right, np.int64))


def _source_order(left, right, scales):
    a, b = (np.float32(scale) for scale in scales)
    return np.rint(np.float32(np.float32(left) * a) + np.float32(np.float32(right) * b)).astype(np.int64)


def test_single_rounding_sum_follows_each_source_f32_operation():
    from merlin.targetgen.composition_corpus import operand_sum_rounding_witnesses

    scales = [float(np.float32(1 / 6)), float(np.float32(1 / 10))]
    # Frozen review counterexample: one rational rounding gives -33, the source -32.
    capsule, env = _single_rounding_capsule([[-120]], [[-125]], scales)
    assert evaluate(capsule, inputs=env)["Y"] == [[-32]]
    witnesses = operand_sum_rounding_witnesses(tuple(scales), 8)
    assert (-120, -125) in witnesses and len(witnesses) == 1080
    left, right = (np.array([[pair[side] for pair in witnesses]]) for side in (0, 1))
    capsule, env = _single_rounding_capsule(left, right, scales)
    np.testing.assert_array_equal(np.asarray(evaluate(capsule, inputs=env)["Y"]), _source_order(left, right, scales))
    assert operand_sum_rounding_witnesses((1.0, 1.0), 8) == ()
