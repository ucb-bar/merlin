"""Independent array evaluation of synthetic composition sources."""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def evaluate(cap):
    from merlin.runtime.exact_readout import f32_readout
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves

    attrs = cap["operation"]["attributes"]
    case, scheme = attrs["case"], attrs["quantization_scheme"]
    env = {key: np.asarray(tensor.to_list(), dtype=np.int64) for key, tensor in materialize_capsule_leaves(cap).items()}
    m, k, n = (attrs[key] for key in ("M", "K", "N"))

    def readout(value, multipliers=None, relu=None):
        scales = multipliers or case["multipliers"]
        width = (
            scheme["columns_per_scale"]
            if scheme["weight_scale_granularity"] == "column_block" and multipliers is None
            else n
        )
        gains = np.array([scales[min(i // width, len(scales) - 1)] for i in range(n)])
        activation = "activation" in case["readout_stages"] if relu is None else relu
        if scheme["readout_strategy"] == "f32_product_rne":
            return f32_readout(value, gains, out_dtype="i8", relu=activation)
        if activation:
            value = np.maximum(value, 0)
        return np.array(
            [
                [min(127, max(-128, round(int(v) * Fraction(float(gains[i]))))) for i, v in enumerate(row)]
                for row in value
            ],
            dtype=np.int64,
        )

    producer = case["producer"]
    if producer == "operand_sum":
        scales = case["operand_multipliers"]
        if scheme["operand_sum_rounding"] == "per_operand":
            # Each load saturates to a byte; the accumulator adds them at full width.
            value = readout(env["Left"], [scales[0]], False).astype(np.int64) + readout(
                env["Right"], [scales[1]], False
            ).astype(np.int64)
        else:
            value = np.array(
                [
                    [
                        round(int(a) * Fraction(scales[0]) + int(b) * Fraction(scales[1]))
                        for a, b in zip(left, right, strict=True)
                    ]
                    for left, right in zip(env["Left"], env["Right"], strict=True)
                ],
                dtype=np.int64,
            )
    elif producer == "movement":
        value = env["Image"]
    else:
        outputs = []
        for index, sibling in enumerate(case.get("siblings", [dict(exit=case["exit"])])):
            weights = (
                np.array(case["constant_weights"], dtype=np.int64)
                if case.get("constant_weights") is not None
                else env[f"W{index}"]
            )
            value = env["A"] @ weights
            if "coefficient" in case:
                value += np.rint(env["Residual"] * case["coefficient"]).astype(np.int64)
            if "preload" in case["accumulator_stages"]:
                value += env["Bias"]
            if case.get("siblings"):
                outputs.append(readout(value) if sibling["exit"] == "requant" else value)
        if outputs:
            return {name: value.tolist() for name, value in zip(attrs["outputs"], outputs, strict=True)}
    if case["exit"] == "requant":
        value = readout(value)
    if "pool" in case["readout_stages"]:
        h, w = case["pool_in_dims"]
        wh, ww = case["pool_window"]
        sh, sw = case["pool_stride"]
        image = value.reshape(h, w, n)
        ph, pw = case["padding"]
        if ph or pw:
            image = np.pad(image, ((ph, ph), (pw, pw), (0, 0)), constant_values=0)
            h, w = h + 2 * ph, w + 2 * pw
        value = np.array(
            [
                image[y : y + wh, x : x + ww].max(axis=(0, 1))
                for y in range(0, h - wh + 1, sh)
                for x in range(0, w - ww + 1, sw)
            ]
        )
    return {attrs["outputs"][0]: value.tolist()}
