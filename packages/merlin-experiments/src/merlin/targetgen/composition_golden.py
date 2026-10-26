"""Independent array evaluation of synthetic composition sources."""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def _input_environment(cap, values):
    declared = {leaf["name"]: leaf for leaf in cap["inputs"]}
    if set(values) != set(declared):
        raise ValueError("composition input names differ from the capsule")
    env = {}
    for name, leaf in declared.items():
        value = np.asarray(values[name])
        if value.shape != tuple(leaf["shape"]):
            raise ValueError(f"composition input {name} shape differs from the capsule")
        dtype = leaf["dtype"]
        if dtype in ("i8", "i32"):
            bits = int(dtype[1:])
            if value.dtype.kind != "i" or np.any(value < -(1 << (bits - 1))) or np.any(value >= 1 << (bits - 1)):
                raise ValueError(f"composition input {name} requires {dtype} signed integer values in range")
            env[name] = value.astype(np.int64)
        elif dtype == "f32":
            if (
                value.dtype.kind != "f"
                or not np.isfinite(value).all()
                or np.any(np.abs(value) > np.finfo(np.float32).max)
            ):
                raise ValueError(f"composition input {name} requires finite f32 values")
            env[name] = value.astype(np.float32)
        else:
            raise ValueError(f"unsupported composition input dtype {dtype}")
    return env


def evaluate(cap, *, inputs=None):
    from merlin.llvmlower.requantization import f32
    from merlin.runtime.exact_readout import f32_readout
    from merlin.targetgen.capsule_golden import materialize_capsule_leaves

    attrs = cap["operation"]["attributes"]
    case, scheme = attrs["case"], attrs["quantization_scheme"]
    values = (
        inputs
        if inputs is not None
        else {key: tensor.to_list() for key, tensor in materialize_capsule_leaves(cap).items()}
    )
    from merlin_experiments.phase0.runtime_scale_reference import runtime_multiplier, validate_runtime_inputs

    # Validate typed storage before the legacy input normalizer can cast f64.
    if inputs is None:
        values = {
            leaf["name"]: np.asarray(values[leaf["name"]], dtype=np.float32)
            if leaf["dtype"] == "f32"
            else values[leaf["name"]]
            for leaf in cap["inputs"]
        }
    validate_runtime_inputs(cap, values)
    env = _input_environment(cap, values)
    if "validated_graph" in attrs:
        outputs = {}
        for index, node in enumerate(attrs["validated_graph"]):
            node_inputs = {
                leaf["name"]: env[
                    leaf["name"]
                    if attrs["validated_structure"] != "shared_input" or leaf["name"] == "A"
                    else f"node{index}_{leaf['name']}"
                ]
                for leaf in node["inputs"]
            }
            outputs.update({f"node{index}_{name}": value for name, value in evaluate(node, inputs=node_inputs).items()})
        return {
            name: np.asarray(outputs[name]).reshape(shape).tolist()
            for name, shape in zip(attrs["outputs"], attrs["validated_output_shapes"], strict=True)
        }
    if runtime_multiplier(cap, env) is not None:
        from merlin.targetgen.exact_quantized_reference import _reference_scheme

        _reference_scheme(None, scheme)
        name = case.get("runtime_scale_input")
        scales = [leaf for leaf in cap["inputs"] if leaf["name"] == name]
        if len(scales) != 1 or scales[0]["dtype"] != "f32" or scales[0]["shape"] != [1] or scales[0]["role"] != "input":
            raise ValueError("runtime readout scale requires one finite f32 input")
        if any(leaf["dtype"] not in ("i8", "i32") for leaf in cap["inputs"] if leaf["name"] != name):
            raise ValueError("composition data operands require signed integer storage")
    m, k, n = (attrs[key] for key in ("M", "K", "N"))

    def readout(value, multipliers=None, relu=None, residual=None):
        scales = multipliers or case["multipliers"]
        if multipliers is None and scheme.get("scale_source") == "runtime_input":
            scales = [runtime_multiplier(cap, env)]
        width = (
            scheme["columns_per_scale"]
            if scheme["weight_scale_granularity"] == "column_block" and multipliers is None
            else n
        )
        gains = np.array([scales[min(i // width, len(scales) - 1)] for i in range(value.shape[-1])])
        activation = "activation" in case["readout_stages"] if relu is None else relu
        if residual is not None or case.get("window_mean"):
            result = np.empty(value.shape, dtype=np.int64)
            for row, column in np.ndindex(value.shape):
                converted = f32(int(value[row, column]))
                if case.get("window_mean"):
                    scaled = f32(converted / f32(k))
                else:
                    gain = f32(gains[column])
                    scaled = f32(converted * gain)
                    residual_scale = f32(float(case["coefficient"]) * gain)
                    dequantized = f32(f32(int(residual[row, column])) * residual_scale)
                    scaled = f32(scaled + dequantized)
                result[row, column] = min(127, max(0 if activation else -128, round(scaled)))
            return result
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
            # The source rounds each operand product to f32, then their f32 sum, then RNE.
            left_scale, right_scale = f32(float(scales[0])), f32(float(scales[1]))
            value = np.array(
                [
                    [
                        round(f32(f32(f32(int(a)) * left_scale) + f32(f32(int(b)) * right_scale)))
                        for a, b in zip(left, right, strict=True)
                    ]
                    for left, right in zip(env["Left"], env["Right"], strict=True)
                ],
                dtype=np.int64,
            )
    elif case.get("window_mean"):
        from merlin.compile.composition_fusion import require_exact_mean_window

        require_exact_mean_window(k)
        value = env["Image"].sum(axis=-1, dtype=np.int64)
    elif producer == "movement":
        value = env["Image"].reshape(m, n)
    else:
        outputs = []
        for index, sibling in enumerate(case.get("siblings", [dict(exit=case["exit"])])):
            weights = (
                np.array(case["constant_weights"], dtype=np.int64)
                if case.get("constant_weights") is not None
                else env[f"W{index}"]
            )
            value = env["A"] @ weights
            if "coefficient" in case and case["exit"] != "requant":
                coefficient = Fraction(case["coefficient"])
                value += np.array(
                    [[round(int(v) * coefficient) for v in row] for row in env["Residual"]], dtype=np.int64
                )
            if "preload" in case["accumulator_stages"]:
                value += np.asarray(case["constant_seed"], dtype=np.int64) if "constant_seed" in case else env["Bias"]
            if case.get("siblings"):
                outputs.append(readout(value, residual=env.get("Residual")) if sibling["exit"] == "requant" else value)
        if outputs:
            return {name: value.tolist() for name, value in zip(attrs["outputs"], outputs, strict=True)}
    if case["exit"] == "requant":
        value = readout(value, residual=env.get("Residual"))
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
