"""Predetermined stimuli and independent, unfused integer/f32 boundaries."""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np

from merlin.targetgen.exact_quantized_reference import round_binary


def stimuli(cap, label, sibling=0):
    attrs = cap["operation"]["attributes"]
    case = attrs["case"]
    env = {}
    from .runtime_scale_reference import needs_runtime_scale, scale_stimulus

    scale_name = case.get("runtime_scale_input") if needs_runtime_scale(cap) else None
    for spec in cap["inputs"]:
        if spec["name"] == scale_name:
            env[scale_name] = scale_stimulus(label)
            continue
        count = int(np.prod(spec["shape"]))
        values = np.arange(count, dtype=np.int64)
        values = ((values * 7 + (0 if spec["name"] == "A" else sibling + 3)) % 16) - 8
        if label in {"rne_saturation", "sum_widening", "coefficient_extreme"}:
            palette = [-128, 127, -1, 1, 0, -3, 3, -5, 5]
            values = np.resize(palette, count)
        if spec["name"] == "Bias":
            palette = (
                [-(2**24) - 1, -(2**24), -(2**24) + 1, 2**24 - 1, 2**24, 2**24 + 1]
                if label == "conversion_boundary"
                else [-259, -257, -255, -5, -3, -1, 1, 3, 5, 253, 255, 257]
            )
            values = np.resize(palette, count)
            if "seed_range" in case:
                values = np.clip(values, *case["seed_range"])
        env[spec["name"]] = values.reshape(spec["shape"])
    if "constant_seed" in case:
        env["Bias"] = np.asarray(case["constant_seed"], dtype=np.int64)
    if label == "rne_saturation" and case["producer"] == "operand_sum":
        env["Right"] = np.roll(env["Right"], 1)
    if label == "conversion_boundary" and "Residual" in env:
        env["Residual"].fill(0)
    if label == "rne_saturation" and case.get("window_mean"):
        # A mean's divisor is its window, so construct half ties in the sum.
        image = env["Image"]
        image.fill(0)
        half = image.shape[-1] // 2
        image[..., 0] = np.resize([-half, half], image.shape[:-1])
    if (
        label in {"rne_saturation", "conversion_boundary"}
        and case["producer"] == "contraction"
        and not case.get("window_mean")
    ):
        # Seeded cases witness the requested accumulator exactly. Other cases
        # construct dot products on the byte grid, without inventing an i32 seed.
        if "Bias" in env and np.any(env["Bias"]):
            env["A"].fill(0)
        elif case.get("constant_weights") is None:
            env["A"].fill(0)
            env["A"][:, 0] = 1
            env["W0"].fill(0)
            env["W0"][0] = np.resize([-5, -3, -1, 1, 3, 5, 127, -128], attrs["N"])
            if attrs["K"] > 1:
                env["A"][:, 1] = 127
                env["W0"][1] = np.resize([0, 0, 0, 0, 0, 0, 127, -128], attrs["N"])
        else:
            env["A"].fill(0)
            env["A"][:, 0] = np.resize([-5, -3, -1, 1, 3, 5, 127, -128], attrs["M"])
            if attrs["K"] >= 3 and attrs["M"] >= 8:
                env["A"][6, :3] = 127
                env["A"][7, :3] = -128
    if label == "residual_rounding" and "coefficient" in case:
        # +/-823 and the opposite residual +/-64 distinguish source f32 ops
        # from an integer-domain extension, including after ReLU or max pooling.
        signs = np.resize([1, -1], attrs["M"])
        env["A"].fill(0)
        env["A"][:, :6] = -127 * signs[:, None]
        env["A"][:, 6] = -61 * signs
        if "W0" in env:
            env["W0"].fill(1)
        if "Bias" in env:
            env["Bias"].fill(0)
        env["Residual"][:] = 64 * signs[:, None]
    return env


@lru_cache(maxsize=65536)
def _source_float(value, multiplier, residual=None, coefficient=0, divisor=None):
    converted = Fraction(round_binary(value))
    if divisor is not None:
        return Fraction(round_binary(converted / Fraction(round_binary(divisor))))
    scale = Fraction(round_binary(Fraction(multiplier)))
    product = Fraction(round_binary(converted * scale))
    if residual is not None:
        residual_scale = Fraction(round_binary(Fraction(coefficient) * scale))
        dequantized = Fraction(round_binary(Fraction(round_binary(residual)) * residual_scale))
        product = Fraction(round_binary(product + dequantized))
    return product


@lru_cache(maxsize=65536)
def boundary(value, multiplier, strategy, activation):
    from .runtime_scale_reference import round_binary32

    value = max(value, 0) if activation else value
    scale = Fraction(round_binary(Fraction(multiplier)))
    if strategy == "f32_product_rne":
        converted = round_binary32(value)
        product = round_binary32(converted * scale)
    elif strategy == "exact_rational_host":
        product = value * scale
    else:
        raise ValueError("UNKNOWN: reference readout strategy")
    if product == float("inf"):
        return 127
    if product == float("-inf"):
        return -128
    return min(127, max(-128, round(product)))


def _readout(value, case, scheme, *, multipliers=None, activation=None, residual=None, divisor=None):
    gains = case["multipliers"] if multipliers is None else multipliers
    width = (
        scheme["columns_per_scale"]
        if multipliers is None and scheme["weight_scale_granularity"] == "column_block"
        else value.shape[1]
    )
    relu = "activation" in case["readout_stages"] if activation is None else activation
    if residual is not None or divisor is not None:
        result = np.empty(value.shape, dtype=object)
        for row, column in np.ndindex(value.shape):
            product = _source_float(
                int(value[row, column]),
                gains[min(column // width, len(gains) - 1)],
                int(residual[row, column]) if residual is not None else None,
                case.get("coefficient", 0),
                divisor,
            )
            result[row, column] = min(127, max(0 if relu else -128, round(product)))
        return result
    return np.array(
        [
            [
                boundary(int(v), gains[min(column // width, len(gains) - 1)], scheme["readout_strategy"], relu)
                for column, v in enumerate(row)
            ]
            for row in value
        ],
        dtype=object,
    )


def _producer(case, scheme, env):
    if case.get("window_mean"):
        return env["Image"].astype(object).sum(axis=-1)
    if case["producer"] == "movement":
        image = env["Image"]
        return image.astype(object).reshape(-1, image.shape[-1])
    if case["producer"] == "operand_sum":
        left, right = env["Left"].astype(object), env["Right"].astype(object)
        a, b = case["operand_multipliers"]
        if scheme["operand_sum_rounding"] == "per_operand":
            return _readout(left, case, scheme, multipliers=[a], activation=False) + _readout(
                right, case, scheme, multipliers=[b], activation=False
            )
        # The source explicitly rounds each f32 product and the addition.
        return np.array(
            [
                [
                    round(
                        round_binary(
                            Fraction(round_binary(int(x) * Fraction(a))) + Fraction(round_binary(int(y) * Fraction(b)))
                        )
                    )
                    for x, y in zip(lrow, rrow, strict=True)
                ]
                for lrow, rrow in zip(left, right, strict=True)
            ],
            dtype=object,
        )
    weights = (
        np.array(case["constant_weights"], dtype=object)
        if case.get("constant_weights") is not None
        else env["W0"].astype(object)
    )
    value = env["A"].astype(object) @ weights
    if "coefficient" in case and case["exit"] != "requant":
        value += env["Residual"].astype(object) * int(case["coefficient"])
    if "preload" in case["accumulator_stages"]:
        value += env["Bias"].astype(object)
    if any(not -(2**31) <= int(v) < 2**31 for v in value.flat):
        raise ValueError("reference accumulator overflow outside fixed profile")
    return value


def unfused_reference(cap, env):
    """Evaluate declared leaves, then independently apply any pooling readout."""
    import copy

    from merlin.targetgen.composition_golden import evaluate

    attrs = cap["operation"]["attributes"]
    case = attrs["case"]
    pooled = "pool" in case["readout_stages"]
    if pooled:
        cap = copy.deepcopy(cap)
        cap["operation"]["attributes"]["case"]["readout_stages"].remove("pool")
    inputs = {leaf["name"]: env[leaf["name"]] for leaf in cap["inputs"]}
    result = evaluate(cap, inputs=inputs)
    if pooled:
        return {name: _unfused_pool(value, case, attrs["N"]) for name, value in result.items()}
    return result


def _unfused_pool(value, case, channels):
    """Ignore out-of-image taps, independently of the explicit -inf padding oracle."""
    h, w = case["pool_in_dims"]
    kh, kw = case["pool_window"]
    sh, sw = case["pool_stride"]
    py, px = case["padding"]
    image = np.asarray(value).reshape(h, w, channels)
    windows = [
        image[max(0, y) : min(h, y + kh), max(0, x) : min(w, x + kw)].max(axis=(0, 1))
        for y in range(-py, h + py - kh + 1, sh)
        for x in range(-px, w + px - kw + 1, sw)
    ]
    return np.asarray(windows).reshape(_pool_output_shape(case, channels)).tolist()


def runtime_multiplier(cap, env):
    from .runtime_scale_reference import runtime_multiplier as read_scale

    return read_scale(cap, env)


def exact_reference(cap, env):
    from .runtime_scale_reference import exact_reference as evaluate

    return evaluate(cap, env, fallback=_legacy_exact_reference)


def _legacy_exact_reference(cap, env):
    attrs = cap["operation"]["attributes"]
    case, scheme = attrs["case"], attrs["quantization_scheme"]
    value = _producer(case, scheme, env)
    witnesses = dict(
        conversion_boundary=any(abs(int(v)) in {2**24 - 1, 2**24, 2**24 + 1} for v in value.flat),
        sum_widening=case["producer"] == "operand_sum" and any(abs(int(v)) > 127 for v in value.flat),
    )
    width = scheme["columns_per_scale"] if scheme["weight_scale_granularity"] == "column_block" else attrs["N"]
    residual = env["Residual"] if "coefficient" in case and case["exit"] == "requant" else None
    divisor = attrs["K"] if case.get("window_mean") else None
    products = [
        _source_float(
            int(v),
            case["multipliers"][min(c // width, len(case["multipliers"]) - 1)],
            int(residual[r, c]) if residual is not None else None,
            case.get("coefficient", 0),
            divisor,
        )
        for r, row in enumerate(value if case["exit"] == "requant" else [])
        for c, v in enumerate(row)
    ]
    if residual is not None:
        witnesses["residual_rounding_difference"] = any(
            round(product)
            != round(
                _source_float(
                    int(value[r, c]) + int(residual[r, c]) * int(case["coefficient"]),
                    case["multipliers"][min(c // width, len(case["multipliers"]) - 1)],
                )
            )
            for (r, c), product in zip(np.ndindex(value.shape), products, strict=True)
        )
    witnesses.update(
        rne_tie=any(p.denominator == 2 for p in products),
        positive_saturation=any(p > 127 for p in products),
        negative_saturation=any(p < -128 for p in products),
    )
    if case["exit"] == "requant":
        value = _readout(value, case, scheme, residual=residual, divisor=divisor)
    if "pool" in case["readout_stages"]:
        h, w = case["pool_in_dims"]
        wh, ww = case["pool_window"]
        sh, sw = case["pool_stride"]
        image = value.reshape(h, w, attrs["N"])
        ph, pw = case["padding"]
        image = np.pad(image.astype(object), ((ph, ph), (pw, pw), (0, 0)), constant_values=-np.inf)
        value = np.array(
            [
                image[y : y + wh, x : x + ww].max(axis=(0, 1))
                for y in range(0, h + 2 * ph - wh + 1, sh)
                for x in range(0, w + 2 * pw - ww + 1, sw)
            ],
            dtype=object,
        )
        value = value.reshape(_pool_output_shape(case, attrs["N"]))
    return {attrs["outputs"][0]: np.asarray(value, dtype=np.int64).tolist()}, witnesses


def _pool_output_shape(case, channels):
    h, w = case["pool_in_dims"]
    kh, kw = case["pool_window"]
    sh, sw = case["pool_stride"]
    py, px = case["padding"]
    return (1, (h + 2 * py - kh) // sh + 1, (w + 2 * px - kw) // sw + 1, channels)
