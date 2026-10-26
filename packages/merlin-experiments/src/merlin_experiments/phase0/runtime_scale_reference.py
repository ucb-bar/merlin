"""Independent binary32 readout oracle, using integer/Fraction arithmetic.

RTL: Configs.scala INToRecFN -> MulAddRecFN -> RecFNToIN (all RNE),
then AccumulatorScale.scala activation-before-scale and signed byte clipping.
No host floating operation rounds the exact oracle's conversion or product.
"""

from __future__ import annotations

import math
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache

import numpy as np

RUNTIME_LABELS = {
    "runtime_scale_zero": 0.0,
    "runtime_scale_quarter": 0.25,
    "runtime_scale_half": 0.5,
    "runtime_scale_nonpow2": 0.3,
}


def scale_stimulus(label):
    return np.array([RUNTIME_LABELS.get(label, 0.5 if label == "rne_saturation" else 0.25)], dtype=np.float32)


def needs_runtime_scale(cap):
    attrs = cap["operation"]["attributes"]
    case = attrs["case"]
    if "validated_graph" in attrs:
        return False
    exits = [s["exit"] for s in case.get("siblings", [case])]
    return attrs["quantization_scheme"].get("scale_source") == "runtime_input" and "requant" in exits


def runtime_multiplier(cap, env):
    """Read the actual f32[1] input; sampled case.multipliers is never authority."""
    if not needs_runtime_scale(cap):
        return None
    name = cap["operation"]["attributes"]["case"].get("runtime_scale_input")
    leaves = [leaf for leaf in cap["inputs"] if leaf["name"] == name]
    if (
        len(leaves) != 1
        or leaves[0].get("dtype") != "f32"
        or leaves[0].get("shape") != [1]
        or leaves[0].get("role") != "input"
    ):
        raise ValueError("runtime readout scale requires one finite f32 input")
    if name not in env:
        raise ValueError("runtime readout scale input is absent")
    value = np.asarray(env[name])
    if value.dtype != np.dtype(np.float32) or value.shape != (1,) or not np.isfinite(value).all():
        raise ValueError("runtime readout scale requires finite f32[1] storage")
    return value[0]


def scale_bits(scale):
    value = np.asarray(scale)
    if value.dtype != np.dtype(np.float32) or value.shape != () or not np.isfinite(value):
        raise ValueError("scale must be a finite np.float32 scalar")
    return int(value.view(np.uint32))


def _pow2(exponent):
    return Fraction(2) ** exponent


@lru_cache(maxsize=4096)
def bits_fraction(bits):
    if type(bits) is not int or not 0 <= bits < 2**32:
        raise ValueError("invalid binary32 bits")
    exponent, mantissa = (bits >> 23) & 255, bits & 0x7FFFFF
    if exponent == 255:
        raise ValueError("scale must be finite f32")
    magnitude = (
        Fraction(mantissa) * _pow2(-149) if exponent == 0 else Fraction((1 << 23) + mantissa) * _pow2(exponent - 150)
    )
    return -magnitude if bits >> 31 else magnitude


def round_binary32(value):
    """RNE to an exact binary32 rational, or signed infinity on overflow.

    The subnormal quantum is 2**-149. A carry at the top binade rounds to
    infinity, including the halfway value above the largest finite number.
    """
    value = Fraction(value)
    if not value:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < _pow2(exponent):
        exponent -= 1
    quantum = _pow2(max(-149, exponent - 23))
    rounded = round(magnitude / quantum) * quantum
    if rounded >= _pow2(128):
        return math.copysign(math.inf, sign)
    return sign * rounded


@lru_cache(maxsize=65536)
def product_fraction(value, bits, relu=False):
    if type(value) is not int or not -(2**31) <= value < 2**31:
        raise ValueError("accumulator must be signed i32")
    converted = round_binary32(max(value, 0) if relu else value)
    return round_binary32(converted * bits_fraction(bits))


def boundary(value, scale, relu=False):
    product = product_fraction(value, scale_bits(scale), relu)
    if isinstance(product, float) and math.isinf(product):
        return 127 if product > 0 else -128
    return min(127, max(-128, round(product)))


def apply_runtime_readout(values, scale, *, relu=False, exact=True):
    scale_bits(scale)
    values = np.asarray(values)
    if not exact:
        from merlin.runtime.exact_readout import f32_readout

        return f32_readout(values, scale, "i8", relu=relu)
    return np.array([boundary(int(v), scale, relu) for v in values.flat], dtype=np.int8).reshape(values.shape)


def _events(value, products):
    return dict(
        conversion_boundary=any(abs(int(v)) in {2**24 - 1, 2**24, 2**24 + 1} for v in value.flat),
        sum_widening=False,
        rne_tie=any(isinstance(p, Fraction) and p.denominator == 2 for p in products),
        positive_saturation=any(p > 127 for p in products),
        negative_saturation=any(p < -128 for p in products),
    )


def _combine_events(events):
    return {key: any(event.get(key, False) for event in events) for key in set().union(*events)}


def _graph_reference(cap, env, fallback):
    attrs = cap["operation"]["attributes"]
    outputs, events = {}, []
    for index, node in enumerate(attrs["validated_graph"]):
        local = {
            leaf["name"]: env[
                leaf["name"]
                if (attrs["validated_structure"] != "shared_input" or leaf["name"] == "A")
                else f"node{index}_{leaf['name']}"
            ]
            for leaf in node["inputs"]
        }
        values, witnesses = exact_reference(node, local, fallback=fallback)
        outputs.update({f"node{index}_{name}": value for name, value in values.items()})
        events.append(witnesses)
    result = {
        name: np.asarray(outputs[name]).reshape(shape).tolist()
        for name, shape in zip(attrs["outputs"], attrs["validated_output_shapes"], strict=True)
    }
    return result, _combine_events(events)


def exact_reference(cap, env, *, fallback):
    """Adapter for runtime contractions, typed siblings, and merged graphs.

    Other arithmetic remains delegated to the existing exact reference.
    Every result is kept in its own shape, with raw exits bypassing readout.
    """
    attrs = cap["operation"]["attributes"]
    if "validated_graph" in attrs:
        return _graph_reference(cap, env, fallback)
    case = attrs["case"]
    if case.get("siblings"):
        outputs, events = {}, []
        for index, (sibling, name) in enumerate(zip(case["siblings"], attrs["outputs"], strict=True)):
            local = deepcopy(cap)
            local_attrs = local["operation"]["attributes"]
            local_attrs["case"].pop("siblings")
            local_attrs["case"].update(sibling)
            local_attrs["outputs"] = [name]
            local_env = dict(env)
            if case.get("constant_weights") is None:
                local_env["W0"] = env[f"W{index}"]
                local_attrs["N"] = np.shape(local_env["W0"])[-1]
            values, witnesses = exact_reference(local, local_env, fallback=fallback)
            outputs.update(values)
            events.append(witnesses)
        return outputs, _combine_events(events)
    scale = runtime_multiplier(cap, env)
    if scale is None:
        return fallback(cap, env)
    from .fusion_reference import _producer

    value = _producer(case, attrs["quantization_scheme"], env)
    relu = "activation" in case["readout_stages"]
    bits = scale_bits(scale)
    products = [product_fraction(int(v), bits, relu) for v in value.flat]
    result = apply_runtime_readout(value, scale, relu=relu)
    return {attrs["outputs"][0]: result.tolist()}, _events(value, products)


def typed_result_dtype(cap, name):
    attrs = cap["operation"]["attributes"]
    case = attrs["case"]
    index = attrs["outputs"].index(name)
    exit_name = case.get("siblings", [case])[index]["exit"]
    return "i8" if exit_name == "requant" else "i32"


def oracle_probe_report(extension, actual, *, repaired=False):
    """Record a real probe's readback; never treat a model mismatch as parity."""
    import hashlib
    from pathlib import Path

    extension = Path(extension).resolve(strict=True)
    expected = boundary(-3, np.float32(-0.5), True)
    status = "parity" if actual == expected else "oracle_mismatch"
    if repaired and status != "parity":
        raise ValueError("repaired Spike still disagrees with pre-scale ReLU")
    return dict(
        extension=str(extension),
        extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
        accumulator=-3,
        scale_bits="bf000000",
        relu=True,
        expected=expected,
        post_scale_relu_witness=2,
        actual=actual,
        status=status,
    )


def effective_scheme(base, node):
    """Overlay this node's semantic regime without mutating global resolution."""
    return {**deepcopy(base), **deepcopy(node.get("regime", {}))}


def split_runtime_readout(cap, reader):
    """Retain original scalar SSA when replacing a closed readout by split ops."""
    from merlin.common import mlir_query as mq
    from merlin.targetgen.composition_capsules import _Source

    attrs = cap["operation"]["attributes"]
    case = attrs["case"]
    if "coefficient" in case or "pool" in case["readout_stages"]:
        raise ValueError("runtime readout v1 excludes residual and pooling")
    m, n = attrs["M"], attrs["N"]
    source = _Source()
    source.input("acc", [m, n], "i32")
    source.input("runtime_gain", [1], "f32")
    value = source.generic(
        "convert", ["acc"], [m, n], "f32", ["%f = arith.sitofp %x0 : i32 to f32", "linalg.yield %f : f32"]
    )
    if "activation" in case["readout_stages"]:
        value = source.generic(
            "activation",
            [value],
            [m, n],
            "f32",
            ["%z = arith.constant 0.0 : f32", "%v = arith.maximumf %x0, %z : f32", "linalg.yield %v : f32"],
        )
    value = source.generic(
        "scale",
        [value, "runtime_gain"],
        [m, n],
        "f32",
        ["%p = arith.mulf %x0, %x1 : f32", "linalg.yield %p : f32"],
        maps=["affine_map<(d0,d1)->(d0,d1)>", "affine_map<(d0,d1)->(0)>", "affine_map<(d0,d1)->(d0,d1)>"],
    )
    source.lines.extend(
        ["%qs = arith.constant dense<1.0> : tensor<f32>", "%qz = arith.constant dense<0> : tensor<i64>"]
    )
    source.types["readout"] = f"tensor<{m}x{n}xi8>"
    source.lines.append(f"""%readout = "quant_ext.quantize_per_tensor"(%{value}, %qs, %qz)
        <{{quant_min=-128:i64, quant_max=127:i64, output_dtype="int8"}}>
        {{prov.composition_stage="readout"}} :
        ({source.types[value]}, tensor<f32>, tensor<i64>) -> {source.types["readout"]}""")
    function = next(op for op in mq.parse(source.render(["readout"])).body.block.ops if op.name == "func.func")
    mapper = dict(zip(function.body.block.args, reader.operands[:2], strict=True))
    for op in list(function.body.block.ops)[:-1]:
        reader.parent.insert_op_before(op.clone(value_mapper=mapper), reader)
    result = mapper[list(function.body.block.ops)[-1].operands[0]]
    reader.results[0].replace_all_uses_with(result)
    reader.parent.erase_op(reader)


def runtime_scale_geometry(entries):
    """The partial runtime smoke includes unequal sibling channel tails."""
    if not any(e["generalization"]["quantization_scheme"].get("scale_source") == "runtime_input" for e in entries):
        return
    if not all(e["composition_instance"]["shape_class"] == "partial" for e in entries):
        return
    for index, entry in enumerate(entries):
        case = entry["composition_instance"]
        entry["N"] += index * 2
        case["output_columns"] = entry["N"]
        if case.get("constant_weights") is not None:
            case["constant_weights"] = [[1] * entry["N"] for _ in range(entry["K"])]


def validate_runtime_inputs(cap, env):
    """Check runtime leaf storage before a fast reference performs any casts."""
    attrs = cap["operation"]["attributes"]
    if "validated_graph" not in attrs:
        runtime_multiplier(cap, env)
        return
    for index, node in enumerate(attrs["validated_graph"]):
        local = {
            leaf["name"]: env[
                leaf["name"]
                if (attrs["validated_structure"] != "shared_input" or leaf["name"] == "A")
                else f"node{index}_{leaf['name']}"
            ]
            for leaf in node["inputs"]
        }
        validate_runtime_inputs(node, local)
