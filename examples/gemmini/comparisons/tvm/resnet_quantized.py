"""Explicit development ResNet50 integer recipe and semantic Relax export.

This is a proposed recipe, not an approved application-quality benchmark. All
contractions use signed symmetric int8 operands and bounded int32 accumulation.
Scales are powers of two; every rounding uses nearest with ties away from zero.
The separate reference executes exact integer-valued float64 contractions (all
products and partial sums are proven below 2**31), then integer host policies.
No accelerated contraction or exported graph is used to obtain its answers.
"""

from dataclasses import dataclass
import hashlib
import math
import operator

import numpy as np


LIMIT = np.iinfo(np.int32).max


def array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(array_digest(value.detach().cpu().numpy()).encode())
    return digest.hexdigest()


def exponent_for(maximum):
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError("Calibration requires finite nonnegative maxima")
    exponent = math.ceil(math.log2(maximum / 127)) if maximum else -24
    if not -30 <= exponent <= 30:
        raise ValueError("Scale exponent lies outside the declared [-30,30] range")
    return exponent


def quantize(value, exponent):
    value = np.asarray(value)
    if not np.isfinite(value).all():
        raise ValueError("Cannot quantize nonfinite values")
    scaled = value.astype(np.float64) / np.exp2(exponent)
    rounded = np.copysign(np.floor(np.abs(scaled) + 0.5), scaled)
    return np.clip(rounded, -127, 127).astype(np.int8)


def activation_scale_errors(value, exponents, *, sample_limit=8192):
    """Bounded, deterministic TRAIN reconstruction errors for candidate scales.

    Uniform flattened indices cover channels and spatial positions. Each image
    contributes at most sample_limit values per site; no activation archive or
    evaluation tensor participates in selection. Saturation is part of the
    error, and the coarse max-abs candidate wins exact ties.
    """
    value = np.asarray(value)
    if value.dtype != np.float32 or not value.size or not np.isfinite(value).all():
        raise ValueError("Activation scale estimation requires finite nonempty FP32 values")
    if type(sample_limit) is not int or not 1 <= sample_limit <= 8192:
        raise ValueError("Invalid activation sampling bound")
    exponents = tuple(exponents)
    if not exponents or any(type(e) is not int or not -30 <= e <= 30 for e in exponents):
        raise ValueError("Invalid candidate activation exponents")
    count = min(value.size, sample_limit)
    indices = np.arange(count, dtype=np.int64) * value.size // count
    sampled = value.reshape(-1)[indices].astype(np.float64)
    errors = [float(np.square(quantize(sampled, exponent).astype(np.float64) * math.ldexp(1.0, exponent) - sampled).sum())
              for exponent in exponents]
    return count, errors


def quantize_parameters(raw_weight, raw_bias, input_exponent):
    """Select the finest max-abs-or-coarser channel scale safe with its bias.

    Increasing a power-of-two exponent cannot increase either rounded magnitude.
    Check the full admitted [-127,127] input range and every contraction partial
    sum, before converting the rounded bias to int32. Never saturate the bias.
    """
    raw_weight, raw_bias = np.asarray(raw_weight), np.asarray(raw_bias)
    if (raw_weight.ndim not in (2, 4) or min(raw_weight.shape) < 1
            or raw_bias.shape != (raw_weight.shape[0],)
            or not np.isfinite(raw_weight).all() or not np.isfinite(raw_bias).all()
            or type(input_exponent) is not int or not -30 <= input_exponent <= 30):
        raise ValueError("Invalid finite contraction coefficients or input scale")
    if math.prod(raw_weight.shape[1:]) * 128 * 128 > LIMIT:
        raise ValueError("Unsafe int32 contraction dimension bound")
    exponents = np.array([exponent_for(float(abs(row).max())) for row in raw_weight], dtype=np.int64)
    weight, bias, adjustments = np.empty_like(raw_weight, dtype=np.int8), np.empty(raw_bias.shape, dtype=np.int32), []
    for channel, row in enumerate(raw_weight):
        initial_exponent = exponent = int(exponents[channel])
        initial = None
        while True:
            quantized = quantize(row, exponent)
            bound = int(127 * np.abs(quantized.astype(np.int64)).sum())
            scaled_bias = float(raw_bias[channel]) / math.ldexp(1.0, input_exponent + exponent)
            rounded_bias = math.copysign(math.floor(abs(scaled_bias) + 0.5), scaled_bias) if math.isfinite(scaled_bias) else scaled_bias
            combined = bound + abs(rounded_bias)
            if initial is None:
                initial = {"initial_exponent": exponent, "initial_accumulation_bound": bound,
                           "initial_bias": rounded_bias, "initial_bound_with_bias": combined}
            if math.isfinite(rounded_bias) and combined <= LIMIT:
                break
            if exponent == 30:
                raise ValueError("Unsafe int32 contraction/bias bound at maximum channel exponent")
            exponent += 1
        weight[channel], bias[channel], exponents[channel] = quantized, int(rounded_bias), exponent
        if exponent != initial_exponent:
            adjustments.append({"channel": channel, **initial, "selected_exponent": exponent,
                                "raw_weight_sha256": array_digest(row), "raw_weight_maximum": float(abs(row).max()),
                                "raw_bias": float(raw_bias[channel]), "accumulation_bound": bound,
                                "bias": int(rounded_bias), "bound_with_bias": int(combined)})
    return weight, exponents, bias, adjustments


def round_divide(value, denominator):
    """Exact signed nearest division, including negative ties."""
    value = np.asarray(value)
    denominator = np.asarray(denominator)
    if denominator.dtype.kind not in "iu":
        raise ValueError("Division denominators must be integers")
    denominator = denominator.astype(np.int64)
    if value.dtype != np.int64 or np.any(denominator <= 0):
        raise ValueError("Division requires int64 values and positive denominators")
    if np.any(value == np.iinfo(np.int64).min):
        raise ValueError("Cannot take int64 minimum magnitude")
    magnitude = np.abs(value)
    if np.any(magnitude > np.iinfo(np.int64).max - denominator // 2):
        raise ValueError("Rounded numerator overflows int64")
    rounded = (magnitude + denominator // 2) // denominator
    return np.where(value < 0, -rounded, rounded)


def rescale(value, exponent):
    """Multiply by 2**exponent with integer rounding; does not saturate."""
    exponents = np.asarray(exponent)
    if exponents.dtype.kind != "i":
        raise ValueError("Rescaling requires integer exponents")
    exponents = exponents.astype(np.int64)
    if np.any(exponents < -30) or np.any(exponents > 30):
        raise ValueError("Requantization shift exceeds the declared 30-bit limit")
    factor = np.left_shift(np.int64(1), np.maximum(exponents, 0))
    divisor = np.left_shift(np.int64(1), np.maximum(-exponents, 0))
    value = np.asarray(value)
    if value.dtype.kind != "i":
        raise ValueError("Rescaling requires signed integer values")
    value = value.astype(np.int64)
    if np.any(value == np.iinfo(np.int64).min):
        raise ValueError("Cannot take int64 minimum magnitude")
    if np.any(np.abs(value) > (np.iinfo(np.int64).max - divisor // 2) // factor):
        raise ValueError("Requantization intermediate overflows int64")
    return round_divide(value * factor, divisor)


def saturate(value):
    return np.clip(value, -127, 127).astype(np.int8)


def residual_add_factors(left_exponent, right_exponent, output_exponent, left_bound, right_bound):
    """Prove exact alignment, addition and one rounding fit signed int64.

    Bounds are absolute per-channel bounds, including contraction biases. Align
    both arms and the destination at their finest exponent before one division;
    never round or saturate an individual arm. Python integers establish the
    bounds before any int64 multiplication. All shifts remain at most 30 bits.
    """
    values = [np.asarray(value) for value in (left_exponent, right_exponent, output_exponent, left_bound, right_bound)]
    if any(value.dtype.kind != "i" for value in values):
        raise ValueError("Residual exponents and bounds must be signed integers")
    values = np.broadcast_arrays(*values)
    left, right, output, lb, rb = values
    if (np.any((left < -60) | (left > 60)) or np.any((right < -60) | (right > 60)) or np.any((output < -30) | (output > 30))
            or np.any(lb < 0) or np.any(rb < 0) or np.any(lb > LIMIT) or np.any(rb > LIMIT)):
        raise ValueError("Invalid residual scale or int32 operand bound")
    common = np.minimum(np.minimum(left, right), output)
    shifts = (left - common, right - common, output - common)
    if any(np.any(shift > 30) for shift in shifts):
        raise ValueError("Residual alignment or rounding shift exceeds 30 bits")
    factors = [np.left_shift(np.int64(1), shift.astype(np.int64)) for shift in shifts]
    rounded_bounds = []
    for a, b, denominator, x, y in zip(*(value.flat for value in (*factors, lb, rb))):
        bound = int(x) * int(a) + int(y) * int(b) + int(denominator) // 2
        if bound > np.iinfo(np.int64).max:
            raise ValueError("Residual aligned sum and rounding offset overflow int64")
        rounded_bounds.append(bound)
    return (*factors, np.asarray(rounded_bounds, dtype=np.int64).reshape(left.shape))


def residual_add(left, right, left_exponent, right_exponent, output_exponent, left_bound, right_bound):
    """Independent joint residual rounding with admitted integer arm bounds."""
    a, b, denominator, _ = residual_add_factors(left_exponent, right_exponent, output_exponent, left_bound, right_bound)
    left, right = np.asarray(left), np.asarray(right)
    if (left.dtype not in (np.dtype("int8"), np.dtype("int32")) or right.dtype not in (np.dtype("int8"), np.dtype("int32"))
            or left.shape != right.shape or np.any(np.abs(left.astype(np.int64)) > left_bound)
            or np.any(np.abs(right.astype(np.int64)) > right_bound)):
        raise ValueError("Residual operands exceed their admitted integer bounds")
    return saturate(round_divide(left.astype(np.int64) * a + right.astype(np.int64) * b, denominator))


def averagepool_scale_factors(area, input_exponent, output_exponent):
    """Prove a single rounded division of the scaled spatial sum fits int64."""
    if type(area) is not int or area < 1 or any(type(exponent) is not int or not -30 <= exponent <= 30 for exponent in (input_exponent, output_exponent)):
        raise ValueError("Invalid averagepool area or scale exponent")
    shift = input_exponent - output_exponent
    if not -30 <= shift <= 30:
        raise ValueError("Averagepool requantization shift exceeds 30 bits")
    factor, denominator = 1 << max(shift, 0), area * (1 << max(-shift, 0))
    limit64 = np.iinfo(np.int64).max
    if 128 * area > limit64 or denominator > limit64 or 127 * area * factor + denominator // 2 > limit64:
        raise ValueError("Averagepool sum, scaled numerator or rounded division may overflow int64")
    return factor, denominator


def averagepool(data, input_exponent, output_exponent):
    """Requantize the spatial mean once, preserving fractional mean precision."""
    if not isinstance(data, np.ndarray) or data.dtype != np.int8 or data.ndim != 4 or min(data.shape) < 1 or np.any(data == -128):
        raise ValueError("Averagepool requires nonempty symmetric int8 NCHW input")
    factor, denominator = averagepool_scale_factors(math.prod(data.shape[2:]), input_exponent, output_exponent)
    total = data.astype(np.int64).sum(axis=(2, 3), keepdims=True)
    return saturate(round_divide(total * factor, denominator))


@dataclass
class Site:
    name: str
    kind: str
    inputs: tuple
    shape: tuple
    exponent: int | None
    attrs: dict
    weight: np.ndarray | None = None
    weight_exponents: np.ndarray | None = None
    bias: np.ndarray | None = None
    retain_accumulator: bool = False


def residual_accumulator_targets(sites):
    """Select Conv -> Identity* -> Add -> Identity* -> ReLU by sole users."""
    users = {site.name: [] for site in sites}
    for site in sites:
        for name in site.inputs:
            if name in users:
                users[name].append(site)

    def terminal(name, kind):
        identities = []
        while len(users[name]) == 1:
            user = users[name][0]
            if user.kind == kind:
                return user, identities
            if user.kind != "identity" or user.inputs != (name,):
                break
            identities.append(user.name)
            name = user.name
        return None, []

    targets = {}
    for site in sites:
        if site.kind != "conv2d":
            continue
        add, identities = terminal(site.name, "add")
        if add is not None and len(add.inputs) == 2:
            relu, after_add = terminal(add.name, "relu")
            if relu is not None:
                targets[site.name] = {"add": add.name, "identities": identities, "relu": relu.name, "add_identities": after_add}
    return targets


def residual_storage(plan):
    """Derive actual carrier scales/bounds; declared Conv scales stay frozen."""
    storage = {plan.provenance["input_name"]: (plan.input_exponent, 127)}
    for site in plan.sites:
        if site.kind == "conv2d" and site.retain_accumulator:
            exponent = storage[site.inputs[0]][0] + site.weight_exponents
            bound = 127 * np.abs(site.weight.astype(np.int64)).reshape(site.weight.shape[0], -1).sum(axis=1) + np.abs(site.bias.astype(np.int64))
            storage[site.name] = (exponent, bound)
        elif site.kind == "identity":
            storage[site.name] = storage[site.inputs[0]]
        else:
            storage[site.name] = (site.exponent, 127)
    return storage


def residual_parameters(site, storage):
    """Broadcast scalar or per-channel arm metadata over NCHW."""
    if len(site.shape) != 4:
        raise ValueError("Accumulator residual policy requires NCHW residuals")
    def channel(value):
        value = np.asarray(value, dtype=np.int64)
        return value.reshape(1, -1, 1, 1) if value.ndim == 1 else value
    (left, lb), (right, rb) = (storage[name] for name in site.inputs)
    return channel(left), channel(right), site.exponent, channel(lb), channel(rb)


@dataclass
class QuantizedResNet:
    sites: list
    input_shape: tuple
    input_exponent: int
    calibration_hashes: tuple
    provenance: dict

    def validate(self):
        """Recheck numeric bounds when a caller hands a plan to an executor."""
        input_name = self.provenance.get("input_name")
        if not isinstance(input_name, str) or not input_name:
            raise ValueError("Invalid plan input name")
        def shape(value):
            if not isinstance(value, (tuple, list)) or not value or any(type(dim) is not int or dim < 1 for dim in value):
                raise ValueError("Plan shapes require positive static integer extents")
            return tuple(value)
        input_shape = shape(self.input_shape)
        if len(input_shape) != 4 or input_shape[:2] != (1, 3) or min(input_shape[2:]) < 32:
            raise ValueError("Invalid ResNet50 input geometry")
        ids = self.provenance.get("calibration_ids", ())
        if (not self.calibration_hashes or len(ids) != len(self.calibration_hashes)
                or any(not isinstance(value, str) or not value.strip() for value in ids)
                or len(set(ids)) != len(ids) or len(set(self.calibration_hashes)) != len(self.calibration_hashes)
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value) for value in self.calibration_hashes)):
            raise ValueError("Invalid frozen calibration provenance")
        scales, shapes = {input_name: self.input_exponent}, {input_name: input_shape}
        counts = {"conv2d": 0, "linear": 0, "add": 0}
        if type(self.input_exponent) is not int or not -30 <= self.input_exponent <= 30:
            raise ValueError("Invalid input scale exponent")
        kinds = {"conv2d", "linear", "add", "relu", "identity", "flatten", "averagepool", "maxpool"}
        def pair(attrs, name, *, minimum):
            value = attrs.get(name)
            if not isinstance(value, (tuple, list)) or len(value) != 2 or any(type(item) is not int or item < minimum for item in value):
                raise ValueError("Invalid " + name + " geometry")
            return tuple(value)
        for site in self.sites:
            if not isinstance(site.name, str) or not site.name or site.name in scales or not site.inputs or any(name not in scales for name in site.inputs):
                raise ValueError("Plan must be an ordered graph with unique site names")
            if site.kind not in kinds or len(site.inputs) != (2 if site.kind == "add" else 1):
                raise ValueError("Invalid plan operation or input arity")
            if not isinstance(site.attrs, dict):
                raise ValueError("Plan operation attributes must be a dictionary")
            output_shape, source_shape = shape(site.shape), shapes[site.inputs[0]]
            if site.kind == "linear":
                if site.exponent is not None:
                    raise ValueError("Classifier logits must not have an activation exponent")
            elif type(site.exponent) is not int or not -30 <= site.exponent <= 30:
                raise ValueError("Invalid activation scale exponent")
            if any(scales[name] is None for name in site.inputs):
                raise ValueError("Classifier must be the terminal graph operation")
            if site.kind in counts:
                counts[site.kind] += 1
            if site.kind in ("conv2d", "linear"):
                if (not all(isinstance(value, np.ndarray) for value in (site.weight, site.bias, site.weight_exponents))
                        or site.weight.dtype != np.int8 or np.any(site.weight == -128)
                        or site.bias.dtype != np.int32 or site.weight_exponents.dtype != np.int64
                        or site.weight.ndim != (4 if site.kind == "conv2d" else 2) or min(site.weight.shape) < 1):
                    raise ValueError("Invalid contraction parameter dtype or symmetric range")
                channels = site.weight.shape[0]
                if (site.bias.shape != (channels,) or site.weight_exponents.shape != (channels,)
                        or np.any(site.weight_exponents < -30) or np.any(site.weight_exponents > 30)):
                    raise ValueError("Invalid per-channel parameters")
                if math.prod(site.weight.shape[1:]) * 128 * 128 > LIMIT:
                    raise ValueError("Unsafe contraction or bias bound")
                bound = 127 * np.abs(site.weight.astype(np.int64)).reshape(channels, -1).sum(axis=1)
                if np.any(bound + np.abs(site.bias.astype(np.int64)) > LIMIT):
                    raise ValueError("Unsafe contraction or bias bound")
                if site.kind == "conv2d":
                    if len(source_shape) != 4 or source_shape[1] != site.weight.shape[1] or set(site.attrs) != {"stride", "padding", "dilation"}:
                        raise ValueError("Incompatible convolution input channels or attributes")
                    stride = pair(site.attrs, "stride", minimum=1)
                    padding = pair(site.attrs, "padding", minimum=0)
                    if pair(site.attrs, "dilation", minimum=1) != (1, 1):
                        raise ValueError("Unsupported convolution dilation")
                    expected_shape = (source_shape[0], channels, *(
                        (extent + 2 * pad - kernel) // step + 1 for extent, pad, kernel, step
                        in zip(source_shape[2:], padding, site.weight.shape[2:], stride)))
                    shifts = scales[site.inputs[0]] + site.weight_exponents - site.exponent
                    if np.any(shifts < -30) or np.any(shifts > 30):
                        raise ValueError("Unsafe convolution rescale")
                else:
                    if source_shape != (1, 2048) or site.weight.shape != (1000, 2048) or site.attrs:
                        raise ValueError("Expected the standard 2048-input 1000-class classifier")
                    expected_shape = (1, 1000)
            elif site.kind == "add":
                if any(shapes[name] != source_shape for name in site.inputs) or site.attrs:
                    raise ValueError("Residual broadcasting or attributes are not admitted")
                if any(not -30 <= scales[name] - site.exponent <= 30 for name in site.inputs):
                    raise ValueError("Invalid residual rescale")
                expected_shape = source_shape
            else:
                if site.kind != "averagepool" and site.exponent != scales[site.inputs[0]]:
                    raise ValueError("Unary integer operations must preserve the input scale")
                if site.kind != "maxpool" and site.attrs:
                    raise ValueError("Unsupported unary operation attributes")
                if site.kind in ("relu", "identity"):
                    expected_shape = source_shape
                elif site.kind == "flatten":
                    expected_shape = (source_shape[0], math.prod(source_shape[1:]))
                elif site.kind in ("averagepool", "maxpool"):
                    if len(source_shape) != 4:
                        raise ValueError("Pooling requires NCHW input")
                    if site.kind == "averagepool":
                        averagepool_scale_factors(math.prod(source_shape[2:]), scales[site.inputs[0]], site.exponent)
                        expected_shape = (*source_shape[:2], 1, 1)
                    else:
                        if set(site.attrs) != {"kernel", "stride", "padding"}:
                            raise ValueError("Unsupported maxpool attributes")
                        kernel = pair(site.attrs, "kernel", minimum=1)
                        stride = pair(site.attrs, "stride", minimum=1)
                        padding = pair(site.attrs, "padding", minimum=0)
                        if any(pad > size // 2 for pad, size in zip(padding, kernel)):
                            raise ValueError("Unsupported maxpool padding")
                        expected_shape = (*source_shape[:2], *(
                            (extent + 2 * pad - size) // step + 1
                            for extent, pad, size, step in zip(source_shape[2:], padding, kernel, stride)))
            if output_shape != expected_shape:
                raise ValueError("Plan output shape differs from operation geometry")
            scales[site.name], shapes[site.name] = site.exponent, output_shape
        if counts != {"conv2d": 53, "linear": 1, "add": 16} or self.sites[-1].kind != "linear":
            raise ValueError("Incomplete ResNet50 graph")
        policy = self.provenance.get("residual_policy", "legacy")
        if policy not in ("legacy", "accumulator"):
            raise ValueError("Invalid residual policy")
        targets = residual_accumulator_targets(self.sites)
        retained = set()
        for site in self.sites:
            if type(site.retain_accumulator) is not bool:
                raise ValueError("Invalid accumulator retention flag")
            if site.retain_accumulator:
                if policy != "accumulator" or site.kind != "conv2d" or site.name not in targets:
                    raise ValueError("Accumulator retention requires a unique Identity chain to Add/ReLU")
                retained.add(site.name)
                retained.update(targets[site.name]["identities"])
            if any(name in retained for name in site.inputs) and site.kind not in ("identity", "add"):
                raise ValueError("Only Identity or residual Add may consume a retained accumulator")
        if policy == "accumulator":
            storage = residual_storage(self)
            for site in self.sites:
                if site.kind == "add":
                    residual_add_factors(*residual_parameters(site, storage))

    def manifest(self):
        self.validate()
        sites, scales, shapes = [], {self.provenance["input_name"]: self.input_exponent}, {self.provenance["input_name"]: self.input_shape}
        for site in self.sites:
            row = {"name": site.name, "kind": site.kind, "inputs": list(site.inputs),
                   "shape": list(site.shape), "activation_exponent": site.exponent, "attributes": site.attrs}
            if site.retain_accumulator:
                row["retain_accumulator"] = True
            if site.weight is not None:
                bound = 127 * np.abs(site.weight.astype(np.int64)).reshape(site.weight.shape[0], -1).sum(axis=1)
                row.update(weight_sha256=array_digest(site.weight), bias_sha256=array_digest(site.bias),
                           weight_exponents=site.weight_exponents.tolist(),
                           accumulation_bound=bound.tolist(),
                           bound_with_bias=(bound + np.abs(site.bias.astype(np.int64))).tolist(),
                           contraction_dimension_bound=math.prod(site.weight.shape[1:]) * 128 * 128)
            if site.kind == "averagepool":
                area = math.prod(shapes[site.inputs[0]][2:])
                factor, denominator = averagepool_scale_factors(area, scales[site.inputs[0]], site.exponent)
                row["mean_requantization"] = {"area": area, "input_exponent": scales[site.inputs[0]], "output_exponent": site.exponent, "factor": factor, "denominator": denominator, "sum_bound": 127 * area, "scaled_numerator_bound": 127 * area * factor, "rounded_numerator_bound": 127 * area * factor + denominator // 2}
            sites.append(row)
            scales[site.name], shapes[site.name] = site.exponent, site.shape
        estimator = self.provenance.get("activation_estimator", "maxabs")
        policy = self.provenance.get("residual_policy", "legacy")
        storage = residual_storage(self) if policy == "accumulator" else {}
        residual_proofs = []
        for site in self.sites:
            if site.kind == "add" and policy == "accumulator":
                (left, lb), (right, rb) = (storage[name] for name in site.inputs)
                a, b, denominator, bound = residual_add_factors(left, right, site.exponent, lb, rb)
                residual_proofs.append({"site": site.name, "left_exponent": np.asarray(left).tolist(), "right_exponent": np.asarray(right).tolist(),
                    "left_bound": np.asarray(lb).tolist(), "right_bound": np.asarray(rb).tolist(), "left_factor": a.tolist(), "right_factor": b.tolist(),
                    "denominator": denominator.tolist(), "rounded_numerator_bound": bound.tolist()})
        recipe = "resnet50-v1.5-symmetric-pow2-development-v4" if estimator == "maxabs" else "resnet50-v1.5-symmetric-pow2-train-mse-v5"
        if policy == "accumulator":
            recipe = "resnet50-v1.5-symmetric-pow2-" + estimator + "-residual-accumulator-v6"
        result = {"recipe": recipe, "paper_quality_approved": False,
                "application_quality": "requires_separate_labeled_evaluation_and_approved_thresholds",
                "input_shape": list(self.input_shape), "input_exponent": self.input_exponent,
                "calibration_tensor_sha256": list(self.calibration_hashes), "provenance": self.provenance,
                "numeric_contract": {"operands": "symmetric signed int8 [-127,127], zero_point=0",
                    "weights": "per-output-channel max-abs power-of-two scale, minimally coarsened until worst-case contraction plus rounded bias fits int32; refuse outside [-30,30]",
                    "activations": "per-tensor TRAIN power-of-two " + estimator + "; Conv/Add with a sole-consumer Identity* chain ending in ReLU use that ReLU's observations; signed residual arms retain signed observations; input remains max-abs",
                    "accumulator": "int32 with worst-case partial-sum and bias bound", "rounding": "nearest ties away from zero",
                    "bias": "round folded FP32 bias / (input_scale * channel_weight_scale), int32",
                    "requantization": "bounded int64 scale then signed rounded division and saturation",
                    "residual": "rescale each arm to calibrated add scale, int64 add, saturate, then ReLU" if policy == "legacy" else "sole-user Conv/Identity residual arms retain biased int32 accumulators; align both arms in bounded int64, add, round once to TRAIN Add scale, saturate then ReLU; no per-arm rounding or saturation",
                    "maxpool": "integer maximum with negative-infinity padding", "averagepool": "int64 spatial sum, multiply/divide to TRAIN-calibrated output scale, round once and saturate; shift and intermediate overflow bounds checked",
                    "classifier": "int8/int32 contraction plus int32 bias, per-channel FP32 dequantized logits"},
                "sites": sites}
        if policy == "accumulator":
            result["residual_integer_proofs"] = residual_proofs
        return result


def post_relu_calibration_targets(graph):
    """Find Conv/Add outputs whose negative values every consumer discards.

    Each edge must have one user, including intermediate Identity nodes. A
    signed residual arm, side branch, or other operation terminates the proof.
    Names identify the result; operation semantics alone select the policy.
    """
    import torch

    targets = {}
    for producer in graph.graph.nodes:
        contraction = producer.op == "call_module" and type(graph.get_submodule(producer.target)) is torch.nn.Conv2d
        residual = producer.op == "call_function" and producer.target is operator.add and len(producer.all_input_nodes) == 2
        if not (contraction or residual):
            continue
        node, identities = producer, []
        while len(node.users) == 1:
            user = next(iter(node.users))
            if user.op != "call_module" or tuple(user.all_input_nodes) != (node,):
                break
            module = graph.get_submodule(user.target)
            if type(module) is torch.nn.ReLU:
                targets[producer.name] = {"relu": user.name, "identities": identities}
                break
            if type(module) is not torch.nn.Identity:
                break
            identities.append(user.name)
            node = user
    return targets


def calibrate_resnet50(model, calibration_images, *, calibration_ids, input_shape=(1, 3, 224, 224), checkpoint_declaration="unverified", activation_estimator="maxabs", residual_policy="legacy"):
    """Fold an unchanged CPU evaluation ResNet and freeze scales on this stream.

    Arrays are already preprocessed FP32 NCHW. This API performs no download or
    preprocessing. Calibration IDs must be stable, unique, and nonempty. Caller
    retains ownership of class order, checkpoint verification and quality gates.
    """
    import torch
    import torchvision
    from verify_resnet import fold_resnet_batchnorm

    if activation_estimator not in ("maxabs", "mse"):
        raise ValueError("Expected maxabs or mse TRAIN activation estimator")
    if residual_policy not in ("legacy", "accumulator"):
        raise ValueError("Expected legacy or accumulator residual policy")
    if type(model) is not torchvision.models.resnet.ResNet or [len(getattr(model, f"layer{i}")) for i in range(1, 5)] != [3, 4, 6, 3]:
        raise ValueError("Expected torchvision ResNet50 v1.5 with [3,4,6,3] bottlenecks")
    if any(type(block) is not torchvision.models.resnet.Bottleneck for i in range(1, 5) for block in getattr(model, f"layer{i}")) or model.fc.out_features != 1000:
        raise ValueError("Expected ResNet50 bottlenecks and 1000-class head")
    if model.conv1.weight.shape != (64, 3, 7, 7) or tuple(model.conv1.stride) != (2, 2) or tuple(model.conv1.padding) != (3, 3) or model.fc.in_features != 2048:
        raise ValueError("Expected the standard ResNet50 stem and 2048-channel classifier input")
    for stage in range(1, 5):
        for index, block in enumerate(getattr(model, f"layer{stage}")):
            stride = (2, 2) if stage > 1 and index == 0 else (1, 1)
            if tuple(block.conv1.stride) != (1, 1) or tuple(block.conv2.stride) != stride or tuple(block.conv3.stride) != (1, 1):
                raise ValueError("Expected torchvision v1.5 stride placement in the 3x3 bottleneck convolution")
    images = list(calibration_images)
    ids = list(calibration_ids)
    if not images or len(images) != len(ids) or any(not isinstance(value, str) or not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Calibration requires distinct nonempty IDs for every image")
    input_shape = tuple(input_shape)
    if len(input_shape) != 4 or input_shape[:2] != (1, 3) or min(input_shape[2:]) < 32:
        raise ValueError("Expected static batch-one NCHW RGB images at least 32 pixels wide/high")
    hashes = []
    for image in images:
        if not isinstance(image, np.ndarray) or image.dtype != np.float32 or image.shape != input_shape or not np.isfinite(image).all():
            raise ValueError("Calibration images must match finite FP32 static input shape")
        hashes.append(array_digest(image))
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate calibration image tensors")
    source_digest = state_digest(model)
    folded, fold_report = fold_resnet_batchnorm(model, torch, state_digest)
    if len(fold_report["sites"]) != 53:
        raise ValueError("Expected all 53 ResNet50 convolution/BatchNorm pairs")
    graph = torch.fx.symbolic_trace(folded)
    maxima, shapes = {}, {}

    class Observer(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if isinstance(result, torch.Tensor):
                if result.dtype != torch.float32 or not torch.isfinite(result).all():
                    raise ValueError(f"Nonfinite or non-FP32 calibration output: {node.name}")
                shape = tuple(result.shape)
                if node.name in shapes and shapes[node.name] != shape:
                    raise ValueError("Dynamic calibration shapes are not admitted")
                shapes[node.name] = shape
                maxima[node.name] = max(maxima.get(node.name, 0.0), float(result.abs().max()))
            return result

    folding_checks = []
    with torch.no_grad():
        for image in images:
            folded_logits = Observer(graph).run(torch.from_numpy(image.copy()))
            source_logits = model(torch.from_numpy(image.copy()))
            passed = bool(torch.allclose(folded_logits, source_logits, rtol=1e-4, atol=1e-4, equal_nan=False))
            folding_checks.append({"max_absolute_error": float((folded_logits - source_logits).abs().max()), "passed": passed})
            if not passed:
                raise AssertionError("Folded FP32 model differs from source on calibration input")
    projections = post_relu_calibration_targets(graph)
    activation_maxima = dict(maxima)
    for name, projection in projections.items():
        # Observer records each node before an in-place ReLU can change it.
        # The terminal ReLU maximum excludes negatives discarded on every use.
        activation_maxima[name] = maxima[projection["relu"]]
    # Only these operations introduce a quantized activation scale. In
    # particular, classifier logits remain FP32 and need no bounded exponent.
    activation_exponents = {node.name: exponent_for(activation_maxima[node.name]) for node in graph.graph.nodes
                            if (node.op == "call_module" and type(graph.get_submodule(node.target)) in (torch.nn.Conv2d, torch.nn.AdaptiveAvgPool2d)
                                or node.op == "call_function" and node.target is operator.add)}
    scale_checks = []
    if activation_estimator == "mse":
        observed_names = set()
        for node in graph.graph.nodes:
            if (node.op == "call_module" and type(graph.get_submodule(node.target)) in (torch.nn.Conv2d, torch.nn.AdaptiveAvgPool2d)
                    or node.op == "call_function" and node.target is operator.add):
                observed_names.add(projections.get(node.name, {}).get("relu", node.name))
        candidates = {name: tuple(range(exponent_for(maxima[name]), max(-31, exponent_for(maxima[name]) - 4), -1))
                      for name in observed_names}
        totals = {name: np.zeros(len(exponents), dtype=np.float64) for name, exponents in candidates.items()}
        counts = {name: 0 for name in candidates}

        class ErrorObserver(torch.fx.Interpreter):
            def run_node(self, node):
                result = super().run_node(node)
                if node.name in candidates:
                    count, errors = activation_scale_errors(result.detach().numpy(), candidates[node.name])
                    counts[node.name] += count
                    totals[node.name] += errors
                return result

        with torch.no_grad():
            for image in images:
                ErrorObserver(graph).run(torch.from_numpy(image.copy()))
        for name, exponents in candidates.items():
            selected = exponents[int(np.argmin(totals[name]))]
            activation_exponents[name] = selected
            scale_checks.append({"observed_site": name, "observed_max_abs": maxima[name], "candidate_exponents": list(exponents),
                                 "sample_count": counts[name], "mean_squared_errors": (totals[name] / counts[name]).tolist(),
                                 "selected_exponent": selected})
        for name, projection in projections.items():
            activation_exponents[name] = activation_exponents[projection["relu"]]
    sites, scales, weight_scale_adjustments = [], {}, []
    input_name = None
    for node in graph.graph.nodes:
        if node.op == "placeholder":
            if input_name is not None:
                raise ValueError("Expected one image input")
            input_name = node.name
            scales[node.name] = exponent_for(maxima[node.name])
            continue
        if node.op == "output":
            if len(node.all_input_nodes) != 1 or node.all_input_nodes[0].name != sites[-1].name or sites[-1].kind != "linear":
                raise ValueError("Expected the classifier as the sole graph output")
            continue
        inputs = tuple(arg.name for arg in node.all_input_nodes)
        attrs, weight, weight_exponents, bias = {}, None, None, None
        if node.op == "call_module":
            module = graph.get_submodule(node.target)
            if type(module) in (torch.nn.Conv2d, torch.nn.Linear):
                kind = "conv2d" if type(module) is torch.nn.Conv2d else "linear"
                if len(inputs) != 1:
                    raise ValueError("Contraction must have one tensor input")
                if kind == "conv2d":
                    if module.groups != 1 or tuple(module.dilation) != (1, 1) or module.padding_mode != "zeros":
                        raise ValueError("Only groups=1 zero-padded undilated convolution is admitted")
                    attrs = {"stride": list(module.stride), "padding": list(module.padding), "dilation": list(module.dilation)}
                raw_weight = module.weight.detach().numpy()
                raw_bias = module.bias.detach().numpy() if module.bias is not None else np.zeros(raw_weight.shape[0])
                try:
                    weight, weight_exponents, bias, adjustments = quantize_parameters(raw_weight, raw_bias, scales[inputs[0]])
                except ValueError as error:
                    raise ValueError(f"{error}: {node.name}") from error
                weight_scale_adjustments.extend({"site": node.name, **adjustment} for adjustment in adjustments)
                output_exponent = activation_exponents[node.name] if kind == "conv2d" else None
                if kind == "conv2d" and np.any(np.abs(scales[inputs[0]] + weight_exponents - output_exponent) > 30):
                    raise ValueError("Requantization shift exceeds 30 bits")
            elif type(module) in (torch.nn.ReLU, torch.nn.Identity):
                kind = "relu" if type(module) is torch.nn.ReLU else "identity"
                output_exponent = scales[inputs[0]]
            elif type(module) is torch.nn.MaxPool2d:
                if module.ceil_mode or module.dilation != 1 or module.return_indices:
                    raise ValueError("Only ordinary maxpool is admitted")
                kind = "maxpool"
                def pair(value):
                    return [value, value] if isinstance(value, int) else list(value)
                attrs = {"kernel": pair(module.kernel_size), "stride": pair(module.stride or module.kernel_size), "padding": pair(module.padding)}
                output_exponent = scales[inputs[0]]
            elif type(module) is torch.nn.AdaptiveAvgPool2d and module.output_size == (1, 1):
                kind, output_exponent = "averagepool", activation_exponents[node.name]
            else:
                raise ValueError(f"Unsupported module: {node.target}/{type(module).__name__}")
        elif node.op == "call_function" and node.target is operator.add and len(inputs) == 2:
            kind, output_exponent = "add", activation_exponents[node.name]
            if shapes[inputs[0]] != shapes[inputs[1]]:
                raise ValueError("Residual broadcasting is not admitted")
            if any(abs(scales[name] - output_exponent) > 30 for name in inputs):
                raise ValueError("Residual rescaling exceeds 30 bits")
        elif node.op == "call_function" and node.target is torch.flatten and node.args[1:] == (1,):
            kind, output_exponent = "flatten", scales[inputs[0]]
        else:
            raise ValueError(f"Unsupported FX operation: {node.op}/{node.target}")
        sites.append(Site(node.name, kind, inputs, shapes[node.name], output_exponent, attrs, weight, weight_exponents, bias))
        scales[node.name] = output_exponent
    if sum(site.kind == "conv2d" for site in sites) != 53 or sum(site.kind == "linear" for site in sites) != 1 or sum(site.kind == "add" for site in sites) != 16:
        raise ValueError("Incomplete ResNet50 graph inventory")
    if state_digest(model) != source_digest:
        raise AssertionError("Quantization mutated the source model")
    residual_targets = residual_accumulator_targets(sites) if residual_policy == "accumulator" else {}
    for site in sites:
        site.retain_accumulator = site.name in residual_targets
    plan = QuantizedResNet(sites, input_shape, scales[input_name], tuple(hashes),
                          {"source_state_dict_sha256": source_digest, "folded_state_dict_sha256": fold_report["derived_state_dict_sha256"],
                           "checkpoint_declaration": checkpoint_declaration, "calibration_ids": ids, "input_name": input_name,
                           "weight_scale_adjustments": weight_scale_adjustments,
                           "activation_estimator": activation_estimator,
                           "activation_scale_checks": sorted(scale_checks, key=lambda row: row["observed_site"]),
                           "activation_sampling": {"maximum_values_per_image_per_site": 8192, "selection": "floor(arange(count)*numel/count)", "candidate_offsets_from_maxabs": [0, -1, -2, -3], "tie_break": "coarsest scale", "data": "calibration stream only"} if activation_estimator == "mse" else None,
                           "post_relu_calibration": [{"site": name, **projection,
                                                       "observed_max_abs": maxima[name],
                                                       "observed_post_relu_maximum": activation_maxima[name],
                                                       "selected_exponent": scales[name]}
                                                      for name, projection in projections.items()],
                           "batchnorm_folding_checks": folding_checks, "batchnorm_folding_tolerance": {"rtol": 1e-4, "atol": 1e-4}})
    if residual_policy == "accumulator":
        plan.provenance.update(residual_policy=residual_policy,
                               residual_accumulator_chains=[{"site": name, **chain} for name, chain in residual_targets.items()])
    plan.validate()
    return plan


def integer_contraction(data, weight, attrs=None):
    """Independent contraction: FP64 exactly represents all bounded integer sums."""
    import torch
    if data.dtype != np.int8 or weight.dtype != np.int8:
        raise ValueError("Contraction requires explicit int8 arrays")
    if np.any(data == -128) or np.any(weight == -128):
        raise ValueError("Recipe uses symmetric [-127,127] int8 operands")
    if np.any(127 * np.abs(weight.astype(np.int64)).reshape(weight.shape[0], -1).sum(axis=1) > LIMIT):
        raise ValueError("Contraction exceeds its int32 exactness proof")
    x = torch.from_numpy(np.ascontiguousarray(data)).to(torch.float64)
    w = torch.from_numpy(np.ascontiguousarray(weight)).to(torch.float64)
    with torch.no_grad():
        product = torch.nn.functional.conv2d(x, w, stride=attrs["stride"], padding=attrs["padding"], dilation=attrs["dilation"]) if attrs is not None else x @ w.T
    result = product.numpy()
    if not np.isfinite(result).all() or np.any(result != np.trunc(result)) or np.any(np.abs(result) > LIMIT):
        raise AssertionError("Independent contraction violated integer exactness")
    return result.astype(np.int32)


def reference(plan, image, *, allow_calibration=False, return_intermediates=False, return_integer_logits=False):
    """Run the frozen recipe; refuse calibration/evaluation overlap by default."""
    plan.validate()
    if not isinstance(image, np.ndarray) or image.dtype != np.float32 or image.shape != plan.input_shape or not np.isfinite(image).all():
        raise ValueError("Expected finite FP32 image with the frozen input shape")
    if not allow_calibration and array_digest(image) in plan.calibration_hashes:
        raise ValueError("Evaluation tensor overlaps calibration")
    if return_integer_logits and return_intermediates:
        raise ValueError("Choose integer logits or intermediates output")
    integer_logits = None
    values = {plan.provenance["input_name"]: quantize(image, plan.input_exponent)}
    scales = {plan.provenance["input_name"]: plan.input_exponent}
    storage = residual_storage(plan) if plan.provenance.get("residual_policy") == "accumulator" else None
    uses = {name: sum(name in site.inputs for site in plan.sites) for name in [*values, *(site.name for site in plan.sites)]}
    for site in plan.sites:
        args = [values[name] for name in site.inputs]
        if site.kind in ("conv2d", "linear"):
            acc = integer_contraction(args[0], site.weight, site.attrs if site.kind == "conv2d" else None)
            channel_shape = (1, -1, 1, 1) if site.kind == "conv2d" else (1, -1)
            acc = (acc.astype(np.int64) + site.bias.reshape(channel_shape)).astype(np.int32)
            if site.kind == "linear":
                integer_logits = acc
            exponent = scales[site.inputs[0]] + site.weight_exponents
            if site.kind == "conv2d":
                result = acc if site.retain_accumulator else saturate(rescale(acc, (exponent - site.exponent).reshape(channel_shape)))
            else:
                result = acc.astype(np.float32) * np.exp2(exponent).astype(np.float32).reshape(channel_shape)
        elif site.kind == "add":
            result = residual_add(*args, *residual_parameters(site, storage)) if storage is not None else saturate(sum(rescale(arg, scales[name] - site.exponent) for arg, name in zip(args, site.inputs)))
        elif site.kind == "relu":
            result = np.maximum(args[0], 0).astype(np.int8)
        elif site.kind == "identity":
            result = args[0]
        elif site.kind == "flatten":
            result = args[0].reshape(site.shape)
        elif site.kind == "averagepool":
            result = averagepool(args[0], scales[site.inputs[0]], site.exponent)
        elif site.kind == "maxpool":
            ph, pw = site.attrs["padding"]
            padded = np.pad(args[0], ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-128)
            windows = np.lib.stride_tricks.sliding_window_view(padded, tuple(site.attrs["kernel"]), axis=(2, 3))
            sh, sw = site.attrs["stride"]
            result = windows[:, :, ::sh, ::sw].max(axis=(-1, -2))
        else:
            raise ValueError(f"Unknown site kind: {site.kind}")
        if result.shape != site.shape:
            raise AssertionError(f"Reference shape mismatch: {site.name}")
        values[site.name], scales[site.name] = result, site.exponent
        if not return_intermediates:
            for name in site.inputs:
                uses[name] -= 1
                if not uses[name]:
                    values.pop(name)
    output = values[plan.sites[-1].name]
    if return_integer_logits:
        return integer_logits, output
    return (output, values) if return_intermediates else output


def export_relax(plan, *, return_integer_logits=False):
    """Export the full mathematical graph; device substitution happens afterward."""
    import tvm
    from tvm import relax

    plan.validate()
    bb = relax.BlockBuilder()
    image = relax.Var("image", relax.TensorStructInfo(plan.input_shape, "int8"))
    values = {plan.provenance["input_name"]: image}
    scales = {plan.provenance["input_name"]: plan.input_exponent}
    storage = residual_storage(plan) if plan.provenance.get("residual_policy") == "accumulator" else None
    def emit(expr):
        return bb.emit(expr)
    def c(value, dtype="int64"):
        return relax.const(np.asarray(value, dtype=dtype), dtype=dtype)
    def rounded(value, divisor):
        mag = emit(relax.op.abs(value))
        result = emit(relax.op.floor_divide(emit(relax.op.add(mag, c(np.asarray(divisor) // 2))), c(divisor)))
        return emit(relax.op.where(emit(relax.op.less(value, c(0))), emit(relax.op.negative(result)), result))
    def scaled(value, exponent):
        exponents = np.asarray(exponent, dtype=np.int64)
        factor = np.left_shift(np.int64(1), np.maximum(exponents, 0))
        divisor = np.left_shift(np.int64(1), np.maximum(-exponents, 0))
        return rounded(emit(relax.op.multiply(emit(relax.op.astype(value, "int64")), c(factor))), divisor)
    def clipped(value):
        return emit(relax.op.astype(emit(relax.op.clip(value, -127, 127)), "int8"))
    integer_logits = None
    with bb.function("main", [image]):
        with bb.dataflow():
            for site in plan.sites:
                args = [values[name] for name in site.inputs]
                if site.kind in ("conv2d", "linear"):
                    shape = (1, -1, 1, 1) if site.kind == "conv2d" else (1, -1)
                    if site.kind == "conv2d":
                        product = emit(relax.op.nn.conv2d(args[0], c(site.weight, "int8"), strides=site.attrs["stride"], padding=site.attrs["padding"], dilation=site.attrs["dilation"], out_dtype="int32"))
                    else:
                        product = emit(relax.op.matmul(args[0], c(site.weight.T.copy(), "int8"), out_dtype="int32"))
                    product = emit(relax.op.add(product, c(site.bias.reshape(shape), "int32")))
                    if site.kind == "linear":
                        integer_logits = product
                    exponents = scales[site.inputs[0]] + site.weight_exponents
                    if site.kind == "conv2d":
                        result = product if site.retain_accumulator else clipped(scaled(product, (exponents - site.exponent).reshape(shape)))
                    else:
                        result = emit(relax.op.multiply(emit(relax.op.astype(product, "float32")), c(np.exp2(exponents).astype(np.float32).reshape(shape), "float32")))
                elif site.kind == "add":
                    if storage is None:
                        result = clipped(emit(relax.op.add(*(scaled(arg, scales[name] - site.exponent) for arg, name in zip(args, site.inputs)))))
                    else:
                        a, b, denominator, _ = residual_add_factors(*residual_parameters(site, storage))
                        arms = [emit(relax.op.multiply(emit(relax.op.astype(arg, "int64")), c(factor))) for arg, factor in zip(args, (a, b))]
                        result = clipped(rounded(emit(relax.op.add(*arms)), denominator))
                elif site.kind == "relu":
                    result = emit(relax.op.maximum(args[0], c(0, "int8")))
                elif site.kind == "identity":
                    result = args[0]
                elif site.kind == "flatten":
                    result = emit(relax.op.reshape(args[0], site.shape))
                elif site.kind == "averagepool":
                    factor, denominator = averagepool_scale_factors(math.prod(tuple(int(dim) for dim in args[0].struct_info.shape)[2:]), scales[site.inputs[0]], site.exponent)
                    total = emit(relax.op.sum(emit(relax.op.astype(args[0], "int64")), axis=[2, 3], keepdims=True))
                    result = clipped(rounded(emit(relax.op.multiply(total, c(factor))), denominator))
                elif site.kind == "maxpool":
                    result = emit(relax.op.nn.max_pool2d(args[0], pool_size=site.attrs["kernel"], strides=site.attrs["stride"], padding=site.attrs["padding"]))
                else:
                    raise ValueError(f"Unsupported site: {site.kind}")
                values[site.name], scales[site.name] = result, site.exponent
            result = values[plan.sites[-1].name]
            out = bb.emit_output(relax.Tuple([integer_logits, result]) if return_integer_logits else result)
        bb.emit_func_output(out)
    mod = bb.get()
    if not tvm.relax.analysis.well_formed(mod):
        raise AssertionError("Exported Relax graph is not well formed")
    return mod


def prepare_device_graph(plan, *, optimize=True, tile_i=1, tile_j=1, return_integer_logits=False):
    """Prepare graph and fail closed if any contraction stays on the CPU."""
    import tvm
    from tvm import relax, tir
    from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
    semantic = export_relax(plan, return_integer_logits=return_integer_logits)
    prepared = prepare_gemmini_graph(semantic, optimize=optimize, tile_i=tile_i, tile_j=tile_j)
    remaining, device_functions = [], []
    for gv, function in prepared.functions.items():
        if isinstance(function, relax.Function):
            def visit(expr):
                if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op) and expr.op.name in ("relax.nn.conv2d", "relax.matmul"):
                    remaining.append(expr.op.name)
            relax.analysis.post_order_visit(function.body, visit)
        elif isinstance(function, tir.PrimFunc):
            calls = []
            def visit_tir(node):
                if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern") and isinstance(node.args[0], tir.StringImm) and node.args[0].value.startswith("tvm_gemmini_"):
                    calls.append(str(node.args[0].value))
            tir.stmt_functor.post_order_visit(function.body, visit_tir)
            if calls:
                device_functions.append(gv.name_hint)
    # Equal-shaped layers share a scheduled PrimFunc. Count invocations, not unique functions.
    invocations = []
    for function in prepared.functions.values():
        if isinstance(function, relax.Function):
            def visit_call(expr):
                if isinstance(expr, relax.Call) and expr.op == tvm.ir.Op.get("relax.call_tir") and expr.args[0].name_hint in device_functions:
                    invocations.append(expr.args[0].name_hint)
            relax.analysis.post_order_visit(function.body, visit_call)
    if remaining or len(invocations) != 54:
        raise AssertionError(f"Incomplete device coverage: {len(invocations)}/54 contractions; remaining={remaining}")
    return semantic, prepared, {"convolution_invocations": 53, "classifier_invocations": 1, "device_invocations": len(invocations), "device_functions": sorted(device_functions)}
