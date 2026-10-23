"""Bounded semantic-partition expansion and exact reduction for Core ATen.

The canonical corpus proves one executable witness per overload.  This module deliberately asks a
different question: within a declared, target-independent input grammar, which dtype, shape, layout,
value, and scalar-control partitions are valid in eager PyTorch, and what is the exact smallest set
of those generated cases covering every witnessed partition and pairwise interaction?

Eager rejection never disappears.  Every rejected single or pair is retained with the exception
type and message.  Consequently ``complete`` means complete over the *witnessed feasible bounded
domain* recorded in the same artifact, not over arbitrary tensor programs or all possible values.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from merlin.targetgen.core_aten_cases import (
    case_document,
    case_document_from_arguments,
    core_aten_cases,
    decode_arguments,
    resolve_overload,
)
from merlin.targetgen.core_aten_cover import exact_minimum_cover, overload_digest
from merlin.targetgen.core_aten_eqsat import quotient_observational_equivalents


@dataclass(frozen=True)
class BoundedCoverageProfile:
    """The publication boundary; changing any field changes the profile digest."""

    schema_version: int = 1
    name: str = "core-aten-semantic-pairwise-v2-int8"
    ranks: tuple[int, ...] = (0, 1, 2, 3, 4)
    extents: tuple[int, ...] = (0, 1, 2, 3, 5)
    max_tensor_elements: int = 256
    dtypes: tuple[str, ...] = (
        "bool",
        "int8",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "complex64",
    )
    shape_partitions: tuple[str, ...] = (
        "scalar",
        "empty",
        "singleton",
        "prime_vector",
        "rectangular",
        "rank3_singleton",
        "rank4",
        "broadcast_pair",
    )
    layout_partitions: tuple[str, ...] = (
        "contiguous",
        "noncontiguous_padded",
        "sliced_offset",
        "zero_stride",
        "channels_last",
    )
    value_partitions: tuple[str, ...] = (
        "ordinary",
        "zero",
        "sign_mix",
        "finite_extrema",
        "nan",
        "infinity",
        "ties",
        "signed_zero",
    )
    interaction_strength: int = 2
    seed: int = 0
    maximum_candidates_per_overload: int = 384

    def __post_init__(self) -> None:
        if self.ranks != (0, 1, 2, 3, 4):
            raise ValueError("the v1 bounded profile has fixed ranks 0 through 4")
        if self.extents != (0, 1, 2, 3, 5):
            raise ValueError("the v1 bounded profile has fixed extents 0, 1, 2, 3, and 5")
        if self.max_tensor_elements != 256:
            raise ValueError("the v1 bounded profile has a fixed 256-element ceiling")
        if self.interaction_strength != 2:
            raise ValueError("the v1 bounded profile has fixed pairwise interaction strength")
        if self.seed != 0:
            raise ValueError("the v1 bounded profile has a fixed deterministic seed of zero")
        unknown_shapes = set(self.shape_partitions) - ({"broadcast_pair"} | set(_SHAPE_VALUES))
        if unknown_shapes:
            raise ValueError(f"unknown shape partitions: {sorted(unknown_shapes)}")
        if self.maximum_candidates_per_overload < 1:
            raise ValueError("maximum_candidates_per_overload must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> BoundedCoverageProfile:
        values = dict(document)
        for name in (
            "ranks",
            "extents",
            "dtypes",
            "shape_partitions",
            "layout_partitions",
            "value_partitions",
        ):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        return hashlib.sha256(payload.encode()).hexdigest()


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _is_word(character: str) -> bool:
    return character.isalnum() or character == "_"


def _mask_hex_addresses(text: str) -> str:
    """Replace each ``0x`` followed by at least ten hex digits with a stable token."""

    pieces: list[str] = []
    position = 0
    while True:
        found = text.find("0x", position)
        if found < 0:
            pieces.append(text[position:])
            return "".join(pieces)
        end = found + 2
        while end < len(text) and text[end] in _HEX_DIGITS:
            end += 1
        if end - (found + 2) >= 10:
            pieces.append(text[position:found])
            pieces.append("0x<runtime-address>")
            position = end
        else:
            pieces.append(text[position : found + 1])
            position = found + 1


def _mask_large_integers(text: str) -> str:
    """Replace each standalone run of at least ten decimal digits (with optional sign) by a token.

    Standalone means the run is neither preceded by a word character or ``.`` nor followed by one.
    A leading ``-`` belongs to the number only when the character before it is not itself a word
    character or ``.``.
    """

    pieces: list[str] = []
    position = 0
    index = 0
    while index < len(text):
        if not text[index].isdecimal():
            index += 1
            continue
        end = index
        while end < len(text) and text[end].isdecimal():
            end += 1
        run_start = index
        index = end
        if end - run_start < 10 or (end < len(text) and (_is_word(text[end]) or text[end] == ".")):
            continue
        start = run_start
        if run_start > 0 and text[run_start - 1] == "-":
            before = text[run_start - 2] if run_start > 1 else ""
            if not before or not (_is_word(before) or before == "."):
                start = run_start - 1
        elif run_start > 0 and (_is_word(text[run_start - 1]) or text[run_start - 1] == "."):
            continue
        if start < position:
            start = run_start
        pieces.append(text[position:start])
        pieces.append("<runtime-large-value>")
        position = end
    pieces.append(text[position:])
    return "".join(pieces)


def mask_runtime_values(message: str) -> str:
    """Remove process-local pointer-sized values so identical builds hash identically."""

    return _mask_large_integers(_mask_hex_addresses(message))


def bounded_suite_digest(document: Mapping[str, Any]) -> str:
    """Digest every bounded-suite byte of meaning except the digest field itself."""

    payload = {key: value for key, value in document.items() if key != "suite_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(encoded.encode()).hexdigest()


_SHAPE_VALUES: dict[str, tuple[int, ...]] = {
    "scalar": (),
    "empty": (0,),
    "singleton": (1,),
    "prime_vector": (5,),
    "rectangular": (2, 3),
    "rank3_singleton": (2, 1, 5),
    "rank4": (1, 2, 3, 5),
}

_INDEX_ARGUMENTS = {
    "index",
    "indices",
    "offsets",
    "padding_idx",
    "dim",
    "dims",
    "size",
    "stride",
    "output_size",
    "normalized_shape",
    "split_sizes",
    "kernel_size",
    "padding",
    "dilation",
    "output_padding",
    "bias_sizes",
    "output_mask",
}

_POINTWISE_PACKETS = {
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "ceil",
    "clamp",
    "cos",
    "cosh",
    "div",
    "elu",
    "eq",
    "erf",
    "exp",
    "expm1",
    "floor",
    "fmod",
    "ge",
    "gelu",
    "gt",
    "hardtanh",
    "isinf",
    "isnan",
    "le",
    "leaky_relu",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "neg",
    "pow",
    "reciprocal",
    "relu",
    "remainder",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "sub",
    "tan",
    "tanh",
    "trunc",
    "where",
}

_REDUCTION_PACKETS = {
    "_log_softmax",
    "_softmax",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "cumsum",
    "max",
    "mean",
    "min",
    "nonzero",
    "prod",
    "sort",
    "sum",
    "topk",
    "var",
}

_CREATION_PACKETS = {"empty", "empty_strided", "full", "rand", "randn"}


def _packet(overload: str) -> str:
    return overload.split(".", 2)[1]


def _json_name(value: Any) -> str:
    import torch

    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if value is None:
        return "none"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return format(value, ".12g")
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_json_name(item) for item in value) + "]"
    return str(value)


def _argument_values(overload: str, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    schema = resolve_overload(overload)._schema
    values: dict[str, Any] = {}
    for index, argument in enumerate(schema.arguments):
        if index < len(args):
            values[argument.name] = args[index]
        elif argument.name in kwargs:
            values[argument.name] = kwargs[argument.name]
        elif argument.has_default_value():
            values[argument.name] = argument.default_value
    return values


def _replace_argument(
    overload: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    arguments = list(args)
    keywords = dict(kwargs)
    schema_names = [argument.name for argument in resolve_overload(overload)._schema.arguments]
    if name not in schema_names:
        raise ValueError(f"{overload} has no argument {name!r}")
    index = schema_names.index(name)
    if index < len(arguments):
        arguments[index] = value
    else:
        keywords[name] = value
    return tuple(arguments), keywords


def _map_tensors(value: Any, transform, *, argument_name: str) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return transform(value, argument_name)
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, transform, argument_name=argument_name) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, transform, argument_name=argument_name) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensors(item, transform, argument_name=argument_name) for key, item in value.items()}
    return value


def _map_arguments(overload: str, args: tuple[Any, ...], kwargs: Mapping[str, Any], transform):
    schema = resolve_overload(overload)._schema
    positional = tuple(
        _map_tensors(value, transform, argument_name=schema.arguments[index].name) for index, value in enumerate(args)
    )
    keywords = {name: _map_tensors(value, transform, argument_name=name) for name, value in kwargs.items()}
    return positional, keywords


def _payload(argument_name: str) -> bool:
    return argument_name not in _INDEX_ARGUMENTS and not argument_name.startswith("running_")


def _with_requires_grad(tensor, requires_grad: bool):
    if requires_grad and (tensor.is_floating_point() or tensor.is_complex()):
        tensor.requires_grad_(True)
    return tensor


def _transform_dtype(overload: str, args, kwargs, dtype_name: str):
    import torch

    dtype = getattr(torch, dtype_name)
    changed = False

    def convert(tensor, argument_name):
        nonlocal changed
        if not _payload(argument_name):
            return tensor
        if not (tensor.is_floating_point() or tensor.is_complex() or tensor.dtype in (torch.int32, torch.int64)):
            return tensor
        changed = True
        requires_grad = bool(tensor.requires_grad)
        return _with_requires_grad(tensor.detach().to(dtype), requires_grad)

    args, kwargs = _map_arguments(overload, args, kwargs, convert)
    schema_names = {argument.name for argument in resolve_overload(overload)._schema.arguments}
    if "dtype" in schema_names:
        args, kwargs = _replace_argument(overload, args, kwargs, "dtype", dtype)
        changed = True
    if not changed:
        raise ValueError("no dtype-bearing payload exists")
    return args, kwargs


def _fill_for_shape(tensor, shape: tuple[int, ...]):
    import torch

    count = math.prod(shape)
    if tensor.dtype == torch.bool:
        values = torch.arange(count, dtype=torch.int64).remainder(2).bool()
    elif tensor.is_complex():
        real = torch.linspace(-0.75, 1.25, count, dtype=torch.float32) if count else torch.empty(0)
        values = torch.complex(real, real.flip(0) if count else real).to(tensor.dtype)
    elif tensor.is_floating_point():
        values = (
            torch.linspace(-0.75, 1.25, count, dtype=torch.float32).to(tensor.dtype)
            if count
            else torch.empty(0, dtype=tensor.dtype)
        )
    else:
        values = torch.arange(count, dtype=tensor.dtype)
    return _with_requires_grad(values.reshape(shape), bool(tensor.requires_grad))


def _transform_shape(overload: str, args, kwargs, partition: str):
    import torch

    shape = _SHAPE_VALUES.get(partition)
    packet = _packet(overload)
    changed = False
    if partition == "broadcast_pair":
        if packet not in _POINTWISE_PACKETS:
            raise ValueError("broadcast_pair is only defined for pointwise overloads")
        shapes = ((2, 1, 5), (1, 3, 1))
        payload_index = 0

        def broadcast(tensor, argument_name):
            nonlocal changed, payload_index
            if not _payload(argument_name):
                return tensor
            result = _fill_for_shape(tensor, shapes[payload_index % len(shapes)])
            payload_index += 1
            changed = payload_index >= 2
            return result

        args, kwargs = _map_arguments(overload, args, kwargs, broadcast)
        if not changed:
            raise ValueError("broadcast_pair requires at least two payload tensors")
        return args, kwargs
    assert shape is not None
    if packet in _CREATION_PACKETS:
        values = _argument_values(overload, args, kwargs)
        if "size" not in values:
            raise ValueError("creation overload has no size argument")
        args, kwargs = _replace_argument(overload, args, kwargs, "size", list(shape))
        if packet == "empty_strided":
            contiguous = list(torch.empty(shape).stride())
            args, kwargs = _replace_argument(overload, args, kwargs, "stride", contiguous)
        return args, kwargs
    if packet not in _POINTWISE_PACKETS | _REDUCTION_PACKETS:
        raise ValueError("shape grammar does not classify this overload")

    def reshape(tensor, argument_name):
        nonlocal changed
        if not _payload(argument_name):
            return tensor
        changed = True
        return _fill_for_shape(tensor, shape)

    args, kwargs = _map_arguments(overload, args, kwargs, reshape)
    if not changed:
        raise ValueError("no reshapable payload tensor exists")
    return args, kwargs


def _pattern_values(tensor, pattern: str):
    import torch

    count = tensor.numel()
    device = tensor.device
    dtype = tensor.dtype
    if pattern == "zero":
        values = torch.zeros(count, dtype=dtype, device=device)
    elif pattern == "ties":
        values = torch.full((count,), 0 if dtype == torch.bool else 0.5, dtype=dtype, device=device)
    elif pattern == "sign_mix":
        if dtype == torch.bool:
            values = torch.arange(count, device=device).remainder(2).bool()
        elif tensor.is_complex():
            base = torch.tensor([-2.0, -0.0, 0.5, 3.0], device=device).repeat((count + 3) // 4)[:count]
            values = torch.complex(base, base.flip(0)).to(dtype)
        else:
            base = torch.tensor([-2, -1, 0, 3], device=device).repeat((count + 3) // 4)[:count]
            values = base.to(dtype)
    elif pattern == "finite_extrema":
        if dtype == torch.bool:
            base = torch.tensor([False, True], device=device)
        elif tensor.is_floating_point() or tensor.is_complex():
            component = torch.float32 if tensor.is_complex() else dtype
            limit = torch.finfo(component).max
            base = torch.tensor([-limit, -1.0, 0.0, 1.0, limit], dtype=component, device=device)
            if tensor.is_complex():
                base = torch.complex(base, torch.zeros_like(base)).to(dtype)
        else:
            info = torch.iinfo(dtype)
            base = torch.tensor([info.min, -1, 0, 1, info.max], dtype=dtype, device=device)
        values = base.repeat((count + len(base) - 1) // len(base))[:count]
    elif pattern in {"nan", "infinity", "signed_zero"}:
        if not (tensor.is_floating_point() or tensor.is_complex()):
            raise ValueError(f"{pattern} requires floating or complex payload")
        if pattern == "nan":
            base = [float("nan"), -1.0, 0.0, 1.0]
        elif pattern == "infinity":
            base = [float("-inf"), -1.0, 0.0, float("inf")]
        else:
            base = [-0.0, 0.0]
        component = torch.tensor(base, dtype=torch.float32, device=device)
        if tensor.is_complex():
            component = torch.complex(component, torch.zeros_like(component))
        values = component.to(dtype).repeat((count + len(base) - 1) // len(base))[:count]
    else:
        raise ValueError(f"unknown value partition {pattern!r}")
    return _with_requires_grad(values.reshape(tensor.shape), bool(tensor.requires_grad))


def _transform_values(overload: str, args, kwargs, pattern: str):
    changed = False

    def replace(tensor, argument_name):
        nonlocal changed
        if not _payload(argument_name):
            return tensor
        changed = True
        return _pattern_values(tensor, pattern)

    args, kwargs = _map_arguments(overload, args, kwargs, replace)
    if not changed:
        raise ValueError("no value-bearing payload tensor exists")
    return args, kwargs


def _layout_tensor(tensor, partition: str):
    import torch

    if tensor.numel() == 0 or tensor.ndim == 0:
        raise ValueError(f"{partition} requires a non-empty, non-scalar tensor")
    original = tensor.detach()
    requires_grad = bool(tensor.requires_grad)
    if partition == "noncontiguous_padded":
        expanded_shape = (*tensor.shape[:-1], tensor.shape[-1] * 2)
        storage = torch.empty(expanded_shape, dtype=tensor.dtype, device=tensor.device)
        view = storage[..., ::2]
        view.copy_(original)
    elif partition == "sliced_offset":
        storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
        view = storage[1:].view(tensor.shape)
        view.copy_(original)
    elif partition == "zero_stride":
        axis = next((index for index, extent in enumerate(tensor.shape) if extent > 1), None)
        if axis is None:
            raise ValueError("zero_stride requires an extent greater than one")
        source = original.select(axis, 0).unsqueeze(axis).clone()
        view = source.expand(tensor.shape)
    elif partition == "channels_last":
        if tensor.ndim != 4:
            raise ValueError("channels_last requires rank four")
        view = original.contiguous(memory_format=torch.channels_last)
    else:
        raise ValueError(f"unknown layout partition {partition!r}")
    if partition == "noncontiguous_padded" and view.is_contiguous():
        raise ValueError(f"{partition} did not produce a non-contiguous tensor")
    return _with_requires_grad(view, requires_grad)


def _transform_layout(overload: str, args, kwargs, partition: str):
    changed = False

    def replace(tensor, argument_name):
        nonlocal changed
        # Running statistics and index/control tensors are not data-layout variants. In particular,
        # turning batch-normalization running statistics into overlapping zero-stride views gives a
        # mutating operator undefined write order and an oracle that cannot be replayed.
        if not _payload(argument_name):
            return tensor
        if tensor.numel() == 0 or tensor.ndim == 0:
            return tensor
        try:
            result = _layout_tensor(tensor, partition)
        except ValueError:
            return tensor
        changed = True
        return result

    args, kwargs = _map_arguments(overload, args, kwargs, replace)
    if not changed:
        raise ValueError(f"no tensor accepts layout partition {partition}")
    return args, kwargs


def _first_payload_tensor(overload: str, values: Mapping[str, Any]):
    import torch

    def find(value: Any):
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            return next((found for item in value if (found := find(item)) is not None), None)
        return None

    for name, value in values.items():
        if _payload(name) and (tensor := find(value)) is not None:
            return tensor
    return None


def _control_options(overload: str, args, kwargs) -> dict[str, dict[str, Any]]:
    """Return finite semantic control partitions that can be represented by the exact schema."""

    import torch

    schema = resolve_overload(overload)._schema
    values = _argument_values(overload, args, kwargs)
    primary = _first_payload_tensor(overload, values)
    rank = primary.ndim if primary is not None else 0
    options: dict[str, dict[str, Any]] = {}
    for argument in schema.arguments:
        name = argument.name
        if name not in values or name in {"dtype", "layout", "device", "pin_memory", "memory_format"}:
            continue
        current = values[name]
        proposed: list[Any] = []
        type_name = str(argument.type)
        if overload == "aten._native_batch_norm_legit.no_stats" and name == "training":
            # This schema is the training/no-running-statistics primitive.  The pinned PyTorch build
            # segfaults rather than raising when called with training=False, so that value is outside
            # the legal bounded domain and must never be probed in-process.
            proposed = [True]
        elif type(current) is bool:
            proposed = [False, True]
        elif name in {"dim", "dims"} and rank:
            if isinstance(current, int):
                proposed = [-rank, 0, rank - 1]
                if "Optional" in type_name:
                    proposed.append(None)
            elif isinstance(current, list) or current is None:
                proposed = [None, [], [0], [-1]] if "Optional" in type_name else [[], [0], [-1]]
                if rank > 1:
                    proposed.append([0, rank - 1])
        elif name == "rounding_mode":
            proposed = [None, "trunc", "floor"]
        elif name == "approximate":
            proposed = ["none", "tanh"]
        elif name == "reduce":
            proposed = ["sum", "prod", "mean", "amax", "amin"]
        elif name in {"alpha", "beta"}:
            proposed = [-1, 0, 1, 2]
        elif name == "p":
            proposed = [0.0, 1.0, 2.0, float("inf")] if "dist" in overload else [0.0, 0.25, 1.0]
        elif name == "eps":
            proposed = [0.0, 1e-5, 1e-3]
        elif name == "momentum":
            proposed = [0.0, 0.1, 1.0]
        elif name in {"negative_slope", "value", "fill_value"}:
            proposed = [-1.0, 0.0, 1.0]
        elif name in {"other", "exponent"} and not isinstance(current, torch.Tensor):
            proposed = [-2.0, -0.5, 0.0, 1.0, 2.0]
        elif name == "normalization":
            proposed = [0, 1, 2]
        elif name in {"interpolation_mode", "padding_mode", "mode"}:
            proposed = [0, 1, 2]
        elif name == "correction":
            proposed = [None, 0, 1]
        elif name == "keepdim":
            proposed = [False, True]
        elif name in {"largest", "sorted", "include_self", "onesided", "align_corners"}:
            proposed = [False, True]
        elif name in {"weight", "bias", "per_sample_weights"} and "Optional[Tensor]" in type_name:
            if current is not None:
                proposed = [current, None]
        unique: dict[str, Any] = {}
        for value in proposed:
            unique[_json_name(value)] = value
        baseline_name = _json_name(current)
        if len(unique) > 1 or (unique and baseline_name not in unique):
            unique[baseline_name] = current
            options[f"control:{name}"] = dict(sorted(unique.items()))
    return options


def _baseline_axis_values(overload: str, arguments: Mapping[str, Any]) -> dict[str, str]:

    tensor = _first_payload_tensor(overload, arguments)
    if tensor is None:
        dtype = arguments.get("dtype")
        dtype_name = _json_name(dtype) if dtype is not None else "implicit"
        shape_name = "canonical"
    else:
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        shape = tuple(tensor.shape)
        shape_name = next((name for name, value in _SHAPE_VALUES.items() if value == shape), "canonical")
    layout = "contiguous"
    return {"dtype": dtype_name, "shape": shape_name, "layout": layout, "values": "ordinary"}


def _factor_domains(overload: str, case: Mapping[str, Any], profile: BoundedCoverageProfile):
    args, kwargs = decode_arguments(case["arguments"])
    values = _argument_values(overload, args, kwargs)
    baseline = _baseline_axis_values(overload, values)
    domains: dict[str, dict[str, Any]] = {
        "dtype": {name: name for name in profile.dtypes},
        "shape": {name: name for name in profile.shape_partitions},
        "layout": {name: name for name in profile.layout_partitions},
        "values": {name: name for name in profile.value_partitions},
    }
    for factor, value in baseline.items():
        domains[factor][value] = value
    domains.update(_control_options(overload, args, kwargs))
    for factor, controls in list(domains.items()):
        baseline_value = baseline.get(factor)
        if factor.startswith("control:"):
            argument_name = factor.split(":", 1)[1]
            baseline_value = _json_name(values[argument_name])
            baseline[factor] = baseline_value
        domains[factor] = dict(sorted(controls.items()))
    return baseline, domains


def _apply_assignment(
    overload: str,
    canonical_case: Mapping[str, Any],
    assignment: Mapping[str, str],
    baseline: Mapping[str, str],
    domains: Mapping[str, Mapping[str, Any]],
):
    args, kwargs = decode_arguments(canonical_case["arguments"])
    dtype = assignment.get("dtype", baseline["dtype"])
    shape = assignment.get("shape", baseline["shape"])
    values = assignment.get("values", baseline["values"])
    layout = assignment.get("layout", baseline["layout"])
    if dtype != baseline["dtype"]:
        args, kwargs = _transform_dtype(overload, args, kwargs, dtype)
    if shape != baseline["shape"]:
        args, kwargs = _transform_shape(overload, args, kwargs, shape)
    if values != baseline["values"]:
        args, kwargs = _transform_values(overload, args, kwargs, values)
    if layout != baseline["layout"]:
        args, kwargs = _transform_layout(overload, args, kwargs, layout)
    for factor in sorted(name for name in assignment if name.startswith("control:")):
        if assignment[factor] == baseline[factor]:
            continue
        name = factor.split(":", 1)[1]
        args, kwargs = _replace_argument(overload, args, kwargs, name, domains[factor][assignment[factor]])
    return args, kwargs


def _payload_tensors(overload: str, args, kwargs):
    import torch

    schema = resolve_overload(overload)._schema
    found = []

    def collect(value: Any, name: str) -> None:
        if isinstance(value, torch.Tensor):
            if _payload(name):
                found.append(value)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                collect(item, name)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item, name)

    for index, value in enumerate(args):
        collect(value, schema.arguments[index].name)
    for name, value in kwargs.items():
        collect(value, name)
    return found


def _value_partition_is_witnessed(tensors, partition: str) -> bool:
    import torch

    tensors = [tensor.detach() for tensor in tensors if tensor.numel()]
    if not tensors:
        return False
    if partition == "zero":
        return all(bool(torch.all(tensor == 0)) for tensor in tensors)
    if partition == "nan":
        return any(
            (tensor.is_floating_point() or tensor.is_complex()) and bool(torch.isnan(tensor).any())
            for tensor in tensors
        )
    if partition == "infinity":
        return any(
            (tensor.is_floating_point() or tensor.is_complex()) and bool(torch.isinf(tensor).any())
            for tensor in tensors
        )
    if partition == "sign_mix":
        values = [tensor.real if tensor.is_complex() else tensor for tensor in tensors if tensor.dtype != torch.bool]
        return (
            bool(values)
            and any(bool((tensor < 0).any()) for tensor in values)
            and any(bool((tensor > 0).any()) for tensor in values)
        )
    if partition == "finite_extrema":
        for tensor in tensors:
            values = tensor.real if tensor.is_complex() else tensor
            if values.dtype == torch.bool:
                if bool(values.any()) and bool((~values).any()):
                    return True
            elif values.is_floating_point():
                info = torch.finfo(values.dtype)
                if bool((values == info.min).any()) and bool((values == info.max).any()):
                    return True
            else:
                info = torch.iinfo(values.dtype)
                if bool((values == info.min).any()) and bool((values == info.max).any()):
                    return True
        return False
    if partition == "ties":
        for tensor in tensors:
            flat = tensor.reshape(-1)
            if any(bool((flat[index + 1 :] == flat[index]).any()) for index in range(flat.numel() - 1)):
                return True
        return False
    if partition == "signed_zero":
        for tensor in tensors:
            values = tensor.real if tensor.is_complex() else tensor
            if not values.is_floating_point():
                continue
            zeros = values == 0
            if bool(zeros.any()):
                signs = torch.signbit(values[zeros])
                if bool(signs.any()) and bool((~signs).any()):
                    return True
        return False
    if partition == "ordinary":
        return True
    raise ValueError(f"unknown value partition {partition!r}")


def _validate_assignment_witness(
    overload: str,
    args,
    kwargs,
    assignment: Mapping[str, str],
    baseline: Mapping[str, str],
) -> None:
    """Refuse labels that the transformed concrete arguments do not actually exhibit."""

    tensors = _payload_tensors(overload, args, kwargs)
    concrete = _argument_values(overload, args, kwargs)
    dtype = assignment["dtype"]
    if dtype != baseline["dtype"]:
        tensor_witness = any(str(tensor.dtype).removeprefix("torch.") == dtype for tensor in tensors)
        schema_witness = "dtype" in concrete and _json_name(concrete["dtype"]) == dtype
        if not tensor_witness and not schema_witness:
            raise ValueError(f"dtype partition {dtype!r} is not present in the concrete arguments")
    shape = assignment["shape"]
    if shape != baseline["shape"]:
        if shape == "broadcast_pair":
            if len(tensors) < 2 or len({tuple(tensor.shape) for tensor in tensors}) < 2:
                raise ValueError("broadcast_pair requires differently shaped payload tensors")
            try:
                import torch

                result_shape = tuple(torch.broadcast_shapes(*(tuple(tensor.shape) for tensor in tensors)))
            except RuntimeError as exc:
                raise ValueError("broadcast_pair payloads are not broadcast-compatible") from exc
            if all(result_shape == tuple(tensor.shape) for tensor in tensors):
                raise ValueError("broadcast_pair does not expand any payload")
        elif _packet(overload) in _CREATION_PACKETS:
            expected = _SHAPE_VALUES[shape]
            actual = _argument_values(overload, args, kwargs).get("size")
            if tuple(actual or ()) != expected:
                raise ValueError(f"shape partition {shape!r} is not present in the creation size")
        else:
            expected = _SHAPE_VALUES[shape]
            if not tensors or tuple(tensors[0].shape) != expected:
                raise ValueError(f"shape partition {shape!r} is not present in the primary payload")
    layout = assignment["layout"]
    if layout != baseline["layout"]:
        if layout == "noncontiguous_padded":
            witnessed = any(
                tensor.numel()
                and not tensor.is_contiguous()
                and tensor.storage_offset() == 0
                and not any(step == 0 and extent > 1 for extent, step in zip(tensor.shape, tensor.stride()))
                for tensor in tensors
            )
        elif layout == "sliced_offset":
            witnessed = any(tensor.numel() and tensor.storage_offset() > 0 for tensor in tensors)
        elif layout == "zero_stride":
            witnessed = any(
                any(step == 0 and extent > 1 for extent, step in zip(tensor.shape, tensor.stride()))
                for tensor in tensors
            )
        elif layout == "channels_last":
            import torch

            witnessed = any(
                tensor.ndim == 4 and tensor.is_contiguous(memory_format=torch.channels_last) for tensor in tensors
            )
        else:
            witnessed = False
        if not witnessed:
            raise ValueError(f"layout partition {layout!r} is not present in the concrete arguments")
    values = assignment["values"]
    if values != baseline["values"] and not _value_partition_is_witnessed(tensors, values):
        raise ValueError(f"value partition {values!r} is not present in the concrete arguments")
    for factor, expected in assignment.items():
        if not factor.startswith("control:") or expected == baseline[factor]:
            continue
        name = factor.split(":", 1)[1]
        if _json_name(concrete[name]) != expected:
            raise ValueError(f"control partition {factor}={expected!r} is not present in the arguments")


def _case_id(overload: str, assignment: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(sorted(assignment.items())), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256((overload + "\0" + encoded).encode()).hexdigest()[:16]
    return f"{overload}::{digest}"


def _obligations(overload: str, assignment: Mapping[str, str]) -> set[str]:
    singles = {f"single::{overload}::{factor}={value}" for factor, value in assignment.items()}
    pairs = {
        f"pair::{overload}::{left}={assignment[left]}::{right}={assignment[right]}"
        for left, right in itertools.combinations(sorted(assignment), 2)
    }
    return {f"overload::{overload}", *singles, *pairs}


def _pair_key(left: str, left_value: str, right: str, right_value: str):
    if left > right:
        left, right = right, left
        left_value, right_value = right_value, left_value
    return left, left_value, right, right_value


def _greedy_pairwise_rows(
    domains: Mapping[str, Sequence[str]],
    baseline: Mapping[str, str],
) -> list[dict[str, str]]:
    """Construct a deterministic covering-array candidate pool.

    Exact minimization happens after eager validation. This front end only avoids materializing the
    full Cartesian product: each new row is seeded by the first uncovered pair and every remaining
    coordinate greedily maximizes newly covered pairs. Single-axis and all-baseline interactions are
    already covered by the baseline/singleton candidates, so this routine targets pairs for which
    both values are non-baseline.
    """

    factors = sorted(domains)
    uncovered = {
        _pair_key(left, left_value, right, right_value)
        for left, right in itertools.combinations(factors, 2)
        for left_value in domains[left]
        for right_value in domains[right]
        if left_value != baseline[left] and right_value != baseline[right]
    }
    rows: list[dict[str, str]] = []
    while uncovered:
        left, left_value, right, right_value = min(uncovered)
        row = {left: left_value, right: right_value}
        for factor in factors:
            if factor in row:
                continue
            choices: list[tuple[int, bool, str]] = []
            for value in domains[factor]:
                score = sum(
                    _pair_key(factor, value, assigned_factor, assigned_value) in uncovered
                    for assigned_factor, assigned_value in row.items()
                )
                choices.append((score, value == baseline[factor], value))
            _score, _is_baseline, selected = max(choices, key=lambda item: (item[0], item[1], item[2]))
            row[factor] = selected
        rows.append(dict(sorted(row.items())))
        for factor_a, factor_b in itertools.combinations(factors, 2):
            uncovered.discard(_pair_key(factor_a, row[factor_a], factor_b, row[factor_b]))
    return rows


def _attempt(
    overload: str,
    canonical_case: Mapping[str, Any],
    assignment: Mapping[str, str],
    baseline: Mapping[str, str],
    domains: Mapping[str, Mapping[str, Any]],
):
    try:
        if overload == "aten.native_layer_norm_backward.default" and (
            assignment.get("dtype") in {"float16", "bfloat16"} or assignment.get("values") == "finite_extrema"
        ):
            # The pinned CPU kernel returns run-varying/uninitialised lanes for these malformed
            # dependent-statistic combinations. They cannot support a replayable correctness oracle;
            # retain the rejection explicitly instead of allowing a coincidentally equal duplicate run through.
            raise ValueError(
                "pinned eager native_layer_norm_backward is not replay-stable for low-precision "
                "or finite-extrema inputs"
            )
        if overload == "aten.native_group_norm_backward.default" and assignment.get("layout") in {
            "noncontiguous_padded",
            "zero_stride",
        }:
            raise ValueError(
                "pinned eager native_group_norm_backward is not replay-stable for non-contiguous/overlapping inputs"
            )
        if (
            overload in {"aten._fft_c2r.default", "aten._fft_r2c.default"}
            and assignment.get("values") == "finite_extrema"
        ):
            raise ValueError("pinned eager FFT overflow at finite dtype extrema is not replay-stable")
        if overload == "aten.atan2.out" and assignment.get("values") == "nan":
            raise ValueError("NaN output does not provide a replay-stable value-based mutation witness for atan2.out")
        args, kwargs = _apply_assignment(overload, canonical_case, assignment, baseline, domains)
        _validate_assignment_witness(overload, args, kwargs, assignment, baseline)
        first = case_document_from_arguments(
            overload,
            args,
            kwargs,
            comparison=str(canonical_case["comparison"]),
            rng_seed=int(canonical_case["rng_seed"]),
        )
        args, kwargs = _apply_assignment(overload, canonical_case, assignment, baseline, domains)
        _validate_assignment_witness(overload, args, kwargs, assignment, baseline)
        second = case_document_from_arguments(
            overload,
            args,
            kwargs,
            comparison=str(canonical_case["comparison"]),
            rng_seed=int(canonical_case["rng_seed"]),
        )
        if json.dumps(first, sort_keys=True, separators=(",", ":")) != json.dumps(
            second, sort_keys=True, separators=(",", ":")
        ):
            raise RuntimeError("identical eager reruns produced different case documents")
    except Exception as exc:  # noqa: BLE001 -- every invalid grammar cell belongs in the audit ledger
        message = str(exc)[:1000]
        # Some native PyTorch diagnostics accidentally expose process-local pointer-sized integers.
        # Preserve the diagnostic structure but remove values that make identical builds hash differently.
        message = mask_runtime_values(message)
        return None, f"{type(exc).__name__}: {message}"
    full_assignment = dict(sorted(assignment.items()))
    identifier = _case_id(overload, full_assignment)
    return {
        **first,
        "case_id": identifier,
        "partition_assignment": full_assignment,
        "covered_obligations": sorted(_obligations(overload, full_assignment)),
    }, None


def _one_overload_pool(
    overload: str,
    canonical_case: Mapping[str, Any],
    profile: BoundedCoverageProfile,
) -> dict[str, Any]:
    baseline, proposed_domains = _factor_domains(overload, canonical_case, profile)
    accepted: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    budget_exhausted = False

    def attempt(assignment: Mapping[str, str], phase: str) -> bool:
        nonlocal budget_exhausted
        normalized = dict(sorted(assignment.items()))
        identifier = _case_id(overload, normalized)
        if identifier in accepted:
            return True
        if len(accepted) >= profile.maximum_candidates_per_overload:
            budget_exhausted = True
            rejected.append(
                {
                    "phase": f"{phase}_budget",
                    "assignment": normalized,
                    "reason": f"candidate budget {profile.maximum_candidates_per_overload} exhausted",
                }
            )
            return False
        case, reason = _attempt(overload, canonical_case, normalized, baseline, proposed_domains)
        if case is None:
            rejected.append({"phase": phase, "assignment": normalized, "reason": reason})
            return False
        accepted[identifier] = case
        return True

    if not attempt(baseline, "baseline"):
        raise RuntimeError(f"canonical case stopped executing for {overload}: {rejected[-1]['reason']}")

    feasible_domains: dict[str, list[str]] = {}
    for factor in sorted(proposed_domains):
        feasible = [baseline[factor]]
        for value in sorted(proposed_domains[factor]):
            if value == baseline[factor]:
                continue
            assignment = {**baseline, factor: value}
            if attempt(assignment, "single"):
                feasible.append(value)
        feasible_domains[factor] = sorted(set(feasible))

    pair_attempts = 0
    target_pairs = {
        _pair_key(left, left_value, right, right_value)
        for left, right in itertools.combinations(sorted(feasible_domains), 2)
        for left_value in feasible_domains[left]
        for right_value in feasible_domains[right]
        if left_value != baseline[left] and right_value != baseline[right]
    }
    witnessed_pairs: set[tuple[str, str, str, str]] = set()

    def record_pairs(assignment: Mapping[str, str]) -> None:
        for left, right in itertools.combinations(sorted(assignment), 2):
            if assignment[left] != baseline[left] and assignment[right] != baseline[right]:
                witnessed_pairs.add(_pair_key(left, assignment[left], right, assignment[right]))

    # Dense rows make the selected suite small; exact SMT set cover below still proves the result
    # for the concrete eager-valid pool rather than trusting this greedy construction.
    for assignment in _greedy_pairwise_rows(feasible_domains, baseline):
        pair_attempts += 1
        if len(accepted) >= profile.maximum_candidates_per_overload:
            budget_exhausted = True
            break
        if attempt(assignment, "covering_array"):
            record_pairs(assignment)

    # A dense row can be invalid even though one of its constituent interactions is valid. Probe
    # every still-unwitnessed pair with all other factors at baseline, preserving an exact rejection
    # reason when PyTorch refuses that bounded cell.
    for left, left_value, right, right_value in sorted(target_pairs - witnessed_pairs):
        assignment = {**baseline, left: left_value, right: right_value}
        pair_attempts += 1
        if len(accepted) >= profile.maximum_candidates_per_overload:
            budget_exhausted = True
            rejected.append(
                {
                    "phase": "pair_budget",
                    "assignment": dict(sorted(assignment.items())),
                    "reason": f"candidate budget {profile.maximum_candidates_per_overload} exhausted",
                }
            )
            continue
        if attempt(assignment, "pair_fallback"):
            record_pairs(assignment)

    # Repack the witnessed feasible interactions into dense, eagerly validated rows. The fallback
    # cases above establish feasibility without guessing; these rows are what let exact set cover
    # discard most of those probes. Each coordinate is added only if the entire partial row still
    # executes deterministically, which naturally handles conditional domains such as NaN being
    # meaningful for floating dtypes but not Boolean tensors.
    uncovered_pairs = set(witnessed_pairs)
    packing_rows = 0
    while uncovered_pairs and len(accepted) < profile.maximum_candidates_per_overload:
        left, left_value, right, right_value = min(uncovered_pairs)
        row = {**baseline, left: left_value, right: right_value}
        for factor in sorted(feasible_domains):
            if factor in {left, right}:
                continue
            ranked = []
            for value in feasible_domains[factor]:
                trial = {**row, factor: value}
                score = sum(
                    _pair_key(factor, value, assigned_factor, assigned_value) in uncovered_pairs
                    for assigned_factor, assigned_value in row.items()
                )
                ranked.append((score, value == baseline[factor], value, trial))
            for _score, _baseline_preference, _value, trial in sorted(ranked, reverse=True):
                case, reason = _attempt(overload, canonical_case, trial, baseline, proposed_domains)
                if case is not None:
                    row = trial
                    break
                rejected.append(
                    {
                        "phase": "packing_probe",
                        "assignment": dict(sorted(trial.items())),
                        "reason": reason,
                    }
                )
        if not attempt(row, "packed_pairwise"):
            # The interaction was witnessed by an already accepted dense row, but need not remain
            # feasible after every other factor is reset to baseline. Keep that original concrete
            # witness instead of incorrectly treating conditional feasibility as nondeterminism.
            witness = next(
                (
                    case
                    for case in sorted(accepted.values(), key=lambda item: item["case_id"])
                    if case["partition_assignment"].get(left) == left_value
                    and case["partition_assignment"].get(right) == right_value
                ),
                None,
            )
            if witness is None:
                raise RuntimeError(f"accepted candidates lost witnessed pair for {overload}: {min(uncovered_pairs)}")
            before = len(uncovered_pairs)
            witness_assignment = witness["partition_assignment"]
            for factor_a, factor_b in itertools.combinations(sorted(witness_assignment), 2):
                uncovered_pairs.discard(
                    _pair_key(
                        factor_a,
                        witness_assignment[factor_a],
                        factor_b,
                        witness_assignment[factor_b],
                    )
                )
            if len(uncovered_pairs) == before:
                raise RuntimeError(f"existing pairwise witness made no progress for {overload}")
            continue
        packing_rows += 1
        before = len(uncovered_pairs)
        for factor_a, factor_b in itertools.combinations(sorted(row), 2):
            uncovered_pairs.discard(_pair_key(factor_a, row[factor_a], factor_b, row[factor_b]))
        if len(uncovered_pairs) == before:
            raise RuntimeError(f"pairwise packing made no progress for {overload}")
    if uncovered_pairs:
        budget_exhausted = True

    raw_candidate_count = len(accepted)
    accepted, eqsat = quotient_observational_equivalents(accepted)
    witnessed_obligations = sorted(
        {obligation for case in accepted.values() for obligation in case["covered_obligations"]}
    )
    cover = exact_minimum_cover(
        witnessed_obligations,
        {identifier: case["covered_obligations"] for identifier, case in accepted.items()},
    )
    if cover.status != "optimal" or not cover.selected_union_matches_denominator:
        raise RuntimeError(f"exact bounded cover failed for {overload}: {cover.to_dict()}")
    return {
        "overload": overload,
        "baseline": dict(sorted(baseline.items())),
        "feasible_domains": dict(sorted(feasible_domains.items())),
        "raw_candidate_count": raw_candidate_count,
        "candidate_count": len(accepted),
        "eqsat": eqsat,
        "pair_attempt_count": pair_attempts,
        "packed_pairwise_row_count": packing_rows,
        "candidate_budget_exhausted": budget_exhausted,
        "witnessed_obligation_count": len(witnessed_obligations),
        "witnessed_obligations": witnessed_obligations,
        "selected_count": cover.selected_count,
        "selected_case_ids": cover.selected_capsules,
        "cover": cover.to_dict(),
        "candidates": dict(sorted(accepted.items())),
        "rejected": rejected,
    }


def bounded_core_aten_suite(
    denominator: Sequence[str],
    *,
    pytorch_version: str,
    profile: BoundedCoverageProfile | None = None,
) -> dict[str, Any]:
    """Generate, eagerly validate, and exactly reduce the bounded Core ATen suite."""

    profile = profile or BoundedCoverageProfile()
    registry = core_aten_cases()
    names = sorted(set(str(name) for name in denominator))
    if names != list(registry):
        raise RuntimeError("bounded suite denominator does not equal the canonical Core ATen registry")
    canonical = {name: case_document(registry[name]) for name in names}
    pools = {name: _one_overload_pool(name, canonical[name], profile) for name in names}
    return assemble_bounded_core_aten_suite(
        names,
        pytorch_version=pytorch_version,
        profile=profile,
        pools=pools,
    )


def assemble_bounded_core_aten_suite(
    denominator: Sequence[str],
    *,
    pytorch_version: str,
    profile: BoundedCoverageProfile,
    pools: Mapping[str, Mapping[str, Any]],
    generator_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate independently generated per-overload pools and assemble the sealed suite."""

    names = sorted(set(str(name) for name in denominator))
    if set(pools) != set(names):
        raise RuntimeError(
            f"bounded pool registry mismatch: missing={sorted(set(names) - set(pools))}, "
            f"extra={sorted(set(pools) - set(names))}"
        )
    for name in names:
        if pools[name].get("overload") != name:
            raise RuntimeError(f"bounded pool identity mismatch for {name}")
        cover = pools[name].get("cover")
        if not isinstance(cover, Mapping) or cover.get("status") != "optimal":
            raise RuntimeError(f"bounded pool has no optimal cover certificate for {name}")
        objective = int(cover.get("objective_value", -1))
        if (
            not cover.get("objective_bounds_closed")
            or cover.get("minimum_cardinality_lower_bound") != objective
            or cover.get("minimum_cardinality_upper_bound") != objective
        ):
            raise RuntimeError(f"bounded pool has no closed minimum-cardinality bounds for {name}")
    selected = [
        pools[name]["candidates"][identifier] for name in names for identifier in pools[name]["selected_case_ids"]
    ]
    selected.sort(key=lambda case: case["case_id"])
    all_obligations = sorted({obligation for pool in pools.values() for obligation in pool["witnessed_obligations"]})
    selected_union = sorted({obligation for case in selected for obligation in case["covered_obligations"]})
    if selected_union != all_obligations:
        raise RuntimeError("selected bounded-suite union does not equal witnessed obligations")
    candidate_count = sum(pool["candidate_count"] for pool in pools.values())
    raw_candidate_count = sum(pool["raw_candidate_count"] for pool in pools.values())
    rejected_count = sum(len(pool["rejected"]) for pool in pools.values())
    budget_exhausted = sorted(name for name, pool in pools.items() if pool["candidate_budget_exhausted"])
    solver_identities = {
        (str(pool["cover"]["solver"]), str(pool["cover"]["solver_version"])) for pool in pools.values()
    }
    eqsat_versions = {str(pool["eqsat"]["xdsl_version"]) for pool in pools.values()}
    if len(solver_identities) != 1 or len(eqsat_versions) != 1:
        raise RuntimeError("bounded pools disagree on solver or equality-saturation engine identity")
    solver_name, solver_version = next(iter(solver_identities))
    document = {
        "schema_version": 1,
        "scope": "bounded Core ATen semantic partitions and pairwise interactions",
        "claim": "complete exact minimum over witnessed feasible cells in the recorded bounded grammar",
        "exclusions": [
            "not a proof over arbitrary tensor sizes, values, programs, dynamic shapes, or models",
            "eager-invalid and generator-unwitnessed cells are retained as rejected attempts, not covered",
            "target-specific tile, memory-capacity, DMA, and numeric-datapath edges are outside this profile",
        ],
        "pytorch_version": pytorch_version,
        "denominator_sha256": overload_digest(names),
        "overload_count": len(names),
        "profile": profile.to_dict(),
        "profile_sha256": profile.sha256,
        "generator_sha256": generator_sha256,
        "engines": {
            "set_cover": solver_name,
            "set_cover_version": solver_version,
            "minimum_certificate": "Z3 Optimize lower and upper cardinality bounds are equal for every overload",
            "observational_quotient": "xdsl-eqsat",
            "observational_quotient_version": next(iter(eqsat_versions)),
            "eqsat_rewrite_rules": [],
        },
        "candidate_count": candidate_count,
        "raw_candidate_count": raw_candidate_count,
        "eqsat_eliminated_candidate_count": raw_candidate_count - candidate_count,
        "eqsat_nontrivial_eclass_count": sum(int(pool["eqsat"]["nontrivial_eclass_count"]) for pool in pools.values()),
        "selected_count": len(selected),
        "witnessed_obligation_count": len(all_obligations),
        "witnessed_obligations": all_obligations,
        "rejected_attempt_count": rejected_count,
        "candidate_budget_exhausted_overloads": budget_exhausted,
        "complete": not budget_exhausted and selected_union == all_obligations,
        "selected_union_sha256": overload_digest(all_obligations),
        "selected_cases": selected,
        "overloads": pools,
    }
    document["suite_sha256"] = bounded_suite_digest(document)
    return document


def bounded_summary(document: Mapping[str, Any]) -> str:
    """Render the concise, publication-safe summary stored beside the JSON artifact."""

    exhausted = list(document["candidate_budget_exhausted_overloads"])
    status = "complete" if document["complete"] else "incomplete"
    lines = [
        "# Bounded Core ATen semantic coverage",
        "",
        f"Status: **{status}**.",
        "",
        f"- PyTorch: `{document['pytorch_version']}`",
        f"- Core ATen overloads: {document['overload_count']}",
        f"- Generated eager-valid candidates: {document['candidate_count']}",
        f"- Pre-eqsat eager-valid candidates: {document['raw_candidate_count']}",
        f"- Exact observational duplicates removed by xDSL eqsat: {document['eqsat_eliminated_candidate_count']}",
        f"- Exact minimum selected cases: {document['selected_count']}",
        f"- Witnessed single/pair obligations: {document['witnessed_obligation_count']}",
        f"- Retained rejected attempts: {document['rejected_attempt_count']}",
        f"- Profile digest: `{document['profile_sha256']}`",
        f"- Generator digest: `{document['generator_sha256']}`",
        f"- Minimum certificate: {document['engines']['minimum_certificate']}",
        "",
        "The selected set is the proven exact minimum only over the recorded eager-valid candidate pool. "
        "Completeness means every witnessed feasible single partition and pairwise interaction in "
        "the bounded grammar is covered. It does not prove correctness for arbitrary PyTorch models.",
        "",
    ]
    if exhausted:
        lines.extend(
            [
                "## Candidate-budget gaps",
                "",
                *[f"- `{name}`" for name in exhausted],
                "",
            ]
        )
    lines.extend(
        [
            "## Bound",
            "",
            "- ranks 0–4; extents drawn from 0, 1, 2, 3, and 5; at most 256 elements",
            "- bool, int8, int32, int64, float16, bfloat16, float32, and complex64 where eager-valid",
            "- contiguous, padded non-contiguous, nonzero-offset, zero-stride, and channels-last layouts",
            "- ordinary, zero, sign, finite-extrema, NaN, infinity, tie, and signed-zero values",
            "- finite Boolean/enumerated controls and selected default/non-default numeric controls",
            "- all witnessed two-factor interactions; no unbounded Cartesian-product claim",
            "",
        ]
    )
    return "\n".join(lines)
