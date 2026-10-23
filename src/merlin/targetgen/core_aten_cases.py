"""Target-agnostic executable cases for the complete Core ATen denominator.

The case corpus is intentionally independent of model2MLIR and every hardware target.  A case exists
when its exact :class:`torch._ops.OpOverload` can be invoked eagerly with fresh deterministic inputs.
Import, lowering, target admission, and hardware execution are later result dimensions; none is
allowed to erase a source-level test from this corpus.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

Arguments = tuple[tuple[Any, ...], dict[str, Any]]


@dataclass(frozen=True)
class CoreAtenCase:
    overload: str
    make_arguments: Callable[[], Arguments]
    comparison: str = "torch_close"


def resolve_overload(name: str):
    """Resolve ``aten.packet.overload`` without collapsing overload identity."""

    import torch

    namespace, packet, overload = name.split(".", 2)
    if namespace != "aten":
        raise ValueError(f"not an ATen overload: {name}")
    return getattr(getattr(torch.ops.aten, packet), overload)


def _args(*positional: Any, **keywords: Any) -> Callable[[], Arguments]:
    def clone(value: Any) -> Any:
        # Each execution must receive independent mutable inputs.  In particular, an
        # in-place case must not contaminate the next backend or reproducibility run.
        try:
            import torch

            if isinstance(value, torch.Tensor):
                result = value.detach().clone(memory_format=torch.preserve_format)
                result.requires_grad_(value.requires_grad)
                return result
        except ImportError:  # pragma: no cover - the corpus is only executable with PyTorch
            pass
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return value

    return lambda: (clone(positional), clone(keywords))


def _factory(function: Callable[[], Arguments]) -> Callable[[], Arguments]:
    return function


def _cases() -> dict[str, CoreAtenCase]:  # noqa: C901 -- the explicit inventory is the point
    import torch

    cases: dict[str, CoreAtenCase] = {}

    def add(name: str, make: Callable[[], Arguments], comparison: str = "torch_close") -> None:
        if name in cases:
            raise RuntimeError(f"duplicate Core ATen case: {name}")
        cases[name] = CoreAtenCase(name, make, comparison)

    def f(shape: tuple[int, ...] = (2, 3)):
        count = 1
        for extent in shape:
            count *= extent
        return torch.linspace(-0.75, 1.25, count, dtype=torch.float32).reshape(shape)

    def positive(shape: tuple[int, ...] = (2, 3)):
        return f(shape).abs() + 0.5

    def integer(shape: tuple[int, ...] = (2, 3)):
        count = 1
        for extent in shape:
            count *= extent
        return torch.arange(count, dtype=torch.int64).reshape(shape)

    def boolean(shape: tuple[int, ...] = (2, 3)):
        return integer(shape).remainder(2).bool()

    for name in (
        "aten.abs.default",
        "aten.asinh.default",
        "aten.atan.default",
        "aten.ceil.default",
        "aten.cos.default",
        "aten.cosh.default",
        "aten.erf.default",
        "aten.exp.default",
        "aten.expm1.default",
        "aten.floor.default",
        "aten.isinf.default",
        "aten.isnan.default",
        "aten.neg.default",
        "aten.relu.default",
        "aten.round.default",
        "aten.sigmoid.default",
        "aten.sign.default",
        "aten.sin.default",
        "aten.sinh.default",
        "aten.tan.default",
        "aten.tanh.default",
        "aten.trunc.default",
    ):
        add(name, lambda f=f: ((f(),), {}))
    for name in (
        "aten.acos.default",
        "aten.asin.default",
        "aten.atanh.default",
    ):
        add(name, lambda f=f: ((f().clamp(-0.75, 0.75),), {}))
    for name in (
        "aten.acosh.default",
        "aten.log.default",
        "aten.log10.default",
        "aten.log2.default",
        "aten.reciprocal.default",
        "aten.rsqrt.default",
        "aten.sqrt.default",
    ):
        add(name, lambda positive=positive: ((positive(),), {}))
    add("aten.log1p.default", lambda f=f: ((f().clamp_min(-0.75),), {}))
    add("aten.bitwise_not.default", lambda integer=integer: ((integer(),), {}))
    add("aten.logical_not.default", lambda boolean=boolean: ((boolean(),), {}))

    for packet in ("add", "sub", "mul"):
        add(
            f"aten.{packet}.Tensor",
            lambda f=f, packet=packet: ((f(), f().flip(1)), {"alpha": 2} if packet != "mul" else {}),
        )
        add(
            f"aten.{packet}.Scalar",
            lambda f=f, packet=packet: ((f(), 0.5), {"alpha": 2} if packet != "mul" else {}),
        )
    for packet in ("eq", "ge", "gt", "le", "lt", "ne"):
        add(f"aten.{packet}.Tensor", lambda f=f: ((f(), f().flip(1)), {}))
        add(f"aten.{packet}.Scalar", lambda f=f: ((f(), 0.25), {}))
    for packet in ("bitwise_and", "bitwise_or", "bitwise_xor"):
        add(f"aten.{packet}.Tensor", lambda integer=integer: ((integer(), integer().flip(1)), {}))
        add(f"aten.{packet}.Scalar", lambda integer=integer: ((integer(), 3), {}))
    for packet in ("logical_and", "logical_or", "logical_xor"):
        add(f"aten.{packet}.default", lambda boolean=boolean: ((boolean(), boolean().flip(1)), {}))
    for packet in ("maximum", "minimum"):
        add(f"aten.{packet}.default", lambda f=f: ((f(), f().flip(1)), {}))
    for packet in ("fmod", "remainder"):
        add(f"aten.{packet}.Tensor", lambda f=f: ((f() + 2.0, positive()), {}))
        add(f"aten.{packet}.Scalar", lambda f=f: ((f() + 2.0, 1.25), {}))
    add("aten.div.Tensor", lambda f=f: ((f(), positive()), {}))
    add("aten.div.Scalar", lambda f=f: ((f(), 2.0), {}))
    add(
        "aten.div.Tensor_mode",
        lambda integer=integer: ((integer() + 1, integer().flip(1) + 1), {"rounding_mode": "floor"}),
    )
    add("aten.div.Scalar_mode", lambda integer=integer: ((integer() + 1, 2), {"rounding_mode": "floor"}))
    add("aten.atan2.default", lambda f=f: ((f(), f().flip(1) + 0.25), {}))
    add("aten.atan2.out", lambda f=f: ((f(), f().flip(1) + 0.25), {"out": torch.zeros(2, 3)}))
    add("aten.pow.Scalar", lambda f=f: ((2.0, f().abs() + 1.0), {}))
    add("aten.pow.Tensor_Scalar", lambda positive=positive: ((positive(), 2.0), {}))
    add("aten.pow.Tensor_Tensor", lambda positive=positive: ((positive(), positive().flip(1)), {}))
    add("aten.where.self", lambda boolean=boolean, f=f: ((boolean(), f(), f().flip(1)), {}))

    add("aten._adaptive_avg_pool2d.default", lambda f=f: ((f((1, 2, 4, 4)), [2, 2]), {}))
    add("aten._adaptive_avg_pool3d.default", lambda f=f: ((f((1, 1, 4, 4, 4)), [2, 2, 2]), {}))
    add("aten.adaptive_avg_pool1d.default", lambda f=f: ((f((1, 2, 6)), [3]), {}))
    add("aten.avg_pool1d.default", lambda f=f: ((f((1, 2, 6)), [2], [2], [0], False, True), {}))
    add("aten.avg_pool2d.default", lambda f=f: ((f((1, 2, 4, 4)), [2, 2], [2, 2], [0, 0], False, True, None), {}))
    add(
        "aten.avg_pool3d.default",
        lambda f=f: ((f((1, 1, 4, 4, 4)), [2, 2, 2], [2, 2, 2], [0, 0, 0], False, True, None), {}),
    )
    add(
        "aten.max_pool2d_with_indices.default",
        lambda f=f: ((f((1, 1, 4, 4)), [2, 2], [2, 2], [0, 0], [1, 1], False), {}),
    )
    add(
        "aten.max_pool3d_with_indices.default",
        lambda f=f: ((f((1, 1, 4, 4, 4)), [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], False), {}),
    )
    add("aten.upsample_bilinear2d.vec", lambda f=f: ((f((1, 1, 3, 3)), [5, 5], False, None), {}))
    add("aten.upsample_nearest2d.vec", lambda f=f: ((f((1, 1, 3, 3)), [6, 6], None), {}))

    add("aten.mm.default", lambda f=f: ((f((3, 4)), f((4, 2))), {}))
    add("aten.bmm.default", lambda f=f: ((f((2, 3, 4)), f((2, 4, 2))), {}))
    add("aten.addmm.default", lambda f=f: ((f((3, 2)), f((3, 4)), f((4, 2))), {"beta": 1.5, "alpha": 0.5}))
    add(
        "aten.convolution.default",
        lambda f=f: ((f((1, 1, 5, 5)), f((2, 1, 3, 3)), torch.zeros(2), [1, 1], [0, 0], [1, 1], False, [0, 0], 1), {}),
    )
    add("aten.col2im.default", lambda f=f: ((f((1, 4, 4)), [3, 3], [2, 2], [1, 1], [0, 0], [1, 1]), {}))
    add("aten.grid_sampler_2d.default", lambda f=f: ((f((1, 1, 2, 2)), f((1, 2, 2, 2)).clamp(-1, 1), 0, 0, True), {}))

    add("aten._fft_r2c.default", lambda f=f: ((f((2, 4)), [1], 0, True), {}))
    add("aten._fft_c2r.default", lambda f=f: ((torch.complex(f((2, 3)), f((2, 3)).flip(1)), [1], 0, 4), {}))
    add("aten._cdist_forward.default", lambda f=f: ((f((2, 3)), f((4, 3)), 2.0, None), {}))
    add("aten._pdist_forward.default", lambda f=f: ((f((4, 3)), 2.0), {}))

    add("aten._local_scalar_dense.default", _args(torch.tensor(2.5)))
    add("aten._log_softmax.default", lambda f=f: ((f(), 1, False), {}))
    add("aten._softmax.default", lambda f=f: ((f(), 1, False), {}))
    add("aten._to_copy.default", lambda f=f: ((f(),), {"dtype": torch.float64}))
    add("aten.alias.default", lambda f=f: ((f(),), {}))
    add("aten.clone.default", lambda f=f: ((f(),), {"memory_format": torch.contiguous_format}))
    add("aten.copy.default", lambda f=f: ((f(), f().flip(1), False), {}))
    add("aten.fill.Scalar", lambda f=f: ((f(), 3.0), {}))
    add("aten.resize_.default", lambda f=f: ((f(), [3, 2]), {}), comparison="metadata")

    add("aten.amax.default", lambda f=f: ((f(), [0], True), {}))
    add("aten.amin.default", lambda f=f: ((f(), [1], False), {}))
    add("aten.any.default", lambda boolean=boolean: ((boolean(),), {}))
    add("aten.any.dim", lambda boolean=boolean: ((boolean(), 1, True), {}))
    add("aten.any.dims", lambda boolean=boolean: ((boolean(), [0, 1], False), {}))
    add("aten.argmax.default", lambda f=f: ((f(), 1, False), {}))
    add("aten.argmin.default", lambda f=f: ((f(), 0, True), {}))
    add("aten.max.dim", lambda f=f: ((f(), 1, False), {}))
    add("aten.min.dim", lambda f=f: ((f(), 0, True), {}))
    add("aten.mean.default", lambda f=f: ((f(),), {}))
    add("aten.mean.dim", lambda f=f: ((f(), [1], True), {}))
    add("aten.prod.default", lambda positive=positive: ((positive(),), {}))
    add("aten.prod.dim_int", lambda positive=positive: ((positive(), 1, True), {}))
    add("aten.sum.dim_IntList", lambda f=f: ((f(), [0], False), {}))
    add("aten.var.correction", lambda f=f: ((f(), [1]), {"correction": 1, "keepdim": True}))
    add("aten.var.dim", lambda f=f: ((f(), [0], False, False), {}))
    add("aten.cumsum.default", lambda f=f: ((f(), 1), {}))
    add("aten.sort.default", lambda f=f: ((f(), 1, True), {}))
    add("aten.topk.default", lambda f=f: ((f(), 2, 1, True, True), {}))
    add("aten.nonzero.default", lambda f=f: (((f() > 0),), {}))

    add("aten.arange.start_step", _args(1, 7, 2, dtype=torch.int64))
    add(
        "aten.empty.memory_format",
        _args([2, 3], dtype=torch.float32, memory_format=torch.contiguous_format),
        comparison="metadata",
    )
    add("aten.empty_strided.default", _args([2, 3], [3, 1], dtype=torch.float32), comparison="metadata")
    add("aten.full.default", _args([2, 3], 1.25, dtype=torch.float32))
    add("aten.full_like.default", lambda f=f: ((f(), 1.25), {"dtype": torch.float64}))
    add("aten.rand.default", _args([2, 3], dtype=torch.float32))
    add("aten.randn.default", _args([2, 3], dtype=torch.float32))
    add("aten.randperm.default", _args(7, dtype=torch.int64))
    add("aten.scalar_tensor.default", _args(2.5, dtype=torch.float32))

    add("aten.as_strided.default", lambda f=f: ((f((3, 4)), [2, 2], [4, 1], 1), {}))
    add("aten.diagonal.default", lambda f=f: ((f((3, 4)), 0, 0, 1), {}))
    add("aten.expand.default", lambda f=f: ((f((1, 3)), [2, 3]), {"implicit": False}))
    add("aten.flip.default", lambda f=f: ((f(), [1]), {}))
    add("aten.permute.default", lambda f=f: ((f((2, 3, 4)), [2, 0, 1]), {}))
    add("aten.repeat.default", lambda f=f: ((f(), [2, 1]), {}))
    add("aten.select.int", lambda f=f: ((f((2, 3, 4)), 1, 1), {}))
    add("aten.slice.Tensor", lambda f=f: ((f((2, 5)), 1, 1, 5, 2), {}))
    add("aten.split_with_sizes.default", lambda f=f: ((f((2, 5)), [2, 3], 1), {}))
    add("aten.squeeze.dim", lambda f=f: ((f((2, 1, 3)), 1), {}))
    add("aten.squeeze.dims", lambda f=f: ((f((1, 2, 1, 3)), [0, 2]), {}))
    add("aten.unsqueeze.default", lambda f=f: ((f(), 1), {}))
    add("aten.view.default", lambda f=f: ((f((2, 3, 4)), [4, 6]), {}))
    add("aten.cat.default", lambda f=f: (([f(), f().flip(1)], 0), {}))
    add("aten.constant_pad_nd.default", lambda f=f: ((f((1, 2, 3)), [1, 1, 1, 1], 0.5), {}))
    add("aten.reflection_pad1d.default", lambda f=f: ((f((1, 1, 4)), [1, 1]), {}))
    add("aten.reflection_pad2d.default", lambda f=f: ((f((1, 1, 3, 3)), [1, 1, 1, 1]), {}))
    add("aten.reflection_pad3d.default", lambda f=f: ((f((1, 1, 3, 3, 3)), [1, 1, 1, 1, 1, 1]), {}))
    add("aten.replication_pad2d.default", lambda f=f: ((f((1, 1, 3, 3)), [1, 1, 1, 1]), {}))
    add("aten.replication_pad3d.default", lambda f=f: ((f((1, 1, 3, 3, 3)), [1, 1, 1, 1, 1, 1]), {}))

    add("aten.clamp.default", lambda f=f: ((f(), -0.25, 0.75), {}))
    add("aten.clamp.Tensor", lambda f=f: ((f(), torch.full((2, 3), -0.25), torch.full((2, 3), 0.75)), {}))
    add("aten.elu.default", lambda f=f: ((f(), 1.0, 1.0, 1.0), {}))
    add("aten.gelu.default", lambda f=f: ((f(),), {"approximate": "none"}))
    add("aten.hardtanh.default", lambda f=f: ((f(), -0.5, 0.5), {}))
    add("aten.leaky_relu.default", lambda f=f: ((f(), 0.1), {}))

    def embedding_bag():
        return ((f((5, 3)), torch.tensor([0, 1, 2, 1]), torch.tensor([0, 2]), False, 0, False, None, False, -1), {})

    add("aten._embedding_bag.default", _factory(embedding_bag))
    add("aten.embedding.default", lambda f=f: ((f((5, 3)), torch.tensor([[0, 2], [1, 3]]), -1, False, False), {}))
    add(
        "aten.embedding_dense_backward.default", lambda f=f: ((f((4, 3)), torch.tensor([0, 2, 1, 2]), 5, -1, False), {})
    )

    def batch_norm(training: bool, with_stats: bool):
        args: list[Any] = [f((2, 4, 2, 2)), torch.ones(4), torch.zeros(4)]
        if with_stats:
            args.extend([torch.zeros(4), torch.ones(4)])
        args.extend([training, 0.1, 1e-5])
        return (tuple(args), {})

    add("aten._native_batch_norm_legit.default", lambda: batch_norm(True, True))
    add("aten._native_batch_norm_legit.no_stats", lambda: batch_norm(True, False))
    add(
        "aten._native_batch_norm_legit_no_training.default",
        lambda: ((f((2, 4, 2, 2)), torch.ones(4), torch.zeros(4), torch.zeros(4), torch.ones(4), 0.1, 1e-5), {}),
    )
    add("aten.native_dropout.default", lambda f=f: ((f(), 0.25, True), {}))
    add(
        "aten.native_group_norm.default",
        lambda f=f: ((f((2, 4, 2, 2)), torch.ones(4), torch.zeros(4), 2, 4, 4, 2, 1e-5), {}),
    )
    add("aten.native_layer_norm.default", lambda f=f: ((f((2, 3)), [3], torch.ones(3), torch.zeros(3), 1e-5), {}))

    add("aten.gather.default", lambda f=f: ((f(), 1, torch.tensor([[2, 1], [0, 2]])), {"sparse_grad": False}))
    add("aten.index.Tensor", lambda f=f: ((f((4, 3)), [torch.tensor([2, 0])]), {}))
    add("aten.index_put.default", lambda f=f: ((f((4, 3)), [torch.tensor([1, 3])], torch.ones(2, 3), False), {}))
    add("aten.index_select.default", lambda f=f: ((f((4, 3)), 0, torch.tensor([2, 0])), {}))
    add(
        "aten.masked_scatter.default",
        lambda f=f, boolean=boolean: ((f(), boolean(), torch.arange(6, dtype=torch.float32)), {}),
    )
    add("aten.scatter.src", lambda f=f: ((f(), 1, torch.tensor([[0, 2], [1, 0]]), torch.ones(2, 2)), {}))
    add("aten.scatter.value", lambda f=f: ((f(), 1, torch.tensor([[0, 2], [1, 0]]), 2.0), {}))
    add("aten.scatter_add.default", lambda f=f: ((f(), 1, torch.tensor([[0, 2], [1, 0]]), torch.ones(2, 2)), {}))
    add(
        "aten.scatter_reduce.two",
        lambda f=f: ((f(), 1, torch.tensor([[0, 2], [1, 0]]), torch.ones(2, 2), "sum"), {"include_self": True}),
    )
    add("aten.select_scatter.default", lambda f=f: ((f((2, 3)), torch.ones(2), 1, 1), {}))
    add("aten.slice_scatter.default", lambda f=f: ((f((2, 5)), torch.ones(2, 2), 1, 1, 5, 2), {}))

    add("aten.sym_is_contiguous.default", lambda f=f: ((f(), torch.contiguous_format), {}))
    add("aten.sym_numel.default", lambda f=f: ((f(),), {}))
    add("aten.sym_size.int", lambda f=f: ((f(), 1), {}))
    add("aten.sym_storage_offset.default", lambda f=f: ((f(),), {}))
    add("aten.sym_stride.int", lambda f=f: ((f(), 0), {}))

    def adaptive_backward():
        x = f((1, 1, 4, 4))
        return ((torch.ones(1, 1, 2, 2), x), {})

    add("aten._adaptive_avg_pool2d_backward.default", _factory(adaptive_backward))
    add(
        "aten.avg_pool2d_backward.default",
        lambda f=f: ((torch.ones(1, 1, 2, 2), f((1, 1, 4, 4)), [2, 2], [2, 2], [0, 0], False, True, None), {}),
    )
    add(
        "aten.convolution_backward.default",
        lambda f=f: (
            (
                torch.ones(1, 2, 3, 3),
                f((1, 1, 5, 5)),
                f((2, 1, 3, 3)),
                [2],
                [1, 1],
                [0, 0],
                [1, 1],
                False,
                [0, 0],
                1,
                [True, True, True],
            ),
            {},
        ),
    )

    def max_pool_backward():
        x = f((1, 1, 4, 4))
        _values, indices = torch.ops.aten.max_pool2d_with_indices.default(x, [2, 2], [2, 2], [0, 0], [1, 1], False)
        return ((torch.ones(1, 1, 2, 2), x, [2, 2], [2, 2], [0, 0], [1, 1], False, indices), {})

    add("aten.max_pool2d_with_indices_backward.default", _factory(max_pool_backward))

    def group_norm_backward():
        x = f((2, 4, 2, 2))
        weight = torch.ones(4)
        _out, mean, rstd = torch.ops.aten.native_group_norm.default(x, weight, torch.zeros(4), 2, 4, 4, 2, 1e-5)
        return ((torch.ones_like(x), x, mean, rstd, weight, 2, 4, 4, 2, [True, True, True]), {})

    add("aten.native_group_norm_backward.default", _factory(group_norm_backward))

    def layer_norm_backward():
        x = f((2, 3))
        weight, bias = torch.ones(3), torch.zeros(3)
        _out, mean, rstd = torch.ops.aten.native_layer_norm.default(x, [3], weight, bias, 1e-5)
        return ((torch.ones_like(x), x, [3], mean, rstd, weight, bias, [True, True, True]), {})

    add("aten.native_layer_norm_backward.default", _factory(layer_norm_backward))
    return cases


def core_aten_cases() -> dict[str, CoreAtenCase]:
    """Return a fresh, name-sorted complete case registry."""

    return dict(sorted(_cases().items()))


def _encode(value: Any, *, values: bool = True) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        encoded: dict[str, Any] = {
            "kind": "tensor",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "storage_offset": int(tensor.storage_offset()),
            "requires_grad": bool(value.requires_grad),
        }
        if values:
            encoded["values"] = _encode(tensor.tolist())
        return encoded
    if isinstance(value, torch.dtype):
        return {"kind": "dtype", "value": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"kind": "device", "value": str(value)}
    if isinstance(value, torch.memory_format):
        return {"kind": "memory_format", "value": str(value).removeprefix("torch.")}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_encode(item, values=values) for item in value]}
    if isinstance(value, list):
        return [_encode(item, values=values) for item in value]
    if isinstance(value, dict):
        return {str(key): _encode(item, values=values) for key, item in sorted(value.items())}
    if isinstance(value, complex):
        return {"kind": "complex", "real": _encode(value.real), "imag": _encode(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return {"kind": "float", "value": str(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot encode Core ATen case value {type(value).__name__}")


def decode_value(document: Any) -> Any:
    """Reconstruct a corpus value without depending on Merlin or a compiler.

    Tensor strides are part of the contract rather than incidental metadata.  Rebuilding through
    ``empty_strided`` retains them while ``copy_`` installs the recorded logical values.
    """

    import torch

    if isinstance(document, list):
        return [decode_value(item) for item in document]
    if not isinstance(document, dict) or "kind" not in document:
        if isinstance(document, dict):
            return {key: decode_value(value) for key, value in document.items()}
        return document
    kind = document["kind"]
    if kind == "tuple":
        return tuple(decode_value(item) for item in document["items"])
    if kind == "dtype":
        return getattr(torch, document["value"])
    if kind == "device":
        return torch.device(document["value"])
    if kind == "memory_format":
        return getattr(torch, document["value"])
    if kind == "complex":
        return complex(decode_value(document["real"]), decode_value(document["imag"]))
    if kind == "float":
        return float(document["value"])
    if kind == "tensor":
        if "values" not in document:
            raise ValueError("cannot decode a metadata-only tensor")
        dtype = getattr(torch, document["dtype"])
        shape = tuple(document["shape"])
        stride = tuple(document["stride"])
        storage_offset = int(document.get("storage_offset", 0))
        if storage_offset < 0:
            raise ValueError("tensor storage_offset must be non-negative")
        logical = torch.tensor(decode_value(document["values"]), dtype=dtype).reshape(shape)
        storage_size = (
            storage_offset
            if any(extent == 0 for extent in shape)
            else storage_offset + sum((extent - 1) * step for extent, step in zip(shape, stride)) + 1
        )
        storage = torch.empty(max(storage_size, 0), dtype=dtype)
        tensor = storage.as_strided(shape, stride, storage_offset)
        if torch._debug_has_internal_overlap(tensor) == 0:
            tensor.copy_(logical)
        else:
            # Expanded views can legitimately have a zero stride.  PyTorch rejects a bulk copy into
            # such a view, so populate its backing storage once per distinct physical element.
            seen: dict[int, Any] = {}
            for index in itertools.product(*(range(extent) for extent in shape)):
                offset = storage_offset + sum(i * step for i, step in zip(index, stride))
                scalar = logical[index]
                if offset in seen:
                    previous = seen[offset].contiguous().reshape(-1).view(torch.uint8)
                    current = scalar.contiguous().reshape(-1).view(torch.uint8)
                    if not torch.equal(previous, current):
                        raise ValueError("recorded logical values are inconsistent with overlapping strides")
                seen[offset] = scalar
                storage[offset] = scalar
            tensor = storage.as_strided(shape, stride, storage_offset)
        tensor.requires_grad_(bool(document.get("requires_grad", False)))
        return tensor
    raise ValueError(f"unknown Core ATen corpus value kind: {kind!r}")


def decode_arguments(document: Mapping[str, Any]) -> Arguments:
    """Decode one case's portable ``arguments`` member into fresh positional/keyword values."""

    args = decode_value(document["args"])
    kwargs = decode_value(document["kwargs"])
    if not isinstance(args, tuple) or not isinstance(kwargs, dict):
        raise ValueError("invalid Core ATen argument document")
    return args, kwargs


def invoke_case(case: CoreAtenCase) -> Any:
    import torch

    torch.manual_seed(0)
    args, kwargs = case.make_arguments()
    return resolve_overload(case.overload)(*args, **kwargs)


def _tensor_items(value: Any, path: str = "") -> list[tuple[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        return [(path or "$", value)]
    if isinstance(value, (tuple, list)):
        return [item for index, child in enumerate(value) for item in _tensor_items(child, f"{path}[{index}]")]
    if isinstance(value, dict):
        return [
            item
            for key, child in sorted(value.items())
            for item in _tensor_items(child, f"{path}.{key}" if path else str(key))
        ]
    return []


def _tensor_snapshot(tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "values": tensor.detach().cpu().clone(),
    }


def _changed(before: Mapping[str, Any], after) -> bool:
    import torch

    before_bytes = before["values"].contiguous().reshape(-1).view(torch.uint8)
    after_bytes = after.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return not (
        before["dtype"] == str(after.dtype)
        and before["shape"] == tuple(after.shape)
        and before["stride"] == tuple(after.stride())
        and torch.equal(before_bytes, after_bytes)
    )


def _aliases(left, right) -> bool:
    import torch

    if left.device != right.device:
        return False
    try:
        return bool(torch._C._is_alias_of(left, right))
    except (AttributeError, RuntimeError):
        return left.untyped_storage()._cdata == right.untyped_storage()._cdata


def case_document(case: CoreAtenCase) -> dict[str, Any]:
    """Execute one eager oracle and emit its portable audit description."""

    args, kwargs = case.make_arguments()
    return case_document_from_arguments(
        case.overload,
        args,
        kwargs,
        comparison=case.comparison,
        rng_seed=0,
    )


def case_document_from_arguments(
    overload: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    comparison: str = "torch_close",
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Execute an exact overload on supplied arguments and emit a portable oracle document.

    This is the common authoring primitive for the one-case-per-overload baseline and bounded
    semantic-partition variants. The arguments are consumed by the call and may therefore be
    mutated; callers that need to retain them must pass fresh values.
    """

    import torch

    torch.manual_seed(rng_seed)
    kwargs = dict(kwargs)
    argument_document = {"args": _encode(args), "kwargs": _encode(kwargs)}
    input_tensors = _tensor_items(args, "args") + _tensor_items(kwargs, "kwargs")
    before = {path: _tensor_snapshot(tensor) for path, tensor in input_tensors}
    result = resolve_overload(overload)(*args, **kwargs)
    include_values = comparison != "metadata"
    post_argument_document = {
        "args": _encode(args, values=include_values),
        "kwargs": _encode(kwargs, values=include_values),
    }
    result_document = _encode(result, values=include_values)
    result_tensors = _tensor_items(result, "result")
    aliases = sorted(
        {
            f"{result_path}->{input_path}"
            for result_path, output in result_tensors
            for input_path, input_ in input_tensors
            if _aliases(output, input_)
        }
    )
    mutated = sorted(path for path, tensor in input_tensors if _changed(before[path], tensor))
    expected_bytes = (
        json.dumps(result_document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    return {
        "overload": overload,
        "schema": str(resolve_overload(overload)._schema),
        "rng_seed": rng_seed,
        "comparison": comparison,
        "comparison_parameters": (
            {"rtol": 1e-5, "atol": 1e-6, "equal_nan": True} if comparison == "torch_close" else {}
        ),
        "arguments": argument_document,
        "post_arguments": post_argument_document,
        "mutated_arguments": mutated,
        "output_input_aliases": aliases,
        "expected": result_document,
        "expected_sha256": hashlib.sha256(expected_bytes).hexdigest(),
    }


def eager_case_observation(case: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a portable case and return the observation schema consumed by suite evaluators."""

    import torch

    torch.manual_seed(int(case["rng_seed"]))
    args, kwargs = decode_arguments(case["arguments"])
    input_tensors = _tensor_items(args, "args") + _tensor_items(kwargs, "kwargs")
    before = {path: _tensor_snapshot(tensor) for path, tensor in input_tensors}
    result = resolve_overload(str(case["overload"]))(*args, **kwargs)
    result_tensors = _tensor_items(result, "result")
    aliases = sorted(
        {
            f"{result_path}->{input_path}"
            for result_path, output in result_tensors
            for input_path, input_ in input_tensors
            if _aliases(output, input_)
        }
    )
    mutated = sorted(path for path, tensor in input_tensors if _changed(before[path], tensor))
    include_values = case["comparison"] != "metadata"
    return {
        "status": "executed",
        "execution_kind": "host",
        "output": _encode(result, values=include_values),
        "post_arguments": {
            "args": _encode(args, values=include_values),
            "kwargs": _encode(kwargs, values=include_values),
        },
        "mutated_arguments": mutated,
        "output_input_aliases": aliases,
        "evidence": {"engine": "torch_eager"},
    }


def corpus_document(denominator: Sequence[str], *, pytorch_version: str) -> dict[str, Any]:
    """Execute and audit the complete source-level corpus, failing on any omission or extra."""

    registry = core_aten_cases()
    expected = set(str(op) for op in denominator)
    observed = set(registry)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(f"Core ATen case registry mismatch: missing={missing}, extra={extra}")
    cases = [case_document(registry[name]) for name in sorted(registry)]
    return {
        "schema_version": 1,
        "scope": "executable eager cases for every overload tagged torch.Tag.core",
        "pytorch_version": pytorch_version,
        "case_count": len(cases),
        "overload_count": len(expected),
        "complete": len(cases) == len(expected),
        "cases": cases,
    }
