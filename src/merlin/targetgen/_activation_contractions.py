#!/usr/bin/env python3
"""Quantize the contractions ``quantize_`` cannot reach -- runs INSIDE the capture venv.

TorchAO's dynamic int8 schemes replace the weight of an ``nn.Linear``, and nothing else. A model's
other contractions keep their floating-point arithmetic under a program declared int8:

* a contraction of two ACTIVATIONS -- attention's ``Q @ K^T`` and ``P @ V`` -- has no module and no
  weight, so there is nothing for ``quantize_`` to replace. It reaches this module either as a
  ``matmul``/``bmm`` or inside a fused ``scaled_dot_product_attention``, which is spelled out here as
  its two matmuls around the masked softmax (the steps the capture decomposes it into anyway);
* a convolution is a contraction against a stored weight, but ``quantize_`` does not transform a
  ``Conv*`` module under these configurations.

On a datapath that consumes int8 on both operands those contractions cannot run at all, so a model
whose every Linear is int8 can still leave most of its multiply-accumulates on the host. This module
quantizes them in the SAME numeric form the scheme gives a Linear, so the program is uniform:

* the first operand is an activation, quantized dynamically, symmetric, one scale per ROW (per
  token) over the reduction axis, range ``[-127, 127]``, ``eps=1e-5`` -- TorchAO's own per-token
  input quantization for this config;
* the second operand takes the WEIGHT side of the scheme: symmetric, one scale per OUTPUT COLUMN
  over the reduction axis, the scheme's default int8 range and ``eps`` = float32 machine epsilon.
  For ``P @ V`` that is one scale per head-dimension channel of ``V``; for ``Q @ K^T`` one per key
  token. It is computed at run time because the operand is produced at run time; the form is the
  weight's, only the moment it is computed differs;
* the product is ``torch._int_mm`` -- int8 x int8 accumulating in int32 -- and is dequantized the
  way TorchAO's own Linear kernel dequantizes it: int32 -> float, times the row scale, times the
  column scale.

Both operands are quantized with TorchAO's own ``to_affine_quantized_intx``, so the scales and
rounding are the framework's, bit for bit, rather than a re-implementation that agrees with it.

WHAT IS LEFT ALONE, AND SAID. A contraction whose reduction extent is 16 or less stays in floating
point: TorchAO applies the same rule to a Linear ("int8 dynamic quantization only has benefit when
in_feature > 16"), and a rank-one outer product such as a rotary-frequency table is exactly the case
it protects. A convolution is quantized only where it is a pure re-layout of a matmul (stride equal
to the kernel, no padding, no dilation, one group -- a patch embedding); any other convolution is
left in floating point and counted. Nothing here names a target, a model, or a scheme; the numeric
form is the scheme's, and the decision to use it is the caller's.

Why a TorchFunctionMode and not a graph rewrite. The capture lowers the module by exporting it, and
the golden is the SAME module run eagerly; a rewrite of one of those two paths would let them drift.
Entering the mode inside ``forward`` puts both through the identical Python, and every decision is
made from shapes alone, which the export's fake tensors carry exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: TorchAO's own threshold for a Linear: at or below this many reduction elements it declines to
#: quantize. Mirrored rather than re-derived, so a Linear and a matmul of the same extent get the
#: same answer.
MIN_REDUCTION = 16


def activation_reduction_minimum(recipe: Mapping[str, Any]) -> int:
    """Inclusive recipe minimum; absence preserves the historical profitability screen.

    This changes capture selection only. Software admission and integer producer/quality
    checks remain independent. The standalone capture interpreter uses this same parser.
    """
    if "activation_contraction_policy" not in recipe:
        return MIN_REDUCTION + 1
    block = recipe["activation_contraction_policy"]
    if not isinstance(block, dict) or set(block) != {"schema", "minimum_reduction"}:
        raise ValueError("activation_contraction_policy requires schema and minimum_reduction only")
    if block["schema"] != "activation_contraction_admission_v1":
        raise ValueError("unsupported activation_contraction_policy schema")
    if "activation_contraction" not in (recipe.get("families") or ()):
        raise ValueError("activation_contraction_policy requires the activation_contraction family")
    minimum = block["minimum_reduction"]
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum_reduction must be a positive integer")
    return minimum

#: Activation (first-operand) form: TorchAO's ``_int8_symm_per_token_reduced_range_quant``.
_ACT_QMIN, _ACT_QMAX, _ACT_EPS = -127, 127, 1e-5

#: Status words recorded per call site.
QUANTIZED = "quantized"
REDUCTION_TOO_SMALL = "reduction_too_small"
NOT_FLOATING = "not_floating"
RANK_BELOW_TWO = "rank_below_two"
CONV_NOT_A_MATMUL = "conv_not_a_matmul"
SDPA_NOT_SPELLED = "attention_not_spelled_as_matmuls"


class Census:
    """What the mode saw on its LAST forward: one row per distinct (op, shape, verdict)."""

    def __init__(self) -> None:
        self.rows: dict[tuple, int] = {}

    def add(self, op: str, shapes: tuple, verdict: str) -> None:
        key = (op, shapes, verdict)
        self.rows[key] = self.rows.get(key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        rows = [
            {"op": op, "shapes": [list(s) for s in shapes], "verdict": verdict, "count": n}
            for (op, shapes, verdict), n in sorted(
                self.rows.items(), key=lambda kv: (kv[0][2], kv[0][0], str(kv[0][1]))
            )
        ]
        by_verdict: dict[str, int] = {}
        for row in rows:
            by_verdict[row["verdict"]] = by_verdict.get(row["verdict"], 0) + row["count"]
        return {"sites": rows, "by_verdict": by_verdict}


def _quantize(x: Any, *, activation: bool) -> tuple[Any, Any]:
    """``(int8 data, float scale)`` of ``x`` quantized over its LAST axis, one scale per leading index.

    ``activation`` selects the scheme's input form; otherwise the scheme's weight form. Each is the
    pair of TorchAO primitives ``to_affine_quantized_intx`` calls for the arguments
    ``Int8DynamicActivationInt8WeightConfig`` passes it (a plain layout, zero preserved): the input
    side keeps a zero point of zero, the weight side has none. The primitives are called directly
    rather than through that constructor because it wraps the result in a tensor subclass, and a
    subclass built outside a dispatch handler is not traceable by the export this capture runs.
    """
    import torch
    from torchao.quantization.quant_primitives import (
        MappingType,
        _quantize_affine_no_zero_point,
        choose_qparams_affine,
        quantize_affine,
    )

    block = tuple([1] * (x.dim() - 1) + [int(x.shape[-1])])
    if activation:
        scale_dtype = torch.float32 if x.dtype == torch.float16 else None
        scale, zero_point = choose_qparams_affine(
            x, MappingType.SYMMETRIC, block, torch.int8, _ACT_QMIN, _ACT_QMAX, _ACT_EPS, scale_dtype, None
        )
        data = quantize_affine(x, block, scale, zero_point, torch.int8, _ACT_QMIN, _ACT_QMAX)
    else:
        scale, _zero = choose_qparams_affine(
            x, MappingType.SYMMETRIC, block, torch.int8, None, None, torch.finfo(torch.float32).eps, None, torch.int64
        )
        data = _quantize_affine_no_zero_point(x, block, scale, None, torch.int8, None, None)
    return data, scale.reshape(*x.shape[:-1])


def _int_matmul(a3: Any, bt3: Any) -> Any:
    """``a3[B,M,K] @ bt3[B,N,K]^T`` in the scheme's form: per-row ``a3``, per-column (per row of ``bt3``).

    Both operands are quantized ONCE over the whole batch -- a per-row scale depends on its row only,
    so this is the same numbers as quantizing slice by slice, in one set of operations instead of B.
    Each slice is then its own ``int8 x int8 -> int32`` contraction (``torch._int_mm`` is 2-d), and
    the int32 results are dequantized together in TorchAO's order: int32 -> float, times the row
    scale, then times the column scale.
    """
    import torch

    a_q, a_s = _quantize(a3, activation=True)
    b_q, b_s = _quantize(bt3, activation=False)
    acc = torch.stack([torch._int_mm(a_q[i], b_q[i].t()) for i in range(int(a3.shape[0]))])
    y = acc.to(a_s.dtype) * a_s.unsqueeze(-1)
    return (y * b_s.unsqueeze(-2)).to(a3.dtype)


def quantized_matmul(a: Any, b: Any) -> Any:
    """``torch.matmul(a, b)`` for rank >= 2 floating operands, every batch slice an int8 contraction."""
    import torch

    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m, k = a.shape[-2], a.shape[-1]
    n = b.shape[-1]
    a3 = a.expand(*batch, m, k).reshape(-1, m, k)
    bt3 = b.expand(*batch, k, n).transpose(-1, -2).reshape(-1, n, k)
    return _int_matmul(a3, bt3).reshape(*batch, m, n)


_SDPA_ARGS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale")


def sdpa_as_matmuls(args: tuple, kwargs: dict) -> tuple[bool, str]:
    """Whether a fused attention call can be spelled as its two matmuls, and why not when it cannot."""
    bound = dict(zip(_SDPA_ARGS, args))
    bound.update(kwargs)
    if float(bound.get("dropout_p") or 0.0) != 0.0:
        return False, "dropout"
    if bound.get("enable_gqa"):
        return False, "grouped-query broadcast"
    q, k, v = bound.get("query"), bound.get("key"), bound.get("value")
    if q is None or k is None or v is None or min(q.dim(), k.dim(), v.dim()) < 2:
        return False, "operands below rank 2"
    return True, ""


def quantized_sdpa(args: tuple, kwargs: dict) -> Any:
    """``scaled_dot_product_attention`` as ``softmax(Q K^T * scale + mask) V`` with both matmuls int8.

    The fused operator hides its two contractions from the matmul interception, and the capture
    decomposes it into exactly these steps anyway (scores, softmax, weighted sum); spelling them out
    here puts both contractions in the scheme's form instead of leaving them in floating point.
    """
    import math

    import torch

    bound = dict(zip(_SDPA_ARGS, args))
    bound.update(kwargs)
    q, k, v = bound["query"], bound["key"], bound["value"]
    mask, causal, scale = bound.get("attn_mask"), bool(bound.get("is_causal")), bound.get("scale")
    scale = float(scale) if scale is not None else 1.0 / math.sqrt(int(q.shape[-1]))
    scores = quantized_matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        rows, cols = int(scores.shape[-2]), int(scores.shape[-1])
        keep = torch.ones(rows, cols, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else scores + mask
    probs = torch.softmax(scores, dim=-1)
    return quantized_matmul(probs, v)


def _pair(v: Any) -> tuple[int, int]:
    return (int(v[0]), int(v[1])) if isinstance(v, (tuple, list)) else (int(v), int(v))


def conv_as_matmul(args: tuple, kwargs: dict) -> tuple[bool, str]:
    """Whether a 2-d convolution call is a pure re-layout of a matmul, and why not when it is not."""
    names = ("input", "weight", "bias", "stride", "padding", "dilation", "groups")
    bound = dict(zip(names, args))
    bound.update(kwargs)
    weight = bound.get("weight")
    if weight is None or weight.dim() != 4:
        return False, "weight is not a 4-d convolution kernel"
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    stride = _pair(bound.get("stride", 1))
    padding = bound.get("padding", 0)
    if isinstance(padding, str):
        if padding != "valid":
            return False, f"padding {padding!r}"
    elif _pair(padding) != (0, 0):
        return False, f"padding {_pair(padding)}"
    if _pair(bound.get("dilation", 1)) != (1, 1):
        return False, "dilated"
    if int(bound.get("groups", 1)) != 1:
        return False, "grouped"
    if stride != (kh, kw):
        return False, f"stride {stride} is not the kernel {(kh, kw)}: overlapping windows"
    return True, ""


def quantized_patch_conv(args: tuple, kwargs: dict) -> Any:
    """A non-overlapping 2-d convolution as patches @ weight^T, in the scheme's int8 form."""
    names = ("input", "weight", "bias", "stride", "padding", "dilation", "groups")
    bound = dict(zip(names, args))
    bound.update(kwargs)
    x, weight, bias = bound["input"], bound["weight"], bound.get("bias")
    oc, c, kh, kw = (int(d) for d in weight.shape)
    n, _, h, w = (int(d) for d in x.shape)
    ho, wo = h // kh, w // kw
    patches = (
        x[:, :, : ho * kh, : wo * kw]
        .reshape(n, c, ho, kh, wo, kw)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(n * ho * wo, c * kh * kw)
    )
    y = _int_matmul(patches.unsqueeze(0), weight.reshape(1, oc, c * kh * kw)).squeeze(0)
    if bias is not None:
        y = y + bias
    return y.reshape(n, ho, wo, oc).permute(0, 3, 1, 2).contiguous()


def _mode_class():
    import torch
    from torch.overrides import TorchFunctionMode

    matmuls = {torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__, torch.bmm, torch.Tensor.bmm}
    convs = {torch.conv2d, torch.nn.functional.conv2d}
    sdpas = {torch.nn.functional.scaled_dot_product_attention}

    class ActivationContractionMode(TorchFunctionMode):
        def __init__(self, census: Census, *, convolutions: bool) -> None:
            super().__init__()
            self.census = census
            self.convolutions = convolutions

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in matmuls and len(args) >= 2:
                a, b = args[0], args[1]
                shapes = (tuple(a.shape), tuple(b.shape))
                if not (a.is_floating_point() and b.is_floating_point()):
                    self.census.add("matmul", shapes, NOT_FLOATING)
                elif a.dim() < 2 or b.dim() < 2:
                    self.census.add("matmul", shapes, RANK_BELOW_TWO)
                elif int(a.shape[-1]) <= MIN_REDUCTION:
                    self.census.add("matmul", shapes, REDUCTION_TOO_SMALL)
                else:
                    self.census.add("matmul", shapes, QUANTIZED)
                    return quantized_matmul(a, b)
            elif func in sdpas and len(args) >= 3:
                q, k, v = args[0], args[1], args[2]
                shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
                ok, why = sdpa_as_matmuls(args, kwargs)
                if not ok:
                    self.census.add("attention", shapes, SDPA_NOT_SPELLED + ": " + why)
                elif not (q.is_floating_point() and k.is_floating_point() and v.is_floating_point()):
                    self.census.add("attention", shapes, NOT_FLOATING)
                elif min(int(q.shape[-1]), int(k.shape[-2])) <= MIN_REDUCTION:
                    # Q K^T reduces over the head dimension, P V over the key sequence.
                    self.census.add("attention", shapes, REDUCTION_TOO_SMALL)
                else:
                    self.census.add("attention", shapes, QUANTIZED)
                    return quantized_sdpa(args, kwargs)
            elif self.convolutions and func in convs and len(args) >= 2:
                x, weight = args[0], args[1]
                shapes = (tuple(x.shape), tuple(weight.shape))
                ok, why = conv_as_matmul(args, kwargs)
                reduction = int(weight[0].numel()) if weight.dim() == 4 else 0
                if not ok:
                    self.census.add("conv2d", shapes, CONV_NOT_A_MATMUL + ": " + why)
                elif reduction <= MIN_REDUCTION:
                    self.census.add("conv2d", shapes, REDUCTION_TOO_SMALL)
                elif not (x.is_floating_point() and weight.is_floating_point()):
                    self.census.add("conv2d", shapes, NOT_FLOATING)
                else:
                    self.census.add("conv2d", shapes, QUANTIZED)
                    return quantized_patch_conv(args, kwargs)
            return func(*args, **kwargs)

    return ActivationContractionMode


def install(model: Any, *, convolutions: bool = True) -> Census:
    """Route ``model``'s forward through the mode; return the census its next forward fills.

    The forward is replaced on the INSTANCE, so export and the eager golden both run it. The census is
    reset at the start of every forward, so it always describes one pass rather than the sum of the
    export trace and the golden run.
    """
    import functools

    census = Census()
    mode_cls = _mode_class()
    original = model.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        census.rows.clear()
        with mode_cls(census, convolutions=convolutions):
            return original(*args, **kwargs)

    model.forward = forward
    return census
