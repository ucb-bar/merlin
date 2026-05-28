"""Export adapter — replace mxGemmini-quantized linears with eager-mode
linears whose weights are pre-dequantized to high precision.

Mirrors ``third_party/Understanding-PI0/understanding_pi0/common/mx_exportable.py``
so iree-turbine's torch-export path doesn't have to know about our
custom Tensor subclasses. After this rewrite the model is a plain
``nn.Module`` graph again, ready for ``aot_module(...)`` / IREE Turbine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_dtype import (
    MxGemminiE2M2Tensor,
    MxGemminiE4M4Tensor,
)


@dataclass
class ReplaceRecord:
    fqn: str
    replaced: bool
    reason: str


def _is_stock_mx_tensor(x) -> bool:
    return (
        isinstance(x, torch.Tensor)
        and hasattr(x, "qdata")
        and hasattr(x, "scale")
        and hasattr(x, "elem_dtype")
        and hasattr(x, "block_size")
        and x.__class__.__name__ == "MXTensor"
    )


def _has_mxgemmini_attr(linear: nn.Linear) -> bool:
    return hasattr(linear, "mxgemmini_weight") and isinstance(
        getattr(linear, "mxgemmini_weight"),
        (MxGemminiE4M4Tensor, MxGemminiE2M2Tensor),
    )


class ExportableMxGemminiLinear(nn.Module):
    """Wraps a quantized linear so torch-export sees only standard ops.

    Holds the per-block quantized data + signed scales as registered
    buffers (so they survive ``deepcopy`` / ``state_dict``); does the
    dequantize inside ``forward`` using only pointwise + matmul ops
    that AOTautograd is happy with.
    """

    def __init__(
        self,
        weight_hp: torch.Tensor,
        bias: torch.Tensor | None,
        in_features: int,
        out_features: int,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.compute_dtype = compute_dtype
        self.register_buffer("weight_buf", weight_hp.detach().to(compute_dtype), persistent=True)
        if bias is None:
            self.register_buffer("bias_buf", None, persistent=True)
        else:
            self.register_buffer("bias_buf", bias.detach().to(compute_dtype), persistent=True)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_buf

    @property
    def bias(self) -> torch.Tensor | None:
        return self.bias_buf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.weight_buf.dtype)
        return F.linear(x, self.weight_buf, self.bias_buf)


def _iter_named_children_with_parent(module: nn.Module, prefix: str = ""):
    for child_name, child in module.named_children():
        fqn = f"{prefix}.{child_name}" if prefix else child_name
        yield module, child_name, child, fqn
        yield from _iter_named_children_with_parent(child, fqn)


def _materialize_high_precision_weight(linear: nn.Linear, compute_dtype: torch.dtype) -> torch.Tensor:
    if _has_mxgemmini_attr(linear):
        # Stage 6.B custom subclass — dequantize from our codes.
        t = linear.mxgemmini_weight  # type: ignore[attr-defined]
        return t.dequantize(target_dtype=compute_dtype)
    w = linear.weight
    if _is_stock_mx_tensor(w):
        # Stage 6.A — torchao MXTensor; lean on its built-in dequantize().
        if hasattr(w, "dequantize"):
            return w.dequantize().to(compute_dtype)
    if hasattr(w, "dequantize"):
        return w.dequantize().to(compute_dtype)
    return w.to(compute_dtype)


def rewrite_quantized_linears_for_export_(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> list[ReplaceRecord]:
    """In-place: replace every quantized ``nn.Linear`` with
    :class:`ExportableMxGemminiLinear` whose weight is materialized HP.

    Picks up both Stage 6.A (torchao ``MXTensor``) and Stage 6.B (our
    ``mxgemmini_weight`` attribute) linears.
    """
    records: list[ReplaceRecord] = []
    to_replace: list[tuple[nn.Module, str, nn.Linear, str]] = []
    for parent, child_name, child, fqn in _iter_named_children_with_parent(model):
        if isinstance(child, nn.Linear):
            if _has_mxgemmini_attr(child) or _is_stock_mx_tensor(child.weight):
                to_replace.append((parent, child_name, child, fqn))
            elif hasattr(child.weight, "dequantize"):
                to_replace.append((parent, child_name, child, fqn))

    for parent, child_name, child, fqn in to_replace:
        try:
            w_hp = _materialize_high_precision_weight(child, compute_dtype)
            new = ExportableMxGemminiLinear(
                weight_hp=w_hp,
                bias=child.bias,
                in_features=child.in_features,
                out_features=child.out_features,
                compute_dtype=compute_dtype,
            )
            setattr(parent, child_name, new)
            records.append(ReplaceRecord(fqn=fqn, replaced=True, reason="mxgemmini->dequant"))
            if verbose:
                print(f"[mxgemmini-export] replaced {fqn}")
        except Exception as e:  # pragma: no cover
            records.append(ReplaceRecord(fqn=fqn, replaced=False, reason=f"{type(e).__name__}: {e}"))
            if verbose:
                print(f"[mxgemmini-export] FAILED {fqn}: {e}")
    return records


def clone_and_rewrite_quantized_linears_for_export(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.float32,
    verbose: bool = True,
) -> tuple[nn.Module, list[ReplaceRecord]]:
    """Deep-copy the model first, then apply the export rewrite. Leaves
    the original (still-quantized) module intact for further reference
    runs."""
    model_copy = copy.deepcopy(model)
    records = rewrite_quantized_linears_for_export_(model_copy, compute_dtype=compute_dtype, verbose=verbose)
    return model_copy, records
