"""Apply mxGemmini quantization to ``nn.Linear`` modules in a model.

Mirrors
``third_party/Understanding-PI0/understanding_pi0/common/torchao_utils.py::safe_quantize_linears_``
so callers used to that pattern can swap in.

Two stages, both available; the caller picks via ``stage=`` keyword:

    Stage 6.A — uses stock torchao MX dtypes (closest available).
    Stage 6.B — uses our bit-exact ``MxGemminiE4M4Tensor`` /
                ``MxGemminiE2M2Tensor``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import torch.nn as nn
from torchao.quantization import (
    FqnToConfig,
    quantize_,
)

from .config import (
    make_mxgemmini_fp4_config,
    make_mxgemmini_fp8_config,
)
from .custom_dtype import quantize_linear_to_mxgemmini


@dataclass
class QuantizeResult:
    fqn: str
    requested: str
    applied: str | None
    ok: bool
    error: str | None = None


def _list_linear_fqns(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def _resolve_module(model: nn.Module, fqn: str) -> nn.Module:
    cur = model
    for part in fqn.split("."):
        cur = getattr(cur, part)
    return cur


def _apply_stock_mx_(model: nn.Module, fqn: str, fmt: str, quant_device: str | None) -> None:
    if fmt == "fp8":
        cfg = make_mxgemmini_fp8_config()
    elif fmt == "fp4":
        cfg = make_mxgemmini_fp4_config()
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")
    fqn_cfg = FqnToConfig(fqn_to_config=OrderedDict([(fqn, cfg)]))
    # torchao requires bf16 weights for the MX path
    mod = _resolve_module(model, fqn)
    if isinstance(mod, nn.Linear):
        # Cast in-place; downstream MX transform handles bf16 weights.
        import torch

        mod.weight = nn.Parameter(mod.weight.detach().to(torch.bfloat16))
        if mod.bias is not None:
            mod.bias = nn.Parameter(mod.bias.detach().to(torch.bfloat16))
    quantize_(model, fqn_cfg, filter_fn=None, device=quant_device)


def _apply_custom_mx_(model: nn.Module, fqn: str, fmt: str) -> None:
    mod = _resolve_module(model, fqn)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{fqn} is not nn.Linear (got {type(mod).__name__})")
    quantize_linear_to_mxgemmini(mod, fmt=fmt)


def safe_quantize_linears_(
    model: nn.Module,
    plan: OrderedDict[str, str | None],
    format: str = "fp8",
    stage: str = "6B",
    quant_device: str | None = None,
    verbose: bool = True,
) -> list[QuantizeResult]:
    """Apply per-FQN quantization using mxGemmini-targeted configs.

    Parameters
    ----------
    plan
        Mapping ``fqn -> "mx" | None``. ``None`` means skip; "mx" means
        apply ``format``.
    format
        ``"fp8"`` or ``"fp4"`` — picks the elem dtype.
    stage
        ``"6A"`` (stock torchao MXDA config) or ``"6B"`` (custom
        E4M4/E2M2 subclasses). Default 6B = bit-exact.
    """
    fmt = format.lower()
    if fmt not in {"fp8", "fp4"}:
        raise ValueError(f"format must be fp8 or fp4, got {format!r}")
    stage = stage.upper()
    if stage not in {"6A", "6B"}:
        raise ValueError(f"stage must be 6A or 6B, got {stage!r}")

    results: list[QuantizeResult] = []
    for fqn, preferred in plan.items():
        if preferred is None:
            results.append(QuantizeResult(fqn=fqn, requested="skip", applied=None, ok=True))
            continue
        if preferred != "mx":
            results.append(
                QuantizeResult(
                    fqn=fqn,
                    requested=preferred,
                    applied=None,
                    ok=False,
                    error=f"unknown requested quant type: {preferred}",
                )
            )
            continue
        if verbose:
            print(f"[mxgemmini-quant] {fqn} -> stage={stage} format={fmt}")
        try:
            if stage == "6A":
                _apply_stock_mx_(model, fqn, fmt, quant_device)
                applied = f"mx_stock_{fmt}"
            else:
                _apply_custom_mx_(model, fqn, fmt)
                applied = f"mx_custom_{fmt}"
            results.append(QuantizeResult(fqn=fqn, requested="mx", applied=applied, ok=True))
        except Exception as e:  # pragma: no cover — surface failure verbatim
            if verbose:
                print(f"  [failed] {type(e).__name__}: {e}")
            results.append(
                QuantizeResult(
                    fqn=fqn,
                    requested="mx",
                    applied=None,
                    ok=False,
                    error=f"{type(e).__name__}: {e}",
                )
            )
    return results


def summarize_results(results: Iterable[QuantizeResult]) -> dict[str, int]:
    summary = {"mx_stock": 0, "mx_custom": 0, "skipped": 0, "failed": 0}
    for r in results:
        if not r.ok:
            summary["failed"] += 1
        elif r.applied and r.applied.startswith("mx_stock"):
            summary["mx_stock"] += 1
        elif r.applied and r.applied.startswith("mx_custom"):
            summary["mx_custom"] += 1
        else:
            summary["skipped"] += 1
    return summary
