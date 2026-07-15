"""Mixed-precision policy — the generic "split" that is ours, no matter the quantization format.

A mixed-precision capture assigns different formats to different parts of a model (e.g. attention in
fp16, MLP/FFN in fp4). The policy is **data**, not code: a default format plus per-module rules, each
naming a target-agnostic :mod:`merlin.common.quant_formats` format. Merlin owns the policy
representation + validation and translates it into the shape model2MLIR's torchAO pipeline consumes
(``QuantizationConfig.per_module``); the actual quantization runs in the frontend. A base float
format (fp16/bf16/fp32) means "leave this module unquantized" (kept in that dtype), represented as a
``None`` scheme so torchAO skips it.
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.common import quant_formats as qf


@dataclass(frozen=True)
class PrecisionRule:
    """Assign ``fmt`` to modules matching ``module`` (a name pattern matched by the frontend)."""

    module: str
    fmt: str


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """A default format plus first-match-wins per-module overrides."""

    default: str
    rules: tuple[PrecisionRule, ...] = ()

    def validate(self) -> "MixedPrecisionPolicy":
        for name in (self.default, *(r.fmt for r in self.rules)):
            if not qf.has(name):
                raise ValueError(f"mixed-precision: unknown format {name!r}; known: {qf.names()}")
        return self

    def formats(self) -> set[str]:
        return {self.default, *(r.fmt for r in self.rules)}

    def to_m2m_per_module(self) -> dict[str, str | None]:
        """Translate to model2MLIR's per-module map ``{module_pattern: torchao_scheme_or_None}``.

        A base float format (fp16/bf16/fp32) maps to ``None`` (torchAO leaves the module in that
        dtype). A quantized format must carry a ``torchao_scheme`` cross-reference; otherwise this
        raises with a hint (e.g. use ``mxfp4``/``nvfp4`` rather than bare ``fp4_e2m1`` for the torch
        path). The reserved key ``"*"`` carries the default.
        """
        out: dict[str, str | None] = {"*": _scheme_for(self.default)}
        for rule in self.rules:
            out[rule.module] = _scheme_for(rule.fmt)
        return out


def _scheme_for(fmt_name: str) -> str | None:
    fmt = qf.get(fmt_name)
    if fmt.kind == "float_ieee":
        return None  # base float: leave unquantized (kept in this dtype by the capture)
    if fmt.torchao_scheme:
        return fmt.torchao_scheme
    raise ValueError(
        f"format {fmt_name!r} has no torchao_scheme for the torch capture path; "
        f"use a torchAO-producible variant (e.g. mxfp4/mxfp6/nvfp4) or ingest via GGUF"
    )


def attention_fp16_mlp_fp4() -> MixedPrecisionPolicy:
    """The request's worked example: attention in fp16, MLP/FFN in (MX)fp4, rest fp16."""
    return MixedPrecisionPolicy(
        default="fp16",
        rules=(
            PrecisionRule(module="self_attn", fmt="fp16"),
            PrecisionRule(module="mlp", fmt="mxfp4"),
        ),
    ).validate()
