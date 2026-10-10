"""Render builtin capture sources from explicit templates and typed input declarations."""

from __future__ import annotations

import math

from . import component_sources


def _permutation(spec: dict, rank: int) -> list[int]:
    """The axis order a ``permute`` program applies: the spec's, else the last two axes swapped (for
    the historic ``[M, K]`` form, the plain transpose)."""
    raw = spec.get("permutation")
    if raw is None:
        return [*range(rank - 2), rank - 1, rank - 2] if rank >= 2 else list(range(rank))
    if (
        not isinstance(raw, (list, tuple))
        or any(isinstance(axis, bool) or not isinstance(axis, int) for axis in raw)
        or sorted(raw) != list(range(rank))
    ):
        raise ValueError(f"permutation must order every axis of the rank-{rank} shape exactly once, got {raw!r}")
    return [int(axis) for axis in raw]


def render_builtin_source(
    spec: dict,
    *,
    preamble: str,
    integer_matmul: str,
    parametric_linear: str,
    bodies: dict[str, str],
    input_names: dict,
) -> str:
    """Render the PyTorch loader source for an op spec. ``spec`` carries ``op`` + the shape fields the op
    needs (M/K/N/Dv) + optional ``eps``/``causal``/``bias`` + ``seed``/``dtype``. Fail closed on an
    unknown op (never silently emit a wrong program)."""
    op = spec["op"]
    if spec.get("input_palette") is not None and (op == "int_matmul" or spec.get("quant_scheme")):
        raise ValueError("selected input palettes require the normal typed float builtin source path")
    if op != "int_matmul" and op not in bodies:
        raise KeyError(f"capsule_source has no PyTorch template for op {op!r} (have {sorted(bodies)})")
    raw_shape = spec.get("shape")
    if raw_shape is None:
        raw_shape = [spec.get("M", 16), spec.get("K", 16)]
    if (
        not isinstance(raw_shape, (list, tuple))
        or not raw_shape
        or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in raw_shape)
    ):
        raise ValueError(f"capsule shape must be a non-empty sequence of positive integers, got {raw_shape!r}")
    fields = {
        "op": op,
        "dtype": spec.get("dtype", "fp32"),
        "seed": int(spec.get("seed", 0)),
        "M": spec.get("M", 16),
        "K": spec.get("K", 16),
        "N": spec.get("N", 16),
        "Dv": spec.get("Dv", spec.get("K", 16)),
        "eps": spec.get("eps", 1e-5),
        "causal": bool(spec.get("causal", False)),
        "bias": bool(spec.get("bias", False)),
        # extra shape/scalar fields for the composite ops (batch, soft-cap, conv geometry)
        "B": spec.get("B", 2),
        "cap": spec.get("cap", 50.0),
        "Cin": spec.get("Cin", 1),
        "Himg": spec.get("Himg", 8),
        "Wimg": spec.get("Wimg", 8),
        "P": spec.get("P", 3 if op == "conv_residual_pool" else 2),
        "Pool": spec.get("Pool", 2),
        # Elementwise programs preserve the captured rank. Most synthetic probes use the historic
        # MxK form; an application-derived probe may instead carry its exact static shape.
        "shape_args": ", ".join(str(int(d)) for d in raw_shape),
        "perm_args": ", ".join(str(axis) for axis in _permutation(spec, len(raw_shape))),
    }
    for name in ("M", "K", "N", "Dv", "B", "Cin", "Himg", "Wimg", "P", "Pool"):
        if type(fields[name]) is not int or fields[name] < 1:
            raise ValueError(f"builtin source field {name} must be an explicit positive integer")
    for name in ("eps", "cap"):
        if type(fields[name]) not in (int, float) or not math.isfinite(fields[name]) or fields[name] <= 0:
            raise ValueError(f"builtin source field {name} must be an explicit positive finite scalar")
    for name in ("causal", "bias"):
        if name in spec and type(spec[name]) is not bool:
            raise ValueError(f"builtin source field {name} must be an explicit Boolean")
    fields["Ho"] = fields["Himg"] - fields["P"] + 1
    fields["Wo"] = fields["Wimg"] - fields["P"] + 1
    if op == "producer_quantizer_observer":
        fields.update(component_sources.parameters(spec))
    if op == "conv_residual_pool" and (
        fields["Ho"] < 1
        or fields["Wo"] < 1
        or type(fields["Pool"]) is not int
        or fields["Pool"] < 1
        or fields["Ho"] % fields["Pool"]
        or fields["Wo"] % fields["Pool"]
    ):
        raise ValueError("conv_residual_pool requires positive convolved extents divisible by the selected pool")
    if op == "int_matmul":
        if spec.get("quant_scheme"):
            raise ValueError("an isolated int_matmul has quantized operands; do not quantize it again")
        return preamble.format(**fields) + integer_matmul.format(**fields)
    # A QUANTIZED capture needs a weight PARAMETER for the scheme to bind to (see parametric_linear).
    # Only the contraction ops have a meaningful weight; asking for a quantized elementwise op is a
    # request that cannot be honoured, and saying so beats emitting an unquantized program under a
    # quantized name.
    if spec.get("quant_scheme"):
        if op not in ("linear", "matmul"):
            raise ValueError(
                f"quant_scheme is set for op {op!r}, but only a contraction carries a weight for a "
                f"torchAO scheme to quantize; an unquantized program under a quantized name is worse "
                f"than a refusal"
            )
        return preamble.format(**fields) + parametric_linear.format(**fields)
    source = preamble.format(**fields) + bodies[op].format(**fields)
    if spec.get("input_palette") is not None:
        from .input_palette import render_source_inputs

        source += render_source_inputs(spec["input_palette"], input_names[op], fields["dtype"])
    return source
