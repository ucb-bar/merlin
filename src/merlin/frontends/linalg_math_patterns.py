"""Closed source-body evidence for unary f32 sine and cosine regions.

This proves the parsed Linalg scalar operation, not host admission, a selected
libm implementation, or bit-exact equivalence with the capture framework.
Power forms require an independently bound scalar-literal trace and are refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.frontends.linalg_patterns import (
    InvalidLinalgPattern,
    _checked_static_linalg_shell,
    _screen_static_linalg_source,
)

SOURCE_ONLY_SCOPE = "source math only; no host admission, selected-libm equivalence, or PyTorch bit-exactness"
STATIC_F32_MATH_SOURCE_BODY_SCHEMA = "merlin.static_f32_math_source_body.v1"
_UNARY_MATH = frozenset({"math.sin", "math.cos"})


def validate_static_f32_math_source_body(declaration: object) -> dict[str, str]:
    """Validate the closed, opt-in source declaration; grant no host support."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "operation"}
        or declaration.get("schema") != STATIC_F32_MATH_SOURCE_BODY_SCHEMA
        or not isinstance(declaration.get("operation"), str)
        or declaration.get("operation") not in _UNARY_MATH
    ):
        raise InvalidLinalgPattern("source_body requires a closed unary f32 math v1 declaration")
    return dict(declaration)


def static_f32_math_ordered_types(declaration: object) -> tuple[str, ...]:
    """Return the exact input/init/result scalar type order of a declaration."""
    validate_static_f32_math_source_body(declaration)
    return ("f32", "f32", "f32")


@dataclass(frozen=True)
class StaticF32MathPattern:
    operation: str
    shape: tuple[int, ...]
    ordered_types: tuple[str, ...]
    scope: str = SOURCE_ONLY_SCOPE


@dataclass(frozen=True)
class StaticF32MathSource:
    raw_sha256: str
    normalized_sha256: str
    ordinals: tuple[tuple[int, StaticF32MathPattern], ...]


def _recognize_static_f32_unary_math_body(
    op, *, admitted: frozenset[str], operation_classes: tuple[type, ...], description: str,
) -> StaticF32MathPattern:
    """Require one unflagged pure unary f32 math op and its direct yield.

    The common shell checks registered Linalg, static equal tensor shapes,
    identity maps, parallel iterators, exact block types and closed attributes.
    This function additionally excludes init reads, captures, extra operations,
    effects and result substitution by checking the scalar SSA edges exactly.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.traits import Pure

    shell = _checked_static_linalg_shell(op)
    if len(op.inputs) != 1 or shell.ordered_types != ("f32", "f32", "f32"):
        raise InvalidLinalgPattern("math source requires one f32 input and f32 init/result")
    if len(shell.body) != 2:
        raise InvalidLinalgPattern("math source requires exactly one scalar operation and yield")
    scalar, yld = shell.body
    if scalar.name not in admitted or type(scalar) not in operation_classes:
        raise InvalidLinalgPattern(f"math source operation is not {description}")
    if not type(scalar).has_trait(Pure) or scalar.regions or scalar.successors:
        raise InvalidLinalgPattern("math source scalar operation is not pure and region-free")
    if scalar.attributes and any(not name.startswith("prov.") for name in scalar.attributes):
        raise InvalidLinalgPattern("math source scalar operation has an unknown attribute")
    if set(scalar.properties) - {"fastmath"}:
        raise InvalidLinalgPattern("math source scalar operation has an unknown property")
    flags = scalar.properties.get("fastmath")
    if flags is not None and (not isinstance(flags, arith.FastMathFlagsAttr) or flags.data):
        raise InvalidLinalgPattern("math source scalar operation has fast-math flags")
    if tuple(scalar.operands) != (shell.args[0],) or len(scalar.results) != 1 or str(scalar.results[0].type) != "f32":
        raise InvalidLinalgPattern("math source does not consume only its input block argument")
    if (
        type(yld) is not YieldOp
        or yld.regions
        or yld.successors
        or yld.results
        or yld.properties
        or any(not name.startswith("prov.") for name in yld.attributes)
        or tuple(yld.operands) != (scalar.results[0],)
    ):
        raise InvalidLinalgPattern("math source does not directly yield its scalar result")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - xDSL verification is a fail-closed source boundary
        raise InvalidLinalgPattern(f"math source failed xDSL verification: {exc}") from exc
    return StaticF32MathPattern(scalar.name, shell.shape, shell.ordered_types)


def recognize_static_f32_math_body(op) -> StaticF32MathPattern:
    """Recognize the existing source-only sine/cosine declaration vocabulary."""
    from xdsl.dialects import math

    return _recognize_static_f32_unary_math_body(
        op, admitted=_UNARY_MATH, operation_classes=(math.SinOp, math.CosOp),
        description="registered unary sine or cosine",
    )


def recognize_static_f32_exp_rsqrt_body(op) -> StaticF32MathPattern:
    """Recognize source exp/rsqrt without extending host math declarations."""
    from xdsl.dialects import math

    return _recognize_static_f32_unary_math_body(
        op, admitted=frozenset({"math.exp", "math.rsqrt"}),
        operation_classes=(math.ExpOp, math.RsqrtOp),
        description="registered unary exp or rsqrt",
    )


def screen_static_f32_math_source(path: Path, ordinals: tuple[int, ...]) -> StaticF32MathSource:
    """Bind every requested normalized ordinal to the read source bytes."""
    raw_sha, normalized_sha, evidence = _screen_static_linalg_source(path, ordinals, recognize_static_f32_math_body)
    return StaticF32MathSource(raw_sha, normalized_sha, evidence)
