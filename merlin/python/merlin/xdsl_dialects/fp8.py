"""Make the granted xDSL kit parse the 8-bit float element types the corpus uses.

The frozen ``merlin_iface`` interface grammar is target-agnostic, so a capsule may declare
any element type the workload needs. Low-precision (fp8) capsules spell their tensors with
MLIR's builtin 8-bit float names — ``f8E4M3FN`` and ``f8E5M2`` — e.g.
``tensor<32x16xf8E4M3FN>``. The pinned xDSL (0.68) predates builtin fp8 types: its type
parser only knows ``f16/f32/f64/f80/f128`` and ``bf16``, so every fp8 capsule interface
fails to parse out of the box (``Expected shape type``). This module fills that gap so an
agent parsing a capsule never has to reverse-engineer the parser to register the type
itself.

It is a small, self-contained addition — two builtin-style float element types plus one
idempotent parser hook — not a fork of xDSL. The types mirror the spelling and print form of
xDSL's own builtin floats (bare ``f8E4M3FN`` / ``f8E5M2``, no ``!`` prefix) so parsed
modules round-trip textually. Byte packing is intentionally left unimplemented: the contract
surface only names these types (shapes/dtypes), it never materializes fp8 buffers here.
"""
from __future__ import annotations

from ._common import HAS_XDSL

# The 8-bit float element-type spellings this kit teaches the parser, keyed by their exact
# MLIR bare-identifier text. These are dtype names (not target names) and match LLVM/MLIR's
# builtin fp8 formats: E4M3 with finite-only semantics, and E5M2.
FP8_TYPE_NAMES = ("f8E4M3FN", "f8E5M2")

if HAS_XDSL:
    from xdsl.dialects.builtin import _FloatType  # noqa: PLC2701 - builtin float base
    from xdsl.ir import ParametrizedAttribute
    from xdsl.irdl import irdl_attr_definition

    class _Fp8Type(ParametrizedAttribute, _FloatType):
        """Common base for the 8-bit builtin-style float element types.

        Inherits ``_FloatType`` so the printer emits the bare spelling (``print_builtin``)
        and the type is recognized as a builtin attribute — the same treatment ``bf16`` and
        ``f16`` get. Values are 8 bits wide; the packing hooks raise because the contract
        surface never encodes fp8 buffers through this path (fail closed, don't fake bytes).
        """

        @property
        def bitwidth(self) -> int:
            return 8

        @property
        def compile_time_size(self) -> int:
            return 1

        def iter_unpack(self, buffer, /):  # pragma: no cover - not used by the contract path
            raise NotImplementedError("fp8 byte packing is not modeled by the interface kit")

        def unpack(self, buffer, num, /):  # pragma: no cover - not used by the contract path
            raise NotImplementedError("fp8 byte packing is not modeled by the interface kit")

        def pack_into(self, buffer, offset, value):  # pragma: no cover - not used here
            raise NotImplementedError("fp8 byte packing is not modeled by the interface kit")

        def pack(self, values):  # pragma: no cover - not used by the contract path
            raise NotImplementedError("fp8 byte packing is not modeled by the interface kit")

    @irdl_attr_definition
    class Float8E4M3FNType(_Fp8Type):
        """!f8E4M3FN — 8-bit float, 4 exponent / 3 mantissa bits, finite-only (no inf)."""
        name = "f8E4M3FN"

    @irdl_attr_definition
    class Float8E5M2Type(_Fp8Type):
        """!f8E5M2 — 8-bit float, 5 exponent / 2 mantissa bits."""
        name = "f8E5M2"

    # Instances keyed by their MLIR bare spelling, for the parser hook.
    _FP8_BY_NAME = {
        "f8E4M3FN": Float8E4M3FNType(),
        "f8E5M2": Float8E5M2Type(),
    }

    # Sentinel marking the AttrParser class we have already patched, so repeated calls (and
    # repeated imports) stay idempotent and never wrap the hook twice.
    _PATCH_FLAG = "_merlin_fp8_parse_hook"

    def register_fp8_types() -> None:
        """Teach xDSL's builtin type parser the fp8 bare idents. Idempotent.

        xDSL resolves bare-ident number types in
        ``AttrParser._parse_optional_integer_or_float_type``; 0.68's table stops at
        ``f128``/``bf16``. We wrap that method so an ``f8E4M3FN`` / ``f8E5M2`` identifier
        resolves to the matching element type before falling back to the original logic.
        Everything else is delegated unchanged, so no other type parsing is affected.
        """
        from xdsl.parser.attribute_parser import AttrParser
        from xdsl.utils.mlir_lexer import MLIRTokenKind

        if getattr(AttrParser, _PATCH_FLAG, False):
            return

        _original = AttrParser._parse_optional_integer_or_float_type

        def _hooked(self):
            token = self._current_token
            if token.kind == MLIRTokenKind.BARE_IDENT:
                fp8 = _FP8_BY_NAME.get(token.text)
                if fp8 is not None:
                    self._consume_token()
                    return fp8
            return _original(self)

        AttrParser._parse_optional_integer_or_float_type = _hooked
        setattr(AttrParser, _PATCH_FLAG, True)

    # Install the hook as a side effect of importing the kit, so any parse path that touches
    # merlin's xDSL dialects (interface load, make_context, roundtrip) can read fp8 capsules.
    register_fp8_types()

else:  # pragma: no cover - exercised only when xDSL is absent

    _FP8_BY_NAME = {}

    def register_fp8_types() -> None:
        return
