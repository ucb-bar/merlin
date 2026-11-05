"""Shared scaffolding for the merlin core xDSL dialects.

Holds the ``HAS_XDSL`` guard, the enums whose value sets are shared across dialects
(``Visibility`` is used by both ``interface`` and ``dse``), and the print/parse helpers
every dialect module reuses (``roundtrip``, ``make_context``, ``text``).

xDSL 0.65 idioms used throughout (proven in targetgen's generated dialects):
- types: ``ParametrizedAttribute, TypeAttribute`` with field-annotation parameters
  (``ParameterDef`` was removed in 0.65);
- closed enums: ``EnumAttribute[StrEnum]`` + ``SpacedOpaqueSyntaxAttribute``;
- region-bearing ops: ``region_def()`` + ``traits_def(NoTerminator())``;
- it's ``func.ReturnOp`` (not ``Return``).
"""

from __future__ import annotations

try:
    from xdsl.utils.str_enum import StrEnum

    HAS_XDSL = True
except Exception:  # noqa: BLE001 - xDSL is an optional prototyping dependency
    HAS_XDSL = False

if HAS_XDSL:

    class Visibility(StrEnum):
        """DSE variant tag shared by interface ops and dse records."""

        BASELINE = "baseline"
        SOFTWARE_VISIBLE = "software_visible"
        HARDWARE_MANAGED = "hardware_managed"
        ORACLE = "oracle"

    def make_context(*dialects):
        """A Context preloaded with Builtin + Func + the given Dialect objects."""
        from xdsl.context import Context
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func

        # Teach the parser the fp8 element types the corpus uses before any parse.
        from .fp8 import register_fp8_types

        register_fp8_types()

        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(Func)
        for d in dialects:
            ctx.load_dialect(d)
        return ctx

    def text(module, *, generic: bool = False) -> str:
        """Print a module; generic form preserves every attribute and property."""
        import io

        from ..common.dense_float_ir import DenseFloatPrinter

        class PortablePrinter(DenseFloatPrinter):
            def print_op(self, op):
                # xDSL's custom yield format puts attributes before operands,
                # which upstream MLIR rejects. Generic syntax retains provenance
                # and avoids depending on that incompatible custom assembly.
                previous = self.print_generic_format
                attributed_declaration = (
                    op.name == "func.func"
                    and not op.body.blocks
                    and (op.arg_attrs is not None or op.res_attrs is not None)
                )
                # ReduceOp's custom printer omits its attribute dictionary entirely.
                # Keep source provenance and transform contracts on reductions too.
                if (op.name in {"linalg.yield", "linalg.reduce"} and op.attributes) or attributed_declaration:
                    self.print_generic_format = True
                try:
                    super().print_op(op)
                finally:
                    self.print_generic_format = previous

        s = io.StringIO()
        PortablePrinter(stream=s, print_generic_format=generic).print_op(module)
        return s.getvalue()

    def roundtrip(module, *dialects):
        """Print and re-parse a module; returns the parsed module."""
        from ..common.dense_float_ir import DenseFloatParser

        return DenseFloatParser(make_context(*dialects), text(module)).parse_module()

else:  # pragma: no cover - exercised only when xDSL is absent
    Visibility = None

    def make_context(*dialects):
        return None

    def text(module, *, generic: bool = False) -> str:
        return ""

    def roundtrip(module, *dialects):
        return module
