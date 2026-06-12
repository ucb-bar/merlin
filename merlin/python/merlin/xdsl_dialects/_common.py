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

        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(Func)
        for d in dialects:
            ctx.load_dialect(d)
        return ctx

    def text(module) -> str:
        """Print a module to a string."""
        import io

        from xdsl.printer import Printer

        s = io.StringIO()
        Printer(stream=s).print_op(module)
        return s.getvalue()

    def roundtrip(module, *dialects):
        """Print and re-parse a module; returns the parsed module."""
        from xdsl.parser import Parser

        return Parser(make_context(*dialects), text(module)).parse_module()

else:  # pragma: no cover - exercised only when xDSL is absent

    Visibility = None

    def make_context(*dialects):
        return None

    def text(module) -> str:
        return ""

    def roundtrip(module, *dialects):
        return module
