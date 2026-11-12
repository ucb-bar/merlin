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

    _LINALG_DIALECT = None

    def linalg_dialect():
        """Keep the named signed i8 matmul's implicit body faithful to MLIR.

        xDSL 0.68 omits the casts into the i32 accumulator domain. Register a
        local operation class instead of mutating the upstream dialect. Generic
        syntax supplies its own region and never calls this implicit builder.
        """
        global _LINALG_DIALECT
        if _LINALG_DIALECT is None:
            from xdsl.builder import Builder
            from xdsl.dialects import arith
            from xdsl.dialects.builtin import UnregisteredAttr, i8, i32
            from xdsl.dialects.linalg import Linalg
            from xdsl.dialects.linalg.ops import MatmulOp, YieldOp
            from xdsl.ir import Dialect

            class SignedAccumulatorMatmulOp(MatmulOp):
                @classmethod
                def get_hidden_region(cls, inputs, outputs):
                    types = cls.body_arg_types((*inputs, *outputs))
                    if len(inputs) != 2 or len(outputs) != 1 or tuple(types) != (i8, i8, i32):
                        return super().get_hidden_region(inputs, outputs)

                    @Builder.implicit_region(types)
                    def region(args):
                        lhs = arith.ExtSIOp(args[0], i32)
                        rhs = arith.ExtSIOp(args[1], i32)
                        product = arith.MuliOp(lhs, rhs)
                        total = arith.AddiOp(args[2], product)
                        YieldOp(total)

                    return region

                @classmethod
                def parse(cls, parser):
                    op = super().parse(parser)
                    types = cls.body_arg_types((*op.inputs, *op.outputs))
                    if tuple(types) == (i8, i8, i32):
                        cast = op.attributes.get("cast", op.properties.get("cast"))
                        if cast is not None and not (
                            isinstance(cast, UnregisteredAttr)
                            and cast.attr_name.data == "linalg.type_fn"
                            and not cast.is_type.data
                            and not cast.is_opaque.data
                            and cast.value.data.strip() == "cast_signed"
                        ):
                            parser.raise_error("implicit i8 matmul requires the default or signed cast")
                    return op

            _LINALG_DIALECT = Dialect(
                Linalg.name,
                [SignedAccumulatorMatmulOp if op is MatmulOp else op for op in Linalg.operations],
                list(Linalg.attributes),
            )
        return _LINALG_DIALECT

    def make_context(*dialects):
        """A Context preloaded with Builtin + Func + the given Dialect objects."""
        from xdsl.context import Context
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func
        from xdsl.dialects.linalg import Linalg

        # Teach the parser the fp8 element types the corpus uses before any parse.
        from merlin.compiler.dialects.fp8 import register_fp8_types

        register_fp8_types()

        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(Func)
        for d in dialects:
            ctx.load_dialect(linalg_dialect() if d is Linalg else d)
        return ctx

    def text(module, *, generic: bool = False) -> str:
        """Print a module; generic form preserves every attribute and property."""
        import io

        from merlin.common.dense_float_ir import DenseFloatPrinter

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
        from merlin.common.dense_float_ir import DenseFloatParser

        return DenseFloatParser(make_context(*dialects), text(module)).parse_module()

else:  # pragma: no cover - exercised only when xDSL is absent
    Visibility = None

    def make_context(*dialects):
        return None

    def text(module, *, generic: bool = False) -> str:
        return ""

    def roundtrip(module, *dialects):
        return module
