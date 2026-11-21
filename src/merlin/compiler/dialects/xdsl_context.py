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

import json
from io import StringIO

# ---- _common -----------------------------------------------------------------------------------
from merlin.compiler.dialects.visibility import HAS_XDSL, Visibility

if HAS_XDSL:
    from merlin.compiler.dialects.standard_linalg import linalg_dialect

    def make_context(*dialects):
        """A Context preloaded with Builtin + Func + the given Dialect objects."""
        from xdsl.context import Context
        from xdsl.dialects.builtin import Builtin
        from xdsl.dialects.func import Func
        from xdsl.dialects.linalg import Linalg

        # Teach the parser the fp8 element types the corpus uses before any parse.

        register_fp8_types()

        ctx = Context()
        ctx.load_dialect(Builtin)
        ctx.load_dialect(Func)
        for d in dialects:
            ctx.load_dialect(linalg_dialect() if d is Linalg else d)
        from merlin.compiler.dialects.standard_arith import register_unsigned_index_cast

        register_unsigned_index_cast(ctx)
        return ctx

    def text(module, *, generic: bool = False) -> str:
        """Print a module; generic form preserves every attribute and property."""
        import io

        from merlin.compiler.ir.xdsl_io import DenseFloatPrinter

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
                if (
                    op.name in {"linalg.yield", "tensor.yield", "linalg.reduce"} and op.attributes
                ) or attributed_declaration:
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
        from merlin.compiler.ir.xdsl_io import DenseFloatParser

        return DenseFloatParser(make_context(*dialects), text(module)).parse_module()

else:  # pragma: no cover - exercised only when xDSL is absent
    Visibility = None

    def make_context(*dialects):
        return None

    def text(module, *, generic: bool = False) -> str:
        return ""

    def roundtrip(module, *dialects):
        return module


# FP8 types and codecs have one registration owner.
from merlin.compiler.dialects.fp8 import _FP8_BY_NAME as _FP8_BY_NAME
from merlin.compiler.dialects.fp8 import FP8_TYPE_NAMES as FP8_TYPE_NAMES
from merlin.compiler.dialects.fp8 import register_fp8_float_constraints as register_fp8_float_constraints
from merlin.compiler.dialects.fp8 import register_fp8_types as register_fp8_types

# ---- ir_inspection -----------------------------------------------------------------------------
# Read-only xDSL inspection serialization; never executable compiler input.


def reconstruct_tensor(directory, descriptor, *, tensor_type):
    """Reconstruct dense storage in an explicitly supplied static, unencoded tensor.

    Existing inspection descriptors do not identify the original shaped container
    or encoding. This therefore reconstructs only the requested supported tensor,
    not an original operation/module or an executable inspection view. Raw storage
    is xDSL's representation, not a portable numeric interchange format.
    """
    from math import prod

    from xdsl.dialects.builtin import (
        BFloat16Type,
        BytesAttr,
        DenseIntOrFPElementsAttr,
        Float16Type,
        Float32Type,
        Float64Type,
        IntegerType,
        NoneAttr,
        TensorType,
    )

    from merlin.compiler.ir.lowering_audit import read_tensor_payload

    payload = read_tensor_payload(directory, descriptor)
    if not isinstance(tensor_type, TensorType) or not isinstance(tensor_type.encoding, NoneAttr):
        raise ValueError("reconstruction requires a static unencoded TensorType")
    shape = tensor_type.get_shape()
    element = tensor_type.get_element_type()
    if any(dim < 0 for dim in shape):
        raise ValueError("reconstruction requires a static unencoded TensorType")
    if isinstance(element, IntegerType):
        if element.width.data not in (8, 16, 32, 64):
            raise ValueError("unsupported inspection tensor element type")
    elif not isinstance(element, (Float16Type, BFloat16Type, Float32Type, Float64Type)):
        raise ValueError("unsupported inspection tensor element type")
    if list(shape) != descriptor["shape"] or str(element) != descriptor["element_type"]:
        raise ValueError("requested tensor type differs from inspection descriptor")
    if len(payload) != prod(shape) * element.compile_time_size:
        raise ValueError("inspection tensor storage does not match its static type")
    return DenseIntOrFPElementsAttr(tensor_type, BytesAttr(payload))


def compact_text(module, *, elements_limit: int = 64, tensor_sink=None) -> str:
    """Generic operation syntax with large dense tensor attributes elided structurally.

    Custom operation printers may bypass attribute dispatch, so inspection deliberately
    uses generic syntax. This is a diagnostic view, not a round-trippable module or
    executable tensor externalization. An optional sink preserves original dense storage
    bytes for inspection alongside the view. Neither attributes nor operations are modified.
    """
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr
    from xdsl.printer import Printer

    if elements_limit < 0:
        raise ValueError("inspection element limit must be nonnegative")

    class InspectionPrinter(Printer):
        def print_attribute(self, attribute):
            if isinstance(attribute, DenseIntOrFPElementsAttr) and len(attribute) > elements_limit:
                if tensor_sink is None:
                    self.print_string("dense<...> : ")
                else:
                    descriptor = tensor_sink(
                        attribute.data.data,
                        element_type=str(attribute.get_element_type()),
                        shape=tuple(attribute.get_shape()),
                    )
                    self.print_string(
                        f"inspection_tensor<file={json.dumps(descriptor['file'])}, "
                        f"sha256={json.dumps(descriptor['sha256'])}> : "
                    )
                self.print_attribute(attribute.get_type())
            else:
                super().print_attribute(attribute)

    output = StringIO()
    storage = "referenced as raw payloads" if tensor_sink is not None else "elided"
    output.write(f"// xDSL generic inspection; dense tensors above {elements_limit} elements are {storage}.\n")
    InspectionPrinter(stream=output, print_generic_format=True).print_op(module)
    return output.getvalue()


def record_stage(audit, name: str, module, *, generic: bool = False) -> None:
    """Serialize in the caller's executable syntax for exact-byte attribution.

    With no audit directory the module is serialized only if an open compile trace wants this stage
    (:func:`merlin.compiler.ir.compile_trace.observe` renders lazily); with one, the audit reports it."""

    if audit.directory is None:
        from merlin.compiler.ir import compile_trace

        compile_trace.observe(name, pipeline=audit.record["producer"], render=lambda: text(module, generic=generic))
        return

    tensors = []

    def tensor_sink(payload, *, element_type, shape):
        descriptor = audit.tensor(payload, element_type=element_type, shape=shape)
        tensors.append(descriptor)
        return descriptor

    inspection = compact_text(module, tensor_sink=tensor_sink) if audit.mode in {"compact", "both"} else None
    audit.stage(
        name,
        text(module, generic=generic),
        inspection=inspection,
        inspection_tensors=tensors if inspection is not None else None,
    )


if HAS_XDSL:
    from merlin.compiler.dialects.fp8 import Float8E4M3FNType as Float8E4M3FNType
    from merlin.compiler.dialects.fp8 import Float8E4M3FNUZType as Float8E4M3FNUZType
    from merlin.compiler.dialects.fp8 import Float8E5M2FNUZType as Float8E5M2FNUZType
    from merlin.compiler.dialects.fp8 import Float8E5M2Type as Float8E5M2Type
