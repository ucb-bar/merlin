"""Bit-preserving dense float IO at Merlin's xDSL boundary.

xDSL 0.68 reads unquoted dense float hex literals as numeric integers, and
its default printer decodes/re-encodes NaNs. Keep actual bytes authoritative.
All grammar, shape checks and non-floating literals remain xDSL-owned. These
adapters confer no numerical equivalence or permission to change arithmetic.
"""

from xdsl.dialects.builtin import AnyFloat, BytesAttr, DenseIntOrFPElementsAttr
from xdsl.parser import Parser
from xdsl.printer import Printer


class DenseFloatPrinter(Printer):
    """Use MLIR's raw-byte form, retaining NaN payloads and signed zeros."""

    def print_attribute(self, attribute):
        if isinstance(attribute, DenseIntOrFPElementsAttr) and isinstance(attribute.get_element_type(), AnyFloat):
            data = attribute.data.data
            width = attribute.get_element_type().compile_time_size
            # Compare bytes: NaN equality and +0/-0 equality cannot identify splats.
            if data and data == data[:width] * (len(data) // width):
                data = data[:width]
            self.print_string(f'dense<"0x{data.hex().upper()}"> : ')
            self.print_attribute(attribute.type)
            return
        super().print_attribute(attribute)


class DenseFloatParser(Parser):
    """Retain typed hex bit patterns without passing them through Python floats.

    The parent parses and validates the complete dense grammar. Hex float leaves
    use a temporary zero during its packing, then their exact original bytes replace
    those slots. This also avoids overflow and signaling-NaN quieting in conversion.
    Nested attribute parsing has its own saved leaf list, never global parser state.
    """

    class _DenseElement(Parser._TensorLiteralElement):
        def to_type(self, parser, type):
            leaves = getattr(parser, "_dense_float_leaves", None)
            if leaves is None:
                return super().to_type(parser, type)
            spelling = self.span.text.lower()
            raw = None
            if isinstance(type, AnyFloat):
                if spelling.startswith("-0x"):
                    parser.raise_error(
                        "Floating-point hex bit patterns cannot have a sign prefix", at_position=self.span
                    )
                if spelling.startswith("0x"):
                    try:
                        raw = self.value.to_bytes(type.compile_time_size, "little")
                    except (OverflowError, ValueError) as exc:
                        parser.raise_error(
                            f"Floating-point bit pattern does not fit {type}: {exc}", at_position=self.span
                        )
            leaves.append(raw)
            return 0.0 if raw is not None else super().to_type(parser, type)

    def _parse_tensor_literal_element(self):
        element = super()._parse_tensor_literal_element()
        return self._DenseElement(element.is_negative, element.value, element.span)

    def parse_dense_int_or_fp_elements_attr(self, type):
        previous = getattr(self, "_dense_float_leaves", None)
        self._dense_float_leaves = []
        try:
            attribute = super().parse_dense_int_or_fp_elements_attr(type)
            leaves = self._dense_float_leaves
            if not any(raw is not None for raw in leaves):
                return attribute
            data = bytearray(attribute.data.data)
            width = attribute.get_element_type().compile_time_size
            if len(leaves) == 1:
                data[:] = leaves[0] * len(attribute)
            else:
                for index, raw in enumerate(leaves):
                    if raw is not None:
                        data[index * width : (index + 1) * width] = raw
            return DenseIntOrFPElementsAttr(attribute.type, BytesAttr(bytes(data)))
        finally:
            self._dense_float_leaves = previous
