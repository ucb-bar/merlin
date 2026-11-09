"""Bound ordinary native memcpy to the admitted dense descriptor ABI.

Descriptor offsets, sizes and strides come from the generated wrapper. A copy
needs static spans inside those ports and nonoverlap under their existing ABI.
Volatile, unknown, escaping, padded-element and nonintegral-pointer forms refuse.
See https://llvm.org/docs/LangRef.html#llvm-memcpy-intrinsic .
"""
from __future__ import annotations

from math import prod

from xdsl.dialects import builtin, llvm

_WIDTH = {"i8": 1, "i16": 2, "f16": 2, "bf16": 2, "i32": 4, "f32": 4, "i64": 8, "f64": 8}


class CopyBounds:
    """Static descriptor substitutions; no assumptions about a loop induction variable."""

    def __init__(self, compute, ports, input_count, layout):
        self.ports, self.input_count, self.layout = ports, input_count, layout
        self.integers, self.spans, self.nulls, self.sizes, self.copies = {}, {}, set(), {}, []
        arguments, cursor = tuple(compute.body.blocks.first.args), 0
        if len(arguments) != sum(3 + 2 * len(port.shape) for port in ports):
            raise ValueError("native copy descriptor roster is incomplete")
        for number, port in enumerate(ports):
            self.spans[arguments[cursor]] = self.spans[arguments[cursor + 1]] = (number, 0)
            self._integer(arguments[cursor + 2], 0)
            strides, stride = [], 1
            for extent in reversed(port.shape):
                strides.append(stride)
                stride *= extent
            for argument, value in zip(arguments[cursor + 3:cursor + 3 + 2 * len(port.shape)],
                                       (*port.shape, *reversed(strides)), strict=True):
                self._integer(argument, value)
            cursor += 3 + 2 * len(port.shape)

    def _integer(self, result, value):
        if type(result.type) is builtin.IntegerType:
            bits = result.type.width.data
            if -(1 << (bits - 1)) <= value < 1 << (bits - 1):
                self.integers[result] = value

    def _width(self, typ):
        dtype = str(typ)
        if dtype not in _WIDTH:
            return None
        # The dense port ABI does not admit physical padding between primitives.
        # LLVM's default entries cover every supported whole-byte scalar here;
        # selected overrides must retain that allocation size.
        bits = _WIDTH[dtype] * 8
        key = ("f" if dtype == "bf16" else dtype[0]) + str(bits)
        for item in self.layout.split("-"):
            head, *values = item.split(":")
            if head == "ni" and "0" in values:
                raise ValueError("native copy cannot infer sizeof through a nonintegral default pointer")
            if head == key:
                if (len(values) not in (1, 2) or any(not part.isdecimal() for part in values)
                        or int(values[0]) <= 0 or int(values[0]) > bits
                        or int(values[0]) & (int(values[0]) - 1)):
                    raise ValueError("native copy element allocation differs from the dense port ABI")
        return _WIDTH[dtype]

    def _index(self, operation):
        indices = operation.properties.get("rawConstantIndices")
        if type(indices) is not builtin.DenseArrayBase:
            return None
        values = tuple(indices.get_values())
        if len(values) != 1:
            return None
        if values[0] == llvm.GEP_USE_SSA_VAL:
            return self.integers.get(operation.operands[1]) if len(operation.operands) == 2 else None
        return values[0] if len(operation.operands) == 1 else None

    def observe(self, operation, name):
        """Track static values and consume only the special sizeof-pointer forms."""
        operands, results = tuple(operation.operands), tuple(operation.results)
        if name == "llvm.mlir.zero":
            if operands or len(results) != 1 or results[0].type != llvm.LLVMPointerType() or operation.properties:
                raise ValueError("native sizeof needs an ordinary null pointer")
            self.nulls.add(results[0])
            return True
        if name == "llvm.mlir.constant" and len(results) == 1:
            value = operation.properties.get("value")
            if type(value) is builtin.IntegerAttr:
                self._integer(results[0], value.value.data)
        elif name in {"llvm.mul", "llvm.add", "llvm.sub"} and len(operands) == 2 and len(results) == 1:
            left, right = (self.integers.get(value) for value in operands)
            flags = operation.properties.get("overflowFlags")
            # A mathematically small signed result can still be poison under
            # unsigned no-wrap (for example -1 + 2). Ordinary loop arithmetic
            # retains its existing admission; COPY sizes need this extra proof.
            if (left is not None and right is not None
                    and type(flags) is builtin.IntegerAttr and flags.value.data == 0):
                value = left * right if name == "llvm.mul" else left + right if name == "llvm.add" else left - right
                self._integer(results[0], value)
        elif name == "llvm.bitcast" and len(operands) == len(results) == 1:
            if operands[0] in self.spans and results[0].type == llvm.LLVMPointerType():
                self.spans[results[0]] = self.spans[operands[0]]
        elif name == "llvm.getelementptr" and operands and len(results) == 1:
            base, index = operands[0], self._index(operation)
            width = self._width(operation.properties.get("elem_type"))
            if base in self.nulls:
                flags = operation.properties.get("noWrapFlags")
                if (index != 1 or width is None or type(flags) is not builtin.IntegerAttr or flags.value.data != 0
                        or results[0].type != llvm.LLVMPointerType()):
                    raise ValueError("native sizeof needs a non-inbounds one-element primitive null GEP")
                self.sizes[results[0]] = width
                return True
            if (base in self.spans and index is not None and width is not None
                    and results[0].type == llvm.LLVMPointerType()):
                port, offset = self.spans[base]
                flags = operation.properties.get("noWrapFlags")
                ordinary = type(flags) is builtin.IntegerAttr and flags.value.data == 0
                extent = prod(self.ports[port].shape) * _WIDTH[self.ports[port].dtype]
                # A flagged out-of-allocation intermediate can be poison even
                # when a later GEP returns inside. Only unchanged, already
                # bounded pointers need no additional target-index proof here.
                if ordinary or (type(flags) is builtin.IntegerAttr and index == 0 and 0 <= offset <= extent):
                    self.spans[results[0]] = (port, offset + index * width)
        elif name == "llvm.ptrtoint":
            if (len(operands) != 1 or len(results) != 1 or operands[0] not in self.sizes
                    or type(results[0].type) is not builtin.IntegerType):
                raise ValueError("native pointer-to-integer use is not a proven primitive sizeof")
            self._integer(results[0], self.sizes[operands[0]])
            return True
        return False

    def admit(self, operation):
        operands = tuple(operation.operands)
        volatile = operation.properties.get("isVolatile")
        if (len(operands) != 3 or operation.results or set(operation.properties) != {"isVolatile"}
                or type(volatile) is not builtin.IntegerAttr or volatile.type != builtin.i1 or volatile.value.data != 0
                or any(value.type != llvm.LLVMPointerType() for value in operands[:2])
                or set(operation.attributes) - {"op_name__"}):
            raise ValueError("native copy requires the exact ordinary nonvolatile memcpy contract")
        destination, source = (self.spans.get(value) for value in operands[:2])
        length = self.integers.get(operands[2])
        if destination is None or source is None or length is None or length <= 0:
            raise ValueError("native copy needs static positive length and proven dense descriptor spans")
        dst, dst_offset = destination
        src, src_offset = source
        if dst < self.input_count:
            raise ValueError("native copy can mutate a read-only input")
        for number, offset in (destination, source):
            port = self.ports[number]
            if offset < 0 or offset + length > prod(port.shape) * _WIDTH[port.dtype]:
                raise ValueError("native copy exceeds its admitted dense tensor extent")
        if dst == src and not (dst_offset + length <= src_offset or src_offset + length <= dst_offset):
            raise ValueError("native copy needs proven disjoint descriptor spans")
        self.copies.append({"destination": dst, "source": src, "destination_offset": dst_offset,
                            "source_offset": src_offset, "bytes": length})
        return dst
