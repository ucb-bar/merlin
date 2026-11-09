"""Native descriptor-copy admission: real LLVM shapes, pointer bounds and effects.

These fixtures model the scalar lowering's descriptor-size/primitive-sizeof copy
without depending on a captured compiler product or claiming native execution.
"""

from io import StringIO

import pytest
from xdsl.dialects import builtin, llvm
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from merlin.compiler.backend.cabi import TensorPort
from merlin.compiler.lower.llvm import call

LAYOUT = "e-p:64:64-i64:64-n32:64-S128"
PORTS = (TensorPort((2, 2), "f32"),) * 3


class NativeCopy:
    """One read-only input and two disjoint dense output descriptors."""

    def __init__(self, layout=LAYOUT):
        pointer, index = llvm.LLVMPointerType(), builtin.i64
        descriptor = [pointer, pointer, index, index, index, index, index]
        self.block = Block(arg_types=descriptor * 3)
        self.function = llvm.FuncOp("compute", llvm.LLVMFunctionType(descriptor * 3, llvm.LLVMVoidType()),
                                    linkage=llvm.LinkageAttr("external"), body=Region(self.block))
        self.module = builtin.ModuleOp([self.function],
                                      attributes={"llvm.data_layout": builtin.StringAttr(layout)})
        self.input, self.output, self.scratch = (self.block.args[i] for i in (1, 8, 15))

    def add(self, operation):
        self.block.add_op(operation)
        return operation.results[0] if operation.results else None

    def integer(self, value, typ=builtin.i64):
        return self.add(llvm.ConstantOp(builtin.IntegerAttr(value, typ), typ))

    def gep(self, pointer, index, *, inbounds=False):
        operation = llvm.GEPOp.from_mixed_indices(pointer, [index], builtin.f32)
        # Native MLIR uses noWrapFlags; xDSL's legacy constructor emits a
        # separate `inbounds` property instead. Exercise the actual transport.
        if inbounds:
            operation.properties["noWrapFlags"] = builtin.IntegerAttr(1, builtin.i32)
        return self.add(operation)

    def length(self):
        # Native lowering multiplies actual descriptor sizes by sizeof(f32).
        null = self.add(llvm.ZeroOp(result_types=[llvm.LLVMPointerType()]))
        end = self.gep(null, 1)
        width = self.add(llvm.PtrToIntOp(end, builtin.i64))
        elements = self.add(llvm.MulOp(self.block.args[10], self.block.args[11]))
        return self.add(llvm.MulOp(elements, width))

    def copy(self, destination, source, length, *, volatile=False, attributes=None):
        return self.add(builtin.UnregisteredOp.with_name("llvm.intr.memcpy").create(
            operands=[destination, source, length], properties={"isVolatile": builtin.IntegerAttr(
                int(volatile), builtin.i1)}, attributes=attributes or {}))

    def admit(self):
        self.block.add_op(llvm.ReturnOp())
        self.module.verify()
        # Exercise the same generic-IR intake and memory boundary as the service.
        stream = StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(self.module)
        module = call._parse_native(stream.getvalue())
        return call._memory(module.body.block.first_op, module, PORTS, 1)


def test_native_descriptor_sizes_sizeof_and_gep_produce_bounded_copy_witness():
    native = NativeCopy()
    length = native.length()
    source = native.gep(native.input, native.block.args[2])
    destination = native.gep(native.output, native.block.args[9])
    native.copy(destination, source, length)
    native.copy(native.scratch, native.output, length)
    record = native.admit()
    assert record["inputs"] == "read_only" and record["retained_pointers"] is False
    assert record["bounded_copies"] == [
        {"destination": 1, "source": 0, "destination_offset": 0, "source_offset": 0, "bytes": 16},
        {"destination": 2, "source": 1, "destination_offset": 0, "source_offset": 0, "bytes": 16},
    ]


def test_nonoverlapping_subspans_of_one_output_are_admitted():
    native = NativeCopy()
    native.copy(native.output, native.input, native.length())
    native.copy(native.gep(native.output, 2), native.output, native.integer(8))
    native.copy(native.scratch, native.output, native.integer(16))
    copies = native.admit()["bounded_copies"]
    assert copies[1] == {"destination": 1, "source": 1, "destination_offset": 8,
                         "source_offset": 0, "bytes": 8}


@pytest.mark.parametrize("case, message", [
    ("oversize", "extent"), ("negative-offset", "extent"), ("past-end", "extent"),
    ("input-write", "read-only input"), ("same-span", "disjoint"), ("overlap", "disjoint"),
    ("volatile", "nonvolatile"), ("semantic-attribute", "nonvolatile"),
    ("zero-length", "positive length"), ("negative-length", "positive length"),
    ("integer-overflow", "positive length"), ("unsigned-overflow", "positive length"),
    ("pointer-escape", "pointer-to-integer"),
    ("padded-layout", "allocation"), ("nonintegral-pointer", "nonintegral"),
    ("sizeof-inbounds", "non-inbounds"), ("sizeof-two-elements", "one-element"),
])
def test_unproved_or_effectful_copy_refuses(case, message):
    layout = LAYOUT + ("-f32:64" if case == "padded-layout" else "-ni:0" if case == "nonintegral-pointer" else "")
    native = NativeCopy(layout)
    destination, source = native.output, native.input
    if case == "pointer-escape":
        length = native.add(llvm.PtrToIntOp(native.input))
    elif case.startswith("sizeof-"):
        null = native.add(llvm.ZeroOp(result_types=[llvm.LLVMPointerType()]))
        end = native.gep(null, 2 if case == "sizeof-two-elements" else 1,
                         inbounds=case == "sizeof-inbounds")
        length = native.add(llvm.PtrToIntOp(end))
    else:
        length = native.length()
    if case in {"oversize", "zero-length", "negative-length"}:
        length = native.integer({"oversize": 17, "zero-length": 0, "negative-length": -1}[case])
    elif case == "integer-overflow":
        length = native.add(llvm.AddOp(native.integer((1 << 63) - 1), native.integer(1)))
    elif case == "unsigned-overflow":
        # Signed arithmetic fits, but the selected native `nuw` promise fails.
        addition = llvm.AddOp(native.integer(-1), native.integer(2))
        addition.properties["overflowFlags"] = builtin.IntegerAttr(2, builtin.i32)
        length = native.add(addition)
    elif case == "negative-offset":
        destination = native.gep(destination, -1)
    elif case == "past-end":
        destination = native.gep(destination, 4)
    elif case == "input-write":
        destination = native.input
    elif case in {"same-span", "overlap"}:
        source = native.output
        if case == "overlap":
            destination, length = native.gep(destination, 1), native.integer(8)
    native.copy(destination, source, length, volatile=case == "volatile",
                attributes={"unknown_policy": builtin.UnitAttr()} if case == "semantic-attribute" else None)
    native.copy(native.scratch, native.input, native.integer(16))
    with pytest.raises(ValueError, match=message):
        native.admit()


def test_dynamic_nonentry_length_is_not_a_descriptor_constant():
    native = NativeCopy()
    other = Block(arg_types=[builtin.i64])
    native.block.add_op(llvm.BrOp(other, native.block.args[3]))
    native.function.body.add_block(other)
    native.block = other
    native.copy(native.output, native.input, other.args[0])
    with pytest.raises(ValueError, match="static positive length"):
        native.admit()


@pytest.mark.parametrize("escape", ["external-call", "store-pointer"])
def test_memcpy_does_not_authorize_pointer_escape(escape):
    native = NativeCopy()
    native.copy(native.output, native.input, native.length())
    native.copy(native.scratch, native.output, native.integer(16))
    if escape == "external-call":
        native.module.body.block.add_op(llvm.FuncOp("unadmitted", llvm.LLVMFunctionType(
            [llvm.LLVMPointerType()], llvm.LLVMVoidType()), linkage=llvm.LinkageAttr("external")))
        native.add(llvm.CallOp(builtin.SymbolRefAttr("unadmitted"), native.output))
    else:
        native.add(llvm.StoreOp(native.input, native.scratch))
    with pytest.raises(ValueError, match="effects|escape pointers"):
        native.admit()


def test_poisoned_inbounds_intermediate_cannot_be_hidden_by_returning_inside():
    native = NativeCopy()
    outside = native.gep(native.output, 8, inbounds=True)
    returned = native.gep(outside, -8)
    native.copy(returned, native.input, native.length())
    native.copy(native.scratch, native.input, native.integer(16))
    with pytest.raises(ValueError, match="GEP|bounds|pointer|spans"):
        native.admit()
