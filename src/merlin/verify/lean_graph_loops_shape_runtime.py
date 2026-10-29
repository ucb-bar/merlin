"""Finite xDSL shape-pass executions, separate from proof and native fidelity.

Only identity casts and dimension reads are excluded from the retained-operation
trace, after checking their descriptor domain. Index constant materialization is
also unobserved. Partial memory operations validate coordinates and initialization.
"""

from __future__ import annotations

from math import prod


def execute(module, inputs, *, function, index_bits, max_cells=4096):
    from xdsl.dialects import arith, builtin, func, memref
    from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
    from xdsl.interpreters.arith import ArithFunctions
    from xdsl.interpreters.func import FuncFunctions
    from xdsl.interpreters.memref import MemRefFunctions, xtype_for_el_type
    from xdsl.interpreters.scf import ScfFunctions
    from xdsl.interpreters.shaped_array import ShapedArray
    from xdsl.interpreters.utils.ptr import TypedPtr

    if index_bits not in (32, 64) or type(max_cells) is not int or max_cells <= 0:
        raise ValueError("finite execution requires explicit index width and positive allocation budget")
    module.verify()
    entry = next(op for op in module.walk() if isinstance(op, func.FuncOp) and op.sym_name.data == function)
    if len(inputs) != len(entry.function_type.inputs):
        raise ValueError("entry argument arity differs")
    memories, private = {}, {}

    def descriptor(array, kind):
        if not isinstance(kind, builtin.MemRefType) or not isinstance(array, ShapedArray):
            raise ValueError("finite view requires a ranked descriptor")
        if not isinstance(kind.layout, builtin.NoneAttr) or not isinstance(kind.memory_space, builtin.NoneAttr):
            raise ValueError("finite view requires identity layout and default space")
        if list(kind.get_shape()) != array.shape or any(size < 0 for size in array.shape):
            raise ValueError("runtime descriptor violates static source type")
        if prod(array.shape) > max_cells:
            raise ValueError("finite runtime refuses large volume; symbolic proof remains separate")

    for index, (value, kind) in enumerate(zip(inputs, entry.function_type.inputs, strict=True)):
        if isinstance(kind, builtin.MemRefType):
            descriptor(value, kind)
            memories.setdefault(id(value), dict(name=f"input{index}", array=value))
        elif not isinstance(kind, (builtin.IntegerType, builtin.IndexType)) or type(value) is not int:
            raise ValueError("finite scalar entry domain unsupported")

    def access(array, indices):
        if id(array) not in memories or len(indices) != len(array.shape):
            raise ValueError("unknown descriptor/rank in finite memory access")
        if any(type(i) is not int or not 0 <= i < size for i, size in zip(indices, array.shape, strict=True)):
            raise ValueError("out-of-bounds finite memory access")
        if id(array) in private and not private[id(array)]["live"]:
            raise ValueError("use after free in finite execution")

    @register_impls
    class CheckedMemory(InterpreterFunctions):
        @impl(memref.CastOp)
        def cast(self, interpreter, op, args):
            (array,) = args
            if op.source.type != op.dest.type:
                raise ValueError("finite shape witness only admits identical casts")
            descriptor(array, op.source.type)
            return (array,)

        @impl(memref.DimOp)
        def dimension(self, interpreter, op, args):
            array, axis = args
            descriptor(array, op.source.type)
            if not 0 <= axis < len(array.shape):
                raise ValueError("dimension axis out of range")
            return (array.shape[axis],)

        @impl(memref.AllocOp)
        def allocate(self, interpreter, op, args):
            kind = op.memref.type
            shape = list(kind.get_shape())
            if args or any(size < 0 for size in shape) or prod(shape) > max_cells:
                raise ValueError("finite allocation outside static bounded domain")
            array = ShapedArray(
                TypedPtr.new([73] * prod(shape), xtype=xtype_for_el_type(kind.get_element_type(), index_bits)), shape
            )
            descriptor(array, kind)
            memories[id(array)] = dict(name=f"private{len(private)}", array=array)
            private[id(array)] = dict(live=True, initialized=set())
            return (array,)

        @impl(memref.DeallocOp)
        def release(self, interpreter, op, args):
            (array,) = args
            if id(array) not in private or not private[id(array)]["live"]:
                raise ValueError("invalid/double private free")
            private[id(array)]["live"] = False
            return ()

        @impl(memref.LoadOp)
        def load(self, interpreter, op, args):
            array, *indices = args
            access(array, indices)
            if id(array) in private and tuple(indices) not in private[id(array)]["initialized"]:
                raise ValueError("uninitialized private read")
            return (array.load(tuple(indices)),)

        @impl(memref.StoreOp)
        def store(self, interpreter, op, args):
            value, array, *indices = args
            access(array, indices)
            array.store(tuple(indices), value)
            if id(array) in private:
                private[id(array)]["initialized"].add(tuple(indices))
            return ()

    def snapshot(value):
        if isinstance(value, ShapedArray):
            return dict(view=memories[id(value)]["name"], shape=list(value.shape), cells=list(value.data))
        if type(value) is not int:
            raise ValueError("unknown finite observation")
        return value

    def tracked(op):
        if isinstance(op, (memref.CastOp, memref.DimOp)):
            return False
        if isinstance(op, arith.ConstantOp) and isinstance(op.result.type, builtin.IndexType):
            return False
        return op.name.startswith(("memref.", "arith.", "func."))

    class Listener(Interpreter.Listener):
        def __init__(self):
            self.trace = []

        def will_interpret_op(self, op, args):
            if tracked(op):
                self.trace.append(dict(phase="before", operation=op.name, values=[snapshot(arg) for arg in args]))

        def did_interpret_op(self, op, results):
            if tracked(op):
                self.trace.append(dict(phase="after", operation=op.name, values=[snapshot(value) for value in results]))

    listener = Listener()
    interpreter = Interpreter(module, index_bitwidth=index_bits, listeners=(listener,))
    for functions in (ArithFunctions(), ScfFunctions(), MemRefFunctions(), FuncFunctions()):
        interpreter.register_implementations(functions)
    interpreter.register_implementations(CheckedMemory(), override=True)
    outputs = interpreter.call_op(function, inputs)
    if any(row["live"] for row in private.values()):
        raise ValueError("finite closed-allocation witness has a missing free")
    return dict(
        outputs=[snapshot(value) for value in outputs],
        trace=listener.trace,
        input_memories=[snapshot(row["array"]) for identity, row in memories.items() if identity not in private],
        initialized_private_reads=True,
        valid_private_frees=True,
        descriptor_conformance_checked_on_finite_entries=True,
        actual_native_execution=False,
        native_or_abi_fidelity_proved=False,
        python_interpreter_and_trace_projection_trusted=True,
    )
