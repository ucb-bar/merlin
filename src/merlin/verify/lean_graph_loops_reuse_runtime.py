"""Finite initialized xDSL witnesses ONLY; no formal/native SCF evaluator bridge.

Instrumented private allocations begin with nonzero poison, and every private
load checks initialization and liveness. Allocation/free traces remain separate
because those observations are explicitly excluded by the qualified model.
"""

from __future__ import annotations

from math import prod


def execute(module, inputs, *, index_bits):
    from xdsl.dialects import arith, func, memref
    from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
    from xdsl.interpreters.arith import ArithFunctions
    from xdsl.interpreters.func import FuncFunctions
    from xdsl.interpreters.memref import MemRefFunctions, xtype_for_el_type
    from xdsl.interpreters.scf import ScfFunctions
    from xdsl.interpreters.shaped_array import ShapedArray
    from xdsl.interpreters.utils.ptr import TypedPtr

    private, allocations = {}, []
    caller = inputs[-1]

    def check(array, indices):
        if len(indices) != len(array.shape) or any(
            not 0 <= index < size for index, size in zip(indices, array.shape, strict=True)
        ):
            raise ValueError("out-of-bounds finite memory witness")
        if id(array) in private and not private[id(array)]["live"]:
            raise ValueError("use after private free in finite memory witness")

    @register_impls
    class PrivateMemory(InterpreterFunctions):
        @impl(memref.AllocOp)
        def allocate(self, interpreter, op, args):
            kind = op.memref.type
            shape = list(kind.get_shape())
            if prod(shape) > 4096:
                raise ValueError("finite execution refuses large-shape dynamic expansion")
            array = ShapedArray(
                TypedPtr.new([73] * prod(shape), xtype=xtype_for_el_type(kind.get_element_type(), index_bits)), shape
            )
            private[id(array)] = dict(
                live=True, written=set(), bytes=prod(shape) * (kind.get_element_type().width.data // 8)
            )
            allocations.append(dict(kind="allocate", bytes=private[id(array)]["bytes"], array=id(array)))
            return (array,)

        @impl(memref.DeallocOp)
        def release(self, interpreter, op, args):
            (array,) = args
            if id(array) not in private or not private[id(array)]["live"]:
                raise ValueError("invalid/double private free in finite memory witness")
            private[id(array)]["live"] = False
            allocations.append(dict(kind="free", bytes=private[id(array)]["bytes"], array=id(array)))
            return ()

        @impl(memref.StoreOp)
        def store(self, interpreter, op, args):
            value, array, *indices = args
            coordinate = tuple(indices)
            check(array, coordinate)
            array.store(coordinate, value)
            if id(array) in private:
                private[id(array)]["written"].add(coordinate)
            return ()

        @impl(memref.LoadOp)
        def load(self, interpreter, op, args):
            array, *indices = args
            coordinate = tuple(indices)
            check(array, coordinate)
            if id(array) in private and coordinate not in private[id(array)]["written"]:
                raise ValueError("uninitialized private load in finite memory witness")
            return (array.load(coordinate),)

        @impl(memref.DimOp)
        def dimension(self, interpreter, op, args):
            array, axis = args
            if id(array) in private and not private[id(array)]["live"]:
                raise ValueError("dimension after private free in finite memory witness")
            if not 0 <= axis < len(array.shape):
                raise ValueError("dimension axis out of range in finite memory witness")
            return (array.shape[axis],)

    def snapshot(value):
        if isinstance(value, ShapedArray):
            if id(value) in private:
                return ("private", tuple(value.shape))
            return ("caller", tuple(value.shape), tuple(value.data))
        assert type(value) is int
        return value

    tracked = (memref.LoadOp, memref.StoreOp, memref.DimOp, arith.AddiOp, arith.TruncIOp, func.CallOp, func.ReturnOp)

    class Listener(Interpreter.Listener):
        def __init__(self):
            self.trace = []

        def will_interpret_op(self, op, args):
            if isinstance(op, tracked):
                self.trace.append(("before", op.name, tuple(snapshot(arg) for arg in args), tuple(caller.data)))

        def did_interpret_op(self, op, results):
            if isinstance(op, tracked):
                self.trace.append(("after", op.name, tuple(snapshot(value) for value in results), tuple(caller.data)))

    listener = Listener()
    interpreter = Interpreter(module, index_bitwidth=index_bits, listeners=(listener,))
    for functions in (ArithFunctions(), ScfFunctions(), MemRefFunctions(), FuncFunctions()):
        interpreter.register_implementations(functions)
    interpreter.register_implementations(PrivateMemory(), override=True)
    result = interpreter.call_op("kernel", inputs)
    assert interpreter.call_op("untouched", (2**31 - 1,)) == (-(2**31),)
    if any(record["live"] for record in private.values()):
        raise ValueError("missing final physical free in finite memory witness")
    used, peak = 0, 0
    for row in allocations:
        used += row["bytes"] if row["kind"] == "allocate" else -row["bytes"]
        peak = max(peak, used)
        del row["array"]
    assert used == 0
    return dict(
        result=result,
        trace=listener.trace,
        caller=list(caller.data),
        allocation_events=allocations,
        allocation_count=sum(row["kind"] == "allocate" for row in allocations),
        peak_live_bytes=peak,
        poisoned_initial_private_memory=True,
        all_private_reads_initialized=True,
        all_private_frees_valid=True,
        actual_native_execution=False,
        formal_SCF_bridge_proved=False,
    )
