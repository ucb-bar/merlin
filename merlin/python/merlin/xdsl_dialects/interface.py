"""The ``interface`` dialect (xDSL): target-independent HW/SW abstractions.

The central research dialect: what should software expose to hardware, and what should
hardware promise to software? Resident packed tensors, streaming tiles, accumulators,
commits, events, FIFOs, command regions. Lower than tensor/linalg, higher than target
instructions. Every major op carries a ``visibility`` DSE variant
(baseline/software_visible/hardware_managed/oracle).
See docs/core_dialects.md and docs/dialects.md.
"""
from __future__ import annotations

from ._common import HAS_XDSL, Visibility
# Importing the fp8 kit installs the parser hook so fp8 capsule interfaces
# (tensor<...xf8E4M3FN> / f8E5M2) parse without the agent registering the type.
from . import fp8 as _fp8  # noqa: F401 - imported for its registration side effect

DIALECT_NAME = "interface"
OPS = ["resident_pack", "resident_evict", "matmul", "accumulator.create", "accumulate",
       "commit", "async_copy", "await", "fifo.create", "fifo.push", "fifo.pop",
       "command.region"]
TYPES = ["resident_tensor", "streaming_tile", "accumulator", "committed_tensor",
         "event", "fifo", "command"]

# Epilogue stages `interface.commit` accepts — must stay aligned with the runtime
# engine's COMMIT semantics (merlin/python/merlin/runtime/simulator.py).
KNOWN_EPILOGUE = {"bias", "bias_add", "requant", "relu"}

# Output dtypes `interface.commit` can produce (engine: to_i8() or raw i32).
KNOWN_OUTPUT_DTYPES = {"i8", "i32"}

if HAS_XDSL:
    from xdsl.ir import (Attribute, Dialect, EnumAttribute, ParametrizedAttribute,
                         SpacedOpaqueSyntaxAttribute, TypeAttribute)
    from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                           operand_def, opt_prop_def, prop_def, region_def, result_def,
                           traits_def)
    from xdsl.traits import NoTerminator
    from xdsl.utils.exceptions import VerifyException
    from xdsl.utils.str_enum import StrEnum
    from xdsl.dialects.builtin import (ArrayAttr, IntegerAttr, IntegerType, StringAttr,
                                       TensorType)

    # -- enums ---------------------------------------------------------------

    class MemoryState(StrEnum):
        STREAMING = "streaming"
        RESIDENT = "resident"
        PACKED = "packed"
        ACCUMULATOR = "accumulator"
        COMMITTED = "committed"
        FIFO = "fifo"
        HARDWARE_MANAGED = "hardware_managed"
        SOFTWARE_VISIBLE = "software_visible"

    class Layout(StrEnum):
        CANONICAL = "canonical"
        PACKED_LHS = "packed_lhs"
        PACKED_RHS = "packed_rhs"
        TARGET_NATIVE = "target_native"
        OPAQUE = "opaque"

    class Lifetime(StrEnum):
        OP = "op"
        REGION = "region"
        DISPATCH = "dispatch"
        INVOCATION = "invocation"
        PERSISTENT_UNTIL_EVICT = "persistent_until_evict"

    @irdl_attr_definition
    class MemoryStateAttr(EnumAttribute[MemoryState], SpacedOpaqueSyntaxAttribute):
        """#interface.memory_state — memory state of an interface object."""
        name = "interface.memory_state"

    @irdl_attr_definition
    class VisibilityAttr(EnumAttribute[Visibility], SpacedOpaqueSyntaxAttribute):
        """#interface.visibility — DSE variant tag (essential for DSE)."""
        name = "interface.visibility"

    @irdl_attr_definition
    class LayoutAttr(EnumAttribute[Layout], SpacedOpaqueSyntaxAttribute):
        """#interface.layout — layout class of an interface object."""
        name = "interface.layout"

    @irdl_attr_definition
    class LifetimeAttr(EnumAttribute[Lifetime], SpacedOpaqueSyntaxAttribute):
        """#interface.lifetime — lifetime of an interface object."""
        name = "interface.lifetime"

    # -- types ---------------------------------------------------------------

    @irdl_attr_definition
    class ResidentTensorType(ParametrizedAttribute, TypeAttribute):
        """!interface.resident_tensor<tensor<...>, "layout"> — resident in
        target-managed or compiler-managed storage."""
        name = "interface.resident_tensor"
        element: Attribute
        layout: StringAttr

    @irdl_attr_definition
    class StreamingTileType(ParametrizedAttribute, TypeAttribute):
        """!interface.streaming_tile<tensor<...>> — consumed in streaming fashion."""
        name = "interface.streaming_tile"
        element: Attribute

    @irdl_attr_definition
    class AccumulatorType(ParametrizedAttribute, TypeAttribute):
        """!interface.accumulator<tensor<...>> — accumulation state, not yet committed."""
        name = "interface.accumulator"
        element: Attribute

    @irdl_attr_definition
    class CommittedTensorType(ParametrizedAttribute, TypeAttribute):
        """!interface.committed_tensor<tensor<...>> — final output after commit."""
        name = "interface.committed_tensor"
        element: Attribute

    @irdl_attr_definition
    class EventType(ParametrizedAttribute, TypeAttribute):
        """!interface.event — abstract dependency token."""
        name = "interface.event"

    @irdl_attr_definition
    class FifoType(ParametrizedAttribute, TypeAttribute):
        """!interface.fifo<tensor<...>> — abstract producer-consumer FIFO."""
        name = "interface.fifo"
        element: Attribute

    @irdl_attr_definition
    class CommandType(ParametrizedAttribute, TypeAttribute):
        """!interface.command — abstract command object before runtime lowering."""
        name = "interface.command"

    def _element_tensor(t: Attribute):
        """The TensorType wrapped by an interface object type, if any."""
        inner = getattr(t, "element", None)
        return inner if isinstance(inner, TensorType) else None

    # -- ops -----------------------------------------------------------------

    @irdl_op_definition
    class ResidentPackOp(IRDLOperation):
        """interface.resident_pack — packs + installs a tensor into resident storage.

        Verifier intent: source must be immutable/stable over the lifetime (a contract
        proof, checked by the contract layer); layout must match the result type.
        """
        name = "interface.resident_pack"
        src = operand_def()
        layout = prop_def(LayoutAttr)
        lifetime = opt_prop_def(LifetimeAttr)
        visibility = opt_prop_def(VisibilityAttr)
        res = result_def(ResidentTensorType)

        def verify_(self) -> None:
            if self.res.type.layout.data != self.layout.data.value:
                raise VerifyException(
                    "interface.resident_pack layout %r does not match result type "
                    "layout %r" % (self.layout.data.value, self.res.type.layout.data))

    @irdl_op_definition
    class ResidentEvictOp(IRDLOperation):
        """interface.resident_evict — evicts a resident object.

        No value may use the handle after eviction — cross-op check in
        lowering/analyses.py (check_no_use_after_evict).
        """
        name = "interface.resident_evict"
        handle = operand_def(ResidentTensorType)

    @irdl_op_definition
    class MatmulOp(IRDLOperation):
        """interface.matmul — target-independent matmul producing an accumulator.

        Consumes normal tensors, resident tensors, or streaming tiles; shape/dtype
        compatibility is checked when both sides carry static tensor types.
        """
        name = "interface.matmul"
        lhs = operand_def()
        rhs = operand_def()
        visibility = opt_prop_def(VisibilityAttr)
        acc = result_def(AccumulatorType)

        def verify_(self) -> None:
            lt = self.lhs.type if isinstance(self.lhs.type, TensorType) \
                else _element_tensor(self.lhs.type)
            rt = self.rhs.type if isinstance(self.rhs.type, TensorType) \
                else _element_tensor(self.rhs.type)
            if lt is not None and rt is not None:
                lshape = lt.get_shape()
                rshape = rt.get_shape()
                if len(lshape) == 2 and len(rshape) == 2 and lshape[1] != rshape[0]:
                    raise VerifyException(
                        "interface.matmul inner dims disagree: %s vs %s"
                        % (list(lshape), list(rshape)))

    @irdl_op_definition
    class AccumulatorCreateOp(IRDLOperation):
        """interface.accumulator.create — explicit accumulator object (usually optional)."""
        name = "interface.accumulator.create"
        acc = result_def(AccumulatorType)

    @irdl_op_definition
    class AccumulateOp(IRDLOperation):
        """interface.accumulate — accumulates a matmul contribution into an accumulator."""
        name = "interface.accumulate"
        acc = operand_def(AccumulatorType)
        lhs = operand_def()
        rhs = operand_def()
        visibility = opt_prop_def(VisibilityAttr)
        out = result_def(AccumulatorType)

        def verify_(self) -> None:
            if self.acc.type != self.out.type:
                raise VerifyException(
                    "interface.accumulate input/output accumulator types must match")

    @irdl_op_definition
    class CommitOp(IRDLOperation):
        """interface.commit — commits an accumulator to a tensor with an epilogue.

        The ``bias`` property names the bias tensor (the runtime engine references
        bias by name, not by SSA value). Lowering intent: target commit op then a
        runtime COMMIT command. Runtime interpretation: epilogue stages run in order,
        then the output dtype conversion.
        """
        name = "interface.commit"
        acc = operand_def(AccumulatorType)
        epilogue = prop_def(ArrayAttr)
        bias = opt_prop_def(StringAttr)
        requant_shift = opt_prop_def(IntegerAttr)
        output_dtype = opt_prop_def(StringAttr)
        visibility = opt_prop_def(VisibilityAttr)
        out = result_def()

        def verify_(self) -> None:
            stages = []
            for entry in self.epilogue:
                stage = entry.data if isinstance(entry, StringAttr) else None
                if stage not in KNOWN_EPILOGUE:
                    raise VerifyException(
                        "interface.commit epilogue stage %r not in %s"
                        % (stage, sorted(KNOWN_EPILOGUE)))
                stages.append(stage)
            if ("bias" in stages or "bias_add" in stages) and self.bias is None:
                raise VerifyException(
                    "interface.commit epilogue has a bias stage but no `bias` tensor name")
            if "requant" in stages and self.requant_shift is None:
                raise VerifyException(
                    "interface.commit epilogue has `requant` but no `requant_shift`")
            if self.output_dtype is not None:
                dtype = self.output_dtype.data
                if dtype not in KNOWN_OUTPUT_DTYPES:
                    raise VerifyException(
                        "interface.commit output_dtype %r not in %s"
                        % (dtype, sorted(KNOWN_OUTPUT_DTYPES)))
                out_t = self.out.type if isinstance(self.out.type, TensorType) \
                    else _element_tensor(self.out.type)
                if out_t is not None:
                    elem = out_t.element_type
                    if isinstance(elem, IntegerType):
                        want = int(dtype[1:])
                        if elem.width.data != want:
                            raise VerifyException(
                                "interface.commit output_dtype %s does not match result "
                                "element type %s" % (dtype, elem))

    @irdl_op_definition
    class AsyncCopyOp(IRDLOperation):
        """interface.async_copy — abstract asynchronous transfer producing an event."""
        name = "interface.async_copy"
        src = operand_def()
        dst = operand_def()
        kind = prop_def(StringAttr)
        bytes_ = opt_prop_def(IntegerAttr, prop_name="bytes")
        event = result_def(EventType)

        def verify_(self) -> None:
            if self.bytes_ is not None and self.bytes_.value.data < 0:
                raise VerifyException("interface.async_copy bytes must be non-negative")

    @irdl_op_definition
    class AwaitOp(IRDLOperation):
        """interface.await — waits on an abstract event."""
        name = "interface.await"
        event = operand_def(EventType)

    @irdl_op_definition
    class FifoCreateOp(IRDLOperation):
        """interface.fifo.create — creates a producer-consumer FIFO."""
        name = "interface.fifo.create"
        depth = prop_def(IntegerAttr)
        fifo = result_def(FifoType)

        def verify_(self) -> None:
            if self.depth.value.data <= 0:
                raise VerifyException("interface.fifo.create depth must be positive")

    @irdl_op_definition
    class FifoPushOp(IRDLOperation):
        """interface.fifo.push — producer side of a FIFO transfer."""
        name = "interface.fifo.push"
        fifo = operand_def(FifoType)
        value = operand_def()

        def verify_(self) -> None:
            if self.value.type != self.fifo.type.element:
                raise VerifyException(
                    "interface.fifo.push value type %s does not match fifo element %s"
                    % (self.value.type, self.fifo.type.element))

    @irdl_op_definition
    class FifoPopOp(IRDLOperation):
        """interface.fifo.pop — consumer side of a FIFO transfer."""
        name = "interface.fifo.pop"
        fifo = operand_def(FifoType)
        value = result_def()

        def verify_(self) -> None:
            if self.value.type != self.fifo.type.element:
                raise VerifyException(
                    "interface.fifo.pop result type %s does not match fifo element %s"
                    % (self.value.type, self.fifo.type.element))

    @irdl_op_definition
    class CommandRegionOp(IRDLOperation):
        """interface.command.region — compiler-level grouping of interface ops.

        Not the runtime command buffer: a grouping before target + runtime lowering.
        """
        name = "interface.command.region"
        sym_name = prop_def(StringAttr)
        visibility = opt_prop_def(VisibilityAttr)
        body = region_def()
        traits = traits_def(NoTerminator())

        def verify_(self) -> None:
            if not self.sym_name.data:
                raise VerifyException("interface.command.region must be named")

    _OP_CLASSES = [ResidentPackOp, ResidentEvictOp, MatmulOp, AccumulatorCreateOp,
                   AccumulateOp, CommitOp, AsyncCopyOp, AwaitOp, FifoCreateOp,
                   FifoPushOp, FifoPopOp, CommandRegionOp]
    _ATTR_CLASSES = [ResidentTensorType, StreamingTileType, AccumulatorType,
                     CommittedTensorType, EventType, FifoType, CommandType,
                     MemoryStateAttr, VisibilityAttr, LayoutAttr, LifetimeAttr]
    INTERFACE_DIALECT = Dialect(DIALECT_NAME, _OP_CLASSES, _ATTR_CLASSES)

    def get_dialect() -> Dialect:
        return INTERFACE_DIALECT

    def build_example(reuse: int = 2):
        """The repeated-RHS workload at the interface level: pack once, matmul/commit
        ``reuse`` times, evict once. Verifiable and round-trippable."""
        from xdsl.ir import Block, Region
        from xdsl.dialects.builtin import FunctionType, ModuleOp, i8, i32
        from xdsl.dialects.func import FuncOp, ReturnOp

        At = TensorType(i8, [64, 128])
        Wt = TensorType(i8, [128, 64])
        Yt = TensorType(i8, [64, 64])
        acc_t = AccumulatorType(TensorType(i32, [64, 64]))
        res_t = ResidentTensorType(Wt, StringAttr("packed_rhs"))

        arg_types = [TensorType(i8, [64, 128]) for _ in range(reuse)] + [Wt]
        blk = Block(arg_types=arg_types)
        a_args, w = list(blk.args[:-1]), blk.args[-1]
        ops = []
        pack = ResidentPackOp(operands=[w], result_types=[res_t], properties={
            "layout": LayoutAttr(Layout.PACKED_RHS),
            "lifetime": LifetimeAttr(Lifetime.REGION),
            "visibility": VisibilityAttr(Visibility.SOFTWARE_VISIBLE)})
        ops.append(pack)
        outs = []
        for a in a_args:
            mm = MatmulOp(operands=[a, pack.res], result_types=[acc_t], properties={
                "visibility": VisibilityAttr(Visibility.SOFTWARE_VISIBLE)})
            cm = CommitOp(operands=[mm.acc], result_types=[Yt], properties={
                "epilogue": ArrayAttr([StringAttr("bias_add"), StringAttr("requant"),
                                       StringAttr("relu")]),
                "bias": StringAttr("bias"),
                "requant_shift": IntegerAttr(4, 64),
                "output_dtype": StringAttr("i8")})
            ops += [mm, cm]
            outs.append(cm.out)
        ev = ResidentEvictOp(operands=[pack.res])
        ret = ReturnOp(*outs)
        blk.add_ops(ops + [ev, ret])
        fn = FuncOp("repeated_rhs_matmul",
                    FunctionType.from_lists(arg_types, [Yt] * reuse), Region([blk]))
        return ModuleOp([fn])

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None

    def build_example(reuse: int = 2):
        return None
