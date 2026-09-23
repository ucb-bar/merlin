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
OPS = ["resident_pack", "resident_evict", "matmul", "elementwise", "accumulator.create",
       "accumulate", "commit", "vector_map", "vector_reduce", "async_copy", "await",
       "fifo.create", "fifo.push", "fifo.pop", "command.region"]
TYPES = ["resident_tensor", "streaming_tile", "accumulator", "committed_tensor",
         "event", "fifo", "command"]

# Epilogue stages `interface.commit` accepts. DERIVED, not restated: the vocabulary is
# `merlin.runtime.commandbuffer.EPILOGUE_STAGES`, the single definition the command-buffer ABI, its JSON
# validator and the three engines all read (see that constant for why it is not derived from RTL facts).
# This module used to carry its own copy, and the copy had drifted in BOTH directions: it admitted
# `maxpool`, which the JSON validator rejected, and it rejected `acc_scale`, which the validator and both
# ABI documents admit and every engine implements. Rejecting a stage here that the runtime implements
# lets a module verify and then compute something the verifier declared impossible; admitting one no
# engine implements is the mirror defect.
from merlin.runtime.commandbuffer import EPILOGUE_STAGE_SET as KNOWN_EPILOGUE  # noqa: E402

#: The properties a ``maxpool`` epilogue stage needs to be executable. Checked by NAME on the raw
#: property dict rather than declared as ``opt_prop_def``s: the op's declared property set is part of
#: the dialect's shape (the target factory reproduces the hand-written dialects byte for byte), and the
#: geometry is carried through to the runtime command whatever the op declares. Same fail-closed rule
#: the ``requant``/``bias`` stages already follow -- a stage whose parameters are missing cannot be
#: executed, and a module that verified without them would fail (or worse, be defaulted) downstream.
POOL_REQUIRED_PROPS = ("pool_in_dims", "pool_size", "pool_stride")


def missing_pool_props(stages, properties) -> list[str]:
    """The ``maxpool`` geometry properties an op is missing, or ``[]`` when it needs none."""
    if "maxpool" not in stages:
        return []
    return [k for k in POOL_REQUIRED_PROPS if properties.get(k) is None]


# Output dtypes `interface.commit` can produce: integer commits (engine to_i8()/raw i32) and
# float commits (whole-model layers that stay in float — the accumulator element type is carried
# through unchanged). Kept as data, not a per-target assumption.
KNOWN_OUTPUT_DTYPES = {"i8", "i32", "f16", "f32", "f64", "bf16"}

# Vector-family (target scalar/vector lane) vocabulary for the non-matmul ops of a whole model —
# elementwise combine + activation + reduction. Must stay aligned with the runtime engine
# (merlin/python/merlin/runtime/{simulator,reference}.py VECTOR_MAP/VREDUCE) and the target factory.
KNOWN_COMBINE = {"add", "mul", "identity"}
KNOWN_ACTIVATION = {"relu"}
KNOWN_REDUCE = {"sum"}

# Combines and activations `interface.elementwise` accepts. These are exactly what the runtime's
# VECTOR_MAP command implements (`runtime/simulator.py`, `runtime/reference.py`) — accepting more
# here would let a module verify at the interface level and then fail, or worse compute nothing, at
# the runtime tier. `linalg.sub` is therefore NOT expressible: the runtime has no subtract combine,
# and lowering it as an add would be a miscompile.
KNOWN_COMBINES = {"add", "mul"}
KNOWN_ACTIVATIONS = {"relu"}

if HAS_XDSL:
    from xdsl.ir import (Attribute, Dialect, EnumAttribute, ParametrizedAttribute,
                         SpacedOpaqueSyntaxAttribute, TypeAttribute)
    from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                           operand_def, opt_operand_def, opt_prop_def, prop_def, region_def,
                           result_def, traits_def)
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
        # int8 weight-only: an optional per-channel scale operand + axis dequantizes the i8 source to
        # a float resident weight at pack time (model2MLIR's dequantize_per_channel idiom).
        scale = opt_operand_def()
        layout = prop_def(LayoutAttr)
        lifetime = opt_prop_def(LifetimeAttr)
        visibility = opt_prop_def(VisibilityAttr)
        dequant_axis = opt_prop_def(IntegerAttr)
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
    class ElementwiseOp(IRDLOperation):
        """interface.elementwise — target-independent elementwise combine of two tensors.

        The counterpart to :class:`MatmulOp` for the SIMD/vector family. It produces a tensor
        directly rather than an accumulator, because an elementwise combine has no accumulation
        phase to commit: there is nothing to hold open across dispatches, so a pack/commit/evict
        shape would be ceremony describing hardware behaviour that does not happen.

        This op is what several generated dialect plans were already declaring coverage for. Without
        it a target could claim an accelerated elementwise unit that the compiler had no way to
        reach, and the payload went silently down the generic path instead.
        """
        name = "interface.elementwise"
        lhs = operand_def()
        rhs = operand_def()
        combine = prop_def(StringAttr)
        activation = opt_prop_def(ArrayAttr)
        visibility = opt_prop_def(VisibilityAttr)
        out = result_def()

        def verify_(self) -> None:
            if self.combine.data not in KNOWN_COMBINES:
                raise VerifyException(
                    f"interface.elementwise combine {self.combine.data!r} not in "
                    f"{sorted(KNOWN_COMBINES)} — the runtime's VECTOR_MAP implements no other")
            for entry in (self.activation or ()):
                stage = entry.data if isinstance(entry, StringAttr) else None
                if stage not in KNOWN_ACTIVATIONS:
                    raise VerifyException(
                        f"interface.elementwise activation {stage!r} not in "
                        f"{sorted(KNOWN_ACTIVATIONS)}")
            types = [self.lhs.type, self.rhs.type, self.out.type]
            shaped = [t for t in types if isinstance(t, TensorType)]
            if len(shaped) == len(types) and len({tuple(t.get_shape()) for t in shaped}) != 1:
                raise VerifyException(
                    "interface.elementwise operands and result must have one shape, got "
                    + ", ".join(str(tuple(t.get_shape())) for t in shaped))

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
            missing = missing_pool_props(stages, self.properties)
            if missing:
                raise VerifyException(
                    "interface.commit epilogue has `maxpool` but no %s; a pooling stage with no window "
                    "cannot be executed, and defaulting one would commit a tensor of the wrong extent"
                    % ", ".join(f"`{k}`" for k in missing))
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
                    elif str(elem) != dtype:
                        # float commit: the token must name the accumulator's float element
                        # type (f16/f32/f64/bf16) — no cast, so they must agree.
                        raise VerifyException(
                            "interface.commit output_dtype %s does not match result "
                            "element type %s" % (dtype, elem))

    @irdl_op_definition
    class VectorMapOp(IRDLOperation):
        """interface.vector_map — an elementwise combine of two equal-shape tensors
        (``combine`` = add/mul) or an identity copy of ``lhs`` (``combine`` = identity),
        followed by an optional activation (relu). This is the non-matmul workhorse of a whole
        model: residual adds, gating multiplies, and pointwise activations run on the target's
        vector/scalar lanes. Lowers to the runtime VECTOR_MAP command."""
        name = "interface.vector_map"
        lhs = operand_def()
        rhs = operand_def()
        combine = prop_def(StringAttr)
        activation = opt_prop_def(ArrayAttr)
        visibility = opt_prop_def(VisibilityAttr)
        out = result_def()

        def verify_(self) -> None:
            if self.combine.data not in KNOWN_COMBINE:
                raise VerifyException(
                    "interface.vector_map combine %r not in %s"
                    % (self.combine.data, sorted(KNOWN_COMBINE)))
            if self.activation is not None:
                for entry in self.activation:
                    act = entry.data if isinstance(entry, StringAttr) else None
                    if act not in KNOWN_ACTIVATION:
                        raise VerifyException(
                            "interface.vector_map activation %r not in %s"
                            % (act, sorted(KNOWN_ACTIVATION)))

    @irdl_op_definition
    class VectorReduceOp(IRDLOperation):
        """interface.vector_reduce — reduce a tensor to a scalar-per-row (sum). Runs on the
        target's vector/scalar lanes; lowers to the runtime VREDUCE command."""
        name = "interface.vector_reduce"
        src = operand_def()
        reduce = prop_def(StringAttr, prop_name="op")
        visibility = opt_prop_def(VisibilityAttr)
        out = result_def()

        def verify_(self) -> None:
            if self.reduce.data not in KNOWN_REDUCE:
                raise VerifyException(
                    "interface.vector_reduce op %r not in %s"
                    % (self.reduce.data, sorted(KNOWN_REDUCE)))

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

    _OP_CLASSES = [ResidentPackOp, ResidentEvictOp, MatmulOp, ElementwiseOp,
                   AccumulatorCreateOp, AccumulateOp, CommitOp, VectorMapOp, VectorReduceOp,
                   AsyncCopyOp, AwaitOp, FifoCreateOp, FifoPushOp, FifoPopOp, CommandRegionOp]
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
        pack = ResidentPackOp(operands=[w, None], result_types=[res_t], properties={
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
