"""The ``schedule`` dialect (xDSL): chosen compiler decisions.

``schedule`` says which decision was selected: layout preservation, pack hoisting,
memory placement, accumulator lifetime, dispatch grouping, vector strategy, interface
selection. It is not a runtime dialect; it never submits work or encodes target commands.
See docs/core_dialects.md and docs/dialects.md.
"""
from __future__ import annotations

from ._common import HAS_XDSL, Visibility

DIALECT_NAME = "schedule"
OPS = ["plan", "bind", "apply_policy", "preserve_layout", "hoist_pack", "place",
       "keep_accumulator_live", "group_dispatch", "vector_strategy", "select_interface"]
TYPES = ["handle", "plan", "policy"]

# Interface abstractions `schedule.select_interface` may materialize (verifier-checked).
KNOWN_INTERFACES = {
    "resident_packed_tensor", "accumulator_commit", "event_token_queue",
    "producer_consumer_fifo", "async_copy", "command_region",
}

if HAS_XDSL:
    from xdsl.ir import (Dialect, EnumAttribute, ParametrizedAttribute,
                         SpacedOpaqueSyntaxAttribute, TypeAttribute)
    from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                           operand_def, opt_operand_def, opt_prop_def, prop_def,
                           region_def, result_def, traits_def, var_operand_def)
    from xdsl.traits import NoTerminator
    from xdsl.utils.exceptions import VerifyException
    from xdsl.utils.str_enum import StrEnum
    from xdsl.dialects.builtin import StringAttr

    from .contract import CapabilityType

    # -- enums ---------------------------------------------------------------

    class MemoryState(StrEnum):
        NORMAL = "normal"
        STREAMING = "streaming"
        PACKED = "packed"
        RESIDENT = "resident"
        ACCUMULATOR = "accumulator"
        COMMITTED = "committed"
        FIFO = "fifo"

    class DispatchGranularity(StrEnum):
        OP = "op"
        TILE = "tile"
        FUSED_REGION = "fused_region"
        COMMAND_BUFFER = "command_buffer"
        PERSISTENT_REGION = "persistent_region"

    class VectorStrategy(StrEnum):
        FIXED_WIDTH = "fixed_width"
        SCALABLE_VL = "scalable_vl"
        PREDICATED_TAIL = "predicated_tail"
        MASKED_TAIL = "masked_tail"
        SCALAR_CLEANUP = "scalar_cleanup"

    class PolicySource(StrEnum):
        MANUAL = "manual"
        KERNEL_MINING = "kernel_mining"
        GRID_SEARCH = "grid_search"
        EVOLUTIONARY_SEARCH = "evolutionary_search"
        MAP_ELITES = "map_elites"
        ORACLE = "oracle"

    @irdl_attr_definition
    class MemoryStateAttr(EnumAttribute[MemoryState], SpacedOpaqueSyntaxAttribute):
        """#schedule.memory_state — the memory state assigned to an object."""
        name = "schedule.memory_state"

    @irdl_attr_definition
    class DispatchGranularityAttr(EnumAttribute[DispatchGranularity],
                                  SpacedOpaqueSyntaxAttribute):
        """#schedule.dispatch_granularity — how ops are grouped into dispatches."""
        name = "schedule.dispatch_granularity"

    @irdl_attr_definition
    class VectorStrategyAttr(EnumAttribute[VectorStrategy], SpacedOpaqueSyntaxAttribute):
        """#schedule.vector_strategy — vector lowering strategy."""
        name = "schedule.vector_strategy"

    @irdl_attr_definition
    class PolicySourceAttr(EnumAttribute[PolicySource], SpacedOpaqueSyntaxAttribute):
        """#schedule.policy_source — provenance of a schedule decision."""
        name = "schedule.policy_source"

    @irdl_attr_definition
    class VisibilityAttr(EnumAttribute[Visibility], SpacedOpaqueSyntaxAttribute):
        """#schedule.visibility — DSE variant requested for the selected interface."""
        name = "schedule.visibility"

    # -- types ---------------------------------------------------------------

    @irdl_attr_definition
    class HandleType(ParametrizedAttribute, TypeAttribute):
        """!schedule.handle — a handle to payload IR or a logical workload object."""
        name = "schedule.handle"

    @irdl_attr_definition
    class PlanType(ParametrizedAttribute, TypeAttribute):
        """!schedule.plan — a schedule plan value."""
        name = "schedule.plan"

    @irdl_attr_definition
    class PolicyType(ParametrizedAttribute, TypeAttribute):
        """!schedule.policy — a reference to a policy rule."""
        name = "schedule.policy"

    # -- ops -----------------------------------------------------------------

    @irdl_op_definition
    class PlanOp(IRDLOperation):
        """schedule.plan — a named schedule plan holding decision ops."""
        name = "schedule.plan"
        sym_name = prop_def(StringAttr)
        source = opt_prop_def(PolicySourceAttr)
        body = region_def()
        traits = traits_def(NoTerminator())

        def verify_(self) -> None:
            if not self.sym_name.data:
                raise VerifyException("schedule.plan must be named")

    @irdl_op_definition
    class BindOp(IRDLOperation):
        """schedule.bind — binds a schedule handle to payload IR (symbolic ref for MVP)."""
        name = "schedule.bind"
        target = prop_def(StringAttr)
        handle = result_def(HandleType)

        def verify_(self) -> None:
            if not self.target.data:
                raise VerifyException("schedule.bind target reference must be non-empty")

    @irdl_op_definition
    class ApplyPolicyOp(IRDLOperation):
        """schedule.apply_policy — applies a named policy to bound handles."""
        name = "schedule.apply_policy"
        handles = var_operand_def(HandleType)
        policy = prop_def(StringAttr)

        def verify_(self) -> None:
            if not self.policy.data:
                raise VerifyException("schedule.apply_policy policy name must be non-empty")

    @irdl_op_definition
    class PreserveLayoutOp(IRDLOperation):
        """schedule.preserve_layout — declares layout preservation over a scope."""
        name = "schedule.preserve_layout"
        value = operand_def()
        layout = prop_def(StringAttr)
        scope = prop_def(StringAttr)

        def verify_(self) -> None:
            if not self.layout.data:
                raise VerifyException("schedule.preserve_layout layout must be non-empty")

    @irdl_op_definition
    class HoistPackOp(IRDLOperation):
        """schedule.hoist_pack — hoist a pack of an immutable operand out of a loop."""
        name = "schedule.hoist_pack"
        value = operand_def()
        outside = prop_def(StringAttr)
        layout = prop_def(StringAttr)

    @irdl_op_definition
    class PlaceOp(IRDLOperation):
        """schedule.place — assigns a logical object to a memory state.

        The optional ``capability`` operand names the contract capability that must
        allow the placement; that legality is a cross-op analysis (lowering/analyses.py).
        """
        name = "schedule.place"
        value = operand_def()
        capability = opt_operand_def(CapabilityType)
        state = prop_def(MemoryStateAttr)
        lifetime = opt_prop_def(StringAttr)

    @irdl_op_definition
    class KeepAccumulatorLiveOp(IRDLOperation):
        """schedule.keep_accumulator_live — no materialization before commit."""
        name = "schedule.keep_accumulator_live"
        value = operand_def()
        until = prop_def(StringAttr)

        def verify_(self) -> None:
            if not self.until.data:
                raise VerifyException(
                    "schedule.keep_accumulator_live requires an `until` marker")

    @irdl_op_definition
    class GroupDispatchOp(IRDLOperation):
        """schedule.group_dispatch — groups ops into one dispatch/command-buffer region."""
        name = "schedule.group_dispatch"
        items = var_operand_def()
        granularity = prop_def(DispatchGranularityAttr)

        def verify_(self) -> None:
            if not self.items:
                raise VerifyException("schedule.group_dispatch needs at least one item")

    @irdl_op_definition
    class VectorStrategyOp(IRDLOperation):
        """schedule.vector_strategy — selects vector lowering + tail strategy."""
        name = "schedule.vector_strategy"
        value = operand_def()
        strategy = prop_def(VectorStrategyAttr)
        tail = opt_prop_def(VectorStrategyAttr)

        def verify_(self) -> None:
            if self.tail is not None and self.tail.data in (
                    VectorStrategy.FIXED_WIDTH, VectorStrategy.SCALABLE_VL):
                raise VerifyException(
                    "schedule.vector_strategy tail must be a tail policy "
                    "(predicated_tail/masked_tail/scalar_cleanup)")

    @irdl_op_definition
    class SelectInterfaceOp(IRDLOperation):
        """schedule.select_interface — selects an interface abstraction to materialize."""
        name = "schedule.select_interface"
        value = operand_def()
        interface = prop_def(StringAttr)
        reason = opt_prop_def(StringAttr)
        visibility = opt_prop_def(VisibilityAttr)

        def verify_(self) -> None:
            if self.interface.data not in KNOWN_INTERFACES:
                raise VerifyException(
                    "schedule.select_interface %r not a known interface abstraction "
                    "(known: %s)" % (self.interface.data, sorted(KNOWN_INTERFACES)))

    _OP_CLASSES = [PlanOp, BindOp, ApplyPolicyOp, PreserveLayoutOp, HoistPackOp, PlaceOp,
                   KeepAccumulatorLiveOp, GroupDispatchOp, VectorStrategyOp,
                   SelectInterfaceOp]
    _ATTR_CLASSES = [HandleType, PlanType, PolicyType, MemoryStateAttr,
                     DispatchGranularityAttr, VectorStrategyAttr, PolicySourceAttr,
                     VisibilityAttr]
    SCHEDULE_DIALECT = Dialect(DIALECT_NAME, _OP_CLASSES, _ATTR_CLASSES)

    def get_dialect() -> Dialect:
        return SCHEDULE_DIALECT

    def build_example():
        """A small, verifiable module exercising every schedule op."""
        from xdsl.ir import Block, Region
        from xdsl.dialects.builtin import (ArrayAttr, FunctionType, ModuleOp, TensorType,
                                           i8)
        from xdsl.dialects.func import FuncOp, ReturnOp

        from .contract import CapabilityOp

        Wt = TensorType(i8, [128, 64])
        blk = Block(arg_types=[Wt])
        (w,) = blk.args
        cap = CapabilityOp(
            result_types=[CapabilityType(StringAttr("toy_npu"))],
            properties={"sym_name": StringAttr("toy_npu"),
                        "features": ArrayAttr([StringAttr("resident_packed_tensor")])})
        bind = BindOp(result_types=[HandleType()],
                      properties={"target": StringAttr("@payload::@main::%matmul0")})
        apply_p = ApplyPolicyOp(operands=[[bind.handle]],
                                properties={"policy": StringAttr("resident_packed_rhs")})
        pres = PreserveLayoutOp(operands=[w], properties={
            "layout": StringAttr("packed_rhs"), "scope": StringAttr("region")})
        hoist = HoistPackOp(operands=[w], properties={
            "outside": StringAttr("@loop_i"), "layout": StringAttr("packed_rhs")})
        place = PlaceOp(operands=[w, [cap.cap]], properties={
            "state": MemoryStateAttr(MemoryState.RESIDENT),
            "lifetime": StringAttr("region")})
        keep = KeepAccumulatorLiveOp(operands=[w], properties={
            "until": StringAttr("epilogue_commit")})
        group = GroupDispatchOp(operands=[[w, w]], properties={
            "granularity": DispatchGranularityAttr(DispatchGranularity.COMMAND_BUFFER)})
        vec = VectorStrategyOp(operands=[w], properties={
            "strategy": VectorStrategyAttr(VectorStrategy.SCALABLE_VL),
            "tail": VectorStrategyAttr(VectorStrategy.PREDICATED_TAIL)})
        sel = SelectInterfaceOp(operands=[w], properties={
            "interface": StringAttr("resident_packed_tensor"),
            "reason": StringAttr("reuse_count >= threshold and capacity fits"),
            "visibility": VisibilityAttr(Visibility.SOFTWARE_VISIBLE)})
        inner = Block()
        plan = PlanOp(regions=[Region([inner])], properties={
            "sym_name": StringAttr("resident_rhs_plan"),
            "source": PolicySourceAttr(PolicySource.KERNEL_MINING)})
        ret = ReturnOp()
        blk.add_ops([cap, bind, apply_p, pres, hoist, place, keep, group, vec, sel,
                     plan, ret])
        fn = FuncOp("decisions", FunctionType.from_lists([Wt], []), Region([blk]))
        return ModuleOp([fn])

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None

    def build_example():
        return None
