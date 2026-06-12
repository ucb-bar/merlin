"""The ``runtime`` dialect (xDSL): target-independent execution.

Merlin owns this dialect; targets implement adapters. Devices, buffers, command
buffers, queues, events, persistent handles, profiling, metrics, traces. Never target
instruction names, packet bitfields, or MMIO registers.

``runtime.command_buffer.append`` carries the abstract command inline (opcode +
operand-name map + attribute map) so the terminal lowering stage can emit a
command-buffer dict for the Python engine (``merlin.runtime`` package) directly.
See docs/core_dialects.md and docs/runtime.md.
"""
from __future__ import annotations

from ._common import HAS_XDSL

DIALECT_NAME = "runtime"
OPS = ["device.get", "buffer.alloc", "buffer_view.create", "handle.create",
       "handle.destroy", "command_buffer.create", "command_buffer.append", "submit",
       "wait", "profile.region", "metrics.read", "trace.emit"]
TYPES = ["device", "backend", "buffer", "buffer_view", "command_buffer", "command",
         "event", "handle", "metrics", "trace"]

# Common metric vocabulary (superset of the engine's COMMON_METRIC_NAMES); targets add
# extras under a `target_specific.` prefix.
KNOWN_METRICS = {
    "cycles", "host_cycles", "device_cycles", "submit_cycles", "wait_cycles",
    "queue_wait_cycles", "dispatch_count", "command_count", "bytes_moved", "bytes_read",
    "bytes_written", "pack_count", "resident_hits", "resident_misses", "evictions",
    "accumulator_commits", "intermediate_write_bytes", "error_count",
}

if HAS_XDSL:
    from xdsl.ir import (Dialect, EnumAttribute, ParametrizedAttribute,
                         SpacedOpaqueSyntaxAttribute, TypeAttribute)
    from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                           operand_def, opt_prop_def, prop_def, region_def, result_def,
                           traits_def)
    from xdsl.traits import NoTerminator
    from xdsl.utils.exceptions import VerifyException
    from xdsl.utils.str_enum import StrEnum
    from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, IntegerAttr,
                                       StringAttr)

    # -- enums ---------------------------------------------------------------

    class Backend(StrEnum):
        SIMULATOR = "simulator"
        HOST = "host"
        BAREMETAL = "baremetal"
        ZEPHYR = "zephyr"
        FIRESIM = "firesim"
        EXTERNAL = "external"

    class QueueKind(StrEnum):
        COMPUTE = "compute"
        DMA = "dma"
        COPY = "copy"
        HOST = "host"
        CONTROL = "control"
        UNIFIED = "unified"

    class SubmitMode(StrEnum):
        BLOCKING = "blocking"
        ASYNC = "async"
        DEFERRED = "deferred"
        BATCHED = "batched"
        PERSISTENT = "persistent"

    class EventKind(StrEnum):
        COMPLETION = "completion"
        DEPENDENCY = "dependency"
        FENCE = "fence"
        PROFILE_MARKER = "profile_marker"
        ERROR = "error"

    @irdl_attr_definition
    class BackendAttr(EnumAttribute[Backend], SpacedOpaqueSyntaxAttribute):
        """#runtime.backend — which runtime backend executes the work."""
        name = "runtime.backend"

    @irdl_attr_definition
    class QueueKindAttr(EnumAttribute[QueueKind], SpacedOpaqueSyntaxAttribute):
        """#runtime.queue_kind — the queue a command is appended to."""
        name = "runtime.queue_kind"

    @irdl_attr_definition
    class SubmitModeAttr(EnumAttribute[SubmitMode], SpacedOpaqueSyntaxAttribute):
        """#runtime.submit_mode — how a command buffer is submitted."""
        name = "runtime.submit_mode"

    @irdl_attr_definition
    class EventKindAttr(EnumAttribute[EventKind], SpacedOpaqueSyntaxAttribute):
        """#runtime.event_kind — what an event represents."""
        name = "runtime.event_kind"

    # -- types ---------------------------------------------------------------

    @irdl_attr_definition
    class DeviceType(ParametrizedAttribute, TypeAttribute):
        """!runtime.device — a logical device (toy_npu0, saturn_cpu0, ...)."""
        name = "runtime.device"

    @irdl_attr_definition
    class BackendType(ParametrizedAttribute, TypeAttribute):
        """!runtime.backend — a runtime backend value."""
        name = "runtime.backend_t"

    @irdl_attr_definition
    class BufferType(ParametrizedAttribute, TypeAttribute):
        """!runtime.buffer — raw memory buffer."""
        name = "runtime.buffer"

    @irdl_attr_definition
    class BufferViewType(ParametrizedAttribute, TypeAttribute):
        """!runtime.buffer_view — typed shape/layout view over a buffer."""
        name = "runtime.buffer_view"

    @irdl_attr_definition
    class CommandBufferType(ParametrizedAttribute, TypeAttribute):
        """!runtime.command_buffer — ordered command collection."""
        name = "runtime.command_buffer"

    @irdl_attr_definition
    class CommandType(ParametrizedAttribute, TypeAttribute):
        """!runtime.command — opaque command before target encoding."""
        name = "runtime.command"

    @irdl_attr_definition
    class EventType(ParametrizedAttribute, TypeAttribute):
        """!runtime.event — runtime synchronization event."""
        name = "runtime.event"

    @irdl_attr_definition
    class HandleType(ParametrizedAttribute, TypeAttribute):
        """!runtime.handle — persistent runtime object handle."""
        name = "runtime.handle"

    @irdl_attr_definition
    class MetricsType(ParametrizedAttribute, TypeAttribute):
        """!runtime.metrics — metrics object."""
        name = "runtime.metrics"

    @irdl_attr_definition
    class TraceType(ParametrizedAttribute, TypeAttribute):
        """!runtime.trace — trace event stream object."""
        name = "runtime.trace"

    # -- ops -----------------------------------------------------------------

    @irdl_op_definition
    class DeviceGetOp(IRDLOperation):
        """runtime.device.get — gets a logical device on a backend.

        Device symbol/backend support is validated against the runtime adapter plan
        by the cross-op analyses, not locally.
        """
        name = "runtime.device.get"
        device = prop_def(StringAttr)
        backend = prop_def(BackendAttr)
        dev = result_def(DeviceType)

        def verify_(self) -> None:
            if not self.device.data:
                raise VerifyException("runtime.device.get needs a device symbol")

    @irdl_op_definition
    class BufferAllocOp(IRDLOperation):
        """runtime.buffer.alloc — allocates a runtime buffer."""
        name = "runtime.buffer.alloc"
        dev = operand_def(DeviceType)
        bytes_ = prop_def(IntegerAttr, prop_name="bytes")
        memory = opt_prop_def(StringAttr)
        buffer = result_def(BufferType)

        def verify_(self) -> None:
            if self.bytes_.value.data <= 0:
                raise VerifyException("runtime.buffer.alloc bytes must be positive")

    @irdl_op_definition
    class BufferViewCreateOp(IRDLOperation):
        """runtime.buffer_view.create — typed view over a buffer."""
        name = "runtime.buffer_view.create"
        buffer = operand_def(BufferType)
        shape = prop_def(ArrayAttr)
        dtype = prop_def(StringAttr)
        layout = opt_prop_def(StringAttr)
        view = result_def(BufferViewType)

        def verify_(self) -> None:
            for d in self.shape:
                if not isinstance(d, IntegerAttr) or d.value.data <= 0:
                    raise VerifyException(
                        "runtime.buffer_view.create shape dims must be positive ints")

    @irdl_op_definition
    class HandleCreateOp(IRDLOperation):
        """runtime.handle.create — creates a persistent handle on a device."""
        name = "runtime.handle.create"
        dev = operand_def(DeviceType)
        kind = prop_def(StringAttr)
        lifetime = opt_prop_def(StringAttr)
        handle = result_def(HandleType)

        def verify_(self) -> None:
            if not self.kind.data:
                raise VerifyException("runtime.handle.create needs a handle kind")

    @irdl_op_definition
    class HandleDestroyOp(IRDLOperation):
        """runtime.handle.destroy — destroys a persistent handle."""
        name = "runtime.handle.destroy"
        handle = operand_def(HandleType)

    @irdl_op_definition
    class CommandBufferCreateOp(IRDLOperation):
        """runtime.command_buffer.create — creates a command buffer on a device.

        ``tensors`` is the command buffer's resource table (leaf tensor name ->
        "shape:dtype"), carried here so the emit stage is a pure function of the module.
        """
        name = "runtime.command_buffer.create"
        dev = operand_def(DeviceType)
        target = prop_def(StringAttr)
        mode = opt_prop_def(SubmitModeAttr)
        tensors = opt_prop_def(DictionaryAttr)
        cb = result_def(CommandBufferType)

        def verify_(self) -> None:
            if not self.target.data:
                raise VerifyException(
                    "runtime.command_buffer.create needs the target name")

    @irdl_op_definition
    class CommandBufferAppendOp(IRDLOperation):
        """runtime.command_buffer.append — appends an abstract command.

        The command is carried inline: ``opcode`` plus an operand-name map (``args``)
        and an attribute map (``attrs``). Queue support / device-target matching is a
        cross-op analysis (check_command_buffer_consistency).
        """
        name = "runtime.command_buffer.append"
        cb = operand_def(CommandBufferType)
        opcode = prop_def(StringAttr)
        args = prop_def(DictionaryAttr)
        attrs = opt_prop_def(DictionaryAttr)
        queue = opt_prop_def(QueueKindAttr)

        def verify_(self) -> None:
            if not self.opcode.data:
                raise VerifyException("runtime.command_buffer.append needs an opcode")
            for key, val in self.args.data.items():
                if not isinstance(val, StringAttr) or not val.data:
                    raise VerifyException(
                        "runtime.command_buffer.append arg %r must name a tensor" % key)

    @irdl_op_definition
    class SubmitOp(IRDLOperation):
        """runtime.submit — submits a command buffer; yields a completion event."""
        name = "runtime.submit"
        dev = operand_def(DeviceType)
        cb = operand_def(CommandBufferType)
        mode = opt_prop_def(SubmitModeAttr)
        event = result_def(EventType)

    @irdl_op_definition
    class WaitOp(IRDLOperation):
        """runtime.wait — waits on an event."""
        name = "runtime.wait"
        event = operand_def(EventType)

    @irdl_op_definition
    class ProfileRegionOp(IRDLOperation):
        """runtime.profile.region — marks a profiled runtime region."""
        name = "runtime.profile.region"
        label = prop_def(StringAttr)
        body = region_def()
        traits = traits_def(NoTerminator())

        def verify_(self) -> None:
            if not self.label.data:
                raise VerifyException("runtime.profile.region needs a label")

    @irdl_op_definition
    class MetricsReadOp(IRDLOperation):
        """runtime.metrics.read — reads named metrics from a device."""
        name = "runtime.metrics.read"
        dev = operand_def(DeviceType)
        metrics = prop_def(ArrayAttr)
        out = result_def(MetricsType)

        def verify_(self) -> None:
            for entry in self.metrics:
                m = entry.data if isinstance(entry, StringAttr) else None
                if m is None or (m not in KNOWN_METRICS
                                 and not m.startswith("target_specific.")):
                    raise VerifyException(
                        "runtime.metrics.read metric %r unknown (known: %s, or "
                        "target_specific.*)" % (m, sorted(KNOWN_METRICS)))

    @irdl_op_definition
    class TraceEmitOp(IRDLOperation):
        """runtime.trace.emit — emits a trace marker."""
        name = "runtime.trace.emit"
        event = prop_def(StringAttr)
        attrs = opt_prop_def(DictionaryAttr)

        def verify_(self) -> None:
            if not self.event.data:
                raise VerifyException("runtime.trace.emit needs an event name")

    _OP_CLASSES = [DeviceGetOp, BufferAllocOp, BufferViewCreateOp, HandleCreateOp,
                   HandleDestroyOp, CommandBufferCreateOp, CommandBufferAppendOp,
                   SubmitOp, WaitOp, ProfileRegionOp, MetricsReadOp, TraceEmitOp]
    _ATTR_CLASSES = [DeviceType, BackendType, BufferType, BufferViewType,
                     CommandBufferType, CommandType, EventType, HandleType, MetricsType,
                     TraceType, BackendAttr, QueueKindAttr, SubmitModeAttr,
                     EventKindAttr]
    RUNTIME_DIALECT = Dialect(DIALECT_NAME, _OP_CLASSES, _ATTR_CLASSES)

    def get_dialect() -> Dialect:
        return RUNTIME_DIALECT

    def build_example():
        """A small, verifiable module exercising every runtime op."""
        from xdsl.ir import Block, Region
        from xdsl.dialects.builtin import FunctionType, ModuleOp
        from xdsl.dialects.func import FuncOp, ReturnOp

        blk = Block()
        dev = DeviceGetOp(result_types=[DeviceType()], properties={
            "device": StringAttr("toy_npu0"),
            "backend": BackendAttr(Backend.SIMULATOR)})
        buf = BufferAllocOp(operands=[dev.dev], result_types=[BufferType()], properties={
            "bytes": IntegerAttr(4096, 64), "memory": StringAttr("device")})
        view = BufferViewCreateOp(operands=[buf.buffer], result_types=[BufferViewType()],
                                  properties={
            "shape": ArrayAttr([IntegerAttr(64, 64), IntegerAttr(64, 64)]),
            "dtype": StringAttr("i8"), "layout": StringAttr("row_major")})
        h = HandleCreateOp(operands=[dev.dev], result_types=[HandleType()], properties={
            "kind": StringAttr("resident_tensor"), "lifetime": StringAttr("region")})
        cb = CommandBufferCreateOp(operands=[dev.dev],
                                   result_types=[CommandBufferType()], properties={
            "target": StringAttr("toy_npu"),
            "mode": SubmitModeAttr(SubmitMode.BATCHED)})
        ap = CommandBufferAppendOp(operands=[cb.cb], properties={
            "opcode": StringAttr("RES_PACK"),
            "args": DictionaryAttr({"src": StringAttr("W"), "dst": StringAttr("W_res")}),
            "attrs": DictionaryAttr({"layout": StringAttr("packed_rhs")}),
            "queue": QueueKindAttr(QueueKind.COMPUTE)})
        sub = SubmitOp(operands=[dev.dev, cb.cb], result_types=[EventType()],
                       properties={"mode": SubmitModeAttr(SubmitMode.BLOCKING)})
        wait = WaitOp(operands=[sub.event])
        prof = ProfileRegionOp(regions=[Region([Block()])],
                               properties={"label": StringAttr("resident_matmul")})
        met = MetricsReadOp(operands=[dev.dev], result_types=[MetricsType()],
                            properties={"metrics": ArrayAttr(
                                [StringAttr("cycles"), StringAttr("bytes_moved"),
                                 StringAttr("resident_hits")])})
        tr = TraceEmitOp(properties={
            "event": StringAttr("resident_hit"),
            "attrs": DictionaryAttr({"object": StringAttr("W_res")})})
        hd = HandleDestroyOp(operands=[h.handle])
        ret = ReturnOp()
        blk.add_ops([dev, buf, view, h, cb, ap, sub, wait, prof, met, tr, hd, ret])
        fn = FuncOp("launch", FunctionType.from_lists([], []), Region([blk]))
        return ModuleOp([fn])

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None

    def build_example():
        return None
