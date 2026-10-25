"""Actual synchronous host row transport for the typed rolling-window IR.

Only input movement is implemented here. The original consumer and its proven
completion remain explicit bindings, including any device DMA lifetime.
"""

from xdsl.dialects import llvm
from xdsl.dialects.builtin import i8
from xdsl.ir import SSAValue

from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.compile.scheduling.rolling_window import RollingWindowPlan, emit_rolling_window
from merlin.llvmlower.host_builder import PTR, FnBuilder


def emit_rolling_row_copy(builder, source, cache, fields, *, volatile=False):
    """Bind one verified stage_row's actual byte addresses and padding predicate.

    This primitive alone supplies no source/address, ownership or completion
    proof. Call it only on fields projected from checked rolling-window IR.
    An explicit volatile binding preserves individual stores when the selected
    freestanding runtime cannot supply a compiler-generated memcpy/memset.
    Volatility supplies neither a target fence nor consumer completion.
    """
    if type(volatile) is not bool:
        raise ValueError("rolling row store volatility must be explicit")
    control = FnBuilderControlAdapter(builder)

    def copy():
        def byte(index):
            origin = builder.gep(source, control.op("add", fields["source_byte"], index), i8)
            target = builder.gep(cache, control.op("add", fields["cache_byte"], index), i8)
            value = builder.add(llvm.LoadOp(origin, i8)).results[0]
            builder.add(llvm.StoreOp(value, target, volatile=volatile))

        control.loop(0, fields["row_bytes"], 1, byte)

    def zero():
        def byte(index):
            target = builder.gep(cache, control.op("add", fields["cache_byte"], index), i8)
            builder.add(llvm.StoreOp(builder.const(0, i8), target, volatile=volatile))

        control.loop(0, fields["row_bytes"], 1, byte)

    control.branch(fields["valid"], copy, zero)


def emit_host_rolling_window(
    module, plan, builder, source, cache, *, consume, complete, source_cache_disjoint, maximum_cache_bytes
):
    """Copy/zero actual guarded rows, then invoke the original checked consumer.

    Cache byte addresses are relative to ``cache`` and include the assigned
    AddressWindow.start. Its caller owns capacity/nonaliasing and the lifetime
    of both pointers. ``consume(fields)`` returns a binding handle and
    ``complete(handle)`` must synchronously finish *all* uses of cached bytes.
    Neither callback is an assertion-only escape/completion qualification.
    """
    if type(plan) is not RollingWindowPlan or type(builder) is not FnBuilder:
        raise ValueError("rolling host transport requires its typed source plan and ordinary LLVM builder")
    plan.__post_init__()
    if any(not isinstance(value, SSAValue) or value.type != PTR for value in (source, cache)):
        raise ValueError("rolling host transport requires byte-addressable input/cache pointers")
    if (
        source_cache_disjoint is not True
        or type(maximum_cache_bytes) is not int
        or maximum_cache_bytes < plan.cache.start + plan.cache.size
    ):
        raise ValueError("rolling host transport requires disjoint input/cache and complete assigned byte capacity")
    if not callable(consume) or not callable(complete):
        raise ValueError("rolling host transport requires actual original-consumer and completion bindings")
    control = FnBuilderControlAdapter(builder)

    def lower(kind, fields):
        if kind == "begin_transport":
            return object()
        if kind == "stage_row":
            emit_rolling_row_copy(builder, source, cache, fields)
        elif kind == "await_transport":
            # The actual LLVM copies/zeros preceding this point are synchronous
            # host memory operations. Device consumer completion is separate.
            return fields["transfer"]
        elif kind == "consume":
            return consume(fields)
        elif kind == "complete":
            complete(fields["consumer"])
        else:
            raise ValueError("unknown rolling-window effect")

    receipt = emit_rolling_window(module, plan, control, lower)
    return {
        **receipt,
        "transport": "synchronous guarded host i8 copy/zero",
        "consumer_qualification": "separate original source/effect contract; not supplied by transport",
    }


def emit_cached_spatial_row_panel(
    plan,
    builder,
    cache,
    destination,
    *,
    producer_y,
    output_x,
    output_pixels,
    first_halo_column,
    rows,
    cache_destination_disjoint,
    maximum_panel_bytes,
):
    """Pack original kh/kw/channel taps from one resident producer row.

    The horizontal border mapping is precisely the existing clipped minimum-pad
    pool halo. It changes no conv padding, Q1, Q2, reduction order or observer.
    The cache must remain resident until the separate dense consumer completes.
    """
    from merlin.compile.scheduling.rolling_spatial import RollingPackedSpatialPlan
    from merlin.compile.scheduling.rolling_window import emit_rolling_window_lookup
    from merlin.llvmlower.convolution_packing import ClippedSpatialHaloRows

    if type(plan) is not RollingPackedSpatialPlan or type(builder) is not FnBuilder:
        raise ValueError("cached packing requires its original typed rolling spatial plan")
    plan.__post_init__()
    source = plan.packed.source
    ClippedSpatialHaloRows(source.spatial, output_pixels)
    w = source.producer.window
    if (
        cache_destination_disjoint is not True
        or type(rows) is not int
        or not 0 < rows <= plan.packed.panel_rows
        or type(maximum_panel_bytes) is not int
        or rows * plan.packed.reduction > maximum_panel_bytes
    ):
        raise ValueError("cached packing requires disjoint cache/panel and complete bounded panel capacity")
    if any(not isinstance(value, SSAValue) or value.type != PTR for value in (cache, destination)):
        raise ValueError("cached packing requires byte-addressable pointers")
    control = FnBuilderControlAdapter(builder)
    width = (output_pixels - 1) * source.spatial.stride[1] + source.spatial.window[1]

    def row_body(row):
        local = control.op("add", first_halo_column, row)
        x = control.op(
            "add",
            control.op("sub", control.op("mul", output_x, source.spatial.stride[1]), source.spatial.padding[1]),
            local,
        )
        x = control.select(control.cmp("slt", x, 0), 0, x)
        x = control.select(control.cmp("slt", x, w.output_spatial[1]), x, w.output_spatial[1] - 1)
        valid = control.op(
            "and",
            control.cmp("ult", local, width),
            control.cmp("ule", output_x, source.spatial.output_hw[1] - output_pixels),
            "i1",
        )

        def ky_body(ky):
            def kx_body(kx):
                def byte(channel):
                    lookup = emit_rolling_window_lookup(plan.rolling, control, producer_y, x, ky, kx, channel)
                    offset = control.op(
                        "add",
                        control.op("mul", row, plan.packed.reduction),
                        control.op(
                            "add",
                            control.op(
                                "mul", control.op("add", control.op("mul", ky, w.kernel[1]), kx), w.input_shape[-1]
                            ),
                            channel,
                        ),
                    )
                    target = builder.gep(destination, offset, i8)

                    def copy():
                        origin = builder.gep(cache, control.value(lookup["cache_byte"]), i8)
                        value = builder.add(llvm.LoadOp(origin, i8)).results[0]
                        builder.add(llvm.StoreOp(value, target))

                    control.branch(
                        control.op("and", valid, lookup["predicate"], "i1"),
                        copy,
                        lambda: builder.add(llvm.StoreOp(builder.const(0, i8), target)),
                    )

                control.loop(0, w.input_shape[-1], 1, byte)

            control.loop(0, w.kernel[1], 1, kx_body)

        control.loop(0, w.kernel[0], 1, ky_body)

    control.loop(0, rows, 1, row_body)
    return dict(
        panel_shape=[rows, plan.packed.reduction],
        source_reduction_order=["kernel_y", "kernel_x", "input_channel"],
        source="original resident rolling row cache",
        target_qualification=False,
    )
