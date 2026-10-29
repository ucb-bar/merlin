"""Execute actual LLVM circular transport and checked lookup on the host CPU."""

import ctypes
import io
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, i8
from xdsl.printer import Printer

from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.rolling_window import (
    emit_rolling_window_lookup,
    materialize_rolling_window,
    propose_rolling_window,
)
from merlin.compile.scheduling.window_transfer import Window2D
from merlin.llvmlower.host_builder import PTR, FnBuilder
from merlin.llvmlower.rolling_window import emit_cached_spatial_row_panel, emit_host_rolling_window


def test_native_counted_initial_frontier_and_modulo_lookup_match_independent_original_windows(tmp_path):
    tools = Path("/work/merlin/toolchains/llvm-a47bddcc/install/bin")
    if not all((tools / name).is_file() for name in ("mlir-translate", "clang")):
        pytest.skip("native LLVM tools unavailable")
    window = Window2D((2, 8, 9, 2), (3, 2), stride=(2, 1), dilation=(2, 2), padding=(2, 1, 1, 2))
    plan = propose_rolling_window(
        window,
        AddressWindow("host_owned", 16, 512),
        source_identity="d" * 64,
        output_x_first=1,
        output_x_count=4,
        slots=7,
        alignment=8,
    )
    builder = FnBuilder([PTR, PTR, PTR])
    source, cache, output = builder.entry.args
    control = FnBuilderControlAdapter(builder)

    def consume(fields):
        def x_body(x):
            def ky_body(ky):
                def kx_body(kx):
                    def channel_body(channel):
                        lookup = emit_rolling_window_lookup(plan, control, fields["output_y"], x, ky, kx, channel)
                        row = control.op(
                            "add",
                            control.op("mul", fields["image"], plan.output_y_count),
                            control.op("sub", fields["output_y"], plan.output_y_first),
                        )
                        offset = control.op(
                            "add",
                            control.op("mul", row, plan.output_x_count),
                            control.op("sub", x, plan.output_x_first),
                        )
                        for index, extent in ((ky, 3), (kx, 2), (channel, 2)):
                            offset = control.op("add", control.op("mul", offset, extent), index)
                        target = builder.gep(output, offset, i8)

                        def copy():
                            origin = builder.gep(cache, control.value(lookup["cache_byte"]), i8)
                            value = builder.add(llvm.LoadOp(origin, i8)).results[0]
                            builder.add(llvm.StoreOp(value, target))

                        control.branch(
                            lookup["predicate"], copy, lambda: builder.add(llvm.StoreOp(builder.const(0, i8), target))
                        )

                    control.loop(0, 2, 1, channel_body)

                control.loop(0, 2, 1, kx_body)

            control.loop(0, 3, 1, ky_body)

        control.loop(plan.output_x_first, plan.output_x_first + plan.output_x_count, 1, x_body)
        # This CPU-only consumer contains only synchronous loads/stores.
        return object()

    receipt = emit_host_rolling_window(
        materialize_rolling_window(plan),
        plan,
        builder,
        source,
        cache,
        consume=consume,
        complete=lambda _: None,
        source_cache_disjoint=True,
        maximum_cache_bytes=528,
    )
    module = ModuleOp(
        [
            llvm.FuncOp(
                "scan",
                llvm.LLVMFunctionType([PTR, PTR, PTR], llvm.LLVMVoidType()),
                linkage=llvm.LinkageAttr("external"),
                body=builder.finish(),
            )
        ]
    )
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    mlir, ll, library = (tmp_path / name for name in ("rolling.mlir", "rolling.ll", "rolling.so"))
    mlir.write_text(stream.getvalue())
    subprocess.run(
        [str(tools / "mlir-translate"), "--mlir-to-llvmir", str(mlir), "-o", str(ll)], check=True, capture_output=True
    )
    subprocess.run(
        [str(tools / "clang"), "-O1", "-shared", "-fPIC", str(ll), "-o", str(library)], check=True, capture_output=True
    )
    fn = ctypes.CDLL(str(library)).scan
    fn.argtypes = [ctypes.c_void_p] * 3
    data = bytes((i * 17 + 19) % 251 for i in range(math.prod(window.input_shape)))
    expected = []
    for b in range(2):
        for y in range(plan.output_y_count):
            for x in range(1, 5):
                for ky in range(3):
                    for kx in range(2):
                        for c in range(2):
                            iy, ix = y * 2 - 2 + ky * 2, x - 1 + kx * 2
                            expected.append(
                                data[((b * 8 + iy) * 9 + ix) * 2 + c] if (0 <= iy < 8 and 0 <= ix < 9) else 0
                            )
    inputs = ctypes.create_string_buffer(data)
    storage = (ctypes.c_ubyte * 544)(*([239] * 544))
    outputs = (ctypes.c_ubyte * (len(expected) + 16))(*([241] * (len(expected) + 16)))
    fn(inputs, ctypes.byref(storage, 8), ctypes.byref(outputs, 8))
    assert list(outputs)[8:-8] == expected
    assert list(outputs)[:8] == list(outputs)[-8:] == [241] * 8
    assert list(storage)[:24] == [239] * 24
    assert list(storage)[8 + 16 + plan.required_bytes :] == [239] * (544 - 24 - plan.required_bytes)
    assert receipt["dynamic_stage_rows"] < receipt["unstaged_reference_rows"]
    assert receipt["target_qualification"] is False


@pytest.mark.parametrize("factor_channel_guards", [False, True])
def test_native_cached_dense_panels_keep_original_k_order_and_pool_border_duplication(tmp_path, factor_channel_guards):
    from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
    from merlin.compile.scheduling.rolling_spatial import RollingPackedSpatialPlan
    from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan
    from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
    from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan

    tools = Path("/work/merlin/toolchains/llvm-a47bddcc/install/bin")
    if not all((tools / name).is_file() for name in ("mlir-translate", "clang")):
        pytest.skip("native LLVM tools unavailable")
    window = Window2D((2, 6, 7, 3), (3, 3), padding=(1, 1, 1, 1))
    producer = CountedConvolutionPlan(window, 5, 4, 4, 4, 3, 32, 64, 0, 8, 3, True)
    chain = SpatialReadoutChain(RoundedReadout((0.5,), True), RoundedReadout((0.75,)), None)
    source_plan = SpatialProducerPlan(
        producer,
        SpatialMaxReadout((6, 7), (3, 3), (2, 2), (1, 1, 1, 1), -128),
        (chain, chain),
        3,
        readout_mode="staged_readouts",
        intermediate=AddressWindow("old", 0, 128),
    )
    packed = PackedSpatialProducerPlan(source_plan, 4, AddressWindow("owned", 0, 8192), 8)
    required = RollingPackedSpatialPlan(packed).required_bytes
    plan = RollingPackedSpatialPlan(replace(packed, workspace=replace(packed.workspace, size=required)))
    builder = FnBuilder([PTR, PTR, PTR])
    source, cache, output = builder.entry.args
    control = FnBuilderControlAdapter(builder)

    def consume(fields):
        row = control.op("add", control.op("mul", fields["image"], 6), fields["output_y"])
        destination = builder.gep(output, control.op("mul", row, 4 * 27), i8)
        emit_cached_spatial_row_panel(
            plan,
            builder,
            cache,
            destination,
            producer_y=fields["output_y"],
            output_x=0,
            output_pixels=3,
            first_halo_column=0,
            rows=4,
            cache_destination_disjoint=True,
            maximum_panel_bytes=4 * 27,
            factor_channel_guards=factor_channel_guards,
        )
        return object()

    emit_host_rolling_window(
        materialize_rolling_window(plan.rolling),
        plan.rolling,
        builder,
        source,
        cache,
        consume=consume,
        complete=lambda _: None,
        source_cache_disjoint=True,
        maximum_cache_bytes=required,
    )
    module = ModuleOp(
        [
            llvm.FuncOp(
                "cached_panels",
                llvm.LLVMFunctionType([PTR, PTR, PTR], llvm.LLVMVoidType()),
                linkage=llvm.LinkageAttr("external"),
                body=builder.finish(),
            )
        ]
    )
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    mlir, ll, library = (tmp_path / name for name in ("cached.mlir", "cached.ll", "cached.so"))
    mlir.write_text(stream.getvalue())
    subprocess.run(
        [str(tools / "mlir-translate"), "--mlir-to-llvmir", str(mlir), "-o", str(ll)], check=True, capture_output=True
    )
    subprocess.run(
        [str(tools / "clang"), "-O1", "-shared", "-fPIC", str(ll), "-o", str(library)], check=True, capture_output=True
    )
    fn = ctypes.CDLL(str(library)).cached_panels
    fn.argtypes = [ctypes.c_void_p] * 3
    data = bytes((i * 7 + 3) % 251 for i in range(2 * 6 * 7 * 3))
    expected = []
    for b in range(2):
        for y in range(6):
            for x in (0, 0, 1, 2):
                for ky in range(3):
                    for kx in range(3):
                        for c in range(3):
                            iy, ix = y - 1 + ky, x - 1 + kx
                            expected.append(
                                data[((b * 6 + iy) * 7 + ix) * 3 + c] if (0 <= iy < 6 and 0 <= ix < 7) else 0
                            )
    inputs = ctypes.create_string_buffer(data)
    storage = (ctypes.c_ubyte * (required + 16))(*([239] * (required + 16)))
    outputs = (ctypes.c_ubyte * (len(expected) + 16))(*([241] * (len(expected) + 16)))
    fn(inputs, ctypes.byref(storage, 8), ctypes.byref(outputs, 8))
    assert list(outputs)[8:-8] == expected
    assert list(outputs)[:8] == list(outputs)[-8:] == [241] * 8
    assert list(storage)[:8] == list(storage)[-8:] == [239] * 8


@pytest.mark.parametrize("channels,invalid_row", [(1, False), (3, False), (17, False), (3, True)])
def test_factored_channel_run_actual_pointer_coverage_and_padding(channels, invalid_row):
    """Walk real emitted CFG; no shape declaration substitutes for its operands."""
    from collections import Counter

    from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
    from merlin.compile.scheduling.rolling_spatial import RollingPackedSpatialPlan
    from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan
    from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
    from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan
    from merlin.perf.finite_pointer_trace import ArgumentAddress, project_finite_pointer_trace

    window = Window2D((1, 5, 6, channels), (3, 2), dilation=(2, 1), padding=(2, 1, 2, 0))
    producer = CountedConvolutionPlan(window, 5, 4, 4, 4, min(4, channels), 64, 128, 0, 8, 3, True)
    chain = SpatialReadoutChain(RoundedReadout((0.5,), True), RoundedReadout((0.75,)), None)
    source = SpatialProducerPlan(
        producer, SpatialMaxReadout(window.output_spatial, (3, 3), (2, 2), (1, 1, 1, 1), -128),
        (chain, chain), 1, readout_mode="staged_readouts", intermediate=AddressWindow("old", 0, 128),
    )
    packed = PackedSpatialProducerPlan(source, 4, AddressWindow("owned", 16, 65536), 8)
    plan = RollingPackedSpatialPlan(packed)
    plan = RollingPackedSpatialPlan(replace(packed, workspace=replace(packed.workspace, size=plan.required_bytes)))
    y = window.output_spatial[0] + 2 if invalid_row else 0
    expected = []
    # Original NHWC reduction order, independent integer address calculation.
    for row in range(4):
        x = min(max(row - 1, 0), window.output_spatial[1] - 1)
        for ky in range(3):
            iy = y - 2 + ky * 2
            for kx in range(2):
                ix = x - 1 + kx
                for channel in range(channels):
                    output = (row * 6 + ky * 2 + kx) * channels + channel
                    if row < 3 and 0 <= y < window.output_spatial[0] and 0 <= iy < 5 and 0 <= ix < 6:
                        offset = (plan.rolling.cache.start
                                  + ((y + ky * 2) % plan.rolling.slots) * plan.rolling.row_stride_bytes
                                  + ix * channels + channel)
                        expected.append(("load", ArgumentAddress(0, offset)))
                    expected.append(("store", ArgumentAddress(1, output)))
    branches = []
    for factored in (False, True):
        builder = FnBuilder([PTR, PTR])
        emit_cached_spatial_row_panel(
            plan, builder, *builder.entry.args, producer_y=y, output_x=0, output_pixels=1,
            first_halo_column=0, rows=4, cache_destination_disjoint=True,
            maximum_panel_bytes=4 * 6 * channels, factor_channel_guards=factored,
        )
        fn = llvm.FuncOp("pack", llvm.LLVMFunctionType([PTR, PTR], llvm.LLVMVoidType()),
                         linkage=llvm.LinkageAttr("external"), body=builder.finish())
        counts = Counter()

        def observe(op, args):
            counts[op.name] += 1
            if op.name == "llvm.load":
                return "load", args[0]
            if op.name == "llvm.store":
                return "store", args[1]
            return None

        actual = project_finite_pointer_trace(fn, observe=observe, pointer_bits=64)
        assert actual == tuple(expected)
        assert len([row for row in actual if row[0] == "store"]) == 4 * 6 * channels
        branches.append(counts["llvm.cond_br"])
    if channels > 1:
        assert branches[1] < branches[0]
