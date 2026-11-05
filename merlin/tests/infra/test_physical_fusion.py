"""Typed arithmetic fusion, actual encoded transfers and independent execution."""

import struct
from dataclasses import replace

import pytest

from merlin.compile.scheduling.fusion import FusionCapability, plan_fusion, verify_fusion_step
from merlin.compile.scheduling.fusion_semantics import ElementwiseSemantics, UnsupportedElementwise, scalar_graph
from merlin.compile.scheduling.program_limits import (
    AddressWindow,
    DecodedInstruction,
    EncodedAccess,
    ProgramDecoder,
    ProgramLimits,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, ScheduleEncoding, compile_schedule
from merlin.compile.scheduling.spill_conformance import verify_storage_dataflow
from merlin.compile.scheduling.temporary_storage import StorageAdapter, StorageEffects, TemporaryObject
from merlin.sched.ir import Call, Const, Kernel, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank
from merlin.verify.epilogue_applicability import ReadoutCapability

SPACES = ("normal", "strict", "input", "output")
WORD = struct.Struct("<BBBQQQ")


def command(name, source, destination, count, read_bank, write_bank):
    return Call(
        name,
        tuple(
            (key, Const(value))
            for key, value in zip(
                ("source", "destination", "count", "read_bank", "write_bank"),
                (source, destination, count, read_bank, write_bank),
                strict=True,
            )
        ),
    )


def scalar_functions(bits, selection=False):
    wide = bits * 2
    bias = (1 << (wide - 1)) - 16
    low, high = -(1 << (bits - 2)), (1 << (bits - 2)) - 1
    producer = f"""%extended = arith.extsi %x : i{bits} to i{wide}
%bias = arith.constant {bias} : i{wide}
%biased = arith.addi %extended, %bias : i{wide}
"""
    consumer = f"""%lo = arith.constant {low} : i{wide}
%hi = arith.constant {high} : i{wide}
%lower = arith.maxsi %biased, %lo : i{wide}
%clamped = arith.minsi %lower, %hi : i{wide}
%out = arith.trunci %clamped : i{wide} to i{bits}
"""

    if selection:
        consumer = f"""%lo = arith.constant {low} : i{wide}
%hi = arith.constant {high} : i{wide}
%too_low = arith.cmpi slt, %biased, %lo : i{wide}
%lower = arith.select %too_low, %lo, %biased : i{wide}
%too_high = arith.cmpi sgt, %lower, %hi : i{wide}
%clamped = arith.select %too_high, %hi, %lower : i{wide}
%out = arith.trunci %clamped : i{wide} to i{bits}
"""

    def function(inputs, output, body, returned):
        return (
            f"builtin.module {{ func.func @forward({inputs}) -> i{output} {{\n{body}"
            f"func.return %{returned} : i{output}\n}} }}"
        ).encode()

    return (
        function(f"%x: i{bits}", wide, producer, "biased"),
        function(f"%biased: i{wide}", bits, consumer, "out"),
        function(f"%x: i{bits}", bits, producer + consumer, "out"),
    )


def fixture(bits=8, count=7, order="little", *, private_output=False, strict=False, separate=False, selection=False):
    width = bits // 8
    functions = scalar_functions(bits, selection)
    kinds = ("producer", "consumer", "fused", "save")

    def project(call):
        if call.instr not in kinds:
            return None
        args = {key: evaluate(value, {}) for key, value in call.args}
        read_width = 2 * width if call.instr == "consumer" else width
        write_width = 2 * width if call.instr == "producer" else width
        accesses = (
            EncodedAccess(SPACES[args["read_bank"]], args["source"], args["count"] * read_width),
            EncodedAccess(SPACES[args["write_bank"]], args["destination"], args["count"] * write_width, True),
        )
        stages = {
            "producer": ("cast", "bias"),
            "consumer": ("clamp", "cast"),
            "fused": ("cast", "bias", "clamp", "cast"),
        }
        semantic = (
            None
            if call.instr == "save"
            else ElementwiseSemantics(
                functions[kinds.index(call.instr)],
                args["count"],
                stages[call.instr],
                (order, order),
                "single-physical-readout" if call.instr == "fused" else None,
            )
        )
        return StorageEffects(
            call.instr, accesses, "synchronous", exact_copy=call.instr == "save", elementwise=semantic
        )

    def rewrite(call, accesses):
        changes = {"source": Const(accesses[0].start), "destination": Const(accesses[1].start)}
        return replace(call, args=tuple((key, changes.get(key, value)) for key, value in call.args))

    adapter = StorageAdapter(
        project,
        rewrite,
        f"neutral-{bits}-{order}-{'split' if separate else 'shared'}",
        f"wrapping-i{2 * bits}-source",
        "synthetic-only:" + str(selection),
    )

    def fuse(producer, consumer):
        if producer.instr != "producer" or consumer.instr != "consumer":
            return None
        p, c = dict(producer.args), dict(consumer.args)
        return command(
            "fused",
            evaluate(p["source"], {}),
            evaluate(c["destination"], {}),
            evaluate(p["count"], {}),
            evaluate(p["read_bank"], {}),
            1 if strict else evaluate(c["write_bank"], {}),
        )

    capability = FusionCapability(
        ReadoutCapability(
            "single-physical-readout", frozenset({"cast", "bias", "clamp"}), "synthetic physical composition"
        ),
        fuse,
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        "synthetic-fused-op",
    )
    capacity = (4 * count * width + 8) if private_output else (2 * count * width)
    banks = (
        StorageBank("normal", "local-backing", capacity, "bytes"),
        StorageBank(
            "strict", "result-backing" if separate else "local-backing", capacity, "bytes", 32 if strict else 1
        ),
    )
    objects = (TemporaryObject("intermediate", "normal", 64, 2 * count * width),)
    body = (
        command("producer", 0, 64, count, 2, 0),
        command(
            "consumer", 64, 256 if private_output else 0, count, 0, (1 if separate else 0) if private_output else 3
        ),
    )
    if private_output:
        objects += (TemporaryObject("result", "strict" if separate else "normal", 256, count * width),)
        body += (command("save", 256, 0, count, 1 if separate else 0, 3),)
    kernel = Kernel(
        "fusion",
        (TensorArg("input", (count * width,), "i8", "read"), TensorArg("output", (2 * count * width,), "i8", "write")),
        body,
    )
    model = dict(
        private_ownership=True,
        external_windows=(AddressWindow("input", 0, count * width), AddressWindow("output", 0, 2 * count * width)),
    )

    def encode(kernel):
        return b"".join(
            WORD.pack(
                kinds.index(call.instr),
                evaluate(call.arg("read_bank"), {}),
                evaluate(call.arg("write_bank"), {}),
                evaluate(call.arg("source"), {}),
                evaluate(call.arg("destination"), {}),
                evaluate(call.arg("count"), {}),
            )
            for call in kernel.body
        )

    def decode(data):
        result = []
        for offset in range(0, len(data), WORD.size):
            opcode, read_bank, write_bank, source, destination, extent = WORD.unpack_from(data, offset)
            if opcode >= len(kinds) or read_bank >= len(SPACES) or write_bank >= len(SPACES):
                raise ValueError("unknown actual instruction")
            result.append(command(kinds[opcode], source, destination, extent, read_bank, write_bank))
        return tuple(result)

    def decode_program(data):
        return tuple(
            DecodedInstruction(
                index * WORD.size, WORD.size, project(call).accesses, completion="synchronous", relocatable=True
            )
            for index, call in enumerate(decode(data))
        )

    encoding = ScheduleEncoding(
        encode,
        decode,
        ProgramDecoder(decode_program, adapter.configuration_identity, "synthetic-byte-decoder"),
        adapter.numerical_policy_identity,
        "synthetic-encoding",
    )
    limits = ProgramLimits(
        WORD.size * 16,
        16,
        1,
        (*model["external_windows"], *(AddressWindow(bank.name, 0, bank.capacity) for bank in banks)),
        (),
    )
    return kernel, objects, banks, adapter, capability, encoding, limits, model


@pytest.mark.parametrize("bits,count,order", [(8, 3, "little"), (16, 7, "big"), (8, 19, "big")])
def test_ordinary_pipeline_fuses_one_physical_readout_and_eliminates_actual_intermediate_effects(bits, count, order):
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture(bits, count, order)
    baseline = compile_schedule(
        kernel,
        objects,
        banks,
        adapter,
        encoding,
        limits,
        fusion_capabilities=(capability,),
        budgets=ScheduleBudgets(max_fusions=0),
        **model,
    )
    optimized = compile_schedule(
        kernel, objects, banks, adapter, encoding, limits, fusion_capabilities=(capability,), **model
    )
    assert baseline.program.report["instruction_count"] == 2
    assert optimized.program.report["instruction_count"] == 1
    assert optimized.report["storage"]["addresses"] == {}
    assert optimized.report["storage"]["fusion"]["status"] == "realized"
    proof = optimized.report["storage"]["fusion"]["checked_steps"][0]
    assert proof["ordered_stages"] == ["cast", "bias", "clamp", "cast"]
    assert proof["original_requested_read_bytes"] == 3 * count * (bits // 8)
    assert proof["actual_requested_read_bytes"] == count * (bits // 8)
    calls = tuple(call for segment in optimized.program.segments for call in encoding.decode_calls(segment))
    actual = tuple(adapter.project(call) for call in calls)
    assert all(access.space in {"input", "output"} for effect in actual for access in effect.accesses)
    assert optimized.report["final_conformance"]["requested_read_bytes"] == count * (bits // 8)
    assert baseline.report["context"]["library"] == optimized.report["context"]["library"]
    assert [record["name"] for record in optimized.report["passes"]] == [
        "semantic_admission",
        "fusion",
        "storage",
        "completion",
        "emission",
        "final_program",
    ]


def test_failed_physical_fusion_allocation_restores_original_and_replans():
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture(
        count=3, private_output=True, strict=True
    )
    model["reservations"] = (Reservation("normal", 0, 1),)
    result = compile_schedule(
        kernel, objects, banks, adapter, encoding, limits, fusion_capabilities=(capability,), **model
    )
    assert result.report["storage"]["fusion"]["status"] == "allocation_fallback"
    assert result.report["repair_rounds"] == 1
    assert result.program.report["instruction_count"] == 3
    assert all(call.instr != "fused" for segment in result.program.segments for call in encoding.decode_calls(segment))
    with pytest.raises(ValueError, match="repair budget exhausted"):
        compile_schedule(
            kernel,
            objects,
            banks,
            adapter,
            encoding,
            limits,
            fusion_capabilities=(capability,),
            budgets=ScheduleBudgets(max_repairs=0),
            **model,
        )


def test_stages_from_different_physical_readouts_cannot_be_combined():
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture()
    first = replace(capability, readout=ReadoutCapability("first", frozenset({"cast", "bias"})))
    second = replace(capability, readout=ReadoutCapability("second", frozenset({"clamp"})))
    result = compile_schedule(
        kernel, objects, banks, adapter, encoding, limits, fusion_capabilities=(first, second), **model
    )
    assert result.program.report["instruction_count"] == 2
    assert "fusion" not in result.report["storage"]


@pytest.mark.parametrize("mutation", ["arithmetic", "conversion", "clamp", "layout", "stages", "readout"])
def test_independent_actual_fusion_check_rejects_changed_numerical_boundaries(mutation):
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture()

    def decode(call):
        effect = adapter.project(call)
        if call.instr != "fused":
            return effect
        semantics = effect.elementwise
        body = semantics.function_ir
        if mutation == "arithmetic":
            body = body.replace(b"32752", b"32751")
        elif mutation == "conversion":
            body = body.replace(b"arith.extsi", b"arith.extui")
        elif mutation == "clamp":
            body = body.replace(b"arith.maxsi", b"arith.minsi")
        updates = {"function_ir": body}
        if mutation == "layout":
            updates["byte_orders"] = ("big", "little")
        if mutation == "stages":
            updates["stages"] = ("bias", "clamp", "cast")
        if mutation == "readout":
            updates["readout"] = "another-physical-path"
        return replace(effect, elementwise=replace(semantics, **updates))

    with pytest.raises(ValueError, match="arithmetic|numerical boundaries|single selected readout"):
        compile_schedule(
            kernel,
            objects,
            banks,
            replace(adapter, project=decode),
            encoding,
            limits,
            fusion_capabilities=(capability,),
            **model,
        )


def test_other_observers_overwrites_and_unknown_arithmetic_keep_original_route():
    kernel, objects, banks, adapter, capability, *_rest, model = fixture()
    side_user = command("save", 64, 0, 7, 0, 3)
    observed = replace(kernel, body=(*kernel.body, side_user))
    assert plan_fusion(observed, objects, banks, adapter, (capability,), **model) is None
    overwritten = replace(kernel, body=(kernel.body[0], kernel.body[0], kernel.body[1]))
    assert plan_fusion(overwritten, objects, banks, adapter, (capability,), **model) is None
    assert plan_fusion(kernel, objects, banks, adapter, (replace(capability, emit=lambda p, c: None),), **model) is None

    def unsupported(call):
        effect = adapter.project(call)
        if call.instr == "producer":
            body = effect.elementwise.function_ir.replace(b"arith.addi", b"arith.divsi")
            return replace(effect, elementwise=replace(effect.elementwise, function_ir=body))
        return effect

    assert plan_fusion(kernel, objects, banks, replace(adapter, project=unsupported), (capability,), **model) is None


def test_mismatched_intermediate_byte_layout_is_a_conservative_barrier():
    kernel, objects, banks, adapter, capability, *_rest, model = fixture()

    def decode(call):
        effect = adapter.project(call)
        if call.instr == "consumer":
            return replace(effect, elementwise=replace(effect.elementwise, byte_orders=("big", "little")))
        return effect

    assert plan_fusion(kernel, objects, banks, replace(adapter, project=decode), (capability,), **model) is None


def test_shifted_streaming_alias_is_not_fused():
    kernel, objects, banks, adapter, capability, *_rest, model = fixture()
    body = (kernel.body[0], command("consumer", 64, 1, 7, 0, 2))
    model["external_windows"] = (AddressWindow("input", 0, 8), AddressWindow("output", 0, 14))
    kernel = replace(kernel, args=(replace(kernel.args[0], shape=(8,), access="readwrite"), kernel.args[1]), body=body)
    assert plan_fusion(kernel, objects, banks, adapter, (capability,), **model) is None


def test_scalar_work_budgets_disable_fusion_without_expanding_elements():
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture()
    result = compile_schedule(
        kernel,
        objects,
        banks,
        adapter,
        encoding,
        limits,
        fusion_capabilities=(capability,),
        budgets=ScheduleBudgets(max_scalar_operations=2),
        **model,
    )
    assert result.program.report["instruction_count"] == 2
    assert plan_fusion(kernel, objects, banks, adapter, (capability,), max_fusions=0, **model) is None
    # Each original function fits five operations, but their composed function
    # does not. Keep the original route rather than accepting an unchecked body.
    limited = compile_schedule(
        kernel,
        objects,
        banks,
        adapter,
        encoding,
        limits,
        fusion_capabilities=(capability,),
        budgets=ScheduleBudgets(max_scalar_operations=5),
        **model,
    )
    assert limited.program.report["instruction_count"] == 2


def test_final_code_and_storage_capacity_limits_use_realized_fusion():
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture()
    small = tuple(replace(bank, capacity=bank.capacity - 1) for bank in banks)
    limits = replace(
        limits,
        max_bytes=WORD.size,
        max_instructions=1,
        windows=(*model["external_windows"], *(AddressWindow(bank.name, 0, bank.capacity) for bank in small)),
    )
    result = compile_schedule(
        kernel, objects, small, adapter, encoding, limits, fusion_capabilities=(capability,), **model
    )
    assert result.program.report["instruction_count"] == 1
    assert result.report["storage"]["addresses"] == {}


def test_multiple_independent_compositions_are_bounded_and_replanned():
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture()
    kernel = replace(
        kernel,
        args=(TensorArg("input", (10,), "i8", "read"), TensorArg("output", (10,), "i8", "write")),
        body=(
            *kernel.body,
            command("producer", 7, 256, 3, 2, 0),
            command("consumer", 256, 7, 3, 0, 3),
        ),
    )
    objects += (TemporaryObject("second_intermediate", "normal", 256, 6),)
    model["external_windows"] = (AddressWindow("input", 0, 10), AddressWindow("output", 0, 10))
    limits = replace(
        limits, windows=(*model["external_windows"], *(AddressWindow(bank.name, 0, bank.capacity) for bank in banks))
    )
    one = compile_schedule(
        kernel,
        objects,
        banks,
        adapter,
        encoding,
        limits,
        fusion_capabilities=(capability,),
        budgets=ScheduleBudgets(max_fusions=1),
        **model,
    )
    both = compile_schedule(
        kernel, objects, banks, adapter, encoding, limits, fusion_capabilities=(capability,), **model
    )
    assert one.program.report["instruction_count"] == 3
    assert both.program.report["instruction_count"] == 2
    assert len(both.report["storage"]["fusion"]["checked_steps"]) == 2
    assert both.report["storage"]["addresses"] == {}


def test_fusion_step_check_accepts_no_selected_pair_or_removed_store_list():
    kernel, objects, banks, adapter, capability, *_rest, model = fixture()
    candidate = plan_fusion(kernel, objects, banks, adapter, (capability,), **model)
    removed, proof = verify_fusion_step(kernel, candidate.kernel, objects, banks, adapter, capability, **model)
    assert removed == "intermediate" and proof == candidate.checks[0]
    bad = replace(candidate.kernel, body=(*candidate.kernel.body, kernel.body[1]))
    with pytest.raises(ValueError, match="exactly one actual instruction"):
        verify_fusion_step(kernel, bad, objects, banks, adapter, capability, **model)


def test_storage_dataflow_cannot_ignore_changed_typed_semantics_with_the_same_opcode_key():
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture()
    result = compile_schedule(kernel, objects, banks, adapter, encoding, limits, **model)

    def changed(call):
        effect = adapter.project(call)
        if call.instr == "producer" and evaluate(call.arg("destination"), {}) == 0:
            return replace(
                effect,
                elementwise=replace(
                    effect.elementwise,
                    function_ir=effect.elementwise.function_ir.replace(b"arith.extsi", b"arith.extui"),
                ),
            )
        return effect

    with pytest.raises(ValueError, match="ordered original semantics"):
        verify_storage_dataflow(kernel, result.kernel, objects, banks, replace(adapter, project=changed), **model)


def test_unknown_floating_semantics_and_asynchronous_completion_are_barriers():
    with pytest.raises(UnsupportedElementwise, match="total fixed-width integers"):
        scalar_graph(b"builtin.module {func.func @f(%x: f32) -> f32 {func.return %x : f32}}", 128, 65536)
    kernel, objects, banks, adapter, capability, *_rest, model = fixture()
    kernel = replace(kernel, body=(kernel.body[0], Call("wait", ()), kernel.body[1]))

    def decode(call):
        if call.instr == "wait":
            return StorageEffects("wait", completion="wait", waits=("ready",))
        effect = adapter.project(call)
        return replace(effect, completion="asynchronous", token="ready") if call.instr == "producer" else effect

    assert plan_fusion(kernel, objects, banks, replace(adapter, project=decode), (capability,), **model) is None


def test_symbolic_large_vectors_keep_compile_space_proportional_to_scalar_ir():
    count = 1 << 35
    kernel, objects, banks, adapter, capability, *_rest, model = fixture(count=count)
    candidate = plan_fusion(kernel, objects, banks, adapter, (capability,), **model)
    assert len(candidate.kernel.body) == 1 and candidate.objects == ()
    assert candidate.checks[0]["elements"] == count
    assert candidate.checks[0]["actual_requested_read_bytes"] == count


def test_extra_consumer_operand_is_preserved_in_actual_decoding_and_execution():
    import numpy as np

    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture()
    count = 7
    _p_body, c_body, f_body = scalar_functions(8)

    def residual_function(body):
        body = (
            body.replace(b"-> i8 {", b"-> i8 {\n%sum = arith.addi %biased, %rhs : i16\n")
            if body == c_body
            else body.replace(b"%lo =", b"%sum = arith.addi %biased, %rhs : i16\n%lo =", 1)
        )
        body = body.replace(b"%biased, %lo", b"%sum, %lo")
        header = b"%biased: i16)" if body.startswith(b"builtin.module { func.func @forward(%biased") else b"%x: i8)"
        return body.replace(header, header[:-1] + b", %rhs: i16)")

    consumer, fused = residual_function(c_body), residual_function(f_body)

    def project(call):
        effect = adapter.project(call)
        if call.instr not in {"consumer", "fused"}:
            return effect
        return replace(
            effect,
            accesses=(*effect.accesses[:-1], EncodedAccess("residual", 0, count * 2), effect.accesses[-1]),
            elementwise=replace(
                effect.elementwise,
                function_ir=consumer if call.instr == "consumer" else fused,
                stages=("bias", *effect.elementwise.stages)
                if call.instr == "consumer"
                else (*effect.elementwise.stages[:2], "bias", *effect.elementwise.stages[2:]),
                byte_orders=("little", "little", "little"),
            ),
        )

    selected = replace(adapter, project=project, configuration_identity=adapter.configuration_identity + ":residual")
    capability = replace(capability, configuration_identity=selected.configuration_identity)
    decoder = replace(
        encoding.program_decoder,
        configuration_identity=selected.configuration_identity,
        decode=lambda image: tuple(
            DecodedInstruction(
                index * WORD.size, WORD.size, project(call).accesses, completion="synchronous", relocatable=True
            )
            for index, call in enumerate(encoding.decode_calls(image))
        ),
    )
    encoding = replace(encoding, program_decoder=decoder)
    kernel = replace(kernel, args=(*kernel.args, TensorArg("residual", (14,), "i8", "read")))
    model["external_windows"] += (AddressWindow("residual", 0, 14),)
    limits = replace(limits, windows=(*limits.windows, model["external_windows"][-1]))
    baseline = compile_schedule(
        kernel,
        objects,
        banks,
        selected,
        encoding,
        limits,
        fusion_capabilities=(capability,),
        budgets=ScheduleBudgets(max_fusions=0),
        **model,
    )
    optimized = compile_schedule(
        kernel, objects, banks, selected, encoding, limits, fusion_capabilities=(capability,), **model
    )
    x = np.asarray([-128, -1, 0, 1, 15, 16, 127], dtype=np.int8)
    residual = np.asarray([-32768, -1, 0, 1, 16, 32767, -999], dtype=np.dtype("<i2"))
    reference = np.clip(
        ((x.astype(np.int64) + 32752).astype(np.int16).astype(np.int64) + residual).astype(np.int16), -64, 63
    ).astype(np.int8).tobytes() + bytes([91] * count)
    for result, expected_reads, expected_writes in ((baseline, 35, 21), (optimized, 21, 7)):
        memory = {
            "input": bytearray(x.tobytes()),
            "residual": bytearray(residual.tobytes()),
            "output": bytearray([91] * 14),
            "local": bytearray(banks[0].capacity),
        }
        reads = writes = 0

        def read(space, address, width):
            nonlocal reads
            reads += width
            return int.from_bytes(memory[space][address : address + width], "little", signed=True)

        def write(space, address, width, value):
            nonlocal writes
            writes += width
            memory[space][address : address + width] = (value % (1 << (8 * width))).to_bytes(width, "little")

        for image in result.program.segments:
            for cursor in range(0, len(image), WORD.size):
                op, rb, wb, source, destination, n = WORD.unpack_from(image, cursor)
                rb, wb = ("local" if rb < 2 else SPACES[rb]), ("local" if wb < 2 else SPACES[wb])
                for element in range(n):
                    value = read(rb, source + element * (2 if op == 1 else 1), 2 if op == 1 else 1)
                    if op in (0, 2):
                        value = ((value + 32752 + 32768) % 65536) - 32768
                    if op in (1, 2):
                        value = ((value + read("residual", element * 2, 2) + 32768) % 65536) - 32768
                        value = min(63, max(-64, value))
                    write(wb, destination + element * (2 if op == 0 else 1), 2 if op == 0 else 1, value)
        assert bytes(memory["output"]) == reference
        assert bytes(memory["input"]) == x.tobytes() and bytes(memory["residual"]) == residual.tobytes()
        assert (reads, writes) == (expected_reads, expected_writes)
        assert result.report["final_conformance"]["requested_read_bytes"] == reads
        assert result.report["final_conformance"]["requested_write_bytes"] == writes


@pytest.mark.parametrize(
    "bits,count,order,fault,separate,selection",
    [
        (8, 3, "little", 0, False, False),
        (16, 7, "big", 0, False, False),
        (8, 19, "big", 0, False, False),
        (16, 7, "little", 0, True, False),
        (16, 7, "big", 0, False, True),
        (8, 7, "little", 1, False, False),
        (8, 7, "little", 2, False, False),
    ],
)
def test_native_execution_of_actual_encoded_programs_and_independent_full_outputs(
    tmp_path, bits, count, order, fault, separate, selection
):
    import ctypes
    import shutil
    import subprocess

    import numpy as np

    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C toolchain unavailable")
    kernel, objects, banks, adapter, capability, encoding, limits, model = fixture(
        bits,
        count,
        order,
        separate=separate,
        private_output=separate,
        selection=selection,
    )
    baseline = compile_schedule(
        kernel,
        objects,
        banks,
        adapter,
        encoding,
        limits,
        fusion_capabilities=(capability,),
        budgets=ScheduleBudgets(max_fusions=0),
        **model,
    )
    optimized = compile_schedule(
        kernel, objects, banks, adapter, encoding, limits, fusion_capabilities=(capability,), **model
    )
    width, wide = bits // 8, bits * 2
    bias = (1 << (wide - 1)) - 16
    low, high = -(1 << (bits - 2)), (1 << (bits - 2)) - 1
    code = f"""#include <stdint.h>
#include <assert.h>
uint64_t reads,writes,commands;
static uint8_t local[{banks[0].capacity}], result_memory[{banks[1].capacity}], *input_ptr, *output_ptr;
static uint8_t *space(int bank) {{
 return bank==0?local:bank==1?({int(separate)}?result_memory:local):bank==2?input_ptr:output_ptr;
}}
static uint64_t word(const uint8_t *p) {{uint64_t x=0;for(int i=0;i<8;i++) x|=((uint64_t)p[i])<<(8*i);return x;}}
static int64_t read_value(int bank,int address,int bytes) {{
 uint64_t x=0; reads+=bytes;
 for(int i=0;i<bytes;i++) x|=((uint64_t)space(bank)[address+i])<<(8*({int(order == "big")}?bytes-1-i:i));
 if(x & (1ULL<<(8*bytes-1))) return (int64_t)x-(1LL<<(8*bytes));return (int64_t)x;
}}
static void write_value(int bank,int address,int bytes,int64_t value) {{
 writes+=bytes;for(int i=0;i<bytes;i++) space(bank)[address+i]=(uint64_t)value>>(8*({int(order == "big")}?bytes-1-i:i));
}}
static int64_t biased(int64_t x) {{
 uint64_t y=(x+{bias}) & ((1ULL<<{wide})-1);
 return y & (1ULL<<{wide - 1}) ? (int64_t)y-(1LL<<{wide}):(int64_t)y;
}}
static int64_t clamp(int64_t x) {{return x<{low}?{low}:x>{high}?{high}:x;}}
void run(const uint8_t *program,int length,uint8_t *input,uint8_t *output) {{
 input_ptr=input;output_ptr=output;reads=writes=commands=0;
 for(int cursor=0;cursor<length;cursor+={WORD.size}) {{
  int op=program[cursor], rb=program[cursor+1],wb=program[cursor+2];
  int source=word(program+cursor+3),dest=word(program+cursor+11),n=word(program+cursor+19);
  commands++;
  if(op==2 && {fault}==2) {{
   for(int i=0;i<n;i++) write_value(0,i*{2 * width},{2 * width},biased(read_value(rb,source+i*{width},{width})));
   for(int i=0;i<n;i++) write_value(wb,dest+i*{width},{width},clamp(read_value(0,i*{2 * width},{2 * width})));
  }} else for(int i=0;i<n;i++) {{
   int rw=op==1?{2 * width}:{width},ww=op==0?{2 * width}:{width};
   int64_t x=read_value(rb,source+i*rw,rw);
   if(op==0 || op==2) x=biased(x);
   if(op==1 || op==2) x=clamp(x);
   if(op==2 && {fault}==1) x=x=={high}?{low}:{high};
   write_value(wb,dest+i*ww,ww,x);
  }}
 }}
}}
"""
    source, library = (
        tmp_path / "execute.c",
        tmp_path / f"execute_{bits}_{count}_{order}_{fault}_{separate}_{selection}.so",
    )
    source.write_text(code)
    subprocess.run(
        [cc, "-std=c11", "-O1", "-shared", "-fPIC", str(source), "-o", str(library)], check=True, capture_output=True
    )
    native = ctypes.CDLL(str(library))
    dtype = np.dtype((">" if order == "big" else "<") + f"i{width}")
    wide_dtype = np.dtype(f"i{2 * width}")
    samples = np.asarray([-(1 << (bits - 1)), 0, (1 << (bits - 1)) - 1] * ((count + 2) // 3), dtype=np.int64)[:count]
    raw = samples.astype(dtype).tobytes()
    # Independent vector reference preserves the intermediate signed wrap,
    # clamp before narrowing, data byte order, complete outputs and input state.
    expected = np.clip((samples + bias).astype(wide_dtype).astype(np.int64), low, high).astype(dtype).tobytes()
    expected += bytes([91] * (count * width))
    for name, result, traffic in (
        ("baseline", baseline, (3 + int(separate)) * count * width),
        ("optimized", optimized, (1 + int(separate)) * count * width),
    ):
        image = b"".join(result.program.segments)
        program = (ctypes.c_uint8 * len(image)).from_buffer_copy(image)
        inputs = (ctypes.c_uint8 * len(raw)).from_buffer_copy(raw)
        outputs = (ctypes.c_uint8 * len(expected))(*([91] * len(expected)))
        native.run(program, len(image), inputs, outputs)
        assert bytes(inputs) == raw
        if fault == 1 and name == "optimized":
            assert bytes(outputs) != expected  # target arithmetic qualification fails
        else:
            assert bytes(outputs) == expected
        actual_traffic = ctypes.c_uint64.in_dll(native, "reads").value
        assert actual_traffic == ctypes.c_uint64.in_dll(native, "writes").value
        if fault == 2 and name == "optimized":
            assert actual_traffic > traffic  # hidden stores/reloads violate decoded effects
        else:
            assert actual_traffic == traffic
        assert ctypes.c_uint64.in_dll(native, "commands").value == (2 if name == "baseline" else 1) + int(separate)
