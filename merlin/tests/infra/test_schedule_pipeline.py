"""Execute the normal schedule pipeline and interpret its final encoded bytes."""

import struct
from dataclasses import replace

import pytest
from test_private_copy_elimination import ADAPTER, SPACES, command, fixture, project

from merlin.compile.scheduling.program_limits import (
    AddressWindow,
    Continuation,
    DecodedInstruction,
    ProgramDecoder,
    ProgramLimits,
)
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, ScheduleEncoding, compile_schedule
from merlin.compile.scheduling.spill_conformance import SpillClass
from merlin.compile.scheduling.temporary_storage import StorageOptimizationUnavailable, TemporaryObject
from merlin.sched.ir import Call, Loop, TensorArg, evaluate
from merlin.transforms.pipeline_state import (
    AnalysisStamp,
    PassSpec,
    PipelineContext,
    PipelineSession,
    validate_pass_order,
)


def encoding(word_format="<BBBQQQ"):
    word = struct.Struct(word_format)

    def encode(kernel):
        data = []
        for call in kernel.body:
            args = {name: evaluate(value, {}) for name, value in call.args}
            data.append(
                word.pack(
                    ("copy", "increment").index(call.instr),
                    args["read_bank"],
                    args["write_bank"],
                    args["source"],
                    args["destination"],
                    args["size"],
                )
            )
        return b"".join(data)

    def decode_calls(data):
        result = []
        for offset in range(0, len(data), word.size):
            op, read_bank, write_bank, source, destination, size = word.unpack_from(data, offset)
            if op not in (0, 1) or read_bank >= len(SPACES) or write_bank >= len(SPACES):
                raise ValueError("unknown actual opcode or memory selector")
            result.append(command(("copy", "increment")[op], source, destination, size, read_bank, write_bank))
        return tuple(result)

    def decode_program(data):
        return tuple(
            DecodedInstruction(
                index * word.size, word.size, project(call).accesses, completion="synchronous", relocatable=True
            )
            for index, call in enumerate(decode_calls(data))
        )

    return ScheduleEncoding(
        encode,
        decode_calls,
        ProgramDecoder(decode_program, ADAPTER.configuration_identity, "neutral-byte-decoder-only:" + word_format),
        ADAPTER.numerical_policy_identity,
        "synthetic-encoding-only:" + word_format,
    ), word


def limits_for(banks, model, word, count=32):
    return ProgramLimits(
        word.size * count,
        count,
        1,
        (*model["external_windows"], *(AddressWindow(bank.name, 0, bank.capacity) for bank in banks)),
        (),
    )


def execute_bytes(segments, word, capacity, inputs, initial_output, *, separate=False):
    # Execution uses the actual encoded bytes directly, separately from both
    # the request stream and the semantic decoder used by the compiler gate.
    local = bytearray(capacity)
    spaces = (local, bytearray(capacity) if separate else local, bytearray(inputs), bytearray(initial_output))
    commands = reads = writes = 0
    for image in segments:
        for cursor in range(0, len(image), word.size):
            opcode, read_bank, write_bank, source, destination, size = word.unpack_from(image, cursor)
            data = bytes(spaces[read_bank][source : source + size])
            assert len(data) == size and destination + size <= len(spaces[write_bank])
            if opcode == 1:
                data = bytes((value + 1) % 256 for value in data)
            else:
                assert opcode == 0
            spaces[write_bank][destination : destination + size] = data
            commands += 1
            reads += size
            writes += size
    return bytes(spaces[2]), bytes(spaces[3]), (commands, reads, writes)


@pytest.mark.parametrize("width,word_format", [(3, "<BBBQQQ"), (17, ">BBBIII")])
def test_ordinary_schedule_path_optimizes_encodes_checks_and_executes_complete_outputs(width, word_format):
    kernel, objects, banks, model = fixture(width)
    codec, word = encoding(word_format)
    limits = limits_for(banks, model, word)
    baseline = compile_schedule(
        kernel, objects, banks, ADAPTER, codec, limits, budgets=ScheduleBudgets(eliminate_copies=False), **model
    )
    optimized = compile_schedule(kernel, objects, banks, ADAPTER, codec, limits, **model)
    data = bytes((i * 31 + 253) % 256 for i in range(2 * width))
    expected = bytes((value + 1) % 256 for value in data[:width]) + bytes([91] * width)
    for result, count in ((baseline, 4), (optimized, 3)):
        actual_input, actual_output, counters = execute_bytes(
            result.program.segments, word, 2 * width, data, bytes([91] * 2 * width)
        )
        assert actual_input == data and actual_output == expected
        assert counters == (count, count * width, count * width)
        assert result.program.report["encoded_bytes"] == count * word.size
        assert result.program.report["instruction_count"] == count
        assert result.report["final_conformance"]["actual_commands"] == count
        assert result.report["final_conformance"]["requested_read_bytes"] == count * width
        assert [record["name"] for record in result.report["passes"]] == [
            "semantic_admission",
            "storage",
            "completion",
            "emission",
            "final_program",
        ]
        assert result.report["passes"][1]["invalidated"] == ("source_effects",)
        assert result.report["final_conformance"]["execution_qualification"] == "not_established"
    assert baseline.report["context"]["library"] == optimized.report["context"]["library"]
    assert baseline.report["context"]["resources"] != optimized.report["context"]["resources"]


@pytest.mark.parametrize(
    "order",
    [
        ("storage", "semantic_admission", "completion", "emission", "final_program"),
        ("semantic_admission", "storage", "emission", "completion", "final_program"),
        ("semantic_admission", "storage", "completion", "final_program", "emission"),
        ("storage",),
    ],
)
def test_explicit_invalid_order_refuses_before_any_target_callback(order):
    kernel, objects, banks, model = fixture()
    codec, word = encoding()
    calls = []
    codec = replace(codec, encode=lambda kernel: calls.append(kernel))
    with pytest.raises(ValueError, match="prerequisites|ordering|every selected"):
        compile_schedule(
            kernel, objects, banks, ADAPTER, codec, limits_for(banks, model, word), pass_order=order, **model
        )
    assert calls == []


@pytest.mark.parametrize("mutation", ["arithmetic", "read_address", "omission", "extra", "unknown", "truncated"])
def test_actual_encoding_mutations_fail_conformance_or_final_address_check(mutation):
    kernel, objects, banks, model = fixture()
    codec, word = encoding()

    def broken(kernel):
        calls = list(kernel.body)
        if mutation == "arithmetic":
            calls[1] = replace(calls[1], instr="copy")
        elif mutation == "read_address":
            calls[-1] = command("copy", 1000, 0, 8, 1, 3)
        elif mutation == "omission":
            calls.pop()
        elif mutation == "extra":
            calls.append(calls[-1])
        data = codec.encode(replace(kernel, body=tuple(calls)))
        if mutation == "unknown":
            data = bytes([255]) + data[1:]
        if mutation == "truncated":
            data = data[:-1]
        return data

    with pytest.raises((ValueError, struct.error)):
        compile_schedule(
            kernel, objects, banks, ADAPTER, replace(codec, encode=broken), limits_for(banks, model, word), **model
        )


def test_cached_request_stream_cannot_substitute_for_actual_operand_decoding():
    kernel, objects, banks, model = fixture()
    codec, word = encoding()
    captured = []

    def broken(kernel):
        captured[:] = kernel.body
        wrong = replace(kernel, body=(*kernel.body[:-1], command("copy", 8, 0, 8, 1, 3)))
        return codec.encode(wrong)

    lying = replace(codec, encode=broken, decode_calls=lambda image: tuple(captured))
    with pytest.raises(ValueError, match="matching straight-line access and completion"):
        compile_schedule(kernel, objects, banks, ADAPTER, lying, limits_for(banks, model, word), **model)


def test_encoder_cannot_mutate_verified_schedule_to_match_its_wrong_output():
    kernel, objects, banks, model = fixture()
    codec, word = encoding()

    def mutate(kernel):
        object.__setattr__(kernel, "body", kernel.body[:-1])
        return codec.encode(kernel)

    with pytest.raises(ValueError, match="analysis pass emission changed IR"):
        compile_schedule(
            kernel, objects, banks, ADAPTER, replace(codec, encode=mutate), limits_for(banks, model, word), **model
        )


def test_final_segmentation_redecodes_numerical_semantics_and_executes_complete_state():
    kernel, objects, banks, model = fixture()
    codec, word = encoding()
    continuation = Continuation("neutral-launches", SPACES, True, True, 3)
    limits = limits_for(banks, model, word, count=2)
    result = compile_schedule(kernel, objects, banks, ADAPTER, codec, limits, continuation=continuation, **model)
    assert len(result.program.segments) == 2
    data = bytes(range(16))
    assert execute_bytes(result.program.segments, word, 16, data, bytes([91] * 16))[1] == bytes(range(1, 9)) + bytes(
        [91] * 8
    )

    def inconsistent(data):
        calls = codec.decode_calls(data)
        if len(data) <= 2 * word.size:
            calls = tuple(replace(call, instr="copy") if call.instr == "increment" else call for call in calls)
        return calls

    with pytest.raises(ValueError, match="actual emitted numerical"):
        compile_schedule(
            kernel,
            objects,
            banks,
            ADAPTER,
            replace(codec, decode_calls=inconsistent),
            limits,
            continuation=continuation,
            **model,
        )


def test_final_program_growth_and_missing_continuation_refuse():
    kernel, objects, banks, model = fixture()
    codec, word = encoding()
    with pytest.raises(ValueError, match="exceeds budget"):
        compile_schedule(kernel, objects, banks, ADAPTER, codec, limits_for(banks, model, word, count=2), **model)
    with pytest.raises(ValueError, match="state-preserving cut"):
        compile_schedule(
            kernel,
            objects,
            banks,
            ADAPTER,
            codec,
            limits_for(banks, model, word, count=2),
            continuation=Continuation("volatile", (), True, True, 4),
            **model,
        )


def test_final_instruction_budget_counts_realized_copy_elimination():
    kernel, objects, banks, model = fixture()
    codec, word = encoding()
    limits = limits_for(banks, model, word, count=3)
    with pytest.raises(ValueError, match="exceeds budget"):
        compile_schedule(
            kernel, objects, banks, ADAPTER, codec, limits, budgets=ScheduleBudgets(eliminate_copies=False), **model
        )
    result = compile_schedule(kernel, objects, banks, ADAPTER, codec, limits, **model)
    assert result.program.report["instruction_count"] == 3


def test_allocation_fallback_consumes_global_repair_budget_and_recomputes():
    kernel, _, banks, model = fixture(capacity=24)
    objects = (
        TemporaryObject("a", "bytes", 0, 16),
        TemporaryObject("b", "bytes", 32, 8),
        TemporaryObject("c", "bytes", 64, 16),
    )
    kernel = replace(
        kernel,
        body=(
            command("copy", 0, 0, 16, 2, 0),
            command("copy", 0, 32, 8),
            command("copy", 0, 64, 16, 2, 0),
            command("copy", 64, 0, 16, 0, 3),
            command("copy", 32, 8, 8, 0, 3),
        ),
    )
    codec, word = encoding()
    limits = limits_for(banks, model, word)
    with pytest.raises(ValueError, match="repair budget exhausted"):
        compile_schedule(
            kernel, objects, banks, ADAPTER, codec, limits, budgets=ScheduleBudgets(max_repairs=0), **model
        )
    result = compile_schedule(
        kernel, objects, banks, ADAPTER, codec, limits, budgets=ScheduleBudgets(max_repairs=1), **model
    )
    assert result.report["repair_rounds"] == 1
    assert result.report["storage"]["copy_elimination"]["status"] == "allocation_fallback"
    assert result.program.report["instruction_count"] == 5
    assert result.report["passes"][1]["repair_round"] == 1


def test_device_spill_insertion_is_rechecked_and_charged_to_the_same_global_budget():
    kernel, _, banks, model = fixture(separate=True, capacity=8)
    kernel = replace(
        kernel,
        args=(TensorArg("input", (24,), "i8", "read"), TensorArg("output", (24,), "i8", "write")),
        body=(
            command("copy", 0, 0, 8, 2, 0),
            command("increment", 0, 0, 8),
            command("copy", 8, 32, 8, 2, 0),
            command("increment", 32, 32, 8),
            command("copy", 32, 8, 8, 0, 3),
            command("copy", 16, 64, 8, 2, 0),
            command("copy", 64, 16, 8, 0, 3),
            command("copy", 0, 0, 8, 0, 3),
        ),
    )
    objects = tuple(TemporaryObject(str(index), "bytes", address, 8) for index, address in enumerate((0, 32, 64)))
    model["external_windows"] = (AddressWindow("input", 0, 24), AddressWindow("output", 0, 24))
    capability = SpillClass(
        "bytes",
        "typed_view",
        8,
        1,
        lambda read, write: command(
            "copy", read.start, write.start, read.size, SPACES.index(read.space), SPACES.index(write.space)
        ),
        "exact-neutral-spills",
    )
    codec, word = encoding()
    with pytest.raises(ValueError, match="repair budget exhausted"):
        compile_schedule(
            kernel,
            objects,
            banks,
            ADAPTER,
            codec,
            limits_for(banks, model, word),
            spill_classes=(capability,),
            budgets=ScheduleBudgets(max_repairs=0),
            **model,
        )
    result = compile_schedule(
        kernel,
        objects,
        banks,
        ADAPTER,
        codec,
        limits_for(banks, model, word),
        spill_classes=(capability,),
        budgets=ScheduleBudgets(max_repairs=1),
        **model,
    )
    assert result.report["repair_rounds"] == 1
    assert result.report["storage"]["spills"]["inserted_exact_copies"] == 2
    assert result.program.report["instruction_count"] == 10
    with pytest.raises(ValueError, match="exceeds budget"):
        compile_schedule(
            kernel,
            objects,
            banks,
            ADAPTER,
            codec,
            limits_for(banks, model, word, count=8),
            spill_classes=(capability,),
            budgets=ScheduleBudgets(max_repairs=1),
            **model,
        )
    data = bytes(range(24))
    before, after, counters = execute_bytes(result.program.segments, word, 8, data, bytes([91] * 24), separate=True)
    assert before == data
    assert after == bytes(range(1, 17)) + bytes(range(16, 24))
    assert counters == (10, 80, 80)


def test_a_lying_numerical_decoder_still_needs_independent_execution_qualification():
    kernel, objects, banks, model = fixture()
    codec, word = encoding()

    def broken_encode(kernel):
        return codec.encode(
            replace(
                kernel,
                body=tuple(replace(call, instr="copy") if call.instr == "increment" else call for call in kernel.body),
            )
        )

    def lying_decode(data):
        calls = list(codec.decode_calls(data))
        calls[1] = replace(calls[1], instr="increment")
        return tuple(calls)

    result = compile_schedule(
        kernel,
        objects,
        banks,
        ADAPTER,
        replace(codec, encode=broken_encode, decode_calls=lying_decode),
        limits_for(banks, model, word),
        **model,
    )
    data = bytes(range(16))
    expected = bytes(range(1, 9)) + bytes([91] * 8)
    assert execute_bytes(result.program.segments, word, 16, data, bytes([91] * 16))[1] != expected
    assert result.report["final_conformance"]["execution_qualification"] == "not_established"


@pytest.mark.parametrize("change", ["ir", "hardware", "numerical_policy", "resources", "execution", "library"])
def test_schedule_analysis_custody_rejects_stale_ir_and_every_contract(change):
    kernel, *_ = fixture()
    session = PipelineSession(
        kernel, ir_level="schedule", fingerprint=lambda value: value.digest(), verify_ir=lambda value: None
    )
    session.run(PassSpec("analysis", 0, produces=("fact",), mutates_ir=False, ir_level="schedule"), lambda: {"fact": 1})
    if change == "ir":
        session.module = replace(kernel, body=kernel.body[:-1])
    else:
        session.context = replace(session.context, **{change: "changed"})
    with pytest.raises(ValueError, match="stale"):
        session.analysis("fact")


@pytest.mark.parametrize("field", ["hardware", "numerical_policy", "resources", "execution", "library"])
def test_retained_analysis_does_not_share_the_original_context_object(field):
    kernel, *_ = fixture()
    context = PipelineContext(**{field: "original"})
    session = PipelineSession(
        kernel, context=context, fingerprint=lambda value: value.digest(), verify_ir=lambda value: None
    )
    session.run(PassSpec("analysis", 0, produces=("fact",), mutates_ir=False), lambda: {"fact": "original allocation"})
    object.__setattr__(context, field, "replacement")
    assert getattr(session.records[0].after.context, field) == "original"
    with pytest.raises(ValueError, match="stale"):
        session.analysis("fact")


@pytest.mark.parametrize("field", ["hardware", "numerical_policy", "resources", "execution", "library"])
def test_action_cannot_mutate_a_live_context_and_relabel_its_analysis(field):
    kernel, *_ = fixture()
    context = PipelineContext(**{field: "original"})
    session = PipelineSession(
        kernel, context=context, fingerprint=lambda value: value.digest(), verify_ir=lambda value: None
    )

    def action():
        object.__setattr__(context, field, "replacement")
        return {"fact": "original allocation"}

    with pytest.raises(ValueError, match="changed the selected contracts"):
        session.run(PassSpec("analysis", 0, produces=("fact",), mutates_ir=False), action)
    with pytest.raises(ValueError, match="session failed"):
        session.analysis("fact")


@pytest.mark.parametrize("field", ["hardware", "numerical_policy", "resources", "execution", "library"])
def test_direct_analysis_stamp_retains_context_values_instead_of_an_alias(field):
    context = PipelineContext(**{field: "original"})
    stamp = AnalysisStamp("0" * 64, context, "schedule")
    object.__setattr__(context, field, "replacement")
    assert getattr(stamp.context, field) == "original"
    assert stamp.context is not context


def test_explicit_context_replacement_can_produce_a_fresh_analysis():
    kernel, *_ = fixture()
    session = PipelineSession(kernel, fingerprint=lambda value: value.digest(), verify_ir=lambda value: None)
    spec = PassSpec("analysis", 0, produces=("fact",), mutates_ir=False)
    session.run(spec, lambda: {"fact": "original allocation"})
    session.context = replace(session.context, resources="new original resources")
    with pytest.raises(ValueError, match="stale"):
        session.analysis("fact")
    session.run(spec, lambda: {"fact": "new allocation"})
    assert session.analysis("fact") == "new allocation"
    assert session.records[0].after.context.resources is None
    assert session.records[1].after.context.resources == "new original resources"


def test_alternate_ir_callbacks_are_explicit_and_failures_poison_the_session():
    kernel, *_ = fixture()
    with pytest.raises(ValueError, match="both explicit"):
        PipelineSession(kernel, fingerprint=lambda value: value.digest())
    with pytest.raises(ValueError, match="SHA-256"):
        PipelineSession(kernel, fingerprint=lambda value: "not-a-digest", verify_ir=lambda value: None).stamp()
    session = PipelineSession(kernel, fingerprint=lambda value: value.digest(), verify_ir=lambda value: False)
    with pytest.raises(ValueError, match="structural verifier"):
        session.run(PassSpec("verify", 0), lambda: {})
    with pytest.raises(ValueError, match="session failed"):
        session.stamp()


def test_instruction_mutation_invalidates_existing_emission_and_size_facts():
    kernel, *_ = fixture()
    session = PipelineSession(
        kernel, ir_level="schedule", fingerprint=lambda value: value.digest(), verify_ir=lambda value: None
    )
    session.run(
        PassSpec("emission", 5, produces=("encoded", "size"), mutates_ir=False, ir_level="schedule"),
        lambda: {"encoded": b"old", "size": 3},
    )

    def rewrite():
        session.module = replace(kernel, body=kernel.body[:-1])
        return {}

    session.run(PassSpec("instruction_rewrite", 5, ir_level="schedule"), rewrite)
    assert session.records[-1].invalidated == ("encoded", "size")
    consumed = []
    with pytest.raises(ValueError, match="stale"):
        session.run(
            PassSpec("final", 6, requires=("encoded", "size"), ir_level="schedule"), lambda: consumed.append(True)
        )
    assert consumed == []


def test_pass_cannot_silently_change_ir_level_and_relabel_analyses():
    kernel, *_ = fixture()
    session = PipelineSession(
        kernel, ir_level="schedule", fingerprint=lambda value: value.digest(), verify_ir=lambda value: None
    )

    def change_level():
        session.ir_level = "encoded"
        return {}

    with pytest.raises(ValueError, match="explicit lowering boundary"):
        session.run(PassSpec("unselected_lowering", 5, ir_level="schedule"), change_level)
    with pytest.raises(ValueError, match="session failed"):
        session.stamp()


def test_replacing_an_encoded_image_invalidates_coproduced_and_transitively_derived_checks():
    kernel, *_ = fixture()
    session = PipelineSession(
        kernel, ir_level="schedule", fingerprint=lambda value: value.digest(), verify_ir=lambda value: None
    )
    specs = (
        PassSpec("emit", 5, produces=("encoded", "size"), mutates_ir=False, ir_level="schedule"),
        PassSpec("check", 5, requires=("size",), produces=("checked",), mutates_ir=False, ir_level="schedule"),
        PassSpec("reemit", 5, produces=("encoded",), mutates_ir=False, ir_level="schedule"),
        PassSpec("final", 6, requires=("encoded", "checked"), mutates_ir=False, ir_level="schedule"),
    )
    with pytest.raises(ValueError, match="lacks prerequisites: checked"):
        validate_pass_order(specs)
    session.run(specs[0], lambda: {"encoded": b"first", "size": 5})
    session.run(specs[1], lambda: {"checked": True})
    session.run(specs[2], lambda: {"encoded": b"replacement"})
    assert session.analysis("encoded") == b"replacement"
    assert session.records[-1].invalidated == ("checked", "encoded", "size")
    for name in ("size", "checked"):
        with pytest.raises(ValueError, match="stale"):
            session.analysis(name)


def test_malformed_analysis_products_poison_the_session():
    kernel, *_ = fixture()
    session = PipelineSession(
        kernel, ir_level="schedule", fingerprint=lambda value: value.digest(), verify_ir=lambda value: None
    )
    with pytest.raises(ValueError, match="mapping of produced analyses"):
        session.run(PassSpec("bad", 0, produces=("fact",), ir_level="schedule"), lambda: ["fact"])
    with pytest.raises(ValueError, match="session failed"):
        session.stamp()


def test_missing_effects_and_configuration_do_not_reach_encoding():
    kernel, objects, banks, model = fixture()
    codec, word = encoding()
    with pytest.raises(ValueError, match="matching selected"):
        compile_schedule(
            kernel,
            objects,
            banks,
            replace(ADAPTER, configuration_identity="other"),
            codec,
            limits_for(banks, model, word),
            **model,
        )
    with pytest.raises(ValueError, match="matching selected"):
        compile_schedule(
            kernel,
            objects,
            banks,
            ADAPTER,
            replace(codec, numerical_policy_identity="unsupported-policy"),
            limits_for(banks, model, word),
            **model,
        )
    with pytest.raises(StorageOptimizationUnavailable, match="unknown instruction"):
        compile_schedule(
            replace(kernel, body=(Call("opaque", ()), *kernel.body)),
            objects,
            banks,
            ADAPTER,
            codec,
            limits_for(banks, model, word),
            **model,
        )


def test_unsupported_loop_execution_refuses_before_target_encoding():
    kernel, objects, banks, model = fixture()
    codec, word = encoding()
    seen = []
    kernel = replace(kernel, body=(Loop("i", 1, kernel.body),))
    with pytest.raises(StorageOptimizationUnavailable, match="straight-line"):
        compile_schedule(
            kernel,
            objects,
            banks,
            ADAPTER,
            replace(codec, encode=lambda kernel: seen.append(kernel)),
            limits_for(banks, model, word),
            **model,
        )
    assert seen == []
