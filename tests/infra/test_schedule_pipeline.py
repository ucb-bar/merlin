import struct
from dataclasses import replace

import pytest
from corefixtures.private_copy_elimination import ADAPTER, SPACES, command, fixture, project

from merlin.compiler.schedule.program_limits import (
    AddressWindow,
    DecodedInstruction,
    ProgramDecoder,
    ProgramLimits,
)
from merlin.compiler.schedule.schedule_pipeline import ScheduleBudgets, ScheduleEncoding, compile_schedule
from merlin.sched.ir import evaluate


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
            (
                DecodedInstruction(
                    index * word.size, word.size, project(call).accesses, completion="synchronous", relocatable=True
                )
                for index, call in enumerate(decode_calls(data))
            )
        )

    return (
        ScheduleEncoding(
            encode,
            decode_calls,
            ProgramDecoder(decode_program, ADAPTER.configuration_identity, "neutral-byte-decoder-only:" + word_format),
            ADAPTER.numerical_policy_identity,
            "synthetic-encoding-only:" + word_format,
        ),
        word,
    )


def limits_for(banks, model, word, count=32):
    return ProgramLimits(
        word.size * count,
        count,
        1,
        (*model["external_windows"], *(AddressWindow(bank.name, 0, bank.capacity) for bank in banks)),
        (),
    )


def execute_bytes(segments, word, capacity, inputs, initial_output, *, separate=False):
    local = bytearray(capacity)
    spaces = (local, bytearray(capacity) if separate else local, bytearray(inputs), bytearray(initial_output))
    commands = reads = writes = 0
    for image in segments:
        for cursor in range(0, len(image), word.size):
            opcode, read_bank, write_bank, source, destination, size = word.unpack_from(image, cursor)
            data = bytes(spaces[read_bank][source : source + size])
            assert len(data) == size and destination + size <= len(spaces[write_bank])
            if opcode == 1:
                data = bytes(((value + 1) % 256 for value in data))
            else:
                assert opcode == 0
            spaces[write_bank][destination : destination + size] = data
            commands += 1
            reads += size
            writes += size
    return (bytes(spaces[2]), bytes(spaces[3]), (commands, reads, writes))


@pytest.mark.parametrize("width,word_format", [(3, "<BBBQQQ")])
def test_ordinary_schedule_path_optimizes_encodes_checks_and_executes_complete_outputs(width, word_format):
    kernel, objects, banks, model = fixture(width)
    codec, word = encoding(word_format)
    limits = limits_for(banks, model, word)
    baseline = compile_schedule(
        kernel, objects, banks, ADAPTER, codec, limits, budgets=ScheduleBudgets(eliminate_copies=False), **model
    )
    optimized = compile_schedule(kernel, objects, banks, ADAPTER, codec, limits, **model)
    data = bytes(((i * 31 + 253) % 256 for i in range(2 * width)))
    expected = bytes(((value + 1) % 256 for value in data[:width])) + bytes([91] * width)
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


@pytest.mark.parametrize("mutation", ["arithmetic", "read_address"])
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
