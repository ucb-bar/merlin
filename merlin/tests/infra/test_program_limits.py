"""Decode and execute actual neutral byte programs, including refused cuts."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.program_limits import (
    AddressWindow,
    BranchRule,
    Continuation,
    DecodedInstruction,
    EncodedAccess,
    EncodedBranch,
    ProgramDecoder,
    ProgramLimits,
    finalize_program,
)


def decode(data):
    instructions = []
    cursor = 0
    while cursor < len(data):
        opcode = data[cursor]
        accesses, branches = (), ()
        completion = "synchronous"
        if opcode == 1:
            size = 1
        elif opcode in (0x10, 0x11, 0x20, 0x21):
            size = 3
            address = data[cursor + 1]
            if opcode == 0x10:
                accesses = (EncodedAccess("data", address, 1, True),)
            elif opcode == 0x11:
                accesses = (EncodedAccess("data", address, 1), EncodedAccess("data", address, 1, True))
            elif opcode == 0x20:
                accesses = (EncodedAccess("registers", address, 1, True),)
            else:
                accesses = (EncodedAccess("registers", address, 1), EncodedAccess("data", data[cursor + 2], 1, True))
        elif opcode == 0x30:
            size = 2
            displacement = int.from_bytes(data[cursor + 1 : cursor + 2], "little", signed=True)
            branches = (EncodedBranch("relative", displacement),)
        elif opcode == 0x31:
            size = 2
            accesses = (EncodedAccess("data", data[cursor + 1], 1, True),)
            completion = "asynchronous"
        elif opcode == 0x32:
            size = 1
            completion = "drain"
        else:
            raise ValueError("unknown instruction")
        instructions.append(DecodedInstruction(cursor, size, accesses, branches, completion, True))
        cursor += size
    return tuple(instructions)


DECODER = ProgramDecoder(decode, "neutral-variable-width", "synthetic-decoder-only")
LIMITS = ProgramLimits(
    12,
    6,
    1,
    (AddressWindow("data", 0, 16), AddressWindow("registers", 0, 8)),
    (BranchRule("relative", -128, 127, 1, "next_instruction"),),
)
CONTINUATION = Continuation("neutral-sequential-launch", ("data",), True, True, 16)


def execute(segments):
    """Independent interpreter reads emitted bytes; each launch resets registers."""
    memory = bytearray(16)
    commands = 0
    for segment in segments:
        registers = bytearray(8)
        pending = []
        pc = 0
        while pc < len(segment):
            commands += 1
            assert commands < 1000
            op = segment[pc]
            if op == 1:
                pc += 1
            elif op in (0x10, 0x11, 0x20, 0x21):
                address, value = segment[pc + 1 : pc + 3]
                if op == 0x10:
                    memory[address] = value
                elif op == 0x11:
                    memory[address] = (memory[address] + value) % 256
                elif op == 0x20:
                    registers[address] = value
                else:
                    memory[value] = registers[address]
                pc += 3
            elif op == 0x30:
                pc += 2 + int.from_bytes(segment[pc + 1 : pc + 2], "little", signed=True)
            elif op == 0x31:
                pending.append(segment[pc + 1])
                pc += 2
            elif op == 0x32:
                for address in pending:
                    memory[address] = (memory[address] + 1) % 256
                pending.clear()
                pc += 1
            else:
                raise ValueError("unsupported actual opcode")
        assert not pending, "launch returned with unfinished work"
    return bytes(memory), commands


def test_final_program_is_segmented_and_actual_bytes_execute_identically():
    image = bytes([0x10, 0, 5] + [0x11, 0, 1] * 8)
    result = finalize_program(image, LIMITS, DECODER, continuation=CONTINUATION, outputs=(EncodedAccess("data", 0, 1),))
    baseline, baseline_commands = execute((image,))
    optimized, commands = execute(result.segments)
    assert baseline == optimized == bytes([13] + [0] * 15)
    assert commands == baseline_commands == result.report["instruction_count"] == 9
    assert result.report["segment_bytes"] == [12, 12, 3]
    assert b"".join(result.segments) == image
    assert result.report["execution_qualification"] == "not_established"


def test_instruction_budget_and_byte_budget_are_separate_final_limits():
    image = bytes([1] * 9)
    result = finalize_program(
        image, replace(LIMITS, max_bytes=100, max_instructions=3), DECODER, continuation=CONTINUATION, outputs=()
    )
    assert result.report["segment_bytes"] == [3, 3, 3]
    assert result.report["instruction_count"] == 9
    with pytest.raises(ValueError, match="no qualified continuation"):
        finalize_program(image, replace(LIMITS, max_instructions=8), DECODER)
    exact = finalize_program(image, replace(LIMITS, max_bytes=9, max_instructions=9), DECODER)
    assert exact.segments == (image,)


@pytest.mark.parametrize("mutation", ["address", "branch_target", "branch_reach", "omitted", "trailing", "unknown"])
def test_corrupted_final_bytes_or_decoding_are_rejected(mutation):
    image, limits, decoder = bytes([0x10, 0, 3, 1]), LIMITS, DECODER
    if mutation == "address":
        image = bytes([0x10, 17, 3, 1])
    elif mutation == "branch_target":
        image = bytes([0x30, 1, 0x10, 0, 3])  # lands inside the store
    elif mutation == "branch_reach":
        image = bytes([0x30, 2, 1, 1, 1])
        limits = replace(limits, branches=(replace(limits.branches[0], maximum=1),))
    elif mutation == "omitted":
        decoder = replace(DECODER, decode=lambda data: decode(data)[:-1])
    elif mutation == "trailing":
        decoder = replace(DECODER, decode=lambda data: (replace(decode(data)[0], size=len(data) + 1),))
    else:
        image = bytes([255])
    with pytest.raises(ValueError):
        finalize_program(image, limits, decoder)


def test_branch_relative_origin_and_internal_edges_survive_relocation():
    # Skip a store, then increment the same external counter. Each closed branch
    # region remains intact and is decoded again at its actual launch address.
    region = bytes([0x30, 3, 0x10, 0, 99, 0x11, 0, 1])
    image = region * 3
    result = finalize_program(
        image, replace(LIMITS, max_bytes=8), DECODER, continuation=CONTINUATION, outputs=(EncodedAccess("data", 0, 1),)
    )
    assert [len(segment) for segment in result.segments] == [8, 8, 8]
    assert execute(result.segments) == execute((image,))
    assert execute(result.segments)[0][0] == 3
    with pytest.raises(ValueError, match="no completion-safe"):
        finalize_program(
            image,
            replace(LIMITS, max_bytes=4),
            DECODER,
            continuation=CONTINUATION,
            outputs=(EncodedAccess("data", 0, 1),),
        )


def test_live_volatile_values_and_published_outputs_prevent_unsafe_cuts():
    image = bytes([0x20, 0, 7, 0x11, 0, 1, 0x21, 0, 1])
    assert execute((image,))[0][:2] == bytes([1, 7])
    with pytest.raises(ValueError, match="no completion-safe"):
        finalize_program(
            image,
            replace(LIMITS, max_bytes=6),
            DECODER,
            continuation=CONTINUATION,
            outputs=(EncodedAccess("data", 0, 2),),
        )
    # A value observable only by the caller must be retained too, even without
    # a later instruction reading it. Only the final cut need not preserve it.
    image = bytes([0x20, 0, 7] + [1] * 7)
    with pytest.raises(ValueError, match="no completion-safe"):
        finalize_program(
            image,
            replace(LIMITS, max_bytes=5),
            DECODER,
            continuation=CONTINUATION,
            outputs=(EncodedAccess("registers", 0, 1),),
        )


def test_actual_completion_is_required_before_each_segment_boundary():
    image = bytes([0x31, 0, 1, 1, 0x32] * 3)
    result = finalize_program(
        image, replace(LIMITS, max_bytes=5), DECODER, continuation=CONTINUATION, outputs=(EncodedAccess("data", 0, 1),)
    )
    assert result.report["segment_bytes"] == [5, 5, 5]
    assert execute(result.segments) == execute((image,))
    assert execute(result.segments)[0][0] == 3
    for maximum, program in ((4, image), (5, image[:-1])):
        with pytest.raises(ValueError, match="completion"):
            finalize_program(
                program,
                replace(LIMITS, max_bytes=maximum),
                DECODER,
                continuation=CONTINUATION,
                outputs=(EncodedAccess("data", 0, 1),),
            )


def test_missing_contracts_and_changed_segment_effects_cannot_be_promoted():
    image = bytes([1] * 9)
    with pytest.raises(ValueError, match="published output"):
        finalize_program(image, replace(LIMITS, max_bytes=4), DECODER, continuation=CONTINUATION)
    for field, value in (("completion", "unknown"), ("relocatable", False)):
        decoder = replace(DECODER, decode=lambda data: tuple(replace(op, **{field: value}) for op in decode(data)))
        with pytest.raises(ValueError, match="unknown completion or position-dependent"):
            finalize_program(image, replace(LIMITS, max_bytes=4), decoder, continuation=CONTINUATION, outputs=())

    def changed(data):
        actual = decode(data)
        if len(data) < len(image):
            actual = (replace(actual[0], accesses=(EncodedAccess("data", 0, 1, True),)), *actual[1:])
        return actual

    with pytest.raises(ValueError, match="instruction effects differ"):
        finalize_program(
            image, replace(LIMITS, max_bytes=4), replace(DECODER, decode=changed), continuation=CONTINUATION, outputs=()
        )
    with pytest.raises(ValueError, match="launch budget"):
        finalize_program(
            image, replace(LIMITS, max_bytes=2), DECODER, continuation=replace(CONTINUATION, max_segments=4), outputs=()
        )


def test_distinct_fixed_word_instruction_model_and_alignment():
    def words(data):
        if len(data) % 4:
            raise ValueError("incomplete word")
        return tuple(
            DecodedInstruction(offset, 4, completion="synchronous", relocatable=True)
            for offset in range(0, len(data), 4)
        )

    limits = ProgramLimits(12, None, 4, (), ())
    decoder = ProgramDecoder(words, "neutral-fixed-word", "synthetic-fixed-word-decoder")
    image = bytes(range(24))
    result = finalize_program(image, limits, decoder, continuation=CONTINUATION, outputs=())
    assert result.report["segment_bytes"] == [12, 12]
    assert result.report["instruction_count"] == 6
    assert result.report["instruction_budget"] == "unknown"
    with pytest.raises(ValueError, match="instruction structure"):
        finalize_program(bytes([1, 1, 1, 1]), limits, DECODER)


def test_closed_forward_and_backward_edges_keep_instruction_boundaries_together():
    # The forward edge skips the backward edge. Both encodings must still
    # remain internal, including the backward edge's target at offset zero.
    region = bytes([0x30, 2, 0x30, 252, 1])
    image = region * 64
    result = finalize_program(
        image,
        replace(LIMITS, max_bytes=5),
        DECODER,
        continuation=replace(CONTINUATION, max_segments=64),
        outputs=(),
    )
    assert result.segments == (region,) * 64
    assert execute(result.segments) == execute((image,))
    with pytest.raises(ValueError, match="no completion-safe"):
        finalize_program(image, replace(LIMITS, max_bytes=4), DECODER, continuation=CONTINUATION, outputs=())


@pytest.mark.parametrize("field,value", [("branches", (object(),)), ("windows", (object(),)), ("branches", [])])
def test_untyped_hardware_facts_have_a_precise_diagnostic(field, value):
    with pytest.raises(ValueError, match="explicit immutable declarations"):
        replace(LIMITS, **{field: value})
