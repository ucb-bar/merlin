import struct
from dataclasses import replace

import pytest
from corefixtures.private_copy_elimination import SPACES, command

from merlin.compiler.schedule.copy_coalescing import coalesce_private_copies
from merlin.compiler.schedule.copy_coalescing_check import CopyMergeLaw
from merlin.compiler.schedule.program_limits import AddressWindow, EncodedAccess
from merlin.compiler.schedule.temporary_storage import StorageAdapter, StorageEffects, TemporaryObject
from merlin.sched.ir import Const, Kernel, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import StorageBank


def project(call):
    if call.instr not in {"copy", "increment", "observe"}:
        return None
    args = {key: evaluate(value, {}) for key, value in call.args}
    source, destination = (SPACES[args["read_bank"]], SPACES[args["write_bank"]])
    return StorageEffects(
        repr((call.instr, source, destination, args.get("mode", 0))),
        (
            EncodedAccess(source, args["source"], args["size"]),
            EncodedAccess(destination, args["destination"], args["size"], True),
        ),
        "synchronous",
        exact_copy=call.instr == "copy",
    )


def rewrite(call, accesses):
    updates = {"source": accesses[0].start, "destination": accesses[1].start, "size": accesses[0].size}
    return replace(
        call, args=tuple(((key, Const(updates[key]) if key in updates else value) for key, value in call.args))
    )


def fixture(width=8, *, base=32, alignment=1, backing="memory"):
    adapter = StorageAdapter(project, rewrite, "neutral-contiguous-bytes-v1", "unchanged-bytes-v1", "neutral-C-v1")
    law = CopyMergeLaw(
        adapter.configuration_identity,
        adapter.numerical_policy_identity,
        adapter.qualification_identity,
        "original-neutral-C-bounded-copy-merge-v1",
        4 * width,
        no_faults=True,
        no_extra_observations=True,
    )
    source = Kernel(
        "merge_copies",
        (TensorArg("input", (4 * width,), "i8", "read"), TensorArg("output", (4 * width,), "i8", "write")),
        (
            command("copy", 0, base, 2 * width, 2, 0),
            command("copy", width, base, width, 2, 0),
            command("copy", 2 * width, base + width, width, 2, 0),
            command("copy", base, 0, 2 * width, 0, 3),
        ),
    )
    objects = (TemporaryObject("scratch", "bytes", base, 2 * width),)
    banks = (StorageBank("bytes", backing, base + 4 * width, "bytes", alignment),)
    kwargs = dict(
        merge_law=law,
        private_ownership=True,
        external_windows=(AddressWindow("input", 0, 4 * width), AddressWindow("output", 0, 4 * width)),
    )
    return (source, objects, banks, adapter, kwargs)


@pytest.mark.parametrize("width,separate,word_format", [(8, True, "<BBBQQQ")])
def test_independent_native_byte_executor_observes_fewer_requests_same_complete_state(
    native, width, separate, word_format
):
    source, objects, banks, adapter, kwargs = fixture(width)
    result = coalesce_private_copies(source, objects, banks, adapter, **kwargs)
    word, histories = (struct.Struct(word_format), [])
    for kernel in (source, result.kernel):
        image = bytearray()
        for call in kernel.body:
            args = {key: evaluate(value, {}) for key, value in call.args}
            image += word.pack(
                0, args["read_bank"], args["write_bank"], args["source"], args["destination"], args["size"]
            )
        state = bytes(range(4 * width))
        history = []
        for epoch in range(3):
            data = bytearray(((x + 19 * epoch) % 256 for x in state))
            original = bytes(data)
            first, second, output = (bytearray([173] * 128), bytearray([149] * 128), bytearray([211] * len(data)))
            counts = native(
                image,
                first,
                second,
                data,
                output,
                width=word.size // 3 - 1,
                big=word_format[0] == ">",
                separate=separate,
            )
            assert output == original[width : 3 * width] + bytes([211] * (2 * width))
            assert data == original and second == bytes([149] * 128)
            assert first[:32] == bytes([173] * 32) and first[32 + 2 * width :] == bytes([173] * (96 - 2 * width))
            assert first[32 : 32 + 2 * width] == output[: 2 * width]
            assert counts[:3] == (len(kernel.body), 6 * width, 6 * width)
            history.append(bytes(output))
            state = bytes(output)
        histories.append(history)
    assert histories[0] == histories[1]
