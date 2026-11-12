from dataclasses import replace

from merlin.compiler.kernel_ir.ir import Call, Const, Kernel, TensorArg, evaluate
from merlin.compiler.schedule.program_limits import AddressWindow, EncodedAccess
from merlin.compiler.schedule.temporary_storage import (
    StorageAdapter,
    StorageEffects,
    TemporaryObject,
)
from merlin.semantic_compiler.allocation_types import StorageBank

SPACES = ("bytes", "typed_view", "input", "output")


def command(op, source, destination, size, read_bank=0, write_bank=0):
    return Call(
        op,
        tuple(
            (
                (key, Const(value))
                for key, value in zip(
                    ("source", "destination", "size", "read_bank", "write_bank"),
                    (source, destination, size, read_bank, write_bank),
                    strict=True,
                )
            )
        ),
    )


def project(call):
    if call.instr not in {"copy", "increment"}:
        return None
    args = {key: evaluate(value, {}) for key, value in call.args}
    accesses = (
        EncodedAccess(SPACES[args["read_bank"]], args["source"], args["size"]),
        EncodedAccess(SPACES[args["write_bank"]], args["destination"], args["size"], True),
    )
    return StorageEffects(
        repr((call.instr, accesses[0].space, accesses[1].space, args["size"])),
        accesses,
        "synchronous",
        exact_copy=call.instr == "copy",
    )


def rewrite(call, accesses):
    updates = {"source": Const(accesses[0].start), "destination": Const(accesses[1].start)}
    return replace(call, args=tuple(((key, updates.get(key, value)) for key, value in call.args)))


ADAPTER = StorageAdapter(project, rewrite, "neutral-byte-views", "wrap-u8-v1", "synthetic-ISA-only")


def fixture(width=8, *, separate=False, capacity=None):
    kernel = Kernel(
        "private_copies",
        (TensorArg("input", (2 * width,), "i8", "read"), TensorArg("output", (2 * width,), "i8", "write")),
        (
            command("copy", 0, 32, width, 2, 0),
            command("increment", 32, 32, width),
            command("copy", 32, 128, width, 0, 1),
            command("copy", 128, 0, width, 1, 3),
        ),
    )
    objects = (TemporaryObject("a", "bytes", 32, width), TemporaryObject("b", "typed_view", 128, width))
    capacity = 2 * width if capacity is None else capacity
    banks = (
        StorageBank("bytes", "memory", capacity, "bytes"),
        StorageBank("typed_view", "other" if separate else "memory", capacity, "bytes"),
    )
    model = dict(
        private_ownership=True,
        external_windows=(AddressWindow("input", 0, 2 * width), AddressWindow("output", 0, 2 * width)),
    )
    return (kernel, objects, banks, model)
