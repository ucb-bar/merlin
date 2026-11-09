from dataclasses import replace

from merlin.compiler.schedule.copy_pipeline import (
    resolve_copy_pipeline_admission,
)
from merlin.compiler.schedule.fusion_semantics import ElementwiseSemantics
from merlin.compiler.schedule.program_limits import AddressWindow, EncodedAccess
from merlin.compiler.schedule.temporary_storage import (
    StorageAdapter,
    StorageEffects,
    TemporaryObject,
)
from merlin.sched.ir import Call, Const, Kernel, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank

AFFINE = b"module {func.func @source(%x:i8)->i8 {\n%three=arith.constant 3:i8\n%seven=arith.constant 7:i8\n%a=arith.muli %x,%three:i8\n%r=arith.addi %a,%seven:i8\nfunc.return %r:i8\n}}"


def op(name, *, address=0, width=1, host=0, token=0, view=0):
    return Call(
        name,
        tuple(
            (
                (key, Const(value))
                for key, value in dict(address=address, width=width, host=host, token=token, view=view).items()
            )
        ),
    )


def project(call):
    values = {key: evaluate(value, {}) for key, value in call.args}
    a, n, h, token = (values[key] for key in ("address", "width", "host", "token"))
    space = ("bytes", "words")[values["view"]]
    local = EncodedAccess(space, a, n)
    if call.instr == "fetch":
        return StorageEffects(
            repr(call),
            (EncodedAccess("input", h, n), replace(local, writes=True)),
            "asynchronous",
            str(token),
            exact_copy=True,
        )
    if call.instr == "await":
        return StorageEffects(repr(call), completion="wait", waits=(str(token),))
    if call.instr == "affine":
        return StorageEffects(
            repr(call),
            (local, replace(local, writes=True)),
            "synchronous",
            elementwise=ElementwiseSemantics(AFFINE, n, ("affine",), ("little", "little")),
        )
    if call.instr == "store":
        return StorageEffects(repr(call), (local, EncodedAccess("output", h, n, True)), "synchronous", exact_copy=True)
    if call.instr in ("setup", "descriptor"):
        return StorageEffects(repr(call), completion="synchronous")
    return None


ADAPTER = StorageAdapter(project, lambda call, accesses: call, "selected-revision", "exact-i8", "checked-byte-effects")


def fixture(*, width=5, panels=7, slots=2, capacity=None, alias=True, mutate=None):
    spacing = width + 3
    base = 8
    end = base + slots * spacing + 8
    banks = (
        StorageBank("bytes", "first", end if capacity is None else capacity, "bytes", 1),
        StorageBank("words", "first" if alias else "second", end, "bytes", 1),
    )
    objects = tuple(
        (TemporaryObject(f"slot{i}", ("bytes", "words")[i % 2], base + i * spacing, width) for i in range(slots))
    )
    body = [op("setup")]
    for i in range(panels):
        address, view = (base + i % slots * spacing, i % slots % 2)
        body.extend(
            (
                op("fetch", address=address, width=width, host=i * width, token=i, view=view),
                op("await", token=i),
                op("affine", address=address, width=width, view=view),
                op("store", address=address, width=width, host=2 + i * width, view=view),
            )
        )
    kernel = Kernel(
        "panels",
        (
            TensorArg("input", (panels * width,), "i8", "read"),
            TensorArg("output", (panels * width + 4,), "i8", "write"),
        ),
        tuple(body),
    )
    facts = {"resources": [{"inflight": 4, "queue": 3}]}
    declaration = {
        "schema": "merlin.copy_pipeline.v1",
        "crossing_pairs": [["fetch", name] for name in ("await", "affine", "store")],
        "concurrent_pairs": [["fetch", "fetch"]],
        "max_in_flight": {"fact": ["resources", 0, "inflight"]},
        "queue_capacity": {"fact": ["resources", 0, "queue"]},
        "independence": "disjoint_memory_preserves_state_and_issue_order",
        "wait_semantics": "token_completion_only_preserves_state",
    }
    if mutate:
        mutate(declaration, facts)
    admission = resolve_copy_pipeline_admission(
        declaration, facts, adapter=ADAPTER, qualification_identity="independent-copy-concurrency-and-waits"
    )
    model = dict(
        external_windows=(AddressWindow("input", 0, panels * width), AddressWindow("output", 0, panels * width + 4)),
        reservations=(Reservation("bytes", 0, base),),
        private_ownership=True,
    )
    return (kernel, objects, banks, admission, model)
