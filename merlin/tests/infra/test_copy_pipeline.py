"""Actual copy launches and waits under an independent byte/event interpreter."""

from dataclasses import replace

import pytest

from merlin.compile.scheduling.copy_pipeline import (
    CopyPipelinePolicy,
    pipeline_private_copies,
    resolve_copy_pipeline_admission,
    verify_copy_pipeline,
)
from merlin.compile.scheduling.fusion_semantics import ElementwiseSemantics
from merlin.compile.scheduling.program_limits import AddressWindow, EncodedAccess
from merlin.compile.scheduling.temporary_storage import (
    StorageAdapter,
    StorageEffects,
    StorageOptimizationUnavailable,
    TemporaryObject,
)
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, evaluate
from merlin.semantic_compiler.allocation_types import Reservation, StorageBank

AFFINE = b"""module {func.func @source(%x:i8)->i8 {
%three=arith.constant 3:i8
%seven=arith.constant 7:i8
%a=arith.muli %x,%three:i8
%r=arith.addi %a,%seven:i8
func.return %r:i8
}}"""


def op(name, *, address=0, width=1, host=0, token=0, view=0):
    return Call(
        name,
        tuple(
            (key, Const(value))
            for key, value in dict(
                address=address,
                width=width,
                host=host,
                token=token,
                view=view,
            ).items()
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
        return StorageEffects(
            repr(call),
            (local, EncodedAccess("output", h, n, True)),
            "synchronous",
            exact_copy=True,
        )
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
        TemporaryObject(f"slot{i}", ("bytes", "words")[i % 2], base + i * spacing, width) for i in range(slots)
    )
    body = [op("setup")]
    for i in range(panels):
        address, view = base + i % slots * spacing, i % slots % 2
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
        declaration,
        facts,
        adapter=ADAPTER,
        qualification_identity="independent-copy-concurrency-and-waits",
    )
    model = dict(
        external_windows=(AddressWindow("input", 0, panels * width), AddressWindow("output", 0, panels * width + 4)),
        reservations=(Reservation("bytes", 0, base),),
        private_ownership=True,
    )
    return kernel, objects, banks, admission, model


def run(kernel, banks, raw, *, capacity=3, latency=30):
    """Decode the actual payload, with delayed DMA and explicit token waits.

    This interpreter does not call StorageAdapter or any scheduling/checker
    helper. A pending DMA reserves its read/write intervals until its actual
    wait. Completion makes bytes visible; issue never does. The clock is a
    synthetic event model, not processor or target timing evidence.
    """
    backings = {bank.backing: bytearray([131] * bank.capacity) for bank in banks}
    memories = {bank.name: backings[bank.backing] for bank in banks}
    output = bytearray([197] * kernel.args[1].nbytes)
    memories.update(input=bytearray(raw), output=output)
    pending, seen, waits = {}, set(), []
    clock, maximum, state = 0, 0, None

    def hazard(accesses):
        for memory, start, n, write in accesses:
            for _ready, source, h, destination, a, width in pending.values():
                for old_memory, old_start, old_write in ((source, h, False), (destination, a, True)):
                    if (
                        memory is old_memory
                        and (write or old_write)
                        and start < old_start + width
                        and old_start < start + n
                    ):
                        raise AssertionError("access to a pending transfer")

    for call in kernel.body:
        v = {name: evaluate(value, {}) for name, value in call.args}
        a, n, h, token = (v[name] for name in ("address", "width", "host", "token"))
        local = memories[("bytes", "words")[v["view"]]]
        if call.instr == "fetch":
            hazard(((memories["input"], h, n, False), (local, a, n, True)))
            assert token not in seen and len(pending) < capacity
            seen.add(token)
            pending[token] = (clock + latency, memories["input"], h, local, a, n)
            maximum = max(maximum, len(pending))
        elif call.instr == "await":
            assert token in pending, "wait has no pending launch"
            ready, source, start, destination, address, width = pending.pop(token)
            clock = max(clock, ready)
            destination[address : address + width] = source[start : start + width]
            waits.append(token)
        elif call.instr == "affine":
            hazard(((local, a, n, True),))
            assert state == "selected-setup"
            local[a : a + n] = bytes((value * 3 + 7) % 256 for value in local[a : a + n])
            clock += n
        elif call.instr == "store":
            hazard(((local, a, n, False), (output, h, n, True)))
            output[h : h + n] = local[a : a + n]
            clock += n
        elif call.instr == "setup":
            state = "selected-setup"
        elif call.instr == "descriptor":
            state = "selected-setup"
        else:
            raise AssertionError("unknown instruction")
        clock += 1
    assert not pending, "return before DMA completion"
    return bytes(output), {key: bytes(value) for key, value in backings.items()}, waits, maximum, clock


@pytest.mark.parametrize("width,panels,slots,alias", [(1, 5, 2, True), (7, 8, 2, False), (13, 11, 3, True)])
def test_real_launches_pipeline_into_bounded_private_slots(width, panels, slots, alias):
    kernel, objects, banks, admission, model = fixture(width=width, panels=panels, slots=slots, alias=alias)
    policy = CopyPipelinePolicy(distance=12, max_moves=panels, max_pending_copies=slots)
    candidate, report = pipeline_private_copies(kernel, objects, banks, ADAPTER, admission, policy=policy, **model)
    raw = bytes((i * 37 + 11) % 256 for i in range(width * panels))
    before, after = run(kernel, banks, raw), run(candidate, banks, raw)
    assert before[:3] == after[:3]
    assert after[0] == bytes([197, 197]) + bytes((value * 3 + 7) % 256 for value in raw) + bytes([197, 197])
    assert before[3] == 1 and after[3] == slots
    assert after[4] < before[4]
    assert report["maximum_pending_copies"] == slots
    assert report["removed_completion_instructions"] == 0
    assert len(candidate.body) == len(kernel.body)
    assert [call for call in candidate.body if call.instr == "await"] == [
        call for call in kernel.body if call.instr == "await"
    ]
    assert (
        verify_copy_pipeline(
            kernel,
            candidate,
            tuple(report["original_positions"]),
            objects,
            banks,
            ADAPTER,
            admission,
            policy=policy,
            **model,
        )
        == report
    )
    assert report["timing"] == "UNKNOWN"


@pytest.mark.parametrize("bound", [1, 2, 3])
def test_policy_queue_and_hardware_limits_independently_bound_overlap(bound):
    kernel, objects, banks, admission, model = fixture(slots=3)
    for field in ("max_in_flight", "queue_capacity", "policy"):
        selected = admission if field == "policy" else replace(admission, **{field: bound})
        policy = CopyPipelinePolicy(distance=20, max_moves=20, max_pending_copies=bound if field == "policy" else 3)
        candidate, report = pipeline_private_copies(kernel, objects, banks, ADAPTER, selected, policy=policy, **model)
        actual = run(candidate, banks, bytes(range(35)), capacity=bound)
        assert actual[3] == report["maximum_pending_copies"] == bound


@pytest.mark.parametrize("missing", ["wait", "concurrency", "affine", "store"])
def test_missing_pair_admission_blocks_only_unsupported_motion(missing):
    kernel, objects, banks, admission, model = fixture()
    if missing == "concurrency":
        admission = replace(admission, concurrent_pairs=())
    else:
        excluded = "await" if missing == "wait" else missing
        admission = replace(
            admission, crossing_pairs=tuple(pair for pair in admission.crossing_pairs if pair[1] != excluded)
        )
    policy = CopyPipelinePolicy(distance=10, max_moves=20, max_pending_copies=2)
    candidate, report = pipeline_private_copies(kernel, objects, banks, ADAPTER, admission, policy=policy, **model)
    actual = run(candidate, banks, bytes(range(35)))
    if missing in ("wait", "concurrency"):
        assert actual[3] == report["maximum_pending_copies"] == 1
    for move in report["moves"]:
        assert all(("fetch", kernel.body[i].instr) in admission.crossing_pairs for i in move["crossed_source_indices"])


@pytest.mark.parametrize("barrier", ["setup", "descriptor"])
def test_state_and_descriptor_operations_are_not_moved_or_crossed(barrier):
    kernel, objects, banks, admission, model = fixture()
    kernel = replace(kernel, body=(*kernel.body[:5], op(barrier), *kernel.body[5:]))
    # Even an erroneous broad pair declaration cannot convert state changes
    # into pure memory operations or token waits.
    admission = replace(admission, crossing_pairs=(*admission.crossing_pairs, ("fetch", barrier)))
    candidate, report = pipeline_private_copies(
        kernel,
        objects,
        banks,
        ADAPTER,
        admission,
        policy=CopyPipelinePolicy(distance=20, max_moves=20, max_pending_copies=2),
        **model,
    )
    positions = report["original_positions"]
    assert positions.index(5) < positions.index(6)
    assert run(candidate, banks, bytes(range(35)))[:3] == run(kernel, banks, bytes(range(35)))[:3]


@pytest.mark.parametrize("mutation", ["remove_wait", "overwrite_live", "reverse_launch", "change_address", "queue"])
def test_independent_final_checker_rejects_hazardous_or_nonmember_candidates(mutation):
    kernel, objects, banks, admission, model = fixture(slots=3)
    policy = CopyPipelinePolicy(distance=20, max_moves=20, max_pending_copies=3)
    candidate, report = pipeline_private_copies(kernel, objects, banks, ADAPTER, admission, policy=policy, **model)
    order = list(report["original_positions"])
    if mutation == "remove_wait":
        order.remove(2)
    elif mutation == "overwrite_live":
        order.remove(13)  # fourth launch overwrites slot zero before its consumer
        order.insert(4, 13)
    elif mutation == "reverse_launch":
        first, second = order.index(1), order.index(5)
        order[first], order[second] = order[second], order[first]
    elif mutation == "queue":
        admission = replace(admission, queue_capacity=1)
    actual = replace(kernel, body=tuple(kernel.body[i] for i in order))
    if mutation == "change_address":
        changed = replace(
            actual.body[1],
            args=tuple((key, Const(999) if key == "address" else value) for key, value in actual.body[1].args),
        )
        actual = replace(actual, body=(actual.body[0], changed, *actual.body[2:]))
    with pytest.raises(ValueError):
        verify_copy_pipeline(kernel, actual, tuple(order), objects, banks, ADAPTER, admission, policy=policy, **model)
    if mutation in ("remove_wait", "overwrite_live"):
        with pytest.raises(AssertionError):
            run(actual, banks, bytes(range(35)))


@pytest.mark.parametrize("mutation", ["capacity", "reservation", "ownership", "window", "unknown", "async_arithmetic"])
def test_actual_storage_and_completion_prerequisites_are_enforced(mutation):
    kernel, objects, banks, admission, model = fixture()
    adapter = ADAPTER
    if mutation == "capacity":
        banks = tuple(replace(bank, capacity=10) for bank in banks)
    elif mutation == "reservation":
        model["reservations"] = (Reservation("bytes", 8, 5),)
    elif mutation == "ownership":
        model["private_ownership"] = False
    elif mutation == "window":
        model["external_windows"] = ()
    elif mutation == "unknown":
        kernel = replace(kernel, body=(op("opaque"), *kernel.body[1:]))
    else:

        def arithmetic(call):
            effect = project(call)
            return replace(effect, exact_copy=False) if call.instr == "fetch" else effect

        adapter = replace(adapter, project=arithmetic)
    with pytest.raises(ValueError):
        pipeline_private_copies(kernel, objects, banks, adapter, admission, **model)


@pytest.mark.parametrize(
    "mutation", ["unknown", "literal", "zero", "float", "missing", "wait_semantics", "extra", "duplicate"]
)
def test_admission_is_structural_fact_bound_and_separately_qualified(mutation):
    def mutate(declaration, facts):
        if mutation in ("unknown", "zero", "float"):
            facts["resources"][0]["queue"] = {"unknown": "UNKNOWN", "zero": 0, "float": 2.0}[mutation]
        elif mutation == "literal":
            declaration["queue_capacity"] = 4
        elif mutation == "missing":
            declaration["max_in_flight"] = {"fact": ["absent"]}
        elif mutation == "wait_semantics":
            declaration["wait_semantics"] = "issue_is_completion"
        elif mutation == "extra":
            declaration["fastest_cycles"] = 1
        else:
            declaration["crossing_pairs"].append(["fetch", "await"])

    with pytest.raises(ValueError):
        fixture(mutate=mutate)


def test_disabled_motion_and_wrong_qualification_are_explicit():
    kernel, objects, banks, admission, model = fixture()
    for policy in (CopyPipelinePolicy(distance=0), CopyPipelinePolicy(max_moves=0)):
        candidate, report = pipeline_private_copies(kernel, objects, banks, ADAPTER, admission, policy=policy, **model)
        assert candidate == kernel and report["status"] == "unchanged"
    for field in ("hardware_identity", "storage_qualification_identity"):
        with pytest.raises(ValueError, match="selected storage"):
            pipeline_private_copies(
                kernel, objects, banks, ADAPTER, replace(admission, **{field: "different"}), **model
            )
    with pytest.raises(StorageOptimizationUnavailable, match="unknown instruction"):
        pipeline_private_copies(
            replace(kernel, body=(op("opaque"), *kernel.body)), objects, banks, ADAPTER, admission, **model
        )


@pytest.mark.parametrize("field", ["admission", "policy", "bank", "source"])
def test_decoder_cannot_mutate_the_selected_context_during_scheduling(field):
    kernel, objects, banks, admission, model = fixture()
    policy = CopyPipelinePolicy(distance=3, max_pending_copies=2)
    changed = False

    def mutate(call):
        nonlocal changed
        if not changed:
            changed = True
            if field == "admission":
                object.__setattr__(admission, "queue_capacity", 999)
            elif field == "policy":
                object.__setattr__(policy, "distance", 999)
            elif field == "bank":
                object.__setattr__(banks[0], "capacity", 999)
            else:
                object.__setattr__(kernel, "name", "different_source")
        return project(call)

    with pytest.raises(ValueError, match="changed during"):
        pipeline_private_copies(
            kernel, objects, banks, replace(ADAPTER, project=mutate), admission, policy=policy, **model
        )


def test_private_and_external_aliases_block_physical_dependencies():
    kernel, objects, banks, admission, model = fixture()

    def external_aliases(call):
        effect = project(call)
        if call.instr == "fetch":
            effect = replace(effect, accesses=(replace(effect.accesses[0], space="host"), effect.accesses[1]))
        elif call.instr == "store":
            # Each result overwrites bytes of the next input panel. A future
            # input copy must stay after this source-visible external write.
            effect = replace(
                effect,
                accesses=(
                    effect.accesses[0],
                    replace(
                        effect.accesses[1],
                        space="host",
                        start=effect.accesses[1].start - 2 + 5,
                    ),
                ),
            )
        return effect

    model["external_windows"] = (AddressWindow("host", 0, 40),)
    adapter = replace(ADAPTER, project=external_aliases)
    candidate, report = pipeline_private_copies(
        kernel,
        objects,
        banks,
        adapter,
        admission,
        policy=CopyPipelinePolicy(distance=20, max_moves=20, max_pending_copies=2),
        **model,
    )
    assert candidate == kernel
    assert report["status"] == "unchanged"


def test_source_and_realized_call_budgets_precede_program_traversal():
    kernel, objects, banks, admission, model = fixture()
    with pytest.raises(StorageOptimizationUnavailable, match="bounded straight-line"):
        pipeline_private_copies(kernel, objects, banks, ADAPTER, admission, max_calls=1, **model)
    with pytest.raises(StorageOptimizationUnavailable, match="bounded straight-line"):
        pipeline_private_copies(
            replace(kernel, body=(Loop("i", 1000000, kernel.body),)),
            objects,
            banks,
            ADAPTER,
            admission,
            **model,
        )
    with pytest.raises(StorageOptimizationUnavailable, match="bounded straight-line"):
        verify_copy_pipeline(
            kernel,
            replace(kernel, body=kernel.body * 2),
            tuple(range(len(kernel.body))),
            objects,
            banks,
            ADAPTER,
            admission,
            max_calls=len(kernel.body),
            **model,
        )
