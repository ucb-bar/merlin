"""Pipeline qualified input copies into existing private buffer slots.

This extends the single-outstanding-copy family without widening its admission.
Concurrent copies and launch motion across pure token waits need a separate
original-owner qualification. Capacity alone grants neither. Launch order,
waits, arithmetic, descriptors and setup remain intact; no fence is removed.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass, replace

from merlin.sched.ir import Call, Kernel
from merlin.targetgen.isa_spec import resolve_fact_integer

from .temporary_storage import (
    StorageAdapter,
    StorageOptimizationUnavailable,
    _completion,
    _overlap,
    _problem,
    verify_temporary_storage,
)


@dataclass(frozen=True)
class CopyPipelineAdmission:
    """Selected software overlap semantics with fact-derived queue bounds.

    crossing_pairs qualify a launch across a synchronous memory-only operation
    or a pure token-completion wait. concurrent_pairs qualify simultaneously
    pending external-to-private exact copies, in their preserved issue order.
    Both require disjoint memory effects, including aliases of the same bank.
    Qualification must cover shared ports, configuration and descriptor state;
    these pairs are never inferred from storage capacity or instruction names.
    """

    crossing_pairs: tuple[tuple[str, str], ...]
    concurrent_pairs: tuple[tuple[str, str], ...]
    max_in_flight: int
    queue_capacity: int
    hardware_identity: str
    storage_qualification_identity: str
    qualification_identity: str
    facts_sha256: str
    software_sha256: str

    def __post_init__(self):
        for pairs in (self.crossing_pairs, self.concurrent_pairs):
            if (
                type(pairs) is not tuple
                or len(pairs) > 4096
                or any(
                    type(pair) is not tuple or len(pair) != 2 or any(type(name) is not str or not name for name in pair)
                    for pair in pairs
                )
                or len(set(pairs)) != len(pairs)
            ):
                raise ValueError("copy pipeline needs bounded unique qualified instruction pairs")
        if any(type(n) is not int or n < 1 for n in (self.max_in_flight, self.queue_capacity)) or any(
            type(identity) is not str or not identity.strip()
            for identity in (
                self.hardware_identity,
                self.storage_qualification_identity,
                self.qualification_identity,
            )
        ):
            raise ValueError("copy pipeline needs positive fact bounds and selected original-owner identities")
        if any(
            type(digest) is not str or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
            for digest in (self.facts_sha256, self.software_sha256)
        ):
            raise ValueError("copy pipeline must record its exact selected facts and software declaration")


def resolve_copy_pipeline_admission(declaration, facts, *, adapter, qualification_identity):
    """Resolve a separately reviewed software contract against selected RTL facts.

    This resolver supplies no qualification or candidate access grant. The
    original execution owner must qualify the exact selected overlap semantics
    and bind the actual hardware revision, as for StorageAdapter itself.
    Missing/UNKNOWN capacity refuses rather than guessing a queue depth.
    """
    if type(adapter) is not StorageAdapter:
        raise ValueError("copy pipeline requires a selected typed storage adapter")
    keys = {
        "schema",
        "crossing_pairs",
        "concurrent_pairs",
        "max_in_flight",
        "queue_capacity",
        "independence",
        "wait_semantics",
    }
    if type(declaration) is not dict or set(declaration) != keys:
        raise ValueError("copy pipeline requires its exact software declaration")
    if (
        declaration["schema"] != "merlin.copy_pipeline.v1"
        or declaration["independence"] != "disjoint_memory_preserves_state_and_issue_order"
        or declaration["wait_semantics"] != "token_completion_only_preserves_state"
    ):
        raise ValueError("unsupported copy concurrency, wait or state semantics")
    pairs = []
    for key in ("crossing_pairs", "concurrent_pairs"):
        rows = declaration[key]
        if type(rows) is not list or len(rows) > 4096 or any(type(pair) is not list for pair in rows):
            raise ValueError("copy pipeline requires explicit bounded software instruction pairs")
        pairs.append(tuple(tuple(pair) for pair in rows))
    return CopyPipelineAdmission(
        *pairs,
        resolve_fact_integer(declaration["max_in_flight"], facts, label="pipeline max in flight", minimum=1),
        resolve_fact_integer(declaration["queue_capacity"], facts, label="pipeline queue capacity", minimum=1),
        adapter.configuration_identity,
        adapter.qualification_identity,
        qualification_identity,
        hashlib.sha256(json.dumps(facts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        hashlib.sha256(
            json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    )


@dataclass(frozen=True)
class CopyPipelinePolicy:
    """Finite search choices, independently capped by the selected admission."""

    distance: int = 1
    max_moves: int = 8
    max_pending_copies: int = 1

    def __post_init__(self):
        if (
            any(type(n) is not int or not 0 <= n <= 1 << 20 for n in (self.distance, self.max_moves))
            or type(self.max_pending_copies) is not int
            or not 1 <= self.max_pending_copies <= 1 << 20
        ):
            raise ValueError("copy pipeline policies need finite distance, moves and pending-copy bounds")


def _input_copy(effect, banks):
    return (
        effect.completion == "asynchronous"
        and effect.exact_copy
        and effect.accesses[0].space not in banks
        and effect.accesses[1].space in banks
    )


def _independent(left, right, banks):
    return not any((a.writes or b.writes) and _overlap(a, b, banks) for a in left.accesses for b in right.accesses)


def _may_cross(launch, crossed, left, right, banks, admission):
    return (
        (launch.instr, crossed.instr) in admission.crossing_pairs
        and _input_copy(left, banks)
        and (
            right.completion == "wait"
            or right.completion == "synchronous"
            and (right.exact_copy or right.elementwise is not None)
        )
        and _independent(left, right, banks)
    )


def _resource_trace(calls, effects, banks, admission, policy):
    """Check actual completion/alias hazards, pending-copy pairs and occupancy."""
    _completion(effects, banks)
    pending, maximum = {}, 0
    capacity = min(admission.max_in_flight, admission.queue_capacity, policy.max_pending_copies)
    for call, effect in zip(calls, effects, strict=True):
        for token in effect.waits:
            del pending[token]
        if effect.completion == "asynchronous":
            if not _input_copy(effect, banks):
                raise StorageOptimizationUnavailable(
                    "copy pipeline does not schedule asynchronous arithmetic or readout"
                )
            for previous, old in pending.values():
                if (previous.instr, call.instr) not in admission.concurrent_pairs or not _independent(
                    old, effect, banks
                ):
                    raise ValueError("simultaneously pending copies lack qualified independent concurrency")
            pending[effect.token] = (call, effect)
            maximum = max(maximum, len(pending))
            if maximum > capacity:
                raise ValueError("copy pipeline exceeds selected queue or pending-copy capacity")
    return maximum


def _check_selection(adapter, admission, policy):
    if type(admission) is not CopyPipelineAdmission or type(policy) is not CopyPipelinePolicy:
        raise ValueError("copy pipeline requires explicit typed admission and search policy")
    if type(adapter) is not StorageAdapter:
        raise ValueError("copy pipeline requires a selected typed storage adapter")
    if (
        admission.hardware_identity != adapter.configuration_identity
        or admission.storage_qualification_identity != adapter.qualification_identity
    ):
        raise ValueError("copy pipeline admission differs from its selected storage qualification or hardware")


def _bounded_kernel(kernel, max_calls):
    if type(kernel) is not Kernel:
        raise ValueError("copy pipeline requires a typed source or realized kernel")
    if type(max_calls) is not int or max_calls <= 0:
        raise ValueError("copy pipeline requires a positive call budget")
    if len(kernel.body) > max_calls or any(type(call) is not Call for call in kernel.body):
        raise StorageOptimizationUnavailable("copy pipeline requires bounded straight-line calls")


def _snapshot(kernel, objects, banks, adapter, admission, policy, model):
    _check_selection(adapter, admission, policy)
    _bounded_kernel(kernel, model["max_calls"])
    return (
        kernel.digest(),
        deepcopy((objects, banks, admission, policy, model)),
        (
            adapter.project,
            adapter.rewrite,
            adapter.configuration_identity,
            adapter.numerical_policy_identity,
            adapter.qualification_identity,
        ),
    )


def _source(kernel, objects, banks, adapter, admission, policy, model):
    _check_selection(adapter, admission, policy)
    bank_map, effects, *_ = _problem(
        kernel,
        objects,
        banks,
        adapter,
        model["external_windows"],
        model["private_ownership"],
        model["max_calls"],
        model["max_objects"],
    )
    # Actual fixed slots must fit. Virtual, pre-allocation addresses cannot
    # establish double-buffer independence or reserve physical storage.
    verify_temporary_storage(kernel, kernel, objects, banks, adapter, **model)
    _resource_trace(kernel.body, effects, bank_map, admission, policy)
    return bank_map, effects


def verify_copy_pipeline(
    original,
    realized,
    permutation,
    objects,
    banks,
    adapter,
    admission,
    *,
    policy=CopyPipelinePolicy(),
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
):
    """Check actual instructions, every inverted dependence and resource trace.

    This does not trust a greedy search, proposed buffer plan or move report.
    It also does not establish that emitted target code has these effects:
    final decoding/binding remains the original target execution owner's job.
    """
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=max_calls,
        max_objects=max_objects,
    )
    snapshot = _snapshot(original, objects, banks, adapter, admission, policy, model)
    _bounded_kernel(realized, max_calls)
    realized_digest = realized.digest()
    bank_map, effects = _source(original, objects, banks, adapter, admission, policy, model)
    if (
        type(permutation) is not tuple
        or any(type(i) is not int for i in permutation)
        or sorted(permutation) != list(range(len(original.body)))
        or (original.name, original.args, original.attrs) != (realized.name, realized.args, realized.attrs)
        or realized.body != tuple(original.body[i] for i in permutation)
    ):
        raise ValueError("copy pipeline changed original call membership or kernel boundary")
    actual = tuple(adapter.project(call) for call in realized.body)
    if actual != tuple(effects[i] for i in permutation):
        raise ValueError("copy pipeline changed independently decoded instruction effects")
    launches = {i for i, effect in enumerate(effects) if effect.completion == "asynchronous"}
    # Never reorder two launches, two waits, setup, descriptors or arithmetic.
    for selected in (launches, set(range(len(effects))) - launches):
        if [i for i in permutation if i in selected] != sorted(selected):
            raise ValueError("copy pipeline reordered launches or nonlaunch instructions")
    positions = {source: index for index, source in enumerate(permutation)}
    moves = []
    for source in sorted(launches):
        crossed = [i for i in range(source) if positions[i] > positions[source]]
        if positions[source] > source:
            raise ValueError("copy pipeline delayed an original launch")
        if not crossed:
            continue
        if (
            len(crossed) > policy.distance
            or len(moves) >= policy.max_moves
            or not all(
                _may_cross(original.body[source], original.body[i], effects[source], effects[i], bank_map, admission)
                for i in crossed
            )
        ):
            raise ValueError("copy pipeline has an unqualified inversion or exceeded its motion budget")
        moves.append(dict(source_index=source, candidate_index=positions[source], crossed_source_indices=crossed))
    maximum = _resource_trace(realized.body, actual, bank_map, admission, policy)
    storage = verify_temporary_storage(realized, realized, objects, banks, adapter, **model)
    if (
        snapshot != _snapshot(original, objects, banks, adapter, admission, policy, model)
        or realized.digest() != realized_digest
    ):
        raise ValueError("copy pipeline source, selected contracts or realized program changed during checking")
    return {
        "schema": "merlin.checked_copy_pipeline.v1",
        "status": "applied" if moves else "unchanged",
        "moves": moves,
        "original_positions": list(permutation),
        "maximum_pending_copies": maximum,
        "policy": asdict(policy),
        "admission": asdict(admission),
        "storage": storage,
        "source_sha256": original.digest(),
        "candidate_sha256": realized.digest(),
        "static_commands": len(realized.body),
        "removed_completion_instructions": 0,
        "final_emitted_program_check": "required from original qualified target owner",
        "timing": "UNKNOWN",
    }


def pipeline_private_copies(
    kernel,
    objects,
    banks,
    adapter,
    admission,
    *,
    policy=CopyPipelinePolicy(),
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
):
    """Hoist existing input-copy launches into qualified fixed private slots.

    A two-slot source and max_pending_copies=2 permit bounded double buffering;
    no storage, async instruction or wait is invented. Unknown state, missing
    pair qualification and capacity exhaustion stop motion conservatively.
    """
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=max_calls,
        max_objects=max_objects,
    )
    snapshot = _snapshot(kernel, objects, banks, adapter, admission, policy, model)
    bank_map, effects = _source(kernel, objects, banks, adapter, admission, policy, model)
    order, moved = list(range(len(kernel.body))), 0
    for source, effect in enumerate(effects):
        if moved == policy.max_moves:
            break
        if not _input_copy(effect, bank_map):
            continue
        position, steps = order.index(source), 0
        while position > 0 and steps < policy.distance:
            previous = order[position - 1]
            if not _may_cross(
                kernel.body[source], kernel.body[previous], effect, effects[previous], bank_map, admission
            ):
                break
            proposed = order.copy()
            proposed[position - 1], proposed[position] = proposed[position], proposed[position - 1]
            try:
                _resource_trace(
                    tuple(kernel.body[i] for i in proposed),
                    tuple(effects[i] for i in proposed),
                    bank_map,
                    admission,
                    policy,
                )
            except ValueError:
                break
            order = proposed
            position -= 1
            steps += 1
        moved += bool(steps)
    candidate = replace(kernel, body=tuple(kernel.body[i] for i in order))
    report = verify_copy_pipeline(
        kernel,
        candidate,
        tuple(order),
        objects,
        banks,
        adapter,
        admission,
        policy=policy,
        **model,
    )
    if snapshot != _snapshot(kernel, objects, banks, adapter, admission, policy, model):
        raise ValueError("copy pipeline source or selected contracts changed during scheduling")
    return candidate, report
