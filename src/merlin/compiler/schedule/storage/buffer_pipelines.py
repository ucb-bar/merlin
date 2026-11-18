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

from merlin.compiler.kernel_ir.ir import (
    Call,
    Const,
    Expr,
    Kernel,
    Loop,
    Var,
    check_integer_domain,
    check_structure,
    select,
)
from merlin.compiler.kernel_ir.primitives import Cursor, _subst_call, divide_loop
from merlin.compiler.schedule.storage import temporaries as _storage
from merlin.compiler.schedule.storage.temporaries import (
    StorageAdapter,
    StorageOptimizationUnavailable,
    TemporaryObject,
    _completion,
    _overlap,
    _problem,
    allocate_temporary_storage,
    verify_temporary_storage,
)
from merlin.target.isa.spec import resolve_fact_integer

# ---- copy_pipeline -----------------------------------------------------------------------------


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


class _CopyPipelineOwner:
    """Original admission and implementation, never restored from a record."""

    def __init__(self, admission, policy, budgets):
        self.admission, self.policy, self.budgets = admission, policy, budgets
        self._original = deepcopy((admission, policy, budgets))
        self._reader = self._functions
        self._reader_code = self._reader.__code__
        self._callbacks = self._reader()

    @staticmethod
    def _functions():
        return (
            ((_storage, None),)
            + tuple(
                (value, value.__code__)
                for value in vars(_storage).values()
                if getattr(value, "__module__", None) == _storage.__name__ and hasattr(value, "__code__")
            )
            + tuple(
                (value, value.__code__)
                for value in globals().values()
                if getattr(value, "__module__", None) == __name__ and hasattr(value, "__code__")
            )
            + tuple(
                (value, value.__code__)
                for value in (
                    _CopyPipelineOwner.__init__,
                    _CopyPipelineOwner._functions,
                    _CopyPipelineOwner.revalidate,
                    _CopyPipelineOwner.record,
                    _CopyPipelineOwner.options,
                    _CopyPipelineOwner.propose,
                    _CopyPipelineOwner.verify,
                    CopyPipelineAdmission.__post_init__,
                    CopyPipelinePolicy.__post_init__,
                    _storage.TemporaryObject.__post_init__,
                    _storage.StorageEffects.__post_init__,
                    _storage.StorageAdapter.__post_init__,
                    _problem,
                    _completion,
                    _overlap,
                    verify_temporary_storage,
                )
            )
        )

    def revalidate(self):
        if self._reader is not _CopyPipelineOwner._functions or self._reader.__code__ is not self._reader_code:
            raise ValueError("original copy pipeline implementation reader changed")
        current = self._reader()
        if len(current) != len(self._callbacks) or any(
            a is not b or ac is not bc for (a, ac), (b, bc) in zip(current, self._callbacks, strict=True)
        ):
            raise ValueError("original copy pipeline implementation changed")
        if (self.admission, self.policy, self.budgets) != self._original:
            raise ValueError("original copy pipeline admission or policy changed")

    def record(self):
        self.revalidate()
        return {
            "admission": asdict(self.admission),
            "policy": asdict(self.policy),
            "budgets": asdict(self.budgets),
            "scope": "qualified copy motion, no timing or hardware qualification supplied",
        }

    def options(self, windows, ownership):
        self.revalidate()
        return dict(
            policy=self.policy,
            external_windows=windows,
            private_ownership=ownership,
            max_calls=self.budgets.max_calls,
            max_objects=self.budgets.max_objects,
        )

    def propose(self, source, objects, banks, adapter, windows, ownership):
        self.revalidate()
        try:
            actual, proof = pipeline_private_copies(
                source, objects, banks, adapter, self.admission, **self.options(windows, ownership)
            )
        except StorageOptimizationUnavailable:
            self.revalidate()
            return None
        self.revalidate()
        return actual, tuple(proof["original_positions"])

    def verify(self, source, actual, positions, objects, banks, adapter, windows, ownership):
        result = verify_copy_pipeline(
            source, actual, positions, objects, banks, adapter, self.admission, **self.options(windows, ownership)
        )
        self.revalidate()
        return result


def copy_pipeline_motion(admission, *, policy=CopyPipelinePolicy(), budgets=None):
    """Bind one original qualified copy pass to the ordinary fixed storage stage.

    This V8 companion adds no import permission to historical SDK inventories.
    Unknown admission leaves the exact original compilation path available.
    """
    from pathlib import Path

    from merlin.compiler.schedule.pipeline.contracts import CopyPipelineMotion, ScheduleBudgets

    if admission is None:
        return None
    if type(admission) is not CopyPipelineAdmission or type(policy) is not CopyPipelinePolicy:
        raise ValueError("copy pipeline needs typed original admission and policy")
    admission.__post_init__()
    policy.__post_init__()
    budgets = ScheduleBudgets() if budgets is None else budgets
    if type(budgets) is not ScheduleBudgets:
        raise ValueError("copy pipeline needs original typed schedule budgets")
    budgets.__post_init__()
    owner = _CopyPipelineOwner(admission, policy, budgets)
    return CopyPipelineMotion(
        owner.propose, owner.verify, owner.revalidate, owner.record, (Path(__file__), Path(_storage.__file__))
    )


# ---- operand_slots -----------------------------------------------------------------------------
# Consume dependency-derived operand storage in a bounded counted pipeline.
#
# Operands group existing private backing objects into immutable generations.
# Actual decoded initialization, reads and completion commands determine their
# live intervals; the ordinary temporary allocator assigns physical addresses.
# Only a verified cyclic affine rotation is reconstructed as counted phase loops. No
# latency, asynchronous overlap permission or target completion is inferred here.


@dataclass(frozen=True)
class PipelineOperand:
    """A source operand's invocation-private, immutable backing generations.

    These are existing TemporaryObjects, not requested buffer depths. One
    generation can serve every iteration; distinct advancing generations may
    share physical storage only after their actual asynchronous users finish.
    """

    name: str
    generations: tuple[TemporaryObject, ...]

    def __post_init__(self):
        if (
            type(self.name) is not str
            or not self.name
            or type(self.generations) is not tuple
            or not self.generations
            or any(type(obj) is not TemporaryObject for obj in self.generations)
        ):
            raise ValueError("pipeline operands require named existing private generations")


@dataclass(frozen=True)
class PipelineSlotResult:
    kernel: Kernel
    report: dict


def _shape(kernel, max_calls, *, source):
    """Check the entire static shape and dynamic budget before closing calls."""
    if type(max_calls) is not int or max_calls <= 0:
        raise ValueError("pipeline expansion requires an explicit positive call budget")
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("pipeline slots require a structurally valid kernel")
    if kernel.attrs:
        raise StorageOptimizationUnavailable("pipeline slots cannot interpret kernel attributes")
    count, loops = 0, []

    def visit(body, multiplier, depth):
        nonlocal count
        for stmt in body:
            if type(stmt) is Loop:
                if depth >= (1 if source else 2) or stmt.iter_args or stmt.yields or stmt.results:
                    raise StorageOptimizationUnavailable(
                        "pipeline slots do not transport nested or SSA-carried asynchronous boundaries"
                    )
                loops.append(stmt)
                visit(stmt.body, multiplier * stmt.extent, depth + 1)
            elif type(stmt) is Call:
                if (
                    stmt.unit
                    or stmt.produces
                    or stmt.awaits
                    or stmt.stages
                    or stmt.sets
                    or stmt.assumes
                    or any(not isinstance(value, Expr) for _name, value in stmt.args)
                ):
                    raise StorageOptimizationUnavailable(
                        "pipeline slots require integer commands with actual decoded completion operands"
                    )
                count += multiplier
                if count > max_calls:
                    raise StorageOptimizationUnavailable("pipeline effect expansion exceeds its call budget")
            else:
                raise StorageOptimizationUnavailable("pipeline slots encountered an unknown statement")

    visit(kernel.body, 1, 0)
    if source and len(loops) != 1:
        raise StorageOptimizationUnavailable("pipeline slots support exactly one source counted loop")
    check_integer_domain(kernel, 64)
    return count


def _close(kernel):
    """Called only after the complete static shape and dynamic budget check."""
    calls = []

    def visit(body, bindings):
        for stmt in body:
            if type(stmt) is Loop:
                for index in range(stmt.extent):
                    visit(stmt.body, (*bindings, (stmt.var, index)))
            else:
                for name, value in bindings:
                    stmt = _subst_call(stmt, name, Const(value))
                calls.append(stmt)

    visit(kernel.body, ())
    return replace(kernel, body=tuple(calls))


def _objects(operands):
    if (
        type(operands) is not tuple
        or not operands
        or any(type(operand) is not PipelineOperand for operand in operands)
        or len({operand.name for operand in operands}) != len(operands)
    ):
        raise ValueError("pipeline operands require unique explicit roles")
    objects = tuple(obj for operand in operands for obj in operand.generations)
    if len({obj.name for obj in objects}) != len(objects):
        raise ValueError("each backing generation must belong to exactly one operand")
    return objects


def _requirements(original, operands, banks, adapter, model):
    objects = _objects(operands)
    bank_map, effects, bindings, intervals, *_rest = _problem(
        original,
        objects,
        banks,
        adapter,
        model["external_windows"],
        model["private_ownership"],
        model["max_calls"],
        model["max_objects"],
    )
    for index, (effect, row) in enumerate(zip(effects, bindings, strict=True)):
        for access, binding in zip(effect.accesses, row, strict=True):
            if binding is not None and access.writes and index != intervals[binding[0]][0]:
                raise StorageOptimizationUnavailable("an operand generation is modified after initialization")
    result = {}
    for operand in operands:
        if len({bank_map[obj.bank].backing for obj in operand.generations}) != 1:
            raise ValueError("one operand's generations must use the same physical backing")
        names = {obj.name for obj in operand.generations}
        # Closed issue intervals are intentional: objects born in a completing
        # command cannot alias its still-live inputs. No service-cycle estimate.
        events = {}
        for name in names:
            first, last = intervals[name]
            events[first] = events.get(first, 0) + 1
            events[last + 1] = events.get(last + 1, 0) - 1
        active = peak = 0
        for index in sorted(events):
            active += events[index]
            peak = max(peak, active)
        result[operand.name] = {
            "generations": [obj.name for obj in operand.generations],
            "required_live_generations": peak,
            "lifetime_calls": {obj.name: intervals[obj.name] for obj in operand.generations},
        }
    return result


def _cyclic_affine(values):
    """One affine progression modulo equally spaced allocated slot bases."""
    bases = sorted(set(values))
    if len(bases) == 1:
        return 0
    stride = bases[1] - bases[0]
    if any(right - left != stride for left, right in zip(bases, bases[1:])):
        return None
    span = stride * len(bases)
    step = values[1] - values[0]
    if any(value != bases[0] + (values[0] - bases[0] + index * step) % span for index, value in enumerate(values)):
        return None
    return step


def _phase_address(values, phase):
    step = _cyclic_affine(values)
    assert step is not None
    result = Const(values[0]) + Var(phase) * step
    # The small index language has no modulo. The checked, bounded phase loop
    # supplies its finite domain; exact wrap corrections remain ordinary selects.
    for index, value in enumerate(values):
        if value != values[0] + index * step:
            result = select(phase, index, value, result)
    return result


def _rotation(kernel, flat_source, flat_actual, max_period):
    """Find a repeated affine phase, then reuse the ordinary loop divider.

    Only relocated operands change. All original indices/tail predicates go
    through divide_loop's checked quotient/remainder substitution. A bounded
    final partial period is peeled using the exact already-checked calls.
    Irregular assignments keep the caller's original route.
    """
    position, loop = next((i, stmt) for i, stmt in enumerate(kernel.body) if type(stmt) is Loop)
    prefix, width, extent = position, len(loop.body), loop.extent
    changed = {}
    for column, call in enumerate(loop.body):
        for name, _value in call.args:
            before = [flat_source.body[prefix + i * width + column].arg(name) for i in range(extent)]
            after = [flat_actual.body[prefix + i * width + column].arg(name) for i in range(extent)]
            if after != before:
                if any(type(value) is not Const for value in after):
                    raise StorageOptimizationUnavailable("relocated pipeline operands must close to integer addresses")
                changed[column, name] = [value.value for value in after]

    def matches(period, whole):
        return all(
            all(values[i] == values[i % period] for i in range(whole)) and _cyclic_affine(values[:period]) is not None
            for values in changed.values()
        )

    selected = None
    for peel in (0, 1):
        for period in range(1, min(max_period, extent) + 1):
            whole = (extent - peel) // period * period
            if whole and (period == 1 or whole >= 2 * period) and matches(period, whole):
                selected = period, whole
                break
        if selected is not None:
            break
    if selected is None:
        raise StorageOptimizationUnavailable("allocation has no supported repeated affine operand rotation")
    period, whole = selected
    shortened = replace(
        kernel, body=(*kernel.body[:position], replace(loop, extent=whole), *kernel.body[position + 1 :])
    )
    if period == 1:
        divided = shortened
        phase = None
        body = loop.body
    else:
        divided = divide_loop(shortened, Cursor((loop.var,)), period)
        outer = divided.body[position]
        phase = outer.body[0].var
        body = outer.body[0].body
    rewritten = []
    for column, call in enumerate(body):
        args = []
        for name, value in call.args:
            values = changed.get((column, name))
            if values is not None:
                value = Const(values[0])
                if period > 1:
                    value = _phase_address(values[:period], phase)
            args.append((name, value))
        rewritten.append(replace(call, args=tuple(args)))
    if period == 1:
        counted = replace(divided.body[position], body=tuple(rewritten))
    else:
        outer = divided.body[position]
        counted = replace(outer, body=(replace(outer.body[0], body=tuple(rewritten)),))
    end = prefix + extent * width
    actual = replace(
        kernel,
        body=(
            *flat_actual.body[:prefix],
            counted,
            *flat_actual.body[prefix + whole * width : end],
            *flat_actual.body[end:],
        ),
    )
    return actual, {"phase_extent": period, "peeled_iterations": extent - whole}


def verify_pipeline_operand_slots(
    original,
    realized,
    operands,
    banks,
    adapter,
    *,
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
) -> dict:
    """Independently recover live generations/addresses from emitted calls."""
    _shape(original, max_calls, source=True)
    _shape(realized, max_calls, source=False)
    source, actual = _close(original), _close(realized)
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=max_calls,
        max_objects=max_objects,
    )
    requirements = _requirements(source, operands, banks, adapter, model)
    storage = verify_temporary_storage(source, actual, _objects(operands), banks, adapter, **model)
    for operand in operands:
        addresses = {obj.name: storage["addresses"][obj.name] for obj in operand.generations}
        requirements[operand.name]["addresses"] = addresses
        # Distinct bases can exceed the peak-live lower bound with different
        # extents/alignment or shared-bank fragmentation; do not call it optimal.
        requirements[operand.name]["assigned_distinct_bases"] = len(set(addresses.values()))
    return {
        "schema": "merlin.pipeline_operand_slots.v1",
        "source_sha256": original.digest(),
        "realized_sha256": realized.digest(),
        "operands": requirements,
        "storage": storage,
        "command_order": "preserved_with_actual_completion_commands",
        "machine_emission": "not_established",
        "execution_qualification": "not_established",
        "timing": "UNKNOWN",
    }


def allocate_pipeline_operand_slots(
    kernel,
    operands,
    banks,
    adapter,
    *,
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_period=8,
    timeout_ms=5000,
) -> PipelineSlotResult | None:
    """Assign existing immutable generations, consume a checked counted rotation.

    Input is a prologue, exactly one nonnested loop of integer commands, and an
    epilogue. Async dependencies use actual encoded launch/wait operands decoded
    by StorageAdapter; SSA-carried/nested async boundaries currently refuse.
    The caller supplies an already legal issue order and qualified target laws.
    This transform neither moves commands nor creates overlap authorization.
    Unknown/irregular effects diagnose conservatively; infeasible storage gives
    None. Expansion is checked before closing any dynamic command. Numerical
    source qualification and actual native/target execution remain independent.
    """
    _shape(kernel, max_calls, source=True)
    if type(max_period) is not int or not 1 <= max_period <= 64:
        raise ValueError("pipeline rotation requires a positive bounded phase-search limit")
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=max_calls,
        max_objects=max_objects,
    )
    flat = _close(kernel)
    _requirements(flat, operands, banks, adapter, model)
    allocated = allocate_temporary_storage(
        flat, _objects(operands), banks, adapter, timeout_ms=timeout_ms, eliminate_copies=False, **model
    )
    if allocated is None:
        return None
    actual, rotation = _rotation(kernel, flat, allocated.kernel, max_period)
    _shape(actual, max_calls, source=False)
    if _close(actual) != allocated.kernel:
        raise ValueError("counted rotation changed the allocated command program")
    report = verify_pipeline_operand_slots(kernel, actual, operands, banks, adapter, **model)
    return PipelineSlotResult(actual, {**report, "rotation": rotation})
