"""Complete-domain row normalization in the existing counted schedule IR.

The source selects the rounded exponential, lane accumulation order, final
binary tree and normalization operation. Hardware selects chunk limits and
private, addressable storage. A chunk never owns a denominator. Unknown costs
select the largest legal chunk; this is a command-count policy, not a cycle
estimate. Encoders and actual-operand projectors require independent qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import lcm

from merlin.compiler.kernel_ir.ir import (
    DTYPE_BYTES,
    Call,
    Const,
    Expr,
    Kernel,
    Loop,
    Ptr,
    Scaled,
    Sum,
    TensorArg,
    Var,
    check_structure,
)
from merlin.compiler.schedule.dense.compact import DenseMemoryView
from merlin.compiler.semantic.allocation_types import CandidateGraph, Reservation, StorageBank, Value
from merlin.compiler.semantic.rules import AddressConstraint
from merlin.compiler.semantic.search.allocation import allocate, check_assignment

# ---- row_softmax -------------------------------------------------------------------------------

ROLES = ("maximum", "exponentials", "lanes", "normalizer")


LIFETIMES = ((0, 2), (2, 6), (3, 5), (5, 6))


PHASE_ROLES = {
    "init_maximum": (0,),
    "maximum": (0,),
    "exponential": (0, 1),
    "init_lanes": (2,),
    "sum": (1, 2),
    "finish": (2, 3),
    "normalize": (1, 3),
}


class RowSoftmaxUnavailable(ValueError):
    """Preserve the existing source route; this family cannot realize it."""


@dataclass(frozen=True)
class RowSoftmaxNumerics:
    """An explicit source algorithm, not an accuracy or hardware certificate.

    Lane j starts with exponential j, then adds j+lanes, j+2*lanes, ... .
    First assignment is not an addition to zero. The tree fixes every rounded
    addition. Finite unmasked scores, RNE, nontrapping operations and unobserved
    floating status are prerequisites of this family. The exponential accepts
    nonpositive values including negative infinity and returns exact zero for it.
    """

    source_policy: str
    exponential: str
    dtype: str
    lanes: int
    tree: int | tuple
    normalization: str
    finite_inputs: bool
    unobserved_status_rne: bool

    def __post_init__(self):
        if any(not isinstance(s, str) or not s for s in (self.source_policy, self.exponential)):
            raise ValueError("row softmax needs explicit source policy and exponential identities")
        if self.dtype not in ("f32", "f64") or type(self.lanes) is not int or not 1 <= self.lanes <= 256:
            raise ValueError("row softmax requires a bounded explicit floating lane algorithm")
        leaves = []
        visited = 0

        def visit(node, depth=0):
            nonlocal visited
            visited += 1
            if visited > 511:
                raise ValueError("source sum tree exceeds its structural budget")
            if depth > 256:
                raise ValueError("source sum tree exceeds its structural budget")
            if type(node) is int:
                leaves.append(node)
            elif type(node) is tuple and len(node) == 2:
                for child in node:
                    visit(child, depth + 1)
            else:
                raise ValueError("source sum tree requires ordered binary additions")

        visit(self.tree)
        if sorted(leaves) != list(range(self.lanes)):
            raise ValueError("source sum tree must cover each lane exactly once")
        if self.normalization not in ("reciprocal_multiply", "divide"):
            raise ValueError("unknown source normalization operation")
        if type(self.finite_inputs) is not bool or type(self.unobserved_status_rne) is not bool:
            raise ValueError("source numerical preconditions must be explicit booleans")

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class CausalRowMask:
    """Column c is active iff c <= row % period + offset; every row is nonempty."""

    period: int
    offset: int = 0

    def __post_init__(self):
        if type(self.period) is not int or self.period <= 0 or type(self.offset) is not int or self.offset < 0:
            raise ValueError("causal mask requires a positive period and nonnegative offset")


@dataclass(frozen=True)
class RowSoftmaxProblem:
    rows: int
    columns: int
    mask: CausalRowMask | None = None

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.rows, self.columns)):
            raise ValueError("row softmax requires nonempty static rows and complete columns")
        if self.mask is not None and type(self.mask) is not CausalRowMask:
            raise ValueError("unknown masks require the original source route")


@dataclass(frozen=True)
class RowSoftmaxResources:
    max_chunk_elements: int
    granularity: int
    partial_chunks: bool
    max_static_calls: int = 16

    def __post_init__(self):
        if any(type(n) is not int or n <= 0 for n in (self.max_chunk_elements, self.granularity)):
            raise ValueError("row softmax requires explicit positive chunk resources")
        if type(self.partial_chunks) is not bool or type(self.max_static_calls) is not int or self.max_static_calls < 0:
            raise ValueError("row softmax requires explicit tail support and static call budget")


@dataclass(frozen=True)
class RowSoftmaxStep:
    """Public synchronous primitive semantics, with actual physical byte operands.

    maximum scans source scores in increasing column order from the retained
    -infinity seed. exponential uses the completed row maximum and writes each
    shifted exponential once. sum resumes the globally indexed source lanes;
    it never resets at a chunk boundary. finish uses the complete source tree,
    storing either reciprocal or denominator as selected. normalize reads that
    single complete-row result. Masked scores are -infinity in both score passes.
    No other effects, reassociation, hidden scratch or state are permitted.

    Payload effects are part of this contract: maximum reads count inputs and
    reads/writes its scalar once; exponential reads count inputs and one maximum
    and writes count scratch elements; sum reads count scratch and reads/writes
    all source lanes once; finish reads all lanes and writes one factor;
    normalize reads count scratch and one factor and writes count outputs.
    Initializers only write their scalar/all lanes. Masked input positions are
    still requested. These are logical requests, not bus or register traffic.
    """

    kind: str
    row: Expr
    column: Expr
    count: int
    local: tuple[tuple[str, Expr], ...]
    source: Ptr | None
    output: Ptr | None
    numerical_identity: str
    mask: CausalRowMask | None


@dataclass(frozen=True)
class RowSoftmaxAdapter:
    encode: Callable[[RowSoftmaxStep], Call]
    project: Callable[[Call], RowSoftmaxStep]
    configuration_identity: str
    numerical_identity: str
    qualification_identity: str
    synchronous_completion: bool
    exact_source_order: bool
    addressable_state: bool
    index_bits: int

    def __post_init__(self):
        if (
            not callable(self.encode)
            or not callable(self.project)
            or any(
                not isinstance(s, str) or not s
                for s in (self.configuration_identity, self.numerical_identity, self.qualification_identity)
            )
        ):
            raise ValueError("row softmax needs selected qualified encode/project contracts")
        if any(
            type(x) is not bool for x in (self.synchronous_completion, self.exact_source_order, self.addressable_state)
        ):
            raise ValueError("row softmax completion, source order and state contracts must be Boolean")
        if type(self.index_bits) is not int or self.index_bits not in (32, 64):
            raise ValueError("row softmax needs explicit 32 or 64-bit index arithmetic")


def _admit(problem, numerics, resources, adapter, private_ownership, disjoint_arguments):
    if (
        type(problem) is not RowSoftmaxProblem
        or type(numerics) is not RowSoftmaxNumerics
        or type(resources) is not RowSoftmaxResources
        or type(adapter) is not RowSoftmaxAdapter
    ):
        raise ValueError("row softmax needs typed current source, resources and primitive contracts")
    if type(private_ownership) is not bool or type(disjoint_arguments) is not bool:
        raise ValueError("row softmax ownership declarations must be Boolean")
    if not private_ownership or not disjoint_arguments:
        raise RowSoftmaxUnavailable(
            "row softmax requires private workspace and disjoint immutable input/output bindings"
        )
    if not numerics.finite_inputs or not numerics.unobserved_status_rne:
        raise RowSoftmaxUnavailable("row softmax requires finite inputs and the explicit RNE/status numerical contract")
    if problem.columns < numerics.lanes:
        raise RowSoftmaxUnavailable("source lane initialization requires at least one source element per lane")
    if not all((adapter.synchronous_completion, adapter.exact_source_order, adapter.addressable_state)):
        raise RowSoftmaxUnavailable("row softmax lacks exact source order, synchronous completion or addressable state")
    if adapter.numerical_identity != numerics.identity:
        raise ValueError("row softmax adapter is stale for the selected numerical algorithm")
    if problem.rows * problem.columns * DTYPE_BYTES[numerics.dtype] > (1 << (adapter.index_bits - 1)) - 1:
        raise RowSoftmaxUnavailable("complete row source/output address range exceeds selected index arithmetic")
    if problem.mask and max(problem.mask.period, problem.mask.offset + problem.mask.period) >= 1 << (
        adapter.index_bits - 1
    ):
        raise RowSoftmaxUnavailable("causal mask arithmetic exceeds selected index representation")


def _storage_problem(problem, numerics, banks, role_banks):
    if type(banks) is not tuple or not banks or any(type(b) is not StorageBank or b.unit != "byte" for b in banks):
        raise ValueError("row softmax requires explicit byte-addressed banks")
    by_name = {b.name: b for b in banks}
    if (
        len(by_name) != len(banks)
        or type(role_banks) is not tuple
        or len(role_banks) != 4
        or any(b not in by_name for b in role_banks)
    ):
        raise ValueError("each row state role must select an existing bank")
    width = DTYPE_BYTES[numerics.dtype]
    counts = (1, problem.columns, numerics.lanes, 1)
    values, order, ids = [], [], {}
    for phase in range(7):
        for role, (start, _end) in enumerate(LIFETIMES):
            if start == phase:
                node = len(values)
                ids[role] = node
                values.append(
                    Value(
                        node,
                        ROLES[role],
                        role_banks[role],
                        counts[role] * width,
                        (),
                        ROLES[role],
                        "instruction",
                        (AddressConstraint("aligned", "out", value=width),),
                    )
                )
                order.append(node)
        node = len(values)
        values.append(
            Value(
                node,
                "completion",
                banks[0].name,
                0,
                tuple(ids[r] for r, (_s, end) in enumerate(LIFETIMES) if end == phase),
                None,
                "event",
            )
        )
        order.append(node)
    return CandidateGraph(tuple(values), ()), tuple(order), ids, counts, by_name


def allocate_row_softmax_storage(problem, numerics, banks, role_banks, *, reservations=(), timeout_ms=2000):
    """Use Merlin's allocator over inclusive completed-phase lifetimes.

    Exponential storage holds one full row and is reused only after synchronous
    normalization finishes. Maximum/normalizer may share physical bytes; lanes
    and normalizer are simultaneously live during finish. Reservations include
    packing scratch/DMA staging and apply across aliased bank views.
    """
    if type(reservations) is not tuple or any(type(r) is not Reservation for r in reservations):
        raise ValueError("row softmax reservations must be explicit immutable intervals")
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("row softmax allocation needs a positive solver budget")
    graph, order, ids, counts, by_name = _storage_problem(problem, numerics, banks, role_banks)
    result = allocate(graph, order, banks, reservations=reservations, timeout_ms=timeout_ms)
    if result.status != "feasible":
        raise RowSoftmaxUnavailable("complete row intermediate storage cannot be allocated: " + result.reason)
    valid, reason = check_assignment(graph, order, result.addresses, banks, reservations=reservations)
    if not valid:
        raise ValueError("invalid row state allocation: " + reason)
    width = DTYPE_BYTES[numerics.dtype]
    return tuple(
        DenseMemoryView(
            ROLES[r],
            by_name[role_banks[r]].backing,
            width,
            result.addresses[ids[r]] // width,
            counts[r],
            lcm(width, by_name[role_banks[r]].alignment) // width,
        )
        for r in range(4)
    )


def verify_row_softmax_storage(problem, numerics, storage, banks, role_banks, reservations=()):
    graph, order, ids, counts, by_name = _storage_problem(problem, numerics, banks, role_banks)
    if type(storage) is not tuple or len(storage) != 4 or any(type(v) is not DenseMemoryView for v in storage):
        raise ValueError("row softmax needs four assigned physical state views")
    width = DTYPE_BYTES[numerics.dtype]
    addresses = {v.id: 0 for v in graph.values if v.kind == "event"}
    for role, view in enumerate(storage):
        bank = by_name[role_banks[role]]
        if (
            view.name != ROLES[role]
            or view.backing != bank.backing
            or view.unit_bytes != width
            or view.capacity < counts[role]
        ):
            raise ValueError("row state view has wrong backing, element width or capacity")
        if view.base * width < bank.start or (view.base + view.capacity) * width > bank.capacity:
            raise ValueError("row state view extends beyond its selected bank")
        addresses[ids[role]] = view.base * width
    valid, reason = check_assignment(graph, order, addresses, banks, reservations=reservations)
    if not valid:
        raise ValueError("invalid completed row state ownership: " + reason)


def _step(kind, row, column, count, problem, numerics, storage):
    width = DTYPE_BYTES[numerics.dtype]
    local = tuple(
        (storage[r].backing, Const(storage[r].base * width) + (column * width if r == 1 else Const(0)))
        for r in PHASE_ROLES[kind]
    )
    pointer = Ptr("input", (row * problem.columns + column) * width) if kind in ("maximum", "exponential") else None
    output = Ptr("output", (row * problem.columns + column) * width) if kind == "normalize" else None
    return RowSoftmaxStep(kind, row, column, count, local, pointer, output, numerics.identity, problem.mask)


@dataclass(frozen=True)
class RowSoftmaxCompilation:
    kernel: Kernel
    storage: tuple[DenseMemoryView, ...]
    report: dict
    index_bits: int
    passes: tuple = ()
    selected_contracts: object = None
    checked_index_bits: int | None = None
    checked_selection: str | None = None

    def revalidate(self):
        from merlin.compiler.schedule.pipeline.selected_contracts import SelectedSchedulingContracts
        from merlin.compiler.schedule.pipeline.stages import shared_library_sources

        if self.selected_contracts is not None:
            if type(self.selected_contracts) is not SelectedSchedulingContracts:
                raise ValueError("row selected RTL/software contracts changed; reselect and replan")
            self.selected_contracts.revalidate()
        selection = (
            None
            if self.selected_contracts is None
            else hashlib.sha256(json.dumps(self.selected_contracts.record(), sort_keys=True).encode()).hexdigest()
        )
        if selection != self.checked_selection:
            raise ValueError("row selected RTL/software contracts changed; reselect and replan")
        # Mutable diagnostic dictionaries grant no analysis custody.
        stamp = self.passes[-1].after if self.passes else None
        library = hashlib.sha256(json.dumps(shared_library_sources(), sort_keys=True).encode()).hexdigest()
        if (
            stamp is None
            or self.kernel.digest() != stamp.source_sha256
            or type(self.index_bits) is not int
            or self.index_bits != self.checked_index_bits
            or stamp.context.execution is None
            or not stamp.context.execution.endswith(
                f":selection={selection or 'unselected'}:index_bits={self.index_bits}"
            )
            or library != stamp.context.library
        ):
            raise ValueError("row checked kernel or shared library changed; replan before emission")

    def emit_c(self, instruction_set, *, symbol):
        from merlin.compiler.kernel_ir.codegen import emit_c_function

        self.revalidate()
        source = emit_c_function(self.kernel, instruction_set, symbol=symbol, index_bits=self.index_bits)
        self.revalidate()
        return source


# ---- row_softmax_verification ------------------------------------------------------------------
# Reconstruct complete row reductions from actual projected counted calls.
#
# This checker admits one bounded grammar, independently of selected chunks:
# row loop; initialized full maximum; full exponentials; initialized source lanes;
# full ordered sum; completed tree; full normalization. Affine byte operands and
# constant chunk counts establish coverage without enumerating tensor elements.
# It proves the schedule under the qualified primitive contract, not that an
# arbitrary primitive implementation obeys that contract.


def _affine(expression):
    if isinstance(expression, Const):
        return {"": expression.value} if expression.value else {}
    if isinstance(expression, Var):
        return {expression.name: 1}
    if isinstance(expression, Scaled):
        return {
            key: value * expression.factor
            for key, value in _affine(expression.expr).items()
            if value * expression.factor
        }
    if isinstance(expression, Sum):
        result = {}
        for term in expression.terms:
            for key, value in _affine(term).items():
                result[key] = result.get(key, 0) + value
        return {key: value for key, value in result.items() if value}
    raise ValueError("row softmax operands require supported affine byte addressing")


def _equal(actual, expected):
    if _affine(actual) != _affine(expected):
        raise ValueError("row softmax actual address/coordinate does not cover its source domain")


def verify_row_softmax(
    kernel,
    problem,
    numerics,
    resources,
    storage,
    banks,
    role_banks,
    adapter,
    *,
    reservations=(),
    private_ownership=False,
    disjoint_arguments=False,
):
    """Validate actual phases, operands, domains, state lifetimes and requests."""
    _admit(problem, numerics, resources, adapter, private_ownership, disjoint_arguments)
    verify_row_softmax_storage(problem, numerics, storage, banks, role_banks, reservations)
    if check_structure(kernel) or kernel.attrs:
        raise ValueError("row softmax requires a valid kernel without unknown attributes")
    expected_args = tuple(
        TensorArg(name, (problem.rows, problem.columns), numerics.dtype, access)
        for name, access in (("input", "read"), ("output", "write"))
    )
    if kernel.args != expected_args or len(kernel.body) != 1 or type(kernel.body[0]) is not Loop:
        raise ValueError("row softmax requires complete disjoint source/output tensors and one row loop")
    outer = kernel.body[0]
    if outer.extent != problem.rows:
        raise ValueError("row softmax row loop omits or duplicates complete rows")
    row = Var(outer.var)
    body = outer.body
    index = 0
    static_calls = 0
    commands = reads = writes = 0
    phase_counts = {}
    width = DTYPE_BYTES[numerics.dtype]

    def projected(call, kind, column, count, instances):
        nonlocal static_calls, commands, reads, writes
        if (
            type(call) is not Call
            or call.unit
            or call.produces
            or call.awaits
            or call.stages
            or call.sets
            or call.assumes
        ):
            raise ValueError("unknown execution/completion/configuration effects are row reduction barriers")
        actual = adapter.project(call)
        if type(actual) is not RowSoftmaxStep or actual.kind != kind:
            raise ValueError("row softmax actual phase order is incomplete or reordered")
        if type(actual.count) is not int or actual.count != count or count <= 0:
            raise ValueError("row softmax primitive count changed its domain")
        if actual.numerical_identity != numerics.identity or actual.mask != problem.mask:
            raise ValueError("row softmax actual numerical algorithm or mask changed")
        _equal(actual.row, row)
        _equal(actual.column, column)
        if type(actual.local) is not tuple or len(actual.local) != len(PHASE_ROLES[kind]):
            raise ValueError("row softmax actual intermediate state roles changed")
        for (space, offset), role in zip(actual.local, PHASE_ROLES[kind], strict=True):
            view = storage[role]
            if space != view.backing:
                raise ValueError("row softmax intermediate uses the wrong physical backing")
            _equal(offset, Const(view.base * width) + (column * width if role == 1 else Const(0)))
        for pointer, name, present in (
            (actual.source, "input", kind in ("maximum", "exponential")),
            (actual.output, "output", kind == "normalize"),
        ):
            if present:
                if type(pointer) is not Ptr or pointer.tensor != name:
                    raise ValueError("row softmax actual external source/output binding changed")
                _equal(pointer.offset, (row * problem.columns + column) * width)
            elif pointer is not None:
                raise ValueError("row softmax has an undeclared external effect")
        static_calls += 1
        if static_calls > resources.max_static_calls:
            raise ValueError("row softmax exceeds its static command budget")
        multiplicity = instances * problem.rows
        commands += multiplicity
        phase_counts[kind] = phase_counts.get(kind, 0) + multiplicity
        # Logical requested payloads of the declared primitive, including state.
        # Physical/native register traffic and hidden effects are not inferred.
        lane = numerics.lanes
        r, w = {
            "init_maximum": (0, 1),
            "maximum": (count + 1, 1),
            "exponential": (count + 1, count),
            "init_lanes": (0, lane),
            "sum": (count + lane, lane),
            "finish": (lane, 1),
            "normalize": (count + 1, count),
        }[kind]
        reads += multiplicity * width * r
        writes += multiplicity * width * w

    def singleton(kind, count):
        nonlocal index
        if index >= len(body):
            raise ValueError("row softmax lost a required initialization/completion phase")
        projected(body[index], kind, Const(0), count, 1)
        index += 1

    def domain(kind):
        nonlocal index
        if index >= len(body):
            raise ValueError("row softmax lost a complete reduction domain")
        statement = body[index]
        covered = 0
        full_chunk = None
        if type(statement) is Loop:
            if len(statement.body) != 1 or type(statement.body[0]) is not Call:
                raise ValueError("row chunk requires one declared primitive with no hidden nested work")
            actual = adapter.project(statement.body[0])
            if type(actual) is not RowSoftmaxStep or type(actual.count) is not int or actual.count <= 0:
                raise ValueError("row chunk requires a positive actual constant extent")
            full_chunk = actual.count
            covered = statement.extent * full_chunk
            if (
                full_chunk > resources.max_chunk_elements
                or full_chunk % resources.granularity
                and (full_chunk != problem.columns or not resources.partial_chunks)
                or covered > problem.columns
            ):
                raise ValueError("row chunk violates capacity, granularity or complete domain")
            projected(statement.body[0], kind, Var(statement.var) * full_chunk, full_chunk, statement.extent)
            index += 1
        remaining = problem.columns - covered
        if remaining:
            if (
                not resources.partial_chunks
                or remaining > resources.max_chunk_elements
                or (full_chunk is not None and remaining >= full_chunk)
            ):
                raise ValueError("row softmax incomplete domain or unsupported tail")
            if index >= len(body):
                raise ValueError("row softmax lost its final reduction tail")
            projected(body[index], kind, Const(covered), remaining, 1)
            index += 1

    singleton("init_maximum", 1)
    domain("maximum")
    domain("exponential")
    singleton("init_lanes", numerics.lanes)
    domain("sum")
    singleton("finish", numerics.lanes)
    domain("normalize")
    if index != len(body):
        raise ValueError("row softmax has extra effects after its complete normalization")
    # The shared IR check bounds every emitted intermediate,
    # not just the final tensor byte extent. Exercise the same check before
    # publishing a scheduling receipt, including local addresses and loop ends.
    from merlin.compiler.kernel_ir.ir import check_integer_domain

    check_integer_domain(kernel, adapter.index_bits)
    return {
        "schema": "complete_row_softmax.v1",
        "kernel_sha256": kernel.digest(),
        "numerical_identity": numerics.identity,
        "configuration_identity": adapter.configuration_identity,
        "qualification_identity": adapter.qualification_identity,
        "problem": asdict(problem),
        "resources": asdict(resources),
        "static_calls": static_calls,
        "executed_commands": commands,
        "requested_read_bytes": reads,
        "requested_write_bytes": writes,
        "phase_commands": phase_counts,
        "private_state_bytes": sum(v.capacity * v.unit_bytes for v in storage),
        "physical_high_water_bytes": {
            b.backing: max((v.base + v.capacity) * v.unit_bytes for v in storage if v.backing == b.backing)
            for b in banks
            if any(v.backing == b.backing for v in storage)
        },
        "scope": "actual projected counted calls; independent primitive and final-byte qualification required",
        "timing": None,
    }
