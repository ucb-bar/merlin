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

from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Ptr, TensorArg, Var
from merlin.sched.ir.kernel import DTYPE_BYTES, check_structure
from merlin.semantic_compiler.allocate import allocate, check_assignment
from merlin.semantic_compiler.allocation_types import CandidateGraph, Reservation, StorageBank, Value
from merlin.semantic_compiler.rules import AddressConstraint

from .compact_dense import DenseMemoryView

ROLES = ("maximum", "exponentials", "lanes", "normalizer")
# Inclusive instruction phases: a source and destination of one instruction
# remain simultaneously owned through its completion.
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

    def emit_c(self, instruction_set, *, symbol):
        from merlin.sched.codegen import emit_c_function

        return emit_c_function(self.kernel, instruction_set, symbol=symbol, index_bits=self.index_bits)


def _choose_chunk(problem, resources, explicit):
    if resources.max_static_calls < 7:
        raise RowSoftmaxUnavailable("row softmax static command budget cannot hold its required seven phases")
    limit = min(problem.columns, resources.max_chunk_elements)
    chunk = (limit // resources.granularity) * resources.granularity
    if resources.partial_chunks and limit == problem.columns:
        chunk = limit
    elif not chunk and resources.partial_chunks:
        chunk = limit
    if explicit is not None:
        chunk = explicit
    elif chunk and (not resources.partial_chunks or resources.max_static_calls < 11) and problem.columns % chunk:
        # A bounded descending search avoids scanning a tensor-sized candidate
        # domain. The primitive granularity is a legal conservative fallback
        # when it divides the complete row. No timing model is implied.
        candidates = (chunk - i * resources.granularity for i in range(64))
        chunk = next(
            (n for n in candidates if n > 0 and problem.columns % n == 0),
            resources.granularity if problem.columns % resources.granularity == 0 else 0,
        )
    if type(chunk) is not int or chunk <= 0 or chunk > limit:
        raise RowSoftmaxUnavailable("no legal row chunk within declared resources")
    if chunk % resources.granularity and (chunk != problem.columns or not resources.partial_chunks):
        raise RowSoftmaxUnavailable("row chunk violates declared primitive granularity")
    if problem.columns % chunk and not resources.partial_chunks:
        raise RowSoftmaxUnavailable("row tail requires declared partial-chunk support")
    if 7 + 4 * bool(problem.columns % chunk) > resources.max_static_calls:
        raise RowSoftmaxUnavailable("row softmax chosen tail exceeds the static command budget")
    return chunk


def _schedule(problem, numerics, storage, chunk, adapter):
    row = Var("row")
    body = []

    def call(kind, column, count):
        return adapter.encode(_step(kind, row, column, count, problem, numerics, storage))

    def chunks(kind):
        trips, tail = divmod(problem.columns, chunk)
        var = kind + "_chunk"
        if trips:
            body.append(Loop(var, trips, (call(kind, Var(var) * chunk, chunk),)))
        if tail:
            body.append(call(kind, Const(trips * chunk), tail))

    body.append(call("init_maximum", Const(0), 1))
    chunks("maximum")
    chunks("exponential")
    body.append(call("init_lanes", Const(0), numerics.lanes))
    chunks("sum")
    body.append(call("finish", Const(0), numerics.lanes))
    chunks("normalize")
    return Kernel(
        "row_softmax",
        tuple(
            TensorArg(name, (problem.rows, problem.columns), numerics.dtype, access)
            for name, access in (("input", "read"), ("output", "write"))
        ),
        (Loop("row", problem.rows, tuple(body)),),
    )


def compile_row_softmax(
    problem,
    numerics,
    resources,
    banks,
    role_banks,
    adapter,
    *,
    reservations=(),
    private_ownership=False,
    disjoint_arguments=False,
    chunk_elements=None,
    timeout_ms=2000,
    pass_order=(),
):
    """Execute admission, tiling, allocation and verified dependency scheduling.

    Explicit chunks are Phase-2 alternatives through the same pipeline, selected
    source algorithm and checks. Invalid explicit ordering fails before any
    encoder/allocation runs. Refusal retains the caller's original source route.
    This synchronous schedule-to-C path is not final machine-byte conformance
    or ordinary full-model target integration.
    """
    from merlin.transforms.pipeline import PassSpec, PipelineContext, PipelineSession, validate_pass_order

    from .row_softmax_verification import verify_row_softmax
    from .schedule_pipeline import shared_library_sources

    specs = (
        PassSpec("row_semantics", 0, produces=("row_source",), mutates_ir=False, ir_level="schedule"),
        PassSpec(
            "row_tiling", 2, requires=("row_source",), produces=("row_chunk",), mutates_ir=False, ir_level="schedule"
        ),
        PassSpec(
            "row_allocation",
            3,
            requires=("row_source", "row_chunk"),
            produces=("row_storage",),
            mutates_ir=False,
            ir_level="schedule",
        ),
        PassSpec(
            "row_schedule",
            4,
            requires=("row_source", "row_chunk", "row_storage"),
            produces=("row_realization",),
            invalidates=("*",),
            ir_level="schedule",
        ),
    )
    if type(pass_order) is not tuple:
        raise ValueError("row reduction pass ordering must be an explicit immutable sequence")
    if pass_order:
        if len(pass_order) != len(specs) or set(pass_order) != {s.name for s in specs}:
            raise ValueError("explicit row reduction ordering must contain every pass exactly once")
        specs = tuple(next(s for s in specs if s.name == name) for name in pass_order)
    validate_pass_order(specs)
    _admit(problem, numerics, resources, adapter, private_ownership, disjoint_arguments)
    _storage_problem(problem, numerics, banks, role_banks)
    if type(reservations) is not tuple or any(type(r) is not Reservation for r in reservations):
        raise ValueError("row softmax reservations must be explicit immutable intervals")

    def identity(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    context = PipelineContext(
        hardware=adapter.configuration_identity,
        numerical_policy=numerics.identity,
        resources=identity(
            dict(
                problem=asdict(problem),
                resources=asdict(resources),
                banks=[asdict(b) for b in banks],
                role_banks=role_banks,
                reservations=[asdict(r) for r in reservations],
                chunk_elements=chunk_elements,
                private_ownership=private_ownership,
                disjoint_arguments=disjoint_arguments,
                index_bits=adapter.index_bits,
            )
        ),
        execution=adapter.qualification_identity,
        library=identity(shared_library_sources()),
    )

    def verify_ir(value):
        if type(value) is not Kernel or check_structure(value):
            raise ValueError("row scheduling produced invalid counted IR")

    session = PipelineSession(
        Kernel("row_softmax", (), ()),
        context=context,
        ir_level="schedule",
        max_repairs=0,
        fingerprint=lambda value: value.digest(),
        verify_ir=verify_ir,
    )

    def scheduled():
        assigned = session.analysis("row_storage")
        chunk = session.analysis("row_chunk")
        session.module = _schedule(problem, numerics, assigned, chunk, adapter)
        report = verify_row_softmax(
            session.module,
            problem,
            numerics,
            resources,
            assigned,
            banks,
            role_banks,
            adapter,
            reservations=reservations,
            private_ownership=private_ownership,
            disjoint_arguments=disjoint_arguments,
        )
        return {"row_realization": report}

    actions = {
        "row_semantics": lambda: {"row_source": numerics.identity},
        "row_tiling": lambda: {"row_chunk": _choose_chunk(problem, resources, chunk_elements)},
        "row_allocation": lambda: {
            "row_storage": allocate_row_softmax_storage(
                problem, numerics, banks, role_banks, reservations=reservations, timeout_ms=timeout_ms
            )
        },
        "row_schedule": scheduled,
    }
    storage = None
    for spec in specs:
        session.run(spec, actions[spec.name])
        if spec.name == "row_allocation":
            storage = session.analysis("row_storage")
    return RowSoftmaxCompilation(
        session.module, storage, session.analysis("row_realization"), adapter.index_bits, tuple(session.records)
    )
