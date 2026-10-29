"""Independent matrix products over explicit shared batch/plane views.

Batch axes describe logical products, not physical accelerator dimensions.
Omitting a logical axis from a read view reuses that original operand. Every
output retains every logical axis; no product or numerical plane is combined.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import prod


@dataclass(frozen=True)
class BatchMatrixView:
    """Leading physical tensor axes select named logical batch axes.

    The last two physical dimensions hold a matrix. ``transposed`` changes
    their interpretation, without claiming a physical transpose or copy.
    """

    shape: tuple[int, ...]
    batch_axes: tuple[int, ...]
    transposed: bool = False

    def __post_init__(self):
        if (
            type(self.shape) is not tuple or not 2 <= len(self.shape) <= 8
            or any(type(n) is not int or n <= 0 for n in self.shape)
            or prod(self.shape) * 4 >= 1 << 63
            or type(self.batch_axes) is not tuple
            or len(self.batch_axes) != len(self.shape) - 2
            or any(type(a) is not int or a < 0 for a in self.batch_axes)
            or len(set(self.batch_axes)) != len(self.batch_axes)
            or type(self.transposed) is not bool
        ):
            raise ValueError("batch view requires bounded static dimensions and distinct explicit axes")

    @property
    def matrix_shape(self):
        return self.shape[-2:][::-1] if self.transposed else self.shape[-2:]


@dataclass(frozen=True)
class BatchedDenseProblem:
    """Complete Cartesian product with independently projected read operands.

    For example, A[lp,head,M,K] and B[rp,head,K,N] use axes (0,2)
    and (1,2), producing Y[lp,rp,head,M,N]. All plane combinations remain.
    Accessible allocations, nonalias and primitive effects need separate checks.
    """

    batch_shape: tuple[int, ...]
    lhs: BatchMatrixView
    rhs: BatchMatrixView

    def __post_init__(self):
        if (
            type(self.batch_shape) is not tuple or not 1 <= len(self.batch_shape) <= 6
            or any(type(n) is not int or n <= 0 for n in self.batch_shape)
            or type(self.lhs) is not BatchMatrixView or type(self.rhs) is not BatchMatrixView
        ):
            raise ValueError("independent products require explicit positive bounded batch axes")
        for view in (self.lhs, self.rhs):
            view.__post_init__()
            if any(axis >= len(self.batch_shape) for axis in view.batch_axes):
                raise ValueError("operand batch projection is outside the logical product")
            if tuple(self.batch_shape[a] for a in view.batch_axes) != view.shape[:-2]:
                raise ValueError("operand batch projection changes an original extent")
        if self.lhs.matrix_shape[1] != self.rhs.matrix_shape[0]:
            raise ValueError("independent products must preserve the complete reduction extent")
        if prod(self.output_shape) * 4 >= 1 << 63:
            raise ValueError("batched output exceeds the signed index range")

    @property
    def m(self):
        return self.lhs.matrix_shape[0]

    @property
    def k(self):
        return self.lhs.matrix_shape[1]

    @property
    def n(self):
        return self.rhs.matrix_shape[1]

    @property
    def output_shape(self):
        return (*self.batch_shape, self.m, self.n)

    @property
    def products(self):
        return prod(self.batch_shape)

    def record(self):
        self.__post_init__()
        return {"schema": "merlin.batched_dense_problem.v1", **asdict(self),
                "output_shape": self.output_shape, "products": self.products,
                "numeric_law": "independent zero-seeded signed-i8 products, modular-i32 accumulation",
                "target_effects": "not established by geometry"}


def counted_batched_dense_products(problem, *, name, instruction):
    """Call a caller-owned matrix implementation once per logical product.

    ``product`` is the row-major logical product index. ``lhs_offset``,
    ``rhs_offset`` and ``output_offset`` are **element offsets**, not bytes,
    into the original contiguous physical tensor shapes. Read offsets project
    the logical batch indices through each view's explicit ``batch_axes``;
    omitted axes share that read operand without combining any products.
    A transposed matrix keeps its physical base; the leaf owns interpretation
    of its two matrix axes. Output always retains every logical batch axis.

    This constructs only counted index/control IR. The caller supplies the
    instruction implementation, element widths, accessible allocation/nonalias
    checks, physical storage, effects and actual completion behavior.
    """
    from merlin.sched.ir import Call, Kernel, Loop, Var, add, check_structure, mul

    if (type(problem) is not BatchedDenseProblem
        or any(type(value) is not str or not value
               or any(c.isspace() or c in "(),:" for c in value)
               for value in (name, instruction))):
        raise ValueError("counted products require an explicit problem, kernel name and instruction")
    problem.__post_init__()
    indices = tuple(Var(f"batch{axis}") for axis in range(len(problem.batch_shape)))
    product = add(*(mul(index, prod(problem.batch_shape[axis + 1:]))
                    for axis, index in enumerate(indices)))

    def offset(view):
        return add(*(mul(indices[axis], prod(view.shape[physical + 1:]))
                     for physical, axis in enumerate(view.batch_axes)))

    body = (Call(instruction, (
        ("product", product), ("lhs_offset", offset(problem.lhs)),
        ("rhs_offset", offset(problem.rhs)),
        ("output_offset", mul(product, problem.m * problem.n)),
    )),)
    for axis in reversed(range(len(indices))):
        body = (Loop(indices[axis].name, problem.batch_shape[axis], body),)
    kernel = Kernel(name, (), body, (("offset_unit", "elements"),))
    if check_structure(kernel):
        raise ValueError("counted batch product indices have invalid binder scope")
    return kernel


def partition_batched_dense_storage(storage, *, groups, row_counts):
    """Slice already owned private windows into disjoint invocation-local groups.

    Row counts and byte widths come from the caller's checked source/layout.
    This assigns no additional memory, external tensor nonalias permission,
    completion law or cross-invocation residency.
    """
    from .compact_dense import DenseLoopStorage

    if (type(storage) is not DenseLoopStorage or storage.accumulator is None
        or type(groups) is not int or not 1 <= groups <= 4096
        or type(row_counts) is not tuple or len(row_counts) != 3
        or any(type(n) is not int or n <= 0 for n in row_counts)):
        raise ValueError("batch storage needs complete owned windows and bounded positive row counts")
    views = (storage.lhs, storage.rhs, storage.accumulator)
    for view, rows in zip(views, row_counts, strict=True):
        view.__post_init__()
        if groups * rows > view.capacity or rows % view.alignment:
            raise ValueError("batch suballocations exceed or misalign the original private window")
    for index, left in enumerate(views):
        for right in views[index + 1:]:
            if (left.backing == right.backing
                and left.base * left.unit_bytes < (right.base + right.capacity) * right.unit_bytes
                and right.base * right.unit_bytes < (left.base + left.capacity) * left.unit_bytes):
                raise ValueError("original private batch windows have possible byte overlap")
    return tuple(DenseLoopStorage(*(
        replace(view, base=view.base + group * rows, capacity=rows)
        for view, rows in zip(views, row_counts, strict=True)
    )) for group in range(groups))


def partition_dense_phases(kernel, *, phases, classify, completion_instruction, max_static_nodes=4096):
    """Propose an exact counted partition; supply no reordering/issue authority.

    ``classify(call)`` must assign every original noncompletion call to one
    named phase under the consumer's selected complete effect contract. Each
    phase preserves its calls, operands and within-group order verbatim. The
    consumer separately proves cross-phase/group dependencies, source outputs,
    configuration persistence, private storage and real replacement completion.
    Original completion calls are counted in the receipt, not silently retained
    as a false claim that the resulting phases are independently executable.
    """
    from merlin.sched.ir import Call, Kernel, Loop, check_structure

    if (type(kernel) is not Kernel or check_structure(kernel)
        or type(phases) is not tuple or not 2 <= len(phases) <= 16
        or any(type(p) is not str or not p for p in phases)
        or len(set(phases)) != len(phases) or not callable(classify)
        or type(completion_instruction) is not str or not completion_instruction
        or type(max_static_nodes) is not int or not 1 <= max_static_nodes <= 65536):
        raise ValueError("phase partition needs checked source and explicit bounded phase/effect inputs")
    counts = dict.fromkeys(phases, 0)
    nodes, completions = 0, 0

    def visit(body, multiplicity=1):
        nonlocal nodes, completions
        selected = {p: [] for p in phases}
        for statement in body:
            nodes += 1
            if nodes > max_static_nodes:
                raise ValueError("phase partition exceeds its original static proof budget")
            if type(statement) is Loop:
                if statement.iter_args or statement.results or statement.yields:
                    raise ValueError("phase partition cannot transport carried completion/state")
                child = visit(statement.body, multiplicity * statement.extent)
                for phase in phases:
                    if child[phase]:
                        selected[phase].append(replace(statement, body=child[phase]))
            elif type(statement) is not Call:
                raise ValueError("phase partition encountered an unknown statement")
            elif statement.instr == completion_instruction:
                if statement != Call(completion_instruction, ()):
                    raise ValueError("phase completion has an unknown payload or observable event")
                completions += multiplicity
            else:
                if (statement.unit or statement.produces or statement.awaits or statement.stages
                    or statement.sets or statement.assumes):
                    raise ValueError("phase partition needs an independent event/state transport proof")
                phase = classify(statement)
                if type(phase) is not str or phase not in selected:
                    raise ValueError("phase partition lacks a complete classification of original effects")
                counts[phase] += multiplicity
                selected[phase].append(statement)
        return {phase: tuple(body) for phase, body in selected.items()}

    bodies = visit(kernel.body)
    if any(not counts[phase] for phase in phases):
        raise ValueError("phase partition cannot substitute an empty phase")
    actual = {phase: replace(kernel, body=bodies[phase]) for phase in phases}
    if any(check_structure(part) for part in actual.values()):
        raise ValueError("phase partition changed original counted binder scope")
    return actual, {
        "schema": "merlin.dense_phase_partition.v1", "original_kernel_sha256": kernel.digest(),
        "phase_kernel_sha256": {p: actual[p].digest() for p in phases},
        "original_dynamic_calls": counts, "original_dynamic_completions": completions,
        "static_nodes": nodes, "execution_authority": "not established",
        "completion_owner": "consumer must prove and emit replacement phase completions",
    }


def verify_dense_phase_partition(kernel, partitions, receipt, **options):
    """Rebuild from the original source and current effect selector before use."""
    expected, record = partition_dense_phases(kernel, **options)
    if type(partitions) is not dict or partitions != expected or receipt != record:
        raise ValueError("phase source, partitioned commands or classification changed")
    return record
