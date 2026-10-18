"""Compact rectangular dense schedules in the existing counted-loop kernel IR.

Merlin owns panel blocking, retention, tails and source reduction order. An
explicit adapter translates typed moves/computes and projects their *actual*
operands back. Instruction implementation and physical row mapping remain OOT.
No target lookup, opaque output inference, or latency model is used here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, TensorArg, Var, select
from merlin.sched.ir.kernel import DTYPE_BYTES, check_structure
from merlin.transforms.dense_planning import DenseBlock, DenseChoice, DenseProblem, DenseResources, choose_dense_block

from .block_schedule import Compute, Load, Preload, Store

DenseOperation = Load | Preload | Compute | Store


@dataclass(frozen=True)
class DenseMemoryView:
    """Assigned rows in an explicitly identified physical backing.

    ``base``/``capacity``/``alignment`` are in this view's row units. Different
    row widths may alias a backing only through their explicit byte mapping.
    These are usable assignments after packing/DMA/other reservations.
    """

    name: str
    backing: str
    unit_bytes: int
    base: int
    capacity: int
    alignment: int = 1

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in (self.name, self.backing)):
            raise ValueError("dense storage needs explicit view and backing identities")
        if any(type(x) is not int or x <= 0 for x in (self.unit_bytes, self.capacity, self.alignment)):
            raise ValueError("dense storage units, capacity and alignment must be positive")
        if type(self.base) is not int or self.base < 0 or self.base % self.alignment:
            raise ValueError("dense storage base is negative or unaligned")


@dataclass(frozen=True)
class DenseLoopAdapter:
    """Public instruction boundary. Neither function is selected by its name.

    Projection must decode the actual call operands, not retained requests.
    The declaration is conditional on the adapter's independent qualification;
    synthetic adapters establish only the shared algorithm's scope.
    """

    encode: Callable[[DenseOperation], Call]
    project: Callable[[Call], DenseOperation]
    configuration_identity: str
    numerical_policy_identity: str
    synchronous_completion: bool
    exact_accumulator_continuation: bool
    index_bits: int
    qualification_identity: str | None = None

    def __post_init__(self):
        if not callable(self.encode) or not callable(self.project):
            raise ValueError("dense lowering requires an encoder and actual-operand projection")
        if any(
            not isinstance(value, str) or not value
            for value in (self.configuration_identity, self.numerical_policy_identity)
        ):
            raise ValueError("dense lowering requires selected configuration and numerical policy")
        if any(type(x) is not bool for x in (self.synchronous_completion, self.exact_accumulator_continuation)):
            raise ValueError("dense completion and continuation declarations must be Boolean")
        if self.qualification_identity is not None and (
            type(self.qualification_identity) is not str or not self.qualification_identity.strip()
        ):
            raise ValueError("dense qualification identity must be nonempty or unknown")
        if type(self.index_bits) is not int or self.index_bits not in (32, 64):
            raise ValueError("dense C index arithmetic requires a selected signed 32 or 64-bit representation")


@dataclass(frozen=True)
class DenseLoopStorage:
    lhs: DenseMemoryView
    rhs: DenseMemoryView
    accumulator: DenseMemoryView | None

    def __post_init__(self):
        views = (self.lhs, self.rhs) + (() if self.accumulator is None else (self.accumulator,))
        if len({view.name for view in views}) != len(views):
            raise ValueError("dense live storage roles require distinct view identities")

    def row_counts(self, problem: DenseProblem, block: DenseBlock):
        return (
            block.m * block.k * problem.tile_m,
            block.k * block.n * problem.tile_k,
            block.m * block.n * problem.tile_m,
        )

    def k_limit(self, problem: DenseProblem, resources: DenseResources, m: int, n: int):
        return min(
            self.lhs.capacity // (m * problem.tile_m),
            self.rhs.capacity // (n * problem.tile_k),
            resources.operand_bytes
            // (m * problem.tile_m * self.lhs.unit_bytes + n * problem.tile_k * self.rhs.unit_bytes),
        )

    def admits(self, problem: DenseProblem, block: DenseBlock, resources: DenseResources):
        if self.accumulator is None:
            return False
        views = (self.lhs, self.rhs, self.accumulator)
        widths = (
            problem.tile_k * problem.lhs_bytes,
            problem.tile_n * problem.rhs_bytes,
            problem.tile_n * problem.accumulator_bytes,
        )
        rows = self.row_counts(problem, block)
        if any(
            stride % view.alignment for stride, view in zip((problem.tile_m, problem.tile_k, problem.tile_m), views)
        ):
            return False
        if any(width > view.unit_bytes or count > view.capacity for width, view, count in zip(widths, views, rows)):
            return False
        used = tuple(view.unit_bytes * count for view, count in zip(views, rows))
        if used[0] + used[1] > resources.operand_bytes or used[2] > resources.accumulator_bytes:
            return False
        intervals = [
            (view.backing, view.base * view.unit_bytes, (view.base + view.capacity) * view.unit_bytes) for view in views
        ]
        return not any(
            a[0] == b[0] and a[1] < b[2] and b[1] < a[2] for i, a in enumerate(intervals) for b in intervals[i + 1 :]
        )


@dataclass(frozen=True)
class CompactDenseResult:
    kernel: Kernel
    choice: DenseChoice
    realization: dict
    index_bits: int
    selected_contracts: object = None

    def emit_c(self, instruction_set, *, symbol):
        """Use the existing emitter with the selected, checked loop index width.

        Target instruction implementations still own their argument widths and
        address windows. This emits C, not a machine conformance receipt.
        """
        from merlin.sched.codegen import emit_c_function

        def fresh():
            if self.selected_contracts is not None:
                self.selected_contracts.revalidate()
            receipt = self.realization.get("pipeline")
            if receipt is not None and (
                self.kernel.digest() != receipt["kernel_sha256"]
                or _dense_library_sources() != receipt["library_sources_sha256"]
            ):
                raise ValueError("dense checked kernel or shared library changed; replan before emission")

        fresh()
        source = emit_c_function(self.kernel, instruction_set, symbol=symbol, index_bits=self.index_bits)
        fresh()
        return source


@dataclass(frozen=True)
class _Group:
    var: str | None
    trips: int
    base: Expr
    blocks: int
    contains_last: bool


def _groups(total, block, name):
    full, tail = divmod(total, block)
    parts = []
    if full:
        parts.append(_Group(name, full, Var(name) * block, block, not tail))
    if tail:
        parts.append(_Group(None, 1, Const(full * block), tail, True))
    return tuple(parts)


def _edge(group, local, logical_extent, primitive):
    tail = logical_extent % primitive
    if not tail or not group.contains_last:
        return Const(primitive)
    last = select(local, group.blocks - 1, tail, primitive)
    return last if group.var is None else select(group.var, group.trips - 1, last, primitive)


def _first(group, local):
    if isinstance(group.base, Const):
        return select(local, 0, 0, 1) if group.base.value == 0 else Const(1)
    return select(group.var, 0, select(local, 0, 0, 1), 1)


def _static_calls(body):
    for statement in body:
        if isinstance(statement, Loop):
            yield from _static_calls(statement.body)
        else:
            yield statement


def emit_compact_dense(
    problem: DenseProblem,
    block: DenseBlock,
    storage: DenseLoopStorage,
    adapter: DenseLoopAdapter,
    *,
    name: str,
    dtypes: tuple[str, str, str],
    outer_order: tuple[str, str] = ("m", "n"),
):
    """Emit bounded panel nests without enumerating tensor elements or tiles.

    The supported family starts a zero-initialized reduction on its first
    compute and keeps each output accumulator through the complete K domain.
    Bias/other seeds retain their original route until explicitly implemented.
    """
    if problem.bias:
        raise ValueError("compact dense family requires an explicit zero seed; bias initialization is unavailable")
    if any(type(value) is not int or value <= 0 for value in (block.m, block.k, block.n)):
        raise ValueError("compact dense block dimensions must be positive integers")
    if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation:
        raise ValueError("compact dense family needs synchronous completion and exact accumulator continuation")
    if storage.accumulator is None:
        raise ValueError("compact dense family has no declared addressable accumulator storage")
    if outer_order not in (("m", "n"), ("n", "m")):
        raise ValueError("compact dense outer order must keep the complete source reduction inside each output block")
    if tuple(DTYPE_BYTES.get(t) for t in dtypes) != (problem.lhs_bytes, problem.rhs_bytes, problem.accumulator_bytes):
        raise ValueError("dense operand and accumulator dtypes disagree with their physical widths")
    mt, kt, nt = problem.tiles
    serial = 0

    def variable(prefix):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def call(operation):
        result = adapter.encode(operation)
        if type(result) is not Call:
            raise ValueError("dense adapter must emit an actual kernel instruction call")
        return result

    def nest(extents, action):
        names = tuple(variable("tile") for _ in extents)
        body = tuple(action(*(Var(name) for name in names)))
        for var, extent in reversed(tuple(zip(names, extents))):
            body = (Loop(var, extent, body),)
        return body

    def tile_group(mg, ng):
        statements = []
        for kg in _groups(kt, block.k, variable("panel")):

            def lhs(i, reduction):
                m, k = mg.base + i, kg.base + reduction
                return (
                    call(
                        Load(
                            "lhs",
                            (m, k),
                            m * problem.tile_m,
                            k * problem.tile_k,
                            _edge(mg, i, problem.m, problem.tile_m),
                            _edge(kg, reduction, problem.k, problem.tile_k),
                            storage.lhs.base + (i * block.k + reduction) * problem.tile_m,
                            1,
                        )
                    ),
                )

            def rhs(reduction, j):
                k, n = kg.base + reduction, ng.base + j
                return (
                    call(
                        Load(
                            "weight",
                            (k, n),
                            k * problem.tile_k,
                            n * problem.tile_n,
                            _edge(kg, reduction, problem.k, problem.tile_k),
                            _edge(ng, j, problem.n, problem.tile_n),
                            storage.rhs.base + (reduction * block.n + j) * problem.tile_k,
                            1,
                        )
                    ),
                )

            def compute(reduction, i, j):
                m, k, n = mg.base + i, kg.base + reduction, ng.base + j
                rows = _edge(mg, i, problem.m, problem.tile_m)
                depth = _edge(kg, reduction, problem.k, problem.tile_k)
                cols = _edge(ng, j, problem.n, problem.tile_n)
                row = storage.accumulator.base + (i * block.n + j) * problem.tile_m
                return (
                    call(
                        Preload(
                            storage.rhs.base + (reduction * block.n + j) * problem.tile_k,
                            (k, n),
                            row,
                            _first(kg, reduction),
                            rows,
                            cols,
                            depth,
                        )
                    ),
                    call(
                        Compute(
                            storage.lhs.base + (i * block.k + reduction) * problem.tile_m, (m, k), rows, depth, True
                        )
                    ),
                )

            body = (
                *nest((mg.blocks, kg.blocks), lhs),
                *nest((kg.blocks, ng.blocks), rhs),
                *nest((kg.blocks, mg.blocks, ng.blocks), compute),
            )
            statements.extend((Loop(kg.var, kg.trips, body),) if kg.var is not None else body)

        def store(i, j):
            return (
                call(
                    Store(
                        (mg.base + i) * problem.tile_m,
                        (ng.base + j) * problem.tile_n,
                        _edge(mg, i, problem.m, problem.tile_m),
                        _edge(ng, j, problem.n, problem.tile_n),
                        storage.accumulator.base + (i * block.n + j) * problem.tile_m,
                    )
                ),
            )

        statements.extend(nest((mg.blocks, ng.blocks), store))
        return tuple(statements)

    groups = {"m": _groups(mt, block.m, "outer_m"), "n": _groups(nt, block.n, "outer_n")}

    def outer(level, selected):
        if level == 2:
            return tile_group(selected["m"], selected["n"])
        axis = outer_order[level]
        body = []
        for group in groups[axis]:
            nested = outer(level + 1, {**selected, axis: group})
            body.extend((Loop(group.var, group.trips, nested),) if group.var is not None else nested)
        return tuple(body)

    kernel = Kernel(
        name,
        (
            TensorArg("lhs", (problem.m, problem.k), dtypes[0], "read"),
            TensorArg("rhs", (problem.k, problem.n), dtypes[1], "read"),
            TensorArg("output", (problem.m, problem.n), dtypes[2], "write"),
        ),
        outer(0, {}),
    )
    problems = check_structure(kernel)
    if problems:
        raise ValueError("invalid compact dense structure: " + "; ".join(problems))
    return kernel


def schedule_compact_dense(
    problem: DenseProblem,
    resources: DenseResources | None,
    storage: DenseLoopStorage,
    adapter: DenseLoopAdapter,
    *,
    name: str,
    dtypes: tuple[str, str, str],
    static_budget: int,
    external_buffers_disjoint: bool,
    outer_order: tuple[str, str] = ("m", "n"),
    verification_timeout_ms: int = 2000,
    max_repairs: int = 1,
    selected_contracts=None,
    reservations=(),
    pass_order=(),
):
    """Ordinary scheduling entry: choose, emit and independently verify output.

    No candidate preserves the caller's existing legal route. A checking-budget
    refusal may try one simple streaming loop; every address and dependency is
    rechecked. A concrete invariant failure never triggers fallback. The explicit
    disjointness premise belongs to the original source/runtime boundary, not
    to an optimization candidate. Missing decoder/completion facts diagnose.
    With selected contracts, geometry, formats and assigned windows are checked
    against both pinned documents; resources=None derives available budgets.
    Reservations name canonical backing bytes after upstream allocation.
    """
    if type(external_buffers_disjoint) is not bool or not external_buffers_disjoint:
        raise ValueError("dense scheduling cannot reorder possibly aliased external buffers")
    if type(max_repairs) is not int or max_repairs not in (0, 1):
        raise ValueError("compact dense supports a bounded zero or one repair")

    from merlin.transforms.pipeline import PassSpec, PipelineContext, PipelineSession, validate_pass_order

    from .compact_verification import DenseVerificationUnavailable, verify_compact_dense
    from .selected_contracts import SelectedSchedulingContracts

    if selected_contracts is not None:
        if type(selected_contracts) is not SelectedSchedulingContracts or selected_contracts.dense is None:
            raise ValueError("compact dense requires a selected dense scheduling contract")
        selected_contracts.revalidate()
        resources, adapter = selected_contracts.dense.bind(
            problem, storage, adapter, dtypes=dtypes, hardware=selected_contracts.hardware_identity,
            policy=selected_contracts.numerical_policy, resources=resources, reservations=reservations,
        )
    elif resources is None or reservations:
        raise ValueError("dense resource derivation and reservations require selected RTL/software contracts")
    sources = _dense_library_sources()
    binding = None if selected_contracts is None else selected_contracts.record(scopes=(
        "dense_geometry", "dense_storage_windows", "dense_numerics_and_execution", "dense_instruction_bindings"
    ))
    context = PipelineContext(
        hardware=adapter.configuration_identity, numerical_policy=adapter.numerical_policy_identity,
        resources=_digest((asdict(resources), asdict(storage), tuple(asdict(r) for r in reservations))),
        execution=_digest((binding, adapter.qualification_identity, adapter.synchronous_completion,
                           adapter.exact_accumulator_continuation, adapter.index_bits)), library=_digest(sources),
    )
    state = {"problem": asdict(problem), "resources": asdict(resources), "storage": asdict(storage),
             "dtypes": dtypes, "outer_order": outer_order, "name": name, "static_budget": static_budget,
             "external_buffers_disjoint": external_buffers_disjoint, "kernel": None}

    def fingerprint(current):
        return _digest({**current, "kernel": None if current["kernel"] is None else current["kernel"].digest()})

    def structure(current):
        if current["kernel"] is not None and check_structure(current["kernel"]):
            raise ValueError("dense pipeline requires structurally valid kernel IR")

    session = PipelineSession(state, context=context, ir_level="compact_dense", max_repairs=max_repairs,
                              fingerprint=fingerprint, verify_ir=structure)
    specs = (
        PassSpec("admission", 0, produces=("legality",), mutates_ir=False, ir_level="compact_dense"),
        PassSpec("blocking", 2, requires=("legality",), produces=("choice",), mutates_ir=False,
                 ir_level="compact_dense"),
        PassSpec("emission", 5, requires=("choice",), produces=("emitted",), invalidates=("*",),
                 ir_level="compact_dense"),
        PassSpec("realization", 5, requires=("emitted",), produces=("realization",), mutates_ir=False,
                 ir_level="compact_dense"),
    )
    names = tuple(spec.name for spec in specs)
    if type(pass_order) is not tuple or (
        pass_order and (len(pass_order) != len(names) or set(pass_order) != set(names))
    ):
        raise ValueError("explicit dense pass order must name each pass exactly once")
    ordered = tuple({spec.name: spec for spec in specs}[name] for name in (pass_order or names))
    validate_pass_order(ordered)

    def fresh():
        if selected_contracts is not None:
            selected_contracts.revalidate()
        if sources != _dense_library_sources():
            raise ValueError("dense shared library changed during compilation; replan")

    def emit(block):
        return emit_compact_dense(problem, block, storage, adapter, name=name, dtypes=dtypes, outer_order=outer_order)

    repairs = []
    search = resources
    for attempt in range(max_repairs + 1):
        choice = None
        unavailable = None

        def admission():
            if not adapter.synchronous_completion or not adapter.exact_accumulator_continuation:
                raise ValueError("compact dense family needs synchronous completion and exact accumulator continuation")
            return {"legality": {"selected_contracts": binding}}

        def blocking():
            nonlocal choice
            choice = choose_dense_block(
                problem, search, static_budget=static_budget,
                price=lambda block: sum(1 for _ in _static_calls(emit(block).body)),
                admits=lambda block: storage.admits(problem, block, resources), require_reuse=False,
                k_limit=lambda m, n: storage.k_limit(problem, resources, m, n),
            )
            return {"choice": choice}

        def emission():
            state["kernel"] = emit(session.analysis("choice").block)
            return {"emitted": state["kernel"].digest()}

        def realization():
            nonlocal unavailable
            try:
                report = verify_compact_dense(
                    state["kernel"], problem, resources, storage, adapter,
                    static_budget=static_budget, timeout_ms=verification_timeout_ms,
                )
            except DenseVerificationUnavailable as exc:
                unavailable = exc
                report = None
            return {"realization": report}

        actions = dict(admission=admission, blocking=blocking, emission=emission, realization=realization)
        for spec in ordered:
            fresh()
            session.run(spec, actions[spec.name])
            fresh()
            if spec.name == "blocking" and choice is None:
                return None
        if unavailable is not None:
            if attempt == max_repairs or isinstance(unavailable.__cause__, ImportError):
                raise unavailable
            repairs.append({"discarded_kernel_sha256": state["kernel"].digest(), "reason": str(unavailable)})
            session.repair(stage=0, reason=str(unavailable))
            state["kernel"] = None
            search = replace(resources, max_m_blocks=1, max_n_blocks=1, max_k_blocks=1)
        else:
            report = session.analysis("realization")
            report["repairs"] = repairs
            report["pipeline"] = {
                "schema": "merlin.compact_dense_pipeline.v1", "context": asdict(context),
                "kernel_sha256": state["kernel"].digest(), "library_sources_sha256": sources,
                "selected_contracts": binding, "passes": [asdict(record) for record in session.records],
                "repair_rounds": session.repair_round,
            }
            return CompactDenseResult(state["kernel"], choice, report, adapter.index_bits, selected_contracts)


def _digest(value):
    import hashlib
    import json

    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dense_library_sources():
    import hashlib
    import importlib
    from pathlib import Path

    names = (
        "merlin.compile.scheduling.compact_dense", "merlin.compile.scheduling.compact_verification",
        "merlin.compile.scheduling.dense_contracts", "merlin.compile.scheduling.selected_contracts",
        "merlin.compile.scheduling.storage_contracts", "merlin.compile.scheduling.temporary_storage",
        "merlin.semantic_compiler.allocation_types", "merlin.semantic_compiler.allocate",
        "merlin.compile.scheduling.block_schedule", "merlin.transforms.dense_planning",
        "merlin.transforms.pipeline", "merlin.sched.ir.kernel", "merlin.sched.ir.expr",
        "merlin.sched.ir.integer_domain", "merlin.sched.codegen", "merlin.sched.isa",
        "merlin.targetgen.isa_spec",
    )
    return {
        name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest() for name in names
    }
