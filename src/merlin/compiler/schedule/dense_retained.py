"""Retained-operand choices and independent dense reduction coverage checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import prod

from merlin.compiler.kernel_ir.ir import (
    Call,
    Const,
    Loop,
    Scaled,
    Select,
    Sum,
    Var,
    check_integer_domain,
    check_structure,
)
from merlin.compiler.schedule.blocks import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compiler.schedule.dense_contracts import DenseTiledIntegerReadout, TiledIntegerStore
from merlin.compiler.schedule.dense_panel_ir import DRAIN, _calls, _loops, _serial_projection


@dataclass(frozen=True)
class DenseRetainedChoice:
    """Unverified canonical residency alternative, never an execution grant.

    Counts include every full/tail family in the counted emitter after its
    invariant hoists. They exclude unchanged bias/readout transfers and do not
    predict physical DMA aggregation, target stalls, configuration or cycles.
    The consumer must execute the selected role/order/block and independently
    check final source products, addresses, effects and completion.
    """

    block: object
    role: str
    outer_order: tuple[str, str]
    lhs_loads: int
    rhs_loads: int
    requested_operand_bytes: int


def dense_m_panel_capacity(problem, resources, storage, *, n_tiles):
    """Explain full-K M-panel bounds using the original private assignments.

    Physical row widths include padding. The caller's emitter limits remain
    limits; unused capacity in another role is never reassigned implicitly.
    These are construction conditions, not a profitability prediction.
    """
    from merlin.transforms.dense_planning import DenseBlock

    mt, kt, nt = problem.tiles
    if type(n_tiles) is not int or not 1 <= n_tiles <= nt:
        raise ValueError("M-panel capacity needs an original positive N tile extent")
    lhs_per_m, acc_per_m = kt * problem.tile_m, n_tiles * problem.tile_m
    rhs_rows = kt * n_tiles * problem.tile_k
    rhs_bytes = rhs_rows * storage.rhs.unit_bytes
    limits = dict(
        source_m_tiles=mt,
        emitter_m_tiles=resources.max_m_blocks,
        lhs_private_rows=storage.lhs.capacity // lhs_per_m,
        accumulator_private_rows=(storage.accumulator.capacity // acc_per_m if storage.accumulator else 0),
        combined_operand_bytes=max(0, resources.operand_bytes - rhs_bytes) // (lhs_per_m * storage.lhs.unit_bytes),
        accumulator_bytes=(
            resources.accumulator_bytes // (acc_per_m * storage.accumulator.unit_bytes) if storage.accumulator else 0
        ),
    )
    one = DenseBlock(
        1,
        kt,
        n_tiles,
        lhs_per_m * storage.lhs.unit_bytes + rhs_bytes,
        acc_per_m * storage.accumulator.unit_bytes if storage.accumulator else 0,
        0,
        0,
        0,
        0,
        0,
    )
    conditions = dict(
        rhs_private_rows=rhs_rows <= storage.rhs.capacity,
        multiple_m_tiles=mt > 1,
        full_k_emitter_domain=kt <= resources.max_k_blocks,
        n_emitter_domain=n_tiles <= resources.max_n_blocks,
        original_private_assignment=storage.admits(problem, one, resources),
        partial_output_tiles=(
            resources.partial_output_tiles or problem.m % problem.tile_m == problem.n % problem.tile_n == 0
        ),
    )
    maximum = min(limits.values()) if all(conditions.values()) else 0
    return dict(
        mechanism="full_k_rhs_reuse_across_m_panels",
        n_tiles=n_tiles,
        k_tiles=kt,
        primitive_m=problem.tile_m,
        maximum_m_tiles=maximum,
        maximum_m_rows=maximum * problem.tile_m,
        m_tile_limits=limits,
        conditions=conditions,
        retained_rhs_rows=rhs_rows,
        retained_rhs_bytes=rhs_bytes,
        lhs_rows_per_m_tile=lhs_per_m,
        accumulator_rows_per_m_tile=acc_per_m,
        limiting_conditions=[key for key, bound in limits.items() if bound == maximum]
        + [key for key, value in conditions.items() if not value],
        profitability="unknown",
        array_weight_retention="requires_separate_selected_array_law",
    )


def dense_retained_choices(problem, resources, storage, *, roles=("lhs", "weight"), n_choices=None):
    """Enumerate both reusable residency families within original assignments.

    No memory is reassigned between roles. The existing bounded geometric
    search and all original partial-tile/block restrictions are retained. For
    LHS retention, the complete N groups share one invariant LHS stage; a
    separate partial N family still reloads it. RHS retention is symmetric in
    M. Thus even the traffic ranking accounts for remainder families without
    expanding the dynamic domain or assuming reuse across unrelated scopes.
    Explicit ``n_choices`` restricts enumeration to original supported widths;
    constructing one selected width does not search every unrelated N panel.
    """
    from merlin.transforms.dense_planning import dense_blocks

    if (
        type(roles) is not tuple
        or not roles
        or len(set(roles)) != len(roles)
        or any(role not in ("lhs", "weight") for role in roles)
    ):
        raise ValueError("retained selection requires distinct explicit LHS/weight residency families")
    mt, kt, nt = problem.tiles
    blocks = dense_blocks(
        problem, resources, k_limit=lambda m, n: storage.k_limit(problem, resources, m, n), n_choices=n_choices
    )
    choices = []
    for block in blocks:
        if block.k != kt or not storage.admits(problem, block, resources):
            continue
        for role in roles:
            lhs_repeats = (
                int(bool(nt // block.n)) + int(bool(nt % block.n)) if role == "lhs" else (nt + block.n - 1) // block.n
            )
            rhs_repeats = (
                int(bool(mt // block.m)) + int(bool(mt % block.m))
                if role == "weight"
                else (mt + block.m - 1) // block.m
            )
            choices.append(
                DenseRetainedChoice(
                    block,
                    role,
                    ("m", "n") if role == "lhs" else ("n", "m"),
                    mt * kt * lhs_repeats,
                    kt * nt * rhs_repeats,
                    problem.m * problem.k * problem.lhs_bytes * lhs_repeats
                    + problem.k * problem.n * problem.rhs_bytes * rhs_repeats,
                )
            )
    return tuple(
        sorted(
            choices,
            key=lambda choice: (
                choice.requested_operand_bytes,
                choice.lhs_loads + choice.rhs_loads,
                -choice.block.m * choice.block.n,
                roles.index(choice.role),
                choice.block.m,
                choice.block.n,
            ),
        )
    )


@dataclass(frozen=True)
class DenseRetainedReductionAdmission:
    """Independent selected premises for read-only operands and ordered sums.

    Ordered command acceptance alone does not prove accumulator commit order.
    The selected implementation must preserve each overlapping accumulator
    update in issue order, including initialization and continuation. No operand
    overwrite, load/compute overlap, or generic RAW/WAR scoreboard is admitted.
    Initializers retain individual drains unless an explicit ordered load
    configuration snapshot and pairwise-disjoint ACC write admission is selected.
    Issued counts are compiler cost bounds, not invented physical capacities.
    """

    hardware_identity: str
    numerical_policy_identity: str
    qualification_identity: str
    max_issued_compute_pairs: int
    max_issued_loads: int
    pending_compute_pairs: int
    pending_loads: int
    execution: str = "ordered_accumulator_commits_read_only_operands_full_drain"
    load_configuration: str = "ordered_snapshot_before_data_issue"
    descriptor: str = "preload_immediately_paired_with_compute"
    max_issued_readouts: int = 1
    pending_readouts: int = 1
    readout_configuration: str = "separate_drained_readout"
    max_issued_initializations: int = 1
    initialization_configuration: str = "separate_drained_initialization"

    def __post_init__(self):
        if (
            any(
                type(v) is not str or not v.strip()
                for v in (self.hardware_identity, self.numerical_policy_identity, self.qualification_identity)
            )
            or any(
                type(v) is not int or v <= 0
                for v in (
                    self.max_issued_compute_pairs,
                    self.max_issued_loads,
                    self.pending_compute_pairs,
                    self.pending_loads,
                    self.max_issued_readouts,
                    self.pending_readouts,
                    self.max_issued_initializations,
                )
            )
            or self.execution != "ordered_accumulator_commits_read_only_operands_full_drain"
            or self.load_configuration != "ordered_snapshot_before_data_issue"
            or self.descriptor != "preload_immediately_paired_with_compute"
            or self.initialization_configuration
            not in ("separate_drained_initialization", "ordered_snapshot_before_disjoint_accumulator_issue")
            or self.max_issued_initializations > 1
            and self.initialization_configuration != "ordered_snapshot_before_disjoint_accumulator_issue"
            or self.readout_configuration not in ("separate_drained_readout", "ordered_snapshot_before_store_issue")
            or self.max_issued_readouts > 1
            and self.readout_configuration != "ordered_snapshot_before_store_issue"
        ):
            raise ValueError("retained reduction needs independent ordered commit, snapshot and completion premises")


def dense_retained_operand_hoists(kernel, adapter, *, role):
    """Propose actual invariant operand motion across canonical output groups.

    The input is the actual compact counted program, not a requested access
    list. Only full-K singleton panels are extracted. Every request executes
    against the preceding result; final source/lifetime verification is still
    required. Tails outside the reused output loop keep their original loads.
    """
    from merlin.compiler.schedule.loop_schedule import LoopTransform, propose_loop_transforms

    if role not in ("lhs", "weight"):
        raise ValueError("retained hoist requires an explicit original operand role")
    crossed = "outer_n" if role == "lhs" else "outer_m"
    requests = []
    actual = kernel

    def candidate(body, path=()):
        for statement in body:
            if type(statement) is not Loop:
                continue
            current = (*path, statement.var)
            if statement.var == crossed:
                for index, panel in enumerate(statement.body):
                    if type(panel) is not Loop or panel.extent != 1:
                        continue
                    types = {type(adapter.project(c)) for c in _calls(panel.body)}
                    if types != {Load, Preload, Compute}:
                        continue
                    for load_index, child in enumerate(panel.body):
                        ops = tuple(adapter.project(c) for c in _calls((child,)))
                        if ops and all(type(op) is Load and op.role == role for op in ops):
                            return (
                                LoopTransform("hoist", (*current, panel.var), index=load_index),
                                LoopTransform("hoist", current, index=index),
                            )
            found = candidate(statement.body, current)
            if found is not None:
                return found
        return None

    while (selected := candidate(actual.body)) is not None:
        if len(requests) + len(selected) > 32:
            raise ValueError("retained operand hoists exceed the original finite transformation budget")
        actual = propose_loop_transforms(actual, selected, index_bits=adapter.index_bits)
        requests.extend(selected)
    return tuple(requests)


def _dense_retained_compute_nests(kernel, adapter):
    """Read independent M/K/N axes from original projected command operands."""
    from merlin.compiler.kernel_ir.ir import free_vars

    streams = []

    def discover(body, path=()):
        for statement in body:
            if type(statement) is not Loop:
                continue
            kinds = {type(adapter.project(call)) for call in _calls((statement,))}
            if kinds != {Preload, Compute}:
                discover(statement.body, (*path, statement.var))
                continue
            chain, tail = [statement], statement.body
            while len(tail) == 1 and type(tail[0]) is Loop:
                chain.append(tail[0])
                tail = tail[0].body
            if (
                len(chain) < 3
                or any(loop.extent != 1 for loop in chain[:-3])
                or len(tail) != 2
                or any(type(call) is not Call for call in tail)
            ):
                raise ValueError("resident compute interchange requires a perfect three-axis paired nest")
            preload, compute = (adapter.project(call) for call in tail)
            if type(preload) is not Preload or type(compute) is not Compute:
                raise ValueError("resident compute interchange cannot split a physical descriptor/compute pair")
            loops = chain[-3:]
            variables = {loop.var for loop in loops}
            coordinates = (compute.input_block[0], compute.input_block[1], preload.weight_block[1])
            names = {}
            for axis, coordinate in zip(("m", "k", "n"), coordinates, strict=True):
                selected = variables & free_vars(coordinate)
                if len(selected) != 1:
                    raise ValueError("resident compute axes are not independent original source coordinates")
                names[axis] = selected.pop()
            if set(names.values()) != variables or variables & free_vars(preload.weight_block[0]) != {names["k"]}:
                raise ValueError("resident compute input/weight reduction axes disagree")
            current = [next(axis for axis, name in names.items() if name == loop.var) for loop in loops]
            streams.append(
                ((*path, *(loop.var for loop in chain[:-3])), names, current, {loop.var: loop.extent for loop in loops})
            )

    discover(kernel.body)
    if not streams:
        raise ValueError("resident compute order has no complete original compute nest to transform")
    return streams


def dense_retained_compute_nests(kernel, adapter):
    """Inspect actual source M/K/N axes of perfect paired compute nests.

    Returns tuples of (enclosing path, axis-to-variable map, current axis
    order, variable-to-extent map). These are proposal coordinates, not a
    source, storage, dependency or target execution proof. Unknown effects
    and noncanonical descriptor/compute nests refuse through the same reader
    used by executable interchange and array-weight retention.
    """
    return tuple(_dense_retained_compute_nests(kernel, adapter))


def dense_retained_compute_order(kernel, adapter, *, order):
    """Propose actual adjacent interchanges of complete resident compute nests.

    Axes come from projected input/weight block coordinates, not loop names or
    descriptive attributes. Only perfect three-axis nests of adjacent physical
    PRELOAD/COMPUTE pairs are handled; singleton surrounding scopes stay put.
    Each chosen axis occurs once, so each output still visits K in increasing
    order. This is a proposal for the unchanged source and completion verifiers,
    never permission to reorder an operand staging or external observer.
    """
    from merlin.compiler.schedule.loop_schedule import LoopTransform, propose_loop_transforms

    if (
        type(order) is not tuple
        or len(order) != 3
        or any(type(axis) is not str for axis in order)
        or set(order) != {"m", "k", "n"}
    ):
        raise ValueError("resident compute order requires an explicit permutation of M/K/N")
    streams = _dense_retained_compute_nests(kernel, adapter)
    actual, requests = kernel, []
    for prefix, names, current, _extents in streams:
        for position, axis in enumerate(order):
            index = current.index(axis)
            while index > position:
                cursor = (*prefix, *(names[axis] for axis in current[:index]))
                request = LoopTransform("reorder", cursor)
                if len(requests) == 32:
                    raise ValueError("resident compute order exceeds original finite transformation budget")
                actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
                requests.append(request)
                current[index - 1], current[index] = current[index], current[index - 1]
                index -= 1
    return tuple(requests)


def dense_retained_compute_unroll(kernel, adapter, *, axis, factor, max_static_operations=4096):
    """Propose bounded unrolling of actual resident compute coordinates.

    Each request targets a semantic source axis, preserving the complete dynamic
    command sequence, descriptor adjacency and increasing reduction order. The
    ordinary typed schedule pipeline still proves source products, storage and
    completion afterward. This does not unroll loads, remove a drain, expand the
    full dynamic program, or infer a performance benefit from code size.
    """
    from merlin.compiler.schedule.loop_schedule import LoopTransform, propose_loop_transforms

    if axis not in ("m", "k", "n") or type(factor) is not int or not 2 <= factor <= 64:
        raise ValueError("resident compute unroll requires a source axis and explicit factor 2..64")
    streams = _dense_retained_compute_nests(kernel, adapter)
    actual, requests = kernel, []
    for prefix, names, current, extents in streams:
        if factor > extents[names[axis]]:
            raise ValueError("resident compute unroll factor exceeds an original source-axis extent")
        cursor = (*prefix, *(names[value] for value in current[: current.index(axis) + 1]))
        request = LoopTransform("unroll", cursor, factor=factor)
        if len(requests) == 32:
            raise ValueError("resident compute unroll exceeds original finite transformation budget")
        actual = propose_loop_transforms(
            actual, (request,), index_bits=adapter.index_bits, max_static_operations=max_static_operations
        )
        requests.append(request)
    return tuple(requests)


def complete_dense_retained_reduction(kernel, problem, block, resources, storage, adapter):
    """Form pure staging/reduction epochs from actual already transformed IR.

    All K tiles must fit the original assigned windows. Consecutive pure loads
    drain once before any reader; a complete pure compute region drains once
    before reuse/readout. Initializers and stores retain individual completion
    boundaries. This function proposes completion; it grants no execution law.
    """
    if block.k != problem.tiles[1] or not storage.admits(problem, block, resources):
        raise ValueError("retained reduction requires full K inside the original assigned capacity")
    used = {statement.var for statement in _loops(kernel.body)}
    serial = 0

    def epoch(body):
        nonlocal serial
        serial += 1
        name = "dense_retained_epoch_" + str(serial)
        if name in used:
            raise ValueError("retained completion scope collides with original loop")
        used.add(name)
        return Loop(name, 1, (*body, Call(DRAIN, ())))

    def form(body):
        result, pending = [], []
        previous = None

        def flush():
            if pending:
                result.append(epoch(tuple(pending)))
                pending.clear()

        for statement in body:
            kinds = {type(adapter.project(c)) for c in _calls((statement,))}
            kind = "loads" if kinds == {Load} else "compute" if kinds == {Preload, Compute} else None
            if kind is not None:
                if previous != kind:
                    flush()
                pending.append(statement)
            else:
                flush()
                if type(statement) is Loop:
                    result.append(replace(statement, body=form(statement.body)))
                elif kinds in ({Store}, {TiledIntegerStore}, {InitializeAccumulator}):
                    result.append(epoch((statement,)))
                else:
                    raise ValueError("retained completion cannot classify an unknown or unpaired effect")
            previous = kind
        flush()
        return tuple(result)

    result = replace(kernel, body=form(kernel.body))
    if check_structure(result):
        raise ValueError("retained completion did not preserve closed structured IR")
    check_integer_domain(result, adapter.index_bits)
    return result


def verify_dense_retained_reduction(
    kernel,
    problem,
    block,
    resources,
    storage,
    adapter,
    admission,
    *,
    static_budget,
    timeout_ms=5000,
    output_readout=None,
    bias_source=None,
):
    """Prove source cover and pure-epoch hazards, with ordered sum commitment.

    Repeated accumulator writes are permitted only in compute-only regions
    under the separate selected ordered-commit premise. The unchanged complete
    source checker proves exact output identity, increasing K, initialization,
    continuations, source residency, aliases, tails and final observers.
    """
    import time

    import z3

    from merlin.compiler.schedule.compact_verification import _project, verify_compact_dense

    if (
        type(admission) is not DenseRetainedReductionAdmission
        or admission.hardware_identity != adapter.configuration_identity
        or admission.numerical_policy_identity != adapter.numerical_policy_identity
    ):
        raise ValueError("retained reduction differs from selected ordered-commit capability")
    admission.__post_init__()
    if (
        block.k != problem.tiles[1]
        or not storage.admits(problem, block, resources)
        or check_structure(kernel)
        or type(static_budget) is not int
        or static_budget <= 0
        or sum(1 for _ in _calls(kernel.body)) > static_budget
        or type(timeout_ms) is not int
        or timeout_ms <= 0
    ):
        raise ValueError("retained reduction exceeds original full-K capacity, structure or checking bounds")
    check_integer_domain(kernel, adapter.index_bits)
    deadline = time.monotonic() + timeout_ms / 1000
    epochs, serial = [], 0

    def structure(body, ancestry=(), inside=False):
        nonlocal serial
        for statement in body:
            if type(statement) is Loop:
                serial += 1
                scope = (serial, statement.var, statement.extent)
                drains = [s for s in statement.body if type(s) is Call and s.instr == DRAIN]
                if drains:
                    if inside or statement.extent != 1 or len(drains) != 1 or statement.body[-1] != Call(DRAIN, ()):
                        raise ValueError("retained epoch requires exactly one final full drain")
                    epochs.append((*ancestry, scope))
                    structure(statement.body[:-1], (*ancestry, scope), True)
                else:
                    structure(statement.body, (*ancestry, scope), inside)
            elif not inside:
                raise ValueError("every retained effect must belong to a completed epoch")

    structure(kernel.body)
    if not epochs:
        raise ValueError("retained reduction has no completion epochs")
    sites = _project(_serial_projection(kernel), adapter, fold_singleton_loops=False)

    def expression(value, env):
        if type(value) is Const:
            return z3.IntVal(value.value)
        if type(value) is int:
            return z3.IntVal(value)
        if type(value) is Var:
            return env[value.name]
        if type(value) is Sum:
            return z3.Sum(*(expression(v, env) for v in value.terms))
        if type(value) is Scaled:
            return expression(value.expr, env) * value.factor
        if type(value) is Select:
            return z3.If(env[value.var] == value.equals, expression(value.then, env), expression(value.other, env))
        raise ValueError("unknown retained address expression")

    def disjoint(a, b, outer):
        def view(site):
            if type(site.operation) is InitializeAccumulator:
                return storage.accumulator
            return storage.lhs if site.operation.role == "lhs" else storage.rhs

        va, vb = view(a), view(b)
        if va.backing != vb.backing:
            return
        ae = {name: z3.Int(f"retained-a:{scope}:{name}") for scope, name, _ in a.scopes}
        be = {name: z3.Int(f"retained-b:{scope}:{name}") for scope, name, _ in b.scopes}
        constraints = [
            z3.And(env[name] >= 0, env[name] < extent)
            for site, env in ((a, ae), (b, be))
            for _, name, extent in site.scopes
        ]
        constraints.extend(ae[name] == be[name] for _, name, _ in outer)
        if a.ordinal == b.ordinal:
            local = [ae[name] != be[name] for scope, name, _ in a.scopes if scope not in {s[0] for s in outer}]
            if not local:
                return
            constraints.append(z3.Or(*local))
        al = expression(a.operation.row, ae) * va.unit_bytes
        bl = expression(b.operation.row, be) * vb.unit_bytes
        ah = al + expression(a.operation.rows, ae) * va.unit_bytes
        bh = bl + expression(b.operation.rows, be) * vb.unit_bytes
        remaining = int((deadline - time.monotonic()) * 1000)
        if remaining <= 0:
            raise ValueError("retained staging proof budget exhausted")
        solver = z3.Solver()
        solver.set(timeout=remaining)
        solver.add(*constraints, al < bh, bl < ah)
        if solver.check() != z3.unsat:
            raise ValueError("retained staging or initialization has overlapping in-flight destinations")

    summaries = []
    for outer in epochs:
        selected = [s for s in sites if outer[-1][0] in {scope for scope, _, _ in s.scopes}]
        kinds = {type(s.operation) for s in selected}
        if not selected or kinds not in (
            {Load},
            {Preload, Compute},
            {InitializeAccumulator},
            {Store},
            {TiledIntegerStore},
        ):
            raise ValueError("retained epoch must contain only staging, paired computation, initialization or readout")

        def count(site):
            return prod(extent for scope, _, extent in site.scopes if scope not in {s[0] for s in outer})

        loads = sum(count(s) for s in selected if type(s.operation) is Load)
        pairs = sum(count(s) for s in selected if type(s.operation) is Compute)
        if loads > admission.max_issued_loads or pairs > admission.max_issued_compute_pairs:
            raise ValueError("retained epoch exceeds original selected issued-command budget")
        if kinds == {Load}:
            if any(s.operation.role not in ("lhs", "weight") for s in selected):
                raise ValueError("retained staging has an unknown source role")
            for index, a in enumerate(selected):
                for b in selected[index:]:
                    disjoint(a, b, outer)
        elif kinds == {Preload, Compute}:
            if len(selected) % 2 or any(
                type(a.operation) is not Preload or type(b.operation) is not Compute or a.scopes != b.scopes
                for a, b in zip(selected[::2], selected[1::2], strict=True)
            ):
                raise ValueError("retained reduction must not split or overwrite a physical preload/compute pair")
        elif kinds == {InitializeAccumulator}:
            if sum(count(s) for s in selected) > admission.max_issued_initializations:
                raise ValueError("retained initializer exceeds its selected completion budget")
            for index, a in enumerate(selected):
                for b in selected[index:]:
                    disjoint(a, b, outer)
        elif kinds & {Store, TiledIntegerStore}:
            readouts = sum(count(s) for s in selected)
            if readouts > admission.max_issued_readouts:
                raise ValueError("retained readout exceeds selected store completion budget")
            if (
                readouts > 1
                and type(output_readout) is DenseTiledIntegerReadout
                and len(set(output_readout.column_scale_bits)) != 1
            ):
                raise ValueError("varying per-tile readout needs an independent configuration snapshot mapping")
        summaries.append(
            dict(
                compute_pairs=pairs,
                future_or_initial_loads=loads,
                initialization=kinds == {InitializeAccumulator},
                readout=bool(kinds & {Store, TiledIntegerStore}),
                readout_commands=sum(count(s) for s in selected) if kinds & {Store, TiledIntegerStore} else 0,
                issue_order=tuple(type(s.operation).__name__ for s in selected),
                completion=DRAIN,
            )
        )
    remaining = int((deadline - time.monotonic()) * 1000)
    if remaining <= 0:
        raise ValueError("retained source proof budget exhausted")
    source = verify_compact_dense(
        _serial_projection(kernel),
        problem,
        resources,
        storage,
        adapter,
        static_budget=static_budget,
        timeout_ms=remaining,
        output_readout=output_readout,
        bias_source=bias_source,
    )
    return dict(
        schema="merlin.dense_retained_reduction.v1",
        kernel_sha256=kernel.digest(),
        source=source,
        epochs=summaries,
        admission=asdict(admission),
        dynamic_epoch_drains=sum(prod(extent for _, _, extent in scopes) for scopes in epochs),
        overlapped_compute_pairs=0,
        qualification="conditional independent ordered accumulator commits; physical execution not established",
        target_execution="not_established",
    )
