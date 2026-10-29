"""Final-K readout frontiers with explicit causal accumulator completion.

Actual counted PRELOAD/COMPUTE pairs remain adjacent. Each original output store
follows its complete final K run in the same epoch; the full accumulator view
and all input staging epochs stay intact. The target must separately establish
producer retirement, pending-write read readiness, store snapshot/backpressure
and the final drain. This module supplies no cycle spacing or execution grant.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compile.scheduling.block_schedule import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compile.scheduling.compact_dense import _digest
from merlin.compile.scheduling.dense_contracts import TiledIntegerStore
from merlin.compile.scheduling.dense_panel_pipeline import (
    DRAIN,
    DensePanelPipelineResult,
    DenseReadyReadoutAdmission,
    DenseRetainedReductionAdmission,
    _calls,
    _remap,
    _verify_dense_epochs,
)
from merlin.compile.scheduling.dense_segmented_pipeline import DenseSegmentedPipelineResult, _assignment
from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms
from merlin.compile.scheduling.tile_load_coalescing import _linear, _plain
from merlin.sched.ir import Call, Kernel, Loop, check_structure
from merlin.sched.ir.expr import free_vars
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import _loop_at, _splice


def _chain(statement, count):
    loops = []
    for index in range(count):
        if type(statement) is not Loop:
            raise ValueError("ready readout requires perfect counted compute and output domains")
        loops.append(statement)
        if index + 1 < count:
            if len(statement.body) != 1:
                raise ValueError("ready readout cannot move effects across an imperfect nest")
            statement = statement.body[0]
    return loops, statement.body


def _strip_store_epochs(statement):
    """Remove only singleton store completion wrappers, substituting their zero."""
    if type(statement) is Call:
        if statement == Call(DRAIN, ()):
            raise ValueError("ready readout encountered a drain outside its original epoch")
        return (statement,)
    if type(statement) is not Loop:
        raise ValueError("unknown ready readout statement")
    if statement.body and statement.body[-1] == Call(DRAIN, ()):
        if statement.extent != 1:
            raise ValueError("ready readout cannot erase a counted completion epoch")
        return tuple(s for child in _remap(statement.body[:-1], statement.var, None)
                     for s in _strip_store_epochs(child))
    return (replace(statement, body=tuple(s for child in statement.body for s in _strip_store_epochs(child))),)


@dataclass(frozen=True)
class DenseReadyInputPlacement:
    """Capacity proposal for the retained-weight ready consumer, not a law.

    The next LHS stage overlaps the current panel's last users. Its two
    generations need separate slots; invariant weights and the drained output
    each need one. Actual projected row windows must corroborate this geometry
    before the consumer can emit it. Original ownership and execution laws are
    still required. No storage is reassigned between roles.
    """

    choice: object
    lhs_rows: int
    rhs_rows: int
    accumulator_rows: int
    lhs_slot_offset: int
    context_sha256: str

    @property
    def block(self):
        return self.choice.block

    def validate_for(self, problem, resources, storage):
        expected = _ready_input_placement(problem, self.choice, resources, storage)
        if self != expected:
            raise ValueError("ready input placement changed its original choice, ownership or footprint")


def _ready_input_placement(problem, choice, resources, storage):
    from .dense_panel_pipeline import DenseRetainedChoice

    if (type(choice) is not DenseRetainedChoice or choice.role != "weight"
            or choice.outer_order != ("n", "m")
            or choice.block.k != problem.tiles[1] or problem.tiles[0] // choice.block.m < 2
            or not storage.admits(problem, choice.block, resources)):
        raise ValueError("ready input placement needs an advancing unbiased retained-weight full-K choice")
    lhs, rhs, acc = storage.row_counts(problem, choice.block)
    offset = (lhs + storage.lhs.alignment - 1) // storage.lhs.alignment * storage.lhs.alignment
    if (offset + lhs > storage.lhs.capacity
            or ((offset + lhs) * storage.lhs.unit_bytes + rhs * storage.rhs.unit_bytes
                > resources.operand_bytes)):
        raise ValueError("ready input placement needs two owned LHS slots and one RHS panel")
    return DenseReadyInputPlacement(choice, lhs, rhs, acc, offset,
        _digest((asdict(problem), asdict(resources), asdict(storage))))


def dense_ready_input_choices(problem, resources, storage):
    """Filter bounded retained choices by their actual consumer's live roles.

    This explicit option keeps the ordinary one-panel selector unchanged.
    Every returned placement consumes two advancing input panels, one retained
    weight panel and one accumulator panel. The caller selects a choice, builds
    its ordinary retained full-K plan, and passes this SAME placement to
    ``schedule_dense_pipeline(input_pipeline_placement=...)``. No dynamic
    command trace is expanded and no numerical or completion law is supplied.
    """
    from .dense_panel_pipeline import dense_retained_choices

    choices = []
    for choice in dense_retained_choices(problem, resources, storage, roles=("weight",)):
        try:
            placement = _ready_input_placement(problem, choice, resources, storage)
        except ValueError:
            continue
        choices.append(placement)
    return tuple(choices)


def _input_group_windows(body, adapter, domains):
    """Bound actual typed accesses without enumerating counted instances."""
    from .dense_readout_overlap import _affine_window_bounds

    windows = {}

    def add(role, row, count, bounds):
        start = _affine_window_bounds(row, bounds)[0]
        end = _affine_window_bounds(row + count, bounds)[1]
        old = windows.get(role, (start, end))
        windows[role] = min(start, old[0]), max(end, old[1])

    def walk(statements, bounds):
        for statement in statements:
            if type(statement) is Loop:
                walk(statement.body, {**bounds, statement.var: (0, statement.extent - 1)})
            elif statement != Call(DRAIN, ()):
                operation = adapter.project(statement)
                if type(operation) is Load:
                    add("lhs_stage" if operation.role == "lhs" else "weight", operation.row, operation.rows, bounds)
                elif type(operation) is Compute:
                    add("lhs_read", operation.input_row, operation.rows, bounds)
                elif type(operation) is Preload:
                    if operation.weight_row is not None:
                        add("weight", operation.weight_row, operation.weight_rows, bounds)
                    add("accumulator", operation.accumulator_row, operation.rows, bounds)
                elif type(operation) in (Store, TiledIntegerStore):
                    add("accumulator", operation.accumulator_row, operation.rows, bounds)
                else:
                    raise ValueError("unknown effect in ready input footprint")

    walk(body, domains)
    return windows


def propose_dense_ready_readout(kernel, problem, adapter, *, final_k_prefix=False, array_prefix_admission=None):
    """Propose source-bound final reduction followed by each original store.

    Nested outer counted groups and primitive/output tails stay counted. Only a
    pure compute epoch immediately followed by one complete original store nest
    is fused. Partial store partitions and unrolled/imperfect compute nests
    decline rather than silently selecting another schedule. Original tiled
    readout descriptors stay intact; varying coefficients require a separately
    selected configuration snapshot in the checked consumer. Complete source
    and actual epoch checks MUST follow this proposal. The explicit
    ``final_k_prefix`` alternative partitions the original K,N,M nest at its
    last K slice. Only its pure prefix may consume the separate typed array
    law; every final-K pair remains fresh before its original causal store.
    """
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("ready readout needs original closed counted commands")
    if type(final_k_prefix) is not bool or array_prefix_admission is not None and not final_k_prefix:
        raise ValueError("array prefix requires an explicit counted final-K frontier")
    check_integer_domain(kernel, adapter.index_bits)

    def expose_epochs(body):
        # A full-K compact panel may wrap completed staging/compute epochs in
        # one singleton reduction group, with its stores immediately outside.
        # Substitute its actual zero index before exposing those siblings.
        # Primitive M/N/K loops and every real completion epoch remain intact.
        result = []
        for statement in body:
            if type(statement) is Loop:
                statement = replace(statement, body=expose_epochs(statement.body))
                if (statement.extent == 1 and statement.body
                        and statement.body[-1] != Call(DRAIN, ())
                        and any(type(child) is Loop and child.body
                                and child.body[-1] == Call(DRAIN, ())
                                and any(c != Call(DRAIN, ()) and type(adapter.project(c)) is Compute
                                        for c in _calls(child.body)) for child in statement.body)):
                    result.extend(_remap(statement.body, statement.var, None))
                    continue
            result.append(statement)
        return tuple(result)

    exposed = replace(kernel, body=expose_epochs(kernel.body))

    def kinds(statement):
        return {type(adapter.project(c)) for c in _calls((statement,)) if c != Call(DRAIN, ())}

    paths = []

    def discover(body, path=()):
        for index, statement in enumerate(body):
            if type(statement) is not Loop:
                continue
            if statement.body and statement.body[-1] == Call(DRAIN, ()):
                if (kinds(statement) == {Preload, Compute} and index + 1 < len(body)
                        and kinds(body[index + 1]) in ({Store}, {TiledIntegerStore})):
                    paths.append((*path, statement.var))
            else:
                discover(statement.body, (*path, statement.var))

    discover(exposed.body)
    if not paths or len(paths) > 32:
        raise ValueError("ready readout needs 1..32 final pure compute/store regions")
    actual, records = exposed, []
    for path in paths:
        epoch = _loop_at(actual.body, path)
        if epoch.extent != 1 or len(epoch.body) != 2:
            raise ValueError("ready readout needs one complete final compute nest and its original drain")
        parent = _loop_at(actual.body, path[:-1]) if path[:-1] else None
        siblings = parent.body if parent is not None else actual.body
        index = next(i for i, s in enumerate(siblings) if s is epoch)
        store_tree = _strip_store_epochs(siblings[index + 1])
        if len(store_tree) != 1:
            raise ValueError("ready readout needs one complete store domain, not partial partitions")
        loops, pair = _chain(epoch.body[0], 3)
        stores, store_calls = _chain(store_tree[0], 2)
        if (len(pair) != 2 or len(store_calls) != 1
                or any(type(c) is not Call or not _plain(c) for c in (*pair, *store_calls))):
            raise ValueError("ready readout requires plain adjacent physical pairs and one original store")
        pre, compute = (adapter.project(c) for c in pair)
        store = adapter.project(store_calls[0])
        if (type(pre) is not Preload or type(compute) is not Compute
                or type(store) not in (Store, TiledIntegerStore)):
            raise ValueError("ready readout cannot split descriptors or vary output numerical configuration")
        variables = {s.var for s in loops}
        names = {}
        for axis, coordinate in zip(("m", "k", "n"),
                                    (compute.input_block[0], compute.input_block[1], pre.weight_block[1]), strict=True):
            selected = variables & free_vars(coordinate)
            if len(selected) != 1:
                raise ValueError("ready readout needs independent original M/K/N coordinates")
            names[axis] = selected.pop()
        if set(names.values()) != variables or variables & free_vars(pre.weight_block[0]) != {names["k"]}:
            raise ValueError("ready readout reduction coordinates disagree")
        extents = {s.var: s.extent for s in loops}
        store_variables = {s.var for s in stores}
        store_names = {}
        for axis, coordinate in (("m", store.dram_row), ("n", store.dram_col)):
            selected = store_variables & free_vars(coordinate)
            if len(selected) != 1:
                raise ValueError("ready readout store coordinates are not independent original M/N axes")
            store_names[axis] = selected.pop()
        if set(store_names.values()) != store_variables or any(
            next(s.extent for s in stores if s.var == store_names[a]) != extents[names[a]] for a in ("m", "n")
        ):
            raise ValueError("ready readout store domain differs from the complete final compute output domain")
        renamed = store_calls
        for axis in ("m", "n"):
            renamed = _remap(renamed, store_names[axis], names[axis])
        renamed_store = adapter.project(renamed[0])
        # Exact original scalar descriptors, including piecewise output tails.
        # Only affine logical coordinates need algebraic normalization.
        if (pre.accumulator_row != renamed_store.accumulator_row or pre.rows != renamed_store.rows
                or pre.cols != renamed_store.cols
                or _linear(renamed_store.dram_row) != _linear(compute.input_block[0] * problem.tile_m)
                or _linear(renamed_store.dram_col) != _linear(pre.weight_block[1] * problem.tile_n)
                or names["k"] in (free_vars(pre.accumulator_row) | free_vars(pre.rows) | free_vars(pre.cols))):
            raise ValueError("ready readout does not match the exact original producer ACC/output footprint")
        if adapter.encode(renamed_store) != renamed[0] or any(adapter.encode(adapter.project(c)) != c for c in pair):
            raise ValueError("ready readout requires complete descriptor/effect roundtrips")
        current = [next(a for a, name in names.items() if name == s.var) for s in loops]
        requests = []
        array_changes = ()
        if final_k_prefix:
            if current != ["k", "n", "m"] or loops[0].extent < 2:
                raise ValueError("final-K prefix requires the original full K,N,M run with at least two K slices")
            k, n, m = loops
            request = LoopTransform("partition", (*path, k.var), cuts=(k.extent - 1,))
            actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
            requests.append(request)
            epoch = _loop_at(actual.body, path)
            prefix, final, drain = epoch.body
            if array_prefix_admission is not None and m.extent > 1:
                from .dense_array_retention import propose_dense_array_retention

                # Only the pure prefix enters the existing fresh/STAY owner.
                # Every final-K consume remains fresh; no store crosses STAY.
                isolated = prefix
                for depth in range(len(path), 0, -1):
                    ancestor = _loop_at(actual.body, path[:depth])
                    isolated = replace(ancestor, body=(isolated, Call(DRAIN, ())))
                proposal = propose_dense_array_retention(replace(actual, body=(isolated,)), adapter,
                                                        array_prefix_admission)
                proposal.revalidate(adapter, array_prefix_admission)
                prefix = _loop_at(proposal.kernel.body, (*path, prefix.var))
                array_changes = proposal.changes
            (last, n, m), final_pair = _chain(final, 3)
            frontier = replace(last, body=(replace(n, body=(replace(m, body=(*final_pair, *renamed)),)),))
            replacement = replace(epoch, body=(prefix, frontier, drain))
        else:
            for position, axis in enumerate(("m", "n", "k")):
                cursor = current.index(axis)
                while cursor > position:
                    request = LoopTransform("reorder", (*path, *(names[a] for a in current[:cursor])))
                    actual = propose_loop_transforms(actual, (request,), index_bits=adapter.index_bits)
                    requests.append(request)
                    current[cursor - 1], current[cursor] = current[cursor], current[cursor - 1]
                    cursor -= 1
            epoch = _loop_at(actual.body, path)
            (m, n, k), _ = _chain(epoch.body[0], 3)
            frontier = replace(m, body=(replace(n, body=(k, *renamed)),))
            replacement = replace(epoch, body=(frontier, Call(DRAIN, ())))
        parent = _loop_at(actual.body, path[:-1]) if path[:-1] else None
        siblings = parent.body if parent is not None else actual.body
        rewritten = (*siblings[:index], replacement, *siblings[index + 2:])
        actual = (replace(actual, body=_splice(actual.body, path[:-1], (replace(parent, body=rewritten),)))
                  if parent is not None else replace(actual, body=rewritten))
        records.append(dict(path=path, loop_transforms=[asdict(r) for r in requests],
                            final_k_tiles=k.extent, output_tiles=m.extent * n.extent,
                            final_k_prefix=final_k_prefix, array_prefix=tuple(array_changes),
                            geometry="unchanged original ACC and operand views",
                            frontier="each original store immediately follows its complete final K paired run",
                            completion="one original full drain after the complete frontier"))
    if check_structure(actual):
        raise ValueError("ready readout changed original binder custody")
    if array_prefix_admission is not None and not any(row["array_prefix"] for row in records):
        raise ValueError("final-K array prefix needs at least one reusable M run")
    check_integer_domain(actual, adapter.index_bits)
    return actual, records


def propose_dense_ready_input_pipeline(kernel, adapter, storage, *, row_offset, placement=None):
    """Rotate only advancing LHS panels across their actual completed users.

    A counted group must contain an LHS staging epoch and a complete
    compute/ready-store epoch, optionally preceded by separately drained bias
    initialization scopes. Those scopes keep every original call and drain
    before their current compute; they never enter a mixed epoch. The next
    input staging moves into the compute/ready-store epoch,
    in a disjoint explicitly owned LHS slot. Its original drain closes every
    input read, future load and pending store before the same ACC rows or an
    input slot are reused. Hoisted RHS generations and ACC addresses stay as
    supplied. Tails outside this exact counted group retain their own epochs.
    This proposal needs the separate load/compute/readout conjunction and the
    actual-address epoch checker; capacity alone supplies neither permission.
    """
    if (type(kernel) is not Kernel or check_structure(kernel)
            or type(row_offset) is not int or row_offset <= 0
            or row_offset % storage.lhs.alignment):
        raise ValueError("input pipeline requires an aligned explicit private LHS slot offset")
    names = {arg.name for arg in kernel.args} | {loop.var for loop in _all_loops(kernel.body)}
    records = []

    def fresh(prefix):
        while prefix in names:
            prefix += "_"
        names.add(prefix)
        return prefix

    def kinds(body):
        return {type(adapter.project(c)) for c in _calls(body) if c != Call(DRAIN, ())}

    def epoch(statement):
        return (type(statement) is Loop and statement.extent == 1 and statement.body
                and statement.body[-1] == Call(DRAIN, ())
                and all(c != Call(DRAIN, ()) for c in _calls(statement.body[:-1])))

    def initialization(statement):
        # Preserve the existing separately completed initializer scopes. A
        # plain initializer or a mixed/undrained tree cannot cross a stage.
        if type(statement) is not Loop:
            return False
        if epoch(statement):
            return kinds(statement.body[:-1]) == {InitializeAccumulator}
        return bool(statement.body) and all(initialization(s) for s in statement.body)

    def drains(statement):
        if type(statement) is Loop:
            return statement.extent * sum(drains(s) for s in statement.body)
        return int(statement == Call(DRAIN, ()))

    def shifted(body, offset):
        result = []
        for statement in body:
            if type(statement) is Loop:
                result.append(replace(statement, body=shifted(statement.body, offset)))
            else:
                operation = adapter.project(statement)
                if (type(operation) is Load and operation.role == "lhs"
                        or type(operation) is Compute):
                    if adapter.encode(operation) != statement:
                        raise ValueError("input slot relocation requires complete physical effect roundtrips")
                    field = "row" if type(operation) is Load else "input_row"
                    operation = replace(operation, **{field: getattr(operation, field) + offset})
                    statement = adapter.encode(operation)
                    if adapter.project(statement) != operation:
                        raise ValueError("input slot relocation changed its physical effect")
                result.append(statement)
        return tuple(result)

    def walk(body, path=(), domains=None):
        domains = {} if domains is None else domains
        result = []
        for statement in body:
            if type(statement) is not Loop:
                result.append(statement)
                continue
            parts = statement.body
            if (statement.extent >= 2 and len(parts) >= 2
                    and all(initialization(s) for s in parts[:-2])
                    and all(epoch(s) for s in parts[-2:])
                    and kinds(parts[-2].body) == {Load}
                    and all(adapter.project(c).role == "lhs" for c in _calls(parts[-2].body[:-1]))
                    and kinds(parts[-1].body) in ({Preload, Compute, Store},
                                                {Preload, Compute, TiledIntegerStore})):
                initializers, (stage, current) = parts[:-2], parts[-2:]
                initialization_drains = statement.extent * sum(drains(s) for s in initializers)
                windows = _input_group_windows((stage, current), adapter,
                    {**domains, statement.var: (0, statement.extent - 1)})
                if placement is not None:
                    if type(placement) is not DenseReadyInputPlacement or row_offset != placement.lhs_slot_offset:
                        raise ValueError("ready input emission must consume the selected placement")
                    lhs_window = (storage.lhs.base, storage.lhs.base + placement.lhs_rows)
                    if (windows.get("lhs_stage") != lhs_window or windows.get("lhs_read") != lhs_window
                            or any(role not in windows or windows[role][0] < view.base
                                   or windows[role][1] > view.base + rows
                                   for role, view, rows in (
                                       ("weight", storage.rhs, placement.rhs_rows),
                                       ("accumulator", storage.accumulator, placement.accumulator_rows)))):
                        raise ValueError("actual ready input accesses disagree with the selected role footprints")
                period = fresh("dense_input_period")

                def selected(tree, slot, new, scale, offset):
                    contents = _remap(tree.body[:-1], tree.var, None)
                    contents = _remap(contents, statement.var, new, scale, offset)
                    return shifted(contents, slot * row_offset)

                def advance(slot, new, scale, offset):
                    return (*_remap(initializers, statement.var, new, scale, offset),
                        Loop(fresh("dense_input_epoch"), 1, (
                        *selected(stage, 1 - slot, new, scale, offset + 1),
                        *selected(current, slot, new, scale, offset), Call(DRAIN, ()))))

                result.append(Loop(fresh("dense_input_prologue"), 1,
                                   (*selected(stage, 0, None, 1, 0), Call(DRAIN, ()))))
                pairs, tail = divmod(statement.extent - 1, 2)
                if pairs:
                    result.append(Loop(period, pairs, (*advance(0, period, 2, 0),
                                                        *advance(1, period, 2, 1))))
                if tail:
                    result.extend(advance(0, None, 1, 2 * pairs))
                result.extend(_remap(initializers, statement.var, None, 1, statement.extent - 1))
                result.append(Loop(fresh("dense_input_epilogue"), 1, (
                    *selected(current, (statement.extent - 1) % 2, None, 1, statement.extent - 1),
                    Call(DRAIN, ()))))
                records.append(dict(path=(*path, statement.var), groups=statement.extent,
                                    lhs_slot_offsets=(0, row_offset), rhs_slots=1,
                                    actual_role_windows=windows,
                                    accumulator="same original rows reused only after full drain",
                                    initializer_scopes=len(initializers),
                                    initialization="all original separate completion scopes retained before current compute",
                                    live_lhs_generations=2,
                                    preserved_initialization_drains=initialization_drains,
                                    remaining_drains=statement.extent + 1 + initialization_drains))
            else:
                result.append(replace(statement, body=walk(parts, (*path, statement.var),
                    {**domains, statement.var: (0, statement.extent - 1)})))
        return tuple(result)

    actual = replace(kernel, body=walk(kernel.body))
    if not records:
        raise ValueError("input pipeline needs a counted advancing LHS staging/ready-frontier group")
    if check_structure(actual):
        raise ValueError("input pipeline changed counted source binder custody")
    check_integer_domain(actual, adapter.index_bits)
    return actual, records


def _all_loops(body):
    for statement in body:
        if type(statement) is Loop:
            yield statement
            yield from _all_loops(statement.body)


@dataclass(frozen=True)
class DenseReadyReadoutCheckingProjection:
    """Exact deterministic transform transport, usable only after actual hazards."""

    original_kernel: Kernel
    problem: object
    canonical_load_runs: bool = False
    input_pipeline_row_offset: int | None = None
    input_pipeline_placement: DenseReadyInputPlacement | None = None
    final_k_prefix: bool = False
    array_prefix_admission: object = None

    def _expected(self, adapter, storage, *, retained=False):
        if (self.input_pipeline_placement is not None
                and self.input_pipeline_row_offset != self.input_pipeline_placement.lhs_slot_offset):
            raise ValueError("ready input emission must consume the selected placement")
        expected, regions = propose_dense_ready_readout(self.original_kernel, self.problem, adapter,
            final_k_prefix=self.final_k_prefix,
            array_prefix_admission=self.array_prefix_admission if retained else None)
        input_pipeline = []
        if self.input_pipeline_row_offset is not None:
            expected, input_pipeline = propose_dense_ready_input_pipeline(
                expected, adapter, storage, row_offset=self.input_pipeline_row_offset,
                placement=self.input_pipeline_placement)
        if type(self.canonical_load_runs) is not bool:
            raise ValueError("explicit exact ready-readout load representation required")
        canonical_changes = []
        if self.canonical_load_runs:
            from merlin.compile.scheduling.panel_tile_loads import _canonical_load_runs

            def original_load(call):
                if call == Call(DRAIN, ()):
                    return None
                operation = adapter.project(call)
                return operation if type(operation) is Load else None

            expected, canonical_changes = _canonical_load_runs(expected, original_load)
        return expected, regions, input_pipeline, canonical_changes

    def hazard_kernel(self, actual, adapter, storage):
        """Check exact prefix STAY, then strengthen its unchanged scratch reads.

        The existing typed array owner rewrites ONLY pure K-prefix pairs. All
        loads, input reads, ACC updates, final fresh pairs, stores and drains
        retain their sequence/addresses. Checking fresh RHS scratch reads is
        stronger than reading their captured array values: the epoch owner
        still proves immutable RHS rows through every original consume. No
        STAY overlaps a store and no target store-preserves-array law is used.
        """
        if self.array_prefix_admission is None:
            return actual, None
        expected, regions, _, _ = self._expected(adapter, storage, retained=True)
        if actual != expected or actual.digest() != expected.digest():
            raise ValueError("actual array prefix differs from the exact pure retained-prefix transformation")
        fresh, _, _, _ = self._expected(adapter, storage)
        return fresh, dict(actual_kernel_sha256=actual.digest(), fresh_read_kernel_sha256=fresh.digest(),
            admission=asdict(self.array_prefix_admission), regions=regions,
            premise="pure prefix only; same input/ACC/store accesses and issue order; stronger immutable RHS reads",
            final_k="all original final-K consumes are fresh; original causal stores and full drain")

    def checking_kernel(self, actual, adapter, storage):
        expected, regions, input_pipeline, canonical_changes = self._expected(adapter, storage)
        if actual != expected or actual.digest() != expected.digest():
            raise ValueError("ready readout actual commands differ from the exact original transformation")
        return self.original_kernel, dict(
            rule="exact_final_k_producer_store_frontier_after_actual_hazards",
            original_kernel_sha256=self.original_kernel.digest(), actual_kernel_sha256=actual.digest(),
            regions=regions, canonical_load_runs=canonical_changes,
            input_pipeline=input_pipeline,
            geometry="original owned views; only explicit checked LHS slot relocation",
            correspondence="same complete counted products/stores; only independent outputs interchanged",
            premise="all overlapping producer instances precede each store; no later ACC reuse in epoch",
            emission="actual frontier commands, never the checking projection",
        )


@dataclass(frozen=True)
class DenseReadyReadoutResult:
    original: DenseSegmentedPipelineResult | DensePanelPipelineResult
    kernel: Kernel
    realization: dict
    admission: DenseReadyReadoutAdmission
    problem: object
    checked_identity: str
    owner_sha256: str
    _adapter: object
    _storage: object
    input_pipeline_row_offset: int | None = None
    input_pipeline_placement: DenseReadyInputPlacement | None = None
    array_prefix_admission: object = None

    @property
    def source_identity(self):
        return self.original.source_identity

    @property
    def output_readout(self):
        return self.original.output_readout

    @property
    def bias_source(self):
        return self.original.bias_source

    @property
    def readout_qualification_identity(self):
        return self.original.readout_qualification_identity

    def revalidate(self):
        self.original.revalidate()
        if (self.input_pipeline_placement is not None
                and self.input_pipeline_row_offset != self.input_pipeline_placement.lhs_slot_offset):
            raise ValueError("ready input emission must consume the selected placement")
        expected, _ = propose_dense_ready_readout(self.original.kernel, self.problem, self._adapter,
            final_k_prefix=self.array_prefix_admission is not None, array_prefix_admission=self.array_prefix_admission)
        if self.input_pipeline_row_offset is not None:
            expected, _ = propose_dense_ready_input_pipeline(
                expected, self._adapter, self._storage, row_offset=self.input_pipeline_row_offset,
                placement=self.input_pipeline_placement)
        if (self.kernel != expected or self.kernel.digest() != self.realization["kernel_sha256"]
                or self.checked_identity != _digest((self.original.kernel.digest(), self.kernel.digest(),
                                                    asdict(self.admission), asdict(self.problem),
                                                    repr(self._adapter), asdict(self._storage),
                                                    None if self.input_pipeline_placement is None
                                                    else asdict(self.input_pipeline_placement),
                                                    None if self.array_prefix_admission is None
                                                    else asdict(self.array_prefix_admission)))
                or sha256(Path(__file__).read_bytes()).hexdigest() != self.owner_sha256):
            raise ValueError("ready readout original source, actual commands, capability or owner changed")

    def materialize_scf(self):
        from merlin.compile.scheduling.scf_commands import materialize_scf_commands

        self.revalidate()
        module = materialize_scf_commands(self.kernel)
        self.revalidate()
        return module

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        """Emit actual checked effects with the same encoder and full-drain callback."""
        from .dense_panel_pipeline import DRAIN
        from .llvm_program import emit_index_effect_program

        self.revalidate()
        report = emit_index_effect_program(self.kernel, control, lower_effect,
            complete_epoch=complete_epoch, completion_instruction=DRAIN)
        self.revalidate()
        return report

    def emit_llvm(self, builder, lower_call):
        from merlin.compile.scheduling.llvm_program import FnBuilderIndexAdapter
        from merlin.compile.scheduling.scf_commands import emit_scf_commands

        module = self.materialize_scf()
        report = emit_scf_commands(module, FnBuilderIndexAdapter(builder), lower_call)
        self.revalidate()
        return report


def _ready_execution_bound(result, problem, block, resources, storage):
    if type(result) not in (DenseSegmentedPipelineResult, DensePanelPipelineResult):
        raise ValueError("ready readout requires its original checked owner")
    result.revalidate()
    original = result.admission
    if type(result) is DenseSegmentedPipelineResult:
        _assignment(problem, block, resources, storage)
        return replace(original.overlap,
                        max_issued_compute_pairs=min(original.overlap.max_issued_compute_pairs,
                                                     original.reduction.max_issued_compute_pairs),
                        max_issued_loads=min(original.overlap.max_issued_loads, original.reduction.max_issued_loads),
                        max_issued_readouts=min(original.overlap.max_issued_readouts,
                                               original.reduction.max_issued_readouts))
    else:
        if (type(original) is not DenseRetainedReductionAdmission
                or block.k != problem.tiles[1]
                or not storage.admits(problem, block, resources)):
            raise ValueError("ready readout requires the original complete full-K storage and reduction owner")
        return original


def schedule_dense_ready_readout(result, problem, block, resources, storage, adapter, admission, *,
                                 static_budget=128, timeout_ms=5000, input_pipeline_row_offset=None,
                                 input_pipeline_placement=None, array_prefix_admission=None):
    """Consume a checked segmented/full-K result and separate causal law.

    The final compute/store frontier retains original ownership and reduction.
    An explicit ``input_pipeline_row_offset`` additionally consumes the selected
    disjoint-load conjunction and two LHS slots inside the original owned view.
    Complete drains retain the original single ACC region between groups.
    """
    if type(admission) is not DenseReadyReadoutAdmission:
        raise ValueError("ready readout requires its original checked owner and separate typed capability")
    if input_pipeline_placement is not None:
        if type(input_pipeline_placement) is not DenseReadyInputPlacement or input_pipeline_row_offset is not None:
            raise ValueError("ready input needs one typed placement or one explicit manual offset")
        input_pipeline_placement.validate_for(problem, resources, storage)
        if input_pipeline_placement.block != block:
            raise ValueError("ready input placement cannot replace the explicit selected block")
        input_pipeline_row_offset = input_pipeline_placement.lhs_slot_offset
    bound = _ready_execution_bound(result, problem, block, resources, storage)
    admission.validate_for(adapter, bound)
    if array_prefix_admission is not None and type(result) is not DensePanelPipelineResult:
        raise ValueError("array prefix requires the original retained full-K owner")
    actual, changes = propose_dense_ready_readout(result.kernel, problem, adapter,
        final_k_prefix=array_prefix_admission is not None, array_prefix_admission=array_prefix_admission)
    input_pipeline = []
    if input_pipeline_row_offset is not None:
        if admission.disjoint_load_qualification_identity is None:
            raise ValueError("input pipeline needs the separately selected load/ready-readout conjunction")
        if type(result) is not DensePanelPipelineResult:
            raise ValueError("input pipeline requires its original retained full-K owner")
        if input_pipeline_placement is None:
            lhs_rows = block.m * block.k * problem.tile_m
            rhs_rows = block.k * block.n * problem.tile_k
            if (2 * lhs_rows * storage.lhs.unit_bytes + rhs_rows * storage.rhs.unit_bytes
                    > resources.operand_bytes):
                raise ValueError("two live LHS panels and one RHS panel exceed the original operand byte budget")
        actual, input_pipeline = propose_dense_ready_input_pipeline(
            actual, adapter, storage, row_offset=input_pipeline_row_offset,
            placement=input_pipeline_placement)
    projection = DenseReadyReadoutCheckingProjection(result.kernel, problem,
        input_pipeline_row_offset=input_pipeline_row_offset, input_pipeline_placement=input_pipeline_placement,
        final_k_prefix=array_prefix_admission is not None, array_prefix_admission=array_prefix_admission)
    fresh, array_proof = projection.hazard_kernel(actual, adapter, storage)
    if (type(static_budget) is not int or static_budget <= 0
            or sum(1 for _ in _calls(actual.body)) > static_budget):
        raise ValueError("actual retained prefix exceeds the original static command budget")
    report = _verify_dense_epochs(
        fresh, problem, resources, storage, adapter, bound,
        static_budget=static_budget, timeout_ms=timeout_ms,
        output_readout=result.output_readout, bias_source=result.bias_source,
        ordered_accumulator_commits=True, source_phase_fission=True,
        readout_compute_admission=admission,
        readout_source_projection=projection,
    )
    report.update(schema="merlin.dense_ready_readout.v1", transformations=changes,
                  original_schedule=result.realization, source_identity=result.source_identity,
                  additional_capability=asdict(admission), schedule_ir=result.realization["schedule_ir"])
    if input_pipeline:
        report["input_pipeline"] = input_pipeline
    if input_pipeline_placement is not None:
        report["input_pipeline_placement"] = asdict(input_pipeline_placement)
    if array_proof is not None:
        report["array_prefix_projection"] = array_proof
        report["kernel_sha256"] = actual.digest()
    checked = DenseReadyReadoutResult(
        result, actual, report, admission, problem,
        _digest((result.kernel.digest(), actual.digest(), asdict(admission), asdict(problem),
                 repr(adapter), asdict(storage), None if input_pipeline_placement is None
                 else asdict(input_pipeline_placement), None if array_prefix_admission is None
                 else asdict(array_prefix_admission))),
        sha256(Path(__file__).read_bytes()).hexdigest(), adapter, storage,
        input_pipeline_row_offset, input_pipeline_placement, array_prefix_admission,
    )
    checked.revalidate()
    return checked


def admit_ready_readout_tile_loads(original, proposal, problem, block, resources, storage, adapter, *,
                                   project, encode_band, static_budget=128, timeout_ms=5000,
                                   max_static_nodes=4096, external_sources=None):
    """Compose final physical wide bands with the SAME causal frontier proof.

    Every actual band is decoded and all actual epochs are verified before the
    exact original counted source representation is used. Emission retains the
    wide commands. No old typed consumer or execution admission is broadened.
    """
    from merlin.compile.scheduling.panel_tile_loads import _admit_dense_tile_loads

    if type(original) is not DenseReadyReadoutResult:
        raise ValueError("wide causal readout requires its exact original checked result")
    original.revalidate()
    if problem != original.problem or storage != original._storage or adapter != original._adapter:
        raise ValueError("wide causal readout cannot change original problem, storage or adapter")
    bound = _ready_execution_bound(original.original, problem, block, resources, storage)
    projection = DenseReadyReadoutCheckingProjection(
        original.original.kernel, problem, canonical_load_runs=True,
        input_pipeline_row_offset=original.input_pipeline_row_offset,
        input_pipeline_placement=original.input_pipeline_placement,
        final_k_prefix=original.array_prefix_admission is not None,
        array_prefix_admission=original.array_prefix_admission)

    def verifier(actual, p, b, r, s, a, admission, **options):
        if admission != original.admission:
            raise ValueError("wide addressing cannot change the causal readout capability")
        _ready_execution_bound(original.original, p, b, r, s)
        fresh, array_proof = projection.hazard_kernel(actual, a, s)
        report = _verify_dense_epochs(
            fresh, p, r, s, a, bound, **options,
            ordered_accumulator_commits=True, source_phase_fission=True,
            readout_compute_admission=admission, readout_source_projection=projection,
        )
        if array_proof is not None:
            report.update(array_prefix_projection=array_proof, kernel_sha256=actual.digest())
        return report

    return _admit_dense_tile_loads(
        original, proposal, problem, block, resources, storage, adapter,
        verifier=verifier, project=project, encode_band=encode_band,
        static_budget=static_budget, timeout_ms=timeout_ms, max_static_nodes=max_static_nodes,
        external_sources=external_sources,
    )

def schedule_dense_pipeline(
    problem, resources, storage, adapter, *, name, dtypes, source_identity=None,
    external_buffers_disjoint, schedule_plan=None, pipeline_admission=None,
    batch_readouts=False, readout_mode="separate", readout_admission=None,
    readout_group_path=(), static_budget=128, timeout_ms=5000,
    output_readout=None, bias_source=None, readout_qualification_identity=None,
    epoch_law=None, input_pipeline_row_offset=None, input_pipeline_placement=None, array_prefix_admission=None,
):
    """Consume one source-bound decision through its original checked owner.

    A compact backend can keep the same problem/resources/storage/adapter and
    lowering callback while trying panel or segmented decisions. None retains
    the ordinary compact planner. Explicit plans never silently choose another
    block/family on a capacity, tail, proof or execution-law refusal.

    The caller supplies independently selected target premises, never inferred
    from capacity or the plan: panel/retained/segmented pipeline_admission and,
    optionally, a separate causal-ready or disjoint-ACC readout_admission.
    ``batch_readouts`` consumes only the pipeline law's existing store budget.
    All original product, effect, alias, numerical, finite-epoch and completion
    checks remain in the original owners. Returned results emit their checked
    actual commands with ``emit_llvm`` or through their counted kernel. A common
    lowering callback accepts (instruction, operands, metadata=None): ready
    emission supplies its actual SCF metadata as the third argument. The target
    must lower every explicit completion, not erase it or fence every data call.
    No preferred schedule, performance score, simulator or new grant is added.

    ``input_pipeline_row_offset`` selects an explicit second owned LHS slot for
    a retained full-K ready-readout decision and requires its separate disjoint
    load conjunction. Other routes reject this option instead of ignoring it.
    Alternatively, ``input_pipeline_placement`` consumes the same capacity
    proposal returned by ``dense_ready_input_choices``. It must match this
    explicit plan and actual projected row windows; it never repairs a plan.
    ``array_prefix_admission`` selects counted final-K peeling with the
    existing pure K-prefix fresh/STAY law. Final-K pairs remain fresh and
    their original stores consume the same causal law. No store/STAY
    conjunction, extra ACC window or implicit execution permission is added.

    ``epoch_law`` (a ``dense_panel_epochs.DensePanelEpochLaw``) selects fact-backed
    epochs for a two-window decision: drains inside each output block's K stream
    (and before its readout, with result visibility) go where the law's ordering
    premises hold, and the result emits only the remaining drains. None keeps
    every original drain; the returned schedule is then unchanged.
    """
    if (type(batch_readouts) is not bool or readout_mode not in ("separate", "ready", "disjoint")
            or type(readout_group_path) is not tuple
            or any(type(v) is not str or not v for v in readout_group_path)
            or readout_mode == "separate" and (readout_admission is not None or readout_group_path)
            or readout_mode == "ready" and (readout_admission is None or readout_group_path)
            or readout_mode == "disjoint" and (readout_admission is None or not readout_group_path)):
        raise ValueError("dense pipeline needs explicit consumed readout mode, law and group path")
    from .schedule_ir import read_dense_schedule_plan

    boundary = dict(output_readout=output_readout, bias_source=bias_source,
                    readout_qualification_identity=readout_qualification_identity)
    common = dict(name=name, dtypes=dtypes, source_identity=source_identity,
                  external_buffers_disjoint=external_buffers_disjoint, static_budget=static_budget,
                  schedule_plan=schedule_plan, **boundary)
    decision = None if schedule_plan is None else read_dense_schedule_plan(
        schedule_plan, problem, resources, storage, adapter, source_identity=source_identity,
        dtypes=dtypes, static_budget=static_budget, output_readout=output_readout, bias_source=bias_source)
    if (input_pipeline_row_offset is not None and input_pipeline_placement is not None):
        raise ValueError("ready input needs one typed placement or one explicit manual offset")
    if (input_pipeline_row_offset is not None or input_pipeline_placement is not None
            or array_prefix_admission is not None) and (
            readout_mode != "ready" or decision is None or decision.pipeline_kind != "retained_full_k"):
        raise ValueError("advancing input slots require an explicit retained full-K ready-readout decision")
    if epoch_law is not None and (decision is None or decision.pipeline_kind != "two_window"):
        raise ValueError("dense panel epochs apply to an explicit two-window panel decision")
    if decision is None or decision.pipeline_prefetch is None:
        if pipeline_admission is not None or batch_readouts or readout_mode != "separate":
            raise ValueError("compact decision cannot consume asynchronous pipeline or readout laws")
        from .compact_dense import schedule_compact_dense

        return schedule_compact_dense(problem, resources, storage, adapter,
            verification_timeout_ms=timeout_ms, max_repairs=0 if decision is not None else 1, **common)
    if pipeline_admission is None:
        raise ValueError("advanced dense decision requires its independently selected pipeline law")
    # Layout decisions may relocate the original assigned windows. Every later
    # proof consumes this same checked actual layout, rather than stale bases.
    chosen_storage = decision.storage
    args = (problem, decision.block, resources, chosen_storage, adapter)
    if decision.pipeline_kind == "retained_lhs_segmented_k":
        from .dense_segmented_pipeline import batch_dense_segmented_readouts, schedule_dense_segmented_pipeline

        result = schedule_dense_segmented_pipeline(problem, resources, storage, adapter,
                                                    pipeline_admission, timeout_ms=timeout_ms, **common)
        if batch_readouts:
            result = batch_dense_segmented_readouts(result, *args, pipeline_admission,
                                                    static_budget=static_budget, timeout_ms=timeout_ms)
        if readout_mode == "ready":
            result = schedule_dense_ready_readout(result, *args, readout_admission,
                                                  static_budget=static_budget, timeout_ms=timeout_ms,
                                                  input_pipeline_row_offset=input_pipeline_row_offset,
                                                  input_pipeline_placement=input_pipeline_placement,
                                                  array_prefix_admission=array_prefix_admission)
        elif readout_mode == "disjoint":
            from .dense_readout_overlap import schedule_dense_readout_overlap

            result = schedule_dense_readout_overlap(result, *args, readout_admission,
                group_path=readout_group_path, static_budget=static_budget, timeout_ms=timeout_ms)
    else:
        if readout_mode == "disjoint" or (readout_mode == "ready"
                and decision.pipeline_kind != "retained_full_k"):
            raise ValueError("readout mode requires its original checked segmented/full-K pipeline")
        from .dense_panel_pipeline import schedule_dense_panel_pipeline

        result = schedule_dense_panel_pipeline(problem, resources, storage, adapter,
                                               pipeline_admission, timeout_ms=timeout_ms, **common)
        if batch_readouts:
            from .dense_panel_readout import batch_dense_panel_readouts

            result = batch_dense_panel_readouts(result, *args, pipeline_admission,
                                                static_budget=static_budget, timeout_ms=timeout_ms)
        if epoch_law is not None:
            from .dense_panel_epochs import schedule_dense_panel_epochs

            result = schedule_dense_panel_epochs(result, *args, pipeline_admission, epoch_law,
                                                 static_budget=static_budget, timeout_ms=timeout_ms)
        if readout_mode == "ready":
            result = schedule_dense_ready_readout(result, *args, readout_admission,
                                                  static_budget=static_budget, timeout_ms=timeout_ms,
                                                  input_pipeline_row_offset=input_pipeline_row_offset,
                                                  input_pipeline_placement=input_pipeline_placement,
                                                  array_prefix_admission=array_prefix_admission)
    result.revalidate()
    return result
