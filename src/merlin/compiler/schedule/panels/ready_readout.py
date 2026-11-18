"""Final-K readout frontiers with explicit causal accumulator completion.

Actual counted PRELOAD/COMPUTE pairs remain adjacent. Each original output store
follows its complete final K run in the same epoch; the full accumulator view
and all input staging epochs stay intact. The target must separately establish
producer retirement, pending-write read readiness, store snapshot/backpressure
and the final drain. This module supplies no cycle spacing or execution grant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.compiler.kernel_ir.ir import Call, Kernel, Loop, check_integer_domain, check_structure, free_vars
from merlin.compiler.kernel_ir.primitives import _loop_at, _splice
from merlin.compiler.schedule.blocks import Compute, InitializeAccumulator, Load, Preload, Store
from merlin.compiler.schedule.dense.compact import _digest
from merlin.compiler.schedule.dense.contracts import TiledIntegerStore
from merlin.compiler.schedule.library_sources import admission_point
from merlin.compiler.schedule.loops.counted_loops import LoopTransform, propose_loop_transforms
from merlin.compiler.schedule.panels.load_coalescing import _linear, _plain
from merlin.compiler.schedule.panels.pipeline import (
    DRAIN,
    DensePanelPipelineResult,
    DenseReadyReadoutAdmission,
    DenseRetainedReductionAdmission,
    _calls,
    _remap,
    _verify_dense_epochs,
)
from merlin.compiler.schedule.panels.segmented_pipeline import DenseSegmentedPipelineResult, _assignment

# ---- dense_ready_readout -----------------------------------------------------------------------


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
        return tuple(
            s for child in _remap(statement.body[:-1], statement.var, None) for s in _strip_store_epochs(child)
        )
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
    from merlin.compiler.schedule.panels.pipeline import DenseRetainedChoice

    if (
        type(choice) is not DenseRetainedChoice
        or choice.role != "weight"
        or choice.outer_order != ("n", "m")
        or choice.block.k != problem.tiles[1]
        or problem.tiles[0] // choice.block.m < 2
        or not storage.admits(problem, choice.block, resources)
    ):
        raise ValueError("ready input placement needs an advancing unbiased retained-weight full-K choice")
    lhs, rhs, acc = storage.row_counts(problem, choice.block)
    offset = (lhs + storage.lhs.alignment - 1) // storage.lhs.alignment * storage.lhs.alignment
    if offset + lhs > storage.lhs.capacity or (
        (offset + lhs) * storage.lhs.unit_bytes + rhs * storage.rhs.unit_bytes > resources.operand_bytes
    ):
        raise ValueError("ready input placement needs two owned LHS slots and one RHS panel")
    return DenseReadyInputPlacement(
        choice, lhs, rhs, acc, offset, _digest((asdict(problem), asdict(resources), asdict(storage)))
    )


def dense_ready_input_choices(problem, resources, storage):
    """Filter bounded retained choices by their actual consumer's live roles.

    This explicit option keeps the ordinary one-panel selector unchanged.
    Every returned placement consumes two advancing input panels, one retained
    weight panel and one accumulator panel. The caller selects a choice, builds
    its ordinary retained full-K plan, and passes this SAME placement to
    ``schedule_dense_pipeline(input_pipeline_placement=...)``. No dynamic
    command trace is expanded and no numerical or completion law is supplied.
    """
    from merlin.compiler.schedule.panels.pipeline import dense_retained_choices

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
    from merlin.compiler.schedule.panels.readouts import _affine_window_bounds

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
                if (
                    statement.extent == 1
                    and statement.body
                    and statement.body[-1] != Call(DRAIN, ())
                    and any(
                        type(child) is Loop
                        and child.body
                        and child.body[-1] == Call(DRAIN, ())
                        and any(
                            c != Call(DRAIN, ()) and type(adapter.project(c)) is Compute for c in _calls(child.body)
                        )
                        for child in statement.body
                    )
                ):
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
                if (
                    kinds(statement) == {Preload, Compute}
                    and index + 1 < len(body)
                    and kinds(body[index + 1]) in ({Store}, {TiledIntegerStore})
                ):
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
        if (
            len(pair) != 2
            or len(store_calls) != 1
            or any(type(c) is not Call or not _plain(c) for c in (*pair, *store_calls))
        ):
            raise ValueError("ready readout requires plain adjacent physical pairs and one original store")
        pre, compute = (adapter.project(c) for c in pair)
        store = adapter.project(store_calls[0])
        if type(pre) is not Preload or type(compute) is not Compute or type(store) not in (Store, TiledIntegerStore):
            raise ValueError("ready readout cannot split descriptors or vary output numerical configuration")
        variables = {s.var for s in loops}
        names = {}
        for axis, coordinate in zip(
            ("m", "k", "n"), (compute.input_block[0], compute.input_block[1], pre.weight_block[1]), strict=True
        ):
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
        if (
            pre.accumulator_row != renamed_store.accumulator_row
            or pre.rows != renamed_store.rows
            or pre.cols != renamed_store.cols
            or _linear(renamed_store.dram_row) != _linear(compute.input_block[0] * problem.tile_m)
            or _linear(renamed_store.dram_col) != _linear(pre.weight_block[1] * problem.tile_n)
            or names["k"] in (free_vars(pre.accumulator_row) | free_vars(pre.rows) | free_vars(pre.cols))
        ):
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
                from merlin.compiler.schedule.panels.array_retention import propose_dense_array_retention

                # Only the pure prefix enters the existing fresh/STAY owner.
                # Every final-K consume remains fresh; no store crosses STAY.
                isolated = prefix
                for depth in range(len(path), 0, -1):
                    ancestor = _loop_at(actual.body, path[:depth])
                    isolated = replace(ancestor, body=(isolated, Call(DRAIN, ())))
                proposal = propose_dense_array_retention(
                    replace(actual, body=(isolated,)), adapter, array_prefix_admission
                )
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
        rewritten = (*siblings[:index], replacement, *siblings[index + 2 :])
        actual = (
            replace(actual, body=_splice(actual.body, path[:-1], (replace(parent, body=rewritten),)))
            if parent is not None
            else replace(actual, body=rewritten)
        )
        records.append(
            dict(
                path=path,
                loop_transforms=[asdict(r) for r in requests],
                final_k_tiles=k.extent,
                output_tiles=m.extent * n.extent,
                final_k_prefix=final_k_prefix,
                array_prefix=tuple(array_changes),
                geometry="unchanged original ACC and operand views",
                frontier="each original store immediately follows its complete final K paired run",
                completion="one original full drain after the complete frontier",
            )
        )
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
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or type(row_offset) is not int
        or row_offset <= 0
        or row_offset % storage.lhs.alignment
    ):
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
        return (
            type(statement) is Loop
            and statement.extent == 1
            and statement.body
            and statement.body[-1] == Call(DRAIN, ())
            and all(c != Call(DRAIN, ()) for c in _calls(statement.body[:-1]))
        )

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
                if type(operation) is Load and operation.role == "lhs" or type(operation) is Compute:
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
            if (
                statement.extent >= 2
                and len(parts) >= 2
                and all(initialization(s) for s in parts[:-2])
                and all(epoch(s) for s in parts[-2:])
                and kinds(parts[-2].body) == {Load}
                and all(adapter.project(c).role == "lhs" for c in _calls(parts[-2].body[:-1]))
                and kinds(parts[-1].body) in ({Preload, Compute, Store}, {Preload, Compute, TiledIntegerStore})
            ):
                initializers, (stage, current) = parts[:-2], parts[-2:]
                initialization_drains = statement.extent * sum(drains(s) for s in initializers)
                windows = _input_group_windows(
                    (stage, current), adapter, {**domains, statement.var: (0, statement.extent - 1)}
                )
                if placement is not None:
                    if type(placement) is not DenseReadyInputPlacement or row_offset != placement.lhs_slot_offset:
                        raise ValueError("ready input emission must consume the selected placement")
                    lhs_window = (storage.lhs.base, storage.lhs.base + placement.lhs_rows)
                    if (
                        windows.get("lhs_stage") != lhs_window
                        or windows.get("lhs_read") != lhs_window
                        or any(
                            role not in windows or windows[role][0] < view.base or windows[role][1] > view.base + rows
                            for role, view, rows in (
                                ("weight", storage.rhs, placement.rhs_rows),
                                ("accumulator", storage.accumulator, placement.accumulator_rows),
                            )
                        )
                    ):
                        raise ValueError("actual ready input accesses disagree with the selected role footprints")
                period = fresh("dense_input_period")

                def selected(tree, slot, new, scale, offset):
                    contents = _remap(tree.body[:-1], tree.var, None)
                    contents = _remap(contents, statement.var, new, scale, offset)
                    return shifted(contents, slot * row_offset)

                def advance(slot, new, scale, offset):
                    return (
                        *_remap(initializers, statement.var, new, scale, offset),
                        Loop(
                            fresh("dense_input_epoch"),
                            1,
                            (
                                *selected(stage, 1 - slot, new, scale, offset + 1),
                                *selected(current, slot, new, scale, offset),
                                Call(DRAIN, ()),
                            ),
                        ),
                    )

                result.append(
                    Loop(fresh("dense_input_prologue"), 1, (*selected(stage, 0, None, 1, 0), Call(DRAIN, ())))
                )
                pairs, tail = divmod(statement.extent - 1, 2)
                if pairs:
                    result.append(Loop(period, pairs, (*advance(0, period, 2, 0), *advance(1, period, 2, 1))))
                if tail:
                    result.extend(advance(0, None, 1, 2 * pairs))
                result.extend(_remap(initializers, statement.var, None, 1, statement.extent - 1))
                result.append(
                    Loop(
                        fresh("dense_input_epilogue"),
                        1,
                        (
                            *selected(current, (statement.extent - 1) % 2, None, 1, statement.extent - 1),
                            Call(DRAIN, ()),
                        ),
                    )
                )
                records.append(
                    dict(
                        path=(*path, statement.var),
                        groups=statement.extent,
                        lhs_slot_offsets=(0, row_offset),
                        rhs_slots=1,
                        actual_role_windows=windows,
                        accumulator="same original rows reused only after full drain",
                        initializer_scopes=len(initializers),
                        initialization="all original separate completion scopes retained before current compute",
                        live_lhs_generations=2,
                        preserved_initialization_drains=initialization_drains,
                        remaining_drains=statement.extent + 1 + initialization_drains,
                    )
                )
            else:
                result.append(
                    replace(
                        statement,
                        body=walk(parts, (*path, statement.var), {**domains, statement.var: (0, statement.extent - 1)}),
                    )
                )
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
