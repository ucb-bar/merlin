"""Composable counted loop proposals for source-verified scheduling.

Split/unroll/ordered partitions preserve command order. Reorder/fuse/fission need final independent
source and dependence verification; proposal construction alone never admits them.
They are proposals for the existing source/effect verifier, not target admission.
Ordered carried-completion split/unroll/partitions thread actual event binders;
dependency-changing moves require a separate completion transport proof.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.sched.ir import Const, Kernel, Loop, Ptr, Var, check_structure
from merlin.sched.ir.expr import Expr, free_vars
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import (
    Cursor,
    NotApplicable,
    _bound_vars,
    _loop_at,
    _map_calls,
    _resolve,
    _splice,
    _subst_call,
    divide_loop,
    unroll,
)


@dataclass(frozen=True)
class LoopTransform:
    action: str
    path: tuple[str, ...]
    factor: int | None = None
    cut: int | None = None
    names: tuple[str, ...] = ()
    index: int | None = None
    cuts: tuple[int, ...] = ()

    def __post_init__(self):
        if (
            self.action not in ("split", "unroll", "reorder", "fuse", "fission", "hoist", "partition")
            or type(self.path) is not tuple
            or not self.path
            or any(type(name) is not str or not name for name in self.path)
        ):
            raise ValueError("loop transform requires an executable action and a current loop path")
        if type(self.cuts) is not tuple or any(type(c) is not int for c in self.cuts):
            raise ValueError("partition cuts must be an exact tuple of integer interior offsets")
        expected_names = (2 if self.action in ("split", "unroll")
                          else len(self.cuts) + 1 if self.action == "partition" else 1)
        if (
            type(self.names) is not tuple
            or any(type(name) is not str or not name for name in self.names)
            or len(set(self.names)) != len(self.names)
            or (self.names and len(self.names) != expected_names)
            or (self.names and self.action in ("reorder", "fuse", "hoist"))
        ):
            raise ValueError("loop result names must name exactly the requested split or fission loops")
        if self.action == "partition":
            if (
                not 1 <= len(self.cuts) <= 15
                or any(not 0 < c < 1 << 63 for c in self.cuts)
                or tuple(sorted(set(self.cuts))) != self.cuts
                or self.factor is not None or self.cut is not None or self.index is not None
            ):
                raise ValueError("partition requires only 1..15 strictly increasing positive interior cuts")
        elif self.cuts:
            raise ValueError("only partition can carry interior cuts")
        if self.action == "hoist":
            if type(self.index) is not int or self.index < 0 or self.factor is not None or self.cut is not None:
                raise ValueError("hoist requires only a nonnegative statement index")
        elif self.index is not None:
            raise ValueError("only hoist can carry a statement index")
        elif self.action in ("split", "unroll"):
            if type(self.factor) is not int or not 2 <= self.factor <= 64 or self.cut is not None:
                raise ValueError("split/unroll requires only an explicit factor 2..64")
        elif self.action == "fission":
            if type(self.cut) is not int or self.cut <= 0 or self.factor is not None:
                raise ValueError("fission requires only a positive body cut position")
        elif self.factor is not None or self.cut is not None:
            raise ValueError("reorder/fuse cannot carry ignored factor or cut decisions")


def _size(body):
    return sum(1 + (_size(s.body) if type(s) is Loop else 0) for s in body)


def _split(kernel, path, factor, names=()):
    target = _loop_at(kernel.body, path)
    from merlin.sched.ir.events import has_carried_events

    if has_carried_events((target,)):
        from merlin.sched.primitives.event_loops import split_event_loop

        return split_event_loop(kernel, path, factor, names or (target.var + "o", target.var + "i"))
    full, remainder = divmod(target.extent, factor)
    if not full:
        raise NotApplicable("split/unroll factor exceeds current loop extent")
    bulk = full * factor
    # Bound arithmetic is constant; no trip-count-sized expansion or dynamic min.
    original = replace(target, extent=bulk)
    temporary = replace(kernel, body=_splice(kernel.body, path, (original,)))
    actual = divide_loop(temporary, Cursor(loops=path), factor, names=names or None)
    if remainder:
        tail_body = _map_calls(target.body, lambda call: _subst_call(call, target.var, Var(target.var) + bulk))
        tail = Loop(target.var, remainder, tail_body)
        outer_path = (*path[:-1], names[0] if names else target.var + "o")
        outer = _loop_at(actual.body, outer_path)
        actual = replace(actual, body=_splice(actual.body, outer_path, (outer, tail)))
    return actual


def striped_fission_transforms(kernel, paths, *, factor, cut):
    """Name explicit split/fission requests for selected current loop bodies.

    Each complete stripe executes the prefix for all stripe indices, then the
    suffix for those same indices. A remaining tail retains its original order.
    This exposes counted suffix loops to staging/transfer passes without
    enumerating the dynamic domain. It is a proposal, not dependence admission:
    the source/effect owner must verify the final transformed program.
    """
    if type(kernel) is not Kernel or check_structure(kernel) or type(paths) is not tuple or not 1 <= len(paths) <= 16:
        raise ValueError("striped fission requires checked IR and 1..16 distinct current loop paths")
    # Validate paths/knobs through the ordinary typed request constructor.
    for path in paths:
        LoopTransform("split", path, factor=factor)
        LoopTransform("fission", path, cut=cut)
    if len(set(paths)) != len(paths):
        raise ValueError("striped fission requires distinct current loop paths")
    if any(a != b and b[: len(a)] == a for a in paths for b in paths):
        raise ValueError("striped fission paths must be independent, not nested")
    used = _bound_vars(kernel.body) | {arg.name for arg in kernel.args}

    def fresh(stem):
        index = 0
        while (name := stem + "_" + str(index)) in used:
            index += 1
        used.add(name)
        return name

    requests = []
    for path in paths:
        target = _loop_at(kernel.body, path)
        if target.extent < factor or cut >= len(target.body):
            raise NotApplicable("stripe needs a complete factor and two nonempty body parts")
        outer, inner, suffix = (fresh(target.var + part) for part in ("_stripe", "_within", "_suffix"))
        requests.extend(
            (
                LoopTransform("split", path, factor=factor, names=(outer, inner)),
                LoopTransform("fission", (*path[:-1], outer, inner), cut=cut, names=(suffix,)),
            )
        )
    return tuple(requests)


def propose_loop_transforms(kernel, transforms, *, index_bits=64, max_static_operations=4096):
    """Apply every requested action or refuse; no ignored cursor or fallback.

    Paths are resolved against each preceding transform's current result. A
    split of ``i`` creates explicit ``names`` (default ``io``/``ii``) and an
    optional remaining ``i`` tail. Explicit names make repeated transformations
    in independent scopes composable without ambiguous generated cursors.
    Partial unroll inlines the new inner factor while retaining an ordered tail.
    Reorder/fuse/fission only propose a different order; they confer no legality.
    Every command argument, pointer, configuration and staging field is kept.
    Original source, resources, alias and completion contracts remain the duty
    of the invoking verified source family, which must check the final result.
    """
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or type(transforms) is not tuple
        or len(transforms) > 32
        or any(type(t) is not LoopTransform for t in transforms)
        or type(max_static_operations) is not int
        or not 1 <= max_static_operations <= 65536
    ):
        raise ValueError("ordered loop composition needs checked IR and explicit bounded typed requests")
    check_integer_domain(kernel, index_bits)
    actual = kernel
    for request in transforms:
        target = _loop_at(actual.body, request.path)
        from merlin.sched.ir.events import has_carried_events

        if request.action in ("reorder", "fuse", "fission", "hoist") and has_carried_events((target,)):
            raise NotApplicable("completion-carrying order/placement changes need a separate "
                                "dependency transport proof")
        # Conservative preflight before cloning: work depends on the explicit
        # factor and static body, never the dynamic loop trip count.
        growth = (len(request.cuts) + 1 if request.action == "partition" else (request.factor or 1) + 1) * (
            _size(target.body) + 2
        )
        if _size(actual.body) + growth > max_static_operations:
            raise NotApplicable("ordered loop transform exceeds original static operation budget")
        if request.action == "partition":
            if request.cuts[-1] >= target.extent:
                raise NotApplicable("partition cuts must stay strictly inside the original loop extent")
            used = _bound_vars(actual.body) | {arg.name for arg in actual.args}
            names = request.names
            if names:
                if any(name in used for name in names):
                    raise NotApplicable("partition loop name collides with an original name")
            else:
                fresh = []
                index = 0
                while len(fresh) < len(request.cuts) + 1:
                    name = target.var + "_part_" + str(index)
                    index += 1
                    if name not in used:
                        fresh.append(name)
                        used.add(name)
                names = tuple(fresh)
            boundaries = (0, *request.cuts, target.extent)
            if has_carried_events((target,)):
                from merlin.sched.primitives.event_loops import partition_event_loop

                actual = partition_event_loop(actual, request.path, boundaries, names)
                check_integer_domain(actual, index_bits)
                continue
            parts = []
            for name, start, stop in zip(names, boundaries[:-1], boundaries[1:], strict=True):
                # Shift on the old variable first, then rename: the existing
                # substitution primitive preserves equality predicates under a
                # SAME-variable shift and under a separate variable rename.
                body = _map_calls(target.body, lambda call: _subst_call(
                    _subst_call(call, target.var, Var(target.var) + start), target.var, Var(name)
                ))
                parts.append(Loop(name, stop - start, body))
            actual = replace(actual, body=_splice(actual.body, request.path, tuple(parts)))
        elif request.action == "split":
            actual = _split(actual, request.path, request.factor, request.names)
        elif request.action == "unroll":
            if target.extent == request.factor:
                if request.names:
                    raise NotApplicable("complete unroll has no result loops to name")
                actual = unroll(actual, Cursor(loops=request.path))
            else:
                actual = _split(actual, request.path, request.factor, request.names)
                outer, inner = request.names or (target.var + "o", target.var + "i")
                actual = unroll(actual, Cursor(loops=(*request.path[:-1], outer, inner)))
        elif request.action == "reorder":
            if len(target.body) != 1 or type(target.body[0]) is not Loop:
                raise NotApplicable("reorder requires a perfect rectangular loop pair")
            inner = target.body[0]
            swapped = Loop(inner.var, inner.extent, (Loop(target.var, target.extent, inner.body),))
            actual = replace(actual, body=_splice(actual.body, request.path, (swapped,)))
        elif request.action == "hoist":
            if target.extent <= 0 or request.index >= len(target.body):
                raise NotApplicable("hoist statement index is outside the current loop body")
            moved = target.body[request.index]

            def calls(body):
                for statement in body:
                    if type(statement) is Loop:
                        yield from calls(statement.body)
                    else:
                        yield statement

            selected = tuple(calls((moved,)))
            if any(
                call.unit or call.produces or call.awaits or call.stages or call.sets or call.assumes
                for call in selected
            ):
                raise NotApplicable("hoist cannot move unknown completion, staging or configuration effects")
            if target.extent == 1:
                moved = _map_calls((moved,), lambda call: _subst_call(call, target.var, Const(0)))[0]
            else:

                def depends(value):
                    if type(value) is Ptr:
                        return depends(value.offset)
                    if type(value) is tuple:
                        return any(depends(item) for item in value)
                    return isinstance(value, Expr) and target.var in free_vars(value)

                for call in selected:
                    for _key, value in call.args:
                        if depends(value):
                            raise NotApplicable("hoisted statement depends on the crossed loop index")
            remaining = (*target.body[: request.index], *target.body[request.index + 1 :])
            replacement = (moved, replace(target, body=remaining)) if remaining else (moved,)
            actual = replace(actual, body=_splice(actual.body, request.path, replacement))
        elif request.action == "fission":
            if request.cut >= len(target.body):
                raise NotApplicable("fission cut must separate two nonempty body parts")
            name = request.names[0] if request.names else target.var + "_fission"
            if name in _bound_vars(actual.body) or name in {arg.name for arg in actual.args}:
                raise NotApplicable("fission loop name collides with an original name")
            second = _map_calls(target.body[request.cut :], lambda call: _subst_call(call, target.var, Var(name)))
            parts = (replace(target, body=target.body[: request.cut]), Loop(name, target.extent, second))
            actual = replace(actual, body=_splice(actual.body, request.path, parts))
        else:  # fuse
            siblings, _ = _resolve(actual.body, Cursor(loops=request.path[:-1]))
            position = next(i for i, child in enumerate(siblings) if child is target)
            if position + 1 >= len(siblings) or type(siblings[position + 1]) is not Loop:
                raise NotApplicable("fuse requires an immediately adjacent following loop")
            second = siblings[position + 1]
            if has_carried_events((second,)):
                raise NotApplicable("completion-carrying fusion needs a separate dependency transport proof")
            if target.extent != second.extent:
                raise NotApplicable("fuse requires identical iteration extents")
            renamed = _map_calls(second.body, lambda call: _subst_call(call, second.var, Var(target.var)))
            from merlin.sched.primitives import _rebuild

            merged = replace(target, body=(*target.body, *renamed))
            body = (*siblings[:position], merged, *siblings[position + 2 :])
            actual = replace(actual, body=_rebuild(actual.body, request.path[:-1], body))
        errors = check_structure(actual)
        if errors:
            raise NotApplicable("ordered loop transform cannot preserve closed completion/name scopes: " + errors[0])
        if _size(actual.body) > max_static_operations:
            raise NotApplicable("ordered loop transform exceeds original static operation budget")
        check_integer_domain(actual, index_bits)
    return actual


def tile_loop_nest_transforms(
    kernel,
    path,
    *,
    factors,
    outer_order=None,
    inner_order=None,
    tail_policy="refuse",
    index_bits=64,
    max_static_operations=4096,
):
    """Compose ordinary split/reorder requests for a perfect rectangular nest.

    Factors and both axis permutations are explicit caller decisions. The
    result tiles 2..4 selected axes, with outer tile loops before within-tile
    loops; it does not choose a resource or performance policy. Each extent
    must divide exactly by default. Explicit ``tail_policy="cut"`` instead
    partitions bulk/edge rectangles and tiles each applicable region, retaining
    thinner edge axes without padding or a masked-command law. This may change
    inter-region order and still requires the final source/effect verifier.

    All returned paths refer to the preceding request's CURRENT result. Fresh
    names avoid unrelated scopes, while the existing primitives substitute
    original indices and retain every argument/configuration/staging field.
    The final source/dependence/effect verifier must admit the composed result:
    rectangular metadata and unchanged products alone do not authorize a
    reordered reduction, storage reuse or completion-token movement.
    """
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or type(factors) is not tuple
        or not 2 <= len(factors) <= 4
        or any(type(f) is not int or not 2 <= f <= 64 for f in factors)
    ):
        raise ValueError("perfect nest tiling needs checked IR and 2..4 explicit factors 2..64")
    depth = len(factors)
    natural = tuple(range(depth))
    outer_order = natural if outer_order is None else outer_order
    inner_order = natural if inner_order is None else inner_order
    if any(
        type(order) is not tuple
        or len(order) != depth
        or any(type(axis) is not int for axis in order)
        or set(order) != set(natural)
        for order in (outer_order, inner_order)
    ):
        raise ValueError("tile orders must be complete permutations of the selected axes")
    if tail_policy not in ("refuse", "cut"):
        raise ValueError("tile tail policy must explicitly refuse or cut exact rectangles")
    if tail_policy == "cut":
        return _tail_tile_loop_nest_transforms(
            kernel, path, factors, outer_order, inner_order,
            index_bits=index_bits, max_static_operations=max_static_operations,
        )
    LoopTransform("split", path, factor=factors[0])
    selected = []
    target = _loop_at(kernel.body, path)
    for axis, factor in enumerate(factors):
        if target.extent <= 0 or target.extent % factor or target.extent < factor:
            raise NotApplicable("perfect nest tiling requires exact divisible extents; tails remain unsupported")
        selected.append(target)
        if axis + 1 < depth:
            if len(target.body) != 1 or type(target.body[0]) is not Loop:
                raise NotApplicable("tiling selected axes requires a perfect rectangular loop nest")
            target = target.body[0]
    used = _bound_vars(kernel.body) | {arg.name for arg in kernel.args}

    def fresh(stem):
        index = 0
        while (name := stem + "_" + str(index)) in used:
            index += 1
        used.add(name)
        return name

    names = tuple((fresh(loop.var + "_tile"), fresh(loop.var + "_within")) for loop in selected)
    # Innermost first leaves each still-original outer path intact. No cursor
    # forwarding or primitive implementation is duplicated here.
    requests = [
        LoopTransform(
            "split",
            (*path[:-1], *(loop.var for loop in selected[: axis + 1])),
            factor=factors[axis],
            names=names[axis],
        )
        for axis in reversed(range(depth))
    ]
    current = [name for pair in names for name in pair]
    wanted = [names[axis][0] for axis in outer_order] + [names[axis][1] for axis in inner_order]
    # Adjacent interchanges are ordinary recorded decisions, not a private
    # rewiring of the final loop tree. Each is consumed by the existing reader.
    for destination, name in enumerate(wanted):
        position = current.index(name)
        while position > destination:
            requests.append(LoopTransform("reorder", (*path[:-1], *current[:position])))
            current[position - 1], current[position] = current[position], current[position - 1]
            position -= 1
    result = tuple(requests)
    propose_loop_transforms(
        kernel, result, index_bits=index_bits, max_static_operations=max_static_operations
    )
    return result


def _tail_tile_loop_nest_transforms(
    kernel, path, factors, outer_order, inner_order, *, index_bits, max_static_operations
):
    """Compose existing executed cuts/fissions/splits; never enumerate trips.

    Each region is an exact Cartesian product of original bulk/edge ranges.
    Fission exposes those rectangles before interchange. Nothing here proves
    that two effects commute, that storage stays live, or a reduction is legal.
    The same 32-decision/static-operation limits and final source checker apply.
    """
    LoopTransform("split", path, factor=factors[0])
    selected, target = [], _loop_at(kernel.body, path)
    for axis in range(len(factors)):
        if target.extent <= 0:
            raise NotApplicable("tail tiling requires positive counted extents")
        selected.append(target)
        if axis + 1 < len(factors):
            if len(target.body) != 1 or type(target.body[0]) is not Loop:
                raise NotApplicable("tail tiling selected axes require a perfect rectangular nest")
            target = target.body[0]
    used = _bound_vars(kernel.body) | {arg.name for arg in kernel.args}

    def fresh(stem):
        index = 0
        while (name := stem + "_" + str(index)) in used:
            index += 1
        used.add(name)
        return name

    actual, requests = kernel, []

    def apply(request):
        nonlocal actual
        if len(requests) == 32:
            raise NotApplicable("tail tiling exceeds the existing 32-decision composition bound")
        actual = propose_loop_transforms(
            actual, (request,), index_bits=index_bits, max_static_operations=max_static_operations
        )
        requests.append(request)

    # Partition inner axes first: original outer cursors still identify their
    # current loops. The primitive performs exact offset substitution, including
    # pointer, Select predicate and all persistent state/command fields.
    roots = (path,)
    for axis in reversed(range(len(factors))):
        extent, factor = selected[axis].extent, factors[axis]
        bulk = extent // factor * factor
        if 0 < bulk < extent:
            names = (fresh(selected[axis].var + "_bulk"), fresh(selected[axis].var + "_edge"))
            cursor = (*path[:-1], *(loop.var for loop in selected[: axis + 1]))
            apply(LoopTransform("partition", cursor, cuts=(bulk,), names=names))
            if axis == 0:
                roots = tuple((*path[:-1], name) for name in names)

    def rectangles(cursor, depth):
        target = _loop_at(actual.body, cursor)
        if depth + 1 == len(factors):
            return (cursor,)
        children = tuple(target.body)
        if not children or any(type(child) is not Loop for child in children):
            raise NotApplicable("tail rectangles lost their exact selected loop shell")
        for child in children:
            rectangles((*cursor, child.var), depth + 1)
        # Child fission may have created siblings. Lift each into its own
        # counted parent by ordinary recorded fission, preserving every field.
        regions = []
        while len(_loop_at(actual.body, cursor).body) > 1:
            suffix = fresh(cursor[-1] + "_region")
            apply(LoopTransform("fission", cursor, cut=1, names=(suffix,)))
            regions.append(cursor)
            cursor = (*cursor[:-1], suffix)
        regions.append(cursor)
        return tuple(regions)

    regions = tuple(region for root in roots for region in rectangles(root, 0))
    for region in regions:
        loops, target = [], _loop_at(actual.body, region)
        for axis in range(len(factors)):
            loops.append(target)
            if axis + 1 < len(factors):
                if len(target.body) != 1 or type(target.body[0]) is not Loop:
                    raise NotApplicable("tail tiling region is not an exact rectangle")
                target = target.body[0]
        names = {}
        for axis in reversed(range(len(factors))):
            loop, factor = loops[axis], factors[axis]
            if loop.extent >= factor:
                if loop.extent % factor:
                    raise NotApplicable("tail tiling region retains an uncut remainder")
                names[axis] = (fresh(loop.var + "_tile"), fresh(loop.var + "_within"))
                apply(LoopTransform("split", (*region[:-1], *(loop.var for loop in loops[: axis + 1])),
                                    factor=factor, names=names[axis]))
        current = [name for axis, loop in enumerate(loops) for name in names.get(axis, (loop.var,))]
        wanted = ([names[axis][0] for axis in outer_order if axis in names]
                  + [names[axis][1] if axis in names else loops[axis].var for axis in inner_order])
        for destination, name in enumerate(wanted):
            position = current.index(name)
            while position > destination:
                apply(LoopTransform("reorder", (*region[:-1], *current[:position])))
                current[position - 1], current[position] = current[position], current[position - 1]
                position -= 1
    result = tuple(requests)
    # Rebuild the complete public decision sequence, not merely its fragments.
    # No caller-supplied projection or interpretation of a tail is accepted.
    rebuilt = propose_loop_transforms(
        kernel, result, index_bits=index_bits, max_static_operations=max_static_operations
    )
    if rebuilt != actual:
        raise ValueError("tail tiling lost custody of its complete executed decisions")
    return result


@dataclass(frozen=True)
class LoopPartitionRegion:
    """An actual counted region mapped to the original half-open interval."""

    path: tuple[str, ...]
    start: int
    stop: int


@dataclass(frozen=True)
class OrderedLoopPartition:
    """Exact ordered-instance custody, not a source or target admission grant."""

    kernel: Kernel
    request: LoopTransform
    regions: tuple[LoopPartitionRegion, ...]
    source_identity: str
    index_bits: int
    max_static_operations: int

    def verify_current(self, original, partitioned):
        """Rebuild the exact proposal from current original bytes and compare.

        No dynamic-domain unfolding. The same positive contiguous intervals
        partition [0,extent), and every call/metadata field is transported by
        the existing substitution primitive. Original dependence, lifetime,
        source and completion proofs remain independently necessary.
        """
        if type(original) is not Kernel or original.digest() != self.source_identity:
            raise ValueError("original ordered partition source identity changed")
        if (type(self.request) is not LoopTransform or self.request.action != "partition"
                or len(self.request.names) != len(self.request.cuts) + 1):
            raise ValueError("ordered partition requires its exact recorded request")
        expected = propose_loop_transforms(original, (self.request,), index_bits=self.index_bits,
                                           max_static_operations=self.max_static_operations)
        target = _loop_at(original.body, self.request.path)
        boundaries = (0, *self.request.cuts, target.extent)
        regions = tuple(
            LoopPartitionRegion((*self.request.path[:-1], name), start, stop)
            for name, start, stop in zip(self.request.names, boundaries[:-1], boundaries[1:], strict=True)
        )
        if (type(partitioned) is not Kernel or partitioned != expected
                or self.kernel != expected or self.regions != regions):
            raise ValueError("actual ordered partition or original interval mapping changed")
        return dict(schema="merlin.ordered_loop_partition.v1", status="pass",
                    source_identity=self.source_identity, partitioned_identity=partitioned.digest(),
                    regions=len(regions), exact_instance_order=True,
                    source_and_target_admission="original independent proofs remain required")


def partition_counted_loop(kernel, path, *, cuts, names=(), index_bits=64, max_static_operations=4096):
    """Partition one current loop at explicit INTERIOR cuts without reordering.

    Sequential counted regions retain each original iteration exactly once.
    All body operands (including pointers, predicates and configurations) use
    original_index = region.start + local_index. Nested scopes and all effect
    fields survive or the ordinary structure checker refuses. No unrolling,
    target law, physical capacity, masked work or dependence permission is
    inferred. This optional proposal does not alter default emission.
    """
    request = LoopTransform("partition", path, names=names, cuts=cuts)
    actual = propose_loop_transforms(kernel, (request,), index_bits=index_bits,
                                     max_static_operations=max_static_operations)
    # Resolve actual generated names, so custody never depends on an implicit
    # naming convention or a caller's unverified member declaration.
    target = _loop_at(kernel.body, path)
    siblings, _ = _resolve(actual.body, Cursor(loops=path[:-1]))
    original_siblings, _ = _resolve(kernel.body, Cursor(loops=path[:-1]))
    position = next(i for i, child in enumerate(original_siblings) if child is target)
    names = tuple(child.var for child in siblings[position:position + len(cuts) + 1])
    request = replace(request, names=names)
    boundaries = (0, *cuts, target.extent)
    regions = tuple(LoopPartitionRegion((*path[:-1], name), start, stop)
                    for name, start, stop in zip(names, boundaries[:-1], boundaries[1:], strict=True))
    result = OrderedLoopPartition(actual, request, regions, kernel.digest(), index_bits, max_static_operations)
    result.verify_current(kernel, actual)
    return result
