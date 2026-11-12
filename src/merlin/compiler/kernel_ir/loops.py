"""Counted loop rewrites with lexical completion ownership, and digest-bound loop handles."""

from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.compiler.kernel_ir.ir import Call, Const, Expr, Kernel, Loop, Ptr, Scaled, Select, Sum, Var, check_structure
from merlin.compiler.kernel_ir.primitives import PrimitiveError

# ---- event_loops -------------------------------------------------------------------------------


def _allocator(kernel):
    used = set()

    def visit(body):
        for op in body:
            if type(op) is Loop:
                used.add(op.var)
                used.update(n for pair in op.iter_args for n in pair)
                used.update((*op.yields, *op.results))
                visit(op.body)
            else:
                used.update(op.awaits)
                if op.produces:
                    used.add(op.produces)

    visit(kernel.body)

    def fresh(stem):
        index = 0
        while (value := f"{stem}_event_{index}") in used:
            index += 1
        used.add(value)
        return value

    return fresh


def _clone(body, bindings, fresh):
    """Alpha-clone each producer and lexical loop binder, using actual bindings."""
    env, result = dict(bindings), []
    for op in body:
        if type(op) is Loop:
            initial = tuple(env.get(n, n) for _, n in op.iter_args)
            local = tuple(fresh(n) for n, _ in op.iter_args)
            child, yielded = _clone(op.body, dict(zip((n for n, _ in op.iter_args), local, strict=True)), fresh)
            outputs = tuple(fresh(n) for n in op.results)
            result.append(
                replace(
                    op,
                    body=child,
                    iter_args=tuple(zip(local, initial, strict=True)),
                    yields=tuple(yielded.get(n, n) for n in op.yields),
                    results=outputs,
                )
            )
            for _, old in op.iter_args:
                env.pop(old, None)
            env.update(zip(op.results, outputs, strict=True))
        else:
            output = fresh(op.produces) if op.produces else None
            result.append(replace(op, awaits=tuple(env.get(n, n) for n in op.awaits), produces=output))
            if output:
                env[op.produces] = output
    return tuple(result), env


def _replace_at(kernel, path, replacement, aliases):
    """Rewrite subsequent uses and enclosing yields after eliminating a binder.

    Aliases never cross a child scope except via that child's explicit inputs.
    Loop results themselves keep their original enclosing names.
    """
    from merlin.compiler.kernel_ir.primitives import NotApplicable

    def tail(body, mapping):
        result = []
        for op in body:
            if type(op) is Loop:
                result.append(replace(op, iter_args=tuple((n, mapping.get(v, v)) for n, v in op.iter_args)))
                # A legal source cannot shadow a live event; results nevertheless
                # terminate any mapping with the same spelling in this scope.
                mapping = {k: v for k, v in mapping.items() if k not in op.results}
            else:
                result.append(replace(op, awaits=tuple(mapping.get(n, n) for n in op.awaits)))
                if op.produces in mapping:
                    mapping = {k: v for k, v in mapping.items() if k != op.produces}
        return tuple(result), mapping

    def visit(body, remaining):
        matches = [i for i, op in enumerate(body) if type(op) is Loop and op.var == remaining[0]]
        if len(matches) != 1:
            raise NotApplicable("completion transform needs an unambiguous current loop path")
        index = matches[0]
        if len(remaining) == 1:
            suffix, mapping = tail(body[index + 1 :], dict(aliases))
            return (*body[:index], *replacement, *suffix), mapping
        op = body[index]
        child, mapping = visit(op.body, remaining[1:])
        rebuilt = replace(op, body=child, yields=tuple(mapping.get(n, n) for n in op.yields))
        return (*body[:index], rebuilt, *body[index + 1 :]), {}

    body, _ = visit(kernel.body, path)
    result = replace(kernel, body=tuple(body))
    errors = check_structure(result)
    if errors:
        raise NotApplicable("ordered completion transform cannot preserve lexical ownership: " + errors[0])
    return result


def unroll_event_loop(kernel, path):
    """Inline bounded copies, threading yields and alpha-renaming all events."""
    from merlin.compiler.kernel_ir.primitives import NotApplicable, _loop_at, _map_calls, _subst_call

    target = _loop_at(kernel.body, path)
    if not 1 <= target.extent <= 4096:
        raise NotApplicable("completion unroll exceeds the existing 4096-copy bound")
    fresh = _allocator(kernel)
    current = tuple(n for _, n in target.iter_args)
    copies = []
    for index in range(target.extent):
        body = _map_calls(target.body, lambda call: _subst_call(call, target.var, Const(index)))
        body, names = _clone(body, dict(zip((n for n, _ in target.iter_args), current, strict=True)), fresh)
        copies.extend(body)
        current = tuple(names.get(n, n) for n in target.yields)
    return _replace_at(kernel, path, tuple(copies), dict(zip(target.results, current, strict=True)))


def partition_event_loop(kernel, path, boundaries, names):
    """Counted adjacent regions with explicit result-to-input ownership edges."""
    from merlin.compiler.kernel_ir.primitives import NotApplicable, _bound_vars, _loop_at, _map_calls, _subst_call

    target = _loop_at(kernel.body, path)
    if (
        type(boundaries) is not tuple
        or not 3 <= len(boundaries) <= 17
        or any(type(n) is not int for n in boundaries)
        or boundaries[0] != 0
        or boundaries[-1] != target.extent
        or tuple(sorted(set(boundaries))) != boundaries
        or type(names) is not tuple
        or len(names) != len(boundaries) - 1
        or len(set(names)) != len(names)
        or any(
            type(n) is not str
            or not n.isidentifier()
            or n in _bound_vars(kernel.body)
            or n in {arg.name for arg in kernel.args}
            for n in names
        )
    ):
        raise NotApplicable("completion partitions require exact complete increasing bounds and fresh region names")
    fresh = _allocator(kernel)
    current = tuple(n for _, n in target.iter_args)
    parts = []
    for number, (name, start, stop) in enumerate(zip(names, boundaries[:-1], boundaries[1:], strict=True)):
        body = _map_calls(
            target.body,
            lambda call: _subst_call(_subst_call(call, target.var, Var(target.var) + start), target.var, Var(name)),
        )
        outputs = target.results if number == len(names) - 1 else tuple(fresh(n) for n in target.results)
        parts.append(
            replace(
                target,
                var=name,
                extent=stop - start,
                body=body,
                iter_args=tuple((n, v) for (n, _), v in zip(target.iter_args, current, strict=True)),
                results=outputs,
            )
        )
        current = outputs
    return _replace_at(kernel, path, tuple(parts), {})


def split_event_loop(kernel, path, factor, names):
    """Quotient/remainder split with carried bulk, nested yield and ordered tail."""
    from merlin.compiler.kernel_ir.primitives import (
        NotApplicable,
        _bound_vars,
        _divide_call,
        _loop_at,
        _map_calls,
        _subst_call,
    )

    target = _loop_at(kernel.body, path)
    full, remainder = divmod(target.extent, factor)
    if not full:
        raise NotApplicable("completion split factor exceeds current loop extent")
    if len(set(names)) != 2 or any(n in _bound_vars(kernel.body) or n in {a.name for a in kernel.args} for n in names):
        raise NotApplicable("completion split loop names collide with original bindings")
    outer, inner = names
    fresh = _allocator(kernel)
    local = tuple(fresh(n) for n, _ in target.iter_args)
    inner_results = tuple(fresh(n) for n in target.results)
    outer_results = tuple(fresh(n) for n in target.results) if remainder else target.results
    body = _map_calls(target.body, lambda call: _divide_call(call, target.var, Var(outer), Var(inner), factor))
    inner_loop = replace(
        target,
        var=inner,
        extent=factor,
        body=body,
        iter_args=tuple((n, v) for (n, _), v in zip(target.iter_args, local, strict=True)),
        results=inner_results,
    )
    outer_loop = Loop(
        outer,
        full,
        (inner_loop,),
        tuple((n, v) for n, (_, v) in zip(local, target.iter_args, strict=True)),
        inner_results,
        outer_results,
    )
    parts = [outer_loop]
    if remainder:
        body = _map_calls(target.body, lambda call: _subst_call(call, target.var, Var(target.var) + full * factor))
        parts.append(
            replace(
                target,
                extent=remainder,
                body=body,
                iter_args=tuple((n, v) for (n, _), v in zip(target.iter_args, outer_results, strict=True)),
            )
        )
    return _replace_at(kernel, path, tuple(parts), {})


# ---- loop_handles ------------------------------------------------------------------------------
# Digest-bound loop occurrences and capture-free names for legacy cursors.
#
# Handles identify an occurrence, not a binder spelling. They expire after any
# rewrite. Explicit result handles are returned for subsequent composition; no
# handle is silently rebound to a similar loop in a different program.


def _valid(kernel):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise PrimitiveError("loop handles require structurally valid kernel IR")


def _at(kernel, path):
    if (
        type(path) is not tuple
        or not path
        or len(path) > 64
        or any(type(index) is not int or index < 0 for index in path)
    ):
        raise PrimitiveError("loop occurrence requires a nonempty bounded structural path")
    body = kernel.body
    for index in path:
        if index >= len(body) or type(body[index]) is not Loop:
            raise PrimitiveError("structural path does not identify a loop occurrence")
        target = body[index]
        body = target.body
    return target


@dataclass(frozen=True)
class LoopHandle:
    kernel_sha256: str
    statement_path: tuple[int, ...]
    loop: Loop

    def resolve(self, kernel):
        _valid(kernel)
        if kernel.digest() != self.kernel_sha256:
            raise PrimitiveError("stale loop handle: original kernel identity changed")
        actual = _at(kernel, self.statement_path)
        if actual != self.loop:
            raise PrimitiveError("loop handle content differs from its selected occurrence")
        return actual


def locate_loop(kernel, statement_path):
    _valid(kernel)
    return LoopHandle(kernel.digest(), statement_path, _at(kernel, statement_path))


def loop_handles(kernel):
    """All occurrences in source order, including identical sibling loops."""
    _valid(kernel)
    result = []

    def walk(body, path):
        for index, statement in enumerate(body):
            if type(statement) is Loop:
                here = (*path, index)
                result.append(locate_loop(kernel, here))
                walk(statement.body, here)

    walk(kernel.body, ())
    return tuple(result)


def _expression(value, names):
    if type(value) is Const:
        return value
    if type(value) is Var:
        return Var(names.get(value.name, value.name))
    if type(value) is Sum:
        return replace(value, terms=tuple(_expression(term, names) for term in value.terms))
    if type(value) is Scaled:
        return replace(value, expr=_expression(value.expr, names))
    if type(value) is Select:
        return replace(
            value,
            var=names.get(value.var, value.var),
            then=_expression(value.then, names),
            other=_expression(value.other, names),
        )
    if isinstance(value, Expr):
        raise PrimitiveError("binder hygiene cannot interpret an unknown index expression")
    if type(value) is Ptr:
        return replace(value, offset=_expression(value.offset, names))
    # Instruction names, tensor identities, state keys and completion tokens
    # are separate namespaces, not strings to substitute as integer binders.
    return value


@dataclass(frozen=True)
class LoopHygiene:
    original: Kernel
    kernel: Kernel
    bindings: tuple[tuple[tuple[int, ...], str, str], ...]

    @property
    def original_sha256(self):
        return self.original.digest()

    def handle(self, original_handle):
        if type(original_handle) is not LoopHandle or original_handle.kernel_sha256 != self.original_sha256:
            raise PrimitiveError("hygiene handle belongs to another original kernel")
        original_handle.resolve(self.original)
        return locate_loop(self.kernel, original_handle.statement_path)


def uniquify_loop_binders(kernel, *, prefix="schedule_loop"):
    """Explicit alpha-renaming of every integer binder, including call state values.

    All non-index fields, event inputs/yields/results and lexical scopes remain
    unchanged. This grants no dependence, alias or target completion permission.
    The ordinary final source and hazard checks remain required after scheduling.
    """
    _valid(kernel)
    if type(prefix) is not str or not prefix.isidentifier() or len(prefix) > 128:
        raise PrimitiveError("binder hygiene requires an identifier prefix")
    used = {arg.name for arg in kernel.args}
    for handle in loop_handles(kernel):
        used.add(handle.loop.var)
        used.update(name for pair in handle.loop.iter_args for name in pair)
        used.update((*handle.loop.yields, *handle.loop.results))

    def reserve_events(body):
        for statement in body:
            if type(statement) is Loop:
                reserve_events(statement.body)
            else:
                used.update(statement.awaits)
                if statement.produces:
                    used.add(statement.produces)

    reserve_events(kernel.body)
    bindings = []
    serial = 0

    def fresh():
        nonlocal serial
        while True:
            name = prefix + "_" + str(serial)
            serial += 1
            if name not in used:
                used.add(name)
                return name

    def walk(body, scope, path):
        result = []
        for index, statement in enumerate(body):
            here = (*path, index)
            if type(statement) is Loop:
                name = fresh()
                bindings.append((here, statement.var, name))
                result.append(
                    replace(statement, var=name, body=walk(statement.body, {**scope, statement.var: name}, here))
                )
            elif type(statement) is Call:
                fields = {
                    field: tuple((key, _expression(value, scope)) for key, value in getattr(statement, field))
                    for field in ("args", "sets", "assumes")
                }
                result.append(replace(statement, **fields))
            else:
                raise PrimitiveError("binder hygiene requires typed Loop/Call statements")
        return tuple(result)

    actual = replace(kernel, body=walk(kernel.body, {}, ()))
    _valid(actual)
    return LoopHygiene(kernel, actual, tuple(bindings))


@dataclass(frozen=True)
class LoopEdit:
    kernel: Kernel
    original: LoopHandle
    hygiene: LoopHygiene
    replacement_loops: tuple[LoopHandle, ...]


def transform_loop(kernel, handle, *, action, factor=None, cuts=(), names=()):
    """Apply existing split/unroll/partition semantics to one exact occurrence.

    Ordered split/partition regions receive explicit new handles. Other moves
    continue to use their existing source-family APIs until their handle mapping
    is specified. This helper does not guess a descendant after reorder/fusion.
    """
    from merlin.compiler.schedule.loop_schedule import LoopTransform, propose_loop_transforms

    if type(handle) is not LoopHandle:
        raise PrimitiveError("typed loop occurrence handle required")
    handle.resolve(kernel)
    if action not in ("split", "unroll", "partition"):
        raise PrimitiveError("loop handle edit supports split, unroll and ordered partition only")
    hygiene = uniquify_loop_binders(kernel)
    path = []
    body = hygiene.kernel.body
    for index in handle.statement_path:
        path.append(body[index].var)
        body = body[index].body
    request = LoopTransform(action, tuple(path), factor=factor, cuts=cuts, names=names)
    actual = propose_loop_transforms(hygiene.kernel, (request,))
    before_body, after_body = hygiene.kernel.body, actual.body
    for index in handle.statement_path[:-1]:
        before_body, after_body = before_body[index].body, after_body[index].body
    start = handle.statement_path[-1]
    suffix = len(before_body) - start - 1
    if before_body[:start] != after_body[:start] or (suffix and before_body[-suffix:] != after_body[-suffix:]):
        raise PrimitiveError("loop edit changed statements outside the selected occurrence")
    stop = len(after_body) - suffix
    replacements = tuple(
        locate_loop(actual, (*handle.statement_path[:-1], index))
        for index in range(start, stop)
        if type(after_body[index]) is Loop
    )
    return LoopEdit(actual, handle, hygiene, replacements)
