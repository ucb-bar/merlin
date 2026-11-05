"""Digest-bound loop occurrences and capture-free names for legacy cursors.

Handles identify an occurrence, not a binder spelling. They expire after any
rewrite. Explicit result handles are returned for subsequent composition; no
handle is silently rebound to a similar loop in a different program.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.sched.ir import Call, Kernel, Loop, Ptr, check_structure
from merlin.sched.ir.expr import Const, Expr, Scaled, Select, Sum, Var
from merlin.sched.primitives import PrimitiveError


def _valid(kernel):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise PrimitiveError("loop handles require structurally valid kernel IR")


def _at(kernel, path):
    if (type(path) is not tuple or not path or len(path) > 64
            or any(type(index) is not int or index < 0 for index in path)):
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
        return replace(value, var=names.get(value.var, value.var),
                       then=_expression(value.then, names), other=_expression(value.other, names))
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
                result.append(replace(statement, var=name,
                    body=walk(statement.body, {**scope, statement.var: name}, here)))
            elif type(statement) is Call:
                fields = {field: tuple((key, _expression(value, scope)) for key, value in getattr(statement, field))
                          for field in ("args", "sets", "assumes")}
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
    from merlin.compile.scheduling.loop_schedule import LoopTransform, propose_loop_transforms

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
    replacements = tuple(locate_loop(actual, (*handle.statement_path[:-1], index))
                         for index in range(start, stop) if type(after_body[index]) is Loop)
    return LoopEdit(actual, handle, hygiene, replacements)
