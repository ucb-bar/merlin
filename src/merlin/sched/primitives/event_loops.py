"""Ordered counted rewrites with lexical completion ownership.

These transformations preserve issue/await order and all command fields. They
neither establish a target's completion law nor authorize dependency reordering;
the actual SCF stage checker and target contracts must still admit emission.
"""
from __future__ import annotations

from dataclasses import replace

from merlin.sched.ir import Const, Loop, Var, check_structure


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
            result.append(replace(op, body=child, iter_args=tuple(zip(local, initial, strict=True)),
                                  yields=tuple(yielded.get(n, n) for n in op.yields), results=outputs))
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
    from . import NotApplicable

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
            suffix, mapping = tail(body[index + 1:], dict(aliases))
            return (*body[:index], *replacement, *suffix), mapping
        op = body[index]
        child, mapping = visit(op.body, remaining[1:])
        rebuilt = replace(op, body=child, yields=tuple(mapping.get(n, n) for n in op.yields))
        return (*body[:index], rebuilt, *body[index + 1:]), {}

    body, _ = visit(kernel.body, path)
    result = replace(kernel, body=tuple(body))
    errors = check_structure(result)
    if errors:
        raise NotApplicable("ordered completion transform cannot preserve lexical ownership: " + errors[0])
    return result


def unroll_event_loop(kernel, path):
    """Inline bounded copies, threading yields and alpha-renaming all events."""
    from . import NotApplicable, _loop_at, _map_calls, _subst_call

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
    from . import NotApplicable, _bound_vars, _loop_at, _map_calls, _subst_call

    target = _loop_at(kernel.body, path)
    if (type(boundaries) is not tuple or not 3 <= len(boundaries) <= 17
            or any(type(n) is not int for n in boundaries)
            or boundaries[0] != 0 or boundaries[-1] != target.extent
            or tuple(sorted(set(boundaries))) != boundaries
            or type(names) is not tuple or len(names) != len(boundaries) - 1
            or len(set(names)) != len(names)
            or any(type(n) is not str or not n.isidentifier() or n in _bound_vars(kernel.body)
                   or n in {arg.name for arg in kernel.args} for n in names)):
        raise NotApplicable("completion partitions require exact complete increasing bounds and fresh region names")
    fresh = _allocator(kernel)
    current = tuple(n for _, n in target.iter_args)
    parts = []
    for number, (name, start, stop) in enumerate(zip(names, boundaries[:-1], boundaries[1:], strict=True)):
        body = _map_calls(target.body, lambda call: _subst_call(
            _subst_call(call, target.var, Var(target.var) + start), target.var, Var(name)))
        outputs = target.results if number == len(names) - 1 else tuple(fresh(n) for n in target.results)
        parts.append(replace(target, var=name, extent=stop - start, body=body,
                             iter_args=tuple((n, v) for (n, _), v in zip(target.iter_args, current, strict=True)),
                             results=outputs))
        current = outputs
    return _replace_at(kernel, path, tuple(parts), {})


def split_event_loop(kernel, path, factor, names):
    """Quotient/remainder split with carried bulk, nested yield and ordered tail."""
    from . import NotApplicable, _bound_vars, _divide_call, _loop_at, _map_calls, _subst_call

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
    inner_loop = replace(target, var=inner, extent=factor, body=body,
                         iter_args=tuple((n, v) for (n, _), v in zip(target.iter_args, local, strict=True)),
                         results=inner_results)
    outer_loop = Loop(outer, full, (inner_loop,),
                      tuple((n, v) for n, (_, v) in zip(local, target.iter_args, strict=True)),
                      inner_results, outer_results)
    parts = [outer_loop]
    if remainder:
        body = _map_calls(target.body, lambda call: _subst_call(call, target.var, Var(target.var) + full * factor))
        parts.append(replace(target, extent=remainder, body=body,
                             iter_args=tuple((n, v) for (n, _), v in
                                             zip(target.iter_args, outer_results, strict=True))))
    return _replace_at(kernel, path, tuple(parts), {})
