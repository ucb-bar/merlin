"""Lexical completion-token binding; no target timing or completion inference."""
from __future__ import annotations

from .kernel import Loop


def has_carried_events(body):
    return any(isinstance(op, Loop) and (op.iter_args or op.yields or op.results or has_carried_events(op.body))
               for op in body)


def lexical_event_errors(kernel):
    errors = []

    def name(value):
        return (type(value) is str and value and (value[0].isalpha() or value[0] == "_")
                and all(c.isalnum() or c == "_" for c in value))

    def body(statements, incoming):
        available = set(incoming)
        for op in statements:
            if isinstance(op, Loop):
                if (type(op.iter_args) is not tuple or type(op.yields) is not tuple
                        or type(op.results) is not tuple
                        or len(op.iter_args) != len(op.yields) or len(op.yields) != len(op.results)
                        or any(type(pair) is not tuple or len(pair) != 2 for pair in op.iter_args)):
                    errors.append(f"loop {op.var!r}: completion input/yield/result arities differ")
                    continue
                inputs = tuple(pair[0] for pair in op.iter_args)
                initial = tuple(pair[1] for pair in op.iter_args)
                if (any(not name(n) for n in (*inputs, *initial, *op.yields, *op.results))
                        or any(len(set(v)) != len(v) for v in (inputs, initial, op.yields, op.results))):
                    errors.append(f"loop {op.var!r}: malformed or aliased completion bindings")
                    continue
                if not set(initial) <= available:
                    errors.append(f"loop {op.var!r}: incoming completion has no dominating producer")
                child = body(op.body, inputs)
                if not set(op.yields) <= child:
                    errors.append(f"loop {op.var!r}: yielded completion is outside its lexical scope")
                # A carry moves ownership. The old name cannot accidentally await a
                # stale handle instead of the result of the final iteration.
                available.difference_update(initial)
                if set(op.results) & available:
                    errors.append(f"loop {op.var!r}: completion result shadows a live lexical name")
                available.update(op.results)
            else:
                if any(not name(n) for n in op.awaits) or not set(op.awaits) <= available:
                    errors.append(f"{op.instr}: awaits an unknown or out-of-scope completion")
                if op.produces:
                    if not name(op.produces) or op.produces in available:
                        errors.append(f"{op.instr}: completion producer shadows a live lexical name")
                    available.add(op.produces)
        return available

    body(kernel.body, ())
    return errors
