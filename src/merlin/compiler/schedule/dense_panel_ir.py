"""Structural traversal and serial projection of dense panel command IR."""

from __future__ import annotations

from dataclasses import replace

from merlin.sched.ir import Call, Loop

DRAIN = "merlin_dense_panel_drain"


def _calls(body):
    for statement in body:
        if type(statement) is Loop:
            yield from _calls(statement.body)
        else:
            yield statement


def _loops(body):
    for statement in body:
        if type(statement) is Loop:
            yield statement
            yield from _loops(statement.body)


def _serial_projection(kernel):
    def body(statements):
        return tuple(
            replace(s, body=body(s.body)) if type(s) is Loop else s
            for s in statements
            if not (type(s) is Call and s.instr == DRAIN)
        )

    return replace(kernel, body=body(kernel.body))
