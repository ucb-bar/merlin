# ruff: noqa: E402
# Staging imports the separately frozen source bundle after its explicit path.
"""Concrete source-only actual full-entry generic-carrier query preparation."""

from __future__ import annotations

import copy
import hashlib
import math

from xdsl.dialects import builtin

from merlin.verify.lean_graph_state_pipeline_capture import index_operation
from merlin.verify.lean_graph_state_scalar import budget

from . import lean_graph_generic_admission as admission
from .gate_errors import GateUnavailable
from .lean_graph_generic_exact import canonical, exact, metadata
from .lean_graph_generic_parser import parse_generic
from .lean_graph_generic_projection import project_pair
from .lean_graph_generic_render import program as program
from .lean_graph_generic_render import types as types
from .lean_graph_generic_roles import bind as bind_control_roles


def frame_growth(records):
    return sum(
        1
        if "pure" in row
        else len(row["outputs"])
        if "effect" in row
        else len(row["loop"]["carries"])
        if "loop" in row
        else 0
        for row in records
    )


def outer_uses(records, count):
    """Extract actual free frame slots, retaining every ordered operation."""
    used = set()

    def visit(rows, environment):
        for row in rows:
            if "loop" in row:
                loop = row["loop"]
                for index in [*loop["bounds"], *loop["initial"]]:
                    if environment[index] is not None:
                        used.add(environment[index])
                visit(loop["body"], [None] * (len(loop["carries"]) + 1) + environment)
                environment = [None] * len(loop["carries"]) + environment
            else:
                for index in row.get("arguments", row.get("ret", [])):
                    if environment[index] is not None:
                        used.add(environment[index])
                if "pure" in row:
                    environment = [None] + environment
                elif "effect" in row:
                    environment = [None] * len(row["outputs"]) + environment

    visit(records, list(range(count)))
    return used


def reindex(records, count, captures):
    """Retain every instruction; Scope must check the actual frame projection."""
    fresh = 0

    def local(n):
        nonlocal fresh
        values = [("local", fresh + i) for i in range(n)]
        fresh += n
        return values

    def go(rows, old, new):
        output = []

        def uses(indices):
            return [new.index(old[i]) for i in indices]

        for row in rows:
            value = copy.deepcopy(row)
            if "loop" in row:
                loop = row["loop"]
                value["loop"]["bounds"] = uses(loop["bounds"])
                value["loop"]["initial"] = uses(loop["initial"])
                args = local(len(loop["carries"]) + 1)
                value["loop"]["body"] = go(loop["body"], args + old, args + new)
                results = local(len(loop["carries"]))
            else:
                field = "ret" if "ret" in row else "arguments"
                value[field] = uses(row[field])
                results = local(1 if "pure" in row else len(row["outputs"]) if "effect" in row else 0)
            output.append(value)
            old, new = results + old, results + new
        return output

    return go(records, [("outer", i) for i in range(count)], [("outer", i) for i in captures])


def controls_from_actual(modules, capture):
    identities = {canonical(row["descriptor"]): i for i, row in enumerate(capture["registry"])}
    controls = {}
    for module in modules:
        for op in module.walk():
            descriptor = {**metadata(op), "location": exact(op.location)}
            identity = identities.get(canonical(descriptor))
            if identity is None or len(op.results) != 1 or type(op.results[0].type) is not builtin.IndexType:
                continue
            actual = index_operation(op)
            if actual is not None:
                value = {"identity": identity, **actual}
                if identity in controls and controls[identity] != value:
                    raise ValueError("one exact actual index key has inconsistent interpretation")
                controls[identity] = value
    return [controls[i] for i in sorted(controls)]


def prepare(before, after, *, factor=2, index_bits=64):
    if type(factor) is not int or factor != 2 or type(index_bits) is not int or index_bits != 64:
        raise ValueError("initial generic bulk+tail profile requires factor2/index64")
    for raw in (before, after):
        if type(raw) is not str or len(raw.encode()) > 4 * 1024 * 1024:
            raise GateUnavailable("complete raw buffer4MiB resource policy required")
        admission.admit_text(raw)
    modules = [parse_generic(raw) for raw in (before, after)]
    # The existing dense adapter retains byte-backed NaN payloads. Scalar
    # FloatData parsing can quiet signaling NaNs, so admitting those literals
    # would make distinct raw bodies share an opaque descriptor. Runtime opaque
    # data remains unrestricted; this concerns only parsed scalar attributes.
    for module in modules:
        for op in module.walk():
            for value, field in admission._fields(op):
                for attribute, path in admission._attributes(value, field):
                    if isinstance(attribute, builtin.FloatData) and math.isnan(attribute.data):
                        raise ValueError("scalar NaN literal requires raw-bit preserving parser at " + path)
    capture = project_pair(*modules)
    if any(before["declaration"] != after["declaration"] for before, after in zip(*capture["functions"], strict=True)):
        raise ValueError("actual complete definition/declaration roster changed")
    normalizer_work = []
    for functions in capture["functions"]:
        for function in functions:
            if not function["declaration"]:
                work = [0]
                budget(function["body"], len(function["inputs"]), work)
                normalizer_work.append(work[0])
    changed = [
        i for i, pair in enumerate(zip(*capture["functions"], strict=True)) if pair[0]["body"] != pair[1]["body"]
    ]
    if len(changed) != 1:
        raise ValueError("first generic profile needs one changed defined entry")
    fi = changed[0]
    if capture["functions"][0][fi]["declaration"] or len(capture["functions"][0][fi]["outputs"]) != 1:
        raise ValueError("initial generic profile needs one defined entry with one returned value")
    left, right = [stage[fi]["body"] for stage in capture["functions"]]
    selected = [i for i, row in enumerate(left) if "loop" in row]
    if len(selected) != 1 or selected[0] < 3 or len(left) != selected[0] + 2:
        raise ValueError("selected actual loop followed by actual return required")
    cut = selected[0] - 3
    prefix = left[:cut]
    count = len(capture["functions"][0][fi]["inputs"]) + frame_growth(prefix)
    source, target = left[cut:], right[cut:]
    captures = sorted(outer_uses(source, count) | outer_uses(target, count))
    controls = controls_from_actual(modules, capture)
    # Initial actual source controls stay conservative. Source index arithmetic
    # outside an opaque structured closure still needs separate range evidence.
    control_map = {row["identity"]: row for row in controls}
    setup = []
    for row in source[:3]:
        actual = control_map.get(row.get("effect"), {})
        if row.get("arguments") != [] or actual.get("kind") != "index-constant":
            raise ValueError("actual three-literal selected-loop setup required")
        setup.append(actual["value"])
    lower, upper, step = setup
    if not 0 <= lower <= upper or step < 1:
        raise ValueError("actual nonnegative selected bounds/positive stride required")
    trips = (upper - lower + step - 1) // step
    groups, remainder = divmod(trips, factor)
    if not groups or not remainder:
        raise ValueError("initial generic profile requires an actual nonempty bulk and tail")
    # The raw interner remains an arbitrary partial meaning for all ordinary
    # events. Only this actual changed control slice receives fresh bounded
    # identities, with a finite occurrence plan checked by the emitted query.
    roles = bind_control_roles(capture, fi, cut, controls)
    controls = roles["index_operations"]
    minimal = [reindex(rows, count, captures) for rows in (source, target)]
    return {
        "authority": False,
        "status": "UNCOMPILED_ACTUAL_GENERIC_QUERY",
        "raw_buffers": [before, after],
        "raw_sha256": [hashlib.sha256(raw.encode()).hexdigest() for raw in (before, after)],
        "capture": capture,
        "index_operations": controls,
        "function": fi,
        "normalizer_work": normalizer_work,
        "roles": roles,
        "bounds": {
            "lower": lower,
            "upper": upper,
            "step": step,
            "factor": factor,
            "trips": trips,
            "groups": groups,
            "remainder": remainder,
            "stride": step * factor,
            "bulk_upper": lower + step * factor * groups,
        },
        "prefix": prefix,
        "frame_count": count,
        "captures": captures,
        "rests": [source, target],
        "minimal": minimal,
    }
