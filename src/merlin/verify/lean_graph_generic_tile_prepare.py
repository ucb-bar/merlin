# ruff: noqa: E402
"""Actual singleton TILE source staging, over the frozen generic value model."""

from __future__ import annotations

import hashlib
import math

from merlin.verify.lean_graph_state_scalar import budget

from .gate_errors import GateUnavailable
from .lean_graph_generic_parser import parse_generic as parse_mlir_text
from .lean_graph_generic_prepare import (
    admission,
    builtin,
    controls_from_actual,
    frame_growth,
    outer_uses,
    project_pair,
    reindex,
)
from .lean_graph_generic_tile_roles import bind


def prepare(before, after, *, tile, index_bits=64):
    if type(tile) is not int or not 2 <= tile <= 65536 or type(index_bits) is not int or index_bits != 64:
        raise ValueError("actual TILE selection needs integer tile2..65536/index64")
    for raw in (before, after):
        if type(raw) is not str or len(raw.encode()) > 4 * 1024 * 1024:
            raise GateUnavailable("complete raw4MiB resource policy required")
        admission.admit_text(raw)
    modules = [parse_mlir_text(raw) for raw in (before, after)]
    for module in modules:
        for op in module.walk():
            for value, field in admission._fields(op):
                for attribute, path in admission._attributes(value, field):
                    if isinstance(attribute, builtin.FloatData) and math.isnan(attribute.data):
                        raise ValueError("scalar NaN literal requires raw-bit preserving parser at " + path)
    capture = project_pair(*modules)
    if any(left["declaration"] != right["declaration"] for left, right in zip(*capture["functions"], strict=True)):
        raise ValueError("complete definition/declaration roster differs")
    changed = [
        i for i, (left, right) in enumerate(zip(*capture["functions"], strict=True)) if left["body"] != right["body"]
    ]
    if len(changed) != 1:
        raise ValueError("initial generic TILE profile needs one changed defined entry")
    fi = changed[0]
    before_fn, after_fn = [stage[fi] for stage in capture["functions"]]
    if before_fn["declaration"] or len(before_fn["outputs"]) != 1:
        raise ValueError("initial generic TILE profile needs one captured return value")
    left, right = before_fn["body"], after_fn["body"]
    loops = [i for i, row in enumerate(left) if "loop" in row]
    if len(loops) != 1 or loops[0] < 3 or len(left) != loops[0] + 2:
        raise ValueError("initial generic TILE singleton source loop + return required")
    cut = loops[0] - 3
    prefix = left[:cut]
    count = len(before_fn["inputs"]) + frame_growth(prefix)
    source, target = left[cut:], right[cut:]
    if len(target) != 6 or "loop" not in target[4]:
        raise ValueError("actual one-block-step/outer-loop target shape required")
    outer_body = target[4]["loop"]["body"]
    if len(outer_body) != 4 or "loop" not in outer_body[2] or outer_body[-1] != {"ret": []}:
        raise ValueError("actual generated add/min/inner/yield target shape required")
    if outer_body[0].get("arguments") != [0, 1] or outer_body[1].get("arguments") != [0, 4]:
        raise ValueError("actual generated bound control operand slots required")
    captures = sorted(outer_uses(source, count) | outer_uses(target, count))
    controls = controls_from_actual(modules, capture)
    control_map = {row["identity"]: row for row in controls}
    setup = []
    for row in source[:3]:
        kind = control_map.get(row.get("effect"), {})
        if row.get("arguments") != [] or kind.get("kind") != "index-constant":
            raise ValueError("actual three-literal selected-loop setup required")
        setup.append(kind["value"])
    lower, upper, step = setup
    if not 0 <= lower <= upper or step < 1:
        raise ValueError("actual static nonnegative controls/positive stride required")
    roles = bind(capture, fi, cut, controls)
    work = []
    for stage in capture["functions"]:
        for function in stage:
            if not function["declaration"]:
                local = [0]
                budget(function["body"], len(function["inputs"]), local)
                work.append(local[0])
    return {
        "authority": False,
        "status": "UNCOMPILED_ACTUAL_GENERIC_TILE",
        "raw_buffers": [before, after],
        "raw_sha256": [hashlib.sha256(raw.encode()).hexdigest() for raw in (before, after)],
        "capture": capture,
        "index_operations": roles["index_operations"],
        "function": fi,
        "normalizer_work": work,
        "roles": roles,
        "prefix": prefix,
        "frame_count": count,
        "captures": captures,
        "rests": [source, target],
        "minimal": [reindex(rows, count, captures) for rows in (source, target)],
        "bounds": {"lower": lower, "upper": upper, "step": step, "tile": tile, "block_step": step * tile},
    }
