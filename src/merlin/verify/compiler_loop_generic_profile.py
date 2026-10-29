"""Pre-producer selection for the bounded generic ordered-loop model.

OutsideProfile is a coverage exclusion: ordinary compilation/optimization continues,
with an explicit UNVERIFIED end-of-compilation record and no proof credit.
Once this predicate succeeds, producer/preparation/Lean failures must block the
selected optimization rather than being converted to a later exclusion.
"""

from __future__ import annotations

import hashlib
import math

from xdsl.dialects import scf

from merlin.verify.lean_graph_state_scalar import budget

from .lean_graph_generic_parser import parse_generic
from .lean_graph_generic_prepare import admission, builtin, controls_from_actual, frame_growth, project_pair

MAX_RAW_BYTES = 4 * 1024 * 1024
MAX_TILE = 65536
DOMAIN = "compiler-static-ordered-scf-generic-value-index64-v1"


class OutsideProfile(ValueError):
    """An original source lies outside this explicit bounded proof model; no gate credit."""


def source_profile(raw, *, kind, factor=2, tile=3, index_bits=64, depth=1):
    if type(raw) is not str or len(raw.encode()) > MAX_RAW_BYTES:
        raise OutsideProfile("generic source must fit the4MiB declared raw intake policy")
    if kind not in ("ordered_loop_unrolling", "ordered_loop_segmentation"):
        raise OutsideProfile("generic profile supports one singleton ordered UNROLL or TILE")
    if type(index_bits) is not int or index_bits != 64 or type(depth) is not int or depth != 1:
        raise OutsideProfile("generic profile supports signed-index64 top-level selected loops")
    if kind == "ordered_loop_unrolling" and (type(factor) is not int or factor != 2):
        raise OutsideProfile("generic UNROLL profile supports factor2 bulk plus one tail")
    if kind == "ordered_loop_segmentation" and (type(tile) is not int or not 2 <= tile <= MAX_TILE):
        raise OutsideProfile("generic TILE profile uses tile2..65536 as finite resource policy")
    try:
        admission.admit_text(raw)
        module = parse_generic(raw)
        for operation in module.walk():
            for value, field in admission._fields(operation):
                for attribute, path in admission._attributes(value, field):
                    if isinstance(attribute, builtin.FloatData) and math.isnan(attribute.data):
                        raise OutsideProfile("scalar NaN literal lacks lossless parser correspondence at " + path)
        if sum(type(operation) is scf.ForOp for operation in module.walk()) != 1:
            raise OutsideProfile("generic profile requires one actual SCF loop including opaque region closure")
        captured = project_pair(module, module)
        functions = captured["functions"][0]
        all_loops = []

        def walk(rows, function, depth):
            for position, row in enumerate(rows):
                if "loop" in row:
                    all_loops.append((function, depth, position, row["loop"]))
                    walk(row["loop"]["body"], function, depth + 1)

        for i, function in enumerate(functions):
            if not function["declaration"]:
                work = [0]
                budget(function["body"], len(function["inputs"]), work)
                walk(function["body"], i, 1)
        if len(all_loops) != 1 or all_loops[0][1] != 1:
            raise OutsideProfile("generic profile requires one module SCF loop, at top level")
        fi, _, position, loop = all_loops[0]
        function = functions[fi]
        rows = function["body"]
        if len(function["outputs"]) != 1 or position < 3 or len(rows) != position + 2:
            raise OutsideProfile("generic source requires three setup literals, loop, one captured return")
        if loop["bounds"] != [2, 1, 0] or loop["carries"] or loop["initial"]:
            raise OutsideProfile("generic source requires exact selected setup slots and zero carries")
        body = loop["body"]
        if len(body) != 2 or body[-1] != {"ret": []}:
            raise OutsideProfile("generic body requires one opaque zero-result event and yield")
        event = body[0]
        arguments = event.get("arguments", [])
        if "effect" not in event or event["outputs"] or len(arguments) != 2 or arguments[0] != 0 or arguments[1] < 4:
            raise OutsideProfile("generic body event requires actual IV and one captured data argument")
        if rows[-1].get("ret") is None or len(rows[-1]["ret"]) != 1 or rows[-1]["ret"][0] < 3:
            raise OutsideProfile("generic entry must return one value captured outside the setup")
        controls = {row["identity"]: row for row in controls_from_actual((module,), captured)}
        setup = []
        for row in rows[position - 3 : position]:
            selected = controls.get(row.get("effect"), {})
            if row.get("arguments") != [] or selected.get("kind") != "index-constant":
                raise OutsideProfile("generic selected controls must be actual index literals")
            setup.append(selected["value"])
        lower, upper, step = setup
        if not 0 <= lower <= upper or step < 1:
            raise OutsideProfile("generic selected controls need nonnegative bounds and positive stride")
        if kind == "ordered_loop_unrolling":
            trips = (upper - lower + step - 1) // step
            if trips < 3 or trips % 2 != 1:
                raise OutsideProfile("generic UNROLL source needs nonempty factor2 bulk and remainder1")
            growth = step * factor
        else:
            trips = (upper - lower + step - 1) // step
            if trips < 1:
                raise OutsideProfile("initial generic TILE source requires at least one iteration")
            growth = step * tile
        if upper + growth >= 2**63 or upper + step >= 2**63:
            raise OutsideProfile("generic selected/generated index controls do not fit signed64")
        frame_count = len(function["inputs"]) + frame_growth(rows[: position - 3])
        if arguments[1] >= frame_count + 4 or rows[-1]["ret"][0] >= frame_count + 3:
            raise OutsideProfile("generic captured data/return slots are outside actual lexical frame")
    except OutsideProfile:
        raise
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise OutsideProfile(str(error)) from error
    return {
        "supported": True,
        "domain": DOMAIN,
        "kind": kind,
        "function": fi,
        "depth": 1,
        "factor": factor,
        "tile": tile,
        "index_bits": 64,
        "lower": lower,
        "upper": upper,
        "step": step,
        "trips": trips,
        "frame_count": frame_count,
        "source_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "projection": "trusted-occurrence-aware-controls-with-closed-role-types-and-Fits",
        "proof_credit": False,
    }
