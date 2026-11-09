"""Closed bounded feedback values; no pickle or caller-selected reconstruction.

Only evaluator-owned scalar/collection data and the two existing cost value types
cross this transport. Limits and deadline checks refuse late or excessive results;
bounded local decoding is not an interruptible OS or physical-runtime deadline.
"""
from __future__ import annotations

import json
import math
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path

from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from .contracts import StageGateError

SCHEMA = "merlin.feedback_value.v1"


@dataclass(frozen=True)
class FeedbackValueLimits:
    max_bytes: int = 8 * 1024 * 1024
    max_nodes: int = 65536
    max_depth: int = 64
    max_string_bytes: int = 128 * 1024

    def verify(self):
        ceilings = (8 * 1024 * 1024, 65536, 64, 128 * 1024)
        values = (self.max_bytes, self.max_nodes, self.max_depth, self.max_string_bytes)
        if any(type(value) is not int or not 0 < value <= ceiling
               for value, ceiling in zip(values, ceilings, strict=True)):
            raise StageGateError("component feedback requires bounded byte/node/depth/string limits")
        return self


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise StageGateError("component feedback exceeded result deadline")


class _Budget:
    def __init__(self, limits, deadline=None):
        if type(limits) is not FeedbackValueLimits:
            raise StageGateError("component feedback requires typed result limits")
        self.limits, self.deadline, self.nodes = limits.verify(), deadline, 0

    def visit(self, depth):
        check_deadline(self.deadline)
        self.nodes += 1
        if depth > self.limits.max_depth or self.nodes > self.limits.max_nodes:
            raise StageGateError("component feedback exceeded result depth or node limit")

    def scalar(self, value):
        if type(value) is str:
            if len(value.encode("utf-8")) > self.limits.max_string_bytes:
                raise StageGateError("component feedback exceeded scalar string limit")
        elif type(value) is int:
            if value.bit_length() > 128:
                raise StageGateError("component feedback exceeded scalar integer limit")
        elif type(value) is float and not math.isfinite(value):
            raise StageGateError("component feedback contains nonfinite scalar")
        elif type(value) not in (type(None), bool, float):
            raise StageGateError("component feedback contains unsupported scalar")
        return value


def _encode(value, budget, depth=0):
    budget.visit(depth)
    kind = type(value)
    if kind in (type(None), bool, int, float, str):
        return ["scalar", budget.scalar(value)]
    if kind in (tuple, list):
        if len(value) > budget.limits.max_nodes:
            raise StageGateError("component feedback exceeded collection node limit")
        return ["tuple" if kind is tuple else "list", [_encode(item, budget, depth + 1) for item in value]]
    if kind is dict:
        if len(value) > budget.limits.max_nodes:
            raise StageGateError("component feedback exceeded mapping node limit")
        return ["dict", [[_encode(key, budget, depth + 1), _encode(item, budget, depth + 1)]
                         for key, item in value.items()]]
    if kind is CycleInterval:
        return ["interval", _encode((value.lo, value.hi, value.provenance, value.missing), budget, depth + 1)]
    # This exact existing evaluator value type has no arbitrary reconstruction hook.
    from .component_analytical import ComponentAnalyticalResults

    if kind is ComponentAnalyticalResults:
        return ["analytical", _encode((dict(value), value.reports, value.screening), budget, depth + 1)]
    raise StageGateError("component feedback contains unsupported result type")


def _decode(value, budget, depth=0):
    budget.visit(depth)
    if type(value) is not list or len(value) != 2 or type(value[0]) is not str:
        raise StageGateError("component feedback value frame is malformed")
    tag, body = value
    if tag == "scalar":
        return budget.scalar(body)
    if tag in ("tuple", "list", "dict"):
        if type(body) is not list or len(body) > budget.limits.max_nodes:
            raise StageGateError("component feedback collection frame exceeds its limit")
        if tag != "dict":
            values = [_decode(item, budget, depth + 1) for item in body]
            return tuple(values) if tag == "tuple" else values
        result = {}
        for pair in body:
            if type(pair) is not list or len(pair) != 2:
                raise StageGateError("component feedback mapping entry is malformed")
            key, item = (_decode(part, budget, depth + 1) for part in pair)
            try:
                if key in result:
                    raise StageGateError("component feedback mapping has duplicate keys")
                result[key] = item
            except TypeError as error:
                raise StageGateError("component feedback mapping key is not plain hashable data") from error
        return result
    if tag == "interval":
        fields = _decode(body, budget, depth + 1)
        if (type(fields) is not tuple or len(fields) != 4
            or any(type(item) not in (type(None), float, int) for item in fields[:2])
            or any(type(item) is not tuple or any(type(s) is not str for s in item) for item in fields[2:])):
            raise StageGateError("component feedback interval frame is malformed")
        return CycleInterval(*fields)
    if tag == "analytical":
        from .component_analytical import ComponentAnalyticalResults

        fields = _decode(body, budget, depth + 1)
        if type(fields) is not tuple or len(fields) != 3 or any(type(item) is not dict for item in fields[:2]):
            raise StageGateError("component feedback analytical frame is malformed")
        return ComponentAnalyticalResults(*fields)
    raise StageGateError("component feedback result tag is outside the closed protocol")


def encode_feedback_value(value, *, limits):
    frame = {"schema": SCHEMA, "value": _encode(value, _Budget(limits))}
    data = json.dumps(frame, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(data) > limits.max_bytes:
        raise StageGateError("component feedback exceeded received byte limit")
    return data


def _syntax_depth(data, limits, deadline):
    """Bound nesting before JSON builds objects, including braces inside strings."""
    depth, units, quoted, escaped = 0, 0, False, False
    for offset, byte in enumerate(data):
        if offset % 4096 == 0:
            check_deadline(deadline)
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
            units += 1
        elif byte in (91, 123):
            depth += 1
            units += 1
            if depth > 4 * limits.max_depth + 8:
                raise StageGateError("component feedback exceeded received syntax depth limit")
        elif byte in (44, 58):
            units += 1
        elif byte in (93, 125):
            depth -= 1
            if depth < 0:
                raise StageGateError("component feedback result syntax is malformed")
        if units > 6 * limits.max_nodes + 16:
            raise StageGateError("component feedback exceeded received syntax node limit")


def decode_feedback_value(data, *, limits, deadline=None):
    budget = _Budget(limits, deadline)
    check_deadline(deadline)
    if type(data) is not bytes or len(data) > limits.max_bytes:
        raise StageGateError("component feedback exceeded received byte limit")
    _syntax_depth(data, limits, deadline)
    try:
        frame = json.loads(data, object_pairs_hook=_unique_object)
        check_deadline(deadline)
        if type(frame) is not dict or set(frame) != {"schema", "value"} or frame["schema"] != SCHEMA:
            raise StageGateError("component feedback result schema differs")
        value = _decode(frame["value"], budget)
        check_deadline(deadline)
        return value
    except (ValueError, TypeError, RecursionError, OverflowError) as error:
        raise StageGateError("component feedback result decoding refused") from error


def receive_feedback_value(path: Path, *, limits, deadline):
    """Read at most the admitted bytes from an owned plain file, then decode data."""
    if type(limits) is not FeedbackValueLimits:
        raise StageGateError("component feedback requires typed result limits")
    limits.verify()
    check_deadline(deadline)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077
            or info.st_size > limits.max_bytes):
            raise StageGateError("component feedback result is not an owned bounded plain file")
        data = bytearray()
        while True:
            check_deadline(deadline)
            block = os.read(descriptor, min(65536, limits.max_bytes + 1 - len(data)))
            if not block:
                break
            data.extend(block)
            if len(data) > limits.max_bytes:
                raise StageGateError("component feedback exceeded received byte limit")
        if len(data) != info.st_size:
            raise StageGateError("component feedback result changed during reception")
        result = decode_feedback_value(bytes(data), limits=limits, deadline=deadline)
        return result, len(data)
    finally:
        os.close(descriptor)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StageGateError("component feedback result has duplicate object keys")
        result[key] = value
    return result
