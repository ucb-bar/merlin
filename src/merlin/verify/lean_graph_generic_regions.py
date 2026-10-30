"""SOURCE CANDIDATE ONLY: exact closures for unchanged structured operations.

This does not grant compiler eligibility or a tensor/floating value model.
Complete input admission is retained before any opaque closure is extracted.
The returned complete descriptor, not its diagnostic hash, is the registry key.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import lean_graph_generic_admission as admission
from .lean_graph_generic_exact import canonical, exact, metadata
from .lean_graph_input_admission import ModelRefusal


@dataclass(frozen=True)
class RegionClosure:
    descriptor: dict
    captures: tuple
    arguments: tuple[int, ...]

    @property
    def key(self):
        return canonical(self.descriptor)

    @property
    def sha256(self):
        return hashlib.sha256(self.key.encode()).hexdigest()


def capture_region(module, operation, lexical, *, syntax_budget=65536):
    """Close a real single-block structured op over every available free SSA use.

    Explicit operands precede implicit captures, preserving first-use order and
    deduplicating by SSA identity. Each argument is an actual lexical-frame slot.
    Nested local definitions use structural positions, never parser SSA names.
    Arbitrary CFG/symbol closure is deliberately outside this first candidate.
    """
    # This complete walk must not be replaced by admission of only selected ops:
    # it rejects internal callees/globals/function values inside nested regions.
    admission.admit_module(module)
    if not any(op is operation for op in module.walk()):
        raise ValueError("opaque operation must belong to complete admitted module")
    return _capture_region(operation, lexical, syntax_budget=syntax_budget)


def capture_regions(module, requests, *, syntax_budget=65536):
    """One complete admission walk for one immutable in-process projection.

    Requests contain real operation/lexical-frame pairs, never saved admission
    records. The caller must not mutate this parsed module during projection.
    No result from this host helper carries checker acceptance authority.
    """
    admission.admit_module(module)
    members = set(module.walk())
    output = []
    for operation, lexical in requests:
        if operation not in members:
            raise ModelRefusal("opaque operation must belong to complete admitted module")
        output.append(_capture_region(operation, lexical, syntax_budget=syntax_budget))
    return tuple(output)


def _capture_region(operation, lexical, *, syntax_budget):
    if type(syntax_budget) is not int or syntax_budget <= 0:
        raise ValueError("positive exact syntax resource budget required")
    if not operation.regions:
        raise ModelRefusal("structured operation required")
    available = tuple(lexical)
    if len(set(available)) != len(available):
        raise ModelRefusal("unique actual lexical SSA slots required")
    available_slots = {value: i for i, value in enumerate(available)}
    captures, capture_slots = [], {}
    work = 0

    def use(value, local):
        if value in local:
            return local[value]
        if value not in available_slots:
            raise ModelRefusal("implicit SSA capture lacks lexical availability/dominance")
        if value not in capture_slots:
            capture_slots[value] = len(captures)
            captures.append(value)
        return ["capture", capture_slots[value]]

    def encode(op, path, inherited):
        nonlocal work
        work += 1 + len(op.operands) + len(op.results)
        if work > syntax_budget:
            raise ModelRefusal("opaque closure syntax resource budget exceeded")
        if op.successors:
            raise ModelRefusal("opaque CFG successor semantics not yet admitted")
        fields = metadata(op)
        fields["location"] = exact(op.location)
        fields["operands"] = [use(value, inherited) for value in op.operands]
        fields["regions"] = []
        for ri, region in enumerate(op.regions):
            if len(region.blocks) != 1:
                raise ModelRefusal("opaque single-block region geometry required")
            blocks = []
            for bi, block in enumerate(region.blocks):
                local = dict(inherited)
                work += len(block.args)
                if work > syntax_budget:
                    raise ModelRefusal("opaque closure syntax resource budget exceeded")
                for ai, value in enumerate(block.args):
                    local[value] = ["block-argument", [*path, ri, bi], ai]
                ops = []
                for oi, nested in enumerate(block.ops):
                    nested_path = [*path, ri, bi, oi]
                    ops.append(encode(nested, nested_path, local))
                    # Results are unavailable in their defining operation's own
                    # regions, and become visible only to following operations.
                    for vi, value in enumerate(nested.results):
                        local[value] = ["result", nested_path, vi]
                blocks.append({"arguments": [exact(v.type) for v in block.args], "operations": ops})
            fields["regions"].append({"blocks": blocks})
        return fields

    # Capture the explicit operand ABI first, including operands unused by the
    # region. Duplicate uses retain their exact operand-position references.
    for value in operation.operands:
        use(value, {})
    body = encode(operation, [], {})
    descriptor = {
        "schema": "opaque-structured-closure-source-candidate-v1",
        "input_types": [exact(value.type) for value in captures],
        "output_types": [exact(value.type) for value in operation.results],
        "operation": body,
    }
    return RegionClosure(descriptor, tuple(captures), tuple(available_slots[value] for value in captures))
