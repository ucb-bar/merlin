"""Merlin-owned buffer-aware enumerative candidate extraction.

The bounded traversal is informed by ACT section 5.3, but is an independently
implemented multi-root extractor. It requests each e-class in a particular
physical storage and keeps alternatives after an initial allocation failure.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic

from .egg_bridge import Exploration
from .model import KernelRequest
from .rules import RuleProgram


class ExtractionTimeout(RuntimeError):
    pass


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise ExtractionTimeout("native candidate extraction exceeded its deadline")


@dataclass(frozen=True)
class Choice:
    eclass: int
    symbol: str
    storage: str
    children: tuple[Choice, ...] = ()

    def key(self) -> tuple:
        return (self.eclass, self.symbol, self.storage, tuple(child.key() for child in self.children))

    def instruction_count(self) -> int:
        return (1 if self.symbol.startswith("i_") else 0) + sum(child.instruction_count() for child in self.children)


@dataclass(frozen=True)
class Candidate:
    outputs: tuple[Choice, ...]

    def digest(self) -> str:
        data = json.dumps([choice.key() for choice in self.outputs], separators=(",", ":"), sort_keys=True)
        return sha256(data.encode()).hexdigest()

    def instruction_count(self) -> int:
        """Charge one shared instruction once across all ordered roots."""
        seen: set[tuple] = set()

        def visit(choice: Choice) -> int:
            key = choice.key()
            if key in seen:
                return 0
            seen.add(key)
            return (1 if choice.symbol.startswith("i_") else 0) + sum(visit(child) for child in choice.children)

        return sum(visit(output) for output in self.outputs)


def _choices(
    exploration: Exploration,
    request: KernelRequest,
    program: RuleProgram,
    eclass: int,
    storage: str,
    budget: int,
    ancestors: frozenset[tuple[int, str]],
    deadline: float | None,
) -> Iterator[Choice]:
    _check_deadline(deadline)
    if budget < 0:
        return
    # A no-op semantic rewrite can put a copy instruction and its source in
    # one e-class. Revisited classes may still terminate at an input/constant
    # leaf; only further instruction expansion would make a recursive cycle.
    recursive = (eclass, storage) in ancestors
    next_ancestors = ancestors | {(eclass, storage)}
    inputs = dict(request.input_storages)
    for node in exploration.classes.get(eclass, ()):
        _check_deadline(deadline)
        metadata = program.symbols.get(node.symbol)
        if metadata is None:
            continue
        kind = metadata["kind"]
        if kind == "input":
            if inputs[metadata["source_node"]] == storage:
                yield Choice(eclass, node.symbol, storage)
            continue
        if kind == "constant":
            # Constant payloads are compiler inputs and must be initialized by
            # the materializer; they are never an uninitialized scratchpad leaf.
            if storage == "external":
                yield Choice(eclass, node.symbol, storage)
            continue
        if kind != "instruction":
            continue
        if recursive:
            continue
        if budget == 0:
            continue
        descriptor = metadata["descriptor"]
        if descriptor["output_storage"] != storage:
            continue
        required = descriptor["input_storages"]
        if len(required) != len(node.children):
            continue
        if not node.children:
            yield Choice(eclass, node.symbol, storage)
            continue

        def combinations(index: int, chosen: tuple[Choice, ...]) -> Iterator[tuple[Choice, ...]]:
            _check_deadline(deadline)
            if index == len(node.children):
                yield chosen
                return
            for child in _choices(
                exploration,
                request,
                program,
                node.children[index],
                required[index],
                budget - 1,
                next_ancestors,
                deadline,
            ):
                # The same selected producer can satisfy two input ports.
                # Charge the prospective DAG, not the unfolded child trees.
                if 1 + Candidate((*chosen, child)).instruction_count() > budget:
                    continue
                yield from combinations(index + 1, (*chosen, child))

        for children in combinations(0, ()):
            yield Choice(eclass, node.symbol, storage, children)


def enumerate_candidates(
    exploration: Exploration,
    request: KernelRequest,
    program: RuleProgram,
    *,
    node_budget: int,
    max_candidates: int,
    deadline: float | None = None,
) -> Iterator[Candidate]:
    if node_budget <= 0 or max_candidates <= 0:
        raise ValueError("candidate work bounds must be positive")
    if len(exploration.roots) != len(request.outputs):
        raise ValueError("e-graph roots differ from source outputs")
    seen: set[str] = set()
    emitted = 0

    def combine(index: int, chosen: tuple[Choice, ...]) -> Iterator[Candidate]:
        nonlocal emitted
        _check_deadline(deadline)
        if emitted >= max_candidates:
            return
        if index == len(exploration.roots):
            candidate = Candidate(chosen)
            digest = candidate.digest()
            if digest not in seen:
                seen.add(digest)
                emitted += 1
                yield candidate
            return
        root = exploration.roots[index]
        storage = request.output_storages[index]
        for choice in _choices(exploration, request, program, root, storage, node_budget, frozenset(), deadline):
            if emitted >= max_candidates:
                return
            if Candidate((*chosen, choice)).instruction_count() <= node_budget:
                yield from combine(index + 1, (*chosen, choice))

    yield from combine(0, ())
