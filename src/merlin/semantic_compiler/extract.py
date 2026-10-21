"""Merlin-owned buffer-aware enumerative candidate extraction.

The bounded traversal is informed by ACT section 5.3, but is an independently
implemented multi-root extractor. It requests each e-class in a particular
physical storage and keeps alternatives after an initial allocation failure.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from graphlib import TopologicalSorter
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
    bundle_instance: str = ""
    bundle_result_index: int = 0
    bundle_result_count: int = 1

    def __post_init__(self) -> None:
        if type(self.children) is not tuple:
            raise ValueError("choice children must be an immutable tuple")
        if any(not isinstance(child, Choice) for child in self.children):
            raise ValueError("choice children must contain selected choices")
        if type(self.bundle_instance) is not str:
            raise ValueError("selected instruction needs a string bundle identity")
        if self.bundle_instance:
            if (
                type(self.bundle_result_index) is not int
                or type(self.bundle_result_count) is not int
                or self.bundle_result_count <= 1
                or not 0 <= self.bundle_result_index < self.bundle_result_count
            ):
                raise ValueError("selected multi-result instruction has malformed projection identity")
        elif self.bundle_result_index != 0 or self.bundle_result_count != 1:
            raise ValueError("single-result choice has a multi-result projection index")

    def key(self) -> tuple:
        return self._key

    @cached_property
    def _key(self) -> tuple:
        # Choices are immutable. Preserve the serialized identity while avoiding
        # repeated expansion of the same shared producer for every consumer.
        base = (self.eclass, self.symbol, self.storage, tuple(child.key() for child in self.children))
        return base if not self.bundle_instance else (
            *base, self.bundle_instance, self.bundle_result_index, self.bundle_result_count,
        )

    def physical_instance(self) -> tuple | None:
        if not self.bundle_instance:
            return None
        return (self.bundle_instance, tuple(child.key() for child in self.children))

    def instruction_count(self) -> int:
        return Candidate((self,)).instruction_count()


@dataclass(frozen=True)
class Candidate:
    outputs: tuple[Choice, ...]

    def digest(self) -> str:
        data = json.dumps([choice.key() for choice in self.outputs], separators=(",", ":"), sort_keys=True)
        return sha256(data.encode()).hexdigest()

    def instruction_count(self) -> int:
        """Charge one shared instruction once across all ordered roots."""
        seen: set[tuple] = set()
        instances: set[tuple] = set()

        def visit(choice: Choice) -> int:
            key = choice.key()
            if key in seen:
                return 0
            seen.add(key)
            instance = choice.physical_instance()
            count = int(choice.symbol.startswith("i_") and (instance is None or instance not in instances))
            if instance is not None:
                instances.add(instance)
            return count + sum(visit(child) for child in choice.children)

        return sum(visit(output) for output in self.outputs)

    def bundles_complete(self) -> bool:
        """A selected bundle retains every result of one physical issue."""
        groups: dict[tuple, tuple[int, dict[int, Choice]]] = {}
        visited: set[tuple] = set()

        def visit(choice: Choice) -> bool:
            if choice.key() in visited:
                return True
            visited.add(choice.key())
            instance = choice.physical_instance()
            if instance is not None:
                count, projections = groups.setdefault(instance, (choice.bundle_result_count, {}))
                if count != choice.bundle_result_count or not 0 <= choice.bundle_result_index < count:
                    return False
                previous = projections.setdefault(choice.bundle_result_index, choice)
                if previous.key() != choice.key():
                    return False
            return all(visit(child) for child in choice.children)

        return all(visit(output) for output in self.outputs) and all(
            set(projections) == set(range(count)) for count, projections in groups.values()
        )


@dataclass
class _ChoiceStream:
    values: list[Choice]
    remaining: Iterator[Choice] | None


_ChoiceRequest = tuple[int, str, str, int, frozenset[tuple[int, str, str]]]


def _cycle_components(exploration: Exploration, deadline: float | None) -> dict[int, frozenset[int]]:
    """Find mutually reachable classes with two iterative depth-first walks."""
    edges = {
        eclass: {child for node in nodes for child in node.children}
        for eclass, nodes in exploration.classes.items()
    }
    reverse: dict[int, set[int]] = {eclass: set() for eclass in edges}
    for eclass, children in tuple(edges.items()):
        _check_deadline(deadline)
        for child in children:
            edges.setdefault(child, set())
            reverse.setdefault(child, set()).add(eclass)
    visited: set[int] = set()
    finished: list[int] = []
    for root in edges:
        if root in visited:
            continue
        visited.add(root)
        stack = [(root, iter(edges[root]))]
        while stack:
            _check_deadline(deadline)
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                finished.append(node)
                stack.pop()
            elif child not in visited:
                visited.add(child)
                stack.append((child, iter(edges[child])))
    components: dict[int, frozenset[int]] = {}
    for root in reversed(finished):
        if root in components:
            continue
        group = {root}
        pending = [root]
        while pending:
            _check_deadline(deadline)
            for predecessor in reverse[pending.pop()]:
                if predecessor not in components and predecessor not in group:
                    group.add(predecessor)
                    pending.append(predecessor)
        frozen = frozenset(group)
        components.update((node, frozen) for node in group)
    return components


def _mandatory_instruction_bound(
    exploration: Exploration,
    program: RuleProgram,
    components: dict[int, frozenset[int]],
    deadline: float | None,
    output_storages: tuple[str, ...] = (),
) -> int:
    """Count instruction components required by every finite realization.

    A finite tree must leave each visited strongly connected component at an
    exit node. Intersect obligations across every exit, then union the roots.
    Count a component once even when several consumers share its producer.
    Ignore storage/layout restrictions, which can only increase the work.
    Multi-result instructions can cover distinct components in one issue, so
    this single-result bound is disabled whenever a bundle is available.
    """
    if any(item.get("bundle_instance") for item in program.symbols.values()):
        return 0
    labels: dict[frozenset[int], int] = {}
    owners = {node: labels.setdefault(group, len(labels)) for node, group in components.items()}
    groups = {label: group for group, label in labels.items()}
    dependencies: dict[int, set[int]] = {label: set() for label in groups}
    for node, alternatives in exploration.classes.items():
        _check_deadline(deadline)
        dependencies[owners[node]].update(
            owners[child] for alternative in alternatives for child in alternative.children
            if owners[child] != owners[node]
        )
    required: dict[int, set[int]] = {}
    for label in TopologicalSorter(dependencies).static_order():
        exits = []
        for node in groups[label]:
            for alternative in exploration.classes.get(node, ()):
                _check_deadline(deadline)
                kind = program.symbols.get(alternative.symbol, {}).get("kind")
                if kind in {"input", "constant"}:
                    exits.append(set())
                elif kind in {"instruction", "view"} and all(
                    owners[child] != label for child in alternative.children
                ):
                    obligations = set().union(*(required[owners[child]] for child in alternative.children))
                    if kind == "instruction" and alternative.symbol.startswith("i_"):
                        obligations.add(label)
                    exits.append(obligations)
        # No known exit need not imply source infeasibility. Keep a zero bound
        # for incomplete or unextractable components and let search diagnose it.
        required[label] = set.intersection(*exits) if exits else set()
    mandatory = set().union(*(required[owners[root]] for root in exploration.roots))
    # A root that can only be produced by a non-exit instruction also needs
    # that physical issue, in addition to the exit issues counted above.
    # Distinct (class, storage) results cannot share a single-result issue.
    boundary_issues = set()
    for root, storage in zip(exploration.roots, output_storages):
        alternatives = exploration.classes.get(root, ())
        if any(program.symbols.get(node.symbol, {}).get("kind") in {"input", "constant", "view"}
               for node in alternatives):
            continue
        matching = [
            node for node in alternatives
            if program.symbols.get(node.symbol, {}).get("kind") == "instruction"
            and program.symbols[node.symbol]["descriptor"].get("output_storage") == storage
        ]
        if matching and all(
            node.symbol.startswith("i_") and any(child in components[root] for child in node.children)
            for node in matching
        ):
            boundary_issues.add((root, storage))
    return len(mandatory) + len(boundary_issues)


@dataclass
class ExtractionSession:
    """Resume bounded extraction only for the same graph, rules and deadline."""

    exploration: Exploration
    request: KernelRequest
    program: RuleProgram
    deadline: float | None = None
    memo: dict[_ChoiceRequest, _ChoiceStream] = field(default_factory=dict, init=False)
    components: dict[int, frozenset[int]] = field(init=False)
    minimum_instructions: int = field(init=False)

    def __post_init__(self) -> None:
        self.components = _cycle_components(self.exploration, self.deadline)
        self.minimum_instructions = _mandatory_instruction_bound(
            self.exploration, self.program, self.components, self.deadline, self.request.output_storages,
        )


def _choices(
    exploration: Exploration,
    request: KernelRequest,
    program: RuleProgram,
    eclass: int,
    storage: str,
    required_layout: str,
    budget: int,
    ancestors: frozenset[tuple[int, str, str]],
    deadline: float | None,
    memo: dict[_ChoiceRequest, _ChoiceStream],
    components: dict[int, frozenset[int]],
) -> Iterator[Choice]:
    # Retain the traversal context: a choice legal on one path can be cyclic on
    # another. Replay only identical subproblems, lazily and in the same order.
    # The memo belongs to one search session; remaining budget is part of its key.
    _check_deadline(deadline)
    # An ancestor can affect a descendant only if they are mutually reachable:
    # the traversal already supplies ancestor -> descendant. Preserve all
    # storage/layout contexts within that component, including copy cycles.
    relevant = frozenset(ancestor for ancestor in ancestors if ancestor[0] in components[eclass])
    key = (eclass, storage, required_layout, budget, relevant)
    stream = memo.get(key)
    if stream is None:
        stream = _ChoiceStream([], _expand_choices(
            exploration, request, program, eclass, storage, required_layout,
            budget, ancestors, deadline, memo, components,
        ))
        memo[key] = stream
    index = 0
    while True:
        _check_deadline(deadline)
        if index == len(stream.values):
            if stream.remaining is None:
                return
            try:
                stream.values.append(next(stream.remaining))
            except StopIteration:
                stream.remaining = None
                return
        yield stream.values[index]
        index += 1


def _expand_choices(
    exploration: Exploration,
    request: KernelRequest,
    program: RuleProgram,
    eclass: int,
    storage: str,
    required_layout: str,
    budget: int,
    ancestors: frozenset[tuple[int, str, str]],
    deadline: float | None,
    memo: dict[_ChoiceRequest, _ChoiceStream],
    components: dict[int, frozenset[int]],
) -> Iterator[Choice]:
    _check_deadline(deadline)
    if budget < 0:
        return
    # A no-op semantic rewrite can put a copy instruction and its source in
    # one e-class. A repeat may finish at a boundary or a producer that leaves
    # the component. This permits a finite store/reload realization while
    # forbidding another traversal around the same cycle.
    recursive = (eclass, storage, required_layout) in ancestors
    next_ancestors = ancestors | {(eclass, storage, required_layout)}
    inputs = dict(request.input_storages)
    # Prefer computation over a storage roundtrip. Copy alternatives remain
    # available after a direct realization fails placement.
    alternatives = sorted(
        exploration.classes.get(eclass, ()),
        key=lambda node: program.symbols.get(node.symbol, {}).get("descriptor", {}).get("value_preserving_copy", False),
    )
    for node in alternatives:
        _check_deadline(deadline)
        metadata = program.symbols.get(node.symbol)
        if metadata is None:
            continue
        kind = metadata["kind"]
        if kind == "input":
            if inputs[metadata["source_node"]] == storage and (
                not required_layout or request.boundary_layout(metadata["source_node"]) == required_layout
            ):
                yield Choice(eclass, node.symbol, storage)
            continue
        if kind == "constant":
            # Constant payloads are compiler inputs and must be initialized by
            # the materializer; they are never an uninitialized scratchpad leaf.
            if storage == "external" and (
                not required_layout or request.boundary_layout(metadata["source_node"]) == required_layout
            ):
                yield Choice(eclass, node.symbol, storage)
            continue
        if kind == "view":
            descriptor = metadata["descriptor"]
            required_extents = (
                tuple(descriptor["source_extents"])
                if descriptor.get("kind") == "concat"
                else (descriptor.get("source_extent", descriptor["extent"]),)
            )
            if (
                recursive or len(node.children) != len(required_extents) or descriptor["storage"] != storage
                or required_layout and descriptor["physical_layout"] != required_layout
            ):
                continue

            def view_children(index: int, chosen: tuple[Choice, ...]) -> Iterator[tuple[Choice, ...]]:
                _check_deadline(deadline)
                if index == len(node.children):
                    yield chosen
                    return
                for child in _choices(
                    exploration, request, program, node.children[index], storage,
                    descriptor["physical_layout"], budget,
                    next_ancestors, deadline, memo, components,
                ):
                    child_metadata = program.symbols[child.symbol]
                    child_extent = (
                        child_metadata["descriptor"]["extent"]
                        if child_metadata["kind"] in {"instruction", "view"}
                        else child_metadata.get("extent")
                    )
                    if child_extent != required_extents[index]:
                        continue
                    if Candidate((*chosen, child)).instruction_count() > budget:
                        continue
                    yield from view_children(index + 1, (*chosen, child))

            for children in view_children(0, ()):
                yield Choice(eclass, node.symbol, storage, children)
            continue
        if kind != "instruction":
            continue
        if recursive and any(child in components[eclass] for child in node.children):
            continue
        if budget == 0:
            continue
        descriptor = metadata["descriptor"]
        if descriptor["output_storage"] != storage or (
            required_layout and descriptor.get("output_layout", "") != required_layout
        ):
            continue
        required = descriptor["input_storages"]
        # Earlier request schemas do not state a boundary representation.
        # Descriptor layouts become obligations only for layout-qualified v4
        # requests (or when a view explicitly requests a physical layout).
        input_layouts = descriptor.get("input_layouts", ()) if (
            request.boundary_layouts or required_layout
        ) else ()
        if request.boundary_layouts and len(input_layouts) != len(required):
            continue
        if len(required) != len(node.children):
            continue
        if not node.children:
            yield Choice(
                eclass, node.symbol, storage,
                bundle_instance=metadata.get("bundle_instance", ""),
                bundle_result_index=metadata.get("bundle_result_index", 0),
                bundle_result_count=metadata.get("bundle_result_count", 1),
            )
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
                input_layouts[index] if input_layouts else "",
                budget - 1,
                next_ancestors,
                deadline,
                memo,
                components,
            ):
                # The same selected producer can satisfy two input ports.
                # Charge the prospective DAG, not the unfolded child trees.
                if 1 + Candidate((*chosen, child)).instruction_count() > budget:
                    continue
                yield from combinations(index + 1, (*chosen, child))

        for children in combinations(0, ()):
            yield Choice(
                eclass, node.symbol, storage, children,
                metadata.get("bundle_instance", ""),
                metadata.get("bundle_result_index", 0),
                metadata.get("bundle_result_count", 1),
            )


def enumerate_candidates(
    exploration: Exploration,
    request: KernelRequest,
    program: RuleProgram,
    *,
    node_budget: int,
    max_candidates: int,
    deadline: float | None = None,
    session: ExtractionSession | None = None,
) -> Iterator[Candidate]:
    if node_budget <= 0 or max_candidates <= 0:
        raise ValueError("candidate work bounds must be positive")
    if len(exploration.roots) != len(request.outputs):
        raise ValueError("e-graph roots differ from source outputs")
    if session is None:
        session = ExtractionSession(exploration, request, program, deadline)
    elif (
        session.exploration is not exploration or session.request is not request
        or session.program is not program or session.deadline != deadline
    ):
        raise ValueError("extraction session belongs to a different graph, request, rules or deadline")
    if node_budget < session.minimum_instructions:
        return
    seen: set[str] = set()
    emitted = 0
    memo, components = session.memo, session.components

    def combine(index: int, chosen: tuple[Choice, ...]) -> Iterator[Candidate]:
        nonlocal emitted
        _check_deadline(deadline)
        if emitted >= max_candidates:
            return
        if index == len(exploration.roots):
            candidate = Candidate(chosen)
            digest = candidate.digest()
            if digest not in seen and candidate.bundles_complete():
                seen.add(digest)
                emitted += 1
                yield candidate
            return
        root = exploration.roots[index]
        storage = request.output_storages[index]
        layout = request.output_layouts[index] if request.output_layouts else ""
        for choice in _choices(
            exploration, request, program, root, storage, layout, node_budget, frozenset(), deadline, memo, components,
        ):
            if emitted >= max_candidates:
                return
            if Candidate((*chosen, choice)).instruction_count() <= node_budget:
                yield from combine(index + 1, (*chosen, choice))

    yield from combine(0, ())
