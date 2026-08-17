"""The generation pipeline as a dependency graph, so structural reuse becomes a measured quantity.

Today the pipeline is a straight line that overwrites every file, and its layer selector is manual rather
than staleness-driven. Nothing models a parent target, a content hash that gates regeneration, or an
invalidation set — so "adding this unit reused most of the existing target" is an assertion, and the
usual way to support it is a line count, which measures the wrong thing.

An edge here means **changing the source invalidates the sink**. That single convention is what makes the
interesting claim falsifiable: "the delta invalidates nothing in the parent's schedule" is not a promise
about intent, it is the claim that *no path exists* from the changed node to that one. A test can look
for the path. If someone later adds an edge that creates one, the claim fails in CI instead of in a
paragraph.

Three properties this refuses to fudge:

* **The changed set is measured, not declared.** :func:`changed_from_hashes` diffs recorded content
  hashes against current ones. A node whose sources cannot be read becomes ``UNKNOWN`` and is treated as
  changed — never as unchanged, because an unreadable source is exactly the case where assuming reuse
  would overstate the result.
* **The denominator is explicit.** ``reuse_ratio = reused / relevant`` is meaningless until someone says
  what was relevant, and a ratio whose denominator is chosen quietly can be made to say anything. So
  :meth:`ReuseMeasurement.of` takes the relevant set as an argument and records it in the result.
* **A cycle is an error, not a fixed point.** In a cyclic graph the invalidation closure grows to
  everything reachable and the reuse ratio silently collapses toward zero, which looks like a
  conservative answer rather than a broken graph.

The node set below describes *merlin's own* generation pipeline (manifest → evidence → plans → emitted
layers, plus the compiler-side nodes the design's DAG names). It contains no target facts: nodes are
roles, and the unit-specific ones are named for their role in the delta rather than for any target.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ArtifactGraph", "Node", "ReuseMeasurement", "TargetDelta", "UNKNOWN_HASH",
           "changed_from_hashes", "content_hashes", "pipeline_graph"]

#: Recorded for a node whose sources could not be read. Never compares equal to itself as "unchanged":
#: :func:`changed_from_hashes` treats it as changed on both sides.
UNKNOWN_HASH = "UNKNOWN"


@dataclass(frozen=True)
class Node:
    """One artifact in the generation graph.

    ``inputs`` names the nodes this one is derived from, so an edge is declared once, at the sink, next
    to the thing it constrains. ``sources`` are repo-relative paths whose content *is* this node — they
    are what gets hashed, which is what makes a change detectable rather than announced.
    """

    name: str
    kind: str
    inputs: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    why: str = ""


@dataclass(frozen=True)
class ArtifactGraph:
    """A validated dependency graph. Construct via :meth:`of` so the checks cannot be skipped."""

    nodes: dict[str, Node] = field(default_factory=dict)

    @classmethod
    def of(cls, nodes: Iterable[Node]) -> "ArtifactGraph":
        table: dict[str, Node] = {}
        for node in nodes:
            if node.name in table:
                raise ValueError(f"duplicate node {node.name!r}; a redefinition would silently replace "
                                 "one set of edges with another")
            table[node.name] = node
        graph = cls(nodes=table)
        problems = graph.problems()
        if problems:
            raise ValueError(f"invalid artifact graph: {problems}")
        return graph

    def problems(self) -> tuple[str, ...]:
        """Dangling inputs and cycles.

        Both are failures rather than warnings. A dangling input means an edge that does not constrain
        anything, so an invalidation would not propagate; a cycle makes every closure grow to the whole
        reachable set, which reads as a conservative answer instead of a broken graph.
        """
        found: list[str] = []
        for node in self.nodes.values():
            for dep in node.inputs:
                if dep not in self.nodes:
                    found.append(f"{node.name!r} declares unknown input {dep!r}")
        cycle = self._find_cycle()
        if cycle:
            found.append("cycle: " + " -> ".join(cycle))
        return tuple(found)

    def _find_cycle(self) -> tuple[str, ...]:
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {n: WHITE for n in self.nodes}

        def walk(name: str, trail: list[str]) -> tuple[str, ...]:
            colour[name] = GREY
            for dep in self.nodes[name].inputs:
                if dep not in self.nodes:
                    continue
                if colour[dep] == GREY:
                    return tuple(trail + [name, dep])
                if colour[dep] == WHITE:
                    got = walk(dep, trail + [name])
                    if got:
                        return got
            colour[name] = BLACK
            return ()

        for name in sorted(self.nodes):
            if colour[name] == WHITE:
                got = walk(name, [])
                if got:
                    return got
        return ()

    # -- structure ----------------------------------------------------------------------------

    def consumers(self) -> dict[str, tuple[str, ...]]:
        """``{node: nodes that declare it as an input}`` — the edge direction invalidation travels."""
        out: dict[str, list[str]] = {n: [] for n in self.nodes}
        for node in self.nodes.values():
            for dep in node.inputs:
                if dep in out:
                    out[dep].append(node.name)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    def downstream(self, names: Iterable[str]) -> frozenset[str]:
        """Every node reachable by following edges forward, EXCLUDING the given ones."""
        consumers = self.consumers()
        seen: set[str] = set()
        stack = [n for n in names if n in self.nodes]
        while stack:
            for nxt in consumers.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return frozenset(seen - set(names))

    def upstream(self, names: Iterable[str]) -> frozenset[str]:
        """Every node the given ones are derived from, transitively."""
        seen: set[str] = set()
        stack = [n for n in names if n in self.nodes]
        while stack:
            for dep in self.nodes[stack.pop()].inputs:
                if dep in self.nodes and dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        return frozenset(seen - set(names))

    def reaches(self, source: str, sink: str) -> bool:
        """Whether invalidating ``source`` would reach ``sink``. The falsifiable form of a reuse claim."""
        return sink in self.downstream([source])

    def topo_order(self) -> tuple[str, ...]:
        """Inputs before consumers. Ties broken by name so the order is reproducible."""
        indegree = {n: len({d for d in node.inputs if d in self.nodes})
                    for n, node in self.nodes.items()}
        consumers = self.consumers()
        ready = sorted(n for n, d in indegree.items() if d == 0)
        out: list[str] = []
        while ready:
            name = ready.pop(0)
            out.append(name)
            for nxt in consumers[name]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)
                    ready.sort()
        if len(out) != len(self.nodes):
            raise ValueError("topological order is undefined: the graph has a cycle")
        return tuple(out)

    def of_kind(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(n for n, node in self.nodes.items() if node.kind == kind))


# ---------------------------------------------------------------------------------------------
# Content hashing — what makes a change measured rather than asserted
# ---------------------------------------------------------------------------------------------


def _hash_path(path: Path) -> str | None:
    """sha256 over a file, or over a directory's files in sorted order. None when unreadable."""
    try:
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
        if path.is_dir():
            h = hashlib.sha256()
            for child in sorted(p for p in path.rglob("*") if p.is_file()):
                h.update(str(child.relative_to(path)).encode("utf-8"))
                h.update(child.read_bytes())
            return h.hexdigest()
    except OSError:
        return None
    return None


def content_hashes(graph: ArtifactGraph, root: "str | Path") -> dict[str, str]:
    """``{node: hash}`` over each node's declared sources.

    A node with no sources, or whose sources are missing, hashes to :data:`UNKNOWN_HASH` rather than to
    the hash of nothing. Hashing an empty input would make every such node compare equal forever, which
    is indistinguishable from "nothing changed" and would inflate the reuse ratio for free.
    """
    base = Path(root)
    out: dict[str, str] = {}
    for name, node in graph.nodes.items():
        if not node.sources:
            out[name] = UNKNOWN_HASH
            continue
        parts: list[str] = []
        for rel in node.sources:
            got = _hash_path(base / rel)
            if got is None:
                parts = []
                break
            parts.append(f"{rel}:{got}")
        out[name] = (hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
                     if parts else UNKNOWN_HASH)
    return out


def changed_from_hashes(recorded: Mapping[str, str], current: Mapping[str, str]) -> frozenset[str]:
    """Nodes whose content differs, plus every node that cannot be shown to be unchanged.

    Fails closed in three ways, each of which would otherwise be reported as reuse: a node that is new,
    a node that has disappeared, and a node whose hash is ``UNKNOWN`` on either side.
    """
    changed: set[str] = set()
    for name in set(recorded) | set(current):
        was, now = recorded.get(name), current.get(name)
        if was is None or now is None:
            changed.add(name)
        elif was == UNKNOWN_HASH or now == UNKNOWN_HASH or was != now:
            changed.add(name)
    return frozenset(changed)


# ---------------------------------------------------------------------------------------------
# The delta
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetDelta:
    """A change to a target, expressed as the set of graph nodes whose sources moved.

    ``label`` is for reports only. The arithmetic uses ``changed``, which is meant to come from
    :func:`changed_from_hashes` rather than from a hand-written list — a hand-written changed set is how a
    reuse number becomes whatever its author wanted.
    """

    changed: frozenset[str]
    label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "changed", frozenset(self.changed))

    def invalidated(self, graph: ArtifactGraph) -> frozenset[str]:
        """The changed nodes plus everything downstream of them — what must be regenerated."""
        unknown = self.changed - set(graph.nodes)
        if unknown:
            raise ValueError(f"delta names nodes absent from the graph: {sorted(unknown)}; refusing to "
                             "compute an invalidation set that silently ignores them")
        return frozenset(self.changed) | graph.downstream(self.changed)

    def reused(self, graph: ArtifactGraph, relevant: Iterable[str]) -> frozenset[str]:
        """The relevant nodes this delta does NOT invalidate."""
        rel = frozenset(relevant)
        unknown = rel - set(graph.nodes)
        if unknown:
            raise ValueError(f"relevant set names nodes absent from the graph: {sorted(unknown)}")
        return rel - self.invalidated(graph)

    def measure(self, graph: ArtifactGraph, relevant: Iterable[str]) -> "ReuseMeasurement":
        return ReuseMeasurement.of(self, graph, relevant)


@dataclass(frozen=True)
class ReuseMeasurement:
    """A reuse ratio together with the sets that produced it, so the number can be argued with."""

    label: str
    changed: tuple[str, ...]
    invalidated: tuple[str, ...]
    reused: tuple[str, ...]
    relevant: tuple[str, ...]

    @classmethod
    def of(cls, delta: TargetDelta, graph: ArtifactGraph,
           relevant: Iterable[str]) -> "ReuseMeasurement":
        rel = frozenset(relevant)
        if not rel:
            raise ValueError("the relevant set is empty, so a reuse ratio would have no denominator; "
                             "name the nodes that were in play")
        unknown = rel - set(graph.nodes)
        if unknown:
            # A node the graph does not have cannot be invalidated, so it would land in `reused` and
            # enlarge the denominator at the same time -- inflating the ratio from both ends.
            raise ValueError(f"relevant set names nodes absent from the graph: {sorted(unknown)}")
        inv = delta.invalidated(graph)
        return cls(label=delta.label, changed=tuple(sorted(delta.changed)),
                   invalidated=tuple(sorted(inv)), reused=tuple(sorted(rel - inv)),
                   relevant=tuple(sorted(rel)))

    @property
    def reuse_ratio(self) -> float:
        return len(self.reused) / len(self.relevant)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "reuse_ratio": round(self.reuse_ratio, 4),
                "n_relevant": len(self.relevant), "n_reused": len(self.reused),
                "n_invalidated": len(self.invalidated),
                "changed": list(self.changed), "invalidated": list(self.invalidated),
                "reused": list(self.reused), "relevant": list(self.relevant)}


# ---------------------------------------------------------------------------------------------
# merlin's generation pipeline, as declared by the pipeline itself
# ---------------------------------------------------------------------------------------------

_PY = "merlin/python/merlin"

#: Each emitted layer: the plan it is generated from, and the generator that writes it. Keyed by the
#: layer names ``pipeline.EMIT_LAYERS`` uses, so a layer added to the pipeline without a node here is a
#: test failure rather than a silent hole in the graph.
_EMIT_LAYER_PLAN = {
    "xdsl": ("dialect_plan", "generate/xdsl.py"),
    "mlir": ("dialect_plan", "generate/mlir_scaffold.py"),
    "zephyr": ("zephyr_plan", "generate/zephyr_module.py"),
    "runtime": ("runtime_adapter_plan", "generate/runtime_adapter.py"),
    "llvm-plan": ("llvm_extension_plan", "generate/llvm_plan.py"),
}

#: The four synthesized plans, each with the module that synthesizes it.
_PLAN_NODES = {
    "dialect_plan": ("synthesize/dialect_plan.py",
                     "the compiler dialect the target's ops are expressed in"),
    "runtime_adapter_plan": ("synthesize/runtime_adapter_plan.py",
                             "how the generated runtime adapter talks to the target"),
    "zephyr_plan": ("synthesize/zephyr_plan.py", "the board/OS module for the target"),
    "llvm_extension_plan": ("synthesize/llvm_extension_plan.py",
                            "the backend extension description"),
}


def pipeline_graph() -> ArtifactGraph:
    """The graph for merlin's own target-generation pipeline plus the compiler-side nodes.

    Edges follow ``targetgen/pipeline.py``: the source manifest feeds hardware evidence, evidence and the
    capability contract feed all five plans, and each emitted layer comes from its plan. The
    compiler-side chain (routing, lowering, codegen, CCA, certification) hangs off the capability
    contract, which is the seam a datapath addition actually enters through.

    The parent's schedule, the generic lowering, board/runtime support and the elementwise path are
    deliberately ROOTS with no path from hardware evidence. That is the structural form of the
    default-OFF-feature invariant: with no feature enabled the emitted pipeline is byte-identical to
    baseline, so a hardware change cannot reach them. If a future edge creates such a path, the reuse
    claim in the design doc is what breaks, and a test says so.
    """
    nodes: list[Node] = [
        Node("source_manifest", "evidence", (),
             (f"{_PY}/targetgen/ingest",),
             "where the target's own sources were read from"),
        Node("hardware_evidence", "evidence", ("source_manifest",),
             (f"{_PY}/targetgen/evidence", f"{_PY}/targetgen/rtl"),
             "facts extracted from the target's RTL and headers"),
        Node("capability_contract", "contract", ("hardware_evidence",),
             (f"{_PY}/targetgen/synthesize/target_contract.py",
              f"{_PY}/targetgen/compute_units.py"),
             "what the target can do, as data the compiler reads"),
    ]
    for plan, (module, why) in _PLAN_NODES.items():
        nodes.append(Node(plan, "plan", ("hardware_evidence", "capability_contract"),
                          (f"{_PY}/targetgen/{module}",), why))
    for layer, (plan, module) in _EMIT_LAYER_PLAN.items():
        nodes.append(Node(f"emit_{layer.replace('-', '_')}", "emit", (plan,),
                          (f"{_PY}/targetgen/{module}",),
                          f"the {layer} artifacts written for the target"))
    nodes += [
        Node("capability_routing", "compiler", ("capability_contract",),
             (f"{_PY}/targetgen/routing.py",),
             "which unit each demand is routed to"),
        Node("unit_lowering", "compiler", ("capability_contract", "dialect_plan"),
             (f"{_PY}/llvmlower/perop_blocks.py",),
             "lowering a tagged contraction onto the added unit"),
        Node("unit_codegen", "compiler", ("unit_lowering",),
             (f"{_PY}/kernels/opu_kernel.py",),
             "the emitted microkernel for the added unit"),
        Node("unit_cca", "compiler", ("unit_codegen",),
             (f"{_PY}/kernels/cca.py", f"{_PY}/kernels/action_catalog.py"),
             "the compiler-capability actions the unit exposes"),
        Node("unit_certification", "test", ("unit_codegen",),
             (f"{_PY}/kernels/opu_cert.py", f"{_PY}/kernels/opu_corpus.py"),
             "the numerical acceptance surface for the unit"),
        # --- roots the delta must not reach ---------------------------------------------------
        Node("parent_schedule", "parent", (),
             (f"{_PY}/llvmlower/impr_features.py",),
             "the certified parent package's schedule, reused literally"),
        Node("generic_lowering", "parent", (),
             (f"{_PY}/llvmlower/passes_quant_int.py",),
             "the target-independent integer lowering"),
        Node("runtime_board_support", "parent", (),
             (f"{_PY}/runtime/boards.py", "merlin/runtime/baremetal"),
             "board bring-up and the bare-metal harness"),
        Node("elementwise_path", "parent", (),
             (f"{_PY}/runtime/tensor.py",),
             "the elementwise/epilogue path"),
    ]
    return ArtifactGraph.of(nodes)
