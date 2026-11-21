"""Choose a bounded, serial cover from original admitted implementation proposals.

This policy deliberately makes no performance claim. Unknown costs remain
unknown, CPU/vector resource sharing never implies concurrency, and the first
scope admits only unchanged logical values in one synchronous memory domain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import prod

from merlin.compiler.backend.source_proposals import AdmittedProposal
from merlin.compiler.partition.global_plan import CycleInterval, GlobalPlan, verify_global_plan
from merlin.compiler.partition.semantic_regions import outline_semantic_regions, semantic_dispatch
from merlin.compiler.semantic.target_binding import NativeCompilationError

# ---- execution ---------------------------------------------------------------------------------


def _record(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _size(buffer, alignment):
    widths = {"i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}
    if buffer.dtype not in widths or any(type(d) is not int or d <= 0 for d in buffer.shape):
        raise ValueError(f"execution memory bound has unsupported static storage type: {buffer.dtype}")
    size = prod(buffer.shape) * widths[buffer.dtype]
    return ((size + alignment - 1) // alignment) * alignment


def _source_peak(program, alignment, selected):
    """Retain complete region lifetimes and original arguments/constants.

    A region call can need its output storage before it releases any input.
    Round every original lifetime outwards to its selected region boundary;
    per-operation peaks alone would undercount that overlap. Internal source
    values remain conservatively counted even if a backend later removes them.
    """
    boundaries = {
        node: (item.alternative.nodes[0], item.alternative.nodes[-1])
        for item in selected
        for node in item.alternative.nodes
    }
    producers = {name: index for index, node in enumerate(program.nodes) for name in node.outputs}
    uses = {name: [] for name in program.buffers}
    for index, node in enumerate(program.nodes):
        for name in node.inputs:
            uses[name].append(index)
    for name in program.results:
        uses[name].append(len(program.nodes))
    intervals = []
    for name, buffer in program.buffers.items():
        start = producers.get(name, -1)
        end = max([start, *uses[name]])
        if buffer.kind in {"arg", "const"}:
            end = len(program.nodes)
        start = boundaries.get(start, (start, start))[0]
        end = boundaries.get(end, (end, end))[1]
        intervals.append((start, end, _size(buffer, alignment)))
    return max(
        (
            sum(size for start, end, size in intervals if start <= step <= end)
            for step in range(-1, len(program.nodes) + 1)
        ),
        default=0,
    )


@dataclass(frozen=True)
class SelectedExecution:
    source: object
    topology: object
    candidates: tuple[AdmittedProposal, ...]
    selected: tuple[AdmittedProposal, ...]
    plan: GlobalPlan
    memory_domain: str
    storage_bound_bytes: int
    rejected: tuple[tuple[str, str], ...]
    policy: tuple[str, ...]
    _plan_record: str
    _decision_record: str
    _owners: tuple

    def revalidate(self):
        owners = (self.source, self.topology, self.candidates, self.selected, self.plan)
        if len(owners) != len(self._owners) or any(a is not b for a, b in zip(owners, self._owners, strict=True)):
            raise ValueError("selected execution owner changed; replan")
        if _record((self.policy, self.rejected)) != self._decision_record:
            raise ValueError("selected execution decision report changed; replan")
        self.source.revalidate()
        self.topology.revalidate()
        for candidate in self.candidates:
            candidate.revalidate()
        if self.plan.selected != tuple(item.alternative for item in self.selected):
            raise ValueError("execution plan differs from its original selected proposals")
        if _record(self.plan.to_dict()) != self._plan_record:
            raise ValueError("execution plan changed; replan")
        domain, bound = _compatible(self.source, self.topology, self.selected)
        if domain != self.memory_domain or bound != self.storage_bound_bytes:
            raise ValueError("execution memory binding changed; replan")

    def record(self):
        self.revalidate()
        return {
            "schema": "merlin.selected_execution.v1",
            "source_sha256": self.source.stamp.source_sha256,
            "topology_sha256": self.topology.sha256,
            "plan": self.plan.to_dict(),
            "policy": {"kind": "bounded-first-compatible", "provider_order": list(self.policy)},
            "memory_domain": self.memory_domain,
            "storage_bound_bytes": self.storage_bound_bytes,
            "storage_scope": (
                "source liveness widened to region boundaries without alias reuse plus maximum serial scratch"
            ),
            "execution_order": [
                {
                    "slot": index,
                    "alternative": item.alternative.id,
                    "provider": item.proposal.provider,
                    "resource": item.proposal.resource,
                    "shared_group": self.topology.resource(item.proposal.resource).shared_group,
                    "source_ordinals": list(item.proposal.operation_ordinals),
                }
                for index, item in enumerate(self.selected)
            ],
            "source_observations": [item.source_observation() for item in self.selected],
            **self._context_record(),
            "rejected": [{"alternative": identity, "reason": reason} for identity, reason in self.rejected],
            "qualification": "not_established",
        }

    def _context_record(self):
        from merlin.compiler.backend.c_abi import context_reservations

        reserved = context_reservations(self.selected, self.topology)
        return {"host_context_reservations": reserved} if reserved["contexts"] else {}


def _compatible(source, topology, selected):
    alternatives = tuple(item.alternative for item in selected)
    plan = GlobalPlan(alternatives, (), CycleInterval.unknown("no comparable cost model selected"))
    problems = verify_global_plan(source.program, plan)
    if problems:
        raise ValueError("execution cover: " + "; ".join(problems))
    domains = {item.proposal.memory_domain for item in selected}
    if len(domains) != 1:
        raise ValueError("execution requires one memory domain; transfers are not materialized by this route")
    domain = next(iter(domains))
    memory = topology.memory(domain)
    bound = _source_peak(source.program, memory.alignment_bytes, selected)
    scratch = max((item.alternative.scratch_bytes for item in selected), default=0)
    scratch = ((scratch + memory.alignment_bytes - 1) // memory.alignment_bytes) * memory.alignment_bytes
    bound += scratch
    from merlin.compiler.backend.c_abi import context_reservations

    reserved = context_reservations(selected, topology)
    bound += reserved["domains"].get(domain, {}).get("total_bytes", 0)
    if bound > memory.capacity_bytes:
        raise ValueError(f"execution storage bound {bound} exceeds selected available capacity {memory.capacity_bytes}")
    return domain, bound


def select_execution(source, candidates, *, preferred_providers=(), max_states=4096):
    """Select a complete admitted cover in explicit provider order, then proposal ID.

    Search is bounded and deterministic. It may backtrack around incompatible
    groupings or memory budgets; it never substitutes a different precision or
    treats an unmeasured implementation as having zero cost.
    """
    if type(candidates) is not tuple or not candidates or len(candidates) > 4096:
        raise ValueError("execution selection requires a nonempty bounded immutable candidate roster")
    if type(max_states) is not int or not 0 < max_states <= 100_000:
        raise ValueError("execution search requires a bounded positive state budget")
    if (
        type(preferred_providers) is not tuple
        or any(type(value) is not str or not value for value in preferred_providers)
        or len(set(preferred_providers)) != len(preferred_providers)
    ):
        raise ValueError("provider preference must be an ordered tuple of distinct names")
    source.revalidate()
    if type(candidates[0]) is not AdmittedProposal:
        raise ValueError("execution candidates must be original admitted proposals")
    topology = candidates[0].topology
    ids = set()
    for item in candidates:
        if type(item) is not AdmittedProposal or item.source is not source or item.topology is not topology:
            raise ValueError("all proposals must retain the same original source and selected topology")
        item.revalidate()
        alternative = item.alternative
        if alternative.id in ids:
            raise ValueError("execution candidate IDs must be unique")
        ids.add(alternative.id)
        nodes = alternative.nodes
        if nodes != tuple(range(nodes[0], nodes[-1] + 1)):
            raise ValueError("this execution route requires contiguous source regions")
    providers = {item.proposal.provider for item in candidates}
    if set(preferred_providers) - providers:
        raise ValueError("provider preference names an absent candidate provider")
    rank = {name: index for index, name in enumerate(preferred_providers)}
    ordered = sorted(
        candidates,
        key=lambda item: (
            rank.get(item.proposal.provider, len(rank)),
            -len(item.alternative.nodes),
            item.alternative.id,
        ),
    )
    by_start = {}
    for item in ordered:
        by_start.setdefault(item.alternative.nodes[0], []).append(item)
    states = 0
    failures = {}
    # Explicit stack avoids Python recursion limits for large flat programs.
    stack = [(0, ())]
    found = None
    while stack:
        states += 1
        if states > max_states:
            raise ValueError("execution search exhausted its declared state budget")
        position, chosen = stack.pop()
        if position == len(source.operations):
            try:
                domain, bound = _compatible(source, topology, chosen)
            except ValueError as error:
                for item in chosen:
                    failures[item.alternative.id] = str(error)
                continue
            found = (chosen, domain, bound)
            break
        for item in reversed(by_start.get(position, ())):
            if chosen and item.proposal.memory_domain != chosen[0].proposal.memory_domain:
                failures[item.alternative.id] = "requires an unmaterialized memory-domain transition"
                continue
            stack.append((item.alternative.nodes[-1] + 1, (*chosen, item)))
    if found is None:
        detail = "; ".join(sorted(set(failures.values()))) or "no complete source cover"
        raise ValueError("no compatible execution plan: " + detail)
    selected, domain, bound = found
    plan = GlobalPlan(
        tuple(item.alternative for item in selected),
        (),
        CycleInterval.unknown("declaration policy; no performance comparison"),
        notes=(
            "provider preference, then larger admitted regions, then stable identity; no performance comparison",
            "serial issue and synchronous completion; no concurrency inferred",
        ),
    )
    selected_ids = {item.alternative.id for item in selected}
    rejected = tuple(
        (item.alternative.id, failures.get(item.alternative.id, "not selected by declared cover policy"))
        for item in ordered
        if item.alternative.id not in selected_ids
    )
    result = SelectedExecution(
        source,
        topology,
        candidates,
        selected,
        plan,
        domain,
        bound,
        rejected,
        preferred_providers,
        _record(plan.to_dict()),
        _record((preferred_providers, rejected)),
        (source, topology, candidates, selected, plan),
    )
    result.revalidate()
    return result


# ---- application -------------------------------------------------------------------------------
# Select a bounded first-valid partition from target-admitted tensor regions.
#
# This existing model-partition mechanism checks compatible source coverage. It
# is not heterogeneous system topology, coherence or execution admission.


def select_application_plan(source, alternatives, *, max_states=10_000):
    """Choose fusion and placement without inventing numerical or cost facts.

    Providers enumerate legal regions, representations and placements. Merlin
    chooses a complete compatible cover, including checking noncontiguous
    fusions for cycles. Larger regions are tried first, with stable tie breaking.
    This is bounded first-valid selection, not a performance optimization.
    """
    if type(max_states) is not int or max_states <= 0:
        raise ValueError("application planning needs a positive work budget")
    program = semantic_dispatch(source, entry="application")
    choices = tuple(sorted(alternatives, key=lambda row: (-len(row.nodes), row.nodes, row.id)))
    if len({row.id for row in choices}) != len(choices):
        raise ValueError("application region alternatives need distinct identities")
    domain = frozenset(range(len(program.nodes)))
    if any(not row.nodes or not set(row.nodes) <= domain for row in choices):
        raise ValueError("application alternative references an invalid source node")
    stack = [((), domain)]
    visited = 0
    while stack:
        if visited >= max_states:
            raise NativeCompilationError("resource_limit", "application partition search work limit reached")
        selected, remaining = stack.pop()
        visited += 1
        if not remaining:
            plan = GlobalPlan(
                tuple(sorted(selected, key=lambda row: row.nodes)),
                (),
                CycleInterval.unknown("first-valid application partition; costs are not ranked"),
            )
            if verify_global_plan(program, plan):
                continue
            try:
                outline_semantic_regions(source, entry="application", plan=plan)
            except ValueError:
                continue
            return plan
        anchor = min(remaining)
        eligible = [row for row in choices if anchor in row.nodes and set(row.nodes) <= remaining]
        for row in reversed(eligible):
            stack.append(((*selected, row), remaining - set(row.nodes)))
    raise NativeCompilationError("unsupported_semantics", "no compatible application partition covers every node")


def application_plan(source, adapter):
    """Use shared selection for region-enumerating providers.

    Legacy providers may still supply a complete plan. A failed enumeration
    never falls back to that route or silently changes placement.
    """
    enumerate_regions = getattr(adapter, "application_region_alternatives", None)
    if enumerate_regions is not None:
        return select_application_plan(source, enumerate_regions(source))
    return adapter.plan(source)
