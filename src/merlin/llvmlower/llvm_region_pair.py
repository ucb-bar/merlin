"""Explicit two-graph syntax accounting; no semantic equivalence is inferred."""

import hashlib
import json


def proof_pair_budget(
    self,
    candidate,
    original_indices,
    candidate_indices,
    *,
    unchanged_lines,
    changed_original,
    changed_candidate,
    max_bytes=65536,
    max_components=4096,
):
    """Account an explicit compressed syntax witness for two actual graphs.

    Graph-local SSA/control environments remain distinct even for shared
    bytes. Consumers must independently check both dependency closures,
    dominance, operation semantics and outside replacement legality.
    """
    from .llvm_region_boundary import LLVMRegionBoundary

    if not isinstance(candidate, LLVMRegionBoundary) or not isinstance(self, LLVMRegionBoundary):
        raise ValueError("two original LLVM boundary indices required")
    if (
        type(max_bytes) is not int
        or not 0 < max_bytes <= 65536
        or type(max_components) is not int
        or not 0 < max_components <= 4096
    ):
        raise ValueError("explicit bounded LLVM pair proof limits required")
    if candidate.scan_limits != self.scan_limits:
        raise ValueError("LLVM pair scan limits differ")
    if (
        self.arguments != candidate.arguments
        or self.argument_types != candidate.argument_types
        or self.words[self.first][:3] != candidate.words[candidate.first][:3]
    ):
        raise ValueError("LLVM pair selected function namespace differs")
    for graph, indices, changed in (
        (self, original_indices, changed_original),
        (candidate, candidate_indices, changed_candidate),
    ):
        for selected in (indices, changed):
            if type(selected) not in (set, frozenset) or any(
                type(i) is not int or not 0 <= i < len(graph.lines) for i in selected
            ):
                raise ValueError("bounded actual LLVM pair line sets required")
    if type(unchanged_lines) is not tuple or len(unchanged_lines) > self.scan_limits["max_scan_lines"]:
        raise ValueError("bounded explicit LLVM outside correspondence required")
    previous = (-1, -1)
    correspondence = {}
    for pair in unchanged_lines:
        if (
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(i) is not int for i in pair)
            or not previous[0] < pair[0] < len(self.lines)
            or not previous[1] < pair[1] < len(candidate.lines)
        ):
            raise ValueError("LLVM outside correspondence is not monotone injective")
        i, j = pair
        if self.lines[i] != candidate.lines[j] or self.line_blocks.get(i) != candidate.line_blocks.get(j):
            raise ValueError("LLVM outside bytes or block namespace differ")
        correspondence[i] = j
        previous = pair
    if (
        set(correspondence) != set(range(len(self.lines))) - changed_original
        or set(correspondence.values()) != set(range(len(candidate.lines))) - changed_candidate
    ):
        raise ValueError("LLVM outside correspondence does not cover both sources")

    # Refuse mutable/stale caller indices rather than account invented CFGs.
    reparsed = []
    for graph in (self, candidate):
        actual = LLVMRegionBoundary(
            graph.text,
            function_name=graph.words[graph.first][2][1:],
            argument_types=graph.argument_types,
            **graph.scan_limits,
        )
        for name in (
            "lines",
            "tokens",
            "words",
            "blocks",
            "edges",
            "definition_lines",
            "line_blocks",
            "arguments",
            "argument_types",
            "first",
            "last",
            "entry",
        ):
            if getattr(graph, name) != getattr(actual, name):
                raise ValueError("LLVM pair index changed from actual source")
        reparsed.append(actual)
    self, candidate = reparsed

    def ranges(values):
        result = []
        for value in sorted(values):
            if result and result[-1][0] + result[-1][1] == value:
                result[-1][1] += 1
            else:
                result.append([value, 1])
        if {v for start, length in result for v in range(start, start + length)} != set(values):
            raise ValueError("LLVM pair membership range reconstruction differs")
        return result

    outside = []
    for i, j in unchanged_lines:
        if outside and outside[-1][0] + outside[-1][2] == i and outside[-1][1] + outside[-1][2] == j:
            outside[-1][2] += 1
        else:
            outside.append([i, j, 1])
    if {(i + k, j + k) for i, j, n in outside for k in range(n)} != set(unchanged_lines):
        raise ValueError("LLVM outside range reconstruction differs")
    selected = (original_indices | changed_original, candidate_indices | changed_candidate)
    base_names = sorted(self.edges)
    base_positions = {name: i for i, name in enumerate(base_names)}
    reuse_cfg, delta_cfg = [], {}

    def frontier(graph, block):
        return [
            i
            for i in graph.blocks[block]
            if graph.words[i] and (graph.words[i][-1] == ":" or graph.words[i][0] in {"br", "ret"})
        ]

    for block, edges in candidate.edges.items():
        old_frontier = frontier(self, block) if block in self.blocks else []
        new_frontier = frontier(candidate, block)
        if (
            block in self.edges
            and self.edges[block] == edges
            and len(old_frontier) == len(new_frontier)
            and all(correspondence.get(i) == j for i, j in zip(old_frontier, new_frontier, strict=True))
        ):
            reuse_cfg.append(base_positions[block])
        else:
            delta_cfg[block] = edges
    reconstructed_cfg = {base_names[i]: self.edges[base_names[i]] for i in reuse_cfg}
    reconstructed_cfg.update(delta_cfg)
    if reconstructed_cfg != candidate.edges:
        raise ValueError("LLVM candidate CFG reconstruction differs")

    def facts(graph, indices):
        return [
            (i, graph.code(i))
            for i in sorted(indices)
            if graph.words[i] and graph.words[i][-1] != ":" and graph.words[i][0] not in {"br", "ret"}
        ]

    base_facts = facts(self, selected[0])
    original_fact_positions = {i: n for n, (i, _) in enumerate(base_facts)}
    reverse = {j: i for i, j in unchanged_lines}
    reuse_facts, delta_facts = [], []
    candidate_facts = facts(candidate, selected[1])
    for j, code in candidate_facts:
        i = reverse.get(j)
        if i in original_fact_positions:
            reuse_facts.append(original_fact_positions[i])
        else:
            delta_facts.append((j, code))
    reconstructed_facts = [(correspondence[base_facts[n][0]], base_facts[n][1]) for n in reuse_facts]
    if sorted(reconstructed_facts + delta_facts) != candidate_facts:
        raise ValueError("LLVM candidate proof facts reconstruction differs")
    witness = dict(
        schema="merlin.llvm_region_pair_syntax.v1",
        graph_sources=[hashlib.sha256(g.text.encode()).hexdigest() for g in (self, candidate)],
        function=[self.words[self.first][2], self.arguments, self.argument_types],
        graph_local_semantics="distinct; shared records are syntax only",
        outside_line_ranges=outside,
        changed_line_ranges=[ranges(changed_original), ranges(changed_candidate)],
        selected_line_ranges=[ranges(x) for x in selected],
        original_cfg={name: self.edges[name] for name in base_names},
        candidate_cfg_references=ranges(reuse_cfg),
        candidate_cfg_delta=delta_cfg,
        original_facts=base_facts,
        candidate_fact_references=ranges(reuse_facts),
        candidate_fact_delta=delta_facts,
    )
    encoded = json.dumps(witness, sort_keys=True, separators=(",", ":")).encode()
    provenance = (
        len(outside)
        + sum(map(len, witness["changed_line_ranges"]))
        + sum(map(len, witness["selected_line_ranges"]))
        + len(witness["candidate_cfg_references"])
        + len(witness["candidate_fact_references"])
        + 4
    )
    count = (
        len(self.blocks)
        + sum(map(len, self.edges.values()))
        + len(delta_cfg)
        + sum(map(len, delta_cfg.values()))
        + len(base_facts)
        + len(delta_facts)
        + provenance
    )
    if len(encoded) > max_bytes or count > max_components:
        raise ValueError("aggregate LLVM pair syntax representation bound exceeded")
    return dict(
        bytes=len(encoded),
        components=count,
        representation=witness,
        physical_records=dict(
            original_cfg_blocks=len(self.blocks),
            original_cfg_edges=sum(map(len, self.edges.values())),
            candidate_delta_blocks=len(delta_cfg),
            candidate_delta_edges=sum(map(len, delta_cfg.values())),
            original_facts=len(base_facts),
            candidate_delta_facts=len(delta_facts),
            provenance=provenance,
        ),
        original=dict(blocks=len(self.blocks), edges=sum(map(len, self.edges.values())), facts=len(base_facts)),
        candidate=dict(
            blocks=len(reconstructed_cfg), edges=sum(map(len, reconstructed_cfg.values())), facts=len(candidate_facts)
        ),
    )
