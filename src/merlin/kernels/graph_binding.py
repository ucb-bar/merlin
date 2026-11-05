"""Provenance hypotheses joining named exact instruction bodies to captured calls.

Names select possible source associations, never compiler policy or numerical
semantics. Even a unique association leaves the expert's operand layouts,
quantizer, fusion and execution correspondence unproved.
"""

from __future__ import annotations

from collections import defaultdict

from merlin.common.jsonio import canonical_sha256


def _identifier(value):
    return "_".join("".join(char if char.isalnum() else " " for char in value).split()).lower()


def _module_path(region):
    stack = region["operation"].get("module_stack", {})
    paths = [row[0].split(".") for row in stack.values() if isinstance(row, list) and row and row[0]]
    return paths[-1] if paths else []


def _common_prefix(paths):
    if not paths:
        return 0
    count = 0
    for values in zip(*paths):
        if len(set(values)) != 1:
            break
        count += 1
    return max(0, count - 1) if all(path == paths[0] for path in paths) else count


def _frontier(members, by_id):
    selected = set(members)
    inputs, outputs = {}, {}
    for identity in members:
        region = by_id[identity]
        for row in region["inputs"]:
            if row["producer_node_id"] not in selected:
                inputs.setdefault(row["value"]["id"], row)
        for value in region["outputs"]:
            consumers = region["output_consumers"].get(value["id"], [])
            if any(row["node_id"] not in selected for row in consumers):
                outputs.setdefault(value["id"], value)
    return list(inputs.values()), list(outputs.values())


def graph_binding_hypotheses(graph_index: dict, listing_inventory: dict, *, max_members: int = 4) -> dict:
    """Cover every listing region with candidate source selections or an explicit gap.

    Complete body matches may associate symbol path tokens with actual captured
    module paths. Indexed operator names retain both zero/one-based conventions.
    Connected successor selections expose alternative fusion frontiers without
    asserting effect closure or an actual fused implementation. Unmatched calls
    retain the complete source roster; no call index forces a graph ordinal.
    """
    if graph_index.get("schema") != "merlin.capture_graph_regions.v1":
        raise ValueError("graph binding requires a complete selected capture region index")
    if type(max_members) is not int or not 1 <= max_members <= 16:
        raise ValueError("graph binding member budget must be between1 and16")
    regions = graph_index["regions"]
    by_id = {region["region_id"]: region for region in regions}
    if len(by_id) != len(regions):
        raise ValueError("graph binding repeats source region identities")
    paths = {identity: _module_path(region) for identity, region in by_id.items()}
    common = _common_prefix([path for path in paths.values() if path])
    names = {identity: _identifier(".".join(path[common:])) for identity, path in paths.items() if path}
    by_family = defaultdict(list)
    for region in regions:
        target = region["operation"].get("target", "")
        pieces = target.split(".")
        family = _identifier(pieces[1] if len(pieces) > 1 else target)
        if family:
            by_family[family].append(region["region_id"])
    bindings = []
    listing_ids = set()
    for listed in listing_inventory["regions"]:
        identity = listed["region_id"]
        if identity in listing_ids:
            raise ValueError("graph binding repeats instruction region identities")
        listing_ids.add(identity)
        witness = listed.get("body_match", {})
        seeds = defaultdict(list)
        complete = witness.get("basis") == "complete_listed_body_bytes" and witness.get("contiguous_body") is True
        for match in witness.get("matches", []) if complete else ():
            symbol = match.get("symbol", "")
            normalized = _identifier(symbol)
            matches = [
                (source_id, name) for source_id, name in names.items() if name and f"_{name}_" in f"_{normalized}_"
            ]
            longest = max((len(name) for _, name in matches), default=0)
            for source_id, name in matches:
                if len(name) == longest:
                    seeds[source_id].append(
                        {
                            "kind": "named_body_module_path",
                            "symbol": symbol,
                            "source_module_path": ".".join(paths[source_id]),
                            "body_match": match,
                        }
                    )
            tokens = normalized.split("_")
            for index, token in enumerate(tokens[:-1]):
                if token not in by_family or not tokens[index + 1].isdigit():
                    continue
                position = int(tokens[index + 1])
                for base in (0, 1):
                    ordinal = position - base
                    if 0 <= ordinal < len(by_family[token]):
                        seeds[by_family[token][ordinal]].append(
                            {
                                "kind": "indexed_operator_name_hypothesis",
                                "symbol": symbol,
                                "operator_family": token,
                                "index_base": base,
                                "body_match": match,
                            }
                        )
        candidates = []
        for source_id, basis in seeds.items():
            members = [source_id]
            while True:
                inputs, outputs = _frontier(members, by_id)
                candidates.append(
                    {
                        "graph_region_ids": list(members),
                        "inputs": inputs,
                        "outputs": outputs,
                        "basis": basis,
                        "fusion_status": "hypothesis_only",
                        "effect_closure": "unknown",
                    }
                )
                if len(members) >= max_members:
                    break
                successors = {
                    row["node_id"]
                    for value in by_id[members[-1]]["outputs"]
                    for row in by_id[members[-1]]["output_consumers"].get(value["id"], [])
                }
                if len(successors) != 1:
                    break
                successor = next(iter(successors))
                if successor not in by_id or successor in members:
                    break
                members.append(successor)
        bindings.append(
            {
                "region_id": identity,
                "entry_pc": listed.get("entry_pc"),
                "call_index": listed.get("call_index"),
                "listing_evidence_identity": listed.get("evidence_identity"),
                "body_match": witness,
                "status": "hypothesized" if candidates else "unbound",
                "candidate_region_ids": list(seeds) if seeds else list(by_id),
                "candidate_bindings": candidates,
                "source_binding_basis": "named exact-body provenance association"
                if candidates
                else "complete source roster; correspondence unbound",
                "numeric_match_status": "unknown",
                "source_region": None,
                "gaps": [
                    {
                        "kind": "reference_alignment_gap",
                        "reason": "Source/expert operand, numerical and fusion equality remains unproved.",
                    },
                    {
                        "kind": "source_region_extraction_gap",
                        "reason": "Source selections require SSA extraction and independent effect checks.",
                    },
                ],
            }
        )
    return {
        "schema": "merlin.graph_binding_hypotheses.v1",
        "graph_index_identity": canonical_sha256(graph_index),
        "listing_inventory_identity": canonical_sha256(listing_inventory),
        "graph_bindings": bindings,
        "max_members": max_members,
        "not_established": [
            "semantic_correspondence",
            "quantizer_equality",
            "fusion_equivalence",
            "complete_elf_identity",
            "numerical_equivalence",
            "performance",
        ],
    }
