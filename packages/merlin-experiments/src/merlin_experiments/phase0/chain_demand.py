"""Derived Phase 0 stage: connected chain test cases from captured programs' graph structure.

Every case the performance-demand stage (:mod:`.performance_demand`) derives is one contraction, so
nothing in its cohort can be fused, share a loaded input or keep an intermediate on the device. This
stage reads the same pinned captured programs, finds their producer-to-consumer chains by structure
(:mod:`.chain_graph`) and writes each selected chain as ONE test case whose program contains every
operation of the chain, so its cycles cover the whole chain:

* ``conv_requant_pool`` -- a convolution whose readout bias, requantization and activation are
  followed by a max pool, declared in the convolution's ordered epilogue;
* ``contraction_residual`` -- a contraction, its requantizing readout and a residual add as separate
  regions (``residual_seam``); a pointwise convolution is written as the contraction over its image
  positions, which is the same arithmetic;
* ``shared_input_siblings`` -- several contractions of one input, as one tensor program;
* ``sibling_feed`` -- two sibling contractions whose readouts merge and feed a third contraction;
* ``shared_kv_heads`` -- every query head's score and context contraction against its shared key and
  value, as one tensor program.

Selection is declared in a per-target policy and works on chain GEOMETRY. Instances are grouped by
signature; within each family group, each program offers its signature with the largest share of the
program's contraction MACs, and the largest shares (one per signature) up to the declared count become
PERFORMANCE members at their exact captured shape when admissible. From each group's leading
signature the stage also derives small CORRECTNESS variants on the target's tile scale (read from the
RTL facts): a reduced one at a few tiles, an edge one with one-element tails, and for a pool a variant
whose pooling windows are mostly padding. Performance members join the public scalability suite and
are scored; correctness variants join the generated correctness cohort.

Nothing here names a model, a target or an ISA constant. Inputs are independent deterministic
stimuli, readout scales are powers of two chosen from the declared stimulus range and reduction
depth, and goldens come from the ordinary writer's integer reference. No timing is read or predicted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import chain_graph as G

POLICY_SCHEMA = "merlin.phase0.chain_demand_policy.v1"
PLAN_SCHEMA = "merlin.phase0.chain_demand_plan.v1"
#: The provenance block each chain case carries.
BLOCK = "chain_demand"
_POLICY_KEYS = {"schema", "families", "performance", "correctness", "stimulus_range", "readout_rms", "name_prefix"}
_PERFORMANCE_KEYS = {"members_per_group", "admissibility", "minimum_input_elements", "max_program_nodes"}
_CORRECTNESS_KEYS = {"reduced_tiles", "edge_tail"}
#: Short, stable spellings used in case names, by selection group.
_SHORT = {
    ("conv_requant_pool", None): "conv_pool",
    ("contraction_residual", "pointwise"): "residual_conv",
    ("contraction_residual", "linear"): "residual_linear",
    ("shared_input_siblings", None): "siblings",
    ("sibling_feed", None): "sibling_feed",
    ("shared_kv_heads", None): "kv_heads",
}


class ChainDemandError(ValueError):
    """The declared policy or a captured program cannot derive a complete chain plan."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _positive_int(value, what):
    if type(value) is not int or value < 1:
        raise ChainDemandError(f"{what} must be a positive integer")
    return value


def load_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a declared chain-demand policy; every knob is explicit, none defaulted."""
    if not isinstance(document, Mapping) or set(document) != _POLICY_KEYS or document["schema"] != POLICY_SCHEMA:
        raise ChainDemandError(f"policy must be {POLICY_SCHEMA} with exactly {sorted(_POLICY_KEYS)}")
    families = document["families"]
    if not isinstance(families, list) or not families or len(set(families)) != len(families) \
            or any(f not in G.FAMILIES for f in families):
        raise ChainDemandError(f"families must be distinct members of {list(G.FAMILIES)}")
    performance = document["performance"]
    if not isinstance(performance, Mapping) or set(performance) != _PERFORMANCE_KEYS:
        raise ChainDemandError(f"performance declares exactly {sorted(_PERFORMANCE_KEYS)}")
    _positive_int(performance["members_per_group"], "performance.members_per_group")
    _positive_int(performance["minimum_input_elements"], "performance.minimum_input_elements")
    _positive_int(performance["max_program_nodes"], "performance.max_program_nodes")
    bounds = performance["admissibility"]
    if not isinstance(bounds, Mapping) or set(bounds) != {"logical_storage_budget_bytes", "max_case_macs"}:
        raise ChainDemandError("performance.admissibility declares logical_storage_budget_bytes and max_case_macs")
    for key, value in bounds.items():
        _positive_int(value, f"performance.admissibility.{key}")
    correctness = document["correctness"]
    if not isinstance(correctness, Mapping) or set(correctness) != _CORRECTNESS_KEYS:
        raise ChainDemandError(f"correctness declares exactly {sorted(_CORRECTNESS_KEYS)}")
    _positive_int(correctness["reduced_tiles"], "correctness.reduced_tiles")
    _positive_int(correctness["edge_tail"], "correctness.edge_tail")
    declared = document["stimulus_range"]
    low, high = declared if isinstance(declared, list) and len(declared) == 2 else (None, None)
    if type(low) is not int or type(high) is not int or not low < 0 < high:
        raise ChainDemandError("stimulus_range is an inclusive signed [low, high] around zero")
    rms = document["readout_rms"]
    if type(rms) not in (int, float) or isinstance(rms, bool) or not 0 < rms < 128:
        raise ChainDemandError("readout_rms is the intended readout magnitude, in operand steps")
    prefix = document["name_prefix"]
    if not isinstance(prefix, str) or not prefix.isalnum() or not prefix[:1].isupper():
        raise ChainDemandError("name_prefix is a short alphanumeric capitalized origin tag")
    return copy.deepcopy(dict(document))


def program_macs(graph: Mapping[str, Any]) -> int:
    """All parameter-weighted contraction MACs of a program plus its attention contractions."""
    total = sum(c["M"] * c["K"] * c["N"] for nid in graph["nodes"] if (c := G.contraction(graph, nid)))
    total += sum(G.macs(item) for item in G.instances(graph) if item["family"] == "shared_kv_heads")
    return total


def _group(row: Mapping[str, Any]) -> tuple[str, str | None]:
    if row["family"] == "contraction_residual":
        return row["family"], row["instance"]["producer"]["kind"]
    return row["family"], None


# --- case geometry ---------------------------------------------------------------------------------


def readout_scale(reduction: int, policy: Mapping[str, Any]) -> float:
    """A power-of-two readout multiplier aiming the readout at ``readout_rms`` operand steps.

    Operands are independent uniform integers over the declared stimulus range, of mean ``mu`` and
    second moment ``m2``; a depth-``reduction`` sum of their products has mean ``reduction * mu**2``
    and variance ``reduction * (m2**2 - mu**4)``, so its root mean square is known before any value
    is drawn. A power of two keeps the multiplier exact in float32.
    """
    low, high = policy["stimulus_range"]
    values = range(low, high + 1)
    mu = sum(values) / len(values)
    m2 = sum(v * v for v in values) / len(values)
    rms = math.sqrt(reduction * (m2 * m2 - mu ** 4) + (reduction * mu * mu) ** 2)
    return 2.0 ** round(math.log2(policy["readout_rms"] / rms))


def _conv_input(out: int, kernel: int, stride: int, pad: int, dilation: int) -> int:
    return (out - 1) * stride - 2 * pad + dilation * (kernel - 1) + 1


def _cap(extent: int, limit: int) -> int:
    return extent if extent == 1 else min(extent, limit)


def variants(row: Mapping[str, Any], *, tile: int, policy: Mapping[str, Any]) -> list[tuple[str, dict]]:
    """Small correctness geometries of one chain signature on the tile scale."""
    reduced, tail = policy["correctness"]["reduced_tiles"] * tile, policy["correctness"]["edge_tail"]
    inst, family = row["instance"], row["family"]
    out: list[tuple[str, dict]] = []
    if family == "conv_requant_pool":
        p, pool = inst["producer"], inst["pool"]
        (kp, _), (sp, _) = pool["size"], pool["stride"]
        cout = min(p["cout"], tile)
        for name, (ho, wo), channels in (("reduced", (4 * sp, 4 * sp), cout),
                                         ("edge", (3 * sp + 1, 2 * sp + 1), cout + tail),
                                         ("window", (kp, kp), cout)):
            if ho < 1 or wo < 1:
                continue
            height = _conv_input(ho, p["kh"], p["stride"][0], p["padding"][0], p["dilation"][0])
            width = _conv_input(wo, p["kw"], p["stride"][1], p["padding"][1], p["dilation"][1])
            if height < 1 or width < 1:
                continue
            out.append((name, {**inst, "producer": {**p, "height": height, "width": width, "cout": channels,
                                                    "out_height": ho, "out_width": wo}}))
        return out
    for name, limit, extra in (("reduced", reduced, 0), ("edge", tile, tail)):
        def fit(extent, limit=limit, extra=extra):
            return 1 if extent == 1 else min(extent, limit) + extra
        if family == "contraction_residual":
            p = inst["producer"]
            out.append((name, {**inst, "producer": {**p, "M": fit(p["M"]), "K": fit(p["K"]), "N": fit(p["N"])}}))
        elif family == "shared_input_siblings":
            widest = max(inst["N"])
            # The widest sibling is capped like any extent; the others keep their proportion to it.
            widths = [max(1, -(-n * min(limit, widest) // widest)) + extra for n in inst["N"]]
            out.append((name, {**inst, "M": fit(inst["M"]), "K": fit(inst["K"]), "N": widths}))
        elif family == "sibling_feed":
            out.append((name, {**inst, **{axis: fit(inst[axis]) for axis in ("M", "K", "H", "N")}}))
        elif family == "shared_kv_heads":
            share = inst["query_heads"] // inst["kv_heads"]
            groups = 1 if name == "reduced" else min(2, inst["kv_heads"])
            out.append((name, {**inst, "query_heads": share * groups, "kv_heads": groups,
                               **{axis: fit(inst[axis], min(limit, tile) if axis != "L" else limit)
                                  for axis in ("S", "L", "d", "dv")}}))
    return out


def stem(family: str, inst: Mapping[str, Any]) -> str:
    """Geometry spelled into a case name."""
    if family == "conv_requant_pool":
        p, pool = inst["producer"], inst["pool"]
        return (f"c{p['cin']}h{p['height']}w{p['width']}k{p['kh']}s{p['stride'][0]}n{p['cout']}"
                f"_p{pool['size'][0]}s{pool['stride'][0]}")
    if family == "contraction_residual":
        p = inst["producer"]
        return f"m{p['M']}k{p['K']}n{p['N']}"
    if family == "shared_input_siblings":
        return f"m{inst['M']}k{inst['K']}n" + "n".join(str(n) for n in inst["N"])
    if family == "sibling_feed":
        return f"m{inst['M']}k{inst['K']}h{inst['H']}n{inst['N']}"
    return f"q{inst['query_heads']}kv{inst['kv_heads']}s{inst['S']}l{inst['L']}d{inst['d']}"


def entry(family: str, inst: Mapping[str, Any], *, name: str, policy: Mapping[str, Any], block: dict,
          narrow_bits: int) -> dict:
    """The ordinary Phase 0 profile entry that writes this chain through an existing builder.

    ``narrow_bits`` is the width of the readout format the target requantizes into; a padded pool
    window pads with that format's least value, so padding never wins a maximum.
    """
    base = {
        "name": name, "kind": "model_slice", "label": "public", "source_role": "model_derived",
        "stimulus_range": list(policy["stimulus_range"]), BLOCK: block,
    }
    if family == "conv_requant_pool":
        p, pool = inst["producer"], inst["pool"]
        stages = (["bias_add"] if inst["affine"] or p["bias"] else []) + ["acc_scale"] \
            + (["relu"] if inst["relu"] else []) + ["maxpool"]
        return {**base, "op": "conv2d", "Himg": p["height"], "Wimg": p["width"], "ci": p["cin"], "N": p["cout"],
                "kh": p["kh"], "kw": p["kw"], "stride": list(p["stride"]),
                "padding": [p["padding"][0], p["padding"][0], p["padding"][1], p["padding"][1]],
                "dilation": list(p["dilation"]), "epilogue": stages,
                "acc_scale": readout_scale(p["cin"] * p["kh"] * p["kw"], policy),
                "pool_size": list(pool["size"]), "pool_stride": list(pool["stride"]),
                "pool_padding": [pool["padding"][0], pool["padding"][0], pool["padding"][1], pool["padding"][1]],
                "pool_pad_value": -(1 << (narrow_bits - 1))}
    if family == "contraction_residual":
        p = inst["producer"]
        return {**base, "op": "residual_seam", "M": p["M"], "K": p["K"], "N": p["N"], "epilogue": ["acc_scale"],
                "acc_scale": readout_scale(p["K"], policy),
                "add_epilogue": ["relu"] if inst["relu_after_add"] else [],
                "lhs_scale": 1.0, "rhs_scale": 1.0, "bound_lsb": 0}
    if family == "sibling_feed":
        scale = readout_scale(inst["K"], policy)
        return {**base, "op": "sibling_feed", "M": inst["M"], "K": inst["K"], "H": inst["H"], "N": inst["N"],
                "gate_acc_scale": scale, "up_acc_scale": scale, "lhs_scale": 1.0, "rhs_scale": 1.0, "bound_lsb": 0}
    return {**base, "op": "component_program", "program": component_program(family, inst)}


def component_program(family: str, inst: Mapping[str, Any]) -> dict:
    """The typed tensor program of a sibling or shared key/value chain (raw accumulator outputs)."""
    from merlin.targetgen.component_program import LOGICAL_IO_V1, SCHEMA_V2

    def tensor(name, role, shape):
        return {"name": name, "role": role, "shape": list(shape), "dtype": "operand"}

    if family == "shared_input_siblings":
        inputs = [tensor("X", "input", (inst["M"], inst["K"]))]
        inputs += [tensor(f"W{i}", "weight", (inst["K"], n)) for i, n in enumerate(inst["N"])]
        nodes = [{"name": f"p{i}", "op": "matmul", "inputs": ["X", f"W{i}"]} for i in range(len(inst["N"]))]
        outputs = [{"name": f"Y{i}", "value": f"p{i}"} for i in range(len(inst["N"]))]
    else:
        share = inst["query_heads"] // inst["kv_heads"]
        inputs, nodes, outputs = [], [], []
        for group in range(inst["kv_heads"]):
            inputs += [tensor(f"KT{group}", "input", (inst["d"], inst["L"])),
                       tensor(f"V{group}", "input", (inst["L"], inst["dv"]))]
        for head in range(inst["query_heads"]):
            group = head // share
            inputs += [tensor(f"Q{head}", "input", (inst["S"], inst["d"])),
                       tensor(f"P{head}", "input", (inst["S"], inst["L"]))]
            nodes += [{"name": f"s{head}", "op": "matmul", "inputs": [f"Q{head}", f"KT{group}"]},
                      {"name": f"o{head}", "op": "matmul", "inputs": [f"P{head}", f"V{group}"]}]
            outputs += [{"name": f"S{head}", "value": f"s{head}"}, {"name": f"O{head}", "value": f"o{head}"}]
    return {"schema": SCHEMA_V2, "logical_io": LOGICAL_IO_V1, "inputs": inputs, "nodes": nodes, "outputs": outputs}


def _volume(family: str, inst: Mapping[str, Any]) -> tuple[int, int, int]:
    """``(input elements, logical bytes, program nodes)`` of the case a geometry writes."""
    if family == "conv_requant_pool":
        p = inst["producer"]
        elements = p["height"] * p["width"] * p["cin"] + p["kh"] * p["kw"] * p["cin"] * p["cout"] + p["cout"]
        return elements, elements + inst["pool"]["out"][0] * inst["pool"]["out"][1] * p["cout"], 1
    if family == "contraction_residual":
        p = inst["producer"]
        elements = p["M"] * p["K"] + p["K"] * p["N"] + p["M"] * p["N"]
        return elements, elements + p["M"] * p["N"], 3
    if family == "sibling_feed":
        m, k, h, n = (inst[a] for a in ("M", "K", "H", "N"))
        elements = m * k + 2 * k * h + h * n
        return elements, elements + 4 * m * n, 5
    program = component_program(family, inst)
    elements = sum(math.prod(row["shape"]) for row in program["inputs"])
    if family == "shared_input_siblings":
        results = sum(inst["M"] * n for n in inst["N"])
    else:
        results = inst["query_heads"] * inst["S"] * (inst["L"] + inst["dv"])
    return elements, elements + 4 * results, len(program["nodes"])


def _admission(family, inst, policy) -> str | None:
    performance = policy["performance"]
    elements, footprint, nodes = _volume(family, inst)
    if footprint > performance["admissibility"]["logical_storage_budget_bytes"]:
        return "logical_storage_budget_bytes"
    if G.macs({**inst, "family": family}) > performance["admissibility"]["max_case_macs"]:
        return "max_case_macs"
    if nodes > performance["max_program_nodes"]:
        return "max_program_nodes"
    if elements < performance["minimum_input_elements"]:
        return "minimum_input_elements"
    return None


def derive(programs: Sequence[Mapping[str, Any]], *, policy: Mapping[str, Any], facts: Mapping[str, Any]) -> dict:
    """The digest-bound chain plan over every declared program.

    ``programs`` rows are ``{"model", "program", "graph", "trace_sha256", "claim"}`` with the
    structural graph (:func:`.chain_graph.read`) of one captured program. A ``claim`` program is a
    held-out evaluation model: like the performance-demand stage, its chain SHAPES may become public
    performance members, but no correctness variant is derived from it, so the generated cohort only
    holds forms of the public applications.
    """
    from .performance_demand import tile_edge

    policy = load_policy(policy)
    tile = tile_edge(facts)
    labels = [(str(row["model"]), str(row["program"])) for row in programs]
    if not programs or len(labels) != len(set(labels)):
        raise ChainDemandError("select each captured program once")
    offered: dict[tuple, list[dict]] = {}
    census = []
    for (model, program), row in zip(labels, programs, strict=True):
        if type(row.get("claim")) is not bool:
            raise ChainDemandError(f"{model}/{program}: declare whether the program is a held-out claim model")
        rows = [r for r in G.inventory(row["graph"]) if r["family"] in policy["families"]]
        total = program_macs(row["graph"])
        census.append({"model": model, "program": program, "claim": row["claim"],
                       "trace_sha256": row["trace_sha256"], "graph_sha256": row["graph"]["graph_sha256"],
                       "contraction_macs": total,
                       "signatures": [{"family": r["family"], "signature": r["signature"][1:],
                                       "occurrences": r["occurrences"], "macs": r["macs"],
                                       "mac_share": r["macs"] / total if total else 0.0} for r in rows]})
        best: dict[tuple, dict] = {}
        for r in rows:
            group = _group(r)
            if group not in best or r["macs"] > best[group]["macs"]:
                best[group] = r
        for group, r in best.items():
            offered.setdefault(group, []).append({**r, "model": model, "program": program, "claim": row["claim"],
                                                  "mac_share": r["macs"] / total if total else 0.0,
                                                  "trace_sha256": row["trace_sha256"]})
    prefix, members, deferred = policy["name_prefix"], [], []
    for group in sorted(offered):
        family, _ = group
        short = _SHORT[group]
        ranked = sorted(offered[group], key=lambda r: (-r["mac_share"], r["model"], r["program"]))
        chosen, seen, models = [], set(), set()
        for r in ranked:
            key = tuple(r["signature"])
            if key in seen or r["model"] in models:
                continue
            seen.add(key)
            refused = _admission(family, r["instance"], policy)
            if refused == "minimum_input_elements":
                continue  # a small chain is a correctness form, not a performance member
            if refused is not None:
                deferred.append({"group": short, "model": r["model"], "program": r["program"],
                                 "signature": r["signature"][1:], "reason": f"not admissible under {refused}"})
                continue
            if len(chosen) < policy["performance"]["members_per_group"]:
                chosen.append(r)
                models.add(r["model"])
        for r in chosen:
            name = f"{prefix}_{short}_{_slug(r['model'])}_{_slug(r['program'])}_{stem(family, r['instance'])}"
            members.append(_member(name, "performance", family, short, r["instance"], [r]))
        public = [r for r in ranked if not r["claim"]]
        if not public:
            deferred.append({"group": short, "reason": "no public application exhibits this chain; "
                                                       "no correctness form is derived from a claim model"})
            continue
        lead = public[0]
        if _admission(family, lead["instance"], policy) == "minimum_input_elements":
            name = f"{prefix}_{short}_{_slug(lead['model'])}_{stem(family, lead['instance'])}"
            members.append(_member(name, "correctness_exact", family, short, lead["instance"], [lead]))
        for variant, inst in variants(lead, tile=tile, policy=policy):
            if G.signature({**inst, "family": family}) == tuple(lead["signature"]):
                continue  # the variant is the captured shape itself
            name = f"{prefix}_{short}_{variant}_{stem(family, inst)}"
            members.append(_member(name, f"correctness_{variant}", family, short, inst, [lead]))
    names = [m["name"] for m in members]
    if len(names) != len(set(names)):
        raise ChainDemandError("two chain members minted the same case name")
    plan = {
        "schema": PLAN_SCHEMA, "generated_by": f"{__name__}.derive", "tile_edge": tile, "policy": policy,
        "programs": census, "members": members, "deferred": deferred,
        "claims": {"measured": False, "captured_boundary_values": False, "full_model_equivalence": False,
                   "timing_predicted": False},
    }
    return {**plan, "sha256": _digest(plan)}


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _member(name, role, family, short, inst, sources) -> dict:
    return {
        "name": name, "role": role, "family": family, "group": short, "geometry": json.loads(json.dumps(inst)),
        "cohort": "scalability" if role == "performance" else "generated",
        "represents": [{"model": r["model"], "program": r["program"], "signature": r["signature"][1:],
                        "occurrences": r["occurrences"], "mac_share": r["mac_share"],
                        "trace_sha256": r["trace_sha256"], "source_node_ids": r["source_node_ids"][:4]}
                       for r in sources],
        "admission": "exact" if role in {"performance", "correctness_exact"} else "derived_variant",
    }


def member_entry(member: Mapping[str, Any], plan_sha256: str, policy: Mapping[str, Any], *, narrow_bits: int) -> dict:
    """The profile entry of one plan member, with its public chain declaration."""
    uses = "; ".join(f"{r['model']}/{r['program']} ({r['occurrences']} instances, "
                     f"{100 * r['mac_share']:.1f}% of contraction MACs)" for r in member["represents"])
    kind = ("exact captured chain shape" if member["admission"] == "exact"
            else f"{member['role'].split('_', 1)[1]} correctness variant on the tile scale")
    block = {"plan_sha256": plan_sha256, "family": member["family"], "role": member["role"],
             "chain": CHAIN_DECLARATIONS[member["family"]],
             "represents": [{k: r[k] for k in ("model", "program", "signature", "occurrences", "mac_share")}
                            for r in member["represents"]]}
    item = entry(member["family"], member["geometry"], name=member["name"], policy=policy, block=block,
                 narrow_bits=narrow_bits)
    item["source_reference"] = (f"Phase 0 chain demand: {kind} of the {member['family']} chain of {uses}. "
                                "Independent deterministic int8 inputs; one program timed as one call; "
                                "not captured-boundary or model equivalence.")
    if member["role"] == "performance":
        item.update(cat="sources", performance_scope="scored", required_oracle_tiers=["L0", "L1", "L2"])
    else:
        item["cat"] = "model_slices"
    return item


#: The operation chain each family's case states, producer first.
CHAIN_DECLARATIONS = {
    "conv_requant_pool": ["conv2d", "bias_add", "requantize", "relu", "maxpool"],
    "contraction_residual": ["matmul", "requantize", "residual_add"],
    "shared_input_siblings": ["matmul (shared input)", "matmul (shared input)"],
    "sibling_feed": ["matmul (shared input)", "requantize", "matmul (shared input)", "requantize",
                     "residual_add", "matmul"],
    "shared_kv_heads": ["matmul (query x shared key, per head)", "matmul (probabilities x shared value, per head)"],
}
