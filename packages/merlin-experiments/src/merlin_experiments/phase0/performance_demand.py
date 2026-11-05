"""Derived Phase 0 stage: performance test cases from captured models' layer inventories.

A performance cohort written by hand samples the shapes its author thought of. The one this replaces
had 228 cases spanning 16 to 2.1 G MACs, yet held the dominant layers of none of the programs it was
meant to stand for: 56% of one network's work sat in tail-M, deep-K contractions that no case had,
and 81% of a vision encoder's work had no case at all. This stage derives the cohort instead.

It reads each declared program's VALUE-FREE prepared-graph projection
(:func:`merlin_experiments.corpus.model_shape_roster.project`), takes every exact integer contraction
call site (:func:`merlin_experiments.corpus.projection_shapes.contractions`) and

1. groups call sites by exact ``(M, K, N)`` shape and by REGIME CLASS -- the bin of each extent on a
   scale of the target's own tile edge, read from the RTL facts (``M = 1`` is its own class);
2. ranks shapes by their share of the program's contraction MACs;
3. holds out, per class with at least two distinct shapes, the class's smallest shape that is not
   already a public case, as a hidden member (holdouts measure transfer within a regime);
4. selects public shapes in rank order until the declared ``coverage_target`` fraction of the
   program's MACs is covered (plus every existing exact case of a layer holding at least
   ``min_scored_share`` of the work), each by its exact shape or, when the exact shape exceeds the declared
   storage or cost bound, by the nearest admissible representative: only the positions (M) shrink,
   in whole tiles, keeping the model's residue modulo the tile; reduction depth and features never
   change (the rule :mod:`.model_forms` applies to reduced layers).

A shape another program also needs becomes ONE member that lists every program it serves. A shape the
existing corpus already holds with the same semantics is recorded as satisfied by that case instead of
being emitted twice. Every member names the model, program and class it represents.

Nothing here names a target, a model or an ISA constant: the tile edge comes from the RTL facts, the
bins and bounds from the declared per-target policy, model and program labels from the caller.
This stage selects shapes only; programs are written through the native interface IR, goldens come
from the grader's integer reference at L0, and no timing is predicted.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "merlin.phase0.performance_demand_policy.v1"
PLAN_SCHEMA = "merlin.phase0.performance_demand_plan.v1"
#: The provenance block each derived member carries onto its capsule.
BLOCK = "performance_demand"
_POLICY_KEYS = {
    "schema",
    "coverage_target",
    "min_scored_share",
    "class_edges",
    "holdout",
    "admissibility",
    "name_prefix",
}
_AXES = ("M", "K", "N")


class PerformanceDemandError(ValueError):
    """The declared policy, projection or template cannot derive a complete, admissible plan."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_policy(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a declared performance-demand policy; every knob is explicit, none defaulted."""
    if not isinstance(document, Mapping) or set(document) != _POLICY_KEYS or document["schema"] != POLICY_SCHEMA:
        raise PerformanceDemandError(f"policy must be {POLICY_SCHEMA} with exactly {sorted(_POLICY_KEYS)}")
    target = document["coverage_target"]
    if not isinstance(target, (int, float)) or isinstance(target, bool) or not 0 < target <= 1:
        raise PerformanceDemandError("coverage_target is a fraction of each program's MACs in (0, 1]")
    floor = document["min_scored_share"]
    if not isinstance(floor, (int, float)) or isinstance(floor, bool) or not 0 <= floor < 1:
        raise PerformanceDemandError("min_scored_share is the program MAC fraction an existing case must reach")
    edges = document["class_edges"]
    if not isinstance(edges, Mapping) or set(edges) != set(_AXES):
        raise PerformanceDemandError("class_edges declares tile multiples for exactly M, K and N")
    for axis, values in edges.items():
        if (
            not isinstance(values, list)
            or not values
            or any(type(v) is not int or v < 1 for v in values)
            or values != sorted(set(values))
        ):
            raise PerformanceDemandError(f"class_edges.{axis} must be strictly increasing positive tile multiples")
    if document["holdout"] not in {"one_per_class", "none"}:
        raise PerformanceDemandError("holdout is one_per_class or none")
    bounds = document["admissibility"]
    if not isinstance(bounds, Mapping) or set(bounds) != {"logical_storage_budget_bytes", "max_case_macs"}:
        raise PerformanceDemandError("admissibility declares logical_storage_budget_bytes and max_case_macs")
    for key, value in bounds.items():
        if value is not None and (type(value) is not int or value < 1):
            raise PerformanceDemandError(f"admissibility.{key} must be a positive integer or null")
    prefix = document["name_prefix"]
    if not isinstance(prefix, str) or not prefix.isalnum() or not prefix[:1].isupper():
        raise PerformanceDemandError("name_prefix is a short alphanumeric capitalized origin tag")
    return copy.deepcopy(dict(document))


def tile_edge(facts: Mapping[str, Any]) -> int:
    """The square compute array's edge from the RTL facts; ambiguity fails closed."""
    body = facts.get("facts", facts) if isinstance(facts, Mapping) else {}
    arrays = [row for row in body.get("arrays") or () if isinstance(row, Mapping)]
    edges = {(row.get("rows"), row.get("cols")) for row in arrays}
    if len(edges) != 1:
        raise PerformanceDemandError("the RTL facts must name exactly one compute array geometry")
    rows, cols = next(iter(edges))
    if type(rows) is not int or rows < 1 or rows != cols:
        raise PerformanceDemandError("performance classes need one square compute array edge")
    return rows


def inventory(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Exact contraction call sites of one program, grouped by shape, with their MAC weight."""
    from merlin_experiments.corpus.projection_shapes import contractions

    rows = []
    for group in contractions(dict(projection)):
        m, k, n = group["shape_mkn"]
        count = len(group["occurrences"])
        rows.append(
            {
                "shape": [m, k, n],
                "occurrences": count,
                "macs": m * k * n * count,
                "source_node_ids": list(group["source_node_ids"]),
                "trace_sha256": group["trace_sha256"],
                "graph_sha256": group.get("graph_sha256"),
                "operation": group["operation"],
            }
        )
    if not rows:
        raise PerformanceDemandError("the selected program has no exact integer contraction call site")
    return sorted(rows, key=lambda row: (-row["macs"], row["shape"]))


def regime_class(shape: Sequence[int], *, tile: int, edges: Mapping[str, Sequence[int]]) -> str:
    """``M<=4t/K<=128t/N>128t``-style bin of each extent; a vector (M = 1) is its own M class."""
    parts = []
    for axis, extent in zip(_AXES, shape, strict=True):
        if axis == "M" and extent == 1:
            parts.append("M=1")
            continue
        bound = next((edge for edge in edges[axis] if extent <= edge * tile), None)
        parts.append(f"{axis}<={bound}t" if bound is not None else f"{axis}>{edges[axis][-1]}t")
    return "/".join(parts)


def footprint(shape: Sequence[int], *, operand_bytes: int = 1, result_bytes: int = 4) -> int:
    """Logical bytes of the raw contract's operands and result (int8 inputs, int32 output)."""
    m, k, n = shape
    return (m * k + k * n) * operand_bytes + m * n * result_bytes


def admissible(shape: Sequence[int], bounds: Mapping[str, int | None]) -> str | None:
    """``None`` when the shape fits every declared bound, else the bound it exceeds."""
    storage, macs = bounds["logical_storage_budget_bytes"], bounds["max_case_macs"]
    if storage is not None and footprint(shape) > storage:
        return "logical_storage_budget_bytes"
    m, k, n = shape
    if macs is not None and m * k * n > macs:
        return "max_case_macs"
    return None


def representative(shape: Sequence[int], *, tile: int, bounds: Mapping[str, int | None]) -> list[int] | None:
    """Nearest admissible shape that shrinks only the positions, in whole tiles, keeping M mod tile."""
    m, k, n = shape
    for candidate in range(m - tile, 0, -tile):
        if admissible((candidate, k, n), bounds) is None:
            return [candidate, k, n]
    return None


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def derive(
    programs: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any],
    facts: Mapping[str, Any],
    existing: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    """The digest-bound plan over every declared program.

    ``programs`` rows are ``{"model", "program", "projection"}`` with the value-free projection of one
    captured program. ``existing`` maps public case names already in the corpus to the ``(M, K, N)`` of
    their raw int8-to-int32 contraction, so a needed shape is reused instead of duplicated and a
    holdout never repeats a public point.
    """
    policy = load_policy(policy)
    tile = tile_edge(facts)
    edges, bounds, target = policy["class_edges"], policy["admissibility"], policy["coverage_target"]
    labels = [(str(row["model"]), str(row["program"])) for row in programs]
    if not programs or len(labels) != len(set(labels)):
        raise PerformanceDemandError("select each captured program once")
    known = {tuple(shape): name for name, shape in sorted((existing or {}).items())}
    tables = {}
    for (model, program), row in zip(labels, programs, strict=True):
        rows = inventory(row["projection"])
        total = sum(item["macs"] for item in rows)
        for item in rows:
            item["share"] = item["macs"] / total
            item["class"] = regime_class(item["shape"], tile=tile, edges=edges)
        tables[(model, program)] = {"rows": rows, "total_macs": total}
    # Tentative holdout per class: its smallest shape that is not already public.
    tentative = {}
    for key, table in tables.items():
        by_class = {}
        for item in table["rows"]:
            by_class.setdefault(item["class"], []).append(item)
        for cls, members in by_class.items():
            if policy["holdout"] == "one_per_class" and len(members) >= 2:
                candidates = [
                    m for m in sorted(members, key=lambda r: (r["macs"], r["shape"])) if tuple(m["shape"]) not in known
                ]
                tentative[(key, cls)] = candidates
    # Public selection in rank order until the covered fraction reaches the target.
    public = {}
    for key, table in tables.items():
        held = {tuple(c[0]["shape"]) for (k, _), c in tentative.items() if k == key and c}
        covered = 0.0
        for item in table["rows"]:
            shape = tuple(item["shape"])
            if shape in held:
                continue
            # An existing exact case of a material layer costs nothing new and keeps its history.
            if covered < target or (shape in known and item["share"] >= policy["min_scored_share"]):
                covered += item["share"]
                public.setdefault(shape, []).append((key, item))
    # Final holdouts: never a public point of this plan or the corpus; only for classes served publicly.
    hidden = {}
    for (key, cls), candidates in sorted(tentative.items()):
        if not any(item["class"] == cls for uses in public.values() for owner, item in uses if owner == key):
            continue
        choice = next((c for c in candidates if tuple(c["shape"]) not in public), None)
        if choice is not None:
            hidden.setdefault(tuple(choice["shape"]), []).append((key, choice))
    members, deferred = [], []
    for split, chosen in (("public", public), ("hidden", hidden)):
        for shape, uses in sorted(chosen.items(), key=lambda kv: (-sum(i["macs"] for _, i in kv[1]), kv[0])):
            exceeded = admissible(shape, bounds)
            written = list(shape) if exceeded is None else representative(shape, tile=tile, bounds=bounds)
            if written is None:
                deferred.append(
                    {"shape": list(shape), "split": split, "reason": f"no admissible representative under {exceeded}"}
                )
                continue
            (model, program), lead = uses[0]
            stem = f"m{written[0]}k{written[1]}n{written[2]}"
            tag = policy["name_prefix"] + ("H" if split == "hidden" else "")
            name = f"{tag}_{_slug(model)}_{_slug(program)}_{stem}_raw"
            satisfied = known.get(tuple(written)) if split == "public" else None
            members.append(
                {
                    "name": satisfied or name,
                    "split": split,
                    "shape": written,
                    "source_shape": list(shape),
                    "admission": "exact" if exceeded is None else "representative",
                    "exceeded_bound": exceeded,
                    "satisfied_by_existing": satisfied is not None,
                    "represents": [
                        {
                            "model": m,
                            "program": p,
                            "class": item["class"],
                            "mac_share": item["share"],
                            "occurrences": item["occurrences"],
                            "trace_sha256": item["trace_sha256"],
                            "graph_sha256": item["graph_sha256"],
                            "operation": item["operation"],
                            "source_node_ids": item["source_node_ids"],
                        }
                        for (m, p), item in uses
                    ],
                }
            )
    coverage = {}
    for key, table in tables.items():
        exact = represented = held_share = 0.0
        public_points = {tuple(m["source_shape"]): m for m in members if m["split"] == "public"}
        hidden_points = {tuple(m["source_shape"]) for m in members if m["split"] == "hidden"}
        for item in table["rows"]:
            member = public_points.get(tuple(item["shape"]))
            if member is not None:
                if member["admission"] == "exact":
                    exact += item["share"]
                else:
                    represented += item["share"]
            elif tuple(item["shape"]) in hidden_points:
                held_share += item["share"]
        coverage[f"{key[0]}/{key[1]}"] = {
            "total_macs": table["total_macs"],
            "distinct_shapes": len(table["rows"]),
            "classes": len({item["class"] for item in table["rows"]}),
            "exact_public_share": exact,
            "represented_public_share": represented,
            "hidden_share": held_share,
            "status": "target_met" if exact + represented >= target - 1e-12 else "target_unreachable",
        }
    names = [m["name"] for m in members if not m["satisfied_by_existing"]]
    if len(names) != len(set(names)):
        raise PerformanceDemandError("two derived members minted the same case name")
    plan = {
        "schema": PLAN_SCHEMA,
        "generated_by": f"{__name__}.derive",
        "tile_edge": tile,
        "policy": policy,
        "programs": [
            {"model": m, "program": p, "trace_sha256": tables[(m, p)]["rows"][0]["trace_sha256"]} for m, p in labels
        ],
        "members": members,
        "deferred": deferred,
        "coverage": coverage,
        "claims": {
            "measured": False,
            "latency_weighting": False,
            "captured_boundary_values": False,
            "full_model_equivalence": False,
        },
    }
    return {**plan, "sha256": _digest(plan)}


def _render(template: Path, member: Mapping[str, Any], plan_sha256: str):
    """One raw int8-to-int32 contract from the reviewed template, with this member's provenance."""
    import yaml
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir, parse_interface_mlir

    cap = yaml.safe_load((template / "capsule.yaml").read_text())
    program = parse_interface_mlir((template / cap["interface_mlir"]).read_text())
    attrs = cap["operation"]["attributes"]
    if (
        cap["operation"]["op"] != "matmul"
        or attrs.get("epilogue") != []
        or attrs.get("output_dtype") != "i32"
        or cap.get("numeric_policy") != {"compare": "exact_int", "dtype": "i32"}
    ):
        raise PerformanceDemandError("template must be the independent raw int8-to-int32 matrix contract")
    m, k, n = member["shape"]
    for item in cap["inputs"]:
        if item["name"] not in {attrs["lhs"], attrs["weight"]} or item["dtype"] != "i8":
            raise PerformanceDemandError("template input ABI is not the raw matrix contract")
        item["shape"] = [m, k] if item["name"] == attrs["lhs"] else [k, n]
        program["tensors"][item["name"]]["shape"] = item["shape"]
    if attrs["out"] in program["tensors"]:
        program["tensors"][attrs["out"]]["shape"] = [m, n]
    for key in ("software_screen", "integer_partial_sum_bound", "extends", "max_oracle_tier", "performance"):
        cap.pop(key, None)
    uses = "; ".join(
        f"{row['model']}/{row['program']} class {row['class']} "
        f"({100 * row['mac_share']:.1f}% of contraction MACs, {row['occurrences']} call sites)"
        for row in member["represents"]
    )
    kind = (
        "exact layer shape"
        if member["admission"] == "exact"
        else (f"nearest admissible representative of {member['source_shape']} under {member['exceeded_bound']}")
    )
    cap.update(
        name=member["name"],
        source_role="handauthored_compiler_test",
        source_reference=f"Phase 0 performance demand: {kind} of {uses}. Independent deterministic "
        "int8 inputs, exact int32 product; not captured-boundary or model equivalence.",
        required_oracle_tiers=["L0", "L1", "L2"],
        label=member["split"],
        performance_scope="scored",
    )
    cap[BLOCK] = {
        "plan_sha256": plan_sha256,
        "admission": member["admission"],
        "source_shape_mkn": member["source_shape"],
        "represents": [
            {key: row[key] for key in ("model", "program", "class", "mac_share", "occurrences")}
            for row in member["represents"]
        ],
    }
    binding = {
        "plan_sha256": plan_sha256,
        "represents": member["represents"],
        "numerical_scope": "exact int32 product of independent deterministic int8 inputs",
        "captured_boundary_values": False,
        "connected_cut": False,
        "full_model": False,
        "execution_status": "not_executed",
    }
    return cap, emit_interface_mlir(program), binding


def write_public_sources(plan: Mapping[str, Any], *, template: Path, destination: Path) -> list[Path]:
    """Render every emitted public member as a standalone source contract for scalability intake."""
    import yaml

    if destination.exists():
        raise PerformanceDemandError("select a new source directory")
    destination.mkdir(parents=True)
    written = []
    for member in plan["members"]:
        if member["split"] != "public" or member["satisfied_by_existing"]:
            continue
        cap, text, binding = _render(Path(template), member, plan["sha256"])
        path = destination / member["name"]
        path.mkdir()
        (path / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False))
        (path / cap["interface_mlir"]).write_text(text)
        (path / "source-shape-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
        written.append(path)
    return written


def hidden_profile(plan: Mapping[str, Any], *, template: Path) -> dict[str, Any]:
    """Held-out members as Phase 0 hidden-profile ``capsules:`` entries (the existing hidden route).

    The generation writer places them under ``hidden/`` with label ``hidden``, computes their goldens
    with the integer engine and checks they repeat no public program point.
    """
    import yaml

    cap = yaml.safe_load((Path(template) / "capsule.yaml").read_text())
    attrs = cap["operation"]["attributes"]
    modes = (cap.get("expected") or {}).get("modes") or {}
    entries = []
    for member in plan["members"]:
        if member["split"] != "hidden":
            continue
        m, k, n = member["shape"]
        entries.append(
            {
                "name": member["name"],
                "kind": "isa",
                "cat": "hidden",
                "label": "hidden",
                "op": "matmul",
                "lhs": attrs["lhs"],
                "weight": attrs["weight"],
                "out": attrs["out"],
                "modes": dict(modes),
                "stimulus_range": list(cap["stimulus_range"]),
                "M": m,
                "K": k,
                "N": n,
                # Derived from a model inventory. A holdout joins the generated hidden cohort, so it carries
                # the target's declared tiers like every other generated member and is held to the
                # descriptor's certification floor; this stage has no basis to cap it below that floor.
                "source_role": "model_derived",
                "source_reference": "Phase 0 performance-demand holdout of one regime class; "
                f"plan {plan['sha256'][:12]}",
            }
        )
    return {"capsules": entries}
