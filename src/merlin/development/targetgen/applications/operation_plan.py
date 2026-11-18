"""Project public source operation obligations onto available capsule writers."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

# ---- application_operation_plan ----------------------------------------------------------------


def _application_operation_plan(demands: dict | None, *, exact_entries: list[dict] | None = None) -> dict:
    """Plan operation obligations without equating family cells to frontend coverage.

    The selected spec carries a compact operation projection, not the complete capture inventory.
    Its groups identify missing writers and host seams. Exact integer-matmul candidates are admitted
    only when digest-bound detailed signatures account for every source ordinal. Emission and compiler
    execution remain explicit, unverified obligations.
    """

    from merlin.development.targetgen.conformance.synthesis_entries import (
        SynthesisError,
        _cost,
        _exact_int_mm_group_candidates,
        _op_family_map,
        _writer_for,
        available_ops,
    )

    if not isinstance(demands, dict):
        return {
            "status": "not_recorded",
            "coverage_status": "unverified",
            "reason": "this conformance spec predates the application operation inventory",
            "obligations": [],
        }

    groups = demands.get("operation_groups")
    base = {
        "status": "unverified",
        "coverage_status": "unverified",
        "inventory_status": demands.get("status"),
        "full_inventory_sha256": demands.get("full_inventory_sha256"),
        "sidecar": demands.get("sidecar"),
        "n_operations": demands.get("n_operations", 0),
        "basis": (
            "operation groups are planning hints from declared application captures; a family/dtype "
            "cell or a writer candidate does not prove frontend lowering, capsule coverage, or execution"
        ),
    }
    if demands.get("status") == "not_declared":
        return {**base, "status": "not_declared", "obligations": []}
    if not isinstance(groups, list):
        return {
            **base,
            "status": "blocked",
            "reason": "compact spec has no operation groups; inspect the digest-checked full inventory sidecar",
            "missing_mapping": ["application signature -> generic writer", "application signature -> exact capsule"],
            "obligations": [],
        }

    from merlin.development.targetgen.conformance.requirements import capsule_dtype

    pool = available_ops()
    families = _op_family_map()
    exact_entries = list(exact_entries or ())
    obligations: list[dict] = []
    for group in groups:
        if not isinstance(group, dict):
            raise SynthesisError("application_demands.operation_groups contains a non-object row")
        operation = str(group.get("operation") or "")
        mlir_operation = str(group.get("mlir_operation") or "")
        family = str(group.get("semantic_family") or "")
        fmt = group.get("operand_format")
        dtype = capsule_dtype(str(fmt)) if fmt else None
        disposition = str(group.get("disposition") or "unclassified")
        # A captured frontend op is not equivalent to the linalg op left after frontend lowering.
        # Only an exact vocabulary name is an operation-specific writer candidate. A named linalg op
        # may use its final segment when no frontend/provenance operation survives in the projection.
        writer_op = operation if operation in pool else None
        if writer_op is None and operation == mlir_operation and mlir_operation.startswith("linalg."):
            leaf = mlir_operation.rpartition(".")[2]
            writer_op = leaf if leaf in pool else None
        writer = _writer_for({"op": writer_op, "operand_dtype": dtype}) if writer_op and dtype else None
        family_writers = sorted(
            (
                op
                for op in pool
                if families.get(op) == family and dtype and _writer_for({"op": op, "operand_dtype": dtype})
            ),
            key=_cost,
        )
        row = {
            "operation": operation,
            "mlir_operation": mlir_operation,
            "semantic_family": family or None,
            "operand_format": fmt,
            "shape_class": group.get("shape_class"),
            "disposition": disposition,
            "count": int(group.get("count") or 0),
            "sources": group.get("sources") or [],
            "coverage_status": "unverified",
        }
        if disposition == "hardware_admitted":
            row["lane"] = "accelerator"
            exact_candidates = (
                _exact_int_mm_group_candidates(
                    group,
                    exact_entries,
                    demands.get("full_inventory_sha256"),
                    tuple(sorted((demands.get("applications") or {}).keys())),
                )
                if (
                    operation == "aten._int_mm.default"
                    and mlir_operation == "linalg.generic"
                    and family == "contraction"
                    and fmt == "int8"
                    and group.get("shape_class") == "contraction:rank_3"
                )
                else []
            )
            if exact_candidates:
                row.update(
                    {
                        "status": "candidate_unverified",
                        "obligation": "exact_signature_capsule_and_compile_route",
                        "writer_candidate": {"op": "matmul", "source": "pytorch", "capture_op": "int_matmul"},
                        "capsule_candidates": exact_candidates,
                        "mapped_occurrences": row["count"],
                        "missing_mapping": [
                            "capture and emit every exact PyTorch integer-matmul capsule candidate",
                            "verify each emitted linalg signature against its digest-bound source",
                            "generated capsule -> executable whole-model compiler route",
                        ],
                    }
                )
            elif writer:
                row.update(
                    {
                        "status": "unverified",
                        "obligation": "exact_signature_capsule_and_compile_route",
                        "writer_candidate": {"op": writer_op, "source": writer},
                        "missing_mapping": [
                            "captured frontend signature -> writer semantics and operand ABI",
                            "writer + exact dtype/shape/layout/quantization -> generated capsule",
                            "generated capsule -> executable whole-model compiler route",
                        ],
                    }
                )
            else:
                row.update(
                    {
                        "status": "refused",
                        "obligation": "no_exact_generic_writer",
                        "writer_candidate": None,
                        "family_writer_candidates_not_equivalent": family_writers,
                        "action": (
                            "add an operation-specific generic writer or a reviewed frontend decomposition "
                            "at this dtype; a family-level capsule cannot stand in for this operation"
                        ),
                    }
                )
        elif disposition == "host_required":
            row.update(
                {
                    "lane": "host",
                    "status": "required_unverified",
                    "obligation": "host_lowering_and_boundary",
                    "action": "retain a host implementation and verify value transfer at the accelerator/host seam",
                }
            )
        elif disposition == "support_required":
            row.update(
                {
                    "lane": "support",
                    "status": "required_unverified",
                    "obligation": "graph_support_lowering",
                    "action": "lower this graph-plumbing operation; it is not a separate compute capsule",
                }
            )
        elif disposition == "unclassified":
            row.update(
                {
                    "lane": "unresolved",
                    "status": "blocked_unclassified",
                    "obligation": "resolve_operation_semantics",
                    "action": "resolve the operation, callee, shape or format before assigning a hardware or host lane",
                }
            )
        else:
            # Structural containers and nested linalg-body components are counted by the inventory,
            # but are not independent capsule demands.
            continue
        obligations.append(row)

    obligations.sort(
        key=lambda row: (
            row["disposition"],
            row["operation"],
            row["mlir_operation"],
            str(row["operand_format"]),
            str(row["shape_class"]),
        )
    )
    excluded = sum(
        int((app.get("counts") or {}).get(kind) or 0)
        for app in (demands.get("applications") or {}).values()
        for kind in ("structural", "component")
    )
    planned = sum(row["count"] for row in obligations)
    expected = int(demands.get("n_operations") or 0) - excluded
    blocked = [row for row in obligations if row["status"] in {"refused", "blocked_unclassified"}]
    return {
        **base,
        "status": (
            "blocked"
            if demands.get("status") != "inventoried" or expected <= 0 or planned != expected or blocked
            else "obligations_pending"
        ),
        "blocked_operations": sum(row["count"] for row in blocked),
        "obligations": obligations,
        "n_planned_operations": planned,
        "n_non_demand_operations": excluded,
        "projection_status": "complete" if planned == expected else "incomplete",
        "projection_note": (
            "every non-structural parsed operation appears in an obligation group"
            if planned == expected
            else f"operation groups account for {planned} of {expected} non-structural parsed operations; "
            "inspect the full inventory before relying on this plan"
        ),
        "counts": {
            key: sum(row["count"] for row in obligations if row["obligation"] == key)
            for key in sorted({row["obligation"] for row in obligations})
        },
        "missing_mapping": ["digest-checked full capture signature -> exact generated capsule and compiler route"],
    }


def _cost(op: str) -> tuple:
    """Rank candidate ops for a family: fewest operands, then smallest default footprint, then name.

    A derived tie-break rather than a preference list, so adding an op cannot silently change which
    capsule a cell gets unless that op is genuinely cheaper.
    """
    from merlin.development.targetgen.corpus.model_builders import BUILDERS

    # Operand ROLES, not tensor count: `linear` carries an optional bias role that `matmul` does not,
    # so the bare contraction is the cheaper way to evidence a contraction cell.
    operands = {
        "matmul": 2,
        "linear": 3,
        "movement": 1,
        "rmsnorm": 1,
        "softmax": 1,
        "layernorm": 1,
        "add": 2,
        "reduce_sum": 1,
        "gelu": 1,
        "silu": 1,
        "bias_add": 2,
    }
    return (operands.get(op, 3), 0 if op in BUILDERS else 1, op)


# ---- framework_operation_partition -------------------------------------------------------------
# Frontend-first registry partitions, distinct from lowered provenance fanout.

_STAGES = ("original", "quantized", "prepared")


def attach_frontend_partitions(universe: dict, source_trace: dict) -> dict:
    """Use verified graph operator identities without replacing them with decompositions."""
    result = copy.deepcopy(universe)
    lowered_keys = (
        "observed_frontend_operators",
        "observed_registered_aten_operators",
        "unobserved_registered_aten_operators",
        "unlisted_frontend_operators",
        "accelerator_candidate_frontend_operators",
        "unresolved_frontend_operators",
        "count_basis",
    )
    result["lowered_provenance"] = {key: result[key] for key in lowered_keys}
    registered = set(result.get("registered_aten_operators") or [])
    catalog_available = result.get("registered_aten_operators") is not None
    stages = {}
    for stage in _STAGES:
        snapshot = source_trace.get("graphs", {}).get(stage) or {}
        observed = snapshot.get("status") == "verified"
        counts = snapshot.get("by_target") if observed else None
        names = set(counts or {})
        version = (snapshot.get("runtime_versions") or {}).get("torch")
        catalog_matches = catalog_available and version is not None and version == result.get("torch_version")
        stages[stage] = {
            "status": "observed" if observed else "unknown",
            "graph_sha256": snapshot.get("sha256"),
            "torch_version": version,
            "static_call_site_count": snapshot.get("call_count"),
            "operator_counts": copy.deepcopy(counts),
            "operator_names": sorted(names) if observed else None,
            "observed_registered_aten_operators": sorted(names & registered) if observed and catalog_matches else None,
            "unobserved_registered_aten_operators": sorted(registered - names)
            if observed and catalog_matches
            else None,
            "unlisted_frontend_operators": sorted(names - registered) if observed and catalog_matches else None,
            "registry_comparison_status": "matched"
            if catalog_matches
            else "unknown"
            if not version or not catalog_available
            else "version_mismatch",
            "count_basis": "exact static call sites in this captured graph, not lowered annotation occurrences",
        }
    result["stages"] = stages
    original = stages["original"]
    result["frontend_denominator_stage"] = "original" if original["status"] == "observed" else "unknown"
    if original["status"] == "observed":
        result["observed_frontend_operators"] = [
            {"operator": name, "static_call_site_count": count}
            for name, count in sorted(original["operator_counts"].items())
        ]
        for key in (
            "observed_registered_aten_operators",
            "unobserved_registered_aten_operators",
            "unlisted_frontend_operators",
        ):
            result[key] = original[key]
        # Source-call acceleration needs all its lowered obligations, not just a
        # similarly named or partially accelerated decomposed operator.
        result["accelerator_candidate_frontend_operators"] = None
        result["unresolved_frontend_operators"] = None
        result["count_basis"] = original["count_basis"]
    return result


def combine_frontend_partitions(universe: dict, applications: dict) -> dict:
    """Aggregate source stages only when every selected application supplies that stage."""
    graphs = {}
    for stage in _STAGES:
        partitions = [application["framework_universe"]["stages"][stage] for application in applications.values()]
        known = bool(partitions) and all(partition["status"] == "observed" for partition in partitions)
        versions = {partition["torch_version"] for partition in partitions}
        counts = Counter()
        if known:
            for partition in partitions:
                counts.update(partition["operator_counts"])
        graphs[stage] = {
            "status": "verified" if known else "unknown",
            "by_target": dict(sorted(counts.items())) if known else None,
            "call_count": sum(counts.values()) if known else None,
            "runtime_versions": {"torch": next(iter(versions))} if len(versions) == 1 else {},
        }
    return attach_frontend_partitions(universe, {"graphs": graphs})


# ---- scope_census ------------------------------------------------------------------------------
# What a capture contains BEYOND one op at a time -- the unit a phase-2 obligation is made of.
#
# The existing census describes a capture one region at a time: family, dtype, extents. That is the right
# unit for a correctness obligation, because a compiler gets one op wrong at a time. It is the wrong unit
# for a performance obligation, because the levers that matter above the tile -- inter-layer scheduling,
# epilogue fusion, the host/accelerator seam, whole-layer residency -- are properties of ADJACENT regions,
# and adjacency is exactly what a per-region descriptor throws away.
#
# Two things are lost, and each has a measured consequence.
#
# **Op configuration.** ``RegionDescriptor`` carries ``(family, dtype, extents)`` and nothing else, so a
# convolution's padding, stride and dilation never reach the requirement. Every convolution capsule in
# this corpus therefore declared the SAME geometry -- zero padding, unit stride, unit dilation, an
# identical window -- four times over, while the builder had accepted all three parameters all along and
# nothing ever passed a non-default. The defect that hid there is a lowering that reuses whatever the
# staging buffer last held instead of reading an out-of-bounds tap as zero: wrong only in the border
# rows, and unreachable by a corpus with no border. A requirement cannot ask what its vocabulary cannot
# express, so the vocabulary is widened here rather than the capsules being hand-written.
#
# **Adjacency.** The optimisation ladder declares rungs for inter-layer scheduling, boundary crossing,
# fusion and global scope. The corpus has ZERO members between them, and it is not an oversight: no
# observation in this repo records that two regions were adjacent, so a requirement for a chain cannot be
# derived at all. ``chains`` recovers it from the use-def graph the capture already carries.
#
# WHAT IS DERIVED AND WHAT IS NOT. Both structures read only what the capture states -- an op's own
# attribute mapping, verbatim, and the producer/consumer edges between region ops. Nothing is normalised
# against a list of known attribute names, because such a list is a per-target fact in disguise: it would
# have to be edited for the next target's spelling, and the edit is what gets forgotten. A configuration
# this module cannot interpret is still RECORDED, so a downstream requirement can ask for coverage of it
# without this module having to understand it.


@dataclass(frozen=True)
class RegionConfig:
    """One region's declared configuration, verbatim.

    ``attrs`` is whatever the op carries. It is deliberately not filtered against a known-key list:
    an unknown key is evidence about the capture, and dropping it is how an axis becomes invisible.
    """

    index: int
    op: str
    family: str | None
    attrs: dict[str, str] = field(default_factory=dict)

    def axis(self, key: str) -> str | None:
        return self.attrs.get(key)


@dataclass(frozen=True)
class Chain:
    """A maximal run of regions where each consumes the previous one's result.

    ``ops`` is the sequence of short op names in program order. ``signature`` is what a requirement
    would be written against -- the shape of the chain rather than its extents -- so two attention
    blocks at different sequence lengths present the same obligation.
    """

    ops: tuple[str, ...]
    indices: tuple[int, ...]
    families: tuple[str | None, ...] = ()

    @property
    def signature(self) -> str:
        """The chain's shape in the vocabulary a requirement is written in.

        An unnamed ``linalg.generic`` carries no information in its op name, and a real capture is
        mostly unnamed generics -- a signature built from op names alone reads "generic -> generic ->
        generic" for every chain in the model, which is a count of nothing. The semantic family is the
        cross-target vocabulary the conformance cells already use, so it is what the signature says
        wherever it resolves.
        """
        parts = []
        for i, op in enumerate(self.ops):
            fam = self.families[i] if i < len(self.families) else None
            parts.append(fam or op)
        return " -> ".join(parts)

    @property
    def length(self) -> int:
        return len(self.ops)


def _region_ops(module) -> list[Any]:
    """The computation-carrying regions, in program order.

    Delegates the predicate to the existing census rather than re-deriving it: two copies of "is this a
    region" drift, and a drifted copy silently changes which ops a chain can span.
    """
    from merlin.development.targetgen.conformance.model_coverage import region_ops

    return list(region_ops(module))


def region_configs(module) -> tuple[RegionConfig, ...]:
    """Every region's declared configuration, in the same order the region census uses."""
    from merlin.compiler.ir import mlir_query as mq
    from merlin.development.targetgen.conformance.model_coverage import _short_op  # the one spelling of "short name"

    out: list[RegionConfig] = []
    for i, op in enumerate(_region_ops(module)):
        short = _short_op(mq.op_name(op))
        attrs: dict[str, str] = {}
        for table in mq._attr_tables(op):
            try:
                items = table.items()
            except AttributeError:
                continue
            for k, v in items:
                key = str(k)
                if key.startswith("prov."):
                    continue  # provenance is recorded elsewhere and is a hint, not configuration
                attrs[key] = str(v)
        out.append(RegionConfig(index=i, op=short, family=_family_of(op, short), attrs=attrs))
    return tuple(out)


def config_axes(configs: Sequence[RegionConfig]) -> dict[str, dict[str, int]]:
    """``op -> attribute -> number of DISTINCT values observed``.

    This is the number that decides whether an axis is worth a capsule. An attribute a capture only
    ever presents at one value is not an axis the corpus is failing to cover; an attribute presenting
    several is one the corpus must span, and today does not.
    """
    seen: dict[str, dict[str, set[str]]] = {}
    for c in configs:
        per_op = seen.setdefault(c.op, {})
        for k, v in c.attrs.items():
            per_op.setdefault(k, set()).add(v)
    return {op: {k: len(vs) for k, vs in sorted(axes.items())} for op, axes in sorted(seen.items())}


def chains(module, *, max_length: int = 8) -> tuple[Chain, ...]:
    """Maximal producer -> consumer runs over the capture's region ops.

    A chain is grown greedily from each region that no other region feeds, following the single
    consumer while there is exactly one. Branching ends a chain: where a value feeds two regions the
    run is no longer a line, and calling it one would invent an ordering the program does not state.

    ``max_length`` bounds the walk so a deep residual stack cannot produce one chain per model. It is a
    reporting bound, not a claim about the program.
    """
    ops = _region_ops(module)
    if not ops:
        return ()
    index_of = {id(op): i for i, op in enumerate(ops)}

    # consumers[i] = the region indices that read a result of region i
    consumers: dict[int, list[int]] = {i: [] for i in range(len(ops))}
    producers: dict[int, list[int]] = {i: [] for i in range(len(ops))}
    for i, op in enumerate(ops):
        for operand in _operands(op):
            owner = _owner(operand)
            j = index_of.get(id(owner)) if owner is not None else None
            if j is not None and j != i:
                consumers[j].append(i)
                producers[i].append(j)

    out: list[Chain] = []
    truncated = False
    for i in range(len(ops)):
        # A region heads a run unless it is the SINGLE continuation of exactly one producer. Starting
        # only at regions with no producer at all missed every run that begins after a branch -- in a
        # residual block that is most of them, so the census saw the stem and none of the arms.
        prods = producers[i]
        if len(prods) == 1 and len(consumers.get(prods[0]) or []) == 1:
            continue
        run = [i]
        cur = i
        while len(run) < max_length:
            nxt = consumers.get(cur) or []
            if len(nxt) != 1:
                break
            cur = nxt[0]
            if cur in run:  # defensive: a cycle is not a chain
                break
            run.append(cur)
        if len(run) == max_length and len(consumers.get(run[-1]) or []) == 1:
            truncated = True  # the bound bound: the real chain continues past what is reported
        if len(run) > 1:
            from merlin.compiler.ir import mlir_query as mq
            from merlin.development.targetgen.conformance.model_coverage import _short_op
            from merlin.target.capabilities import semantic_families as sf

            names = tuple(_short_op(mq.op_name(ops[k])) for k in run)
            out.append(
                Chain(
                    ops=names,
                    indices=tuple(run),
                    families=tuple(_family_of(ops[k], names[j]) for j, k in enumerate(run)),
                )
            )
    _LAST_WALK["truncated"] = truncated
    return tuple(out)


#: Whether the most recent :func:`chains` walk hit its length bound. A census that silently reports a
#: truncated longest-chain as the model's longest chain states a property of the BOUND as a property of
#: the program, so the flag travels with the census rather than being inferred from ``longest``.
_LAST_WALK: dict[str, bool] = {"truncated": False}


def _family_of(op, short: str) -> "str | None":
    """The op's semantic family, resolved the SAME way the region census resolves it.

    Structural first -- an op's own name settles it whatever a tag claims -- then the capture's
    provenance tags for the unnamed ``linalg.generic`` case, which is most of a real capture. Reading
    only the name reported "generic" for 90% of a model, which is a count of nothing.
    """
    from merlin.development.targetgen.conformance.model_coverage import region_family

    return region_family(op, short)


def _operands(op) -> Iterator[Any]:
    try:
        yield from op.operands
    except (AttributeError, TypeError):
        return


def _owner(value) -> Any | None:
    owner = getattr(value, "owner", None)
    if owner is None:
        return None
    # An xDSL result's owner is the op; a block argument's owner is a block, which is not a region op.
    return owner if hasattr(owner, "operands") else None


def chain_census(module, *, max_length: int = 8) -> dict[str, Any]:
    """What a phase-2 requirement would be written against: which chain shapes a capture presents.

    Reports the signature counts rather than the chains themselves, because the obligation is the
    SHAPE of the chain -- two attention blocks at different sequence lengths present one obligation,
    not two -- while the extents belong to the member that witnesses it.
    """
    cs = chains(module, max_length=max_length)
    by_sig: dict[str, int] = {}
    for c in cs:
        by_sig[c.signature] = by_sig.get(c.signature, 0) + 1
    lengths: dict[int, int] = {}
    for c in cs:
        lengths[c.length] = lengths.get(c.length, 0) + 1
    return {
        "n_regions": len(_region_ops(module)),
        "n_chains": len(cs),
        "longest": max((c.length for c in cs), default=0),
        "length_bound": max_length,
        "longest_is_truncated": _LAST_WALK["truncated"],
        "by_length": dict(sorted(lengths.items())),
        "by_signature": dict(sorted(by_sig.items(), key=lambda kv: -kv[1])),
    }


# ---- host_lane_ops -----------------------------------------------------------------------------
# The host operations real captures contain, per ``(family, dtype)``, with their frontend identity.
#
# The host-lane requirement says WHICH ``(family, dtype)`` work a target must leave on the host. A probe
# capsule for that pair has to be built from an operation that work actually consists of: a probe
# written with a stock op of the same family (``gelu`` for an elementwise pair, ``reduce_sum`` for a
# reduction) exercises a program no captured model contains, and it falls outside reviewed host
# declarations that name only the operations the captures carry. So the candidates are read from the
# captures themselves -- each region's model2MLIR op (``prov.op``, the same vocabulary as Merlin's op
# names) and its frontend operator (``prov.aten``) -- and a probe is chosen only from those.
#
# Nothing here names a target or an operation; it counts what the captured regions say they are.


def pair_key(family: str, dtype: str) -> str:
    return f"{family}/{dtype}"


def observed_host_ops(captures: Mapping[str, str | Path], pairs: Iterable[tuple[str, str]]) -> dict[str, list]:
    """``"family/dtype" -> [{op, frontend_op, n_regions}]`` for the requested pairs, most frequent first.

    A region whose provenance names no op is not counted: an anonymous region cannot be reproduced by
    a probe, and guessing its op would invent the very stock program this exists to avoid. An
    unreadable capture contributes nothing (it is reported by the pair census that reads it too).
    """
    from merlin.development.targetgen.conformance import model_coverage as mc
    from merlin.development.targetgen.conformance.requirements import capsule_dtype

    wanted = {(str(family), str(dtype)) for family, dtype in pairs}
    counts: dict[tuple[str, str], Counter] = {pair: Counter() for pair in wanted}
    layouts: dict[tuple[str, str, str, str | None], Counter] = {}
    for _label, path in sorted((captures or {}).items()):
        try:
            module = mc.load_module(path)
            sources = list(mc.region_sources(module))
        except Exception:  # noqa: BLE001 -- an unreadable capture is evidence neither way
            continue
        for family, dtype, op, frontend, layout in _transpose_layouts(module):
            try:
                dtype = capsule_dtype(str(dtype))
            except Exception:  # noqa: BLE001 -- an unmappable token stays as spelled
                dtype = str(dtype)
            if (family, dtype) in counts:
                layouts.setdefault((family, dtype, op, frontend), Counter())[layout] += 1
        for family, dtype, op, frontend in sources:
            if not family or not dtype or not op:
                continue
            try:
                dtype = capsule_dtype(str(dtype))
            except Exception:  # noqa: BLE001 -- an unmappable token stays as spelled
                dtype = str(dtype)
            key = (str(family), str(dtype))
            if key in counts:
                counts[key][(str(op), frontend)] += 1

    def row(pair, op, frontend, n):
        found = layouts.get((*pair, op, frontend))
        out = {"op": op, "frontend_op": frontend, "n_regions": n}
        if found:
            # The exact (input shape, axis order) of each observed region, so a probe reproduces a
            # layout the captures carry: most frequent first, the higher rank first among equals.
            out["layouts"] = [
                {"shape": list(shape), "permutation": list(order), "n_regions": count}
                for (shape, order), count in sorted(found.items(), key=lambda item: (-item[1], -len(item[0][0])))
            ]
        return out

    return {
        pair_key(*pair): [
            row(pair, op, frontend, n)
            for (op, frontend), n in sorted(counter.items(), key=lambda item: (-item[1], item[0][0]))
        ]
        for pair, counter in sorted(counts.items())
    }


def _transpose_layouts(module):
    """``(family, dtype, prov.op, prov.aten, (input shape, permutation))`` of every permutation region."""
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition.groups.group_command import _attr_ints
    from merlin.development.targetgen.conformance import model_coverage as mc

    for op in module.walk():
        if op.name != "linalg.transpose":
            continue
        order = _attr_ints(op, "permutation")
        name = mc._attr_str(op, "prov.op")  # noqa: SLF001 -- the coverage module's provenance reader
        if not order or not name or not op.operands:
            continue
        shape, _dtype = mq.type_shape_dtype(op.operands[0].type)
        if len(shape) != len(order) or any(not isinstance(d, int) or d < 1 for d in shape):
            continue
        yield (
            mc.region_family(op, mc._short_op(op.name)),  # noqa: SLF001
            mc._elem_dtype(op),  # noqa: SLF001
            name,
            mc._attr_str(op, "prov.aten"),  # noqa: SLF001
            (tuple(int(d) for d in shape), tuple(int(a) for a in order)),
        )


def choose(observed: Iterable[Mapping[str, Any]] | None, writable: Iterable[str]) -> dict | None:
    """The most frequent observed host op a writer can express, or ``None`` when none can.

    ``None`` is reported by the caller as an uncovered pair; substituting a writable op that the
    captures do not contain is exactly the failure this module replaces.
    """
    allowed = set(writable)
    for row in observed or ():
        if row.get("op") in allowed:
            return dict(row)
    return None


__all__ = [
    "Chain",
    "RegionConfig",
    "chain_census",
    "chains",
    "config_axes",
    "region_configs",
]
