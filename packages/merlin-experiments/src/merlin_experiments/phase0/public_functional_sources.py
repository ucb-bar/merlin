"""Bounded functional source cuts from independent public graph witnesses.

Selection reads source graph semantics, selected facts and numerical contracts.
Nonunit extent magnitudes, weights, expected outputs, compiler products and
timings do not choose probe sizes. Fresh capture and admission own all results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from merlin.common import strict_json
from merlin.targetgen.frontend_trace import loads_frontend_trace

from . import public_boundaries as B
from . import public_boundary_semantics as S
from . import public_typed_semantics as T


def dimensions(region, quantum, variant):
    if type(quantum) is not int or quantum < 1 or type(variant) is not int or variant not in (0, 1):
        raise ValueError("functional source needs an exact selected extent quantum and variant")
    return {
        name: quantum if variant == 0 else max(1, quantum + (1 if i % 2 else -1))
        for i, name in enumerate(region["dimensions"])
    }


def facets(program):
    """Report proved logical source structure without physical reuse/attention claims."""
    ancestors, projections, result = {}, {}, {"residual_join": [], "mean_reduction": [], "shared_input_projections": []}
    values = {row["name"]: row for row in [*program["inputs"], *program["nodes"]]}
    for node in program["nodes"]:
        # Program operands use constructor-owned names, not frontend value IDs.
        def names(value):
            if isinstance(value, dict) and set(value) == {"value"}:
                yield value["value"]
            elif isinstance(value, list):
                for item in value:
                    yield from names(item)

        refs = list(names(node["args"]))
        upstream = set().union(*(ancestors.get(ref, set()) for ref in refs))
        if node["target"] in T.CONTRACTIONS:
            upstream.add(node["name"])
            projections.setdefault(refs[0], []).append(node["name"])
        if (
            node["target"] == "aten.add.Tensor"
            and len(refs) == 2
            and refs[0] != refs[1]
            and upstream
            and ancestors.get(refs[0], set()) != ancestors.get(refs[1], set())
            and all(values[ref]["shape"] == node["shape"] for ref in refs)
        ):
            result["residual_join"].append(node["name"])
        if node["target"] == "aten.mean.dim":
            result["mean_reduction"].append(node["name"])
        ancestors[node["name"]] = upstream
    result["shared_input_projections"] = [names for names in projections.values() if len(names) > 1]
    return result


def basis(member):
    return {
        "capture_sha256": member["origin"]["capture_sha256"],
        "receipt_sha256": member["origin"]["receipt_sha256"],
        "operations": list(dict.fromkeys(n["target"] for n in member["program"]["nodes"])),
        "constructed_program_sha256": B.digest(member["program"]),
        "raw_facts_sha256": member["raw_facts_sha256"],
        "numerical_semantics_sha256": member["numerical_semantics_sha256"],
    }


def verify_member(member, region, entry):
    policy = B.validate(member["selection"])
    if policy["schema"] != B.SCHEMA_V3 or member["constructor_node_budget"] != policy["max_nodes"]:
        raise ValueError("functional source changed its finite construction policy")
    expected = S.instantiate(region, dimensions(region, member["extent_quantum"], member["variant"]))
    if (
        expected != member["program"]
        or member["logical_facets"] != facets(expected)
        or member["source_execution_basis"] != basis(member)
        or member["stages"] != {"generated": True, "evaluated": False, "admitted": False, "executed": False}
        or entry.get("cat") != "_synthetic"
        or entry.get("label") != "public"
        or "performance" in entry
    ):
        raise ValueError("functional source changed its construction, purpose or logical witness")
    if (
        len(B.render(expected).encode()) > policy["max_source_bytes"]
        or S.logical_elements(expected) > policy["max_logical_elements"]
        or any(d > policy["max_extent"] for d in member["program"]["dimension_values"].values())
    ):
        raise ValueError("functional source exceeds selected complete-source budgets")


def derive(original_entries, outputs, *, evidence, policy):
    policy = B.validate(policy)
    numerical = evidence.software_spec.get("numerical_semantics")
    if not isinstance(numerical, dict) or not numerical:
        raise ValueError("functional source selection requires the selected numerical contract")
    numeric_sha = B.digest(numerical)
    quanta = {
        row.get("rows")
        for row in (evidence.refreshed_facts.get("facts") or {}).get("arrays", [])
        if isinstance(row, dict) and type(row.get("rows")) is int and row["rows"] > 0
    }
    quantum = next(iter(quanta)) if len(quanta) == 1 else None
    entries, products, missing, originals = [], {}, [], []
    requested = 0

    def unknown(origin, reason, **detail):
        missing.append(
            {
                "origin": origin,
                "mandatory": True,
                "state": "unavailable",
                "status": "UNKNOWN",
                "reason": reason,
                **detail,
            }
        )

    for original in original_entries:
        capture = original["materialized_capture"]
        if capture.get("workload_role") != "performance_scale":
            raise ValueError("functional derivation requires the independently selected public source roster")
        prefix = Path(capture["path"]).parent
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("public original source must remain in its selected bundle")
        trace_path, receipt_path = (str(prefix / name) for name in ("frontend-trace.json", "capture_receipt.json"))
        raw, receipt_raw = outputs[trace_path], outputs[receipt_path]
        receipt = strict_json.loads(receipt_raw)
        if (
            hashlib.sha256(receipt_raw).hexdigest() != capture["receipt_sha256"]
            or hashlib.sha256(raw).hexdigest() != receipt["artifacts"]["frontend-trace.json"]["sha256"]
        ):
            raise ValueError("functional original source differs from selected receipt/trace bytes")
        trace = loads_frontend_trace(raw)
        origin = {
            k: capture[k]
            for k in (
                "capture_sha256",
                "receipt_sha256",
                "loader_sha256",
                "full_inventory_sha256",
                "workload_id",
                "workload_role",
            )
        }
        origin["graph_sha256"] = trace["graphs"]["original"]["sha256"]
        context = {"path": trace_path, "sha256": hashlib.sha256(raw).hexdigest()}
        originals.append({"origin": origin, "source_context": context})
        products[trace_path] = raw
        regions, gaps = S.cuts(trace, max_nodes=policy["max_nodes"], typed=True, shared_inputs=True)
        for gap in gaps:
            unknown(origin, gap["reason"], source_gap=gap)
        if not regions:
            unknown(origin, "no exact supported public connected source is constructible")
        if quantum is None:
            unknown(origin, "selected facts do not establish one positive extent quantum")
            continue
        for region in regions:
            for variant in range(2 if region["dimensions"] else 1):
                requested += 1
                sizes = dimensions(region, quantum, variant)
                if requested > policy["max_members"] or any(d > policy["max_extent"] for d in sizes.values()):
                    unknown(origin, "functional member exceeds finite metadata budget", dimensions=sizes)
                    continue
                program = S.instantiate(region, sizes)
                if S.logical_elements(program) > policy["max_logical_elements"]:
                    unknown(
                        origin, "complete logical input/intermediate/output volume exceeds budget", dimensions=sizes
                    )
                    continue
                source = B.render(program).encode()
                if len(source) > policy["max_source_bytes"]:
                    unknown(origin, "complete source exceeds byte budget", dimensions=sizes)
                    continue
                member = {
                    "schema": B.SCHEMA_V3,
                    "selection": policy,
                    "origin": origin,
                    "source_context": context,
                    "constructor_node_budget": policy["max_nodes"],
                    "program": program,
                    "extent_quantum": quantum,
                    "variant": variant,
                    "raw_facts_sha256": evidence.raw_facts_sha256,
                    "numerical_semantics_sha256": numeric_sha,
                    "logical_facets": facets(program),
                    "loader_sha256": hashlib.sha256(source).hexdigest(),
                    "stages": {"generated": True, "evaluated": False, "admitted": False, "executed": False},
                }
                member["source_execution_basis"] = basis(member)
                name = "PF_" + B.digest(member)[:20]
                loader = "public-boundaries/" + name + "/loader.py"
                entry = {
                    "name": name,
                    "kind": "model",
                    "op": "model",
                    "cat": "_synthetic",
                    "label": "public",
                    "source_role": "derived_sweep",
                    "source_reference": "receipt-bound public connected source",
                    "operand_dtype": next(
                        (
                            T.DTYPES[row["dtype"]]
                            for row in program["inputs"]
                            if row["dtype"].startswith(("float", "bfloat"))
                        ),
                        "fp32",
                    ),
                    "loader": loader,
                    "public_boundary": member,
                }
                verify_member(member, region, entry)
                entries.append(entry)
                products[loader] = source
    # These exact semantic/state obligations are not covered by batched dot,
    # shared SSA inputs or finite source numerics; never manufacture attention.
    unknown(
        None,
        "complete attention mask/probability and carried KV state contracts are not established",
        domain="attention_state",
        source_class="not_admitted",
    )
    unknown(
        None,
        "dynamic cut leaves do not establish frozen-weight native readout or fusion admission",
        domain="native_integer_graph",
        source_class="not_admitted",
    )
    return (
        entries,
        products,
        {
            "schema": B.SCHEMA_V3,
            "selection": policy,
            "requested": requested,
            "constructed": len(entries),
            "missing": missing,
            "members_sha256": B.digest(entries),
            "raw_facts_sha256": evidence.raw_facts_sha256,
            "numerical_semantics_sha256": numeric_sha,
            "original_sources": originals,
            "selection_basis": "public original typed graph semantics plus selected facts/numerical contract only",
            "stages": {"generated": len(entries), "evaluated": 0, "admitted": 0, "executed": 0},
            "functional_completeness": "UNKNOWN",
            "physical_reuse": "UNKNOWN",
            "timing": "UNKNOWN",
        },
    )
