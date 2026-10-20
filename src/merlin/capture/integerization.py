"""Validate a complete integer/preserved-floating PT2E contraction partition.

This is frontend evidence, not accelerator or host admission. Preserving a
non-FP32 Q/DQ contraction never makes it an integer contraction or grants a
target implementation for that floating operation.
"""

from __future__ import annotations


def contraction_partition(receipt: dict) -> dict[str, int]:
    """Reject missing work, unresolved rewrites, or unaccounted precision refusals."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "m2m.pt2e-integerize.v1":
        raise ValueError("unsupported integerization receipt")

    def count(value: object) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("invalid integerization contraction census")
        return value

    seen = count(receipt.get("quantized_contractions_seen"))
    integerized = count(receipt.get("quantized_contractions_integerized"))
    remaining = count(receipt.get("quantized_contractions_remaining"))
    if seen < 1 or integerized < 1 or seen != integerized + remaining:
        raise ValueError("incomplete integerization contraction census")
    by_kind = receipt.get("quantized_by_kind")
    kinds = {"linear", "conv2d", "matmul", "unsupported"}
    if not isinstance(by_kind, dict) or set(by_kind) != kinds:
        raise ValueError("missing per-kind integerization contraction census")
    totals = [0, 0, 0]
    for kind in kinds:
        row = by_kind[kind]
        if not isinstance(row, dict):
            raise ValueError("invalid per-kind integerization contraction census")
        values = [count(row.get(key)) for key in ("seen", "integerized", "remaining")]
        if values[0] != values[1] + values[2]:
            raise ValueError("inconsistent per-kind integerization contraction census")
        totals = [total + value for total, value in zip(totals, values)]
    if totals != [seen, integerized, remaining] or by_kind["unsupported"]["seen"]:
        raise ValueError("unresolved or inconsistent integerization contraction census")

    refusals = receipt.get("refusals")
    if not isinstance(refusals, list):
        raise ValueError("missing integerization refusal inventory")
    decisions = receipt.get("precision_decisions")
    # Historical complete-integer receipts did not carry a precision ledger.
    # They can never authorize a preserved floating region through this path.
    if decisions is None:
        if remaining or refusals or receipt.get("precision_decision_counts") is not None:
            raise ValueError("remaining contractions lack an exact preserved-precision ledger")
        return {"seen": seen, "integerized": integerized, "preserved": 0}
    if not isinstance(decisions, list) or len(decisions) != seen:
        raise ValueError("incomplete integerization precision ledger")
    counts = {"integerized_i32": 0, "preserve_float_qdq": 0, "unresolved": 0}
    identities: set[tuple[str, str]] = set()
    preserved: set[tuple[str, str]] = set()
    decision_kinds = {kind: [0, 0, 0] for kind in kinds}
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("invalid integerization precision decision")
        kind, node, decision = row.get("kind"), row.get("node"), row.get("decision")
        if (
            not isinstance(kind, str)
            or kind not in kinds
            or not isinstance(node, str)
            or not node
            or not isinstance(decision, str)
            or decision not in counts
        ):
            raise ValueError("invalid integerization precision decision")
        identity = (kind, node)
        if identity in identities:
            raise ValueError("duplicate integerization precision decision")
        identities.add(identity)
        counts[decision] += 1
        decision_kinds[kind][0] += 1
        if decision == "integerized_i32":
            decision_kinds[kind][1] += 1
        elif decision == "preserve_float_qdq":
            if (
                not isinstance(row.get("source_dtype"), str)
                or row["source_dtype"] not in {"torch.bfloat16", "torch.float16"}
                or row.get("required_numeric_semantics") != "dequantize-before-floating-contraction"
            ):
                raise ValueError("preserved Q/DQ lacks exact non-FP32 floating semantics")
            preserved.add(identity)
            decision_kinds[kind][2] += 1
    declared_counts = receipt.get("precision_decision_counts")
    if not isinstance(declared_counts, dict) or set(declared_counts) != set(counts):
        raise ValueError("missing integerization precision partition counts")
    declared_counts = {key: count(value) for key, value in declared_counts.items()}
    if (
        counts != {"integerized_i32": integerized, "preserve_float_qdq": remaining, "unresolved": 0}
        or declared_counts != counts
        or any(
            decision_kinds[kind] != [by_kind[kind][key] for key in ("seen", "integerized", "remaining")]
            for kind in kinds
        )
    ):
        raise ValueError("unresolved or inconsistent integerization precision partition")
    refused: set[tuple[str, str]] = set()
    for row in refusals:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in ("kind", "node")):
            raise ValueError("invalid integerization refusal")
        identity = (row.get("kind"), row.get("node"))
        if (
            identity not in preserved
            or identity in refused
            or not isinstance(row.get("reason"), str)
            or not row["reason"]
        ):
            raise ValueError("integerization refusal is not an exact preserved-precision decision")
        refused.add(identity)
    if refused != preserved:
        raise ValueError("preserved-precision decisions and refusal inventory differ")
    return {"seen": seen, "integerized": integerized, "preserved": remaining}


_DYNAMIC_TRANSFORMS = frozenset({"static_attention", "dynamic_contraction"})


def _i8_i8_i32(row: dict) -> bool:
    """A traced MLIR op whose first two operands are i8 tensors and whose one result is an i32 tensor."""

    def element(text: object) -> str | None:
        if not isinstance(text, str) or not text.startswith("tensor<") or not text.endswith(">"):
            return None
        return text[len("tensor<") : -1].rsplit("x", 1)[-1]

    operands, results = row.get("operand_types"), row.get("result_types")
    return (
        isinstance(operands, list)
        and len(operands) >= 2
        and [element(t) for t in operands[:2]] == ["i8", "i8"]
        and isinstance(results, list)
        and [element(t) for t in results] == ["i32"]
    )


def traced_integer_owners(trace: dict, program_sha256: str) -> tuple[int, dict[tuple[str, ...], int]]:
    """Owners of every emitted i8 x i8 -> i32 contraction, from the stage's byte-bound frontend trace.

    Each such ``linalg.generic`` must come from exactly one prepared-graph ``aten._int_mm`` node, every
    such node must be emitted exactly once, and the node's transformation names its owner: the PT2E
    integerization (a weight contraction) or a recipe rewrite (a dynamic site, keyed by the original
    nodes of its lineage). Returns (weight contractions, dynamic contractions per lineage).
    """
    from collections import Counter

    mlir = trace.get("mlir") if isinstance(trace, dict) else None
    graphs = trace.get("graphs") if isinstance(trace, dict) else None
    prepared = (graphs or {}).get("prepared") if isinstance(graphs, dict) else None
    if (
        not isinstance(mlir, dict)
        or mlir.get("sha256") != program_sha256
        or not isinstance(mlir.get("operations"), list)
        or not isinstance(prepared, dict)
        or not isinstance(prepared.get("nodes"), list)
    ):
        raise ValueError("dynamic contraction census trace is not bound to the stage program")
    int_mm = {
        node["id"]: node
        for node in prepared["nodes"]
        if isinstance(node, dict) and node.get("target") == "aten._int_mm.default" and isinstance(node.get("id"), str)
    }
    selected_nodes = [
        node for node in prepared["nodes"] if isinstance(node, dict) and node.get("target") == "aten._int_mm.default"
    ]
    if len(selected_nodes) != len(int_mm):
        raise ValueError("traced integer contractions have duplicate or missing node identities")
    seen: Counter = Counter()
    weight, dynamic = 0, Counter()
    for row in mlir["operations"]:
        if not isinstance(row, dict) or row.get("operation") != "linalg.generic" or not _i8_i8_i32(row):
            continue
        sources = row.get("source_node_ids")
        if (
            not isinstance(sources, list)
            or len(sources) != 1
            or not isinstance(sources[0], str)
            or sources[0] not in int_mm
        ):
            raise ValueError("an emitted integer contraction has no unique traced owner")
        node = int_mm[sources[0]]
        seen[sources[0]] += 1
        transform = node.get("transformation")
        if transform == "pt2e_integerization":
            weight += 1
        elif transform in _DYNAMIC_TRANSFORMS:
            origins = node.get("origin_node_ids")
            if not isinstance(origins, list) or any(not isinstance(v, str) for v in origins):
                raise ValueError("a dynamic integer contraction has no original lineage")
            key = tuple(sorted(v for v in origins if v.startswith("g:original:")))
            if not key:
                raise ValueError("a dynamic integer contraction has no original lineage")
            dynamic[key] += 1
        else:
            raise ValueError("an emitted integer contraction has an unknown traced transformation")
    if set(seen) != set(int_mm) or any(count != 1 for count in seen.values()):
        raise ValueError("traced integer contractions are not each emitted exactly once")
    return weight, dict(dynamic)


def _report_only_quality(selection: dict) -> dict:
    """Project an explicit validated numerical selection without publishing recipe details."""
    from merlin.common.digest import is_sha256
    from merlin.targetgen.quant_recipe import digest
    from merlin.targetgen.quant_recipe_policy import validate

    identities = ("source_recipe_sha256", "source_methodology_sha256", "source_output_path_sha256")
    if (
        not isinstance(selection, dict)
        or set(selection) != {"schema", "policy", *identities}
        or selection.get("schema") != "numerical_recipe_selection_v1"
        or not all(is_sha256(selection.get(key)) for key in identities)
    ):
        raise ValueError("report-only dynamic quality requires an exact numerical recipe selection")
    policy = validate(selection["policy"])
    return {
        "schema": "merlin.dynamic_float_quality.v1",
        "mode": "report_only",
        "qualified": False,
        "selection_sha256": digest(selection),
        "policy_sha256": digest(policy),
        **{key: selection[key] for key in identities},
    }


def dynamic_integerization_count(evidence: dict, *, emitted: int, exported: int, program_sha256: str) -> int:
    """Recheck an answer-free observation of recipe-site and traced-owner accounting.

    The caller binds this observation to materialized capture metadata and trace bytes.
    This is no independent source-execution attestation or target numerical verdict.
    """
    import math
    from collections import Counter

    from merlin.common.digest import is_sha256
    from merlin.targetgen.quant_recipe import site_agreement_of

    if not isinstance(evidence, dict) or evidence.get("schema") not in (
        "merlin.dynamic_integerization.v1",
        "merlin.dynamic_integerization.v2",
    ):
        raise ValueError("invalid dynamic integerization observation")
    if not is_sha256(evidence.get("recipe_sha256")):
        raise ValueError("dynamic integerization observation lacks a recipe identity")
    trace_identity = evidence.get("trace")
    if (
        not isinstance(trace_identity, dict)
        or not is_sha256(trace_identity.get("sha256"))
        or type(trace_identity.get("bytes")) is not int
        or trace_identity["bytes"] <= 0
        or evidence.get("program_sha256") != program_sha256
        or not is_sha256(program_sha256)
    ):
        raise ValueError("dynamic contraction census trace is not bound to the stage program")
    bar = site_agreement_of({"dynamic": {"site_agreement": evidence.get("site_agreement")}})
    if evidence["schema"] == "merlin.dynamic_integerization.v1":
        if bar is None:
            raise ValueError("the recipe declares no per-site agreement bar for its dynamic sites")
    else:
        quality = evidence.get("float_quality")
        identities = {
            "selection_sha256",
            "policy_sha256",
            "source_recipe_sha256",
            "source_methodology_sha256",
            "source_output_path_sha256",
        }
        if (
            bar is not None
            or not isinstance(quality, dict)
            or set(quality) != {"schema", "mode", "qualified", *identities}
            or quality.get("schema") != "merlin.dynamic_float_quality.v1"
            or quality.get("mode") != "report_only"
            or quality.get("qualified") is not False
            or not all(is_sha256(quality.get(key)) for key in identities)
        ):
            raise ValueError("invalid explicit report-only dynamic quality observation")
    families, samples = evidence.get("families"), evidence.get("samples")
    if not isinstance(families, list) or not families or any(not isinstance(f, str) for f in families):
        raise ValueError("invalid selected dynamic contraction families")
    if any(type(value) is not int or value <= 0 for value in (emitted, exported)):
        raise ValueError("invalid integer contraction counts")
    weight, roster = evidence.get("weight_contractions"), evidence.get("dynamic_owners")
    if type(weight) is not int or weight < 0 or not isinstance(roster, list):
        raise ValueError("invalid traced integer contraction owners")
    traced = {}
    for row in roster:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("lineage"), list)
            or not row["lineage"]
            or any(not isinstance(v, str) or not v.startswith("g:original:") for v in row["lineage"])
            or type(row.get("calls")) is not int
            or row["calls"] <= 0
        ):
            raise ValueError("invalid traced dynamic contraction owner")
        key = tuple(sorted(row["lineage"]))
        if key in traced:
            raise ValueError("duplicate traced dynamic contraction owner")
        traced[key] = row["calls"]
    block = evidence.get("contractions")

    def need(condition: bool, why: str) -> None:
        if not condition:
            raise ValueError(f"capture dynamic contraction census refused: {why}")

    need(isinstance(block, dict), "invalid dynamic block")
    sites, calls = block.get("sites"), block.get("device_calls")
    need(
        isinstance(sites, list)
        and type(block.get("rewritten")) is int
        and block.get("rewritten") == len(sites)
        and block.get("kept_in_float") == []
        and type(calls) is int
        and calls >= 0,
        "every dynamic site must be rewritten and none kept in floating point",
    )
    need(bool(sites) or (calls == 0 and not traced), "empty dynamic census differs from the traced dynamic owners")
    need(type(samples) is int and samples >= 1, "the recipe agreement records no samples")
    owned: Counter = Counter()
    for site in sites:
        check = site.get("verification") if isinstance(site, dict) else None
        need(isinstance(site, dict), "invalid dynamic site")
        shapes, lineage = site.get("shapes"), site.get("lineage")
        need(
            isinstance(site, dict)
            and site.get("status") == "rewritten"
            and site.get("family") in families
            and type(site.get("device_calls")) is int
            and site["device_calls"] > 0,
            "a dynamic site is not a rewritten contraction of a selected family",
        )
        need(
            isinstance(shapes, list)
            and bool(shapes)
            and isinstance(shapes[0], list)
            and bool(shapes[0])
            and type(shapes[0][-1]) is int
            and 0 < shapes[0][-1]
            and shapes[0][-1] * 128 * 128 <= (1 << 31) - 1,
            "a dynamic site's reduction bound is not proved",
        )
        metrics = ("cosine_min", "cosine_mean", "max_abs_error", "rel_rmse_max")
        need(
            isinstance(check, dict)
            and check.get("schema") == "m2m.dynamic_site_verification.v1"
            and type(check.get("samples")) is int
            and check.get("samples") == samples
            and check.get("finite") is True
            and all(type(check.get(key)) is float and math.isfinite(check[key]) for key in metrics),
            "a dynamic site lacks a finite verification on the recipe agreement samples",
        )
        need(
            check["max_abs_error"] >= 0.0
            and check["rel_rmse_max"] >= 0.0
            and check["cosine_min"] <= check["cosine_mean"],
            "a dynamic site's verification metrics are inconsistent",
        )
        if bar is not None:
            need(
                check["cosine_min"] >= bar["cosine_min"] and check["rel_rmse_max"] <= bar["rel_rmse_max"],
                "a dynamic site is outside the recipe's declared agreement bar",
            )
        key = tuple(sorted(v for v in lineage or () if isinstance(v, str) and v.startswith("g:original:")))
        need(isinstance(lineage, list) and bool(key), "a dynamic site has no original lineage")
        owned[key] += site["device_calls"]
    need(sum(owned.values()) == calls, "dynamic device calls do not sum to the census total")
    need(emitted + calls == exported, "exported integer contractions are not the emitted plus dynamic ones")
    need(weight == emitted and traced == dict(owned), "traced integer contractions differ from their owners")
    return calls


def dynamic_integerization_projection(
    meta: dict, stats: dict, *, trace: dict, trace_identity: dict, program_sha256: str, emitted: int, exported: int
) -> dict:
    """Validate raw recipe/trace identity and project only bounded census evidence.

    Do not publish the private recipe, graph, workload names, source paths or tensor values.
    The receipt owner still has to bind all three artifacts to their exact bytes.
    """
    from merlin.targetgen.quant_recipe import DYNAMIC, digest, site_agreement_of

    recipe = meta.get("recipe")
    if not (isinstance(recipe, dict) and recipe.get("recipe_sha256") == meta.get("recipe_sha256") == digest(recipe)):
        raise ValueError("capture dynamic contraction census refused: the recipe snapshot does not match its digest")
    bar = site_agreement_of(recipe)
    selection = recipe.get("numerical_selection")
    quality = None if selection is None else _report_only_quality(selection)
    if quality is None and bar is None:
        raise ValueError("the recipe declares no per-site agreement bar for its dynamic sites")
    if quality is not None and (bar is not None or recipe.get("software_numerical_engine") != "integer_reference"):
        raise ValueError("report-only dynamic quality requires an unambiguous independent integer-reference recipe")
    block = stats.get("dynamic_contractions")
    if not isinstance(block, dict) or not isinstance(block.get("sites"), list):
        raise ValueError("capture dynamic contraction census refused: invalid dynamic block")
    if not block["sites"] and any(
        type(block.get(key)) is not int or block[key] != 0 for key in ("host_elements", "int8_macs")
    ):
        raise ValueError("capture dynamic contraction census refused: empty domain has invalid work counters")
    sites = []
    for site in block["sites"]:
        if not isinstance(site, dict):
            raise ValueError("capture dynamic contraction census refused: invalid dynamic site")
        # In particular, omit the site node's possibly workload-specific name.
        projected = {key: site.get(key) for key in ("family", "status", "device_calls", "shapes", "lineage")}
        lineage = site.get("lineage")
        projected["lineage"] = (
            [value for value in lineage if isinstance(value, str) and value.startswith("g:original:")]
            if isinstance(lineage, list)
            else None
        )
        verification = site.get("verification")
        projected["verification"] = (
            {
                key: verification.get(key)
                for key in ("schema", "samples", "finite", "cosine_min", "cosine_mean", "max_abs_error", "rel_rmse_max")
            }
            if isinstance(verification, dict)
            else None
        )
        sites.append(projected)
    weight, dynamic = traced_integer_owners(trace, program_sha256)
    agreement = meta.get("recipe_agreement")
    evidence = {
        "schema": "merlin.dynamic_integerization.v1" if quality is None else "merlin.dynamic_integerization.v2",
        **({"float_quality": quality} if quality is not None else {}),
        "recipe_sha256": meta["recipe_sha256"],
        "families": (recipe.get(DYNAMIC) or {}).get("families"),
        "site_agreement": bar,
        "samples": agreement.get("samples") if isinstance(agreement, dict) else None,
        "program_sha256": program_sha256,
        "trace": {key: trace_identity.get(key) for key in ("sha256", "bytes")},
        "weight_contractions": weight,
        "dynamic_owners": [{"lineage": list(key), "calls": calls} for key, calls in sorted(dynamic.items())],
        "contractions": {
            **{key: block.get(key) for key in ("rewritten", "kept_in_float", "device_calls")},
            "sites": sites,
        },
    }
    dynamic_integerization_count(evidence, emitted=emitted, exported=exported, program_sha256=program_sha256)
    return evidence
