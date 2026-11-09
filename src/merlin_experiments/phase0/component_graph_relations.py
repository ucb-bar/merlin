"""Private full-output metamorphic witnesses for independent graph families.

No relation is established from labels or selected final outputs. Reopen each
ordinary source and independent golden, replay its complete original oracle,
then compare both representations on exactly the same admitted leaf semantics.
These witnesses describe tensor semantics, not physical buffer reuse or timing.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from merlin.targetgen import golden_store

from .component_generation import digest
from .component_graph_variants import PARAMETERS, VARIANTS, parameters, program, validate
from .component_graph_variants import SCHEMA as FAMILY_SCHEMA

SCHEMA = "merlin.component_graph_relations.v1"


def _families(report):
    return {row["id"]: row for row in report["declaration"]["obligations"] if "graph_family" in row["base"]}


def _groups(row, declaration):
    family = validate(declaration["base"]["graph_family"])
    groups = {}
    for member in row["members"]:
        variant = member.get("graph_variant")
        if (
            not isinstance(variant, dict)
            or set(variant) != {"schema", "family_sha256", "representation", "parameters"}
            or variant["schema"] != FAMILY_SCHEMA
            or variant["family_sha256"] != digest(family)
            or variant["representation"] not in VARIANTS
            or set(variant["parameters"]) != set(PARAMETERS)
        ):
            raise ValueError("graph relation lost its frozen family/representation identity")
        parameters(variant["parameters"])
        point = member["point_sha256"]
        groups.setdefault(point, []).append(member)
    result = []
    for point, members in groups.items():
        by_variant = {member["graph_variant"]["representation"]: member for member in members}
        if len(members) != len(VARIANTS) or set(by_variant) != set(VARIANTS):
            raise ValueError("graph relation requires both exact requested representations")
        ordered = [by_variant[name] for name in VARIANTS]
        if len({digest(member["graph_variant"]["parameters"]) for member in ordered}) != 1:
            raise ValueError("graph relation representations disagree on selected source parameters")
        result.append((point, ordered))
    return result


def _observe(root, row, members):
    from .component_execution_budget import source_for_capsule
    from .component_numerics import evaluate

    results, inputs, outputs = [], [], []
    for member in members:
        if member["state"] != "generated":
            raise ValueError("graph relation has an unavailable requested representation")
        directory = Path(root) / member["member"]
        capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        stamp = capsule.get("component_coverage") or {}
        variant = member["graph_variant"]
        if (
            stamp.get("graph_variant") != variant
            or stamp.get("point_sha256") != member["point_sha256"]
            or stamp.get("obligation") != row["id"]
            or stamp.get("cohort") != row["cohort"]
        ):
            raise ValueError("graph relation written source lost its frozen requested membership")
        source = source_for_capsule(capsule)
        values = variant["parameters"]
        stages = 2 * values["fanout"] + (3 if variant["representation"] == "logical_epochs" else 2)
        typed = source["program"]
        if (
            len(typed["nodes"]) != 1 + values["depth"] * stages
            or len(typed["outputs"]) != 3 * values["depth"] + 1
            or [item["shape"] for item in typed["inputs"]] != [[values["M"], values["K"]], [values["K"], values["N"]]]
        ):
            # Actual source costs were already admitted. Claimed parameters
            # must match its finite topology before constructing a replay;
            # hostile parameters cannot trigger an unbounded expected graph.
            raise ValueError("graph relation parameters disagree with actual bounded source topology")
        if capsule["operation"]["attributes"]["program"] != program(variant["parameters"], variant["representation"]):
            raise ValueError("graph relation written source differs from independently constructed family")
        leaf_source = {key: source[key] for key in ("input_palette", "stimulus_range")}
        inputs.append(digest({**leaf_source, "inputs": source["program"]["inputs"]}))
        golden = golden_store.load_golden(directory)
        original = evaluate(capsule)
        if digest(golden.get("outputs")) != digest(original):
            raise ValueError("graph relation golden differs from original complete independent oracle")
        outputs.append(digest(original))
        results.append(
            {
                "name": member["name"],
                "member": member["member"],
                "representation": variant["representation"],
                "source_sha256": digest(source),
                "full_outputs_sha256": digest(original),
                "output_roster": sorted(original),
                "generated_effects": source["program"]["effects"],
            }
        )
    if len(set(inputs)) != 1 or len(set(outputs)) != 1:
        raise ValueError("graph relation representations do not have identical leaf semantics and full outputs")
    return {
        "state": "established",
        "input_source_sha256": inputs[0],
        "members": results,
        "scope": "complete independent tensor outputs; physical aliases, lifetime, completion and timing unverified",
    }


def witness(report, *, root):
    """Join the exact requested roster; a missing member keeps its denominator."""
    families = _families(report)
    if not families:
        return None
    rows, seen = [], set()
    for row in report["obligations"]:
        if row["id"] not in families:
            if any("graph_variant" in member for member in row["members"]):
                raise ValueError("graph representation has no selected independent family")
            continue
        seen.add(row["id"])
        pairs = []
        try:
            groups = _groups(row, families[row["id"]])
            for point, members in groups:
                pair = {"point_sha256": point, "requested_members": [member["requested_member"] for member in members]}
                try:
                    pair.update(_observe(root, row, members))
                except (OSError, ValueError) as exc:
                    pair.update(state="unavailable", reason=str(exc))
                pairs.append(pair)
            state = "established" if pairs and all(pair["state"] == "established" for pair in pairs) else "unavailable"
            rows.append({"obligation": row["id"], "state": state, "pairs": pairs})
        except (OSError, ValueError) as exc:
            rows.append({"obligation": row["id"], "state": "unavailable", "pairs": [], "reason": str(exc)})
    if seen != set(families):
        raise ValueError("graph relation lost a selected independent family obligation")
    return {"schema": SCHEMA, "rows": rows}


def finalize(report, *, root):
    ledger = witness(report, root=root)
    if ledger is None:
        return
    report["graph_relations"] = ledger
    unavailable = {row["obligation"] for row in ledger["rows"] if row["state"] != "established"}
    for row in report["obligations"]:
        if row["id"] in unavailable:
            row["state"] = "unavailable"
            row["errors"].append("original complete graph relation is unavailable")


def verify(report, *, root):
    families = _families(report)
    if bool(families) != ("graph_relations" in report):
        raise ValueError("graph relation evidence version changed")
    if not families:
        return
    actual = witness(report, root=root)
    if actual != report["graph_relations"] or any(row["state"] != "established" for row in actual["rows"]):
        raise ValueError("graph relation actual source/complete outputs or roster changed")
    # Budget replay opens the whole generation roster. A re-signed coverage
    # report cannot silently drop a still-issued family member from its pairs.
    expected = {member["name"] for row in report["obligations"] if row["id"] in families for member in row["members"]}
    observed = set()
    for decision in report["execution_admission"]["decisions"]:
        capsule = yaml.safe_load((Path(root) / decision["requested_member"] / "capsule.yaml").read_bytes())
        if "graph_variant" in (capsule.get("component_coverage") or {}):
            observed.add(decision["name"])
    if observed != expected:
        raise ValueError("graph relation lost complete issued family membership")
