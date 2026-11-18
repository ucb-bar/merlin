"""Bounded observation cards and diversity features for explicitly selected references."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from merlin.analysis.mining.references.cca_extraction import observations_only
from merlin.analysis.mining.references.database import digest, safe_member, write_json
from merlin.analysis.mining.references.database import verify as verify_database

# ---- reference_subset_cards --------------------------------------------------------------------


def known_leaves(value, prefix=""):
    """Keep false and zero; null is absence of a reconstructed fact."""
    if isinstance(value, dict):
        result = {}
        for key, item in sorted(value.items()):
            if key not in ("provenance", "instructions"):
                result.update(known_leaves(item, prefix + "." + key if prefix else key))
        return result
    return {prefix: value} if value is not None else {}


def canonical(analysis):
    # Measure semantic signatures separately from witnessed instruction records;
    # evidence IDs must never inflate reported reconstruction diversity.
    return {key: value for key, value in analysis["cca"].items() if key not in ("provenance", "instructions")}


def leaf_paths(value, prefix=""):
    """Every canonical leaf path, known or null, so knownness has a denominator."""
    if isinstance(value, dict):
        paths = set()
        for key, item in value.items():
            if key not in ("provenance", "instructions"):
                paths |= leaf_paths(item, prefix + "." + key if prefix else key)
        return paths
    return {prefix}


def axis_knownness(canonicals):
    """Per-axis count of occurrences that know the axis, beside signature counts.

    A signature over mostly-null axes measures facet sparsity, not identity:
    two occurrences with every axis unknown share one signature trivially.
    """
    canonicals = list(canonicals)
    axes = sorted(set().union(*(leaf_paths(item) for item in canonicals))) if canonicals else []
    known = [set(known_leaves(item)) for item in canonicals]
    per_axis = {axis: sum(axis in row for row in known) for axis in axes}
    counts = sorted(len(row) for row in known)
    return dict(
        occurrences=len(canonicals),
        axes=len(axes),
        known_occurrences_by_axis=per_axis,
        never_known_axes=sum(not n for n in per_axis.values()),
        known_axes_per_occurrence=dict(min=counts[0], median=counts[len(counts) // 2], max=counts[-1])
        if counts
        else None,
        interpretation="signature counts are bounded by these known axes; null is absence of a fact",
    )


def bucket(value):
    return min(max(0, int(value)).bit_length(), 16)


def value_bin(value):
    if type(value) is bool:
        return "bool:" + str(value)
    if type(value) is int:
        return "integer:" + str(-bucket(-value) if value < 0 else bucket(value))
    if isinstance(value, str):
        return "enum:" + value
    if isinstance(value, float) and math.isfinite(value):
        return "float"
    return None


def features(analysis):
    """No candidate, timing, model-gap labels or target-specific field-name rules."""
    projection = analysis["mechanism_projection"]
    result = {"role:" + key for key, n in projection.get("role_sites", {}).items() if n}
    result.update("class:" + key for key, n in projection.get("instruction_classes", {}).items() if n)
    for field in projection.get("field_fingerprints", []):
        label = field["instruction"] + "." + field["field"]
        unresolved, missing = field.get("unresolved_sites", 0), field.get("missing_sites", 0)
        known = field["sites"] - unresolved
        status = "resolved" if not unresolved and not missing else "partial" if known else "unresolved"
        result.add("field:" + label + ":" + status)
        result.add("field:" + label + ":distinct-bin:" + str(bucket(field["distinct_values"])))
    for label, values in projection.get("field_profiles", {}).items():
        for row in values:
            category = value_bin(row["value"])
            if category is not None:
                result.add("field:" + label + ":sample-value-bin:" + category)
    result.add("structure:site-bin:" + str(bucket(analysis["accelerator_sites"])))
    result.add("structure:loop-bin:" + str(bucket(analysis.get("cfg", {}).get("natural_loops", 0))))
    result.add("structure:fence-bin:" + str(bucket(projection.get("fence_sites", 0))))
    result.add("structure:counted-control:" + projection.get("counted_control", {}).get("status", "unknown"))
    return result


#: Fields of :func:`family_signature`, in report order.
FAMILY_FIELDS = ("roles", "instruction_classes", "configured_fields", "loop_nest", "control", "dataflow")


def _loop_shape(analysis):
    """Loop-nest shape without trip counts, level counts or addresses (tail loops vary with shape)."""
    projection = analysis.get("mechanism_projection") or {}
    value = (projection.get("loop_nest") or {}).get("value") or {}
    return dict(depth=value.get("depth"), calls_in_loop=bool(value.get("calls_in_loop")) if value else None)


def family_signature(analysis):
    """Structural CCA signature of one lifted accelerator region; trip counts and shapes excluded.

    The same function serves bank units and an author's emitted kernels, so two signatures compare
    field by field. Kept: which decoded roles and instruction classes occur, which instruction fields
    are configured, the loop-nest shape, whether scalar control proved, and the non-numeric canonical
    CCA facts (dataflow forms, dtypes, flags). Dropped: every count, extent, stride, value and address.
    """
    projection = analysis.get("mechanism_projection") or {}
    dataflow = {}
    if isinstance(analysis.get("cca"), dict):
        for path, value in known_leaves(canonical(analysis)).items():
            if type(value) is bool or isinstance(value, str):
                dataflow[path] = value
            elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                dataflow[path] = sorted(value)
    return dict(
        roles=sorted(key for key, n in (projection.get("role_sites") or {}).items() if n),
        instruction_classes=sorted(key for key, n in (projection.get("instruction_classes") or {}).items() if n),
        configured_fields=sorted(
            {
                row["instruction"] + "." + row["field"]
                for row in projection.get("field_fingerprints") or ()
                if isinstance(row, dict) and "instruction" in row and "field" in row
            }
        ),
        loop_nest=_loop_shape(analysis),
        control=(projection.get("counted_control") or {}).get("status", "unknown"),
        dataflow=dataflow,
    )


def family_key(signature):
    """Stable identity of one structural signature."""
    import hashlib

    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]


def _items(value):
    if isinstance(value, list):
        return {json.dumps(item, sort_keys=True) for item in value}
    if isinstance(value, dict):
        return {json.dumps([key, item], sort_keys=True) for key, item in value.items()}
    return {json.dumps(value, sort_keys=True)}


def signature_similarity(left, right):
    """Per-field agreement of two signatures; the score is the mean per-field Jaccard overlap.

    The basis lists the fields that agree and, for each differing field, what only one side has. It is
    structural agreement only: it never establishes semantic equivalence or relative performance."""
    agree, differ, score = [], {}, 0.0
    for name in FAMILY_FIELDS:
        a, b = _items(left.get(name)), _items(right.get(name))
        union = a | b
        score += len(a & b) / len(union) if union else 1.0
        if a == b:
            agree.append(name)
        else:
            differ[name] = dict(
                only_left=[json.loads(item) for item in sorted(a - b)],
                only_right=[json.loads(item) for item in sorted(b - a)],
            )
    return dict(score=round(score / len(FAMILY_FIELDS), 4), agree=agree, differ=differ)


def card(occurrence, analysis):
    projection = analysis["mechanism_projection"]
    observations = projection.get("mechanism_observations", {})
    fingerprints = projection.get("field_fingerprints", [])
    return dict(
        occurrence_id=occurrence["id"],
        artifact_id=occurrence["artifact_id"],
        # The original's own symbol for this body (ELF symtab), so a unit can be named without its address.
        symbol=occurrence.get("symbol"),
        address=occurrence["address"],
        size_bytes=occurrence["size_bytes"],
        accelerator_sites=occurrence["accelerator_sites"],
        boundary_status=occurrence["boundary_status"],
        canonical_facts=known_leaves(analysis["cca"]),
        role_static_sites=projection.get("role_sites", {}),
        instruction_class_static_sites=projection.get("instruction_classes", {}),
        field_count=len(fingerprints),
        field_sites=sum(row["sites"] for row in fingerprints),
        unresolved_field_sites=sum(row.get("unresolved_sites", 0) for row in fingerprints),
        missing_field_sites=sum(row.get("missing_sites", 0) for row in fingerprints),
        field_fingerprint_status=projection.get("field_fingerprint_status"),
        counted_control=projection.get("counted_control", {}),
        schedule_complete=analysis.get("schedule_model", {}).get("complete", False),
        repetition_census=observations.get("repetition_census"),
        invariant_fields=observations.get("invariant_fields", []),
        repeated_operand_groups=observations.get("repeated_operand_groups", []),
        evidence_id=analysis["evidence"]["evidence_id"],
        unknowns=analysis["cca"].get("provenance", {}).get("unknowns", []),
        qualification="static observations, conditional on selected ISA; no numerical or performance qualification",
    )


def coverage(units):
    cards = [item for unit in units for item in unit["cards"]]
    signatures = Counter(json.dumps(item, sort_keys=True) for unit in units for item in unit["canonical"])
    return dict(
        unique_bodies=len(units),
        occurrences=len(cards),
        model_occurrences=dict(sorted(Counter(item["artifact_id"] for item in cards).items())),
        static_accelerator_sites=sum(item["accelerator_sites"] for item in cards),
        raw_code_bytes=sum(unit["size_bytes"] for unit in units),
        canonical_signatures=len(signatures),
        largest_canonical_signature_occurrences=max(signatures.values(), default=0),
        canonical_axis_knownness=axis_knownness(item for unit in units for item in unit["canonical"]),
        instruction_classes=sorted({name for item in cards for name in item["instruction_class_static_sites"]}),
        roles=sorted({name for item in cards for name in item["role_static_sites"]}),
        observed_features=len(set().union(*(unit["features"] for unit in units))) if units else 0,
        field_sites=sum(item["field_sites"] for item in cards),
        unresolved_field_sites=sum(item["unresolved_field_sites"] for item in cards),
        missing_field_sites=sum(item["missing_field_sites"] for item in cards),
        complete_schedules=sum(bool(item["schedule_complete"]) for item in cards),
        feature_absence="not a proof of semantic absence; projections and value populations can be bounded",
    )


def diversity_order(units):
    """One declared diversity-first curriculum, not a random-subset or cost ablation."""
    remaining = sorted(units, key=lambda unit: unit["kernel_id"])
    selected, seen, models = [], set(), set()
    while remaining:

        def score(unit):
            unseen = unit["features"] - seen
            roles = {name.removeprefix("role:") for name in unit["features"] if name.startswith("role:")}
            return (
                bool(set(unit["models"]) - models),
                bool(roles - {"config", "sync"}),
                sum(name.startswith("role:") for name in unseen),
                sum(name.startswith("class:") for name in unseen),
                sum(name.startswith("field:") for name in unseen),
                sum(name.startswith("structure:") for name in unseen),
                -unit["size_bytes"],
            )

        chosen = max(remaining, key=score)
        chosen["selection_score"] = score(chosen)
        chosen["new_features"] = sorted(chosen["features"] - seen)
        selected.append(chosen)
        seen.update(chosen["features"])
        models.update(chosen["models"])
        remaining.remove(chosen)
    return selected


# ---- reference_subsets -------------------------------------------------------------------------
# Prepare nested observation-only CCA banks; never launch an author experiment.

SCHEMA = "merlin.reference_subset_bank.v1"


def inventory(root, artifact_ids):
    root = Path(root)
    selected = set(artifact_ids)
    if not selected or len(selected) != len(artifact_ids):
        raise ValueError("select explicit distinct original artifact IDs")
    with sqlite3.connect("file:" + str((root / "database.sqlite").absolute()) + "?mode=ro", uri=True) as db:
        artifacts = [dict(id=r[0], path=r[1], sha256=r[2]) for r in db.execute("SELECT id,path,sha256 FROM artifacts")]
        if not selected <= {row["id"] for row in artifacts}:
            raise ValueError("unknown original artifact ID")
        rows = [json.loads(r[0]) for r in db.execute("SELECT metadata_json FROM occurrences WHERE accelerator_sites>0")]
    units = {}
    for row in rows:
        if row["artifact_id"] not in selected:
            continue
        analysis = json.loads((root / safe_member(row["analysis"])).read_bytes())
        unit = units.setdefault(
            row["kernel_id"],
            dict(
                kernel_id=row["kernel_id"],
                size_bytes=row["size_bytes"],
                models=set(),
                features=set(),
                occurrences=[],
                cards=[],
                canonical=[],
            ),
        )
        unit["models"].add(row["artifact_id"])
        unit["features"].update(features(analysis))
        unit["occurrences"].append(row)
        unit["cards"].append(card(row, analysis))
        unit["canonical"].append(canonical(analysis))
    # Derived products (e.g. ahead-of-time compiled sources) contribute units, never raw grants.
    raw = [row for row in artifacts if row["id"] in selected and Path(row["path"]).parts[:1] != ("derived",)]
    return raw, list(units.values())


def prepare(root, out, artifact_ids, sizes):
    root, out = Path(root).absolute(), Path(out).absolute()
    if (out / "selection.json").exists():
        raise ValueError("finished selection exists; allocate a new product")
    verify_database(root)
    raw, units = inventory(root, artifact_ids)
    if not units or len(set(sizes)) != len(sizes) or any(type(n) is not int or n < 1 or n > len(units) for n in sizes):
        raise ValueError("sizes must be distinct positive counts within the selected corpus")
    ordered = diversity_order(units)
    sizes = sorted(set([*sizes, len(ordered)]))
    raw = sorted(raw, key=lambda row: row["id"])
    out.mkdir(parents=True, exist_ok=True)
    # This operator-only descriptor is not an author mount or an original-reference grant.
    write_json(
        out / "raw-mount-selection.json",
        dict(
            corpus_root=str(root),
            files=raw,
            mode="read_only_per_file",
            author_mount_implemented=False,
            warning="Mount only these original files; never expose the global corpus root to an author.",
        ),
    )
    cases = [
        _condition(root, out, "raw" if not count else "cca-" + str(count), raw, ordered[:count])
        for count in [0, *sizes]
    ]
    selection = dict(
        schema="merlin.reference_subset_selection.v1",
        database_sha256=digest(root / "database.sqlite"),
        corpus_manifest_sha256=digest(root / "manifest.json"),
        artifact_ids=sorted(artifact_ids),
        raw=raw,
        conditions=cases,
        order=_order_rows(ordered),
        algorithm="model coverage, execution roles, unseen roles/classes/field bins/control; byte size; SHA tie-break",
        selection_uses="entire precomputed reference/ISA observation bank; no candidate or evaluation results",
        cost_boundary="all-reference preprocessing and selection cost applies even to the smallest bank",
        experiment="CCA coverage conditional on identical full raw references; not total reference availability",
        status="offline_banks_prepared_not_launch_qualified",
        native_broker_canary="not_run",
        author_campaigns_launched=0,
    )
    write_json(out / "selection.json", selection)
    return selection


def _copy_decoder(root, destination):
    """Ship the corpus's pinned operand decoder (contract, facts, sidecar) with every CCA bank."""
    import yaml

    contract = root / "analysis-inputs" / "contract.yaml"
    facts = root / "analysis-inputs" / "facts.json"
    if not contract.is_file() or not facts.is_file():
        return
    selection = (yaml.safe_load(contract.read_bytes()) or {}).get("rocc_operand_selection") or {}
    sidecar = Path((selection.get("sidecar") or {}).get("path") or "")
    if not sidecar.is_file() or digest(sidecar) != selection["sidecar"].get("sha256"):
        return
    if digest(facts) != (selection.get("original_facts") or {}).get("sha256"):
        raise ValueError("corpus decoder facts differ from the contract's operand selection")
    target = destination / "decoder"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract, target / "target_contract.yaml")
    shutil.copyfile(facts, target / "facts.json")
    shutil.copyfile(sidecar, target / "operand-protocol.json")


def _condition(root, out, label, raw, selected):
    destination = out / "conditions" / label
    if destination.exists():
        raise ValueError("condition output already exists")
    destination.mkdir(parents=True)
    write_json(destination / "raw-grant.json", dict(files=raw, identical_for_every_condition=True))
    entries = []
    for unit in selected:
        relative = Path("units") / safe_member(unit["kernel_id"]) / "card.json"
        write_json(destination / relative, dict(kernel_id=unit["kernel_id"], occurrences=unit["cards"]))
        reports = []
        for occurrence in unit["occurrences"]:
            parent = Path(occurrence["analysis"]).parent
            for name in ("analysis.json", "cca.json", "cca.mlir", "instructions.dis"):
                path = parent / name
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / path, target)
                reports.append(path.as_posix())
        entries.append(dict(kernel_id=unit["kernel_id"], card=relative.as_posix(), reports=reports))
    if selected:
        _copy_decoder(root, destination)
    pins = {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    bank = dict(
        schema=SCHEMA,
        label=label,
        cca_enabled=bool(selected),
        raw=raw,
        units=entries,
        coverage=coverage(selected),
        files=pins,
        on_demand_outside_selection="not_authorized",
        runtime_isolation="not_verified",
        semantics="observations_only",
    )
    write_json(destination / "bank.json", bank)
    verify_bank(destination)
    return dict(label=label, bank="conditions/" + label + "/bank.json", coverage=bank["coverage"])


def _order_rows(ordered):
    return [
        dict(
            kernel_id=u["kernel_id"],
            models=sorted(u["models"]),
            occurrences=[o["id"] for o in u["occurrences"]],
            selection_score=u["selection_score"],
            new_features=u["new_features"],
        )
        for u in ordered
    ]


def prepare_stages(root, out, stages):
    """Cumulative source-ordered banks: stage k admits every unit any of stages 1..k's originals use.

    ``stages`` is an ordered list of (name, artifact IDs). A unit shared with an earlier stage's
    originals enters at that earlier stage; inside a stage the declared diversity order applies.
    """
    root, out = Path(root).absolute(), Path(out).absolute()
    if (out / "selection.json").exists():
        raise ValueError("finished selection exists; allocate a new product")
    names = [name for name, _ in stages]
    members = [artifact for _, ids in stages for artifact in ids]
    if not stages or len(set(names)) != len(names) or any(not ids for _, ids in stages):
        raise ValueError("declare distinct non-empty stages")
    if len(set(members)) != len(members):
        raise ValueError("each original belongs to exactly one stage")
    verify_database(root)
    raw, units = inventory(root, members)
    raw = sorted(raw, key=lambda row: row["id"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "raw-mount-selection.json",
        dict(
            corpus_root=str(root),
            files=raw,
            mode="read_only_per_file",
            author_mount_implemented=False,
            warning="Mount only these original files; never expose the global corpus root to an author.",
        ),
    )
    ordered, admitted, cases, plan = [], set(), [_condition(root, out, "raw", raw, [])], []
    for name, ids in stages:
        admitted.update(ids)
        taken = {unit["kernel_id"] for unit in ordered}
        new = [unit for unit in units if unit["kernel_id"] not in taken and unit["models"] & admitted]
        ordered.extend(diversity_order(new))
        # A stage's raw grant is exactly the originals its bank was built from, cumulative in stage order.
        cases.append(_condition(root, out, "cca-" + name, [row for row in raw if row["id"] in admitted], list(ordered)))
        plan.append(dict(stage=name, artifact_ids=sorted(ids), new_units=len(new), cumulative_units=len(ordered)))
    selection = dict(
        schema="merlin.reference_subset_selection.v1",
        database_sha256=digest(root / "database.sqlite"),
        corpus_manifest_sha256=digest(root / "manifest.json"),
        artifact_ids=sorted(members),
        raw=raw,
        conditions=cases,
        stages=plan,
        order=_order_rows(ordered),
        algorithm="cumulative declared source stages; diversity order inside each stage; SHA tie-break",
        selection_uses="entire precomputed reference/ISA observation bank; no candidate or evaluation results",
        cost_boundary="all-reference preprocessing and selection cost applies even to the smallest bank",
        experiment="CCA coverage by declared source order, conditional on identical full raw references",
        status="offline_banks_prepared_not_launch_qualified",
        native_broker_canary="not_run",
        author_campaigns_launched=0,
    )
    write_json(out / "selection.json", selection)
    return selection


def verify_bank(root):
    root = Path(root)
    bank = json.loads((root / "bank.json").read_bytes())
    if bank.get("schema") != SCHEMA:
        raise ValueError("unsupported selected bank")
    for relative, expected in bank["files"].items():
        path = root / safe_member(relative)
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("bank asset pin mismatch")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "bank.json"}
    if actual != set(bank["files"]):
        raise ValueError("unselected asset in bank")
    ids = [unit["kernel_id"] for unit in bank["units"]]
    if len(set(ids)) != len(ids) or bool(ids) != bank["cca_enabled"]:
        raise ValueError("invalid selected unit roster")
    for unit in bank["units"]:
        for relative in [unit["card"], *unit["reports"]]:
            if relative not in bank["files"]:
                raise ValueError("unselected report path")
    return bank


def query(root, *, instruction=None, field=None, role=None, kernel_id=None, limit=4):
    """Retrieve only declared units; include exact known values and missingness, never advice."""
    root = Path(root)
    bank = verify_bank(root)
    if not bank["cca_enabled"]:
        raise ValueError("CCA is unavailable in this condition")
    if not 1 <= limit <= 8 or field and not instruction:
        raise ValueError("query requires a finite limit and an instruction for field lookup")
    allowed = {unit["kernel_id"] for unit in bank["units"]}
    if kernel_id is not None and kernel_id not in allowed:
        raise ValueError("unit is outside the selected CCA bank")
    result = []
    for unit in bank["units"]:
        if kernel_id is not None and unit["kernel_id"] != kernel_id:
            continue
        for relative in unit["reports"]:
            if not relative.endswith("/analysis.json"):
                continue
            analysis = json.loads((root / relative).read_bytes())
            projection = analysis["mechanism_projection"]
            if role and not projection.get("role_sites", {}).get(role):
                continue
            if instruction and not projection.get("instruction_classes", {}).get(instruction):
                continue
            fields = [
                item
                for item in projection.get("field_fingerprints", [])
                if (not instruction or item["instruction"] == instruction) and (not field or item["field"] == field)
            ]
            if field and not fields:
                continue
            denominator = sum(item["sites"] + item.get("missing_sites", 0) for item in fields)
            known = sum(item["sites"] - item.get("unresolved_sites", 0) for item in fields)
            has_values = any(
                projection.get("field_profiles", {}).get(item["instruction"] + "." + item["field"]) for item in fields
            )
            result.append(
                dict(
                    kernel_id=unit["kernel_id"],
                    analysis=relative,
                    evidence_id=analysis["evidence"]["evidence_id"],
                    role_static_sites=projection.get("role_sites", {}),
                    canonical_facts=card_facts(analysis),
                    fields=[
                        dict(
                            item,
                            value_profile=projection.get("field_profiles", {}).get(
                                item["instruction"] + "." + item["field"], []
                            ),
                        )
                        for item in fields[:4]
                    ],
                    matching_fields=len(fields),
                    fields_omitted=max(0, len(fields) - 4),
                    observed_values_available=has_values,
                    known_field_fraction=known / denominator if denominator else 0,
                    profile_scope=(
                        "bounded observed values; fingerprint counts cover more sites; missing values stay unknown"
                    ),
                    count_basis=projection.get("count_basis", "listed_instruction_sites"),
                    unknowns=analysis["cca"].get("provenance", {}).get("unknowns", []),
                    semantic_equivalence="unknown",
                    performance="unknown",
                )
            )
    result.sort(key=lambda item: (item["observed_values_available"], item["known_field_fraction"]), reverse=True)
    selected, seen = [], set()
    for item in result:
        if kernel_id is None and item["kernel_id"] in seen:
            continue
        selected.append(item)
        seen.add(item["kernel_id"])
        if len(selected) == limit:
            break
    return dict(
        results=selected,
        limit=limit,
        matching_occurrences=len(result),
        ordering="observed values and field knownness; ties use frozen curriculum; no performance ranking",
    )


def card_facts(analysis):

    return known_leaves(analysis["cca"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("prepare")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--artifact", action="append", required=True)
    build.add_argument("--sizes", type=int, nargs="+", required=True)
    staged = actions.add_parser("prepare-stages")
    staged.add_argument("--corpus", type=Path, required=True)
    staged.add_argument("--out", type=Path, required=True)
    staged.add_argument("--stage", action="append", required=True, help="NAME=ARTIFACT[,ARTIFACT...] in declared order")
    inspect = actions.add_parser("query")
    inspect.add_argument("--bank", type=Path, required=True)
    for name in ("instruction", "field", "role", "kernel-id"):
        inspect.add_argument("--" + name)
    inspect.add_argument("--limit", type=int, default=4)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.corpus, args.out, args.artifact, args.sizes)
    elif args.action == "prepare-stages":
        stages = [(spec.partition("=")[0], [a for a in spec.partition("=")[2].split(",") if a]) for spec in args.stage]
        result = prepare_stages(args.corpus, args.out, stages)
    else:
        result = query(
            args.bank,
            instruction=args.instruction,
            field=args.field,
            role=args.role,
            kernel_id=args.kernel_id,
            limit=args.limit,
        )
    print(json.dumps(observations_only(result), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
