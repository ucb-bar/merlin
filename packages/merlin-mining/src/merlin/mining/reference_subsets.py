"""Prepare nested observation-only CCA banks; never launch an author experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from .reference_database import verify as verify_database
from .reference_database_analysis import observations_only
from .reference_database_io import digest, safe_member, write_json
from .reference_subset_cards import canonical, card, coverage, diversity_order, features

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
    return [row for row in artifacts if row["id"] in selected], list(units.values())


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
    cases = []
    for count in [0, *sizes]:
        label = "raw" if not count else "cca-" + str(count)
        destination = out / "conditions" / label
        if destination.exists():
            raise ValueError("condition output already exists")
        destination.mkdir(parents=True)
        selected = ordered[:count]
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
        pins = {p.relative_to(destination).as_posix(): digest(p) for p in sorted(destination.rglob("*")) if p.is_file()}
        bank = dict(
            schema=SCHEMA,
            label=label,
            cca_enabled=bool(count),
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
        cases.append(dict(label=label, bank="conditions/" + label + "/bank.json", coverage=bank["coverage"]))
    selection = dict(
        schema="merlin.reference_subset_selection.v1",
        database_sha256=digest(root / "database.sqlite"),
        corpus_manifest_sha256=digest(root / "manifest.json"),
        artifact_ids=sorted(artifact_ids),
        raw=raw,
        conditions=cases,
        order=[
            dict(
                kernel_id=u["kernel_id"],
                models=sorted(u["models"]),
                occurrences=[o["id"] for o in u["occurrences"]],
                selection_score=u["selection_score"],
                new_features=u["new_features"],
            )
            for u in ordered
        ],
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
    from .reference_subset_cards import known_leaves

    return known_leaves(analysis["cca"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("prepare")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--artifact", action="append", required=True)
    build.add_argument("--sizes", type=int, nargs="+", required=True)
    inspect = actions.add_parser("query")
    inspect.add_argument("--bank", type=Path, required=True)
    for name in ("instruction", "field", "role", "kernel-id"):
        inspect.add_argument("--" + name)
    inspect.add_argument("--limit", type=int, default=4)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.corpus, args.out, args.artifact, args.sizes)
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
