"""Build/query a deduplicated reference corpus using the existing offline CCA owners.

Inputs are explicit pinned archives/files. No compiler optimization, author,
numerical workload or hardware execution is selected by this command.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

from .reference_database_io import copy_checked, digest, safe_member, unpack_selected, write_json

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE artifacts(id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, sha256 TEXT NOT NULL,
 format TEXT NOT NULL, origin_json TEXT NOT NULL);
CREATE TABLE models(id TEXT PRIMARY KEY REFERENCES artifacts(id), name TEXT NOT NULL);
CREATE TABLE kernels(id TEXT PRIMARY KEY, kind TEXT NOT NULL, body_sha256 TEXT NOT NULL,
 path TEXT UNIQUE NOT NULL, size_bytes INTEGER NOT NULL, UNIQUE(kind, body_sha256));
CREATE TABLE occurrences(id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL REFERENCES artifacts(id),
 kernel_id TEXT NOT NULL REFERENCES kernels(id), symbol TEXT, address INTEGER, size_bytes INTEGER,
 accelerator_sites INTEGER, boundary_status TEXT NOT NULL, analysis TEXT NOT NULL, metadata_json TEXT NOT NULL);
CREATE TABLE call_edges(id INTEGER PRIMARY KEY, caller_id TEXT NOT NULL REFERENCES occurrences(id),
 callee_id TEXT REFERENCES occurrences(id), metadata_json TEXT NOT NULL);
CREATE INDEX occurrences_by_artifact ON occurrences(artifact_id);
CREATE INDEX occurrences_by_kernel ON occurrences(kernel_id);
CREATE INDEX calls_by_callee ON call_edges(callee_id);
CREATE VIEW model_to_kernel AS SELECT m.name AS model, o.id AS occurrence_id, o.kernel_id,
 o.address, o.size_bytes, o.accelerator_sites, o.boundary_status, o.analysis
 FROM models m JOIN occurrences o ON o.artifact_id=m.id;
CREATE VIEW unique_accelerator_kernels AS SELECT DISTINCT k.* FROM kernels k
 JOIN occurrences o ON o.kernel_id=k.id WHERE o.accelerator_sites>0;
"""


def write_database(path, artifacts, bodies, occurrences, edges):
    if Path(path).exists():
        raise ValueError("database already exists; use a fresh output identity")
    unique = {row["id"]: row for row in bodies}
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        for row in artifacts:
            db.execute(
                "INSERT INTO artifacts VALUES(?,?,?,?,?)",
                (row["id"], row["path"], row["sha256"], row["format"], json.dumps(row["origin"], sort_keys=True)),
            )
            if row.get("model"):
                db.execute("INSERT INTO models VALUES(?,?)", (row["id"], row["model"]))
        for row in unique.values():
            db.execute(
                "INSERT INTO kernels VALUES(?,?,?,?,?)",
                tuple(row[key] for key in ("id", "kind", "body_sha256", "path", "size_bytes")),
            )
        for row in occurrences:
            db.execute(
                "INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    row["id"],
                    row["artifact_id"],
                    row["kernel_id"],
                    row.get("symbol"),
                    row.get("address"),
                    row.get("size_bytes"),
                    row.get("accelerator_sites"),
                    row["boundary_status"],
                    row["analysis"],
                    json.dumps(row, sort_keys=True),
                ),
            )
        for row in edges:
            db.execute(
                "INSERT INTO call_edges(caller_id,callee_id,metadata_json) VALUES(?,?,?)",
                (row["caller_id"], row.get("callee_id"), json.dumps(row, sort_keys=True)),
            )
        if (
            db.execute("PRAGMA integrity_check").fetchone() != ("ok",)
            or db.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise ValueError("database integrity check failed")
    return list(unique.values())


def stage_inputs(spec, root):
    artifacts, origins = [], []
    for archive in spec.get("archives", []):
        source = Path(archive["path"])
        if source.is_symlink() or digest(source) != archive["sha256"]:
            raise ValueError("original archive differs from selected hash")
        relative = safe_member(archive["destination"])
        unpack_selected(
            source,
            root / relative,
            [member["member"] for member in archive["members"]],
            archive_format=archive["format"],
        )
        if digest(source) != archive["sha256"]:
            raise ValueError("archive changed during extraction")
        origins.append({key: value for key, value in archive.items() if key != "members"})
        for member in archive["members"]:
            path = relative / safe_member(member["member"])
            if digest(root / path) != member["sha256"]:
                raise ValueError("extracted member differs from selected hash")
            artifacts.append(
                dict(
                    member,
                    path=path.as_posix(),
                    origin=dict(kind="archive_member", archive_sha256=archive["sha256"], member=member["member"]),
                )
            )
    for source in spec.get("files", []):
        relative = safe_member(source["destination"])
        copy_checked(source["path"], root / relative, source["sha256"])
        artifacts.append(dict(source, path=relative.as_posix(), origin=source["origin"]))
    ids = [row["id"] for row in artifacts]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate artifact IDs")
    # Original input paths are recorded in provenance, not copied executable code.
    write_json(root / "originals" / "manifest.json", dict(artifacts=artifacts, archives=origins))
    return artifacts


def build(spec_path, root):
    from merlin_experiments.phase1.feedback import reference_analysis as analysis

    from .reference_database_analysis import ADVICE_KEYS, machine_regions, source_regions
    from .reference_database_audit import audit

    spec_path, root = Path(spec_path).absolute(), Path(root).absolute()
    spec = json.loads(spec_path.read_bytes())
    if spec.get("schema") != "merlin.reference_database_inputs.v1":
        raise ValueError("unsupported input selection")
    if (root / "database.sqlite").exists() or (root / "manifest.json").exists():
        raise ValueError("finished output already exists")
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    artifacts = stage_inputs(spec, root)
    selected = {}
    for role, suffix in (("contract", ".yaml"), ("facts", ".json")):
        original = spec[role]
        destination = root / "analysis-inputs" / (role + suffix)
        copy_checked(original["path"], destination, original["sha256"])
        selected[role] = destination
    decoder = analysis._operand_decoder(selected["contract"], selected["facts"], spec["target"], include_structure=True)
    if decoder.get("semantics") is None:
        raise ValueError("selected ISA operand observer unavailable")
    write_json(
        root / "analysis-inputs" / "decoder.json",
        dict(
            target=spec["target"],
            selected_input_pins=decoder.get("decoder_inputs", []),
            original_selection=spec,
            interpretation="conditional on selected ISA/facts; no model build-record correspondence",
        ),
    )
    bodies, occurrences, edges, summaries = [], [], [], []
    for artifact in artifacts:
        begin = time.monotonic()
        if artifact["format"] == "elf":
            result = machine_regions(root, artifact, facts=selected["facts"], decoder=decoder)
        elif artifact["format"] in ("c_source", "python_schedule"):
            result = source_regions(root, artifact)
        else:
            continue
        new_bodies, new_occurrences, new_edges, summary = result
        bodies.extend(new_bodies)
        occurrences.extend(new_occurrences)
        edges.extend(new_edges)
        summaries.append(dict(artifact_id=artifact["id"], summary=summary, seconds=time.monotonic() - begin))
        print(json.dumps(summaries[-1], sort_keys=True), flush=True)
    unique = write_database(root / "database.sqlite", artifacts, bodies, occurrences, edges)
    source_symbols = {}
    for row in occurrences:
        if row["boundary_status"] == "parsed_source_definition":
            source_symbols.setdefault(row["symbol"], []).append(row["id"])
    name_matches = []
    for row in occurrences:
        for symbol in row.get("containing_symbols", []):
            for source in source_symbols.get(symbol["name"], []):
                name_matches.append(
                    dict(
                        machine_occurrence=row["id"],
                        source_occurrence=source,
                        symbol=symbol["name"],
                        basis="ELF-contained symbol and parsed source name",
                        status="name_correspondence_only",
                        semantic_equivalence="unknown",
                    )
                )
    write_json(
        root / "mappings" / "source-binary-name-matches.json",
        dict(matches=name_matches, scope="discovery links, not verified source/build or semantic correspondence"),
    )
    by_kernel = {}
    for row in occurrences:
        by_kernel.setdefault(row["kernel_id"], []).append(row)
    for kernel in unique:
        write_json(
            root / Path(kernel["path"]).parent / "index.json",
            dict(
                body=kernel,
                occurrences=by_kernel[kernel["id"]],
                identity_scope="exact code/source/recipe bytes, not context-independent semantic equivalence",
            ),
        )
    # Exact same symbol strings are discoverable, but never upgraded to source/binary correspondence.
    write_json(
        root / "mappings" / "index.json",
        dict(
            occurrences=occurrences,
            call_edges=edges,
            mapping_scope="original-model -> inferred code span -> exact-byte body; source -> parsed definition",
            unavailable=[
                "model layer names for stripped binaries",
                "source/binary semantic equivalence",
                "candidate-compiler matching",
                "dynamic invocation order/multiplicity",
            ],
        ),
    )
    for label, records in (
        ("artifacts", artifacts),
        ("kernels", unique),
        ("occurrences", occurrences),
        ("call_edges", edges),
    ):
        with (root / (label + ".jsonl")).open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
    with (root / "model-to-kernel.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "artifact_id",
                "id",
                "kernel_id",
                "address",
                "size_bytes",
                "accelerator_sites",
                "boundary_status",
                "analysis",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(row for row in occurrences if row.get("address") is not None)
    owners = []
    for name, module in sorted(sys.modules.copy().items()):
        path = getattr(module, "__file__", None)
        if path and name.startswith(("merlin.", "merlin_experiments.")) and Path(path).is_file():
            owners.append(dict(module=name, path=str(Path(path).absolute()), sha256=digest(path)))
    write_json(
        root / "analysis-inputs" / "implementation.json",
        dict(
            modules=owners,
            note="Observed Python implementation files, not an entire interpreter/dependency execution closure.",
        ),
    )
    write_json(root / "FAIRNESS.json", audit(root, artifacts, occurrences, summaries))
    accelerator_occurrences = sum(bool(row.get("accelerator_sites")) for row in occurrences)
    (root / "README.md").write_text(
        "# Expert kernel database\n\n"
        f"{len(artifacts)} original artifacts; {len(occurrences)} code/source occurrences; "
        f"{len(unique)} exact-byte-distinct bodies; "
        f"{len(occurrences) - len(unique)} duplicate occurrences consolidated. "
        f"{accelerator_occurrences} accelerator-containing spans have CCA JSON and MLIR.\n\n"
        "```text\n"
        "originals/\n"
        "  models/                 Jack: resnet50.elf, smolvla.elf, tinyllama.elf\n"
        "  supplied-source/        Supplied Exo LLM C/header\n"
        "  upstream-exo/           Pinned Exo schedules, derived C/ELFs, dependencies, licenses\n"
        "kernels/\n"
        "  machine/<sha>/          body.bin and occurrence index.json\n"
        "  c_source/<sha>/         body.c and occurrence index.json\n"
        "  python_schedule/<sha>/  body.py and occurrence index.json\n"
        "  source_export/<sha>/    Original export recipe and occurrence index.json\n"
        "mappings/                 Model/source occurrences, static calls and source reports\n"
        "  <artifact>/<span>/      instructions.dis, analysis.json, cca.json, cca.mlir\n"
        "analysis-inputs/          Selected ISA/RTL inputs and implementation pins\n"
        "database.sqlite           Queryable bodies, occurrences and mappings\n"
        "model-to-kernel.csv        Machine-span mapping\n"
        "*.jsonl                   Portable database tables\n"
        "manifest.json             Counts, provenance, timings and limitations\n"
        "FAIRNESS.json             Input/output boundary audit\n"
        "SHA256SUMS.json           File pins\n"
        "```\n\n"
        "## Query\n\n"
        "```sql\n"
        "SELECT * FROM model_to_kernel WHERE accelerator_sites > 0 ORDER BY model, address;\n"
        "SELECT kernel_id, COUNT(*) AS uses FROM occurrences GROUP BY kernel_id HAVING COUNT(*) > 1;\n"
        "SELECT * FROM unique_accelerator_kernels;\n"
        "```\n\n"
        "## Evidence boundaries\n\n"
        "Model binaries are stripped. Decomposition recovers inferred static code spans, "
        "not verified layer names or source functions. Bodies are byte slices, not standalone "
        "relinkable executables. Deduplication means identical bytes within a representation; "
        "it does not establish semantic or numerical equivalence across contexts.\n\n"
        "CCA interpretation is conditional on the selected ISA/RTL definitions; matching "
        "original build records were not supplied. Every selected accelerator instruction site "
        "is covered, but individual semantic fields can remain unknown. Host-only spans have "
        "structural reports. C/Python definitions and exported procedures are parsed syntax; "
        "their source-to-command semantic lifts remain unknown. Name matches are discovery "
        "links, not source/binary equivalence proofs.\n\n"
        "This is declared bottom-up reference preprocessing. No candidate compiler, hidden "
        "case, grading answer, or measured winner supplies analysis facts. Optimization advice "
        "is removed from analysis exports; candidate matching and optimization decisions belong "
        "inside the author budget. Both experiment arms must have the same original reference "
        "selection and public target definitions; disclose which receives the derived bank and "
        "report preprocessing cost separately. No numerical, performance or hardware validation "
        "is implied. FAIRNESS.json audits this builder's boundary, not arbitrary dependency behavior.\n\n"
        "The durable input selection and rebuild guide live at "
        "examples/gemmini/reference-kernels/ in the repository.\n"
    )
    # Pin analysis products as well as original/code bytes. The final manifest pins this list.
    file_pins = {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in ("manifest.json", "SHA256SUMS.json")
    }
    write_json(root / "SHA256SUMS.json", file_pins)
    manifest = dict(
        schema="merlin.reference_database.v1",
        status="complete_offline_inventory",
        models=sum(bool(row.get("model")) for row in artifacts),
        artifacts=len(artifacts),
        unique_kernels=len(unique),
        occurrences=len(occurrences),
        duplicate_occurrences=len(occurrences) - len(unique),
        static_call_edges=len(edges),
        unresolved_call_edges=sum(row.get("callee_id") is None for row in edges),
        summaries=summaries,
        seconds=time.monotonic() - started,
        input_spec_sha256=digest(spec_path),
        deduplication="exact bytes within representation; no semantic deduplication",
        source_to_binary_mapping="unknown; matching source/build records were not supplied",
        numerical_checks="not_run",
        hardware_execution="not_run",
        performance="unknown",
        optimization_selection="not_performed",
        removed_advice_fields=sorted(ADVICE_KEYS),
        database_sha256=digest(root / "database.sqlite"),
        file_inventory_sha256=digest(root / "SHA256SUMS.json"),
    )
    write_json(root / "manifest.json", manifest)
    verify(root)
    return manifest


def verify(root):
    root = Path(root).absolute()
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_bytes())
        if digest(root / "database.sqlite") != manifest["database_sha256"]:
            raise ValueError("database pin mismatch")
        if digest(root / "SHA256SUMS.json") != manifest["file_inventory_sha256"]:
            raise ValueError("file inventory pin mismatch")
        for relative, expected in json.loads((root / "SHA256SUMS.json").read_bytes()).items():
            if digest(root / safe_member(relative)) != expected:
                raise ValueError("analysis product pin mismatch: " + relative)
    with sqlite3.connect("file:" + str(root / "database.sqlite") + "?mode=ro", uri=True) as db:
        if (
            db.execute("PRAGMA integrity_check").fetchone() != ("ok",)
            or db.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise ValueError("database integrity check failed")
        for path, expected in db.execute(
            "SELECT path,sha256 FROM artifacts UNION ALL SELECT path,body_sha256 FROM kernels"
        ):
            if digest(root / path) != expected:
                raise ValueError("artifact/body pin mismatch: " + path)
        for relative, raw in db.execute("SELECT analysis,metadata_json FROM occurrences"):
            if not (root / relative).is_file():
                raise ValueError("missing occurrence analysis")
            row = json.loads(raw)
            if row.get("listing") and not (root / row["listing"]).is_file():
                raise ValueError("missing machine listing")
        counts = dict(
            kernels=db.execute("SELECT COUNT(*) FROM kernels").fetchone()[0],
            occurrences=db.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0],
        )
    return dict(status="PASS", **counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("build")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    inspect = sub.add_parser("verify")
    inspect.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    result = build(args.spec, args.out) if args.action == "build" else verify(args.root)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
