"""Shared configuration query execution; artifacts alone confer no authority.

The consumer must supply its canonical freshness reconstruction, check the exact
candidate type/root roster first, and own live completion delivery afterward.
The historical API remains unchanged while the current compiler route is added.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from merlin.common.paths import module_source_path

from .configuration_elision_bridge import canonical, file_ref
from .configuration_elision_emit import MODULE
from .configuration_elision_gate import Unavailable, _audit, _exact, _run, require_query_success
from .configuration_elision_intake import RELATION, proof_sources


def prove(
    candidate,
    *,
    lean,
    destination,
    timeout,
    validate,
    query_module=MODULE,
    source_plan=proof_sources,
    relation=RELATION,
    require_success=require_query_success,
):
    """Return audited artifacts only; no live token or delivery authority."""
    validate(candidate)
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    library = directory / "lib"
    library.mkdir()
    executable = Path(lean).resolve()
    tool = file_ref(executable)
    auditor = Path(module_source_path("merlin.verify")).parent / "lean/Merlin/GraphStateAudit.lean"
    auditor_ref = file_ref(auditor)
    env = {
        **os.environ,
        "LEAN_PATH": str(library),
        "LEAN_NUM_THREADS": "1",
        "PATH": str(executable.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    snapshots = []
    for label, text in (("source", candidate.source), ("final", candidate.final)):
        path = directory / (label + ".mlir")
        path.write_text(text)
        snapshots.append(file_ref(path))
    query = directory / (query_module + ".lean")
    query.write_text(candidate.query)
    query_ref = file_ref(query)
    mapping_records = []
    for label, value in (
        ("Projection", candidate.projection_json),
        ("Resolution", candidate.resolution_json),
        ("SourceRequest", candidate.request_json),
    ):
        path = directory / (label + ".json")
        path.write_text(value)
        mapping_records.append(file_ref(path))
    sources = source_plan()
    for i, row in enumerate(sources):
        obj = library / (row["module"].replace(".", "/") + ".olean")
        obj.parent.mkdir(parents=True, exist_ok=True)
        process = _run(
            [str(executable), "-j1", "-o", str(obj), row["path"]],
            Path(module_source_path("merlin.verify")).parent / "lean",
            env,
            directory,
            "import-" + str(i),
            timeout,
        )
        if process.returncode:
            raise Unavailable("configuration project proof source did not compile: " + row["module"])
    validate(candidate)
    obj = library / (query_module + ".olean")
    process = _run([str(executable), "-j1", "-o", str(obj), str(query)], directory, env, directory, "query", timeout)
    require_success(process)
    validate(candidate)
    if any(path.is_symlink() for path in library.rglob("*")):
        raise Unavailable("configuration private proof objects must not be symlinks")
    library_refs = [file_ref(path) for path in sorted(library.rglob("*")) if path.is_file()]
    values = {}
    for label, args in (
        ("audit", [query_module]),
        ("types", ["--statements", ",".join(candidate.roots), query_module]),
    ):
        process = _run([str(executable), "--run", str(auditor), *args], directory, env, directory, label, timeout)
        if process.returncode:
            raise Unavailable(label + " did not establish configuration proof evidence")
        try:
            values[label] = json.loads(process.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as error:
            raise Unavailable("malformed configuration audit output") from error
    if any(type(value) is not dict for value in values.values()):
        raise Unavailable("malformed configuration audit object")
    audit, statements = values["audit"], values["types"].get("statements")
    modules = {row["module"] for row in sources} | {query_module}
    _audit(audit, statements, candidate, modules, query_module=query_module)
    imports = []
    for row in audit["imports"]:
        if row["module"] in modules:
            expected = library / (row["module"].replace(".", "/") + ".olean")
            if Path(row["olean"]).resolve() != expected.resolve():
                raise Unavailable("configuration project import escaped fresh private object tree")
        imports.append(file_ref(row["olean"]))
    validate(candidate)
    for row in [tool, auditor_ref, query_ref, *snapshots, *mapping_records, *imports, *library_refs]:
        _exact(row)
    if any(path.is_symlink() for path in library.rglob("*")):
        raise Unavailable("configuration private proof object symlink introduced")
    if {str(path.resolve()) for path in library.rglob("*") if path.is_file()} != {row["path"] for row in library_refs}:
        raise Unavailable("configuration private proof object membership changed")
    report = {
        "schema": "merlin.configuration-elision.checked.v1",
        "status": "checked",
        "candidate": candidate.describe(),
        "source_identity": candidate.identity,
        "source_request": json.loads(candidate.request_json),
        "sources": json.loads(candidate.sources_json),
        "buffers": snapshots,
        "mapping_records": mapping_records,
        "query": query_ref,
        "object": file_ref(obj),
        "proof_sources": sources,
        "tool": tool,
        "auditor": auditor_ref,
        "imports": imports,
        "library_root": str(library),
        "library": library_refs,
        "process_evidence": [
            file_ref(path)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.name.endswith((".stdout", ".stderr", ".process.json", ".intent.json"))
        ],
        "audit": audit,
        "statements": statements,
        "relation": relation,
        "interpretation_is_conditional": True,
        "independently_qualified": False,
        "hardware_model_approved": False,
        "production_enforcement": False,
    }
    path = directory / "REPORT.json"
    path.write_text(canonical(report))
    return report, file_ref(path)
