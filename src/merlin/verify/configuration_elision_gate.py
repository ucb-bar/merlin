"""Fresh kernel/audit/live-consumer path for bounded configuration elimination.

No production registration or human/hardware model approval is issued. Saved
status JSON, caller callbacks and copied tokens have no delivery authority.
"""

from __future__ import annotations

import json
import os
import subprocess
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.owned_process import run_owned

from .configuration_elision_bridge import canonical, file_ref
from .configuration_elision_emit import MODULE, emit
from .configuration_elision_intake import DOMAIN, RELATION, Candidate, prepare, proof_sources, revalidate

STD3 = {"propext", "Classical.choice", "Quot.sound"}
_LIVE = weakref.WeakKeyDictionary()


class Unavailable(RuntimeError):
    """Infrastructure, unsupported/stale evidence or incomplete proof execution."""


class NotEstablished(RuntimeError):
    """A known closed proof obligation was rejected; never a successful proof."""


@dataclass(frozen=True, eq=False)
class Completion:
    identity: str
    report: str

    def __reduce_ex__(self, protocol):
        raise TypeError("live configuration completion cannot be serialized")

    def __copy__(self):
        return Completion(self.identity, self.report)

    def __deepcopy__(self, memo):
        return self.__copy__()


@dataclass(frozen=True)
class Result:
    status: str
    reason: str
    completion: Completion | None = None


def _exact(row):
    if file_ref(row["path"]) != row:
        raise Unavailable("bound configuration source/tool/artifact changed")


def _ready(candidate):
    if type(candidate) is not Candidate or candidate.domain != DOMAIN or candidate.relation != RELATION:
        raise Unavailable("exact current complete-entry configuration request required")
    generated = emit(json.loads(candidate.projection_json))
    if candidate.query != generated["query"] or list(candidate.roots) != generated["roots"]:
        raise Unavailable("configuration query/complete root roster changed")
    if json.loads(candidate.proof_sources_json) != proof_sources():
        raise Unavailable("canonical configuration proof closure changed or unavailable")


def _run(command, cwd, env, directory, label, timeout):
    (directory / (label + ".intent.json")).write_text(
        canonical({"command": command, "cwd": str(cwd), "timeout": timeout})
    )
    try:
        process = run_owned(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        for suffix, value in (("stdout", getattr(error, "stdout", "")), ("stderr", getattr(error, "stderr", ""))):
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            (directory / (label + "." + suffix)).write_text(value or "")
        (directory / (label + ".failure.json")).write_text(
            canonical({"reason": str(error), "cleanup": getattr(error, "__notes__", []), "status": "unavailable"})
        )
        raise Unavailable(label + " unavailable; first raw output retained") from error
    (directory / (label + ".stdout")).write_text(process.stdout)
    (directory / (label + ".stderr")).write_text(process.stderr)
    (directory / (label + ".process.json")).write_text(
        canonical({"returncode": process.returncode, "command": command})
    )
    if process.returncode < 0:
        raise Unavailable(label + " signal " + str(-process.returncode) + "; first raw output retained")
    return process


def require_query_success(process):
    if process.returncode == 0:
        if process.stderr.strip():
            raise Unavailable("configuration query unexpected stderr; first raw output retained")
        return
    if process.returncode != 1 or process.stderr.strip():
        raise Unavailable("configuration query unexpected exit/stderr; first raw output retained")
    lines = process.stdout.splitlines()
    headers = []
    for index, line in enumerate(lines):
        location, marker, payload = line.partition(": error")
        if marker:
            fields = location.rsplit(":", 2)
            if (
                not payload.startswith(": ")
                or len(fields) != 3
                or not fields[0].endswith(".lean")
                or not fields[1].isdigit()
                or not fields[2].isdigit()
            ):
                raise Unavailable("unclassified configuration proof error header")
            headers.append((index, payload[2:]))
        elif line.startswith("error"):
            raise Unavailable("unclassified configuration proof error location")
    if not headers:
        raise Unavailable("unclassified configuration query failure; first raw output retained")
    for i, (index, head) in enumerate(headers):
        stop = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
        if head != "Tactic `decide` proved that the proposition" or not any(
            line.strip() == "is false" for line in lines[index + 1 : stop]
        ):
            raise Unavailable("configuration query unknown/mixed/resource diagnostic; first raw output retained")
    raise NotEstablished("closed complete configuration obligation rejected")


def _audit(audit, statements, candidate, modules, *, query_module=MODULE):
    if (
        type(audit) is not dict
        or audit.get("violations") != []
        or audit.get("assurance") != "lean_kernel_std3_no_native_reflection"
        or type(audit.get("theorems_audited")) is not int
        or audit["theorems_audited"] <= 0
        or type(audit.get("axioms_used")) is not list
        or any(type(value) is not str for value in audit["axioms_used"])
        or not set(audit["axioms_used"]) <= STD3
    ):
        raise Unavailable("incomplete/refused configuration std3 audit")
    if (
        type(statements) is not dict
        or set(statements) != set(candidate.roots)
        or any(type(value) is not str or not value.strip() for value in statements.values())
    ):
        raise Unavailable("missing/changed complete configuration root/type roster")
    imports = audit.get("imports")
    if type(imports) is not list or any(
        type(row) is not dict
        or set(row) != {"module", "olean"}
        or any(type(value) is not str or not value for value in row.values())
        for row in imports
    ):
        raise Unavailable("malformed configuration import audit")
    names = [row["module"] for row in imports]
    if (
        len(names) != len(set(names))
        or {name for name in names if name.startswith("Merlin.") or name == query_module} != modules
    ):
        raise Unavailable("actual configuration proof imports differ from selected closure")


def _verify(candidate, *, lean, destination, timeout):
    _ready(candidate)
    revalidate(candidate)
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    library = directory / "lib"
    library.mkdir()
    executable = Path(lean).resolve()
    tool = file_ref(executable)
    auditor = Path(__file__).parent / "lean/Merlin/GraphStateAudit.lean"
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
    query = directory / (MODULE + ".lean")
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
    sources = proof_sources()
    for i, row in enumerate(sources):
        obj = library / (row["module"].replace(".", "/") + ".olean")
        obj.parent.mkdir(parents=True, exist_ok=True)
        process = _run(
            [str(executable), "-j1", "-o", str(obj), row["path"]],
            Path(__file__).parent / "lean",
            env,
            directory,
            "import-" + str(i),
            timeout,
        )
        if process.returncode:
            raise Unavailable("configuration project proof source did not compile: " + row["module"])
    revalidate(candidate)
    obj = library / (MODULE + ".olean")
    process = _run([str(executable), "-j1", "-o", str(obj), str(query)], directory, env, directory, "query", timeout)
    require_query_success(process)
    revalidate(candidate)
    if any(path.is_symlink() for path in library.rglob("*")):
        raise Unavailable("configuration private proof objects must not be symlinks")
    library_refs = [file_ref(path) for path in sorted(library.rglob("*")) if path.is_file()]
    values = {}
    for label, args in (("audit", [MODULE]), ("types", ["--statements", ",".join(candidate.roots), MODULE])):
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
    modules = {row["module"] for row in sources} | {MODULE}
    _audit(audit, statements, candidate, modules)
    imports = []
    for row in audit["imports"]:
        if row["module"] in modules:
            expected = library / (row["module"].replace(".", "/") + ".olean")
            if Path(row["olean"]).resolve() != expected.resolve():
                raise Unavailable("configuration project import escaped fresh private object tree")
        imports.append(file_ref(row["olean"]))
    revalidate(candidate)
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
        "relation": RELATION,
        "interpretation_is_conditional": True,
        "independently_qualified": False,
        "hardware_model_approved": False,
        "production_enforcement": False,
    }
    path = directory / "REPORT.json"
    path.write_text(canonical(report))
    token = Completion(candidate.identity, str(path))
    _LIVE[token] = (candidate, file_ref(path), report)
    validate_completion(token, candidate)
    return Result("checked", "complete conditional configuration entry obligation and std3/root audit checked", token)


def check(candidate, *, lean, destination, timeout=120):
    try:
        return _verify(candidate, lean=lean, destination=destination, timeout=timeout)
    except NotEstablished as error:
        return Result("not_established", str(error))
    except (Unavailable, OSError, ValueError, TypeError, KeyError) as error:
        return Result("unavailable", str(error))


def validate_completion(token, candidate):
    if type(token) is not Completion or token not in _LIVE:
        raise Unavailable("saved/constructed/copied configuration completion has no authority")
    _ready(candidate)
    original, ref, report = _LIVE[token]
    if original is not candidate or token.identity != candidate.identity or file_ref(token.report) != ref:
        raise Unavailable("live configuration candidate/report changed")
    revalidate(candidate)
    library = Path(report["library_root"])
    if any(path.is_symlink() for path in library.rglob("*")):
        raise Unavailable("configuration private proof object symlink introduced")
    if {str(path.resolve()) for path in library.rglob("*") if path.is_file()} != {
        row["path"] for row in report["library"]
    }:
        raise Unavailable("configuration private proof object membership changed")
    for row in [
        *report["buffers"],
        *report["mapping_records"],
        report["query"],
        report["object"],
        report["tool"],
        report["auditor"],
        *report["imports"],
        *report["library"],
        *report["process_evidence"],
    ]:
        _exact(row)
    return True


def deliver(token, candidate, consumer):
    validate_completion(token, candidate)
    return consumer(candidate.final)


def replay(report_path, *, lean, destination, timeout=120):
    try:
        report = json.loads(Path(report_path).read_text())
        if report.get("schema") != "merlin.configuration-elision.checked.v1" or len(report["buffers"]) != 2:
            raise Unavailable("complete configuration replay record required")
        for row in report["buffers"]:
            _exact(row)
        candidate = prepare(*(Path(row["path"]).read_text() for row in report["buffers"]), **report["source_request"])
        if candidate.identity != report["source_identity"] or candidate.describe() != report["candidate"]:
            raise Unavailable("fresh configuration reconstruction differs from replay request")
        return check(candidate, lean=lean, destination=destination, timeout=timeout)
    except (Unavailable, OSError, ValueError, TypeError, KeyError) as error:
        return Result("unavailable", str(error))
