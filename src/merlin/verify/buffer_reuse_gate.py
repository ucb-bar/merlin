"""Development possible-success buffer gate; no production registration.

This is deliberately a different relation/completion type from scalar loop error
and terminal-world preservation. No status JSON or supplied callback authorizes
proof execution or delivery. Production registries are not modified.
"""

from __future__ import annotations

import json
import os
import subprocess
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.owned_process import run_owned

from .buffer_reuse_emit import MODULES, factor
from .buffer_reuse_intake import (
    ALLOCATOR,
    DOMAIN,
    RELATION,
    Candidate,
    canonical,
    digest,
    prepare_chain,
    revalidate,
)

# Eight fresh generated modules replace the retained crashing monolith. This
# enables actual development checking, not receipt-based approval or independent
# qualification. Authority still requires all actual compilations/audits below.
_LIVE = weakref.WeakKeyDictionary()
STD3 = {"propext", "Classical.choice", "Quot.sound"}


class Unavailable(RuntimeError):
    """Proof/process/binding unavailable; never a proved refusal."""


class NotEstablished(RuntimeError):
    """A generated obligation did not typecheck; no successful observation claim."""


@dataclass(frozen=True, eq=False)
class Completion:
    identity: str
    report: str

    def __reduce_ex__(self, protocol):
        raise TypeError("live buffer completion cannot be serialized")

    def __copy__(self):
        return Completion(self.identity, self.report)  # deliberately absent from live ledger

    def __deepcopy__(self, memo):
        return self.__copy__()


@dataclass(frozen=True)
class Result:
    status: str
    reason: str
    completion: Completion | None = None


def _ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": digest(data), "bytes": len(data)}


def _exact(ref):
    if _ref(ref["path"]) != ref:
        raise Unavailable("bound buffer source/artifact changed")


def _parts(candidate):
    if type(candidate) is not Candidate or not candidate.query_parts_json:
        raise Unavailable("exact fresh buffer query bundle required")
    parts = factor(candidate.query, candidate.roots)
    if canonical(parts) != candidate.query_parts_json:
        raise Unavailable("generated query sources/imports/roots changed")
    return parts


def _ready(candidate):
    if (
        type(candidate) is not Candidate
        or not candidate.raw_chain_json
        or candidate.domain != DOMAIN
        or candidate.relation != RELATION
        or candidate.allocator != ALLOCATOR
        or type(candidate.index_bits) is not int
        or candidate.index_bits != 64
    ):
        raise Unavailable("exact actual4 possible-success request required")
    _parts(candidate)


def _run(command, cwd, env, directory, label, timeout):
    try:
        proc = run_owned(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        for suffix, value in (("stdout", getattr(exc, "stdout", "")), ("stderr", getattr(exc, "stderr", ""))):
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            (directory / (label + "." + suffix)).write_text(value or "")
        (directory / (label + ".failure.json")).write_text(
            canonical(
                {
                    "command": command,
                    "reason": str(exc),
                    "status": "unavailable",
                    "cleanup_notes": getattr(exc, "__notes__", []),
                }
            )
        )
        raise Unavailable(f"{label} unavailable: {exc}") from exc
    (directory / (label + ".stdout")).write_text(proc.stdout)
    (directory / (label + ".stderr")).write_text(proc.stderr)
    (directory / (label + ".process.json")).write_text(canonical({"command": command, "returncode": proc.returncode}))
    if proc.returncode < 0:
        raise Unavailable(f"{label} signal {-proc.returncode}; first output retained")
    return proc


def _ordered_sources(candidate):
    base = Path(__file__).parent / "lean"
    bound = {r["path"] for r in json.loads(candidate.sources_json) if r["path"].endswith(".lean")}
    seen, ordered = set(), []

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        if not name.startswith("Merlin.") or any(not p.isidentifier() for p in name.split(".")):
            raise Unavailable("invalid selected project module")
        source = base / (name.replace(".", "/") + ".lean")
        if str(source.resolve()) not in bound:
            raise Unavailable("unbound current proof source")
        for line in source.read_text().splitlines():
            words = line.split()
            if words and words[0] == "import":
                for dep in words[1:]:
                    if dep.startswith("Merlin."):
                        visit(dep)
        ordered.append((name, source))

    for line in candidate.query.splitlines():
        if line.startswith("import "):
            for name in line.split()[1:]:
                visit(name)
    return ordered


def _audit(audit, statements, candidate, modules):
    if (
        type(audit) is not dict
        or audit.get("violations") != []
        or audit.get("assurance") != "lean_kernel_std3_no_native_reflection"
        or type(audit.get("theorems_audited")) is not int
        or audit["theorems_audited"] <= 0
        or type(audit.get("axioms_used")) is not list
        or any(type(a) is not str for a in audit["axioms_used"])
        or not set(audit["axioms_used"]) <= STD3
    ):
        raise Unavailable("incomplete/refused std3 audit")
    if (
        type(statements) is not dict
        or set(statements) != set(candidate.roots)
        or any(type(v) is not str or not v.strip() for v in statements.values())
    ):
        raise Unavailable("missing/changed complete root/type roster")
    rows = audit.get("imports")
    if type(rows) is not list or any(
        type(r) is not dict or set(r) != {"module", "olean"} or any(type(v) is not str or not v for v in r.values())
        for r in rows
    ):
        raise Unavailable("malformed import audit")
    names = [r["module"] for r in rows]
    if len(names) != len(set(names)) or {n for n in names if n.startswith("Merlin.") or n in MODULES} != modules:
        raise Unavailable("actual proof imports differ from selected closure")


def require_query_success(process):
    """Distinguish known closed proof refusals from infrastructure/unknowns.

    This reports a failed obligation, never a semantic counterexample. All raw
    output is already retained by _run. Unknown or mixed diagnostics fail closed.
    """
    if process.returncode == 0:
        return
    if process.returncode < 0:
        raise Unavailable("generated query terminated by signal; first output retained")
    if process.returncode != 1:
        raise Unavailable("generated query has unexpected error exit; first output retained")
    if (process.stderr or "").strip():
        raise Unavailable("query failed with additional stderr diagnostic; first output retained")
    diagnostic = (process.stdout or "") + "\n" + (process.stderr or "")
    lower = diagnostic.lower()
    infrastructure = (
        "maximum number of heartbeats", "maximum recursion depth", "stack overflow",
        "(deterministic) timeout", "out of memory", "cannot allocate", "failed to create thread",
        "unknown module", "unknown identifier", "unknown constant", "unknown namespace",
        "object file", "invalid header", "no executable code", "internal error",
        "uncaught exception", "interrupted", "segmentation fault", "panic",
    )
    if any(marker in lower for marker in infrastructure):
        raise Unavailable("query resource/import/runtime failure; first output retained")
    lines = diagnostic.splitlines()
    headers = []
    for index, line in enumerate(lines):
        location, marker, payload = line.partition(": error")
        if marker:
            if not payload.startswith(": "):
                raise Unavailable("query failed with unclassified/tagged error header; first output retained")
            head = payload[2:]
            fields = location.rsplit(":", 2)
            if (len(fields) != 3 or not fields[0].endswith(".lean")
                    or not fields[1].isdigit() or not fields[2].isdigit()):
                raise Unavailable("query failed with unclassified/mixed error location; first output retained")
            headers.append((index, head))
        elif line.startswith("error"):
            raise Unavailable("query failed with unclassified/mixed error header; first output retained")
    if not headers:
        raise Unavailable("query failed with unclassified diagnostic; first output retained")
    for index, (line_index, head) in enumerate(headers):
        stop = headers[index+1][0] if index+1 < len(headers) else len(lines)
        body = lines[line_index+1:stop]
        closed_false = (
            head == "Tactic `decide` proved that the proposition"
            and any(line.strip() == "is false" for line in body)
        )
        known = (closed_false or head.startswith("Tactic `rfl` failed:") or head == "unsolved goals"
                 or head.startswith("Type mismatch") or head.startswith("type mismatch")
                 or head.startswith("Application type mismatch") or head.startswith("application type mismatch"))
        if not known:
            raise Unavailable("query failed with unclassified/mixed diagnostic; first output retained")
    raise NotEstablished("complete possible-success buffer obligation rejected")


def _verify(candidate, *, lean, destination, timeout):
    _ready(candidate)
    revalidate(candidate)
    if (
        not candidate.raw_chain_json
        or candidate.domain != DOMAIN
        or candidate.relation != RELATION
        or candidate.allocator != ALLOCATOR
    ):
        raise Unavailable("exact actual4 possible-success request required")
    fresh = prepare_chain(tuple(json.loads(candidate.raw_chain_json)), allocator=ALLOCATOR)
    if fresh != candidate:
        raise Unavailable("stale or forged buffer preparation")
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    library = directory / "lib"
    library.mkdir()
    lean = Path(lean).resolve()
    tool = _ref(lean)
    auditor = Path(__file__).parent / "lean/Merlin/GraphStateAudit.lean"
    auditor_ref = _ref(auditor)
    env = {
        **os.environ,
        "LEAN_PATH": str(library),
        "LEAN_NUM_THREADS": "1",
        "PATH": str(lean.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    snapshots = []
    for i, text in enumerate(json.loads(candidate.raw_chain_json)):
        path = directory / f"stage{i}.mlir"
        path.write_text(text)
        snapshots.append(_ref(path))
    # Retain the complete canonical source, but never retry compiling the old
    # monolith. Each generated module is derived and bound by _parts/fresh prepare.
    query = directory / "CanonicalBufferQuery.lean"
    query.write_text(candidate.query)
    query_ref = _ref(query)
    sources = _ordered_sources(candidate)
    for i, (module, source) in enumerate(sources):
        obj = library / (module.replace(".", "/") + ".olean")
        obj.parent.mkdir(parents=True, exist_ok=True)
        p = _run(
            [str(lean), "-j1", "-o", str(obj), str(source)],
            source.parent.parent,
            env,
            directory,
            f"import-{i}",
            timeout,
        )
        if p.returncode:
            raise Unavailable("proof closure did not compile: " + module)
    revalidate(candidate)
    generated = []
    for i, part in enumerate(_parts(candidate)):
        source = directory / (part["module"] + ".lean")
        source.write_text(part["source"])
        reference = _ref(source)
        if reference["sha256"] != part["sha256"]:
            raise Unavailable("written generated query differs from prepared source")
        obj = library / (part["module"] + ".olean")
        checked = _run(
            [str(lean), "-j1", "-o", str(obj), str(source)],
            directory,
            env,
            directory,
            f"query-{i}-{part['module']}",
            timeout,
        )
        require_query_success(checked)
        generated.append(
            {"module": part["module"], "imports": part["imports"], "source": reference, "object": _ref(obj)}
        )
    revalidate(candidate)
    for row in generated:
        _exact(row["source"])
        _exact(row["object"])
    # Freeze the entire freshly built private tree before the actual auditor
    # imports it. A changed object cannot acquire its expected hash after audit.
    if any(path.is_symlink() for path in library.rglob("*")):
        raise Unavailable("private proof objects must not be symlinks")
    library_refs = [_ref(path) for path in sorted(library.rglob("*")) if path.is_file()]
    values = {}
    for label, args in (
        ("audit", list(MODULES)),
        ("types", ["--statements", ",".join(candidate.roots), *MODULES]),
    ):
        p = _run([str(lean), "--run", str(auditor), *args], directory, env, directory, label, timeout)
        if p.returncode:
            raise Unavailable(label + " process did not establish proof evidence")
        try:
            values[label] = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise Unavailable("malformed actual audit output") from exc
    if any(type(value) is not dict for value in values.values()):
        raise Unavailable("malformed actual audit object")
    audit, types = values["audit"], values["types"].get("statements")
    _audit(audit, types, candidate, {n for n, _ in sources} | set(MODULES))
    for row in audit["imports"]:
        if row["module"].startswith("Merlin.") or row["module"] in MODULES:
            expected = library / (row["module"].replace(".", "/") + ".olean")
            if Path(row["olean"]).resolve() != expected.resolve():
                raise Unavailable("selected proof import escaped new object tree")
    imports = [_ref(row["olean"]) for row in audit["imports"]]
    revalidate(candidate)
    if any(p.is_symlink() for p in library.rglob("*")):
        raise Unavailable("private proof objects must not be symlinks")
    for ref in (
        tool,
        auditor_ref,
        query_ref,
        *snapshots,
        *(row["source"] for row in generated),
        *(row["object"] for row in generated),
        *imports,
        *library_refs,
    ):
        _exact(ref)
    report = {
        "schema": "merlin.buffer-reuse.checked.v2",
        "status": "checked",
        "kernel_outcome": "checked",
        "qualification": "development_only_unqualified",
        "candidate": candidate.describe(),
        "source_identity": candidate.identity,
        "sources": json.loads(candidate.sources_json),
        "actual4": snapshots,
        "query": query_ref,
        "generated_queries": generated,
        "audit_modules": list(MODULES),
        "imports": imports,
        "object": _ref(obj),
        "tool": tool,
        "auditor": auditor_ref,
        "library_root": str(library),
        "library": library_refs,
        "process_evidence": [
            _ref(p)
            for p in sorted(directory.iterdir())
            if p.is_file() and p.name.endswith((".stdout", ".stderr", ".process.json"))
        ],
        "audit": audit,
        "statements": types,
        "relation": RELATION,
        "allocator": ALLOCATOR,
        "typed_caller_domain_is_premise": True,
        "full_error_world_equivalence": False,
        "native_allocator_conformance": False,
        "independent_review_approved": False,
        "independently_qualified": False,
        "production_enforcement": False,
    }
    path = directory / "REPORT.json"
    path.write_text(canonical(report))
    token = Completion(candidate.identity, str(path))
    _LIVE[token] = (candidate, _ref(path), report)
    validate_completion(token, candidate)
    return Result("checked", "complete actual4 possible-success query and audit checked", token)


def check(candidate, *, lean, destination, timeout=120):
    """Check every generated module/root; status records never grant authority."""
    try:
        _ready(candidate)
        if type(candidate) is not Candidate:
            raise Unavailable("exact buffer Candidate required")
        return _verify(candidate, lean=lean, destination=destination, timeout=timeout)
    except NotEstablished as exc:
        return Result("not_established", str(exc))
    except (Unavailable, OSError, ValueError, TypeError, KeyError) as exc:
        return Result("unavailable", str(exc))


def validate_completion(token, candidate):
    if type(token) is not Completion or token not in _LIVE:
        raise Unavailable("saved/constructed/copied completion has no live authority")
    _ready(candidate)
    original, reference, report = _LIVE[token]
    if original is not candidate or token.identity != candidate.identity or _ref(token.report) != reference:
        raise Unavailable("live candidate/report changed")
    revalidate(candidate)
    if any(p.is_symlink() for p in Path(report["library_root"]).rglob("*")):
        raise Unavailable("private proof object symlink introduced")
    current_objects = {str(p.resolve()) for p in Path(report["library_root"]).rglob("*") if p.is_file()}
    if current_objects != {ref["path"] for ref in report["library"]}:
        raise Unavailable("private proof object tree membership changed")
    for ref in (
        *report["actual4"],
        report["query"],
        report["object"],
        report["tool"],
        report["auditor"],
        *report["imports"],
        *(row["source"] for row in report["generated_queries"]),
        *(row["object"] for row in report["generated_queries"]),
        *report["library"],
        *report["process_evidence"],
    ):
        _exact(ref)
    return True


def deliver(token, candidate, consumer):
    validate_completion(token, candidate)
    return consumer(candidate.final)


def replay(report_path, *, lean, destination, timeout=120):
    try:
        recorded = json.loads(Path(report_path).read_text())
        refs = recorded["actual4"]
        if len(refs) != 4:
            raise Unavailable("exact4 replay buffers required")
        for ref in refs:
            _exact(ref)
        candidate = prepare_chain(tuple(Path(r["path"]).read_text() for r in refs), allocator=ALLOCATOR)
        if candidate.identity != recorded["source_identity"] or candidate.describe() != recorded["candidate"]:
            raise Unavailable("reconstructed replay differs from recorded request")
        return check(candidate, lean=lean, destination=destination, timeout=timeout)
    except (Unavailable, OSError, ValueError, TypeError, KeyError) as exc:
        return Result("unavailable", str(exc))
