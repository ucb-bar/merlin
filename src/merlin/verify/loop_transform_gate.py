"""Direct scoped loop consumer: fresh kernel proof before final IR delivery.

Ordinary selected Lean execution only. Python projection/process/host transport
and fidelity of the explicit operational interpretation remain trusted premises.
A serialized report is never a live completion and cannot authorize delivery.
"""

from __future__ import annotations

import json
import os
import subprocess
import weakref
from dataclasses import dataclass
from pathlib import Path

from merlin.common.owned_process import run_owned

from .loop_transform_intake import (
    ROOT,
    PreparedLoop,
    Settings,
    Unavailable,
    canonical,
    file_ref,
    prepare,
    produce,
    revalidate,
)

STD3 = {"propext", "Classical.choice", "Quot.sound"}
RELATION = "all defined entries equal returned values/errors/terminal world"
PREMISES = (
    "trusted full-IR Python projection; common total pure primitives; local opaque result arity; explicit index model"
)
_LIVE = weakref.WeakKeyDictionary()


class NotEstablished(RuntimeError):
    """Lean ran and rejected the generated proof; no consumer delivery."""


@dataclass(frozen=True, eq=False)
class Completion:
    identity: str
    report: str


def _run(command, cwd, env, directory, label, timeout):
    try:
        proc = run_owned(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        (directory / (label + ".failure")).write_text(str(exc))
        for suffix, value in (("stdout", getattr(exc, "stdout", "")), ("stderr", getattr(exc, "stderr", ""))):
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            (directory / (label + "." + suffix)).write_text(value or "")
        raise Unavailable(f"{label} execution unavailable: {exc}") from exc
    (directory / (label + ".stdout")).write_text(proc.stdout)
    (directory / (label + ".stderr")).write_text(proc.stderr)
    (directory / (label + ".process.json")).write_text(canonical({"command": command, "returncode": proc.returncode}))
    if proc.returncode < 0:
        raise Unavailable(f"{label} terminated by signal {-proc.returncode}")
    return proc


def _closure(prepared):
    root = Path(__file__).parent / "lean"
    ordered, seen = [], set()
    sources = {row["path"] for row in json.loads(prepared.sources_json) if row["path"].endswith(".lean")}

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        path = root / (name.replace(".", "/") + ".lean")
        if str(path.resolve()) not in sources:
            raise Unavailable("unbound project proof import")
        for line in path.read_text().splitlines():
            fields = line.split()
            if fields and fields[0] == "import":
                for dep in fields[1:]:
                    if dep.startswith("Merlin."):
                        visit(dep)
        ordered.append((name, path))

    for line in prepared.query.splitlines():
        if line.startswith("import "):
            for name in line.split()[1:]:
                if name.startswith("Merlin."):
                    visit(name)
    return ordered


def validate_audits(audit, statements, roots, expected_modules):
    """Pure schema/roster consistency, never authority without an actual process."""
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
        raise Unavailable("transitive std3 audit failed")
    if (
        type(statements) is not dict
        or set(statements) != set(roots)
        or ROOT not in statements
        or any(type(value) is not str or not value.strip() for value in statements.values())
    ):
        raise Unavailable("complete exact root/type roster missing")
    rows = audit.get("imports")
    if type(rows) is not list or any(
        type(row) is not dict
        or set(row) != {"module", "olean"}
        or any(type(value) is not str or not value for value in row.values())
        for row in rows
    ):
        raise Unavailable("malformed actual import roster")
    names = [row["module"] for row in rows]
    if (
        len(set(names)) != len(names)
        or {n for n in names if n.startswith("Merlin.") or n == "LoopTransformQuery"} != expected_modules
    ):
        raise Unavailable("actual project imports differ from complete compiled closure")


def require_query_success(process):
    if not process.returncode:
        return
    diagnostic = process.stdout + process.stderr
    if (
        "maximum number of heartbeats (" in diagnostic
        and ") has been reached" in diagnostic
        or "maximum recursion depth has been reached" in diagnostic
    ):
        raise Unavailable("Lean proof-resource exhaustion; original limit/output retained")
    raise NotEstablished("complete loop query did not typecheck")


def _library_refs(library):
    library = Path(library)
    paths = sorted(library.rglob("*"))
    if library.is_symlink() or any(path.is_symlink() for path in paths):
        raise Unavailable("private proof objects must not be symlinks")
    return [file_ref(path) for path in paths if path.is_file()]


def _validate_library(library, references):
    if _library_refs(library) != references:
        raise Unavailable("private proof object bytes or tree membership changed")


def check(prepared, *, lean, destination, timeout=120):
    """Compile exact selected project closure/query, audit roots, issue live handle.

    destination must be NEW. No historical object/receipt/caller callback supplies
    authority. F qualifies this same code path in a separate private output tree.
    """
    from .compiler_loop_intake import PreparedCompilerLoop, refresh

    if type(prepared) is PreparedLoop:
        fresh = prepare(prepared.buffers, reference=prepared.reference, settings=prepared.settings)
    elif type(prepared) is PreparedCompilerLoop:
        fresh = refresh(prepared)
    else:
        raise Unavailable("exact prepared request required")
    if fresh != prepared:
        raise Unavailable("stale or forged preparation/query/roots/domain")
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    library = directory / "lib"
    library.mkdir()
    lean = Path(lean).resolve()
    tool = file_ref(lean)
    audit_source = Path(__file__).parent / "lean/Merlin/GraphStateAudit.lean"
    auditor = file_ref(audit_source)
    env = {
        **os.environ,
        "LEAN_PATH": str(library),
        "LEAN_NUM_THREADS": "1",
        "PATH": str(lean.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    snapshots = []
    for i, text in enumerate(prepared.buffers):
        path = directory / f"stage{i}.mlir"
        path.write_text(text)
        snapshots.append(file_ref(path))
    query = directory / "LoopTransformQuery.lean"
    query.write_text(prepared.query)
    query_ref = file_ref(query)
    for i, (module, source) in enumerate(_closure(prepared)):
        target = library / (module.replace(".", "/") + ".olean")
        target.parent.mkdir(parents=True, exist_ok=True)
        process = _run(
            [str(lean), "-j1", "-o", str(target), str(source)],
            source.parent.parent,
            env,
            directory,
            f"import-{i}",
            timeout,
        )
        if process.returncode:
            raise Unavailable(f"project proof dependency failed: {module}")
    obj = library / "LoopTransformQuery.olean"
    process = _run([str(lean), "-j1", "-o", str(obj), str(query)], directory, env, directory, "query", timeout)
    require_query_success(process)
    # Freeze bytes and membership before either auditor imports the objects.
    # Audit-time replacement cannot become the expected object after the fact.
    library_refs = _library_refs(library)
    # The canonical existing auditor checks all imported project theorems and
    # forbids unsafe/extern/axiom/reflection overrides; no synthetic Boolean test.
    audits = {}
    for label, args in (
        ("audit", ["LoopTransformQuery"]),
        ("types", ["--statements", ",".join(prepared.roots), "LoopTransformQuery"]),
    ):
        process = _run([str(lean), "--run", str(audit_source), *args], directory, env, directory, label, timeout)
        if process.returncode:
            raise Unavailable(f"{label} could not establish exact audit output")
        try:
            audits[label] = json.loads(process.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise Unavailable("malformed root audit") from exc
    audit, statements = audits["audit"], audits["types"].get("statements")
    expected_modules = {module for module, _ in _closure(prepared)} | {"LoopTransformQuery"}
    validate_audits(audit, statements, prepared.roots, expected_modules)
    imports = []
    for row in audit["imports"]:
        if row["module"].startswith("Merlin.") or row["module"] == "LoopTransformQuery":
            path = Path(row["olean"]).resolve()
            expected = library / (row["module"].replace(".", "/") + ".olean")
            if path != expected.resolve():
                raise Unavailable("project proof object escaped fresh private build")
            imports.append(file_ref(path))
    _revalidate(prepared)
    _validate_library(library, library_refs)
    if file_ref(lean) != tool or file_ref(audit_source) != auditor or file_ref(query) != query_ref:
        raise Unavailable("tool/source/query changed during proof")
    for ref in [*snapshots, *imports, *library_refs]:
        if file_ref(ref["path"]) != ref:
            raise Unavailable("checked artifact changed during proof")
    report = {
        "schema": "merlin.loop-transform.checked.v1",
        "status": "checked",
        "prepared": prepared.describe(),
        "sources": json.loads(prepared.sources_json),
        "buffers": snapshots,
        "query": query_ref,
        "object": file_ref(obj),
        "lean": tool,
        "auditor": auditor,
        "imports": imports,
        "library_root": str(library),
        "built_artifacts": library_refs,
        "audit": audit,
        "statements": statements,
        "scope": "actual compiler loop boundary"
        if type(prepared) is PreparedCompilerLoop
        else "scoped direct loop consumer; not compiler-wide enforcement",
        "relation": RELATION,
        "premises": PREMISES,
        "general_tiling": False,
        "native_fidelity": False,
        "human_approved": False,
    }
    report_path = directory / "REPORT.json"
    report_path.write_text(canonical(report))
    token = Completion(prepared.identity, str(report_path))
    _LIVE[token] = (prepared, file_ref(report_path), report)
    validate_completion(token, prepared)
    return token


def validate_completion(completion, prepared):
    if type(completion) is not Completion or completion not in _LIVE:
        raise Unavailable("missing/forged/saved completion has no live authority")
    original, report_ref, report = _LIVE[completion]
    if (
        original is not prepared
        or completion.identity != prepared.identity
        or file_ref(completion.report) != report_ref
    ):
        raise Unavailable("live completion/request/report changed")
    _revalidate(prepared)
    _validate_library(report["library_root"], report["built_artifacts"])
    for ref in [
        *report["buffers"],
        report["query"],
        report["object"],
        report["lean"],
        report["auditor"],
        *report["imports"],
        *report["built_artifacts"],
    ]:
        if file_ref(ref["path"]) != ref:
            raise Unavailable("checked artifact changed before delivery")
    return True


def _revalidate(prepared):
    from .compiler_loop_intake import PreparedCompilerLoop
    from .compiler_loop_intake import revalidate as compiler_revalidate

    if type(prepared) is PreparedCompilerLoop:
        compiler_revalidate(prepared)
    else:
        revalidate(prepared)


def deliver(completion, prepared, consumer):
    validate_completion(completion, prepared)
    from .compiler_loop_intake import PreparedCompilerLoop

    if type(prepared) is PreparedCompilerLoop:
        return consumer(prepared.actual_target)
    return consumer(prepared.buffers[-1])


def transform(source, *, reference, lean, destination, consumer, settings=Settings()):
    """Actual pinned producer + fresh checker + guarded direct consumer."""
    buffers = produce(source, reference=reference, settings=settings)
    prepared = prepare(buffers, reference=reference, settings=settings)
    completion = check(prepared, lean=lean, destination=destination)
    return deliver(completion, prepared, consumer)


def replay(report_path, *, lean, destination, reference=None, consumer=None, timeout=120):
    """Rebuild the exact scoped workflow; saved status grants no authority."""
    recorded = json.loads(Path(report_path).read_text())
    refs = recorded["buffers"]
    for ref in refs:
        if file_ref(ref["path"]) != ref:
            raise Unavailable("replay input snapshot changed")
    settings = Settings(**recorded["prepared"]["settings"])
    buffers = tuple(Path(r["path"]).read_text() for r in refs)
    if recorded["prepared"].get("schema") == "merlin.compiler-loop.prepared.v1":
        from .compiler_loop_intake import TILE, UNROLL
        from .compiler_loop_intake import prepare as compiler_prepare

        kinds = tuple(recorded["prepared"]["kinds"])
        if kinds == (UNROLL,):
            buffers = buffers[:2]
        elif kinds == (TILE,):
            buffers = (buffers[0], buffers[2])
        prepared = compiler_prepare(
            buffers, kinds=kinds, factor=max(2, settings.factor), tile=settings.tile, index_bits=settings.index_bits
        )
    else:
        if reference is None:
            raise Unavailable("standalone loop replay requires its explicit producer reference")
        prepared = prepare(buffers, reference=reference, settings=settings)
    if canonical(prepared.describe()) != canonical(recorded["prepared"]):
        raise Unavailable("reconstructed replay identity differs")
    completion = check(prepared, lean=lean, destination=destination, timeout=timeout)
    validate_completion(completion, prepared)
    if consumer is not None:
        deliver(completion, prepared, consumer)
    return completion
