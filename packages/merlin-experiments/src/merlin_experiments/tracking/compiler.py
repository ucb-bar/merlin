"""What compiler the agent has built, checkpoint by checkpoint: passes, dialects, ops, entry points, LOC.

The sources are the run's own snapshots, read without touching them:

* each graded snapshot the harness committed to the run's ``oot/`` history repository, named by
  ``oot_commits.jsonl`` -- read with ``git ls-tree`` / ``git cat-file`` only (no checkout, no index, no
  optional locks; the harness's own hardened git environment);
* the final ``submission/`` copy, when the run has one.

From a snapshot's files this module lists, structurally and without regular expressions:

* the ``manifest.yaml`` entry points, command names, components and optimization surfaces;
* TableGen ``def`` records: dialects (``: Dialect`` with ``let name = "..."``), operations (a ``def`` whose
  base class is an op class and whose first template argument is a quoted mnemonic) and passes
  (``: Pass<"name"`` / ``InterfacePass<"name"``);
* Python classes, by ``ast``: a class with a ``name = "..."`` attribute that derives from a ``*Pass``
  base is a pass, one decorated ``irdl_op_definition`` or deriving from ``*Operation`` is an op; a
  ``Dialect("name", ...)`` call is a dialect;
* C++ ``getArgument()`` bodies returning a string (a pass's command-line name);
* non-blank lines per source file.

It is a reading aid: a name this parser does not recognise is simply not listed, and the per-file line
counts stay exact either way.  Results are cached per commit, so a live view parses each snapshot once.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import records as R

SOURCE_SUFFIXES = (".td", ".py", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".c", ".mlir", ".yaml", ".yml", ".txt")
#: Bound on bytes parsed per snapshot (a vendored tree must not make a live view expensive).
MAX_BYTES = 24 << 20
MAX_CHECKPOINTS = 60


# --------------------------------------------------------------------------- parsers
def _quoted(text: str) -> str | None:
    start = text.find('"')
    if start < 0:
        return None
    end = text.find('"', start + 1)
    return text[start + 1 : end] if end > start else None


def parse_td(text: str) -> dict[str, list[str]]:
    dialects, ops, passes = [], [], []
    in_dialect = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("def ") and ":" in line:
            head, _, base = line[4:].partition(":")
            base = base.strip()
            base_name = base.split("<")[0].split("{")[0].split(" ")[0].strip()
            in_dialect = base_name == "Dialect"
            if base_name in ("Pass", "InterfacePass") or base_name.endswith("Pass"):
                name = _quoted(base)
                if name:
                    passes.append(name)
            elif base_name.endswith("Op") or base_name.endswith("_Op") or "Op<" in base:
                name = _quoted(base)
                if name and "<" in base:
                    ops.append(name)
            del head
            continue
        if in_dialect and line.startswith("let name") and "=" in line:
            name = _quoted(line)
            if name:
                dialects.append(name)
            in_dialect = False
    return {"dialects": dialects, "ops": ops, "passes": passes}


def _base_names(node: ast.ClassDef) -> list[str]:
    out = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            out.append(base.id)
        elif isinstance(base, ast.Attribute):
            out.append(base.attr)
    return out


def parse_py(text: str) -> dict[str, list[str]]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {"dialects": [], "ops": [], "passes": []}
    dialects, ops, passes = [], [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            name = None
            for stmt in node.body:
                targets = (
                    stmt.targets
                    if isinstance(stmt, ast.Assign)
                    else [stmt.target]
                    if isinstance(stmt, ast.AnnAssign)
                    else []
                )
                value = getattr(stmt, "value", None)
                if (
                    any(isinstance(t, ast.Name) and t.id == "name" for t in targets)
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    name = value.value
            if not name:
                continue
            bases = _base_names(node)
            decorators = [d.id if isinstance(d, ast.Name) else getattr(d, "attr", "") for d in node.decorator_list]
            if any(b.endswith("Pass") for b in bases):
                passes.append(name)
            elif "irdl_op_definition" in decorators or any(b.endswith("Operation") for b in bases):
                ops.append(name)
        elif isinstance(node, ast.Call):
            func = node.func
            callee = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if callee == "Dialect" and node.args and isinstance(node.args[0], ast.Constant):
                if isinstance(node.args[0].value, str):
                    dialects.append(node.args[0].value)
    return {"dialects": dialects, "ops": ops, "passes": passes}


def parse_cpp(text: str) -> dict[str, list[str]]:
    passes = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "getArgument()" in line:
            for follow in lines[i : i + 4]:
                if "return" in follow:
                    name = _quoted(follow.partition("return")[2])
                    if name:
                        passes.append(name)
                    break
    return {"dialects": [], "ops": [], "passes": passes}


def parse_manifest(text: str) -> dict[str, Any]:
    from merlin.common.yaml import safe_load_text

    try:
        doc = safe_load_text(text)
    except Exception:  # noqa: BLE001 -- an unparsable manifest is reported as such
        return {"error": "manifest.yaml does not parse"}
    if not isinstance(doc, Mapping):
        return {"error": "manifest.yaml is not a mapping"}
    commands = doc.get("commands") if isinstance(doc.get("commands"), Mapping) else {}
    surfaces = [s.get("id") for s in doc.get("optimization_surfaces") or () if isinstance(s, Mapping)]
    return {
        "package_id": doc.get("package_id"),
        "language": doc.get("language"),
        "entrypoints": dict(doc.get("entrypoints") or {}) if isinstance(doc.get("entrypoints"), Mapping) else {},
        "commands": sorted(str(k) for k in commands),
        "components": sorted(str(k) for k in (doc.get("components") or {}))
        if isinstance(doc.get("components"), Mapping)
        else [],
        "surfaces": [str(s) for s in surfaces if s],
    }


def analyse(files: Mapping[str, bytes]) -> dict[str, Any]:
    """Passes, dialects, ops, manifest facts and LOC by file of one snapshot ``{path: bytes}``."""
    found = {"dialects": set(), "ops": set(), "passes": set()}
    loc: dict[str, int] = {}
    manifest = None
    for path, data in files.items():
        text = data.decode("utf-8", "replace")
        loc[path] = sum(1 for line in text.splitlines() if line.strip())
        suffix = Path(path).suffix
        parsed = None
        if suffix == ".td":
            parsed = parse_td(text)
        elif suffix == ".py":
            parsed = parse_py(text)
        elif suffix in (".cpp", ".cc", ".cxx"):
            parsed = parse_cpp(text)
        if Path(path).name == "manifest.yaml" and (manifest is None or path.count("/") < manifest[0].count("/")):
            manifest = (path, parse_manifest(text))
        if parsed:
            for key in found:
                found[key].update(parsed[key])
    return {
        "dialects": sorted(found["dialects"]),
        "ops": sorted(found["ops"]),
        "passes": sorted(found["passes"]),
        "manifest": manifest[1] if manifest else None,
        "loc": dict(sorted(loc.items())),
        "loc_total": sum(loc.values()),
        "files": len(files),
    }


def diff(before: Mapping[str, Any] | None, after: Mapping[str, Any]) -> dict[str, Any]:
    """What changed from one snapshot's analysis to the next."""
    if before is None:
        return {"first": True}
    out: dict[str, Any] = {}
    for key in ("passes", "ops", "dialects"):
        b, a = set(before[key]), set(after[key])
        out[key] = {"added": sorted(a - b), "removed": sorted(b - a)}
    bl, al = before["loc"], after["loc"]
    out["files_added"] = sorted(set(al) - set(bl))
    out["files_removed"] = sorted(set(bl) - set(al))
    changed = {p: al[p] - bl[p] for p in set(al) & set(bl) if al[p] != bl[p]}
    out["files_changed"] = dict(sorted(changed.items(), key=lambda kv: -abs(kv[1]))[:20])
    out["loc_delta"] = after["loc_total"] - before["loc_total"]
    return out


# --------------------------------------------------------------------------- snapshot readers
def _is_source(path: str) -> bool:
    name = Path(path).name
    return Path(path).suffix in SOURCE_SUFFIXES or name in ("CMakeLists.txt", "Makefile")


def read_commit(repo: Path, commit: str) -> dict[str, bytes] | None:
    """``{path: bytes}`` of the source files in one commit, through git's object reader only."""
    from merlin.common import oot_repo as O

    env = {"GIT_OPTIONAL_LOCKS": "0"}
    try:
        listing = O._git(repo, "--no-optional-locks", "ls-tree", "-r", "-l", "-z", commit, env=env)
    except Exception:  # noqa: BLE001 -- an unreadable commit is reported, never fatal
        return None
    wanted: list[tuple[str, str]] = []
    total = 0
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        meta, _, path = entry.partition(b"\t")
        parts = meta.split()
        if len(parts) < 4 or parts[1] != b"blob":
            continue
        name = path.decode("utf-8", "surrogateescape")
        size = int(parts[3]) if parts[3].isdigit() else 0
        if not _is_source(name) or total + size > MAX_BYTES:
            continue
        total += size
        wanted.append((parts[2].decode(), name))
    if not wanted:
        return {}
    request = "".join(f"{sha}\n" for sha, _ in wanted).encode()
    try:
        blob = O._git(repo, "--no-optional-locks", "cat-file", "--batch", input=request, env=env)
    except Exception:  # noqa: BLE001
        return None
    out: dict[str, bytes] = {}
    position = 0
    for sha, name in wanted:
        end = blob.find(b"\n", position)
        if end < 0:
            break
        header = blob[position:end].split()
        if len(header) < 3 or header[1] == b"missing":
            position = end + 1
            continue
        size = int(header[2])
        out[name] = blob[end + 1 : end + 1 + size]
        position = end + 1 + size + 1
        del sha
    return out


def read_tree(root: Path) -> dict[str, bytes]:
    """``{relative path: bytes}`` of the source files under a snapshot directory (bounded)."""
    out: dict[str, bytes] = {}
    total = 0
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part.startswith(".") or part in ("build", "__pycache__") for part in Path(rel).parts):
            continue
        if not _is_source(rel):
            continue
        try:
            size = path.stat().st_size
            if total + size > MAX_BYTES:
                continue
            out[rel] = path.read_bytes()
            total += size
        except OSError:
            continue
    return out


def evolution(
    run_dir: Path,
    commits: Sequence[Mapping[str, Any]] | None,
    inventory: R.Inventory,
    cache: dict[str, Any],
) -> dict[str, Any]:
    """Per checkpoint: what the snapshot holds and what changed since the previous one."""
    run_dir = Path(run_dir)
    repo = run_dir / "oot"
    checkpoints: list[dict[str, Any]] = []
    rows = [c for c in commits or () if c.get("commit")]
    if len(rows) > MAX_CHECKPOINTS:
        step = len(rows) / (MAX_CHECKPOINTS - 1)
        keep = sorted({int(i * step) for i in range(MAX_CHECKPOINTS - 1)} | {len(rows) - 1})
        rows = [rows[i] for i in keep]
    if rows and not (repo / "HEAD").exists() and not (repo / ".git").exists():
        inventory.note("oot/ (snapshot history repository)", repo, "absent")
    elif rows:
        inventory.note("oot/ (snapshot history repository)", repo, "read", f"{len(rows)} checkpoints parsed")
    for row in rows:
        commit = str(row["commit"])
        if commit not in cache:
            files = read_commit(repo, commit)
            cache[commit] = analyse(files) if files is not None else None
        checkpoints.append(
            {
                "source": "oot commit",
                "label": f"{row.get('label') or ''} {row.get('key') or ''}".strip(),
                "commit": commit,
                "at": row.get("at"),
                "n_passed": row.get("n_passed"),
                "n_capsules": row.get("n_capsules"),
                "analysis": cache[commit],
            }
        )
    submission = run_dir / "submission"
    if submission.is_dir():
        key = f"tree:{submission}"
        stamp = max((p.stat().st_mtime for p in submission.rglob("*") if p.is_file()), default=None)
        if cache.get(key, (None,))[0] != stamp:
            cache[key] = (stamp, analyse(read_tree(submission)))
        checkpoints.append(
            {
                "source": "submission/",
                "label": "final submission",
                "commit": None,
                "at": stamp,
                "n_passed": None,
                "n_capsules": None,
                "analysis": cache[key][1],
            }
        )
    previous = None
    for checkpoint in checkpoints:
        analysis = checkpoint["analysis"]
        checkpoint["diff"] = diff(previous, analysis) if analysis else None
        if analysis:
            previous = analysis
    return {"checkpoints": checkpoints, "repo": str(repo), "sampled": len(commits or ()) > MAX_CHECKPOINTS}


__all__ = ["analyse", "diff", "evolution", "parse_cpp", "parse_manifest", "parse_py", "parse_td"]
