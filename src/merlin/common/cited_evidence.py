"""Which paths a seal, release, run record or derivation cites -- so no cleanup deletes them.

On 2026-10-17 and again on 2026-10-24 a disk cleanup deleted files that sealed Phase 0 evidence
pinned by digest: the captures behind a corpus derivation, then the catalogs inside a reviewed
derivation itself. Both times the deletion looked safe from the filesystem ("old", "staged", "a
copy exists elsewhere") and both times the loss surfaced only when a later preparation re-read the
exact path and digest. The knowledge that a path is live was never missing; it was written down in
the records that cite it, and nothing that deletes consulted them.

This module reads those records and answers one question: would removing a path take bytes away
from a record that will outlive the removal? The record roster -- which file names are citing
records, which seal anchors protect a whole tree, which directories are not worth walking -- is
data in ``merlin/contract/storage.yaml`` (``cited_evidence``), not names in code.

What counts as a citation, read structurally from each record's JSON or YAML:

* an absolute path string anywhere in the document (argv lists and ``definition:`` fields included);
* a relative ``path`` / ``*_path`` value in a mapping that also carries a ``*sha256`` digest, resolved
  against the record's own directory;
* the member names of an ``artifacts`` mapping whose rows carry a digest (a producer receipt);
* the whole tree under a declared seal anchor, and every path a record lists under ``protects``.

What a citation forbids: removing the cited path or any directory containing it. Removing something
INSIDE a cited directory is forbidden only when the citation pins the tree's content (it carries a
digest) or anchors it (a seal) -- a bare mention of a cache root is not a claim on each file in it.
A citation never blocks a removal that also removes the record making it (a run unit deleted whole
takes its own run record with it), except a seal anchor, which is the point of sealing.
"""

from __future__ import annotations

import argparse
import bisect
import fnmatch
import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ROOTS_ENV = "MERLIN_CITED_ROOTS"
_MAX_PATH = 4096
_DEFAULT_MAX_RECORD_BYTES = 16 << 20


@dataclass(frozen=True)
class Citation:
    path: str
    record: str
    kind: str  # "path" (bare mention), "pinned" (carries a digest), "anchor" (sealed tree)
    sha256: str | None = None

    def describe(self) -> str:
        pin = f" @ {self.sha256[:12]}" if self.sha256 else ""
        return f"{self.kind} citation of {self.path}{pin} by {self.record}"


def roster() -> dict:
    """The contract's ``cited_evidence`` block: ``records``, ``anchors``, ``skip_dirs``, ``extra_roots``."""
    from merlin.common.artifacts import storage_contract

    block = storage_contract().get("cited_evidence") or {}
    if not isinstance(block, dict):
        raise ValueError("storage cited_evidence must be a mapping")
    return block


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _norm(path: str) -> str:
    return os.path.normpath(path)


def _relative_member(value: str) -> bool:
    if not value or len(value) > _MAX_PATH or "\n" in value or "://" in value or value.startswith("/"):
        return False
    parts = Path(value).parts
    return bool(parts) and ".." not in parts


def _absolute(value: str) -> bool:
    return value.startswith("/") and len(value) < _MAX_PATH and "\n" not in value and value != "/"


def _digest_of(mapping: dict, key: str) -> str | None:
    """The digest a mapping pairs with ``key``: ``x_path`` -> ``x_sha256``, ``path`` -> ``sha256`` or the only one."""
    stem = key[: -len("path")] if key.endswith("path") else None
    if stem is not None and _is_digest(mapping.get(stem + "sha256")):
        return mapping[stem + "sha256"]
    digests = [v for k, v in mapping.items() if isinstance(k, str) and k.endswith("sha256") and _is_digest(v)]
    return digests[0] if key == "path" and len(digests) == 1 else None


def _walk_document(document: object, base: Path, record: str) -> Iterator[Citation]:
    stack = [document]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            has_digest = any(isinstance(k, str) and k.endswith("sha256") and _is_digest(v) for k, v in node.items())
            artifacts = node.get("artifacts")
            if isinstance(artifacts, dict):
                for name, row in artifacts.items():
                    if isinstance(name, str) and _relative_member(name) and isinstance(row, dict):
                        digest = row.get("sha256")
                        if _is_digest(digest):
                            yield Citation(_norm(str(base / name)), record, "pinned", digest)
            for key, value in node.items():
                if isinstance(value, str) and isinstance(key, str):
                    digest = _digest_of(node, key) if (key == "path" or key.endswith("_path")) else None
                    if _absolute(value):
                        yield Citation(_norm(value), record, "pinned" if digest else "path", digest)
                    elif has_digest and (key == "path" or key.endswith("_path")) and _relative_member(value):
                        yield Citation(_norm(str(base / value)), record, "pinned" if digest else "path", digest)
                elif isinstance(value, dict | list):
                    stack.append(value)
        elif isinstance(node, list):
            for value in node:
                if isinstance(value, str):
                    if _absolute(value):
                        yield Citation(_norm(value), record, "path")
                elif isinstance(value, dict | list):
                    stack.append(value)


def _load(path: Path) -> object | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        if path.suffix == ".json":
            return json.loads(raw)
        import yaml

        return yaml.safe_load(raw)
    except Exception:  # noqa: BLE001 -- an unreadable or malformed record cites nothing
        return None


def record_citations(path: Path, *, anchors: Iterable[dict] = ()) -> list[Citation]:
    """Every citation one record file makes, including the trees its seal anchors."""
    path = Path(os.path.abspath(path))
    document = _load(path)
    record = str(path)
    out = list(_walk_document(document, path.parent, record)) if document is not None else []
    if isinstance(document, dict) and isinstance(document.get("protects"), list):
        for value in document["protects"]:
            if isinstance(value, str) and (_absolute(value) or _relative_member(value)):
                out.append(Citation(_norm(str(path.parent / value)), record, "anchor"))
    for anchor in anchors:
        pattern = str(anchor.get("record") or "")
        if pattern and fnmatch.fnmatchcase(path.as_posix(), "*/" + pattern):
            root = path
            for _ in range(int(anchor.get("up", 1))):
                root = root.parent
            out.append(Citation(_norm(str(root)), record, "anchor"))
    return out


def _is_record(name: str, relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) or fnmatch.fnmatchcase(relative, "*/" + p) for p in patterns)


def declared_roots() -> list[Path]:
    """The out/ root, the storage scan roots, the contract's extra roots and ``MERLIN_CITED_ROOTS``."""
    from merlin.common.paths import out_dir

    roots = [out_dir()]
    try:
        from merlin.common.storage_cli import scan_roots

        roots += scan_roots()
    except (OSError, ValueError):
        pass
    roots += extra_roots()
    kept: list[Path] = []
    for path in sorted({Path(os.path.abspath(p)) for p in roots}, key=lambda p: len(p.parts)):
        if path.is_dir() and not any(parent in path.parents or parent == path for parent in kept):
            kept.append(path)
    return kept


class Registry:
    """Citations indexed by normalized path, with the record files they were read from."""

    def __init__(self, citations: Iterable[Citation], *, roots: Iterable[Path] = (), records: int = 0):
        self.by_path: dict[str, list[Citation]] = {}
        for citation in citations:
            self.by_path.setdefault(citation.path, []).append(citation)
            real = os.path.realpath(citation.path)
            if real != citation.path:
                self.by_path.setdefault(real, []).append(citation)
        self._sorted = sorted(self.by_path)
        self.roots = [str(r) for r in roots]
        self.records = records

    def __len__(self) -> int:
        return len(self.by_path)

    def citations(self, candidate: Path | str, *, ignore_records_under: Path | str | None = None) -> list[Citation]:
        """Citations that removing ``candidate`` would violate.

        ``ignore_records_under`` drops citations made by records inside that tree: a scratch root's
        own working receipts are not a seal, release, run record or derivation of it.
        """
        target = _norm(os.path.abspath(candidate))
        real = os.path.realpath(target)
        found: list[Citation] = []
        for probe in dict.fromkeys((target, real)):
            # The candidate itself, or anything inside it.
            found += self.by_path.get(probe, [])
            index = bisect.bisect_left(self._sorted, probe + os.sep)
            while index < len(self._sorted) and self._sorted[index].startswith(probe + os.sep):
                found += self.by_path[self._sorted[index]]
                index += 1
            # A tree containing the candidate, when the citation pins or seals that whole tree.
            for parent in Path(probe).parents:
                for citation in self.by_path.get(str(parent), []):
                    if citation.kind in {"pinned", "anchor"}:
                        found.append(citation)
        ignored = _norm(os.path.abspath(ignore_records_under)) + os.sep if ignore_records_under else None
        out: list[Citation] = []
        for citation in dict.fromkeys(found):
            record = citation.record
            if ignored and record.startswith(ignored) and citation.kind != "anchor":
                continue
            inside = record == real or record.startswith(real + os.sep) or record.startswith(target + os.sep)
            if inside and citation.kind != "anchor":
                continue  # the removal takes the citing record with it
            out.append(citation)
        return out

    def blockers(self, candidate: Path | str, *, limit: int = 3) -> list[str]:
        cited = self.citations(candidate)
        if not cited:
            return []
        shown = "; ".join(c.describe() for c in cited[:limit])
        more = f" (and {len(cited) - limit} more)" if len(cited) > limit else ""
        return [f"cited evidence: {shown}{more}"]


def build(roots: Iterable[Path] | None = None) -> Registry:
    """Walk ``roots`` for citing records and index everything they cite."""
    block = roster()
    patterns = [str(p) for p in block.get("records") or []]
    anchors = [a for a in block.get("anchors") or [] if isinstance(a, dict)]
    skip = {str(s) for s in block.get("skip_dirs") or []}
    limit = int(block.get("max_record_bytes") or _DEFAULT_MAX_RECORD_BYTES)
    roots = list(roots) if roots is not None else declared_roots()
    citations: list[Citation] = []
    records = 0
    for root in roots:
        root = Path(os.path.abspath(root))
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if d not in skip]
            for name in files:
                path = Path(current) / name
                relative = path.relative_to(root).as_posix()
                if not _is_record(name, relative, patterns):
                    continue
                try:
                    if path.is_symlink() or path.stat().st_size > limit:
                        continue
                except OSError:
                    continue
                records += 1
                citations += record_citations(path, anchors=anchors)
    return Registry(citations, roots=roots, records=records)


_CACHE: dict[tuple[str, ...], tuple[float, Registry]] = {}


def registry(roots: Iterable[Path] | None = None, *, refresh: bool = False, max_age: float | None = None) -> Registry:
    """The registry for ``roots`` (default: the declared roots), built once per process.

    ``max_age`` (seconds) rebuilds a cached registry older than that, for long-lived callers that
    must see records written after they started.
    """
    import time

    chosen = list(roots) if roots is not None else declared_roots()
    key = tuple(str(r) for r in chosen)
    cached = _CACHE.get(key)
    if refresh or cached is None or (max_age is not None and time.monotonic() - cached[0] > max_age):
        cached = _CACHE[key] = (time.monotonic(), build(chosen))
    return cached[1]


def extra_roots() -> list[Path]:
    """Only the roots beyond out/: the contract's ``extra_roots`` and ``MERLIN_CITED_ROOTS``."""
    from merlin.common.paths import merlin_dir

    roots = [p for pattern in roster().get("extra_roots") or [] for p in sorted(merlin_dir().glob(str(pattern)))]
    roots += [Path(p) for p in os.environ.get(ROOTS_ENV, "").split(os.pathsep) if p.strip()]
    return [Path(os.path.abspath(p)) for p in roots if Path(p).is_dir()]


def clear() -> None:
    _CACHE.clear()


def blockers(path: Path | str, roots: Iterable[Path] | None = None) -> list[str]:
    """Why removing ``path`` would break a citing record (empty when nothing cites it)."""
    return registry(roots).blockers(path)


def require_uncited(path: Path | str, roots: Iterable[Path] | None = None) -> None:
    reasons = blockers(path, roots)
    if reasons:
        raise PermissionError(f"refusing to remove {path}: {'; '.join(reasons)}")


def add_arguments(sub) -> None:
    """The ``merlin-storage cited`` subcommand: ask before deleting anything by hand."""
    cited = sub.add_parser(
        "cited",
        help="which records cite a path; exit 1 when removing any given path would break one (read-only)",
    )
    cited.add_argument("paths", nargs="+", type=Path, help="paths you intend to delete")
    cited.add_argument("--root", action="append", type=Path, default=[], help="also scan this tree for records")
    cited.add_argument("--json", action="store_true", help="emit the citations instead of a summary")


def run(args: argparse.Namespace) -> int:
    roots = declared_roots() + [Path(os.path.abspath(r)) for r in args.root]
    found = registry(roots)
    rows = {str(p): [c.__dict__ for c in found.citations(p)] for p in args.paths}
    if args.json:
        print(json.dumps({"roots": found.roots, "records": found.records, "paths": rows}, indent=2))
    else:
        print(f"{found.records} citing records under {len(found.roots)} root(s)")
        for path, citations in rows.items():
            verdict = f"CITED ({len(citations)})" if citations else "not cited"
            print(f"  {verdict:>12}  {path}")
            for row in citations[:5]:
                print(f"      {row['kind']}: {row['record']}")
    return 1 if any(rows.values()) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="merlin-cited", description=__doc__.splitlines()[0])
    add_arguments(parser.add_subparsers(dest="command", required=True))
    sys.exit(run(parser.parse_args()))
