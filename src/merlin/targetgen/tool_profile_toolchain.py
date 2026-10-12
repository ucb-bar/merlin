"""Original explicit Lean toolchain grant and identity rules, shared by profile versions."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from merlin.common.path_relations import LexicalPathIndex

_PRIVATE_ROOT_SCHEMA = "merlin.private_toolchain_root.v1"


def _member_path(raw, manifest):
    if not isinstance(raw, str) or not raw or ".." in Path(raw).parts:
        raise ValueError("private toolchain member path is invalid")
    path = Path(raw)
    return path if path.is_absolute() else manifest.parent / path


def _plain_tree_members(root):
    """Inspect every entry without following aliases or overlooking special files."""
    try:
        if root.resolve() != root or not root.is_dir():
            return None
        result, pending = set(), [root]
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        return None
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if stat.S_ISDIR(mode):
                        pending.append(Path(entry.path))
                    elif stat.S_ISREG(mode):
                        result.add(Path(entry.path))
                    else:
                        return None
        return result
    except OSError:
        return None


def _root_members(document, manifest, root):
    rows = document["files"]
    members = {}
    for row in rows:
        path = _member_path(row["path"], manifest)
        if path in members:
            raise ValueError("private toolchain inventory repeats a member")
        members[path] = row["sha256"]
    return {path: sha for path, sha in members.items() if path.is_relative_to(root)}


def _existing_sources(bundle):
    """Unknown public/source resolution cannot establish a safe private batch."""
    from merlin.common.paths import repo_root

    result = []
    for role in ("allowed", "host_inputs"):
        for row in bundle[role]:
            raw = row.get("path")
            if not isinstance(raw, str) or not raw or ".." in Path(raw).parts:
                return None
            path = Path(raw)
            if not path.is_absolute():
                path = repo_root().absolute() / path
                if not path.exists():
                    # Match the existing repository shorthand only when it is
                    # present. Unresolved registry/source conventions abstain.
                    alternate = repo_root().absolute() / "merlin" / raw
                    if not alternate.exists():
                        return None
                    path = alternate
            try:
                result.extend((path, path.resolve(strict=True)))
            except OSError:
                return None
    return result


def _closed_subdirectories(root, members, blocked):
    """One bottom-up walk selects maximal closed private subtrees, never extra files."""
    if root.resolve() != root or not root.is_dir():
        return []
    blocked_index = LexicalPathIndex.from_paths(blocked)
    pending, order, facts = [root], [], {}
    try:
        while pending:
            directory = pending.pop()
            closed = not blocked_index.overlaps(directory)
            children, count = [], 0
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if stat.S_ISDIR(mode):
                        children.append(path)
                    elif stat.S_ISREG(mode):
                        selected = path in members
                        closed &= selected
                        count += selected
                    else:
                        closed = False  # Aliases and special entries cannot enter a closed tree.
            order.append(directory)
            facts[directory] = (closed, count, children)
            pending.extend(children)
    except OSError:
        return []
    complete, sizes = {}, {}
    for directory in reversed(order):
        closed, count, children = facts[directory]
        complete[directory] = closed and all(complete[child] for child in children)
        sizes[directory] = count + sum(sizes[child] for child in children)
    result, pending = [], [root]
    while pending:
        directory = pending.pop()
        if complete[directory] and sizes[directory]:
            result.append(directory)
        else:
            pending.extend(facts[directory][2])
    return sorted(result)


def _private_root_records(bundle, document, manifest, roots, manifest_sha):
    existing = _existing_sources(bundle)
    if existing is None:
        return []
    protected = {manifest, _member_path(document["executable"]["path"], manifest)}
    solver = document.get("sat_solver")
    if solver is not None:
        protected.add(_member_path(solver["path"], manifest))
    record = []
    for root in roots:
        if any(root.is_relative_to(other) or other.is_relative_to(root) for other in roots if other != root):
            continue
        members = _root_members(document, manifest, root)
        for closed in _closed_subdirectories(root, members, (*existing, *protected)):
            record.append({
                "path": str(closed), "note": "closed private formal verification library",
                "closed_toolchain_root": {"schema": _PRIVATE_ROOT_SCHEMA, "manifest": str(manifest),
                                          "manifest_sha256": manifest_sha},
            })
    return record


def validate_private_toolchain_roots(bundle, *, snapshot_root=None, repo=None):
    """Reopen full original file pins before copying and on actual frozen bytes.

    Only host records created by the explicit toolchain selection participate.
    No public grants, leaf records or checker cache policy changes here.
    """
    from merlin.common.bundle_snapshot import snapshot_path

    if (snapshot_root is None) != (repo is None):
        raise ValueError("private root snapshot verification requires its original repository mapping")

    def actual(path):
        return path if snapshot_root is None else snapshot_path(Path(snapshot_root), path, Path(repo))

    for row in bundle.get("host_inputs", []):
        if "closed_toolchain_root" not in row:
            continue
        selection = row["closed_toolchain_root"]
        if (
            type(selection) is not dict or set(selection) != {"schema", "manifest", "manifest_sha256"}
            or selection["schema"] != _PRIVATE_ROOT_SCHEMA
        ):
            raise ValueError("private toolchain root selection is malformed")
        manifest, root = Path(selection["manifest"]), Path(row["path"])
        if not manifest.is_absolute() or not root.is_absolute() or ".." in manifest.parts or ".." in root.parts:
            raise ValueError("private toolchain root selection is not canonical")
        selected = actual(manifest)
        if selected.resolve() != selected or not selected.is_file():
            raise ValueError("private toolchain manifest source is unavailable or indirect")
        payload = selected.read_bytes()
        if hashlib.sha256(payload).hexdigest() != selection["manifest_sha256"]:
            raise ValueError("private toolchain manifest changed before freezing")
        document = json.loads(payload)
        if document.get("schema") != "merlin.lean_toolchain.v1" or not any(
            root.is_relative_to(_member_path(value, manifest)) for value in document["library_roots"]
        ):
            raise ValueError("private root is outside its selected toolchain library roots")
        expected = {actual(path): sha for path, sha in _root_members(document, manifest, root).items()}
        if not expected or _plain_tree_members(actual(root)) != set(expected):
            raise ValueError("private toolchain root membership changed before freezing")
        for path, sha in expected.items():
            with path.open("rb") as stream:
                observed = hashlib.file_digest(stream, "sha256").hexdigest()
            if observed != sha:
                raise ValueError("private toolchain root bytes differ from their full selected file pins")


def attach_selected_toolchain(bundle: dict, tools: tuple[str, ...], manifest: Path | None) -> dict:
    toolchain = {"selection": "not_selected", "readiness": "unavailable"}
    if manifest is not None:
        selected = Path(manifest)
        if (
            not selected.is_absolute()
            or selected.is_symlink()
            or selected.resolve() != selected
            or not selected.is_file()
        ):
            raise ValueError("Lean toolchain manifest must be an ordinary canonical selected file")
        payload = selected.read_bytes()
        manifest_sha = hashlib.sha256(payload).hexdigest()
        document = json.loads(payload)
        if not isinstance(document, dict) or document.get("schema") != "merlin.lean_toolchain.v1":
            raise ValueError("unknown selected Lean toolchain manifest schema")
        executable = document.get("executable")
        files = document.get("files")
        if not isinstance(executable, dict) or not isinstance(files, list) or not files:
            raise ValueError("Lean toolchain requires its explicit executable and closed file membership")
        roots = document.get("library_roots")
        if not isinstance(roots, list) or not roots or any(not isinstance(root, str) or not root for root in roots):
            raise ValueError("Lean toolchain requires its explicit import roots")
        for name in ("timeout_s", "memory_mb"):
            budget = document.get(name, 120 if name == "timeout_s" else 1024)
            if type(budget) is not int or budget < 1:
                raise ValueError("Lean toolchain requires positive execution budgets")

        def resolve_member(raw: str) -> Path:
            path = Path(raw)
            if ".." in path.parts or not raw:
                raise ValueError("Lean toolchain member path is invalid")
            path = path if path.is_absolute() else selected.parent / path
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("Lean toolchain member is linked or noncanonical")
            return path

        rows = [executable, *files]
        paths: list[str] = []
        import_roots = [resolve_member(root) for root in roots]
        missing = any(not root.is_dir() for root in import_roots)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Lean toolchain contains an invalid file member")
            raw, digest = row.get("path"), row.get("sha256")
            if (
                not isinstance(raw, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Lean toolchain member requires an explicit path and SHA-256")
            path = resolve_member(raw)
            if path.exists():
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("selected Lean toolchain bytes differ from its manifest")
            else:
                missing = True
            if str(path) not in paths:
                paths.append(str(path))
        from .tool_profile import _public_runtime_members

        runtime_paths = _public_runtime_members(document, resolve_member)
        for path in sorted(runtime_paths):
            if not any(row.get("path") == path for row in bundle["allowed"]):
                bundle["allowed"].append({"path": path, "mode": "ro",
                    "note": "selected ordinary OS runtime; not formal authoring capability"})
        role = "allowed" if "lean_proof_authoring" in tools else "host_inputs"
        private_roots = (
            _private_root_records(bundle, document, selected, import_roots, manifest_sha)
            if role == "host_inputs" else []
        )
        grouped = [Path(row["path"]) for row in private_roots]
        for path in (str(selected), *paths):
            # The compiled formal closure may also name the packaged foundation
            # already selected as automatic checker data. One source identity
            # must have one materialization record, rather than duplicate grants.
            member_role = "allowed" if path in runtime_paths else role
            if any(row.get("path") == path for row in bundle[member_role]):
                continue
            if any(Path(path).is_relative_to(root) for root in grouped):
                continue
            row = {"path": path, "note": "selected formal verification toolchain"}
            if path == str(selected):
                row["note"] = "selected formal verification toolchain manifest"
            if member_role == "allowed":
                row["mode"] = "ro"
            bundle[member_role].append(row)
        bundle[role].extend(private_roots)
        if selected.read_bytes() != payload:
            raise ValueError("selected Lean toolchain manifest changed during source preparation")
        toolchain = {
            "selection": "explicit_manifest",
            "manifest_sha256": manifest_sha,
            "executable_sha256": executable["sha256"],
            "dependency_revision": document.get("dependency_revision"),
            "readiness": "unavailable" if missing else "declared_unprobed",
        }
    return toolchain


def require_packaged_mathematics(manifest: Path, source_root: Path, modules: tuple[str, ...]) -> None:
    """Presence and exact source pins, not runtime or source/product qualification."""
    selected = Path(manifest)
    document = json.loads(selected.read_text())

    def resolve(raw):
        path = Path(raw)
        return path if path.is_absolute() else selected.parent / path

    members = {resolve(row["path"]): row["sha256"] for row in document["files"]}
    for module in modules:
        source = source_root / "merlin/verify/lean/Merlin" / (module + ".lean")
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        if not any(
            path.name == module + ".lean" and path.parent.name == "Merlin" and sha == expected
            for path, sha in members.items()
        ):
            raise ValueError("V3 toolchain omits the exact packaged mathematical source: " + module)
    roots = [resolve(raw) for raw in document["library_roots"]]
    namespaces = [root for root in roots if any(path.is_relative_to(root / "Merlin") for path in members)]
    if not namespaces or not all(namespaces[0] / "Merlin" / (module + ".olean") in members for module in modules):
        raise ValueError(
            "V3 toolchain requires all packaged mathematical products in the first selected namespace root"
        )
