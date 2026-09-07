"""Host-frozen AST edit authority; candidate manifest declarations confer no new permission."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or str(path) == ".":
        raise ValueError("edit authority requires a scoped relative path")
    return path


def validate_edit_contract(contract: Mapping[str, Any], initial: Path) -> dict[str, Any]:
    body = {key: value for key, value in contract.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if contract.get("schema") != "compiler_edit_contract_v1" or contract.get("sha256") != digest:
        raise ValueError("host edit contract schema or canonical identity is invalid")
    identities = set()
    for owner in contract["existing_symbols"]:
        path, symbol = _relative(owner["path"]), owner["symbol"]
        source = initial / str(path)
        if source.is_symlink() or source.suffix != ".py" or not source.is_file():
            raise ValueError("host edit symbol does not name an initial Python source file")
        tree = ast.parse(source.read_text())
        nodes = tree.body
        found = None
        for part in symbol.split("."):
            found = next((node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                          and node.name == part), None)
            if found is None:
                raise ValueError(f"host edit symbol is absent from initial source: {path}:{symbol}")
            nodes = found.body
        identities.add(owner["surface_id"])
    for extension in contract.get("helper_extensions", []):
        directory = _relative(extension["directory"])
        if not (initial / str(directory)).is_dir() or not extension.get("surface_ids"):
            raise ValueError("helper extension must name an existing initial compiler component")
        if set(extension["surface_ids"]) - identities:
            raise ValueError("helper extension has no host-frozen owning surface")
    for ancillary in contract.get("ancillary_paths", []):
        path = _relative(ancillary)
        if path.parts[0] not in ("docs", "tests"):
            raise ValueError("ancillary authority is restricted to explicit candidate docs/tests paths")
    return copy.deepcopy(dict(contract))


def _definition_names(source: str) -> set[str]:
    found = set()
    def walk(nodes, parents=()):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = (*parents, node.name)
                found.add(".".join(name))
                walk(node.body, name)
    walk(ast.parse(source).body)
    return found


def _remainder(source: str, authorized: set[str], *, ignore_imports: bool) -> str:
    tree = ast.parse(source)

    def prune(nodes, parents=()):
        result = []
        for node in nodes:
            if not parents and ignore_imports and isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = (*parents, node.name)
                if ".".join(name) in authorized:
                    continue
                node.body = prune(node.body, name)
            result.append(node)
        return result

    tree.body = prune(tree.body)
    return ast.dump(tree, include_attributes=False)


def inspect_compiler_edits(initial: Path, candidate: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Check actual submitted changes against the immutable host authority, without importing code."""
    authorized: dict[str, set[str]] = {}
    for owner in contract["existing_symbols"]:
        authorized.setdefault(owner["path"], set()).add(owner["symbol"])
    extensions = [_relative(row["directory"]) for row in contract.get("helper_extensions", [])]
    ancillary = [_relative(row) for row in contract.get("ancillary_paths", [])]

    def files(root):
        return {path.relative_to(root).as_posix(): path for path in root.rglob("*")
                if (path.is_file() or path.is_symlink()) and ".git" not in path.relative_to(root).parts
                and "__pycache__" not in path.parts and path.suffix != ".pyc"}

    before, after = files(initial), files(candidate)
    changes, refused = [], []
    for name in sorted(before.keys() | after.keys()):
        left, right = before.get(name), after.get(name)
        if any(path and path.is_symlink() for path in (left, right)):
            refused.append({"path": name, "reason": "linked compiler source is not editable authority"})
            continue
        if left and right and left.read_bytes() == right.read_bytes():
            continue
        changes.append(name)
        relative = PurePosixPath(name)
        if any(relative == path or path in relative.parents for path in ancillary):
            continue
        if name == "manifest.yaml" and left and right:
            old, new = yaml.safe_load(left.read_text()), yaml.safe_load(right.read_text())
            if isinstance(old, dict) and isinstance(new, dict):
                old.pop("optimization_surfaces", None)
                new.pop("optimization_surfaces", None)
                if old == new:
                    continue
            refused.append({"path": name, "reason": "protected manifest controls changed"})
            continue
        if not left and right and right.suffix == ".py" and any(path in relative.parents for path in extensions):
            ast.parse(right.read_text())
            continue
        if left and right and name in authorized:
            permitted = set(authorized[name])
            if any(path in relative.parents for path in extensions):
                permitted |= _definition_names(right.read_text()) - _definition_names(left.read_text())
            if _remainder(left.read_text(), permitted, ignore_imports=True) == _remainder(
                    right.read_text(), permitted, ignore_imports=True):
                continue
        refused.append({"path": name, "reason": "change outside host-frozen AST symbols/helper extensions"})
    return {"schema": "compiler_submitted_edit_scope_v1", "status": "refused" if refused else "allowed",
            "contract_sha256": contract["sha256"], "changed_paths": changes, "violations": refused,
            "candidate_manifest_grants_authority": False,
            "scope": "edit ownership only; no semantic correctness or performance claim"}
