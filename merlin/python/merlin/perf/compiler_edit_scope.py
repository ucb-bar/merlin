"""Host-frozen AST edit authority; candidate manifest declarations confer no new permission."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

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


_MECHANISM_SELECTOR_KINDS = frozenset({
    "imports", "module", "class", "function", "method", "helper"})


def _definition_inventory(source: str) -> dict[tuple[str, str], ast.AST]:
    """Return exact class/function/method definitions, never a broad class subtree."""
    result: dict[tuple[str, str], ast.AST] = {}

    def visit_class(node: ast.ClassDef, parents: tuple[str, ...]) -> None:
        qualified = ".".join((*parents, node.name))
        result[("class", qualified)] = node
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                visit_class(child, (*parents, node.name))
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[("method", f"{qualified}.{child.name}")] = child

    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            visit_class(node, ())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[("function", node.name)] = node
    return result


def _node_body_structure(node: ast.Module | ast.ClassDef) -> tuple[Any, ...]:
    """Structure owned by the module/class itself, excluding imports and nested definitions."""
    fields = (() if isinstance(node, ast.Module) else tuple(
        (name, ast.dump(value, include_attributes=False) if isinstance(value, ast.AST)
         else [ast.dump(item, include_attributes=False) for item in value]
         if isinstance(value, list) else value)
        for name, value in ast.iter_fields(node) if name != "body"))
    body = tuple(ast.dump(item, include_attributes=False) for item in node.body
                 if not isinstance(item, (ast.Import, ast.ImportFrom, ast.ClassDef,
                                          ast.FunctionDef, ast.AsyncFunctionDef)))
    return fields, body


def _definition_order(node: ast.Module | ast.ClassDef) -> tuple[tuple[str, str], ...]:
    return tuple((type(item).__name__, item.name) for item in node.body
                 if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))


def _python_semantic_units(before: Path | None, after: Path | None, *, path: str
                           ) -> list[dict[str, Any]]:
    """Attribute semantic AST deltas to narrow units; formatting produces no unit."""
    before_text = before.read_text() if before is not None else None
    after_text = after.read_text() if after is not None else None
    before_tree = ast.parse(before_text) if before_text is not None else None
    after_tree = ast.parse(after_text) if after_text is not None else None

    def digest(value: Any) -> str | None:
        if value is None:
            return None
        payload = repr(value).encode()
        return hashlib.sha256(payload).hexdigest()

    units: list[dict[str, Any]] = []
    imports = []
    module = []
    definitions = [
        {} if text is None else _definition_inventory(text)
        for text in (before_text, after_text)]
    for tree in (before_tree, after_tree):
        imports.append(None if tree is None else tuple(
            ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))))
        module.append(None if tree is None else _node_body_structure(tree))
    for kind, values in (("imports", imports), ("module", module)):
        reordered = (kind == "module" and before_tree is not None and after_tree is not None
                     and sorted(_definition_order(before_tree))
                     == sorted(_definition_order(after_tree))
                     and _definition_order(before_tree) != _definition_order(after_tree))
        if values[0] != values[1] or reordered:
            if reordered:
                values = [(values[index], _definition_order(tree))
                          for index, tree in enumerate((before_tree, after_tree))]
            units.append({"kind": kind, "path": path, "symbol": None,
                          "before_ast_sha256": digest(values[0]),
                          "after_ast_sha256": digest(values[1])})
    keys = sorted(set(definitions[0]) | set(definitions[1]))
    for kind, symbol in keys:
        nodes = [definitions[index].get((kind, symbol)) for index in range(2)]
        values = [None if node is None else ast.dump(node, include_attributes=False)
                  for node in nodes]
        if values[0] == values[1]:
            continue
        if kind == "class" and nodes[0] is not None and nodes[1] is not None:
            # A class selector owns bases/decorators/class statements and definition ordering,
            # but never changes hidden inside an otherwise unchanged method.
            values = [_node_body_structure(node) for node in nodes]
            reordered = (sorted(_definition_order(nodes[0]))
                         == sorted(_definition_order(nodes[1]))
                         and _definition_order(nodes[0]) != _definition_order(nodes[1]))
            if values[0] == values[1] and not reordered:
                continue
            if reordered:
                values = [(values[index], _definition_order(node))
                          for index, node in enumerate(nodes)]
        units.append({"kind": kind, "path": path, "symbol": symbol,
                      "before_ast_sha256": digest(values[0]),
                      "after_ast_sha256": digest(values[1])})
    return units


def validate_mechanism_catalog(catalog: Mapping[str, Any], initial: Path,
                               contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a host-frozen one-mechanism catalog without expanding edit authority."""
    validate_edit_contract(contract, initial)
    body = {key: value for key, value in catalog.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    mechanisms = catalog.get("mechanisms")
    if (catalog.get("schema") != "compiler_mechanism_catalog_v1"
            or set(catalog) != {"schema", "contract_sha256", "mechanisms", "sha256"}
            or catalog.get("sha256") != digest
            or catalog.get("contract_sha256") != contract["sha256"]
            or not isinstance(mechanisms, Sequence)
            or isinstance(mechanisms, (str, bytes)) or not mechanisms):
        raise ValueError("mechanism catalog schema, contract, or canonical identity is invalid")
    authority = [(owner["path"], owner["symbol"])
                 for owner in contract["existing_symbols"]]
    authority_paths = {path for path, _symbol in authority}
    extensions = {_relative(row["directory"]): set(row["surface_ids"])
                  for row in contract.get("helper_extensions", [])}
    ids: set[str] = set()
    selector_owners: dict[tuple[Any, ...], str] = {}
    helper_owners: list[tuple[PurePosixPath, str]] = []
    for mechanism in mechanisms:
        ident = mechanism.get("id") if isinstance(mechanism, Mapping) else None
        selectors = mechanism.get("selectors") if isinstance(mechanism, Mapping) else None
        if (not isinstance(ident, str) or not ident.strip() or ident != ident.strip()
                or ident in ids or set(mechanism) != {"id", "selectors"}
                or not isinstance(selectors, Sequence) or isinstance(selectors, (str, bytes))
                or not selectors):
            raise ValueError("mechanism catalog requires unique ids and nonempty selectors")
        ids.add(ident)
        for selector in selectors:
            if not isinstance(selector, Mapping) or selector.get("kind") not in _MECHANISM_SELECTOR_KINDS:
                raise ValueError("mechanism selector kind is invalid")
            kind = selector["kind"]
            if kind == "helper":
                if set(selector) != {"kind", "directory"}:
                    raise ValueError("helper selector requires only an exact directory")
                directory = _relative(selector["directory"])
                if directory not in extensions:
                    raise ValueError("helper selector lies outside host-frozen helper authority")
                if any(directory == other or directory in other.parents or other in directory.parents
                       for other, owner in helper_owners if owner != ident):
                    raise ValueError("helper selector directories overlap across mechanisms")
                helper_owners.append((directory, ident))
                key = (kind, str(directory))
            else:
                required = {"kind", "path"} | ({"symbol"} if kind in {
                    "class", "function", "method"} else set())
                if set(selector) != required:
                    raise ValueError("mechanism selector fields are not exact for its kind")
                path = _relative(selector["path"])
                source = initial / str(path)
                if (str(path) not in authority_paths or source.is_symlink()
                        or source.suffix != ".py" or not source.is_file()):
                    raise ValueError("mechanism selector path has no host-frozen edit authority")
                symbol = selector.get("symbol")
                key = (kind, str(path), symbol)
                if kind in {"class", "function", "method"}:
                    if not isinstance(symbol, str) or not symbol:
                        raise ValueError("symbol selector requires a qualified symbol")
                    inventory = _definition_inventory(source.read_text())
                    if (kind, symbol) not in inventory:
                        raise ValueError("mechanism selector does not resolve to its exact AST kind")
                    if not any(str(path) == owner_path and (
                            symbol == owner_symbol or symbol.startswith(owner_symbol + "."))
                               for owner_path, owner_symbol in authority):
                        raise ValueError("mechanism symbol selector exceeds host-frozen authority")
            previous = selector_owners.get(key)
            if previous is not None:
                raise ValueError("one narrow selector cannot be repeated or belong to two mechanisms")
            selector_owners[key] = ident
    return copy.deepcopy(dict(catalog))


def inspect_round_mechanism_edits(initial: Path, round_start: Path, candidate: Path,
                                  contract: Mapping[str, Any],
                                  catalog: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce one host-catalogued mechanism on the round-local semantic AST delta."""
    validated = validate_mechanism_catalog(catalog, initial, contract)
    start_scope = inspect_compiler_edits(initial, round_start, contract)
    candidate_scope = inspect_compiler_edits(initial, candidate, contract)
    selectors: dict[tuple[Any, ...], set[str]] = {}
    helpers: list[tuple[PurePosixPath, str]] = []
    for mechanism in validated["mechanisms"]:
        for selector in mechanism["selectors"]:
            if selector["kind"] == "helper":
                helpers.append((_relative(selector["directory"]), mechanism["id"]))
            else:
                key = (selector["kind"], selector["path"], selector.get("symbol"))
                selectors.setdefault(key, set()).add(mechanism["id"])

    def python_files(root: Path) -> dict[str, Path]:
        return {path.relative_to(root).as_posix(): path for path in root.rglob("*.py")
                if "__pycache__" not in path.parts and ".git" not in path.relative_to(root).parts}

    before, after = python_files(round_start), python_files(candidate)
    initial_files = python_files(initial)
    units = []
    mechanisms: set[str] = set()
    violations = []
    for path in sorted(before.keys() | after.keys()):
        left, right = before.get(path), after.get(path)
        if any(item is not None and item.is_symlink() for item in (left, right)):
            violations.append({"path": path, "reason": "round delta contains linked Python source"})
            continue
        if left is not None and right is not None and left.read_bytes() == right.read_bytes():
            continue
        for unit in _python_semantic_units(left, right, path=path):
            owners = set(selectors.get((unit["kind"], path, unit["symbol"]), ()))
            # A host-frozen helper directory owns only code absent from the original authority
            # snapshot. It does not turn existing unrelated code into a broad editable surface.
            relative = PurePosixPath(path)
            helper_code = (path not in initial_files or unit["symbol"] is not None
                           and (unit["kind"], unit["symbol"]) not in (
                               _definition_inventory(initial_files[path].read_text())
                               if path in initial_files else {}))
            if helper_code:
                owners.update(owner for directory, owner in helpers
                              if directory in relative.parents)
            unit["mechanism_ids"] = sorted(owners)
            units.append(unit)
            mechanisms.update(owners)
            if not owners:
                violations.append({"path": path, "kind": unit["kind"],
                                   "symbol": unit["symbol"],
                                   "reason": "semantic AST delta has no narrow mechanism selector"})
            elif len(owners) != 1:
                violations.append({"path": path, "kind": unit["kind"],
                                   "symbol": unit["symbol"],
                                   "reason": "semantic AST delta is ambiguous across mechanisms"})
    if start_scope["status"] != "allowed":
        violations.append({"reason": "round start already exceeds cumulative edit authority",
                           "details": start_scope["violations"]})
    if candidate_scope["status"] != "allowed":
        violations.append({"reason": "candidate exceeds cumulative edit authority",
                           "details": candidate_scope["violations"]})
    if len(mechanisms) > 1:
        violations.append({"reason": "round contains more than one optimization mechanism",
                           "mechanism_ids": sorted(mechanisms)})
    raw_changed = sorted(path for path in before.keys() | after.keys()
                         if path not in before or path not in after
                         or before[path].read_bytes() != after[path].read_bytes())
    return {
        "schema": "compiler_round_mechanism_attribution_v1",
        "status": "refused" if violations else "allowed",
        "contract_sha256": contract["sha256"], "catalog_sha256": catalog["sha256"],
        "selected_mechanism_id": next(iter(mechanisms)) if len(mechanisms) == 1 else None,
        "mechanism_ids": sorted(mechanisms), "semantic_ast_units": units,
        "semantic_noop": not units, "formatting_only_noop": bool(raw_changed) and not units,
        "round_changed_paths": raw_changed, "violations": violations,
        "cumulative_start_scope": start_scope, "cumulative_candidate_scope": candidate_scope,
        "candidate_manifest_grants_authority": False,
        "scope": "round-local AST mechanism attribution plus unchanged cumulative edit authority",
    }
