"""Pinned, import-free inventory of external kernel sources and reference artifacts.

Source syntax is evidence of source structure, not instruction semantics or timing.
Family labels select analysis representatives only; they never select compiler behavior.
"""

from __future__ import annotations

import ast
import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

SCHEMA = "merlin.kernel_reference_manifest.v1"
INVENTORY_SCHEMA = "merlin.kernel_reference_inventory.v1"
MAX_ARTIFACT_BYTES = 256 << 20
MAX_SOURCE_BYTES = 16 << 20


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_pinned(path, sha256, *, max_bytes=MAX_ARTIFACT_BYTES):
    """Read the exact selected ordinary file, detecting replacement during the read."""
    path = Path(path)
    before = path.stat()
    if not _pin(sha256) or not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError("reference requires an exact SHA256 and a bounded ordinary file")
    data = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or digest(data) != sha256:
        raise ValueError("selected reference changed or differs from its SHA256")
    return data


def load_manifest(path):
    path = Path(path).absolute()
    if path.stat().st_size > 4 << 20:
        raise ValueError("reference manifest exceeds metadata bound")
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("unsupported kernel reference manifest")
    if document.get("mode") != "reference_guided":
        raise ValueError("kernel references require an explicitly declared reference_guided workflow")
    references = document.get("references")
    if not isinstance(references, list) or not 1 <= len(references) <= 1024:
        raise ValueError("reference manifest needs a bounded nonempty selection")
    ids = set()
    for reference in references:
        if not isinstance(reference, dict) or not isinstance(reference.get("id"), str) or not reference["id"]:
            raise ValueError("each selected reference requires an identity")
        if reference["id"] in ids:
            raise ValueError("reference identities must be unique")
        ids.add(reference["id"])
        if reference.get("visibility") not in ("operator_only", "public_reference"):
            raise ValueError("reference visibility must be explicit")
        if reference.get("format") not in (
            "c_source",
            "python_schedule",
            "elf",
            "disassembly",
            "normalized_commands",
            "semantic_region",
        ):
            raise ValueError("unsupported reference format")
        artifact = reference.get("artifact")
        if not isinstance(artifact, dict) or not _pin(artifact.get("sha256")) or not artifact.get("path"):
            raise ValueError("reference artifact needs a path and exact SHA256")
        if "symbols" in reference and (
            not isinstance(reference["symbols"], list)
            or any(not isinstance(v, str) or not v for v in reference["symbols"])
            or len(set(reference["symbols"])) != len(reference["symbols"])
        ):
            raise ValueError("symbol selection must contain distinct nonempty names")
    return document, path.parent


def _selected_bytes(artifact, root, cache):
    path = (root / artifact["path"]).resolve()
    key = (str(path), artifact["sha256"])
    if key not in cache:
        cache[key] = read_pinned(path, artifact["sha256"])
    raw = cache[key]
    pin = {"kind": "file", "uri": str(path), "sha256": digest(raw), "size_bytes": len(raw)}
    if "member" not in artifact:
        return raw, pin, None
    from io import BytesIO

    member = artifact["member"]
    if (
        not isinstance(member, str)
        or not member
        or PurePosixPath(member).is_absolute()
        or ".." in PurePosixPath(member).parts
    ):
        raise ValueError("archive member must be an explicit relative path")
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        matches = [entry for entry in archive.infolist() if entry.filename == member]
        if len(matches) != 1:
            raise ValueError("selected archive member is absent or ambiguous")
        entry = matches[0]
        if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size > MAX_ARTIFACT_BYTES:
            raise ValueError("selected archive member is indirect or exceeds its bound")
        payload = archive.read(entry)
    if "member_sha256" in artifact and digest(payload) != artifact["member_sha256"]:
        raise ValueError("archive member differs from its selected SHA256")
    return (
        payload,
        pin,
        {"kind": "archive_member", "uri": f"{path}!/{member}", "sha256": digest(payload), "size_bytes": len(payload)},
    )


def _python_source(text):
    tree = ast.parse(text)
    functions = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = Counter(ast.unparse(child.func) for child in ast.walk(node) if isinstance(child, ast.Call))
        loops = [
            {
                "kind": type(child).__name__,
                "line": child.lineno,
                "header": ast.unparse(child.target) + " in " + ast.unparse(child.iter)
                if isinstance(child, (ast.For, ast.AsyncFor))
                else ast.unparse(child.test),
            }
            for child in ast.walk(node)
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While))
        ]
        source = ast.get_source_segment(text, node) or ""
        functions.append(
            {
                "symbol": node.name,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "body_sha256": digest(source.encode()),
                "parameters": [ast.unparse(v) for v in node.args.args],
                "decorators": [ast.unparse(v) for v in node.decorator_list],
                "loops": sorted(loops, key=lambda v: v["line"]),
                "calls": dict(sorted(calls.items())),
            }
        )
    return {"parser": "python.ast", "functions": functions, "diagnostics": []}


def _c_source(text):
    try:
        import tree_sitter_c
        from tree_sitter import Language, Parser
    except ImportError:
        return {
            "parser": "unavailable",
            "functions": [],
            "diagnostics": [
                {
                    "kind": "missing_evidence",
                    "next_action": "install_kernels_ast_extra",
                    "reason": "C structural analysis requires the existing kernels-ast extra",
                }
            ],
        }
    raw = text.encode()
    language = Language(tree_sitter_c.language())
    parser = Parser(language)
    tree = parser.parse(raw)

    def txt(node):
        return raw[node.start_byte : node.end_byte].decode() if node is not None else ""

    def descendants(node):
        cursor = node.walk()
        while True:
            yield cursor.node
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return

    functions = []
    for node in descendants(tree.root_node):
        if node.type != "function_definition":
            continue
        ancestor = node.parent
        guards = []
        nested = False
        while ancestor is not None:
            if ancestor.type == "function_definition":
                nested = True
            if ancestor.type.startswith("preproc_"):
                guards.append(txt(ancestor).splitlines()[0])
            ancestor = ancestor.parent
        if nested:
            raise ValueError("nested function definitions require a selected source-language adapter")
        declaration = node.child_by_field_name("declarator")
        while declaration is not None and declaration.type != "function_declarator":
            declaration = declaration.child_by_field_name("declarator")
        if declaration is None:
            raise ValueError("C function declarator could not be placed")
        name = declaration.child_by_field_name("declarator")
        while name is not None and name.type != "identifier":
            name = name.child_by_field_name("declarator")
        if name is None:
            raise ValueError("C function has no structural identifier")
        body = node.child_by_field_name("body")
        calls = Counter()
        sites, loops = [], []
        for child in descendants(body):
            if child.type == "call_expression":
                target = txt(child.child_by_field_name("function"))
                calls[target] += 1
                sites.append(
                    {
                        "callee": target,
                        "line": child.start_point.row + 1,
                        "arguments": txt(child.child_by_field_name("arguments"))[:2048],
                    }
                )
            if child.type in ("for_statement", "while_statement", "do_statement"):
                loops.append(
                    {
                        "kind": child.type,
                        "line": child.start_point.row + 1,
                        "initializer": txt(child.child_by_field_name("initializer")),
                        "condition": txt(child.child_by_field_name("condition")),
                        "update": txt(child.child_by_field_name("update")),
                    }
                )
        if len(sites) > 20000 or len(loops) > 4096:
            raise ValueError("C function exceeds structural analysis budget")
        parameters = declaration.child_by_field_name("parameters")
        functions.append(
            {
                "symbol": txt(name),
                "line": node.start_point.row + 1,
                "end_line": node.end_point.row + 1,
                "body_sha256": digest(raw[body.start_byte : body.end_byte]),
                "parameters": [txt(v) for v in parameters.named_children] if parameters else [],
                "loops": loops,
                "calls": dict(sorted(calls.items())),
                "call_sites": sites,
                "syntax_complete": not node.has_error,
                "conditional_guards": list(reversed(guards)),
            }
        )
    if len(functions) > 1024:
        raise ValueError("C source exceeds function inventory budget")
    return {
        "parser": "tree_sitter.c",
        "functions": functions,
        "diagnostics": []
        if not tree.root_node.has_error
        else [
            {
                "kind": "missing_evidence",
                "next_action": "repair_source_parse",
                "reason": "source contains syntax not completely parsed; syntax is not a semantic reconstruction",
            }
        ],
    }


def inspect_source(raw, format):
    """Inspect without importing/executing source or interpreting instruction names."""
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds structural analysis bound")
    text = raw.decode("utf-8")
    if format == "python_schedule":
        return _python_source(text)
    if format == "c_source":
        # Native parser failures must become evidence gaps, not kill a campaign.
        # Source is stdin data; it is never imported or passed as executable code.
        import os

        # A host may select parser wheels after interpreter startup. Preserve
        # those dependency roots without allowing them to replace the currently
        # imported implementation roots in the isolated child.
        roots = tuple(dict.fromkeys((*sys.path, *(root for root in
            os.environ.get("PYTHONPATH", "").split(os.pathsep) if root))))
        try:
            child = subprocess.run(
                [sys.executable, "-m", __name__, "--inspect-c"],
                input=raw,
                capture_output=True,
                timeout=30,
                env={**os.environ, "PYTHONPATH": os.pathsep.join(roots), "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return {
                "parser": "unavailable",
                "functions": [],
                "diagnostics": [
                    {
                        "kind": "missing_evidence",
                        "next_action": "repair_source_parser",
                        "reason": "C parser exceeded its bounded execution time",
                    }
                ],
            }
        if child.returncode != 0:
            return {
                "parser": "unavailable",
                "functions": [],
                "diagnostics": [
                    {
                        "kind": "missing_evidence",
                        "next_action": "repair_source_parser",
                        "reason": f"isolated C parser failed with exit status {child.returncode}",
                    }
                ],
            }
        if len(child.stdout) > 32 << 20:
            raise ValueError("source parser metadata exceeds its output bound")
        return json.loads(child.stdout)
    raise ValueError("structural source inspection requires a supported source format")


def inventory(document, *, root):
    """Resolve an explicitly selected corpus and retain missing evidence as tasks."""
    if document.get("schema") != SCHEMA or document.get("mode") != "reference_guided":
        raise ValueError("explicit reference_guided manifest required")
    cache, rows = {}, []
    for selection in document["references"]:
        raw, container, member = _selected_bytes(selection["artifact"], Path(root), cache)
        row = {
            "id": selection["id"],
            "family": selection.get("family", "unclassified"),
            "visibility": selection["visibility"],
            "format": selection["format"],
            "artifacts": [container] + ([member] if member else []),
            "analysis": {},
            "gaps": [],
        }
        if selection["format"] in ("c_source", "python_schedule"):
            row["analysis"] = inspect_source(raw, selection["format"])
            selected = selection.get("symbols")
            if selected is not None:
                found = {f["symbol"] for f in row["analysis"]["functions"]}
                for missing in sorted(set(selected) - found):
                    row["gaps"].append(
                        {
                            "kind": "missing_evidence",
                            "symbol": missing,
                            "next_action": "locate_selected_symbol",
                            "reason": "selected symbol absent from parsed source",
                        }
                    )
                row["analysis"]["functions"] = [f for f in row["analysis"]["functions"] if f["symbol"] in selected]
            row["gaps"].extend(row["analysis"]["diagnostics"])
        elif selection["format"] in ("normalized_commands", "semantic_region"):
            if len(raw) > MAX_SOURCE_BYTES:
                raise ValueError("structured reference metadata exceeds bound")
            row["analysis"] = {"record": json.loads(raw)}
        else:
            row["gaps"].append(
                {
                    "kind": "missing_evidence",
                    "next_action": "apply_selected_isa_projector",
                    "reason": "binary/disassembly needs its own pinned ISA and operand projection",
                }
            )
        if "semantic_region" in selection:
            row["semantic_region"] = selection["semantic_region"]
        else:
            row["gaps"].append(
                {
                    "kind": "missing_evidence",
                    "next_action": "recover_semantic_region",
                    "reason": "source syntax or binary identity alone does not establish "
                    "a complete fused semantic region",
                }
            )
        rows.append(row)
    return {
        "schema": INVENTORY_SCHEMA,
        "mode": "reference_guided",
        "references": rows,
        "manifest_sha256": digest(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()),
        "qualification": "pinned reference inventory and source syntax only; no execution or performance claim",
    }


def select_representatives(index, *, per_family=2):
    """Deterministic structural diversity; unclassified entries remain uncovered."""
    if type(per_family) is not int or not 1 <= per_family <= 2:
        raise ValueError("select one or two references per declared semantic/schedule family")
    groups = {}
    for row in index["references"]:
        if row["family"] == "unclassified":
            continue
        fingerprint = digest(
            json.dumps(
                [(f["calls"], len(f["loops"])) for f in row["analysis"].get("functions", [])], sort_keys=True
            ).encode()
        )
        groups.setdefault(row["family"], []).append((fingerprint, row["id"]))
    selected = {}
    for family, rows in sorted(groups.items()):
        unique, remaining = [], []
        fingerprints = set()
        for fingerprint, identity in sorted(rows):
            (remaining if fingerprint in fingerprints else unique).append(identity)
            fingerprints.add(fingerprint)
        selected[family] = (unique + remaining)[:per_family]
    return {
        "per_family": per_family,
        "selected": selected,
        "unclassified": [r["id"] for r in index["references"] if r["family"] == "unclassified"],
        "selection_scope": "analysis representatives only; no compiler policy or generalization claim",
    }


if __name__ == "__main__":
    if sys.argv[1:] != ["--inspect-c"]:
        raise SystemExit("internal source-inspection worker only")
    source = sys.stdin.buffer.read(MAX_SOURCE_BYTES + 1)
    if len(source) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds structural analysis bound")
    record = _c_source(source.decode("utf-8"))
    from importlib.metadata import PackageNotFoundError, version

    for package in ("tree-sitter", "tree-sitter-c"):
        try:
            record.setdefault("dependencies", {})[package] = version(package)
        except PackageNotFoundError:
            pass
    serialized = json.dumps(record)
    if len(serialized.encode()) > 32 << 20:
        raise ValueError("source parser metadata exceeds its output bound")
    print(serialized)
