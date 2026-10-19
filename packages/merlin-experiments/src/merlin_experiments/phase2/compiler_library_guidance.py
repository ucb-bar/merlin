"""Answer-free source inventory and projections of existing host edit authority.

AST declarations describe reviewed import availability. They do not identify
target capabilities, execute library code or authorize compiler edits.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from merlin.targetgen.compiler_library import active_compiler_library

from .contracts import StageGateError, document_sha256, mapping_file, write_json

MAX_DECLARATIONS = 256
MAX_PREVIEW = 320


def _declarations(body, prefix=""):
    rows = []
    for node in body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") and not (prefix and node.name == "__init__"):
                continue
            row = {
                "symbol": prefix + node.name,
                "line": node.lineno,
                "kind": "class" if isinstance(node, ast.ClassDef) else "callable",
            }
            doc = ast.get_docstring(node)
            if doc:
                row["documentation_preview"] = doc[:MAX_PREVIEW]
                row["documentation_truncated"] = len(doc) > MAX_PREVIEW
            if isinstance(node, ast.ClassDef):
                rows.append(row)
                rows.extend(_declarations(node.body, prefix + node.name + "."))
            else:
                signature = ast.unparse(node.args)
                row["parameters_preview"] = signature[:MAX_PREVIEW]
                row["parameters_truncated"] = len(signature) > MAX_PREVIEW
                rows.append(row)
        elif not prefix and isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if name != "*" and not name.startswith("_"):
                    rows.append({"symbol": name, "kind": "imported_name", "line": node.lineno})
    return rows


def selected_inventory():
    """Read exact reviewed source bytes without importing their implementation."""
    selection = active_compiler_library()
    if selection is None:
        return None
    from .compiler_libraries import guest_root

    before = selection.record()
    members = {row.module: row for row in selection.contract.members}
    modules, remaining = [], MAX_DECLARATIONS
    for module in sorted(selection.contract.public_modules):
        member = members[module]
        declarations = _declarations(ast.parse((selection.root / member.path).read_bytes()).body)
        retained = declarations[:remaining]
        remaining -= len(retained)
        modules.append(
            {
                "module": module,
                "source": str(guest_root(selection) / member.path),
                "source_sha256": member.sha256,
                "declarations": retained,
                "declaration_count": len(declarations),
                "omitted_declarations": len(declarations) - len(retained),
            }
        )
    if selection.record() != before:
        raise StageGateError("selected compiler library changed during public API inventory")
    return {
        "schema": "merlin.phase2.public_compiler_library_inventory.v1",
        "review_id": selection.contract.review_id,
        "contract_sha256": selection.contract.sha256,
        "public_modules": modules,
        "derivation": "AST declarations from exact reviewed source; no library imports executed",
        "declaration_scope": (
            "top-level definitions/imported names and public class methods; "
            "dynamic exports and value aliases unresolved"
        ),
        "library_edit_authority": "read-only reviewed algorithms, identical for both compared arms",
        "candidate_edit_authority": "only the host-frozen compiler edit contract; inventory grants no edits",
        "target_adapter_applicability": "UNKNOWN from source availability; requires selected software/RTL checks",
        "independent_qualification": "required; source inventory supplies no numerical, runtime or timing verdict",
        "algorithm_origin": (
            "reviewed authored library code; selected software effects and RTL facts constrain legal policy"
        ),
    }


def prompt_text(inventory):
    if inventory is None:
        return ""
    return (
        "\n## Additional reviewed compiler library\n\n"
        "The JSON below lists this selected release's public modules, exact read-only source paths, "
        "declaration lines and bounded API/docstring previews. Imported names are descriptive; inspect "
        "the exact source for signatures and guarantees. This is import availability, not target support. "
        "Generic algorithms are reviewed authored code. Legal sizes, layouts, resource bounds and effects "
        "derive from the selected software/ISA specification and RTL facts; a qualified adapter must connect them "
        "to independently checked emitted instructions. Missing applicability or timing remains UNKNOWN.\n\n"
        "Tune or modify candidate-owned policies and transformation bodies, or add components only where "
        "the host-frozen edit contract allows them. Read `STAGE_CONTEXT.json` -> "
        "`frozen_compiler_edit_authority` for exact existing path/symbol, helper and v2 new-component authority "
        "when queued search is enabled. Its absence grants no new component. The automatic optimization "
        "inventory/template and candidate manifest are descriptive, not permission. Selected shared-library "
        "leaves remain read-only; changing that release needs a separate host review and same-release comparison.\n\n"
        "For each hypothesis, identify the owned entrypoint, expected emitted-program delta and affected "
        "public shapes; freeze the source revision, inspect emitted bytes/work and independent correctness, "
        "then inspect original paired measurements or explicitly qualified predictions. Use search-status "
        "for pending/refused/failed cells and the unchanged objective. Roofline gaps guide experiments; "
        "UNKNOWN or partial floors cannot establish convergence, and an IR annotation alone is no improvement.\n"
    )


def prompt_context(inputs):
    inventory = getattr(inputs, "additional_compiler_library", None)
    return {"additional_compiler_library": copy.deepcopy(inventory)} if inventory is not None else {}


def publish_edit_authority(authority, candidate):
    """Project the already-frozen authority, excluding private seed/host paths."""
    authority.check_integrity()
    if not authority.configured or authority.contract is None:
        raise StageGateError("public edit guidance requires the actual frozen compiler edit authority")
    context_path = Path(candidate).parent / "STAGE_CONTEXT.json"
    if context_path.is_symlink() or not context_path.is_file():
        raise StageGateError("queued author context is absent or linked")
    projection = {
        "schema": "public_frozen_compiler_edit_authority_v1",
        "initial_candidate_sha256": authority.binding["initial_candidate_sha256"],
        "contract_document_sha256": authority.binding["contract_document_sha256"],
        "contract": copy.deepcopy(authority.contract),
        "scope": "description of existing host authority; editing this projection grants nothing",
    }
    projection["sha256"] = document_sha256(projection)
    context = mapping_file(context_path)
    context["frozen_compiler_edit_authority"] = projection
    write_json(context_path, context)
    authority.check_integrity()
    return projection
