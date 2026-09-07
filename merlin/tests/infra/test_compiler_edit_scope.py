import hashlib
import json

import pytest

from merlin.perf.compiler_edit_scope import inspect_compiler_edits, validate_edit_contract


def setup(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    for root in (before, after):
        (root / "compiler").mkdir(parents=True)
        (root / "compiler/codegen.py").write_text("LIMIT=16\ndef allowed():\n return 1\ndef protected():\n return 2\n")
        (root / "manifest.yaml").write_text("entrypoints: {compile: old}\noptimization_surfaces: []\n")
    contract = {"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": "compile", "path": "compiler/codegen.py", "symbol": "allowed"}],
        "helper_extensions": [{"directory": "compiler", "surface_ids": ["compile"], "reason": "explicit host helper scope"}]}
    contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    validate_edit_contract(contract, before)
    return before, after, contract


@pytest.mark.parametrize("change,allowed", [
    ("owned", True), ("new_helper", True), ("import", True),
    ("protected", False), ("constant", False), ("manifest", False), ("self_authorize", False)])
def test_actual_ast_scope_and_no_manifest_self_authority(tmp_path, change, allowed):
    before, after, contract = setup(tmp_path)
    file = after / "compiler/codegen.py"
    if change == "owned":
        file.write_text(file.read_text().replace("return 1", "return 3"))
    elif change == "new_helper":
        file.write_text(file.read_text()+"\ndef new_helper():\n return 4\n")
        (after / "compiler/new.py").write_text("def new():\n return 1\n")
    elif change == "import":
        file.write_text("import math\n"+file.read_text())
    elif change in ("protected", "constant"):
        file.write_text(file.read_text().replace("return 2" if change == "protected" else "LIMIT=16", "return 7" if change == "protected" else "LIMIT=8"))
    elif change == "manifest":
        (after / "manifest.yaml").write_text("entrypoints: {compile: malicious}\n")
    else:
        (after / "unauthorized.py").write_text("def new():\n return 1\n")
        (after / "manifest.yaml").write_text("entrypoints: {compile: old}\noptimization_surfaces: [{path: unauthorized.py, symbol: new}]\n")
    assert (inspect_compiler_edits(before, after, contract)["status"] == "allowed") is allowed


def test_contract_mutation_and_linked_source_fail_closed(tmp_path):
    before, after, contract = setup(tmp_path)
    contract["helper_extensions"][0]["directory"] = "../outside"
    with pytest.raises(ValueError, match="identity"):
        validate_edit_contract(contract, before)
    (after / "compiler/link.py").symlink_to(before / "compiler/codegen.py")
    assert inspect_compiler_edits(before, after, {**contract, "helper_extensions": []})["status"] == "refused"
