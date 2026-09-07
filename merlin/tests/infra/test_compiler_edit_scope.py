import hashlib
import json
import shutil

import pytest

from merlin.perf.compiler_edit_scope import (
    inspect_compiler_edits,
    inspect_round_mechanism_edits,
    validate_edit_contract,
    validate_mechanism_catalog,
)


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


def _seal(document):
    document = dict(document)
    document["sha256"] = hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return document


def round_setup(tmp_path):
    initial = tmp_path / "initial"
    (initial / "compiler").mkdir(parents=True)
    (initial / "compiler/a.py").write_text(
        "import os\nCONFIG = 1\n\n"
        "def tune_a():\n    return 1\n\n"
        "class Planner:\n"
        "    mode = 'base'\n\n"
        "    def choose(self):\n        return 1\n\n"
        "    def unrelated(self):\n        return 2\n")
    (initial / "compiler/b.py").write_text("def tune_b():\n    return 1\n")
    (initial / "manifest.yaml").write_text(
        "entrypoints: {compile: old}\noptimization_surfaces: []\n")
    round_start, candidate = tmp_path / "round_start", tmp_path / "candidate"
    shutil.copytree(initial, round_start)
    shutil.copytree(initial, candidate)
    contract = _seal({"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": "a", "path": "compiler/a.py", "symbol": "tune_a"},
        {"surface_id": "planner", "path": "compiler/a.py", "symbol": "Planner"},
        {"surface_id": "b", "path": "compiler/b.py", "symbol": "tune_b"},
    ], "helper_extensions": [{"directory": "compiler",
        "surface_ids": ["a", "b"], "reason": "host-owned shared helper component"}]})
    validate_edit_contract(contract, initial)
    return initial, round_start, candidate, contract


def catalog(contract, mechanisms):
    return _seal({"schema": "compiler_mechanism_catalog_v1",
                  "contract_sha256": contract["sha256"],
                  "mechanisms": mechanisms})


def test_one_mechanism_may_span_multiple_exact_functions_and_paths(tmp_path):
    initial, start, candidate, contract = round_setup(tmp_path)
    frozen = catalog(contract, [{"id": "shared-layout", "selectors": [
        {"kind": "function", "path": "compiler/a.py", "symbol": "tune_a"},
        {"kind": "function", "path": "compiler/b.py", "symbol": "tune_b"},
    ]}])
    (candidate / "compiler/a.py").write_text(
        (candidate / "compiler/a.py").read_text().replace(
            "def tune_a():\n    return 1", "def tune_a():\n    return 3"))
    (candidate / "compiler/b.py").write_text(
        (candidate / "compiler/b.py").read_text().replace("return 1", "return 4"))

    result = inspect_round_mechanism_edits(initial, start, candidate, contract, frozen)
    assert result["status"] == "allowed"
    assert result["selected_mechanism_id"] == "shared-layout"
    assert {(row["path"], row["symbol"]) for row in result["semantic_ast_units"]} == {
        ("compiler/a.py", "tune_a"), ("compiler/b.py", "tune_b")}


def test_two_host_catalogued_mechanisms_in_one_round_are_rejected(tmp_path):
    initial, start, candidate, contract = round_setup(tmp_path)
    frozen = catalog(contract, [
        {"id": "layout", "selectors": [
            {"kind": "function", "path": "compiler/a.py", "symbol": "tune_a"}]},
        {"id": "latency", "selectors": [
            {"kind": "method", "path": "compiler/a.py", "symbol": "Planner.choose"}]},
    ])
    source = candidate / "compiler/a.py"
    source.write_text(source.read_text().replace("return 1", "return 9"))

    result = inspect_round_mechanism_edits(initial, start, candidate, contract, frozen)
    assert result["status"] == "refused"
    assert result["mechanism_ids"] == ["latency", "layout"]
    assert any(row["reason"] == "round contains more than one optimization mechanism"
               for row in result["violations"])


def test_broad_class_authority_does_not_hide_unrelated_method_edit(tmp_path):
    initial, start, candidate, contract = round_setup(tmp_path)
    frozen = catalog(contract, [{"id": "planner-shape", "selectors": [
        {"kind": "class", "path": "compiler/a.py", "symbol": "Planner"},
    ]}])
    source = candidate / "compiler/a.py"
    source.write_text(source.read_text().replace(
        "def unrelated(self):\n        return 2",
        "def unrelated(self):\n        return 7"))

    result = inspect_round_mechanism_edits(initial, start, candidate, contract, frozen)
    assert result["cumulative_candidate_scope"]["status"] == "allowed"
    assert result["status"] == "refused"
    assert [(row["kind"], row["symbol"], row["mechanism_ids"])
            for row in result["semantic_ast_units"]] == [
                ("method", "Planner.unrelated", [])]


@pytest.mark.parametrize("unit,selector,edit", [
    ("method", {"kind": "method", "path": "compiler/a.py", "symbol": "Planner.choose"},
     lambda text: text.replace("def choose(self):\n        return 1",
                               "def choose(self):\n        return 8")),
    ("class", {"kind": "class", "path": "compiler/a.py", "symbol": "Planner"},
     lambda text: text.replace("mode = 'base'", "mode = 'tiled'")),
    ("imports", {"kind": "imports", "path": "compiler/a.py"},
     lambda text: "import math\n" + text),
])
def test_method_class_and_import_selectors_are_explicit_and_narrow(
        tmp_path, unit, selector, edit):
    initial, start, candidate, contract = round_setup(tmp_path)
    frozen = catalog(contract, [{"id": unit, "selectors": [selector]}])
    source = candidate / "compiler/a.py"
    source.write_text(edit(source.read_text()))
    result = inspect_round_mechanism_edits(initial, start, candidate, contract, frozen)
    assert result["status"] == "allowed", result
    assert result["selected_mechanism_id"] == unit
    assert {row["kind"] for row in result["semantic_ast_units"]} == {unit}


def test_module_selector_attributes_but_does_not_expand_cumulative_authority(tmp_path):
    initial, start, candidate, contract = round_setup(tmp_path)
    frozen = catalog(contract, [{"id": "module-state", "selectors": [
        {"kind": "module", "path": "compiler/a.py"},
    ]}])
    source = candidate / "compiler/a.py"
    source.write_text(source.read_text().replace("CONFIG = 1", "CONFIG = 2"))

    result = inspect_round_mechanism_edits(initial, start, candidate, contract, frozen)
    assert [(row["kind"], row["mechanism_ids"])
            for row in result["semantic_ast_units"]] == [("module", ["module-state"])]
    assert result["selected_mechanism_id"] == "module-state"
    assert result["status"] == "refused"
    assert result["cumulative_candidate_scope"]["status"] == "refused"


def test_host_frozen_helper_directory_attributes_new_and_cumulative_helpers(tmp_path):
    initial, start, candidate, contract = round_setup(tmp_path)
    frozen = catalog(contract, [{"id": "shared-helper", "selectors": [
        {"kind": "helper", "directory": "compiler"},
    ]}])
    (start / "compiler/generated.py").write_text("def tile():\n    return 1\n")
    shutil.copy2(start / "compiler/generated.py", candidate / "compiler/generated.py")
    (candidate / "compiler/generated.py").write_text("def tile():\n    return 16\n")

    result = inspect_round_mechanism_edits(initial, start, candidate, contract, frozen)
    assert result["status"] == "allowed", result
    assert result["selected_mechanism_id"] == "shared-helper"
    assert result["cumulative_start_scope"]["status"] == "allowed"
    assert result["cumulative_candidate_scope"]["status"] == "allowed"


def test_formatting_only_round_is_an_explicit_noop(tmp_path):
    initial, start, candidate, contract = round_setup(tmp_path)
    frozen = catalog(contract, [{"id": "layout", "selectors": [
        {"kind": "function", "path": "compiler/a.py", "symbol": "tune_a"},
    ]}])
    source = candidate / "compiler/a.py"
    source.write_text(source.read_text().replace("return 1", "return    1"))

    result = inspect_round_mechanism_edits(initial, start, candidate, contract, frozen)
    assert result["status"] == "allowed"
    assert result["semantic_noop"] is True
    assert result["formatting_only_noop"] is True
    assert result["selected_mechanism_id"] is None


def test_round_start_cannot_launder_a_prior_out_of_authority_edit(tmp_path):
    initial, start, candidate, contract = round_setup(tmp_path)
    frozen = catalog(contract, [{"id": "layout", "selectors": [
        {"kind": "function", "path": "compiler/a.py", "symbol": "tune_a"},
    ]}])
    for root in (start, candidate):
        source = root / "compiler/a.py"
        source.write_text(source.read_text().replace("CONFIG = 1", "CONFIG = 9"))

    result = inspect_round_mechanism_edits(initial, start, candidate, contract, frozen)
    assert result["semantic_noop"] is True
    assert result["status"] == "refused"
    assert result["cumulative_start_scope"]["status"] == "refused"
    assert result["cumulative_candidate_scope"]["status"] == "refused"
    assert any(row["reason"] == "round start already exceeds cumulative edit authority"
               for row in result["violations"])


def test_catalog_is_host_bound_and_rejects_kind_mismatch_or_cross_group_overlap(tmp_path):
    initial, _start, _candidate, contract = round_setup(tmp_path)
    wrong_kind = catalog(contract, [{"id": "bad", "selectors": [
        {"kind": "function", "path": "compiler/a.py", "symbol": "Planner.choose"},
    ]}])
    with pytest.raises(ValueError, match="exact AST kind"):
        validate_mechanism_catalog(wrong_kind, initial, contract)

    overlap = catalog(contract, [
        {"id": "one", "selectors": [{"kind": "imports", "path": "compiler/a.py"}]},
        {"id": "two", "selectors": [{"kind": "imports", "path": "compiler/a.py"}]},
    ])
    with pytest.raises(ValueError, match="two mechanisms"):
        validate_mechanism_catalog(overlap, initial, contract)

    wrong_contract = dict(catalog(contract, [{"id": "one", "selectors": [
        {"kind": "imports", "path": "compiler/a.py"}]}]))
    wrong_contract["contract_sha256"] = "0" * 64
    wrong_contract = _seal({key: value for key, value in wrong_contract.items()
                            if key != "sha256"})
    with pytest.raises(ValueError, match="contract"):
        validate_mechanism_catalog(wrong_contract, initial, contract)
