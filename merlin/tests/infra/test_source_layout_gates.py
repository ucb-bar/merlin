"""Architecture gates must examine relocated code, not pass because the old tree is empty."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root


def _gate(name):
    script = repo_root() / "build_tools/scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"layout_test_{name}", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(root, rel, contents="x = 1\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def test_reference_data_is_research_owned_not_a_core_dependency(tmp_path, monkeypatch):
    gate = _gate("check_structure")
    monkeypatch.setattr(gate, "ROOT", str(tmp_path))
    paths = {
        "src/merlin/core.py": 'root / "experiments/reference-data/study"',
        "packages/merlin-dse/src/merlin/dse/data.py": 'root / "experiments/reference-data/study"',
        "packages/merlin-dse/src/merlin/dse/engine.py": 'root / "experiments/engine.py"',
    }
    for name, text in paths.items():
        _write(tmp_path, name, text)
    errors = []
    gate.check_library_boundary(errors)
    assert len(errors) == 2, errors
    assert any("core.py" in error for error in errors)
    assert any("engine.py" in error for error in errors)


def test_example_owned_target_tests_do_not_require_native_implementations(tmp_path, monkeypatch):
    gate = _gate("check_structure")
    monkeypatch.setattr(gate, "ROOT", str(tmp_path))
    assert gate._example_test_buckets() == set()
    _write(tmp_path, "examples/device_a/target/descriptor.yaml", "target: fixture_a\n")
    _write(tmp_path, "examples/device_b/target/contracts/target_contract.yaml", "name: fixture_b\n")
    _write(tmp_path, "examples/unrelated/README.md", "not a target\n")
    assert gate._example_test_buckets() == {"device_a", "device_b"}
    assert gate._discovered_targets() == []


def test_overfit_inventory_preserves_identity_not_arbitrary_new_modules(monkeypatch):
    gate = _gate("check_overfit_register")
    inventory = [
        "src/merlin/driver.py:12: coupling",
        "packages/merlin-analysis/src/merlin/compare/driver.py:5: coupling",
        "packages/merlin-experiments/src/merlin_experiments/driver.py:9: coupling",
    ]
    monkeypatch.setattr(gate, "_load_name_gate", lambda: SimpleNamespace(coupling_inventory=lambda: inventory))
    assert gate._coupled_files() == {
        "merlin/python/merlin/driver.py",
        "merlin/python/merlin/compare/driver.py",
        "packages/merlin-experiments/src/merlin_experiments/driver.py",
    }


def test_overfit_register_matches_moved_identity_and_rejects_new_coupling(monkeypatch, capsys):
    gate = _gate("check_overfit_register")
    entry = {
        "id": "existing-debt",
        "kind": "target_coupling",
        "status": "triaged",
        "owner": "test",
        "items": ["packages/merlin-analysis/src/merlin/compare/driver.py"],
        "why": "Existing dependency",
        "blocks": "Other targets",
        "removal_condition": "Replace dependency",
    }
    monkeypatch.setattr(gate, "load_register", lambda: [entry])
    live = {"merlin/python/merlin/compare/driver.py"}
    monkeypatch.setattr(gate, "_coupled_files", lambda: live)
    assert gate.main([]) == 0
    assert "no longer couple" not in capsys.readouterr().out
    live.add("packages/merlin-experiments/src/merlin_experiments/driver.py")
    assert gate.main([]) == 1
    assert "merlin_experiments/driver.py" in capsys.readouterr().out


def test_retired_checkout_paths_do_not_match_new_in_package_adapters():
    gate = _gate("check_doc_paths")
    for needle, pre, post, _ in gate.RETIRED:
        if needle == "merlin/integrations":
            assert gate._match_retired("use merlin/integrations/tool", needle, pre, post)
            assert gate._match_retired("use src/merlin/integrations/tool", needle, pre, post) is None


@pytest.mark.parametrize(
    "name", ["check_no_regex", "check_no_target_name", "check_no_assumed_constants", "check_fact_provenance"]
)
def test_scanners_cover_src_and_extensions_once(tmp_path, monkeypatch, name):
    gate = _gate(name)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    _write(tmp_path, "src/merlin/probe.py")
    _write(tmp_path, "packages/merlin-analysis/src/merlin_analysis/probe.py")
    _write(tmp_path, "packages/merlin-analysis/tests/test_probe.py")
    _write(tmp_path, "src/merlin/_data/copied.py")
    legacy = tmp_path / "merlin/python/merlin"
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to("../../src/merlin", target_is_directory=True)
    actual = [str(path) for path in gate._iter_targets(False)]
    assert actual == ["src/merlin/probe.py", "packages/merlin-analysis/src/merlin_analysis/probe.py"]


def test_staged_renames_are_scanned_and_an_unreadable_index_still_fails(tmp_path, monkeypatch):
    gate = _gate("check_no_regex")
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    def renamed(command, **kwargs):
        assert "--diff-filter=ACMR" in command and kwargs["check"] is True
        return SimpleNamespace(stdout="src/merlin/renamed.py\npackages/merlin-analysis/tests/test_x.py\n")

    monkeypatch.setattr(gate.subprocess, "run", renamed)
    assert gate._iter_targets(True) == [Path("src/merlin/renamed.py")]

    def unreadable(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(gate.subprocess, "run", unreadable)
    with pytest.raises(subprocess.CalledProcessError):
        gate._iter_targets(True)


def test_src_uses_existing_exemption_but_extensions_do_not_inherit_it():
    gate = _gate("check_no_target_name")
    exact = {"merlin/python/merlin/old_debt.py"}
    assert gate._allowed("src/merlin/old_debt.py", exact, [])
    assert not gate._allowed("src/merlin/new_debt.py", exact, [])
    assert not gate._allowed("packages/merlin-analysis/src/merlin_analysis/old_debt.py", exact, [])
    assert gate._allowed("packages/merlin-dse/src/merlin/old_debt.py", exact, [])
    assert not gate._allowed("packages/merlin-dse/src/merlin/new_debt.py", exact, [])


def test_relocated_regex_violation_still_fails_without_growing_allowlist(tmp_path, monkeypatch, capsys):
    gate = _gate("check_no_regex")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    allowlist = _write(tmp_path, "allow.txt", "merlin/python/merlin/old_debt.py\n")
    monkeypatch.setattr(gate, "ALLOW_FILE", allowlist)
    _write(tmp_path, "src/merlin/old_debt.py", "import re\nre.compile('old')\n")
    assert gate.main([]) == 0
    _write(tmp_path, "packages/merlin-analysis/src/merlin_analysis/new.py", "import re\nre.compile('new')\n")
    assert gate.main([]) == 1
    assert "merlin_analysis/new.py:2" in capsys.readouterr().out
    assert allowlist.read_text() == "merlin/python/merlin/old_debt.py\n"


def test_module_size_ratchet_follows_core_but_not_new_extension(tmp_path, monkeypatch):
    gate = _gate("check_structure")
    monkeypatch.setattr(gate, "ROOT", str(tmp_path))
    monkeypatch.setattr(gate, "MODULE_SIZE_LIMIT", 2)
    ratchet = _write(tmp_path, "size.txt", "merlin/python/merlin/large.py\n")
    monkeypatch.setattr(gate, "MODULE_SIZE_RATCHET", str(ratchet))
    _write(tmp_path, "src/merlin/large.py", "x=1\ny=2\nz=3\n")
    _write(tmp_path, "packages/merlin-analysis/src/merlin_analysis/large.py", "x=1\ny=2\nz=3\n")
    errors = []
    gate.check_module_size(errors)
    assert len(errors) == 1 and "merlin_analysis/large.py" in errors[0]


def test_schema_usage_and_boundary_examine_src_without_compatibility_link(tmp_path, monkeypatch):
    gate = _gate("check_structure")
    monkeypatch.setattr(gate, "ROOT", str(tmp_path))
    monkeypatch.setattr(gate, "BOUNDARY_RATCHET", str(tmp_path / "no_ratchet"))
    _write(tmp_path, "merlin/schemas/example.schema.yaml", "type: object\n")
    _write(tmp_path, "src/merlin/probe.py", 'SCHEMA = "example"\nP = "experiments/private"\n')
    errors = []
    gate.check_schema_usage(errors)
    assert errors == []
    gate.check_library_boundary(errors)
    assert len(errors) == 1 and "src/merlin/probe.py:2" in errors[0]


def test_package_docs_include_extensions_and_canonical_module_names(tmp_path, monkeypatch):
    gate = _gate("gen_package_docs")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    _write(tmp_path, "src/merlin/__init__.py", '"""Core package."""\n')
    _write(tmp_path, "packages/merlin-analysis/src/merlin_analysis/__init__.py", '"""Optional reports."""\n')
    text = gate.gen_index()
    assert "| `merlin` | Core package. |" in text
    assert "| `merlin_analysis` | Optional reports. |" in text


def test_doc_freshness_follows_source_behind_legacy_alias(tmp_path, monkeypatch):
    gate = _gate("check_docs_freshness")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    _write(tmp_path, "src/merlin/probe.py")
    legacy = tmp_path / "merlin/python/merlin"
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to("../../src/merlin", target_is_directory=True)

    def history(command, **kwargs):
        assert command[-2:] == ["merlin/python/merlin/probe.py", "src/merlin/probe.py"]
        return SimpleNamespace(stdout="2026-09-20\n")

    monkeypatch.setattr(gate.subprocess, "run", history)
    assert gate._last_commit_date("merlin/python/merlin/probe.py") == "2026-09-20"


def test_formatter_checks_edited_renames_but_preserves_byte_identical_moves(monkeypatch):
    gate = _gate("check_format")

    def changed(*command):
        if command[0] in {"ls-files", "ls-tree"}:
            return "120000 deadbeef 0\tpackages/optional/setup.py\0"
        assert "--name-status" in command and "-z" in command and "--diff-filter=ACMR" in command
        return (
            "R100\0legacy.py\0src/untouched.py\0R090\0old.py\0src/edited.py\0"
            "A\0src/new.py\0A\0packages/optional/setup.py\0"
        )

    monkeypatch.setattr(gate, "_git", changed)
    assert gate._changed(True, None) == ["src/edited.py", "src/new.py"]
    assert gate._changed(False, "base") == ["src/edited.py", "src/new.py"]
