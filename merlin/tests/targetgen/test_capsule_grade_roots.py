"""capsule_grade must resolve its roots absolutely before grading starts.

`run_capsule` chdirs into the build tree, so a relative `--contract` / `--capsules` stops resolving
partway through. The failure surfaces as `FileNotFoundError: contract schema not found: merlin/
contract/schemas/capsule_result.schema.json` raised inside a worker thread -- which reads as a broken
submission rather than a bad path, and it fired even when the CLI was invoked from the repo root,
because the defaults were relative string literals.

Both defaults now come from `merlin.common.paths.data_path`, per the repo rule that roots are asked
for rather than spelled out.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common.paths import data_path
from merlin.targetgen import capsule_grade


def test_defaults_are_absolute():
    """The parser's own defaults must not be relative strings."""
    import inspect
    src = inspect.getsource(capsule_grade.main)
    assert 'default="merlin/contract"' not in src, "relative contract default reintroduced"
    assert 'default="merlin/contract/capsules"' not in src, "relative capsules default reintroduced"
    assert "data_path(" in src, "roots must come from the paths helper"
    assert ".resolve()" in src, "a user-supplied relative root must be resolved"


def test_the_contract_root_actually_holds_the_schema_that_broke():
    schema = data_path("contract") / "schemas" / "capsule_result.schema.json"
    assert schema.is_file(), f"the schema the grader loads must exist at {schema}"
    assert schema.is_absolute()


def test_a_relative_root_resolves_to_the_same_place(tmp_path, monkeypatch):
    """Resolution must not depend on the CWD at the moment the runner happens to chdir."""
    root = data_path("contract")
    monkeypatch.chdir(root.parent)
    assert Path("contract").resolve() == root
    monkeypatch.chdir(tmp_path)
    # the previously-relative form no longer resolves here -- which is exactly the bug
    assert not (Path("merlin/contract") / "schemas").is_dir()
