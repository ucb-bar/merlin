import importlib.util
import json

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def audit():
    path = repo_root() / "build_tools/scripts/audit_compiler_integration.py"
    spec = importlib.util.spec_from_file_location("compiler_integration_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mlir_byte_escapes_and_missing_attribute(audit):
    text = r'module attributes {test.schedule = "{\22family\22:\22a\\nb\22,\22result\22:\22applied\22}"}'
    assert audit.json_string_attribute(text, "test.schedule") == {"family": "a\nb", "result": "applied"}
    assert audit.json_string_attribute(text, "missing") is None
    with pytest.raises(ValueError):
        audit.json_string_attribute('test.schedule = "unterminated', "test.schedule")


def test_source_mismatch_never_credits_pass(audit, tmp_path):
    case = tmp_path / "case"
    generated = case / "generated"
    generated.mkdir(parents=True)
    source = "a" * 64
    (case / "capsule_result.json").write_text(json.dumps({"compiler_source": {"candidate_sha256": source}}))
    (generated / "lowered.target.mlir").write_text(
        r'test.schedule = "{\22family\22:\22tile\22,\22result\22:\22applied\22}"')
    snapshot = generated / "compiler-source" / source
    snapshot.mkdir(parents=True)
    file = snapshot / "tool.py"
    file.write_text("# original\n")
    sources = [{"path": "tool.py", "sha256": audit.digest(file)}]
    assert audit.receipt_inventory(tmp_path, "test.schedule", sources)["matched_sources"] == 1
    file.write_text("# different compiler\n")
    result = audit.receipt_inventory(tmp_path, "test.schedule", sources)
    assert result["per_family_matched_cases"] == []
    assert result["cases"][0]["families"] == [{"family": "tile", "result": "applied"}]


def test_unchanged_override_is_not_counted_as_a_modification(audit, tmp_path):
    submission, baseline = tmp_path / "submission", tmp_path / "baseline"
    override = submission / "compiler_library/merlin/example.py"
    original = baseline / "merlin/example.py"
    override.parent.mkdir(parents=True)
    original.parent.mkdir(parents=True)
    override.write_text("import math\n")
    original.write_text(override.read_text())
    sources, overrides = audit.source_inventory(submission, baseline)
    assert sources[0]["imports"] == ["math"]
    assert overrides[0]["comparison"] == "unchanged"
