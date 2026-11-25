"""Explicit metadata indexing cannot promote an experiment verdict."""

import importlib.util
import json

import pytest

from merlin.common.digest import sha256_bytes
from merlin.common.paths import repo_root


def tool():
    path = repo_root() / "examples/gemmini/model-study/conditions.py"
    spec = importlib.util.spec_from_file_location("model_study_conditions_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def selected(tmp_path):
    module = tool()
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "failed", "nested": {"path": str(tmp_path / "absent.npz")}}))
    digest = sha256_bytes(path.read_bytes())
    return module, path, digest


def index_file(module, path, digest, output):
    document = module.make_index("trial", [("result", "run", path, digest)])
    raw = json.dumps(document).encode()
    output.write_bytes(raw)
    return document, sha256_bytes(raw)


def test_failed_record_and_missing_nested_payload_are_identity_only(selected, tmp_path):
    module, path, digest = selected
    before = path.read_bytes()
    document, expected = index_file(module, path, digest, tmp_path / "index.json")
    assert module.verify_index(tmp_path / "index.json", expected) == document
    assert set(document) == {"schema", "scope", "label", "entries"}
    assert document["scope"] == module.SCOPE
    assert set(document["entries"][0]) == {"name", "role", "pin"}
    assert '"status":' not in json.dumps(document)
    assert not (tmp_path / "absent.npz").exists()
    assert path.read_bytes() == before


def test_changed_named_file_refuses_even_with_same_size(selected, tmp_path):
    module, path, digest = selected
    _, expected = index_file(module, path, digest, tmp_path / "index.json")
    path.write_bytes(path.read_bytes().replace(b"failed", b"passed"))
    with pytest.raises(ValueError, match="digest differs"):
        module.verify_index(tmp_path / "index.json", expected)


def test_changed_index_requires_independently_selected_digest(selected, tmp_path):
    module, path, digest = selected
    _, expected = index_file(module, path, digest, tmp_path / "index.json")
    with (tmp_path / "index.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="digest differs"):
        module.verify_index(tmp_path / "index.json", expected)


@pytest.mark.parametrize("change", ["verdict", "scope", "bytes", "duplicate", "role"])
def test_repinned_index_cannot_add_claims_or_ambiguous_entries(selected, tmp_path, change):
    module, path, digest = selected
    document = module.make_index("trial", [("result", "run", path, digest)])
    if change == "verdict":
        document["numerical_status"] = "pass"
    elif change == "scope":
        document["scope"] = "hardware_verified"
    elif change == "bytes":
        document["entries"][0]["pin"]["bytes"] = True
    elif change == "duplicate":
        document["entries"].append(document["entries"][0])
    else:
        document["entries"][0]["role"] = "tensor"
    raw = json.dumps(document).encode()
    output = tmp_path / "index.json"
    output.write_bytes(raw)
    with pytest.raises(ValueError):
        module.verify_index(output, sha256_bytes(raw))


def test_duplicate_json_keys_refuse_before_indexing(selected):
    module, path, _ = selected
    path.write_bytes(b'{"status":"pass","status":"fail"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        module.make_index("trial", [("result", "run", path, sha256_bytes(path.read_bytes()))])


def test_config_is_pinned_without_parsing_or_copying_values(tmp_path):
    module = tool()
    path = tmp_path / "models.yaml"
    path.write_text("private_value: retained-only-at-input\n")
    document = module.make_index("trial", [("config", "models", path, sha256_bytes(path.read_bytes()))])
    assert "retained-only-at-input" not in json.dumps(document)


@pytest.mark.parametrize("kind", ["directory", "symlink", "symlink_parent", "payload", "too_large"])
def test_only_bounded_ordinary_explicit_metadata_files(selected, tmp_path, kind):
    module, path, digest = selected
    if kind == "directory":
        path = tmp_path
    elif kind == "symlink":
        alias = tmp_path / "alias.json"
        alias.symlink_to(path)
        path = alias
    elif kind == "symlink_parent":
        alias = tmp_path / "alias"
        alias.symlink_to(tmp_path, target_is_directory=True)
        path = alias / path.name
    elif kind == "payload":
        path = tmp_path / "expected.npz"
        path.write_bytes(b"not metadata")
    else:
        path.write_bytes(b"x" * (module.MAX_FILE_BYTES + 1))
    with pytest.raises((OSError, ValueError)):
        module.make_index("trial", [("result", "run", path, digest)])


def test_entry_and_aggregate_limits_apply_to_create_and_verify(selected, tmp_path, monkeypatch):
    module, path, digest = selected
    with pytest.raises(ValueError, match="1..64"):
        module.make_index("trial", [("result", str(i), path, digest) for i in range(65)])
    _, expected = index_file(module, path, digest, tmp_path / "index.json")
    monkeypatch.setattr(module, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="total byte limit"):
        module.verify_index(tmp_path / "index.json", expected)


def test_cli_creation_is_exclusive_and_verification_does_not_report_pass(selected, tmp_path, capsys):
    module, path, digest = selected
    output = tmp_path / "index.json"
    argv = ["create", "--label", "trial", "--result", "run", str(path), digest, "--output", str(output)]
    assert module.main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "recorded"
    before = output.read_bytes()
    with pytest.raises(SystemExit) as error:
        module.main(argv)
    assert error.value.code == 2 and output.read_bytes() == before
    capsys.readouterr()
    assert module.main(["verify", "--index", str(output), "--index-sha256", report["index_sha256"]]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "identities_verified"


def test_wrong_explicit_pin_leaves_no_output(selected, tmp_path):
    module, path, _ = selected
    output = tmp_path / "index.json"
    with pytest.raises(SystemExit):
        module.main(["create", "--label", "trial", "--result", "run", str(path), "0" * 64, "--output", str(output)])
    assert not output.exists()
