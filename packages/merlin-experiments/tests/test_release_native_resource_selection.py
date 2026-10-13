"""Host resource changes preserve original contract bytes and semantic declarations."""

import copy

import pytest
import yaml
from merlin.common.digest import sha256_file
from merlin_experiments.corpus.preparation import _stage_capability_contract
from merlin_experiments.spec import SpecError


def _source(tmp_path):
    document = {
        "name": "fixture", "numeric_policy": {"kind": "exact"},
        "runner": {"backend": "chipyard_rocc", "chipyard_rocc": {
            "version": 1, "config": "fixture", "toolchain": {"selected": "original"},
            "engines": {
                "gsim": {"max_timeout_s": 600, "argv": ["{elf}", "+max-cycles=20000000"]},
                "spike": {"max_timeout_s": 600, "argv": ["{elf}"]},
            },
        }},
    }
    path = tmp_path / "original.yaml"
    path.write_text(yaml.safe_dump(document))
    return path, document


def test_explicit_resource_selection_preserves_source_and_all_nonresource_fields(tmp_path):
    source, original = _source(tmp_path)
    before = source.read_bytes()
    destination = tmp_path / "selected.yaml"
    record = _stage_capability_contract(source, sha256_file(source), destination, 1800)
    selected = yaml.safe_load(destination.read_text())
    restored = copy.deepcopy(selected)
    block = restored["runner"]["chipyard_rocc"]
    assert block.pop("execution_resources") == {"max_native_timeout_s": 1800}
    assert block["engines"]["gsim"]["max_timeout_s"] == 1800
    block["engines"]["gsim"]["max_timeout_s"] = 600
    assert restored == original
    assert source.read_bytes() == before
    assert record["sha256"] == sha256_file(source)
    assert record["execution_resource_selection"]["selected_contract_sha256"] == sha256_file(destination)
    assert record["execution_resource_selection"]["engines"] == ["gsim"]


def test_default_selection_keeps_exact_original_bytes(tmp_path):
    source, _ = _source(tmp_path)
    destination = tmp_path / "selected.yaml"
    record = _stage_capability_contract(source, sha256_file(source), destination, None)
    assert destination.read_bytes() == source.read_bytes()
    assert "execution_resource_selection" not in record


@pytest.mark.parametrize("budget", [True, 0, -1, 3601, 1800.0])
def test_invalid_resource_selection_publishes_no_contract(tmp_path, budget):
    source, _ = _source(tmp_path)
    destination = tmp_path / "selected.yaml"
    with pytest.raises(SpecError):
        _stage_capability_contract(source, sha256_file(source), destination, budget)
    assert not destination.exists()


def test_changed_original_contract_refuses_before_publication(tmp_path):
    source, _ = _source(tmp_path)
    digest = sha256_file(source)
    source.write_text("changed original\n")
    destination = tmp_path / "selected.yaml"
    with pytest.raises(SpecError, match="differs from verified evidence"):
        _stage_capability_contract(source, digest, destination, 1800)
    assert not destination.exists()
