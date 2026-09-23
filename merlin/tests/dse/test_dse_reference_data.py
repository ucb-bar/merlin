"""Curated DSE evidence stays byte-stable and separate from writable analysis outputs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.dse_guidance import cli, reference_data


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "new-output"))
    monkeypatch.delenv("MERLIN_DSE_REFERENCE_DIR", raising=False)
    return tmp_path


def _snapshot(root: Path, marker: str) -> Path:
    root.mkdir(parents=True)
    (root / "dse_contract.json").write_text(
        json.dumps({"workloads": [marker], "per_workload": {}, "what_is_not_claimed": "speedup"})
    )
    return root


def test_committed_evidence_is_present_and_byte_identical():
    parent = repo_root() / "experiments" / "reference-data" / "dse"
    migration = json.loads((parent / "MIGRATION.json").read_text())
    snapshot = parent / "case_study"
    assert (snapshot / "dse_contract.json").is_file(), "missing references must fail, never skip tests"
    assert len(migration["files"]) == 265
    for row in migration["files"]:
        assert hashlib.sha256((snapshot / row["path"]).read_bytes()).hexdigest() == row["sha256"], row["path"]


def test_git_preserves_reference_bytes_in_attributes_and_index():
    root = repo_root()
    migration = json.loads((root / "experiments/reference-data/dse/MIGRATION.json").read_text())
    paths = [f"{migration['destination_root']}/{row['path']}" for row in migration["files"]]
    for options in ([], ["--cached"]):
        attributes = (
            subprocess.run(
                ["git", "check-attr", *options, "-z", "--stdin", "text", "eol"],
                cwd=root,
                input="\0".join(paths).encode() + b"\0",
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .split("\0")[:-1]
        )
        assert len(attributes) == 6 * len(paths)
        for index in range(0, len(attributes), 3):
            path, attribute, value = attributes[index : index + 3]
            assert value == "unset", f"{path}: {attribute}={value}; archived bytes must not be normalized"
    objects = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input="".join(f":{path}\n" for path in paths).encode(),
        capture_output=True,
        check=True,
    ).stdout
    offset = 0
    for path, row in zip(paths, migration["files"], strict=True):
        end = objects.index(b"\n", offset)
        header = objects[offset:end].split()
        assert len(header) == 3 and header[1] == b"blob", f"missing indexed reference payload: {path}"
        size = int(header[2])
        payload = objects[end + 1 : end + 1 + size]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"], f"indexed bytes changed: {path}"
        offset = end + 1 + size + 1
    assert offset == len(objects)


def test_reference_wins_over_both_generated_and_historical_output(rooted):
    canonical = _snapshot(rooted / "experiments" / "reference-data" / "dse" / "case_study", "reference")
    _snapshot(rooted / "out" / "artifacts" / "dse-guidance" / "case_study", "historical")
    _snapshot(rooted / "new-output" / "artifacts" / "dse-guidance" / "case_study", "new-output")
    assert reference_data.case_study_dir() == canonical


def test_historical_checkout_fallback_ignores_output_override(rooted):
    historical = _snapshot(rooted / "out" / "artifacts" / "dse-guidance" / "case_study", "historical")
    assert reference_data.case_study_dir() == historical
    assert not (rooted / "new-output").exists()


def test_explicit_and_environment_inputs_do_not_fall_back(rooted, monkeypatch):
    canonical = _snapshot(rooted / "experiments" / "reference-data" / "dse" / "case_study", "reference")
    selected = rooted / "operator-selection"
    monkeypatch.setenv("MERLIN_DSE_REFERENCE_DIR", str(selected))
    assert reference_data.case_study_dir() == selected
    assert reference_data.case_study_dir(canonical) == canonical
    assert not selected.exists()


def test_query_reads_reference_without_creating_output(rooted, capsys):
    _snapshot(rooted / "experiments" / "reference-data" / "dse" / "case_study", "reference")
    assert cli.main(["--query", "summary"]) == 0
    assert "workloads: reference" in capsys.readouterr().out
    assert not (rooted / "new-output").exists()


@pytest.mark.parametrize("flag", ["--case-study-dir", "--out"])
def test_query_can_explicitly_consume_generated_output(rooted, capsys, flag):
    selected = _snapshot(rooted / "new-output" / "study", "generated")
    assert cli.main(["--query", "summary", flag, str(selected)]) == 0
    assert "workloads: generated" in capsys.readouterr().out


def test_case_study_generation_never_defaults_to_reference(rooted, monkeypatch):
    from merlin.dse_guidance import case_study

    reference = _snapshot(rooted / "experiments" / "reference-data" / "dse" / "case_study", "reference")
    before = (reference / "dse_contract.json").read_bytes()
    destinations = []

    def generate(path):
        destinations.append(path)
        return {"workloads": []}

    monkeypatch.setattr(case_study, "run_case_study", generate)
    assert cli.main(["--case-study"]) == 0
    assert destinations == [rooted / "new-output" / "artifacts" / "dse-guidance" / "case_study"]
    assert (reference / "dse_contract.json").read_bytes() == before
