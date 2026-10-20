"""Staging the declared Phase 0 capture inputs from a workload roster (no capture runs)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import workload_roster as WR

from merlin.common.paths import repo_root

TARGET = repo_root() / "examples" / "gemmini" / "target"


def _roster(tmp_path: Path, document: dict) -> Path:
    loader = tmp_path / "src" / "net" / "loader.py"
    loader.parent.mkdir(parents=True, exist_ok=True)
    loader.write_text("MODEL = 1\n")
    path = tmp_path / "roster.yaml"
    path.write_text(yaml.safe_dump({"schema": WR.SCHEMA, **document}))
    return path


def test_the_target_roster_stages_exactly_the_declared_labels(tmp_path):
    record = WR.stage(TARGET / "phase0-workloads.yaml", tmp_path / "staged", descriptor=TARGET / "descriptor.yaml")
    spec = yaml.safe_load((TARGET / "descriptor.yaml").read_text())["workload_spec"]
    for section in WR.SECTIONS:
        assert sorted(record[section]) == sorted(spec[section])
        for label, row in record[section].items():
            root = Path(row["workload_root"])
            assert root == tmp_path / "staged" / section / label
            loader = (root / "loader.py").read_bytes()
            assert loader == Path(row["source_loader"]).read_bytes()
            assert hashlib.sha256(loader).hexdigest() == row["loader_sha256"]
            profile = root / "profile.json"
            assert profile.is_file() == (row["profile_sha256"] is not None)
            if profile.is_file():
                assert hashlib.sha256(profile.read_bytes()).hexdigest() == row["profile_sha256"]
    # The decode step is the prefill loader under its own profile.
    step = record["applications"]["causal_decoder_decode_step"]
    assert step["source_loader"] == record["applications"]["causal_decoder"]["source_loader"]
    assert json.loads((Path(step["workload_root"]) / "profile.json").read_text())["decode_step"] is True
    assert json.loads((tmp_path / "staged" / "staged-workloads.json").read_text()) == record


def test_profiles_are_accepted_by_the_loaders_that_read_them(tmp_path):
    import importlib.util

    pytest.importorskip("torch")
    record = WR.stage(TARGET / "phase0-workloads.yaml", tmp_path / "staged")
    for rows in (record["applications"], record["performance_applications"]):
        for label, row in rows.items():
            root = Path(row["workload_root"])
            spec = importlib.util.spec_from_file_location(f"staged_{label}", root / "loader.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            validate = getattr(module, "validate_profile", None)
            if validate is not None and (root / "profile.json").is_file():
                validate(json.loads((root / "profile.json").read_text()))


def test_a_roster_that_differs_from_the_descriptor_is_refused(tmp_path):
    roster = _roster(tmp_path, {"applications": {"net": {"loader": "src/net/loader.py"}}})
    with pytest.raises(ValueError, match="applications: the roster stages"):
        WR.stage(roster, tmp_path / "out", descriptor=TARGET / "descriptor.yaml")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ({}, "declares no workload"),
        ({"applications": {"Net": {"loader": "src/net/loader.py"}}}, "lower-case identifiers"),
        (
            {
                "applications": {"net": {"loader": "src/net/loader.py"}},
                "performance_applications": {"net": {"loader": "src/net/loader.py"}},
            },
            "unique",
        ),
        ({"applications": {"net": {"loader": "src/net/absent.py"}}}, "existing loader.py"),
        ({"applications": {"net": {"loader": "src/net/loader.py", "extra": 1}}}, "exactly loader"),
        ({"applications": {"net": {"loader": "src/net/loader.py", "profile": {"schema": "x"}}}}, "schema"),
        ({"applications": {}, "holdout": {}}, "unknown key"),
    ],
)
def test_a_malformed_roster_is_refused(tmp_path, document, match):
    with pytest.raises(ValueError, match=match):
        WR.load(_roster(tmp_path, document))


def test_staging_needs_a_fresh_output_outside_the_loaders(tmp_path):
    roster = _roster(tmp_path, {"applications": {"net": {"loader": "src/net/loader.py", "profile": {"a": 1}}}})
    (tmp_path / "taken").mkdir()
    with pytest.raises(ValueError, match="fresh directory"):
        WR.stage(roster, tmp_path / "taken")
    with pytest.raises(ValueError, match="inside a selected loader directory"):
        WR.stage(roster, tmp_path / "src" / "net" / "staged")
    record = WR.stage(roster, tmp_path / "out")
    staged = tmp_path / "out" / "applications" / "net"
    assert (staged / "profile.json").read_bytes() == WR.profile_bytes({"a": 1})
    assert record["applications"]["net"]["profile_sha256"] == hashlib.sha256(WR.profile_bytes({"a": 1})).hexdigest()
