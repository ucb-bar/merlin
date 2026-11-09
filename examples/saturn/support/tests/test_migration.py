"""Offline migration commitments, not compiler or hardware qualification."""

import hashlib
import json
from pathlib import Path

from merlin.targetgen.providers import ProviderRole, read_provider


ROOT = Path(__file__).resolve().parents[1]


def test_recorded_source_members_match_exact_bytes():
    record = json.loads((ROOT / "provenance.json").read_text())
    assert record["schema"] == "merlin.support_migration.v1"
    assert record["target"] == "saturn"
    members = record["files"]
    expected = {member["path"]: member["sha256"] for member in members}
    assert len(expected) == len(members)
    for member in members:
        assert member["source_revision"] == record["merlin_source_baseline"]
        assert member["source"] == "merlin/targets/saturn/" + member["path"]
    for filename, modified, added in (
        ("matrix_adapter_migration.json", {"contracts/target_contract.yaml"}, {"matrix_lowering.py"}),
        ("opu_shim_migration.json", {"matrix_lowering.py", "contracts/target_contract.yaml"},
         {"opu_shim.py", "contracts/matrix_units.yaml"}),
    ):
        transition = json.loads((ROOT / filename).read_text())
        assert transition["schema"] == "merlin.support_transformation.v1"
        assert {entry["path"] for entry in transition["modified"]} == modified
        assert {entry["path"] for entry in transition["added"]} == added
        assert len(transition["modified"]) == len(modified)
        assert len(transition["added"]) == len(added)
        for entry in transition["modified"]:
            assert expected[entry["path"]] == entry["source_sha256"]
            expected[entry["path"]] = entry["sha256"]
        for entry in transition["added"]:
            assert entry["path"] not in expected
            expected[entry["path"]] = entry["sha256"]
    actual = {
        path.relative_to(ROOT).as_posix()
        for directory in ("backend", "dialect", "contracts")
        for path in (ROOT / directory).rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml"}
    } | {path.name for path in ROOT.glob("*.py")}
    assert set(expected) == actual
    for name, digest in expected.items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT)
        assert not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_provider_identity_remains_distinct_from_rvv():
    provider = read_provider(ROOT)
    assert provider is not None
    assert provider.target == "saturn"
    assert provider.role == ProviderRole.SUPPORT
    assert read_provider(ROOT.parent / "merlin-support").target == "rvv"
