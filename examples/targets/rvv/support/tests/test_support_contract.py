"""Read-only qualification of migrated support resources, not hardware/candidate certification."""

import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.schemas import validate_or_raise
from merlin.common.schemas import validate as validate_schema
from merlin.targetgen.plugins import validate
from merlin.targetgen.providers import ProviderRole, read_provider

ROOT = Path(__file__).resolve().parents[1]


def test_support_identity_schema_diagnostic_parity_and_plugin_references():
    provider = read_provider(ROOT)
    assert provider.role == ProviderRole.SUPPORT
    contract = yaml.safe_load(provider.contract_path.read_text())
    qualification = json.loads((ROOT / "provenance.json").read_text())["qualification"]
    problems = validate_schema(contract, "target_contract")
    assert problems == qualification["schema_diagnostics"]
    assert qualification["schema_valid"] == (not problems)
    assert contract["name"] == provider.target
    assert validate(contract.get("plugin"), root=ROOT) == []
    plan = ROOT / "contracts/dialect_plan.yaml"
    if plan.is_file():
        validate_or_raise(yaml.safe_load(plan.read_text()), "dialect_plan")


def test_snapshots_match_recorded_content_hashes():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert provenance["files"]
    assert not provenance["qualification"]["candidate_certification"]
    assert not provenance["qualification"]["hardware_executed"]
    for record in provenance["files"]:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"]
