"""Package checks: identity, layering, contract schema, plugin references, boards, content hashes.

No hardware is executed and nothing here certifies a compiler candidate.
"""

import hashlib
import json
from pathlib import Path

import yaml

from merlin.common.schemas import validate as validate_schema
from merlin.runtime import boards
from merlin.targetgen import provider_compose
from merlin.targetgen.plugins import validate
from merlin.targetgen.providers import ProviderRole, read_provider

ROOT = Path(__file__).resolve().parents[1]


def test_identity_layering_and_plugin_references():
    provider = read_provider(ROOT)
    assert provider.role == ProviderRole.SUPPORT
    assert provider.extends is not None and provider.extends.target == "rvv"
    contract = yaml.safe_load(provider.contract_path.read_text())
    assert contract["name"] == provider.target
    assert validate_schema(contract, "target_contract") == []
    assert validate(contract.get("plugin"), root=ROOT) == []
    assert (ROOT / contract["matrix_contract"]).is_file()


def test_effective_contract_is_fresh_against_the_pinned_base():
    """The committed contract is exactly base @ pinned commit + overlay. Needs the rvv base selected
    (MERLIN_TARGET_PATH) or vendored in the Merlin checkout running this test."""
    fresh, comp = provider_compose.materialize(ROOT, check=True)
    assert fresh, f"stale: run `merlin-target-compose {ROOT}` (base {comp.base.root} @ {comp.base_revision})"
    assert comp.base_revision == read_provider(ROOT).extends.commit


def test_boards_load_and_describe_the_silicon():
    found = boards.load_boards(str(ROOT))
    chip = found["chipyard_kodiak"]
    assert (chip.harts, chip.vector_harts, chip.vlen) == (3, 2, 512)
    assert chip.dram_bytes == 512 << 20
    assert not chip.fpu_sharing and chip.zephyr_vector_ext


def test_files_match_recorded_content_hashes():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert provenance["files"]
    assert not provenance["qualification"]["candidate_certification"]
    assert not provenance["qualification"]["hardware_executed"]
    for record in provenance["files"]:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"], record["path"]
