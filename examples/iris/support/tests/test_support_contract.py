"""Package checks: identity, layering, contract schema, boards, content hashes. No hardware executed."""

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


def test_identity_layering_and_contract():
    provider = read_provider(ROOT)
    assert provider.role == ProviderRole.SUPPORT
    assert provider.extends is not None and provider.extends.target == "rvv"
    contract = yaml.safe_load(provider.contract_path.read_text())
    assert contract["name"] == provider.target
    assert validate_schema(contract, "target_contract") == []
    assert validate(contract.get("plugin"), root=ROOT) == []
    assert contract["matrix_contract"] == "contracts/matrix_units.yaml"
    assert (ROOT / contract["matrix_contract"]).is_file()
    assert contract["plugin"]["matrix_lowering"] == "matrix_lowering.py"
    assert "matrix_unit" in contract["features"]
    assert any(u["name"] == "vme" for u in contract["compute_units"])


def test_effective_contract_is_fresh_against_the_pinned_base():
    fresh, comp = provider_compose.materialize(ROOT, check=True)
    assert fresh, f"stale: run `merlin-target-compose {ROOT}` (base {comp.base.root} @ {comp.base_revision})"
    assert comp.base_revision == read_provider(ROOT).extends.commit


def test_boards_describe_irisconfig():
    rtl = boards.load_boards(str(ROOT))["iris_rtl"]
    assert (rtl.harts, rtl.vector_harts, rtl.vlen) == (3, 3, 256)
    assert (rtl.dram_base, rtl.dram_bytes) == (0x80000000, 4 << 30)
    # The Rocket DMA hart has a vector unit and no matrix unit.
    assert rtl.matrix_harts == (0, 1)
    rtl.check_matrix_image(2)
    try:
        rtl.check_matrix_image(3)
    except boards.BoardRegistryError:
        pass
    else:
        raise AssertionError("a 3-hart matrix image would reach the hart without the unit")


def test_files_match_recorded_content_hashes():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert provenance["files"]
    assert not provenance["qualification"]["hardware_executed"]
    for record in provenance["files"]:
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record["sha256"], record["path"]
