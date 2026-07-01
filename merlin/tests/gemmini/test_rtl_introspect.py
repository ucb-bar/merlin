"""merlin-rtl-introspect: structure-only FIRRTL fact extraction + contract reproduction."""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

from pathlib import Path

import pytest
import yaml

from merlin.targetgen.rtl import introspect

REPO = repo_root()
CONTRACT = REPO / "merlin/targets/gemmini/contracts/target_contract.yaml"


def _contract() -> dict:
    return yaml.safe_load(CONTRACT.read_text())


def test_validate_logic_unit():
    """Validator logic, no chipyard needed: matching facts pass, mismatched facts are caught."""
    contract = _contract()
    good = {
        "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
        "memories": [{"name": "scratchpad", "bytes": 262144}],
        "datapaths": [{"name": "input", "dtype": "i8"},
                      {"name": "accumulator", "dtype": "i32"}],
    }
    assert introspect.validate_against_contract(good, contract) == []

    bad = {
        "arrays": [{"name": "mesh", "rows": 8, "cols": 8}],
        "memories": [{"name": "scratchpad", "bytes": 999}],
        "datapaths": [{"name": "input", "dtype": "i8"},
                      {"name": "accumulator", "dtype": "i32"}],
    }
    problems = introspect.validate_against_contract(bad, contract)
    assert any("mesh" in p for p in problems)
    assert any("scratchpad" in p for p in problems)


def test_packages_declare_honest_authoring_mode():
    """The hand-authored packages MUST be labeled hand_curated with target_generation NOT
    reproducible — so they are never implicitly presented as RTL-derived/generated targets."""
    for pkg in ("gemmini", "saturn_vec"):
        man = yaml.safe_load((REPO / "artifacts/targets" / pkg / "hand_v0" / "manifest.yaml").read_text())
        assert man["authoring"]["mode"] == "hand_curated"
        assert man["authoring"]["generated_from_rtl_facts"] is False
        assert man["reproducibility"]["target_generation_reproducible"] is False
        # certification IS reproducible even though generation is not
        assert man["reproducibility"]["certified_execution_reproducible"] is True


_ART = introspect.find_artifacts()
_HAVE_FIRRTL = _ART["fir"].is_file() and _ART["hierarchy"].is_file()


@pytest.mark.skipif(not _HAVE_FIRRTL, reason="Gemmini FIRRTL artifacts not present")
def test_dump_facts_is_reproducible_record(tmp_path):
    """dump_facts writes a recorded, attributable rtl_facts.yaml: generator version + source SHAs
    + facts. This is the recorded INPUT an RTL-derived target-gen experiment consumes."""
    rec = introspect.dump_facts(tmp_path / "rtl_facts.yaml")
    assert (tmp_path / "rtl_facts.yaml").is_file()
    assert rec["generator"]["version"] == introspect.GENERATOR_VERSION
    assert "grep" in rec["generator"]["method"].lower()        # honest: NOT a CIRCT pass yet
    assert set(rec["source_shas"]) == {"chipyard", "gemmini"}
    assert introspect.validate_against_contract(rec["facts"], _contract()) == []


@pytest.mark.skipif(not _HAVE_FIRRTL, reason="Gemmini FIRRTL artifacts not present")
def test_extracted_facts_reproduce_hand_curated_contract():
    """The structure-only facts extracted from real FIRRTL reproduce the hand-curated capacities."""
    facts = introspect.extract_facts(_ART["fir"], _ART["hierarchy"])
    mesh = next(a for a in facts["arrays"] if a["name"] == "mesh")
    assert mesh["rows"] == 16 and mesh["cols"] == 16 and mesh["square"]
    sp = next(m for m in facts["memories"] if m["name"] == "scratchpad")
    assert sp["bytes"] == 262144
    # structure-only: the extractor must NOT emit semantic role/kind guesses
    for m in facts["memories"]:
        assert "role" not in m and "role_guess" not in m
    for a in facts["arrays"]:
        assert "kind" not in a and "kind_guess" not in a
    # the headline: facts reproduce the hand-curated contract
    assert introspect.validate_against_contract(facts, _contract()) == []
