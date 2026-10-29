"""None is explicit absence for clean profiles, not inherited expert exposure."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def source_owner():
    spec=importlib.util.spec_from_file_location("no_expert_source_owner",
                                              Path(__file__).with_name("prepare_source_release.py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("level", (1, 2, 3))
def test_explicit_no_expert_profiles_do_not_inherit_old_scaffold_bank(source_owner, level):
    scaffold={"public_reference_sources": {"path": "/prior/expert/reference-manifest.json"}}
    assert source_owner.selected_public_references(scaffold, None, f"xs-el-no-expert-v1/el{level}") is None
    # The same declaration also works when refreshing an already clean release.
    assert source_owner.selected_public_references({}, None, f"xs-el-no-expert-v1/el{level}") is None
    with pytest.raises(ValueError, match="cannot select public references"):
        source_owner.selected_public_references(scaffold, "/injected/bank.json", f"xs-el-no-expert-v1/el{level}")


def test_historical_inheritance_and_explicit_reviewed_bank_are_preserved(source_owner):
    scaffold={"public_reference_sources": {"path": "/prior/expert/reference-manifest.json"}}
    assert source_owner.selected_public_references(scaffold, None, "xs-el-v1/el1") == Path(scaffold["public_reference_sources"]["path"])
    assert source_owner.selected_public_references(scaffold, "/reviewed/new.json") == Path("/reviewed/new.json")
