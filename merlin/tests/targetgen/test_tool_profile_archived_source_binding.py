"""Archived source-binding grants survive unrelated scheduling API growth."""

import copy

import pytest

from merlin.targetgen import tool_profile_shared as S


def archived():
    modules = (*S._V6_CONFIGURATION_MODULES, *S.SOURCE_BINDING_MODULES)
    public = (*S._V6_CONFIGURATION_PUBLIC, *S.SOURCE_BINDING_MODULES)
    return {"schema": "merlin.shared_scheduling_library.v2",
            "source_binding_protocol": S.SOURCE_BINDING_PROTOCOL,
            "members": [{"module": name} for name in (*modules, None, None)],
            "public_modules": list(public)}


def test_archived_release_preserves_exact_grants_as_live_api_grows(monkeypatch):
    document = archived()
    before = copy.deepcopy(document)
    expected = S._release_inventory(document)
    monkeypatch.setattr(S, "LIBRARY_MODULES", (*S.LIBRARY_MODULES, "merlin.compile.scheduling.future"))
    monkeypatch.setattr(S, "PUBLIC_MODULES", (*S.PUBLIC_MODULES, "merlin.compile.scheduling.future"))
    assert S._release_inventory(document) == expected
    assert document == before
    assert "merlin.compile.scheduling.panel_transfers" not in expected[1]


@pytest.mark.parametrize("damage", ["missing", "extra", "order", "public", "protocol"])
def test_archived_release_rejects_changed_grants(damage):
    document = archived()
    if damage == "missing":
        del document["members"][8]
    elif damage == "extra":
        document["members"].insert(-2, {"module": "merlin.compile.scheduling.panel_transfers"})
    elif damage == "order":
        document["members"][8:10] = reversed(document["members"][8:10])
    elif damage == "public":
        document["public_modules"].append("merlin.compile.scheduling.panel_transfers")
    else:
        document["source_binding_protocol"] = "unreviewed"
    with pytest.raises(ValueError, match="inventory changed"):
        S._release_inventory(document)
