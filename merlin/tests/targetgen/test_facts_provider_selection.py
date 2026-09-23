"""Selected support owns structural facts; same-name artifacts are not interchangeable."""

import json

import pytest

from merlin.targetgen import capability_discovery as discovery
from merlin.targetgen.rtl import facts


def _pin(root, marker):
    path = root / "contracts/rtl_facts/facts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"facts": {"target": "synthetic", "marker": marker}}))
    return path


@pytest.fixture
def selected(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(repo))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT", raising=False)
    native = repo / "merlin/targets/synthetic"
    _pin(native, "native")
    provider = tmp_path / "provider"
    pin = _pin(provider, "selected")
    (provider / "contracts/target_contract.yaml").write_text("name: synthetic\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    cache = facts.rtl_facts_path("synthetic")
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"facts": {"target": "synthetic", "marker": "stale-cache"}}))
    monkeypatch.setattr(facts, "_written_by_another_family", lambda *args: False)
    monkeypatch.setattr(facts, "_dump_facts_for_kind", lambda *args: pytest.fail("unexpected extraction"))
    facts.clear_resolution_cache()
    yield provider, pin, native
    facts.clear_resolution_cache()


def test_selected_provider_owns_target_base(selected):
    provider, _, _ = selected
    assert facts.target_base("synthetic") == provider


def test_selected_pin_beats_same_name_cache_and_native_pin(selected):
    _, pin, _ = selected
    assert facts.ensure_facts("synthetic") == pin
    assert list(facts._committed_facts_candidates("synthetic")) == [pin]


def test_discovery_reads_same_selected_facts_without_extraction(selected):
    _, pin, _ = selected
    doc, source = discovery._facts_if_present("synthetic")
    assert doc["facts"]["marker"] == "selected"
    assert source == str(pin)


def test_discovery_does_not_require_reconciling_an_unselected_provider(selected):
    notes = []
    discovery._from_facts("synthetic", [], notes)
    assert not any("Reconcile" in note for note in notes)


def test_missing_selected_facts_never_mix_legacy_evidence(selected):
    _, pin, _ = selected
    pin.unlink()
    with pytest.raises(FileNotFoundError):
        facts.ensure_facts("synthetic")
    assert discovery._facts_if_present("synthetic") == (None, "")


@pytest.mark.parametrize("bad", ["malformed", "wrong-target", "nonmapping"])
def test_bad_selected_facts_refuse_instead_of_cache_fallback(selected, bad):
    _, pin, _ = selected
    pin.write_text({"malformed": "{", "wrong-target": '{"facts":{"target":"other"}}', "nonmapping": "[]"}[bad])
    with pytest.raises(ValueError):
        facts.ensure_facts("synthetic")


def test_switching_provider_invalidates_same_target_memo(selected, tmp_path, monkeypatch):
    _, first, _ = selected
    assert facts.ensure_facts("synthetic") == first
    second = tmp_path / "second"
    pin = _pin(second, "second")
    (second / "contracts/target_contract.yaml").write_text("name: synthetic\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(second))
    assert facts.ensure_facts("synthetic") == pin


def test_discovery_enumerates_external_only_target(selected, monkeypatch):
    provider, _, _ = selected
    monkeypatch.setattr(discovery, "_descriptor_targets", lambda: set())
    monkeypatch.setattr(discovery, "targets_dir", lambda: provider / "absent")
    assert "synthetic" in discovery.targets_with_facts()


def test_selected_pin_mutation_is_not_hidden_by_resolution_memo(selected):
    _, pin, _ = selected
    assert facts.ensure_facts("synthetic") == pin
    pin.write_text('{"facts":{"target":"other"}}')
    with pytest.raises(ValueError, match="differs from selected"):
        facts.ensure_facts("synthetic")
    with pytest.raises(ValueError, match="differs from selected"):
        discovery._facts_if_present("synthetic")


@pytest.mark.parametrize("escape", [False, True])
def test_selected_pin_rejects_dangling_or_escaping_symlink(selected, tmp_path, escape):
    _, pin, _ = selected
    pin.unlink()
    outside = tmp_path / "outside.json"
    if escape:
        outside.write_text('{"facts":{"target":"synthetic"}}')
    pin.symlink_to(outside)
    with pytest.raises(ValueError):
        facts.ensure_facts("synthetic")


@pytest.mark.parametrize("environment", [False, True])
def test_explicit_facts_keep_actual_source_without_provider_or_alias_mixing(
    selected, tmp_path, monkeypatch, environment
):
    override = tmp_path / "explicit.json"
    document = {"facts": {"target": "declared-other-design", "marker": "explicit"}}
    override.write_text(json.dumps(document))
    monkeypatch.setattr(facts, "facts_alias", lambda target: pytest.fail("unrelated alias resolution"))
    if environment:
        monkeypatch.setenv("MERLIN_RTL_FACTS", str(override))
        assert facts.load_facts("synthetic") == document
    else:
        assert facts.load_facts("synthetic", explicit=override) == document


def test_custom_provider_contract_path_is_resolved(selected):
    provider, _, _ = selected
    custom = provider / "contracts/custom.yaml"
    (provider / "contracts/target_contract.yaml").rename(custom)
    (provider / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: custom\ntarget: synthetic\nrole: support\ncontract: contracts/custom.yaml\n"
    )
    assert facts.target_contract_path("synthetic") == custom


def test_selected_provider_does_not_inherit_legacy_alias_cache(selected, monkeypatch):
    monkeypatch.setitem(facts._FACTS_ALIAS_CACHE, "synthetic", ("other", "legacy residual"))
    assert facts.facts_alias("synthetic") == "synthetic"
    assert "served_for" not in facts.load_facts("synthetic")


def test_registry_directory_alias_preserves_actual_selected_design(selected, monkeypatch):
    from merlin.targetgen import target_registry

    monkeypatch.setattr(target_registry, "declared_target_for", lambda target: "synthetic")
    document = facts.load_facts("directory_alias")
    assert document["facts"]["marker"] == "selected"
    assert document["served_for"]["target"] == "directory_alias"
    assert document["served_for"]["artifact_of"] == "synthetic"
