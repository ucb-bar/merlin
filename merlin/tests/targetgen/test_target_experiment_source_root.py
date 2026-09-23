"""Explicit descriptor ownership works outside a checkout without ambient discovery."""

from types import SimpleNamespace

import pytest

from merlin.targetgen import target_experiment as TE


def _descriptor(root):
    descriptor = root / "provider/descriptor.yaml"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text(
        "target: fixture_device\ncapsule_corpus: corpus/public\n"
        "resources_root: authored\ntask_root: prompts\ncontracts_root: contracts\n"
        "hardware_spec:\n  target_contract: provider/contract.yaml\n"
    )
    (descriptor.parent / "contract.yaml").write_text("target: fixture_device\n")
    for category in ("public", "layers", "hidden", "_perf"):
        member = root / "corpus" / category / "member"
        member.mkdir(parents=True)
        (member / "capsule.yaml").write_text("name: member\n")
    return descriptor


def test_explicit_source_owner_survives_all_descriptor_path_methods(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path)
    monkeypatch.setattr(TE, "repo_root", lambda: pytest.fail("ambient repository lookup"))
    target = TE.load_target_experiment(descriptor.relative_to(tmp_path), source_root=tmp_path)
    assert target.path == descriptor
    assert target.declared_contract_path() == descriptor.parent / "contract.yaml"
    assert target.corpus_rel() == "corpus/public/"
    assert target.corpus_siblings() == ["corpus/layers/"]
    assert target.graded_roots() == [tmp_path / "corpus/public", tmp_path / "corpus/layers"]
    assert target.hidden_roots() == [tmp_path / "corpus/hidden"]
    assert target.perf_roots() == [tmp_path / "corpus/_perf"]
    assert target.resource_path("task/prompt.md") == tmp_path / "prompts/prompt.md"
    assert target.resource_path("contracts/abi.h") == tmp_path / "contracts/abi.h"
    assert target.experiment_resource("other") == "authored/other"
    seen = []
    lane = SimpleNamespace(resolve=lambda **kw: seen.append(kw) or (tmp_path, {}))
    object.__setattr__(target, "host_lanes", SimpleNamespace(for_dtype=lambda _: lane))
    target.resolve_host_lane()
    assert seen == [{"root": tmp_path, "descriptor": descriptor}]


def test_legacy_loader_keeps_ambient_default(tmp_path, monkeypatch):
    descriptor = _descriptor(tmp_path)
    monkeypatch.setattr(TE, "repo_root", lambda: tmp_path)
    target = TE.load_target_experiment(descriptor)
    assert target.source_root is None
    assert target.graded_roots() == [tmp_path / "corpus/public", tmp_path / "corpus/layers"]


def test_holdout_binding_supplies_explicit_descriptor_owner(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import declarations, profiles
    from merlin_experiments.phase2 import holdout_corpus as HC

    from merlin.targetgen import corpus_spec

    descriptor = _descriptor(tmp_path)
    monkeypatch.setattr(TE, "repo_root", lambda: pytest.fail("ambient repository lookup"))
    declaration = SimpleNamespace(descriptor=descriptor, profile="fixture", profile_inputs=lambda: {})
    monkeypatch.setattr(declarations, "for_target", lambda *a, **k: declaration)
    monkeypatch.setattr(profiles, "load_profile", lambda *a, **k: {"datapath": {}})

    def derive(target, datapath):
        assert target.source_root == tmp_path
        assert target.declared_contract_path() == descriptor.parent / "contract.yaml"
        return SimpleNamespace(tile_dim=4, operand_dtype="int8", accum_dtype="int32")

    monkeypatch.setattr(corpus_spec, "derive_binding", derive)
    context = HC.HoldoutSourceContext(
        tmp_path, tmp_path / "catalog", tmp_path / "core", tmp_path / "optional", tmp_path / "namespace"
    )
    result = HC._binding(
        "fixture_device",
        {"bounds": {"mesh": {"rows": 4}, "operand_dtype": "int8", "accumulator_dtype": "int32"}},
        context=context,
    )
    assert result.tile_dim == 4
