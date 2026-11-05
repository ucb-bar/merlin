"""A CCA bank arm stages exactly its pinned bank and declares its broker dependency."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def campaign():
    spec = importlib.util.spec_from_file_location(
        "cca_bank_arm_campaign", Path(__file__).with_name("fresh_campaign.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bank(tmp_path):
    root = tmp_path / "conditions/cca-x"
    root.mkdir(parents=True)
    (root / "bank.json").write_text(json.dumps({"units": []}))
    return root, hashlib.sha256((root / "bank.json").read_bytes()).hexdigest()


def context():
    return SimpleNamespace(path=Path)


def test_bank_arm_requires_its_pinned_bank(campaign, tmp_path):
    root, pin = bank(tmp_path)
    arm = SimpleNamespace(bank_sha256=pin)
    assert campaign.cca_bank_selection(context(), {"cca_bank": {"path": str(root), "sha256": pin}}, arm) == (
        root.resolve(), pin)
    with pytest.raises(campaign.SpecError, match="requires inputs.cca_bank"):
        campaign.cca_bank_selection(context(), {}, arm)
    with pytest.raises(campaign.SpecError, match="must match"):
        campaign.cca_bank_selection(context(), {"cca_bank": {"path": str(root), "sha256": "0" * 64}}, arm)
    with pytest.raises(campaign.SpecError, match="must match"):
        campaign.cca_bank_selection(context(), {"cca_bank": {"path": str(root), "sha256": pin}},
                                    SimpleNamespace(bank_sha256=None))
    (root / "bank.json").write_text("{}")
    with pytest.raises(campaign.SpecError, match="differs from its pin"):
        campaign.cca_bank_selection(context(), {"cca_bank": {"path": str(root), "sha256": pin}}, arm)


def test_other_profiles_select_no_bank(campaign):
    assert campaign.cca_bank_selection(context(), {}, None) is None
    assert campaign.cca_bank_selection(context(), {}, SimpleNamespace(bank_sha256=None)) is None


@pytest.mark.parametrize("change", [dict(reference_map={"path": "m", "sha256": "0" * 64}),
                                    dict(reference_experiments={"path": "p", "sha256": "0" * 64}),
                                    dict(guidance="on"), dict(environment={"MERLIN_REFERENCE_MAP": "{}"})])
def test_coverage_arms_take_no_reference_map_or_guidance(campaign, change):
    for coverage in (SimpleNamespace(coverage=True, bank_sha256=None), SimpleNamespace(coverage=True,
                                                                                       bank_sha256="a" * 64)):
        campaign.require_coverage_isolation({"reference_map": None}, {}, coverage, "off")
        plan = {key: value for key, value in change.items() if key not in ("guidance", "environment")}
        with pytest.raises(campaign.SpecError, match="no reference map"):
            campaign.require_coverage_isolation(plan, change.get("environment", {}), coverage,
                                                change.get("guidance", "off"))
    campaign.require_coverage_isolation({"reference_map": {"path": "m"}}, {}, SimpleNamespace(coverage=False), "on")


def test_bank_arm_plan_needs_merlin_mining_on_the_broker_pythonpath(campaign):
    arm = SimpleNamespace(bank_sha256="a" * 64)
    campaign.require_bank_dependencies({"PYTHONPATH": "/x/src:/x/packages/merlin-mining/src:/deps"}, arm)
    with pytest.raises(campaign.SpecError, match="merlin-mining"):
        campaign.require_bank_dependencies({"PYTHONPATH": "/x/src:/x/packages/merlin-experiments/src"}, arm)
    campaign.require_bank_dependencies({"PYTHONPATH": "/x/src"}, SimpleNamespace(bank_sha256=None))
    campaign.require_bank_dependencies({}, None)


def test_bank_arm_references_are_exactly_its_banks_raw_list(campaign, tmp_path):
    """Planning and input staging refuse a raw set that differs from the pinned bank's raw list."""
    from merlin_experiments.phase1 import fresh10_profiles as P

    own, other = sorted(P.COVERAGE_RAW_REFERENCE_SHA256)[:2], sorted(P.COVERAGE_RAW_REFERENCE_SHA256)[2]
    root = tmp_path / "conditions/cca-x"
    root.mkdir(parents=True)
    (root / "bank.json").write_text(json.dumps({"units": [], "raw": [
        {"id": str(index), "path": "originals/" + str(index), "sha256": sha} for index, sha in enumerate(own)]}))
    pin = hashlib.sha256((root / "bank.json").read_bytes()).hexdigest()
    inputs = {"cca_bank": {"path": str(root), "sha256": pin}}
    ctx = SimpleNamespace(path=Path, section=lambda name: inputs)
    arm = SimpleNamespace(bank_sha256=pin)

    def manifest(digests):
        path = tmp_path / "reference-manifest.json"
        path.write_text(json.dumps({"schema": "operator.allowed_reference_sources.v2", "review_id": "r", "origins": [
            {"name": "o", "files": [{"member": str(i), "destination": str(i), "sha256": sha}
                                    for i, sha in enumerate(digests)]}]}))
        return path

    campaign.require_bank_raw_references(ctx, manifest(own), arm)
    for digests in ([*own, other], own[:1], sorted(P.COVERAGE_RAW_REFERENCE_SHA256)):
        with pytest.raises(campaign.SpecError, match="exactly its bank's raw originals"):
            campaign.require_bank_raw_references(ctx, manifest(digests), arm)
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"members": []}))
    with pytest.raises(campaign.SpecError, match="v2 reviewed reference manifest"):
        campaign.require_bank_raw_references(ctx, legacy, arm)
    # The Raw arm and every other profile are unaffected.
    campaign.require_bank_raw_references(ctx, legacy, SimpleNamespace(bank_sha256=None))
    campaign.require_bank_raw_references(ctx, legacy, None)


def test_raw_originals_arms_take_no_map_and_exactly_the_pinned_originals(campaign, tmp_path):
    """The from-scratch raw-originals arms share the coverage isolation and the originals check."""
    from merlin_experiments.phase1 import fresh10_profiles as P
    from merlin_experiments.phase1.author_profiles import resolve

    for name in P.RAW_ORIGINALS_NAMES:
        arm = resolve(name)
        campaign.require_coverage_isolation({"reference_map": None}, {}, arm, "off")
        for plan, environment, guidance in (({"reference_map": {"path": "m"}}, {}, "off"),
                                            ({}, {"MERLIN_REFERENCE_MAP": "{}"}, "off"), ({}, {}, "on")):
            with pytest.raises(campaign.SpecError, match="no reference map"):
                campaign.require_coverage_isolation(plan, environment, arm, guidance)
    path = tmp_path / "reference-manifest.json"

    def manifest(digests):
        path.write_text(json.dumps({"schema": "operator.allowed_reference_sources.v2", "origins": [
            {"name": "o", "files": [{"member": str(i), "destination": str(i), "sha256": sha}
                                    for i, sha in enumerate(digests)]}]}))
        return path

    full = resolve(P.RAW_ORIGINALS_NAMES[0])
    campaign.require_bank_raw_references(None, manifest(sorted(P.COVERAGE_RAW_REFERENCE_SHA256)), full)
    for digests in (sorted(P.COVERAGE_RAW_REFERENCE_SHA256)[1:], [*P.COVERAGE_RAW_REFERENCE_SHA256, "0" * 64]):
        with pytest.raises(campaign.SpecError, match="exactly the pinned originals"):
            campaign.require_bank_raw_references(None, manifest(digests), full)
    bank = resolve(P.RAW_ORIGINALS_NAMES[1])
    assert bank.bank_sha256 == P.COVERAGE_BANKS["cca-coverage-v1/bank-cca-all"]
    with pytest.raises(campaign.SpecError, match="requires inputs.cca_bank"):
        campaign.cca_bank_selection(context(), {}, bank)
    with pytest.raises(campaign.SpecError, match="merlin-mining"):
        campaign.require_bank_dependencies({"PYTHONPATH": "/x/src"}, bank)
