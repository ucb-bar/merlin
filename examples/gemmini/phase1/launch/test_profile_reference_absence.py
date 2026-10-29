"""A declared no-reference treatment cannot accidentally stage examples."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def campaign():
    spec = importlib.util.spec_from_file_location(
        "profile_reference_campaign", Path(__file__).with_name("fresh_campaign.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(*, guidance="off", reference_map=None):
    return SimpleNamespace(
        reference_guidance=guidance,
        section=lambda _name: {"reference_map": reference_map},
        path=Path,
    )


def test_no_reference_profile_passes_none_without_staging(campaign, tmp_path, monkeypatch):
    def unexpected(*_args):
        raise AssertionError("reference staging must not run")

    monkeypatch.setattr(campaign, "stage_launch_references", unexpected)
    destination = tmp_path / "references"
    result = campaign.profile_reference_manifest(
        context(), {"public_references": None}, destination,
        SimpleNamespace(raw_kernel_references=False),
    )
    assert result is None
    assert not destination.exists()


@pytest.mark.parametrize("profile", [None, SimpleNamespace(), SimpleNamespace(raw_kernel_references=True)])
def test_reference_absence_requires_the_explicit_capability(campaign, tmp_path, profile):
    with pytest.raises(campaign.SpecError, match="requires its reviewed public references"):
        campaign.profile_reference_manifest(
            context(), {"public_references": None}, tmp_path / "references", profile,
        )


@pytest.mark.parametrize("violation", ["bank", "map", "guidance", "destination", "symlink"])
def test_no_reference_profile_refuses_residual_exposure(campaign, tmp_path, violation):
    destination = tmp_path / "references"
    selected = str(tmp_path / "bank") if violation == "bank" else None
    reference_map = {"path": "map.json"} if violation == "map" else None
    if violation == "destination":
        destination.mkdir()
    elif violation == "symlink":
        destination.symlink_to(tmp_path / "absent-bank", target_is_directory=True)
    with pytest.raises(campaign.SpecError):
        campaign.profile_reference_manifest(
            context(guidance="on" if violation == "guidance" else "off", reference_map=reference_map),
            {"public_references": selected}, destination,
            SimpleNamespace(raw_kernel_references=False),
        )


def test_reference_profiles_retain_the_reviewed_staging_path(campaign, tmp_path, monkeypatch):
    calls = []
    manifest = tmp_path / "bank/reference-manifest.json"
    destination = tmp_path / "references"

    def stage(path, target, forbidden):
        calls.append((path, target, forbidden))
        return manifest

    monkeypatch.setattr(campaign, "stage_launch_references", stage)
    result = campaign.profile_reference_manifest(
        context(), {"public_references": str(manifest.parent), "forbidden_reference_prefixes": ["private/"]},
        destination, SimpleNamespace(raw_kernel_references=True),
    )
    assert result == manifest
    assert calls == [(manifest, destination, ["private/"])]


def test_no_reference_plan_does_not_require_a_manifest(campaign):
    profile = SimpleNamespace(raw_kernel_references=False)
    assert campaign.planned_reference_manifest(context(), {}, profile) is None


@pytest.mark.parametrize("violation", ["manifest", "bank", "map", "guidance"])
def test_no_reference_plan_refuses_residual_exposure(campaign, violation):
    plan = {"public_reference_manifest": "reference-manifest.json"} if violation == "manifest" else {}
    if violation == "map":
        plan["reference_map"] = {"path": "map.json"}
    ctx = SimpleNamespace(
        reference_guidance="on" if violation == "guidance" else "off",
        section=lambda _name: {"public_references": "bank" if violation == "bank" else None},
        path=Path,
    )
    with pytest.raises(campaign.SpecError, match="no-reference plan"):
        campaign.planned_reference_manifest(ctx, plan, SimpleNamespace(raw_kernel_references=False))


def test_reference_plan_still_requires_positive_review(campaign, tmp_path):
    import json

    manifest = tmp_path / "reference-manifest.json"
    manifest.write_text(json.dumps({"members": [{"path": "private/kernel.c"}]}))
    ctx = SimpleNamespace(
        reference_guidance="off",
        section=lambda _name: {"forbidden_reference_prefixes": ["private/"]},
        path=Path,
    )
    profile = SimpleNamespace(raw_kernel_references=True)
    with pytest.raises(campaign.SpecError, match="requires its reviewed"):
        campaign.planned_reference_manifest(ctx, {}, profile)
    with pytest.raises(campaign.SpecError, match="forbidden members"):
        campaign.planned_reference_manifest(ctx, {"public_reference_manifest": str(manifest)}, profile)


@pytest.mark.parametrize("transforms", [False, True])
def test_fresh10_uses_the_projected_sdk_without_requiring_withheld_passes(campaign, transforms):
    profile = SimpleNamespace(
        name="fresh10-tooling-v1/no-reference-kernels" if transforms else "fresh10-tooling-v1/infra",
        seeded=False, level=4, allows=lambda capability: transforms,
    )
    argv = ["--compiler-access", "full_compiler", "--compiler-library-root", "selected-sdk",
            "--compiler-library-contract", "contract.json", "--compiler-library-contract-sha256", "hash",
            "--full-compiler-root", "selected-source", "--required-shared-pass", "optimized-pass",
            "--private-full-model-spec", "models.json"]
    expected = argv if transforms else [value for value in argv if value not in {
        "--required-shared-pass", "optimized-pass"}]
    assert campaign.profile_campaign_argv(profile, argv) == expected
    assert "--required-shared-pass" in argv  # caller's original spec remains intact


@pytest.mark.parametrize("transforms", [False, True])
def test_fresh10_admission_requires_the_selected_sdk(campaign, transforms, tmp_path, monkeypatch):
    from merlin_experiments.phase1 import author_profiles

    profile = SimpleNamespace(name="fresh10-tooling-v1/infra", seeded=False, level=4,
                              allows=lambda capability: transforms)
    monkeypatch.setattr(author_profiles, "resolve", lambda name: profile)
    plan = {"author_profile": profile.name, "required_flags": ["--compiler-library-root"],
            "input_flags": ["--compiler-library-root"]}
    with pytest.raises(campaign.SpecError, match="requires --compiler-library-root"):
        campaign._check_campaign(None, [], plan)
    with pytest.raises(campaign.SpecError, match="missing input for --compiler-library-root"):
        campaign._check_campaign(None, ["--compiler-library-root", str(tmp_path / "missing-sdk")], plan)
    campaign._check_campaign(None, ["--compiler-library-root", str(tmp_path)], plan)


def test_fresh10_admission_requires_pass_reuse_only_when_library_is_supplied(campaign, monkeypatch):
    from merlin_experiments.phase1 import author_profiles

    profile = SimpleNamespace(name="fresh10-tooling-v1/no-reference-kernels", seeded=False,
                              allows=lambda capability: True)
    monkeypatch.setattr(author_profiles, "resolve", lambda name: profile)
    plan = {"author_profile": profile.name, "required_flags": ["--required-shared-pass"]}
    with pytest.raises(campaign.SpecError, match="requires --required-shared-pass"):
        campaign._check_campaign(None, [], plan)
    profile.allows = lambda capability: False
    campaign._check_campaign(None, [], plan)
