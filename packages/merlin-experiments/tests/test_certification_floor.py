"""A selected certification floor refuses capped members without shrinking a cohort."""

import pytest
from merlin_experiments.corpus import preparation
from merlin_experiments.phase0.certification_floor import require_direct_tier, selected_floor
from merlin_experiments.spec import SpecError

from merlin.targetgen import applications, conformance, corpus_synth


def test_selected_floor_must_belong_to_explicit_recipe_tiers():
    assert selected_floor({}, ["L2", "L3"]) is None
    assert selected_floor({"certification_floor": "L3"}, ["L2", "L3"]) == "L3"
    for invalid in ("L4", "l3", True, 3):
        with pytest.raises(ValueError, match="selected recipe"):
            selected_floor({"certification_floor": invalid}, ["L2", "L3"])


def test_derived_profile_refuses_l2_extension_without_dropping_it():
    capsules = [
        {"name": "certified"},
        {"name": "large", "max_oracle_tier": "L2", "extends": "certified"},
    ]
    with pytest.raises(ValueError, match="1 of 2.*large"):
        require_direct_tier(capsules, floor="L3", declared_tiers=["L2", "L3"])
    assert len(capsules) == 2 and capsules[1]["max_oracle_tier"] == "L2"
    assert require_direct_tier(capsules, floor=None, declared_tiers=["L2", "L3"])["checked"] == 2


def test_materialized_corpus_requires_own_direct_tiers_and_rechecks_ceiling():
    direct = {"name": "member", "required_oracle_tiers": ["L2", "L3"]}
    assert require_direct_tier([direct], floor="L3")["checked"] == 1
    for candidate in (
        {"name": "member"},
        {**direct, "required_oracle_tiers": ["L2"]},
        {**direct, "max_oracle_tier": "L2"},
        {**direct, "max_oracle_tier": "unknown"},
    ):
        with pytest.raises(ValueError, match="member"):
            require_direct_tier([candidate], floor="L3")


@pytest.mark.parametrize("floor", ["L3", "L4"])
def test_explicit_floor_keeps_functional_anchor_and_full_shape_without_cost_claim(floor):
    evidence = applications.ClassEvidence(
        region_class=applications.RegionClass("contraction", "i8", "partial", "fits_double", 2, "tall_skinny"),
        m=256,
        k=72,
        n=8,
        batch=1,
        multiplicity=9,
        work=256 * 72 * 8 * 9,
        work_complete=True,
        source="selected_capture",
    )
    sized, refused = applications.size_class(
        evidence, target="not_a_target", budget_s=1, tile=16, certification_floor=floor
    )
    assert refused is None
    assert [(row.m, row.k, row.n, row.tier) for row in sized] == [(16, 72, 8, floor), (256, 72, 8, floor)]
    assert {row.basis["cost_status"] for row in sized} == {"unknown_not_priced"}
    assert all("cost_fit" not in row.basis and "fitted_seconds" not in row.basis for row in sized)
    assert sized[-1].basis["source"] == "selected_capture"


def test_explicit_floor_refuses_unknown_tile_instead_of_inventing_unit_anchor():
    evidence = applications.ClassEvidence(
        region_class=applications.RegionClass("contraction", "i8", "partial", "fits_double", 2, "tall_skinny"),
        m=256,
        k=72,
        n=8,
        batch=1,
        multiplicity=9,
        work=256 * 72 * 8 * 9,
        work_complete=True,
        source="selected_capture",
    )
    for tile in (None, 0, -1):
        sized, refused = applications.size_class(
            evidence, target="not_a_target", budget_s=300, tile=tile, certification_floor="L3"
        )
        assert sized == []
        assert "tile" in refused


def test_explicit_floor_never_reads_ambient_application_cost_history(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("ambient cost history was consulted")

    from merlin.targetgen import cert_cost, corpus_spec, target_registry

    monkeypatch.setattr(conformance, "_binding_engine_fit", unexpected)
    monkeypatch.setattr(conformance, "admitted", lambda _target: {"contraction": ("i8",)})
    monkeypatch.setattr(cert_cost, "fit_for", unexpected)
    monkeypatch.setattr(target_registry, "load_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(corpus_spec, "_tile_dim", lambda *_args, **_kwargs: 16)
    monkeypatch.setattr(
        applications,
        "classify_captures",
        lambda *_args, **_kwargs: {
            "classes": [
                {
                    "class": "contraction/i8/partial/fits_double/rank2/tall_skinny",
                    "family": "contraction",
                    "dtype": "i8",
                    "alignment": "partial",
                    "regime": "fits_double",
                    "rank": 2,
                    "geometry": "tall_skinny",
                    "M": 256,
                    "K": 72,
                    "N": 8,
                    "batch": 1,
                    "multiplicity": 9,
                    "work": 256 * 72 * 8 * 9,
                    "work_complete": True,
                    "source": "selected_capture",
                }
            ],
            "n_classes": 1,
        },
    )
    got = conformance._application_axis("gemmini", captures={"selected_capture": "unused"}, certification_floor="L3")
    assert got["cost_model"] is None
    assert [row["tier"] for row in got["required"]] == ["L3", "L3"]


def test_application_axis_reports_missing_selected_tile_as_refusal(monkeypatch):
    from merlin.targetgen import corpus_spec

    def missing_tile(*_args, **_kwargs):
        raise ValueError("selected fixed mesh has no derived tile geometry")

    monkeypatch.setattr(corpus_spec, "_tile_dim", missing_tile)
    monkeypatch.setattr(conformance, "admitted", lambda _target: {"contraction": ("i8",)})
    monkeypatch.setattr(
        applications,
        "classify_captures",
        lambda *_args, **_kwargs: {
            "classes": [
                {
                    "class": "contraction/i8/partial/fits_double/rank2/tall_skinny",
                    "family": "contraction",
                    "dtype": "i8",
                    "alignment": "partial",
                    "regime": "fits_double",
                    "rank": 2,
                    "geometry": "tall_skinny",
                    "M": 256,
                    "K": 72,
                    "N": 8,
                    "batch": 1,
                    "multiplicity": 9,
                    "work": 256 * 72 * 8 * 9,
                    "work_complete": True,
                    "source": "selected_capture",
                }
            ],
            "n_classes": 1,
        },
    )
    with pytest.raises(ValueError, match="positive tile geometry"):
        conformance._application_axis("gemmini", captures={"selected_capture": "unused"}, certification_floor="L3")


def test_direct_core_requirement_carries_floor_metadata_for_l4_synthesis():
    requirement = conformance.derive_spec("gemmini", {}, oracle_tiers=[], certification_floor="L4")
    assert requirement["certification_floor"]["tier"] == "L4"
    assert (
        requirement["certification_floor"]["scope"]
        == "direct per-capsule certification; no execution or affordability claim"
    )
    with pytest.raises(corpus_synth.SynthesisError, match="selected oracle-tier declarations"):
        corpus_synth.synthesize(requirement, workload_spec={"certification_floor": "L4"})


def test_full_requirement_floor_does_not_consult_ambient_depth_or_affordability(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("ambient certification history was consulted")

    monkeypatch.setattr(conformance, "_certified_depth", unexpected)
    monkeypatch.setattr(conformance, "_cert_affordability", unexpected)
    requirement = conformance.derive_spec("gemmini", {}, oracle_tiers=[], certification_floor="L3")
    assert requirement["cert_affordability"]["status"] == "unknown_not_priced"
    assert requirement["cert_affordability"]["max_elements"] is None


def test_floor_disables_size_cap_but_retains_legacy_cost_cap():
    member = {"name": "large", "op": "matmul", "M": 256, "K": 72, "N": 8}
    floor = {"cert_affordability": {"status": "unknown_not_priced", "max_elements": None}}
    assert corpus_synth.cap_to_affordable(member, floor) is None
    assert "max_oracle_tier" not in member
    legacy = dict(member)
    priced = {
        "cert_affordability": {"max_elements": 100, "budget_s": 300},
        "boundaries": {"tile_edge": 16},
        "oracle_tiers_declared": ["L2", "L3"],
    }
    assert corpus_synth.cap_to_affordable(legacy, priced)
    assert legacy["max_oracle_tier"] == "L2"


def test_release_preflight_refuses_existing_l2_member_before_materialization(monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen import capsule_runner

    public = {"name": "old_extension", "required_oracle_tiers": ["L2", "L3"], "max_oracle_tier": "L2"}
    hidden = {"name": "hidden", "required_oracle_tiers": ["L2", "L3"]}
    monkeypatch.setattr(
        capsule_runner,
        "discover_capsules",
        lambda _roots, *, labels: [public] if "public" in labels else [hidden],
    )
    te = SimpleNamespace(
        workload_spec={"certification_floor": "L3"},
        graded_roots=lambda: [],
        hidden_roots=lambda: [],
    )
    with pytest.raises(SpecError, match="old_extension"):
        preparation._admission(te, coverage_output=None)
