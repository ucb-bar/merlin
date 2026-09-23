"""Joining a tier record's design keys to the ledger's design string, without guessing.

WHY THIS EXISTS. Two halves of the repo named a design differently and nothing joined them:
`cost_plane` carries a five-key identity off a tier record and refuses to compare unless all five
match and are stated; `target_reference` carries one `design` string and refuses to score across
designs. Both refusals are correct and neither could reach the other, so a measured cycle count
could not be scored against a measured destination at all.

WHAT THE JOIN IS NOT. It is not the configuration name. Measured 2026-09-19: two FireSim bitstreams
in this repo elaborate the same `FireSimGemminiRocketConfig` onto the same U250 platform and are
different devices -- different sizes, different digests, different Gemmini revisions. A week of
whole-model cycle numbers was quoted under an hw-config the registry had never heard of, and
resolving them by configuration name would have compared them against references from the other
device. So the join is by the declared queue hw-config name, CONFIRMED against the digest of the
HWDB CONFIG ENTRY the submission pointed at.

The central test below is the one that mistake would have failed: right name, wrong bytes.

WHICH BYTES, AND WHY IT MATTERS. A record states the sha256 of the hwdb config entry; the registry
also declares ``digest``, the sha256 of the built IMAGE. Those are different files. Comparing one
against the other -- which this resolver did until 2026-09-21 -- can never agree, so the check
inverted: a record that supplied its digest was refused as "a different device under a reused name",
and a record that supplied nothing resolved by name alone. Supplying evidence was punished. The last
test in this file is that inversion, so it cannot come back.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from merlin.perf import design_identity as D

DIGEST_A = "0" * 64
DIGEST_B = "1" * 64
#: The built image's hash -- a DIFFERENT file from the hwdb entry, and never the one compared.
IMAGE_DIGEST = "2" * 64


def test_declared_hwdb_digest_survives_real_registry_loading(tmp_path):
    from merlin.common.provenance import load_artifacts

    registry = tmp_path / "pins.yaml"
    registry.write_text(
        "artifacts:\n  fixture:\n    path: image.tar\n    role: "
        + D.BITSTREAM_ROLE
        + "\n    hw_configs: [queue_name_a]\n    config: ConfigX\n"
        + f'    digest: "{IMAGE_DIGEST}"\n    hwdb_digest: "{DIGEST_A}"\n'
    )
    artifacts = load_artifacts(registry)
    assert artifacts["fixture"].digest == IMAGE_DIGEST
    assert artifacts["fixture"].hwdb_digest == DIGEST_A
    matched = D.design_string(
        {"hw_config": "queue_name_a", "hwdb_config_artifact_sha256": DIGEST_A}, artifacts=artifacts
    )
    assert matched.resolved and matched["confirmed_by"] == "hw_config and hwdb digest"
    wrong = D.design_string(
        {"hw_config": "queue_name_a", "hwdb_config_artifact_sha256": IMAGE_DIGEST}, artifacts=artifacts
    )
    assert not wrong.resolved


def test_registry_without_hwdb_digest_remains_explicitly_name_only(tmp_path):
    from merlin.common.provenance import load_artifacts

    registry = tmp_path / "pins.yaml"
    registry.write_text(
        "artifacts:\n  fixture:\n    path: image.tar\n    role: "
        + D.BITSTREAM_ROLE
        + "\n    hw_configs: [queue_name_a]\n    config: ConfigX\n"
    )
    result = D.design_string(
        {"hw_config": "queue_name_a", "hwdb_config_artifact_sha256": DIGEST_A},
        artifacts=load_artifacts(registry),
    )
    assert result.resolved and "declares no hwdb digest" in result["confirmed_by"]


def test_registry_rejects_non_string_hwdb_digest(tmp_path):
    from merlin.common.provenance import PinsError, load_artifacts

    registry = tmp_path / "pins.yaml"
    registry.write_text("artifacts:\n  fixture:\n    path: image.tar\n    hwdb_digest: 123\n")
    with pytest.raises(PinsError, match="hwdb_digest must be a quoted string"):
        load_artifacts(registry)


@dataclass
class _Artifact:
    """Only the fields the resolver reads, so the test does not depend on the rest."""

    role: str = D.BITSTREAM_ROLE
    hw_configs: tuple[str, ...] = ()
    #: sha256 of the built IMAGE. Present because the registry declares it -- and deliberately NOT
    #: what the resolver compares against, which is the distinction this file exists to hold.
    digest: str = ""
    #: sha256 of the HWDB CONFIG ENTRY, the quantity a record actually states.
    hwdb_digest: str = ""
    config: str = ""


def _registry(**kwargs):
    return kwargs


class TestItResolvesOnTwoAgreeingFacts:
    def test_a_matching_hw_config_and_digest_names_the_design(self):
        got = D.design_string(
            {"hw_config": "queue_name_a", "hwdb_config_artifact_sha256": DIGEST_A},
            artifacts=_registry(
                dev_a=_Artifact(
                    hw_configs=("queue_name_a",), digest=IMAGE_DIGEST, hwdb_digest=DIGEST_A, config="ConfigX"
                )
            ),
        )
        assert got.resolved and got["config"] == "ConfigX" and got["name"] == "dev_a"
        assert got["confirmed_by"] == "hw_config and hwdb digest"

    def test_a_record_with_no_digest_resolves_but_says_it_settled_for_less(self):
        """Weaker evidence is allowed and must be VISIBLE; it is never skipped silently."""
        got = D.design_string(
            {"hw_config": "queue_name_a"},
            artifacts=_registry(dev_a=_Artifact(hw_configs=("queue_name_a",), digest=DIGEST_A, config="ConfigX")),
        )
        assert got.resolved
        assert "hw_config only" in got["confirmed_by"]

    def test_stating_a_digest_the_registry_cannot_match_is_not_held_against_the_record(self):
        """THE INVERSION, and why this test is here.

        Until 2026-09-21 the resolver compared the record's HWDB digest against ``digest``, the hash
        of the built image -- two different files, which can never agree. The consequence was exactly
        backwards: a record that supplied its digest was refused as "a different device under a
        reused name", while a record that supplied nothing sailed through on the name alone. Evidence
        was punished.

        A registry that has not recorded the hwdb hash is a gap in the REGISTRY. The record is still
        as good as one that stayed silent, so it resolves -- and the result says the check could not
        be made rather than claiming a byte match it never performed.
        """
        got = D.design_string(
            {"hw_config": "queue_name_a", "hwdb_config_artifact_sha256": DIGEST_A},
            artifacts=_registry(dev_a=_Artifact(hw_configs=("queue_name_a",), digest=IMAGE_DIGEST, config="ConfigX")),
        )
        assert got.resolved, "supplying a digest must never be worse than supplying none"
        assert got["config"] == "ConfigX"
        assert "declares no hwdb digest" in got["confirmed_by"]
        assert "hw_config and hwdb digest" != got["confirmed_by"], (
            "an unmade check must not be reported as a byte match"
        )

    def test_the_image_digest_is_never_what_gets_compared(self):
        """A record whose stated digest happens to equal the IMAGE hash is still unconfirmed.

        This pins the type distinction directly: if the resolver ever reverts to comparing against
        ``digest``, this record would resolve as a byte match, and that is the bug.
        """
        got = D.design_string(
            {"hw_config": "queue_name_a", "hwdb_config_artifact_sha256": IMAGE_DIGEST},
            artifacts=_registry(
                dev_a=_Artifact(
                    hw_configs=("queue_name_a",), digest=IMAGE_DIGEST, hwdb_digest=DIGEST_A, config="ConfigX"
                )
            ),
        )
        assert not got.resolved, "matching the image hash is not matching the hwdb entry"


class TestItRefusesRatherThanGuessing:
    def test_the_right_name_with_the_wrong_bytes_is_refused(self):
        """THE MUTATION. This is the real mistake: a reused configuration name over different
        silicon. The name matches a registered device and the digest does not, and a resolver that
        trusted the name would attribute the result to the wrong hardware."""
        got = D.design_string(
            {"hw_config": "queue_name_a", "hwdb_config_artifact_sha256": DIGEST_B},
            artifacts=_registry(
                dev_a=_Artifact(
                    hw_configs=("queue_name_a",), digest=IMAGE_DIGEST, hwdb_digest=DIGEST_A, config="ConfigX"
                )
            ),
        )
        assert not got.resolved
        assert "different device under a reused name" in got["reason"]

    def test_an_unregistered_hw_config_is_refused_and_says_what_is_declared(self):
        got = D.design_string(
            {"hw_config": "never_registered"},
            artifacts=_registry(dev_a=_Artifact(hw_configs=("queue_name_a",), digest=DIGEST_A, config="ConfigX")),
        )
        assert not got.resolved
        assert "queue_name_a" in got["reason"], "the refusal should name what IS declared"

    def test_a_record_with_no_hw_config_is_refused(self):
        got = D.design_string({"hwdb_config_artifact_sha256": DIGEST_A}, artifacts=_registry())
        assert not got.resolved

    def test_two_devices_claiming_one_hw_config_is_reported_not_resolved(self):
        """A registry defect. Picking one would make a single submission string mean two devices,
        which is exactly the confusion this module exists to end."""
        got = D.design_string(
            {"hw_config": "shared", "hwdb_config_artifact_sha256": DIGEST_A},
            artifacts=_registry(
                dev_a=_Artifact(hw_configs=("shared",), digest=DIGEST_A, config="ConfigX"),
                dev_b=_Artifact(hw_configs=("shared",), digest=DIGEST_B, config="ConfigY"),
            ),
        )
        assert not got.resolved
        assert "cannot mean two devices" in got["reason"]

    def test_a_bitstream_with_no_config_names_no_design(self):
        got = D.design_string(
            {"hw_config": "queue_name_a", "hwdb_config_artifact_sha256": DIGEST_A},
            artifacts=_registry(dev_a=_Artifact(hw_configs=("queue_name_a",), digest=DIGEST_A, config="")),
        )
        assert not got.resolved

    def test_only_bitstreams_are_indexed(self):
        """A simulator binary is not a design. Indexing one would let a GSIM run resolve to an FPGA."""
        index = D.hw_config_index(_registry(sim=_Artifact(role="gsim_binary", hw_configs=("queue_name_a",))))
        assert index == {}


class TestTheShippedRegistry:
    def test_every_registered_bitstream_declares_a_queue_name_and_a_config(self):
        from merlin.common.provenance import load_artifacts

        bitstreams = {n: a for n, a in load_artifacts().items() if a.role == D.BITSTREAM_ROLE}
        assert bitstreams, "the registry must declare at least one bitstream for this join to mean anything"
        for name, artifact in bitstreams.items():
            assert artifact.hw_configs, f"{name} declares no hw_configs, so no run can resolve to it"
            assert artifact.config, f"{name} declares no config, so it names no design for the ledger"

    def test_no_two_bitstreams_claim_the_same_queue_name(self):
        collisions = {hw: names for hw, names in D.hw_config_index().items() if len(names) > 1}
        assert not collisions, f"one submission string would mean two devices: {collisions}"

    @pytest.mark.parametrize(
        "hw_config,expected",
        [
            ("alveo_u250_firesim_gemmini_rocket_30mhz", "firesim_gemmini_rocket_u250_30mhz"),
            ("resnet50_merlin_warm_measured_full_gemmini_u250_pinned", "firesim_gemmini_rocket_u250"),
        ],
    )
    def test_the_two_gemmini_queue_names_resolve_to_different_devices(self, hw_config, expected):
        """They share a `config`, so only the hw-config join tells them apart -- which is the whole
        reason this module is not a lookup on `config`."""
        got = D.design_string({"hw_config": hw_config})
        assert got.resolved and got["name"] == expected
