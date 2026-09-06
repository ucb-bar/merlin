"""The engine set a cross-validation capture is ABOUT, resolved from the pin registry.

A capture asserts two engines agree on one ELF, so it is valid only for those engines. Until this
existed the set lived solely inside a certificate produced by a command line that named all five
artifacts by hand -- so a grade that had just produced the ELF could not say what a capture of it would
be about, and a cache could not tell whether an earlier capture answered today's question.

These tests pin the two properties that make the resolution worth trusting: it is keyed on the DECLARED
role rather than a naming convention, and every artifact is verified against its digest before the set
is returned. Each refusal is exercised, because a resolver that cannot fail is worth nothing.
"""
from __future__ import annotations

import copy
import pathlib
import tempfile

import pytest
import yaml

from merlin.common import provenance as prov
from merlin.perf.engine_pins import REQUIRED_ROLES, EnginePinsUnavailable, engine_pins

TARGET = "gemmini"


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.safe_load(pathlib.Path(prov.pins_path()).read_text(encoding="utf-8"))


def _written(body: dict) -> pathlib.Path:
    path = pathlib.Path(tempfile.mkstemp(suffix=".yaml", prefix="pins-")[1])
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def _mutated(registry: dict, mutate) -> pathlib.Path:
    body = copy.deepcopy(registry)
    mutate(body["artifacts"])
    return _written(body)


def test_every_required_role_resolves_and_is_verified(registry):
    pins = engine_pins(TARGET, registry=_mutated(registry, lambda arts: None))
    assert set(pins) == set(REQUIRED_ROLES)
    for role, entry in pins.items():
        assert len(entry["sha256"]) == 64, f"{role} resolved without a digest"
        assert pathlib.Path(entry["path"]).is_file(), f"{role} resolved to a path that is not a file"


def test_a_role_no_artifact_claims_is_refused_by_name(registry):
    path = _mutated(registry, lambda arts: arts.pop("gemmini_verilator_simulator"))
    with pytest.raises(EnginePinsUnavailable, match="verilator_binary"):
        engine_pins(TARGET, registry=path)


def test_a_digest_that_disagrees_with_the_bytes_refuses_the_whole_set(registry):
    """Four good pins and one lie is the wrong-device hazard, so the set fails rather than degrades."""
    def mutate(arts):
        arts["gemmini_gsim_emulator"]["digest"] = "0" * 64
    with pytest.raises(EnginePinsUnavailable, match="does not match its declared digest"):
        engine_pins(TARGET, registry=_mutated(registry, mutate))


def test_two_artifacts_claiming_one_role_is_refused_rather_than_tie_broken(registry):
    """Picking one would make the capture describe an engine chosen by sort order."""
    def mutate(arts):
        arts["gemmini_gsim_emulator_copy"] = dict(arts["gemmini_gsim_emulator"])
    with pytest.raises(EnginePinsUnavailable, match="claim it"):
        engine_pins(TARGET, registry=_mutated(registry, mutate))


def test_an_artifact_with_no_declared_digest_would_certify_itself(registry):
    def mutate(arts):
        arts["gemmini_verilator_firrtl"]["digest"] = ""
    with pytest.raises(EnginePinsUnavailable, match="certify itself"):
        engine_pins(TARGET, registry=_mutated(registry, mutate))


def test_roles_are_declared_not_read_out_of_the_artifact_name(registry):
    """One target's elaboration is named after its own config, which no other target shares."""
    arts = prov.load_artifacts()
    firrtl = [n for n, a in arts.items() if a.target == TARGET and a.role == "gsim_firrtl"]
    assert firrtl == ["gemmini_gsim_model_serialclk"], (
        "the gsim FIRRTL is found by its DECLARED role; its name states a configuration, not a role")
