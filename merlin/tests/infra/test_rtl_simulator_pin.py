"""A selected RTL executable must be a byte-checked build product of its pin."""

import subprocess
from types import SimpleNamespace

import pytest
import yaml

from merlin.common import provenance
from merlin.targetgen import rtl_simulator_pin
from merlin.targetgen.rtl_simulator_pin import verify_selected


def _committed_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config").write_text("design")
    env = {
        "PATH": "/usr/bin:/bin",
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    for args in (("init", "-q"), ("add", "config"), ("commit", "-qm", "initial")):
        subprocess.run(["git", "-C", str(source), *args], env=env, check=True)
    return source, subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()


def _write_registry(registry, revision, binary):
    pin = {
        "commit": revision,
        "root_env": "TEST_RTL_SOURCE",
        "targets": ["example"],
        "content_check": True,
        "requires_paths": ["config"],
        "build_products": {"simulator": provenance.file_digest(binary)},
    }
    registry.write_text(yaml.safe_dump({"version": 1, "pins": {"selected": pin}}))


@pytest.fixture
def selected(tmp_path, monkeypatch):
    source, revision = _committed_source(tmp_path)
    binary = source / "simulator"
    binary.write_bytes(b"compiled design")
    registry = tmp_path / "pins.yaml"
    _write_registry(registry, revision, binary)
    monkeypatch.setattr(provenance, "pins_path", lambda: registry)
    monkeypatch.setenv("TEST_RTL_SOURCE", str(source))
    monkeypatch.setenv("MERLIN_RTL_SIM_PIN", "selected")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "verilator")
    monkeypatch.setattr(rtl_simulator_pin, "_verify_facts", lambda *args: (registry,))
    return SimpleNamespace(verilator_path=lambda: binary), binary


def test_byte_verified_product_is_recorded(selected):
    backend, binary = selected
    record = verify_selected(backend, "verilator", "example")
    assert record["all_pins_ok"] is True
    assert record["artifact_digests"]["simulator"] == provenance.file_digest(binary)


def test_replaced_build_product_is_refused_even_in_observation_scope(selected):
    backend, binary = selected
    with provenance.observation_scope():
        verify_selected(backend, "verilator", "example")
        binary.write_bytes(b"different design")
        with pytest.raises(provenance.PinsError, match="build product"):
            verify_selected(backend, "verilator", "example")


def test_other_backend_path_is_not_authorized(selected):
    backend, binary = selected
    other = binary.with_name("another-simulator")
    other.write_bytes(binary.read_bytes())
    backend.verilator_path = lambda: other
    with pytest.raises(ValueError, match="not a build product"):
        verify_selected(backend, "verilator", "example")


def test_target_and_engine_must_agree(selected, monkeypatch):
    backend, _ = selected
    with pytest.raises(ValueError, match="does not declare target"):
        verify_selected(backend, "verilator", "other")
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE")
    with pytest.raises(ValueError, match="matching MERLIN_REQUIRED_RTL_ENGINE"):
        verify_selected(backend, "verilator", "example")


def test_unselected_pin_is_inert(monkeypatch):
    monkeypatch.delenv("MERLIN_RTL_SIM_PIN", raising=False)
    assert verify_selected(object(), "verilator", "example") is None


def test_unbound_facts_are_refused(selected, monkeypatch):
    backend, _ = selected

    def refuse(*args):
        raise ValueError("RTL facts FIRRTL is not a byte-checked product")

    monkeypatch.setattr(rtl_simulator_pin, "_verify_facts", refuse)
    with pytest.raises(ValueError, match="RTL facts FIRRTL"):
        verify_selected(backend, "verilator", "example")


@pytest.fixture
def pinned_facts(tmp_path, monkeypatch):
    from merlin.targetgen import target_experiment
    from merlin.targetgen.rtl import facts, source_selection

    firrtl = tmp_path / "selected.fir"
    firrtl.write_text("circuit example")
    digest = provenance.file_digest(firrtl)
    selected = {
        "selection_sha256": "bundle-digest",
        "config": "ExampleConfig",
        "sources": {"firrtl": {"path": str(firrtl), "sha256": digest}},
    }
    inputs = {
        "source_bundle_path": "bundle.json",
        "source_bundle_sha256": "bundle-digest",
        "fir_sha256": digest,
    }
    monkeypatch.setattr(facts, "load_facts", lambda _: {"inputs": inputs})
    monkeypatch.setattr(source_selection, "load_selection", lambda *args, **kwargs: selected)
    monkeypatch.setattr(source_selection, "production_consistency", lambda _: {"status": "verified"})
    monkeypatch.setattr(
        target_experiment,
        "load_capability_manifest",
        lambda _: SimpleNamespace(contract={"runtime": {"rtl_sim_config": "ExampleConfig"}}),
    )
    monkeypatch.setattr(facts, "find_facts", lambda _: firrtl)
    pin = SimpleNamespace(build_products=((firrtl.name, digest),))
    return pin, selected


def test_facts_must_bind_the_pinned_firrtl_and_runtime(tmp_path, pinned_facts):
    pin, selected = pinned_facts
    rtl_simulator_pin._verify_facts(pin, "example", tmp_path)
    selected["config"] = "DifferentConfig"
    with pytest.raises(ValueError, match="configuration differs"):
        rtl_simulator_pin._verify_facts(pin, "example", tmp_path)
    selected["config"] = "ExampleConfig"
    selected["sources"]["firrtl"]["sha256"] = "unbound"
    with pytest.raises(ValueError, match="input differs|not a byte-checked product"):
        rtl_simulator_pin._verify_facts(pin, "example", tmp_path)
