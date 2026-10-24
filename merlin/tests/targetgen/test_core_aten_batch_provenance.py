"""Batch verdict attribution follows the selected target and executed artifacts."""

from types import SimpleNamespace

from merlin.common import provenance
from merlin.targetgen.core_aten_provenance import batch_provenance


def test_provenance_derives_pins_and_records_actual_binaries(tmp_path, monkeypatch):
    from merlin.targetgen import provenance as target_provenance
    from merlin.targetgen import target_registry
    from merlin.targetgen.rtl import facts

    selected = []
    monkeypatch.setattr(target_provenance, "declared_pins", lambda target: [target + "_rtl", target + "_headers"])
    monkeypatch.setattr(target_registry, "load_contract", lambda target: {"hardware_pins": [target + "_rtl"]})
    monkeypatch.setattr(provenance, "verify", lambda name: selected.append(name) or SimpleNamespace(ok=True))
    observed = {}
    monkeypatch.setattr(provenance, "record", lambda **kwargs: observed.update(kwargs) or kwargs)
    simulator = tmp_path / "simulator"
    extension = tmp_path / "extension"
    rtl = tmp_path / "facts.json"
    for path in (simulator, extension, rtl):
        path.write_bytes(b"observed content")
    monkeypatch.setattr(facts, "ensure_facts", lambda target, explicit=None: explicit or rtl)
    build = tmp_path / "spike-build"
    build.mkdir()
    (build / "model.elf").write_bytes(b"executed content")
    batch_provenance(tmp_path, target="fixture", runner_options={"spike_binary": simulator, "extlib": extension})
    assert selected == ["fixture_rtl"]
    assert observed["artifacts"] == {
        "spike": simulator,
        "extension": extension,
        "rtl_facts": rtl,
        "spike-build/model.elf": build / "model.elf",
    }
    assert rtl in observed["sources"]
