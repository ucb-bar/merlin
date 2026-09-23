"""Selected runtime configuration cannot substitute for certificate pin checks."""

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import heldout_qualification as Q
from merlin_experiments.phase2 import revealed_corpus as RC


@pytest.mark.parametrize("provider", ["array_fixture", "vector_fixture"])
@pytest.mark.parametrize("mismatch", [None, "gsim", "verilator"])
def test_selected_runtime_checks_both_pins_and_exits_context(tmp_path, monkeypatch, provider, mismatch):
    from merlin.runtime.backends import base

    binaries = {}
    for engine in ("gsim", "verilator"):
        path = tmp_path / engine
        path.write_text(f"{provider}:{engine}")
        binaries[engine] = path
    pins = {f"{engine}_binary": {"path": str(path), "sha256": RC.sha_file(path)} for engine, path in binaries.items()}
    certificate = SimpleNamespace(target=provider, pins=pins)
    events = []

    @contextlib.contextmanager
    def configure(*, binaries, gsim_max_cycles):
        assert binaries == {engine: Path(pins[f"{engine}_binary"]["path"]) for engine in ("gsim", "verilator")}
        assert gsim_max_cycles is None
        events.append("enter")
        try:
            yield selected
        finally:
            events.append("exit")

    selected = SimpleNamespace(
        pinned_runtime=configure,
        gsim_path=lambda: binaries["gsim"],
        verilator_path=lambda: binaries["verilator"],
    )

    def get_backend(target):
        assert target == provider
        return selected

    monkeypatch.setattr(base, "get_backend", get_backend)
    if mismatch:
        binaries[mismatch].write_text("substituted engine")
        with pytest.raises(RC.QualificationError, match=f"runtime {mismatch} binary differs"):
            with Q._pinned_runtime(certificate, gsim_max_cycles=None):
                pytest.fail("capture reached despite changed binary")
    else:
        with pytest.raises(RuntimeError, match="capture failed"):
            with Q._pinned_runtime(certificate, gsim_max_cycles=None) as backend:
                assert backend is selected
                raise RuntimeError("capture failed")
    assert events == ["enter", "exit"]


def test_runtime_without_configuration_capability_refuses(monkeypatch):
    from merlin.runtime.backends import base

    monkeypatch.setattr(base, "get_backend", lambda _: object())
    with pytest.raises(RC.QualificationError, match="does not expose pinned runtime"):
        with Q._pinned_runtime(SimpleNamespace(target="fixture"), gsim_max_cycles=10):
            pytest.fail("unsupported backend was admitted")


def test_baseline_lowering_threads_selected_contract(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_common, oot_runner

    tool = tmp_path / "tool.py"
    tool.write_text("# fixture")
    package = SimpleNamespace(manifest={}, tool=tool)
    contract = tmp_path / "selected-contract"
    seen = []

    def load_package(path, *, contract):
        seen.append(("package", contract))
        return package

    def load_capsule(path, *, contract):
        seen.append(("capsule", contract))
        return {}

    def run_entrypoints(package, baseline, capsule, paths, *, contract, **kwargs):
        seen.append(("entrypoints", contract))
        for name in ("command_buffer.json", "lowered.llvm.mlir"):
            (paths.generated / name).write_text("fixture")

    monkeypatch.setattr(oot_runner, "load_package", load_package)
    monkeypatch.setattr(oot_runner, "integrity_scan", lambda _: None)
    monkeypatch.setattr(capsule_common, "load_capsule", load_capsule)
    monkeypatch.setattr(capsule_common, "run_entrypoints", run_entrypoints)
    output = tmp_path / "artifacts"
    member = SimpleNamespace(name="fixture", source_dir=tmp_path / "capsule")
    assert Q.lower_with_functional_baseline(tmp_path, member, output, 10, contract_root=contract) == output
    assert seen == [(name, contract) for name in ("package", "capsule", "entrypoints")]
