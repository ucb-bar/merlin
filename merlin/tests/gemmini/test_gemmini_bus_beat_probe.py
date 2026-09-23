import importlib
from pathlib import Path

from merlin.runtime.backends.base import get_backend


def _module():
    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.bus_beat_probe")


def test_generated_verilator_metadata_is_derived_from_selected_binary(tmp_path: Path) -> None:
    module = _module()
    simulator = tmp_path / "simulator-chipyard.harness-ConfigFromBinary"
    simulator.write_bytes(b"model")
    root = tmp_path / "generated-src" / "chipyard.harness.TestHarness.ConfigFromBinary"
    objects = root / root.name
    objects.mkdir(parents=True)
    public = objects / "VTop.h"
    public.write_text("class VTop {};\n", encoding="utf-8")
    metadata = objects / "VTop_classes.mk"
    metadata.write_text("VM_TRACE = 0\n", encoding="utf-8")
    (objects / "VTop___024root.h").write_text("private internals\n", encoding="utf-8")

    got_public, got_metadata, problem = module._compiled_model_files(simulator)
    assert problem is None
    assert got_public == public
    assert got_metadata == metadata


def test_ambiguous_generated_model_refuses(tmp_path: Path) -> None:
    module = _module()
    simulator = tmp_path / "simulator-chipyard.harness-Same"
    simulator.write_bytes(b"model")
    for prefix in ("one", "two"):
        (tmp_path / "generated-src" / f"{prefix}.Same").mkdir(parents=True)
    public, metadata, problem = module._compiled_model_files(simulator)
    assert public is None and metadata is None
    assert "ambiguous" in problem
