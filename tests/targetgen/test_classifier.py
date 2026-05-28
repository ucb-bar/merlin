from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from targetgen.intake import (  # noqa: E402
    SOURCE_TO_TARGETGEN,
    build_source_inventory,
    classify_inventory,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _classify(name: str):
    inv = build_source_inventory(target=name, sources=[FIXTURES / name])
    return classify_inventory(inv)


def test_cuda_tile_routes_to_external_mlir_bridge() -> None:
    cls = _classify("external_mlir_cuda_tile")
    assert "external_mlir_bridge" in cls.source_styles
    assert "post_global_plugin" in cls.targetgen_styles
    assert cls.primary_integration == "post_global_plugin"


def test_gemmini_routes_to_rocc_and_chipyard() -> None:
    cls = _classify("chipyard_gemmini_rocc")
    assert "rocc_accelerator" in cls.source_styles
    assert "chipyard_generator" in cls.source_styles
    # rocc → post_global_plugin + llvm_ukernel; chipyard → runtime_hal
    assert "post_global_plugin" in cls.targetgen_styles
    assert "llvm_ukernel" in cls.targetgen_styles
    assert "runtime_hal" in cls.targetgen_styles


def test_radiance_routes_to_gpu_codegen_stack() -> None:
    cls = _classify("radiance_gluon_gpu")
    assert "gpu_codegen_stack" in cls.source_styles
    assert set(cls.targetgen_styles) == {
        "post_global_plugin",
        "structured_text_isa",
        "runtime_hal",
    }
    assert cls.primary_integration == "runtime_hal"


def test_fft_generator_routes_to_mmio_runtime_hal() -> None:
    cls = _classify("fft_generator_mmio")
    assert "mmio_accelerator" in cls.source_styles
    assert "chipyard_generator" in cls.source_styles
    assert cls.targetgen_styles == ["runtime_hal"]
    assert cls.primary_integration == "runtime_hal"


def test_classifier_records_rationales_and_confidence() -> None:
    cls = _classify("radiance_gluon_gpu")
    assert cls.rationales
    assert 0.0 < cls.confidence <= 1.0


def test_classifier_default_when_no_signals(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    inv = build_source_inventory(target="empty", sources=[empty])
    cls = classify_inventory(inv)
    assert cls.source_styles == []
    assert cls.targetgen_styles == ["llvm_ukernel"]
    assert cls.confidence == 0.0
    assert any("No source-facing styles matched" in m for m in cls.missing_information)


def test_source_to_targetgen_table_covers_all_styles() -> None:
    expected = {
        "external_mlir_bridge",
        "external_toolchain_bridge",
        "chipyard_generator",
        "rocc_accelerator",
        "mmio_accelerator",
        "rtl_or_systemc_model",
        "llvm_backend_extension",
        "gpu_codegen_stack",
    }
    assert set(SOURCE_TO_TARGETGEN.keys()) == expected
    valid_targets = {"runtime_hal", "structured_text_isa", "post_global_plugin", "llvm_ukernel"}
    for src, tgts in SOURCE_TO_TARGETGEN.items():
        assert set(tgts).issubset(valid_targets), f"{src} maps to unknown style"
